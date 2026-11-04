"""Coverage metrics for the owner scope defined in configs/owner_aliases.yaml.

Every number is a request count from the window start onward. No label, no-action or
outcome column is read, so the metrics can be computed before the scope is frozen.

Metrics:
  scope share       requests with at least one in-scope owner key / all window requests
  network recall    the same share among requests filed by the protection-network reporters
  agency recall     the same share among requests filed by the other selected film/TV
                    agencies (protection-network reporters excluded)
  gap               out-of-scope requests of each reporter set, split into owner keys that
                    were reviewed and left out, keys that were never candidates, and requests
                    without an owner key
  breakdowns        in-scope requests by group, market, evidence basis and candidate source
  threshold check   keys a lower ranking threshold would add as unreviewed candidates

A request counts once in every share, even when its owner name maps to several keys.

Usage (from the repo root):
    python -m piracylens.data.scope_metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from piracylens.data.aliases import load_aliases, missing_keys
from piracylens.data.scope_candidates import (
    DEFAULT_OWNER_KEYS,
    DEFAULT_PARAMS,
    DEFAULT_REPORTER_ALIASES,
    DEFAULT_REQUESTS,
    _prepare,
    _scalar,
)

DEFAULT_OWNER_ALIASES = Path("configs/owner_aliases.yaml")
DEFAULT_CANDIDATES = Path("data/interim/scope_candidates.csv")
DEFAULT_OUT_DIR = Path("data/interim")
DEFAULT_ALT_RANKING_MIN = 2000
KEY_COLUMNS = ("owner_key", "in_scope", "group", "requests", "network_requests", "agency_requests")
_CANDIDATE_FLAGS = ("source_a", "source_b", "source_c", "source_d", "in_seed")


def _share(part: int, whole: int) -> float:
    return round(part / whole, 6) if whole else 0.0


def build_metrics(
    params_path: Path,
    reporter_aliases: Path,
    owner_aliases: Path,
    candidates: Path,
    requests: Path,
    owner_keys: Path,
    out_dir: Path,
    alt_ranking_min: int = DEFAULT_ALT_RANKING_MIN,
    threads: int = 16,
    memory_limit: str = "14GB",
) -> dict:
    """Compute the coverage metrics, write them to out_dir and return the summary."""
    for path in (owner_aliases, candidates):
        if not path.is_file():
            raise SystemExit(f"input not found: {path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_out = out_dir / "scope_metrics.json"
    keys_out = out_dir / "scope_metrics_keys.csv"
    prep = _prepare(
        params_path,
        reporter_aliases,
        requests,
        owner_keys,
        out_dir / "scope_metrics.reporters.tsv.tmp",
        threads,
        memory_limit,
        True,
    )
    con, params = prep.con, prep.params
    if not 0 < alt_ranking_min < params.ranking_min_requests:
        raise SystemExit(
            f"alternative threshold must be below {params.ranking_min_requests}: {alt_ranking_min}"
        )

    # 1. Inputs: alias keys must exist in the mapping and every alias key must be a candidate.
    alias_rows = load_aliases(owner_aliases, "owner")
    known = [
        r[0]
        for r in con.execute(
            f"SELECT DISTINCT owner_key FROM read_parquet('{owner_keys}')"
        ).fetchall()
    ]
    missing = missing_keys(alias_rows, known)
    if missing:
        raise SystemExit(f"owner alias keys missing from the mapping: {missing[:10]}")
    with candidates.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        absent_cols = [
            c for c in ("owner_key", *_CANDIDATE_FLAGS) if c not in (reader.fieldnames or [])
        ]
        if absent_cols:
            raise SystemExit(f"candidate file lacks columns: {absent_cols}")
        cand_rows = [(r["owner_key"], *(r[c] == "true" for c in _CANDIDATE_FLAGS)) for r in reader]
    cand_keys = {r[0] for r in cand_rows}
    not_candidates = sorted({r.key for r in alias_rows} - cand_keys)
    if not_candidates:
        raise SystemExit(f"owner alias keys that are not candidates: {not_candidates[:10]}")

    con.execute(
        "CREATE TABLE scope AS SELECT unnest(?::VARCHAR[]) AS owner_key, "
        "unnest(?::VARCHAR[]) AS grp, unnest(?::VARCHAR[]) AS market, "
        "unnest(?::VARCHAR[]) AS basis",
        [
            [r.key for r in alias_rows],
            [r.group for r in alias_rows],
            [r.market for r in alias_rows],
            [r.basis for r in alias_rows],
        ],
    ).fetchall()
    cols = list(zip(*cand_rows, strict=True))
    con.execute(
        "CREATE TABLE cand AS SELECT unnest(?::VARCHAR[]) AS owner_key, "
        "unnest(?::BOOLEAN[]) AS a, unnest(?::BOOLEAN[]) AS b, unnest(?::BOOLEAN[]) AS c, "
        "unnest(?::BOOLEAN[]) AS d, unnest(?::BOOLEAN[]) AS seed",
        [list(c) for c in cols],
    ).fetchall()

    # 2. Reporter sets: the protection network, and the other selected agencies.
    network = list(params.network_reporters)
    agencies = [g for g in params.agency_selected if g not in params.network_reporters]
    # Views cannot hold query parameters, so the reporter sets live in small tables.
    for table, values in (("net_set", network), ("agency_set", agencies)):
        con.execute(
            f"CREATE TABLE {table} AS SELECT unnest(?::VARCHAR[]) AS v", [values]
        ).fetchall()
    # The window start is a validated date, so writing it into the SQL text is safe.
    con.execute(
        f"""
        CREATE VIEW win AS
        SELECT q.request_id, q.owner_name,
               coalesce(p.reporter_group IN (SELECT v FROM net_set)
                        OR p.reporter_key IN (SELECT v FROM net_set), false) AS in_network,
               coalesce(p.reporter_group IN (SELECT v FROM agency_set), false)
                   AND NOT coalesce(p.reporter_group IN (SELECT v FROM net_set)
                                    OR p.reporter_key IN (SELECT v FROM net_set), false)
                   AS in_agencies
        FROM read_parquet('{requests}') AS q
        LEFT JOIN reporters AS p USING (reporter_name)
        WHERE q.requested_at_utc >= TIMESTAMP '{params.window_start.isoformat()}'
        """
    ).fetchall()
    con.execute(
        f"""
        CREATE VIEW win_keys AS
        SELECT w.request_id, w.in_network, w.in_agencies, k.owner_key
        FROM win AS w LEFT JOIN read_parquet('{owner_keys}') AS k USING (owner_name)
        """
    ).fetchall()

    total = _scalar(con, "SELECT count(*) FROM win")
    in_scope = _scalar(
        con,
        "SELECT count(DISTINCT request_id) FROM win_keys JOIN scope USING (owner_key)",
    )

    # 3. Recall and gap per reporter set. Rank: 3 in scope, 2 reviewed and left out,
    # 1 never a candidate, 0 no owner key. A request takes its highest rank.
    def reporter_set(flag: str) -> dict:
        n, s, r, nc, nk = con.execute(
            f"""
            WITH ranked AS (
                SELECT request_id,
                       max(CASE WHEN s.owner_key IS NOT NULL THEN 3
                                WHEN c.owner_key IS NOT NULL THEN 2
                                WHEN w.owner_key IS NOT NULL THEN 1 ELSE 0 END) AS rnk
                FROM win_keys AS w
                LEFT JOIN scope AS s USING (owner_key)
                LEFT JOIN cand AS c USING (owner_key)
                WHERE w.{flag}
                GROUP BY request_id
            )
            SELECT count(*), count(*) FILTER (WHERE rnk = 3), count(*) FILTER (WHERE rnk = 2),
                   count(*) FILTER (WHERE rnk = 1), count(*) FILTER (WHERE rnk = 0)
            FROM ranked
            """
        ).fetchall()[0]
        return {
            "requests": n,
            "in_scope": s,
            "recall": _share(s, n),
            "gap": {"reviewed_out": r, "never_candidate": nc, "no_owner_key": nk},
        }

    network_metrics = {"reporters": network, **reporter_set("in_network")}
    agency_metrics = {"reporters": agencies, **reporter_set("in_agencies")}

    # 4. Breakdowns of in-scope requests (each request once per category value).
    def breakdown(column: str) -> dict:
        rows = con.execute(
            f"""
            SELECT {column}, count(DISTINCT request_id) AS n
            FROM win_keys JOIN scope USING (owner_key) JOIN cand USING (owner_key)
            GROUP BY 1 ORDER BY n DESC, 1
            """
        ).fetchall()
        return {k: {"requests": n, "share": _share(n, in_scope)} for k, n in rows}

    source_case = (
        "CASE WHEN a OR d THEN 'ranking_or_prefix' "
        "WHEN b OR c THEN 'network_or_agency_only' ELSE 'seed_only' END"
    )
    by_group = breakdown("grp")
    breakdowns = {
        "group_top20": dict(list(by_group.items())[:20]),
        "groups": len(by_group),
        "vgtrk_share": by_group.get("vgtrk", {"share": 0.0})["share"],
        "market": breakdown("market"),
        "evidence_basis": breakdown("basis"),
        "source": breakdown(source_case),
    }

    # 5. Per-key volumes for the candidate keys (joined with private review notes later).
    con.execute(
        """
        CREATE TABLE key_volumes AS
        SELECT owner_key, count(*) AS requests,
               count(*) FILTER (WHERE in_network) AS network_requests,
               count(*) FILTER (WHERE in_agencies) AS agency_requests
        FROM win_keys WHERE owner_key IS NOT NULL GROUP BY owner_key
        """
    ).fetchall()
    tmp = keys_out.with_name(keys_out.name + ".tmp")
    con.execute(
        f"""
        COPY (
            SELECT c.owner_key, s.owner_key IS NOT NULL AS in_scope, s.grp AS "group",
                   coalesce(v.requests, 0) AS requests,
                   coalesce(v.network_requests, 0) AS network_requests,
                   coalesce(v.agency_requests, 0) AS agency_requests
            FROM cand AS c LEFT JOIN scope AS s USING (owner_key)
            LEFT JOIN key_volumes AS v USING (owner_key)
            ORDER BY requests DESC, c.owner_key
        ) TO '{tmp}' (FORMAT csv, HEADER true)
        """
    ).fetchall()
    n_csv = _scalar(con, f"SELECT count(*) FROM read_csv('{tmp}', header = true)")
    if n_csv != len(cand_keys):
        tmp.unlink()
        raise SystemExit(f"row count mismatch: expected {len(cand_keys)}, csv has {n_csv}")
    tmp.replace(keys_out)

    # 6. Threshold check: keys a lower ranking threshold would add (unreviewed upper bound).
    alt_keys, alt_requests = con.execute(
        """
        SELECT count(*), coalesce(sum(requests), 0) FROM key_volumes AS v
        WHERE requests >= ? AND requests < ?
          AND NOT EXISTS (SELECT 1 FROM cand AS c WHERE c.owner_key = v.owner_key)
        """,
        [alt_ranking_min, params.ranking_min_requests],
    ).fetchall()[0]

    summary = {
        "window_start": params.window_start.isoformat(),
        "window_requests": total,
        "scope_share": {
            "in_scope_requests": in_scope,
            "share": _share(in_scope, total),
            "groups": len(by_group),
            "keys": len(alias_rows),
        },
        "network_recall": network_metrics,
        "agency_recall": agency_metrics,
        "breakdowns": breakdowns,
        "threshold_check": {
            "ranking_min_requests": params.ranking_min_requests,
            "alternative_min_requests": alt_ranking_min,
            "added_unreviewed_keys": alt_keys,
            "added_requests_upper_bound": alt_requests,
            "added_share_upper_bound": _share(alt_requests, total),
        },
    }
    tmp = summary_out.with_name(summary_out.name + ".tmp")
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(summary_out)

    print(
        f"window requests {total:,} | in scope {in_scope:,} ({summary['scope_share']['share']:.2%})"
        f" | {len(alias_rows)} keys, {len(by_group)} groups | VGTRK share of scope "
        f"{breakdowns['vgtrk_share']:.1%} | {time.perf_counter() - prep.t0:.1f}s"
    )
    for name, m in (("network", network_metrics), ("agencies", agency_metrics)):
        g = m["gap"]
        print(
            f"{name} recall {m['recall']:.1%} of {m['requests']:,} | gap: reviewed out "
            f"{g['reviewed_out']:,}, never candidate {g['never_candidate']:,}, "
            f"no owner key {g['no_owner_key']:,}"
        )
    print(
        f"threshold >= {alt_ranking_min:,}: +{alt_keys:,} unreviewed keys, "
        f"{alt_requests:,} requests ({summary['threshold_check']['added_share_upper_bound']:.2%})"
        f" -> {summary_out}, {keys_out}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute scope coverage metrics.")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--reporter-aliases", type=Path, default=DEFAULT_REPORTER_ALIASES)
    parser.add_argument("--owner-aliases", type=Path, default=DEFAULT_OWNER_ALIASES)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--requests", type=Path, default=DEFAULT_REQUESTS)
    parser.add_argument("--owner-keys", type=Path, default=DEFAULT_OWNER_KEYS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--alt-ranking-min", type=int, default=DEFAULT_ALT_RANKING_MIN)
    args = parser.parse_args()
    build_metrics(
        args.params,
        args.reporter_aliases,
        args.owner_aliases,
        args.candidates,
        args.requests,
        args.owner_keys,
        args.out_dir,
        args.alt_ranking_min,
    )


if __name__ == "__main__":
    main()
