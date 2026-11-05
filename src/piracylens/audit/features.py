"""Feature table for the takedown audit model.

One row per decided in-window request x domain row of the scoped tables, with the raw
model inputs fixed in the design file and both labels. Inputs that must be fitted (the
top-k category mapping and the domain n-gram vocabulary) are not built here: they are
fitted on training rows inside the model pipeline, so validation and test rows cannot
shape them.

Domain history comes from every owner's rows in the full domains table from the history
start. The past rate of a row uses only decided rows filed at least the configured gap
earlier (compared on filing timestamps), so the row itself, its own request and anything
filed later cannot reach it. The domain age uses the first appearance of the domain in any
row filed strictly earlier.

The command prints row counts and feature coverage only: no label sums, rates or shares.

Usage (from the repo root):
    python -m piracylens.audit.features --dry-run   # sizes only, writes nothing
    python -m piracylens.audit.features
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterable
from pathlib import Path

import duckdb

from piracylens.audit.preregistration import Preregistration, load_preregistration
from piracylens.data.aliases import AliasRow, load_aliases

DEFAULT_OWNER_ALIASES = Path("configs/owner_aliases.yaml")
DEFAULT_REPORTER_ALIASES = Path("configs/reporter_aliases.yaml")
DEFAULT_SCOPED_REQUESTS = Path("data/interim/scoped_requests.parquet")
DEFAULT_SCOPED_DOMAINS = Path("data/interim/scoped_domains.parquet")
DEFAULT_DOMAINS = Path("data/processed/domains.parquet")
DEFAULT_REQUESTS = Path("data/processed/requests.parquet")
DEFAULT_OUT = Path("data/interim/audit_features.parquet")

COUNTERS = ("urls_removed", "urls_not_in_index", "urls_no_action", "urls_pending")
KEY_COLUMNS = ("request_id", "domain", "requested_at_utc", "split")
LABEL_COLUMNS = ("label_primary", "label_sensitivity")
# Every input except the n-grams, which the pipeline builds from the domain column.
INPUT_COLUMNS = (
    "tld",
    "domain_length",
    "domain_labels",
    "domain_digit_share",
    "domain_hyphens",
    "domain_punycode",
    "domains_in_request",
    "urls_in_request",
    "urls_for_domain",
    "owner_group",
    "market",
    "content_type",
    "region",
    "reporter_group",
    "reporter_brand",
    "self_filed",
    "domain_past_rate",
    "has_domain_history",
    "domain_past_requests",
    "domain_age_days",
    "has_domain_age",
)
OUTPUT_COLUMNS = (*KEY_COLUMNS, *INPUT_COLUMNS, *LABEL_COLUMNS)


def _q(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> list[tuple]:
    return con.execute(sql, params or []).fetchall()


def _split_case(p: Preregistration, col: str) -> str:
    # Dates come from the strict loader (validated date objects), so inlining them is safe.
    day = f"CAST({col} AS DATE)"
    return (
        "CASE "
        f"WHEN {day} BETWEEN DATE '{p.train.start}' AND DATE '{p.train.end}' THEN 'train' "
        f"WHEN {day} BETWEEN DATE '{p.validation.start}' AND DATE '{p.validation.end}' "
        "THEN 'validation' "
        f"WHEN {day} BETWEEN DATE '{p.test.start}' AND DATE '{p.test.end}' THEN 'test' END"
    )


def _check_rules(p: Preregistration) -> None:
    """Stop if the design file asks for a rule this builder does not implement."""
    expected = {
        "past_rate_unit": "decided_rows",
        "tld_rule": "last_label",
        "time_basis": "timestamp",
        "domain_normalization": "lowercase_trim",
        "request_size_rows": "all",
        "self_filed_level": "owner_group",
        "past_rate_history": "full_domains_table",
        "no_history": "missing_with_flag",
    }
    wrong = {k: getattr(p, k) for k, v in expected.items() if getattr(p, k) != v}
    if wrong:
        raise SystemExit(f"design rules not implemented by this builder: {wrong}")
    missing = sorted(set(INPUT_COLUMNS) - set(p.inputs))
    if missing or "domain_ngrams" not in p.inputs or len(p.inputs) != len(INPUT_COLUMNS) + 1:
        raise SystemExit(f"design inputs differ from the builder's columns: missing {missing}")


def _register_aliases(
    con: duckdb.DuckDBPyConnection,
    owner_rows: Iterable[AliasRow],
    reporter_rows: Iterable[AliasRow],
) -> None:
    owners = [(r.group, r.key) for r in owner_rows]
    reporters = [(r.key, r.brand) for r in reporter_rows]
    if not owners or not reporters:
        raise SystemExit("alias files must not be empty")
    _q(
        con,
        "CREATE TABLE owner_keys AS SELECT unnest(?::VARCHAR[]) AS owner_group, "
        "unnest(?::VARCHAR[]) AS key",
        [[g for g, _ in owners], [k for _, k in owners]],
    )
    _q(
        con,
        "CREATE TABLE reporter_brands AS SELECT unnest(?::VARCHAR[]) AS key, "
        "unnest(?::VARCHAR[]) AS brand",
        [[k for k, _ in reporters], [b for _, b in reporters]],
    )


def build_features(
    prereg: Preregistration,
    owner_rows: Iterable[AliasRow],
    reporter_rows: Iterable[AliasRow],
    scoped_requests: Path,
    scoped_domains: Path,
    domains: Path,
    requests: Path,
    out: Path,
    *,
    dry_run: bool = False,
    threads: int = 16,
    memory_limit: str = "14GB",
) -> int:
    """Write the feature table (or, with dry_run, only print its sizes); return its rows."""
    t0 = time.perf_counter()
    _check_rules(prereg)
    for path in (scoped_requests, scoped_domains, domains, requests):
        if not path.is_file():
            raise SystemExit(f"input not found: {path}")
    out.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    _q(con, f"SET threads = {threads}")
    _q(con, f"SET memory_limit = '{memory_limit}'")
    _q(con, f"SET temp_directory = '{out.parent / 'duckdb_tmp'}'")
    _register_aliases(con, owner_rows, reporter_rows)

    def decided(prefix: str = "") -> str:
        return " + ".join(f"{prefix}{c}" for c in prereg.decided_counters) + " > 0"

    def lab(prefix: str = "") -> str:
        return f"{prefix}{prereg.label_counter} >= {prereg.label_min_count}"

    norm = "lower(trim({}))"
    gap = f"INTERVAL {prereg.past_rate_min_gap_days} DAY"
    hist_start = f"TIMESTAMP '{prereg.history_start}'"
    data_stop = f"TIMESTAMP '{prereg.data_end}' + INTERVAL 1 DAY"

    # 1. Target rows: decided, in the window.
    _q(
        con,
        f"""CREATE TABLE tgt AS
        SELECT d.request_id, {norm.format("d.domain")} AS domain, d.requested_at_utc,
               {_split_case(prereg, "d.requested_at_utc")} AS split,
               {", ".join(f"d.{c}" for c in COUNTERS)}
        FROM read_parquet('{scoped_domains}') AS d
        WHERE d.in_window AND {decided("d.")}""",
    )
    if _q(con, "SELECT count(*) FROM tgt WHERE split IS NULL")[0][0]:
        raise SystemExit("target rows outside every split")
    _q(
        con,
        "CREATE TABLE tdom AS SELECT row_number() OVER (ORDER BY domain) AS did, domain "
        "FROM (SELECT DISTINCT domain FROM tgt)",
    )

    # 2. History events of the target domains, one row per (domain, filing time).
    _q(
        con,
        f"""CREATE TABLE ev AS
        SELECT m.did, q.requested_at_utc AS t, count(*) AS n_all,
               count(*) FILTER (WHERE {decided("d.")}) AS n_dec,
               count(*) FILTER (WHERE {decided("d.")} AND {lab("d.")}) AS n_lab
        FROM read_parquet('{domains}') AS d
        JOIN read_parquet('{requests}') AS q USING (request_id)
        JOIN tdom AS m ON m.domain = {norm.format("d.domain")}
        WHERE q.requested_at_utc >= {hist_start} AND q.requested_at_utc < {data_stop}
        GROUP BY 1, 2""",
    )

    n_tgt, n_dom = _q(con, "SELECT count(*), (SELECT count(*) FROM tdom) FROM tgt")[0]
    n_ev, n_hist, n_dec = _q(con, "SELECT count(*), sum(n_all), sum(n_dec) FROM ev")[0]
    n_www = _q(con, "SELECT count(*) FROM tdom WHERE domain LIKE 'www.%'")[0][0]
    print(
        f"target rows {n_tgt:,}, distinct domains {n_dom:,} ({n_www:,} start with www., "
        f"{100 * n_www / max(n_dom, 1):.2f}%)\n"
        f"history rows on these domains {n_hist or 0:,} (decided {n_dec or 0:,}), "
        f"distinct (domain, time) events {n_ev:,}"
    )
    for split, n in _q(
        con, "SELECT split, count(*) FROM tgt GROUP BY 1 ORDER BY min(requested_at_utc)"
    ):
        print(f"  {split:<11}{n:>14,}")
    if dry_run:
        print(f"dry run: nothing written ({time.perf_counter() - t0:.1f}s)")
        return n_tgt

    # 3. Cumulative decided history per domain, and the first appearance of each domain.
    _q(
        con,
        """CREATE TABLE cum AS
        SELECT did, t,
               sum(n_dec) OVER w AS past_n, sum(n_lab) OVER w AS past_k
        FROM ev WHERE n_dec > 0
        WINDOW w AS (PARTITION BY did ORDER BY t ROWS UNBOUNDED PRECEDING)""",
    )
    _q(con, "CREATE TABLE first_seen AS SELECT did, min(t) AS first_t FROM ev GROUP BY 1")
    _q(con, "DROP TABLE ev")  # every later step reads cum and first_seen only

    # 4. Request size over every row of the request, decided or not.
    _q(
        con,
        f"""CREATE TABLE rsize AS
        SELECT request_id, count(*) AS domains_in_request,
               sum({" + ".join(COUNTERS)}) AS urls_in_request
        FROM read_parquet('{scoped_domains}')
        WHERE request_id IN (SELECT DISTINCT request_id FROM tgt)
        GROUP BY 1""",
    )

    # 5. Assemble and write in one pass; the past rate joins the last cumulative event at or
    # before t - gap. The row count is checked on the written file before it is put in place.
    sens = (
        f"b.{prereg.label_counter} / ({' + '.join(f'b.{c}' for c in prereg.decided_counters)}) "
        f">= {prereg.sensitivity_min_share}"
    )
    tmp = out.with_name(out.name + ".tmp")
    _q(
        con,
        f"""COPY (
        WITH base AS (
            SELECT g.*, m.did, g.requested_at_utc - {gap} AS cutoff
            FROM tgt AS g JOIN tdom AS m USING (domain)
        )
        SELECT b.request_id, b.domain, b.requested_at_utc, b.split,
               regexp_extract(b.domain, '[^.]*$') AS tld,
               length(b.domain) AS domain_length,
               len(string_split(b.domain, '.')) AS domain_labels,
               length(regexp_replace(b.domain, '[^0-9]', '', 'g'))
                   / greatest(length(b.domain), 1) AS domain_digit_share,
               length(b.domain) - length(replace(b.domain, '-', '')) AS domain_hyphens,
               regexp_matches(b.domain, '(^|\\.)xn--') AS domain_punycode,
               s.domains_in_request, s.urls_in_request,
               {" + ".join(f"b.{c}" for c in COUNTERS)} AS urls_for_domain,
               r.owner_group, r.market, r.content_type, r.region,
               r.reporter_group,
               coalesce(rb.brand, r.reporter_key) AS reporter_brand,
               ok.key IS NOT NULL AS self_filed,
               c.past_k / c.past_n AS domain_past_rate,
               c.past_n IS NOT NULL AS has_domain_history,
               coalesce(c.past_n, 0) AS domain_past_requests,
               CASE WHEN f.first_t < b.requested_at_utc
                    THEN epoch(b.requested_at_utc - f.first_t) / 86400.0 END AS domain_age_days,
               coalesce(f.first_t < b.requested_at_utc, false) AS has_domain_age,
               {lab("b.")} AS label_primary,
               {sens} AS label_sensitivity
        FROM base AS b
        ASOF LEFT JOIN cum AS c ON b.did = c.did AND b.cutoff >= c.t
        LEFT JOIN first_seen AS f ON f.did = b.did
        JOIN rsize AS s ON s.request_id = b.request_id
        JOIN read_parquet('{scoped_requests}') AS r ON r.request_id = b.request_id
        LEFT JOIN reporter_brands AS rb ON rb.key = r.reporter_key
        LEFT JOIN owner_keys AS ok ON ok.owner_group = r.owner_group AND ok.key = r.reporter_key
        ) TO '{tmp}' (FORMAT parquet, COMPRESSION zstd)""",
    )
    n = _q(con, f"SELECT count(*) FROM read_parquet('{tmp}')")[0][0]
    if n != n_tgt:
        tmp.unlink()
        raise SystemExit(f"row count changed while joining: {n_tgt} targets, {n} written rows")
    tmp.replace(out)

    print("coverage per split (share of rows): history, age, self-filed")
    for split, h, a, sf in _q(
        con,
        """SELECT split, avg(has_domain_history::INT), avg(has_domain_age::INT),
                  avg(self_filed::INT)
           FROM read_parquet(?) GROUP BY 1 ORDER BY min(requested_at_utc)""",
        [str(out)],
    ):
        print(f"  {split:<11}{100 * h:>8.2f}%{100 * a:>8.2f}%{100 * sf:>8.2f}%")
    print(
        f"{n:,} rows -> {out} ({out.stat().st_size / 1e6:.0f} MB, {time.perf_counter() - t0:.1f}s)"
    )
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the audit feature table.")
    parser.add_argument("--dry-run", action="store_true", help="print sizes, write nothing")
    parser.add_argument("--design", type=Path, default=Path("configs/audit_preregistration.yaml"))
    parser.add_argument("--owner-aliases", type=Path, default=DEFAULT_OWNER_ALIASES)
    parser.add_argument("--reporter-aliases", type=Path, default=DEFAULT_REPORTER_ALIASES)
    parser.add_argument("--scoped-requests", type=Path, default=DEFAULT_SCOPED_REQUESTS)
    parser.add_argument("--scoped-domains", type=Path, default=DEFAULT_SCOPED_DOMAINS)
    parser.add_argument("--domains", type=Path, default=DEFAULT_DOMAINS)
    parser.add_argument("--requests", type=Path, default=DEFAULT_REQUESTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--memory-limit", default="14GB")
    args = parser.parse_args()
    build_features(
        load_preregistration(args.design),
        load_aliases(args.owner_aliases, "owner"),
        load_aliases(args.reporter_aliases, "reporter"),
        args.scoped_requests,
        args.scoped_domains,
        args.domains,
        args.requests,
        args.out,
        dry_run=args.dry_run,
        memory_limit=args.memory_limit,
    )


if __name__ == "__main__":
    main()
