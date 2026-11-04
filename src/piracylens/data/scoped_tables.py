"""Scoped sub-tables: the film/TV takedown rows every model reads.

Two Parquet files in data/interim/:
  scoped_requests.parquet  one row per in-scope request from the history start onward, with
                           owner group, brand and labels, reporter group, candidate source
                           flags and the four URL counters
  scoped_domains.parquet   the request x domain rows of those requests, with the four counters
                           and the request time; other attributes are joined by request_id

Rows before the window start are feature history only and carry in_window = false. A request
is in scope when its owner name maps to an in-scope key; a request that maps to more than one
in-scope key stops the build. The command prints row counts only: no sums, rates or shares of
the URL counters, so the audit label is not looked at before its pre-registration.

Usage (from the repo root; requests first, domains reads its output):
    python -m piracylens.data.scoped_tables requests
    python -m piracylens.data.scoped_tables domains
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import time
from pathlib import Path

import duckdb

from piracylens.data.aliases import load_aliases, missing_keys
from piracylens.data.scope_candidates import (
    DEFAULT_OWNER_KEYS,
    DEFAULT_PARAMS,
    DEFAULT_REPORTER_ALIASES,
    DEFAULT_REQUESTS,
    _prepare,
    _scalar,
)

# Requests from 2019 are kept as feature history for the start of the window; they are never
# training or test examples.
HISTORY_START = dt.date(2019, 1, 1)

DEFAULT_OWNER_ALIASES = Path("configs/owner_aliases.yaml")
DEFAULT_CANDIDATES = Path("data/interim/scope_candidates.csv")
DEFAULT_METRICS = Path("data/interim/scope_metrics.json")
DEFAULT_DOMAINS = Path("data/processed/domains.parquet")
DEFAULT_REQUESTS_OUT = Path("data/interim/scoped_requests.parquet")
DEFAULT_DOMAINS_OUT = Path("data/interim/scoped_domains.parquet")

COUNTERS = ("urls_removed", "urls_not_in_index", "urls_no_action", "urls_pending")
REQUEST_COLUMNS = (
    "request_id",
    "requested_at_utc",
    "in_window",
    "owner_key",
    "owner_group",
    "brand",
    "market",
    "region",
    "content_type",
    "evidence_basis",
    "reporter_key",
    "reporter_group",
    "reporter_id",
    "source_a",
    "source_b",
    "source_c",
    "source_d",
    "in_seed",
    "multi_owner",
    *COUNTERS,
)
DOMAIN_COLUMNS = ("request_id", "domain", "requested_at_utc", "in_window", *COUNTERS)
_FLAGS = ("source_a", "source_b", "source_c", "source_d", "in_seed")


def _write_parquet(con: duckdb.DuckDBPyConnection, table: str, cols: tuple, out: Path) -> int:
    """Copy a table to Parquet through a temp file; rename only if the row count matches."""
    expected = _scalar(con, f"SELECT count(*) FROM {table}")
    tmp = out.with_name(out.name + ".tmp")
    con.execute(
        f"COPY (SELECT {', '.join(cols)} FROM {table}) TO '{tmp}' "
        "(FORMAT parquet, COMPRESSION zstd)"
    ).fetchall()
    written = _scalar(con, f"SELECT count(*) FROM read_parquet('{tmp}')")
    if written != expected:
        tmp.unlink()
        raise SystemExit(f"row count mismatch: expected {expected}, parquet has {written}")
    tmp.replace(out)
    return written


def _print_counts(con: duckdb.DuckDBPyConnection, table: str, label: str) -> None:
    rows = con.execute(f"SELECT in_window, count(*) FROM {table} GROUP BY 1 ORDER BY 1").fetchall()
    print(" | ".join(f"{label} in_window={w}: {n:,}" for w, n in rows))
    years = con.execute(
        f"SELECT year(requested_at_utc), count(*) FROM {table} GROUP BY 1 ORDER BY 1"
    ).fetchall()
    print("  by year: " + ", ".join(f"{y} {n:,}" for y, n in years))


def build_requests(
    params_path: Path,
    reporter_aliases: Path,
    owner_aliases: Path,
    candidates: Path,
    metrics: Path,
    requests: Path,
    owner_keys: Path,
    out: Path,
    threads: int = 16,
    memory_limit: str = "14GB",
) -> int:
    """Write scoped_requests.parquet and return its row count."""
    for path in (owner_aliases, candidates, metrics):
        if not path.is_file():
            raise SystemExit(f"input not found: {path}")
    out.parent.mkdir(parents=True, exist_ok=True)
    prep = _prepare(
        params_path,
        reporter_aliases,
        requests,
        owner_keys,
        out.with_name(out.name + ".reporters.tsv.tmp"),
        threads,
        memory_limit,
        True,
        HISTORY_START.isoformat(),
    )
    con, params = prep.con, prep.params
    if HISTORY_START >= params.window_start:
        raise SystemExit(f"history start {HISTORY_START} must be before the window start")
    expected_window = json.loads(metrics.read_text(encoding="utf-8"))["scope_share"][
        "in_scope_requests"
    ]

    # 1. In-scope keys with their labels; every key must exist in the mapping.
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
    con.execute(
        "CREATE TABLE scope AS SELECT unnest(?::VARCHAR[]) AS owner_key, "
        "unnest(?::VARCHAR[]) AS owner_group, unnest(?::VARCHAR[]) AS brand, "
        "unnest(?::VARCHAR[]) AS market, unnest(?::VARCHAR[]) AS region, "
        "unnest(?::VARCHAR[]) AS content_type, unnest(?::VARCHAR[]) AS evidence_basis",
        [
            [r.key for r in alias_rows],
            [r.group for r in alias_rows],
            [r.brand for r in alias_rows],
            [r.market for r in alias_rows],
            [r.region for r in alias_rows],
            [r.content_type for r in alias_rows],
            [r.basis for r in alias_rows],
        ],
    ).fetchall()

    # 2. Candidate source flags; every in-scope key must be a candidate.
    with candidates.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        absent = [c for c in ("owner_key", *_FLAGS) if c not in (reader.fieldnames or [])]
        if absent:
            raise SystemExit(f"candidate file lacks columns: {absent}")
        cand_rows = [(r["owner_key"], *(r[c] == "true" for c in _FLAGS)) for r in reader]
    not_candidates = sorted({r.key for r in alias_rows} - {r[0] for r in cand_rows})
    if not_candidates:
        raise SystemExit(f"owner alias keys that are not candidates: {not_candidates[:10]}")
    cols = list(zip(*cand_rows, strict=True))
    con.execute(
        "CREATE TABLE cand AS SELECT unnest(?::VARCHAR[]) AS owner_key, "
        + ", ".join(f"unnest(?::BOOLEAN[]) AS {c}" for c in _FLAGS),
        [list(c) for c in cols],
    ).fetchall()

    # 3. One row per in-scope request. The request columns are part of the grouping, so a
    # request_id that appeared with different values would show up as a duplicate below.
    # Both dates are validated dates, so writing them into the SQL text is safe.
    counters = ", ".join(f"q.{c}" for c in COUNTERS)
    con.execute(
        f"""
        CREATE TABLE per_request AS
        SELECT q.request_id, q.requested_at_utc, q.reporter_id, q.reporter_name, {counters},
               count(DISTINCT k.owner_key) AS n_keys,
               count(DISTINCT s.owner_key) AS n_scope,
               any_value(s.owner_key) AS owner_key
        FROM read_parquet('{requests}') AS q
        LEFT JOIN read_parquet('{owner_keys}') AS k USING (owner_name)
        LEFT JOIN scope AS s ON s.owner_key = k.owner_key
        WHERE q.requested_at_utc >= TIMESTAMP '{HISTORY_START.isoformat()}'
        GROUP BY q.request_id, q.requested_at_utc, q.reporter_id, q.reporter_name, {counters}
        HAVING count(DISTINCT s.owner_key) > 0
        """
    ).fetchall()
    n_rows, n_ids, n_multi_scope = con.execute(
        "SELECT count(*), count(DISTINCT request_id), count(*) FILTER (WHERE n_scope > 1) "
        "FROM per_request"
    ).fetchall()[0]
    if n_rows != n_ids:
        raise SystemExit(f"duplicate request_id in the requests table: {n_rows - n_ids} rows")
    if n_multi_scope:
        raise SystemExit(f"requests mapping to more than one in-scope key: {n_multi_scope}")

    window = params.window_start.isoformat()
    con.execute(
        f"""
        CREATE TABLE scoped AS
        SELECT r.request_id, r.requested_at_utc,
               r.requested_at_utc >= TIMESTAMP '{window}' AS in_window,
               r.owner_key, s.owner_group, s.brand, s.market, s.region, s.content_type,
               s.evidence_basis, p.reporter_key, p.reporter_group, r.reporter_id,
               {", ".join(f"c.{f}" for f in _FLAGS)},
               r.n_keys > 1 AS multi_owner,
               {", ".join(f"r.{c}" for c in COUNTERS)}
        FROM per_request AS r
        JOIN scope AS s USING (owner_key)
        JOIN cand AS c USING (owner_key)
        LEFT JOIN reporters AS p USING (reporter_name)
        ORDER BY r.requested_at_utc, r.request_id
        """
    ).fetchall()
    in_window = _scalar(con, "SELECT count(*) FROM scoped WHERE in_window")
    if in_window != expected_window:
        raise SystemExit(
            f"in-window requests {in_window} differ from the scope metrics ({expected_window})"
        )
    n = _write_parquet(con, "scoped", REQUEST_COLUMNS, out)
    n_multi = _scalar(con, "SELECT count(*) FROM scoped WHERE multi_owner")
    n_groups = _scalar(con, "SELECT count(DISTINCT owner_group) FROM scoped")
    _print_counts(con, "scoped", "requests")
    print(
        f"{n:,} requests, {n_groups} groups, {n_multi:,} also naming an out-of-scope owner"
        f" -> {out} ({out.stat().st_size / 1e6:.0f} MB, {time.perf_counter() - prep.t0:.1f}s)"
    )
    return n


def build_domains(
    scoped_requests: Path,
    domains: Path,
    out: Path,
    threads: int = 16,
    memory_limit: str = "14GB",
) -> int:
    """Write scoped_domains.parquet from the scoped requests and return its row count."""
    for path in (scoped_requests, domains):
        if not path.is_file():
            raise SystemExit(f"input not found: {path}")
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    con = duckdb.connect()
    con.execute(f"SET threads = {threads}").fetchall()
    con.execute(f"SET memory_limit = '{memory_limit}'").fetchall()
    con.execute(f"SET temp_directory = '{out.parent / 'duckdb_tmp'}'").fetchall()
    con.execute(
        f"""
        CREATE VIEW scoped AS
        SELECT d.request_id, d.domain, r.requested_at_utc, r.in_window,
               {", ".join(f"d.{c}" for c in COUNTERS)}
        FROM read_parquet('{domains}') AS d
        JOIN read_parquet('{scoped_requests}') AS r USING (request_id)
        """
    ).fetchall()
    n = _write_parquet(con, "scoped", DOMAIN_COLUMNS, out)
    con.execute(f"CREATE VIEW written AS SELECT * FROM read_parquet('{out}')").fetchall()
    _print_counts(con, "written", "domain rows")
    print(
        f"{n:,} domain rows -> {out} ({out.stat().st_size / 1e6:.0f} MB, "
        f"{time.perf_counter() - t0:.1f}s)"
    )
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the scoped sub-tables.")
    parser.add_argument("mode", choices=("requests", "domains"))
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--reporter-aliases", type=Path, default=DEFAULT_REPORTER_ALIASES)
    parser.add_argument("--owner-aliases", type=Path, default=DEFAULT_OWNER_ALIASES)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--requests", type=Path, default=DEFAULT_REQUESTS)
    parser.add_argument("--domains", type=Path, default=DEFAULT_DOMAINS)
    parser.add_argument("--owner-keys", type=Path, default=DEFAULT_OWNER_KEYS)
    parser.add_argument("--requests-out", type=Path, default=DEFAULT_REQUESTS_OUT)
    parser.add_argument("--domains-out", type=Path, default=DEFAULT_DOMAINS_OUT)
    args = parser.parse_args()
    if args.mode == "requests":
        build_requests(
            args.params,
            args.reporter_aliases,
            args.owner_aliases,
            args.candidates,
            args.metrics,
            args.requests,
            args.owner_keys,
            args.requests_out,
        )
    else:
        build_domains(args.requests_out, args.domains, args.domains_out)


if __name__ == "__main__":
    main()
