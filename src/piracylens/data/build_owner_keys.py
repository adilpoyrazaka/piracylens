"""Build the owner_name -> owner_key mapping used to group rights owners.

Usage (from the repo root):
    python -m piracylens.data.build_owner_keys
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb

from piracylens.data.owners import split_owners

DEFAULT_REQUESTS = Path("data/processed/requests.parquet")
DEFAULT_OUT = Path("data/interim/owner_name_keys.parquet")


def _scalar(con: duckdb.DuckDBPyConnection, sql: str):
    """Run a single-value query and read its result completely (DuckDB 2.0 rule)."""
    return con.execute(sql).fetchall()[0][0]


def build(requests: Path, out: Path, threads: int = 16, memory_limit: str = "14GB") -> None:
    """Normalize every distinct owner_name and write one row per (owner_name, owner_key)."""
    if not requests.is_file():
        raise SystemExit(f"input not found: {requests}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tsv = out.parent / f"{out.stem}.tsv.tmp"
    tmp = out.parent / f"{out.stem}.parquet.tmp"

    con = duckdb.connect()
    con.execute(f"SET threads = {threads}").fetchall()
    con.execute(f"SET memory_limit = '{memory_limit}'").fetchall()

    t0 = time.perf_counter()
    con.execute(f"""
        CREATE TABLE names AS
        SELECT row_number() OVER (ORDER BY owner_name) AS name_id, owner_name
        FROM (SELECT DISTINCT owner_name FROM read_parquet('{requests}')
              WHERE owner_name IS NOT NULL)
    """).fetchall()
    rows = con.execute("SELECT name_id, owner_name FROM names").fetchall()

    n_written = 0
    n_without_key = 0
    with tsv.open("w", encoding="utf-8") as f:
        f.write("name_id\towner_key\n")
        for name_id, name in rows:
            keys = split_owners(name)
            if not keys:
                n_without_key += 1
            for key in keys:
                # Plain TSV is safe only because keys never contain these characters.
                if "\t" in key or "\n" in key or '"' in key:
                    raise SystemExit(f"unsafe key for TSV: {key!r} (name_id {name_id})")
                f.write(f"{name_id}\t{key}\n")
                n_written += 1

    con.execute(f"""
        COPY (
            SELECT n.owner_name, k.owner_key
            FROM read_csv('{tsv}', delim = '\t', header = true,
                          columns = {{'name_id': 'BIGINT', 'owner_key': 'VARCHAR'}}) AS k
            JOIN names AS n USING (name_id)
            ORDER BY n.owner_name, k.owner_key
        ) TO '{tmp}' (FORMAT parquet, COMPRESSION zstd)
    """).fetchall()

    n_rows = _scalar(con, f"SELECT count(*) FROM read_parquet('{tmp}')")
    n_keys = _scalar(con, f"SELECT count(DISTINCT owner_key) FROM read_parquet('{tmp}')")
    if n_rows != n_written:
        tmp.unlink()
        raise SystemExit(f"row count mismatch: wrote {n_written}, parquet has {n_rows}")

    tmp.replace(out)
    tsv.unlink()
    print(
        f"names {len(rows):,} | without key {n_without_key:,} | rows {n_rows:,} | "
        f"keys {n_keys:,} | {time.perf_counter() - t0:.1f}s -> {out}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--requests", type=Path, default=DEFAULT_REQUESTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build(args.requests, args.out)


if __name__ == "__main__":
    main()
