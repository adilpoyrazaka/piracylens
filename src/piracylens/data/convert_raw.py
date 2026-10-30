"""Convert raw Google Transparency Report CSVs to Parquet with DuckDB.

Usage:
    python src/piracylens/data/convert_raw.py requests [--force]

Guards (assert-before-write):
  * input file size must match the zip listing exactly
  * header row must match the expected header exactly
  * output is written to a .tmp file and renamed only on success
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw" / "google-websearch-copyright-removals"
OUT_DIR = REPO_ROOT / "data" / "processed"
TMP_DIR = REPO_ROOT / "data" / "interim" / "duckdb_tmp"

COUNTS = {
    "urls_removed": "INTEGER",
    "urls_not_in_index": "INTEGER",
    "urls_no_action": "INTEGER",
    "urls_pending": "INTEGER",
}
COUNT_HEADERS = [
    "URLs removed",
    "URLs that were not in Google's search index",
    "URLs for which we took no action",
    "URLs pending review",
]

TABLES = {
    "requests": {
        "file": "requests.csv",
        "bytes": 7_738_066_978,
        "header": ["Request ID", "Date", "Lumen URL", "Copyright owner ID",
                   "Copyright owner name", "Reporting organization ID",
                   "Reporting organization name", *COUNT_HEADERS, "From Abuser"],
        "columns": {"request_id": "BIGINT", "date_raw": "VARCHAR",
                    "lumen_url": "VARCHAR", "owner_id": "BIGINT",
                    "owner_name": "VARCHAR", "reporter_id": "BIGINT",
                    "reporter_name": "VARCHAR", **COUNTS,
                    "from_abuser": "BOOLEAN"},
        # Strict ISO-8601 UTC parse: any malformed date fails the run loudly.
        "select": ("SELECT request_id, "
                   "strptime(date_raw, '%Y-%m-%dT%H:%M:%SZ') AS requested_at_utc, "
                   "* EXCLUDE (request_id, date_raw)"),
    },
    "domains": {
        "file": "domains.csv",
        "bytes": 36_409_779_308,
        "header": ["Request ID", "Domain", *COUNT_HEADERS, "From Abuser"],
        "columns": {"request_id": "BIGINT", "domain": "VARCHAR", **COUNTS,
                    "from_abuser": "BOOLEAN"},
        "select": "SELECT *",
    },
    "urls_no_action": {
        "file": "urls-no-action-taken.csv",
        "bytes": 45_524_457_061,
        "header": ["Request ID", "Domain", "URL", "From Abuser"],
        "columns": {"request_id": "BIGINT", "domain": "VARCHAR",
                    "url": "VARCHAR", "from_abuser": "BOOLEAN"},
        "select": "SELECT *",
    },
}


def fail(msg: str) -> None:
    sys.exit(f"ABORT: {msg}")


def check_input(spec: dict) -> Path:
    src = RAW_DIR / spec["file"]
    if not src.exists():
        fail(f"missing input {src}")
    size = src.stat().st_size
    if size != spec["bytes"]:
        fail(f"{src.name} is {size} bytes, expected {spec['bytes']}")
    with src.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    if header != spec["header"]:
        fail(f"unexpected header in {src.name}:\n  got      {header}\n"
             f"  expected {spec['header']}")
    return src


def convert(name: str, force: bool) -> None:
    spec = TABLES[name]
    src = check_input(spec)
    out = OUT_DIR / f"{name}.parquet"
    tmp = out.with_suffix(".parquet.tmp")
    if out.exists() and not force:
        fail(f"{out} already exists (use --force to overwrite)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp.unlink(missing_ok=True)

    con = duckdb.connect()
    con.execute("SET threads = 16")
    con.execute("SET memory_limit = '14GB'")
    con.execute(f"SET temp_directory = '{TMP_DIR}'")
    con.execute("SET preserve_insertion_order = false")

    cols = ", ".join(f"'{c}': '{t}'" for c, t in spec["columns"].items())
    source = (f"read_csv('{src}', header = true, delim = ',', quote = '\"', "
              f"escape = '\"', columns = {{{cols}}})")

    t0 = time.perf_counter()
    con.execute(f"COPY ({spec['select']} FROM {source}) "
                f"TO '{tmp}' (FORMAT parquet, COMPRESSION zstd)")
    elapsed = time.perf_counter() - t0

    os.replace(tmp, out)
    
    rows = con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchall()[0][0]
    print(f"OK {name}: {rows:,} rows -> {out} "
          f"({out.stat().st_size / 1e9:.2f} GB) in {elapsed / 60:.1f} min")
    summary = con.execute(
        f"SELECT column_name, column_type, min, max, null_percentage "
        f"FROM (SUMMARIZE SELECT * FROM read_parquet('{out}'))").fetchall()
    for col, typ, lo, hi, nulls in summary:
        print(f"  {col:<20} {typ:<10} min={lo!s:<22} max={hi!s:<22} null={nulls}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("table", choices=TABLES)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    convert(a.table, a.force)