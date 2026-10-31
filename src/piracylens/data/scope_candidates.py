"""Owner scope candidates from pre-registered parameters.

The parameter file fixes every threshold, reporter list and name prefix before
any candidate list is produced. This module loads that file under a strict
schema and provides the matching helpers used by the candidate queries.

Usage (from the repo root):
    python -m piracylens.data.scope_candidates agencies
    python -m piracylens.data.scope_candidates candidates
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import duckdb
import yaml

from piracylens.data.aliases import AliasRow, load_aliases, missing_keys, to_lookup
from piracylens.data.owners import normalize_owner

Mode = Literal["prefix", "word"]
MODES: tuple[Mode, ...] = ("prefix", "word")

_TOP = frozenset({"window", "sources", "seed_owner_keys"})
_SOURCES = frozenset({"ranking", "protection_network", "film_tv_agencies", "prefix_scan"})


class ParamsError(ValueError):
    """Raised when the candidate parameter file violates the schema."""


@dataclass(frozen=True)
class Params:
    """Validated candidate generation parameters."""

    window_start: dt.date
    ranking_min_requests: int
    network_reporters: tuple[str, ...]
    network_min_owner_requests: int
    agency_min_reporter_requests: int
    agency_seed_share: float
    agency_min_owner_requests: int
    agency_selected: tuple[str, ...]
    prefix_min_requests: int
    prefixes: tuple[tuple[str, Mode], ...]
    seed_owner_keys: tuple[str, ...]


def load_params(path: str | Path) -> Params:
    """Read and validate a parameter file."""
    with Path(path).open(encoding="utf-8") as f:
        return parse_params(yaml.safe_load(f))


def parse_params(data: Any) -> Params:
    """Validate an already parsed parameter document."""
    top = _mapping(data, _TOP, "top level")
    window = _mapping(top["window"], {"start"}, "window")
    sources = _mapping(top["sources"], _SOURCES, "sources")
    ranking = _mapping(sources["ranking"], {"min_requests"}, "sources.ranking")
    network = _mapping(
        sources["protection_network"],
        {"reporters", "min_owner_requests"},
        "sources.protection_network",
    )
    agencies = _mapping(
        sources["film_tv_agencies"],
        {"min_reporter_requests", "seed_share", "min_owner_requests", "selected"},
        "sources.film_tv_agencies",
    )
    scan = _mapping(
        sources["prefix_scan"], {"min_requests", "prefix", "word"}, "sources.prefix_scan"
    )

    prefixes: list[tuple[str, Mode]] = []
    seen: dict[str, Mode] = {}
    for mode in MODES:
        for prefix in _key_list(scan, mode, "sources.prefix_scan"):
            if prefix in seen:
                raise ParamsError(
                    f"prefix {prefix!r} is listed under both {seen[prefix]!r} and {mode!r}"
                )
            seen[prefix] = mode
            prefixes.append((prefix, mode))

    w_net = "sources.protection_network"
    w_agency = "sources.film_tv_agencies"
    return Params(
        window_start=_iso_date(window, "start", "window"),
        ranking_min_requests=_positive_int(ranking, "min_requests", "sources.ranking"),
        network_reporters=_key_list(network, "reporters", w_net),
        network_min_owner_requests=_positive_int(network, "min_owner_requests", w_net),
        agency_min_reporter_requests=_positive_int(agencies, "min_reporter_requests", w_agency),
        agency_seed_share=_share(agencies, "seed_share", w_agency),
        agency_min_owner_requests=_positive_int(agencies, "min_owner_requests", w_agency),
        agency_selected=_key_list(agencies, "selected", w_agency, allow_empty=True),
        prefix_min_requests=_positive_int(scan, "min_requests", "sources.prefix_scan"),
        prefixes=tuple(prefixes),
        seed_owner_keys=_key_list(top, "seed_owner_keys", ""),
    )


def matches_prefix(key: str, prefix: str, mode: Mode) -> bool:
    """Return True if key starts with prefix under the given mode.

    "prefix" is a plain startswith. "word" also requires that the prefix is not
    followed by a letter or digit, so "sts" matches "sts" and "sts tv" but not "stsoft".
    """
    if mode not in MODES:
        raise ValueError(f"unknown prefix mode: {mode!r}")
    if not key.startswith(prefix):
        return False
    rest = key[len(prefix) :]
    return mode == "prefix" or not rest or not rest[0].isalnum()


def find_prefix_matches(key: str, prefixes: Iterable[tuple[str, Mode]]) -> list[tuple[str, Mode]]:
    """Return every (prefix, mode) pair that matches the key."""
    return [(p, m) for p, m in prefixes if matches_prefix(key, p, m)]


def resolve_reporter(key: str, lookup: Mapping[str, AliasRow]) -> str:
    """Return the reporter group for a normalized reporter key, or the key itself."""
    row = lookup.get(key)
    return row.group if row is not None else key


def _name(where: str, field: str) -> str:
    return f"{where}.{field}" if where else field


def _mapping(obj: Any, required: set[str] | frozenset[str], where: str) -> Mapping[str, Any]:
    if not isinstance(obj, dict):
        raise ParamsError(f"{where} must be a mapping")
    keys = set(obj)
    missing = sorted(required - keys)
    extra = sorted(str(k) for k in keys - required)
    if missing or extra:
        raise ParamsError(f"{where}: missing fields {missing}, unexpected fields {extra}")
    return obj


def _positive_int(obj: Mapping[str, Any], field: str, where: str) -> int:
    value = obj[field]
    if type(value) is not int or value <= 0:
        raise ParamsError(f"{_name(where, field)} must be a positive integer, got {value!r}")
    return value


def _share(obj: Mapping[str, Any], field: str, where: str) -> float:
    value = obj[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
        raise ParamsError(f"{_name(where, field)} must be a number in (0, 1], got {value!r}")
    return float(value)


def _iso_date(obj: Mapping[str, Any], field: str, where: str) -> dt.date:
    value = obj[field]
    name = _name(where, field)
    if not isinstance(value, str):
        raise ParamsError(f"{name} must be a quoted ISO date string, got {value!r}")
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ParamsError(f"{name} is not an ISO date: {value!r}") from exc


def _key_list(
    obj: Mapping[str, Any], field: str, where: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    value = obj[field]
    name = _name(where, field)
    if not isinstance(value, list):
        raise ParamsError(f"{name} must be a list")
    if not value and not allow_empty:
        raise ParamsError(f"{name} must be a non-empty list")
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item:
            raise ParamsError(
                f"{name}: every entry must be a non-empty quoted string, got {item!r}"
            )
        if item != item.strip() or item != item.casefold():
            raise ParamsError(f"{name}: {item!r} must be stripped and casefolded")
        if item in seen:
            raise ParamsError(f"{name}: duplicate entry {item!r}")
        seen.add(item)
    return tuple(value)


# --- Shared preparation for both modes --------------------------------------------------

DEFAULT_PARAMS = Path("configs/scope_candidates.yaml")
DEFAULT_REPORTER_ALIASES = Path("configs/reporter_aliases.yaml")
DEFAULT_REQUESTS = Path("data/processed/requests.parquet")
DEFAULT_OWNER_KEYS = Path("data/interim/owner_name_keys.parquet")
DEFAULT_AGENCIES_OUT = Path("data/interim/scope_agencies.csv")
DEFAULT_CANDIDATES_OUT = Path("data/interim/scope_candidates.csv")


def absent_keys(keys: Iterable[str], known: Iterable[str]) -> list[str]:
    """Return the keys that are not in the known set, sorted."""
    known_set = set(known)
    return sorted(k for k in set(keys) if k not in known_set)


def unknown_reporters(
    reporters: Iterable[str], lookup: Mapping[str, AliasRow], reporter_keys: Iterable[str]
) -> list[str]:
    """Return reporters that are neither an alias group nor a normalized key in the data."""
    groups = {row.group for row in lookup.values()}
    return absent_keys(reporters, groups | set(reporter_keys))


def format_prefix_matches(matches: Iterable[tuple[str, Mode]]) -> str:
    """Render prefix matches as "prefix:mode" items joined by semicolons."""
    return ";".join(f"{p}:{m}" for p, m in matches)


def _scalar(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None):
    """Run a single-value query and read its result completely (DuckDB 2.0 rule)."""
    return con.execute(sql, params or []).fetchall()[0][0]


@dataclass(frozen=True)
class _Prepared:
    """State shared by both modes after the input checks have passed."""

    params: Params
    con: duckdb.DuckDBPyConnection
    start: str
    n_names: int
    n_empty: int
    reporter_keys: frozenset[str]
    t0: float


def _prepare(
    params_path: Path,
    aliases_path: Path,
    requests: Path,
    owner_keys: Path,
    tsv: Path,
    threads: int,
    memory_limit: str,
    require_selected: bool,
) -> _Prepared:
    """Check inputs, normalize reporter names and register the `reporters` table.

    Stops before any query output is produced if an input file is missing, if a seed
    owner key is not in the owner key mapping, or if a reporter alias key or
    protection-network reporter is not in the data.
    """
    for path in (params_path, aliases_path, requests, owner_keys):
        if not path.is_file():
            raise SystemExit(f"input not found: {path}")
    params = load_params(params_path)
    if require_selected and not params.agency_selected:
        raise SystemExit(
            "sources.film_tv_agencies.selected is empty: record the agency decision first"
        )
    alias_rows = load_aliases(aliases_path, "reporter")
    lookup = to_lookup(alias_rows)
    start = params.window_start.isoformat()
    tsv.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET threads = {threads}").fetchall()
    con.execute(f"SET memory_limit = '{memory_limit}'").fetchall()
    t0 = time.perf_counter()

    # 1. Seed owner keys must exist in the owner key mapping.
    owner_key_set = {
        row[0]
        for row in con.execute(
            f"SELECT DISTINCT owner_key FROM read_parquet('{owner_keys}')"
        ).fetchall()
    }
    missing_seed = absent_keys(params.seed_owner_keys, owner_key_set)
    if missing_seed:
        raise SystemExit(f"seed owner keys missing from the mapping: {missing_seed}")

    # 2. Normalize every distinct reporter name in the window and resolve its group.
    con.execute(
        f"""
        CREATE TABLE names AS
        SELECT row_number() OVER (ORDER BY reporter_name) AS name_id, reporter_name
        FROM (SELECT DISTINCT reporter_name FROM read_parquet('{requests}')
              WHERE requested_at_utc >= CAST(? AS TIMESTAMP) AND reporter_name IS NOT NULL)
        """,
        [start],
    ).fetchall()
    names = con.execute("SELECT name_id, reporter_name FROM names").fetchall()
    reporter_keys: set[str] = set()
    n_empty = 0
    with tsv.open("w", encoding="utf-8") as f:
        f.write("name_id\treporter_key\treporter_group\n")
        for name_id, name in names:
            key = normalize_owner(name)
            if not key:
                n_empty += 1
                continue
            group = resolve_reporter(key, lookup)
            for value in (key, group):
                # Plain TSV is safe only because keys never contain these characters.
                if "\t" in value or "\n" in value or '"' in value:
                    raise SystemExit(f"unsafe key for TSV: {value!r} (name_id {name_id})")
            reporter_keys.add(key)
            f.write(f"{name_id}\t{key}\t{group}\n")

    # 3. Alias keys and protection-network reporters must exist in the data.
    missing_alias = missing_keys(alias_rows, reporter_keys)
    if missing_alias:
        tsv.unlink()
        raise SystemExit(f"reporter alias keys missing from the data: {missing_alias}")
    missing_network = unknown_reporters(params.network_reporters, lookup, reporter_keys)
    if missing_network:
        tsv.unlink()
        raise SystemExit(f"protection-network reporters missing from the data: {missing_network}")

    con.execute(
        f"""
        CREATE TABLE reporters AS
        SELECT n.reporter_name, r.reporter_key, r.reporter_group
        FROM read_csv('{tsv}', delim = '\t', header = true,
                      columns = {{'name_id': 'BIGINT', 'reporter_key': 'VARCHAR',
                                  'reporter_group': 'VARCHAR'}}) AS r
        JOIN names AS n USING (name_id)
        """
    ).fetchall()
    tsv.unlink()
    return _Prepared(params, con, start, len(names), n_empty, frozenset(reporter_keys), t0)


def _create_agency_table(prep: _Prepared, requests: Path, owner_keys: Path) -> None:
    """Create the `agencies` table: volume and seed share per reporter group."""
    prep.con.execute(
        f"""
        CREATE TABLE agencies AS
        WITH req AS (
            SELECT q.request_id, q.owner_name, p.reporter_group, p.reporter_key
            FROM read_parquet('{requests}') AS q
            JOIN reporters AS p USING (reporter_name)
            WHERE q.requested_at_utc >= CAST(? AS TIMESTAMP)
        ),
        seeded AS (
            SELECT DISTINCT req.request_id
            FROM req JOIN read_parquet('{owner_keys}') AS k USING (owner_name)
            WHERE list_contains(?::VARCHAR[], k.owner_key)
        )
        SELECT req.reporter_group,
               count(*) AS requests,
               count(s.request_id) AS seed_requests,
               round(count(s.request_id) / count(*), 4) AS seed_share,
               count(DISTINCT req.reporter_key) AS reporter_keys
        FROM req LEFT JOIN seeded AS s USING (request_id)
        GROUP BY req.reporter_group
        HAVING count(*) >= ?
        """,
        [
            prep.start,
            list(prep.params.seed_owner_keys),
            prep.params.agency_min_reporter_requests,
        ],
    ).fetchall()


def _replace_checked(con: duckdb.DuckDBPyConnection, tmp: Path, out: Path, expected: int) -> None:
    """Move tmp to out only if the CSV has the expected number of data rows."""
    n_csv = _scalar(con, f"SELECT count(*) FROM read_csv('{tmp}', header = true)")
    if n_csv != expected:
        tmp.unlink()
        raise SystemExit(f"row count mismatch: expected {expected}, csv has {n_csv}")
    tmp.replace(out)


# --- Agency table (first stage of the film/TV agency source) -------------------------


def build_agency_table(
    params_path: Path,
    aliases_path: Path,
    requests: Path,
    owner_keys: Path,
    out: Path,
    threads: int = 16,
    memory_limit: str = "14GB",
) -> None:
    """Write one row per reporter group above the agency volume threshold.

    The table holds only reporter-level volumes and the seed share. It contains no
    per-owner results, so the film/TV decision on agencies is made before any owner
    candidate is seen.
    """
    tsv = out.parent / f"{out.stem}.reporters.tsv.tmp"
    tmp = out.parent / f"{out.stem}.csv.tmp"
    prep = _prepare(
        params_path, aliases_path, requests, owner_keys, tsv, threads, memory_limit, False
    )
    con, params = prep.con, prep.params
    total = _scalar(
        con,
        f"SELECT count(*) FROM read_parquet('{requests}') "
        "WHERE requested_at_utc >= CAST(? AS TIMESTAMP)",
        [prep.start],
    )
    _create_agency_table(prep, requests, owner_keys)
    con.execute(
        f"""
        COPY (
            SELECT reporter_group, requests, seed_requests, seed_share,
                   seed_share >= ? AS seed_flag, reporter_keys
            FROM agencies ORDER BY requests DESC, reporter_group
        ) TO '{tmp}' (FORMAT csv, HEADER true)
        """,
        [params.agency_seed_share],
    ).fetchall()

    rows = con.execute(
        "SELECT reporter_group, requests, seed_share FROM agencies "
        "ORDER BY requests DESC, reporter_group"
    ).fetchall()
    covered = sum(r[1] for r in rows)
    _replace_checked(con, tmp, out, len(rows))

    print(
        f"reporter names {prep.n_names:,} | empty key {prep.n_empty:,} | reporter keys "
        f"{len(prep.reporter_keys):,} | requests {total:,} | groups >= "
        f"{params.agency_min_reporter_requests:,}: {len(rows)} ({covered / total:.1%}) | "
        f"{time.perf_counter() - prep.t0:.1f}s -> {out}"
    )
    print("checks: seed keys, reporter alias keys, network reporters all present")
    for group, requests_n, share in rows:
        flag = "*" if share >= params.agency_seed_share else " "
        print(f"{flag} {group[:48]:48s} {requests_n:>11,} {share:7.1%}")


# --- Owner candidates (all four sources) ---------------------------------------------

CANDIDATE_COLUMNS = (
    "owner_key",
    "requests",
    "network_requests",
    "agency_requests",
    "first_year_all",
    "last_year_all",
    "variants",
    "top_raw_name",
    "source_a",
    "source_b",
    "source_c",
    "source_d",
    "prefixes",
    "top_reporter_group",
    "top_reporter_share",
    "in_seed",
)


def build_candidates(
    params_path: Path,
    aliases_path: Path,
    requests: Path,
    owner_keys: Path,
    out: Path,
    threads: int = 16,
    memory_limit: str = "14GB",
) -> None:
    """Write one row per owner key caught by any source, plus every seed key.

    Sources, all counted on requests from the window start onward:
      a) the owner key's request volume reaches the ranking threshold;
      b) requests filed through the protection-network reporters reach their threshold;
      c) requests filed through the hand-picked film/TV agencies reach their threshold;
      d) the key matches a pre-registered prefix and reaches the prefix-scan threshold.

    A request whose owner name maps to several keys counts once for each key. Seed keys
    stay in the table even when no source catches them, so seed recall can be measured.
    `first_year_all` and `last_year_all` use the whole data range; every other count
    uses the window. The table carries volumes and evidence only: no label or
    no-action column, so the scope review cannot be steered by outcomes.
    """
    tsv = out.parent / f"{out.stem}.reporters.tsv.tmp"
    tmp = out.parent / f"{out.stem}.csv.tmp"
    prep = _prepare(
        params_path, aliases_path, requests, owner_keys, tsv, threads, memory_limit, True
    )
    con, params = prep.con, prep.params

    # 1. Every selected agency must be a reporter group above the agency threshold.
    _create_agency_table(prep, requests, owner_keys)
    agency_groups = {
        row[0] for row in con.execute("SELECT reporter_group FROM agencies").fetchall()
    }
    not_agencies = absent_keys(params.agency_selected, agency_groups)
    if not_agencies:
        raise SystemExit(f"selected agencies not in the agency table: {not_agencies}")

    # 2. Per owner key: volume, volume through each reporter set, name variants.
    # A view, not a table: the joined window has tens of millions of rows, and each
    # query below aggregates it in a streaming scan instead of holding it in memory.
    # The window start is a validated date, so writing it into the SQL text is safe
    # (views cannot hold query parameters).
    con.execute(
        f"""
        CREATE VIEW owner_req AS
        SELECT k.owner_key, q.owner_name, p.reporter_group, p.reporter_key
        FROM read_parquet('{requests}') AS q
        JOIN read_parquet('{owner_keys}') AS k USING (owner_name)
        LEFT JOIN reporters AS p USING (reporter_name)
        WHERE q.requested_at_utc >= TIMESTAMP '{params.window_start.isoformat()}'
        """
    ).fetchall()
    con.execute(
        """
        CREATE TABLE owners AS
        SELECT owner_key,
               count(*) AS requests,
               count(*) FILTER (WHERE list_contains(?::VARCHAR[], reporter_group)
                                   OR list_contains(?::VARCHAR[], reporter_key))
                   AS network_requests,
               count(*) FILTER (WHERE list_contains(?::VARCHAR[], reporter_group))
                   AS agency_requests,
               count(DISTINCT owner_name) AS variants
        FROM owner_req
        GROUP BY owner_key
        """,
        [
            list(params.network_reporters),
            list(params.network_reporters),
            list(params.agency_selected),
        ],
    ).fetchall()

    # 3. Prefix scan in Python, so matching follows `matches_prefix` exactly.
    scan_keys = con.execute(
        "SELECT owner_key FROM owners WHERE requests >= ?", [params.prefix_min_requests]
    ).fetchall()
    all_prefixes = tuple(p for p, _ in params.prefixes)
    hit_keys: list[str] = []
    hit_values: list[str] = []
    for (key,) in scan_keys:
        if key.startswith(all_prefixes):
            matches = find_prefix_matches(key, params.prefixes)
            if matches:
                hit_keys.append(key)
                hit_values.append(format_prefix_matches(matches))
    con.execute(
        "CREATE TABLE prefix_hits AS "
        "SELECT unnest(?::VARCHAR[]) AS owner_key, unnest(?::VARCHAR[]) AS prefixes",
        [hit_keys, hit_values],
    ).fetchall()

    # 4. Candidate set: any source flag, plus every seed key.
    con.execute(
        """
        CREATE TABLE flagged AS
        SELECT o.owner_key,
               o.requests >= ? AS source_a,
               o.network_requests >= ? AS source_b,
               o.agency_requests >= ? AS source_c,
               h.owner_key IS NOT NULL AS source_d,
               h.prefixes
        FROM owners AS o LEFT JOIN prefix_hits AS h USING (owner_key)
        """,
        [
            params.ranking_min_requests,
            params.network_min_owner_requests,
            params.agency_min_owner_requests,
        ],
    ).fetchall()
    con.execute(
        """
        CREATE TABLE cand AS
        SELECT owner_key FROM flagged WHERE source_a OR source_b OR source_c OR source_d
        UNION
        SELECT unnest(?::VARCHAR[]) AS owner_key
        """,
        [list(params.seed_owner_keys)],
    ).fetchall()

    # 5. Evidence columns for the candidates only.
    con.execute(
        f"""
        CREATE TABLE years AS
        SELECT k.owner_key,
               min(year(q.requested_at_utc)) AS first_year_all,
               max(year(q.requested_at_utc)) AS last_year_all
        FROM read_parquet('{requests}') AS q
        JOIN read_parquet('{owner_keys}') AS k USING (owner_name)
        JOIN cand USING (owner_key)
        GROUP BY k.owner_key
        """
    ).fetchall()
    con.execute(
        """
        CREATE TABLE top_name AS
        SELECT owner_key, owner_name AS top_raw_name FROM (
            SELECT owner_key, owner_name,
                   row_number() OVER (PARTITION BY owner_key
                                      ORDER BY count(*) DESC, owner_name) AS rn
            FROM owner_req JOIN cand USING (owner_key)
            GROUP BY owner_key, owner_name
        ) WHERE rn = 1
        """
    ).fetchall()
    con.execute(
        """
        CREATE TABLE top_reporter AS
        SELECT owner_key, reporter_group AS top_reporter_group, n FROM (
            SELECT owner_key, reporter_group, count(*) AS n,
                   row_number() OVER (PARTITION BY owner_key
                                      ORDER BY count(*) DESC, reporter_group) AS rn
            FROM owner_req JOIN cand USING (owner_key)
            WHERE reporter_group IS NOT NULL
            GROUP BY owner_key, reporter_group
        ) WHERE rn = 1
        """
    ).fetchall()
    con.execute(
        """
        CREATE TABLE result AS
        SELECT c.owner_key,
               coalesce(o.requests, 0) AS requests,
               coalesce(o.network_requests, 0) AS network_requests,
               coalesce(o.agency_requests, 0) AS agency_requests,
               y.first_year_all,
               y.last_year_all,
               coalesce(o.variants, 0) AS variants,
               t.top_raw_name,
               coalesce(f.source_a, false) AS source_a,
               coalesce(f.source_b, false) AS source_b,
               coalesce(f.source_c, false) AS source_c,
               coalesce(f.source_d, false) AS source_d,
               f.prefixes,
               r.top_reporter_group,
               round(r.n / o.requests, 4) AS top_reporter_share,
               list_contains(?::VARCHAR[], c.owner_key) AS in_seed
        FROM cand AS c
        LEFT JOIN owners AS o USING (owner_key)
        LEFT JOIN flagged AS f USING (owner_key)
        LEFT JOIN years AS y USING (owner_key)
        LEFT JOIN top_name AS t USING (owner_key)
        LEFT JOIN top_reporter AS r USING (owner_key)
        """,
        [list(params.seed_owner_keys)],
    ).fetchall()
    columns = tuple(row[0] for row in con.execute("DESCRIBE result").fetchall())
    if columns != CANDIDATE_COLUMNS:
        raise SystemExit(f"unexpected candidate columns: {columns}")
    con.execute(
        f"""
        COPY (SELECT * FROM result ORDER BY requests DESC, owner_key)
        TO '{tmp}' (FORMAT csv, HEADER true)
        """
    ).fetchall()

    n_rows, n_a, n_b, n_c, n_d, n_seed, n_seed_unflagged, covered = con.execute(
        """
        SELECT count(*),
               count(*) FILTER (WHERE source_a), count(*) FILTER (WHERE source_b),
               count(*) FILTER (WHERE source_c), count(*) FILTER (WHERE source_d),
               count(*) FILTER (WHERE in_seed),
               count(*) FILTER (WHERE in_seed AND NOT (source_a OR source_b
                                                       OR source_c OR source_d)),
               sum(requests)
        FROM result
        """
    ).fetchall()[0]
    _replace_checked(con, tmp, out, n_rows)
    print(
        f"reporter names {prep.n_names:,} | owner keys {_scalar(con, 'SELECT count(*) FROM owners'):,}"
        f" | candidates {n_rows:,} | {time.perf_counter() - prep.t0:.1f}s -> {out}"
    )
    print(
        "checks: selected not empty, seed keys, reporter alias keys, network reporters, "
        "selected agencies all present"
    )
    print(
        f"source a {n_a:,} | b {n_b:,} | c {n_c:,} | d {n_d:,} | seed keys {n_seed:,} "
        f"(no source: {n_seed_unflagged:,}) | candidate requests {covered:,} "
        "(multi-key requests counted once per key)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate owner scope candidates.")
    sub = parser.add_subparsers(dest="mode", required=True)
    commands = {
        "agencies": ("write the reporter-group table for review", DEFAULT_AGENCIES_OUT),
        "candidates": ("write the owner candidate table for review", DEFAULT_CANDIDATES_OUT),
    }
    for name, (help_text, default_out) in commands.items():
        mode = sub.add_parser(name, help=help_text)
        mode.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
        mode.add_argument("--reporter-aliases", type=Path, default=DEFAULT_REPORTER_ALIASES)
        mode.add_argument("--requests", type=Path, default=DEFAULT_REQUESTS)
        mode.add_argument("--owner-keys", type=Path, default=DEFAULT_OWNER_KEYS)
        mode.add_argument("--out", type=Path, default=default_out)
    args = parser.parse_args()
    build = build_agency_table if args.mode == "agencies" else build_candidates
    build(args.params, args.reporter_aliases, args.requests, args.owner_keys, args.out)


if __name__ == "__main__":
    main()
