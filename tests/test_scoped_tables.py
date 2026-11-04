"""Tests for the scoped sub-tables on synthetic data."""

import csv
import json
from pathlib import Path

import duckdb
import pytest
import yaml

from piracylens.data.scoped_tables import (
    DOMAIN_COLUMNS,
    REQUEST_COLUMNS,
    build_domains,
    build_requests,
)

PARAMS = {
    "window": {"start": "2021-01-01"},
    "sources": {
        "ranking": {"min_requests": 4},
        "protection_network": {"reporters": ["f6"], "min_owner_requests": 2},
        "film_tv_agencies": {
            "min_reporter_requests": 2,
            "seed_share": 0.5,
            "min_owner_requests": 2,
            "selected": ["f6", "agency x"],
        },
        "prefix_scan": {"min_requests": 2, "prefix": ["disney"], "word": ["sts"]},
    },
    "seed_owner_keys": ["vgtrk"],
}
REPORTER_ALIASES = {
    "groups": {
        "f6": {
            "evidence": {"basis": "reporter", "status": "verified"},
            "brands": [
                {"brand": "Group-IB", "keys": ["group-ib"]},
                {"brand": "F6", "keys": ["f6"]},
            ],
        }
    }
}


def _group(region, basis, brand, market, keys):
    return {
        "region": region,
        "content_type": "tv",
        "evidence": {"basis": basis, "status": "carried"},
        "brands": [{"brand": brand, "market": market, "keys": keys}],
    }


OWNER_ALIASES = {
    "groups": {
        "vgtrk": _group("ru", "company_knowledge", "VGTRK", "ru", ["vgtrk"]),
        "studio-y": _group("us", "reporter", "Studio Y", "global", ["studio y"]),
    }
}
REQUESTS = [  # id, time, reporter_id, reporter, owner
    (1, "2018-12-31", 10, "Group-IB", "VGTRK"),  # before the history start
    (2, "2019-05-01", 10, "Group-IB", "VGTRK"),  # history, in scope
    (3, "2021-02-01", 11, "F6", "VGTRK, Other"),  # window, also names an out-of-scope owner
    (4, "2022-01-02", 11, "F6", "Music Co"),  # out of scope
    (5, "2023-01-01", 12, "Agency X", "Studio Y"),  # window, in scope
    (6, "2023-01-03", 0, None, "Studio Y"),  # window, no reporter
    (7, "2024-01-01", 12, "Agency X", "Unmapped"),  # no owner key
]
OWNER_KEYS = [
    ("VGTRK", "vgtrk"),
    ("VGTRK, Other", "vgtrk"),
    ("VGTRK, Other", "other"),
    ("Music Co", "music co"),
    ("Studio Y", "studio y"),
    ("VGTRK and Studio Y", "vgtrk"),
    ("VGTRK and Studio Y", "studio y"),
]
DOMAINS = [(1, "a.example"), (2, "a.example"), (2, "b.example"), (3, "c.example")]
DOMAINS += [(4, "d.example"), (5, "e.example"), (7, "f.example")]


def _write(tmp_path: Path, requests=REQUESTS, window_in_scope: int = 3) -> dict[str, Path]:
    p = {
        name: tmp_path / file
        for name, file in (
            ("params", "params.yaml"),
            ("reporter_aliases", "reporter_aliases.yaml"),
            ("owner_aliases", "owner_aliases.yaml"),
            ("candidates", "candidates.csv"),
            ("metrics", "scope_metrics.json"),
            ("requests", "requests.parquet"),
            ("domains", "domains.parquet"),
            ("owner_keys", "owner_keys.parquet"),
            ("requests_out", "out/scoped_requests.parquet"),
            ("domains_out", "out/scoped_domains.parquet"),
        )
    }
    p["params"].write_text(yaml.safe_dump(PARAMS), encoding="utf-8")
    p["reporter_aliases"].write_text(yaml.safe_dump(REPORTER_ALIASES), encoding="utf-8")
    p["owner_aliases"].write_text(yaml.safe_dump(OWNER_ALIASES), encoding="utf-8")
    p["metrics"].write_text(
        json.dumps({"scope_share": {"in_scope_requests": window_in_scope}}), encoding="utf-8"
    )
    flags = {  # source_a, source_b, source_c, source_d, in_seed
        "vgtrk": ("false", "true", "false", "false", "true"),
        "studio y": ("true", "false", "false", "false", "false"),
        "music co": ("true", "false", "false", "false", "false"),
    }
    with p["candidates"].open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["owner_key", "source_a", "source_b", "source_c", "source_d", "in_seed"])
        for key, values in flags.items():
            w.writerow([key, *values])
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE r (request_id BIGINT, requested_at_utc TIMESTAMP, reporter_id BIGINT, "
        "reporter_name VARCHAR, owner_name VARCHAR, urls_removed BIGINT, "
        "urls_not_in_index BIGINT, urls_no_action BIGINT, urls_pending BIGINT)"
    ).fetchall()
    con.executemany("INSERT INTO r VALUES (?, CAST(? AS TIMESTAMP), ?, ?, ?, 3, 1, 1, 0)", requests)
    con.execute(f"COPY r TO '{p['requests']}' (FORMAT parquet)").fetchall()
    con.execute("CREATE TABLE k (owner_name VARCHAR, owner_key VARCHAR)").fetchall()
    con.executemany("INSERT INTO k VALUES (?, ?)", OWNER_KEYS)
    con.execute(f"COPY k TO '{p['owner_keys']}' (FORMAT parquet)").fetchall()
    con.execute(
        "CREATE TABLE d (request_id BIGINT, domain VARCHAR, urls_removed BIGINT, "
        "urls_not_in_index BIGINT, urls_no_action BIGINT, urls_pending BIGINT)"
    ).fetchall()
    con.executemany("INSERT INTO d VALUES (?, ?, 1, 0, 1, 0)", DOMAINS)
    con.execute(f"COPY d TO '{p['domains']}' (FORMAT parquet)").fetchall()
    return p


def _requests(p: dict[str, Path]) -> int:
    return build_requests(
        p["params"],
        p["reporter_aliases"],
        p["owner_aliases"],
        p["candidates"],
        p["metrics"],
        p["requests"],
        p["owner_keys"],
        p["requests_out"],
        threads=1,
        memory_limit="1GB",
    )


def _rows(path: Path) -> tuple[list[str], list[dict]]:
    rel = duckdb.connect().execute(f"SELECT * FROM read_parquet('{path}')")
    names = [c[0] for c in rel.description]
    return names, [dict(zip(names, row, strict=True)) for row in rel.fetchall()]


def test_scoped_requests_and_domains(tmp_path):
    p = _write(tmp_path)
    assert _requests(p) == 4
    names, rows = _rows(p["requests_out"])
    assert tuple(names) == REQUEST_COLUMNS
    by_id = {r["request_id"]: r for r in rows}
    assert [r["request_id"] for r in rows] == [2, 3, 5, 6]  # ordered by time
    assert [by_id[i]["in_window"] for i in (2, 3, 5, 6)] == [False, True, True, True]
    assert (by_id[2]["owner_group"], by_id[2]["reporter_group"]) == ("vgtrk", "f6")
    assert (by_id[2]["reporter_key"], by_id[3]["reporter_key"]) == ("group-ib", "f6")
    assert by_id[3]["multi_owner"] and not by_id[5]["multi_owner"]
    assert (by_id[5]["brand"], by_id[5]["market"], by_id[5]["region"]) == (
        "Studio Y",
        "global",
        "us",
    )
    assert by_id[5]["reporter_group"] == "agency x"
    assert by_id[6]["reporter_group"] is None and by_id[6]["reporter_id"] == 0
    assert (by_id[2]["source_b"], by_id[2]["in_seed"], by_id[5]["source_a"]) == (True, True, True)
    assert by_id[5]["evidence_basis"] == "reporter" and by_id[5]["urls_removed"] == 3

    assert build_domains(p["requests_out"], p["domains"], p["domains_out"], 1, "1GB") == 4
    names, drows = _rows(p["domains_out"])
    assert tuple(names) == DOMAIN_COLUMNS
    assert sorted((r["request_id"], r["domain"], r["in_window"]) for r in drows) == [
        (2, "a.example", False),
        (2, "b.example", False),
        (3, "c.example", True),
        (5, "e.example", True),
    ]
    assert not any(f.name.endswith(".tmp") for f in p["requests_out"].parent.iterdir())


def test_refuses_window_count_different_from_scope_metrics(tmp_path):
    p = _write(tmp_path, window_in_scope=2)
    with pytest.raises(SystemExit, match="differ from the scope metrics"):
        _requests(p)
    assert not p["requests_out"].exists()


def test_refuses_request_with_two_in_scope_keys(tmp_path):
    p = _write(
        tmp_path, requests=[*REQUESTS, (8, "2025-01-01", 12, "Agency X", "VGTRK and Studio Y")]
    )
    with pytest.raises(SystemExit, match="more than one in-scope key: 1"):
        _requests(p)
    assert not p["requests_out"].exists()
