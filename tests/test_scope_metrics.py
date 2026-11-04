"""Tests for scope coverage metrics on synthetic data."""

import csv
import json
from pathlib import Path

import duckdb
import pytest
import yaml

from piracylens.data.scope_metrics import KEY_COLUMNS, build_metrics


def _params() -> dict:
    return {
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


def _owner_aliases() -> dict:
    def group(region, basis, brand, market, keys):
        return {
            "region": region,
            "content_type": "tv",
            "evidence": {"basis": basis, "status": "carried"},
            "brands": [{"brand": brand, "market": market, "keys": keys}],
        }

    return {
        "groups": {
            "vgtrk": group("ru", "company_knowledge", "VGTRK", "ru", ["vgtrk"]),
            "studio-y": group("us", "reporter", "Studio Y", "global", ["studio y"]),
        }
    }


def _write(tmp_path: Path, candidate_keys: list[str]) -> dict[str, Path]:
    paths = {
        "params": tmp_path / "params.yaml",
        "reporter_aliases": tmp_path / "reporter_aliases.yaml",
        "owner_aliases": tmp_path / "owner_aliases.yaml",
        "candidates": tmp_path / "candidates.csv",
        "requests": tmp_path / "requests.parquet",
        "owner_keys": tmp_path / "owner_keys.parquet",
        "out_dir": tmp_path / "out",
    }
    paths["params"].write_text(yaml.safe_dump(_params()), encoding="utf-8")
    paths["reporter_aliases"].write_text(
        yaml.safe_dump(
            {
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
        ),
        encoding="utf-8",
    )
    paths["owner_aliases"].write_text(yaml.safe_dump(_owner_aliases()), encoding="utf-8")
    flags = {  # source_a, source_b, source_c, source_d, in_seed
        "vgtrk": (False, True, False, False, True),
        "studio y": (True, False, False, False, False),
        "music co": (True, False, False, False, False),
    }
    with paths["candidates"].open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["owner_key", "requests", "source_a", "source_b", "source_c", "source_d", "in_seed"]
        )
        for key in candidate_keys:
            w.writerow([key, 0, *("true" if v else "false" for v in flags[key])])
    requests = [
        (1, "2019-05-01", "Group-IB", "VGTRK"),  # before the window
        (2, "2021-02-01", "Group-IB", "VGTRK"),  # network, in scope
        (3, "2022-01-01", "F6", "VGTRK, Other"),  # network, two keys, counts once
        (4, "2022-01-02", "F6", "Music Co"),  # network, reviewed and left out
        (5, "2022-01-03", "F6", "Tiny"),  # network, never a candidate
        (6, "2022-01-04", "F6", "Unmapped"),  # network, no owner key
        (7, "2023-01-01", "Agency X", "Studio Y"),  # agency, in scope
        (8, "2023-01-02", "Agency X", "Music Co"),  # agency, reviewed and left out
        (9, "2023-01-03", None, "Studio Y"),  # no reporter, in scope
        (10, "2024-01-01", "Small Rep", "Mid Owner"),
        (11, "2024-01-02", "Small Rep", "Mid Owner"),
        (12, "2024-01-03", "Small Rep", "Mid Owner"),
    ]
    owner_keys = [
        ("VGTRK", "vgtrk"),
        ("VGTRK, Other", "vgtrk"),
        ("VGTRK, Other", "other"),
        ("Music Co", "music co"),
        ("Tiny", "tiny"),
        ("Studio Y", "studio y"),
        ("Mid Owner", "mid owner"),
    ]
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE r (request_id INTEGER, requested_at_utc TIMESTAMP, "
        "reporter_name VARCHAR, owner_name VARCHAR)"
    ).fetchall()
    con.executemany("INSERT INTO r VALUES (?, CAST(? AS TIMESTAMP), ?, ?)", requests)
    con.execute(f"COPY r TO '{paths['requests']}' (FORMAT parquet)").fetchall()
    con.execute("CREATE TABLE k (owner_name VARCHAR, owner_key VARCHAR)").fetchall()
    con.executemany("INSERT INTO k VALUES (?, ?)", owner_keys)
    con.execute(f"COPY k TO '{paths['owner_keys']}' (FORMAT parquet)").fetchall()
    return paths


def _run(paths: dict[str, Path]) -> dict:
    return build_metrics(
        paths["params"],
        paths["reporter_aliases"],
        paths["owner_aliases"],
        paths["candidates"],
        paths["requests"],
        paths["owner_keys"],
        paths["out_dir"],
        alt_ranking_min=2,
        threads=1,
        memory_limit="1GB",
    )


def test_build_metrics_on_synthetic_data(tmp_path):
    paths = _write(tmp_path, ["vgtrk", "studio y", "music co"])
    m = _run(paths)
    assert m["window_requests"] == 11
    assert m["scope_share"]["in_scope_requests"] == 4
    assert m["scope_share"]["share"] == pytest.approx(4 / 11, abs=1e-6)
    net = m["network_recall"]
    assert (net["requests"], net["in_scope"], net["recall"]) == (5, 2, 0.4)
    assert net["gap"] == {"reviewed_out": 1, "never_candidate": 1, "no_owner_key": 1}
    agency = m["agency_recall"]
    assert agency["reporters"] == ["agency x"]
    assert (agency["requests"], agency["in_scope"]) == (2, 1)
    assert agency["gap"] == {"reviewed_out": 1, "never_candidate": 0, "no_owner_key": 0}
    b = m["breakdowns"]
    assert b["vgtrk_share"] == 0.5
    assert {k: v["requests"] for k, v in b["market"].items()} == {"global": 2, "ru": 2}
    assert {k: v["requests"] for k, v in b["source"].items()} == {
        "network_or_agency_only": 2,
        "ranking_or_prefix": 2,
    }
    check = m["threshold_check"]
    assert (check["added_unreviewed_keys"], check["added_requests_upper_bound"]) == (1, 3)

    assert json.loads((paths["out_dir"] / "scope_metrics.json").read_text(encoding="utf-8")) == m
    with (paths["out_dir"] / "scope_metrics_keys.csv").open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames) == KEY_COLUMNS
        rows = {r["owner_key"]: r for r in reader}
    assert set(rows) == {"vgtrk", "studio y", "music co"}
    assert (rows["vgtrk"]["in_scope"], rows["vgtrk"]["group"]) == ("true", "vgtrk")
    assert (rows["vgtrk"]["requests"], rows["vgtrk"]["network_requests"]) == ("2", "2")
    music = rows["music co"]
    assert (music["in_scope"], music["network_requests"], music["agency_requests"]) == (
        "false",
        "1",
        "1",
    )
    assert not any(p.suffix == ".tmp" for p in paths["out_dir"].iterdir())


def test_build_metrics_refuses_alias_key_outside_candidates(tmp_path):
    paths = _write(tmp_path, ["vgtrk", "music co"])
    with pytest.raises(SystemExit, match=r"not candidates: \['studio y'\]"):
        _run(paths)
