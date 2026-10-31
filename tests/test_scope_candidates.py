"""Tests for scope candidate parameters and matching helpers."""

import csv
import datetime as dt
import re
from pathlib import Path

import duckdb
import pytest
import yaml

from piracylens.data.aliases import AliasRow
from piracylens.data.scope_candidates import (
    CANDIDATE_COLUMNS,
    ParamsError,
    absent_keys,
    build_agency_table,
    build_candidates,
    find_prefix_matches,
    format_prefix_matches,
    load_params,
    matches_prefix,
    parse_params,
    resolve_reporter,
    unknown_reporters,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
_DELETE = object()


def _valid() -> dict:
    return {
        "window": {"start": "2021-01-01"},
        "sources": {
            "ranking": {"min_requests": 5000},
            "protection_network": {"reporters": ["contentscan", "f6"], "min_owner_requests": 500},
            "film_tv_agencies": {
                "min_reporter_requests": 50000,
                "seed_share": 0.5,
                "min_owner_requests": 500,
                "selected": [],
            },
            "prefix_scan": {
                "min_requests": 5,
                "prefix": ["disney", "canal"],
                "word": ["sts", "hbo"],
            },
        },
        "seed_owner_keys": ["vgtrk", "ivi.ru"],
    }


@pytest.mark.parametrize(
    ("key", "prefix", "mode", "expected"),
    [
        ("sts", "sts", "word", True),
        ("sts tv", "sts", "word", True),
        ("sts.", "sts", "word", True),
        ("stsoft", "sts", "word", False),
        ("sky deutschland", "sky", "word", True),
        ("skydance", "sky", "word", False),
        ("canalplus", "canal", "prefix", True),
        ("canalplus", "canal", "word", False),
        ("groupe canal", "canal", "prefix", False),
        ("北京爱奇艺科技有限公司", "北京爱奇艺", "prefix", True),
        ("北京爱奇艺科技有限公司", "北京爱奇艺", "word", False),
    ],
)
def test_matches_prefix(key, prefix, mode, expected):
    assert matches_prefix(key, prefix, mode) is expected


def test_matches_prefix_rejects_unknown_mode():
    with pytest.raises(ValueError):
        matches_prefix("sts", "sts", "fuzzy")


def test_find_prefix_matches_returns_every_match():
    prefixes = [("ivi", "word"), ("ivi.ru", "word"), ("okko", "word")]
    assert find_prefix_matches("ivi.ru online", prefixes) == [("ivi", "word"), ("ivi.ru", "word")]


def test_resolve_reporter_uses_group_then_falls_back_to_key():
    row = AliasRow(
        key="group-ib", group="f6", brand="Group-IB", basis="reporter", status="verified"
    )
    lookup = {row.key: row}
    assert resolve_reporter("group-ib", lookup) == "f6"
    assert resolve_reporter("contentscan", lookup) == "contentscan"


def test_parse_params_accepts_valid_document():
    params = parse_params(_valid())
    assert params.window_start == dt.date(2021, 1, 1)
    assert params.network_reporters == ("contentscan", "f6")
    assert params.agency_seed_share == 0.5
    assert params.agency_selected == ()
    assert params.prefixes == (
        ("disney", "prefix"),
        ("canal", "prefix"),
        ("sts", "word"),
        ("hbo", "word"),
    )


def test_parse_params_keeps_selected_agencies_in_order():
    doc = _valid()
    doc["sources"]["film_tv_agencies"]["selected"] = ["f6", "contentscan"]
    assert parse_params(doc).agency_selected == ("f6", "contentscan")


def test_parse_params_rejects_non_mapping():
    with pytest.raises(ParamsError, match="must be a mapping"):
        parse_params(None)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("sources.ranking.max_requests", 1, "unexpected fields"),
        ("window.start", _DELETE, "missing fields"),
        ("window.start", dt.date(2021, 1, 1), "quoted ISO date"),
        ("window.start", "2021-13-01", "not an ISO date"),
        ("sources.ranking.min_requests", True, "positive integer"),
        ("sources.ranking.min_requests", 0, "positive integer"),
        ("sources.film_tv_agencies.seed_share", 1.5, "(0, 1]"),
        ("sources.prefix_scan.prefix", ["Disney"], "casefolded"),
        ("sources.prefix_scan.prefix", [" disney"], "stripped"),
        ("sources.prefix_scan.word", ["sts", "sts"], "duplicate"),
        ("sources.prefix_scan.word", ["sts", "disney"], "both"),
        ("sources.prefix_scan.word", [False], "quoted string"),
        ("seed_owner_keys", [], "non-empty list"),
        ("sources.film_tv_agencies.selected", _DELETE, "missing fields"),
        ("sources.film_tv_agencies.selected", "f6", "must be a list"),
        ("sources.film_tv_agencies.selected", ["F6"], "casefolded"),
        ("sources.film_tv_agencies.selected", ["f6", "f6"], "duplicate"),
    ],
)
def test_parse_params_rejects_invalid_documents(path, value, message):
    doc = _valid()
    *parents, last = path.split(".")
    node = doc
    for part in parents:
        node = node[part]
    if value is _DELETE:
        del node[last]
    else:
        node[last] = value
    with pytest.raises(ParamsError, match=re.escape(message)):
        parse_params(doc)


def test_repo_parameter_file_is_valid():
    params = load_params(REPO_ROOT / "configs" / "scope_candidates.yaml")
    modes = [mode for _, mode in params.prefixes]
    assert modes.count("prefix") == 78
    assert modes.count("word") == 49
    assert len(params.seed_owner_keys) == 104
    assert params.network_reporters == ("contentscan", "f6", "starmedia", "vindex")


def test_absent_keys_returns_sorted_missing():
    assert absent_keys(["b", "a", "c", "a"], ["c"]) == ["a", "b"]
    assert absent_keys([], ["c"]) == []


def test_unknown_reporters_accepts_groups_and_data_keys():
    lookup = {"group-ib": AliasRow("group-ib", "f6", "Group-IB", "reporter", "verified")}
    result = unknown_reporters(["f6", "contentscan", "vindex"], lookup, ["contentscan"])
    assert result == ["vindex"]


def _write_synthetic(tmp_path: Path, seed: list[str]) -> dict[str, Path]:
    params = _valid()
    params["sources"]["protection_network"]["reporters"] = ["f6"]
    params["sources"]["film_tv_agencies"]["min_reporter_requests"] = 2
    params["seed_owner_keys"] = seed
    paths = {
        "params": tmp_path / "params.yaml",
        "aliases": tmp_path / "aliases.yaml",
        "requests": tmp_path / "requests.parquet",
        "owner_keys": tmp_path / "owner_keys.parquet",
        "out": tmp_path / "out" / "agencies.csv",
    }
    paths["params"].write_text(yaml.safe_dump(params), encoding="utf-8")
    paths["aliases"].write_text(
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
    con = duckdb.connect()
    con.execute(
        f"""
        COPY (SELECT * FROM (VALUES
            (1, TIMESTAMP '2020-06-01', 'Group-IB', 'VGTRK'),
            (2, TIMESTAMP '2021-02-01', 'Group-IB', 'VGTRK'),
            (3, TIMESTAMP '2022-03-01', 'F6', 'Ivi.ru'),
            (4, TIMESTAMP '2023-04-01', 'F6', 'Other Studio'),
            (5, TIMESTAMP '2023-05-01', 'Lone Agency', 'Other Studio'),
            (6, TIMESTAMP '2023-06-01', NULL, 'VGTRK')
        ) AS t(request_id, requested_at_utc, reporter_name, owner_name))
        TO '{paths["requests"]}' (FORMAT parquet)
        """
    ).fetchall()
    con.execute(
        f"""
        COPY (SELECT * FROM (VALUES
            ('VGTRK', 'vgtrk'), ('Ivi.ru', 'ivi.ru'), ('Other Studio', 'other studio')
        ) AS t(owner_name, owner_key))
        TO '{paths["owner_keys"]}' (FORMAT parquet)
        """
    ).fetchall()
    return paths


def _run(paths: dict[str, Path]) -> None:
    build_agency_table(
        paths["params"],
        paths["aliases"],
        paths["requests"],
        paths["owner_keys"],
        paths["out"],
        threads=1,
        memory_limit="1GB",
    )


def test_build_agency_table_on_synthetic_data(tmp_path):
    paths = _write_synthetic(tmp_path, ["vgtrk", "ivi.ru"])
    _run(paths)
    with paths["out"].open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # Request 1 is before the window, request 6 has no reporter, "lone agency" is below
    # the volume threshold; Group-IB and F6 are merged into one group.
    assert [r["reporter_group"] for r in rows] == ["f6"]
    assert rows[0]["requests"] == "3"
    assert rows[0]["seed_requests"] == "2"
    assert float(rows[0]["seed_share"]) == pytest.approx(0.6667)
    assert rows[0]["seed_flag"] == "true"
    assert rows[0]["reporter_keys"] == "2"
    assert not any(p.suffix == ".tmp" for p in paths["out"].parent.iterdir())


def test_build_agency_table_stops_on_missing_seed_key(tmp_path):
    paths = _write_synthetic(tmp_path, ["vgtrk", "not in mapping"])
    with pytest.raises(SystemExit, match="seed owner keys missing"):
        _run(paths)
    assert not paths["out"].exists()


def test_format_prefix_matches():
    assert format_prefix_matches([("sts", "word"), ("disney", "prefix")]) == (
        "sts:word;disney:prefix"
    )
    assert format_prefix_matches([]) == ""


def _write_candidate_synthetic(tmp_path: Path, selected: list[str]) -> dict[str, Path]:
    params = _valid()
    params["sources"]["ranking"]["min_requests"] = 4
    params["sources"]["protection_network"] = {"reporters": ["f6"], "min_owner_requests": 2}
    params["sources"]["film_tv_agencies"].update(
        {"min_reporter_requests": 3, "min_owner_requests": 2, "selected": selected}
    )
    params["sources"]["prefix_scan"] = {"min_requests": 2, "prefix": ["disney"], "word": ["sts"]}
    params["seed_owner_keys"] = ["vgtrk", "quiet seed", "old seed"]
    paths = _write_synthetic(tmp_path, ["vgtrk"])
    paths["params"].write_text(yaml.safe_dump(params), encoding="utf-8")
    paths["out"] = tmp_path / "out" / "candidates.csv"
    big = 'Big "Owner"\nLtd'
    requests = [
        (1, "2019-05-01", "Group-IB", "VGTRK"),
        (2, "2021-02-01", "Group-IB", "VGTRK"),
        (3, "2022-02-01", "Group-IB", "VGTRK"),
        (4, "2022-03-01", "F6", "Vgtrk"),
        (5, "2023-01-01", "Small Rep", big),
        (6, "2023-01-02", "Small Rep", big),
        (7, "2023-01-03", "Small Rep", big),
        (8, "2023-01-04", None, big),
        (9, "2023-02-01", "Agency X", "Studio Y"),
        (10, "2023-02-02", "Agency X", "Studio Y"),
        (11, "2023-02-03", "Agency X", "Other"),
        (12, "2024-01-01", "Small Rep", "STS TV"),
        (13, "2024-01-02", "Small Rep", "STS TV"),
        (14, "2024-01-03", "Small Rep", "STS TV, Disney Plus"),
        (15, "2024-01-04", "Small Rep", "Disney Plus"),
        (16, "2024-02-01", "Small Rep", "STSoft"),
        (17, "2024-02-02", "Small Rep", "STSoft"),
        (18, "2024-02-03", "Small Rep", "STSoft"),
        (19, "2024-03-01", "Small Rep", "Quiet Seed"),
        (20, "2019-07-01", "Small Rep", "Old Seed"),
    ]
    owner_keys = [
        ("VGTRK", "vgtrk"),
        ("Vgtrk", "vgtrk"),
        (big, "big owner"),
        ("Studio Y", "studio y"),
        ("Other", "other"),
        ("STS TV", "sts tv"),
        ("STS TV, Disney Plus", "sts tv"),
        ("STS TV, Disney Plus", "disney plus"),
        ("Disney Plus", "disney plus"),
        ("STSoft", "stsoft"),
        ("Quiet Seed", "quiet seed"),
        ("Old Seed", "old seed"),
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


def _run_candidates(paths: dict[str, Path]) -> list[dict[str, str]]:
    build_candidates(
        paths["params"],
        paths["aliases"],
        paths["requests"],
        paths["owner_keys"],
        paths["out"],
        threads=1,
        memory_limit="1GB",
    )
    with paths["out"].open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_build_candidates_on_synthetic_data(tmp_path):
    paths = _write_candidate_synthetic(tmp_path, ["agency x"])
    rows = _run_candidates(paths)
    by_key = {r["owner_key"]: r for r in rows}
    # "other" and "stsoft" are caught by no source and are not seeds; "stsoft" shows that
    # word mode rejects a prefix followed by a letter.
    assert [r["owner_key"] for r in rows] == [
        "big owner",
        "sts tv",
        "vgtrk",
        "disney plus",
        "studio y",
        "quiet seed",
        "old seed",
    ]
    flags = {
        k: tuple(r[f"source_{s}"] == "true" for s in "abcd") + (r["in_seed"] == "true",)
        for k, r in by_key.items()
    }
    assert flags == {
        "big owner": (True, False, False, False, False),
        "sts tv": (False, False, False, True, False),
        "vgtrk": (False, True, False, False, True),
        "disney plus": (False, False, False, True, False),
        "studio y": (False, False, True, False, False),
        "quiet seed": (False, False, False, False, True),
        "old seed": (False, False, False, False, True),
    }
    vgtrk = by_key["vgtrk"]
    # The 2019 request is outside the window for counts but inside the full year range.
    assert (vgtrk["requests"], vgtrk["network_requests"], vgtrk["agency_requests"]) == (
        "3",
        "3",
        "0",
    )
    assert (vgtrk["first_year_all"], vgtrk["last_year_all"]) == ("2019", "2022")
    assert (vgtrk["variants"], vgtrk["top_raw_name"]) == ("2", "VGTRK")
    assert (vgtrk["top_reporter_group"], float(vgtrk["top_reporter_share"])) == ("f6", 1.0)
    big = by_key["big owner"]
    # A request without a reporter counts in the volume but never as the top reporter.
    assert big["top_raw_name"] == 'Big "Owner"\nLtd'
    assert (big["top_reporter_group"], float(big["top_reporter_share"])) == ("small rep", 0.75)
    assert by_key["studio y"]["agency_requests"] == "2"
    # The compound name counts once for each of its keys.
    assert (by_key["sts tv"]["requests"], by_key["sts tv"]["prefixes"]) == ("3", "sts:word")
    assert (by_key["disney plus"]["requests"], by_key["disney plus"]["prefixes"]) == (
        "2",
        "disney:prefix",
    )
    old = by_key["old seed"]
    assert (old["requests"], old["first_year_all"], old["top_raw_name"]) == ("0", "2019", "")
    assert not any(p.suffix == ".tmp" for p in paths["out"].parent.iterdir())


def test_build_candidates_has_no_outcome_columns(tmp_path):
    paths = _write_candidate_synthetic(tmp_path, ["agency x"])
    _run_candidates(paths)
    with paths["out"].open(encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    assert tuple(header) == CANDIDATE_COLUMNS
    assert not [c for c in header if re.search("label|action|removed|index|pending", c)]


def test_build_candidates_refuses_empty_selection(tmp_path):
    paths = _write_candidate_synthetic(tmp_path, [])
    with pytest.raises(SystemExit, match="selected is empty"):
        _run_candidates(paths)
    assert not paths["out"].exists()


def test_build_candidates_refuses_agency_outside_table(tmp_path):
    paths = _write_candidate_synthetic(tmp_path, ["agency x", "ghost agency"])
    with pytest.raises(SystemExit, match=r"not in the agency table: \['ghost agency'\]"):
        _run_candidates(paths)
    assert not paths["out"].exists()
    assert not any(p.suffix == ".tmp" for p in paths["out"].parent.iterdir())
