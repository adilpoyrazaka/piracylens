"""Tests for scope candidate parameters and matching helpers."""

import datetime as dt
import re
from pathlib import Path

import pytest

from piracylens.data.aliases import AliasRow
from piracylens.data.scope_candidates import (
    ParamsError,
    find_prefix_matches,
    load_params,
    matches_prefix,
    parse_params,
    resolve_reporter,
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
