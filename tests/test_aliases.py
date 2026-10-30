"""Tests for alias file loading and validation (D-037, D-039)."""

import copy

import pytest
import yaml

from piracylens.data.aliases import (
    AliasError,
    load_aliases,
    missing_keys,
    parse_aliases,
    to_lookup,
)

OWNER = {
    "groups": {
        "sony": {
            "region": "global",
            "content_type": "mixed",
            "evidence": {
                "basis": "company_knowledge",
                "status": "carried",
                "note": "Columbia and Screen Gems are Sony studios",
            },
            "brands": [
                {
                    "brand": "Columbia Pictures",
                    "market": "global",
                    "keys": ["columbia pictures industries", "columbia pictures"],
                },
                {"brand": "Screen Gems", "market": "global", "keys": ["screen gems"]},
            ],
        },
        "zee": {
            "region": "in",
            "content_type": "tv",
            "evidence": {"basis": "reporter", "status": "verified"},
            "brands": [{"brand": "Zee CIS", "market": "ru", "keys": ["zee cis"]}],
        },
    }
}

REPORTER = {
    "groups": {
        "f6": {
            "evidence": {"basis": "reporter", "status": "verified"},
            "brands": [
                {"brand": "Group-IB", "keys": ["group-ib"]},
                {"brand": "F.A.C.C.T.", "keys": ["f.a.c.c.t", "f.a.k.k.t"]},
                {"brand": "F6", "keys": ["f6"]},
            ],
        }
    }
}


def owner():
    return copy.deepcopy(OWNER)


def reporter():
    return copy.deepcopy(REPORTER)


def test_owner_file_flattens_to_one_row_per_key():
    rows = parse_aliases(owner(), "owner")
    assert len(rows) == 4
    row = to_lookup(rows)["screen gems"]
    assert (row.group, row.brand, row.market) == ("sony", "Screen Gems", "global")
    assert (row.region, row.content_type, row.status) == ("global", "mixed", "carried")
    assert to_lookup(rows)["zee cis"].market == "ru"


def test_reporter_file_flattens_to_one_row_per_key():
    rows = parse_aliases(reporter(), "reporter")
    assert len(rows) == 4
    row = to_lookup(rows)["f.a.k.k.t"]
    assert (row.group, row.brand, row.market, row.region) == ("f6", "F.A.C.C.T.", None, None)


def test_same_key_in_two_brands_is_rejected():
    data = owner()
    data["groups"]["zee"]["brands"][0]["keys"].append("screen gems")
    with pytest.raises(AliasError, match="in both"):
        parse_aliases(data, "owner")


def test_duplicate_brand_is_rejected():
    data = owner()
    data["groups"]["sony"]["brands"][1]["brand"] = "Columbia Pictures"
    with pytest.raises(AliasError, match="duplicate brand"):
        parse_aliases(data, "owner")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["groups"]["f6"]["brands"][0].update(market="ru"),
        lambda d: d["groups"]["f6"].update(region="ru"),
    ],
    ids=["brand_market", "group_region"],
)
def test_reporter_file_rejects_owner_only_fields(mutate):
    data = reporter()
    mutate(data)
    with pytest.raises(AliasError, match="unexpected"):
        parse_aliases(data, "reporter")


def test_owner_brand_requires_market():
    data = owner()
    del data["groups"]["sony"]["brands"][0]["market"]
    with pytest.raises(AliasError, match="missing"):
        parse_aliases(data, "owner")


def test_misspelled_field_is_rejected():
    data = owner()
    brand = data["groups"]["sony"]["brands"][0]
    brand["markt"] = brand.pop("market")
    with pytest.raises(AliasError, match="unexpected"):
        parse_aliases(data, "owner")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["groups"]["sony"]["evidence"].update(status="likely"),
        lambda d: d["groups"]["sony"]["evidence"].update(basis="guess"),
        lambda d: d["groups"]["sony"].update(content_type="music"),
    ],
    ids=["status", "basis", "content_type"],
)
def test_invalid_enum_values_are_rejected(mutate):
    data = owner()
    mutate(data)
    with pytest.raises(AliasError):
        parse_aliases(data, "owner")


def test_unquoted_yaml_booleans_are_rejected():
    text = """
groups:
  test:
    evidence: {basis: web, status: carried}
    brands:
      - brand: Test
        keys: [no, on]
"""
    with pytest.raises(AliasError, match="trimmed string"):
        parse_aliases(yaml.safe_load(text), "reporter")


def test_quoted_yaml_keys_are_accepted():
    text = """
groups:
  test:
    evidence: {basis: web, status: carried}
    brands:
      - brand: Test
        keys: ["no", "on"]
"""
    rows = parse_aliases(yaml.safe_load(text), "reporter")
    assert [r.key for r in rows] == ["no", "on"]


def test_key_must_be_normalized():
    data = owner()
    data["groups"]["sony"]["brands"][1]["keys"] = ["Screen Gems"]
    with pytest.raises(AliasError, match="not a normalized key"):
        parse_aliases(data, "owner")


def test_group_name_must_be_a_slug():
    data = owner()
    data["groups"]["Sony Pictures"] = data["groups"].pop("sony")
    with pytest.raises(AliasError, match="must match"):
        parse_aliases(data, "owner")


def test_top_level_must_only_contain_groups():
    data = owner()
    data["version"] = 1
    with pytest.raises(AliasError, match="single 'groups'"):
        parse_aliases(data, "owner")


def test_load_aliases_reads_utf8_file(tmp_path):
    data = owner()
    data["groups"]["iqiyi"] = {
        "region": "cn",
        "content_type": "mixed",
        "evidence": {"basis": "company_knowledge", "status": "carried"},
        "brands": [
            {"brand": "iQIYI", "market": "cn", "keys": ["北京爱奇艺科技有限公司", "iqiyi"]},
        ],
    }
    path = tmp_path / "owner_aliases.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    rows = load_aliases(path, "owner")
    assert to_lookup(rows)["北京爱奇艺科技有限公司"].group == "iqiyi"


def test_missing_keys_reports_keys_absent_from_mapping():
    rows = parse_aliases(reporter(), "reporter")
    known = ["group-ib", "f6", "f.a.c.c.t"]
    assert missing_keys(rows, known) == ["f.a.k.k.t"]
