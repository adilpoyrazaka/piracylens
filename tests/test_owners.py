"""Tests for rights-owner name normalization."""

import pytest

from piracylens.data.owners import normalize_owner, split_owners

IVI_VARIANTS = [
    "IVI.RU",
    "Ivi.ru",
    "Ivi.ru LLC.",
    "Ivi.ru Ltd.",
    'LLC "IVI.RU"',
    "LCC IVI.RU",
    "IVI.ru LLC / ООО «Иви.ру»",
    "ООО «Иви.ру»",
    "ИBИ.РУ",  # Latin "B" inside a Cyrillic word
    "ИВИ.РУ",
]


@pytest.mark.parametrize("name", IVI_VARIANTS)
def test_ivi_variants_share_one_key(name):
    assert normalize_owner(name) == "ivi.ru"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("BPI", "bpi"),  # all-Latin words are not touched by the homoglyph fix
        ("Okko Co Productions", "okko productions"),  # legal forms only as whole tokens
        ("Cocoon Records", "cocoon records"),
        ("LLC", "llc"),  # a name made only of a legal form is kept
        ('ООО "Окко"', "okko"),
        ("北京爱奇艺科技有限公司", "北京爱奇艺科技有限公司"),  # non-Cyrillic scripts pass through
    ],
)
def test_normalize_owner_cases(name, expected):
    assert normalize_owner(name) == expected


@pytest.mark.parametrize("name", [None, "", "   "])
def test_normalize_owner_empty(name):
    assert normalize_owner(name) == ""


def test_r1_variants_stay_separate():
    # Left to the manual alias table on purpose; the rules must not guess this merge.
    assert normalize_owner("R1") != normalize_owner("R-1")


def test_split_owners_dedupes_and_drops_empty_lines():
    assert split_owners("Defected\nDefected\n\nSkint") == ["defected", "skint"]
    assert split_owners(None) == []
