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
        (
            "BluTV İletişim ve Dijital Yayın Hizmetleri A.Ş.",
            "blutv iletişim ve dijital yayın hizmetleri",
        ),
        (
            "Demirören TV Radyo Yayıncılık Yapımcılık A.Ş.",
            "demirören tv radyo yayıncılık yapımcılık",
        ),
        ("Örnek Film Ltd. Şti.", "örnek film"),
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


@pytest.mark.parametrize(
    ("encoded", "plain"),
    [
        ("Covington &amp; Burling LLP", "Covington & Burling LLP"),
        ("zeetv &amp;wayaak", "zeetv &wayaak"),
        ("R&eacute;cits de Mars", "Récits de Mars"),
        ("Taylor &amp;amp; Francis Group", "Taylor & Francis Group"),
        ("&#116;&#101;&#115;&#116; Media", "test Media"),
        ("SM Culture &amp;amp;amp;amp;amp; Contents", "SM Culture & Contents"),
    ],
)
def test_html_entities_are_decoded(encoded, plain):
    assert normalize_owner(encoded) == normalize_owner(plain)


def test_entities_without_semicolon_are_not_decoded():
    assert normalize_owner("CÔNG TY TNHH TT&GT 789F") == "công ty tnhh tt gt 789f"
    assert normalize_owner("Smith&notes") == normalize_owner("Smith notes")


def test_invisible_format_characters_are_removed():
    assert normalize_owner("100 Billion\u200b Wires") == normalize_owner("100 Billion Wires")
    assert normalize_owner("Star\u00adMedia") == normalize_owner("StarMedia")


def test_invisible_marks_do_not_block_downstream_steps():
    # Bidi marks glued to the first/last token used to block legal-suffix removal and dedup
    assert normalize_owner("\u202aXFC Inc\u202c\u200f") == "xfc"
    assert normalize_owner("\u202acat cat\u202c\u200f") == "cat"
