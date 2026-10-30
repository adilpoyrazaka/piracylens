"""Normalization of rights-owner names from the Google Transparency Report.

`owner_id` does not identify an owner: one name maps to many ids (verified 2026-10-30).
Owners are therefore grouped by a rule-based key derived from `owner_name`.
"""

from __future__ import annotations

import html
import re
import unicodedata
from html.entities import html5

__all__ = ["normalize_owner", "split_owners"]

_CYRILLIC = re.compile(r"[\u0400-\u04FF]")

# Latin characters that look identical to Cyrillic ones; fixed only inside mixed-script words.
_HOMOGLYPHS = str.maketrans(
    {
        "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М",
        "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
        "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у",
    }
)  # fmt: skip

_TRANSLIT = str.maketrans(
    {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
        "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
        "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
        "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
        "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
        "і": "i", "ї": "yi", "є": "ye", "ґ": "g",
    }
)  # fmt: skip

_QUOTES = re.compile(r"[«»\"“”„'‘’`]")
_SEPARATORS = re.compile(r"[/|,;:()\[\]{}&+]")

# Legal-form tokens, compared after removing dots (heuristic list).
_LEGAL_FORMS = frozenset(
    {
        "llc", "lcc", "ltd", "limited", "inc", "corp", "corporation", "co", "gmbh",
        "sa", "srl", "bv", "nv", "plc", "ag", "ooo", "ao", "oao", "zao", "pao", "tov",
        "aş", "şti",
    }
)  # fmt: skip


def _fix_homoglyphs(text: str) -> str:
    """Replace Latin look-alikes with Cyrillic letters in words that already contain Cyrillic."""
    return " ".join(
        word.translate(_HOMOGLYPHS) if _CYRILLIC.search(word) else word for word in text.split()
    )


_ENTITY = re.compile(r"&(?:#[0-9]+|#[xX][0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]*);")


def _decode_entity(match: re.Match[str]) -> str:
    entity = match.group(0)
    if entity[1] == "#":
        return html.unescape(entity)
    return html5.get(entity[1:], entity)  # unknown names such as '&BassArena;' stay as they are


def _unescape_html(text: str) -> str:
    """Decode ';'-terminated HTML entities, repeating for nested input such as '&amp;amp;'.

    Entities without a closing ';' are left alone: lenient decoding turns the real name
    'TT&GT' into 'TT>'. Each decoding pass shortens the text, so the loop always ends.
    """
    while True:
        decoded = _ENTITY.sub(_decode_entity, text)
        if decoded == text:
            return text
        text = decoded


def normalize_owner(name: str | None) -> str:
    """Return a grouping key for a single owner name ('' for empty input)."""
    if not name:
        return ""
    text = unicodedata.normalize("NFKC", _unescape_html(name))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")  # zero-width etc.
    text = _fix_homoglyphs(text).casefold()
    text = text.replace("\u0307", "")  # Turkish "İ" casefolds to "i" + combining dot
    text = _QUOTES.sub("", text).translate(_TRANSLIT)
    text = _SEPARATORS.sub(" ", text)

    tokens = [tok.strip(".-") for tok in text.split()]
    tokens = [tok for tok in tokens if tok]
    kept = [tok for tok in tokens if tok.replace(".", "") not in _LEGAL_FORMS]
    if not kept:  # the name is only a legal form; keep it rather than return ''
        kept = tokens

    return " ".join(dict.fromkeys(kept))  # drop repeated tokens, keep order


def split_owners(name: str | None) -> list[str]:
    """Split a multi-owner field on line breaks and return unique normalized keys."""
    if not name:
        return []
    keys = (normalize_owner(part) for part in name.splitlines())
    return list(dict.fromkeys(key for key in keys if key))
