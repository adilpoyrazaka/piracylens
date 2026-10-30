"""Owner scope candidates from pre-registered parameters.

The parameter file fixes every threshold, reporter list and name prefix before
any candidate list is produced. This module loads that file under a strict
schema and provides the matching helpers used by the candidate queries.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from piracylens.data.aliases import AliasRow

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
        {"min_reporter_requests", "seed_share", "min_owner_requests"},
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


def _key_list(obj: Mapping[str, Any], field: str, where: str) -> tuple[str, ...]:
    value = obj[field]
    name = _name(where, field)
    if not isinstance(value, list) or not value:
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
