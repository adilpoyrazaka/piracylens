"""Load and validate alias files that group normalized keys (D-037, D-039).

Both files share one nested layout:

    groups:
      <group>:
        evidence: {basis: ..., status: ..., note: ...}
        brands:
          - brand: <display name>
            keys: [<normalized key>, ...]

Owner files also carry ``region`` and ``content_type`` per group and ``market`` per brand.
Reporter files must not carry those fields.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

Kind = Literal["owner", "reporter"]

BASES = frozenset({"reporter", "company_knowledge", "web"})
STATUSES = frozenset({"verified", "carried"})
CONTENT_TYPES = frozenset({"tv", "film", "mixed"})

_GROUP_FIELDS = {
    "owner": frozenset({"region", "content_type", "evidence", "brands"}),
    "reporter": frozenset({"evidence", "brands"}),
}
_BRAND_FIELDS = {
    "owner": frozenset({"brand", "market", "keys"}),
    "reporter": frozenset({"brand", "keys"}),
}
_EVIDENCE_REQUIRED = frozenset({"basis", "status"})
_EVIDENCE_ALLOWED = _EVIDENCE_REQUIRED | {"note"}
_NAME = re.compile(r"[a-z0-9][a-z0-9._-]*")


class AliasError(ValueError):
    """Raised when an alias file violates the schema."""


@dataclass(frozen=True)
class AliasRow:
    """One normalized key and the group/brand it belongs to."""

    key: str
    group: str
    brand: str
    basis: str
    status: str
    region: str | None = None
    content_type: str | None = None
    market: str | None = None


def load_aliases(path: str | Path, kind: Kind) -> list[AliasRow]:
    """Read a YAML alias file and return one validated row per key."""
    text = Path(path).read_text(encoding="utf-8")
    return parse_aliases(yaml.safe_load(text), kind)


def parse_aliases(data: Any, kind: Kind) -> list[AliasRow]:
    """Validate parsed YAML and flatten it to one row per key."""
    if kind not in _GROUP_FIELDS:
        raise AliasError(f"unknown kind: {kind!r}")
    if not isinstance(data, Mapping) or set(data) != {"groups"}:
        raise AliasError("top level must be a mapping with a single 'groups' entry")
    groups = data["groups"]
    if not isinstance(groups, Mapping) or not groups:
        raise AliasError("'groups' must be a non-empty mapping")

    rows: list[AliasRow] = []
    seen: dict[str, str] = {}
    for group, spec in groups.items():
        _check_name(group, "group name")
        where = f"group {group!r}"
        _check_fields(spec, _GROUP_FIELDS[kind], _GROUP_FIELDS[kind], where)
        basis, status = _check_evidence(spec["evidence"], where)

        region = content_type = None
        if kind == "owner":
            region = _check_name(spec["region"], f"{where} region")
            content_type = spec["content_type"]
            if content_type not in CONTENT_TYPES:
                raise AliasError(f"{where}: content_type must be one of {sorted(CONTENT_TYPES)}")

        brands = spec["brands"]
        if not isinstance(brands, list) or not brands:
            raise AliasError(f"{where}: 'brands' must be a non-empty list")
        brand_names: set[str] = set()
        for item in brands:
            _check_fields(item, _BRAND_FIELDS[kind], _BRAND_FIELDS[kind], f"{where} brand")
            brand = _check_text(item["brand"], f"{where} brand")
            if brand in brand_names:
                raise AliasError(f"{where}: duplicate brand {brand!r}")
            brand_names.add(brand)
            market = _check_name(item["market"], f"{where} market") if kind == "owner" else None

            keys = item["keys"]
            if not isinstance(keys, list) or not keys:
                raise AliasError(f"{where} brand {brand!r}: 'keys' must be a non-empty list")
            for key in keys:
                _check_key(key, f"{where} brand {brand!r}")
                if key in seen:
                    raise AliasError(f"key {key!r} in both {seen[key]} and {group}/{brand}")
                seen[key] = f"{group}/{brand}"
                rows.append(
                    AliasRow(key, group, brand, basis, status, region, content_type, market)
                )
    return rows


def to_lookup(rows: Iterable[AliasRow]) -> dict[str, AliasRow]:
    """Map each key to its row (keys are unique after validation)."""
    return {row.key: row for row in rows}


def missing_keys(rows: Iterable[AliasRow], known_keys: Iterable[str]) -> list[str]:
    """Return alias keys that do not exist in the generated key mapping."""
    known = set(known_keys)
    return sorted(row.key for row in rows if row.key not in known)


def _check_fields(obj: Any, required: frozenset, allowed: frozenset, where: str) -> None:
    if not isinstance(obj, Mapping):
        raise AliasError(f"{where}: expected a mapping")
    missing = required - set(obj)
    extra = set(obj) - allowed
    if missing or extra:
        raise AliasError(f"{where}: missing {sorted(missing)}, unexpected {sorted(extra)}")


def _check_evidence(evidence: Any, where: str) -> tuple[str, str]:
    _check_fields(evidence, _EVIDENCE_REQUIRED, _EVIDENCE_ALLOWED, f"{where} evidence")
    basis, status = evidence["basis"], evidence["status"]
    if basis not in BASES:
        raise AliasError(f"{where}: evidence basis must be one of {sorted(BASES)}")
    if status not in STATUSES:
        raise AliasError(f"{where}: evidence status must be one of {sorted(STATUSES)}")
    if "note" in evidence:
        _check_text(evidence["note"], f"{where} evidence note")
    return basis, status


def _check_text(value: Any, where: str) -> str:
    # Rejects YAML 1.1 surprises (yes/no/on -> bool, 1 -> int, null -> None)
    if not isinstance(value, str) or not value or value != value.strip():
        raise AliasError(f"{where}: expected a non-empty trimmed string, got {value!r}")
    return value


def _check_name(value: Any, where: str) -> str:
    _check_text(value, where)
    if not _NAME.fullmatch(value):
        raise AliasError(f"{where}: {value!r} must match {_NAME.pattern}")
    return value


def _check_key(value: Any, where: str) -> str:
    _check_text(value, f"{where} key")
    if value != value.casefold() or "\n" in value:
        raise AliasError(f"{where}: key {value!r} is not a normalized key (casefolded, one line)")
    return value
