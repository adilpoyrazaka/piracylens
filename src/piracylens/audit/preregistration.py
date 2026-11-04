"""Pre-registered design of the takedown audit model.

The design file fixes the label, the time split, the metrics, the baselines and the
mandatory checks before any label rate or model output is looked at. This module loads
it under a strict schema: unknown or missing fields, broken date orders, out-of-range
thresholds and unknown method names are rejected, so a typo cannot silently change the
design. The training size starts empty and must be filled before any model runs.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path("configs/audit_preregistration.yaml")

COUNTERS = frozenset({"urls_removed", "urls_not_in_index", "urls_no_action", "urls_pending"})
SOURCE_FLAGS = frozenset({"source_a", "source_b", "source_c", "source_d"})
BASELINES = frozenset({"base_rate", "domain_past_rate"})
CANDIDATES = frozenset({"logistic_regression", "linear_svm_sgd", "xgboost"})
IMBALANCE_VARIANTS = frozenset({"none", "class_weight"})
CALIBRATIONS = frozenset({"isotonic", "sigmoid", "none"})
PRIMARY_METRICS = frozenset({"pr_auc"})
OTHER_METRICS = frozenset({"recall_at_precision", "brier"})

_TOP = frozenset(
    {
        "data",
        "population",
        "label",
        "split",
        "features",
        "sampling",
        "models",
        "metrics",
        "business_metric",
        "checks",
    }
)
_PERIODS = ("train", "validation", "test")


class PreregistrationError(ValueError):
    """Raised when the audit design file violates the schema."""


@dataclass(frozen=True)
class Period:
    """An inclusive date range of request filing dates."""

    start: dt.date
    end: dt.date

    def contains(self, day: dt.date) -> bool:
        return self.start <= day <= self.end


@dataclass(frozen=True)
class Preregistration:
    """Validated audit design."""

    table: str
    window_start: dt.date
    data_end: dt.date
    history_start: dt.date
    decided_counters: tuple[str, ...]
    label_counter: str
    label_min_count: int
    sensitivity_min_share: float
    train: Period
    validation: Period
    test: Period
    group_by: str
    report_unseen_domains: bool
    past_rate_min_gap_days: int
    excluded_features: tuple[str, ...]
    train_unit: str
    train_requests: int | None
    rebalance_eval: bool
    seed: int
    baselines: tuple[str, ...]
    candidates: tuple[str, ...]
    imbalance_variants: tuple[str, ...]
    calibration: str
    tune_on: str
    primary_metric: str
    other_metrics: tuple[str, ...]
    precision_target: float
    bootstrap_resamples: int
    bootstrap_unit: str
    bootstrap_confidence: float
    business_metric: str
    business_at_precision: float
    stability_train_start: dt.date
    excluded_groups: tuple[str, ...]
    per_group_min_test_rows: int
    reporter_ablation: bool
    source_subset: tuple[str, ...]
    no_reporter_model: bool
    large_gap_pr_auc: float

    def require_train_requests(self) -> int:
        """Return the training size, or stop if it has not been fixed yet."""
        if self.train_requests is None:
            raise PreregistrationError(
                "sampling.train_requests is empty: measure memory without labels, "
                "write the size into the design file and commit it before training"
            )
        return self.train_requests


def load_preregistration(path: str | Path = DEFAULT_PATH) -> Preregistration:
    """Read and validate the audit design file."""
    with Path(path).open(encoding="utf-8") as f:
        return parse_preregistration(yaml.safe_load(f))


def parse_preregistration(data: Any) -> Preregistration:
    """Validate an already parsed design document."""
    top = _mapping(data, _TOP, "top level")

    d = _mapping(top["data"], {"table", "window_start", "data_end", "history_start"}, "data")
    table = _text(d, "table", "data")
    window_start = _date(d, "window_start", "data")
    data_end = _date(d, "data_end", "data")
    history_start = _date(d, "history_start", "data")
    if not history_start < window_start < data_end:
        raise PreregistrationError("data: need history_start < window_start < data_end")

    pop = _mapping(top["population"], {"decided_counters"}, "population")
    decided = _names(pop, "decided_counters", "population", COUNTERS)

    lab = _mapping(top["label"], {"primary", "sensitivity_min_share"}, "label")
    primary = _mapping(lab["primary"], {"counter", "min_count"}, "label.primary")
    label_counter = _choice(primary, "counter", "label.primary", COUNTERS)
    if label_counter not in decided:
        raise PreregistrationError("label.primary.counter must be one of the decided counters")
    label_min_count = _int(primary, "min_count", "label.primary", minimum=1)
    sensitivity_min_share = _share(lab, "sensitivity_min_share", "label", allow_one=True)

    sp = _mapping(top["split"], {*_PERIODS, "group_by", "report_unseen_domains"}, "split")
    train, validation, test = (_period(sp, name) for name in _PERIODS)
    if train.start != window_start:
        raise PreregistrationError("split.train.start must equal data.window_start")
    one_day = dt.timedelta(days=1)
    if validation.start != train.end + one_day or test.start != validation.end + one_day:
        raise PreregistrationError("split: periods must follow each other without gaps")
    if test.end != data_end:
        raise PreregistrationError("split.test.end must equal data.data_end")
    group_by = _choice(sp, "group_by", "split", frozenset({"domain"}))
    report_unseen = _bool(sp, "report_unseen_domains", "split")

    feat = _mapping(top["features"], {"past_rate_min_gap_days", "excluded"}, "features")
    gap_days = _int(feat, "past_rate_min_gap_days", "features", minimum=1)
    excluded_features = _names(feat, "excluded", "features")
    if "from_abuser" not in excluded_features:
        raise PreregistrationError("features.excluded must list from_abuser")

    sam = _mapping(
        top["sampling"], {"train_unit", "train_requests", "rebalance_eval", "seed"}, "sampling"
    )
    train_unit = _choice(sam, "train_unit", "sampling", frozenset({"request"}))
    train_requests = (
        None
        if sam["train_requests"] is None
        else _int(sam, "train_requests", "sampling", minimum=1)
    )
    rebalance_eval = _bool(sam, "rebalance_eval", "sampling")
    if rebalance_eval:
        raise PreregistrationError(
            "sampling.rebalance_eval must be false: evaluation keeps the real rate"
        )
    seed = _int(sam, "seed", "sampling", minimum=0)

    mod = _mapping(
        top["models"],
        {"baselines", "candidates", "imbalance_variants", "calibration", "tune_on"},
        "models",
    )
    baselines = _names(mod, "baselines", "models", BASELINES)
    candidates = _names(mod, "candidates", "models", CANDIDATES)
    variants = _names(mod, "imbalance_variants", "models", IMBALANCE_VARIANTS)
    if "none" not in variants:
        raise PreregistrationError("models.imbalance_variants must include none as the reference")
    calibration = _choice(mod, "calibration", "models", CALIBRATIONS)
    tune_on = _choice(mod, "tune_on", "models", frozenset({"validation"}))

    met = _mapping(
        top["metrics"], {"primary", "others", "precision_target", "bootstrap"}, "metrics"
    )
    primary_metric = _choice(met, "primary", "metrics", PRIMARY_METRICS)
    other_metrics = _names(met, "others", "metrics", OTHER_METRICS)
    precision_target = _share(met, "precision_target", "metrics")
    boot = _mapping(met["bootstrap"], {"resamples", "unit", "confidence"}, "metrics.bootstrap")
    resamples = _int(boot, "resamples", "metrics.bootstrap", minimum=1)
    boot_unit = _choice(boot, "unit", "metrics.bootstrap", frozenset({"domain"}))
    confidence = _share(boot, "confidence", "metrics.bootstrap")

    bus = _mapping(top["business_metric"], {"name", "at_precision"}, "business_metric")
    business_name = _text(bus, "name", "business_metric")
    at_precision = _share(bus, "at_precision", "business_metric")
    if at_precision != precision_target:
        raise PreregistrationError(
            "business_metric.at_precision must equal metrics.precision_target"
        )

    chk = _mapping(
        top["checks"],
        {
            "stability_train_start",
            "excluded_groups",
            "per_group_min_test_rows",
            "reporter_ablation",
            "source_subset",
            "no_reporter_model",
            "large_gap_pr_auc",
        },
        "checks",
    )
    stability_start = _date(chk, "stability_train_start", "checks")
    if not train.start < stability_start <= train.end:
        raise PreregistrationError(
            "checks.stability_train_start must fall after the train start, inside the train period"
        )

    return Preregistration(
        table=table,
        window_start=window_start,
        data_end=data_end,
        history_start=history_start,
        decided_counters=decided,
        label_counter=label_counter,
        label_min_count=label_min_count,
        sensitivity_min_share=sensitivity_min_share,
        train=train,
        validation=validation,
        test=test,
        group_by=group_by,
        report_unseen_domains=report_unseen,
        past_rate_min_gap_days=gap_days,
        excluded_features=excluded_features,
        train_unit=train_unit,
        train_requests=train_requests,
        rebalance_eval=rebalance_eval,
        seed=seed,
        baselines=baselines,
        candidates=candidates,
        imbalance_variants=variants,
        calibration=calibration,
        tune_on=tune_on,
        primary_metric=primary_metric,
        other_metrics=other_metrics,
        precision_target=precision_target,
        bootstrap_resamples=resamples,
        bootstrap_unit=boot_unit,
        bootstrap_confidence=confidence,
        business_metric=business_name,
        business_at_precision=at_precision,
        stability_train_start=stability_start,
        excluded_groups=_names(chk, "excluded_groups", "checks"),
        per_group_min_test_rows=_int(chk, "per_group_min_test_rows", "checks", minimum=1),
        reporter_ablation=_bool(chk, "reporter_ablation", "checks"),
        source_subset=_names(chk, "source_subset", "checks", SOURCE_FLAGS),
        no_reporter_model=_bool(chk, "no_reporter_model", "checks"),
        large_gap_pr_auc=_share(chk, "large_gap_pr_auc", "checks"),
    )


def _mapping(obj: Any, required: set[str] | frozenset[str], where: str) -> Mapping[str, Any]:
    if not isinstance(obj, dict):
        raise PreregistrationError(f"{where} must be a mapping")
    keys = set(obj)
    missing = sorted(required - keys)
    extra = sorted(str(k) for k in keys - required)
    if missing or extra:
        raise PreregistrationError(f"{where}: missing fields {missing}, unexpected fields {extra}")
    return obj


def _text(obj: Mapping[str, Any], field: str, where: str) -> str:
    value = obj[field]
    if not isinstance(value, str) or not value.strip():
        raise PreregistrationError(f"{where}.{field} must be a non-empty string, got {value!r}")
    return value


def _choice(obj: Mapping[str, Any], field: str, where: str, allowed: frozenset[str]) -> str:
    value = _text(obj, field, where)
    if value not in allowed:
        raise PreregistrationError(f"{where}.{field}: {value!r} is not one of {sorted(allowed)}")
    return value


def _names(
    obj: Mapping[str, Any], field: str, where: str, allowed: frozenset[str] | None = None
) -> tuple[str, ...]:
    value = obj[field]
    name = f"{where}.{field}"
    if not isinstance(value, list) or not value:
        raise PreregistrationError(f"{name} must be a non-empty list")
    if not all(isinstance(v, str) and v.strip() for v in value):
        raise PreregistrationError(f"{name} must contain non-empty strings only")
    if len(set(value)) != len(value):
        raise PreregistrationError(f"{name} contains duplicates")
    if allowed is not None and (unknown := sorted(set(value) - allowed)):
        raise PreregistrationError(f"{name}: unknown names {unknown}")
    return tuple(value)


def _date(obj: Mapping[str, Any], field: str, where: str) -> dt.date:
    value = obj[field]
    name = f"{where}.{field}"
    if not isinstance(value, str):
        raise PreregistrationError(f"{name} must be a quoted ISO date string, got {value!r}")
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise PreregistrationError(f"{name} is not an ISO date: {value!r}") from exc


def _period(obj: Mapping[str, Any], name: str) -> Period:
    where = f"split.{name}"
    p = _mapping(obj[name], {"start", "end"}, where)
    period = Period(_date(p, "start", where), _date(p, "end", where))
    if period.start > period.end:
        raise PreregistrationError(f"{where}: start is after end")
    return period


def _int(obj: Mapping[str, Any], field: str, where: str, *, minimum: int) -> int:
    value = obj[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PreregistrationError(
            f"{where}.{field} must be an integer >= {minimum}, got {value!r}"
        )
    return value


def _bool(obj: Mapping[str, Any], field: str, where: str) -> bool:
    value = obj[field]
    if not isinstance(value, bool):
        raise PreregistrationError(f"{where}.{field} must be true or false, got {value!r}")
    return value


def _share(obj: Mapping[str, Any], field: str, where: str, *, allow_one: bool = False) -> float:
    value = obj[field]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PreregistrationError(f"{where}.{field} must be a number, got {value!r}")
    if not (0 < value < 1 or (allow_one and value == 1)):
        bound = "(0, 1]" if allow_one else "(0, 1)"
        raise PreregistrationError(f"{where}.{field} must be in {bound}, got {value!r}")
    return float(value)
