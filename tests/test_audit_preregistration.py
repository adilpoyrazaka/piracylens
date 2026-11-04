"""Tests for the pre-registered audit design file and its loader."""

import copy
import datetime as dt
from pathlib import Path

import pytest
import yaml

from piracylens.audit.preregistration import (
    PreregistrationError,
    load_preregistration,
    parse_preregistration,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DESIGN = REPO_ROOT / "configs" / "audit_preregistration.yaml"


@pytest.fixture
def raw():
    with DESIGN.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _edit(doc, path, value):
    doc = copy.deepcopy(doc)
    node = doc
    for key in path[:-1]:
        node = node[key]
    if value is _DROP:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return doc


_DROP = object()


def test_repo_design_file_is_valid():
    p = load_preregistration(DESIGN)
    assert p.label_counter == "urls_no_action" and p.label_min_count == 1
    assert p.train.start == p.window_start == dt.date(2021, 1, 1)
    assert p.test.end == p.data_end
    assert p.validation.start == p.train.end + dt.timedelta(days=1)
    assert p.test.start == p.validation.end + dt.timedelta(days=1)
    assert "from_abuser" in p.excluded_features
    assert p.rebalance_eval is False
    assert p.business_at_precision == p.precision_target
    assert p.train.contains(p.stability_train_start)


def test_empty_training_size_stops_training(raw):
    p = parse_preregistration(_edit(raw, ["sampling", "train_requests"], None))
    with pytest.raises(PreregistrationError, match="train_requests is empty"):
        p.require_train_requests()


def test_filled_training_size_is_returned(raw):
    p = parse_preregistration(_edit(raw, ["sampling", "train_requests"], 250_000))
    assert p.require_train_requests() == 250_000


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (["label", "extra"], 1, "unexpected fields"),
        (["metrics", "precision_target"], _DROP, "missing fields"),
        (["split", "validation", "start"], "2025-01-02", "without gaps"),
        (["split", "test", "end"], "2026-10-23", "test.end must equal"),
        (["split", "train", "start"], "2021-01-02", "train.start must equal"),
        (["data", "history_start"], "2021-06-01", "history_start < window_start"),
        (["data", "window_start"], "2021/01/01", "not an ISO date"),
        (["label", "primary", "min_count"], True, "integer"),
        (["label", "primary", "counter"], "urls_pending", "decided counters"),
        (["label", "sensitivity_min_share"], 1.5, r"\(0, 1\]"),
        (["metrics", "precision_target"], 1, r"\(0, 1\)"),
        (["models", "candidates"], ["xgboost", "random_forest"], "unknown names"),
        (["models", "candidates"], ["xgboost", "xgboost"], "duplicates"),
        (["models", "imbalance_variants"], ["class_weight"], "include none"),
        (["sampling", "train_requests"], 0, "integer >= 1"),
        (["sampling", "rebalance_eval"], True, "real rate"),
        (["features", "excluded"], ["url_file"], "from_abuser"),
        (["business_metric", "at_precision"], 0.9, "must equal metrics.precision_target"),
        (["checks", "stability_train_start"], "2025-03-01", "inside the train period"),
        (["checks", "source_subset"], ["source_e"], "unknown names"),
        (["checks", "reporter_ablation"], "yes", "true or false"),
    ],
)
def test_schema_violations_are_rejected(raw, path, value, message):
    with pytest.raises(PreregistrationError, match=message):
        parse_preregistration(_edit(raw, path, value))


def test_non_mapping_document_is_rejected():
    with pytest.raises(PreregistrationError, match="top level must be a mapping"):
        parse_preregistration(["not", "a", "mapping"])
