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
    assert p.past_rate_history == "full_domains_table"
    assert p.no_history == "missing_with_flag"
    assert p.past_rate_fallback == "train_base_rate"
    assert p.base_rate_rows == "all_train_rows"
    assert p.pr_auc_method == "average_precision"
    assert p.calibration_rows == "validation_cross_fitted"
    assert p.calibration_report_folds == 2
    assert p.threshold_rule == "precision_target_on_validation"
    assert p.report_by_history is True
    assert p.rebalance_eval is False
    assert p.business_at_precision == p.precision_target
    assert p.train.contains(p.stability_train_start)
    assert len(p.inputs) == 22 and "self_filed" in p.inputs
    assert {"requested_at_utc", p.label_counter} <= set(p.excluded_features)
    assert not set(p.inputs) & set(p.excluded_features)
    assert p.request_size_rows == "all"
    assert p.past_rate_unit == "decided_rows" and p.time_basis == "timestamp"
    assert p.tld_rule == "last_label" and p.domain_normalization == "lowercase_trim"
    assert p.ngram_min <= p.ngram_max
    assert p.require_train_requests() == 150_000
    svm = p.linear_svm
    assert (svm.loss, svm.penalty, svm.early_stopping) == ("hinge", "l2", False)
    assert svm.alpha_grid == (5e-4, 5e-5, 5e-6, 5e-7, 5e-8)
    assert (svm.max_epochs, svm.tol, svm.n_iter_no_change) == (50, 1e-4, 5)
    assert svm.tie_break == "larger_alpha_then_no_class_weight"


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
        (["features", "past_rate_history"], "lumen", "is not one of"),
        (["features", "no_history"], "drop_rows", "is not one of"),
        (["models", "past_rate_fallback"], "zero", "is not one of"),
        (["models", "base_rate_rows"], "validation_rows", "is not one of"),
        (["models", "base_rate_rows"], _DROP, "missing fields"),
        (["metrics", "pr_auc_method"], "trapezoid", "is not one of"),
        (["metrics", "pr_auc_method"], _DROP, "missing fields"),
        (["models", "calibration_rows"], "test_rows", "is not one of"),
        (["models", "calibration_rows"], _DROP, "missing fields"),
        (["models", "calibration_report_folds"], 1, "integer >= 2"),
        (["models", "threshold_rule"], "best_f1", "is not one of"),
        (["models", "threshold_rule"], _DROP, "missing fields"),
        (["split", "report_by_history"], _DROP, "missing fields"),
        (["features", "inputs"], ["domain_ngrams", "calendar_month"], "unknown names"),
        (["features", "inputs"], ["domain_past_rate"], "needs has_domain_history"),
        (["features", "inputs"], ["domain_age_days"], "needs has_domain_age"),
        (["features", "domain_ngrams", "min_n"], 6, "min_n <= max_n"),
        (["features", "domain_ngrams", "analyzer"], "word", "is not one of"),
        (["features", "domain_ngrams", "min_df"], 0, "integer >= 1"),
        (["features", "categorical_top_k"], 0, "integer >= 1"),
        (["features", "request_size_rows"], "decided", "is not one of"),
        (["features", "self_filed_level"], "owner_key", "is not one of"),
        (["features", "past_rate_unit"], "urls", "is not one of"),
        (["features", "tld_rule"], "public_suffix", "is not one of"),
        (["features", "time_basis"], "date", "is not one of"),
        (["features", "domain_normalization"], "strip_www", "is not one of"),
        (["features", "time_basis"], _DROP, "missing fields"),
        (["features", "excluded"], ["from_abuser", "url_file"], "requested_at_utc"),
        (["features", "excluded"], ["from_abuser", "requested_at_utc"], "urls_no_action"),
        (["models", "linear_svm_sgd"], _DROP, "missing fields"),
        (["models", "linear_svm_sgd", "loss"], "log_loss", "is not one of"),
        (["models", "linear_svm_sgd", "penalty"], "elasticnet", "is not one of"),
        (["models", "linear_svm_sgd", "alpha_grid"], [], "non-empty list"),
        (["models", "linear_svm_sgd", "alpha_grid"], [1e-4, 0.0], "positive numbers"),
        (["models", "linear_svm_sgd", "alpha_grid"], [1e-4, True], "positive numbers"),
        (["models", "linear_svm_sgd", "alpha_grid"], [1e-4, 1e-4], "duplicates"),
        (["models", "linear_svm_sgd", "max_epochs"], 0, "integer >= 1"),
        (["models", "linear_svm_sgd", "tol"], 0, "positive number"),
        (["models", "linear_svm_sgd", "n_iter_no_change"], 0, "integer >= 1"),
        (["models", "linear_svm_sgd", "early_stopping"], True, "must be false"),
        (["models", "linear_svm_sgd", "early_stopping"], "no", "true or false"),
        (["models", "linear_svm_sgd", "tie_break"], "best_recall", "is not one of"),
        (["models", "linear_svm_sgd", "extra"], 1, "unexpected fields"),
    ],
)
def test_schema_violations_are_rejected(raw, path, value, message):
    with pytest.raises(PreregistrationError, match=message):
        parse_preregistration(_edit(raw, path, value))


def test_non_mapping_document_is_rejected():
    with pytest.raises(PreregistrationError, match="top level must be a mapping"):
        parse_preregistration(["not", "a", "mapping"])


def test_input_that_is_also_excluded_is_rejected(raw):
    excluded = [*raw["features"]["excluded"], "tld"]
    with pytest.raises(PreregistrationError, match=r"also excluded \['tld'\]"):
        parse_preregistration(_edit(raw, ["features", "excluded"], excluded))
