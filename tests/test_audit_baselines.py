"""Tests for the audit baselines on synthetic data, against sklearn and hand-computed values."""

import json
from pathlib import Path

import duckdb
import numpy as np
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss

from piracylens.audit.baselines import (
    Groups,
    _cases,
    batch_metrics,
    domain_matrices,
    metrics,
    ranked,
    run,
)
from piracylens.audit.preregistration import load_preregistration

DESIGN = Path(__file__).resolve().parents[1] / "configs" / "audit_preregistration.yaml"
COLUMNS = (
    "split",
    "domain",
    "has_domain_history",
    "domain_past_rate",
    "label_primary",
    "label_sensitivity",
)


@pytest.fixture
def design():
    return load_preregistration(DESIGN)


def _rowwise(scores, labels, weight=None):
    rank, n_ranks = ranked(np.asarray(scores, dtype=float))
    y = np.asarray(labels, dtype=float)
    w = np.ones(len(y)) if weight is None else np.asarray(weight, dtype=float)
    return metrics(rank, n_ranks, np.asarray(scores, dtype=float), y, 1 - y, w, 0.8)


def _write(path, rows):
    con = duckdb.connect()
    con.execute(
        """CREATE TABLE t (split VARCHAR, domain VARCHAR, has_domain_history BOOLEAN,
           domain_past_rate DOUBLE, label_primary BOOLEAN, label_sensitivity BOOLEAN)"""
    )
    con.executemany("INSERT INTO t VALUES (?, ?, ?, ?, ?, ?)", rows)
    con.execute(f"COPY t TO '{path}' (FORMAT parquet)")
    return path


# Training: 4 rows, primary rate 0.5, sensitivity rate 0.25.
TRAIN = [
    ("train", "a.com", False, None, True, True),
    ("train", "a.com", True, 1.0, True, False),
    ("train", "b.com", False, None, False, False),
    ("train", "c.com", True, 0.0, False, False),
]
# Validation: two domains with history, one without.
VALID = [
    ("validation", "a.com", True, 0.9, True, True),
    ("validation", "a.com", True, 0.9, True, False),
    ("validation", "b.com", True, 0.1, False, False),
    ("validation", "b.com", True, 0.1, True, False),
    ("validation", "d.com", False, None, False, False),
    ("validation", "d.com", False, None, True, True),
]
TEST = [
    ("test", "a.com", True, 0.8, True, True),
    ("test", "e.com", False, None, False, False),
]


def _by(report, label, baseline, subset):
    (hit,) = [
        r
        for r in report["results"]
        if (r["label"], r["baseline"], r["subset"]) == (label, baseline, subset)
    ]
    return hit


def test_pr_auc_matches_sklearn_average_precision_with_ties():
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 6, 400) / 5  # many ties
    labels = rng.random(400) < 0.3 + 0.4 * scores
    ap, _, brier = _rowwise(scores, labels)
    assert ap == pytest.approx(average_precision_score(labels, scores))
    assert brier == pytest.approx(brier_score_loss(labels, scores))


def test_weights_equal_repeated_rows():
    rng = np.random.default_rng(1)
    scores = rng.random(60).round(1)
    labels = rng.random(60) < 0.5
    weight = rng.integers(0, 4, 60)
    ap, rec, brier = _rowwise(scores, labels, weight)
    rep_s, rep_y = np.repeat(scores, weight), np.repeat(labels, weight)
    ap2, rec2, brier2 = _rowwise(rep_s, rep_y)
    assert (ap, rec, brier) == pytest.approx((ap2, rec2, brier2))
    assert ap == pytest.approx(average_precision_score(rep_y, rep_s))


def test_batch_metrics_equal_the_reference_per_resample():
    rng = np.random.default_rng(2)
    n, n_domains = 300, 40
    history = rng.random(n) < 0.6
    rows = rng.integers(1, 5, n).astype(float)
    g = Groups(
        domain=rng.integers(0, n_domains, n),
        history=history,
        past_rate=np.where(history, rng.integers(0, 6, n) / 5, np.nan),
        rows=rows,
        positives={
            "label_primary": np.floor(rows * rng.random(n)),
            "label_sensitivity": np.floor(rows * rng.random(n) * 0.5),
        },
        n_domains=n_domains,
    )
    draws = np.stack(
        [np.bincount(rng.integers(0, n_domains, n_domains), minlength=n_domains)
         for _ in range(7)] + [np.zeros(n_domains)],
        axis=1,
    ).astype(float)  # fmt: skip
    for c in _cases(g, {"label_primary": 0.4, "label_sensitivity": 0.2}):
        batch = batch_metrics(domain_matrices(c, n_domains), draws, 0.8)
        for k in range(draws.shape[1]):
            ref = metrics(
                c["rank"], c["n_ranks"], c["scores"], c["pos"], c["neg"],
                draws[c["domain"], k], 0.8,
            )  # fmt: skip
            np.testing.assert_allclose(batch[k], ref, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_recall_at_precision_by_hand():
    # Thresholds 0.9 -> P 1, R 1/3; 0.5 -> P 1/2, R 2/3; 0.1 -> P 1/2, R 1.
    scores = [0.9, 0.5, 0.5, 0.5, 0.1, 0.1]
    labels = [1, 1, 0, 0, 1, 0]
    ap, rec, _ = _rowwise(scores, labels)
    assert rec == pytest.approx(1 / 3)
    assert ap == pytest.approx(1 / 3 * 1 + 1 / 3 * (2 / 4) + 1 / 3 * (3 / 6))


def test_precision_exactly_at_the_target_counts():
    # Threshold 0.9 flags 4 positives and 1 negative: precision 0.8, recall 4/6.
    scores = [0.9] * 5 + [0.1] * 3
    labels = [1, 1, 1, 1, 0, 1, 1, 0]
    _, rec, _ = _rowwise(scores, labels)
    assert rec == pytest.approx(4 / 6)
    # The batched path, one domain per row and every domain drawn once.
    s, y = np.array(scores, dtype=float), np.array(labels, dtype=float)
    rank, n_ranks = ranked(s)
    case = {
        "rank": rank, "n_ranks": n_ranks, "domain": np.arange(len(s)), "pos": y, "neg": 1 - y,
        "losses": ((s - 1) ** 2, s**2),
    }  # fmt: skip
    batch = batch_metrics(domain_matrices(case, len(s)), np.ones((len(s), 1)), 0.8)
    assert batch[0, 1] == pytest.approx(4 / 6)


def test_constant_score_is_the_label_rate_and_never_reaches_the_target():
    labels = [1, 0, 0, 1, 0]
    ap, rec, _ = _rowwise([0.3] * 5, labels)
    assert ap == pytest.approx(0.4)
    assert rec == 0.0


def test_no_positive_rows_leave_ranking_metrics_undefined():
    ap, rec, brier = _rowwise([0.2, 0.4], [0, 0])
    assert np.isnan(ap) and np.isnan(rec)
    assert brier == pytest.approx((0.04 + 0.16) / 2)


def test_end_to_end_values(tmp_path, design):
    features = _write(tmp_path / "f.parquet", TRAIN + VALID + TEST)
    out = tmp_path / "b.json"
    report = run(design, features, out, resamples=30)
    assert json.loads(out.read_text()) == report
    assert report["train_rows"] == 4
    assert report["train_base_rate"] == {"label_primary": 0.5, "label_sensitivity": 0.25}
    assert len(report["results"]) == 12
    assert report["validation_domains"] == 3

    base = _by(report, "label_primary", "base_rate", "all")
    assert (base["rows"], base["domains"], base["label_rate"]) == (6, 3, pytest.approx(4 / 6))
    assert base["pr_auc"]["value"] == pytest.approx(4 / 6)
    assert base["recall_at_precision"]["value"] == 0.0
    assert base["brier"]["value"] == pytest.approx(0.25)

    # Past rate: 0.9 (2 pos), 0.5 fallback for d.com (1 pos, 1 neg), 0.1 (1 pos, 1 neg).
    past = _by(report, "label_primary", "domain_past_rate", "all")
    assert past["pr_auc"]["value"] == pytest.approx(0.5 * 1 + 0.25 * 3 / 4 + 0.25 * 4 / 6)
    assert past["recall_at_precision"]["value"] == pytest.approx(0.5)
    expected_brier = (2 * 0.01 + 0.25 + 0.25 + 0.81 + 0.01) / 6
    assert past["brier"]["value"] == pytest.approx(expected_brier)

    # Without history the past-rate baseline is the base rate, by construction.
    for label in ("label_primary", "label_sensitivity"):
        a = _by(report, label, "base_rate", "without_history")
        b = _by(report, label, "domain_past_rate", "without_history")
        for name in ("pr_auc", "recall_at_precision", "brier"):
            assert a[name] == b[name]

    sens = _by(report, "label_sensitivity", "domain_past_rate", "with_history")
    assert sens["rows"] == 4 and sens["label_rate"] == pytest.approx(0.25)


def test_intervals_cover_the_point_estimate_and_are_reproducible(tmp_path, design):
    features = _write(tmp_path / "f.parquet", TRAIN + VALID * 5 + TEST)
    first = run(design, features, tmp_path / "1.json", resamples=200)
    second = run(design, features, tmp_path / "2.json", resamples=200)
    assert first["results"] == second["results"]
    for r in first["results"]:
        for name in ("pr_auc", "recall_at_precision", "brier"):
            m = r[name]
            assert m["ci_low"] <= m["ci_high"]
            if r["subset"] == "all" and r["label"] == "label_primary":
                assert m["undefined_resamples"] == 0
                assert m["ci_low"] <= m["value"] <= m["ci_high"] or name == "recall_at_precision"
    # Every domain has a primary positive, so "all" is always defined for that label; b.com has
    # no sensitivity positive, so a resample of b.com only leaves that label undefined.
    # "without_history" holds one domain (d.com), missed by about (2/3)^3 of the resamples,
    # which leaves every metric undefined there.
    wo = _by(first, "label_primary", "base_rate", "without_history")
    for name in ("pr_auc", "recall_at_precision", "brier"):
        assert 20 < wo[name]["undefined_resamples"] < 100


def test_test_labels_never_reach_the_output(tmp_path, design):
    flipped = [(s, d, h, r, not p, not q) for s, d, h, r, p, q in TEST]
    a = run(design, _write(tmp_path / "a.parquet", TRAIN + VALID + TEST), tmp_path / "a.json",
            resamples=20)  # fmt: skip
    b = run(design, _write(tmp_path / "b.parquet", TRAIN + VALID + flipped), tmp_path / "b.json",
            resamples=20)  # fmt: skip
    for key in ("train_base_rate", "train_rows", "results"):
        assert a[key] == b[key]


def test_dry_run_reads_no_label_column(tmp_path, design, capsys):
    path = tmp_path / "nolabels.parquet"
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE t (split VARCHAR, domain VARCHAR, has_domain_history BOOLEAN, "
        "domain_past_rate DOUBLE)"
    )
    con.executemany("INSERT INTO t VALUES (?, ?, ?, ?)", [r[:4] for r in TRAIN + VALID + TEST])
    con.execute(f"COPY t TO '{path}' (FORMAT parquet)")
    out = tmp_path / "never.json"
    assert run(design, path, out, dry_run=True) is None
    assert not out.exists()
    printed = capsys.readouterr().out
    assert "validation groups 3 over 3 domains" in printed
    assert not [line for line in printed.splitlines() if line.strip().startswith("test")]


def test_history_flag_disagreeing_with_past_rate_stops(tmp_path, design):
    bad = [("validation", "x.com", True, None, True, True)]
    features = _write(tmp_path / "f.parquet", TRAIN + VALID + bad)
    with pytest.raises(SystemExit, match="disagree"):
        run(design, features, tmp_path / "b.json", resamples=5)
    assert not (tmp_path / "b.json").exists()
