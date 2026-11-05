"""Tests for the linear audit models on a synthetic feature table with the real columns."""

import dataclasses
import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from piracylens.audit.baselines import metrics, ranked
from piracylens.audit.features import OUTPUT_COLUMNS
from piracylens.audit.models import (
    OTHER,
    AuditEncoder,
    bootstrap,
    cross_fit_probabilities,
    draw_sample,
    linear_svm_spec,
    logistic_regression_spec,
    rank_rows,
    row_metrics,
    run,
    select_winner,
)
from piracylens.audit.preregistration import load_preregistration

DESIGN = Path(__file__).resolve().parents[1] / "configs" / "audit_preregistration.yaml"
SPLITS = {"train": (2023, 60), "validation": (2025, 30), "test": (2026, 20)}


@pytest.fixture
def design():
    p = load_preregistration(DESIGN)
    return dataclasses.replace(p, train_requests=40, ngram_min_df=1, bootstrap_resamples=20)


def _table(seed=0, *, flip_test=False, labels=True):
    """Feature-table rows: whole requests per split; the label leans on the domain."""
    rng = np.random.default_rng(seed)
    pool = [f"site{k}.example" for k in range(12)] + ["xn--80ak6aa92e.com", "free-movies-1.net"]
    lean = {d: (k % 4) / 4 for k, d in enumerate(pool)}
    rows, rid = [], 0
    for split, (year, n_req) in SPLITS.items():
        for _ in range(n_req):
            rid += 1
            for d in rng.choice(pool, size=rng.integers(1, 4), replace=False):
                hist = bool(rng.random() < 0.6)
                label = bool(rng.random() < 0.1 + 0.7 * lean[d])
                if flip_test and split == "test":
                    label = not label
                row = {
                    "request_id": rid,
                    "domain": d,
                    "requested_at_utc": f"{year}-03-0{1 + rid % 8} 10:00:00",
                    "split": split,
                    "tld": d.rsplit(".", 1)[-1],
                    "domain_length": len(d),
                    "domain_labels": d.count(".") + 1,
                    "domain_digit_share": sum(ch.isdigit() for ch in d) / len(d),
                    "domain_hyphens": d.count("-"),
                    "domain_punycode": d.startswith("xn--"),
                    "domains_in_request": 3,
                    "urls_in_request": int(rng.integers(1, 500)),
                    "urls_for_domain": int(rng.integers(1, 50)),
                    "owner_group": ["studio", "channel", "label"][rid % 3],
                    "market": ["us", "ru"][rid % 2],
                    "content_type": "film",
                    "region": None if rid % 7 == 0 else "us",
                    "reporter_group": ["agency", "studio"][rid % 2],
                    "reporter_brand": ["agency x", "studio a"][rid % 2],
                    "self_filed": rid % 2 == 1,
                    "domain_past_rate": float(lean[d]) if hist else None,
                    "has_domain_history": hist,
                    "domain_past_requests": int(rng.integers(1, 9)) if hist else 0,
                    "domain_age_days": float(rng.integers(1, 900)) if rid % 5 else None,
                    "has_domain_age": rid % 5 != 0,
                    "label_primary": label,
                    "label_sensitivity": label,
                }
                rows.append(row)
    columns = [c for c in OUTPUT_COLUMNS if labels or not c.startswith("label_")]
    return {c: [r[c] for r in rows] for c in columns}


def _write(path, table):
    con = duckdb.connect()
    con.register("t", {k: np.array(v, dtype=object) for k, v in table.items()})
    casts = {
        "requested_at_utc": "TIMESTAMP",
        "domain_past_rate": "DOUBLE",
        "domain_age_days": "DOUBLE",
        "domain_digit_share": "DOUBLE",
    }
    select = ", ".join(f"{c}::{casts[c]} AS {c}" if c in casts else c for c in table)
    con.execute(f"COPY (SELECT {select} FROM t) TO '{path}' (FORMAT parquet)")
    return path


def _cols(**overrides):
    base = {
        "domain": np.array(["a.com", "b.net", "a.com", "c.org"], dtype=object),
        "tld": np.array(["com", "net", "com", "org"], dtype=object),
        "owner_group": np.array(["x", "y", "x", "z"], dtype=object),
        "reporter_group": np.array(["r", "r", "s", "r"], dtype=object),
        "reporter_brand": np.array(["r", "r", "s", "r"], dtype=object),
        "market": np.array(["us", "ru", "us", "us"], dtype=object),
        "content_type": np.array(["film"] * 4, dtype=object),
        "region": np.array(["us", "ru", "us", "us"], dtype=object),
        "domains_in_request": np.array([1.0, 2, 3, 4]),
        "urls_in_request": np.array([1.0, 10, 100, 1000]),
        "urls_for_domain": np.array([1.0, 1, 2, 3]),
        "domain_past_requests": np.array([0.0, 2, 3, 0]),
        "domain_age_days": np.array([np.nan, 5, 10, 20]),
        "domain_length": np.array([5.0, 5, 5, 5]),
        "domain_labels": np.array([2.0, 2, 2, 2]),
        "domain_digit_share": np.array([0.0, 0, 0, 0]),
        "domain_hyphens": np.array([0.0, 0, 0, 0]),
        "domain_past_rate": np.array([np.nan, 0.5, 0.25, np.nan]),
        "domain_punycode": np.array([False] * 4),
        "self_filed": np.array([True, False, False, True]),
        "has_domain_history": np.array([False, True, True, False]),
        "has_domain_age": np.array([False, True, True, True]),
    }
    base.update(overrides)
    return base


def test_sample_is_whole_training_requests_and_reproducible(tmp_path, design):
    features = _write(tmp_path / "f.parquet", _table())
    con = duckdb.connect()
    a = draw_sample(con, features, 40, design.seed)
    b = draw_sample(con, features, 40, design.seed)
    assert np.array_equal(a, b) and len(np.unique(a)) == 40
    train_ids = set(range(1, SPLITS["train"][1] + 1))
    assert set(a.tolist()) <= train_ids
    with pytest.raises(SystemExit, match="only 60 training requests"):
        draw_sample(con, features, 61, design.seed)


def test_encoder_fills_and_encodes_from_training_rows_only():
    enc = AuditEncoder(3, 5, 1, True, top_k=1, base_rate=0.2).fit(_cols())
    assert enc.categories["owner_group"] == ["x", OTHER]  # x is most frequent
    assert enc.categories["tld"] == ["com", OTHER]
    num = enc._numeric(_cols())
    assert num[0, 4] == 0.0  # missing age -> log1p(0)
    assert num[0, -1] == pytest.approx(0.2) and num[3, -1] == pytest.approx(0.2)
    assert num[1, -1] == pytest.approx(0.5)
    assert num[3, 1] == pytest.approx(np.log1p(1000))
    x = enc.transform(_cols())
    assert x.shape == (4, enc.n_columns)
    # Scaling uses the training mean: training rows average to zero in every numeric column.
    block = x[:, enc.n_columns - 4 - 10 : enc.n_columns - 4].toarray()
    np.testing.assert_allclose(block.mean(axis=0), 0, atol=1e-6)
    new = _cols(
        owner_group=np.array(["unseen"] * 4, dtype=object),
        market=np.array(["jp"] * 4, dtype=object),
    )
    y = enc.transform(new).toarray()
    start = len(enc.ngrams.vocabulary_) + 2  # after the tld block
    assert np.array_equal(y[:, start : start + 2], [[0, 1]] * 4)  # unseen owner -> other
    market_start = start + 2 + 2 + 2  # owner, reporter group, reporter brand blocks
    assert y[:, market_start : market_start + 2].sum() == 0  # unseen market -> no column


def test_encoder_refuses_to_transform_before_fit():
    with pytest.raises(RuntimeError, match="not fitted"):
        AuditEncoder(3, 5, 1, True, 1, 0.2).transform(_cols())


def test_row_metrics_match_the_reference_with_ties_and_weights():
    rng = np.random.default_rng(3)
    s = rng.integers(0, 7, 300) / 6
    y = (rng.random(300) < 0.2 + 0.5 * s).astype(float)
    domain = rng.integers(0, 50, 300)
    w = rng.integers(0, 3, 50)[domain].astype(float)
    r = rank_rows(s, y, s, domain)
    order = np.argsort(-s, kind="stable")
    got = row_metrics(r, w[order], 0.8)[:3]
    rank, n_ranks = ranked(s)
    ref = metrics(rank, n_ranks, s, y, 1 - y, w, 0.8)
    np.testing.assert_allclose(got, ref, rtol=1e-12)


def test_bootstrap_does_not_depend_on_the_number_of_workers(tmp_path, design):
    rng = np.random.default_rng(5)
    n, n_domains = 500, 40
    domain = rng.integers(0, n_domains, n)
    s = rng.random(n)
    y = (rng.random(n) < s).astype(float)
    cases = {("model", "all"): rank_rows(s, y, s, domain)}
    one = bootstrap(cases, n_domains, design, 30, workers=1)
    many = bootstrap(cases, n_domains, design, 30, workers=3, scratch=tmp_path)
    np.testing.assert_array_equal(one[("model", "all")], many[("model", "all")])
    assert list(tmp_path.iterdir()) == []  # the memory-mapped scratch folder is removed


def test_threshold_is_the_highest_recall_at_the_target_or_none():
    s = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.5, 0.5, 0.1])
    y = np.array([1, 1, 1, 1, 0, 1, 0, 0], dtype=float)
    r = rank_rows(s, y, s, np.arange(8))
    _, rec, _, idx = row_metrics(r, np.ones(8), 0.8)
    assert rec == pytest.approx(4 / 5) and r.thresholds[idx] == 0.9
    _, rec, _, idx = row_metrics(r, np.ones(8), 0.95)
    assert rec == 0.0 and idx == -1


def test_cross_fitted_probabilities_never_use_their_own_fold():
    rng = np.random.default_rng(4)
    n_domains, n = 30, 400
    domain = rng.integers(0, n_domains, n)
    s = rng.random(n)
    y = (rng.random(n) < s).astype(float)
    prob = cross_fit_probabilities(s, y, domain, n_domains, 2, seed=7)
    fold = (np.random.default_rng(7).permutation(n_domains) % 2)[domain]
    for k in (0, 1):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(s[fold != k], y[fold != k])
        np.testing.assert_allclose(prob[fold == k], iso.predict(s[fold == k]))


def test_end_to_end(tmp_path, design):
    features = _write(tmp_path / "f.parquet", _table())
    report = run(
        design, features, tmp_path / "sample.parquet", tmp_path / "out",
        grid_c=(0.1, 1.0), class_weights=(None, "balanced"), chunk_rows=37,
    )  # fmt: skip
    saved = json.loads((tmp_path / "out" / "logreg.json").read_text())
    assert saved["results"] == report["results"]
    assert (tmp_path / "out" / "logreg.joblib").exists()
    assert len(report["grid"]) == 4 and report["sample"]["requests"] == 40
    winner = report["winner"]
    best = max(g["validation_pr_auc"] for g in report["grid"])
    assert any(
        (g["C"], g["class_weight"]) == (winner["C"], winner["class_weight"])
        and g["validation_pr_auc"] == best
        for g in report["grid"]
    )
    rows = {e["subset"]: e["rows"] for e in report["results"]}
    assert (
        rows["all"] == rows["with_history"] + rows["without_history"] == report["validation_rows"]
    )
    for e in report["results"]:
        for metric in ("pr_auc", "brier"):
            m = e[metric]
            if m["model"]["value"] is not None and m["domain_past_rate"]["value"] is not None:
                assert m["difference"]["value"] == pytest.approx(
                    m["model"]["value"] - m["domain_past_rate"]["value"]
                )


def test_test_labels_never_reach_the_output(tmp_path, design):
    kwargs = {"grid_c": (1.0,), "class_weights": (None,)}
    a = run(design, _write(tmp_path / "a.parquet", _table()), tmp_path / "sa.parquet",
            tmp_path / "a", **kwargs)  # fmt: skip
    b = run(design, _write(tmp_path / "b.parquet", _table(flip_test=True)),
            tmp_path / "sb.parquet", tmp_path / "b", **kwargs)  # fmt: skip
    for key in ("results", "threshold", "winner", "train_base_rate_all_rows", "sample"):
        if key == "sample":
            assert a[key]["requests"] == b[key]["requests"] and a[key]["rows"] == b[key]["rows"]
        else:
            assert a[key] == b[key]
    assert [g["validation_pr_auc"] for g in a["grid"]] == [
        g["validation_pr_auc"] for g in b["grid"]
    ]


def test_dry_run_reads_no_label_column(tmp_path, design, capsys):
    features = _write(tmp_path / "f.parquet", _table(labels=False))
    out = tmp_path / "out"
    assert run(design, features, tmp_path / "s.parquet", out, dry_run=True, chunk_rows=25) is None
    assert not out.exists()
    printed = capsys.readouterr().out
    assert "sample: 40 training requests" in printed and "no label read" in printed


def test_design_inputs_must_match_the_encoding(tmp_path, design):
    p = dataclasses.replace(design, inputs=tuple(i for i in design.inputs if i != "tld"))
    with pytest.raises(SystemExit, match="does not list"):
        run(p, tmp_path / "missing.parquet", tmp_path / "s.parquet", tmp_path / "o")


def _lr_then(tmp_path, design, features, out, **svm):
    """Run the logistic regression, then the linear SVM against it, in one output folder."""
    sample = tmp_path / f"{out}-sample.parquet"
    lr = run(design, features, sample, tmp_path / out, grid_c=(0.1, 1.0), class_weights=(None,))
    kw = {"alpha_grid": (1e-2, 1e-3), "class_weights": (None, "balanced"), "chunk_rows": 37}
    svm_report = run(design, features, sample, tmp_path / out, model="linear_svm", **(kw | svm))
    return lr, svm_report


def test_linear_svm_is_paired_with_the_stored_logistic_regression(tmp_path, design):
    features = _write(tmp_path / "f.parquet", _table())
    lr, report = _lr_then(tmp_path, design, features, "out")
    saved = json.loads((tmp_path / "out" / "linear_svm.json").read_text())
    assert saved["results"] == report["results"] and report["model"] == "linear_svm_sgd"
    assert (tmp_path / "out" / "linear_svm.joblib").exists()
    assert (tmp_path / "out" / "logreg.json").exists()  # the reference report is untouched
    assert [(g["alpha"], g["class_weight"]) for g in report["grid"]] == [
        (1e-2, None), (1e-2, "balanced"), (1e-3, None), (1e-3, "balanced"),
    ]  # fmt: skip
    assert set(report["winner"]) == {"alpha", "class_weight"}
    assert report["sample"] == lr["sample"] and report["reference"]["sha256"]
    stored = {e["subset"]: e for e in lr["results"]}
    for e in report["results"]:
        for metric in ("pr_auc", "recall_at_precision", "brier"):
            m = e[metric]
            # The logistic regression column reproduces its own run on the same resamples.
            assert m["logistic_regression"] == stored[e["subset"]][metric]["model"]
            assert m["domain_past_rate"] == stored[e["subset"]][metric]["domain_past_rate"]
            if m["model"]["value"] is not None and m["logistic_regression"]["value"] is not None:
                assert m["difference_vs_logistic_regression"]["value"] == pytest.approx(
                    m["model"]["value"] - m["logistic_regression"]["value"]
                )


def test_ties_go_to_the_stronger_regularization_then_no_weighting():
    def tied(spec):
        return [{**g, "validation_pr_auc": 0.5} for g in spec.grid]

    svm = linear_svm_spec(load_preregistration(DESIGN), alpha_grid=(1e-6, 1e-4, 1e-5))
    grid = tied(svm)
    winner = grid[select_winner(grid, svm.prefer)]
    assert (winner["alpha"], winner["class_weight"]) == (1e-4, None)
    grid[0]["validation_pr_auc"] = 0.6  # a strictly better fit wins regardless of the tie-break
    assert select_winner(grid, svm.prefer) == 0
    lr = logistic_regression_spec(grid_c=(1.0, 0.1, 10.0))
    grid = tied(lr)
    winner = grid[select_winner(grid, lr.prefer)]
    assert (winner["C"], winner["class_weight"]) == (0.1, None)


def test_linear_svm_test_labels_never_reach_the_output(tmp_path, design):
    a = _lr_then(tmp_path, design, _write(tmp_path / "a.parquet", _table()), "a")[1]
    b = _lr_then(tmp_path, design, _write(tmp_path / "b.parquet", _table(flip_test=True)), "b")[1]
    for key in ("results", "threshold", "winner", "train_base_rate_all_rows"):
        assert a[key] == b[key]
    assert [g["validation_pr_auc"] for g in a["grid"]] == [
        g["validation_pr_auc"] for g in b["grid"]
    ]


def test_linear_svm_needs_the_same_encoding_as_its_reference(tmp_path, design):
    features = _write(tmp_path / "f.parquet", _table())
    sample = tmp_path / "s.parquet"
    run(design, features, sample, tmp_path / "out", grid_c=(1.0,), class_weights=(None,))
    other = dataclasses.replace(design, ngram_max=4)
    with pytest.raises(SystemExit, match="encoding differs"):
        run(other, features, sample, tmp_path / "out", model="linear_svm", alpha_grid=(1e-3,))
    with pytest.raises(SystemExit, match="encoding differs"):
        run(other, features, sample, tmp_path / "out", model="linear_svm", dry_run=True)
    with pytest.raises(SystemExit, match="reference model not found"):
        run(design, features, sample, tmp_path / "empty", model="linear_svm")


def test_linear_svm_needs_the_reference_training_matrix(tmp_path, design):
    features = _write(tmp_path / "f.parquet", _table())
    sample = tmp_path / "s.parquet"
    run(design, features, sample, tmp_path / "out", grid_c=(1.0,), class_weights=(None,))
    path = tmp_path / "out" / "logreg.joblib"
    stored = joblib.load(path)
    stored["encoder"].mean = stored["encoder"].mean + 1.0  # same vocabulary, other scaling
    joblib.dump(stored, path)
    with pytest.raises(SystemExit, match="training matrix differs"):
        run(design, features, sample, tmp_path / "out", model="linear_svm", alpha_grid=(1e-3,))


def test_linear_svm_stops_when_the_reference_does_not_reproduce(tmp_path, design):
    features = _write(tmp_path / "f.parquet", _table())
    sample = tmp_path / "s.parquet"
    run(design, features, sample, tmp_path / "out", grid_c=(1.0,), class_weights=(None,))
    path = tmp_path / "out" / "logreg.json"
    stored = json.loads(path.read_text())
    stored["results"][0]["pr_auc"]["model"]["value"] += 0.01
    path.write_text(json.dumps(stored))
    with pytest.raises(SystemExit, match="does not reproduce its report: all pr_auc"):
        run(design, features, sample, tmp_path / "out", model="linear_svm", alpha_grid=(1e-3,))
    assert not (tmp_path / "out" / "linear_svm.json").exists()


def test_a_stored_sample_must_equal_the_draw(tmp_path, design):
    features = _write(tmp_path / "f.parquet", _table())
    sample = tmp_path / "s.parquet"
    con = duckdb.connect()
    con.execute(f"COPY (SELECT range::BIGINT AS request_id FROM range(1, 41)) TO '{sample}'")
    with pytest.raises(SystemExit, match="different sample"):
        run(design, features, sample, tmp_path / "out", dry_run=True)


def test_model_specs_follow_the_fixed_settings(design):
    lr = logistic_regression_spec()
    assert len(lr.grid) == 8 and lr.stem == "logreg"
    params = lr.build({"C": 0.01, "class_weight": None}).get_params()
    assert (params["C"], params["solver"], params["max_iter"]) == (0.01, "lbfgs", 1000)
    assert max(lr.grid, key=lr.prefer) == {"C": 0.01, "class_weight": None}
    svm = linear_svm_spec(design)
    assert len(svm.grid) == 10 and svm.stem == "linear_svm" and svm.max_iter == 50
    assert [g["alpha"] for g in svm.grid[::2]] == list(design.linear_svm.alpha_grid)
    params = svm.build({"alpha": 5e-4, "class_weight": "balanced"}).get_params()
    expected = {
        "loss": "hinge", "penalty": "l2", "alpha": 5e-4, "class_weight": "balanced",
        "max_iter": 50, "tol": 1e-4, "n_iter_no_change": 5, "early_stopping": False,
        "random_state": design.seed,
    }  # fmt: skip
    assert {k: params[k] for k in expected} == expected
    assert max(svm.grid, key=svm.prefer) == {"alpha": 5e-4, "class_weight": None}
