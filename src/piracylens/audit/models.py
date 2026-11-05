"""Logistic regression for the takedown audit model, tuned and evaluated on validation.

The training sample is a fixed number of whole training requests drawn uniformly without
replacement with the design seed; the drawn request ids are written to a file so every model
trains on the same rows. Every fitted part of the input encoding is fitted on the sample rows
only: the domain character n-gram vocabulary, the most frequent categories (the rest and unseen
values become one "other" column) and the scaling of numeric inputs. Counts and the domain age
are log-transformed before scaling. A missing past rate is filled with the training base rate
(the label mean over every training row); a missing domain age, which happens only at a domain's
first appearance, is filled with zero days and keeps its flag.

A small grid of regularization strengths and class weightings is fitted on the same encoded
sample. The winner has the highest validation PR-AUC; ties go to the smaller strength, then to
no class weighting. Validation metrics of the winner are therefore optimistic (they were used
for selection) and are labelled as such; the test set is not read here.

Ranking metrics (PR-AUC as average precision, recall at the precision target) use the raw
model scores. The Brier score uses isotonic-calibrated probabilities that are cross-fitted on
validation domain folds, so no row is scored by a calibrator that saw it; a final calibrator
fitted on every validation row is stored with the model. The decision threshold is the score
whose threshold reaches the precision target with the highest recall on validation, or none if
the target is not reached. Intervals come from domain-level bootstrap resamples with the design
seed, shared with a row-level evaluation of the domain past-rate baseline, so the difference
between the model and that baseline has a paired interval.

Only training and validation rows are read; test rows are filtered out in every query.

Usage (from the repo root):
    python -m piracylens.audit.models --dry-run   # sample, encoding size and timing; no label
    python -m piracylens.audit.models
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import resource
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from piracylens.audit.preregistration import Preregistration, load_preregistration

DEFAULT_FEATURES = Path("data/interim/audit_features.parquet")
DEFAULT_SAMPLE = Path("data/interim/audit_train_sample.parquet")
DEFAULT_OUT_DIR = Path("data/interim/audit_models")

# Grid for the logistic regression (heuristic values).
GRID_C = (0.01, 0.1, 1.0, 10.0)
CLASS_WEIGHTS = (None, "balanced")
MAX_ITER = 1000
CHUNK_ROWS = 1_000_000
# Worker processes for the bootstrap; each holds its own temporaries of about 0.5 GB, and
# the shared arrays are memory-mapped from a temporary folder in the output directory.
BOOTSTRAP_WORKERS = 8

TOP_K_COLUMNS = ("tld", "owner_group", "reporter_group", "reporter_brand")
ONE_HOT_COLUMNS = ("market", "content_type", "region")
LOG_COLUMNS = (
    "domains_in_request",
    "urls_in_request",
    "urls_for_domain",
    "domain_past_requests",
    "domain_age_days",
)
PLAIN_COLUMNS = (
    "domain_length",
    "domain_labels",
    "domain_digit_share",
    "domain_hyphens",
    "domain_past_rate",
)
FLAG_COLUMNS = ("domain_punycode", "self_filed", "has_domain_history", "has_domain_age")
STRING_COLUMNS = ("domain", *TOP_K_COLUMNS, *ONE_HOT_COLUMNS)
NUMERIC_COLUMNS = (*LOG_COLUMNS, *PLAIN_COLUMNS)
# The design input names this module encodes; checked against the design file.
ENCODED_INPUTS = frozenset(
    {"domain_ngrams", *TOP_K_COLUMNS, *ONE_HOT_COLUMNS, *NUMERIC_COLUMNS, *FLAG_COLUMNS}
)
OTHER = "__other__"
MISSING = "__missing__"


def _select(columns: tuple[str, ...]) -> str:
    """SQL select list: strings without NULL, numerics with NaN for NULL, flags as booleans."""
    parts = []
    for c in columns:
        if c in STRING_COLUMNS:
            parts.append(f"coalesce({c}, '{MISSING}') AS {c}")
        elif c in NUMERIC_COLUMNS:
            parts.append(f"coalesce({c}::DOUBLE, 'NaN'::DOUBLE) AS {c}")
        else:
            parts.append(c)
    return ", ".join(parts)


INPUT_SELECT = (*STRING_COLUMNS, *NUMERIC_COLUMNS, *FLAG_COLUMNS)


@dataclass
class AuditEncoder:
    """Turns the raw input columns into one sparse matrix; fitted on training rows only."""

    ngram_min: int
    ngram_max: int
    ngram_min_df: int
    ngram_sublinear_tf: bool
    top_k: int
    base_rate: float
    ngrams: TfidfVectorizer | None = None
    categories: dict[str, list[str]] = field(default_factory=dict)
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None

    def _numeric(self, cols: dict[str, np.ndarray]) -> np.ndarray:
        past = np.where(
            np.isnan(cols["domain_past_rate"]), self.base_rate, cols["domain_past_rate"]
        )
        age = np.nan_to_num(cols["domain_age_days"], nan=0.0)
        out = []
        for c in LOG_COLUMNS:
            v = age if c == "domain_age_days" else cols[c]
            out.append(np.log1p(np.asarray(v, dtype=float)))
        for c in PLAIN_COLUMNS:
            out.append(past if c == "domain_past_rate" else np.asarray(cols[c], dtype=float))
        return np.column_stack(out)

    def fit(self, cols: dict[str, np.ndarray]) -> AuditEncoder:
        self.ngrams = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(self.ngram_min, self.ngram_max),
            min_df=self.ngram_min_df,
            sublinear_tf=self.ngram_sublinear_tf,
            dtype=np.float32,
        ).fit(cols["domain"])
        for c in TOP_K_COLUMNS:
            counts = Counter(cols[c].tolist())
            # Most frequent first; equal counts in name order, so the choice is deterministic.
            ranked = sorted(counts, key=lambda v: (-counts[v], v))
            self.categories[c] = [*ranked[: self.top_k], OTHER]
        for c in ONE_HOT_COLUMNS:
            self.categories[c] = sorted(set(cols[c].tolist()))
        num = self._numeric(cols)
        self.mean = num.mean(axis=0)
        std = num.std(axis=0)
        self.scale = np.where(std > 0, std, 1.0)
        return self

    def _one_hot(self, values: np.ndarray, cats: list[str]) -> sparse.csr_matrix:
        index = {v: i for i, v in enumerate(cats)}
        other = index.get(OTHER)
        cols = np.array([index.get(v, -1 if other is None else other) for v in values.tolist()])
        rows = np.arange(len(values))
        keep = cols >= 0  # a value unseen in training without an "other" column gets no column
        return sparse.csr_matrix(
            (np.ones(keep.sum(), dtype=np.float32), (rows[keep], cols[keep])),
            shape=(len(values), len(cats)),
        )

    def transform(self, cols: dict[str, np.ndarray]) -> sparse.csr_matrix:
        if self.ngrams is None or self.mean is None or self.scale is None:
            raise RuntimeError("the encoder is not fitted")
        blocks = [self.ngrams.transform(cols["domain"])]
        for c in (*TOP_K_COLUMNS, *ONE_HOT_COLUMNS):
            blocks.append(self._one_hot(cols[c], self.categories[c]))
        blocks.append(sparse.csr_matrix((self._numeric(cols) - self.mean) / self.scale))
        flags = np.column_stack([np.asarray(cols[c], dtype=np.float32) for c in FLAG_COLUMNS])
        blocks.append(sparse.csr_matrix(flags))
        return sparse.hstack(blocks, format="csr", dtype=np.float32)

    @property
    def n_columns(self) -> int:
        n = len(self.ngrams.vocabulary_) if self.ngrams is not None else 0
        return n + sum(len(v) for v in self.categories.values()) + len(NUMERIC_COLUMNS) + 4


def draw_sample(con: duckdb.DuckDBPyConnection, features: Path, n: int, seed: int) -> np.ndarray:
    """Request ids of `n` whole training requests, uniform without replacement (sorted)."""
    ids = con.execute(
        """SELECT DISTINCT request_id FROM read_parquet(?) WHERE split = 'train'
           ORDER BY request_id""",
        [str(features)],
    ).fetchnumpy()["request_id"]
    if len(ids) < n:
        raise SystemExit(f"only {len(ids):,} training requests, the design asks for {n:,}")
    rng = np.random.default_rng(seed)
    return np.sort(ids[rng.choice(len(ids), size=n, replace=False)])


def write_sample(con: duckdb.DuckDBPyConnection, ids: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    con.register("sample_ids", {"request_id": ids})
    con.execute(f"COPY (SELECT request_id FROM sample_ids) TO '{tmp}' (FORMAT parquet)")
    con.unregister("sample_ids")
    n = con.execute("SELECT count(*) FROM read_parquet(?)", [str(tmp)]).fetchone()[0]
    if n != len(ids):
        tmp.unlink()
        raise SystemExit("sample file row count differs from the draw")
    tmp.replace(path)


def read_sample_rows(
    con: duckdb.DuckDBPyConnection, features: Path, sample: Path, *, labels: bool
) -> dict[str, np.ndarray]:
    extra = ", label_primary" if labels else ""
    return con.execute(
        f"""SELECT {_select(INPUT_SELECT)}{extra}
            FROM read_parquet(?) AS f SEMI JOIN read_parquet(?) AS s USING (request_id)
            WHERE f.split = 'train' ORDER BY request_id, domain""",
        [str(features), str(sample)],
    ).fetchnumpy()


def stage_validation(con: duckdb.DuckDBPyConnection, features: Path, *, labels: bool) -> int:
    """Validation rows in a temp table with a row number and a dense domain id."""
    extra = ", label_primary, has_domain_history AS history_flag" if labels else ""
    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE val AS
            SELECT (row_number() OVER (ORDER BY domain, request_id) - 1)::BIGINT AS rn,
                   (dense_rank() OVER (ORDER BY domain) - 1)::BIGINT AS did,
                   {_select(INPUT_SELECT)}{extra}
            FROM read_parquet(?) WHERE split = 'validation'""",
        [str(features)],
    )
    return con.execute("SELECT count(*) FROM val").fetchone()[0]


def validation_chunks(con: duckdb.DuckDBPyConnection, n: int, rows: int = CHUNK_ROWS):
    for start in range(0, n, rows):
        yield con.execute(
            "SELECT * FROM val WHERE rn >= ? AND rn < ? ORDER BY rn", [start, start + rows]
        ).fetchnumpy()


# Row-level evaluation ---------------------------------------------------------------------


@dataclass(frozen=True)
class Ranked:
    """One score over one subset, sorted once: thresholds are the ends of tied-score blocks."""

    domain: np.ndarray
    y: np.ndarray
    squared_error: np.ndarray  # (probability - label) squared, for the Brier score
    ends: np.ndarray
    thresholds: np.ndarray


def rank_rows(score: np.ndarray, y: np.ndarray, prob: np.ndarray, domain: np.ndarray) -> Ranked:
    order = np.argsort(-score, kind="stable")
    s = score[order]
    ends = np.flatnonzero(np.r_[s[1:] != s[:-1], True])
    y_sorted = y[order]
    return Ranked(domain[order], y_sorted, (prob[order] - y_sorted) ** 2, ends, s[ends])


def row_metrics(r: Ranked, weight: np.ndarray, target: float) -> tuple[float, float, float, int]:
    """PR-AUC, recall at the target, Brier and the index of the chosen threshold (-1: none)."""
    total = weight.sum()
    if total == 0:
        return float("nan"), float("nan"), float("nan"), -1
    brier = float(weight @ r.squared_error / total)
    tp = np.cumsum(weight * r.y)[r.ends]
    fp = np.cumsum(weight)[r.ends] - tp
    n_pos = tp[-1]
    if n_pos == 0:
        return float("nan"), float("nan"), brier, -1
    flagged = tp + fp
    precision = np.divide(tp, flagged, out=np.zeros_like(tp), where=flagged > 0)
    recall = tp / n_pos
    ap = float(np.sum(np.diff(recall, prepend=0.0) * precision))
    reached = np.flatnonzero(precision >= target)
    if len(reached) == 0:
        return ap, 0.0, brier, -1
    best = reached[np.argmax(recall[reached])]
    return ap, float(recall[best]), brier, int(best)


def cross_fit_probabilities(
    score: np.ndarray, y: np.ndarray, domain: np.ndarray, n_domains: int, folds: int, seed: int
) -> np.ndarray:
    """Isotonic probabilities where each domain fold is scored by a calibrator fitted on the rest."""
    fold_of_domain = np.random.default_rng(seed).permutation(n_domains) % folds
    fold = fold_of_domain[domain]
    out = np.empty(len(score))
    for k in range(folds):
        test = fold == k
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(score[~test], y[~test])
        out[test] = iso.predict(score[test])
    return out


def _resample_block(
    cases: dict[tuple[str, str], Ranked], draws: np.ndarray, start: int, stop: int, target: float
) -> dict[tuple[str, str], np.ndarray]:
    out = {k: np.full((stop - start, 3), np.nan) for k in cases}
    for b in range(start, stop):
        weight = draws[b].astype(float)
        for k, r in cases.items():
            out[k][b - start] = row_metrics(r, weight[r.domain], target)[:3]
    return out


_FIELDS = ("domain", "y", "squared_error", "ends", "thresholds")


def _worker(task: tuple[str, list[tuple[str, str]], int, int, float]) -> dict:
    """Run one block of resamples on arrays memory-mapped from disk (no copy per worker)."""
    folder, keys, start, stop, target = task
    root = Path(folder)
    cases = {
        k: Ranked(*(np.load(root / f"{i}_{f}.npy", mmap_mode="r") for f in _FIELDS))
        for i, k in enumerate(keys)
    }
    draws = np.load(root / "draws.npy", mmap_mode="r")
    return _resample_block(cases, draws, start, stop, target)


def bootstrap(
    cases: dict[tuple[str, str], Ranked],
    n_domains: int,
    p: Preregistration,
    resamples: int,
    workers: int = 1,
    scratch: Path | None = None,
) -> dict[tuple[str, str], np.ndarray]:
    """Metrics per resample (resamples x 3) for every case, from shared domain draws.

    All draws are made up front from the design seed, so the result does not depend on how
    many worker processes share the resamples. Workers are started fresh (not forked from this
    multi-threaded process) and read the arrays through memory-mapped files in `scratch`.
    """
    rng = np.random.default_rng(p.seed)
    draws = np.empty((resamples, n_domains), dtype=np.int32)
    for b in range(resamples):
        draws[b] = np.bincount(rng.integers(0, n_domains, n_domains), minlength=n_domains)
    step = -(-resamples // max(1, workers))
    blocks = [(s, min(s + step, resamples)) for s in range(0, resamples, step)]
    if len(blocks) == 1:
        return _resample_block(cases, draws, 0, resamples, p.precision_target)
    keys = list(cases)
    with tempfile.TemporaryDirectory(dir=scratch, prefix="bootstrap-") as folder:
        for i, k in enumerate(keys):
            for f in _FIELDS:
                np.save(Path(folder) / f"{i}_{f}.npy", getattr(cases[k], f))
        np.save(Path(folder) / "draws.npy", draws)
        tasks = [(folder, keys, s, e, p.precision_target) for s, e in blocks]
        with multiprocessing.get_context("spawn").Pool(len(blocks)) as pool:
            parts = pool.map(_worker, tasks)
    return {k: np.concatenate([part[k] for part in parts]) for k in keys}


def _interval(values: np.ndarray, confidence: float) -> dict:
    """Percentile interval over the defined resamples, and how many were undefined."""
    ok = values[~np.isnan(values)]
    alpha = (1 - confidence) / 2
    return {
        "ci_low": float(np.quantile(ok, alpha)) if len(ok) else None,
        "ci_high": float(np.quantile(ok, 1 - alpha)) if len(ok) else None,
        "undefined_resamples": int(len(values) - len(ok)),
    }


def _value(x: float) -> float | None:
    return None if np.isnan(x) else float(x)


# Run --------------------------------------------------------------------------------------


def _train_base_rate(con: duckdb.DuckDBPyConnection, features: Path) -> float:
    return float(
        con.execute(
            "SELECT avg(label_primary::DOUBLE) FROM read_parquet(?) WHERE split = 'train'",
            [str(features)],
        ).fetchone()[0]
    )


def _peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def run(
    p: Preregistration,
    features: Path,
    sample_path: Path,
    out_dir: Path,
    *,
    dry_run: bool = False,
    memory_limit: str = "14GB",
    grid_c: tuple[float, ...] = GRID_C,
    class_weights: tuple[str | None, ...] = CLASS_WEIGHTS,
    resamples: int | None = None,
    chunk_rows: int = CHUNK_ROWS,
    workers: int = BOOTSTRAP_WORKERS,
) -> dict | None:
    if missing := sorted(set(p.inputs) - ENCODED_INPUTS):
        raise SystemExit(f"design inputs this module does not encode: {missing}")
    if extra := sorted(ENCODED_INPUTS - set(p.inputs)):
        raise SystemExit(f"encoded inputs that the design does not list: {extra}")
    if (p.calibration, p.calibration_rows, p.threshold_rule) != (
        "isotonic",
        "validation_cross_fitted",
        "precision_target_on_validation",
    ):
        raise SystemExit("the design asks for a calibration or threshold rule not built here")
    if p.base_rate_rows != "all_train_rows" or p.pr_auc_method != "average_precision":
        raise SystemExit("the design asks for a base-rate source or PR-AUC method not built here")
    n_requests = p.require_train_requests()
    n_boot = p.bootstrap_resamples if resamples is None else resamples
    t0 = time.perf_counter()
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute(f"SET threads = {os.cpu_count() or 1}")

    ids = draw_sample(con, features, n_requests, p.seed)
    write_sample(con, ids, sample_path)
    base_rate = 0.5 if dry_run else _train_base_rate(con, features)
    train = read_sample_rows(con, features, sample_path, labels=not dry_run)
    t1 = time.perf_counter()
    encoder = AuditEncoder(
        p.ngram_min, p.ngram_max, p.ngram_min_df, p.ngram_sublinear_tf, p.categorical_top_k,
        base_rate,
    ).fit(train)  # fmt: skip
    x_train = encoder.transform(train)
    t_encode = time.perf_counter() - t1
    n_val = stage_validation(con, features, labels=not dry_run)

    if dry_run:
        t2 = time.perf_counter()
        chunk = next(validation_chunks(con, n_val, chunk_rows))
        encoder.transform(chunk)
        t_chunk = time.perf_counter() - t2
        n_chunks = -(-n_val // chunk_rows)
        print(f"sample: {len(ids):,} training requests, {x_train.shape[0]:,} rows -> {sample_path}")
        print(
            f"encoding: {encoder.n_columns:,} columns, {x_train.nnz / x_train.shape[0]:.1f} "
            f"non-zeros per row, fitted in {t_encode:.1f}s"
        )
        print(
            f"validation: {n_val:,} rows in {n_chunks} chunks, about {t_chunk:.1f}s to encode "
            f"one chunk (about {t_chunk * n_chunks / 60:.1f} min for all)"
        )
        print(f"peak memory so far {_peak_gb():.1f} GB")
        print(
            "dry run: no label read, no model fitted; only the sample file is written "
            f"({time.perf_counter() - t0:.1f}s)"
        )
        return None

    y_train = np.asarray(train["label_primary"], dtype=float)
    del train
    grid = []
    for c in grid_c:
        for cw in class_weights:
            t = time.perf_counter()
            model = LogisticRegression(C=c, class_weight=cw, solver="lbfgs", max_iter=MAX_ITER)
            model.fit(x_train, y_train)
            n_iter = int(model.n_iter_[0])
            grid.append(
                {
                    "C": c,
                    "class_weight": cw,
                    "fit_seconds": round(time.perf_counter() - t, 1),
                    "n_iter": n_iter,
                    "converged": n_iter < MAX_ITER,
                    "model": model,
                }
            )
            print(
                f"fitted C={c:g} class_weight={cw}: {n_iter} iterations, {grid[-1]['fit_seconds']}s"
            )
    train_peak = _peak_gb()
    del x_train

    # One pass over validation: encode each chunk once and score it with every model.
    scores = np.empty((len(grid), n_val))
    y = np.empty(n_val)
    domain = np.empty(n_val, dtype=np.int64)
    history = np.empty(n_val, dtype=bool)
    past = np.empty(n_val)
    pos = 0
    for chunk in validation_chunks(con, n_val, chunk_rows):
        x = encoder.transform(chunk)
        n = x.shape[0]
        for i, g in enumerate(grid):
            scores[i, pos : pos + n] = g["model"].decision_function(x)
        y[pos : pos + n] = chunk["label_primary"]
        domain[pos : pos + n] = chunk["did"]
        history[pos : pos + n] = chunk["history_flag"]
        past[pos : pos + n] = chunk["domain_past_rate"]
        pos += n
    if pos != n_val:
        raise SystemExit("validation rows changed while scoring")
    n_domains = int(domain.max()) + 1
    for i, g in enumerate(grid):
        r = rank_rows(scores[i], y, scores[i], domain)
        g["validation_pr_auc"] = row_metrics(r, np.ones(n_val), p.precision_target)[0]
    best = max(
        range(len(grid)),
        key=lambda i: (
            grid[i]["validation_pr_auc"],
            -grid[i]["C"],
            grid[i]["class_weight"] is None,
        ),
    )
    winner = grid[best]
    s = scores[best]
    prob = cross_fit_probabilities(s, y, domain, n_domains, p.calibration_report_folds, p.seed)
    final_calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(s, y)
    baseline = np.where(history, past, base_rate)

    subsets = {"all": slice(None), "with_history": history, "without_history": ~history}
    cases = {}
    for name, mask in subsets.items():
        cases[("model", name)] = rank_rows(s[mask], y[mask], prob[mask], domain[mask])
        cases[("domain_past_rate", name)] = rank_rows(
            baseline[mask], y[mask], baseline[mask], domain[mask]
        )
    point = {k: row_metrics(r, np.ones(len(r.y)), p.precision_target) for k, r in cases.items()}
    t3 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    boot = bootstrap(cases, n_domains, p, n_boot, workers, scratch=out_dir)
    t_boot = time.perf_counter() - t3

    chosen = point[("model", "all")][3]
    threshold = None if chosen < 0 else float(cases[("model", "all")].thresholds[chosen])
    conf = p.bootstrap_confidence
    results = []
    for name in subsets:
        entry = {"subset": name, "rows": len(cases[("model", name)].y)}
        for j, metric in enumerate(("pr_auc", "recall_at_precision", "brier")):
            m = point[("model", name)][j]
            b = point[("domain_past_rate", name)][j]
            diff = boot[("model", name)][:, j] - boot[("domain_past_rate", name)][:, j]
            entry[metric] = {
                "model": {"value": _value(m), **_interval(boot[("model", name)][:, j], conf)},
                "domain_past_rate": {
                    "value": _value(b),
                    **_interval(boot[("domain_past_rate", name)][:, j], conf),
                },
                "difference": {"value": _value(m - b), **_interval(diff, conf)},
            }
        results.append(entry)

    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / "logreg.joblib"
    tmp = artifact.with_name(artifact.name + ".tmp")
    joblib.dump(
        {
            "encoder": encoder,
            "model": winner["model"],
            "calibrator": final_calibrator,
            "threshold": threshold,
            "base_rate": base_rate,
        },
        tmp,
    )
    tmp.replace(artifact)
    report = {
        "model": "logistic_regression",
        "features": str(features),
        "sample": {"requests": len(ids), "rows": len(y_train), "file": str(sample_path)},
        "train_label_rate_sample": float(y_train.mean()),
        "train_base_rate_all_rows": base_rate,
        "encoding_columns": encoder.n_columns,
        "grid": [{k: v for k, v in g.items() if k != "model"} for g in grid],
        "winner": {"C": winner["C"], "class_weight": winner["class_weight"]},
        "selection": "highest validation PR-AUC; validation metrics are optimistic",
        "validation_rows": n_val,
        "validation_domains": n_domains,
        "precision_target": p.precision_target,
        "threshold": threshold,
        "bootstrap": {
            "resamples": n_boot,
            "unit": p.bootstrap_unit,
            "confidence": p.bootstrap_confidence,
            "seed": p.seed,
        },
        "brier_probabilities": f"isotonic, cross-fitted on {p.calibration_report_folds} domain folds",
        "results": results,
        "train_peak_gb": round(train_peak, 2),
        "peak_gb": round(_peak_gb(), 2),
        "artifact": str(artifact),
    }
    for metric_set in report["grid"]:
        metric_set["validation_pr_auc"] = _value(metric_set["validation_pr_auc"])
    path = out_dir / "logreg.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)

    print(f"winner C={winner['C']:g} class_weight={winner['class_weight']} (validation PR-AUC)")
    print(
        f"validation, used for selection (point [{100 * conf:.0f}% interval]); "
        "model, domain past rate, difference:"
    )
    for entry in results:
        print(f"  {entry['subset']:<16}rows {entry['rows']:>10,}")
        for metric in ("pr_auc", "recall_at_precision", "brier"):
            parts = []
            for who in ("model", "domain_past_rate", "difference"):
                m = entry[metric][who]
                v = "n/a" if m["value"] is None else f"{m['value']:+.4f}"
                ci = (
                    "[n/a]" if m["ci_low"] is None else f"[{m['ci_low']:+.4f}, {m['ci_high']:+.4f}]"
                )
                parts.append(f"{who} {v} {ci}")
            print(f"      {metric:<20}" + "  ".join(parts))
    print(f"threshold at precision {p.precision_target}: {threshold}")
    print(
        f"train peak {train_peak:.1f} GB, overall peak {_peak_gb():.1f} GB, bootstrap "
        f"{t_boot:.0f}s, total {time.perf_counter() - t0:.0f}s -> {path}"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and evaluate the audit logistic regression.")
    parser.add_argument("--dry-run", action="store_true", help="sample and sizes, read no label")
    parser.add_argument("--design", type=Path, default=Path("configs/audit_preregistration.yaml"))
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--memory-limit", default="14GB")
    args = parser.parse_args()
    run(
        load_preregistration(args.design),
        args.features,
        args.sample,
        args.out_dir,
        dry_run=args.dry_run,
        memory_limit=args.memory_limit,
    )


if __name__ == "__main__":
    main()
