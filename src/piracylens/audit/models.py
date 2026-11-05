"""Models for the takedown audit (logistic regression, linear SVM, gradient-boosted trees).

The training sample is a fixed number of whole training requests drawn uniformly without
replacement with the design seed; the drawn request ids are written to a file so every model
trains on the same rows. Every fitted part of the input encoding is fitted on the sample rows
only: the domain character n-gram vocabulary, the most frequent categories (the rest and unseen
values become one "other" column) and the scaling of numeric inputs. Counts and the domain age
are log-transformed before scaling. A missing past rate is filled with the training base rate
(the label mean over every training row); a missing domain age, which happens only at a domain's
first appearance, is filled with zero days and keeps its flag.

A small grid of regularization strengths and class weightings is fitted on the same encoded
sample. The winner has the highest validation PR-AUC; ties go to the stronger regularization,
then to no class weighting. Validation metrics of the winner are therefore optimistic (they were
used for selection) and are labelled as such; the test set is not read here. The logistic
regression grid is fixed in this module; the linear SVM (hinge loss, trained by stochastic
gradient descent) takes its grid and stopping rule from the design file.

Gradient-boosted trees (XGBoost, histogram method) take the same encoded matrix, so the same
guards prove identical inputs. The filled past rate keeps its history flag, which marks exactly
the filled rows, so one split separates them. XGBoost treats zeros that a sparse matrix does not
store as missing; for the n-gram, one-hot and flag columns the learned missing branch plays the
role of zero, and scaled numeric inputs are almost never exactly zero. Each fit grows a fixed
number of trees and is scored on validation at fixed tree checkpoints without refitting; the
depth, class weighting and number of trees are chosen together by validation PR-AUC. The
number of cached node histograms is capped to bound memory; the cap does not change the model.
Each finished tree fit is saved under `<stem>_fits/` in the output directory, keyed by the
training sample file, the training matrix size, the training label mean, the design's
gradient-boosting settings, the fit's own settings and the library version; a rerun loads a
fit whose key matches instead of refitting it, so a crash late in the grid keeps earlier fits.

The linear SVM and the trees are compared with the stored logistic regression on the same rows:
the training sample must equal the stored one, the refitted encoding must reproduce the stored encoder's
training matrix exactly, and the stored model's validation metrics must reproduce its own
report before the bootstrap starts. Both models then share the resamples, so their difference
has a paired interval.

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
    python -m piracylens.audit.models --model linear_svm --dry-run
    python -m piracylens.audit.models --model linear_svm
    python -m piracylens.audit.models --model xgboost --dry-run   # adds a short timing fit
    python -m piracylens.audit.models --model xgboost
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import multiprocessing
import os
import resource
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import xgboost
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, SGDClassifier

from piracylens.audit.preregistration import Preregistration, load_preregistration

DEFAULT_FEATURES = Path("data/interim/audit_features.parquet")
DEFAULT_SAMPLE = Path("data/interim/audit_train_sample.parquet")
DEFAULT_OUT_DIR = Path("data/interim/audit_models")

# Grid for the logistic regression (heuristic values).
GRID_C = (0.01, 0.1, 1.0, 10.0)
CLASS_WEIGHTS = (None, "balanced")
MAX_ITER = 1000
CHUNK_ROWS = 1_000_000
# Threads for gradient boosting, leaving some of the machine's threads free.
XGB_N_JOBS = 16
# How many node histograms XGBoost keeps instead of recomputing them. A histogram covers
# every column, so with about 260,000 columns the library default cache reached 16.7 GB on
# its own in a label-free measurement at depth 10, while 64 nodes peaked at 8.2 GB at the same
# speed and gave bit-identical margins on all training rows. The setting changes memory only,
# never the model.
XGB_MAX_CACHED_HIST_NODE = 64
# Trees of the dry run's timing fit, on placeholder labels, at the deepest grid depth.
XGB_TIMING_TREES = 20
# Worker processes for the bootstrap; each held about 1.3 GB with two evaluated scores and
# needs more with three, so six keep the total within a 23 GB machine. The shared arrays are
# memory-mapped from a temporary folder in the output directory. The result does not depend
# on the number of workers.
BOOTSTRAP_WORKERS = 6

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


# Model families -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """One model family: its grid, how one fit is built, and its tie-break."""

    name: str  # the design's candidate name
    stem: str  # output file name stem
    grid: tuple[dict[str, Any], ...]
    build: Callable[[dict[str, Any]], Any]
    iterations: Callable[[Any], int] | None  # optimizer iterations; None for trees
    max_iter: int | None
    # Among candidates with equal validation PR-AUC, the largest value of this key wins.
    prefer: Callable[[dict[str, Any]], tuple]
    # Trees: each fit is also scored with its first k trees for every k listed here.
    checkpoints: tuple[int, ...] | None = None
    # Trees: class weighting is set from the training sample's class balance at fit time.
    weight_from_sample: bool = False


def logistic_regression_spec(
    grid_c: tuple[float, ...] = GRID_C, class_weights: tuple[str | None, ...] = CLASS_WEIGHTS
) -> ModelSpec:
    return ModelSpec(
        name="logistic_regression",
        stem="logreg",
        grid=tuple({"C": c, "class_weight": cw} for c in grid_c for cw in class_weights),
        build=lambda g: LogisticRegression(
            C=g["C"], class_weight=g["class_weight"], solver="lbfgs", max_iter=MAX_ITER
        ),
        iterations=lambda m: int(m.n_iter_[0]),
        max_iter=MAX_ITER,
        prefer=lambda g: (-g["C"], g["class_weight"] is None),
    )


def linear_svm_spec(
    p: Preregistration,
    alpha_grid: tuple[float, ...] | None = None,
    class_weights: tuple[str | None, ...] | None = None,
) -> ModelSpec:
    s = p.linear_svm
    if s.tie_break != "larger_alpha_then_no_class_weight":
        raise SystemExit(f"the design asks for a tie-break not built here: {s.tie_break}")
    if class_weights is None:
        class_weights = tuple(None if v == "none" else "balanced" for v in p.imbalance_variants)
    alphas = s.alpha_grid if alpha_grid is None else alpha_grid
    return ModelSpec(
        name="linear_svm_sgd",
        stem="linear_svm",
        grid=tuple({"alpha": a, "class_weight": cw} for a in alphas for cw in class_weights),
        build=lambda g: SGDClassifier(
            loss=s.loss,
            penalty=s.penalty,
            alpha=g["alpha"],
            class_weight=g["class_weight"],
            max_iter=s.max_epochs,
            tol=s.tol,
            n_iter_no_change=s.n_iter_no_change,
            early_stopping=False,
            random_state=p.seed,
        ),
        iterations=lambda m: int(m.n_iter_),
        max_iter=s.max_epochs,
        prefer=lambda g: (g["alpha"], g["class_weight"] is None),
    )


def xgboost_spec(
    p: Preregistration,
    depth_grid: tuple[int, ...] | None = None,
    checkpoints: tuple[int, ...] | None = None,
    class_weights: tuple[str | None, ...] | None = None,
) -> ModelSpec:
    x = p.xgboost
    fixed = (x.tree_method, x.class_weighting, x.input_matrix, x.tie_break)
    if fixed != (
        "hist",
        "scale_pos_weight_from_sample",
        "shared_linear_encoding",
        "fewer_trees_then_shallower_then_no_class_weight",
    ):
        raise SystemExit(f"the design asks for gradient-boosting settings not built here: {fixed}")
    if class_weights is None:
        class_weights = tuple(None if v == "none" else "balanced" for v in p.imbalance_variants)
    depths = x.max_depth_grid if depth_grid is None else depth_grid
    marks = x.tree_checkpoints if checkpoints is None else checkpoints
    return ModelSpec(
        name="xgboost",
        stem="xgboost",
        grid=tuple({"max_depth": d, "class_weight": cw} for d in depths for cw in class_weights),
        build=lambda g: xgboost.XGBClassifier(
            tree_method=x.tree_method,
            learning_rate=x.learning_rate,
            max_depth=g["max_depth"],
            n_estimators=marks[-1],
            random_state=p.seed,
            n_jobs=XGB_N_JOBS,
            max_cached_hist_node=XGB_MAX_CACHED_HIST_NODE,
        ),
        iterations=None,
        max_iter=None,
        prefer=lambda g: (-g["n_trees"], -g["max_depth"], g["class_weight"] is None),
        checkpoints=marks,
        weight_from_sample=True,
    )


def score(model: Any, x: sparse.csr_matrix, n_trees: int | None = None) -> np.ndarray:
    """Ranking score: the linear decision value, or the trees' margin with the first n trees."""
    if n_trees is None:
        return model.decision_function(x)
    return model.predict(x, output_margin=True, iteration_range=(0, n_trees))


def _fit_key(
    p: Preregistration, sample_sha256: str, x: sparse.csr_matrix, y: np.ndarray, settings: dict
) -> dict[str, Any]:
    """Everything that determines one tree fit; equal keys give identical fits."""
    return {
        "sample_sha256": sample_sha256,
        "rows": int(x.shape[0]),
        "columns": int(x.shape[1]),
        "nnz": int(x.nnz),
        "label_mean": float(y.mean()),
        "design": dataclasses.asdict(p.xgboost),
        "seed": p.seed,
        "settings": settings,
        "xgboost_version": xgboost.__version__,
    }


def _fit_or_load(
    estimator: Any, key: dict[str, Any], folder: Path, x: sparse.csr_matrix, y: np.ndarray
) -> tuple[Any, float, bool]:
    """Fit and save, or load a saved fit with the same key. Returns (model, fit seconds, loaded)."""
    text = json.dumps(key, sort_keys=True)
    name = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    model_path, meta_path = folder / f"{name}.ubj", folder / f"{name}.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # Compared after a JSON round trip, as stored (tuples become lists).
        if meta["key"] != json.loads(text) or not model_path.is_file():
            raise SystemExit(f"saved fit {meta_path} does not match its key, stopped")
        estimator.load_model(model_path)
        return estimator, meta["fit_seconds"], True
    t = time.perf_counter()
    estimator.fit(x, y)
    seconds = round(time.perf_counter() - t, 1)
    folder.mkdir(parents=True, exist_ok=True)
    tmp = model_path.with_name(f"{name}.tmp.ubj")  # the extension selects the format
    estimator.save_model(tmp)
    tmp.replace(model_path)
    # The metadata file is written last: it marks the saved fit as complete.
    tmp = meta_path.with_name(meta_path.name + ".tmp")
    tmp.write_text(json.dumps({"key": key, "fit_seconds": seconds}, indent=2), encoding="utf-8")
    tmp.replace(meta_path)
    return estimator, seconds, False


def _sample_pos_weight(y: np.ndarray) -> float:
    """Negatives per positive in the training sample (scale_pos_weight of "balanced")."""
    pos = float(y.sum())
    if pos == 0 or pos == len(y):
        raise SystemExit("the training sample has a single class")
    return (len(y) - pos) / pos


def select_winner(grid: list[dict[str, Any]], prefer: Callable[[dict], tuple]) -> int:
    """Index of the fit with the highest validation PR-AUC; ties go to the preferred fit."""
    return max(range(len(grid)), key=lambda i: (grid[i]["validation_pr_auc"], prefer(grid[i])))


def _params_text(g: dict[str, Any]) -> str:
    return " ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in g.items())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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


def use_sample(con: duckdb.DuckDBPyConnection, ids: np.ndarray, path: Path) -> str:
    """Write the drawn sample, or check that an existing sample file holds the same ids."""
    if not path.exists():
        write_sample(con, ids, path)
        return "written"
    stored = con.execute(
        "SELECT request_id FROM read_parquet(?) ORDER BY request_id", [str(path)]
    ).fetchnumpy()["request_id"]
    if not np.array_equal(stored, ids):
        raise SystemExit(f"{path} holds a different sample than the design draws, stopped")
    return "matches the stored file"


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


def _load_reference(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"reference model not found: {path} (run the logistic regression first)")
    ref = joblib.load(path)
    if not isinstance(ref, dict) or not {"encoder", "model", "base_rate"} <= set(ref):
        raise SystemExit(f"{path} is not a stored audit model")
    return ref


def _same_vocabulary(a: AuditEncoder, b: AuditEncoder) -> bool:
    return a.ngrams.vocabulary_ == b.ngrams.vocabulary_ and a.categories == b.categories


def _same_matrix(a: sparse.csr_matrix, b: sparse.csr_matrix) -> bool:
    return a.shape == b.shape and (a != b).nnz == 0


def run(
    p: Preregistration,
    features: Path,
    sample_path: Path,
    out_dir: Path,
    *,
    model: str = "logistic_regression",
    dry_run: bool = False,
    memory_limit: str = "14GB",
    grid_c: tuple[float, ...] = GRID_C,
    alpha_grid: tuple[float, ...] | None = None,
    depth_grid: tuple[int, ...] | None = None,
    checkpoints: tuple[int, ...] | None = None,
    class_weights: tuple[str | None, ...] | None = None,
    reference: Path | None = None,
    resamples: int | None = None,
    chunk_rows: int = CHUNK_ROWS,
    workers: int = BOOTSTRAP_WORKERS,
) -> dict | None:
    if model == "logistic_regression":
        spec = logistic_regression_spec(
            grid_c, CLASS_WEIGHTS if class_weights is None else class_weights
        )
        ref_path = None
    elif model in ("linear_svm", "xgboost"):
        if model == "linear_svm":
            spec = linear_svm_spec(p, alpha_grid, class_weights)
        else:
            spec = xgboost_spec(p, depth_grid, checkpoints, class_weights)
        ref_path = out_dir / "logreg.joblib" if reference is None else reference
    else:
        raise SystemExit(f"unknown model {model!r}")
    if spec.name not in p.candidates:
        raise SystemExit(f"the design does not list {spec.name} as a candidate")
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
    ref = None if ref_path is None else _load_reference(ref_path)
    t0 = time.perf_counter()
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute(f"SET threads = {os.cpu_count() or 1}")

    ids = draw_sample(con, features, n_requests, p.seed)
    sample_state = use_sample(con, ids, sample_path)
    base_rate = 0.5 if dry_run else _train_base_rate(con, features)
    train = read_sample_rows(con, features, sample_path, labels=not dry_run)
    t1 = time.perf_counter()
    encoder = AuditEncoder(
        p.ngram_min, p.ngram_max, p.ngram_min_df, p.ngram_sublinear_tf, p.categorical_top_k,
        base_rate,
    ).fit(train)  # fmt: skip
    x_train = encoder.transform(train)
    t_encode = time.perf_counter() - t1
    if ref is not None:
        # Without labels the base rate is a placeholder, so only the label-free parts compare.
        if not _same_vocabulary(encoder, ref["encoder"]):
            raise SystemExit("the encoding differs from the reference model's encoder, stopped")
        if not dry_run:
            if ref["base_rate"] != base_rate:
                raise SystemExit("the training base rate differs from the reference model's")
            if not _same_matrix(x_train, ref["encoder"].transform(train)):
                raise SystemExit("the training matrix differs from the reference encoder's")
    n_val = stage_validation(con, features, labels=not dry_run)

    if dry_run:
        t2 = time.perf_counter()
        chunk = next(validation_chunks(con, n_val, chunk_rows))
        encoder.transform(chunk)
        t_chunk = time.perf_counter() - t2
        n_chunks = -(-n_val // chunk_rows)
        print(
            f"sample: {len(ids):,} training requests, {x_train.shape[0]:,} rows -> {sample_path} "
            f"({sample_state})"
        )
        print(
            f"encoding: {encoder.n_columns:,} columns, {x_train.nnz / x_train.shape[0]:.1f} "
            f"non-zeros per row, fitted in {t_encode:.1f}s"
        )
        if ref is not None:
            print(f"reference {ref_path}: same n-gram vocabulary and categories")
        print(f"grid: {len(spec.grid)} fits of {spec.name}")
        if spec.checkpoints is not None:
            _time_trees(spec, x_train, p.seed)
        print(
            f"validation: {n_val:,} rows in {n_chunks} chunks, about {t_chunk:.1f}s to encode "
            f"one chunk (about {t_chunk * n_chunks / 60:.1f} min for all)"
        )
        print(f"peak memory so far {_peak_gb():.1f} GB")
        print(
            "dry run: no label read, no model fitted; only a missing sample file is written "
            f"({time.perf_counter() - t0:.1f}s)"
        )
        return None

    y_train = np.asarray(train["label_primary"], dtype=float)
    del train
    grid = []
    sample_sha = _sha256(sample_path) if spec.checkpoints is not None else ""
    for params in spec.grid:
        t = time.perf_counter()
        estimator = spec.build(params)
        extra: dict[str, Any] = {}
        if spec.weight_from_sample and params["class_weight"] == "balanced":
            extra["scale_pos_weight"] = _sample_pos_weight(y_train)
            estimator.set_params(scale_pos_weight=extra["scale_pos_weight"])
        if spec.checkpoints is None:
            fitted = estimator.fit(x_train, y_train)
            seconds = round(time.perf_counter() - t, 1)
        else:
            key = _fit_key(p, sample_sha, x_train, y_train, {**params, **extra})
            fitted, seconds, loaded = _fit_or_load(
                estimator, key, out_dir / f"{spec.stem}_fits", x_train, y_train
            )
        if spec.checkpoints is None:
            n_iter = spec.iterations(fitted)
            grid.append(
                {
                    **params,
                    "fit_seconds": seconds,
                    "n_iter": n_iter,
                    "converged": n_iter < spec.max_iter,
                    "model": fitted,
                }
            )
            print(f"fitted {_params_text(params)}: {n_iter} iterations, {seconds}s")
        else:
            for k in spec.checkpoints:
                grid.append(
                    {**params, "n_trees": k, **extra, "fit_seconds": seconds, "model": fitted}
                )
            how = "loaded saved fit" if loaded else "fitted"
            print(
                f"{how} {_params_text(params)}: {spec.checkpoints[-1]} trees, {seconds}s; "
                f"peak memory so far {_peak_gb():.1f} GB",
                flush=True,
            )
    train_peak = _peak_gb()
    del x_train

    # One pass over validation: encode each chunk once and score it with every model.
    scores = np.empty((len(grid), n_val))
    ref_scores = np.empty(n_val) if ref is not None else None
    y = np.empty(n_val)
    domain = np.empty(n_val, dtype=np.int64)
    history = np.empty(n_val, dtype=bool)
    past = np.empty(n_val)
    pos = 0
    for chunk in validation_chunks(con, n_val, chunk_rows):
        x = encoder.transform(chunk)
        n = x.shape[0]
        for i, g in enumerate(grid):
            scores[i, pos : pos + n] = score(g["model"], x, g.get("n_trees"))
        if ref is not None:
            ref_scores[pos : pos + n] = score(ref["model"], x)
        y[pos : pos + n] = chunk["label_primary"]
        domain[pos : pos + n] = chunk["did"]
        history[pos : pos + n] = chunk["history_flag"]
        past[pos : pos + n] = chunk["domain_past_rate"]
        pos += n
    if pos != n_val:
        raise SystemExit("validation rows changed while scoring")
    if bad := [
        _params_text(_candidate(spec, g))
        for i, g in enumerate(grid)
        if not np.isfinite(scores[i]).all()
    ]:
        raise SystemExit(f"non-finite validation scores from {bad}, stopped before any output")
    n_domains = int(domain.max()) + 1
    for i, g in enumerate(grid):
        r = rank_rows(scores[i], y, scores[i], domain)
        g["validation_pr_auc"] = row_metrics(r, np.ones(n_val), p.precision_target)[0]
    best = select_winner(grid, spec.prefer)
    winner = grid[best]
    s = scores[best]
    prob = cross_fit_probabilities(s, y, domain, n_domains, p.calibration_report_folds, p.seed)
    final_calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(s, y)
    baseline = np.where(history, past, base_rate)
    others = ["domain_past_rate"]
    if ref is not None:
        others.append("logistic_regression")
        ref_prob = cross_fit_probabilities(
            ref_scores, y, domain, n_domains, p.calibration_report_folds, p.seed
        )

    subsets = {"all": slice(None), "with_history": history, "without_history": ~history}
    cases = {}
    for name, mask in subsets.items():
        cases[("model", name)] = rank_rows(s[mask], y[mask], prob[mask], domain[mask])
        cases[("domain_past_rate", name)] = rank_rows(
            baseline[mask], y[mask], baseline[mask], domain[mask]
        )
        if ref is not None:
            cases[("logistic_regression", name)] = rank_rows(
                ref_scores[mask], y[mask], ref_prob[mask], domain[mask]
            )
    point = {k: row_metrics(r, np.ones(len(r.y)), p.precision_target) for k, r in cases.items()}
    if ref is not None:
        _check_reference_reproduces(point, ref_path.with_suffix(".json"), subsets)
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
            entry[metric] = {
                "model": {"value": _value(m), **_interval(boot[("model", name)][:, j], conf)}
            }
            for other in others:
                b = point[(other, name)][j]
                diff = boot[("model", name)][:, j] - boot[(other, name)][:, j]
                key = "difference" if other == "domain_past_rate" else f"difference_vs_{other}"
                entry[metric][other] = {
                    "value": _value(b),
                    **_interval(boot[(other, name)][:, j], conf),
                }
                entry[metric][key] = {"value": _value(m - b), **_interval(diff, conf)}
        results.append(entry)

    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / f"{spec.stem}.joblib"
    tmp = artifact.with_name(artifact.name + ".tmp")
    joblib.dump(
        {
            "encoder": encoder,
            "model": winner["model"],
            "calibrator": final_calibrator,
            "threshold": threshold,
            "base_rate": base_rate,
            "n_trees": winner.get("n_trees"),
        },
        tmp,
    )
    tmp.replace(artifact)
    params = list(_candidate(spec, winner))
    report = {
        "model": spec.name,
        "features": str(features),
        "sample": {"requests": len(ids), "rows": len(y_train), "file": str(sample_path)},
        "train_label_rate_sample": float(y_train.mean()),
        "train_base_rate_all_rows": base_rate,
        "encoding_columns": encoder.n_columns,
        "grid": [{k: v for k, v in g.items() if k != "model"} for g in grid],
        "winner": {k: winner[k] for k in params},
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
    if ref is not None:
        report["reference"] = {"artifact": str(ref_path), "sha256": _sha256(ref_path)}
    for metric_set in report["grid"]:
        metric_set["validation_pr_auc"] = _value(metric_set["validation_pr_auc"])
    path = out_dir / f"{spec.stem}.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)

    print(f"winner {_params_text(report['winner'])} (validation PR-AUC)")
    print(
        f"validation, used for selection (point [{100 * conf:.0f}% interval]); "
        f"model, then each comparison and the model's difference from it:"
    )
    for entry in results:
        print(f"  {entry['subset']:<16}rows {entry['rows']:>10,}")
        for metric in ("pr_auc", "recall_at_precision", "brier"):
            parts = []
            for who in entry[metric]:
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


def _candidate(spec: ModelSpec, g: dict[str, Any]) -> dict[str, Any]:
    """The settings that identify one candidate: its grid values, and its trees if any."""
    keys = [*spec.grid[0], *(["n_trees"] if spec.checkpoints is not None else [])]
    return {k: g[k] for k in keys}


def _time_trees(spec: ModelSpec, x_train: sparse.csr_matrix, seed: int) -> None:
    """Fit a few trees at the deepest depth on placeholder labels; prints time and memory."""
    labels = np.random.default_rng(seed).integers(0, 2, size=x_train.shape[0]).astype(float)
    deepest = max(spec.grid, key=lambda g: g["max_depth"])
    est = spec.build(deepest).set_params(n_estimators=XGB_TIMING_TREES)
    t = time.perf_counter()
    est.fit(x_train, labels)
    per_tree = (time.perf_counter() - t) / XGB_TIMING_TREES
    total = per_tree * spec.checkpoints[-1] * len(spec.grid)
    print(
        f"timing fit: {XGB_TIMING_TREES} trees at depth {deepest['max_depth']} on placeholder "
        f"labels, {per_tree:.2f}s per tree; the grid at that depth would take about "
        f"{total / 60:.0f} min (an upper bound; shallower trees are faster); "
        f"peak memory {_peak_gb():.1f} GB"
    )


def _check_reference_reproduces(point: dict, report_path: Path, subsets: dict) -> None:
    """Stop unless the reference model's validation metrics equal its own stored report."""
    if not report_path.is_file():
        raise SystemExit(f"reference report not found: {report_path}")
    stored = {
        e["subset"]: e for e in json.loads(report_path.read_text(encoding="utf-8"))["results"]
    }
    for name in subsets:
        for j, metric in enumerate(("pr_auc", "recall_at_precision", "brier")):
            want = stored[name][metric]["model"]["value"]
            got = _value(point[("logistic_regression", name)][j])
            if (want is None) != (got is None) or (
                want is not None and not np.isclose(got, want, rtol=1e-9, atol=0.0)
            ):
                raise SystemExit(
                    f"the reference model does not reproduce its report: {name} {metric} "
                    f"{got} vs {want}"
                )
    print(f"reference model reproduces {report_path} on every subset and metric")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and evaluate an audit model.")
    parser.add_argument(
        "--model",
        choices=("logistic_regression", "linear_svm", "xgboost"),
        default="logistic_regression",
    )
    parser.add_argument("--dry-run", action="store_true", help="sample and sizes, read no label")
    parser.add_argument("--design", type=Path, default=Path("configs/audit_preregistration.yaml"))
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="stored logistic regression for the linear SVM and XGBoost comparison "
        "(default: logreg.joblib in the output directory)",
    )
    parser.add_argument("--memory-limit", default="14GB")
    args = parser.parse_args()
    run(
        load_preregistration(args.design),
        args.features,
        args.sample,
        args.out_dir,
        model=args.model,
        dry_run=args.dry_run,
        memory_limit=args.memory_limit,
        reference=args.reference,
    )


if __name__ == "__main__":
    main()
