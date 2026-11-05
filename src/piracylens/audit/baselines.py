"""Baselines for the takedown audit model, evaluated on the validation split.

Two baselines set the bar every model must clear:

- base rate: every row gets the training base rate, the label mean over every training row
  (the design file fixes which rows give it). A constant score carries no ranking skill, so its
  PR-AUC equals the validation label rate and its recall at the precision target is zero unless
  that rate itself reaches the target. Both are true by construction, not findings.
- domain past rate: the row's past no-action rate for its domain; rows without domain history
  get the training base rate instead. Among rows without history this baseline is therefore
  identical to the base rate, by construction.

Metrics follow the design file. PR-AUC is average precision: a step function over distinct
scores, with tied scores forming one threshold. Recall at the precision target is the highest
recall over thresholds whose precision reaches the target, and zero when none does. The Brier
score is the mean squared difference between score and label. Each metric gets a percentile
interval from domain-level bootstrap resamples: validation domains are drawn with replacement
and every row of a drawn domain is kept, so rows of one domain never count as independent.
One set of resamples is shared by every baseline, label and subset, so their intervals are
paired. Subsets (rows with and without domain history) are evaluated on the rows of the same
resampled domains. A resample that draws no row of a subset leaves every metric undefined there,
and one without a positive row leaves PR-AUC and recall undefined; such resamples are counted and
left out of that interval.

Both labels are evaluated. The domain past rate counts earlier rows with the primary label, so
for the sensitivity label it is a mismatched predictor; the output says so.

Only training and validation rows are read. Test rows are filtered out in the query, so no
test label can reach the output. No per-owner, per-group or per-reporter figure is computed.

Usage (from the repo root):
    python -m piracylens.audit.baselines --dry-run   # sizes and timing, reads no label
    python -m piracylens.audit.baselines
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
from scipy import sparse

from piracylens.audit.preregistration import Preregistration, load_preregistration

DEFAULT_FEATURES = Path("data/interim/audit_features.parquet")
DEFAULT_OUT = Path("data/interim/audit_baselines.json")
# Resamples evaluated together; the draw matrix holds domains x batch counts.
BATCH = 25
LABELS = ("label_primary", "label_sensitivity")
BASELINES = ("base_rate", "domain_past_rate")
SUBSETS = ("all", "with_history", "without_history")
NOTES = {
    "base_rate": (
        "Constant score: PR-AUC equals the label rate and recall at the precision target is "
        "zero unless the rate reaches the target, both by construction."
    ),
    "domain_past_rate": (
        "Rows without domain history get the training base rate, so on those rows this "
        "baseline equals the base rate by construction."
    ),
    "label_sensitivity": (
        "The domain past rate counts earlier rows with the primary label, so it is a "
        "mismatched predictor for this label."
    ),
}


@dataclass(frozen=True)
class Groups:
    """Validation rows compressed to one entry per domain, history flag and past rate."""

    domain: np.ndarray  # dense domain id per group
    history: np.ndarray  # bool: the domain had history at filing time
    past_rate: np.ndarray  # float, NaN without history
    rows: np.ndarray  # rows in the group
    positives: dict[str, np.ndarray]  # rows with the label, per label
    n_domains: int


def ranked(scores: np.ndarray) -> tuple[np.ndarray, int]:
    """Rank of each score among the distinct scores, highest first; ties share a rank."""
    distinct, inverse = np.unique(scores, return_inverse=True)
    return len(distinct) - 1 - inverse, len(distinct)


def metrics(
    rank: np.ndarray,
    n_ranks: int,
    scores: np.ndarray,
    pos: np.ndarray,
    neg: np.ndarray,
    weight: np.ndarray,
    precision_target: float,
    losses: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[float, float, float]:
    """Weighted PR-AUC (average precision), recall at the precision target and Brier score.

    `pos`/`neg` are positive and negative row counts per group and `weight` how many times each
    group is drawn. Thresholds are the distinct scores, highest first. `losses` optionally holds
    the precomputed squared errors of a positive and a negative row per group.
    """
    wpos = pos * weight
    wneg = neg * weight
    n_pos = float(wpos.sum())
    n_neg = float(wneg.sum())
    total = n_pos + n_neg
    if total == 0:
        return float("nan"), float("nan"), float("nan")
    loss_pos, loss_neg = ((scores - 1.0) ** 2, scores**2) if losses is None else losses
    brier = float((wpos @ loss_pos + wneg @ loss_neg) / total)
    if n_pos == 0:
        return float("nan"), float("nan"), brier
    if n_ranks == 1:  # a constant score: one threshold that flags every row
        tp, fp = np.array([n_pos]), np.array([n_neg])
    else:
        tp = np.cumsum(np.bincount(rank, weights=wpos, minlength=n_ranks))
        fp = np.cumsum(np.bincount(rank, weights=wneg, minlength=n_ranks))
    flagged = tp + fp
    precision = np.divide(tp, flagged, out=np.zeros_like(tp), where=flagged > 0)
    recall = tp / n_pos
    ap = float(np.sum(np.diff(recall, prepend=0.0) * precision))
    reached = precision >= precision_target
    recall_at = float(recall[reached].max()) if reached.any() else 0.0
    return ap, recall_at, brier


@dataclass(frozen=True)
class DomainMatrices:
    """One case's groups summed per (score rank, domain), for many resamples at once."""

    pos: sparse.csr_matrix  # ranks x domains: positive rows
    neg: sparse.csr_matrix  # ranks x domains: negative rows
    loss: np.ndarray  # per domain: squared error summed over its rows
    rows: np.ndarray  # per domain: rows


def domain_matrices(c: dict, n_domains: int) -> DomainMatrices:
    shape = (c["n_ranks"], n_domains)
    loss_pos, loss_neg = c["losses"]
    return DomainMatrices(
        pos=sparse.csr_matrix((c["pos"], (c["rank"], c["domain"])), shape=shape),
        neg=sparse.csr_matrix((c["neg"], (c["rank"], c["domain"])), shape=shape),
        loss=np.bincount(
            c["domain"], weights=c["pos"] * loss_pos + c["neg"] * loss_neg, minlength=n_domains
        ),
        rows=np.bincount(c["domain"], weights=c["pos"] + c["neg"], minlength=n_domains),
    )


def batch_metrics(m: DomainMatrices, draws: np.ndarray, precision_target: float) -> np.ndarray:
    """The three metrics for each column of `draws` (domains x resamples); NaN where undefined.

    Same definitions as `metrics`, with each group weighted by how often its domain is drawn.
    """
    tp = np.cumsum(m.pos @ draws, axis=0)
    fp = np.cumsum(m.neg @ draws, axis=0)
    total = m.rows @ draws
    n_pos = tp[-1]
    out = np.full((draws.shape[1], 3), np.nan)
    has_rows = total > 0
    out[has_rows, 2] = (m.loss @ draws)[has_rows] / total[has_rows]
    flagged = tp + fp
    precision = np.divide(tp, flagged, out=np.zeros_like(tp), where=flagged > 0)
    recall = np.divide(tp, n_pos, out=np.zeros_like(tp), where=n_pos > 0)
    step = np.diff(recall, axis=0, prepend=0.0)
    ok = n_pos > 0
    out[ok, 0] = (step * precision).sum(axis=0)[ok]
    reached_recall = np.where(precision >= precision_target, recall, 0.0)
    out[ok, 1] = reached_recall.max(axis=0)[ok]
    return out


def load_groups(con: duckdb.DuckDBPyConnection, features: Path, *, blind: bool = False) -> Groups:
    """Read validation rows, compressed, with both labels.

    With `blind`, no label column is read: every group gets half its rows as positives, which
    keeps the shapes and the timing of the real run for a dry run.
    """
    if blind:
        positives = (
            "(count(*) // 2)::DOUBLE AS pos_primary, (count(*) // 2)::DOUBLE AS pos_sensitivity"
        )
        labels = ""
    else:
        positives = (
            "(count(*) FILTER (WHERE label_primary))::DOUBLE AS pos_primary, "
            "(count(*) FILTER (WHERE label_sensitivity))::DOUBLE AS pos_sensitivity"
        )
        labels = ", label_primary, label_sensitivity"
    # A missing past rate becomes NaN so every column arrives as a plain numpy array.
    cols = con.execute(
        f"""
        WITH v AS (
            SELECT domain, has_domain_history, domain_past_rate{labels}
            FROM read_parquet(?) WHERE split = 'validation'
        )
        SELECT (dense_rank() OVER (ORDER BY domain) - 1)::BIGINT AS did,
               has_domain_history AS history,
               coalesce(domain_past_rate, 'NaN'::DOUBLE) AS past_rate,
               count(*)::DOUBLE AS n, {positives}
        FROM v GROUP BY domain, has_domain_history, domain_past_rate
        """,
        [str(features)],
    ).fetchnumpy()
    if len(cols["did"]) == 0:
        raise SystemExit("no validation rows in the feature table")
    history = np.asarray(cols["history"], dtype=bool)
    past_rate = np.asarray(cols["past_rate"], dtype=float)
    if np.any(history == np.isnan(past_rate)):
        raise SystemExit("the history flag and a missing past rate disagree")
    domain = np.asarray(cols["did"], dtype=np.int64)
    return Groups(
        domain=domain,
        history=history,
        past_rate=past_rate,
        rows=np.asarray(cols["n"], dtype=float),
        positives={
            "label_primary": np.asarray(cols["pos_primary"], dtype=float),
            "label_sensitivity": np.asarray(cols["pos_sensitivity"], dtype=float),
        },
        n_domains=int(domain.max()) + 1,
    )


def train_base_rates(con: duckdb.DuckDBPyConnection, features: Path) -> tuple[int, dict]:
    """Label mean over every training row, per label."""
    n, r1, r2 = con.execute(
        """SELECT count(*), avg(label_primary::DOUBLE), avg(label_sensitivity::DOUBLE)
           FROM read_parquet(?) WHERE split = 'train'""",
        [str(features)],
    ).fetchone()
    if not n:
        raise SystemExit("no training rows in the feature table")
    return int(n), {"label_primary": float(r1), "label_sensitivity": float(r2)}


def _cases(g: Groups, base_rates: dict[str, float]) -> list[dict]:
    """Every (label, baseline, subset) combination with its fixed score ranking."""
    masks = {"all": None, "with_history": g.history, "without_history": ~g.history}
    cases = []
    for label in LABELS:
        rate = base_rates[label]
        for baseline in BASELINES:
            if baseline == "base_rate":
                full = np.full(len(g.rows), rate)
            else:
                full = np.where(g.history, g.past_rate, rate)
            for subset in SUBSETS:
                idx = (
                    np.arange(len(g.rows))
                    if masks[subset] is None
                    else np.flatnonzero(masks[subset])
                )
                scores = full[idx]
                rank, n_ranks = ranked(scores)
                pos = g.positives[label][idx]
                cases.append(
                    {
                        "label": label,
                        "baseline": baseline,
                        "subset": subset,
                        "domain": g.domain[idx],
                        "scores": scores,
                        "rank": rank,
                        "n_ranks": n_ranks,
                        "pos": pos,
                        "neg": g.rows[idx] - pos,
                        "losses": ((scores - 1.0) ** 2, scores**2),
                    }
                )
    return cases


def evaluate(
    g: Groups,
    base_rates: dict[str, float],
    p: Preregistration,
    resamples: int | None = None,
) -> list[dict]:
    """Point estimates and domain-bootstrap intervals for every combination."""
    n_boot = p.bootstrap_resamples if resamples is None else resamples
    cases = _cases(g, base_rates)
    point = []
    for c in cases:
        point.append(
            metrics(
                c["rank"], c["n_ranks"], c["scores"], c["pos"], c["neg"],
                np.ones(len(c["pos"])), p.precision_target,
            )
        )  # fmt: skip
    boot = np.full((len(cases), n_boot, 3), np.nan)
    mats = [domain_matrices(c, g.n_domains) for c in cases]
    rng = np.random.default_rng(p.seed)
    for start in range(0, n_boot, BATCH):
        size = min(BATCH, n_boot - start)
        draws = np.empty((g.n_domains, size))
        for k in range(size):
            draws[:, k] = np.bincount(
                rng.integers(0, g.n_domains, g.n_domains), minlength=g.n_domains
            )
        for i, m in enumerate(mats):
            boot[i, start : start + size] = batch_metrics(m, draws, p.precision_target)
    alpha = (1 - p.bootstrap_confidence) / 2
    results = []
    for i, c in enumerate(cases):
        rows = float(c["pos"].sum() + c["neg"].sum())
        entry = {
            "label": c["label"],
            "baseline": c["baseline"],
            "subset": c["subset"],
            "rows": int(rows),
            "domains": len(np.unique(c["domain"])),
            "label_rate": float(c["pos"].sum() / rows) if rows else None,
        }
        for j, name in enumerate(("pr_auc", "recall_at_precision", "brier")):
            values = boot[i, :, j]
            ok = values[~np.isnan(values)]
            entry[name] = {
                "value": None if np.isnan(point[i][j]) else point[i][j],
                "ci_low": float(np.quantile(ok, alpha)) if len(ok) else None,
                "ci_high": float(np.quantile(ok, 1 - alpha)) if len(ok) else None,
                "undefined_resamples": int(n_boot - len(ok)),
            }
        results.append(entry)
    return results


def run(
    p: Preregistration,
    features: Path,
    out: Path,
    *,
    dry_run: bool = False,
    memory_limit: str = "14GB",
    resamples: int | None = None,
) -> dict | None:
    if p.base_rate_rows != "all_train_rows" or p.pr_auc_method != "average_precision":
        raise SystemExit("the design asks for a base-rate source or PR-AUC method not built here")
    if not set(p.baselines) <= set(BASELINES):
        raise SystemExit(f"unknown baselines in the design: {p.baselines}")
    t0 = time.perf_counter()
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute(f"SET threads = {os.cpu_count() or 1}")

    if dry_run:
        # Sizes only: no label column is read.
        sizes = con.execute(
            """SELECT split, count(*) FROM read_parquet(?)
               WHERE split IN ('train', 'validation') GROUP BY 1 ORDER BY 1""",
            [str(features)],
        ).fetchall()
        g = load_groups(con, features, blind=True)
        t1 = time.perf_counter()
        # One batch of resamples on placeholder labels; the setup is included, so this is
        # an upper bound per resample.
        evaluate(g, {"label_primary": 0.5, "label_sensitivity": 0.5}, p, resamples=BATCH)
        per_resample = (time.perf_counter() - t1) / BATCH
        print("rows read (test rows are never read):")
        for split, n in sizes:
            print(f"  {split:<11}{n:>14,}")
        print(f"validation groups {len(g.rows):,} over {g.n_domains:,} domains")
        print(
            f"timing on placeholder labels: {per_resample:.2f}s per resample, about "
            f"{per_resample * p.bootstrap_resamples / 60:.1f} min for {p.bootstrap_resamples:,}"
        )
        print(f"dry run: nothing written ({time.perf_counter() - t0:.1f}s)")
        return None

    n_train, base_rates = train_base_rates(con, features)
    g = load_groups(con, features)
    t1 = time.perf_counter()
    results = evaluate(g, base_rates, p, resamples)
    report = {
        "features": str(features),
        "features_bytes": features.stat().st_size,
        "split": {
            "train": [str(p.train.start), str(p.train.end)],
            "validation": [str(p.validation.start), str(p.validation.end)],
        },
        "train_rows": n_train,
        "train_base_rate": base_rates,
        "validation_groups": len(g.rows),
        "validation_domains": g.n_domains,
        "precision_target": p.precision_target,
        "bootstrap": {
            "resamples": p.bootstrap_resamples if resamples is None else resamples,
            "unit": p.bootstrap_unit,
            "confidence": p.bootstrap_confidence,
            "seed": p.seed,
        },
        "pr_auc_method": p.pr_auc_method,
        "notes": NOTES,
        "results": results,
    }
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    json.loads(tmp.read_text(encoding="utf-8"))
    tmp.replace(out)

    print(f"training base rate over {n_train:,} rows:")
    for label in LABELS:
        print(f"  {label:<18}{base_rates[label]:>9.4f}")
    print(f"validation (point estimate [{100 * p.bootstrap_confidence:.0f}% interval]):")
    for r in results:
        print(
            f"  {r['label']:<18}{r['baseline']:<17}{r['subset']:<16}"
            f"rows {r['rows']:>10,}  rate "
            + ("n/a" if r["label_rate"] is None else f"{r['label_rate']:.4f}")
        )
        for name in ("pr_auc", "recall_at_precision", "brier"):
            m = r[name]
            interval = (
                f"[{m['ci_low']:.4f}, {m['ci_high']:.4f}]" if m["ci_low"] is not None else "[n/a]"
            )
            undefined = (
                f"  undefined {m['undefined_resamples']}" if m["undefined_resamples"] else ""
            )
            value = "n/a" if m["value"] is None else f"{m['value']:.4f}"
            print(f"      {name:<20}{value:>8} {interval}{undefined}")
    print(
        f"-> {out} (bootstrap {time.perf_counter() - t1:.1f}s, total "
        f"{time.perf_counter() - t0:.1f}s)"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the audit baselines on validation.")
    parser.add_argument("--dry-run", action="store_true", help="print sizes, read no label")
    parser.add_argument("--design", type=Path, default=Path("configs/audit_preregistration.yaml"))
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--memory-limit", default="14GB")
    args = parser.parse_args()
    run(
        load_preregistration(args.design),
        args.features,
        args.out,
        dry_run=args.dry_run,
        memory_limit=args.memory_limit,
    )


if __name__ == "__main__":
    main()
