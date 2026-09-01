"""Confidence intervals for the discrimination metrics (Sec. 6).

* AUROC, fixed score vector -> DeLong (DeLong et al. 1988).
* Average precision -> stratified bootstrap percentile CI, cases and controls
  resampled separately.
* Exceedance proportion ``r/K`` -> Clopper-Pearson.

No interval is ever computed for a p-value itself (Sec. 6, last bullet).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..types import BoolArray, FloatArray

__all__ = [
    "ConfidenceInterval",
    "midrank",
    "delong_auc_variance",
    "delong_auroc_ci",
    "stratified_bootstrap_ci",
    "clopper_pearson",
]


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A two-sided interval with the method that produced it recorded.

    ``method`` travels with the numbers because a DeLong interval and a
    bootstrap interval are not interchangeable evidence.
    """

    low: float
    high: float
    level: float
    method: str

    def contains(self, value: float) -> bool:
        """``True`` when ``value`` lies inside the closed interval."""
        return self.low <= value <= self.high

    def as_tuple(self) -> tuple[float, float]:
        """``(low, high)``."""
        return (self.low, self.high)


def midrank(x: FloatArray) -> FloatArray:
    """Ranks of ``x`` with ties resolved to their mid-rank, 1-based.

    This is the tie convention that makes the rank form of the AUC agree
    exactly with the trapezoidal ROC area, and hence with
    ``sklearn.metrics.roc_auc_score``.
    """
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = x.shape[0]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n, dtype=np.float64)
    out[order] = ranks
    return out


def delong_auc_variance(scores: FloatArray, labels: BoolArray) -> tuple[float, float]:
    """Return ``(auc, var_auc)`` by the fast DeLong algorithm.

    The variance is the sum of the two placement-value components,
    ``S01 / n1 + S10 / n0``; both use ``ddof=1`` sample variances of the
    placement values, which is the standard DeLong estimator.
    """
    pos = scores[labels]
    neg = scores[~labels]
    n1 = pos.shape[0]
    n0 = neg.shape[0]
    if n1 == 0 or n0 == 0:
        raise ValueError("DeLong variance is undefined for a single-class outcome")

    tz = midrank(np.concatenate([pos, neg]))
    tx = midrank(pos)
    ty = midrank(neg)
    auc = (float(tz[:n1].sum()) - n1 * (n1 + 1) / 2.0) / (n1 * n0)

    v01 = (tz[:n1] - tx) / n0          # placement values of the positives
    v10 = 1.0 - (tz[n1:] - ty) / n1    # placement values of the negatives

    s01 = float(np.var(v01, ddof=1)) if n1 > 1 else 0.0
    s10 = float(np.var(v10, ddof=1)) if n0 > 1 else 0.0
    var_auc = s01 / n1 + s10 / n0
    return auc, float(var_auc)


def delong_auroc_ci(
    scores: FloatArray,
    labels: BoolArray,
    *,
    level: float = 0.95,
    logit: bool = True,
) -> ConfidenceInterval:
    """DeLong confidence interval for a fixed-score AUROC.

    Parameters
    ----------
    logit:
        Build the normal interval on the ``logit`` scale and transform back.
        Default ``True``: the AUC is bounded in ``[0, 1]`` and its sampling
        distribution is skewed near the boundaries, where the untransformed
        Wald interval routinely leaves the unit interval.  Set ``False`` for
        the literal 1988 formulation.
    """
    auc, var_auc = delong_auc_variance(scores, labels)
    z = float(stats.norm.ppf(0.5 + level / 2.0))
    if not logit or var_auc <= 0.0 or auc <= 0.0 or auc >= 1.0:
        half = z * float(np.sqrt(max(var_auc, 0.0)))
        return ConfidenceInterval(
            low=float(np.clip(auc - half, 0.0, 1.0)),
            high=float(np.clip(auc + half, 0.0, 1.0)),
            level=level,
            method="delong",
        )
    eta = np.log(auc / (1.0 - auc))
    se_eta = np.sqrt(var_auc) / (auc * (1.0 - auc))
    lo = 1.0 / (1.0 + np.exp(-(eta - z * se_eta)))
    hi = 1.0 / (1.0 + np.exp(-(eta + z * se_eta)))
    return ConfidenceInterval(
        low=float(lo), high=float(hi), level=level, method="delong_logit"
    )


def stratified_bootstrap_ci(
    statistic,
    scores: FloatArray,
    labels: BoolArray,
    *,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> ConfidenceInterval:
    """Percentile CI resampling cases and controls **separately**.

    Stratifying the resample holds the prevalence fixed across replicates,
    which matters for average precision because its chance level *is* the
    prevalence: an unstratified bootstrap would mix sampling noise in the
    statistic with sampling noise in its own baseline.

    ``statistic`` is called as ``statistic(scores, labels)`` and must return a
    float.
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(labels)
    neg_idx = np.flatnonzero(~labels)
    if pos_idx.size == 0 or neg_idx.size == 0:
        raise ValueError("stratified bootstrap requires both classes to be present")

    values = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        take = np.concatenate(
            [
                rng.choice(pos_idx, size=pos_idx.size, replace=True),
                rng.choice(neg_idx, size=neg_idx.size, replace=True),
            ]
        )
        values[b] = statistic(scores[take], labels[take])
    alpha = 1.0 - level
    lo, hi = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return ConfidenceInterval(
        low=float(lo),
        high=float(hi),
        level=level,
        method=f"stratified_bootstrap_percentile[B={n_resamples},seed={seed}]",
    )


def clopper_pearson(
    successes: int, trials: int, *, level: float = 0.95
) -> ConfidenceInterval:
    """Exact Clopper-Pearson interval for the exceedance proportion ``r / K``.

    Sec. 3.4 asks for this on the null exceedance count.  It is the interval on
    the *proportion*, never on the p-value itself.
    """
    if trials < 0 or not (0 <= successes <= trials):
        raise ValueError(f"invalid binomial counts: {successes}/{trials}")
    alpha = 1.0 - level
    low = 0.0 if successes == 0 else float(stats.beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    high = 1.0 if successes == trials else float(
        stats.beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes)
    )
    return ConfidenceInterval(low=low, high=high, level=level, method="clopper_pearson")
