"""AUROC and average precision -- the two discrimination metrics of Sec. 6.

AUROC is the default primary statistic: it is prevalence-invariant, so the
candidate and every null draw sit on a common scale whose chance value is
exactly 0.5.  Average precision is *always* reported alongside it, and never
on its own -- :class:`AveragePrecisionReport` bundles the raw AP with its
baseline prevalence and the normalised ``AP_norm`` so that a bare AP number
cannot escape into a report without its own chance level attached.

Both classes satisfy the :class:`~signull.types.Metric` protocol.  Neither
applies any direction policy: folding about the chance level is the job of
:class:`~signull.metrics.direction.DirectedMetric`, which is the only object in
the codebase allowed to do it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from ..types import BinaryOutcome, BoolArray, FloatArray, MetricName, SampleScores
from .floors import validate_scores
from .intervals import (
    ConfidenceInterval,
    delong_auroc_ci,
    midrank,
    stratified_bootstrap_ci,
)

__all__ = [
    "auroc_from_arrays",
    "average_precision_from_arrays",
    "tie_fraction",
    "AurocMetric",
    "AveragePrecisionMetric",
    "AveragePrecisionReport",
    "AurocReport",
]


def auroc_from_arrays(scores: FloatArray, labels: BoolArray) -> float:
    """AUROC by the Mann-Whitney rank statistic with mid-rank tie handling.

    Equivalent to ``sklearn.metrics.roc_auc_score`` including on ties, but
    implemented here so the project's primary statistic does not depend on a
    third-party tie convention that could change under it.
    """
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    n1 = int(np.count_nonzero(labels))
    n0 = labels.shape[0] - n1
    if n1 == 0 or n0 == 0:
        raise ValueError("AUROC is undefined for a single-class outcome")
    ranks = midrank(scores)
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def average_precision_from_arrays(scores: FloatArray, labels: BoolArray) -> float:
    """Step-wise average precision ``sum_i (R_i - R_{i-1}) * P_i``.

    Sec. 6 forbids the trapezoidal interpolation of the PR curve, which is
    optimistically biased.  Tied scores are collapsed into a single operating
    point, matching ``sklearn.metrics.average_precision_score``.
    """
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(np.count_nonzero(labels))
    if n_pos == 0 or n_pos == labels.shape[0]:
        raise ValueError("average precision is undefined for a single-class outcome")

    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    y = labels[order]
    # One operating point per *distinct* score: ties must not be split.
    distinct = np.flatnonzero(np.diff(s))
    cut = np.concatenate([distinct, np.array([s.shape[0] - 1])])
    tp = np.cumsum(y)[cut].astype(np.float64)
    fp = np.cumsum(~y)[cut].astype(np.float64)
    precision = tp / (tp + fp)
    recall = tp / n_pos
    prev_recall = np.concatenate([np.array([0.0]), recall[:-1]])
    return float(np.sum((recall - prev_recall) * precision))


def tie_fraction(scores: FloatArray) -> float:
    """Fraction of unordered sample pairs that are tied in ``scores``.

    Sec. 6 asks for a diagnostic when ties are a meaningful fraction of pairs;
    with small ``n1`` a heavily tied score makes the AUC coarse and the
    permutation null granular.
    """
    n = scores.shape[0]
    if n < 2:
        return 0.0
    _, counts = np.unique(scores, return_counts=True)
    tied_pairs = float(np.sum(counts * (counts - 1) / 2.0))
    return tied_pairs / (n * (n - 1) / 2.0)


@dataclass(frozen=True, slots=True)
class AurocReport:
    """AUROC together with everything Sec. 6 requires be shown with it."""

    auroc: float
    ci: ConfidenceInterval | None
    chance_level: float
    n_samples: int
    n_positive: int
    tie_fraction: float


@dataclass(frozen=True, slots=True)
class AveragePrecisionReport:
    """Average precision with its baseline and normalised form.

    Sec. 6: AP is only interpretable next to the prevalence, because the
    prevalence *is* its chance level.  This object exists so that AP cannot be
    handed to a report layer stripped of that context.

    ``ap_norm = (ap - baseline) / (1 - baseline)`` is 0 at chance and 1 at a
    perfect ranking, which makes it comparable across cohorts of different
    balance in the way raw AP is not.
    """

    average_precision: float
    baseline: float
    ap_norm: float
    ci: ConfidenceInterval | None
    n_samples: int
    n_positive: int

    @property
    def chance_level(self) -> float:
        """Alias for :attr:`baseline`; the prevalence."""
        return self.baseline


@dataclass(frozen=True, slots=True)
class AurocMetric:
    """Area under the ROC curve.  Chance level 0.5, independent of prevalence.

    Attributes
    ----------
    enforce_floors:
        Apply the Sec. 6 cohort floors (default).  Setting ``False`` is an
        explicit, recorded decision to compute a descriptive number on a cohort
        too small to support a verdict.
    ci_level, ci_logit:
        DeLong interval configuration used by :meth:`report`.
    """

    enforce_floors: bool = True
    ci_level: float = 0.95
    ci_logit: bool = True

    @property
    def name(self) -> MetricName:
        """:attr:`~signull.types.MetricName.AUROC`."""
        return MetricName.AUROC

    @property
    def greater_is_better(self) -> bool:
        """``True``."""
        return True

    @property
    def params(self) -> Mapping[str, object]:
        """Serialisable configuration, for config capture."""
        return {
            "enforce_floors": self.enforce_floors,
            "ci_level": self.ci_level,
            "ci_logit": self.ci_logit,
        }

    def chance_level(self, outcome: BinaryOutcome) -> float:
        """0.5 for every outcome."""
        return 0.5

    def __call__(self, scores: SampleScores, outcome: BinaryOutcome) -> float:
        """Raw AUROC, with no direction policy applied."""
        validate_scores(scores, outcome, enforce_floors=self.enforce_floors)
        return auroc_from_arrays(scores.values, outcome.labels)

    def negated(self, scores: SampleScores, outcome: BinaryOutcome) -> float:
        """AUROC of the sign-flipped score, i.e. ``1 - AUROC``.

        Used only by :class:`~signull.metrics.direction.DirectedMetric`.
        """
        return 1.0 - self(scores, outcome)

    def report(self, scores: SampleScores, outcome: BinaryOutcome) -> AurocReport:
        """AUROC plus DeLong CI and the tie diagnostic."""
        value = self(scores, outcome)
        ci = delong_auroc_ci(
            scores.values, outcome.labels, level=self.ci_level, logit=self.ci_logit
        )
        return AurocReport(
            auroc=value,
            ci=ci,
            chance_level=0.5,
            n_samples=outcome.n_samples,
            n_positive=outcome.n_positive,
            tie_fraction=tie_fraction(scores.values),
        )


@dataclass(frozen=True, slots=True)
class AveragePrecisionMetric:
    """Step-wise average precision; chance level is the prevalence.

    Reported alongside AUROC always (Sec. 6), and made the headline only when
    the stated use is ranking / triage of a top-k list.
    """

    enforce_floors: bool = True
    ci_level: float = 0.95
    n_bootstrap: int = 2000
    bootstrap_seed: int = 0

    @property
    def name(self) -> MetricName:
        """:attr:`~signull.types.MetricName.AVERAGE_PRECISION`."""
        return MetricName.AVERAGE_PRECISION

    @property
    def greater_is_better(self) -> bool:
        """``True``."""
        return True

    @property
    def params(self) -> Mapping[str, object]:
        """Serialisable configuration, for config capture."""
        return {
            "enforce_floors": self.enforce_floors,
            "ci_level": self.ci_level,
            "n_bootstrap": self.n_bootstrap,
            "bootstrap_seed": self.bootstrap_seed,
        }

    def chance_level(self, outcome: BinaryOutcome) -> float:
        """The positive-class prevalence."""
        return outcome.prevalence

    def __call__(self, scores: SampleScores, outcome: BinaryOutcome) -> float:
        """Raw average precision, with no direction policy applied."""
        validate_scores(scores, outcome, enforce_floors=self.enforce_floors)
        return average_precision_from_arrays(scores.values, outcome.labels)

    def negated(self, scores: SampleScores, outcome: BinaryOutcome) -> float:
        """Average precision of the sign-flipped score.

        Unlike AUROC, average precision is **not** recovered by reflecting the
        value about its chance level: ``AP(-s) != 2*pi - AP(s)``.  The direction
        policy therefore has to re-rank, which is what this does.
        """
        validate_scores(scores, outcome, enforce_floors=self.enforce_floors)
        return average_precision_from_arrays(-scores.values, outcome.labels)

    def report(
        self, scores: SampleScores, outcome: BinaryOutcome, *, with_ci: bool = True
    ) -> AveragePrecisionReport:
        """AP, its baseline prevalence, ``AP_norm`` and a stratified bootstrap CI."""
        value = self(scores, outcome)
        baseline = outcome.prevalence
        ap_norm = (
            (value - baseline) / (1.0 - baseline) if baseline < 1.0 else float("nan")
        )
        ci = None
        if with_ci:
            ci = stratified_bootstrap_ci(
                average_precision_from_arrays,
                scores.values,
                outcome.labels,
                n_resamples=self.n_bootstrap,
                level=self.ci_level,
                seed=self.bootstrap_seed,
            )
        return AveragePrecisionReport(
            average_precision=value,
            baseline=baseline,
            ap_norm=ap_norm,
            ci=ci,
            n_samples=outcome.n_samples,
            n_positive=outcome.n_positive,
        )
