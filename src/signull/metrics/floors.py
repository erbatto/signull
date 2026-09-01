"""Cohort-size floors and input validation shared by every metric.

``docs/statistical-design.md`` Sec. 6 fixes the floors: ``n1 >= 8``,
``n0 >= 8``, ``N >= 30``.  Below them the AUROC takes too few distinct values
for the permutation null to be usable, so the tool must refuse rather than
emit a number that looks like evidence.

The floors live here -- not in the caller -- so that *every* metric evaluation
in the codebase passes through the same gate.  A caller that genuinely wants a
descriptive number on an undersized cohort must say so explicitly by
constructing the metric with ``enforce_floors=False``; there is no way to do it
by accident.
"""

from __future__ import annotations

from typing import Final

from ..types import BinaryOutcome, SampleScores

__all__ = [
    "MIN_POSITIVES",
    "MIN_NEGATIVES",
    "MIN_SAMPLES",
    "CohortTooSmallError",
    "DegenerateOutcomeError",
    "MisalignedScoresError",
    "check_cohort_floors",
    "validate_scores",
]

#: Minimum positives required before a discrimination metric may be emitted.
MIN_POSITIVES: Final[int] = 8
#: Minimum negatives required before a discrimination metric may be emitted.
MIN_NEGATIVES: Final[int] = 8
#: Minimum cohort size required before a discrimination metric may be emitted.
MIN_SAMPLES: Final[int] = 30


class CohortTooSmallError(ValueError):
    """Raised when a cohort falls below the Sec. 6 floors.

    Carries the observed counts so a report layer can render the refusal
    without recomputing them.
    """

    def __init__(self, n_samples: int, n_positive: int, n_negative: int) -> None:
        self.n_samples = n_samples
        self.n_positive = n_positive
        self.n_negative = n_negative
        super().__init__(
            "cohort below the statistical-design Sec. 6 floors: "
            f"N={n_samples} (need >= {MIN_SAMPLES}), "
            f"n1={n_positive} (need >= {MIN_POSITIVES}), "
            f"n0={n_negative} (need >= {MIN_NEGATIVES})"
        )


class DegenerateOutcomeError(ValueError):
    """Raised when one outcome class is empty, so no discrimination exists."""


class MisalignedScoresError(ValueError):
    """Raised when scores and outcome do not share sample identity and order.

    Never repaired by reordering: a silent reorder is how a score vector gets
    quietly paired with the wrong labels.
    """


def check_cohort_floors(outcome: BinaryOutcome) -> None:
    """Raise unless ``outcome`` satisfies the Sec. 6 floors.

    Raises
    ------
    DegenerateOutcomeError
        When one class is empty.
    CohortTooSmallError
        When any of ``N``, ``n1``, ``n0`` is below its floor.
    """
    if outcome.is_degenerate:
        raise DegenerateOutcomeError(
            f"outcome {outcome.name!r} has n1={outcome.n_positive}, "
            f"n0={outcome.n_negative}; no discrimination metric is defined"
        )
    if (
        outcome.n_samples < MIN_SAMPLES
        or outcome.n_positive < MIN_POSITIVES
        or outcome.n_negative < MIN_NEGATIVES
    ):
        raise CohortTooSmallError(
            outcome.n_samples, outcome.n_positive, outcome.n_negative
        )


def validate_scores(
    scores: SampleScores, outcome: BinaryOutcome, *, enforce_floors: bool = True
) -> None:
    """Validate a (scores, outcome) pair before any metric touches it.

    Checks sample alignment, score finiteness and -- unless explicitly waived --
    the cohort-size floors.
    """
    if scores.sample_ids != outcome.sample_ids:
        raise MisalignedScoresError(
            "scores.sample_ids != outcome.sample_ids; refusing to reorder "
            f"({scores.n_samples} scored samples vs {outcome.n_samples} labelled)"
        )
    values = scores.values
    if values.shape != (outcome.n_samples,):
        raise MisalignedScoresError(
            f"score vector has shape {values.shape}, expected ({outcome.n_samples},)"
        )
    import numpy as np

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "score vector contains non-finite values; this is a contract "
            "violation of SampleScores.values"
        )
    if enforce_floors:
        check_cohort_floors(outcome)
    elif outcome.is_degenerate:
        raise DegenerateOutcomeError(
            f"outcome {outcome.name!r} is single-class; not waivable"
        )
