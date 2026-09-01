"""Discrimination metrics, direction policy and confidence intervals.

Public surface
--------------

Metrics (both satisfy :class:`signull.types.Metric`)
    :class:`AurocMetric` -- the default primary statistic.
    :class:`AveragePrecisionMetric` -- always reported alongside AUROC, and
    only ever through :class:`AveragePrecisionReport`, which carries the
    baseline prevalence and ``AP_norm`` with it.

Direction
    :class:`DirectedMetric` is the only object permitted to fold a statistic
    about its chance level.  See :mod:`signull.metrics.direction` for why that
    makes asymmetric application structurally impossible.

Floors
    :func:`check_cohort_floors` enforces ``n1, n0 >= 8`` and ``N >= 30``
    (Sec. 6) inside every metric call by default.

Intervals
    DeLong for AUROC, stratified bootstrap for AP, Clopper-Pearson for the null
    exceedance proportion.  Never a bootstrap CI on a p-value.

This package depends only on :mod:`signull.types`.  It imports nothing from
``data``, ``nulls``, ``scoring`` or ``report``.
"""

from __future__ import annotations

from typing import Final, Mapping

from ..types import MetricName
from .direction import (
    AsymmetricDirectionError,
    DirectedMetric,
    MetricFingerprint,
    NullStatisticSummary,
    ReflectableMetric,
    StatisticValue,
    assert_comparable,
    summarise_null_statistics,
)
from .discrimination import (
    AurocMetric,
    AurocReport,
    AveragePrecisionMetric,
    AveragePrecisionReport,
    auroc_from_arrays,
    average_precision_from_arrays,
    tie_fraction,
)
from .floors import (
    MIN_NEGATIVES,
    MIN_POSITIVES,
    MIN_SAMPLES,
    CohortTooSmallError,
    DegenerateOutcomeError,
    MisalignedScoresError,
    check_cohort_floors,
    validate_scores,
)
from .intervals import (
    ConfidenceInterval,
    clopper_pearson,
    delong_auc_variance,
    delong_auroc_ci,
    midrank,
    stratified_bootstrap_ci,
)

__all__ = [
    "REGISTRY",
    "get",
    "AsymmetricDirectionError",
    "AurocMetric",
    "AurocReport",
    "AveragePrecisionMetric",
    "AveragePrecisionReport",
    "CohortTooSmallError",
    "ConfidenceInterval",
    "DegenerateOutcomeError",
    "DirectedMetric",
    "MIN_NEGATIVES",
    "MIN_POSITIVES",
    "MIN_SAMPLES",
    "MetricFingerprint",
    "MisalignedScoresError",
    "NullStatisticSummary",
    "ReflectableMetric",
    "StatisticValue",
    "assert_comparable",
    "auroc_from_arrays",
    "average_precision_from_arrays",
    "check_cohort_floors",
    "clopper_pearson",
    "delong_auc_variance",
    "delong_auroc_ci",
    "midrank",
    "stratified_bootstrap_ci",
    "summarise_null_statistics",
    "tie_fraction",
    "validate_scores",
]

#: ``MetricName -> factory``.  Mirrors ``scoring.REGISTRY`` and
#: ``nulls.REGISTRY`` so the CLI resolves every strategy the same way
#: (``docs/architecture.md`` Sec. 2).
REGISTRY: Final[Mapping[MetricName, type]] = {
    MetricName.AUROC: AurocMetric,
    MetricName.AVERAGE_PRECISION: AveragePrecisionMetric,
}


def get(name: MetricName | str, **params: object):
    """Instantiate the metric registered under ``name``."""
    key = MetricName(name)
    return REGISTRY[key](**params)  # type: ignore[operator]
