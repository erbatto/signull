"""Direction handling -- the single place in the codebase allowed to fold a metric.

Sec. 3.1 and Sec. 5(A.1): when the candidate carries no pre-specified
direction, the statistic is ``T* = max(AUC, 1 - AUC)``.  The ``max`` absorbs the
one bit of sign information that would otherwise be fitted on ``y``.  Applying
it to the candidate but not to the nulls is a real, easy-to-make leak that
silently inflates significance -- an unsigned random gene set's score points
either way, so a one-sided AUROC systematically *understates* the null and
flatters the candidate.

How asymmetry is made structurally impossible
---------------------------------------------

1. The raw metrics in :mod:`signull.metrics.discrimination` have no folding
   code at all.  There is no ``symmetrize=True`` argument anywhere on them, so
   no call site can turn folding on for one input and off for another.

2. Folding exists only inside :class:`DirectedMetric`, which is a frozen,
   slotted value object holding the policy.  The policy is fixed at
   construction, not passed per call, so the same object cannot fold one input
   and not the next.

3. :class:`DirectedMetric` refuses to wrap another :class:`DirectedMetric`, so
   a value cannot be folded twice by accidental nesting.

4. Every value it produces is returned as a :class:`StatisticValue` stamped
   with a :class:`MetricFingerprint` recording the metric, the policy, and the
   evaluation context.  :func:`assert_comparable` -- which the null-comparison
   path must call before a candidate statistic is compared to a null
   distribution -- raises :class:`AsymmetricDirectionError` when those stamps
   disagree.  Building two differently-configured metrics and mixing their
   outputs is therefore detected rather than silently wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from ..types import BinaryOutcome, DirectionPolicy, MetricName, SampleScores

__all__ = [
    "AsymmetricDirectionError",
    "ReflectableMetric",
    "MetricFingerprint",
    "StatisticValue",
    "DirectedMetric",
    "assert_comparable",
    "NullStatisticSummary",
    "summarise_null_statistics",
]


class AsymmetricDirectionError(ValueError):
    """Raised when statistics computed under different policies are compared.

    The canonical trigger is a candidate statistic folded with
    ``max(AUC, 1-AUC)`` being compared against an unfolded null distribution.
    """


@runtime_checkable
class ReflectableMetric(Protocol):
    """A :class:`~signull.types.Metric` that can also score the negated ranking.

    ``negated`` is what makes a direction-free statistic computable for metrics
    whose chance level is not a reflection centre.  For AUROC,
    ``negated == 1 - value``, so ``max(value, negated)`` is exactly the
    ``max(m, 2 * chance - m)`` of the contract.  For average precision the
    reflection identity does not hold, so the ranking is genuinely reversed.
    """

    @property
    def name(self) -> MetricName: ...

    @property
    def greater_is_better(self) -> bool: ...

    def chance_level(self, outcome: BinaryOutcome) -> float: ...

    def __call__(self, scores: SampleScores, outcome: BinaryOutcome) -> float: ...

    def negated(self, scores: SampleScores, outcome: BinaryOutcome) -> float: ...


@dataclass(frozen=True, slots=True)
class MetricFingerprint:
    """Identity of *how* a statistic was produced.

    Two statistics are comparable only if their fingerprints are equal.  This
    covers invariant I2 of the design document: candidate and nulls must share
    the identical statistic, direction policy, scorer, and adjustment.

    Attributes
    ----------
    metric:
        Which metric was computed.
    policy:
        Which :class:`~signull.types.DirectionPolicy` was applied.
    scorer:
        ``ScoringMethodName`` value of the scorer that produced the scores, or
        ``None`` when the statistic was computed from a bare score vector.
    scorer_params:
        Canonical string form of the scorer hyper-parameters.
    adjustment:
        Identifier of the score adjustment applied (e.g. ``"none"``,
        ``"pc1"``, ``"meta_pcna"``).
    metric_params:
        Canonical string form of the metric configuration.
    """

    metric: MetricName
    policy: DirectionPolicy
    scorer: str | None = None
    scorer_params: str = ""
    adjustment: str = "none"
    metric_params: str = ""

    def with_context(
        self,
        *,
        scorer: str | None = None,
        scorer_params: str | None = None,
        adjustment: str | None = None,
    ) -> "MetricFingerprint":
        """Return a copy with the scoring context filled in."""
        return replace(
            self,
            scorer=self.scorer if scorer is None else scorer,
            scorer_params=self.scorer_params if scorer_params is None else scorer_params,
            adjustment=self.adjustment if adjustment is None else adjustment,
        )


@dataclass(frozen=True, slots=True)
class StatisticValue:
    """A metric value plus the fingerprint of how it was produced.

    ``raw`` is the metric before any direction policy; ``value`` is after.  Both
    are kept because Sec. 9 requires the report to show what was actually
    compared, and because a large gap between them is itself informative (it
    means the signature ranks the *wrong* way round -- Sec. 8, F10).
    """

    value: float
    raw: float
    chance_level: float
    fingerprint: MetricFingerprint

    @property
    def was_reflected(self) -> bool:
        """``True`` when the direction policy flipped this particular value."""
        return self.value != self.raw

    def __float__(self) -> float:
        """The policy-applied statistic."""
        return float(self.value)


@dataclass(frozen=True, slots=True)
class DirectedMetric:
    """A metric bound to a :class:`~signull.types.DirectionPolicy`.

    Implements the :class:`~signull.types.Metric` protocol, so it drops into
    any place a metric is expected -- but unlike the raw metrics it carries the
    policy as immutable state rather than as a per-call argument.  An evaluator
    holds exactly one of these and routes candidate *and* every null draw
    through it, which is what makes asymmetric application impossible by
    construction rather than by discipline.

    Attributes
    ----------
    base:
        The underlying :class:`ReflectableMetric`.  Must not itself be a
        ``DirectedMetric``.
    policy:
        :attr:`~signull.types.DirectionPolicy.SYMMETRIZED` (default) folds the
        statistic to be direction-free.
        :attr:`~signull.types.DirectionPolicy.AS_GIVEN` is correct only when
        the candidate carries a published, pre-specified direction *and* the
        nulls inherit that same signed structure.
    """

    base: ReflectableMetric
    policy: DirectionPolicy = DirectionPolicy.SYMMETRIZED

    def __post_init__(self) -> None:
        if isinstance(self.base, DirectedMetric):
            raise TypeError(
                "DirectedMetric must not wrap another DirectedMetric: folding "
                "a statistic twice is never correct"
            )
        if not isinstance(self.policy, DirectionPolicy):
            raise TypeError(f"policy must be a DirectionPolicy, got {self.policy!r}")
        if not hasattr(self.base, "negated"):
            raise TypeError(
                f"{type(self.base).__name__} does not implement ReflectableMetric."
                "negated(); a direction policy cannot be applied to it"
            )

    # -- Metric protocol -----------------------------------------------------

    @property
    def name(self) -> MetricName:
        """Identity of the underlying metric."""
        return self.base.name

    @property
    def greater_is_better(self) -> bool:
        """Direction of merit of the underlying metric."""
        return self.base.greater_is_better

    @property
    def params(self) -> Mapping[str, object]:
        """Serialisable configuration including the direction policy."""
        base_params = getattr(self.base, "params", {})
        return {"policy": self.policy.value, "base": dict(base_params)}

    def chance_level(self, outcome: BinaryOutcome) -> float:
        """Chance level of the underlying metric.

        Unchanged by the policy: ``max(m, 2*chance - m)`` has the same chance
        value ``chance``, it merely truncates the lower tail.
        """
        return self.base.chance_level(outcome)

    def __call__(self, scores: SampleScores, outcome: BinaryOutcome) -> float:
        """The policy-applied statistic as a plain float."""
        return self.evaluate(scores, outcome).value

    # -- Fingerprinted evaluation -------------------------------------------

    def fingerprint(self) -> MetricFingerprint:
        """Fingerprint for statistics produced by this metric, sans scoring context."""
        base_params = getattr(self.base, "params", {})
        canonical = ";".join(f"{k}={base_params[k]!r}" for k in sorted(base_params))
        return MetricFingerprint(
            metric=self.base.name,
            policy=self.policy,
            metric_params=canonical,
        )

    def evaluate(self, scores: SampleScores, outcome: BinaryOutcome) -> StatisticValue:
        """Compute the statistic and stamp it with this metric's fingerprint."""
        raw = self.base(scores, outcome)
        chance = self.base.chance_level(outcome)
        if self.policy is DirectionPolicy.AS_GIVEN:
            value = raw
        elif self.policy is DirectionPolicy.SYMMETRIZED:
            # For AUROC this is exactly max(AUC, 1 - AUC); for average
            # precision it is max over the two rankings, since AP is not
            # recovered by reflecting about its chance level.
            value = max(raw, self.base.negated(scores, outcome))
        else:  # pragma: no cover - exhaustive over the enum
            raise ValueError(f"unhandled direction policy {self.policy!r}")
        return StatisticValue(
            value=float(value),
            raw=float(raw),
            chance_level=float(chance),
            fingerprint=self.fingerprint(),
        )


def assert_comparable(
    observed: StatisticValue, nulls: Sequence[StatisticValue]
) -> None:
    """Raise unless every null statistic shares ``observed``'s fingerprint.

    This is the runtime enforcement of invariant I2.  The failure it is built
    to catch is the classic one: ``max(AUC, 1-AUC)`` on the candidate and a
    plain AUC on the nulls.

    Raises
    ------
    AsymmetricDirectionError
        On any fingerprint mismatch, naming the first offending draw and the
        component that differs.
    """
    want = observed.fingerprint
    for i, null in enumerate(nulls):
        got = null.fingerprint
        if got == want:
            continue
        differing = [
            field
            for field in (
                "metric",
                "policy",
                "scorer",
                "scorer_params",
                "adjustment",
                "metric_params",
            )
            if getattr(got, field) != getattr(want, field)
        ]
        raise AsymmetricDirectionError(
            f"null draw {i} was not evaluated like the candidate; differs on "
            f"{differing}: candidate={want!r} null={got!r}"
        )


@dataclass(frozen=True, slots=True)
class NullStatisticSummary:
    """Descriptive summary of a null statistic distribution (Sec. 3.4).

    Deliberately contains **no p-value**: converting exceedances into a
    p-value, choosing ``K``, and the ``+1`` correction all belong to the
    ``nulls`` package.  What lives here is the part that is a property of the
    metric -- the location, spread and upper tail of the null on the metric's
    own scale, plus the standardised effect.
    """

    n_draws: int
    median: float
    q25: float
    q75: float
    p95: float
    mean: float
    sd: float
    observed: float
    standardised_effect: float
    chance_level: float

    @property
    def iqr(self) -> float:
        """``q75 - q25``."""
        return self.q75 - self.q25


def summarise_null_statistics(
    observed: StatisticValue,
    nulls: Sequence[StatisticValue],
    *,
    check_comparable: bool = True,
) -> NullStatisticSummary:
    """Sec. 3.4 descriptive statistics of a null distribution.

    Calls :func:`assert_comparable` first by default: a summary of a null
    distribution that was not produced the same way as the observed statistic
    is worse than no summary, because it looks authoritative.
    """
    if check_comparable:
        assert_comparable(observed, nulls)
    if not nulls:
        raise ValueError("cannot summarise an empty null distribution")
    values = np.asarray([n.value for n in nulls], dtype=np.float64)
    sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    z = (
        (observed.value - float(np.mean(values))) / sd
        if sd > 0.0
        else float("nan")
    )
    q25, median, q75, p95 = (
        float(v) for v in np.quantile(values, [0.25, 0.5, 0.75, 0.95])
    )
    return NullStatisticSummary(
        n_draws=values.size,
        median=median,
        q25=q25,
        q75=q75,
        p95=p95,
        mean=float(np.mean(values)),
        sd=sd,
        observed=observed.value,
        standardised_effect=float(z),
        chance_level=observed.chance_level,
    )
