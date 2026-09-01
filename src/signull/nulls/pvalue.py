"""The empirical (Monte Carlo) p-value estimator.

This module implements one formula and defends one property::

    p_hat = (1 + #{k : T_k at least as extreme as T_obs}) / (1 + K)

**It is structurally incapable of returning zero.**  With ``r >= 0`` and
``K >= 1`` the numerator is at least 1 and the denominator is finite, so
``p_hat >= 1 / (K + 1) > 0``.  This is not a cosmetic detail.  The plug-in
estimator ``r / K`` used by comparable tools -- Bioconductor ``SigCheck``'s
``$checkPval`` among them -- returns exactly ``0`` whenever no null draw beats
the candidate, which asserts infinite evidence from a finite number of draws.
``(1 + r) / (1 + K)`` is the exactly valid Monte Carlo p-value: under the null
it satisfies ``P(p_hat <= alpha) <= alpha`` for *every* ``K``
(North, Curtis & Sham 2002; Phipson & Smyth 2010, "Permutation p-values should
never be zero").

**Ties count as exceedances** (``>=``, not ``>``).  With a small positive class
the AUROC takes few distinct values and ties are common; counting them makes the
test conservative, which is the only safe direction.

**One-sided, upper tail, by default.**  ``docs/statistical-design.md`` Sec. 3.1:
"significantly worse than random" is not a claim this tool makes.

**Draw-count floor.**  Sec. 3.3 and contract amendment 1.1.0 set a hard floor of
:data:`signull.types.MIN_DRAWS` draws for any null that gates a claim; below it
the relative Monte Carlo SE at ``p = 0.05`` exceeds 10% and this estimator
refuses rather than emitting a number that reads as precise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np

from ..types import (
    MIN_DRAWS,
    Alternative,
    Diagnostic,
    FloatArray,
    Severity,
)

__all__ = [
    "EmpiricalPValue",
    "empirical_p_value",
    "p_value_floor",
]

#: Diagnostic code emitted when some null draws produced a non-finite statistic.
INVALID_DRAWS_CODE: Final[str] = "invalid_null_statistics"
#: Diagnostic code emitted when the estimate is pinned at its attainable floor.
AT_FLOOR_CODE: Final[str] = "p_value_at_resolution_floor"


def p_value_floor(n_valid_draws: int) -> float:
    """Smallest p-value attainable with ``n_valid_draws`` draws: ``1/(K+1)``."""
    return 1.0 / (n_valid_draws + 1)


@dataclass(frozen=True, slots=True)
class EmpiricalPValue:
    """Result of the add-one Monte Carlo p-value estimator.

    Attributes
    ----------
    p_value:
        ``(1 + n_exceedances) / (1 + n_valid_draws)``.  Always strictly
        positive and always ``<= 1``.
    n_exceedances:
        Number of null statistics at least as extreme as the observed one, ties
        included.
    n_valid_draws:
        Null statistics that were finite and therefore usable.
    n_draws_requested:
        Draws the caller asked the null model for.  A gap to ``n_valid_draws``
        is a red flag the report must show.
    observed_statistic:
        The candidate's statistic, echoed for traceability.
    alternative:
        Tail used.
    diagnostics:
        Non-fatal conditions, as data rather than log lines.
    """

    p_value: float
    n_exceedances: int
    n_valid_draws: int
    n_draws_requested: int
    observed_statistic: float
    alternative: Alternative = Alternative.GREATER
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    @property
    def floor(self) -> float:
        """Smallest p-value attainable at this number of valid draws."""
        return p_value_floor(self.n_valid_draws)

    @property
    def at_resolution_floor(self) -> bool:
        """``True`` when no null draw was as extreme, so ``p`` is only an upper bound.

        The report must render this as ``p < 1/(K+1)`` rather than quoting the
        number as a point estimate.
        """
        return self.n_exceedances == 0

    @property
    def monte_carlo_se(self) -> float:
        """``sqrt(p (1 - p) / K)`` -- the sampling error of the estimate itself."""
        k = self.n_valid_draws
        if k <= 0:
            return float("nan")
        p = self.p_value
        return float(np.sqrt(p * (1.0 - p) / k))

    def __post_init__(self) -> None:
        # The whole point of the module.  Cheap, and it fails at construction
        # rather than in a report six steps later.
        if not (0.0 < self.p_value <= 1.0):
            raise AssertionError(
                f"empirical p-value must lie in (0, 1]; got {self.p_value!r}. "
                "The add-one estimator cannot produce this -- something bypassed it."
            )


def _count_exceedances(
    observed: float, null_statistics: FloatArray, alternative: Alternative
) -> int:
    """Count null draws at least as extreme as ``observed``; ties count."""
    if alternative is Alternative.GREATER:
        return int(np.count_nonzero(null_statistics >= observed))
    if alternative is Alternative.LESS:
        return int(np.count_nonzero(null_statistics <= observed))
    raise ValueError(f"unhandled alternative {alternative!r}")


def empirical_p_value(
    observed_statistic: float,
    null_statistics: np.ndarray | list[float] | tuple[float, ...],
    *,
    alternative: Alternative = Alternative.GREATER,
    n_draws_requested: int | None = None,
    min_draws: int = MIN_DRAWS,
    enforce_min_draws: bool = True,
) -> EmpiricalPValue:
    """Add-one empirical p-value of ``observed_statistic`` against a null sample.

    Parameters
    ----------
    observed_statistic:
        The candidate's statistic, after the run's
        :class:`~signull.types.DirectionPolicy` has been applied.  Must be
        finite: a non-finite observed value means the evaluation upstream failed
        and no p-value is defined for it.
    null_statistics:
        Statistics of the null draws, *after the identical* direction policy.
        Non-finite entries are treated as invalid draws: they are dropped from
        the denominator and reported, never counted as non-exceedances (which
        would bias the p-value downward) and never re-drawn (which would change
        the null distribution).
    alternative:
        Tail.  :attr:`~signull.types.Alternative.GREATER` (default) is the only
        one ``docs/statistical-design.md`` sanctions for a headline claim;
        ``LESS`` and ``TWO_SIDED`` exist for contract completeness and
        diagnostics.
    n_draws_requested:
        Draws asked for.  Defaults to ``len(null_statistics)``.
    min_draws:
        Floor on *valid* draws; defaults to :data:`signull.types.MIN_DRAWS`.
    enforce_min_draws:
        Whether to refuse below ``min_draws``.  Set ``False`` only for
        exploratory or diagnostic (non-gating) nulls; the report must then say
        the number is not gating.

    Returns
    -------
    EmpiricalPValue

    Raises
    ------
    ValueError
        Non-finite ``observed_statistic``; zero valid draws; or fewer than
        ``min_draws`` valid draws while ``enforce_min_draws``.
    """
    stats = np.asarray(null_statistics, dtype=np.float64).ravel()
    requested = int(len(stats) if n_draws_requested is None else n_draws_requested)

    if not np.isfinite(observed_statistic):
        raise ValueError(
            "observed_statistic must be finite to compare against a null "
            f"distribution; got {observed_statistic!r}"
        )

    finite = np.isfinite(stats)
    n_invalid = int(np.count_nonzero(~finite))
    valid = stats[finite]
    n_valid = int(valid.size)

    diagnostics: list[Diagnostic] = []
    if n_invalid:
        diagnostics.append(
            Diagnostic(
                code=INVALID_DRAWS_CODE,
                severity=Severity.WARNING,
                message=(
                    f"{n_invalid} of {len(stats)} null statistics were not finite and "
                    "were excluded from the p-value denominator."
                ),
                context={"n_invalid": n_invalid, "n_supplied": int(len(stats))},
            )
        )

    if n_valid == 0:
        raise ValueError(
            "no valid null statistics: an empirical p-value needs at least one draw"
        )

    if enforce_min_draws and n_valid < min_draws:
        raise ValueError(
            f"refusing to emit an empirical p-value from {n_valid} valid null draws: "
            f"docs/statistical-design.md Sec. 3.3 sets a hard floor of {min_draws} "
            "(contract amendment 1.1.0, signull.types.MIN_DRAWS). Raise n_draws, or "
            "pass enforce_min_draws=False for an explicitly non-gating diagnostic."
        )

    if alternative is Alternative.TWO_SIDED:
        r_up = _count_exceedances(observed_statistic, valid, Alternative.GREATER)
        r_dn = _count_exceedances(observed_statistic, valid, Alternative.LESS)
        n_exceed = min(r_up, r_dn)
        p_one = (1.0 + n_exceed) / (1.0 + n_valid)
        p = min(1.0, 2.0 * p_one)
    else:
        n_exceed = _count_exceedances(observed_statistic, valid, alternative)
        p = (1.0 + n_exceed) / (1.0 + n_valid)

    if n_exceed == 0:
        diagnostics.append(
            Diagnostic(
                code=AT_FLOOR_CODE,
                severity=Severity.INFO,
                message=(
                    f"no null draw reached the observed statistic; p is pinned at its "
                    f"attainable floor 1/(K+1) = {p_value_floor(n_valid):.3g} and must "
                    "be reported as an upper bound, not a point estimate."
                ),
                context={"n_valid_draws": n_valid, "floor": p_value_floor(n_valid)},
            )
        )

    return EmpiricalPValue(
        p_value=float(p),
        n_exceedances=int(n_exceed),
        n_valid_draws=n_valid,
        n_draws_requested=requested,
        observed_statistic=float(observed_statistic),
        alternative=alternative,
        diagnostics=tuple(diagnostics),
    )
