"""Construction of the eligible background ``B`` -- the universe nulls are drawn from.

``docs/statistical-design.md`` Sec. 2.1::

    B = { g in rows(X) : g passes the expression filter
                       : sd_g > 0
                       : g was measurable on the platform S was derived from (if declared)
                       : (optionally) g not in S }

Two traps the spec calls out explicitly and this module implements:

* **Platform restriction.**  A candidate selected on a 10k-feature platform must
  be tested against draws from those 10k features.  Drawing "random" sets from
  20k genes the original authors could never have picked makes the null easier
  and flatters the candidate.
* **Excluding the candidate.**  Whether ``S`` itself stays in ``B`` is a real
  choice: the statistical spec removes it, while the published random-signature
  benchmarks (and therefore ``MatchingSpec.exclude_candidate_genes=False``,
  the contract default) keep it.  Both are supported; the choice is recorded.

The floors of Sec. 8 F4 (``|B| >= 2000`` and ``|B| >= 20 m``) live here as
:func:`check_background_floors`, which *refuses* rather than warns: a competitive
null drawn from a background too small to be diverse is not a null.

This module never looks at the outcome.  A ``y``-aware expression filter is
leakage (Sec. 5, Path A trap 2), so the filter reads gene statistics only.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from signull.types import (
    AlignedDataset,
    Diagnostic,
    ExpressionMatrix,
    GeneId,
    GeneStats,
    Severity,
    Signature,
)

from .diagnostics import DiagnosticCode, DiagnosticLog

__all__ = [
    "MIN_BACKGROUND_GENES",
    "BACKGROUND_PER_CANDIDATE_GENE",
    "DEFAULT_MIN_DETECTION_RATE",
    "CONTROL_PROBE_PREFIXES",
    "BackgroundTooSmallError",
    "EligibleBackground",
    "eligible_background",
    "check_background_floors",
]

#: ``|B|`` floor from ``docs/statistical-design.md`` Sec. 8 F4.
MIN_BACKGROUND_GENES: Final[int] = 2_000
#: ``|B| >= 20 m`` -- background genes required per candidate gene (F4).
BACKGROUND_PER_CANDIDATE_GENE: Final[int] = 20
#: Sec. 2.1: detected in at least 20 % of samples.
DEFAULT_MIN_DETECTION_RATE: Final[float] = 0.20
#: Affymetrix control probesets; never biological features.
CONTROL_PROBE_PREFIXES: Final[tuple[str, ...]] = ("AFFX-", "AFFX_")


class BackgroundTooSmallError(ValueError):
    """The eligible background cannot support a competitive null (Sec. 8 F4).

    Raised, not warned: with ``|B| < 2000`` or ``|B| < 20 m`` the random gene
    sets overlap each other so heavily that the null distribution is not a
    distribution over *independent* gene sets.  The label-permutation null (N2)
    is unaffected and may still be run.
    """


@dataclass(frozen=True, eq=False, slots=True)
class EligibleBackground:
    """The universe a random gene-set null may sample from, plus why it shrank.

    Attributes
    ----------
    genes:
        Eligible identifiers, in matrix row order.  This is what is handed to
        :meth:`signull.types.NullModel.eligible_universe` consumers.
    n_input:
        Rows the matrix started with.
    excluded:
        ``reason -> identifiers`` for every gene removed, so the report can say
        what the null was drawn from without recomputing anything.
    filters:
        The thresholds actually applied, for the captured config.
    diagnostics:
        Conditions recorded while filtering.
    """

    genes: tuple[GeneId, ...]
    n_input: int
    excluded: dict[str, tuple[GeneId, ...]] = field(default_factory=dict)
    filters: dict[str, object] = field(default_factory=dict)
    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def size(self) -> int:
        """``|B|``."""
        return len(self.genes)

    def __len__(self) -> int:
        return len(self.genes)

    def n_excluded(self, reason: str) -> int:
        """How many genes a given filter removed."""
        return len(self.excluded.get(reason, ()))


def eligible_background(
    source: AlignedDataset | ExpressionMatrix,
    *,
    candidate: Signature | None = None,
    exclude_candidate_genes: bool = False,
    min_detection_rate: float = DEFAULT_MIN_DETECTION_RATE,
    min_median: float | None = None,
    platform_features: Iterable[GeneId] | None = None,
    drop_control_probes: bool = True,
    log: DiagnosticLog | None = None,
) -> EligibleBackground:
    """Build ``B`` from the **aligned** cohort's gene statistics.

    Parameters
    ----------
    source:
        An :class:`~signull.types.AlignedDataset` (the intended input) or a bare
        matrix.  Statistics are read from ``matrix.gene_stats()``, which is
        computed over exactly the samples in that object -- passing a raw,
        unaligned matrix here silently matches the null against statistics from
        a different sample set, which is why the aligned form is the documented
        one.
    candidate:
        The **resolved** candidate.  Required when ``exclude_candidate_genes``,
        and used for the ``|B| >= 20 m`` floor check by the caller.
    exclude_candidate_genes:
        Sec. 2.1 removes ``S`` from ``B``; the contract default
        (``MatchingSpec.exclude_candidate_genes=False``) keeps it, matching the
        published benchmarks.  Whichever is chosen is recorded in
        :attr:`EligibleBackground.filters`.
    min_detection_rate:
        Keep a gene detected in at least this fraction of samples.  Set to
        ``0.0`` to disable.  When the detection dimension is degenerate (every
        gene detected everywhere, the usual case for background-corrected array
        data) the filter removes nothing and says so.
    min_median:
        Alternative admission route: a gene with ``median >= min_median`` is kept
        even if it fails the detection filter (Sec. 2.1's "median CPM >= 1 OR
        detected in >= 20 % of samples").  ``None`` disables this route.
    platform_features:
        Identifiers measurable on the platform the candidate was derived from.
        When given, ``B`` is restricted to their intersection with the matrix.
    drop_control_probes:
        Remove ``AFFX-`` control probesets.

    Returns
    -------
    EligibleBackground
    """
    log = log if log is not None else DiagnosticLog()
    matrix = source.matrix if isinstance(source, AlignedDataset) else source
    stats: GeneStats = matrix.gene_stats()
    table = stats.table
    gene_ids = tuple(matrix.gene_ids)
    n_input = len(gene_ids)

    keep = np.ones(n_input, dtype=bool)
    excluded: dict[str, tuple[GeneId, ...]] = {}

    def _drop(mask: np.ndarray, reason: str) -> None:
        """Record ``mask`` (True = drop) against ``reason`` and apply it."""
        dropping = mask & keep
        if not dropping.any():
            return
        excluded[reason] = tuple(
            gene_ids[i] for i in np.flatnonzero(dropping)
        )
        keep[dropping] = False

    sd = table["sd"].to_numpy(dtype=np.float64)
    _drop(~(sd > 0.0), "zero_variance")

    if drop_control_probes:
        control = np.array(
            [gene.upper().startswith(CONTROL_PROBE_PREFIXES) for gene in gene_ids],
            dtype=bool,
        )
        _drop(control, "control_probe")
        if control.any():
            log.record(
                DiagnosticCode.CONTROL_PROBES_EXCLUDED,
                f"removed {int(control.sum())} control probeset(s) from the eligible "
                "background",
                context={"n_control": int(control.sum())},
            )

    detection = table["detection_rate"].to_numpy(dtype=np.float64)
    detection_is_degenerate = float(detection.std()) < 0.01 and float(detection.min()) >= 1.0
    if min_detection_rate > 0.0 and not detection_is_degenerate:
        admitted = detection >= min_detection_rate
        if min_median is not None:
            admitted |= table["median"].to_numpy(dtype=np.float64) >= min_median
        _drop(~admitted, "expression_filter")
    elif min_detection_rate > 0.0:
        log.record(
            DiagnosticCode.DETECTION_DIMENSION_DEGENERATE,
            "every gene is detected in every sample, so the expression filter admits "
            "the whole matrix; the detection dimension carries no information",
            context={"detection_rate_sd": float(detection.std())},
        )
    elif min_median is not None:
        _drop(table["median"].to_numpy(dtype=np.float64) < min_median, "expression_filter")

    if platform_features is not None:
        allowed = {str(g) for g in platform_features}
        if not allowed:
            raise ValueError("platform_features is empty; pass None to disable the filter")
        _drop(
            np.array([gene not in allowed for gene in gene_ids], dtype=bool),
            "off_platform",
        )

    if exclude_candidate_genes:
        if candidate is None:
            raise ValueError(
                "exclude_candidate_genes=True requires the resolved candidate signature"
            )
        in_candidate = set(candidate.genes)
        _drop(
            np.array([gene in in_candidate for gene in gene_ids], dtype=bool),
            "candidate_gene",
        )

    n_filtered = int((~keep).sum())
    if n_filtered:
        log.record(
            DiagnosticCode.BACKGROUND_GENES_FILTERED,
            f"eligible background is {int(keep.sum())}/{n_input} genes; removed "
            + ", ".join(f"{len(v)} by {k}" for k, v in excluded.items()),
            context={
                "n_input": n_input,
                "n_eligible": int(keep.sum()),
                "removed_by": {k: len(v) for k, v in excluded.items()},
            },
        )

    return EligibleBackground(
        genes=tuple(gene_ids[i] for i in np.flatnonzero(keep)),
        n_input=n_input,
        excluded=excluded,
        filters={
            "min_detection_rate": float(min_detection_rate),
            "min_median": None if min_median is None else float(min_median),
            "drop_control_probes": bool(drop_control_probes),
            "platform_restricted": platform_features is not None,
            "exclude_candidate_genes": bool(exclude_candidate_genes),
            "detection_dimension_degenerate": bool(detection_is_degenerate),
        },
        diagnostics=log.diagnostics,
    )


def check_background_floors(
    background: EligibleBackground | Sequence[GeneId],
    candidate_size: int,
    *,
    min_background: int = MIN_BACKGROUND_GENES,
    per_candidate_gene: int = BACKGROUND_PER_CANDIDATE_GENE,
    log: DiagnosticLog | None = None,
) -> None:
    """Enforce Sec. 8 F4: ``|B| >= 2000`` and ``|B| >= 20 m``.

    Raises
    ------
    BackgroundTooSmallError
        Either floor violated.  Refusing the competitive null is the specified
        behaviour; N2 (label permutation) does not depend on ``B`` and may still
        be run by the caller.
    ValueError
        Non-positive ``candidate_size``.
    """
    if candidate_size <= 0:
        raise ValueError(f"candidate_size must be positive, got {candidate_size}")
    size = len(background)
    required = max(min_background, per_candidate_gene * candidate_size)
    if size >= required:
        return
    message = (
        f"eligible background has {size} genes, below the floor of {required} "
        f"(max of {min_background} absolute and {per_candidate_gene} x {candidate_size} "
        "candidate genes; docs/statistical-design.md Sec. 8 F4). Refusing the "
        "competitive null: draws from a background this small overlap each other "
        "heavily and are not independent gene sets. The label-permutation null is "
        "unaffected."
    )
    if log is not None:
        log.record(
            DiagnosticCode.BACKGROUND_TOO_SMALL,
            message,
            severity=Severity.ERROR,
            context={
                "n_background": size,
                "required": required,
                "candidate_size": candidate_size,
            },
        )
    raise BackgroundTooSmallError(message)
