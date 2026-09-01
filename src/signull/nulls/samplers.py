"""The two null models: random matched gene sets (N0/N1/N3) and label permutation (N2).

They answer **different questions** and are not interchangeable
(``docs/statistical-design.md`` Sec. 2):

* :class:`RandomGeneSetNull` -- *competitive*.  "Is this gene set special among
  gene sets of the same size drawn from this dataset?"  Labels are fixed; the
  gene set varies.  Matched on size unconditionally, and on mean expression,
  variance and detection rate by default: an unmatched null is misspecified in
  the direction that flatters the candidate, because signature genes are
  typically higher-expressed and more variable than a uniform draw.
* :class:`LabelPermutationNull` -- *self-contained*.  "Is there any outcome
  signal at all?"  The gene set is fixed; the labels are permuted.  This is the
  null the calibration acceptance test (T1) drives.

Both yield :class:`~signull.types.NullDraw` objects carrying **both** a
signature and an outcome, so a single downstream evaluation path serves the
candidate and every null.  Neither model knows how signatures are scored --
that is what makes "the same scoring method is applied to candidate and nulls"
structurally true rather than a convention (``docs/architecture.md`` Sec. 2).

This module contains no scoring, no metrics and no evaluation loop.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from signull.types import (
    MIN_DRAWS,
    AlignedDataset,
    BinaryOutcome,
    Diagnostic,
    GeneId,
    NullDraw,
    NullSpec,
    NullType,
    Provenance,
    Severity,
    Signature,
    SignatureOrigin,
)

from .binning import BackgroundExhaustedError, MatchedBackground, build_matched_background
from .seeding import draw_seeds, generator_from_seed

__all__ = [
    "DrawFloorError",
    "RandomGeneSetNull",
    "LabelPermutationNull",
    "REGISTRY",
    "get",
    "null_signature_name",
    "SET_LEVEL_CONSTRAINTS",
    "WIDENED_BIN_CODE",
    "CONSTRAINT_RETRIES_CODE",
]

#: Diagnostic code: this draw needed neighbouring bins to fill some cell.
WIDENED_BIN_CODE: Final[str] = "draw_used_neighbour_bins"
#: Diagnostic code: rejection sampling for a set-level constraint needed retries.
CONSTRAINT_RETRIES_CODE: Final[str] = "set_level_constraint_retries"
#: Set-level constraints understood by :class:`RandomGeneSetNull`.
SET_LEVEL_CONSTRAINTS: Final[tuple[str, ...]] = ("mean_abs_correlation",)


class DrawFloorError(ValueError):
    """``n_draws`` is below :data:`~signull.types.MIN_DRAWS` for a gating null.

    Contract amendment 1.1.0 and ``docs/statistical-design.md`` Sec. 3.3: the
    relative standard error of an empirical p-value at ``p = 0.05`` is 9.7 % at
    ``K = 2000`` and 13.8 % at ``K = 999``.  A p-value the design refuses to
    interpret must not be emitted, so the refusal lives at the point of
    generation.  Pass ``enforce_draw_floor=False`` only for a deliberately
    reduced-resolution run (the supervised path) that says so in its report.
    """


def null_signature_name(candidate: Signature, index: int) -> str:
    """Stable, sortable name for the ``index``-th null draw of ``candidate``."""
    return f"{candidate.name}::null_{index:06d}"


@dataclass(frozen=True, eq=False, slots=True)
class RandomGeneSetNull:
    """Property-matched random gene-set null.  Implements ``NullModel``.

    The background is binned **once** per run by
    :func:`~signull.nulls.binning.build_matched_background` and reused for every
    draw, so a ``K = 10000`` run pays the binning cost once.  Each draw takes one
    gene per candidate gene from that gene's own bin, expanding to L1-neighbour
    bins only when a bin runs out -- and recording a diagnostic when it does.
    There is no unmatched fallback: exhausting the lattice raises
    :class:`~signull.nulls.binning.BackgroundExhaustedError`, because silently
    topping a draw up from anywhere in the universe turns a matched null back
    into an unmatched one without telling anyone.

    Attributes
    ----------
    spec:
        Complete configuration.  ``spec.matching.properties`` empty means the
        uniform N0 baseline; the default
        :func:`~signull.nulls.binning.default_matching_spec` gives N1.
    universe:
        The eligible background, from
        :func:`signull.data.eligible_background`.  ``None`` derives a fallback
        universe from the matrix index (rows with ``sd > 0``) -- correct but
        unfiltered, so callers that care about the expression filter or platform
        restriction must pass one.
    enforce_draw_floor:
        Refuse ``n_draws < MIN_DRAWS``.  Default ``True``.
    weights_from_candidate:
        Reuse the candidate's weight multiset on every draw, element-wise in
        candidate order, so a signed candidate's nulls share its signed
        structure.  Default ``True``; irrelevant for unsigned candidates.
    """

    spec: NullSpec = field(
        default_factory=lambda: NullSpec(null_type=NullType.RANDOM_GENE_SET)
    )
    universe: tuple[GeneId, ...] | None = None
    enforce_draw_floor: bool = True
    weights_from_candidate: bool = True

    def __post_init__(self) -> None:
        if self.spec.null_type is not NullType.RANDOM_GENE_SET:
            raise ValueError(
                f"RandomGeneSetNull requires NullSpec.null_type=RANDOM_GENE_SET, "
                f"got {self.spec.null_type!r}"
            )
        unknown = set(self.spec.matching.set_level_constraints) - set(SET_LEVEL_CONSTRAINTS)
        if unknown:
            raise ValueError(
                f"unsupported set-level constraint(s) {sorted(unknown)!r}; "
                f"understood: {list(SET_LEVEL_CONSTRAINTS)!r}"
            )

    @property
    def null_type(self) -> NullType:
        """:attr:`~signull.types.NullType.RANDOM_GENE_SET`."""
        return NullType.RANDOM_GENE_SET

    def eligible_universe(self, dataset: AlignedDataset) -> tuple[GeneId, ...]:
        """Genes a draw may sample from, in matrix row order.

        Returns the universe supplied at construction, intersected with the
        matrix index so a stale universe cannot smuggle in genes the matrix does
        not have.  Without one, falls back to every matrix row with ``sd > 0``
        (a constant gene has no z-score and cannot be background).

        The universe is dataset-specific by construction: the share of random
        signatures reaching significance ranges from ~1 % to ~40 % across
        datasets, so one borrowed from elsewhere makes the p-value meaningless.
        """
        gene_ids = tuple(dataset.matrix.gene_ids)
        if self.universe is None:
            sd = dataset.matrix.gene_stats().table["sd"].to_numpy(dtype=np.float64)
            return tuple(g for g, s in zip(gene_ids, sd, strict=True) if s > 0.0)
        allowed = set(self.universe)
        selected = tuple(g for g in gene_ids if g in allowed)
        if not selected:
            raise ValueError(
                "the supplied universe shares no identifier with the matrix index; "
                "the background must be built from this dataset "
                "(docs/statistical-design.md Sec. 8 F7)"
            )
        return selected

    def background(
        self, candidate: Signature, dataset: AlignedDataset
    ) -> MatchedBackground:
        """Bin the universe against ``candidate``.  Cheap to call, not cached.

        Exposed because the report needs the bin layout and the build-time
        diagnostics, and because it is the natural unit test seam.
        """
        matching = self.spec.matching
        universe = self.eligible_universe(dataset)
        if matching.exclude_candidate_genes:
            excluded = set(candidate.genes)
            universe = tuple(g for g in universe if g not in excluded)
        return build_matched_background(
            gene_stats=dataset.matrix.gene_stats(),
            universe=universe,
            candidate_genes=candidate.genes,
            matching=matching,
        )

    def draw(
        self,
        candidate: Signature,
        dataset: AlignedDataset,
        n_draws: int,
        rng: np.random.Generator,
    ) -> Iterator[NullDraw]:
        """Yield exactly ``n_draws`` size- and property-matched gene sets.

        Draw ``i`` is generated from its own seed, taken from a single
        vectorised call on ``rng``, so it is reproducible and independent of how
        many draws were consumed before it.  The observed outcome is attached
        unchanged to every draw.
        """
        self._check_draw_floor(n_draws)
        background = self.background(candidate, dataset)
        build_diagnostics = background.diagnostics
        seeds = draw_seeds(rng, n_draws)
        constraint = _CorrelationConstraint.build(
            self.spec.matching.set_level_constraints, candidate, dataset
        )
        weights = candidate.weights if self.weights_from_candidate else None

        for index in range(n_draws):
            draw_rng = generator_from_seed(int(seeds[index]))
            genes, diagnostics = _sample_once(
                background=background,
                constraint=constraint,
                dataset=dataset,
                rng=draw_rng,
                max_attempts=self.spec.matching.max_resample_attempts,
                index=index,
            )
            yield NullDraw(
                index=index,
                signature=Signature(
                    genes=genes,
                    name=null_signature_name(candidate, index),
                    namespace=candidate.namespace,
                    origin=SignatureOrigin.RANDOM_NULL,
                    weights=weights,
                    provenance=Provenance(
                        source="signull.nulls.RandomGeneSetNull",
                        identifier=str(int(seeds[index])),
                        notes=(
                            f"draw {index}, size {len(genes)}, "
                            f"matched on {[p.value for p in background.properties]}"
                        ),
                    ),
                ),
                outcome=dataset.outcome,
                null_type=NullType.RANDOM_GENE_SET,
                diagnostics=(build_diagnostics if index == 0 else ()) + diagnostics,
            )

    def _check_draw_floor(self, n_draws: int) -> None:
        if n_draws < 1:
            raise ValueError(f"n_draws must be >= 1, got {n_draws}")
        if self.enforce_draw_floor and n_draws < MIN_DRAWS:
            raise DrawFloorError(
                f"n_draws={n_draws} is below the hard floor MIN_DRAWS={MIN_DRAWS} "
                "(docs/statistical-design.md Sec. 3.3). At K=999 the relative "
                "standard error of p at p=0.05 is 13.8 %, which the design refuses "
                "to interpret. Raise n_draws, or set enforce_draw_floor=False for a "
                "deliberately reduced-resolution run that says so in its report."
            )


@dataclass(frozen=True, eq=False, slots=True)
class LabelPermutationNull:
    """Label-permutation null (N2).  Implements ``NullModel``.

    ``X``, ``S`` and the scorer are untouched; only ``y`` is permuted, uniformly
    at random.  Class sizes are preserved by construction (a permutation of the
    label vector), so no draw can be degenerate if the observed outcome is not.

    The design (Sec. 2.3) allows exhaustive enumeration when
    ``C(N, n1) <= 20000``.  With the cohort floors of Sec. 8 F3 in force
    (``N >= 30``, ``n1, n0 >= 8``) the smallest reachable count is
    ``C(30, 8) = 5852925``, so that branch is unreachable in any run this tool
    will accept and is deliberately not implemented: Monte Carlo with the add-one
    estimator is exact here in the sense that matters, and dead enumeration code
    would be untested code.

    Attributes
    ----------
    spec:
        Configuration.  ``spec.matching`` is ignored -- nothing is drawn from the
        gene universe.
    enforce_draw_floor:
        Refuse ``n_draws < MIN_DRAWS``.  Default ``True``.
    """

    spec: NullSpec = field(
        default_factory=lambda: NullSpec(null_type=NullType.LABEL_PERMUTATION)
    )
    enforce_draw_floor: bool = True

    def __post_init__(self) -> None:
        if self.spec.null_type is not NullType.LABEL_PERMUTATION:
            raise ValueError(
                f"LabelPermutationNull requires NullSpec.null_type=LABEL_PERMUTATION, "
                f"got {self.spec.null_type!r}"
            )

    @property
    def null_type(self) -> NullType:
        """:attr:`~signull.types.NullType.LABEL_PERMUTATION`."""
        return NullType.LABEL_PERMUTATION

    def eligible_universe(self, dataset: AlignedDataset) -> tuple[GeneId, ...]:
        """The candidate's own genes: this null samples labels, never genes.

        The gene set is fixed under N2, so there is no sampling universe.  The
        contract still defines the method, and returning the matrix index would
        wrongly suggest genes are drawn -- so this returns an empty tuple and the
        docstring says why.
        """
        return ()

    def draw(
        self,
        candidate: Signature,
        dataset: AlignedDataset,
        n_draws: int,
        rng: np.random.Generator,
    ) -> Iterator[NullDraw]:
        """Yield exactly ``n_draws`` permuted-label realisations of ``candidate``.

        Each draw carries the candidate signature unchanged and a
        :class:`~signull.types.BinaryOutcome` with ``is_permuted=True`` and the
        seed that produced it, so any single draw can be rebuilt on its own.
        """
        if n_draws < 1:
            raise ValueError(f"n_draws must be >= 1, got {n_draws}")
        if self.enforce_draw_floor and n_draws < MIN_DRAWS:
            raise DrawFloorError(
                f"n_draws={n_draws} is below MIN_DRAWS={MIN_DRAWS} "
                "(docs/statistical-design.md Sec. 3.3); set enforce_draw_floor=False "
                "for a deliberately reduced-resolution run"
            )
        outcome = dataset.outcome
        labels = np.asarray(outcome.labels, dtype=bool)
        seeds = draw_seeds(rng, n_draws)

        for index in range(n_draws):
            seed = int(seeds[index])
            permuted = generator_from_seed(seed).permutation(labels)
            permuted.setflags(write=False)
            yield NullDraw(
                index=index,
                signature=candidate,
                outcome=BinaryOutcome(
                    sample_ids=outcome.sample_ids,
                    labels=permuted,
                    name=f"{outcome.name}::permuted_{index:06d}",
                    positive_label=outcome.positive_label,
                    negative_label=outcome.negative_label,
                    is_permuted=True,
                    permutation_seed=seed,
                    provenance=outcome.provenance,
                ),
                null_type=NullType.LABEL_PERMUTATION,
            )


# ---------------------------------------------------------------------------
# Draw-time helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _CorrelationConstraint:
    """Rejection-sampling constraint on a draw's mean absolute inter-gene correlation.

    A set-level property cannot be expressed per gene, so it cannot be binned:
    it is enforced by drawing again.  Kept private because the vocabulary of
    supported constraints is :data:`SET_LEVEL_CONSTRAINTS`, not this class.
    """

    target: float
    tolerance: float
    values: np.ndarray  # genes x samples, row-standardised

    @classmethod
    def build(
        cls,
        constraints: Mapping[str, float],
        candidate: Signature,
        dataset: AlignedDataset,
    ) -> "_CorrelationConstraint | None":
        """``None`` when no correlation constraint was requested."""
        tolerance = constraints.get("mean_abs_correlation")
        if tolerance is None:
            return None
        if tolerance <= 0.0:
            raise ValueError(
                f"mean_abs_correlation tolerance must be positive, got {tolerance}"
            )
        values = np.asarray(dataset.matrix.values, dtype=np.float64)
        gene_index = {g: i for i, g in enumerate(dataset.matrix.gene_ids)}
        rows = np.fromiter(
            (gene_index[g] for g in candidate.genes),
            dtype=np.intp,
            count=len(candidate.genes),
        )
        target = cls._mean_abs_correlation(values[rows, :])
        return cls(target=target, tolerance=float(tolerance), values=values)

    @staticmethod
    def _mean_abs_correlation(block: np.ndarray) -> float:
        """Mean ``|r|`` over the distinct gene pairs of a block; 0 for one gene."""
        if block.shape[0] < 2:
            return 0.0
        corr = np.corrcoef(block)
        upper = np.triu_indices(corr.shape[0], k=1)
        return float(np.nanmean(np.abs(corr[upper])))

    def accepts(self, rows: np.ndarray) -> bool:
        """``True`` when the drawn rows match the candidate's coherence."""
        drawn = self._mean_abs_correlation(self.values[rows, :])
        return abs(drawn - self.target) <= self.tolerance


def _sample_once(
    *,
    background: MatchedBackground,
    constraint: "_CorrelationConstraint | None",
    dataset: AlignedDataset,
    rng: np.random.Generator,
    max_attempts: int,
    index: int,
) -> tuple[tuple[GeneId, ...], tuple[Diagnostic, ...]]:
    """One accepted draw plus its diagnostics, in candidate gene order."""
    diagnostics: list[Diagnostic] = []
    gene_index = {g: i for i, g in enumerate(dataset.matrix.gene_ids)}

    attempts = 0
    while True:
        attempts += 1
        picks, radius = background.sample(rng)
        genes = tuple(background.universe[i] for i in picks)
        if radius > 0:
            diagnostics.append(
                Diagnostic(
                    code=WIDENED_BIN_CODE,
                    severity=Severity.INFO,
                    message=(
                        f"draw {index} exhausted at least one matching bin and took "
                        f"replacements from bins up to L1 radius {radius}"
                    ),
                    context={"draw": index, "max_radius": radius},
                )
            )
        if constraint is None:
            return genes, tuple(diagnostics)
        rows = np.fromiter(
            (gene_index[g] for g in genes), dtype=np.intp, count=len(genes)
        )
        if constraint.accepts(rows):
            if attempts > 1:
                diagnostics.append(
                    Diagnostic(
                        code=CONSTRAINT_RETRIES_CODE,
                        severity=Severity.INFO,
                        message=(
                            f"draw {index} needed {attempts} attempts to satisfy "
                            "mean_abs_correlation"
                        ),
                        context={"draw": index, "attempts": attempts},
                    )
                )
            return genes, tuple(diagnostics)
        if attempts >= max_attempts:
            raise BackgroundExhaustedError(
                f"draw {index}: {attempts} attempts failed to satisfy the set-level "
                f"constraint mean_abs_correlation = {constraint.target:.4f} "
                f"+- {constraint.tolerance:.4f}. Refusing to emit an unconstrained "
                "draw in its place; widen the tolerance or drop the constraint."
            )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: ``NullType -> factory``.  The CLI resolves strings to models only through
#: this map, so adding a null model is one entry rather than a new branch.
REGISTRY: Final[Mapping[NullType, type]] = {
    NullType.RANDOM_GENE_SET: RandomGeneSetNull,
    NullType.LABEL_PERMUTATION: LabelPermutationNull,
}


def get(null_type: NullType | str, **params: object):
    """Instantiate the null model registered for ``null_type``.

    Parameters
    ----------
    **params:
        Forwarded to the model's constructor (``spec``, ``universe``,
        ``enforce_draw_floor``, ...).  A ``spec`` whose ``null_type`` disagrees
        with ``null_type`` raises, so the registry key and the captured config
        can never drift apart.
    """
    key = NullType(null_type)
    factory = REGISTRY[key]
    if "spec" not in params:
        params["spec"] = NullSpec(null_type=key)
    return factory(**params)  # type: ignore[operator]
