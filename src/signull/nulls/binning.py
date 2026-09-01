"""Nested conditional quantile binning -- the property-matched null's machinery.

This is the module that makes ``signull`` different from what already exists.
Bioconductor ``SigCheck`` and ``singscore`` build their random-signature nulls by
sampling *uniformly at random from all available features*.  That null is
misspecified in a direction that flatters the candidate: real signature genes
are systematically higher-expressed, more variable and more reliably detected
than a uniform draw, and each of those marginal properties buys discrimination
on its own.  ``scanpy.tl.score_genes`` and ``Seurat::AddModuleScore`` do match
control gene sets on expression bins, and ``nullranges`` does covariate matching
for genomic intervals -- but neither connects property matching to *signature
significance testing*.  That connection is what lives here.

What this module implements, per ``docs/statistical-design.md`` Sec. 2.2:

1. **Nested (conditional) quantile bins, not a product of marginals.**  Mean
   expression and variance are strongly dependent (the mean-variance trend), so
   a product grid leaves most cells empty and the matcher silently degrades into
   an unmatched one.  Bins are therefore formed conditionally: mean expression
   over the whole background, then variance *within each mean stratum*, then
   detection rate *within each (mean, variance) cell*.
2. **An adequacy rule**: ``pool[c] >= max(50, 10 * need[c])`` for every cell the
   candidate occupies.  A cell that cannot supply ten alternatives per candidate
   gene is not a matched cell, it is a lookup table.
3. **A deterministic coarsening ladder** when adequacy fails, ending at a hard
   floor.
4. **L1 neighbour expansion** in bin-index space when a cell is exhausted
   *during* a draw, and a loud failure -- never a silent unmatched fallback --
   when expansion runs out of lattice.

Deliberately **not** matched: association with the outcome.  Matching on that
would remove exactly the signal the test is trying to detect and is forbidden by
Sec. 2.2.  Nothing in this module reads ``y``.

Monotone transforms are irrelevant here: binning is rank-based, so binning on
``variance`` and on the spec's ``log(variance + eps)`` produce identical cells.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Final

import numpy as np
import pandas as pd

from ..types import (
    DEFAULT_BINS_BY_PROPERTY,
    Diagnostic,
    GeneId,
    GeneStats,
    IntArray,
    MatchingProperty,
    MatchingSpec,
    Severity,
    default_mean_expression_bins,
)

__all__ = [
    "MIN_CELL_POOL",
    "POOL_PER_CANDIDATE_GENE",
    "COARSENING_FLOOR_SECOND",
    "quantile_bins",
    "assign_cells",
    "resolve_bin_counts",
    "coarsening_ladder",
    "default_matching_spec",
    "MatchedBackground",
    "build_matched_background",
    "BackgroundExhaustedError",
]

#: Absolute floor on the number of background genes in an occupied cell.
MIN_CELL_POOL: Final[int] = 50
#: Alternatives required per candidate gene in a cell.
POOL_PER_CANDIDATE_GENE: Final[int] = 10
#: Ladder steps for the second matching property (``docs/...`` Sec. 2.2: 5 -> 3 -> 2).
COARSENING_FLOOR_SECOND: Final[tuple[int, ...]] = (3, 2)
#: Lower bound on the first property's bin count while halving.
COARSENING_FLOOR_FIRST: Final[int] = 5

#: Diagnostic codes owned by this module.
BIN_COARSENED_CODE: Final[str] = "sparse_matching_bin_widened"
MARGINAL_GRID_CODE: Final[str] = "marginal_matching_grid_requested"
BIN_LAYOUT_CODE: Final[str] = "matching_bin_layout"
DEGENERATE_PROPERTY_CODE: Final[str] = "degenerate_matching_property"

#: ``sd(detection_rate) < 0.01`` collapses that dimension (Sec. 2.2).
DEGENERATE_SD: Final[float] = 0.01


class BackgroundExhaustedError(RuntimeError):
    """The matching constraint could not be satisfied.

    Raised at build time when the adequacy rule still fails at the coarsening
    floor, and at draw time when L1 neighbour expansion exhausts the lattice.
    It is a :class:`RuntimeError` because ``docs/architecture.md`` Sec. 5 files
    "matching constraint unsatisfiable after widening" under tier 1 -- fail
    fast, never degrade to unmatched sampling behind the caller's back.
    """


# ---------------------------------------------------------------------------
# Binning primitives
# ---------------------------------------------------------------------------


def quantile_bins(values: np.ndarray, n_bins: int) -> IntArray:
    """Rank-based equal-count quantile bins, with ties kept together.

    Genes sharing a value always share a bin: the whole tie group is placed in
    the bin of its mean rank.  Splitting identical genes across bins would shrink
    every pool for no statistical gain, and would make the assignment depend on
    row order.  Bin counts are therefore only *approximately* equal when ties are
    present, and some bins may be empty -- both are handled downstream by the
    adequacy check and the neighbour walk.

    Parameters
    ----------
    values:
        1-D array of a per-gene property.  Must be finite.
    n_bins:
        Requested bins; ``<= 1`` returns a single bin.

    Returns
    -------
    IntArray
        Bin index per element, in ``[0, n_bins)``.
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    n = x.size
    if n_bins <= 1 or n == 0:
        return np.zeros(n, dtype=np.int64)
    uniq, inverse = np.unique(x, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    if uniq.size == 1:
        return np.zeros(n, dtype=np.int64)
    group_sizes = np.bincount(inverse, minlength=uniq.size)
    cum = np.cumsum(group_sizes)
    # Mean 0-based rank of each tie group in ascending order.
    mean_rank = (cum - group_sizes + cum - 1) / 2.0
    group_bin = np.floor(mean_rank * n_bins / n).astype(np.int64)
    np.clip(group_bin, 0, n_bins - 1, out=group_bin)
    return group_bin[inverse]


def assign_cells(
    property_values: Sequence[np.ndarray],
    bin_counts: Sequence[int],
    *,
    nested: bool = True,
) -> IntArray:
    """Assign every gene a multi-index cell over the matching properties.

    Parameters
    ----------
    property_values:
        One finite 1-D array per matching property, all the same length, in the
        property priority order of :attr:`~signull.types.MatchingSpec.properties`.
    bin_counts:
        Bins per property, same order and length.
    nested:
        ``True`` (default, and the only construction the statistical spec
        accepts) bins each property *within the strata of the preceding ones*.
        ``False`` bins each property marginally and takes the product grid --
        a diagnostic escape hatch only: mean and variance are strongly
        dependent, so the marginal cells go sparse and the matcher degrades.

    Returns
    -------
    IntArray
        Shape ``(n_genes, n_properties)`` of bin indices.
    """
    counts = tuple(int(c) for c in bin_counts)
    if len(counts) != len(property_values):
        raise ValueError(
            f"got {len(property_values)} property arrays but {len(counts)} bin counts"
        )
    if not counts:
        return np.zeros((0 if not property_values else len(property_values[0]), 0), dtype=np.int64)

    n = len(property_values[0])
    for arr in property_values:
        if len(arr) != n:
            raise ValueError("all property arrays must have the same length")
    cells = np.zeros((n, len(counts)), dtype=np.int64)

    cells[:, 0] = quantile_bins(property_values[0], counts[0])
    for j in range(1, len(counts)):
        if not nested:
            cells[:, j] = quantile_bins(property_values[j], counts[j])
            continue
        prefix = np.ravel_multi_index(
            tuple(cells[:, i] for i in range(j)), tuple(counts[:j])
        )
        for key in np.unique(prefix):
            member = prefix == key
            cells[member, j] = quantile_bins(property_values[j][member], counts[j])
    return cells


def resolve_bin_counts(
    properties: Sequence[MatchingProperty],
    matching: MatchingSpec,
    n_background: int,
) -> tuple[int, ...]:
    """Bin count per property: explicit spec first, then the design default.

    Precedence:

    1. :attr:`~signull.types.MatchingSpec.bins_by_property` when the property is
       listed there (contract amendment 1.1.0);
    2. the ``docs/statistical-design.md`` Sec. 2.2 default for that property --
       ``clip(|B|/500, 10, 40)`` for ``MEAN_EXPRESSION``, 5 for ``VARIANCE``,
       3 for ``DETECTION_RATE``;
    3. :attr:`~signull.types.MatchingSpec.n_bins` for any property with no design
       default.

    Note the deviation from the contract docstring, which nominates ``n_bins``
    as the sole fallback: a flat ``n_bins`` (default 10) applied to all three
    levels is exactly the marginal-ish scheme Sec. 2.2 rejects, so the design
    default wins over the flat one.  Callers that want full control set
    ``bins_by_property`` explicitly -- :func:`default_matching_spec` does.
    """
    resolved: list[int] = []
    for prop in properties:
        if prop in matching.bins_by_property:
            resolved.append(int(matching.bins_by_property[prop]))
        elif prop is MatchingProperty.MEAN_EXPRESSION:
            resolved.append(default_mean_expression_bins(n_background))
        elif prop in DEFAULT_BINS_BY_PROPERTY:
            resolved.append(int(DEFAULT_BINS_BY_PROPERTY[prop]))
        else:  # pragma: no cover - every current member has a design default
            resolved.append(int(matching.n_bins))
    return tuple(max(1, c) for c in resolved)


def coarsening_ladder(initial: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    """Deterministic coarsening sequence, per ``docs/statistical-design.md`` Sec. 2.2.

    "Coarsen in this order -- ``K_d -> 1``, then ``K_v: 5 -> 3 -> 2``, then
    ``K_a`` halved (min 5)", floor ``(K_a, K_v, K_d) = (5, 2, 1)``.

    Generalised by *position*, so it also covers property tuples other than the
    canonical ``(mean, variance, detection_rate)``: trailing properties (index
    >= 2) collapse to a single bin, last one first; then the second property
    steps down through :data:`COARSENING_FLOOR_SECOND`; then the first halves
    down to :data:`COARSENING_FLOOR_FIRST`.

    Returns
    -------
    tuple
        The initial configuration first, then each successive coarsening.  The
        last element is the floor: if adequacy fails there, the run must refuse.
    """
    current = [int(c) for c in initial]
    ladder: list[tuple[int, ...]] = [tuple(current)]
    if not current:
        return tuple(ladder)

    for i in range(len(current) - 1, 1, -1):
        if current[i] > 1:
            current[i] = 1
            ladder.append(tuple(current))
    if len(current) >= 2:
        for target in COARSENING_FLOOR_SECOND:
            if current[1] > target:
                current[1] = target
                ladder.append(tuple(current))
    while current[0] > COARSENING_FLOOR_FIRST:
        current[0] = max(COARSENING_FLOOR_FIRST, current[0] // 2)
        ladder.append(tuple(current))
    return tuple(ladder)


def default_matching_spec(
    n_background: int,
    *,
    exclude_candidate_genes: bool = False,
) -> MatchingSpec:
    """The N1 default: nested mean -> variance -> detection-rate matching.

    Bin counts are written into ``bins_by_property`` explicitly so the resulting
    spec is self-describing and reproduces bit-identically without consulting
    :func:`resolve_bin_counts` fallbacks.
    """
    properties = (
        MatchingProperty.MEAN_EXPRESSION,
        MatchingProperty.VARIANCE,
        MatchingProperty.DETECTION_RATE,
    )
    return MatchingSpec(
        properties=properties,
        nested=True,
        bins_by_property={
            MatchingProperty.MEAN_EXPRESSION: default_mean_expression_bins(n_background),
            MatchingProperty.VARIANCE: DEFAULT_BINS_BY_PROPERTY[MatchingProperty.VARIANCE],
            MatchingProperty.DETECTION_RATE: DEFAULT_BINS_BY_PROPERTY[
                MatchingProperty.DETECTION_RATE
            ],
        },
        exclude_candidate_genes=exclude_candidate_genes,
    )


# ---------------------------------------------------------------------------
# L1 lattice neighbourhoods
# ---------------------------------------------------------------------------


@lru_cache(maxsize=512)
def l1_offsets(n_dim: int, radius: int) -> tuple[tuple[int, ...], ...]:
    """All integer offsets with ``sum |d| == radius``, in tie-break order.

    Order is by ``(|d_0|, |d_1|, ..., |d_k|)`` then lexicographic -- Sec. 2.2's
    "ties are resolved by smaller ``|delta a_bin|`` first, then ``|delta
    v_bin|``, then ``|delta d_bin|``".  Fixing the order makes the union of
    neighbour pools a deterministic array, so the draw is reproducible.
    """
    if n_dim == 0:
        return ((),) if radius == 0 else ()
    if radius == 0:
        return (tuple([0] * n_dim),)
    out: list[tuple[int, ...]] = []
    for magnitudes in itertools.product(range(radius + 1), repeat=n_dim):
        if sum(magnitudes) != radius:
            continue
        signs = itertools.product(*[((1, -1) if m else (0,)) for m in magnitudes])
        for sign in signs:
            out.append(tuple(s * m for s, m in zip(sign, magnitudes)))
    out.sort(key=lambda d: (tuple(abs(v) for v in d), d))
    return tuple(out)


# ---------------------------------------------------------------------------
# The background object
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False, slots=True)
class MatchedBackground:
    """A binned sampling universe, ready to produce matched gene sets.

    Built once per run by :func:`build_matched_background` and then reused for
    every draw, so the binning cost is paid once for ``K = 10000`` draws.

    Attributes
    ----------
    universe:
        Gene identifiers a draw may sample, in a fixed sorted order.
    properties:
        Matching properties, in priority order.  Empty means an *unmatched*
        (uniform) null -- N0 -- which is a diagnostic baseline, not a gating
        null.
    bin_counts:
        Bins per property after coarsening.
    nested:
        Whether the bins are conditional (``True``) or a marginal product grid.
    candidate_cells:
        Linear cell id per candidate gene, aligned to ``candidate_genes``.
    candidate_genes:
        The resolved candidate, in the order the caller supplied.  Null draws
        are returned in this same order so that a signed candidate's weight
        multiset can be reused element-wise.
    cell_of_universe:
        Linear cell id per universe gene.
    pool_members / pool_offsets:
        CSR-style pools: universe indices of cell ``c`` are
        ``pool_members[pool_offsets[c]:pool_offsets[c + 1]]``.
    need_by_cell:
        Candidate genes per occupied cell, as a ``{cell: count}`` mapping.
    diagnostics:
        Build-time conditions: coarsening applied, degenerate properties,
        the final layout.
    """

    universe: tuple[GeneId, ...]
    properties: tuple[MatchingProperty, ...]
    bin_counts: tuple[int, ...]
    nested: bool
    candidate_genes: tuple[GeneId, ...]
    candidate_cells: IntArray
    cell_of_universe: IntArray
    pool_members: IntArray
    pool_offsets: IntArray
    need_by_cell: Mapping[int, int]
    min_cell_pool: int
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    # -- derived ---------------------------------------------------------
    @property
    def n_universe(self) -> int:
        """Number of genes available to sample."""
        return len(self.universe)

    @property
    def n_cells(self) -> int:
        """Size of the cell lattice (product of ``bin_counts``)."""
        return int(len(self.pool_offsets) - 1)

    @property
    def is_matched(self) -> bool:
        """``False`` for the uniform (N0) background."""
        return bool(self.properties)

    @property
    def size(self) -> int:
        """Observed candidate size -- the size every null draw will have."""
        return len(self.candidate_genes)

    @property
    def max_radius(self) -> int:
        """Largest L1 radius the neighbour walk may reach before failing."""
        return int(sum(self.bin_counts)) if self.bin_counts else 0

    def cell_multi_index(self, cell: int) -> tuple[int, ...]:
        """Decode a linear cell id back to its per-property bin indices."""
        if not self.bin_counts:
            return ()
        return tuple(int(v) for v in np.unravel_index(int(cell), self.bin_counts))

    def pool(self, cell: int) -> IntArray:
        """Universe indices belonging to ``cell``."""
        start = int(self.pool_offsets[cell])
        stop = int(self.pool_offsets[cell + 1])
        return self.pool_members[start:stop]

    def pool_size(self, cell: int) -> int:
        """Number of background genes in ``cell``."""
        return int(self.pool_offsets[cell + 1] - self.pool_offsets[cell])

    # -- sampling --------------------------------------------------------
    def sample(self, rng: np.random.Generator) -> tuple[IntArray, int]:
        """Draw one matched gene set.

        Returns
        -------
        (IntArray, int)
            Universe indices, one per candidate gene *in candidate order*, and
            the largest L1 radius the walk had to reach (0 when every gene was
            matched inside its own cell).

        Raises
        ------
        BackgroundExhaustedError
            When neighbour expansion passes :attr:`max_radius` without finding
            enough unused genes.  The alternative -- quietly topping the draw up
            from anywhere in the universe -- would turn a matched null back into
            an unmatched one without saying so, which is the exact failure this
            package exists to prevent.
        """
        m = self.size
        result = np.empty(m, dtype=np.int64)
        if m == 0:
            return result, 0

        used = np.zeros(self.n_universe, dtype=bool)
        # Pools are disjoint, so only cells another cell has expanded into need
        # a used-mask filter.  This keeps the common path O(m) rather than
        # O(|universe|) per draw.
        touched: set[int] = set()
        max_radius_used = 0

        occupied = np.fromiter(self.need_by_cell.keys(), dtype=np.int64, count=len(self.need_by_cell))
        occupied.sort()
        # Seeded processing order: which cell gets first pick of a scarce
        # neighbourhood must not be a fixed function of the bin index.
        order = rng.permutation(occupied.size)

        for pos in order:
            cell = int(occupied[pos])
            positions = np.flatnonzero(self.candidate_cells == cell)
            n_needed = int(positions.size)
            chosen: list[np.ndarray] = []
            n_have = 0
            radius = 0
            while True:
                if radius == 0:
                    candidates = self.pool(cell)
                    if cell in touched:
                        candidates = candidates[~used[candidates]]
                else:
                    parts = []
                    base = self.cell_multi_index(cell)
                    for offset in l1_offsets(len(self.bin_counts), radius):
                        neighbour = tuple(b + d for b, d in zip(base, offset))
                        if any(v < 0 or v >= k for v, k in zip(neighbour, self.bin_counts)):
                            continue
                        n_id = int(np.ravel_multi_index(neighbour, self.bin_counts))
                        block = self.pool(n_id)
                        if block.size:
                            parts.append(block)
                            touched.add(n_id)
                    candidates = (
                        np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
                    )
                    if candidates.size:
                        candidates = candidates[~used[candidates]]

                if candidates.size:
                    take = min(n_needed - n_have, int(candidates.size))
                    picked = _choose_without_replacement(rng, candidates, take)
                    used[picked] = True
                    chosen.append(picked)
                    n_have += take
                    if n_have == n_needed:
                        break
                radius += 1
                max_radius_used = max(max_radius_used, radius)
                if radius > self.max_radius:
                    raise BackgroundExhaustedError(
                        "background exhausted while drawing a property-matched null: "
                        f"cell {self.cell_multi_index(cell)} needed {n_needed} genes, "
                        f"found {n_have} within L1 radius {self.max_radius} of it "
                        f"(bin counts {self.bin_counts}, |universe| = {self.n_universe}). "
                        "Refusing to top the draw up from unmatched genes."
                    )
            result[positions] = np.concatenate(chosen)[:n_needed]
        return result, max_radius_used


def _choose_without_replacement(
    rng: np.random.Generator, population: IntArray, size: int
) -> IntArray:
    """Uniform sample of ``size`` distinct entries from ``population``.

    Rejection sampling on the sparse path (the adequacy rule guarantees
    ``pool >= 10 * need``, so rejections are rare) and a permutation otherwise.
    Both are exact and both are deterministic given ``rng``.
    """
    n_pop = int(population.size)
    if size >= n_pop:
        return rng.permutation(population)
    if size * 4 <= n_pop:
        seen: set[int] = set()
        out = np.empty(size, dtype=np.int64)
        filled = 0
        while filled < size:
            j = int(rng.integers(n_pop))
            if j in seen:
                continue
            seen.add(j)
            out[filled] = population[j]
            filled += 1
        return out
    return rng.permutation(population)[:size]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _property_vector(
    table: pd.DataFrame, prop: MatchingProperty, genes: Sequence[GeneId]
) -> np.ndarray:
    column = prop.column
    if column not in table.columns:
        raise ValueError(
            f"gene_stats table has no column {column!r} required by matching property "
            f"{prop.value!r}; available columns: {list(table.columns)}"
        )
    values = np.asarray(table.loc[list(genes), column].to_numpy(), dtype=np.float64)
    if not np.all(np.isfinite(values)):
        bad = int(np.count_nonzero(~np.isfinite(values)))
        raise ValueError(
            f"{bad} genes have a non-finite {column!r} statistic; the eligible universe "
            "must be filtered before matching (see NullModel.eligible_universe)"
        )
    return values


def build_matched_background(
    *,
    gene_stats: GeneStats,
    universe: Sequence[GeneId],
    candidate_genes: Sequence[GeneId],
    matching: MatchingSpec,
    min_cell_pool: int = MIN_CELL_POOL,
    pool_per_candidate_gene: int = POOL_PER_CANDIDATE_GENE,
) -> MatchedBackground:
    """Bin the background and validate that it can actually match the candidate.

    Runs the coarsening ladder from :func:`coarsening_ladder`, accepting the
    first configuration where every occupied cell satisfies
    ``pool[c] >= max(min_cell_pool, pool_per_candidate_gene * need[c])``.

    Bins are computed over ``universe | candidate_genes`` so that a candidate
    gene excluded from the sampling pool
    (:attr:`~signull.types.MatchingSpec.exclude_candidate_genes`) still has a
    well-defined cell; pool counts only ever include ``universe`` members.

    Raises
    ------
    ValueError
        Empty candidate, universe smaller than the candidate, or a candidate /
        universe gene missing from ``gene_stats``.
    BackgroundExhaustedError
        Adequacy still violated at the coarsening floor.  ``F4`` in
        ``docs/statistical-design.md`` Sec. 8: refuse the competitive null
        rather than report a p-value from a null that is not matched.
    """
    universe_ids = tuple(dict.fromkeys(str(g) for g in universe))
    candidate_ids = tuple(str(g) for g in candidate_genes)
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate_genes contains duplicates; resolve the signature first")
    m = len(candidate_ids)
    if m == 0:
        raise ValueError("candidate has no genes; nothing to size- or property-match")
    if len(universe_ids) < m:
        raise ValueError(
            f"sampling universe has {len(universe_ids)} genes, fewer than the observed "
            f"candidate size {m}; a size-matched null cannot be drawn"
        )

    table = gene_stats.table
    missing = [g for g in universe_ids if g not in table.index]
    missing += [g for g in candidate_ids if g not in table.index]
    if missing:
        raise ValueError(
            f"{len(missing)} genes are absent from the gene_stats table, e.g. {missing[:5]}"
        )

    diagnostics: list[Diagnostic] = []
    properties = tuple(matching.properties)

    # Binning universe: pool plus any candidate gene held out of the pool.
    universe_index = {g: i for i, g in enumerate(universe_ids)}
    extra = tuple(g for g in candidate_ids if g not in universe_index)
    binning_ids = universe_ids + extra
    n_universe = len(universe_ids)

    if not properties:
        # N0: one cell, the whole universe.  No adequacy rule applies -- nothing
        # is being matched and there is nothing to coarsen.
        diagnostics.append(
            Diagnostic(
                code="unmatched_null_requested",
                severity=Severity.WARNING,
                message=(
                    "MatchingSpec.properties is empty: gene sets will be drawn uniformly "
                    "from the universe, matched on size only. This null is a diagnostic "
                    "baseline (N0); it is biased toward declaring the candidate "
                    "significant because it does not control mean expression, variance "
                    "or detection rate."
                ),
                context={"n_universe": n_universe, "candidate_size": m},
            )
        )
        return MatchedBackground(
            universe=universe_ids,
            properties=(),
            bin_counts=(),
            nested=bool(matching.nested),
            candidate_genes=candidate_ids,
            candidate_cells=np.zeros(m, dtype=np.int64),
            cell_of_universe=np.zeros(n_universe, dtype=np.int64),
            pool_members=np.arange(n_universe, dtype=np.int64),
            pool_offsets=np.array([0, n_universe], dtype=np.int64),
            need_by_cell={0: m},
            min_cell_pool=n_universe,
            diagnostics=tuple(diagnostics),
        )

    if not matching.nested:
        diagnostics.append(
            Diagnostic(
                code=MARGINAL_GRID_CODE,
                severity=Severity.WARNING,
                message=(
                    "MatchingSpec.nested=False: bins form a product grid of marginal "
                    "quantiles. docs/statistical-design.md Sec. 2.2 rejects this "
                    "construction -- mean and variance are strongly dependent, so most "
                    "product cells are sparse or empty. Diagnostic use only."
                ),
                context={"properties": [p.value for p in properties]},
            )
        )

    values = [_property_vector(table, p, binning_ids) for p in properties]

    requested_counts = list(resolve_bin_counts(properties, matching, n_universe))
    for j, prop in enumerate(properties):
        sd = float(np.std(values[j][:n_universe]))
        if sd < DEGENERATE_SD and requested_counts[j] > 1:
            diagnostics.append(
                Diagnostic(
                    code=DEGENERATE_PROPERTY_CODE,
                    severity=Severity.INFO,
                    message=(
                        f"matching property {prop.value!r} is degenerate on this cohort "
                        f"(sd = {sd:.4g} < {DEGENERATE_SD}); collapsing it to a single "
                        "bin rather than splitting near-identical genes."
                    ),
                    context={"property": prop.value, "sd": sd},
                )
            )
            requested_counts[j] = 1

    ladder = coarsening_ladder(requested_counts)
    accepted: tuple[int, ...] | None = None
    cells_all: IntArray | None = None
    worst: tuple[int, int, int] | None = None

    for counts in ladder:
        cells_all = assign_cells(values, counts, nested=bool(matching.nested))
        linear = np.ravel_multi_index(
            tuple(cells_all[:, j] for j in range(len(counts))), counts
        )
        pool_counts = np.bincount(linear[:n_universe], minlength=int(np.prod(counts)))
        need_counts = np.bincount(
            linear[[binning_ids.index(g) for g in candidate_ids]]
            if extra
            else linear[[universe_index[g] for g in candidate_ids]],
            minlength=int(np.prod(counts)),
        )
        occupied = np.flatnonzero(need_counts > 0)
        required = np.maximum(min_cell_pool, pool_per_candidate_gene * need_counts[occupied])
        deficit = pool_counts[occupied] - required
        if deficit.size == 0 or int(deficit.min()) >= 0:
            accepted = counts
            break
        j = int(np.argmin(deficit))
        worst = (
            int(occupied[j]),
            int(pool_counts[occupied[j]]),
            int(required[j]),
        )

    if accepted is None:
        assert worst is not None
        floor = ladder[-1]
        raise BackgroundExhaustedError(
            "property matching is not achievable on this background: at the coarsening "
            f"floor {floor} the cell {tuple(int(v) for v in np.unravel_index(worst[0], floor))} "
            f"holds only {worst[1]} background genes but needs {worst[2]} "
            f"(rule: pool >= max({min_cell_pool}, {pool_per_candidate_gene} * need)). "
            f"|universe| = {n_universe}, candidate size = {m}. "
            "Refuse the competitive null (docs/statistical-design.md F4) rather than "
            "reporting a p-value from a null that is not matched."
        )

    if accepted != ladder[0]:
        diagnostics.append(
            Diagnostic(
                code=BIN_COARSENED_CODE,
                severity=Severity.WARNING,
                message=(
                    f"matching bins coarsened from {ladder[0]} to {accepted} so that every "
                    f"occupied cell holds at least max({min_cell_pool}, "
                    f"{pool_per_candidate_gene} x need) background genes."
                ),
                context={
                    "requested_bins": list(ladder[0]),
                    "final_bins": list(accepted),
                    "ladder_steps": int(ladder.index(accepted)),
                    "properties": [p.value for p in properties],
                },
            )
        )

    assert cells_all is not None
    linear = np.ravel_multi_index(
        tuple(cells_all[:, j] for j in range(len(accepted))), accepted
    )
    cell_of_universe = np.asarray(linear[:n_universe], dtype=np.int64)
    if extra:
        binning_index = {g: i for i, g in enumerate(binning_ids)}
        candidate_cells = np.asarray(
            [linear[binning_index[g]] for g in candidate_ids], dtype=np.int64
        )
    else:
        candidate_cells = np.asarray(
            [linear[universe_index[g]] for g in candidate_ids], dtype=np.int64
        )

    n_cells = int(np.prod(accepted))
    order = np.argsort(cell_of_universe, kind="stable")
    pool_members = np.asarray(order, dtype=np.int64)
    pool_offsets = np.zeros(n_cells + 1, dtype=np.int64)
    np.cumsum(np.bincount(cell_of_universe, minlength=n_cells), out=pool_offsets[1:])

    need_by_cell = {
        int(c): int(n) for c, n in zip(*np.unique(candidate_cells, return_counts=True))
    }
    min_pool = min(int(pool_offsets[c + 1] - pool_offsets[c]) for c in need_by_cell)

    diagnostics.append(
        Diagnostic(
            code=BIN_LAYOUT_CODE,
            severity=Severity.INFO,
            message=(
                f"matched background: properties {[p.value for p in properties]}, "
                f"{'nested' if matching.nested else 'marginal'} bins {accepted}, "
                f"{len(need_by_cell)} occupied cells, smallest occupied-cell pool "
                f"{min_pool}, |universe| = {n_universe}, candidate size = {m}."
            ),
            context={
                "properties": [p.value for p in properties],
                "bins": list(accepted),
                "nested": bool(matching.nested),
                "n_occupied_cells": len(need_by_cell),
                "min_cell_pool": min_pool,
                "n_universe": n_universe,
                "candidate_size": m,
            },
        )
    )

    return MatchedBackground(
        universe=universe_ids,
        properties=properties,
        bin_counts=accepted,
        nested=bool(matching.nested),
        candidate_genes=candidate_ids,
        candidate_cells=candidate_cells,
        cell_of_universe=cell_of_universe,
        pool_members=pool_members,
        pool_offsets=pool_offsets,
        need_by_cell=need_by_cell,
        min_cell_pool=min_pool,
        diagnostics=tuple(diagnostics),
    )
