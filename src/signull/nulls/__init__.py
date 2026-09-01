"""Null models, matched backgrounds, seeding and the empirical p-value estimator.

Public surface
--------------
Models (both satisfy :class:`signull.types.NullModel`)
    :class:`RandomGeneSetNull` -- the competitive null (N0/N1/N3), matched on
    size unconditionally and on gene properties by default.
    :class:`LabelPermutationNull` -- the self-contained null (N2), which the
    calibration test drives.
    :data:`REGISTRY` / :func:`get` resolve a :class:`~signull.types.NullType` to
    a model, so the CLI never branches on the null.

Matching
    :func:`build_matched_background` bins the eligible background once per run;
    :func:`default_matching_spec` is the normative N1 configuration (nested
    mean -> variance -> detection-rate bins).

p-value
    :func:`empirical_p_value` implements the add-one estimator
    ``(1 + r) / (1 + K)``.  It cannot return 0: an uncorrected zero claims
    infinite evidence from a finite number of draws.

Seeding
    :func:`generator_for` derives a component's root generator from the run
    seed; :func:`draw_seeds` gives every draw its own independent seed.

This package depends only on :mod:`signull.types`.  It imports nothing from
``data``, ``scoring``, ``metrics`` or ``report``, and it contains no evaluation
loop -- joining draws to a scorer is wave-3 pipeline work.
"""

from __future__ import annotations

from .binning import (
    COARSENING_FLOOR_SECOND,
    MIN_CELL_POOL,
    POOL_PER_CANDIDATE_GENE,
    BackgroundExhaustedError,
    MatchedBackground,
    assign_cells,
    build_matched_background,
    coarsening_ladder,
    default_matching_spec,
    l1_offsets,
    quantile_bins,
    resolve_bin_counts,
)
from .pvalue import EmpiricalPValue, empirical_p_value, p_value_floor
from .samplers import (
    REGISTRY,
    SET_LEVEL_CONSTRAINTS,
    DrawFloorError,
    LabelPermutationNull,
    RandomGeneSetNull,
    get,
    null_signature_name,
)
from .seeding import (
    N_SEED_STREAMS,
    SeedStream,
    draw_seeds,
    generator_for,
    generator_from_seed,
    seed_sequence_for,
    spawn_seed_sequences,
)

__all__ = [
    "BackgroundExhaustedError",
    "COARSENING_FLOOR_SECOND",
    "DrawFloorError",
    "EmpiricalPValue",
    "LabelPermutationNull",
    "MIN_CELL_POOL",
    "MatchedBackground",
    "N_SEED_STREAMS",
    "POOL_PER_CANDIDATE_GENE",
    "REGISTRY",
    "RandomGeneSetNull",
    "SET_LEVEL_CONSTRAINTS",
    "SeedStream",
    "assign_cells",
    "build_matched_background",
    "coarsening_ladder",
    "default_matching_spec",
    "draw_seeds",
    "empirical_p_value",
    "generator_for",
    "generator_from_seed",
    "get",
    "l1_offsets",
    "null_signature_name",
    "p_value_floor",
    "quantile_bins",
    "resolve_bin_counts",
    "seed_sequence_for",
    "spawn_seed_sequences",
]
