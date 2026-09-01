"""Deterministic seed derivation for every stochastic component of a run.

One integer -- :attr:`signull.types.NullTestConfig.seed` -- is the root of all
randomness in ``signull``.  Per-component streams are derived from it with
:class:`numpy.random.SeedSequence` spawning, never with a global
``np.random.*`` call and never with a bare ``default_rng()``.

The spawn order is fixed by ``docs/architecture.md`` Sec. 6 and must not change,
because changing it silently changes every previously published number::

    root = np.random.SeedSequence(seed)
    gene_ss, perm_ss, cv_ss, score_ss = root.spawn(4)

``0`` gene-set sampling, ``1`` label permutation, ``2`` CV folds, ``3``
stochastic scoring.  ``nulls/`` owns streams 0 and 1; 2 and 3 are derived here
so that the two packages that need them cannot disagree about the order.

Within a null model, draw ``i`` gets its own generator seeded from a
pre-materialised seed vector (:func:`draw_seeds`).  That makes draw ``i``
reproducible *without* replaying draws ``0..i-1``, as the
:class:`~signull.types.NullModel` contract requires, and it gives the label
permutation null an integer to record in
:attr:`~signull.types.BinaryOutcome.permutation_seed`.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final

import numpy as np

__all__ = [
    "SeedStream",
    "N_SEED_STREAMS",
    "spawn_seed_sequences",
    "seed_sequence_for",
    "generator_for",
    "draw_seeds",
    "generator_from_seed",
]


class SeedStream(IntEnum):
    """Fixed spawn index of each stochastic component.

    The integer values *are* the contract: they index into
    ``SeedSequence(seed).spawn(4)``.
    """

    GENE_SET = 0
    PERMUTATION = 1
    CV = 2
    SCORING = 3


#: Number of children spawned from the root seed sequence.
N_SEED_STREAMS: Final[int] = len(SeedStream)

#: Upper bound (exclusive) for per-draw seed integers; keeps them int64-safe and
#: therefore JSON- and report-friendly.
_MAX_DRAW_SEED: Final[int] = int(np.iinfo(np.int64).max)


def spawn_seed_sequences(seed: int | None) -> tuple[np.random.SeedSequence, ...]:
    """Spawn the four component seed sequences from the run's root ``seed``.

    Parameters
    ----------
    seed:
        Root seed.  ``None`` draws fresh OS entropy -- valid, but the run is
        then not reproducible, so callers that care must pass an integer.

    Returns
    -------
    tuple
        Four :class:`numpy.random.SeedSequence` objects in :class:`SeedStream`
        order.
    """
    root = np.random.SeedSequence(seed)
    return tuple(root.spawn(N_SEED_STREAMS))


def seed_sequence_for(seed: int | None, stream: SeedStream) -> np.random.SeedSequence:
    """Seed sequence of one component, derived by the fixed spawn order."""
    return spawn_seed_sequences(seed)[int(stream)]


def generator_for(seed: int | None, stream: SeedStream) -> np.random.Generator:
    """Root :class:`numpy.random.Generator` for one component of a run.

    This is the object a caller hands to :meth:`signull.types.NullModel.draw`.
    """
    return np.random.default_rng(seed_sequence_for(seed, stream))


def draw_seeds(rng: np.random.Generator, n_draws: int) -> np.ndarray:
    """Materialise one independent integer seed per draw.

    Drawn as a single vectorised call so that the seed of draw ``i`` is
    ``draw_seeds(rng, n)[i]`` regardless of how many draws have been consumed --
    the "draw ``i`` without replaying ``0..i-1``" clause of the
    :class:`~signull.types.NullModel` contract.

    Parameters
    ----------
    rng:
        The model's root generator for this run.
    n_draws:
        Number of draws; must be >= 1.

    Returns
    -------
    numpy.ndarray
        ``int64`` array of shape ``(n_draws,)`` with strictly positive seeds.
    """
    if n_draws < 1:
        raise ValueError(f"n_draws must be >= 1, got {n_draws}")
    return np.asarray(rng.integers(1, _MAX_DRAW_SEED, size=n_draws, dtype=np.int64))


def generator_from_seed(seed: int) -> np.random.Generator:
    """Per-draw generator from one of the integers produced by :func:`draw_seeds`."""
    return np.random.default_rng(int(seed))
