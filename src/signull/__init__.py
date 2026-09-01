"""signull -- is this signature better than a size-matched random one?

Given a candidate gene signature, a cohort expression matrix and a binary
outcome, ``signull`` answers that question with a defensible number: an
empirical p-value against a null distribution of signatures drawn from *this*
dataset and matched to the candidate on size and on per-gene properties.

Why the matching matters: 60 % of 47 published breast-cancer outcome signatures
were no better than size-matched random signatures, and more than 90 % of random
signatures longer than 100 genes were significant outcome predictors (Venet et
al. 2011).  Existing tools sample the null uniformly from all features, which is
biased in the direction that flatters the candidate.

Package map (``docs/architecture.md`` Sec. 2)::

    types    the contracts; imports nothing from signull
    data     loading, alignment, identifier resolution, eligible background
    nulls    random gene-set and label-permutation samplers, the p-value
    scoring  signature scoring strategies
    metrics  AUROC, average precision, direction policy, intervals
    report   rendering (wave 3)

``data``, ``nulls``, ``scoring`` and ``metrics`` all depend on ``types`` and on
nothing else in the package, so the evaluation loop that joins a sampler to a
scorer can only live above them.  That is what makes "the same scoring method is
applied to the candidate and to every null draw" structurally true.
"""

from __future__ import annotations

from .types import CONTRACT_VERSION

__version__ = "0.2.0"

__all__ = ["CONTRACT_VERSION", "__version__"]
