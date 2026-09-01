"""Signature scoring strategies and the name -> strategy registry.

Every strategy satisfies :class:`signull.types.ScoringMethod` and is applied by
the *same instance* to the candidate and to every null draw.  The strategies are
frozen dataclasses holding no fitted state, so there is no candidate-only code
path to write even by accident -- the structural version of "the scoring method
must be identical for candidate and nulls", which the literature shows to be a
results-changing choice.

Two paths (``docs/statistical-design.md`` Sec. 5):

Path A -- unsupervised (default)
    :class:`MeanZScoreScorer`, :class:`EigengeneScorer`.  Nothing is fitted on
    ``y``, so no cross-validation is required for validity and both nulls are
    exact.

Path B -- supervised
    :class:`SupervisedModelScorer`.  Cross-validation is mandatory, every fit --
    including gene selection -- happens inside the folds, and the whole CV is
    re-run for every null draw.

This package depends only on :mod:`signull.types`.  It never imports ``nulls``
or ``metrics``, and it reads ``dataset.outcome`` only inside a CV fold of
Path B.
"""

from __future__ import annotations

from typing import Final, Mapping

from ..types import ScoringMethodName
from .base import (
    ConstantGeneError,
    EmptySignatureError,
    make_scores,
    row_standardise,
    signature_block,
    split_weights,
)
from .supervised import SUPPORTED_MODELS, SupervisedModelScorer, make_estimator
from .unsupervised import EigengeneScorer, MeanZScoreScorer

__all__ = [
    "ConstantGeneError",
    "EigengeneScorer",
    "EmptySignatureError",
    "MeanZScoreScorer",
    "REGISTRY",
    "SUPPORTED_MODELS",
    "SupervisedModelScorer",
    "get",
    "make_estimator",
    "make_scores",
    "row_standardise",
    "signature_block",
    "split_weights",
]

#: ``ScoringMethodName -> factory``.  The CLI resolves a scorer name only through
#: this map, so a new strategy is one entry rather than a new branch in the
#: pipeline.  ``SSGSEA`` is deliberately absent until it is implemented: a
#: missing key raises here, where the run can still be fixed, rather than
#: silently scoring with something else.
REGISTRY: Final[Mapping[ScoringMethodName, type]] = {
    ScoringMethodName.MEAN_Z_SCORE: MeanZScoreScorer,
    ScoringMethodName.FIRST_PRINCIPAL_COMPONENT: EigengeneScorer,
    ScoringMethodName.SUPERVISED_MODEL: SupervisedModelScorer,
}


def get(name: ScoringMethodName | str, **params: object):
    """Instantiate the scorer registered under ``name``.

    Raises
    ------
    ValueError
        ``name`` is not a known :class:`~signull.types.ScoringMethodName`, or it
        names a strategy that has no implementation yet.
    """
    key = ScoringMethodName(name)
    try:
        factory = REGISTRY[key]
    except KeyError:
        raise ValueError(
            f"no scoring strategy is registered for {key.value!r}; "
            f"available: {[k.value for k in REGISTRY]!r}"
        ) from None
    return factory(**params)  # type: ignore[operator]
