"""Unsupervised signature scorers -- Path A of Sec. 5, the default path.

Both strategies here satisfy the :class:`~signull.types.ScoringMethod`
protocol, are pure functions of ``(matrix, signature)``, and never touch
``dataset.outcome``.  That is invariant I1: because nothing is fitted on ``y``,
no cross-validation is required for validity, the permutation null is exact and
the competitive null is exact.

The *same instance* is applied to the candidate and to every null draw.  These
objects are frozen and hold no fitted state, so there is no candidate-only code
path to write even by accident.

* :class:`MeanZScoreScorer` -- design S1.  The MVP.
* :class:`EigengeneScorer` -- design S2, the default for unsigned signatures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..types import (
    AlignedDataset,
    SampleScores,
    ScoringMethodName,
    ScoringSpec,
    Signature,
)
from .base import EmptySignatureError, make_scores, row_standardise, signature_block, split_weights

__all__ = ["MeanZScoreScorer", "EigengeneScorer"]


@dataclass(frozen=True, slots=True)
class MeanZScoreScorer:
    """Design S1: per-gene z across samples, then the mean across signature genes.

    Unsigned::

        z = row_standardise(X[S, :])          # mean 0, sd 1 per gene
        score_j = mean_{g in S} z[g, j]

    Signed (``Signature.weights`` present)::

        score_j = wmean_{w>0} z[g, j] - wmean_{w<0} z[g, j]

    which reduces exactly to the design's ``mean(S+) - mean(S-)`` when the
    weights are +-1.  Weight magnitudes act as emphasis inside each sign group,
    so the two groups stay on a common scale regardless of their sizes.

    Note the known bias this carries for *unsigned* heterogeneous signatures:
    an arithmetic mean cancels anti-correlated members, under-scoring the
    candidate while weakly-correlated random sets are unaffected.  That bias
    runs *against* the candidate, so it is conservative -- but
    :class:`EigengeneScorer` is the design's default when no directions are
    supplied.

    Attributes
    ----------
    constant_gene_policy:
        ``"raise"`` (default) or ``"drop"`` for zero-variance genes.
    ddof:
        Degrees of freedom for the per-gene standard deviation.
    """

    constant_gene_policy: str = "raise"
    ddof: int = 0

    @property
    def name(self) -> ScoringMethodName:
        """:attr:`~signull.types.ScoringMethodName.MEAN_Z_SCORE`."""
        return ScoringMethodName.MEAN_Z_SCORE

    @property
    def params(self) -> Mapping[str, object]:
        """Serialisable hyper-parameters."""
        return {"constant_gene_policy": self.constant_gene_policy, "ddof": self.ddof}

    @property
    def is_supervised(self) -> bool:
        """``False``.  This scorer never sees the outcome."""
        return False

    def spec(self) -> ScoringSpec:
        """Config-capture form of this scorer."""
        return ScoringSpec(name=self.name, params=dict(self.params))

    def score(
        self,
        dataset: AlignedDataset,
        signature: Signature,
        rng: np.random.Generator,
    ) -> SampleScores:
        """Score every sample.  ``rng`` is unused: the strategy is deterministic."""
        block, gene_ids = signature_block(dataset, signature)
        z, kept = row_standardise(
            block,
            gene_ids=gene_ids,
            constant_gene_policy=self.constant_gene_policy,
            ddof=self.ddof,
        )
        pos, neg = split_weights(signature, kept)
        if pos is None:
            values = z.mean(axis=0)
        else:
            values = np.zeros(z.shape[1], dtype=np.float64)
            if pos.sum() > 0.0:
                values += (pos[:, None] * z).sum(axis=0) / pos.sum()
            if neg.sum() > 0.0:
                values -= (neg[:, None] * z).sum(axis=0) / neg.sum()
        return make_scores(values, dataset, self.name)


@dataclass(frozen=True, slots=True)
class EigengeneScorer:
    """Design S2: the signature eigengene -- default for *unsigned* signatures.

    ::

        Z_S = row_standardise(X[S, :])
        u1  = first left singular vector of Z_S (rows centred over samples)
        score = u1' Z_S
        flip sign so corr(score, colMeans(Z_S)) >= 0

    Rationale (Sec. 1): an unsigned arithmetic mean cancels anti-correlated
    members and systematically under-scores heterogeneous signatures, while
    weakly-correlated random sets are unaffected.  That is a bias *against* the
    candidate.  The eigengene is still ``y``-blind, so invariant I1 holds.

    The sign convention is fixed against the signature's own mean profile,
    never against ``y``.  Fixing it against ``y`` would be exactly the one-bit
    fit that :class:`~signull.types.DirectionPolicy` exists to absorb.
    """

    constant_gene_policy: str = "raise"
    ddof: int = 0

    @property
    def name(self) -> ScoringMethodName:
        """:attr:`~signull.types.ScoringMethodName.FIRST_PRINCIPAL_COMPONENT`."""
        return ScoringMethodName.FIRST_PRINCIPAL_COMPONENT

    @property
    def params(self) -> Mapping[str, object]:
        """Serialisable hyper-parameters."""
        return {"constant_gene_policy": self.constant_gene_policy, "ddof": self.ddof}

    @property
    def is_supervised(self) -> bool:
        """``False``.  This scorer never sees the outcome."""
        return False

    def spec(self) -> ScoringSpec:
        """Config-capture form of this scorer."""
        return ScoringSpec(name=self.name, params=dict(self.params))

    def score(
        self,
        dataset: AlignedDataset,
        signature: Signature,
        rng: np.random.Generator,
    ) -> SampleScores:
        """Score every sample.  ``rng`` is unused: the SVD is deterministic."""
        block, gene_ids = signature_block(dataset, signature)
        z, _ = row_standardise(
            block,
            gene_ids=gene_ids,
            constant_gene_policy=self.constant_gene_policy,
            ddof=self.ddof,
        )
        if z.shape[0] == 0:
            raise EmptySignatureError("no scorable genes remain")
        mean_profile = z.mean(axis=0)
        if z.shape[0] == 1:
            return make_scores(z[0], dataset, self.name)

        centred = z - z.mean(axis=1, keepdims=True)
        u, _, _ = np.linalg.svd(centred, full_matrices=False)
        u1 = u[:, 0]
        values = u1 @ z
        # Sign convention: agree with the signature's mean profile.  y is never
        # consulted here.
        sd_v = values.std()
        sd_m = mean_profile.std()
        if sd_v > 0.0 and sd_m > 0.0:
            if float(np.corrcoef(values, mean_profile)[0, 1]) < 0.0:
                values = -values
        elif float(u1.sum()) < 0.0:
            values = -values
        return make_scores(values, dataset, self.name)
