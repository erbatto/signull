"""Path B: supervised scoring, where every fit lives inside the cross-validation.

``docs/statistical-design.md`` Sec. 5 Path B.  A supervised score consults ``y``,
so in-sample predictions are not a valid input to a metric and cross-validation
is mandatory for validity rather than a variance-reduction nicety.  Three rules
carry the whole path, and each is enforced structurally here:

1. **Everything that touches ``y`` goes inside the fold.**  Standardisation,
   gene selection and the model fit are one scikit-learn ``Pipeline`` fitted on
   the training part only; the held-out fold is transformed, never fitted.
   Selection outside the loop is the classic catastrophic version of this error
   (Ambroise & McLachlan 2002).
2. **The same folds for the candidate and every null draw.**  Fold assignment is
   derived from :attr:`~signull.types.CVSpec.seed`, not from the per-draw
   generator, so fold randomness contributes nothing to the null's spread.  If
   it did, the null would be inflated and the test would lose power.
3. **Out-of-fold predictions only.**  :attr:`SampleScores.is_out_of_fold` is set;
   with repeats, the per-sample out-of-fold predictions are averaged over
   repeats after being rank-normalised within each repeat, so repeats are
   comparable on a common scale before averaging.

Cost: the entire CV is re-run for every null draw, which is what makes the
supervised path expensive enough that Sec. 3.3 permits ``K = 1000`` there with an
explicit reduced-resolution note.  Reusing the candidate's fitted model with a
swapped gene set is not a null and is not offered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Mapping

import numpy as np
from scipy.stats import rankdata
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..types import (
    AlignedDataset,
    CVSpec,
    SampleScores,
    ScoringMethodName,
    ScoringSpec,
    Signature,
)
from .base import EmptySignatureError, make_scores, signature_block

__all__ = ["SupervisedModelScorer", "SUPPORTED_MODELS", "make_estimator"]

#: Model keys understood by :class:`SupervisedModelScorer`.
SUPPORTED_MODELS: Final[tuple[str, ...]] = ("logistic", "lasso", "ridge", "lda")


def make_estimator(model: str, *, c: float, random_state: int):
    """Build the unfitted estimator for ``model``.

    ``logistic`` is unpenalised-ish (large ``C``); ``ridge`` is L2 and ``lasso``
    is L1 with ``C`` as the inverse penalty strength.  ``lda`` is a shrinkage
    linear discriminant, which is the classic ``m >> n`` choice for expression
    signatures and needs no tuning.
    """
    if model == "logistic":
        return LogisticRegression(C=c, max_iter=5000, random_state=random_state)
    if model == "ridge":
        return LogisticRegression(
            C=c, penalty="l2", solver="lbfgs", max_iter=5000, random_state=random_state
        )
    if model == "lasso":
        return LogisticRegression(
            C=c, penalty="l1", solver="liblinear", max_iter=5000, random_state=random_state
        )
    if model == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    raise ValueError(f"unknown model {model!r}; supported: {list(SUPPORTED_MODELS)!r}")


@dataclass(frozen=True, slots=True)
class SupervisedModelScorer:
    """Cross-validated supervised score.  Satisfies ``ScoringMethod``.

    Attributes
    ----------
    model:
        One of :data:`SUPPORTED_MODELS`.
    cv:
        Fold scheme.  ``CVSpec.seed`` fixes the partition **globally**: the same
        seed must be used for the candidate and for every null draw, which is
        why it lives on the scorer (a run-level object shared by all draws) and
        not on the per-draw generator.
    c:
        Inverse regularisation strength for the penalised models.
    n_select:
        When set, the top ``n_select`` genes by absolute two-sample t statistic
        are selected **inside each training fold**.  ``None`` uses every
        signature gene, which is the honest default: the signature is the
        hypothesis, and re-selecting within it is a second, undeclared model.
    """

    model: str = "logistic"
    cv: CVSpec = field(default_factory=CVSpec)
    c: float = 1.0
    n_select: int | None = None

    def __post_init__(self) -> None:
        if self.model not in SUPPORTED_MODELS:
            raise ValueError(
                f"unknown model {self.model!r}; supported: {list(SUPPORTED_MODELS)!r}"
            )
        if self.cv.n_folds < 2:
            raise ValueError(f"CVSpec.n_folds must be >= 2, got {self.cv.n_folds}")
        if self.cv.n_repeats < 1:
            raise ValueError(f"CVSpec.n_repeats must be >= 1, got {self.cv.n_repeats}")
        if not self.cv.stratified:
            raise ValueError(
                "unstratified folds are not supported for a binary endpoint: an "
                "unstratified fold can be single-class, which leaves the metric "
                "undefined on that fold"
            )
        if self.n_select is not None and self.n_select < 1:
            raise ValueError(f"n_select must be >= 1 or None, got {self.n_select}")

    @property
    def name(self) -> ScoringMethodName:
        """:attr:`~signull.types.ScoringMethodName.SUPERVISED_MODEL`."""
        return ScoringMethodName.SUPERVISED_MODEL

    @property
    def params(self) -> Mapping[str, object]:
        """Serialisable hyper-parameters, complete enough to reconstruct the run."""
        return {
            "model": self.model,
            "c": float(self.c),
            "n_select": self.n_select,
            "n_folds": self.cv.n_folds,
            "n_repeats": self.cv.n_repeats,
            "stratified": self.cv.stratified,
            "cv_seed": self.cv.seed,
        }

    @property
    def is_supervised(self) -> bool:
        """``True``.  Consumers must treat the scores as out-of-fold predictions."""
        return True

    def spec(self) -> ScoringSpec:
        """Config-capture form of this scorer."""
        return ScoringSpec(name=self.name, params=dict(self.params))

    def score(
        self,
        dataset: AlignedDataset,
        signature: Signature,
        rng: np.random.Generator,
    ) -> SampleScores:
        """Return out-of-fold predicted probabilities, averaged over repeats.

        ``rng`` is deliberately **unused**: fold assignment comes from
        ``self.cv.seed`` so that the candidate and every null draw see the same
        partition.  Drawing folds from the per-draw generator would let fold
        noise inflate the null.

        Raises
        ------
        ValueError
            A class too small for the requested number of folds, or a signature
            with no genes.
        """
        block, gene_ids = signature_block(dataset, signature)
        if block.shape[0] == 0:
            raise EmptySignatureError(f"signature {signature.name!r} has no genes")

        x = np.ascontiguousarray(block.T)  # samples x genes
        y = np.asarray(dataset.outcome.labels, dtype=int)
        n_positive = int(y.sum())
        n_negative = int(y.size - n_positive)
        if min(n_positive, n_negative) < self.cv.n_folds:
            raise ValueError(
                f"{self.cv.n_folds}-fold stratified CV needs at least {self.cv.n_folds} "
                f"samples in each class; this cohort has {n_positive} positive and "
                f"{n_negative} negative"
            )

        seed = 0 if self.cv.seed is None else int(self.cv.seed)
        splitter = (
            StratifiedKFold(n_splits=self.cv.n_folds, shuffle=True, random_state=seed)
            if self.cv.n_repeats == 1
            else RepeatedStratifiedKFold(
                n_splits=self.cv.n_folds, n_repeats=self.cv.n_repeats, random_state=seed
            )
        )

        n_samples = x.shape[0]
        pooled = np.zeros(n_samples, dtype=np.float64)
        repeat_scores = np.empty(n_samples, dtype=np.float64)
        fold_number = 0
        for train_idx, test_idx in splitter.split(x, y):
            repeat_scores[test_idx] = self._fit_predict(
                x[train_idx], y[train_idx], x[test_idx], seed=seed + fold_number
            )
            fold_number += 1
            if fold_number % self.cv.n_folds == 0:
                # One repeat is complete: every sample has exactly one
                # out-of-fold prediction.  Rank-normalise before pooling so
                # repeats with different score scales contribute equally.
                pooled += rankdata(repeat_scores) / n_samples
        values = pooled / self.cv.n_repeats
        return make_scores(values, dataset, self.name, is_out_of_fold=True)

    def _fit_predict(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        *,
        seed: int,
    ) -> np.ndarray:
        """Fit on the training fold only and predict the held-out fold.

        Gene selection, standardisation and the model fit all happen here, on
        ``x_train`` alone.  Nothing computed from ``x_test`` or from the full
        cohort enters the fit.
        """
        columns = self._select(x_train, y_train)
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", make_estimator(self.model, c=self.c, random_state=seed)),
            ]
        )
        pipeline.fit(x_train[:, columns], y_train)
        return np.asarray(
            pipeline.predict_proba(x_test[:, columns])[:, 1], dtype=np.float64
        )

    def _select(self, x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Top-``n_select`` gene columns by ``|t|`` on the training fold.

        Uses Welch's t statistic without a p-value, since only the ordering
        matters.  Genes constant within the training fold get ``|t| = 0`` and
        therefore sort last.
        """
        n_genes = x_train.shape[1]
        if self.n_select is None or self.n_select >= n_genes:
            return np.arange(n_genes)
        positive = x_train[y_train == 1]
        negative = x_train[y_train == 0]
        numerator = positive.mean(axis=0) - negative.mean(axis=0)
        denominator = np.sqrt(
            positive.var(axis=0, ddof=1) / positive.shape[0]
            + negative.var(axis=0, ddof=1) / negative.shape[0]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(denominator > 0.0, np.abs(numerator) / denominator, 0.0)
        # Stable order: ties broken by gene position, so the selection is
        # reproducible regardless of the sort implementation.
        return np.sort(np.argsort(-t, kind="stable")[: self.n_select])
