"""Campaign 6: scoring strategies, metrics and the direction policy."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_dataset, make_matrix, make_outcome, make_signature
from signull.metrics import (
    AurocMetric,
    AveragePrecisionMetric,
    CohortTooSmallError,
    DirectedMetric,
    check_cohort_floors,
    delong_auroc_ci,
)
from signull.metrics import get as get_metric
from signull.scoring import (
    ConstantGeneError,
    EigengeneScorer,
    MeanZScoreScorer,
    SupervisedModelScorer,
    get,
    row_standardise,
)
from signull.types import (
    CVSpec,
    DirectionPolicy,
    MetricName,
    ScoringMethodName,
)


def score_of(scorer, dataset, signature, seed: int = 0):
    """Score with a throwaway generator; every scorer here is deterministic."""
    return scorer.score(dataset, signature, np.random.default_rng(seed))


# ---------------------------------------------------------------------------
# Unsupervised scoring
# ---------------------------------------------------------------------------


def test_mean_z_score_returns_one_finite_value_per_sample(dataset, candidate):
    scores = score_of(MeanZScoreScorer(), dataset, candidate)
    assert scores.sample_ids == dataset.matrix.sample_ids
    assert scores.values.shape == (dataset.n_samples,)
    assert np.isfinite(scores.values).all()
    assert scores.is_out_of_fold is False
    assert scores.method is ScoringMethodName.MEAN_Z_SCORE


def test_mean_z_score_is_invariant_to_gene_order(dataset, candidate):
    shuffled = make_signature(list(candidate.genes)[::-1], name=candidate.name)
    a = score_of(MeanZScoreScorer(), dataset, candidate)
    b = score_of(MeanZScoreScorer(), dataset, shuffled)
    np.testing.assert_allclose(a.values, b.values)


def test_mean_z_score_is_invariant_to_per_gene_affine_rescaling(dataset, candidate):
    frame = dataset.matrix.to_frame()
    scale = np.linspace(1.0, 3.0, frame.shape[0])[:, None]
    rescaled = make_matrix(frame * scale + 7.0).align_to(dataset.outcome)
    a = score_of(MeanZScoreScorer(), dataset, candidate)
    b = score_of(MeanZScoreScorer(), rescaled, candidate)
    np.testing.assert_allclose(a.values, b.values, atol=1e-10)


def test_signed_weights_flip_the_score(dataset):
    genes = list(dataset.matrix.gene_ids[:10])
    up = make_signature(genes, weights=[1.0] * 10)
    down = make_signature(genes, weights=[-1.0] * 10)
    a = score_of(MeanZScoreScorer(), dataset, up)
    b = score_of(MeanZScoreScorer(), dataset, down)
    np.testing.assert_allclose(a.values, -b.values, atol=1e-12)


def test_a_constant_gene_raises_rather_than_producing_a_silent_nan():
    frame = make_matrix().to_frame()
    frame.iloc[0, :] = 4.0
    matrix = make_matrix(frame)
    dataset = matrix.align_to(make_outcome(matrix.sample_ids))
    signature = make_signature(list(matrix.gene_ids[:5]))
    with pytest.raises(ConstantGeneError):
        score_of(MeanZScoreScorer(), dataset, signature)


def test_the_constant_gene_can_be_dropped_when_the_caller_opts_in():
    values = np.vstack(
        [np.full(40, 4.0), np.random.default_rng(0).normal(size=(4, 40))]
    )
    z, kept = row_standardise(values, constant_gene_policy="drop")
    assert kept == (1, 2, 3, 4)
    assert z.shape == (4, 40)


def test_the_eigengene_agrees_with_the_mean_on_a_coherent_signature():
    dataset = make_dataset(n_genes=500, n_samples=50, dominant_axis=3.0, seed=2)
    coherent = make_signature(list(dataset.matrix.gene_ids[:20]))
    mean = score_of(MeanZScoreScorer(), dataset, coherent).values
    eigen = score_of(EigengeneScorer(), dataset, coherent).values
    assert float(np.corrcoef(mean, eigen)[0, 1]) > 0.9


def test_the_eigengene_sign_is_fixed_against_the_signature_not_the_outcome():
    dataset = make_dataset(n_genes=500, n_samples=50, dominant_axis=3.0, seed=2)
    signature = make_signature(list(dataset.matrix.gene_ids[:20]))
    scores = score_of(EigengeneScorer(), dataset, signature)
    mean = score_of(MeanZScoreScorer(), dataset, signature)
    assert float(np.corrcoef(scores.values, mean.values)[0, 1]) > 0.0


def test_unsupervised_scorers_ignore_the_outcome(dataset, candidate):
    permuted = make_outcome(dataset.matrix.sample_ids, seed=999)
    other = dataset.matrix.align_to(permuted)
    for scorer in (MeanZScoreScorer(), EigengeneScorer()):
        np.testing.assert_allclose(
            score_of(scorer, dataset, candidate).values,
            score_of(scorer, other, candidate).values,
        )


# ---------------------------------------------------------------------------
# Supervised scoring (Path B)
# ---------------------------------------------------------------------------


def test_supervised_scores_are_marked_out_of_fold(dataset, candidate):
    scorer = SupervisedModelScorer(cv=CVSpec(n_folds=5, seed=1))
    scores = score_of(scorer, dataset, candidate)
    assert scores.is_out_of_fold is True
    assert scores.method is ScoringMethodName.SUPERVISED_MODEL
    assert np.isfinite(scores.values).all()


def test_supervised_folds_do_not_depend_on_the_per_draw_generator(dataset, candidate):
    """Fold noise must not contribute to the null's spread (Sec. 5 Path B)."""
    scorer = SupervisedModelScorer(cv=CVSpec(n_folds=5, seed=1))
    np.testing.assert_allclose(
        score_of(scorer, dataset, candidate, seed=0).values,
        score_of(scorer, dataset, candidate, seed=12345).values,
    )


def test_a_different_cv_seed_gives_a_different_partition(dataset, candidate):
    a = score_of(SupervisedModelScorer(cv=CVSpec(seed=1)), dataset, candidate)
    b = score_of(SupervisedModelScorer(cv=CVSpec(seed=2)), dataset, candidate)
    assert not np.allclose(a.values, b.values)


def test_supervised_scores_on_pure_noise_do_not_beat_chance():
    """The out-of-fold guard: an in-sample fit would score far above 0.5 here."""
    dataset = make_dataset(n_genes=400, n_samples=80, seed=21)
    signature = make_signature(list(dataset.matrix.gene_ids[:40]))
    scorer = SupervisedModelScorer(cv=CVSpec(n_folds=5, n_repeats=2, seed=3))
    auc = AurocMetric()(score_of(scorer, dataset, signature), dataset.outcome)
    assert 0.3 < auc < 0.7


def test_supervised_scores_recover_a_planted_signal():
    matrix = make_matrix(n_genes=400, n_samples=80, seed=5)
    values = matrix.values
    signal = values[:20].mean(axis=0)
    dataset = matrix.align_to(make_outcome(matrix.sample_ids, signal=signal))
    signature = make_signature(list(matrix.gene_ids[:20]))
    scorer = SupervisedModelScorer(cv=CVSpec(n_folds=5, seed=3))
    auc = AurocMetric()(score_of(scorer, dataset, signature), dataset.outcome)
    assert auc > 0.75


def test_gene_selection_happens_inside_the_folds(dataset, candidate):
    """A selecting scorer still runs; the guard is that it stays near chance."""
    scorer = SupervisedModelScorer(cv=CVSpec(n_folds=5, seed=3), n_select=5)
    auc = AurocMetric()(score_of(scorer, dataset, candidate), dataset.outcome)
    assert 0.25 < auc < 0.75


def test_supervised_scoring_refuses_folds_larger_than_the_smaller_class():
    matrix = make_matrix(n_genes=200, n_samples=40, seed=8)
    labels = np.zeros(40, dtype=bool)
    labels[:8] = True
    from signull.types import BinaryOutcome

    outcome = BinaryOutcome(sample_ids=matrix.sample_ids, labels=labels, name="tiny")
    dataset = matrix.align_to(outcome)
    scorer = SupervisedModelScorer(cv=CVSpec(n_folds=10, seed=1))
    with pytest.raises(ValueError, match="at least 10 samples in each class"):
        score_of(scorer, dataset, make_signature(list(matrix.gene_ids[:10])))


def test_an_unknown_supervised_model_is_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown model"):
        SupervisedModelScorer(model="random_forest_of_hope")


# ---------------------------------------------------------------------------
# Metrics and direction
# ---------------------------------------------------------------------------


def test_auroc_matches_sklearn_on_the_same_inputs(dataset, candidate):
    from sklearn.metrics import roc_auc_score

    scores = score_of(MeanZScoreScorer(), dataset, candidate)
    assert AurocMetric()(scores, dataset.outcome) == pytest.approx(
        roc_auc_score(np.asarray(dataset.outcome.labels), scores.values)
    )


def test_average_precision_chance_level_is_the_prevalence(dataset):
    assert AveragePrecisionMetric().chance_level(dataset.outcome) == pytest.approx(
        dataset.outcome.prevalence
    )


def test_symmetrized_direction_folds_a_below_chance_auroc(dataset, candidate):
    scores = score_of(MeanZScoreScorer(), dataset, candidate)
    flipped = type(scores)(
        values=-scores.values,
        sample_ids=scores.sample_ids,
        method=scores.method,
    )
    directed = DirectedMetric(AurocMetric(), DirectionPolicy.SYMMETRIZED)
    a = directed(scores, dataset.outcome)
    b = directed(flipped, dataset.outcome)
    assert a == pytest.approx(b)
    assert a >= 0.5


def test_as_given_direction_keeps_the_sign(dataset, candidate):
    scores = score_of(MeanZScoreScorer(), dataset, candidate)
    raw = AurocMetric()(scores, dataset.outcome)
    directed = DirectedMetric(AurocMetric(), DirectionPolicy.AS_GIVEN)
    assert directed(scores, dataset.outcome) == pytest.approx(raw)


def test_folding_a_folded_metric_is_refused():
    with pytest.raises(TypeError, match="twice"):
        DirectedMetric(DirectedMetric(AurocMetric()))


def test_cohort_floors_refuse_a_tiny_cohort():
    matrix = make_matrix(n_genes=100, n_samples=20, seed=4)
    labels = np.zeros(20, dtype=bool)
    labels[:9] = True
    from signull.types import BinaryOutcome

    outcome = BinaryOutcome(sample_ids=matrix.sample_ids, labels=labels, name="tiny")
    with pytest.raises(CohortTooSmallError):
        check_cohort_floors(outcome)


def test_delong_interval_brackets_the_point_estimate(dataset, candidate):
    scores = score_of(MeanZScoreScorer(), dataset, candidate)
    auc = AurocMetric()(scores, dataset.outcome)
    interval = delong_auroc_ci(scores.values, np.asarray(dataset.outcome.labels))
    assert interval.contains(auc)
    assert interval.method


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------


def test_the_scoring_registry_resolves_every_implemented_strategy():
    assert isinstance(get("mean_z_score"), MeanZScoreScorer)
    assert isinstance(get(ScoringMethodName.FIRST_PRINCIPAL_COMPONENT), EigengeneScorer)
    assert isinstance(get("supervised_model", model="lda"), SupervisedModelScorer)


def test_an_unimplemented_strategy_raises_rather_than_substituting_another():
    with pytest.raises(ValueError, match="no scoring strategy is registered"):
        get(ScoringMethodName.SSGSEA)


def test_the_metric_registry_resolves_both_metrics():
    assert get_metric("auroc").name is MetricName.AUROC
    assert get_metric(MetricName.AVERAGE_PRECISION).name is MetricName.AVERAGE_PRECISION
