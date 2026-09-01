"""Campaign 5: samplers, matching, seeding and the empirical p-value."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_dataset, make_signature
from signull.nulls import (
    BackgroundExhaustedError,
    DrawFloorError,
    LabelPermutationNull,
    RandomGeneSetNull,
    SeedStream,
    build_matched_background,
    default_matching_spec,
    draw_seeds,
    empirical_p_value,
    generator_for,
    get,
    p_value_floor,
)
from signull.types import (
    MIN_DRAWS,
    Alternative,
    MatchingProperty,
    MatchingSpec,
    NullSpec,
    NullType,
    SignatureOrigin,
)

UNMATCHED = NullSpec(null_type=NullType.RANDOM_GENE_SET, matching=MatchingSpec())


def matched_spec(n_background: int) -> NullSpec:
    """N1: nested mean -> variance -> detection-rate matching."""
    return NullSpec(
        null_type=NullType.RANDOM_GENE_SET, matching=default_matching_spec(n_background)
    )


def draws(model, candidate, dataset, n, seed=3):
    """Materialise ``n`` draws from ``model`` with a fixed seed."""
    return list(model.draw(candidate, dataset, n, generator_for(seed, SeedStream.GENE_SET)))


# ---------------------------------------------------------------------------
# Random gene-set null
# ---------------------------------------------------------------------------


def test_draws_have_the_effective_candidate_size_and_no_duplicates(dataset, candidate):
    model = RandomGeneSetNull(spec=UNMATCHED, enforce_draw_floor=False)
    for draw in draws(model, candidate, dataset, 20):
        assert len(draw.signature.genes) == candidate.size
        assert len(set(draw.signature.genes)) == candidate.size
        assert draw.signature.origin is SignatureOrigin.RANDOM_NULL


def test_draws_carry_the_observed_outcome_unchanged(dataset, candidate):
    model = RandomGeneSetNull(spec=UNMATCHED, enforce_draw_floor=False)
    for draw in draws(model, candidate, dataset, 5):
        assert draw.outcome is dataset.outcome
        assert draw.null_type is NullType.RANDOM_GENE_SET


def test_the_same_seed_reproduces_the_same_draws(dataset, candidate):
    model = RandomGeneSetNull(spec=matched_spec(3000), enforce_draw_floor=False)
    first = [d.signature.genes for d in draws(model, candidate, dataset, 10)]
    second = [d.signature.genes for d in draws(model, candidate, dataset, 10)]
    assert first == second


def test_a_different_seed_gives_different_draws(dataset, candidate):
    model = RandomGeneSetNull(spec=UNMATCHED, enforce_draw_floor=False)
    first = [d.signature.genes for d in draws(model, candidate, dataset, 10, seed=1)]
    second = [d.signature.genes for d in draws(model, candidate, dataset, 10, seed=2)]
    assert first != second


def test_draw_i_does_not_depend_on_how_many_draws_precede_it(dataset, candidate):
    model = RandomGeneSetNull(spec=UNMATCHED, enforce_draw_floor=False)
    long_run = [d.signature.genes for d in draws(model, candidate, dataset, 12)]
    rng = generator_for(3, SeedStream.GENE_SET)
    iterator = model.draw(candidate, dataset, 12, rng)
    next(iterator)
    assert next(iterator).signature.genes == long_run[1]


def test_matched_draws_track_the_candidate_marginals_and_uniform_draws_do_not(
    dataset, candidate
):
    """D1, the product claim: an unmatched null is biased toward the candidate.

    The candidate here is deliberately drawn from the high-expression tail, so
    a uniform draw sits far below it in mean expression while the matched draw
    stays with it.  That gap is exactly the advantage the uniform null credits
    to the candidate for free.
    """
    stats = dataset.matrix.gene_stats().table
    candidate_mean = float(stats.loc[list(candidate.genes), "mean"].mean())

    uniform = RandomGeneSetNull(spec=UNMATCHED, enforce_draw_floor=False)
    matched = RandomGeneSetNull(spec=matched_spec(3000), enforce_draw_floor=False)

    def mean_of(model):
        values = [
            float(stats.loc[list(d.signature.genes), "mean"].mean())
            for d in draws(model, candidate, dataset, 25)
        ]
        return float(np.mean(values))

    matched_gap = abs(mean_of(matched) - candidate_mean)
    uniform_gap = abs(mean_of(uniform) - candidate_mean)
    assert matched_gap < 0.2 * uniform_gap


def test_matched_draws_also_track_variance(dataset, candidate):
    stats = dataset.matrix.gene_stats().table
    candidate_var = float(stats.loc[list(candidate.genes), "variance"].mean())
    model = RandomGeneSetNull(spec=matched_spec(3000), enforce_draw_floor=False)
    drawn = [
        float(stats.loc[list(d.signature.genes), "variance"].mean())
        for d in draws(model, candidate, dataset, 25)
    ]
    assert float(np.mean(drawn)) == pytest.approx(candidate_var, rel=0.15)


def test_unmatched_spec_warns_that_it_is_a_diagnostic_baseline(dataset, candidate):
    model = RandomGeneSetNull(spec=UNMATCHED, enforce_draw_floor=False)
    first = draws(model, candidate, dataset, 1)[0]
    assert any(d.code == "unmatched_null_requested" for d in first.diagnostics)


def test_exclude_candidate_genes_keeps_the_candidate_out_of_its_own_null(
    dataset, candidate
):
    spec = NullSpec(
        null_type=NullType.RANDOM_GENE_SET,
        matching=MatchingSpec(exclude_candidate_genes=True),
    )
    model = RandomGeneSetNull(spec=spec, enforce_draw_floor=False)
    drawn = {g for d in draws(model, candidate, dataset, 20) for g in d.signature.genes}
    assert drawn.isdisjoint(candidate.genes)


def test_candidate_genes_are_sampleable_by_default(dataset, candidate):
    model = RandomGeneSetNull(spec=UNMATCHED, enforce_draw_floor=False)
    universe = model.eligible_universe(dataset)
    assert set(candidate.genes) <= set(universe)


def test_a_signed_candidate_hands_its_weight_multiset_to_every_draw(dataset):
    genes = list(dataset.matrix.gene_ids[:20])
    weights = [1.0] * 10 + [-1.0] * 10
    signed = make_signature(genes, weights=weights)
    model = RandomGeneSetNull(spec=UNMATCHED, enforce_draw_floor=False)
    for draw in draws(model, signed, dataset, 5):
        assert draw.signature.weights == signed.weights


def test_the_universe_is_restricted_to_this_dataset(dataset, candidate):
    model = RandomGeneSetNull(
        spec=UNMATCHED, universe=("GENE_FROM_ANOTHER_COHORT",), enforce_draw_floor=False
    )
    with pytest.raises(ValueError, match="F7"):
        model.eligible_universe(dataset)


def test_a_supplied_universe_is_intersected_with_the_matrix(dataset, candidate):
    subset = dataset.matrix.gene_ids[:1000] + ("NOT_IN_MATRIX",)
    model = RandomGeneSetNull(spec=UNMATCHED, universe=subset, enforce_draw_floor=False)
    universe = model.eligible_universe(dataset)
    assert universe == dataset.matrix.gene_ids[:1000]


def test_zero_variance_genes_are_not_eligible_background(dataset):
    frame = dataset.matrix.to_frame()
    frame.iloc[:4, :] = 1.0
    from conftest import make_matrix

    aligned = make_matrix(frame).align_to(dataset.outcome)
    model = RandomGeneSetNull(spec=UNMATCHED, enforce_draw_floor=False)
    universe = model.eligible_universe(aligned)
    assert set(aligned.matrix.gene_ids[:4]).isdisjoint(universe)


def test_a_background_that_cannot_match_refuses_rather_than_degrading(dataset):
    """F4: no silent fallback to unmatched sampling."""
    tiny = dataset.matrix.gene_ids[:60]
    candidate = make_signature(tiny[:50])
    with pytest.raises(BackgroundExhaustedError):
        build_matched_background(
            gene_stats=dataset.matrix.gene_stats(),
            universe=tiny,
            candidate_genes=candidate.genes,
            matching=default_matching_spec(60),
        )


def test_the_draw_floor_is_enforced_by_default(dataset, candidate):
    model = RandomGeneSetNull(spec=matched_spec(3000))
    with pytest.raises(DrawFloorError, match="MIN_DRAWS"):
        list(model.draw(candidate, dataset, 999, generator_for(1, SeedStream.GENE_SET)))


def test_the_default_spec_asks_for_ten_thousand_draws():
    assert NullSpec(null_type=NullType.RANDOM_GENE_SET).n_draws == 10_000
    assert NullSpec(null_type=NullType.RANDOM_GENE_SET).n_draws >= MIN_DRAWS


def test_an_unsupported_set_level_constraint_is_rejected_at_construction():
    spec = NullSpec(
        null_type=NullType.RANDOM_GENE_SET,
        matching=MatchingSpec(set_level_constraints={"mean_abs_covariance": 0.1}),
    )
    with pytest.raises(ValueError, match="unsupported set-level constraint"):
        RandomGeneSetNull(spec=spec)


def test_a_coherence_constraint_holds_on_every_accepted_draw():
    dataset = make_dataset(n_genes=2000, n_samples=60, dominant_axis=1.5, seed=5)
    stats = dataset.matrix.gene_stats().table
    candidate = make_signature(list(dataset.matrix.gene_ids[:15]))
    spec = NullSpec(
        null_type=NullType.RANDOM_GENE_SET,
        matching=MatchingSpec(
            set_level_constraints={"mean_abs_correlation": 0.05},
            max_resample_attempts=200,
        ),
    )
    model = RandomGeneSetNull(spec=spec, enforce_draw_floor=False)

    values = dataset.matrix.values
    index = {g: i for i, g in enumerate(dataset.matrix.gene_ids)}

    def coherence(genes):
        rows = [index[g] for g in genes]
        corr = np.corrcoef(values[rows, :])
        return float(np.abs(corr[np.triu_indices(len(rows), k=1)]).mean())

    target = coherence(candidate.genes)
    for draw in draws(model, candidate, dataset, 10):
        assert abs(coherence(draw.signature.genes) - target) <= 0.05


# ---------------------------------------------------------------------------
# Label permutation null
# ---------------------------------------------------------------------------


def test_permutation_preserves_class_sizes_and_flags_itself(dataset, candidate):
    model = LabelPermutationNull(enforce_draw_floor=False)
    for draw in model.draw(candidate, dataset, 20, generator_for(4, SeedStream.PERMUTATION)):
        assert draw.outcome.n_positive == dataset.outcome.n_positive
        assert draw.outcome.is_permuted is True
        assert draw.outcome.permutation_seed is not None
        assert draw.signature is candidate


def test_permutation_actually_moves_the_labels(dataset, candidate):
    model = LabelPermutationNull(enforce_draw_floor=False)
    observed = np.asarray(dataset.outcome.labels)
    identical = sum(
        np.array_equal(np.asarray(d.outcome.labels), observed)
        for d in model.draw(candidate, dataset, 50, generator_for(4, SeedStream.PERMUTATION))
    )
    assert identical == 0


def test_permutation_draws_are_reproducible(dataset, candidate):
    model = LabelPermutationNull(enforce_draw_floor=False)

    def labels(seed):
        return [
            np.asarray(d.outcome.labels).tobytes()
            for d in model.draw(candidate, dataset, 8, generator_for(seed, SeedStream.PERMUTATION))
        ]

    assert labels(9) == labels(9)
    assert labels(9) != labels(10)


def test_the_permutation_null_draws_no_genes(dataset):
    assert LabelPermutationNull(enforce_draw_floor=False).eligible_universe(dataset) == ()


# ---------------------------------------------------------------------------
# Registry and seeding
# ---------------------------------------------------------------------------


def test_the_registry_resolves_both_nulls_from_their_enum_value():
    assert isinstance(get("random_gene_set"), RandomGeneSetNull)
    assert isinstance(get(NullType.LABEL_PERMUTATION), LabelPermutationNull)


def test_a_model_rejects_a_spec_for_the_other_null():
    with pytest.raises(ValueError, match="RANDOM_GENE_SET"):
        RandomGeneSetNull(spec=NullSpec(null_type=NullType.LABEL_PERMUTATION))


def test_seed_streams_are_independent():
    a = generator_for(42, SeedStream.GENE_SET).integers(0, 2**31, size=5)
    b = generator_for(42, SeedStream.PERMUTATION).integers(0, 2**31, size=5)
    assert not np.array_equal(a, b)


def test_per_draw_seeds_are_distinct_and_reproducible():
    first = draw_seeds(generator_for(1, SeedStream.GENE_SET), 500)
    second = draw_seeds(generator_for(1, SeedStream.GENE_SET), 500)
    np.testing.assert_array_equal(first, second)
    assert len(set(first.tolist())) == 500


# ---------------------------------------------------------------------------
# Empirical p-value
# ---------------------------------------------------------------------------


def test_the_p_value_can_never_be_zero():
    # SigCheck's $checkPval returns exactly 0 when no random signature does as
    # well; the add-one estimator cannot, by construction.
    result = empirical_p_value(10.0, np.zeros(9999))
    assert result.p_value == pytest.approx(1.0 / 10_000)
    assert result.p_value > 0.0
    assert result.at_resolution_floor is True


def test_the_p_value_is_the_add_one_estimator():
    nulls = np.arange(100, dtype=float)
    result = empirical_p_value(90.0, nulls, enforce_min_draws=False)  # 10 draws >= 90
    assert result.n_exceedances == 10
    assert result.p_value == pytest.approx(11 / 101)


def test_the_floor_matches_the_number_of_valid_draws():
    assert p_value_floor(999) == pytest.approx(0.001)
    assert empirical_p_value(
        1.0, np.zeros(999), enforce_min_draws=False
    ).floor == pytest.approx(0.001)


def test_a_uniform_observation_gives_a_uniform_p_value():
    rng = np.random.default_rng(0)
    nulls = rng.normal(size=5000)
    p_values = [
        empirical_p_value(float(rng.normal()), nulls).p_value for _ in range(400)
    ]
    assert 0.02 <= float(np.mean(np.asarray(p_values) <= 0.05)) <= 0.12


def test_the_less_alternative_uses_the_other_tail():
    nulls = np.arange(100, dtype=float)
    result = empirical_p_value(
        10.0, nulls, alternative=Alternative.LESS, enforce_min_draws=False
    )
    assert result.n_exceedances == 11  # 0..10 inclusive
