"""Campaign 4: loaders, resolution, alignment and the eligible background."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_frame, make_matrix, make_outcome, make_signature
from signull.data import (
    AnnotationTable,
    BackgroundTooSmallError,
    DenseExpressionMatrix,
    DiagnosticCode,
    DiagnosticLog,
    MatrixIndexResolver,
    SignullDataWarning,
    check_background_floors,
    eligible_background,
    load_matrix_tsv,
    resolve_signature,
    signature_from_ids,
    strip_version_suffix,
)
from signull.types import (
    DuplicateHandling,
    GeneIdNamespace,
    MissingGenePolicy,
    REQUIRED_GENE_STAT_COLUMNS,
    ResolutionSpec,
)


# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------


def test_gene_stats_has_the_contract_columns_and_matches_numpy(matrix):
    stats = matrix.gene_stats()
    assert list(stats.table.columns) == list(REQUIRED_GENE_STAT_COLUMNS)
    values = matrix.values
    np.testing.assert_allclose(stats.table["mean"].to_numpy(), values.mean(axis=1))
    np.testing.assert_allclose(
        stats.table["sd"].to_numpy(), values.std(axis=1, ddof=1)
    )
    assert stats.n_samples == matrix.n_samples


def test_gene_stats_are_recomputed_after_subsetting_samples(matrix):
    half = matrix.sample_ids[: matrix.n_samples // 2]
    subset = matrix.subset_samples(half)
    assert subset.gene_stats().n_samples == len(half)
    assert not np.allclose(
        subset.gene_stats().table["mean"].to_numpy(),
        matrix.gene_stats().table["mean"].to_numpy(),
    )


def test_align_to_intersects_and_orders_by_matrix_columns(matrix):
    keep = list(matrix.sample_ids[5:])[::-1]  # reversed, and missing the first five
    outcome = make_outcome(keep)
    aligned = matrix.align_to(outcome)
    assert aligned.matrix.sample_ids == aligned.outcome.sample_ids
    assert aligned.matrix.sample_ids == tuple(matrix.sample_ids[5:])
    assert set(aligned.dropped_samples) == set(matrix.sample_ids[:5])


def test_matrix_rejects_non_finite_values():
    frame = make_frame(n_genes=50, n_samples=30)
    frame.iloc[3, 4] = np.nan
    with pytest.raises(ValueError, match="finite"):
        make_matrix(frame)


def test_load_matrix_tsv_roundtrips_and_records_a_checksum(tmp_path):
    frame = make_frame(n_genes=100, n_samples=30)
    path = tmp_path / "matrix.tsv"
    frame.to_csv(path, sep="\t")
    loaded = load_matrix_tsv(path, units="log2 RMA", dataset_id="round_trip")
    np.testing.assert_allclose(loaded.values, frame.to_numpy())
    assert loaded.gene_ids == tuple(frame.index)
    assert loaded.descriptor.provenance.checksum is not None


def test_load_matrix_tsv_transposes_a_declared_samples_x_genes_file(tmp_path):
    frame = make_frame(n_genes=40, n_samples=30)
    path = tmp_path / "t.tsv"
    frame.T.to_csv(path, sep="\t")
    loaded = load_matrix_tsv(path, units="log2 RMA", orientation="samples_x_genes")
    assert loaded.shape == (40, 30)
    np.testing.assert_allclose(loaded.values, frame.to_numpy())


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_perfect_resolution_reports_everything_matched(matrix):
    genes = list(matrix.gene_ids[:20])
    resolved, resolution = resolve_signature(make_signature(genes), matrix)
    assert resolved.genes == tuple(genes)
    assert resolution.n_matched == 20
    assert resolution.overlap_fraction == 1.0
    assert resolution.missing == () and resolution.unmapped == ()


def test_matched_genes_come_back_in_matrix_row_order(matrix):
    genes = [matrix.gene_ids[9], matrix.gene_ids[2], matrix.gene_ids[5]]
    resolved, _ = resolve_signature(make_signature(genes), matrix)
    assert resolved.genes == (matrix.gene_ids[2], matrix.gene_ids[5], matrix.gene_ids[9])


def test_weights_follow_their_genes_through_reordering(matrix):
    genes = [matrix.gene_ids[9], matrix.gene_ids[2], matrix.gene_ids[5]]
    signature = make_signature(genes, weights=[1.0, -2.0, 3.0])
    resolved, _ = resolve_signature(signature, matrix)
    assert resolved.weights == (-2.0, 3.0, 1.0)


def test_missing_genes_are_dropped_and_warned_about(matrix):
    genes = list(matrix.gene_ids[:18]) + ["NOT_A_GENE_1", "NOT_A_GENE_2"]
    with pytest.warns(SignullDataWarning, match="low_signature_overlap"):
        resolved, resolution = resolve_signature(make_signature(genes), matrix)
    assert resolution.n_matched == 18
    assert set(resolution.missing) == {"NOT_A_GENE_1", "NOT_A_GENE_2"}
    assert resolution.n_matched == resolved.size
    assert resolution.overlap_fraction == pytest.approx(0.9)


def test_overlap_below_the_floor_raises_rather_than_shrinking_the_signature(matrix):
    genes = list(matrix.gene_ids[:6]) + [f"ABSENT_{i}" for i in range(14)]
    # The floor is tier 1: it raises before the low-overlap warning is reached,
    # so a 30 % signature never becomes a quietly smaller analysis.
    with pytest.raises(ValueError, match="F5"):
        resolve_signature(make_signature(genes), matrix)


def test_missing_gene_policy_raise_refuses_any_loss(matrix):
    genes = list(matrix.gene_ids[:19]) + ["ABSENT"]
    spec = ResolutionSpec(missing_gene_policy=MissingGenePolicy.RAISE)
    with pytest.raises(ValueError, match="missing_gene_policy is RAISE"):
        resolve_signature(make_signature(genes), matrix, spec)


def test_case_insensitive_matching_is_on_by_default(matrix):
    genes = [g.lower() for g in matrix.gene_ids[:10]]
    resolved, resolution = resolve_signature(make_signature(genes), matrix)
    assert resolution.n_matched == 10
    assert resolved.genes == tuple(matrix.gene_ids[:10])


def test_case_sensitive_spec_reports_the_same_genes_as_missing(matrix):
    genes = [g.lower() for g in matrix.gene_ids[:10]]
    spec = ResolutionSpec(case_insensitive=False, min_overlap_fraction=0.0, min_matched_genes=0)
    with pytest.warns(SignullDataWarning):
        _, resolution = resolve_signature(make_signature(genes), matrix, spec)
    assert resolution.n_matched == 0


def test_annotation_table_bridges_namespaces_and_records_aliases(matrix):
    targets = matrix.gene_ids[:12]
    annotation = AnnotationTable.from_pairs(
        [(f"probe_{i}", gene) for i, gene in enumerate(targets)],
        source_namespace=GeneIdNamespace.PROBE_ID,
        version="test-annotation-1",
    )
    signature = signature_from_ids(
        [f"probe_{i}" for i in range(12)],
        name="probes",
        namespace=GeneIdNamespace.PROBE_ID,
    )
    resolved, resolution = resolve_signature(signature, matrix, annotation=annotation)
    assert resolved.genes == tuple(targets)
    assert resolution.aliased["probe_3"] == targets[3]
    assert resolution.mapping_source == "test-annotation-1"


def test_ambiguous_annotation_entries_are_dropped_not_guessed(matrix):
    annotation = AnnotationTable.from_pairs(
        [("probe_x", matrix.gene_ids[0]), ("probe_x", matrix.gene_ids[1])]
        + [(f"probe_{i}", matrix.gene_ids[i]) for i in range(9)],
        version="v1",
    )
    assert annotation.ambiguous["probe_x"] == tuple(sorted(matrix.gene_ids[:2]))
    signature = signature_from_ids(
        ["probe_x"] + [f"probe_{i}" for i in range(9)],
        name="probes",
        namespace=GeneIdNamespace.PROBE_ID,
    )
    with pytest.warns(SignullDataWarning):
        _, resolution = resolve_signature(signature, matrix, annotation=annotation)
    assert resolution.unmapped == ("probe_x",)
    assert resolution.missing == ()


def test_unmapped_and_missing_are_tracked_separately(matrix):
    annotation = AnnotationTable.from_pairs(
        [(f"probe_{i}", matrix.gene_ids[i]) for i in range(10)]
        + [("probe_dead", "GENE_NOT_ON_ARRAY")],
        version="v1",
    )
    signature = signature_from_ids(
        [f"probe_{i}" for i in range(10)] + ["probe_dead", "probe_unknown"],
        name="probes",
        namespace=GeneIdNamespace.PROBE_ID,
    )
    with pytest.warns(SignullDataWarning):
        _, resolution = resolve_signature(signature, matrix, annotation=annotation)
    # translatable but not on this platform vs. not translatable at all
    assert resolution.missing == ("probe_dead",)
    assert resolution.unmapped == ("probe_unknown",)


def test_two_identifiers_hitting_one_row_collapse_and_are_recorded(matrix):
    annotation = AnnotationTable.from_pairs(
        [("probe_a", matrix.gene_ids[0]), ("probe_b", matrix.gene_ids[0])]
        + [(f"probe_{i}", matrix.gene_ids[i]) for i in range(1, 10)],
        version="v1",
    )
    signature = signature_from_ids(
        ["probe_a", "probe_b"] + [f"probe_{i}" for i in range(1, 10)],
        name="probes",
        namespace=GeneIdNamespace.PROBE_ID,
    )
    with pytest.warns(SignullDataWarning, match="duplicate_probes_collapsed"):
        resolved, resolution = resolve_signature(signature, matrix, annotation=annotation)
    assert resolved.size == 10
    assert resolution.collapsed[matrix.gene_ids[0]] == ("probe_a", "probe_b")


def test_duplicate_handling_raise_refuses_the_collapse(matrix):
    annotation = AnnotationTable.from_pairs(
        [("probe_a", matrix.gene_ids[0]), ("probe_b", matrix.gene_ids[0])]
        + [(f"probe_{i}", matrix.gene_ids[i]) for i in range(1, 10)],
        version="v1",
    )
    signature = signature_from_ids(
        ["probe_a", "probe_b"] + [f"probe_{i}" for i in range(1, 10)],
        name="probes",
        namespace=GeneIdNamespace.PROBE_ID,
    )
    spec = ResolutionSpec(duplicate_handling=DuplicateHandling.RAISE)
    with pytest.raises(ValueError, match="duplicate_handling is RAISE"):
        resolve_signature(signature, matrix, spec, annotation=annotation)


def test_version_suffixes_are_stripped_before_mapping():
    assert strip_version_suffix("ENSG00000141510.14") == "ENSG00000141510"
    assert strip_version_suffix("ENSG00000141510") == "ENSG00000141510"
    assert strip_version_suffix("TP53") == "TP53"


def test_constant_genes_in_a_signature_are_flagged():
    frame = make_frame(n_genes=100, n_samples=40)
    frame.iloc[0, :] = 5.0
    matrix = make_matrix(frame)
    log = DiagnosticLog()
    with pytest.warns(SignullDataWarning, match="zero_variance"):
        resolve_signature(
            make_signature(list(matrix.gene_ids[:10])), matrix, log=log
        )
    assert log.has(DiagnosticCode.ZERO_VARIANCE_GENES_IN_SIGNATURE)


# ---------------------------------------------------------------------------
# Eligible background
# ---------------------------------------------------------------------------


def test_background_drops_constant_genes(dataset):
    frame = dataset.matrix.to_frame()
    frame.iloc[:5, :] = 3.0
    matrix = make_matrix(frame)
    aligned = matrix.align_to(dataset.outcome)
    background = eligible_background(aligned)
    assert background.n_excluded("zero_variance") == 5
    assert set(matrix.gene_ids[:5]).isdisjoint(background.genes)


def test_background_drops_affymetrix_control_probes(dataset):
    frame = dataset.matrix.to_frame()
    frame.index = ["AFFX-BioB-5_at"] + list(frame.index[1:])
    matrix = make_matrix(frame)
    background = eligible_background(matrix.align_to(dataset.outcome))
    assert background.n_excluded("control_probe") == 1


def test_background_restricts_to_the_declared_platform(dataset):
    allowed = dataset.matrix.gene_ids[:500]
    background = eligible_background(dataset, platform_features=allowed)
    assert background.size == 500
    assert background.filters["platform_restricted"] is True


def test_background_can_exclude_the_candidate(dataset, candidate):
    kept = eligible_background(dataset, candidate=candidate)
    dropped = eligible_background(
        dataset, candidate=candidate, exclude_candidate_genes=True
    )
    assert kept.size - dropped.size == candidate.size
    assert set(candidate.genes).isdisjoint(dropped.genes)


def test_background_is_in_matrix_row_order(dataset):
    background = eligible_background(dataset)
    positions = [dataset.matrix.gene_ids.index(g) for g in background.genes]
    assert positions == sorted(positions)


def test_background_floors_refuse_a_small_universe(dataset):
    background = eligible_background(dataset, platform_features=dataset.matrix.gene_ids[:1500])
    with pytest.raises(BackgroundTooSmallError, match="F4"):
        check_background_floors(background, 30)


def test_background_floors_scale_with_the_candidate(dataset):
    background = eligible_background(dataset)  # 3000 genes
    check_background_floors(background, 100)  # 20 * 100 = 2000 <= 3000
    with pytest.raises(BackgroundTooSmallError):
        check_background_floors(background, 200)  # 20 * 200 = 4000 > 3000
