"""A self-contained tour of what signull does, on a synthetic cohort.

Run it::

    .venv/bin/python examples/demo.py

No downloads, no external data, fixed seeds: the numbers below are reproducible
byte for byte.  The cohort is built to contain the confound the tool exists to
expose -- one dominant latent axis that most genes load on, and an outcome that
is genuinely associated with it -- and then three candidate signatures are tested
against it:

``AXIS_50``
    50 genes taken from those loading on the dominant axis -- the synthetic
    stand-in for a published proliferation-flavoured signature.  It looks
    excellent by every conventional check: a high AUROC and a permutation
    p-value at the resolution floor.
``RANDOM_50``
    50 genes drawn uniformly at random.  A negative control that should embarrass
    the tool if it ever comes out significant against the matched null.
``CAUSAL_25``
    25 genes that actually drive the outcome through a factor of their own.
    The positive control.

Each is put through all three nulls.  The point of the demo is the *pattern* of
the three p-values, not any one of them.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from signull.data import (
    DenseExpressionMatrix,
    check_background_floors,
    eligible_background,
    resolve_signature,
)
from signull.metrics import AurocMetric, DirectedMetric
from signull.nulls import (
    DrawFloorError,
    LabelPermutationNull,
    RandomGeneSetNull,
    SeedStream,
    default_matching_spec,
    empirical_p_value,
    generator_for,
)
from signull.scoring import MeanZScoreScorer
from signull.types import (
    BinaryOutcome,
    DirectionPolicy,
    MatchingSpec,
    NullSpec,
    NullType,
    ResolutionSpec,
    Signature,
    SignatureOrigin,
)

N_GENES = 5000
N_SAMPLES = 150
N_DRAWS = 2_000  # the hard floor; the design's default is 10 000
SEED = 20260901


# ---------------------------------------------------------------------------
# A cohort with a dominant axis, the way real expression cohorts come
# ---------------------------------------------------------------------------


def build_cohort() -> tuple[DenseExpressionMatrix, BinaryOutcome, list[str], list[str]]:
    """Genes x samples matrix, a binary outcome, the causal genes and axis genes.

    Structure, in decreasing order of importance:

    * a **dominant axis** (think proliferation) that 60 % of genes load on, with
      loadings increasing with expression level -- this is what makes almost any
      large random gene set look predictive;
    * 20 minor shared factors, so the correlation structure is not rank-one;
    * a **private causal factor** carried by 25 genes only;
    * dropout to zero in low-expression genes, so detection rate varies and the
      third matching level has something to match on.

    The outcome is driven by both the dominant axis and the private factor, so a
    signature can be predictive for either the interesting reason or the boring
    one.  Telling those two apart is the entire job.
    """
    rng = np.random.default_rng(SEED)
    gene_ids = [f"GENE{i:05d}" for i in range(N_GENES)]
    sample_ids = [f"PT{j:03d}" for j in range(N_SAMPLES)]

    means = rng.uniform(3.0, 13.0, size=N_GENES)
    sds = 0.10 + 0.10 * (means - means.min())
    values = rng.normal(means[:, None], sds[:, None], size=(N_GENES, N_SAMPLES))

    dominant = rng.normal(size=N_SAMPLES)
    loads_on_axis = rng.random(N_GENES) < 0.6
    loadings = np.where(loads_on_axis, 0.25 * sds * rng.uniform(2.0, 6.0, N_GENES), 0.0)
    values += loadings[:, None] * dominant[None, :]

    for _ in range(20):
        factor = rng.normal(size=N_SAMPLES)
        members = rng.random(N_GENES) < 0.05
        values[members] += 0.4 * sds[members, None] * factor[None, :]

    # Detection: low-expression genes drop out in some samples, the way real
    # assays behave.  Without this the detection-rate dimension is constant and
    # the matcher correctly collapses it -- with it, all three levels of the
    # nested binning carry information.
    dropout = 0.7 * np.exp(-(means - means.min()) / 2.5)
    dropped = rng.random((N_GENES, N_SAMPLES)) < dropout[:, None]
    values = np.where(dropped, 0.0, values)

    # A "published signature": 50 genes sampled from the strong loaders on the
    # dominant axis.  Nothing about them is special among strong loaders, which
    # is precisely the point.
    strong = np.flatnonzero(loadings >= np.quantile(loadings[loads_on_axis], 0.60))
    axis_genes = [gene_ids[i] for i in rng.choice(strong, size=50, replace=False)]

    private = rng.normal(size=N_SAMPLES)
    causal = list(range(N_GENES - 25, N_GENES))
    values[causal] += 1.2 * sds[causal, None] * private[None, :]

    matrix = DenseExpressionMatrix.from_frame(
        pd.DataFrame(values, index=gene_ids, columns=sample_ids),
        dataset_id="demo_cohort",
        units="log2 arbitrary",
    )

    risk = 1.0 * dominant + 1.0 * private + rng.normal(scale=1.0, size=N_SAMPLES)
    labels = np.zeros(N_SAMPLES, dtype=bool)
    labels[np.argsort(-risk)[: int(0.3 * N_SAMPLES)]] = True  # 30 % prevalence
    labels.setflags(write=False)
    outcome = BinaryOutcome(
        sample_ids=tuple(sample_ids),
        labels=labels,
        name="relapse",
        positive_label="relapse",
        negative_label="no_relapse",
    )
    return matrix, outcome, [gene_ids[i] for i in causal], axis_genes


def candidate_signatures(
    matrix: DenseExpressionMatrix, causal: list[str], axis_genes: list[str]
) -> list[Signature]:
    """The three candidates, in increasing order of how much they deserve to pass."""
    uniform = list(
        np.random.default_rng(SEED + 1).choice(matrix.gene_ids, size=50, replace=False)
    )
    return [
        Signature(genes=tuple(axis_genes), name="AXIS_50", origin=SignatureOrigin.CANDIDATE),
        Signature(genes=tuple(uniform), name="RANDOM_50", origin=SignatureOrigin.CANDIDATE),
        Signature(genes=tuple(causal), name="CAUSAL_25", origin=SignatureOrigin.CANDIDATE),
    ]


# ---------------------------------------------------------------------------
# The test itself
# ---------------------------------------------------------------------------


def run_null(model, candidate, dataset, scorer, metric, observed, stream):
    """Score every draw of one null with the *same* scorer and metric, then test."""
    rng = np.random.default_rng(0)  # unused by the deterministic scorers
    statistics = np.fromiter(
        (
            metric(scorer.score(dataset, draw.signature, rng), draw.outcome)
            for draw in model.draw(candidate, dataset, N_DRAWS, generator_for(SEED, stream))
        ),
        dtype=np.float64,
        count=N_DRAWS,
    )
    return empirical_p_value(observed, statistics), statistics


def main() -> None:
    warnings.simplefilter("ignore")  # the demo prints its own diagnostics
    started = time.perf_counter()

    matrix, outcome, causal, axis_genes = build_cohort()
    dataset = matrix.align_to(outcome)
    print(
        f"cohort: {dataset.matrix.n_genes} genes x {dataset.n_samples} samples, "
        f"{dataset.outcome.n_positive} positive "
        f"(prevalence {dataset.outcome.prevalence:.2f})"
    )

    scorer = MeanZScoreScorer()
    metric = DirectedMetric(AurocMetric(), DirectionPolicy.SYMMETRIZED)
    stats_table = dataset.matrix.gene_stats().table

    whole_background = eligible_background(dataset)
    rows = []
    for signature in candidate_signatures(dataset.matrix, causal, axis_genes):
        candidate, resolution = resolve_signature(
            signature, dataset.matrix, ResolutionSpec()
        )
        background = eligible_background(dataset, candidate=candidate)
        check_background_floors(background, resolution.n_matched)

        observed = metric(
            scorer.score(dataset, candidate, np.random.default_rng(0)), dataset.outcome
        )

        uniform_null = RandomGeneSetNull(
            spec=NullSpec(null_type=NullType.RANDOM_GENE_SET, matching=MatchingSpec()),
            universe=background.genes,
        )
        matched_null = RandomGeneSetNull(
            spec=NullSpec(
                null_type=NullType.RANDOM_GENE_SET,
                matching=default_matching_spec(background.size),
            ),
            universe=background.genes,
        )
        permutation_null = LabelPermutationNull()

        p0, s0 = run_null(
            uniform_null, candidate, dataset, scorer, metric, observed, SeedStream.GENE_SET
        )
        p1, s1 = run_null(
            matched_null, candidate, dataset, scorer, metric, observed, SeedStream.GENE_SET
        )
        p2, _ = run_null(
            permutation_null,
            candidate,
            dataset,
            scorer,
            metric,
            observed,
            SeedStream.PERMUTATION,
        )

        rows.append(
            {
                "signature": candidate.name,
                "m": resolution.n_matched,
                "AUROC": observed,
                "p_N0_uniform": p0.p_value,
                "p_N1_matched": p1.p_value,
                "p_N2_permutation": p2.p_value,
                "N0_median": float(np.median(s0)),
                "N1_median": float(np.median(s1)),
                "N0_q95": float(np.quantile(s0, 0.95)),
                "N1_q95": float(np.quantile(s1, 0.95)),
                "cand_mean_expr": float(stats_table.loc[list(candidate.genes), "mean"].mean()),
            }
        )

    report(rows, whole_background)
    show_matching(dataset, stats_table, axis_genes)
    show_refusals(dataset)
    print(f"\ntotal runtime {time.perf_counter() - started:.1f} s "
          f"({3 * 3 * N_DRAWS} draws scored)")


def report(rows, background) -> None:
    """The three p-values per signature, side by side.  No verdict column."""
    print(f"eligible background |B| = {background.size} genes, K = {N_DRAWS} draws per null\n")
    header = f"{'signature':<18}{'m':>4}{'AUROC':>8}{'p N0':>9}{'p N1':>9}{'p N2':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['signature']:<18}{r['m']:>4}{r['AUROC']:>8.3f}"
            f"{r['p_N0_uniform']:>9.4f}{r['p_N1_matched']:>9.4f}{r['p_N2_permutation']:>9.4f}"
        )
    print("\nnull distributions (symmetrized AUROC):")
    print(f"{'signature':<18}{'N0 median':>11}{'N0 95th':>10}{'N1 median':>11}{'N1 95th':>10}")
    for r in rows:
        print(
            f"{r['signature']:<18}{r['N0_median']:>11.3f}{r['N0_q95']:>10.3f}"
            f"{r['N1_median']:>11.3f}{r['N1_q95']:>10.3f}"
        )
    axis, random_set, causal = rows
    print(
        "\nRead the pattern, not any single number.\n"
        f"  AXIS_50    AUROC {axis['AUROC']:.3f}, permutation p {axis['p_N2_permutation']:.4f}: by the\n"
        "             conventional check this signature is a hit. The uniform null agrees\n"
        f"             (p {axis['p_N0_uniform']:.4f}). The matched null does not (p {axis['p_N1_matched']:.4f}): among gene\n"
        "             sets with the same expression and variance profile, it is ordinary.\n"
        f"  RANDOM_50  permutation p {random_set['p_N2_permutation']:.4f} on 50 genes picked out of a hat. This is the\n"
        "             Venet effect: with a dominant axis in the data, self-contained tests\n"
        "             call random gene sets significant.\n"
        f"  CAUSAL_25  small against all three nulls (p_N1 {causal['p_N1_matched']:.4f}). This is what a signature\n"
        "             that carries its own signal looks like."
    )


def show_matching(dataset, stats_table, axis_genes) -> None:
    """What the matched sampler actually matches, measured on its own draws."""
    print("\n" + "=" * 72)
    print("what the property matching does, measured on 200 draws")
    print("=" * 72)
    candidate = Signature(genes=tuple(axis_genes), name="AXIS_50")
    background = eligible_background(dataset, candidate=candidate)

    def marginals(model, n=200):
        means, variances = [], []
        for draw in model.draw(candidate, dataset, n, generator_for(SEED, SeedStream.GENE_SET)):
            block = stats_table.loc[list(draw.signature.genes)]
            means.append(float(block["mean"].mean()))
            variances.append(float(block["variance"].mean()))
        return float(np.mean(means)), float(np.mean(variances))

    uniform = RandomGeneSetNull(
        spec=NullSpec(null_type=NullType.RANDOM_GENE_SET, matching=MatchingSpec()),
        universe=background.genes,
        enforce_draw_floor=False,
    )
    matched = RandomGeneSetNull(
        spec=NullSpec(
            null_type=NullType.RANDOM_GENE_SET,
            matching=default_matching_spec(background.size),
        ),
        universe=background.genes,
        enforce_draw_floor=False,
    )
    candidate_block = stats_table.loc[list(candidate.genes)]
    rows = [
        ("candidate", float(candidate_block["mean"].mean()), float(candidate_block["variance"].mean())),
        ("uniform draws (N0)", *marginals(uniform)),
        ("matched draws (N1)", *marginals(matched)),
    ]
    print(f"{'':<22}{'mean expression':>17}{'mean variance':>16}")
    for name, mean, var in rows:
        print(f"{name:<22}{mean:>17.3f}{var:>16.3f}")
    print(
        "\nThe uniform null compares a high-expression, high-variance signature against\n"
        "low-expression, low-variance gene sets. Every bit of that gap is discrimination\n"
        "the candidate gets credited for free."
    )

    first = next(matched.draw(candidate, dataset, 1, generator_for(SEED, SeedStream.GENE_SET)))
    print("\nbuild diagnostics from the matched background:")
    for diagnostic in first.diagnostics:
        print(f"  [{diagnostic.severity.value}] {diagnostic.code}: {diagnostic.message[:200]}")


def show_refusals(dataset) -> None:
    """Four things the tool declines to do, each demonstrated live."""
    print("\n" + "=" * 72)
    print("refusals")
    print("=" * 72)

    print("1. a p-value can never be 0, however extreme the candidate:")
    result = empirical_p_value(99.0, np.zeros(N_DRAWS))
    print(f"   observed beats every one of {N_DRAWS} draws -> p = {result.p_value:.6f} "
          f"(floor {result.floor:.6f}, at_resolution_floor={result.at_resolution_floor})")

    print("2. too few draws to interpret:")
    try:
        model = RandomGeneSetNull(spec=NullSpec(null_type=NullType.RANDOM_GENE_SET))
        next(model.draw(
            Signature(genes=dataset.matrix.gene_ids[:20], name="x"),
            dataset, 999, generator_for(1, SeedStream.GENE_SET),
        ))
    except DrawFloorError as error:
        print(f"   DrawFloorError: {str(error).split('.')[0]}.")

    print("3. a signature that half misses the platform:")
    try:
        resolve_signature(
            Signature(
                genes=tuple(dataset.matrix.gene_ids[:10]) + tuple(f"ABSENT{i}" for i in range(10)),
                name="half_missing",
            ),
            dataset.matrix,
        )
    except ValueError as error:
        print(f"   ValueError: {str(error).split('.')[0]}.")

    print("4. a background too small to support a competitive null:")
    try:
        small = eligible_background(dataset, platform_features=dataset.matrix.gene_ids[:1200])
        check_background_floors(small, 50)
    except ValueError as error:
        print(f"   {type(error).__name__}: {str(error).split('.')[0]}.")


if __name__ == "__main__":
    main()
