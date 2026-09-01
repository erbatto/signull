"""Shared synthetic fixtures.

Every fixture is deterministic: seeds are literals, never drawn from the clock.
The cohorts are small enough for a fast suite but respect the floors the code
enforces (``N >= 30``, ``n1, n0 >= 8``, ``|B| >= 2000`` where a competitive null
is actually built).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signull.data import DenseExpressionMatrix, binary_outcome_from_frame
from signull.types import AlignedDataset, BinaryOutcome, Signature, SignatureOrigin


def make_frame(
    n_genes: int = 3000,
    n_samples: int = 60,
    *,
    seed: int = 7,
    dominant_axis: float = 0.0,
) -> pd.DataFrame:
    """Genes x samples log-like matrix with a realistic mean/variance spread.

    Gene means are spread over a wide range and the per-gene sd grows with the
    mean, which is what makes property matching non-trivial: a uniform draw has
    systematically different marginals from a high-expression signature.

    Parameters
    ----------
    dominant_axis:
        Loading of a single shared latent factor, mimicking the proliferation
        axis that dominates real expression cohorts.  ``0.0`` gives independent
        genes.
    """
    rng = np.random.default_rng(seed)
    means = rng.uniform(2.0, 12.0, size=n_genes)
    sds = 0.15 + 0.08 * (means - means.min())
    values = rng.normal(
        loc=means[:, None], scale=sds[:, None], size=(n_genes, n_samples)
    )
    if dominant_axis:
        factor = rng.normal(size=n_samples)
        loadings = rng.uniform(0.0, dominant_axis, size=n_genes)
        values = values + loadings[:, None] * factor[None, :]
    genes = [f"G{i:05d}" for i in range(n_genes)]
    samples = [f"S{j:03d}" for j in range(n_samples)]
    return pd.DataFrame(values, index=genes, columns=samples)


def make_matrix(frame: pd.DataFrame | None = None, **kwargs) -> DenseExpressionMatrix:
    """Wrap a frame (default :func:`make_frame`) as a matrix in log2 RMA units."""
    frame = make_frame(**kwargs) if frame is None else frame
    return DenseExpressionMatrix.from_frame(
        frame, dataset_id="synthetic", units="log2 RMA"
    )


def make_outcome(
    sample_ids, *, prevalence: float = 0.4, seed: int = 11, signal: np.ndarray | None = None
) -> BinaryOutcome:
    """Binary labels with a fixed positive count.

    ``signal`` assigns the positives to the highest-signal samples instead of at
    random, which is how the tests plant a detectable association.
    """
    sample_ids = list(sample_ids)
    n_positive = max(8, int(round(len(sample_ids) * prevalence)))
    labels = np.zeros(len(sample_ids), dtype=bool)
    if signal is None:
        rng = np.random.default_rng(seed)
        labels[rng.choice(len(sample_ids), size=n_positive, replace=False)] = True
    else:
        labels[np.argsort(-np.asarray(signal))[:n_positive]] = True
    labels.setflags(write=False)
    return BinaryOutcome(
        sample_ids=tuple(sample_ids),
        labels=labels,
        name="synthetic_outcome",
        positive_label="event",
        negative_label="no_event",
    )


def make_dataset(**kwargs) -> AlignedDataset:
    """A matrix and outcome already aligned, ready for scoring or drawing."""
    matrix = make_matrix(**kwargs)
    outcome = make_outcome(matrix.sample_ids)
    return matrix.align_to(outcome)


def make_signature(genes, *, name: str = "candidate", weights=None) -> Signature:
    """A candidate signature over the given identifiers."""
    return Signature(
        genes=tuple(genes),
        name=name,
        origin=SignatureOrigin.CANDIDATE,
        weights=None if weights is None else tuple(float(w) for w in weights),
    )


@pytest.fixture
def matrix() -> DenseExpressionMatrix:
    """3000 x 60 synthetic matrix."""
    return make_matrix()


@pytest.fixture
def dataset() -> AlignedDataset:
    """3000 x 60 aligned dataset with a 40 % prevalence outcome."""
    return make_dataset()


@pytest.fixture
def candidate(dataset) -> Signature:
    """A 30-gene candidate biased toward high-expression genes.

    Biased on purpose: the marginal-property advantage is exactly what the
    matched null must remove and the uniform null must not.
    """
    table = dataset.matrix.gene_stats().table
    top = table["mean"].to_numpy().argsort()[::-1][:300]
    genes = [dataset.matrix.gene_ids[i] for i in top[::10]]
    return make_signature(genes)
