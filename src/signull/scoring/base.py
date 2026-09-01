"""Shared helpers for the scoring strategies.

Everything here is ``y``-blind: nothing in this module may take an outcome
argument.  Invariant I1 of ``docs/statistical-design.md`` -- the scoring
function is ``score(X, S) -> R^N`` and never sees ``y`` on the default path.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..types import (
    AlignedDataset,
    FloatArray,
    GeneId,
    SampleScores,
    ScoringMethodName,
    Signature,
)

__all__ = [
    "ConstantGeneError",
    "EmptySignatureError",
    "row_standardise",
    "signature_block",
    "split_weights",
    "make_scores",
]


class ConstantGeneError(ValueError):
    """Raised when a signature gene has zero variance across the cohort.

    A constant gene has no z-score.  Sec. 2.1 requires ``sd_g > 0`` for every
    gene in the eligible background, so this reaching a scorer means the matrix
    was not filtered.  Sec. 7 (T7) requires the behaviour be *defined*, and
    raising is the defined behaviour unless the caller opts into dropping.
    """


class EmptySignatureError(ValueError):
    """Raised when a signature resolves to zero scorable genes."""


def row_standardise(
    values: FloatArray,
    *,
    gene_ids: Sequence[GeneId] | None = None,
    constant_gene_policy: str = "raise",
    ddof: int = 0,
) -> tuple[FloatArray, tuple[int, ...]]:
    """Z-score each row of a ``genes x samples`` block over all samples.

    Returns ``(z, kept_row_indices)``.  Standardisation is a per-gene
    operation, so standardising the signature block gives bit-identical rows to
    standardising the whole matrix and then subsetting -- the cheaper route is
    taken here.

    Parameters
    ----------
    constant_gene_policy:
        ``"raise"`` (default) or ``"drop"``.  Dropping is silent-by-design only
        in the sense that the kept-row indices are returned to the caller,
        which must record them.
    ddof:
        Delta degrees of freedom for the standard deviation.  ``0`` matches the
        population form used throughout the design document.
    """
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D genes x samples block, got {values.shape}")
    if values.shape[0] == 0:
        raise EmptySignatureError("signature block has no rows")
    if not np.all(np.isfinite(values)):
        raise ValueError("expression block contains non-finite values")

    mean = values.mean(axis=1, keepdims=True)
    sd = values.std(axis=1, ddof=ddof, keepdims=True)
    flat_sd = sd.ravel()
    constant = flat_sd <= 0.0
    if np.any(constant):
        names = (
            [str(gene_ids[i]) for i in np.flatnonzero(constant)[:10]]
            if gene_ids is not None
            else [str(i) for i in np.flatnonzero(constant)[:10]]
        )
        if constant_gene_policy == "raise":
            raise ConstantGeneError(
                f"{int(constant.sum())} signature gene(s) have zero variance "
                f"across the cohort and cannot be z-scored: {names}"
            )
        if constant_gene_policy != "drop":
            raise ValueError(
                f"unknown constant_gene_policy {constant_gene_policy!r}; "
                "expected 'raise' or 'drop'"
            )
        keep = np.flatnonzero(~constant)
        if keep.size == 0:
            raise EmptySignatureError(
                "every signature gene is constant across the cohort"
            )
        z = (values[keep] - mean[keep]) / sd[keep]
        return z, tuple(int(i) for i in keep)

    z = (values - mean) / sd
    return z, tuple(range(values.shape[0]))


def signature_block(
    dataset: AlignedDataset, signature: Signature
) -> tuple[FloatArray, tuple[GeneId, ...]]:
    """Extract the ``m x N`` expression block for ``signature``, in signature order.

    The signature must already be resolved against the matrix index (the
    ``data`` package owns resolution); an unknown identifier here is a
    programming error and surfaces as ``KeyError`` from
    :meth:`ExpressionMatrix.subset_genes`.
    """
    if signature.size == 0:
        raise EmptySignatureError(
            f"signature {signature.name!r} resolves to zero genes"
        )
    sub = dataset.matrix.subset_genes(list(signature.genes))
    return np.asarray(sub.values, dtype=np.float64), tuple(sub.gene_ids)


def split_weights(
    signature: Signature, kept: tuple[int, ...]
) -> tuple[FloatArray | None, FloatArray | None]:
    """Return ``(positive_weights, negative_weights)`` aligned to the kept rows.

    ``None`` for both when the signature is unsigned.  Magnitudes are kept:
    ``|w|`` is emphasis, ``sign(w)`` is direction.
    """
    if signature.weights is None:
        return None, None
    w = np.asarray(signature.weights, dtype=np.float64)[list(kept)]
    if not np.all(np.isfinite(w)):
        raise ValueError(f"signature {signature.name!r} has non-finite weights")
    pos = np.where(w > 0.0, w, 0.0)
    neg = np.where(w < 0.0, -w, 0.0)
    if pos.sum() == 0.0 and neg.sum() == 0.0:
        raise ValueError(
            f"signature {signature.name!r} has all-zero weights; no direction "
            "or emphasis is expressible"
        )
    return pos, neg


def make_scores(
    values: FloatArray,
    dataset: AlignedDataset,
    method: ScoringMethodName,
    *,
    is_out_of_fold: bool = False,
) -> SampleScores:
    """Wrap a raw score vector in a validated :class:`SampleScores`."""
    values = np.asarray(values, dtype=np.float64).ravel()
    n = dataset.n_samples
    if values.shape != (n,):
        raise ValueError(
            f"scorer produced {values.shape} values for {n} samples"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "scorer produced non-finite values; SampleScores.values must be finite"
        )
    return SampleScores(
        values=values,
        sample_ids=tuple(dataset.matrix.sample_ids),
        method=method,
        is_out_of_fold=is_out_of_fold,
    )
