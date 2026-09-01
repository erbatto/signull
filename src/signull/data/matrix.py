"""Dense, pandas-backed :class:`~signull.types.ExpressionMatrix` implementation.

Orientation
-----------
**Always genes x samples.**  ``values.shape == (n_genes, n_samples)``; rows are
genes, columns are samples.  The orientation is *declared*, not inferred: a
loader takes an explicit ``orientation`` argument and transposes at ingest if
told to, and :class:`DenseExpressionMatrix` then checks that the array shape
agrees with the two identifier tuples and with the descriptor.  A matrix with
fewer rows than columns is legal (a small test fixture) but suspicious for a
real cohort, so it raises a tier-2 diagnostic rather than passing silently.

Units
-----
Values are floats on a *log-like normalised* scale -- ``log2(TPM+1)``,
``log2(CPM+1)``, log2 MAS5/RMA intensities, VST/rlog.  They may be negative
(centred data) and must be finite.  Raw integer counts are not acceptable input:
mean/variance matching and z-score scoring both assume the log-like scale.  The
scale actually present is recorded verbatim in
:attr:`~signull.types.DatasetDescriptor.units` and travels with every result.

Gene statistics
---------------
:meth:`DenseExpressionMatrix.gene_stats` returns the columns required by
:data:`~signull.types.REQUIRED_GENE_STAT_COLUMNS` -- ``mean`` (matrix units),
``sd`` (matrix units, ``ddof=1``), ``variance`` (matrix units squared),
``detection_rate`` (dimensionless, in ``[0, 1]``) and ``median`` (matrix units)
-- computed over *this* matrix's samples and cached on the instance.  The cache
is per-instance and never shared: :meth:`subset_genes` and :meth:`align_to`
return new objects with an empty cache, so statistics can never be inherited
from a larger raw cohort (``docs/architecture.md`` Sec. 3 step 3).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Final, Literal

import numpy as np
import pandas as pd

from signull.types import (
    REQUIRED_GENE_STAT_COLUMNS,
    AlignedDataset,
    BinaryOutcome,
    DatasetDescriptor,
    GeneId,
    GeneIdNamespace,
    GeneStats,
    SampleId,
)

from .diagnostics import DiagnosticCode, DiagnosticLog

__all__ = [
    "Orientation",
    "GENES_X_SAMPLES",
    "SAMPLES_X_GENES",
    "DenseExpressionMatrix",
    "detection_dimension_is_degenerate",
    "emit_detection_degeneracy",
    "DETECTION_DEGENERACY_SD",
    "SMALL_COHORT_N",
    "SMALL_CLASS_N",
]

Orientation = Literal["genes_x_samples", "samples_x_genes"]

GENES_X_SAMPLES: Final[Orientation] = "genes_x_samples"
SAMPLES_X_GENES: Final[Orientation] = "samples_x_genes"

#: ``sd(detection_rate)`` below this makes the detection dimension carry no
#: information; ``docs/statistical-design.md`` Sec. 2.2 collapses ``K_d -> 1``.
DETECTION_DEGENERACY_SD: Final[float] = 0.01

#: A cohort with fewer than this many samples cannot support a p-value
#: (``docs/statistical-design.md`` Sec. 8 F3).  The refusal itself is a wave-3
#: decision; the data layer only flags it.
SMALL_COHORT_N: Final[int] = 30

#: Minimum size of the smaller outcome class before F3 applies.
SMALL_CLASS_N: Final[int] = 8


def detection_dimension_is_degenerate(
    detection_rate: np.ndarray,
    threshold: float = DETECTION_DEGENERACY_SD,
) -> bool:
    """``True`` when detection rate is effectively constant across genes.

    Parameters
    ----------
    detection_rate:
        Per-gene detection rates in ``[0, 1]``.
    threshold:
        Standard-deviation floor; ``docs/statistical-design.md`` Sec. 2.2 uses
        0.01.
    """
    if detection_rate.size < 2:
        return True
    return bool(np.std(detection_rate, ddof=1) < threshold)


class DenseExpressionMatrix:
    """A cohort expression matrix held densely in memory, genes x samples.

    Satisfies the :class:`~signull.types.ExpressionMatrix` Protocol.  Instances
    are immutable: :attr:`values` is handed out with the NumPy write flag
    cleared, and every transformation returns a new object.

    Parameters
    ----------
    frame:
        ``pandas.DataFrame`` with genes on the index and samples on the columns.
        Values must be numeric and finite; the frame is copied to ``float64``,
        so the caller's object is never mutated or aliased.
    descriptor:
        Identity of the cohort.  ``descriptor.n_genes`` / ``n_samples`` must
        agree with the frame shape, and ``descriptor.namespace`` declares the
        namespace of the index.
    collapse_map:
        When several source rows (e.g. probesets) were collapsed onto one
        canonical identifier at ingest, the map canonical -> sources.  Carried
        so :class:`~signull.data.resolve.MatrixIndexResolver` can fold it into a
        :class:`~signull.types.SignatureResolution`.
    log:
        Optional :class:`~signull.data.diagnostics.DiagnosticLog` for
        construction-time tier-2 conditions.

    Raises
    ------
    ValueError
        Non-2-D input, duplicate gene or sample identifiers, non-numeric or
        non-finite values, a shape that disagrees with the descriptor, or fewer
        than two samples (per-gene ``sd`` with ``ddof=1`` is undefined).
    """

    __slots__ = (
        "_values",
        "_gene_ids",
        "_sample_ids",
        "_descriptor",
        "_collapse_map",
        "_gene_index",
        "_sample_index",
        "_gene_stats_cache",
    )

    def __init__(
        self,
        frame: pd.DataFrame,
        descriptor: DatasetDescriptor,
        *,
        collapse_map: Mapping[GeneId, tuple[GeneId, ...]] | None = None,
        log: DiagnosticLog | None = None,
    ) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(
                f"expression matrix must be a pandas DataFrame, got {type(frame)!r}"
            )
        if frame.ndim != 2:
            raise ValueError(
                f"expression matrix must be 2-D (genes x samples), got ndim={frame.ndim}"
            )

        gene_ids = tuple(str(g) for g in frame.index)
        sample_ids = tuple(str(s) for s in frame.columns)

        _reject_duplicates(gene_ids, "gene")
        _reject_duplicates(sample_ids, "sample")

        non_numeric = [
            str(col)
            for col, dtype in zip(frame.columns, frame.dtypes, strict=True)
            if not pd.api.types.is_numeric_dtype(dtype)
        ]
        if non_numeric:
            raise ValueError(
                "expression values must be numeric; non-numeric columns: "
                f"{non_numeric[:10]!r}"
                f"{' (+%d more)' % (len(non_numeric) - 10) if len(non_numeric) > 10 else ''}"
            )

        values = np.ascontiguousarray(frame.to_numpy(dtype=np.float64, copy=True))
        if values.shape != (len(gene_ids), len(sample_ids)):
            raise ValueError(
                f"matrix shape {values.shape} disagrees with index "
                f"({len(gene_ids)} genes) x columns ({len(sample_ids)} samples)"
            )

        finite = np.isfinite(values)
        if not finite.all():
            bad_rows = np.flatnonzero(~finite.all(axis=1))
            n_bad_cells = int((~finite).sum())
            offenders = [gene_ids[i] for i in bad_rows[:10]]
            raise ValueError(
                f"expression values must be finite; found {n_bad_cells} non-finite "
                f"cell(s) across {bad_rows.size} gene(s), e.g. {offenders!r}. "
                "Missing values must be resolved at ingest, never mean-imputed "
                "silently (docs/statistical-design.md Sec. 8 F12)."
            )

        if len(gene_ids) == 0:
            raise ValueError("expression matrix has zero genes")
        if len(sample_ids) < 2:
            raise ValueError(
                f"expression matrix needs at least 2 samples to define per-gene "
                f"variance (ddof=1); got {len(sample_ids)}"
            )

        if descriptor.n_genes != len(gene_ids) or descriptor.n_samples != len(sample_ids):
            raise ValueError(
                "DatasetDescriptor disagrees with the matrix: descriptor says "
                f"{descriptor.n_genes} x {descriptor.n_samples}, matrix is "
                f"{len(gene_ids)} x {len(sample_ids)} (genes x samples)"
            )

        values.setflags(write=False)

        self._values = values
        self._gene_ids = gene_ids
        self._sample_ids = sample_ids
        self._descriptor = descriptor
        self._collapse_map: Mapping[GeneId, tuple[GeneId, ...]] = dict(collapse_map or {})
        self._gene_index = {g: i for i, g in enumerate(gene_ids)}
        self._sample_index = {s: i for i, s in enumerate(sample_ids)}
        self._gene_stats_cache: GeneStats | None = None

        if len(gene_ids) < len(sample_ids) and log is not None:
            log.warn(
                DiagnosticCode.MATRIX_ORIENTATION_SUSPICIOUS,
                f"matrix has fewer rows ({len(gene_ids)}) than columns "
                f"({len(sample_ids)}); expression matrices are genes x samples, so "
                "this may be transposed",
                context={"n_genes": len(gene_ids), "n_samples": len(sample_ids)},
            )

    # -- construction --------------------------------------------------------

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        dataset_id: str,
        units: str,
        namespace: GeneIdNamespace = GeneIdNamespace.HGNC_SYMBOL,
        orientation: Orientation = GENES_X_SAMPLES,
        preprocessing: tuple[str, ...] = (),
        detection_threshold: float = 0.0,
        provenance=None,
        collapse_map: Mapping[GeneId, tuple[GeneId, ...]] | None = None,
        log: DiagnosticLog | None = None,
    ) -> "DenseExpressionMatrix":
        """Build a matrix from a frame, transposing if ``orientation`` says so.

        Parameters
        ----------
        orientation:
            ``"genes_x_samples"`` (default) or ``"samples_x_genes"``.  The
            latter is transposed here, once, at ingest; nothing downstream ever
            transposes again.  Any other value raises ``ValueError`` -- the
            orientation is declared by the caller and never guessed from shape.
        detection_threshold:
            Expression value *strictly above* which a gene counts as detected in
            a sample, in matrix units.  Defines the ``detection_rate`` gene stat.
        """
        if orientation not in (GENES_X_SAMPLES, SAMPLES_X_GENES):
            raise ValueError(
                f"orientation must be {GENES_X_SAMPLES!r} or {SAMPLES_X_GENES!r}, "
                f"got {orientation!r}"
            )
        oriented = frame.T if orientation == SAMPLES_X_GENES else frame
        descriptor = DatasetDescriptor(
            dataset_id=dataset_id,
            n_genes=int(oriented.shape[0]),
            n_samples=int(oriented.shape[1]),
            namespace=namespace,
            units=units,
            preprocessing=preprocessing,
            detection_threshold=detection_threshold,
            provenance=provenance,
        )
        return cls(oriented, descriptor, collapse_map=collapse_map, log=log)

    # -- ExpressionMatrix Protocol -------------------------------------------

    @property
    def values(self) -> np.ndarray:
        """Expression values, shape ``(n_genes, n_samples)``, finite ``float64``.

        Returned read-only (``flags.writeable is False``) so a caller cannot
        mutate the matrix in place.  Units are
        :attr:`~signull.types.DatasetDescriptor.units`.
        """
        return self._values

    @property
    def gene_ids(self) -> tuple[GeneId, ...]:
        """Gene identifiers, unique, in row order."""
        return self._gene_ids

    @property
    def sample_ids(self) -> tuple[SampleId, ...]:
        """Sample identifiers, unique, in column order."""
        return self._sample_ids

    @property
    def namespace(self) -> GeneIdNamespace:
        """Namespace of :attr:`gene_ids`.

        Canonically :attr:`~signull.types.GeneIdNamespace.HGNC_SYMBOL`; the
        GPL96 benchmark cohorts are :attr:`~signull.types.GeneIdNamespace.PROBE_ID`
        because no pinned probe-to-symbol annotation ships with them.
        """
        return self._descriptor.namespace

    @property
    def descriptor(self) -> DatasetDescriptor:
        """Identity and provenance of this matrix, for config capture."""
        return self._descriptor

    @property
    def collapse_map(self) -> Mapping[GeneId, tuple[GeneId, ...]]:
        """Canonical identifier -> the >1 source identifiers collapsed onto it."""
        return self._collapse_map

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_genes, n_samples)``."""
        return self._values.shape  # type: ignore[return-value]

    @property
    def n_genes(self) -> int:
        """Number of gene rows."""
        return len(self._gene_ids)

    @property
    def n_samples(self) -> int:
        """Number of sample columns."""
        return len(self._sample_ids)

    def to_frame(self) -> pd.DataFrame:
        """A fresh, writable ``DataFrame`` copy, genes x samples.

        A copy on every call, so handing it to a caller cannot alias the
        matrix's internal buffer.
        """
        return pd.DataFrame(
            np.array(self._values, copy=True),
            index=pd.Index(self._gene_ids, name="gene_id"),
            columns=pd.Index(self._sample_ids, name="sample_id"),
        )

    def gene_stats(self) -> GeneStats:
        """Per-gene summary statistics over this matrix's samples, cached.

        Computed once per instance and memoised; :meth:`subset_genes` and
        :meth:`align_to` return new instances with an empty cache, so statistics
        are always recomputed when the sample set changes.

        Returns
        -------
        GeneStats
            ``table`` is indexed by :attr:`gene_ids` in row order with columns
            :data:`~signull.types.REQUIRED_GENE_STAT_COLUMNS`:

            ``mean``
                per-gene mean expression, matrix units.
            ``sd``
                per-gene standard deviation, matrix units, ``ddof=1``.
            ``variance``
                ``sd ** 2``, matrix units squared.
            ``detection_rate``
                fraction of samples in ``[0, 1]`` strictly above
                :attr:`~signull.types.DatasetDescriptor.detection_threshold`.
            ``median``
                per-gene median expression, matrix units.
        """
        cached = self._gene_stats_cache
        if cached is not None:
            return cached

        values = self._values
        threshold = float(self._descriptor.detection_threshold)
        mean = values.mean(axis=1)
        sd = values.std(axis=1, ddof=1)
        table = pd.DataFrame(
            {
                "mean": mean,
                "sd": sd,
                "variance": sd**2,
                "detection_rate": (values > threshold).mean(axis=1),
                "median": np.median(values, axis=1),
            },
            index=pd.Index(self._gene_ids, name="gene_id"),
            columns=list(REQUIRED_GENE_STAT_COLUMNS),
        )
        stats = GeneStats(
            table=table,
            n_samples=self.n_samples,
            units=self._descriptor.units,
            detection_threshold=threshold,
        )
        self._gene_stats_cache = stats
        return stats

    def subset_genes(self, genes: Sequence[GeneId]) -> "DenseExpressionMatrix":
        """Return the sub-matrix for ``genes``, in the order given.

        Raises
        ------
        KeyError
            Any identifier absent from :attr:`gene_ids`.  Callers must resolve
            signatures first, so an unknown gene here is a programming error,
            not a data condition.
        ValueError
            ``genes`` contains duplicates (which would break the unique-gene
            invariant) or is empty.
        """
        requested = [str(g) for g in genes]
        if not requested:
            raise ValueError("subset_genes requires at least one gene")
        unknown = [g for g in requested if g not in self._gene_index]
        if unknown:
            raise KeyError(
                f"{len(unknown)} gene(s) not in the matrix index, e.g. "
                f"{unknown[:10]!r}; resolve the signature first "
                "(signull.data.resolve.MatrixIndexResolver)"
            )
        _reject_duplicates(tuple(requested), "requested gene")

        rows = np.fromiter(
            (self._gene_index[g] for g in requested), dtype=np.intp, count=len(requested)
        )
        frame = pd.DataFrame(
            self._values[rows, :],
            index=pd.Index(requested, name="gene_id"),
            columns=pd.Index(self._sample_ids, name="sample_id"),
        )
        wanted = set(requested)
        return DenseExpressionMatrix(
            frame,
            dataclasses.replace(self._descriptor, n_genes=len(requested)),
            collapse_map={
                k: v for k, v in self._collapse_map.items() if k in wanted
            },
        )

    def subset_samples(self, samples: Sequence[SampleId]) -> "DenseExpressionMatrix":
        """Return the sub-matrix for ``samples``, in the order given.

        Gene statistics are *not* inherited: the returned object recomputes them
        over the new sample set.
        """
        requested = [str(s) for s in samples]
        unknown = [s for s in requested if s not in self._sample_index]
        if unknown:
            raise KeyError(
                f"{len(unknown)} sample(s) not in the matrix columns, e.g. "
                f"{unknown[:10]!r}"
            )
        _reject_duplicates(tuple(requested), "requested sample")

        cols = np.fromiter(
            (self._sample_index[s] for s in requested),
            dtype=np.intp,
            count=len(requested),
        )
        frame = pd.DataFrame(
            self._values[:, cols],
            index=pd.Index(self._gene_ids, name="gene_id"),
            columns=pd.Index(requested, name="sample_id"),
        )
        return DenseExpressionMatrix(
            frame,
            dataclasses.replace(self._descriptor, n_samples=len(requested)),
            collapse_map=self._collapse_map,
        )

    def align_to(self, outcome: BinaryOutcome) -> AlignedDataset:
        """Intersect samples with ``outcome`` and return the aligned pair.

        The shared order is *matrix column order* restricted to the
        intersection, which makes alignment deterministic and independent of the
        outcome file's row order.  Both sides are reordered to it, so the
        invariant ``matrix.sample_ids == outcome.sample_ids`` holds element-wise
        on the returned :class:`~signull.types.AlignedDataset`.

        Samples present on only one side are dropped and reported as tier-2
        diagnostics on the returned object (``samples_dropped_in_alignment``,
        ``outcome_samples_not_in_matrix``).  Cohort-size and class-balance flags
        (``small_cohort``, ``extreme_class_imbalance``) are raised here too,
        because this is the first point at which the *analysis* cohort exists.

        Raises
        ------
        ValueError
            Empty intersection, or an outcome with one empty class after
            alignment (no discrimination metric is defined).
        """
        log = DiagnosticLog()
        outcome_ids = set(outcome.sample_ids)
        shared = [s for s in self._sample_ids if s in outcome_ids]

        if not shared:
            raise ValueError(
                "matrix and outcome share no sample identifiers "
                f"(matrix has {self.n_samples}, e.g. {self._sample_ids[:3]!r}; "
                f"outcome has {outcome.n_samples}, e.g. {outcome.sample_ids[:3]!r})"
            )

        shared_set = set(shared)
        dropped_matrix = tuple(s for s in self._sample_ids if s not in shared_set)
        dropped_outcome = tuple(s for s in outcome.sample_ids if s not in shared_set)

        if dropped_matrix:
            log.warn(
                DiagnosticCode.SAMPLES_DROPPED_IN_ALIGNMENT,
                f"{len(dropped_matrix)} matrix sample(s) have no outcome and were "
                f"dropped from the analysis cohort",
                context={
                    "n_dropped": len(dropped_matrix),
                    "n_retained": len(shared),
                    "examples": list(dropped_matrix[:10]),
                },
            )
        if dropped_outcome:
            log.warn(
                DiagnosticCode.OUTCOME_SAMPLES_NOT_IN_MATRIX,
                f"{len(dropped_outcome)} outcome sample(s) have no matrix column "
                "and were dropped from the analysis cohort",
                context={
                    "n_dropped": len(dropped_outcome),
                    "examples": list(dropped_outcome[:10]),
                },
            )

        matrix = self.subset_samples(shared)

        position = {s: i for i, s in enumerate(outcome.sample_ids)}
        order = np.fromiter(
            (position[s] for s in shared), dtype=np.intp, count=len(shared)
        )
        labels = np.asarray(outcome.labels, dtype=bool)[order]
        labels.setflags(write=False)
        aligned_outcome = BinaryOutcome(
            sample_ids=tuple(shared),
            labels=labels,
            name=outcome.name,
            positive_label=outcome.positive_label,
            negative_label=outcome.negative_label,
            is_permuted=outcome.is_permuted,
            permutation_seed=outcome.permutation_seed,
            provenance=outcome.provenance,
        )

        if aligned_outcome.is_degenerate:
            raise ValueError(
                "outcome is degenerate after alignment: "
                f"{aligned_outcome.n_positive} positive / "
                f"{aligned_outcome.n_negative} negative over "
                f"{aligned_outcome.n_samples} samples; no discrimination metric "
                "is defined"
            )

        _flag_cohort_size(aligned_outcome, log)

        return AlignedDataset(
            matrix=matrix,
            outcome=aligned_outcome,
            dropped_samples=tuple(sorted(set(dropped_matrix) | set(dropped_outcome))),
            diagnostics=log.diagnostics,
        )

    def __repr__(self) -> str:
        return (
            f"DenseExpressionMatrix(dataset_id={self._descriptor.dataset_id!r}, "
            f"shape=({self.n_genes}, {self.n_samples}), "
            f"namespace={self.namespace.value!r}, units={self._descriptor.units!r})"
        )


def _reject_duplicates(ids: tuple[str, ...], kind: str) -> None:
    """Raise ``ValueError`` naming the duplicated identifiers, if any."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in ids:
        if value in seen:
            duplicates.append(value)
        else:
            seen.add(value)
    if duplicates:
        unique_duplicates = sorted(set(duplicates))
        raise ValueError(
            f"duplicate {kind} identifier(s): {unique_duplicates[:10]!r}"
            + (
                f" (+{len(unique_duplicates) - 10} more)"
                if len(unique_duplicates) > 10
                else ""
            )
        )


def _flag_cohort_size(outcome: BinaryOutcome, log: DiagnosticLog) -> None:
    """Raise tier-2 flags for cohorts too small or too imbalanced to trust.

    Thresholds follow ``docs/statistical-design.md`` Sec. 8 F3.  Refusing to
    emit a p-value is a wave-3 decision; the data layer only records the
    condition so the decision can be made from the stored result alone.
    """
    n = outcome.n_samples
    smaller_class = min(outcome.n_positive, outcome.n_negative)
    if n < SMALL_COHORT_N or smaller_class < SMALL_CLASS_N:
        log.warn(
            DiagnosticCode.SMALL_COHORT,
            f"cohort has {n} samples with {outcome.n_positive} positive / "
            f"{outcome.n_negative} negative; docs/statistical-design.md Sec. 8 F3 "
            f"refuses a p-value below N={SMALL_COHORT_N} or a class of "
            f"{SMALL_CLASS_N}",
            context={
                "n_samples": n,
                "n_positive": outcome.n_positive,
                "n_negative": outcome.n_negative,
                "min_n_samples": SMALL_COHORT_N,
                "min_class_size": SMALL_CLASS_N,
            },
        )
    prevalence = outcome.prevalence
    if prevalence < 0.10 or prevalence > 0.90:
        log.warn(
            DiagnosticCode.EXTREME_CLASS_IMBALANCE,
            f"positive-class prevalence is {prevalence:.4f}; AUROC is still "
            "defined but average precision has a correspondingly extreme chance "
            "level",
            context={
                "prevalence": float(prevalence),
                "n_positive": outcome.n_positive,
                "n_samples": n,
            },
        )


def emit_detection_degeneracy(stats: GeneStats, log: DiagnosticLog) -> None:
    """Record whether the detection dimension is informative for this matrix.

    Tier 3 (record only).  When the detection rate is effectively constant --
    which is the expected state for an array cohort with no per-probe background
    call -- ``docs/statistical-design.md`` Sec. 2.2 collapses the matching
    dimension ``K_d`` to 1.  The null sampler makes that decision; the data layer
    records the fact so the report can explain it.
    """
    rates = stats.table["detection_rate"].to_numpy(dtype=np.float64)
    if detection_dimension_is_degenerate(rates):
        log.record(
            DiagnosticCode.DETECTION_DIMENSION_DEGENERATE,
            "detection rate is effectively constant across genes "
            f"(sd={float(np.std(rates, ddof=1)):.5f} < {DETECTION_DEGENERACY_SD}); "
            "the detection dimension carries no matching information "
            "(docs/statistical-design.md Sec. 2.2, K_d -> 1)",
            context={
                "detection_threshold": stats.detection_threshold,
                "detection_rate_sd": float(np.std(rates, ddof=1)),
                "detection_rate_min": float(rates.min()),
                "threshold": DETECTION_DEGENERACY_SD,
            },
        )
