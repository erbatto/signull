"""Core data contracts for :mod:`signull`.

This module is the single source of truth for the shapes that flow between the
``data``, ``nulls``, ``scoring``, ``metrics`` and ``report`` subpackages.  It
contains **contracts only**: enums, frozen value objects and :class:`Protocol`
definitions.  There is deliberately no analysis logic here beyond trivial
derived properties -- the implementing packages own all behaviour.

Global conventions enforced by every contract below
---------------------------------------------------

**Orientation.** Expression data is *always* ``genes x samples``: rows are
genes, columns are samples.  ``matrix.values.shape == (n_genes, n_samples)``.
This orientation is checked at construction time by the ``data`` package and is
never re-derived downstream.  Anything that transposes must do so locally and
must not hand a transposed array back across a package boundary.

**Units.** Expression values are assumed to be *normalised and variance-
stabilised on a log-like scale* -- e.g. ``log2(TPM + 1)``, ``log2(CPM + 1)``,
RMA/MAS5 log2 intensities, or VST/rlog counts.  They are floats, may be
negative (centred data), and must be finite.  Raw integer counts are **not**
acceptable input: mean/variance matching and z-score scoring both assume the
log-like scale.  The transformation actually applied is recorded verbatim in
:class:`DatasetDescriptor.preprocessing` and travels with every result.

**Gene identifiers.** The canonical internal namespace is
:attr:`GeneIdNamespace.HGNC_SYMBOL` -- uppercase, unversioned, HGNC-approved
symbols.  Translation from probe IDs / Ensembl IDs / Entrez IDs happens exactly
once, inside the ``data`` package, at load time.  No other package converts
identifiers.

**Missing genes.** Signature genes absent from the matrix are never silently
ignored.  Resolution of a :class:`Signature` against an
:class:`ExpressionMatrix` produces a :class:`SignatureResolution` that names
every missing and every collapsed identifier; that object is embedded in the
:class:`NullTestResult` and surfaced in the report.  Whether missing genes
raise or are dropped is governed by :class:`MissingGenePolicy` plus the overlap
floors in :class:`ResolutionSpec`; falling below a floor always raises.

**Sample order.** Every per-sample array (scores, labels) is ordered exactly as
``matrix.sample_ids``.  Alignment happens once, via
:meth:`ExpressionMatrix.align_to`, and produces an :class:`AlignedDataset`
whose invariant is ``matrix.sample_ids == outcome.sample_ids``.

**Equality.** Value objects that hold NumPy arrays or pandas objects are
declared ``@dataclass(frozen=True, eq=False)``: element-wise comparison would
make the generated ``__eq__`` raise, so identity comparison is used instead and
structural comparison is left to explicit helpers in the owning package.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Final, Protocol, TypeAlias, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = [
    "default_mean_expression_bins",
    "DEFAULT_BINS_BY_PROPERTY",
    "MIN_DRAWS",
    "DEFAULT_DRAWS",
    "CONTRACT_VERSION",
    "REQUIRED_GENE_STAT_COLUMNS",
    "GeneId",
    "SampleId",
    "FloatArray",
    "BoolArray",
    "IntArray",
    "GeneIdNamespace",
    "SignatureOrigin",
    "NullType",
    "MatchingProperty",
    "MetricName",
    "ScoringMethodName",
    "DirectionPolicy",
    "Alternative",
    "MissingGenePolicy",
    "DuplicateHandling",
    "Severity",
    "Provenance",
    "Diagnostic",
    "DatasetDescriptor",
    "OutcomeDescriptor",
    "GeneStats",
    "Signature",
    "BinaryOutcome",
    "SignatureResolution",
    "ExpressionMatrix",
    "AlignedDataset",
    "GeneIdResolver",
    "SampleScores",
    "ScoringMethod",
    "Metric",
    "NullDraw",
    "NullModel",
    "ScoringSpec",
    "MatchingSpec",
    "CVSpec",
    "ResolutionSpec",
    "NullSpec",
    "NullTestConfig",
    "NullTestResult",
    "EvidenceSummary",
]

# ---------------------------------------------------------------------------
# Versioning and constants
# ---------------------------------------------------------------------------

#: Version of the contracts in this module.  Bump the MAJOR component whenever a
#: field is removed or its meaning changes; MINOR when fields are added.  Every
#: :class:`NullTestConfig` records the version it was built under so that stored
#: results remain interpretable after the contract evolves.
CONTRACT_VERSION: Final[str] = "1.1.0"

#: Columns that :meth:`ExpressionMatrix.gene_stats` must provide.  All are
#: computed on the *aligned analysis cohort*, in the same units as the matrix.
#:
#: - ``mean``           : per-gene mean expression (matrix units).
#: - ``sd``             : per-gene standard deviation (matrix units, ddof=1).
#: - ``variance``       : ``sd ** 2`` (matrix units squared).
#: - ``detection_rate`` : fraction of samples in ``[0, 1]`` with expression
#:                        strictly above :attr:`DatasetDescriptor.detection_threshold`.
#: - ``median``         : per-gene median expression (matrix units).
REQUIRED_GENE_STAT_COLUMNS: Final[tuple[str, ...]] = (
    "mean",
    "sd",
    "variance",
    "detection_rate",
    "median",
)

GeneId: TypeAlias = str
SampleId: TypeAlias = str
FloatArray: TypeAlias = npt.NDArray[np.float64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]
IntArray: TypeAlias = npt.NDArray[np.int64]


# ---------------------------------------------------------------------------
# Fixed vocabularies
# ---------------------------------------------------------------------------


class GeneIdNamespace(str, Enum):
    """Identifier namespace of a gene list or matrix index.

    :attr:`HGNC_SYMBOL` is the canonical internal namespace.  Everything else is
    an *input* namespace that the ``data`` package maps to symbols at load time.
    """

    HGNC_SYMBOL = "hgnc_symbol"
    ENSEMBL_GENE = "ensembl_gene"
    ENTREZ_GENE = "entrez_gene"
    PROBE_ID = "probe_id"
    UNKNOWN = "unknown"


class SignatureOrigin(str, Enum):
    """Where a :class:`Signature` came from.

    Guards against a null draw being reported as if it were the candidate.
    """

    CANDIDATE = "candidate"
    RANDOM_NULL = "random_null"
    DERIVED = "derived"


class NullType(str, Enum):
    """The two non-interchangeable null hypotheses this tool supports.

    :attr:`RANDOM_GENE_SET`
        "Is *this gene set* special among gene sets of the same size drawn from
        this dataset?"  Labels are held fixed; the gene set varies.

    :attr:`LABEL_PERMUTATION`
        "Is there *any* outcome signal to detect at all?"  The gene set is held
        fixed; the outcome labels are permuted.  This is also the null used by
        the calibration acceptance test.
    """

    RANDOM_GENE_SET = "random_gene_set"
    LABEL_PERMUTATION = "label_permutation"


class MatchingProperty(str, Enum):
    """Per-gene property a random gene-set null is matched on.

    Set size is matched *unconditionally* and is therefore not a member here.
    Every member's value is the :data:`REQUIRED_GENE_STAT_COLUMNS` column it
    reads.
    """

    MEAN_EXPRESSION = "mean"
    DETECTION_RATE = "detection_rate"
    VARIANCE = "variance"

    @property
    def column(self) -> str:
        """Name of the :meth:`ExpressionMatrix.gene_stats` column to match on."""
        return self.value


class MetricName(str, Enum):
    """Discrimination metrics defined for a binary endpoint.

    :attr:`AUROC`
        Area under the ROC curve; chance level is 0.5 independent of prevalence.

    :attr:`AVERAGE_PRECISION`
        Area under the precision-recall curve; chance level equals the positive
        class prevalence, so it is only comparable across cohorts of equal
        balance.
    """

    AUROC = "auroc"
    AVERAGE_PRECISION = "average_precision"


class ScoringMethodName(str, Enum):
    """Fixed vocabulary of signature scoring strategies.

    :attr:`SUPERVISED_MODEL` marks any strategy that consults the outcome; such
    strategies must return out-of-fold predictions only (see
    :class:`ScoringMethod`).
    """

    MEAN_Z_SCORE = "mean_z_score"
    FIRST_PRINCIPAL_COMPONENT = "first_principal_component"
    SSGSEA = "ssgsea"
    SUPERVISED_MODEL = "supervised_model"


class DirectionPolicy(str, Enum):
    """How the sign of an unsupervised signature score is handled.

    A randomly drawn gene set has no a-priori direction: its mean z-score may
    point either way, so a one-sided AUROC systematically understates random
    performance and flatters the candidate.  The chosen policy must be applied
    **identically** to the candidate and to every null draw.

    :attr:`AS_GIVEN`
        Use the metric as computed.  Correct only when the candidate carries a
        published, pre-specified direction (signed
        :attr:`Signature.weights`) *and* nulls inherit that same signed
        structure.

    :attr:`SYMMETRIZED`
        Fold the metric about its chance level: ``max(m, 2 * chance - m)`` for
        AUROC.  Direction-free and the safe default for unsigned gene sets.
    """

    AS_GIVEN = "as_given"
    SYMMETRIZED = "symmetrized"


class Alternative(str, Enum):
    """Tail used when converting a null distribution into a p-value."""

    GREATER = "greater"
    LESS = "less"
    TWO_SIDED = "two_sided"


class MissingGenePolicy(str, Enum):
    """What to do with signature genes absent from the matrix index.

    Independent of this choice, missing identifiers are *always* enumerated in
    :class:`SignatureResolution` and always reach the report; and the overlap
    floors in :class:`ResolutionSpec` raise regardless of policy.

    :attr:`RAISE`
        Any missing identifier is an error.  Use for regulated / audited runs.

    :attr:`DROP`
        Continue with the intersection.  The *effective* signature size --
        :attr:`SignatureResolution.n_matched`, not the nominal size -- is what
        null gene sets are size-matched against.
    """

    RAISE = "raise"
    DROP = "drop"


class DuplicateHandling(str, Enum):
    """Collapse rule when several matrix rows map to one canonical symbol.

    Common with probe-level microarray data.  The choice changes results and is
    therefore part of the captured config.
    """

    RAISE = "raise"
    KEEP_FIRST = "keep_first"
    MEAN = "mean"
    MAX_VARIANCE = "max_variance"


class Severity(str, Enum):
    """Severity of a :class:`Diagnostic` recorded in a result."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Provenance and descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where an artefact came from, in human- and machine-readable form.

    Attributes
    ----------
    source:
        Free-text origin, e.g. ``"GEO:GSE2034"``, ``"MSigDB c2.cp.v2023.1"``,
        ``"supplementary table 3 of doi:10.1371/journal.pcbi.1002240"``.
    identifier:
        Stable accession within ``source`` when one exists, else ``None``.
    retrieved_at:
        ISO-8601 date or timestamp of retrieval, or ``None`` if unknown.
    citation:
        DOI or full citation for the artefact, when published.
    checksum:
        Hex digest (algorithm named in ``checksum_algorithm``) of the raw bytes
        the artefact was parsed from.  Used to detect silent input drift between
        runs that share a config.
    checksum_algorithm:
        Digest algorithm name, e.g. ``"sha256"``.
    notes:
        Anything a reader needs in order to trust the artefact.
    """

    source: str
    identifier: str | None = None
    retrieved_at: str | None = None
    citation: str | None = None
    checksum: str | None = None
    checksum_algorithm: str | None = None
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """A non-fatal condition worth recording alongside a result.

    Diagnostics are data, not log lines: they are serialised with the result so
    that a stored result can be audited without the original console output.

    Attributes
    ----------
    code:
        Stable machine-readable identifier, e.g. ``"low_signature_overlap"``.
        Codes are owned by the package that emits them and must not be reused
        with a different meaning.
    severity:
        See :class:`Severity`.
    message:
        Human-readable one-liner.
    context:
        Small JSON-serialisable payload (counts, gene names, thresholds).
    """

    code: str
    severity: Severity
    message: str
    context: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DatasetDescriptor:
    """Identity of the cohort matrix, without the matrix itself.

    Embedded in :class:`NullTestConfig` so a result stays interpretable when the
    matrix is not at hand.  Two results are comparable only if their descriptors
    agree on ``dataset_id``, ``preprocessing`` and ``checksum``.

    Attributes
    ----------
    dataset_id:
        Short stable handle, e.g. ``"GSE2034"``.
    n_genes, n_samples:
        Dimensions of the matrix *as analysed* (post filtering, post alignment).
    namespace:
        Namespace of the matrix index; canonical pipelines report
        :attr:`GeneIdNamespace.HGNC_SYMBOL`.
    units:
        Verbatim description of the value scale, e.g. ``"log2(TPM+1)"``.
    preprocessing:
        Ordered list of transformations applied before analysis, e.g.
        ``("quantile_normalise", "log2(x+1)", "drop_genes_detected_in_<10pct")``.
    detection_threshold:
        Expression value strictly above which a gene counts as detected in a
        sample, in matrix units.  Defines the ``detection_rate`` gene stat.
    provenance:
        See :class:`Provenance`.
    """

    dataset_id: str
    n_genes: int
    n_samples: int
    namespace: GeneIdNamespace
    units: str
    preprocessing: tuple[str, ...] = ()
    detection_threshold: float = 0.0
    provenance: Provenance | None = None


@dataclass(frozen=True, slots=True)
class OutcomeDescriptor:
    """Identity of the endpoint, without the label vector itself.

    Attributes
    ----------
    name:
        Endpoint name, e.g. ``"distant_metastasis_within_5y"``.
    positive_label, negative_label:
        Human-readable names of the two classes.  ``positive_label`` is the
        class encoded as ``True`` in :attr:`BinaryOutcome.labels` and the class
        the metric treats as positive.
    n_samples, n_positive:
        Cohort size and positive count as analysed.
    is_permuted:
        ``True`` when the labels were permuted (calibration runs).  Present so
        that a permuted-label result can never be mistaken for a real one.
    provenance:
        See :class:`Provenance`.
    """

    name: str
    positive_label: str
    negative_label: str
    n_samples: int
    n_positive: int
    is_permuted: bool = False
    provenance: Provenance | None = None

    @property
    def prevalence(self) -> float:
        """Fraction of samples in the positive class; chance-level average precision."""
        return self.n_positive / self.n_samples if self.n_samples else float("nan")


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False, slots=True)
class GeneStats:
    """Per-gene summary statistics of the analysis cohort.

    This is the object the matched random-gene-set sampler reads.  It **must**
    be computed on the aligned analysis cohort (the samples that actually enter
    the test), never on a raw superset: matching on statistics from a different
    sample set silently misspecifies the null.

    Attributes
    ----------
    table:
        ``pandas.DataFrame`` indexed by canonical gene identifier, containing at
        least :data:`REQUIRED_GENE_STAT_COLUMNS`.  Units follow the matrix; see
        the column documentation on :data:`REQUIRED_GENE_STAT_COLUMNS`.  Extra
        columns are permitted and ignored by the contract.
    n_samples:
        Number of samples the statistics were computed over.
    units:
        Value scale of the underlying matrix, copied from
        :attr:`DatasetDescriptor.units`.
    detection_threshold:
        Threshold used to derive ``detection_rate``, in matrix units.
    """

    table: pd.DataFrame
    n_samples: int
    units: str
    detection_threshold: float = 0.0

    @property
    def gene_ids(self) -> pd.Index:
        """Index of the statistics table, in table order."""
        return self.table.index


@dataclass(frozen=True, slots=True)
class Signature:
    """A candidate or null gene signature.

    Attributes
    ----------
    genes:
        Ordered tuple of gene identifiers.  **Invariant: no duplicates.**  A
        tuple rather than a ``frozenset`` so the object is hashable *and* stays
        aligned with :attr:`weights`; set semantics are available through
        :attr:`gene_set`.
    name:
        Short human-readable name, unique within a run.
    namespace:
        Namespace of ``genes``.  After the ``data`` package has resolved the
        signature this is :attr:`GeneIdNamespace.HGNC_SYMBOL`.
    origin:
        See :class:`SignatureOrigin`.  Null draws must set
        :attr:`SignatureOrigin.RANDOM_NULL`.
    weights:
        Optional per-gene weights aligned element-wise to ``genes``.  Sign
        encodes direction (``+1`` up in the positive class, ``-1`` down);
        magnitude encodes emphasis.  ``None`` means unweighted / unsigned.
        A random gene-set null of a *signed* candidate must reuse the
        candidate's weight multiset so that the null shares its signed
        structure, otherwise the comparison is not like-for-like.
    provenance:
        See :class:`Provenance`.  For null draws, records the sampler and draw
        index rather than a publication.
    description:
        Optional free text.
    """

    genes: tuple[GeneId, ...]
    name: str
    namespace: GeneIdNamespace = GeneIdNamespace.HGNC_SYMBOL
    origin: SignatureOrigin = SignatureOrigin.CANDIDATE
    weights: tuple[float, ...] | None = None
    provenance: Provenance | None = None
    description: str | None = None

    @property
    def size(self) -> int:
        """Number of genes in the signature as declared."""
        return len(self.genes)

    @property
    def gene_set(self) -> frozenset[GeneId]:
        """Set view of :attr:`genes`, for membership and overlap tests."""
        return frozenset(self.genes)

    @property
    def is_signed(self) -> bool:
        """``True`` when per-gene directions/weights are attached."""
        return self.weights is not None


@dataclass(frozen=True, eq=False, slots=True)
class BinaryOutcome:
    """The binary endpoint for a cohort.

    Attributes
    ----------
    sample_ids:
        Sample identifiers, ordered.  After alignment this equals
        ``matrix.sample_ids`` element-wise.
    labels:
        Boolean array of shape ``(n_samples,)``; ``True`` denotes the positive
        class named by ``positive_label``.  No missing values: samples with an
        unknown endpoint are dropped during loading and reported as a
        :class:`Diagnostic`, never encoded as ``False``.
    name, positive_label, negative_label:
        See :class:`OutcomeDescriptor`.
    is_permuted:
        ``True`` for label-permutation draws and calibration runs.
    permutation_seed:
        Seed of the permutation that produced these labels, when
        ``is_permuted``; ``None`` otherwise.
    provenance:
        See :class:`Provenance`.
    """

    sample_ids: tuple[SampleId, ...]
    labels: BoolArray
    name: str
    positive_label: str = "positive"
    negative_label: str = "negative"
    is_permuted: bool = False
    permutation_seed: int | None = None
    provenance: Provenance | None = None

    @property
    def n_samples(self) -> int:
        """Cohort size."""
        return len(self.sample_ids)

    @property
    def n_positive(self) -> int:
        """Number of samples in the positive class."""
        return int(np.count_nonzero(self.labels))

    @property
    def n_negative(self) -> int:
        """Number of samples in the negative class."""
        return self.n_samples - self.n_positive

    @property
    def prevalence(self) -> float:
        """Positive class fraction; the chance level for average precision."""
        return self.n_positive / self.n_samples if self.n_samples else float("nan")

    @property
    def is_degenerate(self) -> bool:
        """``True`` when one class is empty, so no discrimination metric exists."""
        return self.n_positive == 0 or self.n_negative == 0

    def describe(self) -> OutcomeDescriptor:
        """Config-sized summary of this outcome."""
        return OutcomeDescriptor(
            name=self.name,
            positive_label=self.positive_label,
            negative_label=self.negative_label,
            n_samples=self.n_samples,
            n_positive=self.n_positive,
            is_permuted=self.is_permuted,
            provenance=self.provenance,
        )


@dataclass(frozen=True, slots=True)
class SignatureResolution:
    """Outcome of matching a :class:`Signature` onto a matrix index.

    Signature/matrix identifier mismatch is the dominant silent failure mode of
    this kind of tool, so the record of what happened is a first-class result
    field, not a log message.  It is embedded in :class:`NullTestResult` and
    must be rendered by the report layer even when nothing went wrong.

    Attributes
    ----------
    requested:
        Identifiers as supplied by the user, before any mapping.
    matched:
        Canonical identifiers present in the matrix, in matrix row order.  These
        are the genes actually scored.
    missing:
        Requested identifiers with no row in the matrix after mapping.
    unmapped:
        Requested identifiers that could not even be translated into the
        canonical namespace (e.g. retired symbols, unknown probes).  A subset of
        ``missing`` in effect, tracked separately because the remedy differs:
        ``unmapped`` means "fix the annotation", plain ``missing`` means "this
        gene is not measured in this platform".
    collapsed:
        Mapping from canonical identifier to the >1 source identifiers that
        collapsed onto it, together with the rule applied.  Empty when the input
        index was already unique.
    aliased:
        Mapping from requested identifier to the canonical identifier it was
        renamed to (symbol alias / namespace conversion).
    source_namespace, target_namespace:
        Namespaces before and after resolution.
    duplicate_handling:
        Rule applied to ``collapsed`` rows.
    mapping_source:
        Identity and version of the annotation used, e.g.
        ``"HGNC complete set 2024-07-01"``.  Results are not reproducible
        without it.
    """

    requested: tuple[GeneId, ...]
    matched: tuple[GeneId, ...]
    missing: tuple[GeneId, ...] = ()
    unmapped: tuple[GeneId, ...] = ()
    collapsed: Mapping[GeneId, tuple[GeneId, ...]] = field(default_factory=dict)
    aliased: Mapping[GeneId, GeneId] = field(default_factory=dict)
    source_namespace: GeneIdNamespace = GeneIdNamespace.HGNC_SYMBOL
    target_namespace: GeneIdNamespace = GeneIdNamespace.HGNC_SYMBOL
    duplicate_handling: DuplicateHandling = DuplicateHandling.RAISE
    mapping_source: str | None = None

    @property
    def n_requested(self) -> int:
        """Number of identifiers supplied by the user."""
        return len(self.requested)

    @property
    def n_matched(self) -> int:
        """Effective signature size; this is what nulls are size-matched to."""
        return len(self.matched)

    @property
    def overlap_fraction(self) -> float:
        """``n_matched / n_requested``; 1.0 for a clean resolution."""
        return self.n_matched / self.n_requested if self.n_requested else float("nan")


@dataclass(frozen=True, eq=False, slots=True)
class SampleScores:
    """Per-sample signature score vector.

    Attributes
    ----------
    values:
        Float array of shape ``(n_samples,)``, ordered as ``sample_ids``.
        Non-finite entries are a contract violation and must raise in the
        producing package.  The scale is method-defined and never compared
        across methods; only the induced ranking enters AUROC / average
        precision.
    sample_ids:
        Sample order, equal to the matrix's ``sample_ids``.
    method:
        Which scoring strategy produced the values.
    is_out_of_fold:
        ``True`` when the values are cross-validated out-of-fold predictions
        from a supervised method.  In-sample predictions from a supervised
        method are never a valid input to a metric here.
    """

    values: FloatArray
    sample_ids: tuple[SampleId, ...]
    method: ScoringMethodName
    is_out_of_fold: bool = False

    @property
    def n_samples(self) -> int:
        """Length of the score vector."""
        return len(self.sample_ids)


# ---------------------------------------------------------------------------
# Structural interfaces
# ---------------------------------------------------------------------------


@runtime_checkable
class ExpressionMatrix(Protocol):
    """A cohort expression matrix, oriented ``genes x samples``.

    Declared as a Protocol rather than a concrete dataclass because the ``data``
    package may back it with a dense array, a memory-mapped store or an AnnData
    object; every other package depends only on this surface.  Implementations
    are expected to be immutable: all methods return new objects.
    """

    @property
    def values(self) -> FloatArray:
        """Expression values, shape ``(n_genes, n_samples)``, finite floats.

        Units are :attr:`DatasetDescriptor.units` -- a log-like normalised
        scale.  Row order matches :attr:`gene_ids`, column order matches
        :attr:`sample_ids`.
        """
        ...

    @property
    def gene_ids(self) -> tuple[GeneId, ...]:
        """Canonical gene identifiers, unique, in row order."""
        ...

    @property
    def sample_ids(self) -> tuple[SampleId, ...]:
        """Sample identifiers, unique, in column order."""
        ...

    @property
    def namespace(self) -> GeneIdNamespace:
        """Namespace of :attr:`gene_ids`; canonically ``HGNC_SYMBOL``."""
        ...

    @property
    def descriptor(self) -> DatasetDescriptor:
        """Identity/provenance of this matrix, for config capture."""
        ...

    def gene_stats(self) -> GeneStats:
        """Per-gene summary statistics over *this* matrix's samples.

        Implementations should cache: the matched null sampler calls this once
        per run but scoring code may call it repeatedly.  Statistics must be
        recomputed (not inherited) whenever the sample set changes, e.g. after
        :meth:`align_to`.
        """
        ...

    def subset_genes(self, genes: Sequence[GeneId]) -> "ExpressionMatrix":
        """Return the sub-matrix for ``genes``, in the order given.

        Raises ``KeyError`` on any identifier absent from :attr:`gene_ids`:
        callers must resolve signatures first (see :class:`GeneIdResolver`), so
        an unknown gene reaching this method is a programming error rather than
        a data condition.
        """
        ...

    def align_to(self, outcome: BinaryOutcome) -> "AlignedDataset":
        """Intersect samples with ``outcome`` and return the aligned pair.

        The intersection is taken on sample identifiers and both sides are
        reordered to a single common order.  Samples present on only one side
        are dropped and reported as a :class:`Diagnostic`.  An empty or
        single-class intersection raises.
        """
        ...


@dataclass(frozen=True, eq=False, slots=True)
class AlignedDataset:
    """A matrix and an outcome guaranteed to share sample identity and order.

    Invariant: ``matrix.sample_ids == outcome.sample_ids``.  Every downstream
    computation (gene statistics, eligible universe, scoring, metrics, null
    draws) operates on this object, so that the null and the candidate are
    provably evaluated on identical samples.

    Attributes
    ----------
    matrix:
        Genes x samples expression, restricted to the shared samples.
    outcome:
        Binary endpoint, restricted and reordered to the shared samples.
    dropped_samples:
        Identifiers dropped by the intersection, for the report.
    diagnostics:
        Conditions noted during alignment.
    """

    matrix: ExpressionMatrix
    outcome: BinaryOutcome
    dropped_samples: tuple[SampleId, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def n_samples(self) -> int:
        """Number of samples shared by matrix and outcome."""
        return self.outcome.n_samples


@runtime_checkable
class GeneIdResolver(Protocol):
    """Maps user-supplied identifiers onto a matrix index. Owned by ``data``."""

    def resolve(
        self,
        signature: Signature,
        matrix: ExpressionMatrix,
        spec: "ResolutionSpec",
    ) -> tuple[Signature, SignatureResolution]:
        """Return the resolved signature and a full record of what happened.

        The returned signature contains only :attr:`SignatureResolution.matched`
        genes, in the canonical namespace, with weights subset accordingly.
        Raises when ``spec`` demands it (see :class:`MissingGenePolicy` and the
        overlap floors on :class:`ResolutionSpec`); otherwise every deviation is
        recorded in the resolution rather than raised.
        """
        ...


@runtime_checkable
class ScoringMethod(Protocol):
    """Turns (matrix, signature) into one score per sample.

    The pluggable strategy at the heart of the tool.  The *same instance* is
    applied to the candidate and to every null draw; any state that differs
    between those calls (a fitted scaler, a cached gene list) breaks the
    comparison and is forbidden.  Implementations must therefore be pure with
    respect to their arguments and depend on randomness only through the
    supplied generator.

    Results are known to be scoring-method dependent, so
    :attr:`name` and :attr:`params` are captured verbatim in every config.
    """

    @property
    def name(self) -> ScoringMethodName:
        """Fixed-vocabulary identity of the strategy."""
        ...

    @property
    def params(self) -> Mapping[str, object]:
        """JSON-serialisable hyper-parameters, complete enough to reconstruct."""
        ...

    @property
    def is_supervised(self) -> bool:
        """``True`` when the strategy consults the outcome to build scores."""
        ...

    def score(
        self,
        dataset: AlignedDataset,
        signature: Signature,
        rng: np.random.Generator,
    ) -> SampleScores:
        """Score every sample of ``dataset`` under ``signature``.

        Contract
        --------
        * Returns exactly ``dataset.n_samples`` finite values, ordered as
          ``dataset.matrix.sample_ids``.
        * Higher values mean "more like the positive class" only when the
          signature is signed; for unsigned signatures the sign is arbitrary and
          the caller must apply :class:`DirectionPolicy`.
        * Unsupervised strategies must ignore ``dataset.outcome`` entirely.
        * Supervised strategies must return out-of-fold predictions with
          :attr:`SampleScores.is_out_of_fold` set, performing all fitting --
          including any feature selection -- strictly inside the folds.
        * Missing genes never reach here: the signature has already been
          resolved against the matrix.
        * A signature that resolves to zero genes raises ``ValueError``.
        """
        ...


@runtime_checkable
class Metric(Protocol):
    """Reduces scores plus outcome to one scalar."""

    @property
    def name(self) -> MetricName:
        """Fixed-vocabulary identity of the metric."""
        ...

    @property
    def greater_is_better(self) -> bool:
        """Direction of merit; ``True`` for both AUROC and average precision."""
        ...

    def chance_level(self, outcome: BinaryOutcome) -> float:
        """Value expected from an uninformative score on this outcome.

        0.5 for AUROC; :attr:`BinaryOutcome.prevalence` for average precision.
        Used by :attr:`DirectionPolicy.SYMMETRIZED` and by the report.
        """
        ...

    def __call__(self, scores: SampleScores, outcome: BinaryOutcome) -> float:
        """Compute the metric.

        Requires ``scores.sample_ids == outcome.sample_ids``; raises otherwise
        rather than reordering silently.  Raises on a degenerate outcome.  Ties
        in ``scores`` are handled by the standard trapezoidal/rank convention of
        the underlying implementation and are reported as a diagnostic when they
        exceed a meaningful fraction of pairs.
        """
        ...


@dataclass(frozen=True, eq=False, slots=True)
class NullDraw:
    """One realisation of a null hypothesis.

    Both fields are always populated, whatever the :class:`NullType`, so that a
    single evaluation path -- ``metric(scoring.score(dataset', signature),
    outcome')`` -- serves candidate and null alike.  For
    :attr:`NullType.RANDOM_GENE_SET` the outcome is the observed one; for
    :attr:`NullType.LABEL_PERMUTATION` the signature is the resolved candidate.

    Attributes
    ----------
    index:
        Zero-based draw index within the run; stable across reruns with the same
        seed.
    signature:
        Gene set to score for this draw.
    outcome:
        Labels to evaluate against for this draw.
    null_type:
        Which null this draw realises.
    diagnostics:
        Conditions noted while producing the draw, e.g. a matching bin that had
        to be widened to find enough candidate genes.
    """

    index: int
    signature: Signature
    outcome: BinaryOutcome
    null_type: NullType
    diagnostics: tuple[Diagnostic, ...] = ()


@runtime_checkable
class NullModel(Protocol):
    """Generates null realisations for a candidate on a dataset."""

    @property
    def null_type(self) -> NullType:
        """Which null hypothesis this model realises."""
        ...

    @property
    def spec(self) -> "NullSpec":
        """Full, serialisable configuration of this null model."""
        ...

    def eligible_universe(self, dataset: AlignedDataset) -> tuple[GeneId, ...]:
        """Genes a random draw may sample from.

        Defined for gene-set nulls; the label-permutation null returns the
        resolved candidate's genes.  The universe is the matrix index after the
        run's filters, so it is dataset-specific by construction: the fraction
        of random signatures that reach significance varies by an order of
        magnitude between datasets, and a universe borrowed from elsewhere would
        make the p-value uninterpretable.
        """
        ...

    def draw(
        self,
        candidate: Signature,
        dataset: AlignedDataset,
        n_draws: int,
        rng: np.random.Generator,
    ) -> Iterator[NullDraw]:
        """Yield exactly ``n_draws`` null realisations.

        Contract
        --------
        * ``candidate`` is already resolved against ``dataset.matrix``.
        * Gene-set draws have the same size as the *resolved* candidate and,
          when :attr:`NullSpec.matching` is non-empty, are matched on those
          per-gene properties using ``dataset.matrix.gene_stats()``.
        * Draws are a lazy iterator so that large ``n_draws`` need not be
          materialised.
        * Consuming the iterator with the same seed and inputs must reproduce
          identical draws, and draw ``i`` must not depend on how many draws were
          consumed before it beyond ordinary generator advancement.
        * If the matching constraint cannot be satisfied the model must widen
          bins and record a :class:`Diagnostic` on the draw, or raise -- never
          silently fall back to unmatched sampling.
        """
        ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScoringSpec:
    """Serialisable description of the scoring strategy used."""

    name: ScoringMethodName
    params: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CVSpec:
    """Cross-validation scheme, required whenever scoring is supervised.

    Attributes
    ----------
    n_folds:
        Number of folds; stratified on the outcome when ``stratified``.
    n_repeats:
        Repeats of the whole scheme; out-of-fold scores are averaged over
        repeats.
    stratified:
        Preserve class balance per fold.  Strongly recommended for imbalanced
        endpoints.
    seed:
        Fold-assignment seed.  Derived from the run seed; recorded so the exact
        partition can be rebuilt.
    """

    n_folds: int = 5
    n_repeats: int = 1
    stratified: bool = True
    seed: int | None = None


#: Default number of null realisations (docs/statistical-design.md Sec. 3.3).
DEFAULT_DRAWS: Final[int] = 10_000
#: Hard floor below which an implementation MUST refuse to emit a p-value.
MIN_DRAWS: Final[int] = 2_000
#: Per-property bin counts for nested conditional matching (Sec. 2.2).  The count for
#: MEAN_EXPRESSION is data-dependent -- see ``default_mean_expression_bins``.
DEFAULT_BINS_BY_PROPERTY: Final[Mapping[MatchingProperty, int]] = MappingProxyType(
    {MatchingProperty.VARIANCE: 5, MatchingProperty.DETECTION_RATE: 3}
)


def default_mean_expression_bins(n_background: int) -> int:
    """Bin count for MEAN_EXPRESSION under nested matching: ``clip(n/500, 10, 40)``."""
    return max(10, min(40, n_background // 500))


@dataclass(frozen=True, slots=True)
class MatchingSpec:
    """How random gene sets are matched to the candidate.

    Size is always matched exactly and is not listed in ``properties``.  An
    unmatched null is misspecified in a direction that flatters the candidate,
    because signature genes are typically higher-expressed and more variable
    than a uniform random draw.

    Attributes
    ----------
    properties:
        Per-gene properties to match on, in priority order.  Empty means
        size-only matching -- permitted, but the report must say so.
    n_bins:
        Number of quantile bins per property.  Sampling draws, for each
        candidate gene, a replacement gene from the same joint bin.
    tolerance:
        Optional absolute tolerance per property, as an alternative to binning.
    bins_by_property:
        CONTRACT AMENDMENT 1.1.0 (Fleet, wave-1 merge).  Per-property bin counts.  A single
        ``n_bins`` integer cannot express the nested conditional scheme required by
        ``docs/statistical-design.md`` Sec. 2.2, which uses a different count at each level:
        ``clip(|B|/500, 10, 40)`` for MEAN_EXPRESSION, then 5 for VARIANCE, then 3 for
        DETECTION_RATE.  See ``DEFAULT_BINS_BY_PROPERTY`` / ``default_mean_expression_bins``.
        Properties absent from this mapping fall back to ``n_bins``.
    nested:
        CONTRACT AMENDMENT 1.1.0.  When True (default) bins are formed CONDITIONALLY, each
        property binned within the strata of the previous one in ``properties`` order.  When
        False they form a product grid of marginal bins.  The statistical spec REJECTS the
        product grid: mean and variance are strongly dependent, so marginal cells go sparse
        and the matcher silently degrades.  ``nested=False`` is a diagnostic escape hatch,
        never a default.
    with_replacement:
        Whether a single draw may reuse a gene.  Default ``False``: a null
        signature has the same size in *distinct* genes as the candidate.
    exclude_candidate_genes:
        Whether the candidate's own genes are removed from the sampling
        universe.  Default ``False``, matching the published random-signature
        benchmarks: the candidate competes against gene sets drawn from the full
        universe it was itself drawn from.
    max_resample_attempts:
        Attempts before a bin is widened (and a diagnostic recorded).
    set_level_constraints:
        Optional constraints on *set-level* properties that cannot be expressed
        per gene, mapped to their absolute tolerance -- e.g.
        ``{"mean_abs_correlation": 0.02}`` for a coherence-matched null.  These
        are applied by rejection sampling on top of the per-gene matching.  Kept
        separate from :class:`MatchingProperty` precisely because they are not
        gene statistics and cannot be binned from
        :meth:`ExpressionMatrix.gene_stats`.
    """

    properties: tuple[MatchingProperty, ...] = ()
    n_bins: int = 10
    bins_by_property: Mapping[MatchingProperty, int] = field(default_factory=dict)
    nested: bool = True
    tolerance: Mapping[str, float] = field(default_factory=dict)
    with_replacement: bool = False
    exclude_candidate_genes: bool = False
    max_resample_attempts: int = 100
    set_level_constraints: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResolutionSpec:
    """Policy for mapping signature identifiers onto the matrix index.

    Attributes
    ----------
    target_namespace:
        Canonical namespace; :attr:`GeneIdNamespace.HGNC_SYMBOL` by default.
    missing_gene_policy:
        See :class:`MissingGenePolicy`.
    duplicate_handling:
        See :class:`DuplicateHandling`.
    min_overlap_fraction:
        Run aborts when ``matched / requested`` falls below this, regardless of
        ``missing_gene_policy``.  Guards against a namespace mismatch quietly
        turning a 200-gene signature into a 7-gene one.  Default 0.70, matching
        the refusal threshold fixed in ``docs/statistical-design.md``; between
        the floor and 1.0 the run proceeds on the observed subset and nulls are
        sized to :attr:`SignatureResolution.n_matched`.
    min_matched_genes:
        Absolute floor on the effective signature size.
    case_insensitive:
        Whether symbol matching upper-cases both sides before comparing.
    mapping_source:
        Identity/version of the annotation table to use; recorded in the result.
    """

    target_namespace: GeneIdNamespace = GeneIdNamespace.HGNC_SYMBOL
    missing_gene_policy: MissingGenePolicy = MissingGenePolicy.DROP
    duplicate_handling: DuplicateHandling = DuplicateHandling.MAX_VARIANCE
    min_overlap_fraction: float = 0.70
    min_matched_genes: int = 3
    case_insensitive: bool = True
    mapping_source: str | None = None


@dataclass(frozen=True, slots=True)
class NullSpec:
    """Complete description of one null model.

    Attributes
    ----------
    null_type:
        See :class:`NullType`.
    n_draws:
        Number of realisations.  The attainable p-value floor is
        ``1 / (n_draws + 1)``; a claim of ``p < 0.001`` needs at least 999 draws.

        CONTRACT AMENDMENT 1.1.0 (Fleet, wave-1 merge).  ``docs/statistical-design.md``
        Sec. 3.3 sets the default to 10000 and imposes a HARD FLOOR of 2000, below which
        an implementation MUST refuse to emit a p-value: the relative SE of the empirical
        p at p=0.05 is 9.7% at K=2000 and 13.8% at K=999.  The previous default of 1000
        sat below that refusal threshold.  Implementations MUST enforce
        ``n_draws >= MIN_DRAWS`` for gating nulls.  The supervised path may use 1000 with
        an explicit reduced-resolution note in the report.
    matching:
        Gene-set matching policy; ignored for the label-permutation null.
    params:
        Extra model-specific, JSON-serialisable options.
    """

    null_type: NullType
    n_draws: int = DEFAULT_DRAWS
    matching: MatchingSpec = field(default_factory=MatchingSpec)
    params: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NullTestConfig:
    """Everything needed to reproduce and interpret one null test.

    A statistic from this tool is meaningless without its configuration: the
    answer depends on the dataset, the scoring method, the matching policy and
    the metric.  Every :class:`NullTestResult` therefore embeds a complete
    config, and the report renders it.  Nothing that changes the number may live
    outside this object.

    Attributes
    ----------
    dataset, outcome:
        Identity of the inputs, without the bulk data.
    candidate:
        The candidate signature *as supplied*.  What was actually scored is
        :attr:`NullTestResult.resolution`.
    scoring, metric, direction_policy, alternative:
        The statistic and how its sign is treated.
    null:
        The null model.
    resolution:
        Identifier-mapping policy.
    cv:
        Cross-validation scheme; ``None`` for unsupervised scoring.
    seed:
        Root seed for the run.  All randomness -- gene sampling, label
        permutation, CV folds, any stochastic scoring -- is derived from it via
        :class:`numpy.random.SeedSequence` spawning, so a run is reproducible
        from this integer alone.
    contract_version:
        :data:`CONTRACT_VERSION` at run time.
    signull_version, code_revision:
        Package version and VCS revision that produced the result.
    created_at:
        ISO-8601 timestamp of the run.
    notes:
        Free text for the analyst.
    """

    dataset: DatasetDescriptor
    outcome: OutcomeDescriptor
    candidate: Signature
    scoring: ScoringSpec
    metric: MetricName
    null: NullSpec
    resolution: ResolutionSpec = field(default_factory=ResolutionSpec)
    direction_policy: DirectionPolicy = DirectionPolicy.SYMMETRIZED
    alternative: Alternative = Alternative.GREATER
    cv: CVSpec | None = None
    seed: int = 0
    contract_version: str = CONTRACT_VERSION
    signull_version: str | None = None
    code_revision: str | None = None
    created_at: str | None = None
    notes: str | None = None


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False, slots=True)
class NullTestResult:
    """The observed statistic against one null distribution, plus its config.

    Attributes
    ----------
    config:
        The complete :class:`NullTestConfig` that produced this result.
        Mandatory: a result without its config is not interpretable and must
        never be constructed.
    observed_statistic:
        Metric value for the candidate, after :class:`DirectionPolicy` has been
        applied.
    null_statistics:
        Metric values for the null draws, same policy applied, shape
        ``(n_valid_draws,)``.  Order matches :attr:`NullDraw.index` for draws
        that succeeded.
    p_value:
        Empirical p-value with the add-one correction
        ``(1 + #{null at least as extreme as observed}) / (n_valid_draws + 1)``,
        with "extreme" defined by :attr:`NullTestConfig.alternative`.  The
        add-one is required: an uncorrected zero would claim infinite evidence
        from a finite sample of draws.
    n_draws_requested, n_valid_draws:
        Draws asked for versus draws that produced a finite statistic.  A gap
        between them is a red flag and must appear in the report.
    resolution:
        How the candidate's identifiers mapped onto the matrix.
    diagnostics:
        Everything worth knowing that did not raise.
    elapsed_seconds:
        Wall-clock cost of the run.
    label:
        Short human-readable name of this arm when several results share a null
        type, e.g. ``"N1 matched, unadjusted"`` vs
        ``"N1 matched, proliferation-adjusted"``.  Results are keyed by
        ``(null type, scoring spec, label)`` in the report, not by null type
        alone.
    result_version:
        Schema version of this record, for stored-result migration.
    """

    config: NullTestConfig
    observed_statistic: float
    null_statistics: FloatArray
    p_value: float
    n_draws_requested: int
    n_valid_draws: int
    resolution: SignatureResolution
    diagnostics: tuple[Diagnostic, ...] = ()
    elapsed_seconds: float | None = None
    label: str | None = None
    result_version: str = CONTRACT_VERSION

    @property
    def null_mean(self) -> float:
        """Mean of the null distribution, in metric units."""
        return float(np.mean(self.null_statistics)) if self.null_statistics.size else float("nan")

    @property
    def null_sd(self) -> float:
        """Standard deviation of the null distribution (ddof=1), in metric units."""
        return (
            float(np.std(self.null_statistics, ddof=1))
            if self.null_statistics.size > 1
            else float("nan")
        )

    @property
    def standardized_effect(self) -> float:
        """``(observed - null_mean) / null_sd``; NaN when the null has no spread.

        A descriptive effect size, not a test statistic: the null is generally
        not Gaussian, so the p-value comes from the empirical distribution.
        """
        sd = self.null_sd
        if not np.isfinite(sd) or sd == 0.0:
            return float("nan")
        return (self.observed_statistic - self.null_mean) / sd

    @property
    def p_value_floor(self) -> float:
        """Smallest p-value attainable at this number of draws."""
        return 1.0 / (self.n_valid_draws + 1) if self.n_valid_draws >= 0 else float("nan")

    @property
    def at_resolution_floor(self) -> bool:
        """``True`` when the p-value is pinned at the floor, so only bounded."""
        return bool(np.isclose(self.p_value, self.p_value_floor))


@dataclass(frozen=True, eq=False, slots=True)
class EvidenceSummary:
    """All null tests run for one candidate on one dataset.

    A defensible answer needs both nulls: the random gene-set null says whether
    the gene set is special, the label-permutation null says whether the cohort
    carries any signal at all.  They answer different questions and neither
    substitutes for the other, so they are reported side by side and this object
    deliberately exposes no single verdict field.

    Attributes
    ----------
    candidate:
        The signature as supplied.
    results:
        One :class:`NullTestResult` per null model, in run order.
    caveats:
        Interpretation limits attached to this particular run (unmatched null,
        low overlap, small cohort, single dataset, ...).  Rendered verbatim next
        to the numbers.
    diagnostics:
        Run-level conditions not owned by any single test.
    created_at:
        ISO-8601 timestamp of the run.
    """

    candidate: Signature
    results: tuple[NullTestResult, ...]
    caveats: tuple[str, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    created_at: str | None = None

    def results_for(self, null_type: NullType) -> tuple[NullTestResult, ...]:
        """All results produced by ``null_type``.

        More than one is normal: e.g. an unadjusted and a
        dominant-axis-adjusted arm of the same gene-set null, distinguished by
        :attr:`NullTestResult.label`.
        """
        return tuple(r for r in self.results if r.config.null.null_type is null_type)

    def result_for(self, null_type: NullType) -> NullTestResult | None:
        """First result produced by ``null_type``, or ``None`` if not run."""
        results = self.results_for(null_type)
        return results[0] if results else None
