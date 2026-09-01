"""Mapping signature identifiers onto a matrix index -- the single conversion point.

``docs/architecture.md`` Sec. 4: identifier mismatch is the dominant silent
failure mode of this class of tool, because a namespace mismatch turns a
200-gene signature into a 7-gene one while every downstream number stays
superficially plausible.  Conversion therefore happens **exactly once**, here,
against a specific matrix, and every requested identifier ends up in exactly one
of :attr:`~signull.types.SignatureResolution.matched`, ``missing`` or
``unmapped``.

Nothing in this module is silent: the returned
:class:`~signull.types.SignatureResolution` is embedded in the result and
rendered by the report even when the overlap is perfect.

Public surface
--------------
:class:`AnnotationTable`
    A pinned source-identifier -> canonical-identifier map.  Ambiguous entries
    are dropped rather than guessed (Sec. 4 rule 3).
:class:`MatrixIndexResolver`
    Satisfies :class:`~signull.types.GeneIdResolver`.
:func:`resolve_signature`
    One-call convenience wrapper.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from signull.types import (
    DuplicateHandling,
    ExpressionMatrix,
    GeneId,
    GeneIdNamespace,
    MissingGenePolicy,
    ResolutionSpec,
    Signature,
    SignatureResolution,
)

from .diagnostics import DiagnosticCode, DiagnosticLog

__all__ = [
    "AnnotationTable",
    "MatrixIndexResolver",
    "load_annotation_table",
    "resolve_signature",
    "strip_version_suffix",
]

#: Namespaces whose identifiers are compared case-insensitively when
#: ``ResolutionSpec.case_insensitive`` is set.  Ensembl and Entrez identifiers
#: are canonical in one case already, and probe identifiers are platform
#: literals, so folding them is not obviously safe -- symbols are.
_CASE_FOLDABLE: Final[frozenset[GeneIdNamespace]] = frozenset(
    {GeneIdNamespace.HGNC_SYMBOL, GeneIdNamespace.UNKNOWN}
)

_ENSEMBL_VERSIONED: Final[re.Pattern[str]] = re.compile(
    r"^(ENS[A-Z]*[GTP]\d{6,})\.\d+$", re.IGNORECASE
)


def strip_version_suffix(identifier: str) -> str:
    """Drop an Ensembl version suffix: ``ENSG00000141510.14 -> ENSG00000141510``.

    Sec. 4 rule 1.  Identifiers that do not look like versioned Ensembl
    accessions are returned unchanged, so this is safe to apply to any string.
    """
    match = _ENSEMBL_VERSIONED.match(identifier)
    return match.group(1) if match else identifier


@dataclass(frozen=True, slots=True)
class AnnotationTable:
    """A pinned map from input identifiers to canonical identifiers.

    Attributes
    ----------
    mapping:
        ``source id -> canonical id``.  Built by :meth:`from_pairs` /
        :func:`load_annotation_table`, which drop ambiguous sources (Sec. 4
        rule 3) into :attr:`ambiguous` instead of picking one.
    ambiguous:
        Source identifiers that mapped to more than one canonical identifier
        and were therefore dropped, with the candidates they mapped to.
    source_namespace, target_namespace:
        Namespaces this table bridges.
    version:
        Identity and version of the annotation, e.g.
        ``"HGNC complete set 2024-07-01"``.  A result is not reproducible
        without it, so a table built without one triggers an
        ``unpinned_annotation_source`` warning at resolution time.
    """

    mapping: Mapping[str, str]
    ambiguous: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    source_namespace: GeneIdNamespace = GeneIdNamespace.UNKNOWN
    target_namespace: GeneIdNamespace = GeneIdNamespace.HGNC_SYMBOL
    version: str | None = None

    @classmethod
    def from_pairs(
        cls,
        pairs: Iterable[tuple[str, str]],
        *,
        source_namespace: GeneIdNamespace = GeneIdNamespace.UNKNOWN,
        target_namespace: GeneIdNamespace = GeneIdNamespace.HGNC_SYMBOL,
        version: str | None = None,
    ) -> "AnnotationTable":
        """Build from ``(source, target)`` pairs, dropping ambiguous sources.

        Repeated identical pairs are collapsed silently -- they carry no
        ambiguity.  A source with two *different* targets is ambiguous and is
        dropped from :attr:`mapping`.
        """
        collected: dict[str, set[str]] = defaultdict(set)
        for source, target in pairs:
            source = str(source).strip()
            target = str(target).strip()
            if not source or not target:
                continue
            collected[source].add(target)

        mapping: dict[str, str] = {}
        ambiguous: dict[str, tuple[str, ...]] = {}
        for source, targets in collected.items():
            if len(targets) == 1:
                mapping[source] = next(iter(targets))
            else:
                ambiguous[source] = tuple(sorted(targets))
        return cls(
            mapping=mapping,
            ambiguous=ambiguous,
            source_namespace=source_namespace,
            target_namespace=target_namespace,
            version=version,
        )

    def __len__(self) -> int:
        return len(self.mapping)


def load_annotation_table(
    path: Path | str,
    *,
    source_column: str,
    target_column: str,
    sep: str = "\t",
    source_namespace: GeneIdNamespace = GeneIdNamespace.PROBE_ID,
    target_namespace: GeneIdNamespace = GeneIdNamespace.HGNC_SYMBOL,
    version: str | None = None,
) -> AnnotationTable:
    """Read a two-column annotation from a delimited file.

    Parameters
    ----------
    version:
        Defaults to ``"file:<name>"``, which pins the *file* but not its
        content version; pass the real annotation release where you know it.
    """
    path = Path(path)
    frame = pd.read_csv(path, sep=sep, dtype=str)
    for column in (source_column, target_column):
        if column not in frame.columns:
            raise ValueError(
                f"{path.name} has no column {column!r}; found "
                f"{list(frame.columns)[:10]!r}"
            )
    pairs = zip(
        frame[source_column].fillna("").tolist(),
        frame[target_column].fillna("").tolist(),
        strict=True,
    )
    return AnnotationTable.from_pairs(
        pairs,
        source_namespace=source_namespace,
        target_namespace=target_namespace,
        version=version or f"file:{path.name}",
    )


@dataclass(frozen=True, slots=True)
class MatrixIndexResolver:
    """Resolve a signature against one matrix index.  Implements ``GeneIdResolver``.

    The resolution target is always the **matrix index namespace**, because the
    matrix index is what nulls are drawn from and what the scorer subsets.  When
    ``spec.target_namespace`` names a different namespace, that is recorded as a
    diagnostic rather than obeyed: converting the signature into a namespace the
    matrix does not use would match nothing.

    Attributes
    ----------
    annotation:
        Optional pinned :class:`AnnotationTable` bridging the signature's
        namespace to the matrix's.  Without one, only identifiers already in the
        matrix namespace can match, and cross-namespace failures are reported as
        ``unmapped`` rather than ``missing``.
    """

    annotation: AnnotationTable | None = None

    def resolve(
        self,
        signature: Signature,
        matrix: ExpressionMatrix,
        spec: ResolutionSpec,
        log: DiagnosticLog | None = None,
    ) -> tuple[Signature, SignatureResolution]:
        """Return ``(resolved_signature, resolution)``.

        The resolved signature carries only matched genes, **in matrix row
        order**, in the matrix namespace, with weights subset (and, for
        many-to-one collapses, combined per ``spec.duplicate_handling``).

        Raises
        ------
        ValueError
            ``spec.missing_gene_policy is RAISE`` and something did not match;
            overlap below ``spec.min_overlap_fraction``; fewer than
            ``spec.min_matched_genes`` matched genes;
            ``spec.duplicate_handling is RAISE`` and two requested identifiers
            collapsed onto one matrix row.
        """
        log = log if log is not None else DiagnosticLog()
        target_namespace = matrix.namespace
        requested = tuple(str(g) for g in signature.genes)
        if not requested:
            raise ValueError(f"signature {signature.name!r} has no identifiers")

        if spec.target_namespace is not target_namespace:
            log.record(
                DiagnosticCode.PROBE_LEVEL_NAMESPACE_RETAINED,
                f"ResolutionSpec asks for {spec.target_namespace.value!r} but the "
                f"matrix index is {target_namespace.value!r}; resolving against the "
                "matrix index, which is the namespace nulls are drawn in",
                context={
                    "requested_namespace": spec.target_namespace.value,
                    "matrix_namespace": target_namespace.value,
                },
            )

        fold = bool(spec.case_insensitive) and target_namespace in _CASE_FOLDABLE
        index_lookup = _index_lookup(matrix.gene_ids, fold=fold)
        needs_translation = signature.namespace is not target_namespace

        if needs_translation and self.annotation is None:
            log.warn(
                DiagnosticCode.SIGNATURE_NAMESPACE_MISMATCH,
                f"signature {signature.name!r} declares namespace "
                f"{signature.namespace.value!r} but the matrix index is "
                f"{target_namespace.value!r} and no annotation table was supplied; "
                "only identifiers that happen to be literal matrix row names can match",
                context={
                    "signature_namespace": signature.namespace.value,
                    "matrix_namespace": target_namespace.value,
                },
            )
        if self.annotation is not None and not self.annotation.version:
            log.warn(
                DiagnosticCode.UNPINNED_ANNOTATION_SOURCE,
                "annotation table carries no version; the resolution, and therefore "
                "the result, is not reproducible from the recorded config alone",
            )

        matched_rows: dict[int, list[int]] = {}  # matrix row -> requested positions
        missing: list[GeneId] = []
        unmapped: list[GeneId] = []
        aliased: dict[GeneId, GeneId] = {}
        version_stripped = 0

        for position, raw in enumerate(requested):
            probe = raw
            stripped = strip_version_suffix(probe)
            if stripped != probe:
                version_stripped += 1
                aliased[raw] = stripped
                probe = stripped

            row = index_lookup.get(_fold(probe, fold))
            if row is None and self.annotation is not None:
                canonical = self.annotation.mapping.get(probe)
                if canonical is None and probe in self.annotation.ambiguous:
                    log.record(
                        DiagnosticCode.AMBIGUOUS_MAPPING_DROPPED,
                        f"identifier {raw!r} maps to several canonical identifiers "
                        f"{self.annotation.ambiguous[probe]!r}; dropped rather than guessed",
                        context={"identifier": raw},
                    )
                    unmapped.append(raw)
                    continue
                if canonical is not None:
                    row = index_lookup.get(_fold(canonical, fold))
                    if canonical != probe:
                        aliased[raw] = canonical
            if row is None:
                if needs_translation and (
                    self.annotation is None or probe not in self.annotation.mapping
                ):
                    unmapped.append(raw)
                else:
                    missing.append(raw)
                continue
            matched_rows.setdefault(row, []).append(position)

        if version_stripped:
            log.record(
                DiagnosticCode.VERSION_SUFFIX_STRIPPED,
                f"stripped Ensembl version suffixes from {version_stripped} identifier(s) "
                "before mapping",
                context={"n_stripped": version_stripped},
            )
        if not needs_translation and not aliased:
            log.record(
                DiagnosticCode.IDENTITY_RESOLUTION,
                f"signature {signature.name!r} is already in the matrix namespace "
                f"({target_namespace.value!r}); no annotation table was consulted",
            )

        gene_ids = tuple(matrix.gene_ids)
        rows_in_order = sorted(matched_rows)
        matched = tuple(gene_ids[row] for row in rows_in_order)

        collapsed = _collapsed_map(
            matrix=matrix,
            matched_rows=matched_rows,
            rows_in_order=rows_in_order,
            requested=requested,
            gene_ids=gene_ids,
            spec=spec,
            signature_name=signature.name,
            log=log,
        )

        resolution = SignatureResolution(
            requested=requested,
            matched=matched,
            missing=tuple(missing),
            unmapped=tuple(unmapped),
            collapsed=collapsed,
            aliased=dict(aliased),
            source_namespace=signature.namespace,
            target_namespace=target_namespace,
            duplicate_handling=spec.duplicate_handling,
            mapping_source=(
                spec.mapping_source
                or (self.annotation.version if self.annotation is not None else None)
            ),
        )

        _enforce_policies(resolution, spec, signature.name, log)
        weights = _resolved_weights(
            signature, matched_rows, rows_in_order, spec, signature.name
        )
        _flag_constant_genes(matrix, matched, signature.name, log)

        resolved = Signature(
            genes=matched,
            name=signature.name,
            namespace=target_namespace,
            origin=signature.origin,
            weights=weights,
            provenance=signature.provenance,
            description=signature.description,
        )
        return resolved, resolution


def resolve_signature(
    signature: Signature,
    matrix: ExpressionMatrix,
    spec: ResolutionSpec | None = None,
    *,
    annotation: AnnotationTable | None = None,
    log: DiagnosticLog | None = None,
) -> tuple[Signature, SignatureResolution]:
    """Convenience wrapper around :class:`MatrixIndexResolver`.

    ``spec`` defaults to :class:`~signull.types.ResolutionSpec`'s own defaults:
    drop missing genes, collapse duplicates by max variance, refuse below 70 %
    overlap.
    """
    resolver = MatrixIndexResolver(annotation=annotation)
    return resolver.resolve(signature, matrix, spec or ResolutionSpec(), log=log)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fold(identifier: str, fold: bool) -> str:
    return identifier.upper() if fold else identifier


def _index_lookup(gene_ids: Sequence[GeneId], *, fold: bool) -> dict[str, int]:
    """Map (optionally case-folded) matrix identifier -> row index.

    On a case-fold collision the *first* row wins and the exact-case spelling
    keeps its own entry, so an exact match is never shadowed by a folded one.
    """
    lookup: dict[str, int] = {}
    if fold:
        for row, gene in enumerate(gene_ids):
            lookup.setdefault(gene.upper(), row)
    else:
        for row, gene in enumerate(gene_ids):
            lookup.setdefault(gene, row)
    return lookup


def _collapsed_map(
    *,
    matrix: ExpressionMatrix,
    matched_rows: Mapping[int, list[int]],
    rows_in_order: Sequence[int],
    requested: Sequence[GeneId],
    gene_ids: Sequence[GeneId],
    spec: ResolutionSpec,
    signature_name: str,
    log: DiagnosticLog,
) -> dict[GeneId, tuple[GeneId, ...]]:
    """Collapses seen at resolution, plus the ones the matrix carries from ingest."""
    collapsed: dict[GeneId, tuple[GeneId, ...]] = {}

    ingest_map = dict(getattr(matrix, "collapse_map", {}) or {})
    for row in rows_in_order:
        gene = gene_ids[row]
        sources = ingest_map.get(gene)
        if sources and len(sources) > 1:
            collapsed[gene] = tuple(sources)

    signature_side = {
        gene_ids[row]: tuple(requested[p] for p in positions)
        for row, positions in matched_rows.items()
        if len(positions) > 1
    }
    if signature_side:
        if spec.duplicate_handling is DuplicateHandling.RAISE:
            example = next(iter(signature_side.items()))
            raise ValueError(
                f"signature {signature_name!r}: {len(signature_side)} matrix row(s) "
                f"are claimed by more than one requested identifier, e.g. {example[0]!r} "
                f"<- {example[1]!r}; duplicate_handling is RAISE"
            )
        log.warn(
            DiagnosticCode.DUPLICATE_PROBES_COLLAPSED,
            f"signature {signature_name!r}: {len(signature_side)} matrix row(s) were "
            f"claimed by several requested identifiers and were collapsed under "
            f"{spec.duplicate_handling.value!r}",
            context={
                "n_collapsed": len(signature_side),
                "examples": {k: list(v) for k, v in list(signature_side.items())[:5]},
            },
        )
        for gene, sources in signature_side.items():
            collapsed[gene] = tuple(dict.fromkeys(collapsed.get(gene, ()) + sources))
    return collapsed


def _resolved_weights(
    signature: Signature,
    matched_rows: Mapping[int, list[int]],
    rows_in_order: Sequence[int],
    spec: ResolutionSpec,
    signature_name: str,
) -> tuple[float, ...] | None:
    """Subset the candidate's weights onto the matched rows, in matrix row order.

    When several requested identifiers collapsed onto one row, their weights are
    combined by the same rule as the rows: ``MEAN`` averages them, every other
    rule keeps the first.  ``RAISE`` has already raised in
    :func:`_collapsed_map`.
    """
    if signature.weights is None:
        return None
    weights = np.asarray(signature.weights, dtype=np.float64)
    if weights.shape[0] != len(signature.genes):
        raise ValueError(
            f"signature {signature_name!r} has {weights.shape[0]} weights for "
            f"{len(signature.genes)} genes"
        )
    out: list[float] = []
    for row in rows_in_order:
        positions = matched_rows[row]
        if len(positions) == 1 or spec.duplicate_handling is not DuplicateHandling.MEAN:
            out.append(float(weights[positions[0]]))
        else:
            out.append(float(weights[positions].mean()))
    return tuple(out)


def _enforce_policies(
    resolution: SignatureResolution,
    spec: ResolutionSpec,
    signature_name: str,
    log: DiagnosticLog,
) -> None:
    """Tier-1 floors of ``docs/architecture.md`` Sec. 4 rule 8, then tier-2 warnings."""
    n_lost = len(resolution.missing) + len(resolution.unmapped)
    if n_lost and spec.missing_gene_policy is MissingGenePolicy.RAISE:
        raise ValueError(
            f"signature {signature_name!r}: {len(resolution.missing)} identifier(s) "
            f"absent from the matrix and {len(resolution.unmapped)} untranslatable, "
            "and missing_gene_policy is RAISE; e.g. "
            f"{(resolution.missing + resolution.unmapped)[:10]!r}"
        )

    overlap = resolution.overlap_fraction
    if overlap < spec.min_overlap_fraction:
        raise ValueError(
            f"signature {signature_name!r} resolves to {resolution.n_matched}/"
            f"{resolution.n_requested} genes (overlap {overlap:.3f}), below the "
            f"floor of {spec.min_overlap_fraction:.2f} "
            "(docs/statistical-design.md Sec. 8 F5). A namespace mismatch is the "
            "usual cause; fix the identifiers rather than lowering the floor."
        )
    if resolution.n_matched < spec.min_matched_genes:
        raise ValueError(
            f"signature {signature_name!r} resolves to {resolution.n_matched} gene(s), "
            f"below min_matched_genes={spec.min_matched_genes}"
        )
    if n_lost:
        log.warn(
            DiagnosticCode.LOW_SIGNATURE_OVERLAP,
            f"signature {signature_name!r} resolves to {resolution.n_matched}/"
            f"{resolution.n_requested} genes (overlap {overlap:.3f}); nulls are "
            f"size-matched to the effective size {resolution.n_matched}, not the "
            "nominal one",
            context={
                "n_requested": resolution.n_requested,
                "n_matched": resolution.n_matched,
                "n_missing": len(resolution.missing),
                "n_unmapped": len(resolution.unmapped),
                "overlap_fraction": float(overlap),
            },
        )


def _flag_constant_genes(
    matrix: ExpressionMatrix,
    matched: Sequence[GeneId],
    signature_name: str,
    log: DiagnosticLog,
) -> None:
    """Warn about zero-variance matched genes: they have no z-score (Sec. 2.1)."""
    if not matched:
        return
    table = matrix.gene_stats().table
    sd = table.loc[list(matched), "sd"].to_numpy(dtype=np.float64)
    constant = [gene for gene, value in zip(matched, sd, strict=True) if value <= 0.0]
    if constant:
        log.warn(
            DiagnosticCode.ZERO_VARIANCE_GENES_IN_SIGNATURE,
            f"signature {signature_name!r} has {len(constant)} gene(s) with zero "
            f"variance across the analysis cohort, e.g. {constant[:5]!r}; they carry "
            "no z-score and are not eligible background either",
            context={"n_constant": len(constant), "examples": constant[:10]},
        )
