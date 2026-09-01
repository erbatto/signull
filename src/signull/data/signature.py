"""Constructing :class:`~signull.types.Signature` objects from gene/probe lists.

A signature arrives as *identifiers in some namespace the user chose*.  This
module's only job is to turn that into a well-formed :class:`Signature` with the
declared namespace recorded honestly.  It performs **no** namespace conversion:
that happens exactly once, in :mod:`signull.data.resolve`, against a specific
matrix (``docs/architecture.md`` Sec. 4).

Supported inputs
----------------
* an in-memory sequence of identifiers (:func:`signature_from_ids`);
* a plain text list, one identifier per line, ``#`` comments allowed
  (:func:`load_signature_list`);
* GMT, the MSigDB format ``name <tab> description <tab> gene ...``
  (:func:`load_signatures_gmt`);
* a column of a delimited table (:func:`load_signature_table`).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

from signull.types import (
    GeneId,
    GeneIdNamespace,
    Provenance,
    Signature,
    SignatureOrigin,
)

from .diagnostics import DiagnosticCode, DiagnosticLog
from .provenance import file_provenance

__all__ = [
    "signature_from_ids",
    "load_signature_list",
    "load_signatures_gmt",
    "load_signature_table",
    "looks_excel_mangled",
]

_MONTHS: Final[str] = "JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC"

#: Patterns that a human gene symbol never matches but an Excel-mangled one does.
#: ``SEPT7`` becomes ``7-Sep``/``Sep-07``/``2007-09-07`` depending on locale;
#: roughly 40 human symbols are affected (``docs/architecture.md`` Sec. 4 rule 5).
_EXCEL_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(rf"^\d{{1,2}}[-/](?:{_MONTHS})[A-Z]*$", re.IGNORECASE),
    re.compile(rf"^(?:{_MONTHS})[A-Z]*[-/]\d{{1,2}}$", re.IGNORECASE),
    re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ].*)?$"),
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$"),
    re.compile(r"^\d+(?:\.\d+)?E\+\d+$", re.IGNORECASE),
)


def looks_excel_mangled(identifier: str) -> bool:
    """``True`` when ``identifier`` looks like a date or scientific-notation cell.

    Spreadsheet round-tripping silently rewrites ``SEPT7`` to ``7-Sep`` and
    ``MARCH1`` to ``1-Mar``.  Such an identifier will never match a matrix index,
    so the signature quietly shrinks -- exactly the failure mode this package is
    built to make loud.
    """
    token = identifier.strip()
    return any(pattern.match(token) for pattern in _EXCEL_PATTERNS)


def signature_from_ids(
    identifiers: Iterable[GeneId],
    *,
    name: str,
    namespace: GeneIdNamespace = GeneIdNamespace.HGNC_SYMBOL,
    origin: SignatureOrigin = SignatureOrigin.CANDIDATE,
    weights: Sequence[float] | None = None,
    provenance: Provenance | None = None,
    description: str | None = None,
    log: DiagnosticLog | None = None,
) -> Signature:
    """Build a :class:`~signull.types.Signature` from raw identifiers.

    Whitespace is stripped and blank entries removed.  Duplicates are removed
    keeping the *first* occurrence (the :class:`Signature` contract forbids
    them) and the removal is recorded.  No case folding, no version stripping,
    no alias lookup: those are resolution-time concerns and depend on the
    matrix.

    Parameters
    ----------
    namespace:
        The namespace the caller believes the identifiers are in.  Declared, not
        sniffed -- a wrong declaration surfaces as a zero-overlap resolution
        error rather than as a silently shrunken signature.
    weights:
        Optional per-identifier weights, aligned element-wise to ``identifiers``
        *before* de-duplication.  Sign encodes direction, magnitude emphasis.

    Raises
    ------
    ValueError
        Empty signature, or ``weights`` of the wrong length.
    """
    log = log if log is not None else DiagnosticLog()

    raw = [str(value).strip() for value in identifiers]
    weight_list = None if weights is None else [float(w) for w in weights]
    if weight_list is not None and len(weight_list) != len(raw):
        raise ValueError(
            f"weights has length {len(weight_list)} but {len(raw)} identifiers "
            "were supplied; weights are aligned element-wise to the gene list"
        )

    genes: list[GeneId] = []
    kept_weights: list[float] = []
    seen: set[GeneId] = set()
    blank = 0
    duplicates: list[GeneId] = []
    for position, value in enumerate(raw):
        if not value:
            blank += 1
            continue
        if value in seen:
            duplicates.append(value)
            continue
        seen.add(value)
        genes.append(value)
        if weight_list is not None:
            kept_weights.append(weight_list[position])

    if duplicates or blank:
        log.record(
            DiagnosticCode.DUPLICATE_SIGNATURE_IDS_DROPPED,
            f"signature {name!r}: dropped {len(duplicates)} duplicate and {blank} "
            "blank identifier(s) at construction",
            context={
                "n_duplicates": len(duplicates),
                "n_blank": blank,
                "examples": sorted(set(duplicates))[:10],
            },
        )

    if not genes:
        raise ValueError(
            f"signature {name!r} contains no usable identifiers "
            f"(received {len(raw)} entries, all blank or duplicate)"
        )

    mangled = [g for g in genes if looks_excel_mangled(g)]
    if mangled:
        log.warn(
            DiagnosticCode.SIGNATURE_LOOKS_EXCEL_MANGLED,
            f"signature {name!r} contains {len(mangled)} identifier(s) that look "
            f"like spreadsheet-mangled dates, e.g. {mangled[:5]!r}; symbols such "
            "as SEPT7 and MARCH1 are rewritten by Excel and will not match any "
            "matrix index",
            context={"n_suspect": len(mangled), "examples": mangled[:10]},
        )

    return Signature(
        genes=tuple(genes),
        name=name,
        namespace=namespace,
        origin=origin,
        weights=tuple(kept_weights) if weight_list is not None else None,
        provenance=provenance,
        description=description,
    )


def load_signature_list(
    path: Path | str,
    *,
    name: str | None = None,
    namespace: GeneIdNamespace = GeneIdNamespace.HGNC_SYMBOL,
    origin: SignatureOrigin = SignatureOrigin.CANDIDATE,
    description: str | None = None,
    source: str | None = None,
    citation: str | None = None,
    log: DiagnosticLog | None = None,
) -> Signature:
    """Load a plain gene/probe list: one identifier per line.

    Blank lines and lines whose first non-space character is ``#`` are ignored,
    so a list can carry a provenance header.  The file's sha256 is recorded in
    the returned signature's provenance.

    Parameters
    ----------
    name:
        Defaults to the file stem.
    namespace:
        Declared namespace of the identifiers in the file.  For the GPL96
        benchmark cohorts a probeset list must declare
        :attr:`~signull.types.GeneIdNamespace.PROBE_ID`.
    """
    path = Path(path)
    log = log if log is not None else DiagnosticLog()
    text = path.read_text(encoding="utf-8")
    identifiers = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    provenance = file_provenance(
        path,
        source=source or f"file:{path.name}",
        citation=citation,
        notes=f"plain identifier list, declared namespace {namespace.value!r}",
        log=log,
    )
    return signature_from_ids(
        identifiers,
        name=name or path.stem,
        namespace=namespace,
        origin=origin,
        provenance=provenance,
        description=description,
        log=log,
    )


def load_signatures_gmt(
    path: Path | str,
    *,
    namespace: GeneIdNamespace = GeneIdNamespace.HGNC_SYMBOL,
    origin: SignatureOrigin = SignatureOrigin.CANDIDATE,
    source: str | None = None,
    log: DiagnosticLog | None = None,
) -> dict[str, Signature]:
    """Load every set in a GMT file, keyed by set name.

    GMT is tab-delimited: ``set_name <tab> description <tab> gene1 <tab> ...``.
    Sets are returned in file order (``dict`` preserves insertion order), so
    iteration is deterministic.

    Raises
    ------
    ValueError
        A malformed line (fewer than three fields) or a duplicated set name.
    """
    path = Path(path)
    log = log if log is not None else DiagnosticLog()
    provenance = file_provenance(
        path,
        source=source or f"file:{path.name}",
        notes=f"GMT, declared namespace {namespace.value!r}",
        log=log,
    )

    signatures: dict[str, Signature] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 3:
            raise ValueError(
                f"{path.name}:{line_number}: GMT lines need at least "
                f"name, description and one gene; got {len(fields)} field(s)"
            )
        set_name, set_description, *genes = fields
        set_name = set_name.strip()
        if set_name in signatures:
            raise ValueError(
                f"{path.name}:{line_number}: duplicate gene-set name {set_name!r}"
            )
        signatures[set_name] = signature_from_ids(
            genes,
            name=set_name,
            namespace=namespace,
            origin=origin,
            provenance=provenance,
            description=set_description.strip() or None,
            log=log,
        )
    if not signatures:
        raise ValueError(f"{path.name} contains no gene sets")
    return signatures


def load_signature_table(
    path: Path | str,
    *,
    name: str | None = None,
    gene_column: str = "gene",
    weight_column: str | None = None,
    sep: str = "\t",
    namespace: GeneIdNamespace = GeneIdNamespace.HGNC_SYMBOL,
    origin: SignatureOrigin = SignatureOrigin.CANDIDATE,
    source: str | None = None,
    log: DiagnosticLog | None = None,
) -> Signature:
    """Load a signature from one column of a delimited table.

    Parameters
    ----------
    weight_column:
        Optional column of per-gene weights.  Sign encodes direction; a signed
        signature must be evaluated with
        :attr:`~signull.types.DirectionPolicy.AS_GIVEN` only when its nulls
        inherit the same signed structure.
    """
    import pandas as pd  # local import: keeps the module light for list-only use

    path = Path(path)
    log = log if log is not None else DiagnosticLog()
    frame = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    if gene_column not in frame.columns:
        raise ValueError(
            f"{path.name} has no column {gene_column!r}; available: "
            f"{list(frame.columns)[:20]!r}"
        )
    if weight_column is not None and weight_column not in frame.columns:
        raise ValueError(
            f"{path.name} has no weight column {weight_column!r}; available: "
            f"{list(frame.columns)[:20]!r}"
        )
    weights = (
        [float(value) for value in frame[weight_column]]
        if weight_column is not None
        else None
    )
    provenance = file_provenance(
        path,
        source=source or f"file:{path.name}",
        notes=f"table column {gene_column!r}, declared namespace {namespace.value!r}",
        log=log,
    )
    return signature_from_ids(
        list(frame[gene_column]),
        name=name or path.stem,
        namespace=namespace,
        origin=origin,
        weights=weights,
        provenance=provenance,
        log=log,
    )
