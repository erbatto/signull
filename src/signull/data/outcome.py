"""Loading a binary patient endpoint into :class:`~signull.types.BinaryOutcome`.

Sample order is *the table's row order* at load time; it becomes irrelevant once
:meth:`~signull.data.matrix.DenseExpressionMatrix.align_to` imposes the matrix's
column order.  Missing endpoints are dropped and reported, never encoded as
``False`` (which would silently move patients into the negative class).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from signull.types import BinaryOutcome, Provenance, SampleId

from .diagnostics import DiagnosticCode, DiagnosticLog
from .provenance import file_provenance

__all__ = ["binary_outcome_from_frame", "load_outcome_tsv"]

#: Strings that count as a missing endpoint in a delimited text file.  ``"NA"``
#: is the R convention the GEO export uses.
_NA_STRINGS: tuple[str, ...] = ("", "NA", "N/A", "na", "nan", "NaN", "null", "None", ".")

_TRUE_TOKENS: frozenset[str] = frozenset({"1", "1.0", "true", "yes", "y", "t"})
_FALSE_TOKENS: frozenset[str] = frozenset({"0", "0.0", "false", "no", "n", "f"})


def binary_outcome_from_frame(
    frame: pd.DataFrame,
    *,
    name: str,
    sample_id_column: str = "sample_id",
    outcome_column: str = "outcome",
    label_column: str | None = None,
    positive_label: str = "positive",
    negative_label: str = "negative",
    provenance: Provenance | None = None,
    log: DiagnosticLog | None = None,
) -> BinaryOutcome:
    """Build a :class:`~signull.types.BinaryOutcome` from a tabular frame.

    Parameters
    ----------
    frame:
        One row per sample.  Never mutated: all work is done on extracted
        arrays, so the caller's frame is untouched (copy-on-write safe).
    name:
        Endpoint name, e.g. ``"pathologic_complete_response"``.
    sample_id_column, outcome_column:
        Column names.  ``outcome_column`` may hold ``{0, 1}``, ``{True, False}``
        or the usual textual spellings (``yes``/``no``, ``true``/``false``);
        anything else raises.
    label_column:
        Optional column carrying the human-readable class name for each row
        (e.g. ``pCR`` / ``RD``).  When given, ``positive_label`` and
        ``negative_label`` are *derived* from it and the mapping is checked for
        consistency, so the label printed in the report is the one in the file
        rather than a caller-supplied guess.
    positive_label, negative_label:
        Used when ``label_column`` is ``None``.
    log:
        Optional diagnostic log.

    Returns
    -------
    BinaryOutcome
        ``labels[i] is True`` exactly when sample ``i`` is in the positive class.

    Raises
    ------
    ValueError
        Missing column, duplicate sample identifiers, un-coercible outcome
        values, an inconsistent ``label_column``, or a degenerate endpoint (one
        class empty).
    """
    log = log if log is not None else DiagnosticLog()

    for column in (sample_id_column, outcome_column):
        if column not in frame.columns:
            raise ValueError(
                f"outcome table has no column {column!r}; available columns: "
                f"{list(frame.columns)[:20]!r}"
            )
    if label_column is not None and label_column not in frame.columns:
        raise ValueError(
            f"outcome table has no label column {label_column!r}; available "
            f"columns: {list(frame.columns)[:20]!r}"
        )

    sample_ids_raw = [str(value).strip() for value in frame[sample_id_column]]
    raw_outcome = frame[outcome_column]
    raw_labels = frame[label_column] if label_column is not None else None

    keep: list[int] = []
    dropped: list[SampleId] = []
    for i, value in enumerate(raw_outcome):
        if _is_missing(value):
            dropped.append(sample_ids_raw[i])
        else:
            keep.append(i)

    if dropped:
        log.warn(
            DiagnosticCode.OUTCOME_MISSING_DROPPED,
            f"{len(dropped)} sample(s) have a missing {name!r} endpoint and were "
            "dropped; a missing endpoint is never encoded as the negative class",
            context={
                "n_dropped": len(dropped),
                "n_retained": len(keep),
                "examples": dropped[:10],
            },
        )

    if not keep:
        raise ValueError(
            f"every row of the outcome table has a missing {outcome_column!r} value"
        )

    sample_ids = tuple(sample_ids_raw[i] for i in keep)
    _reject_duplicate_samples(sample_ids)

    labels = np.fromiter(
        (_coerce_bool(raw_outcome.iloc[i], sample_ids_raw[i]) for i in keep),
        dtype=bool,
        count=len(keep),
    )

    if raw_labels is not None:
        positive_label, negative_label = _derive_class_names(
            [str(raw_labels.iloc[i]).strip() for i in keep], labels, label_column
        )

    labels.setflags(write=False)
    outcome = BinaryOutcome(
        sample_ids=sample_ids,
        labels=labels,
        name=name,
        positive_label=positive_label,
        negative_label=negative_label,
        provenance=provenance,
    )

    if outcome.is_degenerate:
        raise ValueError(
            f"outcome {name!r} is degenerate: {outcome.n_positive} positive / "
            f"{outcome.n_negative} negative over {outcome.n_samples} samples; "
            "no discrimination metric is defined"
        )

    return outcome


def load_outcome_tsv(
    path: Path | str,
    *,
    name: str,
    sample_id_column: str = "sample_id",
    outcome_column: str = "outcome",
    label_column: str | None = None,
    positive_label: str = "positive",
    negative_label: str = "negative",
    sep: str = "\t",
    source: str | None = None,
    identifier: str | None = None,
    expected_checksum: str | None = None,
    provenance: Provenance | None = None,
    log: DiagnosticLog | None = None,
) -> BinaryOutcome:
    """Load a delimited outcome table from disk.

    Duplicate column names in the file (the GEO exports have two ``sample_id``
    columns) are de-duplicated by pandas with a ``.1`` suffix, so
    ``sample_id_column="sample_id"`` always addresses the *first* occurrence.

    Parameters
    ----------
    expected_checksum:
        Optional sha256 of the file's raw bytes; a mismatch is recorded as a
        tier-2 diagnostic.
    provenance:
        Supply to override the automatically constructed provenance.
    """
    path = Path(path)
    log = log if log is not None else DiagnosticLog()
    if provenance is None:
        provenance = file_provenance(
            path,
            source=source or f"file:{path.name}",
            identifier=identifier,
            notes=f"outcome column {outcome_column!r}",
            expected_checksum=expected_checksum,
            log=log,
        )
    frame = pd.read_csv(
        path,
        sep=sep,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    return binary_outcome_from_frame(
        frame,
        name=name,
        sample_id_column=sample_id_column,
        outcome_column=outcome_column,
        label_column=label_column,
        positive_label=positive_label,
        negative_label=negative_label,
        provenance=provenance,
        log=log,
    )


def _is_missing(value: object) -> bool:
    """``True`` for NaN/None and for the textual NA spellings."""
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str):
        return value.strip() in _NA_STRINGS
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # pragma: no cover - exotic object columns
        return False


def _coerce_bool(value: object, sample_id: SampleId) -> bool:
    """Coerce one outcome cell to ``bool`` or raise naming the offender."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return bool(int(value))
        raise ValueError(
            f"outcome for sample {sample_id!r} is {value!r}; a binary endpoint "
            "must be 0 or 1"
        )
    if isinstance(value, (float, np.floating)):
        if float(value) in (0.0, 1.0):
            return bool(float(value))
        raise ValueError(
            f"outcome for sample {sample_id!r} is {value!r}; a binary endpoint "
            "must be 0 or 1"
        )
    token = str(value).strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    raise ValueError(
        f"outcome for sample {sample_id!r} is {value!r}, which is not a binary "
        f"value; accepted: {sorted(_TRUE_TOKENS | _FALSE_TOKENS)!r}"
    )


def _derive_class_names(
    label_values: Sequence[str],
    labels: np.ndarray,
    label_column: str | None,
) -> tuple[str, str]:
    """Derive ``(positive_label, negative_label)`` from a text label column.

    Raises ``ValueError`` when the text label is not a bijection with the binary
    outcome -- that would mean the file disagrees with itself and the report
    would print a class name that does not describe the class.
    """
    by_class: dict[bool, set[str]] = {True: set(), False: set()}
    for text, flag in zip(label_values, labels, strict=True):
        by_class[bool(flag)].add(text)
    for flag, names in by_class.items():
        if len(names) != 1:
            raise ValueError(
                f"label column {label_column!r} is not consistent with the binary "
                f"outcome: class {int(flag)} carries labels {sorted(names)!r}"
            )
    positive = next(iter(by_class[True]))
    negative = next(iter(by_class[False]))
    if positive == negative:
        raise ValueError(
            f"label column {label_column!r} uses the same text {positive!r} for "
            "both classes"
        )
    return positive, negative


def _reject_duplicate_samples(sample_ids: tuple[SampleId, ...]) -> None:
    """Raise ``ValueError`` naming duplicated sample identifiers."""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in sample_ids:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise ValueError(
            f"duplicate sample identifier(s) in the outcome table: "
            f"{sorted(duplicates)[:10]!r}"
        )
