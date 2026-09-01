"""Diagnostic codes and the instance-scoped log used by :mod:`signull.data`.

The architecture document (``docs/architecture.md`` Sec. 5) defines a three-tier
error taxonomy.  This module implements tiers 2 and 3:

* **Tier 2** -- ``warnings.warn`` **and** a recorded :class:`~signull.types.Diagnostic`.
  Console output does not survive into a stored result, so every warning is also
  data.  Use :meth:`DiagnosticLog.warn`.
* **Tier 3** -- recorded only, no warning.  Normal operating detail the report
  must still show.  Use :meth:`DiagnosticLog.record`.

Tier 1 (raise) needs no machinery: the offending values go in the exception
message.

There is no module-level log.  A :class:`DiagnosticLog` is created by a caller
and threaded explicitly through the loaders, so nothing here is global mutable
state and two concurrent loads never interleave.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from enum import Enum

from signull.types import Diagnostic, Severity

__all__ = [
    "SignullDataWarning",
    "DiagnosticCode",
    "DiagnosticLog",
]


class SignullDataWarning(UserWarning):
    """Category for every tier-2 warning emitted by :mod:`signull.data`.

    Declared so that callers (and tests) can filter precisely rather than
    catching bare ``UserWarning``.
    """


class DiagnosticCode(str, Enum):
    """Stable machine-readable diagnostic codes owned by :mod:`signull.data`.

    Codes are part of the wire format: they are serialised with results and may
    be matched on by the report layer, so a code is never reused with a
    different meaning.  The first block is the shared vocabulary listed in
    ``docs/architecture.md`` Sec. 5; the second block is owned exclusively by
    this package.
    """

    # -- shared vocabulary from docs/architecture.md Sec. 5 ------------------
    LOW_SIGNATURE_OVERLAP = "low_signature_overlap"
    SAMPLES_DROPPED_IN_ALIGNMENT = "samples_dropped_in_alignment"
    OUTCOME_MISSING_DROPPED = "outcome_missing_dropped"
    EXTREME_CLASS_IMBALANCE = "extreme_class_imbalance"
    SMALL_COHORT = "small_cohort"
    DUPLICATE_PROBES_COLLAPSED = "duplicate_probes_collapsed"
    UNPINNED_ANNOTATION_SOURCE = "unpinned_annotation_source"
    SIGNATURE_LOOKS_EXCEL_MANGLED = "signature_looks_excel_mangled"
    ZERO_VARIANCE_GENES_IN_SIGNATURE = "zero_variance_genes_in_signature"

    # -- codes owned by signull.data -----------------------------------------
    #: Matrix index is not in the canonical HGNC-symbol namespace and no pinned
    #: annotation table was supplied, so no conversion was attempted.
    PROBE_LEVEL_NAMESPACE_RETAINED = "probe_level_namespace_retained"
    #: Signature namespace differs from the matrix namespace with no annotation
    #: table available to bridge them.
    SIGNATURE_NAMESPACE_MISMATCH = "signature_namespace_mismatch"
    #: One source identifier mapped to several canonical identifiers; dropped
    #: rather than guessed (``docs/architecture.md`` Sec. 4 rule 3).
    AMBIGUOUS_MAPPING_DROPPED = "ambiguous_mapping_dropped"
    #: Ensembl version suffixes were stripped before mapping (Sec. 4 rule 1).
    VERSION_SUFFIX_STRIPPED = "version_suffix_stripped"
    #: Identity resolution: source and target namespaces already agree, so no
    #: annotation table was consulted.
    IDENTITY_RESOLUTION = "identity_resolution"
    #: Duplicate identifiers were removed from a signature at construction.
    DUPLICATE_SIGNATURE_IDS_DROPPED = "duplicate_signature_ids_dropped"
    #: ``sd(detection_rate)`` is below 0.01, so the detection dimension carries
    #: no information (``docs/statistical-design.md`` Sec. 2.2: ``K_d -> 1``).
    DETECTION_DIMENSION_DEGENERATE = "detection_dimension_degenerate"
    #: Affymetrix ``AFFX-`` control probesets removed from the eligible
    #: background.
    CONTROL_PROBES_EXCLUDED = "control_probes_excluded"
    #: Genes removed from the eligible background by an expression filter.
    BACKGROUND_GENES_FILTERED = "background_genes_filtered"
    #: ``|B|`` fell below the floors of ``docs/statistical-design.md`` Sec. 8 F4.
    BACKGROUND_TOO_SMALL = "background_too_small"
    #: Matrix has fewer rows than columns, which is the wrong way round for
    #: nearly all real cohorts.
    MATRIX_ORIENTATION_SUSPICIOUS = "matrix_orientation_suspicious"
    #: The on-disk checksum did not match the recorded provenance checksum.
    CHECKSUM_MISMATCH = "checksum_mismatch"
    #: Samples present in the outcome table but absent from the matrix.
    OUTCOME_SAMPLES_NOT_IN_MATRIX = "outcome_samples_not_in_matrix"


class DiagnosticLog:
    """Ordered, append-only collector of :class:`~signull.types.Diagnostic`.

    Instance-scoped by design.  Create one per load or per run and pass it into
    the loaders; the accumulated :attr:`diagnostics` are what the report layer
    renders.  A log is never shared implicitly and is never module-level state.

    Reusing one log across several loads is supported and is the intended way to
    collect a whole run's data-layer diagnostics in one place.
    """

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: list[Diagnostic] = []

    @property
    def diagnostics(self) -> tuple[Diagnostic, ...]:
        """Everything recorded so far, in emission order."""
        return tuple(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def codes(self) -> tuple[str, ...]:
        """The recorded codes, in emission order.  Convenient for assertions."""
        return tuple(entry.code for entry in self._entries)

    def has(self, code: DiagnosticCode | str) -> bool:
        """``True`` when ``code`` has been recorded at least once."""
        wanted = code.value if isinstance(code, DiagnosticCode) else code
        return any(entry.code == wanted for entry in self._entries)

    def get(self, code: DiagnosticCode | str) -> tuple[Diagnostic, ...]:
        """Every diagnostic recorded under ``code``, in emission order."""
        wanted = code.value if isinstance(code, DiagnosticCode) else code
        return tuple(entry for entry in self._entries if entry.code == wanted)

    def record(
        self,
        code: DiagnosticCode,
        message: str,
        *,
        severity: Severity = Severity.INFO,
        context: Mapping[str, object] | None = None,
    ) -> Diagnostic:
        """Tier 3: record without warning.

        Parameters
        ----------
        code:
            Stable code from :class:`DiagnosticCode`.
        message:
            Human-readable one-liner.
        severity:
            Defaults to :attr:`~signull.types.Severity.INFO`.
        context:
            Small JSON-serialisable payload (counts, identifiers, thresholds).
        """
        entry = Diagnostic(
            code=code.value,
            severity=severity,
            message=message,
            context=dict(context or {}),
        )
        self._entries.append(entry)
        return entry

    def warn(
        self,
        code: DiagnosticCode,
        message: str,
        *,
        context: Mapping[str, object] | None = None,
        stacklevel: int = 3,
    ) -> Diagnostic:
        """Tier 2: ``warnings.warn`` **and** record at ``WARNING`` severity."""
        entry = self.record(
            code, message, severity=Severity.WARNING, context=context
        )
        warnings.warn(
            f"[{code.value}] {message}",
            SignullDataWarning,
            stacklevel=stacklevel,
        )
        return entry

    def extend(self, diagnostics: tuple[Diagnostic, ...]) -> None:
        """Append already-built diagnostics (e.g. from ``AlignedDataset``)."""
        self._entries.extend(diagnostics)
