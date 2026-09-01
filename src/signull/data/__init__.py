"""Loading, alignment, resolution and background construction.

This package owns every step between files on disk and an
:class:`~signull.types.AlignedDataset` whose sample identity is guaranteed.  It
performs no scoring, no sampling, no metrics and no p-values
(``docs/architecture.md`` Sec. 2).

Typical call order
------------------
1. :class:`DenseExpressionMatrix` (``from_frame`` or :func:`load_matrix_tsv`) and
   :func:`load_outcome_tsv` / :func:`binary_outcome_from_frame`;
2. :meth:`DenseExpressionMatrix.align_to` -> ``AlignedDataset``;
3. :func:`resolve_signature` against the aligned matrix -> resolved signature
   plus a :class:`~signull.types.SignatureResolution`;
4. :func:`eligible_background` on the aligned cohort, then
   :func:`check_background_floors` against the *effective* candidate size.

Order matters: gene statistics and the background must be computed after
alignment, on the analysis cohort only.

Diagnostics
-----------
Every loader accepts a :class:`DiagnosticLog`.  Pass one log through the whole
sequence to collect the run's data-layer diagnostics in emission order.
"""

from __future__ import annotations

from .diagnostics import DiagnosticCode, DiagnosticLog, SignullDataWarning
from .matrix import (
    GENES_X_SAMPLES,
    SAMPLES_X_GENES,
    DenseExpressionMatrix,
    detection_dimension_is_degenerate,
    emit_detection_degeneracy,
    load_matrix_tsv,
)
from .outcome import binary_outcome_from_frame, load_outcome_tsv
from .provenance import CHECKSUM_ALGORITHM, file_provenance, sha256_file
from .resolve import (
    AnnotationTable,
    MatrixIndexResolver,
    load_annotation_table,
    resolve_signature,
    strip_version_suffix,
)
from .signature import (
    load_signature_list,
    load_signature_table,
    load_signatures_gmt,
    looks_excel_mangled,
    signature_from_ids,
)
from .universe import (
    BACKGROUND_PER_CANDIDATE_GENE,
    CONTROL_PROBE_PREFIXES,
    DEFAULT_MIN_DETECTION_RATE,
    MIN_BACKGROUND_GENES,
    BackgroundTooSmallError,
    EligibleBackground,
    check_background_floors,
    eligible_background,
)

__all__ = [
    "AnnotationTable",
    "BACKGROUND_PER_CANDIDATE_GENE",
    "BackgroundTooSmallError",
    "CHECKSUM_ALGORITHM",
    "CONTROL_PROBE_PREFIXES",
    "DEFAULT_MIN_DETECTION_RATE",
    "DenseExpressionMatrix",
    "DiagnosticCode",
    "GENES_X_SAMPLES",
    "SAMPLES_X_GENES",
    "detection_dimension_is_degenerate",
    "load_matrix_tsv",
    "DiagnosticLog",
    "EligibleBackground",
    "MIN_BACKGROUND_GENES",
    "MatrixIndexResolver",
    "SignullDataWarning",
    "binary_outcome_from_frame",
    "check_background_floors",
    "eligible_background",
    "emit_detection_degeneracy",
    "file_provenance",
    "load_annotation_table",
    "load_outcome_tsv",
    "load_signature_list",
    "load_signature_table",
    "load_signatures_gmt",
    "looks_excel_mangled",
    "resolve_signature",
    "sha256_file",
    "signature_from_ids",
    "strip_version_suffix",
]
