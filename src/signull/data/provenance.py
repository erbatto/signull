"""Checksumming and :class:`~signull.types.Provenance` construction.

``docs/architecture.md`` Sec. 6 requires the sha256 of the *raw bytes* of every
input to be recorded, so a rerun that silently picked up a different file is
detectable from the stored result alone.  Everything here is pure: no caches, no
module-level state.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

from signull.types import Provenance

from .diagnostics import DiagnosticCode, DiagnosticLog

__all__ = ["CHECKSUM_ALGORITHM", "sha256_file", "file_provenance"]

#: Digest algorithm recorded in :attr:`~signull.types.Provenance.checksum_algorithm`.
CHECKSUM_ALGORITHM: Final[str] = "sha256"

_CHUNK_BYTES: Final[int] = 1 << 20


def sha256_file(path: Path | str) -> str:
    """Hex sha256 of a file's raw bytes, read in 1 MiB chunks.

    The digest is over the file *as stored* -- for a ``.tsv.gz`` that is the
    compressed bytes, matching what the fetch script recorded.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def file_provenance(
    path: Path | str,
    *,
    source: str,
    identifier: str | None = None,
    retrieved_at: str | None = None,
    citation: str | None = None,
    notes: str | None = None,
    expected_checksum: str | None = None,
    log: DiagnosticLog | None = None,
) -> Provenance:
    """Build a :class:`~signull.types.Provenance` for a file on disk.

    Parameters
    ----------
    expected_checksum:
        When given, the computed digest is compared against it.  A mismatch is
        tier 2 (``checksum_mismatch``): the file is still loaded, because
        refusing would strand a user whose only copy differs, but the condition
        is recorded so it can never be discovered only from console scrollback.
    """
    checksum = sha256_file(path)
    if expected_checksum is not None and checksum != expected_checksum:
        message = (
            f"sha256 of {Path(path).name} is {checksum}, expected "
            f"{expected_checksum}; the file on disk is not the one the recorded "
            "provenance describes"
        )
        if log is not None:
            log.warn(
                DiagnosticCode.CHECKSUM_MISMATCH,
                message,
                context={
                    "path": str(path),
                    "observed": checksum,
                    "expected": expected_checksum,
                },
            )
    return Provenance(
        source=source,
        identifier=identifier,
        retrieved_at=retrieved_at,
        citation=citation,
        checksum=checksum,
        checksum_algorithm=CHECKSUM_ALGORITHM,
        notes=notes,
    )
