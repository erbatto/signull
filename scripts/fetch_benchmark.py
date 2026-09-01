#!/usr/bin/env python3
"""Download and prepare the signull public benchmark cohort(s) from NCBI GEO.

First-choice benchmark: **GSE25055** (Hatzis et al. 2011, JAMA 305:1873) -- the
MDACC/I-SPY/Peru discovery cohort of neoadjuvant taxane-anthracycline treated
HER2-negative breast cancer.  310 pre-treatment tumour biopsies on Affymetrix
HG-U133A (GPL96), with a genuinely binary, clinically adjudicated outcome:
pathologic complete response (pCR) vs residual disease (RD).

Its paired external validation cohort **GSE25065** (n=198, same platform, same
outcome definition) is also supported via ``--accession``.

What this script does
---------------------
1. Downloads the GEO *series matrix* file to ``data/raw/`` (idempotent -- an
   existing, checksum-verified file is reused).
2. Parses the header block into per-sample clinical characteristics and the
   table block into an expression matrix.
3. Writes to ``data/processed/``:
     ``<ACC>_expression.tsv.gz``  probes x samples, log2 MAS5 intensities
     ``<ACC>_outcome.tsv``        sample_id, outcome (1=pCR), + covariates
     ``<ACC>_provenance.json``    URL, accession, dates, sha256, shape, balance
4. Validates the parsed shape and class balance against the values recorded in
   ``docs/prior-art-and-data.md`` and fails loudly if GEO has changed.

Dependencies: Python 3.11 standard library + pandas + numpy.

Usage
-----
    python scripts/fetch_benchmark.py                     # GSE25055
    python scripts/fetch_benchmark.py --accession GSE25065
    python scripts/fetch_benchmark.py --force-download
    python scripts/fetch_benchmark.py --strict-checksum   # hard-fail on drift
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Dataset registry.  Every number here is an assertion about the remote file;
# a mismatch means GEO changed and the caller must be told, not silently served
# a different dataset.  Values verified against GEO on 2026-09-01.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Dataset:
    accession: str
    title: str
    platform: str
    pubmed: str
    # Expected remote artefact
    sha256: str
    size_bytes: int
    series_last_update: str
    # Expected parsed shape
    n_probes: int
    n_samples: int
    # Outcome definition
    outcome_key: str
    positive_label: str
    negative_label: str
    missing_labels: tuple[str, ...] = ("NA", "", "na", "N/A")
    expected_balance: dict[str, int] = field(default_factory=dict)
    # Extra clinical columns carried into the outcome table when present
    covariate_keys: tuple[str, ...] = ()

    @property
    def url(self) -> str:
        # GEO buckets series into <prefix>nnn directories, e.g. GSE25nnn.
        prefix = f"{self.accession[:-3]}nnn"
        return (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/"
            f"{prefix}/{self.accession}/matrix/{self.accession}_series_matrix.txt.gz"
        )

    @property
    def geo_page(self) -> str:
        return f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={self.accession}"


# GEO characteristic keys carried through to the outcome table.  Keys are
# renamed on the way out; "sample id" in particular is the *study* patient ID
# and must not collide with the GSM index, which is also called sample_id.
_COVARIATE_RENAME = {"sample id": "study_patient_id"}

_COVARIATES = (
    "sample id",
    "source",
    "age_years",
    "er_status_ihc",
    "pr_status_ihc",
    "her2_status",
    "grade",
    "clinical_t_stage",
    "clinical_nodal_status",
    "clinical_ajcc_stage",
    "pathologic_response_rcb_class",
    "drfs_1_event_0_censored",
    "drfs_even_time_years",
    "esr1_status",
    "erbb2_status",
    "set_class",
    "ggi_class",
    "pam50_class",
    "dlda30_prediction",
    "chemosensitivity_prediction",
    "rcb_0_i_prediction",
)

DATASETS: dict[str, Dataset] = {
    "GSE25055": Dataset(
        accession="GSE25055",
        title=(
            "Discovery cohort for genomic predictor of response and survival "
            "following neoadjuvant taxane-anthracycline chemotherapy in breast cancer"
        ),
        platform="GPL96 (Affymetrix HG-U133A)",
        pubmed="21558518",
        sha256="9f8a94a9226f38d16380d776ab007c0453c3d31c9915d69562d5854baf3d6777",
        size_bytes=37718581,
        series_last_update="Nov 02 2022",
        n_probes=22283,
        n_samples=310,
        outcome_key="pathologic_response_pcr_rd",
        positive_label="pCR",
        negative_label="RD",
        expected_balance={"pCR": 57, "RD": 249, "NA": 4},
        covariate_keys=_COVARIATES,
    ),
    "GSE25065": Dataset(
        accession="GSE25065",
        title=(
            "Validation cohort for genomic predictor of response and survival "
            "following neoadjuvant taxane-anthracycline chemotherapy in breast cancer"
        ),
        platform="GPL96 (Affymetrix HG-U133A)",
        pubmed="21558518",
        sha256="adac6710a8452be19e0f73070668b186a282ebeb34f9aaec09c884906034f6e9",
        size_bytes=24194802,
        series_last_update="Nov 02 2022",
        n_probes=22283,
        n_samples=198,
        outcome_key="pathologic_response_pcr_rd",
        positive_label="pCR",
        negative_label="RD",
        expected_balance={"pCR": 42, "RD": 140, "NA": 16},
        covariate_keys=_COVARIATES,
    ),
}

DEFAULT_ACCESSION = "GSE25055"
TABLE_BEGIN = "!series_matrix_table_begin"
TABLE_END = "!series_matrix_table_end"


class FetchError(RuntimeError):
    """Raised with an actionable message when the remote source has changed."""


def _fail(*lines: str) -> "FetchError":
    body = "\n".join(f"    {line}" for line in lines)
    return FetchError("\n" + "=" * 78 + "\nFETCH FAILED\n" + body + "\n" + "=" * 78)


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(ds: Dataset, raw_dir: Path, *, force: bool) -> tuple[Path, dict]:
    """Fetch the series matrix, or reuse a verified existing copy.

    Returns (path, sidecar) where sidecar records the *original* download event
    so provenance survives idempotent re-runs.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"{ds.accession}_series_matrix.txt.gz"
    sidecar_path = dest.with_suffix(dest.suffix + ".meta.json")

    if dest.exists() and not force:
        digest = _sha256(dest)
        if sidecar_path.exists():
            sidecar = json.loads(sidecar_path.read_text())
        else:
            sidecar = {}
        if sidecar.get("sha256") == digest:
            print(f"[skip]     {dest.name} already present and checksum-verified")
            return dest, sidecar
        print(
            f"[re-fetch] {dest.name} exists but its sidecar checksum is missing or "
            "stale; downloading again"
        )

    print(f"[GET]      {ds.url}")
    started = datetime.now(timezone.utc)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(raw_dir), suffix=".part")
    tmp = Path(tmp_name)
    try:
        req = urllib.request.Request(
            ds.url, headers={"User-Agent": "signull-fetch-benchmark/1.0"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_fd, "wb") as out:
            if getattr(resp, "status", 200) != 200:
                raise _fail(
                    f"GEO returned HTTP {resp.status} for {ds.accession}.",
                    f"URL: {ds.url}",
                    f"Check the series still exists: {ds.geo_page}",
                )
            shutil.copyfileobj(resp, out, length=1 << 20)
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise _fail(
            f"GEO returned HTTP {exc.code} ({exc.reason}) for {ds.accession}.",
            f"URL: {ds.url}",
            "The NCBI FTP layout may have changed, or the series was withdrawn.",
            f"Verify by hand: {ds.geo_page}",
        ) from exc
    except urllib.error.URLError as exc:
        tmp.unlink(missing_ok=True)
        raise _fail(
            f"Could not reach NCBI to download {ds.accession}: {exc.reason}",
            f"URL: {ds.url}",
            "Check network/proxy access to ftp.ncbi.nlm.nih.gov, then re-run.",
        ) from exc

    size = tmp.stat().st_size
    if size < 1_000_000:
        preview = tmp.read_bytes()[:200]
        tmp.unlink(missing_ok=True)
        raise _fail(
            f"Downloaded {ds.accession} file is only {size} bytes -- expected "
            f"~{ds.size_bytes:,}.",
            "This is almost certainly an error page, not the series matrix.",
            f"First bytes: {preview!r}",
            f"URL: {ds.url}",
        )

    digest = _sha256(tmp)
    tmp.chmod(0o644)  # mkstemp defaults to 0600; these are public data
    tmp.replace(dest)
    sidecar = {
        "accession": ds.accession,
        "url": ds.url,
        "downloaded_at_utc": started.isoformat(timespec="seconds"),
        "size_bytes": size,
        "sha256": digest,
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"[ok]       {size:,} bytes -> {dest}")
    return dest, sidecar


def verify_artefact(ds: Dataset, path: Path, sidecar: dict, *, strict: bool) -> list[str]:
    """Compare the downloaded bytes with what docs/prior-art-and-data.md records."""
    warnings: list[str] = []
    digest = sidecar.get("sha256") or _sha256(path)
    size = path.stat().st_size

    if digest != ds.sha256:
        msg = (
            f"sha256 of {path.name} is {digest} but {ds.sha256} was recorded on "
            f"2026-09-01 (size {size:,} vs {ds.size_bytes:,} expected). "
            "GEO has regenerated the series matrix."
        )
        if strict:
            raise _fail(
                msg,
                "Re-verify the cohort against the GEO record and update",
                "  scripts/fetch_benchmark.py DATASETS and docs/prior-art-and-data.md",
                f"GEO page: {ds.geo_page}",
            )
        warnings.append(msg + " Content checks below still apply.")
    else:
        print(f"[verify]   sha256 matches the recorded value ({digest[:16]}...)")
    return warnings


# --------------------------------------------------------------------------- #
# Parse
# --------------------------------------------------------------------------- #


def _split_row(line: str) -> list[str]:
    return [cell.strip().strip('"') for cell in line.rstrip("\n").split("\t")]


def parse_series_matrix(ds: Dataset, path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (expression, sample_metadata, series_header)."""
    series_header: dict[str, list[str]] = {}
    sample_rows: dict[str, list[str]] = {}
    characteristics: list[list[str]] = []
    table_lines: list[str] = []
    in_table = False
    saw_end = False

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not in_table:
                if line.startswith(TABLE_BEGIN):
                    in_table = True
                    continue
                if line.startswith("!Series_"):
                    cells = _split_row(line)
                    series_header.setdefault(cells[0].lstrip("!"), []).extend(cells[1:])
                elif line.startswith("!Sample_characteristics_ch1"):
                    characteristics.append(_split_row(line))
                elif line.startswith("!Sample_"):
                    cells = _split_row(line)
                    sample_rows.setdefault(cells[0].lstrip("!"), cells[1:])
            else:
                if line.startswith(TABLE_END):
                    saw_end = True
                    break
                table_lines.append(line)

    if not in_table:
        raise _fail(
            f"No '{TABLE_BEGIN}' marker found in {path.name}.",
            "The GEO series matrix format has changed, or the file is truncated.",
            "Delete data/raw/ and re-run with --force-download; if that fails, "
            "inspect the file by hand.",
            f"URL: {ds.url}",
        )
    if not saw_end:
        raise _fail(
            f"'{TABLE_END}' marker missing from {path.name} -- the download is truncated.",
            "Re-run with --force-download.",
        )
    if len(table_lines) < 2:
        raise _fail(f"Expression table in {path.name} is empty.", f"URL: {ds.url}")

    # --- expression ------------------------------------------------------- #
    expr = pd.read_csv(
        io.StringIO("".join(table_lines)),
        sep="\t",
        index_col=0,
        na_values=["", "null", "NA"],
    )
    expr.index = expr.index.astype(str).str.strip('"')
    expr.index.name = "probe_id"
    expr.columns = [str(c).strip('"') for c in expr.columns]
    expr.columns.name = "sample_id"
    expr = expr.astype(np.float64)

    # --- sample metadata --------------------------------------------------- #
    gsm = sample_rows.get("Sample_geo_accession")
    if not gsm:
        raise _fail(
            f"No '!Sample_geo_accession' row in {path.name}; cannot identify samples.",
            f"URL: {ds.url}",
        )

    # Characteristics keys are NOT guaranteed to sit at the same row index for
    # every sample (they do in GSE25055/GSE25065, they do not in e.g. GSE20194),
    # so parse "key: value" per cell rather than trusting row position.
    meta: list[dict[str, str]] = [{} for _ in gsm]
    for row in characteristics:
        for j, cell in enumerate(row[1:]):
            if j >= len(meta) or not cell:
                continue
            key, sep, value = cell.partition(":")
            if not sep:
                continue
            meta[j][key.strip()] = value.strip()

    md = pd.DataFrame(meta, index=pd.Index(gsm, name="sample_id"))
    if "Sample_title" in sample_rows:
        md.insert(0, "sample_title", sample_rows["Sample_title"])
    return expr, md, series_header


# --------------------------------------------------------------------------- #
# Validate + build outcome table
# --------------------------------------------------------------------------- #


def build_outcome(ds: Dataset, md: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    if ds.outcome_key not in md.columns:
        raise _fail(
            f"Outcome field '{ds.outcome_key}' is absent from {ds.accession} "
            "sample characteristics.",
            "GEO has re-curated the sample metadata.",
            "Available characteristic keys: " + ", ".join(sorted(md.columns)),
            f"GEO page: {ds.geo_page}",
        )

    raw = md[ds.outcome_key].fillna("NA").astype(str).str.strip()
    balance = {k: int(v) for k, v in raw.value_counts().items()}

    known = {ds.positive_label, ds.negative_label, *ds.missing_labels}
    unexpected = sorted(set(raw.unique()) - known)
    if unexpected:
        raise _fail(
            f"Unexpected values in '{ds.outcome_key}' for {ds.accession}: {unexpected}.",
            f"Expected only {ds.positive_label!r}, {ds.negative_label!r} or missing.",
            "The outcome coding has changed -- do not silently binarise it.",
            f"GEO page: {ds.geo_page}",
        )

    mapping = {ds.positive_label: 1, ds.negative_label: 0}
    outcome = raw.map(mapping)
    keep = outcome.notna()

    cols = {
        "outcome": outcome[keep].astype("int64"),
        "outcome_label": raw[keep],
    }
    out = pd.DataFrame(cols)
    for key in ds.covariate_keys:
        if key in md.columns:
            name = _COVARIATE_RENAME.get(key, key.replace(" ", "_"))
            if name in out.columns or name == "sample_id":
                name = f"geo_{name}"
            out[name] = md.loc[keep, key]
    out.index.name = "sample_id"
    return out, balance


def validate(ds: Dataset, expr: pd.DataFrame, md: pd.DataFrame, balance: dict[str, int],
             header: dict[str, list[str]]) -> list[str]:
    warnings: list[str] = []
    problems: list[str] = []

    if expr.shape[0] != ds.n_probes:
        problems.append(
            f"probe count is {expr.shape[0]:,}, documented value is {ds.n_probes:,}"
        )
    if expr.shape[1] != ds.n_samples:
        problems.append(
            f"sample count is {expr.shape[1]:,}, documented value is {ds.n_samples:,}"
        )
    if len(md) != expr.shape[1]:
        problems.append(
            f"metadata rows ({len(md)}) do not match expression columns ({expr.shape[1]})"
        )
    if list(md.index) != list(expr.columns):
        problems.append("sample order differs between the metadata and the matrix")

    for label, expected in ds.expected_balance.items():
        got = balance.get(label, 0)
        if got != expected:
            problems.append(
                f"outcome class {label!r}: {got} samples, documented value is {expected}"
            )

    if problems:
        raise _fail(
            f"{ds.accession} no longer matches what docs/prior-art-and-data.md records:",
            *[f"  - {p}" for p in problems],
            "",
            "Either GEO revised the series, or this script's parser is wrong.",
            f"Check the GEO record: {ds.geo_page}",
            "If the change is legitimate, update the Dataset entry in",
            "  scripts/fetch_benchmark.py AND the table in docs/prior-art-and-data.md.",
        )

    got_update = (header.get("Series_last_update_date") or ["<absent>"])[0]
    if got_update != ds.series_last_update:
        warnings.append(
            f"Series_last_update_date is {got_update!r}, was {ds.series_last_update!r} "
            "on 2026-09-01. Shape and class balance still check out, but the GEO "
            "record was touched -- re-read the sample metadata before trusting it."
        )

    platform = (header.get("Series_platform_id") or ["<absent>"])[0]
    if platform not in ds.platform:
        warnings.append(f"platform is {platform!r}, expected {ds.platform!r}")

    finite = np.isfinite(expr.to_numpy())
    if not finite.all():
        warnings.append(
            f"{int((~finite).sum()):,} non-finite expression values "
            f"({100 * (~finite).mean():.3f}%) -- downstream scoring must handle them"
        )
    vmax = float(np.nanmax(expr.to_numpy()))
    if vmax > 40:
        warnings.append(
            f"max expression value is {vmax:,.1f}; these look like linear-scale "
            "intensities, not the log2 values this cohort is documented to carry"
        )
    return warnings


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def run(ds: Dataset, repo_root: Path, *, force: bool, strict: bool) -> int:
    raw_dir = repo_root / "data" / "raw"
    proc_dir = repo_root / "data" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nsignull benchmark fetch -- {ds.accession}")
    print(f"  {ds.title}")
    print(f"  platform: {ds.platform} | PubMed: {ds.pubmed}\n")

    path, sidecar = download(ds, raw_dir, force=force)
    warnings = verify_artefact(ds, path, sidecar, strict=strict)

    print("[parse]    reading series matrix ...")
    expr, md, header = parse_series_matrix(ds, path)
    outcome, balance = build_outcome(ds, md)
    warnings += validate(ds, expr, md, balance, header)

    n_pos = int((outcome["outcome"] == 1).sum())
    n_neg = int((outcome["outcome"] == 0).sum())
    print(
        f"[validate] {expr.shape[0]:,} probes x {expr.shape[1]:,} samples; "
        f"outcome usable on {len(outcome)} "
        f"({n_pos} {ds.positive_label} / {n_neg} {ds.negative_label}, "
        f"{100 * n_pos / len(outcome):.1f}% positive)"
    )

    expr_path = proc_dir / f"{ds.accession}_expression.tsv.gz"
    out_path = proc_dir / f"{ds.accession}_outcome.tsv"
    prov_path = proc_dir / f"{ds.accession}_provenance.json"

    expr.to_csv(expr_path, sep="\t", float_format="%.6f")
    outcome.to_csv(out_path, sep="\t")

    provenance = {
        "tool": "scripts/fetch_benchmark.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "accession": ds.accession,
        "title": ds.title,
        "geo_page": ds.geo_page,
        "source_url": ds.url,
        "pubmed_id": ds.pubmed,
        "platform": ds.platform,
        "series_last_update_date": (header.get("Series_last_update_date") or [None])[0],
        "raw_file": str(path.relative_to(repo_root)),
        "raw_sha256": sidecar.get("sha256"),
        "raw_size_bytes": path.stat().st_size,
        "downloaded_at_utc": sidecar.get("downloaded_at_utc"),
        "expression": {
            "path": str(expr_path.relative_to(repo_root)),
            "n_probes": int(expr.shape[0]),
            "n_samples": int(expr.shape[1]),
            "scale": "log2 MAS5 (global scaling, trimmed mean target 600)",
            "sha256": _sha256(expr_path),
        },
        "outcome": {
            "path": str(out_path.relative_to(repo_root)),
            "field": ds.outcome_key,
            "definition": (
                f"1 = {ds.positive_label} (pathologic complete response), "
                f"0 = {ds.negative_label} (residual disease)"
            ),
            "n_samples": int(len(outcome)),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "positive_rate": round(n_pos / len(outcome), 4),
            "n_dropped_missing": int(expr.shape[1] - len(outcome)),
            "raw_value_counts": balance,
            "sha256": _sha256(out_path),
        },
        "warnings": warnings,
    }
    prov_path.write_text(json.dumps(provenance, indent=2) + "\n")

    print("\n--- provenance " + "-" * 62)
    print(f"  accession        {ds.accession}   ({ds.geo_page})")
    print(f"  source URL       {ds.url}")
    print(f"  downloaded       {sidecar.get('downloaded_at_utc')}")
    print(f"  raw sha256       {sidecar.get('sha256')}")
    print(f"  raw size         {path.stat().st_size:,} bytes")
    print(f"  prepared         {provenance['generated_at_utc']}")
    print(f"  expression       {expr_path.relative_to(repo_root)}")
    print(f"                   {expr.shape[0]:,} probes x {expr.shape[1]:,} samples, log2 MAS5")
    print(f"  outcome          {out_path.relative_to(repo_root)}")
    print(f"                   {ds.outcome_key}: 1={ds.positive_label}, 0={ds.negative_label}")
    print(f"                   n={len(outcome)} ({n_pos} pos / {n_neg} neg), "
          f"{int(expr.shape[1] - len(outcome))} dropped as missing")
    print(f"  provenance       {prov_path.relative_to(repo_root)}")
    print("-" * 77)

    if warnings:
        print("\nWARNINGS (recorded in the provenance file):")
        for w in warnings:
            print(f"  ! {w}")
    print("\nDone.\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--accession",
        default=DEFAULT_ACCESSION,
        choices=sorted(DATASETS),
        help=f"GEO series to fetch (default: {DEFAULT_ACCESSION})",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repository root containing data/raw and data/processed",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="re-download even if a checksum-verified copy exists",
    )
    parser.add_argument(
        "--strict-checksum",
        action="store_true",
        help="fail (rather than warn) if the remote file's sha256 has changed",
    )
    args = parser.parse_args(argv)

    try:
        return run(
            DATASETS[args.accession],
            args.repo_root.resolve(),
            force=args.force_download,
            strict=args.strict_checksum,
        )
    except FetchError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
