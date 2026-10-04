"""
download_sevir_subset.py — Download a small subset of the SEVIR dataset from AWS S3.

SEVIR is hosted publicly at s3://sevir/ (no credentials required).
Reference: Veillette et al., NeurIPS 2020 — https://github.com/MIT-AI-Accelerator/neurips-2020-sevir

Run via:
    make sevir-download
or:
    python scripts/download_sevir_subset.py --config configs/config.yaml

What this downloads:
    - The VIL (Vertically Integrated Liquid) channel catalog file (~1 MB)
    - The first N HDF5 data files listed in the catalog
    - Typically each file is ~1-4 GB, so this is a significant download.
    - Set sevir.num_events in config.yaml to limit how many events to keep.

If you don't want to download, use `make synthetic` instead.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Public SEVIR S3 paths (no auth required, anonymous access).
SEVIR_S3_BASE = "s3://sevir"
VIL_CATALOG = f"{SEVIR_S3_BASE}/CATALOG.csv"
VIL_DATA_PREFIX = f"{SEVIR_S3_BASE}/data/vil/"


def _aws_cli_available() -> bool:
    """Check whether the AWS CLI is installed (needed for S3 download)."""
    try:
        result = subprocess.run(
            ["aws", "--version"], capture_output=True, timeout=5, check=False
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def download_catalog(output_dir: Path) -> Path:
    """Download the SEVIR catalog CSV to identify available VIL files."""
    catalog_path = output_dir / "CATALOG.csv"
    if catalog_path.exists():
        log.info("Catalog already exists at %s, skipping download.", catalog_path)
        return catalog_path

    log.info("Downloading SEVIR catalog from %s ...", VIL_CATALOG)
    subprocess.run(
        ["aws", "s3", "cp", "--no-sign-request", VIL_CATALOG, str(catalog_path)],
        check=True,
    )
    return catalog_path


def list_vil_files(catalog_path: Path, max_files: int) -> list[str]:
    """
    Parse the catalog CSV to get the S3 paths of VIL HDF5 files.

    The catalog has a 'file_name' and 'img_type' column. We filter for
    img_type == 'vil' and return up to max_files paths.
    """
    import csv

    vil_files: list[str] = []
    with catalog_path.open("r") as f:
        reader = csv.DictReader(f)
        seen: set[str] = set()
        for row in reader:
            if row.get("img_type", "").strip().lower() != "vil":
                continue
            fname = row.get("file_name", "").strip()
            if fname and fname not in seen:
                seen.add(fname)
                vil_files.append(fname)
            if len(vil_files) >= max_files:
                break
    return vil_files


def download_vil_files(s3_paths: list[str], output_dir: Path) -> None:
    """Download each VIL HDF5 file from S3 to output_dir."""
    for s3_path in s3_paths:
        filename = Path(s3_path).name
        local_path = output_dir / filename
        if local_path.exists():
            log.info("Already exists: %s, skipping.", filename)
            continue
        s3_url = f"{SEVIR_S3_BASE}/{s3_path.lstrip('/')}"
        log.info("Downloading %s (~1-4 GB) ...", filename)
        subprocess.run(
            ["aws", "s3", "cp", "--no-sign-request", s3_url, str(local_path)],
            check=True,
        )
        log.info("Saved to %s", local_path)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    output_dir = Path(cfg.paths.sevir_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not _aws_cli_available():
        log.error(
            "AWS CLI not found. Install it with:\n"
            "    pip install awscli\n"
            "or use `make synthetic` for offline use (no download needed)."
        )
        sys.exit(1)

    # How many HDF5 files to download (one file ≈ several hundred events).
    # We download just enough to cover sevir.num_events.
    max_files = max(1, cfg.sevir.num_events // 100)
    log.info("Will download up to %d VIL file(s) (≈%d events each).", max_files, 100)

    catalog_path = download_catalog(output_dir)
    vil_files = list_vil_files(catalog_path, max_files=max_files)

    if not vil_files:
        log.error("No VIL files found in catalog. Check the catalog format.")
        sys.exit(1)

    log.info(
        "Found %d VIL file(s) in catalog. Downloading %d ...",
        len(vil_files),
        min(len(vil_files), max_files),
    )
    download_vil_files(vil_files[:max_files], output_dir)
    log.info("SEVIR download complete. Files in: %s", output_dir)
    log.info("Now run: make train  (or make demo)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SEVIR VIL subset from S3.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
