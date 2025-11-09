import os
import re
import sys
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable

import requests


KAGGLE_DATASETS_DOWNLOAD = (
    "https://www.kaggle.com/api/v1/datasets/download/{owner}/{dataset}"
)


class KaggleAuthError(RuntimeError):
    pass


@dataclass
class DownloadResult:
    dataset: str
    zip_path: Path
    extract_dir: Path


def _require_env(var: str) -> str:
    val = os.getenv(var)
    if not val:
        raise KaggleAuthError(
            f"Environment variable {var} is not set. Export KAGGLE_USERNAME and KAGGLE_KEY."
        )
    return val


def _filename_from_cd(content_disposition: Optional[str]) -> Optional[str]:
    if not content_disposition:
        return None
    match = re.search(r"filename\*=UTF-8" "([^;\n]+)", content_disposition)
    if match:
        return match.group(1)
    match = re.search(r'filename\s*=\s*"?([^;\n"]+)"?', content_disposition)
    if match:
        return match.group(1)
    return None


def download_dataset(
    slug: str,
    dest_zip_dir: Path,
    dest_extract_dir: Path,
    filename: Optional[str] = None,
    overwrite: bool = False,
    timeout: int = 1800,
) -> DownloadResult:
    """
    Download a Kaggle dataset ZIP via REST API using env vars KAGGLE_USERNAME/KAGGLE_KEY.

    Args:
        slug: "owner/dataset" (e.g., "wenewone/cub2002011").
        dest_zip_dir: where to save the .zip.
        dest_extract_dir: where to extract after download.
        filename: optional zip filename; if None, inferred from headers.
        overwrite: if False and file exists, skip re-download; still re-extract if needed.
        timeout: request timeout seconds.

    Returns:
        DownloadResult with paths.
    """
    owner, dataset = slug.split("/", 1)
    username = _require_env("KAGGLE_USERNAME")
    key = _require_env("KAGGLE_KEY")

    dest_zip_dir = Path(dest_zip_dir)
    dest_extract_dir = Path(dest_extract_dir)
    dest_zip_dir.mkdir(parents=True, exist_ok=True)
    dest_extract_dir.mkdir(parents=True, exist_ok=True)

    url = KAGGLE_DATASETS_DOWNLOAD.format(owner=owner, dataset=dataset)

    with requests.get(
        url, auth=(username, key), stream=True, timeout=timeout, allow_redirects=True
    ) as r:
        r.raise_for_status()
        inferred = (
            _filename_from_cd(r.headers.get("Content-Disposition")) or f"{dataset}.zip"
        )
        zip_name = filename or inferred
        zip_path = dest_zip_dir / zip_name
        if zip_path.exists() and not overwrite:
            pass
        else:
            tmp = zip_path.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            tmp.replace(zip_path)

    # Extract fresh each time into a dataset-specific subdir
    dataset_dir = dest_extract_dir / dataset
    if dataset_dir.exists():
        # Keep idempotency but allow clean re-extract
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dataset_dir)

    return DownloadResult(dataset=slug, zip_path=zip_path, extract_dir=dataset_dir)


def bulk_download(
    slugs: Iterable[str],
    raw_dir: Path,
    extracted_dir: Path,
    overwrite: bool = False,
) -> list[DownloadResult]:
    results = []
    for slug in slugs:
        results.append(
            download_dataset(
                slug,
                dest_zip_dir=raw_dir,
                dest_extract_dir=extracted_dir,
                overwrite=overwrite,
            )
        )
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download Kaggle datasets via env auth."
    )
    parser.add_argument("slugs", nargs="+", help="dataset slugs like owner/name")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--extracted-dir", default="data/extracted")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        results = bulk_download(
            args.slugs, Path(args.raw_dir), Path(args.extracted_dir), args.overwrite
        )
        for r in results:
            print(f"Downloaded {r.dataset} → {r.zip_path}")
            print(f"Extracted  {r.dataset} → {r.extract_dir}")
    except KaggleAuthError as e:
        print(f"Auth error: {e}", file=sys.stderr)
        sys.exit(2)
    except requests.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        sys.exit(3)
