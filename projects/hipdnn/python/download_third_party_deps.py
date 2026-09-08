#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Download pinned third-party source archives for hipDNN Python CI builds."""

import argparse
import dataclasses
import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com"


@dataclasses.dataclass(frozen=True)
class Dependency:
    archive_name: str
    extracted_dir_name: str
    sha256: str


DEPENDENCIES = (
    Dependency(
        archive_name="nanobind-2.12.0.tar.gz",
        extracted_dir_name="nanobind-2.12.0",
        sha256="01f1f0cd0398743c18f33d07ae36ad410bd7f4a1e90683b508504de897d6e629",
    ),
    Dependency(
        archive_name="robin-map-1.4.1.tar.gz",
        extracted_dir_name="robin-map-1.4.1",
        sha256="0e3f53a377fdcdc5f9fed7a4c0d4f99e82bbb64175233bd13427fef9a771f4a1",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download pinned third-party source archives for hipDNN Python builds."
    )
    parser.add_argument("--deps-dir", required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    return parser.parse_args()


def download_and_extract_dependency(
    *, deps_dir: Path, base_url: str, dependency: Dependency
) -> Path:
    deps_dir.mkdir(parents=True, exist_ok=True)

    archive_path = deps_dir / dependency.archive_name
    urllib.request.urlretrieve(f"{base_url}/{dependency.archive_name}", archive_path)

    actual_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if actual_sha256 != dependency.sha256:
        raise RuntimeError(
            f"{dependency.archive_name} SHA256 mismatch: expected {dependency.sha256}, got {actual_sha256}"
        )

    extracted_dir = deps_dir / dependency.extracted_dir_name
    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)

    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(deps_dir, filter="data")

    if not extracted_dir.is_dir():
        raise RuntimeError(
            f"{dependency.extracted_dir_name} source directory not found after extracting {dependency.archive_name}"
        )

    return extracted_dir


def main() -> int:
    args = parse_args()
    deps_dir = Path(args.deps_dir)
    base_url = args.base_url.rstrip("/")

    for dependency in DEPENDENCIES:
        extracted_dir = download_and_extract_dependency(
            deps_dir=deps_dir, base_url=base_url, dependency=dependency
        )
        print(f"{dependency.archive_name}: extracted to {extracted_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
