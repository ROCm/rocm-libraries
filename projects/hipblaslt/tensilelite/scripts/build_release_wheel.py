#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Build and publish the canonical wheel from one refreshed metadata record."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from check_release_wheel_contents import validate
from stage_release_source import stage_source


def build_wheel(source: Path, metadata_file: Path, stage: Path, destination: Path) -> Path:
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    revision = metadata["source_revision"]
    version = metadata["distribution_version"]
    environment = dict(os.environ)
    for key in ("TENSILELITE_SOURCE_REVISION", "TENSILELITE_ROCM_VERSION"):
        environment.pop(key, None)
    if revision != "none":
        environment["TENSILELITE_SOURCE_REVISION"] = revision
        if "SOURCE_DATE_EPOCH" not in environment and shutil.which("git"):
            # Ask for this resolved commit's timestamp, even if HEAD has moved.
            timestamp = subprocess.run(
                ["git", "-c", "safe.directory=*", "show", "-s", "--format=%ct", revision[:40]],
                cwd=source,
                capture_output=True,
                text=True,
            )
            if timestamp.returncode == 0 and timestamp.stdout.strip().isdigit():
                environment["SOURCE_DATE_EPOCH"] = timestamp.stdout.strip()
    if metadata.get("rocm_identity"):
        environment["TENSILELITE_ROCM_VERSION"] = metadata["rocm_identity"]
    if stage.exists():
        shutil.rmtree(stage)
    staged_source = stage / "source"
    stage_source(source, staged_source)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--disable-pip-version-check",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(stage),
            str(staged_source),
        ],
        env=environment,
        check=True,
    )
    wheel = stage / f"tensilelite-{version}-py3-none-any.whl"
    problems = validate(wheel, staged_source, version, revision)
    if problems:
        raise RuntimeError("Invalid canonical wheel: " + "; ".join(problems))
    destination.mkdir(parents=True, exist_ok=True)
    published = destination / wheel.name
    shutil.copy2(wheel, published)
    # This directory is owned by the canonical wheel target. A commit/version
    # change must not leave an older wheel available for accidental selection.
    for old in destination.glob("tensilelite-*.whl"):
        if old != published:
            old.unlink()
    (destination / ".published").write_text(
        json.dumps({"wheel": published.name, "metadata": metadata}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return published


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args(argv)
    print(build_wheel(args.source, args.metadata, args.stage, args.destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
