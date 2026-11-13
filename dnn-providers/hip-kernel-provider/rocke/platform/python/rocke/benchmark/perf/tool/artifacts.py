# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Opt-in, portable measurement bundles. Original profiler files are never rewritten."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rocke.benchmark.perf import report, schema

BUNDLE_SCHEMA = "rocke.bench.artifacts/v1"

# Exact filenames this writer controls; everything else the profiler wrote is
# `profiler_output` so a consumer never has to guess what it may parse.
_KINDS = {
    "pmc.txt": "counter_config",
    "measurement.json": "measurement",
    "comparison.json": "comparison",
}


class ArtifactBundle:
    """Own one new directory; only a finalized manifest denotes a complete bundle."""

    def __init__(self, path: str, *, repeats: int, warmup: int, match: str | None):
        self.path = Path(path).expanduser().absolute()
        self.manifest: dict[str, Any] = {
            "schema": BUNDLE_SCHEMA,
            "measurement_schema": schema.SCHEMA_VERSION,
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "requested_samples": repeats,
            "selection": {"match_kernel": match, "warmup_per_pass": warmup},
            "semantics": {
                "raw": "Original profiler output; includes warmup and other kernels. Do not combine repeats for WaveScope import.",
                "measurement": "Normalized target-kernel counter medians after per-pass warmup removal; repeated runs reduced to median and spread.",
                "att_association": "unbound",
            },
            "samples": [],
            "files": [],
        }

    def _write(self, relative: str, value: dict) -> None:
        dest = self.path / relative
        temporary = dest.with_name(dest.name + ".tmp")
        try:
            temporary.write_text(report.to_json(value) + "\n", encoding="utf-8")
            temporary.replace(dest)
        finally:
            temporary.unlink(missing_ok=True)

    def __enter__(self):
        self.path.mkdir(parents=True, exist_ok=False)
        self._write("manifest.json", self.manifest)
        return self

    def sample_dir(self, index: int) -> Path:
        relative = f"samples/{index:04d}"
        directory = self.path / relative
        directory.mkdir(parents=True, exist_ok=False)
        self.manifest["samples"].append(
            {
                "sample_index": index,
                "status": "running",
                "raw_dir": relative + "/raw",
            }
        )
        self._write("manifest.json", self.manifest)
        return directory / "raw"

    def add_sample(self, index: int, record: dict) -> None:
        relative = f"samples/{index:04d}/measurement.json"
        self._write(relative, record)
        sample = self.manifest["samples"][index]
        sample.update(
            {
                "status": "complete",
                "run_id": record["run"]["run_id"],
                "measurement": relative,
                "profile_capture": record.get("profile_capture"),
            }
        )
        self._write("manifest.json", self.manifest)

    def finish(self, record: dict, comparison: dict) -> None:
        # `profile_capture` is per-repeat and the CLI drops it from the aggregate
        # before this point; samples/ keeps each capture's own status.
        self._write("measurement.json", record)
        self._write("comparison.json", comparison)
        self.manifest.update(
            {
                "measurement": "measurement.json",
                "comparison": "comparison.json",
            }
        )
        self._inventory()
        self.manifest["status"] = "complete"
        self._write("manifest.json", self.manifest)

    def _inventory(self) -> None:
        files = []
        for path in sorted(self.path.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"artifact symlink cannot be exported: {path}")
            if not path.is_file() or path.name == "manifest.json":
                continue
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            relative = path.relative_to(self.path).as_posix()
            kind = _KINDS.get(path.name, "profiler_output")
            if path.name.endswith("counter_collection.csv"):
                # rocprofv3 prefixes these per run/pass, so match the suffix.
                kind = "pmc_csv"
            files.append(
                {
                    "path": relative,
                    "kind": kind,
                    "bytes": path.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
            )
        self.manifest["files"] = files

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None or self.manifest["status"] != "complete":
            self.manifest["status"] = "failed"
            self.manifest["error"] = {
                "type": exc_type.__name__ if exc_type else "IncompleteBundle"
            }
            for sample in self.manifest["samples"]:
                if sample["status"] == "running":
                    sample["status"] = "failed"
            try:
                self._inventory()
                self._write("manifest.json", self.manifest)
            except (OSError, ValueError):
                # The initial running manifest still refuses a complete-capture claim.
                if exc_type is None:
                    raise
        return False
