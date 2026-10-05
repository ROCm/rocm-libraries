# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Remove stored inputs only after proving exact reproduction of a locked corpus."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from .architectures import get_architecture
from .artifact import validate_bundle
from .cli import _validate_cases, load_bundle
from .contract import (
    INPUT_GENERATOR,
    SCHEMA_VERSION,
    checked_inputs,
    file_digest,
    payload_digests,
    write_json,
)


def remove_stored_inputs(
    bundle: Path, lock: Path, output: Path, architecture: str = "gfx942"
) -> None:
    """Preserve kernels and qualification evidence while changing only storage.

    The old lock must come from trusted version control. No independent answers
    or error budgets are recomputed: every generated tensor must reproduce the
    stored dtype, shape, and bytes as well as its qualified digest. The original
    frozen runner/runtime are copied unchanged and consume temporary input files.
    """
    if output.exists():
        raise FileExistsError(output)
    if output.resolve().is_relative_to(bundle.resolve()):
        raise ValueError("migration output must be outside the original bundle")
    validate_bundle(bundle, lock)
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest["schema"] != 1:
        raise ValueError("input storage migration requires a schema-1 bundle")
    _validate_cases(manifest, architecture)
    removed = set()
    for case in get_architecture(architecture).CASES:
        entry = manifest["cases"][case.id]
        arrays = checked_inputs(case, entry["input_digests"])
        relative = f"cases/{case.id}/inputs.npz"
        with np.load(bundle / "payload" / relative, allow_pickle=False) as stored:
            if set(stored.files) != set(arrays):
                raise ValueError(f"unexpected stored input names: {case.id}")
            for name, array in arrays.items():
                original = stored[name]
                if (
                    original.shape != array.shape
                    or original.dtype != array.dtype
                    or original.tobytes() != array.tobytes()
                ):
                    raise ValueError(
                        f"stored SDPA input differs from generator: {case.id}/{name}"
                    )
        removed.add(relative)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix="sdpa-migrate-"
    ) as temporary:
        staged = Path(temporary) / "bundle"
        payload = staged / "payload"
        payload.mkdir(parents=True)
        for relative in manifest["files"]:
            if relative not in removed:
                destination = payload / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(bundle / "payload" / relative, destination)
        manifest["schema"] = SCHEMA_VERSION
        manifest["input_generation"] = INPUT_GENERATOR
        manifest["storage_migration"] = {
            "source_manifest_sha256": file_digest(bundle / "manifest.json"),
            "method": "regenerate-and-compare-quantized-input-bytes-v1",
            "numpy_version": np.__version__,
        }
        manifest["files"] = payload_digests(payload)
        write_json(staged / "manifest.json", manifest)
        write_json(
            staged / "qualification-lock.json",
            {
                "schema": SCHEMA_VERSION,
                "baseline_revision": manifest["baseline_revision"],
                "manifest_sha256": file_digest(staged / "manifest.json"),
            },
        )
        load_bundle(
            staged, staged / "qualification-lock.json", architecture=architecture
        )
        staged.rename(output)
    print(
        f"Removed {len(removed)} stored input files; kernels and qualification evidence unchanged."
    )
