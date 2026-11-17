# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The gfx950 scaled-MFMA emitter follows the differential batch protocol."""

import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_diff


@pytest.mark.parametrize("flavor", ["llvm20", "llvm22", "llvm23"])
def test_scaled_mfma_batch_reenters_and_terminates(monkeypatch, flavor):
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", flavor)
    batch = run_diff._run_python_batch("gfx950_block_scaled_gemm", [0, 49, 0, 50])
    assert batch.error is None
    assert len(batch.emissions) == 4
    first, last, repeated, ended = batch.emissions
    assert all(
        emission.returncode == 0 and emission.has_output
        for emission in (first, last, repeated)
    )
    assert first.output_sha == repeated.output_sha
    assert first.output_sha != last.output_sha
    assert run_diff._is_end_emission(ended)

    # Compare the in-process batch with the original single-config CLI path.
    process = subprocess.run(
        [
            sys.executable,
            str(run_diff.PARITY / "gfx950_block_scaled_gemm_emit.py"),
            "0",
            "ll",
        ],
        env=run_diff._python_emitter_env(run_diff.PARITY),
        capture_output=True,
        check=True,
        timeout=run_diff.TIMEOUT,
    )
    assert hashlib.sha256(process.stdout).hexdigest() == first.output_sha
