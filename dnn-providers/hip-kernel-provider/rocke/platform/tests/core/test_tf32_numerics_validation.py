# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-only CLI checks with injected outputs; no GPU correctness evidence."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from rocke.examples.gfx942 import tf32_numerics as numerics
from rocke.runtime import comgr, hip_module


def _run_mocked_cli(fault: str, output_dir: str) -> None:
    launches = 0

    def corpus(m):
        bits = np.array([0x3F801001], dtype=np.uint32)
        a = np.zeros((3, m, 128 // m), dtype=np.float32)
        b = np.zeros_like(a)
        a[:, :, 0] = bits.view(np.float32)[0]
        b[:, :, 0] = 1
        c = np.zeros((3, m, m), dtype=np.float32)
        return a, b, c, bits, 2

    def launch(blob, name, a, b, c):
        nonlocal launches
        launches += 1
        mode = name.rsplit("_", 1)[1]
        pa, pb = a.view(np.uint32).copy(), b.view(np.uint32).copy()
        if mode == "rne":
            pa, pb = numerics.rne_bits(pa), numerics.rne_bits(pb)
        ra, rb = pa, pb
        if mode != "fp32":
            ra, rb = pa & np.uint32(0xFFFFE000), pb & np.uint32(0xFFFFE000)
        d = np.einsum("bik,bjk->bij", ra.view(np.float32), rb.view(np.float32)) + c
        if fault == "A preparation":
            pa.flat[0] ^= np.uint32(1)
        elif fault == "B preparation":
            pb.flat[0] ^= np.uint32(1)
        elif fault == "dense reference":
            d[-1, 0, 0] += 1
        elif fault == "single product":
            d[0, 0, 0] += 1
        elif (fault == "variant parity" and mode == "carrier") or (
            fault == "backend parity" and launches > 6
        ):
            # Stay within the dense-reference tolerance while breaking exact parity.
            d[-1, 0, 0] = np.nextafter(d[-1, 0, 0], np.float32(np.inf))
        return d, pa, pb

    def native_ir(m, mode):
        kernel = numerics.build_tf32_mma_probe(numerics.Tf32MmaProbeSpec(m, mode))
        return "wrong IR" if fault == "builder parity" else numerics.serialize(kernel)

    native = SimpleNamespace(
        tf32_mma_probe_serialize_ir=native_ir,
        lower_serialized_ir=lambda *a, **k: (
            "wrong LLVM" if fault == "lowerer parity" else "mock LLVM"
        ),
    )
    argv = ["tf32_numerics", "--output-dir", output_dir, "--shape", "16"]
    with (
        patch.object(numerics, "corpus", corpus),
        patch.object(numerics, "launch", launch),
        patch.object(numerics, "_resolve_llvm_flavor", return_value="llvm23"),
        patch.object(
            numerics, "_lower_kernel_to_llvm_python", return_value="mock LLVM"
        ),
        patch.object(numerics, "lower_kernel_to_hip", return_value="mock HIP"),
        patch.object(hip_module, "get_device_arch", return_value="gfx942"),
        patch.object(comgr, "build_hsaco_from_llvm_ir", return_value=(b"mock", None)),
        patch.object(comgr, "resolved_lib_path", return_value="mock COMGR"),
        patch.dict(sys.modules, {"rocke_engine": native}),
        patch.object(sys, "argv", argv),
    ):
        numerics.main()


@pytest.mark.parametrize("optimization", ["normal", "flag", "environment"])
@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "A preparation",
        "B preparation",
        "dense reference",
        "single product",
        "builder parity",
        "lowerer parity",
        "variant parity",
        "backend parity",
    ],
)
def test_cli_validation_survives_optimization(tmp_path, optimization, fault):
    env = os.environ.copy()
    env.pop("PYTHONOPTIMIZE", None)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    env["ROCKE_BACKEND"] = "python"
    if optimization == "environment":
        env["PYTHONOPTIMIZE"] = "1"
    command = [sys.executable]
    if optimization == "flag":
        command.append("-O")
    command += [
        "-c",
        (
            "import runpy, sys; "
            "runpy.run_path(sys.argv[1])['_run_mocked_cli'](sys.argv[2], sys.argv[3])"
        ),
        str(Path(__file__).resolve()),
        fault,
        str(tmp_path),
    ]
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=60, check=False
    )
    report = json.loads((tmp_path / "results.json").read_text())
    if fault == "none":
        assert result.returncode == 0, result.stderr
        assert report["status"] == "pass"
        assert len(report["results"]) == 12
    else:
        assert result.returncode != 0, result.stdout
        assert report["status"] == "fail"
        assert fault in report["error"]
        assert fault in result.stderr
