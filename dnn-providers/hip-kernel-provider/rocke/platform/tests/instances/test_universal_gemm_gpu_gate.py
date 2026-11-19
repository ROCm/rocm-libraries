# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-only checks of the explicit UniversalGemm gate's verdicts."""

import pytest

from rocke.runtime import comgr, hip_module
from rocke.tests import gpu
from rocke.tests import universal_gemm_gpu_gate as gate


@pytest.mark.parametrize("arch", ("gfx942", "gfx950", "gfx1250"))
def test_gate_probes_current_device_and_enforces_required_arch(monkeypatch, arch):
    checked = []
    monkeypatch.setattr(hip_module, "current_device", lambda: 3)

    def device_arch(device):
        checked.append(device)
        return "gfx90a"

    monkeypatch.setattr(hip_module, "get_device_arch", device_arch)
    required = f"ROCKE_REQUIRE_{arch.upper()}"
    monkeypatch.delenv(required, raising=False)
    assert gate.main(["--arch", arch]) == gate.SKIP
    monkeypatch.setenv(required, "1")
    assert gate.main(["--arch", arch]) == 1
    assert checked == [3, 3]


def test_gate_rejects_missing_compiler_and_zero_comparisons(monkeypatch):
    monkeypatch.setattr(gpu, "probe_device_arch", lambda: ("gfx950", ""))
    monkeypatch.setattr(comgr, "loaded_compiler_info", lambda: None)
    assert gate.main(["--arch", "gfx950"]) == 1
    monkeypatch.setattr(comgr, "loaded_compiler_info", object)
    monkeypatch.setattr(hip_module, "Runtime", object)
    monkeypatch.setitem(gate._GPU_CASES, "gfx950", lambda rt, rng: 0)
    assert gate.main(["--arch", "gfx950"]) == 1
    monkeypatch.setitem(gate._GPU_CASES, "gfx950", lambda rt, rng: 1)
    assert gate.main(["--arch", "gfx950"]) == 0
