# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-only checks of the explicit UniversalGemm gate's verdicts."""

import struct
from types import SimpleNamespace

import numpy as np
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


@pytest.mark.parametrize("omit_d0", [False, True])
def test_gfx950_gate_checks_fused_multi_abd_addend(monkeypatch, capsys, omit_d0):
    from rocke.instances.common import gemm_multi_abd

    build = gemm_multi_abd.build_gemm_multi_abd

    def build_swizzled(spec, *, arch):
        tile = spec.base.tile
        assert (tile.tile_m, tile.tile_n, tile.tile_k) == (64, 128, 32)
        assert spec.base.trait.lds_swizzle and spec.base.trait.epilogue == "cshuffle"
        assert spec.d_operands == (("D0", "add"),)
        return build(spec, arch=arch)

    monkeypatch.setattr(gemm_multi_abd, "build_gemm_multi_abd", build_swizzled)
    monkeypatch.setattr(gate, "tile_copy_cases", tuple)
    monkeypatch.setattr(gate, "_grouped_gfx950", lambda rt, rng: 0)
    monkeypatch.setattr(gate, "_batched_gfx950", lambda rt, rng: 0)
    monkeypatch.setattr(gate, "_compile", lambda kernel, arch: kernel)

    def launch(rt, kernel, arrays, grid, block_size, pack):
        a, b, c, d0 = arrays
        assert a.shape == (128, 96) and b.shape == (256, 96)
        assert c.shape == d0.shape == (128, 256)
        assert np.isnan(c).all() and np.all(d0 != 0)
        assert grid == (2, 2, 1) and block_size == 128
        assert "multi_abd_swizzled" in kernel.name
        assert struct.unpack("<QQQiii4xQ", pack((1, 2, 3, 4))) == (
            1,
            2,
            3,
            128,
            256,
            96,
            4,
        )
        c[:] = a.astype(np.float32) @ b.astype(np.float32).T
        if not omit_d0:
            c += d0

    monkeypatch.setattr(gate, "_launch_buffers", launch)
    rng = np.random.default_rng(6)
    if omit_d0:
        with pytest.raises(AssertionError, match="multi_abd_swizzled"):
            gate._gfx950(None, rng)
        assert "numeric_pass" not in capsys.readouterr().out
    else:
        assert gate._gfx950(None, rng) == 1
        assert '"case": "multi_abd_swizzled"' in capsys.readouterr().out


@pytest.mark.parametrize("damage_canary", [False, True])
def test_gfx1250_ragged_case_uses_exact_inputs_and_checks_output_guard(
    monkeypatch, damage_canary
):
    monkeypatch.setattr(gate, "build_universal_gemm", lambda spec, *, arch: spec)
    monkeypatch.setattr(
        gate,
        "_compile",
        lambda spec, arch: SimpleNamespace(
            spec=spec, llvm_text="call void @llvm.amdgcn.tensor.load.to.lds"
        ),
    )
    compared = []
    monkeypatch.setattr(
        gate, "_check_fp16", lambda label, a, b, c: compared.append(label)
    )

    def launch(rt, artifact, a, b, output, grid, block_size, scalars):
        if not artifact.spec.trait.pad_m and not artifact.spec.trait.pad_n:
            return
        assert artifact.spec.trait.pad_m and artifact.spec.trait.pad_n
        m, n, k = scalars
        assert (m, n, k) == (259, 387, 80)
        assert a.shape == (m, k) and b.shape == (n, k)
        assert a.nbytes == 2 * m * k and b.nbytes == 2 * n * k
        assert grid == (4, 3, 1)
        assert np.isnan(output[: m * n]).all()
        assert output.size > m * n
        np.testing.assert_array_equal(output[m * n :], 137)
        if damage_canary:
            output[m * n] = 0

    monkeypatch.setattr(gate, "_run_buffers", launch)
    if damage_canary:
        with pytest.raises(AssertionError, match="TDM output canary was overwritten"):
            gate._gfx1250(None, np.random.default_rng(6))
        assert "ragged_mn" not in compared
    else:
        assert gate._gfx1250(None, np.random.default_rng(6)) == 15
        assert compared[-1] == "ragged_mn"
