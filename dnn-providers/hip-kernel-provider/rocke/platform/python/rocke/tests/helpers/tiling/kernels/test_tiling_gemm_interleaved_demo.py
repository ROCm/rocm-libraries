# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""End-to-end tests for the speed-run interleaved/blocked GEMM demo.

GPU-gated: skips cleanly where the rocke substrate / GPU are unavailable. On the gfx90a host it
proves the register-blocked GEMM (authored from the basic tiling primitives + manual make_tile_desc
layouts) is numerically bit-exact against a numpy golden across register-tile sizes. torch-free.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "rocke.helpers.compile", reason="rocke substrate required (run on the gfx90a host)"
)
_hip = pytest.importorskip(
    "rocke.runtime.hip_module", reason="rocke HIP runtime required (run on the gfx90a host)"
)

try:
    _hip.get_device_arch(0)
except Exception as exc:  # noqa: BLE001 - any HIP/driver failure means "no usable GPU here"
    pytest.skip(f"no GPU available for the demo: {exc}", allow_module_level=True)

from rocke.helpers.tiling.kernels.tiling_gemm_interleaved_demo import (  # noqa: E402
    build_interleaved_gemm,
    run_and_verify_interleaved,
    run_and_verify_lds,
)


def test_build_interleaved_gemm_resolves() -> None:
    kernel, mma = build_interleaved_gemm(256, 256, 256, arch="gfx90a", tile_m=32, tile_n=32)
    assert mma.op_id == "mfma_f32_16x16x16f16"
    assert kernel.name.startswith("tiling_gemm_interleaved_")


@pytest.mark.parametrize("tile", [(16, 16, 16), (32, 32, 16), (64, 64, 16), (128, 128, 16), (32, 32, 32)])
def test_interleaved_gemm_bit_exact_on_gfx90a(tile: tuple[int, int, int]) -> None:
    tm, tn, tk = tile
    report = run_and_verify_interleaved(256, 256, 256, arch="gfx90a", tile_m=tm, tile_n=tn, tile_k=tk)
    assert report["op_id"] == "mfma_f32_16x16x16f16"
    assert report["bit_exact"], report
    assert report["max_abs_diff"] == 0.0, report


@pytest.mark.parametrize("macro,waves", [(128, (2, 2)), (256, (2, 2))])  # square warp tiles
def test_interleaved_gemm_cooperative_multiwave_on_gfx90a(macro: int, waves: tuple[int, int]) -> None:
    # Cooperative multi-wave: waves cooperatively load the macro tile into LDS, each reads its warp
    # tile (canonical) and drives it. Bit-exact vs numpy.
    wm, wn = waves
    report = run_and_verify_interleaved(
        256, 256, 256, arch="gfx90a", tile_m=macro, tile_n=macro, waves_m=wm, waves_n=wn
    )
    assert report["bit_exact"], report


def test_interleaved_gemm_deeper_k_on_gfx90a() -> None:
    # tile_k=32 -> k_iter=2 subtiles accumulated per wave tile (square M/N).
    report = run_and_verify_interleaved(256, 256, 256, arch="gfx90a", tile_m=64, tile_n=64, tile_k=32)
    assert report["bit_exact"], report

# NOTE: rectangular wave tiles (m_sub != n_sub) currently trip the driver's whole-fragment
# validate_operands (A/B wave fragments differ in register count, so their K-dists differ in
# length). The K-alignment should be checked per-ATOM. Deferred -- square tiles are the focus.


def test_lds_staged_gemm_bit_exact_on_gfx90a() -> None:
    # LDS round-trip (single wave / single atom) through the unified load/store_fragment verbs.
    report = run_and_verify_lds(256, 256, 256, arch="gfx90a")
    assert report["bit_exact"], report
