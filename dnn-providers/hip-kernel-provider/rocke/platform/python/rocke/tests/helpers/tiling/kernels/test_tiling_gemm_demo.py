# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""End-to-end M1 test: build + compile + run the tiling GEMM demo on the GPU.

GPU-gated: skips cleanly where the rocke substrate / GPU are unavailable, so the CPU suite
stays green. On the gfx90a host it proves rocke.helpers.tiling's TileMma resolution drives a numerically
correct GEMM on real hardware. torch-free -- golden references are numpy, device I/O is DeviceMem.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "rocke.helpers.compile", reason="rocke substrate required (run on the gfx90a host)"
)
_hip = pytest.importorskip(
    "rocke.runtime.hip_module", reason="rocke HIP runtime required (run on the gfx90a host)"
)

# GPU probe without torch: get_device_arch(0) raises if no device is present.
try:
    _hip.get_device_arch(0)
except Exception as exc:  # noqa: BLE001 - any HIP/driver failure means "no usable GPU here"
    pytest.skip(f"no GPU available for the end-to-end demo: {exc}", allow_module_level=True)

from rocke.helpers.tiling.kernels.tiling_gemm_demo import (  # noqa: E402
    TilingGemmSpec,
    build_tiling_gemm,
    run_and_verify,
    run_and_verify_within_valid_space,
)


def test_build_tiling_gemm_resolves_and_builds() -> None:
    spec = TilingGemmSpec(tile=(16, 16, 16))
    kernel, mma_op = build_tiling_gemm(spec, 256, 256, 256, arch="gfx90a")
    assert mma_op.op_id == "mfma_f32_16x16x16f16"
    assert kernel.name.startswith("tiling_gemm_demo_")


def test_tiling_gemm_numeric_on_gfx90a() -> None:
    report = run_and_verify(256, 256, 256, arch="gfx90a")
    assert report["op_id"] == "mfma_f32_16x16x16f16"
    # Integer inputs -> the correct kernel is bit-exact against the reference.
    assert report["bit_exact"], report
    assert report["max_abs_diff"] == 0.0, report


def test_tiling_gemm_16x16x64_subtiled_k_on_gfx90a() -> None:
    # TILE_K=64 over a 16x16x16 atom -> 4 K subtiles accumulated per tile.
    spec = TilingGemmSpec(tile=(16, 16, 64), atom=(16, 16, 16))
    report = run_and_verify(256, 256, 256, spec=spec, arch="gfx90a")
    assert report["op_id"] == "mfma_f32_16x16x16f16"
    assert report["tile"] == (16, 16, 64)
    assert report["bit_exact"], report
    assert report["max_abs_diff"] == 0.0, report


def test_tiling_gemm_interleaved_a_transform_on_gfx90a() -> None:
    # Load A in the interleaved (AOS) layout, then transform_fragment (a register `reorder`) to the
    # MMA form. k_iter=2 makes the interleave non-trivial. Bit-exact vs the canonical-load path.
    spec = TilingGemmSpec(tile=(16, 16, 32), atom=(16, 16, 16))
    report = run_and_verify(256, 256, 256, spec=spec, arch="gfx90a", interleave_a=True)
    assert report["op_id"] == "mfma_f32_16x16x16f16"
    assert report["bit_exact"], report
    assert report["max_abs_diff"] == 0.0, report


def test_tiling_gemm_32x32x16_subtiled_mn_on_gfx90a() -> None:
    # 2x2 M/N subtile grid over a 16x16x16 atom -> the mma walks the M x N grid internally.
    spec = TilingGemmSpec(tile=(32, 32, 16), atom=(16, 16, 16))
    report = run_and_verify(256, 256, 256, spec=spec, arch="gfx90a")
    assert report["op_id"] == "mfma_f32_16x16x16f16"
    assert report["tile"] == (32, 32, 16)
    assert report["bit_exact"], report
    assert report["max_abs_diff"] == 0.0, report


def test_tiling_gemm_32x32x32_full_grid_on_gfx90a() -> None:
    # 2x2x2 M/N/K grid -> exercises the full subtile driver (M, N, and K > 1).
    spec = TilingGemmSpec(tile=(32, 32, 32), atom=(16, 16, 16))
    report = run_and_verify(256, 256, 256, spec=spec, arch="gfx90a")
    assert report["tile"] == (32, 32, 32)
    assert report["bit_exact"], report
    assert report["max_abs_diff"] == 0.0, report


def test_tiling_gemm_ragged_255_on_gfx90a() -> None:
    # 255x255x255 under a 16-tile: every edge tile overhangs by 1 in M, N, AND K -> the load
    # zero-pads and the store drops OOB. Bit-exact proves the clip is correct (no OOB touch).
    report = run_and_verify(255, 255, 255, arch="gfx90a")
    assert report["op_id"] == "mfma_f32_16x16x16f16"
    assert report["shape"] == (255, 255, 255)
    assert report["bit_exact"], report
    assert report["max_abs_diff"] == 0.0, report


def test_tiling_gemm_clips_within_valid_space_on_gfx90a() -> None:
    # Compute 250^3 into 256-allocated tensors: rows/cols 250..255 are VALID data the clip must
    # exclude by coordinate, and the C tail there must stay untouched (NaN). This isolates the
    # mask/store-guard from the OOB tensor-edge case.
    report = run_and_verify_within_valid_space(compute=250, alloc=256, arch="gfx90a")
    assert report["computed_bit_exact"], report   # valid rows 250..255 correctly excluded
    assert report["tail_untouched"], report        # store dropped OOB-of-compute (still NaN)


def test_tiling_gemm_ragged_k_only_on_gfx90a() -> None:
    # Only K is ragged (M, N tile-aligned): exercises the K-edge zero-pad (masked K loads 0 ->
    # MMA sums 0) and the per-axis fast path (M/N emit no check).
    report = run_and_verify(256, 256, 250, arch="gfx90a")
    assert report["shape"] == (256, 256, 250)
    assert report["bit_exact"], report
    assert report["max_abs_diff"] == 0.0, report


def test_tiling_gemm_atom_by_name_is_bit_exact_on_gfx90a() -> None:
    # Force the atom by explicit intrinsic name (escape hatch); same 2x2 grid, bit-exact.
    spec = TilingGemmSpec(tile=(32, 32, 16), atom="mfma_f32_16x16x16f16")
    report = run_and_verify(256, 256, 256, spec=spec, arch="gfx90a")
    assert report["op_id"] == "mfma_f32_16x16x16f16"
    assert report["bit_exact"], report
    assert report["max_abs_diff"] == 0.0, report


def test_tiling_gemm_subtile_order_is_bit_exact_on_gfx90a() -> None:
    # The subtile-iteration order is a knob; every order must give the same (bit-exact) result.
    spec = TilingGemmSpec(tile=(32, 32, 32), atom=(16, 16, 16), order="KNM")
    report = run_and_verify(256, 256, 256, spec=spec, arch="gfx90a")
    assert report["bit_exact"], report
    assert report["max_abs_diff"] == 0.0, report
