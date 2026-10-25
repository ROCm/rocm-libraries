# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Qualification for the toolchain-free direct gfx1151 GEMM slice."""

from __future__ import annotations

import pytest
from rocke.core.ir_serialize import serialize
from rocke.helpers import (
    compile_gfx1151_wmma_gemm_direct,
    make_gemm_manifest,
    write_artifact,
)
from rocke.instances.gfx1151.wmma_gemm import WmmaGemmSpec, build_wmma_gemm
from rocke.run_manifest import run_manifest
from rocke.runtime import code_object
from rocke.runtime.code_object_elf import inspect_hsaco_semantics
from rocke.runtime.hip_module import HipError, get_device_arch


def _require_direct_library() -> str:
    path = code_object._direct_library_path()
    if path is None:
        pytest.skip("build the Rust cdylib or set ROCKE_CODEOBJ_DIRECT_LIB")
    return path


def test_direct_gemm_compiles_without_comgr_toolchain_or_cache(monkeypatch) -> None:
    _require_direct_library()
    monkeypatch.delenv("ROCKE_COMGR_LIB", raising=False)

    kernel = build_wmma_gemm(
        WmmaGemmSpec(name="rocke_direct_wmma_gemm"), arch="gfx1151"
    )
    hsaco, timings, compiler = code_object.build_direct_gfx1151_wmma_gemm_hsaco(
        serialize(kernel)
    )
    semantics = inspect_hsaco_semantics(hsaco)

    assert timings.executable > 0.0
    assert compiler.backend == "rust-direct-gemm"
    assert compiler.details["uses_comgr"] is False
    assert compiler.details["uses_llvm"] is False
    assert compiler.details["uses_external_tools"] is False
    assert compiler.details["uses_cache"] is False
    assert b"Linker: AMD LLVM" not in hsaco
    assert dict(semantics.kernels[0])[".name"] == (
        code_object.DIRECT_GFX1151_WMMA_GEMM_SYMBOL
    )
    assert semantics.flags == 0x4A


def test_direct_gemm_typed_pipeline_has_no_llvm_text() -> None:
    _require_direct_library()
    artifact = compile_gfx1151_wmma_gemm_direct(capture_ir_text=False)
    assert artifact.kernel_name == code_object.DIRECT_GFX1151_WMMA_GEMM_SYMBOL
    assert artifact.llvm_text == ""
    assert artifact.ir_text == ""
    assert artifact.isa == "amdgcn-amd-amdhsa--gfx1151"
    assert artifact.timings["ir_lower_llvm"] == 0.0


@pytest.mark.parametrize(
    ("block_x_is_m", "grid_order"),
    [(True, "MN"), (False, "NM")],
)
def test_direct_gemm_numeric_on_gfx1151(
    tmp_path, block_x_is_m: bool, grid_order: str
) -> None:
    _require_direct_library()
    try:
        arch = get_device_arch()
    except (HipError, OSError) as error:
        pytest.skip(f"HIP device is unavailable: {error}")
    if arch != "gfx1151":
        pytest.skip(f"direct WMMA GEMM requires gfx1151, found {arch}")
    pytest.importorskip("numpy")

    artifact = compile_gfx1151_wmma_gemm_direct(
        capture_ir_text=False,
        block_x_is_m=block_x_is_m,
    )
    manifest = make_gemm_manifest(
        artifact=artifact,
        block_m=16,
        block_n=16,
        block_k=16,
        threads_per_block=32,
        default_shape=(128, 128, 128),
        grid_order=grid_order,
        atoms=["wmma_f32_16x16x16_f16"],
    )
    manifest["warmup_iters"] = 2
    manifest["timed_iters"] = 5
    write_artifact(artifact, tmp_path, manifest)
    summary = run_manifest(
        tmp_path / "manifest.json",
        tmp_path / f"{artifact.kernel_name}.hsaco",
        shape=(128, 128, 128),
        verify=True,
    )
    assert summary.bad_count == 0
    assert summary.max_abs_diff <= 1e-2
