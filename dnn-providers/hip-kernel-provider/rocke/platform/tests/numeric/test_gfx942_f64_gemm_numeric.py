# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx942 numeric regressions.

Named for the fp64 GEMM work this file was created for; the f16/bf16
``lds_swizzle`` granule regression also lives here so the gfx94X CI lane runs
it on real gfx942 hardware. The fp64 GEMM numerics run on every target that
admits fp64 GEMM and has run them on real hardware.
"""
import ctypes
import importlib.util
import os
import struct

import numpy as np
import pytest
from rocke.core.arch import ArchTarget, known_arches
from rocke.core.backend import universal_gemm_spec_to_dict
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
    is_valid_spec,
)
from rocke.numeric.bf16 import bf16_to_f32, f32_to_bf16
from rocke.runtime.hip_module import get_device_arch


def _require_gfx942():
    arch = get_device_arch()
    if not arch or arch.split(":")[0] != "gfx942":
        if os.getenv("ROCKE_REQUIRE_GFX942") == "1":
            pytest.fail(f"Required gfx942 device absent: {arch}")
        pytest.skip("requires gfx942")


# Targets whose fp64 GEMM numerics have run on real hardware.
_F64_GEMM_HARDWARE_RUN = ("gfx90a", "gfx942", "gfx950")


def _require_f64_gemm_arch():
    arch = (get_device_arch() or "").split(":")[0]
    if os.getenv("ROCKE_REQUIRE_GFX942") == "1" and arch != "gfx942":
        pytest.fail(f"Required gfx942 device absent: {arch}")
    if arch in _F64_GEMM_HARDWARE_RUN:
        return arch
    if arch in known_arches() and any(
        op.a_dtype == "fp64" for op in ArchTarget.from_gfx(arch).mma.ops
    ):
        pytest.skip(f"fp64 GEMM has not run on {arch} hardware yet; enable once it has")
    pytest.skip("requires a GPU with fp64 GEMM")


def _engines():
    # The installed wheel has no pybind extension; demand native parity
    # whenever the binding is supplied.
    if importlib.util.find_spec("rocke_engine") is None:
        return ("python",)
    return ("python", "cpp")


def _spec(
    tile,
    pipeline,
    epilogue,
    pad=False,
    warp_tile=(16, 16, 4),
    acc="fp64",
    batched=False,
    **trait,
):
    tile_m, tile_n, tile_k, warp_m, warp_n = tile
    return UniversalGemmSpec(
        name="f64_gemm",
        tile=TileSpec(tile_m, tile_n, tile_k, warp_m, warp_n, 1, *warp_tile),
        trait=TraitSpec(
            pipeline=pipeline,
            epilogue=epilogue,
            pad_m=pad,
            pad_n=pad,
            pad_k=pad,
            **trait,
        ),
        data=DataSpec("fp64", "fp64", "fp64", acc),
        batched=batched,
    )


def _llvm(spec, engine, arch="gfx942"):
    if engine == "python":
        kernel = build_universal_gemm(spec, arch=arch)
        return _lower_kernel_to_llvm_python(kernel, arch=arch)
    import rocke_engine as native

    return native.gemm_lower_llvm(universal_gemm_spec_to_dict(spec), arch=arch)


def _launch(ll, spec, a, bt, b_device=None, sorted_ids=None, c=None, arch="gfx942"):
    from rocke.runtime.comgr import build_hsaco_from_llvm_ir
    from rocke.runtime.hip_module import Runtime

    *batch, m, k = a.shape
    n = bt.shape[-2]
    batches = batch[0] if batch else 1
    blob, _ = build_hsaco_from_llvm_ir(ll, isa=f"amdgcn-amd-amdhsa--{arch}")
    runtime = Runtime()
    module = runtime.load_module(blob)
    if c is None:
        c = np.full((*batch, m, n), np.nan)
    arrays = [
        a.copy(),
        (bt if b_device is None else b_device).copy(),
        c.copy(),
    ]
    if sorted_ids is not None:
        arrays.append(sorted_ids.copy())
    ptrs = []
    try:
        for array in arrays:
            ptr = runtime.alloc(array.nbytes)
            ptrs.append(ptr)
            runtime.memcpy_h2d(
                ptr, (ctypes.c_ubyte * array.nbytes).from_buffer(array), array.nbytes
            )
        t = spec.tile
        grid = (-(-n // t.tile_n), -(-m // t.tile_m), batches)
        if sorted_ids is None:
            args = struct.pack("3Q3i", *ptrs, m, n, k)
        else:
            args = struct.pack(
                "3Q6iQi", *ptrs[:3], m, n, k, m * k, n * k, m * n, ptrs[3], m
            )
        runtime.launch_blocking(
            module.get_function(spec.kernel_name()), grid, (spec.block_size, 1, 1), args
        )
        c = arrays[2]
        runtime.memcpy_d2h(
            (ctypes.c_ubyte * c.nbytes).from_buffer(c), ptrs[2], c.nbytes
        )
        return c
    finally:
        for ptr in ptrs:
            runtime.free(ptr)
        module.unload()


def _preshuffle_b(bt, tile_n, tile_k):
    n, k = bt.shape
    return np.ascontiguousarray(
        bt.reshape(n // tile_n, tile_n, k // tile_k, tile_k).transpose(2, 0, 1, 3)
    )


@pytest.mark.parametrize(
    "tile, pipeline, epilogue, pad, trait",
    [
        ((64, 64, 16, 2, 2), "compv4", "default", True, {}),
        ((64, 64, 16, 2, 2), "compv3", "cshuffle", False, {}),
        ((32, 32, 8, 1, 1), "mem", "default", False, {}),
        ((16, 16, 4, 1, 1), "mem", "cshuffle", True, {}),
        ((64, 64, 16, 2, 2), "compv3", "cshuffle", False, {"chiplet_swizzle": True}),
        ((64, 64, 32, 2, 2), "compv4", "default", False, {"lds_k_pad": 2}),
        ((64, 64, 16, 2, 2), "mem", "default", False, {"lds_k_pad": 4}),
        ((64, 64, 32, 2, 2), "compv4", "default", False, {"preshuffle_b": True}),
        ((32, 32, 8, 1, 1), "compv3", "cshuffle", False, {"preshuffle_b": True}),
        ((64, 64, 16, 2, 2), "compv4", "default", False, {"lds_swizzle": True}),
        ((32, 32, 8, 1, 1), "mem", "default", False, {"lds_swizzle": True}),
        ((64, 64, 16, 2, 2), "compv3", "cshuffle", False, {"direct_to_lds": True}),
        (
            (64, 64, 32, 2, 2),
            "compv4",
            "default",
            False,
            {"direct_to_lds": True, "dtl_prefetch": True},
        ),
        (
            (64, 64, 16, 2, 2),
            "compv4",
            "default",
            False,
            {"direct_to_lds": True, "lds_swizzle": True},
        ),
        (
            (32, 32, 8, 1, 1),
            "compv3",
            "cshuffle",
            False,
            {"direct_to_lds": True, "dtl_prefetch": True, "lds_swizzle": True},
        ),
    ],
)
def test_gfx942_f64_gemm_numeric(tile, pipeline, epilogue, pad, trait):
    arch = _require_f64_gemm_arch()
    spec = _spec(tile, pipeline, epilogue, pad, **trait)
    if spec.trait.direct_to_lds and ArchTarget.from_gfx(arch).async_lds_max_dwords < 1:
        assert is_valid_spec(spec, arch=arch) == (
            False,
            f"direct_to_lds: {arch} has no DRAM->LDS DMA instruction",
        )
        pytest.skip(f"{arch} has no DRAM->LDS DMA instruction")
    assert is_valid_spec(spec, arch=arch) == (True, "ok")
    tile_m, tile_n, tile_k = tile[:3]
    # Padded configs use extents that are not tile multiples.
    trim = 1 if pad else 0
    m, n, k = 2 * tile_m - trim, 3 * tile_n - trim, 4 * tile_k - trim
    rng = np.random.default_rng(tile_m * 100 + tile_k)

    def b_device(bt):
        if spec.trait.preshuffle_b:
            return _preshuffle_b(bt, tile_n, tile_k)
        return None

    for engine in _engines():
        ll = _llvm(spec, engine, arch)
        a = rng.integers(-8, 9, (m, k)).astype(np.float64)
        bt = rng.integers(-8, 9, (n, k)).astype(np.float64)
        c = _launch(ll, spec, a, bt, b_device(bt), arch=arch)
        np.testing.assert_array_equal(c, a @ bt.T)
        a = rng.standard_normal((m, k))
        bt = rng.standard_normal((n, k))
        c = _launch(ll, spec, a, bt, b_device(bt), arch=arch)
        error = np.abs(c - a @ bt.T) / (np.abs(a) @ np.abs(bt).T)
        assert np.max(error) < k * np.finfo(np.float64).eps


@pytest.mark.parametrize(
    "tile, pipeline, epilogue",
    [
        ((64, 64, 16, 2, 2), "compv4", "default"),
        ((32, 32, 8, 1, 1), "mem", "default"),
    ],
)
def test_gfx942_f64_batched_active_tile_skip_numeric(tile, pipeline, epilogue):
    arch = _require_f64_gemm_arch()
    spec = _spec(tile, pipeline, epilogue, batched=True, active_tile_skip=True)
    assert is_valid_spec(spec, arch=arch) == (True, "ok")
    tile_m, tile_n, tile_k = tile[:3]
    batches, m_tiles, n_tiles = 3, 2, 3
    m, n, k = m_tiles * tile_m, n_tiles * tile_n, 4 * tile_k
    rng = np.random.default_rng(tile_m)
    a = rng.integers(-8, 9, (batches, m, k)).astype(np.float64)
    bt = rng.integers(-8, 9, (batches, n, k)).astype(np.float64)
    expected = np.einsum("bmk,bnk->bmn", a, bt)
    # A tile is skipped when its first sorted token id is -1.
    skipped = {(0, 1), (2, 0)}
    ids = np.arange(batches * m, dtype=np.int32).reshape(batches, m)
    for z, mi in skipped:
        ids[z, mi * tile_m] = -1
    for engine in _engines():
        c = _launch(_llvm(spec, engine, arch), spec, a, bt, sorted_ids=ids, arch=arch)
        for z in range(batches):
            for mi in range(m_tiles):
                rows = slice(mi * tile_m, (mi + 1) * tile_m)
                if (z, mi) in skipped:
                    assert np.isnan(c[z, rows]).all()
                else:
                    np.testing.assert_array_equal(c[z, rows], expected[z, rows])


def _swizzle_spec(tile, warp_tile, pipeline, dtype):
    tile_m, tile_n, tile_k, warp_m, warp_n = tile
    return UniversalGemmSpec(
        name="swizzle",
        tile=TileSpec(tile_m, tile_n, tile_k, warp_m, warp_n, 1, *warp_tile),
        trait=TraitSpec(pipeline=pipeline, epilogue="default", lds_swizzle=True),
        data=DataSpec(dtype, dtype, dtype, "fp32"),
    )


# The LDS store granule must cover max(a_per_lane, load_vec); when it tracked
# only a_per_lane a load_vec-wide store landed at a different swizzle column
# than the per-element ds_reads. These are the tiles that regressed.
@pytest.mark.parametrize(
    "tile, warp_tile, pipeline, dtype",
    [
        ((64, 64, 32, 2, 2), (16, 16, 16), "compv4", "f16"),
        ((64, 64, 64, 2, 2), (16, 16, 16), "compv4", "f16"),
        ((64, 64, 32, 2, 2), (16, 16, 16), "compv3", "f16"),
        ((64, 64, 32, 2, 2), (32, 32, 8), "compv4", "f16"),
        ((32, 32, 8, 1, 1), (32, 32, 8), "mem", "f16"),
        ((64, 64, 32, 2, 2), (16, 16, 16), "compv4", "bf16"),
        ((64, 64, 64, 2, 2), (16, 16, 16), "compv4", "bf16"),
        ((64, 64, 32, 2, 2), (32, 32, 8), "compv4", "bf16"),
        ((32, 32, 8, 1, 1), (32, 32, 8), "mem", "bf16"),
    ],
)
def test_gfx942_f16_bf16_lds_swizzle_numeric(tile, warp_tile, pipeline, dtype):
    _require_gfx942()
    spec = _swizzle_spec(tile, warp_tile, pipeline, dtype)
    assert is_valid_spec(spec, arch="gfx942") == (True, "ok")
    tile_m, tile_n, tile_k = tile[:3]
    m, n, k = 2 * tile_m, 3 * tile_n, 4 * tile_k
    rng = np.random.default_rng(tile_m * 100 + tile_k)
    a_int = rng.integers(-1, 2, (m, k))
    bt_int = rng.integers(-1, 2, (n, k))
    # A, B entries are -1/0/1, exact in f16/bf16, so the product is exact.
    expected = a_int.astype(np.float64) @ bt_int.astype(np.float64).T
    if dtype == "bf16":
        a = f32_to_bf16(a_int.astype(np.float32))
        bt = f32_to_bf16(bt_int.astype(np.float32))
        c = np.full((m, n), 0x7FC0, dtype=np.uint16)
    else:
        a = a_int.astype(np.float16)
        bt = bt_int.astype(np.float16)
        c = np.full((m, n), np.nan, dtype=np.float16)
    for engine in _engines():
        result = _launch(_llvm(spec, engine), spec, a, bt, c=c)
        got = bf16_to_f32(result).astype(np.float64) if dtype == "bf16" else result
        np.testing.assert_array_equal(got.astype(np.float64), expected)


@pytest.mark.parametrize(
    "spec, reason",
    [
        (
            _spec((64, 64, 16, 2, 2), "mem", "default", warp_tile=(4, 4, 4)),
            "fp64 GEMM supports only warp_tile (16, 16, 4)",
        ),
        (
            _spec((64, 64, 16, 2, 2), "mem", "default", acc="fp32"),
            "only supports fp64 accumulation for fp64",
        ),
        (
            _spec((64, 64, 16, 2, 2), "compv3", "default", True, direct_to_lds=True),
            "fp64 GEMM does not support direct_to_lds with pad_k",
        ),
        (
            _spec((64, 64, 16, 2, 2), "compv4", "default", lds_k_pad=64),
            "LDS budget 81920 > 65536 cap",
        ),
    ],
)
def test_f64_gemm_rejects_unsupported(spec, reason):
    ok, why = is_valid_spec(spec, arch="gfx942")
    assert not ok and reason in why
    if "cpp" in _engines():
        import rocke_engine as native

        sd = universal_gemm_spec_to_dict(spec)
        assert native.gemm_is_valid(sd, arch="gfx942") == (ok, why)


@pytest.mark.parametrize(
    "lds_k_pad, smem_bytes", [(16, 32768), (32, 49152), (48, 65536)]
)
def test_f64_gemm_lds_k_pad_budget(lds_k_pad, smem_bytes):
    # The padded row stride must count towards the double-buffer decision: with
    # lds_k_pad >= 16 a doubled A/B region no longer fits and compv4 falls back
    # to a single buffer.
    spec = _spec((64, 64, 16, 2, 2), "compv4", "default", lds_k_pad=lds_k_pad)
    assert is_valid_spec(spec, arch="gfx942") == (True, "ok")
    lowered = {engine: _llvm(spec, engine) for engine in _engines()}
    for ll in lowered.values():
        assert f"[{smem_bytes} x i8] poison" in ll
    assert len(set(lowered.values())) == 1
