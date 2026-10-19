# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import ctypes
import importlib.util
import os
import struct

import numpy as np
import pytest
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
from rocke.runtime.hip_module import get_device_arch


def _require_gfx942():
    arch = get_device_arch()
    if not arch or arch.split(":")[0] != "gfx942":
        if os.getenv("ROCKE_REQUIRE_GFX942") == "1":
            pytest.fail(f"Required gfx942 device absent: {arch}")
        pytest.skip("requires gfx942")


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


def _llvm(spec, engine):
    if engine == "python":
        kernel = build_universal_gemm(spec, arch="gfx942")
        return _lower_kernel_to_llvm_python(kernel, arch="gfx942")
    import rocke_engine as native

    return native.gemm_lower_llvm(universal_gemm_spec_to_dict(spec), arch="gfx942")


def _launch(ll, spec, a, bt, b_device=None, sorted_ids=None):
    from rocke.runtime.comgr import build_hsaco_from_llvm_ir
    from rocke.runtime.hip_module import Runtime

    *batch, m, k = a.shape
    n = bt.shape[-2]
    batches = batch[0] if batch else 1
    blob, _ = build_hsaco_from_llvm_ir(ll, isa="amdgcn-amd-amdhsa--gfx942")
    runtime = Runtime()
    module = runtime.load_module(blob)
    arrays = [
        a.copy(),
        (bt if b_device is None else b_device).copy(),
        np.full((*batch, m, n), np.nan),
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
    _require_gfx942()
    spec = _spec(tile, pipeline, epilogue, pad, **trait)
    assert is_valid_spec(spec, arch="gfx942") == (True, "ok")
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
        ll = _llvm(spec, engine)
        a = rng.integers(-8, 9, (m, k)).astype(np.float64)
        bt = rng.integers(-8, 9, (n, k)).astype(np.float64)
        np.testing.assert_array_equal(_launch(ll, spec, a, bt, b_device(bt)), a @ bt.T)
        a = rng.standard_normal((m, k))
        bt = rng.standard_normal((n, k))
        c = _launch(ll, spec, a, bt, b_device(bt))
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
    _require_gfx942()
    spec = _spec(tile, pipeline, epilogue, batched=True, active_tile_skip=True)
    assert is_valid_spec(spec, arch="gfx942") == (True, "ok")
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
        c = _launch(_llvm(spec, engine), spec, a, bt, sorted_ids=ids)
        for z in range(batches):
            for mi in range(m_tiles):
                rows = slice(mi * tile_m, (mi + 1) * tile_m)
                if (z, mi) in skipped:
                    assert np.isnan(c[z, rows]).all()
                else:
                    np.testing.assert_array_equal(c[z, rows], expected[z, rows])


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
    ],
)
def test_f64_gemm_rejects_unsupported(spec, reason):
    ok, why = is_valid_spec(spec, arch="gfx942")
    assert not ok and reason in why
    if "cpp" in _engines():
        import rocke_engine as native

        sd = universal_gemm_spec_to_dict(spec)
        assert native.gemm_is_valid(sd, arch="gfx942") == (ok, why)
