# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""On-device regressions for BF16 normalization and empty causal rows."""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from rocke.runtime.hip_module import get_device_arch


@pytest.mark.gpu
@pytest.mark.skipif(get_device_arch() != "gfx1151", reason="needs a gfx1151 GPU")
@pytest.mark.parametrize(
    "mask,bottom_right,sq,sk,v_staging",
    [
        ("none", False, 16, 32, False),
        ("causal", False, 16, 32, False),
        ("causal", True, 32, 16, True),
    ],
)
def test_large_negative_bf16_logits(mask, bottom_right, sq, sk, v_staging):
    """Constant V must survive normalization; invisible query rows are zero.

    These finite dot products are below the former -1e30 initial maximum.
    A finite sentinel incorrectly underflowed every probability to zero.
    """
    bf16 = pytest.importorskip("ml_dtypes").bfloat16
    from benchmarks.gfx1151.attention.benchmark_sdpa import DeviceBuffers
    from benchmarks.gfx1151.attention.cases import CaseInputs
    from kernels.gfx1151.wmma_fmha_fwd import (
        WmmaFmhaFwdSpec,
        build_wmma_fmha_fwd,
        wmma_fmha_fwd_grid,
    )
    from rocke.helpers import compile_kernel
    from rocke.runtime.hip_module import Runtime

    inputs = CaseInputs(
        q=np.full((1, sq, 1, 64), 1e15, dtype=bf16),
        k=np.full((1, sk, 1, 64), -1e15, dtype=bf16),
        v=np.ones((1, sk, 1, 64), dtype=bf16),
    )
    spec = WmmaFmhaFwdSpec(
        head_size=64, num_query_heads=1, dtype="bf16", mask_mode=mask,
        causal_bottom_right=bottom_right, v_lds_stage=v_staging,
    )
    rt = Runtime()
    buffers = DeviceBuffers(rt, inputs)
    module = None
    try:
        artifact = compile_kernel(
            build_wmma_fmha_fwd(spec, arch="gfx1151"),
            arch="gfx1151", backend="python",
        )
        module = rt.load_module(artifact.hsaco)
        args = struct.pack(
            "<QQQQfiiiiiiiiii",
            buffers.ptrs["q"], buffers.ptrs["k"], buffers.ptrs["v"],
            buffers.ptrs["rocke_out"], math.log2(math.e) / 8,
            sq, sk, 64, 64, 64, 64, 64, 64, 64, 64,
        )
        rt.launch(
            module.get_function(artifact.kernel_name),
            wmma_fmha_fwd_grid(spec, seqlen_q=sq, batch=1),
            (spec.block_size, 1, 1), args,
        )
        rt.sync()
        actual = buffers.read_output("rocke_out")
        expected = np.ones_like(actual)
        if bottom_right and sq > sk:
            expected[:, :sq - sk] = 0
        np.testing.assert_array_equal(actual, expected)
    finally:
        rt.sync()
        if module is not None:
            module.unload()
        buffers.close()


@pytest.mark.gpu
@pytest.mark.skipif(get_device_arch() != "gfx1151", reason="needs a gfx1151 GPU")
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("v_staging", [False, True])
def test_sequence_tails_preserve_guards_and_ignore_poison(dtype, v_staging):
    if dtype == "bf16":
        pytest.importorskip("ml_dtypes")
    from benchmarks.gfx1151.attention.benchmark_sdpa import DeviceBuffers
    from benchmarks.gfx1151.attention.cases import CaseInputs, SdpaCase, make_inputs, reference
    from kernels.gfx1151.wmma_fmha_fwd import (
        WmmaFmhaFwdSpec, build_wmma_fmha_fwd, wmma_fmha_fwd_grid,
    )
    from rocke.helpers import compile_kernel
    from rocke.runtime.hip_module import Runtime

    case = SdpaCase(
        "tail_guards", "boundary", dtype, 1, 17, 19, 4, 2, 64,
        mask="causal_bottomright", seed=395,
    )
    logical = make_inputs(case)

    def padded(array):
        storage = np.full((1, 32, *array.shape[2:]), np.nan, dtype=array.dtype)
        storage[:, :array.shape[1]] = array
        return storage

    rt = Runtime()
    buffers = DeviceBuffers(
        rt, CaseInputs(q=padded(logical.q), k=padded(logical.k), v=padded(logical.v)),
    )
    guard = 16 * case.heads_q * case.head_dim
    count = logical.q.size
    guarded = np.full(count + 2 * guard, 37, dtype=logical.q.dtype)
    guarded[guard:guard + count] = np.nan
    buffers.add("guarded", guarded)
    module = None
    try:
        spec = WmmaFmhaFwdSpec(
            head_size=64, num_query_heads=4, num_kv_heads=2, dtype=dtype,
            mask_mode="causal", causal_bottom_right=True,
            query_tail=True, kv_tail=True, v_lds_stage=v_staging,
        )
        artifact = compile_kernel(
            build_wmma_fmha_fwd(spec, arch="gfx1151"), arch="gfx1151", backend="python",
        )
        module = rt.load_module(artifact.hsaco)
        output = buffers.ptrs["guarded"] + guard * guarded.itemsize
        args = struct.pack(
            "<QQQQfiiiiiiiiii",
            buffers.ptrs["q"], buffers.ptrs["k"], buffers.ptrs["v"], output,
            math.log2(math.e) / 8, 17, 19, 256, 64, 128, 64, 128, 64, 256, 64,
        )
        rt.launch(
            module.get_function(artifact.kernel_name),
            wmma_fmha_fwd_grid(spec, seqlen_q=17, batch=1),
            (spec.block_size, 1, 1), args,
        )
        rt.sync()
        actual = buffers.read_output("guarded")
        np.testing.assert_array_equal(actual[:guard], np.full(guard, 37, dtype=np.float32))
        np.testing.assert_array_equal(actual[-guard:], np.full(guard, 37, dtype=np.float32))
        np.testing.assert_allclose(
            actual[guard:guard + count].reshape(logical.q.shape),
            reference(case, logical), rtol=0, atol=case.atol, equal_nan=False,
        )
    finally:
        rt.sync()
        if module is not None:
            module.unload()
        buffers.close()
