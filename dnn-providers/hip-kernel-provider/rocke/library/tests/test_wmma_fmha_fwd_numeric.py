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


def _run_score_case(case, inputs):
    from benchmarks.gfx1151.attention.benchmark_sdpa import DeviceBuffers
    from benchmarks.gfx1151.attention.candidate import RockeKernels
    from rocke.runtime.hip_module import Runtime

    rt = Runtime()
    buffers = DeviceBuffers(rt, inputs)
    kernels = RockeKernels(rt)
    try:
        launch, _ = kernels.prepare(case, buffers)
        launch(0)
        rt.sync()
        return buffers.read_output("rocke_out")
    finally:
        kernels.close()
        buffers.close()


@pytest.mark.gpu
@pytest.mark.skipif(get_device_arch() != "gfx1151", reason="needs a gfx1151 GPU")
def test_softcap_saturates_large_bf16_logits():
    bf16 = pytest.importorskip("ml_dtypes").bfloat16
    from benchmarks.gfx1151.attention.cases import CaseInputs, SdpaCase

    case = SdpaCase("softcap_limit", "softcap", "bf16", 1, 16, 32, 1, 1, 64, mask="none", softcap=30)
    inputs = CaseInputs(
        q=np.full((1, 16, 1, 64), 1e15, dtype=bf16),
        k=np.full((1, 32, 1, 64), -1e15, dtype=bf16),
        v=np.ones((1, 32, 1, 64), dtype=bf16),
    )
    np.testing.assert_array_equal(_run_score_case(case, inputs), np.ones((1, 16, 1, 64), np.float32))


@pytest.mark.gpu
@pytest.mark.skipif(get_device_arch() != "gfx1151", reason="needs a gfx1151 GPU")
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_sink_adds_denominator_without_value(dtype):
    from benchmarks.gfx1151.attention.cases import CaseInputs, SdpaCase

    dt = np.float16 if dtype == "fp16" else pytest.importorskip("ml_dtypes").bfloat16
    case = SdpaCase("sink_mass", "sinks", dtype, 1, 1, 1, 1, 1, 64, mask="none", sinks=True)
    inputs = CaseInputs(
        q=np.zeros((1, 1, 1, 64), dtype=dt),
        k=np.zeros((1, 1, 1, 64), dtype=dt),
        v=np.ones((1, 1, 1, 64), dtype=dt),
        sinks=np.zeros(1, dtype=dt),
    )
    np.testing.assert_array_equal(_run_score_case(case, inputs), np.full((1, 1, 1, 64), 0.5, np.float32))


@pytest.mark.gpu
@pytest.mark.skipif(get_device_arch() != "gfx1151", reason="needs a gfx1151 GPU")
def test_alibi_context_offset_preserves_sink_mass():
    from benchmarks.gfx1151.attention.cases import CaseInputs, SdpaCase

    case = SdpaCase(
        "alibi_sink", "combo", "fp16", 1, 2, 4, 1, 1, 64,
        mask="causal_bottomright", sinks=True, alibi=True,
    )
    inputs = CaseInputs(
        q=np.zeros((1, 2, 1, 64), np.float16),
        k=np.zeros((1, 4, 1, 64), np.float16),
        v=np.ones((1, 4, 1, 64), np.float16),
        sinks=np.zeros(1, np.float16),
        alibi_slopes=np.array([math.log(2)], np.float32),
    )
    expected = np.broadcast_to(np.array([7 / 11, 15 / 19], np.float32)[None, :, None, None], (1, 2, 1, 64))
    np.testing.assert_allclose(_run_score_case(case, inputs), expected, atol=1e-3, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(get_device_arch() != "gfx1151", reason="needs a gfx1151 GPU")
def test_qq_bias_bounds_compose_with_window():
    from benchmarks.gfx1151.attention.cases import CaseInputs, SdpaCase

    case = SdpaCase(
        "qq_window", "combo", "fp16", 1, 3, 5, 1, 1, 64,
        mask="causal_bottomright", window=2, qq_bias=True,
    )
    values = np.repeat(np.arange(5, dtype=np.float16)[:, None], 64, axis=1).reshape(1, 5, 1, 64)
    inputs = CaseInputs(
        q=np.zeros((1, 3, 1, 64), np.float16),
        k=np.zeros((1, 5, 1, 64), np.float16),
        v=values,
        qq_bias=np.array([[math.log(3)]], np.float32),
    )
    expected = np.broadcast_to(np.array([1.75, 2.5, 3.5], np.float32)[None, :, None, None], (1, 3, 1, 64))
    np.testing.assert_allclose(_run_score_case(case, inputs), expected, atol=1e-3, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(get_device_arch() != "gfx1151", reason="needs a gfx1151 GPU")
@pytest.mark.parametrize("layout", ["ragged", "paged"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("v_staging", [False, True])
@pytest.mark.parametrize("kv_dtype", ["", "fp8e4m3"])
def test_packed_boundaries_preserve_guards_and_ignore_poison(layout, dtype, v_staging, kv_dtype):
    if dtype == "bf16" or kv_dtype:
        pytest.importorskip("ml_dtypes")
    from benchmarks.gfx1151.attention.benchmark_sdpa import DeviceBuffers
    from benchmarks.gfx1151.attention.cases import SdpaCase, make_inputs, reference
    from kernels.gfx1151.wmma_fmha_fwd import (
        WmmaFmhaFwdSpec, build_wmma_fmha_fwd, wmma_fmha_fwd_grid,
    )
    from rocke.helpers import compile_kernel
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.packing import pack_args

    case = SdpaCase(
        "packed_boundaries", "boundary", dtype, 5, 0, 0, 4, 2, 64,
        layout=layout, mask="causal_bottomright", block_size=16 if layout == "paged" else 0,
        q_lengths=(0, 17, 1, 3, 0), k_lengths=(9, 19, 1, 0, 0), seed=27,
        kv_dtype=kv_dtype,
    )
    inputs = make_inputs(case)
    if kv_dtype:
        inputs.k_scale = np.asarray(0.3, np.float32)
        inputs.v_scale = np.asarray(0.7, np.float32)
    expected = reference(case, inputs)
    if layout == "paged":
        used = np.zeros(inputs.k.shape[:2], dtype=bool)
        for sequence, length in enumerate(case.k_lengths):
            for token in range(length):
                used[inputs.block_table[sequence, token // case.block_size], token % case.block_size] = True
        inputs.k[~used] = np.nan
        inputs.v[~used] = np.nan
    else:
        # The first sequence has keys but no queries. No other sequence may
        # use that storage, even through an invalid query tile.
        inputs.k[:case.k_lengths[0]] = np.nan
        inputs.v[:case.k_lengths[0]] = np.nan

    rt = Runtime()
    buffers = DeviceBuffers(rt, inputs)
    module = None
    guard = 64
    count = inputs.q.size
    rt.free(buffers.ptrs.pop("rocke_out"))
    output = np.full(count + 2 * guard, 37, dtype=inputs.q.dtype)
    output[guard:guard + count] = np.nan
    buffers.add("rocke_out", output)
    try:
        spec = WmmaFmhaFwdSpec(
            head_size=64, num_query_heads=4, num_kv_heads=2, dtype=dtype,
            layout=layout, page_block_size=case.block_size, v_lds_stage=v_staging,
            mask_mode="causal", causal_bottom_right=True,
            kv_dtype=kv_dtype,
        )
        kernel = build_wmma_fmha_fwd(spec)
        artifact = compile_kernel(kernel, arch="gfx1151", backend="python")
        module = rt.load_module(artifact.hsaco)
        values = {
            "Q": buffers.ptrs["q"], "K": buffers.ptrs["k"], "V": buffers.ptrs["v"],
            "O": buffers.ptrs["rocke_out"] + guard * output.itemsize,
            "scale_log2": case.scale * math.log2(math.e),
            "seqlen_q": max(case.q_lengths), "seqlen_k": max(case.k_lengths),
            "cu_seqlens_q": buffers.ptrs["cu_seqlens_q"],
        }
        if kv_dtype:
            values.update(k_scale=float(inputs.k_scale), v_scale=float(inputs.v_scale))
        for name, array in (("q", inputs.q), ("k", inputs.k), ("v", inputs.v), ("o", inputs.q)):
            values[f"stride_{name}_token"] = array.strides[-3] // array.itemsize
            values[f"stride_{name}_head"] = array.strides[-2] // array.itemsize
        if layout == "ragged":
            values["cu_seqlens_k"] = buffers.ptrs["cu_seqlens_k"]
        else:
            values.update(
                seqused_k=buffers.ptrs["seqused_k"], block_table=buffers.ptrs["block_table"],
                block_table_stride=inputs.block_table.shape[1],
                stride_k_block=inputs.k.strides[0] // inputs.k.itemsize,
                stride_v_block=inputs.v.strides[0] // inputs.v.itemsize,
            )
        signature = [{"name": param.name, "type": param.type.name} for param in kernel.params]
        rt.launch(
            module.get_function(artifact.kernel_name),
            wmma_fmha_fwd_grid(spec, seqlen_q=max(case.q_lengths), batch=case.batch),
            (spec.block_size, 1, 1), pack_args(signature, values),
        )
        rt.sync()
        actual = buffers.read_output("rocke_out")
        np.testing.assert_array_equal(actual[:guard], np.full(guard, 37, np.float32))
        np.testing.assert_array_equal(actual[-guard:], np.full(guard, 37, np.float32))
        actual = actual[guard:guard + count].reshape(inputs.q.shape)
        np.testing.assert_allclose(actual, expected, atol=case.atol, rtol=0, equal_nan=False)
        np.testing.assert_array_equal(actual[-3:], np.zeros((3, 4, 64), np.float32))
    finally:
        rt.sync()
        if module is not None:
            module.unload()
        buffers.close()


@pytest.mark.gpu
@pytest.mark.skipif(get_device_arch() != "gfx1151", reason="needs a gfx1151 GPU")
def test_compiled_wmma_preserves_declared_workgroup_limit():
    """The driver must see the requested limit, not a discarded LLVM hint."""
    import ctypes
    from kernels.gfx1151.wmma_fmha_fwd import WmmaFmhaFwdSpec, build_wmma_fmha_fwd
    from rocke.helpers import compile_kernel
    from rocke.runtime import hip_module

    spec = WmmaFmhaFwdSpec(head_size=128, num_query_heads=4)
    artifact = compile_kernel(build_wmma_fmha_fwd(spec), arch="gfx1151", backend="python")
    rt = hip_module.Runtime()
    module = rt.load_module(artifact.hsaco)
    try:
        function = module.get_function(artifact.kernel_name)
        query = hip_module._resolve_hip().hipFuncGetAttribute
        query.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p]
        query.restype = ctypes.c_int
        maximum = ctypes.c_int()
        # HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK is ordinal zero.
        status = query(ctypes.byref(maximum), 0, ctypes.c_void_p(function.p))
        if status:
            raise RuntimeError(f"hipFuncGetAttribute failed: {status}")
        assert maximum.value == spec.block_size
    finally:
        module.unload()
