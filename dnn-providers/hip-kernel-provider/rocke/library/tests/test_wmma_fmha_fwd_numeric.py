# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""On-device regressions for BF16 normalization and empty causal rows."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rocke.core.arch import generic_arch_from_target_id
from rocke.runtime.hip_module import get_device_arch
from rocke.runtime.packing import pack_args

# Every gfx11 (gfx12) device runs one gfx11-generic (gfx12-generic) code object set.
_ARCH = generic_arch_from_target_id(get_device_arch() or "")
_NEEDS_RDNA = pytest.mark.skipif(
    _ARCH not in ("gfx11-generic", "gfx12-generic"),
    reason="needs an RDNA3/3.5 (gfx11) or RDNA4 (gfx12) GPU",
)
# The transposed-QK body uses the gfx11 WMMA operand layout.
_NEEDS_GFX11 = pytest.mark.skipif(
    _ARCH != "gfx11-generic", reason="needs an RDNA3/3.5 (gfx11) GPU"
)


def _dense_values(
    q,
    k,
    v,
    out,
    *,
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    scale_log2,
    seqlen_q,
    seqlen_k,
    num_query_heads,
    num_kv_heads,
    bottom_right=False,
):
    def stride(array, axis):
        return array.strides[axis] // array.itemsize

    return {
        "Q": q_ptr,
        "K": k_ptr,
        "V": v_ptr,
        "O": out_ptr,
        "LSE": out_ptr,
        "scale_log2": scale_log2,
        "seqlen_q": seqlen_q,
        "seqlen_k": seqlen_k,
        "num_query_heads": num_query_heads,
        "num_kv_heads": num_kv_heads,
        "bottom_right": int(bottom_right),
        "window_left": -1,
        "window_right": 0,
        "write_lse": 0,
        "stride_q_batch": stride(q, 0),
        "stride_q_token": stride(q, -3),
        "stride_q_head": stride(q, -2),
        "stride_k_batch": stride(k, 0),
        "stride_k_token": stride(k, -3),
        "stride_k_head": stride(k, -2),
        "stride_v_batch": stride(v, 0),
        "stride_v_token": stride(v, -3),
        "stride_v_head": stride(v, -2),
        "stride_o_batch": stride(out, 0),
        "stride_o_token": stride(out, -3),
        "stride_o_head": stride(out, -2),
        "stride_lse_batch": 0,
        "stride_lse_token": 0,
        "stride_lse_head": 0,
    }


def _reference_attention(q, k, v, *, causal):
    """FP32 dense attention over [B, S, H, D] inputs; K/V heads repeat for GQA."""
    group = q.shape[2] // k.shape[2]
    q32 = q.astype(np.float32)
    k32 = np.repeat(k.astype(np.float32), group, axis=2)
    v32 = np.repeat(v.astype(np.float32), group, axis=2)
    scores = np.einsum("bqhd,bkhd->bhqk", q32, k32) / math.sqrt(q.shape[-1])
    if causal:
        keep = np.arange(k.shape[1])[None, :] <= np.arange(q.shape[1])[:, None]
        scores = np.where(keep, scores, -np.inf)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    return np.einsum("bhqk,bkhd->bqhd", probs, v32)


@pytest.mark.gpu
@_NEEDS_RDNA
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_size,heads_q,heads_kv", [(64, 4, 4), (128, 8, 2)])
def test_dense_attention_matches_reference(head_size, heads_q, heads_kv, causal):
    """Direct-builder FP16 MHA/GQA attention against an FP32 reference."""
    from benchmarks.gfx1151.attention.benchmark_sdpa import DeviceBuffers
    from benchmarks.gfx1151.attention.cases import CaseInputs
    from kernels.gfx1151.wmma_fmha_fwd import (
        WmmaFmhaFwdSpec,
        build_wmma_fmha_fwd,
        wmma_fmha_fwd_grid,
        wmma_fmha_fwd_signature,
    )
    from rocke.helpers import compile_kernel
    from rocke.runtime.hip_module import Runtime

    batch, seqlen = 2, 64
    rng = np.random.default_rng(0xA11E)

    def tensor(heads):
        shape = (batch, seqlen, heads, head_size)
        return (rng.standard_normal(shape) * 0.3).astype(np.float16)

    inputs = CaseInputs(q=tensor(heads_q), k=tensor(heads_kv), v=tensor(heads_kv))
    spec = WmmaFmhaFwdSpec(
        head_size=head_size, mask_mode="causal" if causal else "none"
    )
    rt = Runtime()
    buffers = DeviceBuffers(rt, inputs)
    module = None
    try:
        artifact = compile_kernel(
            build_wmma_fmha_fwd(spec, arch=_ARCH),
            arch=_ARCH,
            backend="python",
        )
        module = rt.load_module(artifact.hsaco)
        values = _dense_values(
            inputs.q,
            inputs.k,
            inputs.v,
            buffers.arrays["rocke_out"],
            q_ptr=buffers.ptrs["q"],
            k_ptr=buffers.ptrs["k"],
            v_ptr=buffers.ptrs["v"],
            out_ptr=buffers.ptrs["rocke_out"],
            scale_log2=math.log2(math.e) / math.sqrt(head_size),
            seqlen_q=seqlen,
            seqlen_k=seqlen,
            num_query_heads=heads_q,
            num_kv_heads=heads_kv,
        )
        rt.launch(
            module.get_function(artifact.kernel_name),
            wmma_fmha_fwd_grid(
                spec, seqlen_q=seqlen, num_query_heads=heads_q, batch=batch
            ),
            (spec.block_size, 1, 1),
            pack_args(wmma_fmha_fwd_signature(spec), values),
        )
        rt.sync()
        expected = _reference_attention(inputs.q, inputs.k, inputs.v, causal=causal)
        np.testing.assert_allclose(
            buffers.read_output("rocke_out"), expected, rtol=0, atol=2e-2
        )
    finally:
        rt.sync()
        if module is not None:
            module.unload()
        buffers.close()


@pytest.mark.gpu
@_NEEDS_RDNA
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
        wmma_fmha_fwd_signature,
    )
    from rocke.helpers import compile_kernel
    from rocke.runtime.hip_module import Runtime

    inputs = CaseInputs(
        q=np.full((1, sq, 1, 64), 1e15, dtype=bf16),
        k=np.full((1, sk, 1, 64), -1e15, dtype=bf16),
        v=np.ones((1, sk, 1, 64), dtype=bf16),
    )
    spec = WmmaFmhaFwdSpec(
        head_size=64,
        dtype="bf16",
        mask_mode=mask,
        v_lds_stage=v_staging,
    )
    rt = Runtime()
    buffers = DeviceBuffers(rt, inputs)
    module = None
    try:
        artifact = compile_kernel(
            build_wmma_fmha_fwd(spec, arch=_ARCH),
            arch=_ARCH,
            backend="python",
        )
        module = rt.load_module(artifact.hsaco)
        values = _dense_values(
            inputs.q,
            inputs.k,
            inputs.v,
            buffers.arrays["rocke_out"],
            q_ptr=buffers.ptrs["q"],
            k_ptr=buffers.ptrs["k"],
            v_ptr=buffers.ptrs["v"],
            out_ptr=buffers.ptrs["rocke_out"],
            scale_log2=math.log2(math.e) / 8,
            seqlen_q=sq,
            seqlen_k=sk,
            num_query_heads=1,
            num_kv_heads=1,
            bottom_right=bottom_right,
        )
        args = pack_args(wmma_fmha_fwd_signature(spec), values)
        rt.launch(
            module.get_function(artifact.kernel_name),
            wmma_fmha_fwd_grid(spec, seqlen_q=sq, num_query_heads=1, batch=1),
            (spec.block_size, 1, 1),
            args,
        )
        rt.sync()
        actual = buffers.read_output("rocke_out")
        expected = np.ones_like(actual)
        if bottom_right and sq > sk:
            expected[:, : sq - sk] = 0
        np.testing.assert_array_equal(actual, expected)
    finally:
        rt.sync()
        if module is not None:
            module.unload()
        buffers.close()


@pytest.mark.gpu
@_NEEDS_RDNA
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("v_staging", [False, True])
def test_sequence_tails_preserve_guards_and_ignore_poison(dtype, v_staging):
    if dtype == "bf16":
        pytest.importorskip("ml_dtypes")
    from benchmarks.gfx1151.attention.benchmark_sdpa import DeviceBuffers
    from benchmarks.gfx1151.attention.cases import (
        CaseInputs,
        SdpaCase,
        make_inputs,
        reference,
    )
    from kernels.gfx1151.wmma_fmha_fwd import (
        WmmaFmhaFwdSpec,
        build_wmma_fmha_fwd,
        wmma_fmha_fwd_grid,
        wmma_fmha_fwd_signature,
    )
    from rocke.helpers import compile_kernel
    from rocke.runtime.hip_module import Runtime

    case = SdpaCase(
        "tail_guards",
        "boundary",
        dtype,
        1,
        17,
        19,
        4,
        2,
        64,
        mask="causal_bottomright",
        seed=395,
    )
    logical = make_inputs(case)

    def padded(array):
        storage = np.full((1, 32, *array.shape[2:]), np.nan, dtype=array.dtype)
        storage[:, : array.shape[1]] = array
        return storage

    rt = Runtime()
    buffers = DeviceBuffers(
        rt,
        CaseInputs(q=padded(logical.q), k=padded(logical.k), v=padded(logical.v)),
    )
    guard = 16 * case.heads_q * case.head_dim
    count = logical.q.size
    guarded = np.full(count + 2 * guard, 37, dtype=logical.q.dtype)
    guarded[guard : guard + count] = np.nan
    buffers.add("guarded", guarded)
    module = None
    try:
        spec = WmmaFmhaFwdSpec(
            head_size=64,
            dtype=dtype,
            mask_mode="causal",
            query_tail=True,
            kv_tail=True,
            v_lds_stage=v_staging,
        )
        artifact = compile_kernel(
            build_wmma_fmha_fwd(spec, arch=_ARCH),
            arch=_ARCH,
            backend="python",
        )
        module = rt.load_module(artifact.hsaco)
        output = buffers.ptrs["guarded"] + guard * guarded.itemsize
        values = _dense_values(
            buffers.arrays["q"],
            buffers.arrays["k"],
            buffers.arrays["v"],
            logical.q,
            q_ptr=buffers.ptrs["q"],
            k_ptr=buffers.ptrs["k"],
            v_ptr=buffers.ptrs["v"],
            out_ptr=output,
            scale_log2=math.log2(math.e) / 8,
            seqlen_q=17,
            seqlen_k=19,
            num_query_heads=4,
            num_kv_heads=2,
            bottom_right=True,
        )
        args = pack_args(wmma_fmha_fwd_signature(spec), values)
        rt.launch(
            module.get_function(artifact.kernel_name),
            wmma_fmha_fwd_grid(spec, seqlen_q=17, num_query_heads=4, batch=1),
            (spec.block_size, 1, 1),
            args,
        )
        rt.sync()
        actual = buffers.read_output("guarded")
        np.testing.assert_array_equal(
            actual[:guard], np.full(guard, 37, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            actual[-guard:], np.full(guard, 37, dtype=np.float32)
        )
        np.testing.assert_allclose(
            actual[guard : guard + count].reshape(logical.q.shape),
            reference(case, logical),
            rtol=0,
            atol=case.atol,
            equal_nan=False,
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
    kernels = RockeKernels(rt, _ARCH)
    try:
        launch, _ = kernels.prepare(case, buffers)
        launch(0)
        rt.sync()
        return buffers.read_output("rocke_out")
    finally:
        kernels.close()
        buffers.close()


@pytest.mark.gpu
@_NEEDS_RDNA
def test_softcap_saturates_large_bf16_logits():
    bf16 = pytest.importorskip("ml_dtypes").bfloat16
    from benchmarks.gfx1151.attention.cases import CaseInputs, SdpaCase

    case = SdpaCase(
        "softcap_limit", "softcap", "bf16", 1, 16, 32, 1, 1, 64, mask="none", softcap=30
    )
    inputs = CaseInputs(
        q=np.full((1, 16, 1, 64), 1e15, dtype=bf16),
        k=np.full((1, 32, 1, 64), -1e15, dtype=bf16),
        v=np.ones((1, 32, 1, 64), dtype=bf16),
    )
    np.testing.assert_array_equal(
        _run_score_case(case, inputs), np.ones((1, 16, 1, 64), np.float32)
    )


@pytest.mark.gpu
@_NEEDS_RDNA
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_sink_adds_denominator_without_value(dtype):
    from benchmarks.gfx1151.attention.cases import CaseInputs, SdpaCase

    dt = np.float16 if dtype == "fp16" else pytest.importorskip("ml_dtypes").bfloat16
    case = SdpaCase(
        "sink_mass", "sinks", dtype, 1, 1, 1, 1, 1, 64, mask="none", sinks=True
    )
    inputs = CaseInputs(
        q=np.zeros((1, 1, 1, 64), dtype=dt),
        k=np.zeros((1, 1, 1, 64), dtype=dt),
        v=np.ones((1, 1, 1, 64), dtype=dt),
        sinks=np.zeros(1, dtype=dt),
    )
    np.testing.assert_array_equal(
        _run_score_case(case, inputs), np.full((1, 1, 1, 64), 0.5, np.float32)
    )


@pytest.mark.gpu
@_NEEDS_RDNA
def test_alibi_context_offset_preserves_sink_mass():
    from benchmarks.gfx1151.attention.cases import CaseInputs, SdpaCase

    case = SdpaCase(
        "alibi_sink",
        "combo",
        "fp16",
        1,
        2,
        4,
        1,
        1,
        64,
        mask="causal_bottomright",
        sinks=True,
        alibi=True,
    )
    inputs = CaseInputs(
        q=np.zeros((1, 2, 1, 64), np.float16),
        k=np.zeros((1, 4, 1, 64), np.float16),
        v=np.ones((1, 4, 1, 64), np.float16),
        sinks=np.zeros(1, np.float16),
        alibi_slopes=np.array([math.log(2)], np.float32),
    )
    expected = np.broadcast_to(
        np.array([7 / 11, 15 / 19], np.float32)[None, :, None, None], (1, 2, 1, 64)
    )
    np.testing.assert_allclose(
        _run_score_case(case, inputs), expected, atol=1e-3, rtol=0
    )


@pytest.mark.gpu
@_NEEDS_RDNA
def test_qq_bias_bounds_compose_with_window():
    from benchmarks.gfx1151.attention.cases import CaseInputs, SdpaCase

    case = SdpaCase(
        "qq_window",
        "combo",
        "fp16",
        1,
        3,
        5,
        1,
        1,
        64,
        mask="causal_bottomright",
        window=2,
        qq_bias=True,
    )
    values = np.repeat(np.arange(5, dtype=np.float16)[:, None], 64, axis=1).reshape(
        1, 5, 1, 64
    )
    inputs = CaseInputs(
        q=np.zeros((1, 3, 1, 64), np.float16),
        k=np.zeros((1, 5, 1, 64), np.float16),
        v=values,
        qq_bias=np.array([[math.log(3)]], np.float32),
    )
    expected = np.broadcast_to(
        np.array([1.75, 2.5, 3.5], np.float32)[None, :, None, None], (1, 3, 1, 64)
    )
    np.testing.assert_allclose(
        _run_score_case(case, inputs), expected, atol=1e-3, rtol=0
    )


@pytest.mark.gpu
@_NEEDS_RDNA
@pytest.mark.parametrize("layout", ["ragged", "paged"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("v_staging", [False, True])
@pytest.mark.parametrize("kv_dtype", ["", "fp8e4m3"])
def test_packed_boundaries_preserve_guards_and_ignore_poison(
    layout, dtype, v_staging, kv_dtype
):
    if dtype == "bf16" or kv_dtype:
        pytest.importorskip("ml_dtypes")
    from benchmarks.gfx1151.attention.benchmark_sdpa import DeviceBuffers
    from benchmarks.gfx1151.attention.cases import SdpaCase, make_inputs, reference
    from kernels.gfx1151.wmma_fmha_fwd import (
        WmmaFmhaFwdSpec,
        build_wmma_fmha_fwd,
        wmma_fmha_fwd_grid,
        wmma_fmha_fwd_signature,
    )
    from rocke.helpers import compile_kernel
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.packing import pack_args

    case = SdpaCase(
        "packed_boundaries",
        "boundary",
        dtype,
        5,
        0,
        0,
        4,
        2,
        64,
        layout=layout,
        mask="causal_bottomright",
        block_size=16 if layout == "paged" else 0,
        q_lengths=(0, 17, 1, 3, 0),
        k_lengths=(9, 19, 1, 0, 0),
        seed=27,
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
                used[
                    inputs.block_table[sequence, token // case.block_size],
                    token % case.block_size,
                ] = True
        inputs.k[~used] = np.nan
        inputs.v[~used] = np.nan
    else:
        # The first sequence has keys but no queries. No other sequence may
        # use that storage, even through an invalid query tile.
        inputs.k[: case.k_lengths[0]] = np.nan
        inputs.v[: case.k_lengths[0]] = np.nan

    rt = Runtime()
    buffers = DeviceBuffers(rt, inputs)
    module = None
    guard = 64
    count = inputs.q.size
    rt.free(buffers.ptrs.pop("rocke_out"))
    output = np.full(count + 2 * guard, 37, dtype=inputs.q.dtype)
    output[guard : guard + count] = np.nan
    buffers.add("rocke_out", output)
    try:
        spec = WmmaFmhaFwdSpec(
            head_size=64,
            dtype=dtype,
            layout=layout,
            page_block_size=case.block_size,
            v_lds_stage=v_staging,
            mask_mode="causal",
            kv_dtype=kv_dtype,
        )
        kernel = build_wmma_fmha_fwd(spec, arch=_ARCH)
        artifact = compile_kernel(kernel, arch=_ARCH, backend="python")
        module = rt.load_module(artifact.hsaco)
        values = {
            "Q": buffers.ptrs["q"],
            "K": buffers.ptrs["k"],
            "V": buffers.ptrs["v"],
            "O": buffers.ptrs["rocke_out"] + guard * output.itemsize,
            "LSE": buffers.ptrs["rocke_out"],
            "scale_log2": case.scale * math.log2(math.e),
            "seqlen_q": max(case.q_lengths),
            "seqlen_k": max(case.k_lengths),
            "num_query_heads": case.heads_q,
            "num_kv_heads": case.heads_kv,
            "bottom_right": 1,
            "window_left": -1,
            "window_right": 0,
            "write_lse": 0,
            "stride_q_batch": 0,
            "stride_k_batch": 0,
            "stride_v_batch": 0,
            "stride_o_batch": 0,
            "stride_lse_batch": 0,
            "stride_lse_token": 0,
            "stride_lse_head": 0,
            "cu_seqlens_q": buffers.ptrs["cu_seqlens_q"],
        }
        if kv_dtype:
            values.update(k_scale=float(inputs.k_scale), v_scale=float(inputs.v_scale))
        for name, array in (
            ("q", inputs.q),
            ("k", inputs.k),
            ("v", inputs.v),
            ("o", inputs.q),
        ):
            values[f"stride_{name}_token"] = array.strides[-3] // array.itemsize
            values[f"stride_{name}_head"] = array.strides[-2] // array.itemsize
        if layout == "ragged":
            values["cu_seqlens_k"] = buffers.ptrs["cu_seqlens_k"]
        else:
            values.update(
                seqused_k=buffers.ptrs["seqused_k"],
                block_table=buffers.ptrs["block_table"],
                block_table_stride=inputs.block_table.shape[1],
                stride_k_block=inputs.k.strides[0] // inputs.k.itemsize,
                stride_v_block=inputs.v.strides[0] // inputs.v.itemsize,
            )
        signature = wmma_fmha_fwd_signature(spec)
        rt.launch(
            module.get_function(artifact.kernel_name),
            wmma_fmha_fwd_grid(
                spec,
                seqlen_q=max(case.q_lengths),
                num_query_heads=case.heads_q,
                batch=case.batch,
            ),
            (spec.block_size, 1, 1),
            pack_args(signature, values),
        )
        rt.sync()
        actual = buffers.read_output("rocke_out")
        np.testing.assert_array_equal(actual[:guard], np.full(guard, 37, np.float32))
        np.testing.assert_array_equal(actual[-guard:], np.full(guard, 37, np.float32))
        actual = actual[guard : guard + count].reshape(inputs.q.shape)
        np.testing.assert_allclose(
            actual, expected, atol=case.atol, rtol=0, equal_nan=False
        )
        np.testing.assert_array_equal(actual[-3:], np.zeros((3, 4, 64), np.float32))
    finally:
        rt.sync()
        if module is not None:
            module.unload()
        buffers.close()


@pytest.mark.gpu
@_NEEDS_RDNA
def test_compiled_wmma_preserves_declared_workgroup_limit():
    """The driver must see the requested limit, not a discarded LLVM hint."""
    import ctypes
    from kernels.gfx1151.wmma_fmha_fwd import WmmaFmhaFwdSpec, build_wmma_fmha_fwd
    from rocke.helpers import compile_kernel
    from rocke.runtime import hip_module

    spec = WmmaFmhaFwdSpec(head_size=128)
    artifact = compile_kernel(
        build_wmma_fmha_fwd(spec, arch=_ARCH), arch=_ARCH, backend="python"
    )
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


@pytest.mark.gpu
@_NEEDS_GFX11
@pytest.mark.parametrize(
    "dimension,block_n,waves,mask,sq,sk,hq,hkv",
    [
        (64, 32, 1, "none", 16, 32, 8, 1),
        (64, 64, 2, "causal", 64, 128, 4, 2),
        (128, 32, 2, "none", 32, 64, 6, 2),
        (128, 64, 1, "causal", 16, 64, 4, 4),
        (64, 32, 1, "bottom_right", 16, 64, 8, 2),
        (64, 64, 2, "bottom_right", 128, 64, 4, 2),
        (128, 32, 2, "bottom_right", 64, 32, 6, 2),
        (128, 64, 1, "bottom_right", 16, 128, 8, 1),
    ],
)
def test_transposed_qk_preserves_output_coordinates_and_guards(
    dimension, block_n, waves, mask, sq, sk, hq, hkv
):
    from benchmarks.gfx1151.attention.benchmark_sdpa import DeviceBuffers
    from benchmarks.gfx1151.attention.cases import CaseInputs
    from kernels.gfx1151.wmma_fmha_fwd import (
        WmmaFmhaFwdSpec,
        build_wmma_fmha_fwd,
        wmma_fmha_fwd_grid,
        wmma_fmha_fwd_signature,
    )
    from rocke.helpers import compile_kernel
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.packing import pack_args

    batch = 2
    q = np.zeros((batch, sq, hq, dimension), np.float16)
    k = np.zeros((batch, sk, hkv, dimension), np.float16)
    columns = np.arange(dimension).astype(np.float32) / 4
    v = (
        np.arange(batch)[:, None, None, None] * 32
        + np.arange(sk)[None, :, None, None]
        + np.arange(hkv)[None, None, :, None] * 8
        + columns[None, None, None, :]
    ).astype(np.float16)
    context = sk - sq if mask == "bottom_right" else 0
    visible = (
        np.full(sq, sk)
        if mask == "none"
        else np.clip(np.arange(sq) + context + 1, 0, sk)
    )
    mean_key = (visible - 1) / 2
    expected = (
        np.arange(batch)[:, None, None, None] * 32
        + mean_key[None, :, None, None]
        + (np.arange(hq) // (hq // hkv))[None, None, :, None] * 8
        + columns[None, None, None, :]
    ).astype(np.float32)
    expected[:, visible == 0] = 0
    rt = Runtime()
    buffers = DeviceBuffers(rt, CaseInputs(q=q, k=k, v=v))
    guard = 64
    rt.free(buffers.ptrs.pop("rocke_out"))
    output = np.full(q.size + 2 * guard, 37, np.float16)
    output[guard:-guard] = np.nan
    buffers.add("rocke_out", output)
    module = None
    try:
        spec = WmmaFmhaFwdSpec(
            head_size=dimension,
            mask_mode="none" if mask == "none" else "causal",
            transposed_qk=True,
            block_n=block_n,
            num_waves=waves,
        )
        artifact = compile_kernel(
            build_wmma_fmha_fwd(spec, arch=_ARCH), arch=_ARCH, backend="python"
        )
        module = rt.load_module(artifact.hsaco)
        values = _dense_values(
            q,
            k,
            v,
            q,
            q_ptr=buffers.ptrs["q"],
            k_ptr=buffers.ptrs["k"],
            v_ptr=buffers.ptrs["v"],
            out_ptr=buffers.ptrs["rocke_out"] + guard * 2,
            scale_log2=math.log2(math.e) / math.sqrt(dimension),
            seqlen_q=sq,
            seqlen_k=sk,
            num_query_heads=hq,
            num_kv_heads=hkv,
            bottom_right=mask == "bottom_right",
        )
        rt.launch(
            module.get_function(artifact.kernel_name),
            wmma_fmha_fwd_grid(spec, seqlen_q=sq, num_query_heads=hq, batch=batch),
            (spec.block_size, 1, 1),
            pack_args(wmma_fmha_fwd_signature(spec), values),
        )
        rt.sync()
        actual = buffers.read_output("rocke_out")
        np.testing.assert_array_equal(actual[:guard], np.full(guard, 37, np.float32))
        np.testing.assert_array_equal(actual[-guard:], np.full(guard, 37, np.float32))
        actual_output = actual[guard:-guard].reshape(q.shape)
        np.testing.assert_allclose(actual_output, expected, rtol=0, atol=2e-2)
        if np.any(visible == 0):
            np.testing.assert_array_equal(
                actual_output[:, visible == 0], expected[:, visible == 0]
            )
    finally:
        rt.sync()
        if module is not None:
            module.unload()
        buffers.close()


def _attention_reference(
    q, k, v, *, scale, ctx=0, left=0, right=-1, bias=None, num_v_heads=None
):
    """FP64 ``(out, lse)`` with independent K and V head groups.

    ``ctx`` is the bottom-right alignment offset; ``left``/``right`` follow the
    kernel's two-sided window (``left > 0`` / ``right >= 0`` enable a bound)."""
    _, sq, hq, _ = q.shape
    sk, hkv = k.shape[1], k.shape[2]
    hv = num_v_heads or hkv
    q64, k64, v64 = (x.astype(np.float64) for x in (q, k, v))
    k64 = np.repeat(k64, hq // hkv, axis=2)
    v64 = np.repeat(v64, hq // hv, axis=2)
    scores = np.einsum("bqhd,bkhd->bhqk", q64 * scale, k64)
    if bias is not None:
        scores = scores + bias.astype(np.float64)
    qi = np.arange(sq).reshape(sq, 1)
    ki = np.arange(sk).reshape(1, sk)
    keep = np.ones((sq, sk), dtype=bool)
    if right >= 0:
        keep &= ki <= qi + ctx + right
    if left > 0:
        keep &= ki > qi + ctx - left
    scores = np.where(keep.reshape(1, 1, sq, sk), scores, -np.inf)
    row_max = scores.max(axis=-1, keepdims=True)
    safe_max = np.where(np.isfinite(row_max), row_max, 0.0)
    e = np.exp(scores - safe_max)
    denom = e.sum(axis=-1, keepdims=True)
    probs = e / np.where(denom == 0, 1.0, denom)
    out = np.einsum("bhqk,bkhd->bqhd", probs, v64).astype(np.float32)
    with np.errstate(divide="ignore"):
        lse = (np.log(denom) + safe_max)[..., 0].astype(np.float32)
    return out, lse


def _run_wmma_fwd(spec, q, k, v, *, scale, bottom_right=0, left=-1, right=0, **extra):
    """Launch the dense kernel directly; return ``(out, lse)`` in fp32.

    ``extra`` may hold ``bias`` (``[Bb, Hb, Sq, Sk]``) and ``num_v_heads``.
    LSE is always written, as ``[B, Hq, Sq]``."""
    from benchmarks.gfx1151.attention.benchmark_sdpa import _host_bytes
    from kernels.gfx1151.wmma_fmha_fwd import (
        build_wmma_fmha_fwd,
        wmma_fmha_fwd_grid,
        wmma_fmha_fwd_signature,
    )
    from rocke.helpers import compile_kernel
    from rocke.runtime.hip_module import Runtime

    batch, sq, hq, dq = q.shape
    sk, hkv = k.shape[1], k.shape[2]
    bias = extra.get("bias")
    num_v_heads = extra.get("num_v_heads", hkv)
    out = np.full((batch, sq, hq, v.shape[-1]), np.nan, dtype=q.dtype)
    lse = np.full((batch, hq, sq), np.nan, dtype=np.float32)

    def stride(array, axis):
        return array.strides[axis] // array.itemsize

    rt = Runtime()
    pointers = {}
    module = None
    try:
        for name, array in (("q", q), ("k", k), ("v", v), ("o", out), ("lse", lse)):
            pointers[name] = rt.alloc(array.nbytes)
            rt.memcpy_h2d(pointers[name], _host_bytes(array), array.nbytes)
        values = {
            "Q": pointers["q"],
            "K": pointers["k"],
            "V": pointers["v"],
            "O": pointers["o"],
            "LSE": pointers["lse"],
            "scale_log2": math.log2(math.e) * scale,
            "seqlen_q": sq,
            "seqlen_k": sk,
            "num_query_heads": hq,
            "num_kv_heads": hkv,
            "bottom_right": bottom_right,
            "window_left": left,
            "window_right": right,
            "write_lse": 1,
            "stride_lse_batch": hq * sq,
            "stride_lse_token": 1,
            "stride_lse_head": sq,
        }
        for name, array in (("q", q), ("k", k), ("v", v), ("o", out)):
            values[f"stride_{name}_batch"] = stride(array, 0)
            values[f"stride_{name}_token"] = stride(array, 1)
            values[f"stride_{name}_head"] = stride(array, 2)
        if spec.runtime_head_dims:
            values.update(
                head_dim_q=dq, head_dim_v=v.shape[-1], num_v_heads=num_v_heads
            )
        if bias is not None:
            pointers["bias"] = rt.alloc(bias.nbytes)
            rt.memcpy_h2d(pointers["bias"], _host_bytes(bias), bias.nbytes)
            values.update(
                attn_bias_ptr=pointers["bias"],
                bias_stride_b=stride(bias, 0) if bias.shape[0] > 1 else 0,
                bias_stride_h=stride(bias, 1) if bias.shape[1] > 1 else 0,
                bias_stride_q=stride(bias, 2) if bias.shape[2] > 1 else 0,
            )
        artifact = compile_kernel(
            build_wmma_fmha_fwd(spec, arch=_ARCH), arch=_ARCH, backend="python"
        )
        module = rt.load_module(artifact.hsaco)
        rt.launch(
            module.get_function(artifact.kernel_name),
            wmma_fmha_fwd_grid(spec, seqlen_q=sq, num_query_heads=hq, batch=batch),
            (spec.block_size, 1, 1),
            pack_args(wmma_fmha_fwd_signature(spec), values),
        )
        rt.sync()
        rt.memcpy_d2h(_host_bytes(out), pointers["o"], out.nbytes)
        rt.memcpy_d2h(_host_bytes(lse), pointers["lse"], lse.nbytes)
        return out.astype(np.float32), lse
    finally:
        rt.sync()
        if module is not None:
            module.unload()
        for ptr in pointers.values():
            rt.free(ptr)


def _random_fp16(seed, *shapes):
    rng = np.random.default_rng(seed)
    return tuple(rng.standard_normal(shape).astype(np.float16) for shape in shapes)


@pytest.mark.gpu
@_NEEDS_RDNA
@pytest.mark.parametrize("bias_dtype", ["f32", "q"])
def test_attn_bias_and_lse_match_reference(bias_dtype):
    """A broadcast-batch additive bias with causal bottom-right alignment."""
    from kernels.gfx1151.wmma_fmha_fwd import WmmaFmhaFwdSpec

    batch, sq, sk, hq, hkv, dim = 2, 37, 53, 4, 2, 64
    q, k, v = _random_fp16(
        601, (batch, sq, hq, dim), (batch, sk, hkv, dim), (batch, sk, hkv, dim)
    )
    bias = (np.random.default_rng(602).standard_normal((1, hq, sq, sk)) * 2.0).astype(
        np.float32 if bias_dtype == "f32" else np.float16
    )
    spec = WmmaFmhaFwdSpec(
        head_size=dim,
        mask_mode="causal",
        query_tail=True,
        kv_tail=True,
        use_attn_bias=True,
        bias_dtype=bias_dtype,
    )
    scale = 1.0 / math.sqrt(dim)
    actual, actual_lse = _run_wmma_fwd(
        spec, q, k, v, scale=scale, bottom_right=1, bias=bias
    )
    expected, expected_lse = _attention_reference(
        q, k, v, scale=scale, ctx=sk - sq, right=0, bias=bias
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-2)
    np.testing.assert_array_equal(np.isneginf(actual_lse), np.isneginf(expected_lse))
    finite = np.isfinite(expected_lse)
    np.testing.assert_allclose(
        actual_lse[finite], expected_lse[finite], rtol=0, atol=2e-2
    )


@pytest.mark.gpu
@_NEEDS_RDNA
def test_runtime_head_dims_use_independent_v_heads():
    """Hv != Hkv: V heads are ``head / (Hq / num_v_heads)``, not the K group."""
    from kernels.gfx1151.wmma_fmha_fwd import WmmaFmhaFwdSpec

    batch, sq, sk, hq, hkv, hv, dq, dv = 2, 37, 53, 8, 4, 2, 64, 32
    q, k, v = _random_fp16(
        603, (batch, sq, hq, dq), (batch, sk, hkv, dq), (batch, sk, hv, dv)
    )
    spec = WmmaFmhaFwdSpec(
        head_size=128,
        v_head_size=64,
        query_tail=True,
        kv_tail=True,
        runtime_head_dims=True,
    )
    scale = 1.0 / math.sqrt(dq)
    actual, _ = _run_wmma_fwd(spec, q, k, v, scale=scale, num_v_heads=hv)
    expected, _ = _attention_reference(q, k, v, scale=scale, num_v_heads=hv)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-2)


@pytest.mark.gpu
@_NEEDS_RDNA
@pytest.mark.parametrize(
    "bottom_right,sq,sk,left,right",
    [(0, 48, 64, 24, 8), (1, 33, 49, 20, 4)],
)
def test_two_sided_window_matches_reference(bottom_right, sq, sk, left, right):
    from kernels.gfx1151.wmma_fmha_fwd import WmmaFmhaFwdSpec

    batch, hq, hkv, dim = 1, 4, 2, 64
    q, k, v = _random_fp16(
        604 + bottom_right,
        (batch, sq, hq, dim),
        (batch, sk, hkv, dim),
        (batch, sk, hkv, dim),
    )
    spec = WmmaFmhaFwdSpec(
        head_size=dim, mask_mode="window", query_tail=True, kv_tail=True
    )
    scale = 1.0 / math.sqrt(dim)
    actual, actual_lse = _run_wmma_fwd(
        spec, q, k, v, scale=scale, bottom_right=bottom_right, left=left, right=right
    )
    expected, expected_lse = _attention_reference(
        q,
        k,
        v,
        scale=scale,
        ctx=(sk - sq) * bottom_right,
        left=left + 1,  # the kernel keeps keys at ``center - left`` and later
        right=right,
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-2)
    finite = np.isfinite(expected_lse)
    np.testing.assert_allclose(
        actual_lse[finite], expected_lse[finite], rtol=0, atol=2e-2
    )
