# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Actual tiled split-KV launches on contiguous and padded non-paged caches.

Set ROCKE_REQUIRE_DECODE_GPU=gfx942/gfx950 to require hardware (no skips).
Reference attention is the FP64 numpy decode over the fp16/bf16-rounded inputs.

The decode launches run on ``rocke.runtime`` device arrays and HIP streams.
"""

import math
import os
from dataclasses import replace

import numpy as np
import pytest
from rocke.numeric import storage
from rocke.numeric.compare import Tolerance, assert_within_tolerance
from rocke.runtime.device_array import DeviceArray
from rocke.runtime.hip_module import Runtime, get_device_arch

from ._attention_unified_support import (
    decode_reference,
    paged_cache,
    random_half,
)


@pytest.fixture(scope="module")
def gpu():
    required = os.environ.get("ROCKE_REQUIRE_DECODE_GPU")
    arch = get_device_arch()
    if arch is None:
        if required:
            pytest.fail("required decode GPU is unavailable")
        pytest.skip("requires a ROCm GPU")
    if required and arch != required:
        pytest.fail(f"requested {required}, found {arch}")
    if arch not in ("gfx942", "gfx950", "gfx1250"):
        if required:
            pytest.fail(f"unsupported required decode target {arch}")
        pytest.skip("decode validation targets gfx942/gfx950/gfx1250")
    return arch


def _reference(q, k, v, lengths, window):
    return decode_reference(
        q, k, v, lengths, 1 / math.sqrt(q.shape[2]), sliding_window=window
    )


def _strided_storage(logical, layout):
    """Host storage of ``logical`` ``[B, H, S, D]`` in ``layout`` and the element
    strides that view it as ``[B, H, S, D]``: ``bhsd`` as is, ``bshd`` token-major,
    ``padded`` with unit-D rows 8 unused (NaN) elements apart plus batch/head
    padding."""
    batch, heads, capacity, dim = logical.shape
    if layout == "bhsd":
        return logical, None
    if layout == "bshd":
        seq = capacity
        return logical.transpose(0, 2, 1, 3), (seq * heads * dim, dim, heads * dim, 1)
    row_stride = dim + 8
    head_stride = (capacity + 1) * row_stride
    batch_stride = (heads + 1) * head_stride
    span = (
        (batch - 1) * batch_stride
        + (heads - 1) * head_stride
        + (capacity - 1) * row_stride
        + dim
    )
    backing = np.full(span, np.nan, np.float32)
    strides = (batch_stride, head_stride, row_stride, 1)
    np.lib.stride_tricks.as_strided(
        backing, logical.shape, tuple(4 * x for x in strides)
    )[...] = logical
    return backing, strides


class _RockeArrays:
    """Device arrays and HIP streams from ``rocke.runtime``."""

    def half(self, values, dtype):
        return DeviceArray(storage.encode(values, dtype), dtype)

    def ints(self, values):
        return DeviceArray(np.asarray(values, np.int32))

    def set_ints(self, array, values):
        array.write(np.asarray(values, np.int32))

    def strided_kv(self, logical, layout, dtype):
        host, strides = _strided_storage(logical, layout)
        array = self.half(host, dtype)
        return array if strides is None else array.view(logical.shape, strides)

    def paged_kv(self, logical, block_size, dtype):
        pages, table = paged_cache(logical, block_size, dtype)
        return pages, self.ints(table)

    def guarded_output(self, shape, dtype, guard, fill):
        """An output view with ``guard`` elements of ``fill`` on each side."""
        count = math.prod(shape)
        backing = self.half(np.full(count + 2 * guard, fill, np.float32), dtype)
        return backing.view(shape, offset=guard), backing

    def read(self, array, dtype):
        return storage.decode(array.values(), dtype)

    def stream(self):
        return Runtime().create_stream()

    def sync(self, stream):
        Runtime().wait_stream(stream)

    def release(self, stream):
        if stream:
            Runtime().destroy_stream(stream)


def _decode_inputs(rng, dtype, batch, q_heads, kv_heads, capacity, dim):
    q = random_half(rng, (batch, q_heads, dim), dtype)
    k = random_half(rng, (batch, kv_heads, capacity, dim), dtype)
    v = random_half(rng, (batch, kv_heads, capacity, dim), dtype)
    return q, k, v


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "dim,block,tile",
    [(256, 16, None), (256, 32, None), (128, 64, None), (256, 64, 16), (256, 64, None)],
)
def test_strided_lds_admission_numeric(gpu, dtype, dim, block, tile):
    from dispatch.attention.common import attention_problem
    from kernels.common import attention_unified as au

    from dispatch.attention import AttentionRequest, dispatch_attention

    arch = gpu
    if arch == "gfx950" and tile is not None and tile < block:
        pytest.skip("gfx950 tile_size_override is a multiple of block_size")
    request = AttentionRequest(
        batch=3,
        nhead_q=8,
        nhead_k=2,
        seqlen_q=1,
        seqlen_k=65,
        hdim_q=dim,
        hdim_v=dim,
        arch=arch,
        dtype=dtype,
        kv_layout="strided",
        kv_block_size=block,
        target_ctas=8,
    )
    problem = attention_problem(request)
    q, k, v = _decode_inputs(np.random.default_rng(314), dtype, 3, 8, 2, 65, dim)
    # Written to the output before each launch: a row the kernel leaves
    # unwritten fails the comparison instead of matching a zero reference.
    unwritten = np.full(q.shape, np.nan, np.float32)
    lengths = [0, 64, 65]
    reference = _reference(q, k, v, lengths, 0)
    arrays = _RockeArrays()
    tensors = dict(
        q=arrays.half(q, dtype),
        k=arrays.strided_kv(k, "bshd", dtype),
        v=arrays.strided_kv(v, "bhsd", dtype),
        out=arrays.half(unwritten, dtype),
        cu_seqlens_q=arrays.ints(range(4)),
        seqused_k=arrays.ints(lengths),
    )
    spec = None
    if tile is not None:
        supported = dispatch_attention(replace(request, kv_block_size=32)).spec
        spec = replace(
            supported,
            kernel_spec=replace(
                supported.kernel_spec, block_size=block, tile_size_override=tile
            ),
        )
    kwargs = dict(
        problem=problem,
        **tensors,
        block_table=None,
        softmax_scale=1 / math.sqrt(dim),
        softcap=0,
        kv_layout="strided",
    )
    if tile is None:
        selected = dispatch_attention(request)
        binding = selected.candidate.bind_torch(
            request, selected.spec, dict(problem=problem, **tensors)
        )
        binding.launch()
    else:
        au.run_unified_attention_torch(**kwargs, tuning_spec=spec)
    atol, rtol = (0.003, 0.005) if dtype == "fp16" else (0.01, 0.02)
    out = arrays.read(tensors["out"], dtype)
    assert_within_tolerance(out, reference, Tolerance(atol=atol, rtol=rtol))
    if tile is None:
        tensors["out"].write(storage.encode(unwritten, dtype))
        au.run_unified_attention_torch(**kwargs)
        out = arrays.read(tensors["out"], dtype)
        assert_within_tolerance(out, reference, Tolerance(atol=atol, rtol=rtol))


# Distinct batches, empty/partial segments, both physical orders, independent
# K/V strides, and final tiles without allocation padding. Repeated passes change
# the lengths in place to exercise binding cache reuse.
_CASES = [
    (1, 64, 2, 4, "bhsd", "bhsd", 0),
    (17, 64, 2, 4, "bshd", "bshd", 0),
    (65, 128, 2, 8, "bhsd", "bshd", 0),
    (97, 128, 2, 8, "bshd", "bhsd", 17),
    (65, 256, 1, 8, "bhsd", "bhsd", 0),
    (129, 256, 2, 8, "padded", "bshd", 33),
    (1031, 64, 2, 4, "bshd", "padded", 0),
]


def _strided_kv_decode(arch, monkeypatch, dtype, case, kv_layout, block_size=16):
    from kernels.common import attention_unified as au

    if arch == "gfx1250" and (
        kv_layout == "strided"
        or dtype != "bf16"
        or case[1] != 64
        or case[3] // case[2] != 8
    ):
        pytest.skip("gfx1250 coverage is paged BF16 D64 GQA-8 decode")
    for target in ("GFX942", "GFX950", "GFX1250"):
        monkeypatch.setenv(f"HIPDNN_{target}_3D_GRAPH", "0")
    capacity, dim, kv_heads, q_heads, k_layout, v_layout, window = case
    batch = 3
    arrays = _RockeArrays()
    rng = np.random.default_rng(701 + capacity + dim)
    q_host, k_host, v_host = _decode_inputs(
        rng, dtype, batch, q_heads, kv_heads, capacity, dim
    )
    # Large per-batch shifts make shared-cache bugs observable.
    for b in range(batch):
        v_host[b] = storage.round_to(v_host[b] + b, dtype)
    q = arrays.half(q_host, dtype)
    if kv_layout == "strided":
        k = arrays.strided_kv(k_host, k_layout, dtype)
        v = arrays.strided_kv(v_host, v_layout, dtype)
        block_table = None
    else:
        k, block_table = arrays.paged_kv(k_host, block_size, dtype)
        v, _ = arrays.paged_kv(v_host, block_size, dtype)
    cu_q = arrays.ints(range(batch + 1))
    lengths_host = [capacity, max(0, capacity - 3), 0]
    lengths = arrays.ints(lengths_host)
    output, guarded = arrays.guarded_output(q_host.shape, dtype, 32, 117)
    problem = au.UnifiedAttentionProblem(
        total_q=batch,
        num_seqs=batch,
        num_query_heads=q_heads,
        num_kv_heads=kv_heads,
        head_size=dim,
        block_size=block_size,
        max_seqlen_q=1,
        max_seqlen_k=capacity,
        dtype=dtype,
        sliding_window=window,
        target_ctas=8,
    )
    stream = arrays.stream()
    try:
        # Paged cases exercise the direct runtime; strided cases also test dispatch.
        if kv_layout == "strided":
            from dispatch.attention.common import AttentionMaskType

            from dispatch.attention import AttentionRequest, dispatch_attention

            # A decode window is anchored at the end of the sequence (bottom-right
            # causal); NO_MASK + sliding_window is the non-causal top-left band.
            request = AttentionRequest(
                mask_type=(
                    AttentionMaskType.BOTTOM_RIGHT_CAUSAL
                    if window
                    else AttentionMaskType.NO_MASK
                ),
                batch=batch,
                nhead_q=q_heads,
                nhead_k=kv_heads,
                seqlen_q=1,
                seqlen_k=capacity,
                hdim_q=dim,
                hdim_v=dim,
                arch=arch,
                dtype=dtype,
                kv_layout="strided",
                kv_block_size=block_size,
                sliding_window=window,
                target_ctas=8,
            )
            result = dispatch_attention(request)
            assert result.candidate.name == "attention_strided_decode"
            binding = result.candidate.bind_torch(
                request,
                result.spec,
                dict(
                    problem=problem,
                    q=q,
                    k=k,
                    v=v,
                    out=output,
                    cu_seqlens_q=cu_q,
                    seqused_k=lengths,
                ),
                stream=stream,
            )
        for step in range(3):
            if step:
                lengths_host = (
                    [max(0, capacity - 1), capacity, min(1, capacity)]
                    if step == 1
                    else [capacity, min(1, capacity), capacity]
                )
                arrays.set_ints(lengths, lengths_host)
            if step and kv_layout == "strided":
                binding.launch()
            else:
                au.run_unified_attention_torch(
                    problem=problem,
                    q=q,
                    k=k,
                    v=v,
                    out=output,
                    cu_seqlens_q=cu_q,
                    seqused_k=lengths,
                    softmax_scale=1 / math.sqrt(dim),
                    block_table=block_table,
                    softcap=0.0,
                    backend="3d",
                    kv_layout=kv_layout,
                    stream=stream,
                )
            arrays.sync(stream)
            expected = _reference(q_host, k_host, v_host, lengths_host, window)
            atol, rtol = (0.01, 0.02) if dtype == "bf16" else (0.003, 0.005)
            assert_within_tolerance(
                arrays.read(output, dtype), expected, Tolerance(atol=atol, rtol=rtol)
            )
            guards = arrays.read(guarded, dtype)
            np.testing.assert_array_equal(guards[:32], 117)
            np.testing.assert_array_equal(guards[-32:], 117)
        if kv_layout == "strided":
            binding.launch.close()
    finally:
        arrays.release(stream)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("kv_layout", ["strided", "paged"])
def test_strided_kv_decode(gpu, monkeypatch, dtype, case, kv_layout):
    _strided_kv_decode(gpu, monkeypatch, dtype, case, kv_layout)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("block_size", [32, 64])
@pytest.mark.parametrize("window", [0, 17])
def test_strided_decode_tile_policy(gpu, monkeypatch, dtype, block_size, window):
    case = (*_CASES[2][:-1], window)
    _strided_kv_decode(gpu, monkeypatch, dtype, case, "strided", block_size=block_size)
