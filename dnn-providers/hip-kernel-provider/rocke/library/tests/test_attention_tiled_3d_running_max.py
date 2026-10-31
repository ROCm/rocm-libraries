# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Running softmax max of the gfx942 and gfx950 3D split-KV segment kernels.

A segment kernel rescales its running sum and partial output by
``alpha = exp2(m_old - m_new)`` once per KV tile, so ``m_new`` has to be the max
of everything the segment has seen (an attention sink included), not of the
current tile alone. A tile-local max overflowed ``alpha`` (NaN) once a tile
scored about 88 nats below an earlier one, and a fully masked tile, whose max
was replaced by 0, overflowed it (NaN) when the row's max was above about 88
nats or flushed it to 0 (exact zeros) when it was below about -87 nats. Each
GPU test below hit one of these.

One sequence, paged KV (identity block table), bf16, head size 128, 16 query
heads over 2 KV heads, 16-token pages. Decode: Q is all ones and one K row is
all ``hot``, every other K row 0, so that token scores ``hot * sqrt(128)``
nats above the rest. Prefill: Q is all +1 or all -1 and every K row is 9. The
reference is fp64 attention over the bf16 inputs.

GPU lane: ``run_all.py --gpu`` (or ``pytest -m gpu``). The GPU tests launch
the segment and reduce kernels through ``rocke.runtime`` from numpy arrays,
since the CI GPU lane has no torch. Set ``ROCKE_REQUIRE_DECODE_GPU=gfx942``
(or ``gfx950``) to fail instead of skip when that GPU is missing.
``test_native_builder_matches_python`` needs the ``rocke_engine`` binding
(``ROCKE_BUILD_PYBIND=ON``), which CI does not build, so it skips there and
runs only where the binding is built: the C++ builders must emit the same IR
as the Python ones for these specs.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict
from importlib import import_module

import numpy as np
import pytest

from dispatch.attention.common import AttentionTuningSpec
from dispatch.attention.tuning_specs import (
    ExplicitAttention3DConfig,
    make_explicit_attention_3d_specs,
)
from kernels.common import attention_unified as au
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.numeric.bf16 import bf16_to_f32, f32_to_bf16
from rocke.numeric.sdpa_reference import sdpa_reference
from rocke.runtime.hip_module import Runtime, get_device_arch, get_device_num_cus
from rocke.runtime.host_buffers import as_u8_buffer
from rocke.runtime.launcher import DeviceMem

HEAD_SIZE = 128
NUM_QUERY_HEADS = 16
NUM_KV_HEADS = 2
BLOCK_SIZE = 16
# The bf16 tolerance of the 3D numeric tests.
ATOL, RTOL = 0.01, 0.02


@pytest.fixture(scope="module")
def gpu():
    """``(arch, CU count)`` of a gfx942 or gfx950 GPU."""
    required = os.environ.get("ROCKE_REQUIRE_DECODE_GPU")
    arch = get_device_arch()
    if required and arch != required:
        pytest.fail(f"requested {required}, found {arch or 'no GPU'}")
    if arch not in ("gfx942", "gfx950"):
        pytest.skip("needs a gfx942 or gfx950 GPU")
    num_cus = get_device_num_cus()
    if num_cus is None:
        pytest.fail(f"cannot read the CU count of the {arch} GPU")
    return arch, num_cus


def _bf16(x):
    """``x`` rounded to bf16, as float32."""
    return bf16_to_f32(f32_to_bf16(x))


def _problem(num_cus, *, q_len, kv_len, use_sinks=False):
    return au.UnifiedAttentionProblem(
        total_q=q_len,
        num_seqs=1,
        num_query_heads=NUM_QUERY_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        block_size=BLOCK_SIZE,
        max_seqlen_q=q_len,
        max_seqlen_k=kv_len,
        dtype="bf16",
        use_sinks=use_sinks,
        num_cus=num_cus,
        num_kv_blocks=math.ceil(kv_len / BLOCK_SIZE),
    )


def _explicit_spec(problem, arch, num_segments):
    """A 3D tuning spec with ``num_segments`` segments and one page per KV tile."""
    segment, reduce = make_explicit_attention_3d_specs(
        problem,
        ExplicitAttention3DConfig(num_segments=num_segments, tile_policy="1x"),
        arch=arch,
    )
    return AttentionTuningSpec(
        path="3d",
        arch=arch,
        builder_kind="tiled_3d",
        compile_backend="llvm",
        candidate_name="running_max_test",
        tuning_id="running_max_test",
        kernel_spec=segment,
        reduce_spec=reduce,
        num_kv_blocks=problem.num_kv_blocks,
    )


def _tiles_per_segment(segment, kv_len):
    # The segment kernel's split: ceil(kv_len / (num_segments * tile)) tiles.
    return math.ceil(kv_len / (segment.num_segments * segment.tile_size))


def _decode_inputs(kv_len, hot, hot_token=0):
    rng = np.random.default_rng(0)
    q = np.ones((1, NUM_QUERY_HEADS, HEAD_SIZE), np.float32)
    k = np.zeros((NUM_KV_HEADS, kv_len, HEAD_SIZE), np.float32)
    k[:, hot_token] = hot
    v = rng.standard_normal((NUM_KV_HEADS, kv_len, HEAD_SIZE), np.float32)
    return q, _bf16(k), _bf16(v)


def _device(array):
    """A device copy of ``array``."""
    array = np.ascontiguousarray(array)
    mem = DeviceMem(array.nbytes)
    Runtime().memcpy_h2d(mem.ptr(), as_u8_buffer(array), array.nbytes)
    return mem


def _nan_filled(nbytes):
    """``nbytes`` of device memory, every byte 0xFF: NaN as bf16 (0xFFFF) and
    as f32 (0xFFFFFFFF). A blocking copy, so it has landed before the launch
    whichever stream the launcher resolves."""
    return _device(np.full(nbytes, 0xFF, np.uint8))


def _attention(problem, arch, q, k, v, *, tuning_spec=None, sinks=None):
    """The 3D path on one paged sequence: ``q`` [Lq, Hq, D], ``k``/``v``
    [Hkv, Lk, D], float32 arrays of bf16 values; without ``tuning_spec`` it
    uses the dispatcher's split rule.

    Launches the segment and reduce kernels as the 3D path of
    ``run_unified_attention_torch`` does (same support check, cache key,
    segment count and launch grids), on device buffers filled from numpy."""
    q_len, kv_len = q.shape[0], k.shape[1]
    blocks = problem.num_kv_blocks
    pad = blocks * BLOCK_SIZE - kv_len

    def pages(x):
        x = np.pad(x, ((0, 0), (0, pad), (0, 0)))
        x = x.transpose(1, 0, 2).reshape(blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
        return _device(f32_to_bf16(x))

    if tuning_spec is not None:
        # run_unified_attention_torch rebinds an explicit spec to the real cache.
        tuning_spec = tuning_spec.with_num_kv_blocks(blocks)
    ok, why = au._explicit_path_supported(problem, tuning_spec, "3d", arch)
    assert ok, why
    if tuning_spec is None:
        cache_key = au._tiled_3d_cache_key(problem, arch)
        num_segments = au._num_segments(problem, arch)
    else:
        cache_key = tuning_spec.cache_key()
        num_segments = tuning_spec.kernel_spec.num_segments
    prepared = au._get_3d_pipeline(
        problem, cache_key, num_segments, arch=arch, tuning_spec=tuning_spec
    )
    # NaN until the kernels write it: an element of the output, or of the
    # workspace (the reduce reads every segment's slot), that a kernel skips
    # fails the comparison instead of passing on a previous case's recycled
    # buffer.
    segm_output, segm_max, segm_expsum = (
        _nan_filled(spec.nbytes())
        for spec in au._attention_3d_workspace_specs(problem, num_segments, device=None)
    )
    bits = np.empty(q.shape, np.uint16)
    out = _nan_filled(bits.nbytes)
    seq_lens = _device(np.array([kv_len], np.int32))
    segment_values = {
        "segm_output_ptr": segm_output,
        "segm_max_ptr": segm_max,
        "segm_expsum_ptr": segm_expsum,
        "query_ptr": _device(f32_to_bf16(q)),
        "key_cache_ptr": pages(k),
        "value_cache_ptr": pages(v),
        "sink_ptr": None if sinks is None else _device(f32_to_bf16(sinks)),
        "block_tables_ptr": _device(np.arange(blocks, dtype=np.int32)),
        "seq_lens_ptr": seq_lens,
        "alibi_slopes_ptr": 0,
        "qq_bias_ptr": 0,
        "query_start_len_ptr": _device(np.array([0, q_len], np.int32)),
        "scale": 1 / math.sqrt(HEAD_SIZE),
        "k_scale": 1.0,
        "v_scale": 1.0,
        "softcap": 0.0,
        "num_seqs": 1,
        "block_table_stride": blocks,
        "qq_bias_stride_0": 0,
    }
    reduce_values = {
        "output_ptr": out,
        "segm_output_ptr": segm_output,
        "segm_max_ptr": segm_max,
        "segm_expsum_ptr": segm_expsum,
        "seq_lens_ptr": seq_lens,
    }
    # The launch fences the stream, so ``out`` is complete when it returns.
    au._launch_3d_pipeline(
        prepared, segment_values, reduce_values, stream=0, capturing=False
    )
    Runtime().memcpy_d2h(as_u8_buffer(bits), out.ptr(), bits.nbytes)
    return bf16_to_f32(bits)


def _reference(q, k, v, sinks=None):
    """fp64 attention over the bf16 values ``_attention`` uploads, with the
    bottom-right causal mask (query p sees keys up to p + Lk - Lq); a sink
    logit joins each row's softmax with no value."""
    # Rounding is idempotent, so inputs already in bf16 are unchanged.
    q, k, v = _bf16(q), _bf16(k), _bf16(v)
    sinks = None if sinks is None else _bf16(sinks)
    q_len, kv_len = q.shape[0], k.shape[1]
    out, lse = sdpa_reference(
        q,
        k.transpose(1, 0, 2),
        v.transpose(1, 0, 2),
        causal_bottom_right=True,
        q_offsets=[0, q_len],
        kv_offsets=[0, kv_len],
    )
    if sinks is not None:
        out = out / (1.0 + np.exp(np.asarray(sinks, np.float64) - lse))[..., None]
    return out


def _assert_close(out, expected):
    np.testing.assert_allclose(out, expected, rtol=RTOL, atol=ATOL, equal_nan=False)


# K row 0 values. Rounded to bf16 they put token 0 at 0, 22.6, 45.3, 67.9, 79.2,
# 84.9, 87.0, 88.4, 90.5, 101.8 and 135.8 nats above every other token.
@pytest.mark.gpu
@pytest.mark.parametrize(
    "hot", [0.0, 2.0, 4.0, 6.0, 7.0, 7.5, 7.7, 7.8, 8.0, 9.0, 12.0]
)
def test_decode_score_gap(gpu, hot):
    arch, num_cus = gpu
    kv_len = 4096
    problem = _problem(num_cus, q_len=1, kv_len=kv_len)
    spec = _explicit_spec(problem, arch, num_segments=2)
    # Token 0 opens segment 0; the rescale that overflowed needs a later tile
    # in the same segment.
    assert _tiles_per_segment(spec.kernel_spec, kv_len) >= 2
    q, k, v = _decode_inputs(kv_len, hot)
    out = _attention(problem, arch, q, k, v, tuning_spec=spec)
    _assert_close(out, _reference(q, k, v))


# The dispatcher's own split (it follows the CU count). For one sequence its
# segments span 2 or more KV tiles from 2048 tokens on gfx942 and from 8192 on
# gfx950, at any CU count; below 8192 a full gfx950 runs one 64-token tile per
# segment, so those cases only check the default path.
MULTI_TILE_SPLIT_FROM = {"gfx942": 2048, "gfx950": 8192}


@pytest.mark.gpu
@pytest.mark.parametrize("kv_len", [2048, 4096, 8192, 16384, 32768, 65536])
def test_decode_default_split(gpu, kv_len):
    arch, num_cus = gpu
    problem = _problem(num_cus, q_len=1, kv_len=kv_len)
    if kv_len >= MULTI_TILE_SPLIT_FROM[arch]:
        # The segment spec the launch below builds (no tuning spec).
        segment = au._tiled_3d_spec_from_problem(problem, arch)
        assert _tiles_per_segment(segment, kv_len) >= 2, (
            num_cus,
            segment.num_segments,
            segment.tile_size,
        )
    q, k, v = _decode_inputs(kv_len, hot=9.0)
    out = _attention(problem, arch, q, k, v)
    _assert_close(out, _reference(q, k, v))


# 64 query tokens after one context token. A q-block's 16 rows are 2 query
# tokens (8 query heads per KV head) whose causal limits are one key apart; for
# tokens 14, 30, 46 and 62 a 16-key tile boundary falls between them, so the
# q-block's last tile is fully masked for that token. Every K row is 9, so the
# scores are all +101.8 or all -101.8 nats and only such a tile moves the max.
# With 2 segments of 3 tiles, segment 1 starts at key 48, so token 46 (keys
# up to 47) has no key in it at all.
@pytest.mark.gpu
@pytest.mark.parametrize("q_value", [1.0, -1.0], ids=["q_pos", "q_neg"])
@pytest.mark.parametrize("num_segments", [1, 2], ids=["seg1", "seg2"])
def test_prefill_masked_tile(gpu, num_segments, q_value):
    arch, num_cus = gpu
    q_len, kv_len = 64, 65
    problem = _problem(num_cus, q_len=q_len, kv_len=kv_len)
    spec = _explicit_spec(problem, arch, num_segments)
    # 2-token q-blocks, 16-key tiles, and 5 tiles in 1 segment or 3 per
    # segment in 2 (segment 1 then starts at key 48, past token 46's keys).
    assert spec.kernel_spec.block_q == 2
    assert spec.kernel_spec.tile_size == BLOCK_SIZE
    assert _tiles_per_segment(spec.kernel_spec, kv_len) == {1: 5, 2: 3}[num_segments]
    rng = np.random.default_rng(0)
    q = np.full((q_len, NUM_QUERY_HEADS, HEAD_SIZE), q_value, np.float32)
    k = np.full((NUM_KV_HEADS, kv_len, HEAD_SIZE), 9.0, np.float32)
    v = _bf16(rng.standard_normal((NUM_KV_HEADS, kv_len, HEAD_SIZE), np.float32))
    out = _attention(problem, arch, q, k, v, tuning_spec=spec)
    _assert_close(out, _reference(q, k, v))


@pytest.mark.gpu
def test_decode_sink(gpu):
    # Segment 0 starts its max at the sink logit, 100 nats above the scores of
    # its first tile. The last token (in segment 1) scores 99.7 nats, so the
    # output is about 0.43 of its V row.
    arch, num_cus = gpu
    kv_len = 4096
    problem = _problem(num_cus, q_len=1, kv_len=kv_len, use_sinks=True)
    spec = _explicit_spec(problem, arch, num_segments=2)
    q, k, v = _decode_inputs(kv_len, hot=8.8125, hot_token=kv_len - 1)
    sinks = np.full(NUM_QUERY_HEADS, 100.0, np.float32)
    out = _attention(problem, arch, q, k, v, tuning_spec=spec, sinks=sinks)
    _assert_close(out, _reference(q, k, v, sinks))


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize(
    "q_len, kv_len, num_segments, use_sinks",
    [(1, 4096, 2, False), (1, 4096, 2, True), (64, 65, 1, False), (64, 65, 2, False)],
)
def test_native_builder_matches_python(arch, q_len, kv_len, num_segments, use_sinks):
    # The GPU tests run the Python builders; the C++ builders, which serve
    # kernels without Python, must emit the same IR for their specs.
    engine = pytest.importorskip("rocke_engine")
    # The explicit specs do not depend on the CU count.
    problem = _problem(1, q_len=q_len, kv_len=kv_len, use_sinks=use_sinks)
    spec = _explicit_spec(problem, arch, num_segments)
    module = import_module(f"kernels.{arch}.attention_tiled_3d")
    expected = _lower_kernel_to_llvm_python(
        module.build_unified_attention_3d_tiled(spec.kernel_spec, arch=arch), arch=arch
    )
    if arch == "gfx950":
        # The gfx950 binding lowers the reduce kernel after the segment kernel.
        expected += _lower_kernel_to_llvm_python(
            module.build_unified_attention_reduce_tiled(spec.reduce_spec, arch=arch),
            arch=arch,
        )
    lower = getattr(engine, f"{arch}_attention_tiled_3d_lower_llvm")
    assert lower(asdict(spec.kernel_spec), arch) == expected
