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

GPU lane: ``run_all.py --gpu`` (or ``pytest -m gpu``). Set
``ROCKE_REQUIRE_DECODE_GPU=gfx942`` (or ``gfx950``) to fail instead of skip
when that GPU is missing. ``test_native_builder_matches_python`` runs in the
CPU lane: the C++ builders must emit the same IR as the Python ones for these
specs.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict
from importlib import import_module

import pytest

from dispatch.attention.common import AttentionTuningSpec
from dispatch.attention.tuning_specs import (
    ExplicitAttention3DConfig,
    make_explicit_attention_3d_specs,
)
from kernels.common import attention_unified as au
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.runtime.hip_module import get_device_arch

HEAD_SIZE = 128
NUM_QUERY_HEADS = 16
NUM_KV_HEADS = 2
BLOCK_SIZE = 16
# The bf16 tolerance of the 3D numeric tests.
ATOL, RTOL = 0.01, 0.02


@pytest.fixture(scope="module")
def gpu():
    """``(torch, arch, CU count)`` of a gfx942 or gfx950 GPU."""
    required = os.environ.get("ROCKE_REQUIRE_DECODE_GPU")
    if required:
        import torch
    else:
        torch = pytest.importorskip("torch")

    if not torch.cuda.is_available():
        if required:
            pytest.fail("required decode GPU is unavailable")
        pytest.skip("requires a ROCm GPU")
    arch = get_device_arch()
    if required and arch != required:
        pytest.fail(f"requested {required}, found {arch}")
    if arch not in ("gfx942", "gfx950"):
        # The required GPU is present (checked above); these kernels do not
        # target it.
        pytest.skip("the tiled 3D kernels target gfx942/gfx950")
    return torch, arch, torch.cuda.get_device_properties(0).multi_processor_count


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


def _decode_inputs(torch, kv_len, hot, hot_token=0):
    rng = torch.Generator().manual_seed(0)
    q = torch.ones(1, NUM_QUERY_HEADS, HEAD_SIZE, dtype=torch.bfloat16)
    k = torch.zeros(NUM_KV_HEADS, kv_len, HEAD_SIZE, dtype=torch.bfloat16)
    k[:, hot_token] = hot
    v = torch.randn(NUM_KV_HEADS, kv_len, HEAD_SIZE, generator=rng)
    return q, k, v.to(torch.bfloat16)


def _attention(torch, problem, q, k, v, *, tuning_spec=None, sinks=None):
    """The 3D path on one paged sequence: ``q`` [Lq, Hq, D], ``k``/``v``
    [Hkv, Lk, D]; without ``tuning_spec`` it uses the dispatcher's split rule."""
    q_len, kv_len = q.shape[0], k.shape[1]
    blocks = problem.num_kv_blocks
    pad = blocks * BLOCK_SIZE - kv_len

    def pages(x):
        x = torch.nn.functional.pad(x, (0, 0, 0, pad))
        x = x.permute(1, 0, 2).reshape(blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
        return x.contiguous().cuda()

    # NaN until the kernels write it, so an element they skip fails the
    # comparison instead of passing on a previous case's recycled output.
    out = torch.full_like(q, float("nan"), device="cuda")
    au.run_unified_attention_torch(
        problem=problem,
        q=q.cuda(),
        k=pages(k),
        v=pages(v),
        out=out,
        cu_seqlens_q=torch.tensor([0, q_len], dtype=torch.int32).cuda(),
        seqused_k=torch.tensor([kv_len], dtype=torch.int32).cuda(),
        softmax_scale=1 / math.sqrt(HEAD_SIZE),
        block_table=torch.arange(blocks, dtype=torch.int32).view(1, blocks).cuda(),
        softcap=0.0,
        sinks=None if sinks is None else sinks.cuda(),
        tuning_spec=tuning_spec,
        backend="3d",
    )
    torch.cuda.synchronize()
    return out.cpu().float()


def _reference(torch, q, k, v, sinks=None):
    """fp64 attention with the bottom-right causal mask (query p sees keys up
    to p + Lk - Lq); a sink logit joins each row's softmax with no value."""
    q_len, kv_len = q.shape[0], k.shape[1]
    hidden = torch.arange(kv_len)[None, :] > (
        torch.arange(q_len)[:, None] + kv_len - q_len
    )
    ratio = NUM_QUERY_HEADS // NUM_KV_HEADS
    out = torch.empty(q_len, NUM_QUERY_HEADS, HEAD_SIZE, dtype=torch.float64)
    for h in range(NUM_QUERY_HEADS):
        keys, values = k[h // ratio].double(), v[h // ratio].double()
        scores = q[:, h].double() @ keys.T / math.sqrt(HEAD_SIZE)
        scores = scores.masked_fill(hidden, float("-inf"))
        if sinks is not None:
            sink = torch.full((q_len, 1), float(sinks[h]), dtype=torch.float64)
            scores = torch.cat([scores, sink], dim=1)
        out[:, h] = scores.softmax(dim=1)[:, :kv_len] @ values
    return out.float()


# K row 0 values. Rounded to bf16 they put token 0 at 0, 22.6, 45.3, 67.9, 79.2,
# 84.9, 87.0, 88.4, 90.5, 101.8 and 135.8 nats above every other token.
@pytest.mark.gpu
@pytest.mark.parametrize(
    "hot", [0.0, 2.0, 4.0, 6.0, 7.0, 7.5, 7.7, 7.8, 8.0, 9.0, 12.0]
)
def test_decode_score_gap(gpu, hot):
    torch, arch, num_cus = gpu
    kv_len = 4096
    problem = _problem(num_cus, q_len=1, kv_len=kv_len)
    spec = _explicit_spec(problem, arch, num_segments=2)
    # Token 0 opens segment 0; the rescale that overflowed needs a later tile
    # in the same segment.
    assert _tiles_per_segment(spec.kernel_spec, kv_len) >= 2
    q, k, v = _decode_inputs(torch, kv_len, hot)
    out = _attention(torch, problem, q, k, v, tuning_spec=spec)
    torch.testing.assert_close(out, _reference(torch, q, k, v), atol=ATOL, rtol=RTOL)


# The dispatcher's own split (it follows the CU count). For one sequence its
# segments span 2 or more KV tiles from 2048 tokens on gfx942 and from 8192 on
# gfx950, at any CU count; below 8192 a full gfx950 runs one 64-token tile per
# segment, so those cases only check the default path.
MULTI_TILE_SPLIT_FROM = {"gfx942": 2048, "gfx950": 8192}


@pytest.mark.gpu
@pytest.mark.parametrize("kv_len", [2048, 4096, 8192, 16384, 32768, 65536])
def test_decode_default_split(gpu, kv_len):
    torch, arch, num_cus = gpu
    problem = _problem(num_cus, q_len=1, kv_len=kv_len)
    if kv_len >= MULTI_TILE_SPLIT_FROM[arch]:
        # The segment spec the launch below builds (no tuning spec).
        segment = au._tiled_3d_spec_from_problem(problem, arch)
        assert _tiles_per_segment(segment, kv_len) >= 2, (
            num_cus,
            segment.num_segments,
            segment.tile_size,
        )
    q, k, v = _decode_inputs(torch, kv_len, hot=9.0)
    out = _attention(torch, problem, q, k, v)
    torch.testing.assert_close(out, _reference(torch, q, k, v), atol=ATOL, rtol=RTOL)


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
    torch, arch, num_cus = gpu
    q_len, kv_len = 64, 65
    problem = _problem(num_cus, q_len=q_len, kv_len=kv_len)
    spec = _explicit_spec(problem, arch, num_segments)
    # 2-token q-blocks, 16-key tiles, and 5 tiles in 1 segment or 3 per
    # segment in 2 (segment 1 then starts at key 48, past token 46's keys).
    assert spec.kernel_spec.block_q == 2
    assert spec.kernel_spec.tile_size == BLOCK_SIZE
    assert _tiles_per_segment(spec.kernel_spec, kv_len) == {1: 5, 2: 3}[num_segments]
    rng = torch.Generator().manual_seed(0)
    q = torch.full((q_len, NUM_QUERY_HEADS, HEAD_SIZE), q_value, dtype=torch.bfloat16)
    k = torch.full((NUM_KV_HEADS, kv_len, HEAD_SIZE), 9.0, dtype=torch.bfloat16)
    v = torch.randn(NUM_KV_HEADS, kv_len, HEAD_SIZE, generator=rng)
    v = v.to(torch.bfloat16)
    out = _attention(torch, problem, q, k, v, tuning_spec=spec)
    torch.testing.assert_close(out, _reference(torch, q, k, v), atol=ATOL, rtol=RTOL)


@pytest.mark.gpu
def test_decode_sink(gpu):
    # Segment 0 starts its max at the sink logit, 100 nats above the scores of
    # its first tile. The last token (in segment 1) scores 99.7 nats, so the
    # output is about 0.43 of its V row.
    torch, arch, num_cus = gpu
    kv_len = 4096
    problem = _problem(num_cus, q_len=1, kv_len=kv_len, use_sinks=True)
    spec = _explicit_spec(problem, arch, num_segments=2)
    q, k, v = _decode_inputs(torch, kv_len, hot=8.8125, hot_token=kv_len - 1)
    sinks = torch.full((NUM_QUERY_HEADS,), 100.0, dtype=torch.bfloat16)
    out = _attention(torch, problem, q, k, v, tuning_spec=spec, sinks=sinks)
    expected = _reference(torch, q, k, v, sinks)
    torch.testing.assert_close(out, expected, atol=ATOL, rtol=RTOL)


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
