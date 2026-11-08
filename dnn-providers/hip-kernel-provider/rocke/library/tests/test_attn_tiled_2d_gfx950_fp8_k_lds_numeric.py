# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU numeric check of the gfx950 tiled-2D QK paths with fp8 K kept in LDS.

With an fp8 KV cache and ``use_fp8_mfma_qk`` the kernel stores K in LDS as raw
fp8 bytes and dequantizes it (``* k_scale``) in registers before the bf16 QK
MFMA. Production dispatch does not pick this knob for every QK path, but tuning
and explicit knob pins can, so each path is launched here through an explicit
tuning spec and compared with an fp64 reference over the dequantized cache.

A ``k_scale`` that is not a power of two makes a missing or wrong scale visible.
The non-transposed 32x32 path once read the fp8 buffer as bf16; its output was
then far off the reference.

The kernels run on ``rocke.runtime`` device buffers filled from numpy, so the
test runs in GPU lanes without torch. ``ml_dtypes`` (an optional test
dependency) is the independent fp8 e4m3fn codec. Skipped unless a gfx950 GPU and
``ml_dtypes`` are present.
"""

from __future__ import annotations

import numpy as np
import pytest
from dispatch.attention.common import AttentionTuningSpec
from kernels import UnifiedAttentionProblem, run_unified_attention_torch
from kernels.gfx950.attention_tiled_2d import UnifiedAttention2DTiledSpec
from rocke.numeric.bf16 import bf16_to_f32, f32_to_bf16
from rocke.numeric.sdpa_reference import sdpa_reference
from rocke.runtime.hip_module import get_device_arch

from ._attention_device_array import DeviceArray
from ._attn_tiled_2d_gfx950_qk_paths import FP8_K_IN_LDS, QK_PATHS

# Optional test dependency (platform/tests/README.md): skip, not fail, without it.
ml_dtypes = pytest.importorskip("ml_dtypes")

_ARCH = get_device_arch()
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(_ARCH != "gfx950", reason=f"needs a gfx950 GPU, found {_ARCH}"),
]

_HEAD_SIZE = 64
_BLOCK_SIZE = 32
_HQ, _HKV = 8, 8
# (query_len, kv_len) per sequence: a partial last KV tile and a short query.
_SEQ_LENS = [(96, 160), (40, 77)]
_K_SCALE = 0.75
_V_SCALE = 1.25
# bf16 output after fp8 K/V; the reference runs over the same dequantized values.
_TOL = 4e-2


@pytest.fixture(scope="module")
def rt():
    from rocke.runtime.hip_module import Runtime

    return Runtime()


def _query_starts():
    return np.cumsum([0] + [q_len for q_len, _ in _SEQ_LENS], dtype=np.int32)


def _kv_starts():
    return np.cumsum([0] + [kv_len for _, kv_len in _SEQ_LENS], dtype=np.int32)


def _fp8_cache(rng, shape):
    """A random fp8 e4m3fn cache: ``(raw bytes, the values they encode)``."""
    values = rng.standard_normal(shape).astype(np.float32)
    fp8 = values.astype(ml_dtypes.float8_e4m3fn)
    return fp8.view(np.uint8), fp8.astype(np.float32)


def _inputs(seed=0):
    """Q (bf16 values as float32), fp8 K and V caches as ``(bytes, values)``,
    and a shuffled block table."""
    rng = np.random.default_rng(seed)
    max_blocks = -(-max(kv_len for _, kv_len in _SEQ_LENS) // _BLOCK_SIZE)
    num_blocks = len(_SEQ_LENS) * max_blocks
    total_q = int(_query_starts()[-1])
    q = bf16_to_f32(f32_to_bf16(rng.standard_normal((total_q, _HQ, _HEAD_SIZE))))
    shape = (num_blocks, _BLOCK_SIZE, _HKV, _HEAD_SIZE)
    k_cache = _fp8_cache(rng, shape)
    v_cache = _fp8_cache(rng, shape)
    block_table = rng.permutation(num_blocks).astype(np.int32)
    return q, k_cache, v_cache, block_table.reshape(len(_SEQ_LENS), max_blocks)


def _reference(q, k_values, v_values, block_table, scale):
    """fp64 paged attention with a bottom-right causal mask over the
    dequantized cache (``k * k_scale``, ``v * v_scale``)."""

    def sequence_tokens(cache, seq):
        kv_len = _SEQ_LENS[seq][1]
        pages = block_table[seq, : -(-kv_len // _BLOCK_SIZE)]
        return cache[pages].reshape(-1, _HKV, _HEAD_SIZE)[:kv_len]

    def packed(cache):
        return np.concatenate(
            [sequence_tokens(cache, seq) for seq in range(len(_SEQ_LENS))]
        )

    out, _lse = sdpa_reference(
        q,
        packed(k_values) * _K_SCALE,
        packed(v_values) * _V_SCALE,
        scale=scale,
        causal_bottom_right=True,
        q_offsets=_query_starts(),
        kv_offsets=_kv_starts(),
    )
    return out


def _tuning_spec(path, num_kv_blocks):
    kernel_spec = UnifiedAttention2DTiledSpec(
        head_size=_HEAD_SIZE,
        block_size=_BLOCK_SIZE,
        num_query_heads=_HQ,
        num_kv_heads=_HKV,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        tile_size=64,
        **FP8_K_IN_LDS,
        **QK_PATHS[path],
    )
    return AttentionTuningSpec(
        path="2d",
        arch="gfx950",
        builder_kind="tiled",
        compile_backend="llvm",
        candidate_name="fp8_k_lds_numeric",
        tuning_id="fp8_k_lds_numeric",
        kernel_spec=kernel_spec,
        num_kv_blocks=num_kv_blocks,
    )


@pytest.mark.parametrize("path", sorted(QK_PATHS))
def test_fp8_k_in_lds_matches_reference(rt, path):
    q, (k_bytes, k_values), (v_bytes, v_values), block_table = _inputs()
    num_kv_blocks = int(k_bytes.shape[0])
    problem = UnifiedAttentionProblem(
        total_q=q.shape[0],
        num_seqs=len(_SEQ_LENS),
        num_query_heads=_HQ,
        num_kv_heads=_HKV,
        head_size=_HEAD_SIZE,
        block_size=_BLOCK_SIZE,
        max_seqlen_q=max(q_len for q_len, _ in _SEQ_LENS),
        max_seqlen_k=max(kv_len for _, kv_len in _SEQ_LENS),
        dtype="bf16",
        use_fp8=True,
        num_kv_blocks=num_kv_blocks,
    )
    host_q = f32_to_bf16(q)
    # 0xFFFF is NaN in bf16: an element the kernel does not store fails the
    # check instead of passing on an earlier case's output in reused memory.
    out = DeviceArray(rt, np.full_like(host_q, 0xFFFF))
    kv_lens = np.diff(_kv_starts()).astype(np.int32)
    scale = _HEAD_SIZE**-0.5
    run_unified_attention_torch(
        problem=problem,
        q=DeviceArray(rt, host_q),
        k=DeviceArray(rt, k_bytes),
        v=DeviceArray(rt, v_bytes),
        out=out,
        cu_seqlens_q=DeviceArray(rt, _query_starts()),
        seqused_k=DeviceArray(rt, kv_lens),
        softmax_scale=scale,
        block_table=DeviceArray(rt, block_table),
        softcap=0.0,
        k_scale=_K_SCALE,
        v_scale=_V_SCALE,
        tuning_spec=_tuning_spec(path, num_kv_blocks),
    )
    result = bf16_to_f32(out.read())
    assert np.isfinite(result).all(), f"{path}: output not fully written"
    expected = _reference(q, k_values, v_values, block_table, scale)
    max_abs = np.abs(expected - result).max()
    assert max_abs < _TOL, f"{path}: max_abs={max_abs:.3e}"
