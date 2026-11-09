# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Additive attention bias on gfx950, one case per path, without torch.

``test_attention_additive_bias_gfx950_numeric.py`` is the full numeric lane, but
it needs ROCm torch, which the installed CTest tree does not have, so it skips
there. This file runs one bf16 case on each path without torch: the 2D 16x16 and
transposed 32x32 score paths, the 3D split-KV path, and 3D with the LSE output.
Both run in ``rocke_library_attention_bias_gpu_pytest``, which needs a gfx950
device; on any other device every case skips and the entry reports skipped.

The 2D cases go through ``run_unified_attention_torch`` on ``rocke.runtime``
device buffers (``DeviceArray``). The 3D path allocates its workspace with
torch and the LSE output must be a torch tensor, so the 3D cases launch the
segment and reduce kernels the way that path does, through the same support
check, cache key, segment count and pipeline. Every output and workspace byte
starts as NaN, so an element a kernel skips fails the comparison.

The reference is ``rocke.numeric.sdpa_reference`` (fp64) over the bf16-rounded
inputs: ``softmax(Q @ K^T * scale + bias, bottom-right causal) @ V``.
"""

import math

import numpy as np
import pytest
from rocke.numeric.bf16 import bf16_to_f32, f32_to_bf16
from rocke.numeric.sdpa_reference import sdpa_reference
from rocke.runtime.hip_module import get_device_arch

from kernels.common import attention_unified as au

from ._attention_device_array import DeviceArray

_ARCH = get_device_arch()
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(_ARCH != "gfx950", reason=f"needs gfx950, found {_ARCH}"),
]

_HQ, _HKV, _HEAD_SIZE, _BLOCK_SIZE = 8, 2, 128, 16
_TOL = 4e-2  # bf16, as in the torch lane

# case -> (path, [(query_len, kv_len) per sequence], emit_lse, 2D score path).
_CASES = {
    "2d_16x16": ("2d", [(37, 101), (64, 64), (5, 130)], False, "16x16"),
    "2d_transposed32": ("2d", [(384, 640)], False, "transposed32"),
    "3d": ("3d", [(1, 300), (3, 517), (1, 64)], False, None),
    "3d_lse": ("3d", [(1, 300), (3, 517), (1, 64)], True, None),
}


class _F32DeviceArray(DeviceArray):
    """A float32 ``DeviceArray`` with the attributes the bias check reads."""

    dtype = "torch.float32"

    def element_size(self):
        return 4

    def storage_offset(self):
        return 0

    def untyped_storage(self):
        nbytes = self._host.nbytes

        class _Storage:
            def nbytes(self):
                return nbytes

        return _Storage()


@pytest.fixture(scope="module")
def rt():
    from rocke.runtime.hip_module import Runtime

    return Runtime()


def _bf16(x):
    return bf16_to_f32(f32_to_bf16(np.asarray(x, np.float32)))


def _nan_bytes(rt, nbytes):
    return DeviceArray(rt, np.full(nbytes, 0xFF, np.uint8))


def _inputs(seq_lens):
    rng = np.random.default_rng(0)
    max_blocks = -(-max(kv for _, kv in seq_lens) // _BLOCK_SIZE)
    num_blocks = len(seq_lens) * max_blocks
    q = _bf16(rng.standard_normal((sum(q for q, _ in seq_lens), _HQ, _HEAD_SIZE)))
    kv_shape = (num_blocks, _BLOCK_SIZE, _HKV, _HEAD_SIZE)
    k = _bf16(rng.standard_normal(kv_shape))
    v = _bf16(rng.standard_normal(kv_shape))
    block_table = rng.permutation(num_blocks).astype(np.int32)
    block_table = block_table.reshape(len(seq_lens), max_blocks)
    max_q = max(q_len for q_len, _ in seq_lens)
    max_k = max(kv for _, kv in seq_lens)
    bias = rng.standard_normal((len(seq_lens), _HQ, max_q, max_k)).astype(np.float32)
    return q, k, v, block_table, bias


def _problem(seq_lens, num_blocks, emit_lse):
    return au.UnifiedAttentionProblem(
        total_q=sum(q for q, _ in seq_lens),
        num_seqs=len(seq_lens),
        num_query_heads=_HQ,
        num_kv_heads=_HKV,
        head_size=_HEAD_SIZE,
        block_size=_BLOCK_SIZE,
        max_seqlen_q=max(q for q, _ in seq_lens),
        max_seqlen_k=max(kv for _, kv in seq_lens),
        dtype="bf16",
        use_additive_bias=True,
        num_kv_blocks=num_blocks,
        emit_lse=emit_lse,
    )


def _reference(q, k, v, block_table, seq_lens, scale, bias):
    """``sdpa_reference`` over the paged K/V gathered into THD: fp64 O
    ``[total_q, Hq, D]`` and LSE ``[total_q, Hq]``; ``bias`` may be None."""
    ks, vs = [], []
    for i, (_, kv_len) in enumerate(seq_lens):
        blocks = block_table[i, : -(-kv_len // _BLOCK_SIZE)]
        ks.append(k[blocks].reshape(-1, _HKV, _HEAD_SIZE)[:kv_len])
        vs.append(v[blocks].reshape(-1, _HKV, _HEAD_SIZE)[:kv_len])
    return sdpa_reference(
        q,
        np.concatenate(ks),
        np.concatenate(vs),
        scale=scale,
        bias=bias,
        causal_bottom_right=True,
        q_offsets=np.cumsum([0] + [q_len for q_len, _ in seq_lens]),
        kv_offsets=np.cumsum([0] + [kv_len for _, kv_len in seq_lens]),
    )


def _bias_strides(bias):
    _, hq, max_q, max_k = bias.shape
    return hq * max_q * max_k, max_q * max_k, max_k


def _run_2d(rt, problem, q, k, v, block_table, seq_lens, scale, bias):
    out_host = np.full(f32_to_bf16(q).shape, 0xFFFF, np.uint16)
    out = DeviceArray(rt, out_host)
    b_stride, h_stride, sq_stride = _bias_strides(bias)
    cu_q = np.cumsum([0] + [q for q, _ in seq_lens]).astype(np.int32)
    au.run_unified_attention_torch(
        problem=problem,
        q=DeviceArray(rt, f32_to_bf16(q)),
        k=DeviceArray(rt, f32_to_bf16(k)),
        v=DeviceArray(rt, f32_to_bf16(v)),
        out=out,
        cu_seqlens_q=DeviceArray(rt, cu_q),
        seqused_k=DeviceArray(rt, np.array([kv for _, kv in seq_lens], np.int32)),
        softmax_scale=scale,
        block_table=DeviceArray(rt, block_table),
        softcap=0.0,
        additive_bias=_F32DeviceArray(rt, bias),
        additive_bias_batch_stride=b_stride,
        additive_bias_head_stride=h_stride,
        additive_bias_sq_stride=sq_stride,
        backend="tiled",
    )
    return bf16_to_f32(out.read()), None


def _run_3d(rt, problem, q, k, v, block_table, seq_lens, scale, bias):
    arch = "gfx950"
    ok, why = au._explicit_path_supported(problem, None, "3d", arch)
    assert ok, why
    num_segments = au._num_segments(problem, arch)
    prepared = au._get_3d_pipeline(
        problem, au._tiled_3d_cache_key(problem, arch), num_segments, arch=arch
    )
    segm_output, segm_max, segm_expsum = (
        _nan_bytes(rt, spec.nbytes())
        for spec in au._attention_3d_workspace_specs(problem, num_segments, device=None)
    )
    out = DeviceArray(rt, np.full(f32_to_bf16(q).shape, 0xFFFF, np.uint16))
    seq_lens_dev = DeviceArray(rt, np.array([kv for _, kv in seq_lens], np.int32))
    b_stride, h_stride, sq_stride = _bias_strides(bias)
    cu_q = np.cumsum([0] + [q for q, _ in seq_lens]).astype(np.int32)
    segment_values = {
        "segm_output_ptr": segm_output,
        "segm_max_ptr": segm_max,
        "segm_expsum_ptr": segm_expsum,
        "query_ptr": DeviceArray(rt, f32_to_bf16(q)),
        "key_cache_ptr": DeviceArray(rt, f32_to_bf16(k)),
        "value_cache_ptr": DeviceArray(rt, f32_to_bf16(v)),
        "sink_ptr": None,
        "block_tables_ptr": DeviceArray(rt, block_table),
        "seq_lens_ptr": seq_lens_dev,
        "alibi_slopes_ptr": 0,
        "qq_bias_ptr": 0,
        "additive_bias_ptr": DeviceArray(rt, bias),
        "query_start_len_ptr": DeviceArray(rt, cu_q),
        "scale": scale,
        "k_scale": 1.0,
        "v_scale": 1.0,
        "softcap": 0.0,
        "num_seqs": problem.num_seqs,
        "block_table_stride": block_table.shape[1],
        "qq_bias_stride_0": 0,
        "additive_bias_batch_stride": b_stride,
        "additive_bias_head_stride": h_stride,
        "additive_bias_sq_stride": sq_stride,
        **au._runtime_kernarg_values(problem),
    }
    reduce_values = {
        "output_ptr": out,
        "segm_output_ptr": segm_output,
        "segm_max_ptr": segm_max,
        "segm_expsum_ptr": segm_expsum,
        "seq_lens_ptr": seq_lens_dev,
        **au._runtime_kernarg_values(problem),
    }
    lse = None
    if problem.emit_lse:
        lse = DeviceArray(rt, np.full((problem.total_q, _HQ, 1), np.nan, np.float32))
        reduce_values["lse_ptr"] = lse
    au._launch_3d_pipeline(
        prepared, segment_values, reduce_values, stream=0, capturing=False
    )
    return bf16_to_f32(out.read()), None if lse is None else lse.read()[..., 0]


@pytest.mark.parametrize("case", list(_CASES))
def test_additive_bias_matches_reference_without_torch(rt, case):
    path, seq_lens, emit_lse, score_path = _CASES[case]
    q, k, v, block_table, bias = _inputs(seq_lens)
    problem = _problem(seq_lens, k.shape[0], emit_lse)
    if path == "2d":
        spec = au._tiled_spec_from_problem(problem, "gfx950")
        transposed = spec.use_mfma_32x32 and spec.use_transposed_qk_32x32
        assert ("transposed32" if transposed else "16x16") == score_path
    scale = 1.0 / math.sqrt(_HEAD_SIZE)
    run = _run_2d if path == "2d" else _run_3d
    out, lse = run(rt, problem, q, k, v, block_table, seq_lens, scale, bias)

    ref, ref_lse = _reference(q, k, v, block_table, seq_lens, scale, bias)
    max_abs = np.abs(out - ref).max()
    assert max_abs < _TOL, f"{case}: max_abs={max_abs:.3e}"
    # A kernel that ignored the bias would still be close to attention without it.
    ref_nobias, _ = _reference(q, k, v, block_table, seq_lens, scale, None)
    assert np.abs(ref - ref_nobias).max() > 10 * _TOL, "bias too weak to observe"
    if emit_lse:
        lse_err = np.abs(lse - ref_lse).max()
        assert lse_err < _TOL, f"{case}: LSE max_abs={lse_err:.3e}"
