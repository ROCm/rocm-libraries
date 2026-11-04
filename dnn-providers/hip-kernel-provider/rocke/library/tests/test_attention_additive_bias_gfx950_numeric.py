# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU numeric lane for additive attention bias on gfx950.

Drives ``run_unified_attention_torch`` with an fp32 additive bias on both the 2D
tiled (``backend="tiled"``) and 3D split-KV (``backend="3d"``) paths and checks
max_abs against ``rocke.numeric.sdpa_reference`` (fp64) of
``softmax(softcap(Q @ K^T * scale) + bias, causal/window) @ V`` -- the bias is
added after the softcap, as for ALiBi/QQ bias. The 2D lane covers both score
paths: short prefill (16x16) and long prefill (max_seqlen_q > 256, multi- and
single-batch), which selects the transposed 32x32 path, plus a multi-batch
group past max_seqlen_q=512 that compiles through hipcc. Softcap and
sliding-window groups run on 2D and 3D, and a d64 / block 32 GQA-8 bf16 group
covers the transposed 32x32 combo stack. The score path and compile backend of
every 2D group, and the bias knobs of the combo and fp8 cohorts, are pinned
without a GPU in ``test_attention_additive_bias_routes.py``. A dispatcher group
launches an ``AttentionRequest(use_attn_bias=True)`` through the gfx950 tuning
candidates' binding with ``tensors["attn_bias"]``.

Bias addressing (element strides; Sk stride is always 1):
``bias[seq_idx*batch_stride + q_head*head_stride + q_local*sq_stride + k_abs]``.
A stride of 0 broadcasts that dimension. ``q_head`` is the *query* head, so
the GQA cases check that a per-query-head bias is not collapsed onto KV heads.

The numeric tests are marked ``gpu`` and gated with a device skipif, so they
are a graceful skip on a CPU CI box. The host-side argument-validation tests at
the bottom need torch for real CPU tensors (``_validate_additive_bias`` checks
a torch-specific dtype string), so they are gated with ``requires_torch`` and
skip, rather than erroring the whole module, on an image with no torch (e.g.
TheRock's gfx94X CI). The installed CTest tree has no torch either, so this
file skips there; ``test_attention_additive_bias_gfx950_device_numeric.py``
runs one case per path in it without torch, on a gfx950 device. Run
standalone:

    HIP_VISIBLE_DEVICES=0 python -m pytest tests/test_attention_additive_bias_gfx950_numeric.py
"""

from __future__ import annotations

import numpy as np
import pytest
from rocke.numeric.sdpa_reference import sdpa_reference

from kernels import UnifiedAttentionProblem, run_unified_attention_torch

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False

requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="ROCm torch required")


def _gpu_ready():
    """True only on a gfx950 box with ROCm torch (gate on ``gcnArchName``)."""
    if not _HAS_TORCH or not torch.cuda.is_available():
        return False
    arch = torch.cuda.get_device_properties(0).gcnArchName.lower()
    return "gfx950" in arch


requires_gfx950_gpu = pytest.mark.skipif(
    not _gpu_ready(), reason="needs a gfx950 GPU with ROCm torch"
)

_TORCH_DT = {"fp16": "float16", "bf16": "bfloat16"}
_TOL = {"fp16": 2e-2, "bf16": 4e-2}

_HEAD_SIZE = 128
_BLOCK_SIZE = 16

# case -> (backend, [(query_len, kv_len) per sequence]). ``tiled`` is short
# mixed prefill with a context prefix (16x16 score path). ``tiled_long``
# (multi-batch) and ``tiled_long_1seq`` (single-batch) have max_seqlen_q > 256,
# which routes the 2D kernel onto the transposed 32x32 score path; the test
# asserts that routing. ``3d`` covers decode-style split-KV with a tail block.
_CASES = {
    "tiled": ("tiled", [(37, 101), (64, 64), (5, 130)]),
    "tiled_long": ("tiled", [(300, 420), (512, 512)]),
    "tiled_long_1seq": ("tiled", [(384, 640)]),
    "3d": ("3d", [(1, 300), (3, 517), (1, 64)]),
}

# Extra paths: case -> (backend, seq_lens, features). ``tiled_long_hipcc`` is
# multi-batch prefill past max_seqlen_q=512, which compiles through hipcc
# instead of LLVM-direct. The softcap and sliding-window groups pin the score
# order and the masked bias load under a window, on 2D and 3D; the single-seq
# window case lands on the transposed 32x32 path (fp16 and bf16). The softcap is small
# enough that ``softcap(QK*scale) + bias`` and ``softcap(QK*scale + bias)``
# differ. Each group runs on two bias layouts (a padded full layout and a
# broadcast one) to bound the lane's runtime; _CASES covers all seven layouts.
_EXTRA_CASES = {
    "tiled_long_hipcc": ("tiled", [(640, 640), (768, 900)], {}),
    "tiled_softcap": ("tiled", [(300, 420), (512, 512)], {"softcap": 2.0}),
    "tiled_window": ("tiled", [(300, 420), (512, 512)], {"sliding_window": 64}),
    "tiled_window_1seq": ("tiled", [(384, 640)], {"sliding_window": 64}),
    "3d_softcap": ("3d", [(1, 300), (3, 517), (1, 64)], {"softcap": 2.0}),
    "3d_window": ("3d", [(1, 300), (3, 517), (1, 64)], {"sliding_window": 64}),
}
_EXTRA_BIAS_MODES = ("full_padded_sq", "bcast_hsq")

# The d64 / block 32 / GQA-8 bf16 prefill cohort takes the transposed 32x32 combo
# stack (scalar softmax state, skipped legacy Q gather; 64/8 heads also use the
# fast paged-KV descriptor). With bias its mask-once / mask-limit opts stay off.
_COMBO_SEQ_LENS = [(512, 512), (300, 420)]
_COMBO_HEADS = {"gqa8_h32": (32, 4), "gqa8_h64": (64, 8)}

# bias layouts: name -> (broadcast batch, broadcast head, broadcast Sq,
# extra Sk padding). ``bcast_hsq`` is the [B, 1, 1, Sk] key-padding shape.
_BIAS_MODES = {
    "full": (False, False, False, 0),
    "full_padded_sq": (False, False, False, 7),
    "bcast_b": (True, False, False, 0),
    "bcast_h": (False, True, False, 0),
    "bcast_bh": (True, True, False, 0),
    "bcast_sq": (False, False, True, 0),
    "bcast_hsq": (False, True, True, 0),
}

_HEADS = {"mha": (8, 8), "gqa": (8, 2)}


def _make_inputs(
    seq_lens, hq, hkv, dtype, seed=0, head_size=_HEAD_SIZE, block_size=_BLOCK_SIZE
):
    torch.manual_seed(seed)
    tdt = getattr(torch, _TORCH_DT[dtype])
    query_lens = [s[0] for s in seq_lens]
    kv_lens = [s[1] for s in seq_lens]
    max_blocks = (max(kv_lens) + block_size - 1) // block_size
    num_blocks = len(seq_lens) * max_blocks
    q = torch.randn(sum(query_lens), hq, head_size, dtype=tdt, device="cuda")
    k = torch.randn(num_blocks, block_size, hkv, head_size, dtype=tdt, device="cuda")
    v = torch.randn_like(k)
    # Disjoint shuffled pages so every sequence reads distinct KV.
    block_table = (
        torch.randperm(num_blocks, device="cuda")
        .to(torch.int32)
        .view(len(seq_lens), max_blocks)
        .contiguous()
    )
    cu_q = torch.tensor([0] + query_lens, dtype=torch.int32, device="cuda").cumsum(
        0, dtype=torch.int32
    )
    seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    return q, k, v, block_table, cu_q, seqused_k


def _make_bias(mode, num_seqs, hq, max_q, max_k):
    """Return (storage, batch_stride, head_stride, sq_stride, full_view).

    ``full_view`` is the logical ``[num_seqs, hq, max_q, max_k]`` bias the
    reference adds; ``storage`` is the compact tensor the kernel reads.
    """
    bcast_b, bcast_h, bcast_sq, pad = _BIAS_MODES[mode]
    rows = 1 if bcast_sq else max_q
    row_pitch = max_k + pad
    shape = [1 if bcast_b else num_seqs, 1 if bcast_h else hq, rows, row_pitch]
    storage = torch.randn(*shape, dtype=torch.float32, device="cuda")
    sq_stride = 0 if bcast_sq else row_pitch
    head_stride = 0 if bcast_h else rows * row_pitch
    batch_stride = 0 if bcast_b else (1 if bcast_h else hq) * rows * row_pitch
    full_view = storage[..., :max_k].expand(num_seqs, hq, max_q, max_k)
    return storage, batch_stride, head_stride, sq_stride, full_view


def _host(x):
    return x.detach().double().cpu().numpy()


def _unexpanded(bias):
    """``bias`` with each broadcast (stride 0) axis cut to size 1;
    ``sdpa_reference`` broadcasts size-1 axes itself."""
    for dim in range(bias.dim()):
        if bias.stride(dim) == 0:
            bias = bias.narrow(dim, 0, 1)
    return bias


def _reference(
    q,
    k,
    v,
    block_table,
    seq_lens,
    scale,
    bias_full,
    softcap=0.0,
    sliding_window=0,
    bias_inside_softcap=False,
    top_left=False,
    right_bound=0,
    sinks=None,
    return_lse=False,
):
    """``sdpa_reference`` (fp64) over the paged K/V gathered into THD, in the
    kernels' order: ``softcap(QK*scale) + bias``, then the mask, then the
    softmax (with ``sinks`` [hq] as extra logits with no value).

    The mask is the band around the diagonal ``d = q + context`` (``context =
    kv_len - q_len`` bottom-right, 0 with ``top_left``): key ``k`` is kept iff
    ``k <= d + right_bound`` (``-1``: no right bound) and, with a window,
    ``d - k < sliding_window``. A row with no kept key, from the mask or from
    -inf bias entries, outputs 0 with LSE -inf. ``return_lse`` also returns the
    natural-log normalizer ``[total_q, hq, 1]`` (without the sink).

    ``bias_inside_softcap`` computes the other order, ``softcap(QK*scale +
    bias)``, only to show a softcap case can tell the two apart.
    """
    hq, hkv = q.shape[1], k.shape[2]
    block_size, head_size = k.shape[1], k.shape[3]
    ks, vs = [], []
    for i, (_, kv_len) in enumerate(seq_lens):
        blocks = block_table[i, : -(-kv_len // block_size)].long()
        ks.append(k[blocks].reshape(-1, hkv, head_size)[:kv_len])
        vs.append(v[blocks].reshape(-1, hkv, head_size)[:kv_len])
    qs, ks, vs = _host(q), _host(torch.cat(ks)), _host(torch.cat(vs))
    q_offsets = np.cumsum([0] + [q_len for q_len, _ in seq_lens])
    kv_offsets = np.cumsum([0] + [kv_len for _, kv_len in seq_lens])
    bias = None if bias_full is None else _host(_unexpanded(bias_full))
    if softcap > 0:
        # sdpa_reference adds its bias to Q @ K^T * scale and has no softcap,
        # so hand it the capped logits as the bias, with Q = 0.
        max_q = max(q_len for q_len, _ in seq_lens)
        max_k = max(kv_len for _, kv_len in seq_lens)
        logits = np.zeros((len(seq_lens), hq, max_q, max_k))
        for i, (q_len, kv_len) in enumerate(seq_lens):
            rows = slice(q_offsets[i], q_offsets[i] + q_len)
            keys = slice(kv_offsets[i], kv_offsets[i] + kv_len)
            kh = np.repeat(ks[keys], hq // hkv, axis=1)
            logits[i, :, :q_len, :kv_len] = (
                np.einsum("qhd,khd->hqk", qs[rows], kh) * scale
            )
        if bias is not None and bias_inside_softcap:
            logits = logits + bias
        logits = softcap * np.tanh(logits / softcap)
        if bias is not None and not bias_inside_softcap:
            logits = logits + bias
        qs, bias = np.zeros_like(qs), logits
    out, lse = sdpa_reference(
        qs,
        ks,
        vs,
        scale=scale,
        bias=bias,
        # The kernels keep sliding_window keys; sdpa_reference keeps left + 1.
        left_bound=sliding_window - 1 if sliding_window > 0 else None,
        right_bound=right_bound,
        top_left=top_left,
        q_offsets=q_offsets,
        kv_offsets=kv_offsets,
    )
    if sinks is not None:
        # The sink adds exp(sink) to each row's denominator; an empty row stays 0.
        out = out / (1.0 + np.exp(_host(sinks)[None, :] - lse))[..., None]
    out = torch.from_numpy(out).float().to(q.device)
    if return_lse:
        return out, torch.from_numpy(lse).float().to(q.device).unsqueeze(-1)
    return out


def _check_case(
    case,
    backend,
    seq_lens,
    features,
    dtype,
    bias_mode,
    heads,
    head_size=_HEAD_SIZE,
    block_size=_BLOCK_SIZE,
):
    hq, hkv = {**_HEADS, **_COMBO_HEADS}[heads]
    softcap = float(features.get("softcap", 0.0))
    sliding_window = int(features.get("sliding_window", 0))
    q, k, v, block_table, cu_q, seqused_k = _make_inputs(
        seq_lens, hq, hkv, dtype, head_size=head_size, block_size=block_size
    )
    max_q = max(s[0] for s in seq_lens)
    max_k = max(s[1] for s in seq_lens)
    scale = head_size**-0.5
    storage, b_stride, h_stride, sq_stride, bias_full = _make_bias(
        bias_mode, len(seq_lens), hq, max_q, max_k
    )

    problem = UnifiedAttentionProblem(
        total_q=q.shape[0],
        num_seqs=len(seq_lens),
        num_query_heads=hq,
        num_kv_heads=hkv,
        head_size=head_size,
        block_size=block_size,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        dtype=dtype,
        sliding_window=sliding_window,
        softcap=softcap,
        use_additive_bias=True,
        num_kv_blocks=int(k.shape[0]),
    )
    out = torch.empty_like(q)
    run_unified_attention_torch(
        problem=problem,
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_q,
        seqused_k=seqused_k,
        softmax_scale=scale,
        block_table=block_table,
        softcap=softcap,
        additive_bias=storage,
        additive_bias_batch_stride=b_stride,
        additive_bias_head_stride=h_stride,
        additive_bias_sq_stride=sq_stride,
        backend=backend,
    )
    torch.cuda.synchronize()

    ref_args = (q, k, v, block_table, seq_lens, scale)
    feats = dict(softcap=softcap, sliding_window=sliding_window)
    ref = _reference(*ref_args, bias_full, **feats)
    ref_nobias = _reference(*ref_args, None, **feats)
    tol = _TOL[dtype]
    max_abs = (ref - out.float()).abs().max().item()
    assert (
        max_abs < tol
    ), f"{case}/{dtype}/{bias_mode}/{heads}: max_abs={max_abs:.3e} >= {tol}"
    # Guard against a silently ignored bias: the bias must move the output.
    shift = (ref - ref_nobias).abs().max().item()
    assert shift > 10 * tol, f"bias too weak to be observable (shift={shift:.3e})"
    if softcap > 0:
        # Guard the order: a kernel that put the bias inside the softcap would be
        # off by about ``gap``, so the gap must clear the tolerance with margin
        # for this case to catch the other order.
        wrong = _reference(*ref_args, bias_full, bias_inside_softcap=True, **feats)
        gap = (ref - wrong).abs().max().item()
        assert gap > 3 * tol, f"softcap too weak to pin the order (gap={gap:.3e})"


@requires_gfx950_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("heads", sorted(_HEADS))
@pytest.mark.parametrize("bias_mode", list(_BIAS_MODES))
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("case", list(_CASES))
def test_additive_bias_matches_reference(case, dtype, bias_mode, heads):
    backend, seq_lens = _CASES[case]
    _check_case(case, backend, seq_lens, {}, dtype, bias_mode, heads)


@requires_gfx950_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("heads", sorted(_HEADS))
@pytest.mark.parametrize("bias_mode", _EXTRA_BIAS_MODES)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("case", list(_EXTRA_CASES))
def test_additive_bias_with_features_matches_reference(case, dtype, bias_mode, heads):
    backend, seq_lens, features = _EXTRA_CASES[case]
    _check_case(case, backend, seq_lens, features, dtype, bias_mode, heads)


@requires_gfx950_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("bias_mode", _EXTRA_BIAS_MODES)
@pytest.mark.parametrize("heads", sorted(_COMBO_HEADS))
def test_additive_bias_combo_cohort_matches_reference(heads, bias_mode):
    _check_case(
        "tiled_combo",
        "tiled",
        _COMBO_SEQ_LENS,
        {},
        "bf16",
        bias_mode,
        heads,
        head_size=64,
        block_size=32,
    )


# Bias combined with the other features the routed candidates declare: the
# top-left causal and no-mask alignments, attention sinks, an fp8 K/V cache and
# the LSE output (2D and split-KV). feature -> (problem fields, paths).
_FEATURES = {
    "top_left": ({"causal_top_left": True, "right_bound": 0}, ("2d", "3d")),
    "no_mask": ({"right_bound": -1}, ("2d", "3d")),
    "sinks": ({"use_sinks": True}, ("2d", "3d")),
    "fp8_kv": ({"use_fp8": True}, ("2d", "3d")),
    "lse": ({"emit_lse": True}, ("2d", "3d")),
}
_FEATURE_PATHS = {
    "2d": ("tiled", [(37, 101), (64, 64), (5, 130)]),
    "3d": ("3d", [(4, 300), (16, 517), (1, 64)]),
}
_FEATURE_CASES = [
    (feature, path) for feature, (_, paths) in _FEATURES.items() for path in paths
]
_FP8_K_SCALE, _FP8_V_SCALE = 0.5, 0.25


def _check_feature_case(
    feature,
    path,
    dtype,
    bias_mode,
    heads,
    *,
    seq_lens=None,
    head_size=_HEAD_SIZE,
    block_size=_BLOCK_SIZE,
):
    fields, _ = _FEATURES[feature]
    backend, default_lens = _FEATURE_PATHS[path]
    seq_lens = seq_lens or default_lens
    hq, hkv = {**_HEADS, **_COMBO_HEADS}[heads]
    q, k, v, block_table, cu_q, seqused_k = _make_inputs(
        seq_lens, hq, hkv, dtype, head_size=head_size, block_size=block_size
    )
    k_scale = v_scale = 1.0
    if fields.get("use_fp8"):
        # Quantise the cache; the reference reads the same dequantised values.
        k_scale, v_scale = _FP8_K_SCALE, _FP8_V_SCALE
        k = (k.float() / k_scale).to(torch.float8_e4m3fn)
        v = (v.float() / v_scale).to(torch.float8_e4m3fn)
        k_ref, v_ref = k.float() * k_scale, v.float() * v_scale
    else:
        k_ref, v_ref = k, v
    max_q = max(s[0] for s in seq_lens)
    max_k = max(s[1] for s in seq_lens)
    scale = head_size**-0.5
    storage, b_stride, h_stride, sq_stride, bias_full = _make_bias(
        bias_mode, len(seq_lens), hq, max_q, max_k
    )
    # The kernels read one sink logit per query head in the Q dtype.
    sinks = (
        torch.randn(hq, dtype=q.dtype, device="cuda") + 2.0
        if fields.get("use_sinks")
        else None
    )
    problem = UnifiedAttentionProblem(
        total_q=q.shape[0],
        num_seqs=len(seq_lens),
        num_query_heads=hq,
        num_kv_heads=hkv,
        head_size=head_size,
        block_size=block_size,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        dtype=dtype,
        use_additive_bias=True,
        num_kv_blocks=int(k.shape[0]),
        **fields,
    )
    lse = (
        torch.full((q.shape[0], hq, 1), float("nan"), device="cuda")
        if fields.get("emit_lse")
        else None
    )
    out = torch.empty_like(q)
    run_unified_attention_torch(
        problem=problem,
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_q,
        seqused_k=seqused_k,
        softmax_scale=scale,
        block_table=block_table,
        softcap=0.0,
        sinks=sinks,
        additive_bias=storage,
        additive_bias_batch_stride=b_stride,
        additive_bias_head_stride=h_stride,
        additive_bias_sq_stride=sq_stride,
        backend=backend,
        k_scale=k_scale,
        v_scale=v_scale,
        lse=lse,
    )
    torch.cuda.synchronize()

    mask_kw = {
        "top_left": bool(fields.get("causal_top_left", False)),
        "right_bound": int(fields.get("right_bound", 0)),
        "sinks": sinks,
    }
    ref_args = (q, k_ref, v_ref, block_table, seq_lens, scale)
    ref, ref_lse = _reference(*ref_args, bias_full, return_lse=True, **mask_kw)
    ref_nobias = _reference(*ref_args, None, **mask_kw)
    tol = _TOL[dtype]
    label = f"{feature}/{path}/{dtype}/{bias_mode}/{heads}"
    max_abs = (ref - out.float()).abs().max().item()
    assert max_abs < tol, f"{label}: max_abs={max_abs:.3e} >= {tol}"
    shift = (ref - ref_nobias).abs().max().item()
    assert shift > 10 * tol, f"{label}: bias too weak to observe (shift={shift:.3e})"
    if lse is not None:
        assert not torch.isnan(lse).any(), f"{label}: NaN in the LSE"
        assert torch.equal(torch.isneginf(lse), torch.isneginf(ref_lse)), label
        finite = torch.isfinite(ref_lse)
        lse_err = (lse[finite] - ref_lse[finite]).abs().max().item()
        assert lse_err < tol, f"{label}: lse max_abs={lse_err:.3e} >= {tol}"


@requires_gfx950_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("heads", sorted(_HEADS))
@pytest.mark.parametrize("bias_mode", _EXTRA_BIAS_MODES)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(("feature", "path"), _FEATURE_CASES)
def test_additive_bias_with_more_features_matches_reference(
    feature, path, dtype, bias_mode, heads
):
    _check_feature_case(feature, path, dtype, bias_mode, heads)


# -inf bias entries: whole key columns, one fully masked query row, and on 3D
# one whole split-KV segment. case -> (backend, seq_lens, emit_lse).
_NEG_INF_CASES = {
    "2d": ("tiled", [(37, 101), (64, 64), (5, 130)], False),
    "2d_transposed32": ("tiled", [(300, 420), (512, 512)], False),
    "2d_lse": ("tiled", [(37, 101), (64, 64), (5, 130)], True),
    "3d": ("3d", [(1, 300), (3, 517), (1, 64)], False),
    "3d_lse": ("3d", [(1, 300), (3, 517), (1, 64)], True),
}


def _neg_inf_bias(problem, backend, seq_lens, hq, max_q, max_k):
    """A full-layout bias with -inf entries; returns (bias, masked segment).

    The segment is ``(seq, first key, end key)`` of split-KV segment 1 of the
    longest sequence on 3D (segments cover ``cdiv(kv_len, num_segments * T)``
    tiles of ``T`` keys), else None."""
    bias = torch.randn(
        len(seq_lens), hq, max_q, max_k, dtype=torch.float32, device="cuda"
    )
    bias[:, :, :, [0, 7]] = float("-inf")
    bias[0, :, 0, :] = float("-inf")
    if backend != "3d":
        return bias, None
    from kernels.common.attention_unified import _tiled_3d_spec_from_problem

    spec = _tiled_3d_spec_from_problem(problem, "gfx950")
    seq = max(range(len(seq_lens)), key=lambda i: seq_lens[i][1])
    kv_len = seq_lens[seq][1]
    seg_keys = -(-kv_len // (spec.num_segments * spec.tile_size)) * spec.tile_size
    assert 2 * seg_keys <= kv_len, (seg_keys, kv_len)
    bias[seq, :, :, seg_keys : 2 * seg_keys] = float("-inf")
    return bias, (seq, seg_keys, 2 * seg_keys)


@requires_gfx950_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("case", list(_NEG_INF_CASES))
def test_neg_inf_bias_masks_rows_and_segments(case, dtype):
    backend, seq_lens, emit_lse = _NEG_INF_CASES[case]
    hq, hkv = _HEADS["gqa"]
    q, k, v, block_table, cu_q, seqused_k = _make_inputs(seq_lens, hq, hkv, dtype)
    max_q = max(s[0] for s in seq_lens)
    max_k = max(s[1] for s in seq_lens)
    scale = _HEAD_SIZE**-0.5
    problem = UnifiedAttentionProblem(
        total_q=q.shape[0],
        num_seqs=len(seq_lens),
        num_query_heads=hq,
        num_kv_heads=hkv,
        head_size=_HEAD_SIZE,
        block_size=_BLOCK_SIZE,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        dtype=dtype,
        use_additive_bias=True,
        num_kv_blocks=int(k.shape[0]),
        emit_lse=emit_lse,
    )
    bias, segment = _neg_inf_bias(problem, backend, seq_lens, hq, max_q, max_k)
    lse = (
        torch.full((q.shape[0], hq, 1), float("nan"), device="cuda")
        if emit_lse
        else None
    )
    out = torch.empty_like(q)
    run_unified_attention_torch(
        problem=problem,
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_q,
        seqused_k=seqused_k,
        softmax_scale=scale,
        block_table=block_table,
        softcap=0.0,
        additive_bias=bias,
        additive_bias_batch_stride=hq * max_q * max_k,
        additive_bias_head_stride=max_q * max_k,
        additive_bias_sq_stride=max_k,
        backend=backend,
        lse=lse,
    )
    torch.cuda.synchronize()

    ref, ref_lse = _reference(
        q, k, v, block_table, seq_lens, scale, bias, return_lse=True
    )
    empty = torch.isneginf(ref_lse[..., 0])
    label = f"{case}/{dtype}"
    assert empty.any(), f"{label}: no fully masked row"
    assert not torch.isnan(out).any(), f"{label}: NaN in the output"
    got = out.float()
    assert torch.equal(got[empty], torch.zeros_like(got[empty])), label
    max_abs = (ref - got)[~empty].abs().max().item()
    assert max_abs < _TOL[dtype], f"{label}: max_abs={max_abs:.3e}"
    if segment is not None:
        # The masked segment is a whole one, yet the sequence's rows still see
        # keys elsewhere, so the reduce must drop that segment, not the rows.
        seq, _, _ = segment
        start = sum(s[0] for s in seq_lens[:seq])
        rows = slice(start, start + seq_lens[seq][0])
        assert not empty[rows].any(), f"{label}: masked segment emptied a row"
    if lse is not None:
        assert not torch.isnan(lse).any(), f"{label}: NaN in the LSE"
        assert torch.equal(torch.isneginf(lse[..., 0]), empty), label
        lse_err = (lse[..., 0] - ref_lse[..., 0])[~empty].abs().max().item()
        assert lse_err < _TOL[dtype], f"{label}: LSE max_abs={lse_err:.3e}"


@requires_gfx950_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("head_size", [64, 128])
def test_additive_bias_fp8_single_batch_combo_matches_reference(head_size):
    # The fp8 K/V single-batch cohort (d64/d128, block 64, 64/8 heads) takes the
    # transposed 32x32 combo; bias must turn its mask-once/mask-limit opts off.
    _check_feature_case(
        "fp8_kv",
        "2d",
        "bf16",
        "full_padded_sq",
        "gqa8_h64",
        seq_lens=[(384, 640)],
        head_size=head_size,
        block_size=64,
    )


# Dispatcher lane: an AttentionRequest(use_attn_bias=True) pinned to a gfx950
# tuning geometry, bound with tensors["attn_bias"] and launched through the
# candidate's binding. One geometry per score path.
_DISPATCH_GEOMETRIES = (
    "attention_gfx950_u2d_narrow_nw2_mw16_t4xb_llvm",
    "attention_gfx950_u2d_transposed32_nw2_mw32_t2xb_llvm",
    "attention_gfx950_u3d_splitkv_seg8_t1xb",
)


@requires_gfx950_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("bias_shape", ["full", "key_padding"])
@pytest.mark.parametrize("geometry", _DISPATCH_GEOMETRIES)
def test_dispatched_attn_bias_matches_reference(geometry, bias_shape):
    from dispatch.attention import AttentionRequest, attention_execution_candidates
    from dispatch.attention.common import AttentionMaskType

    batch, seqlen_q, seqlen_k, hq, hkv, dtype = 2, 96, 200, 8, 2, "bf16"
    seq_lens = [(seqlen_q, seqlen_k)] * batch
    q, k, v, block_table, cu_q, seqused_k = _make_inputs(seq_lens, hq, hkv, dtype)
    shape = (batch, hq, seqlen_q, seqlen_k)
    if bias_shape == "key_padding":
        shape = (batch, 1, 1, seqlen_k)
    bias = torch.randn(*shape, dtype=torch.float32, device="cuda")

    candidate = next(
        c for c in attention_execution_candidates() if c.name.startswith(geometry)
    )
    request = AttentionRequest(
        batch=batch,
        nhead_q=hq,
        nhead_k=hkv,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        hdim_q=_HEAD_SIZE,
        hdim_v=_HEAD_SIZE,
        kv_block_size=_BLOCK_SIZE,
        arch="gfx950",
        dtype=dtype,
        mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
        use_attn_bias=True,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
    )
    ok, why = candidate.admits(request)
    assert ok, why
    spec = candidate.select_spec(request)
    assert spec.kernel_spec.use_additive_bias
    out = torch.empty_like(q)
    tensors = {
        "q": q,
        "k": k,
        "v": v,
        "out": out,
        "cu_seqlens_q": cu_q,
        "seqused_k": seqused_k,
        "block_table": block_table,
        "attn_bias": bias,
    }
    candidate.bound_torch(request, spec, tensors).launch()
    torch.cuda.synchronize()

    scale = _HEAD_SIZE**-0.5
    bias_full = bias.expand(batch, hq, seqlen_q, seqlen_k)
    ref_args = (q, k, v, block_table, seq_lens, scale)
    ref = _reference(*ref_args, bias_full)
    tol = _TOL[dtype]
    max_abs = (ref - out.float()).abs().max().item()
    assert max_abs < tol, f"{geometry}/{bias_shape}: max_abs={max_abs:.3e} >= {tol}"
    shift = (ref - _reference(*ref_args, None)).abs().max().item()
    assert shift > 10 * tol, f"bias too weak to be observable (shift={shift:.3e})"


# Host-side argument validation. These raise before any compile or launch, so
# they run on CPU tensors and need no GPU.


def _bias_problem(use_additive_bias=True):
    return UnifiedAttentionProblem(
        total_q=4,
        num_seqs=1,
        num_query_heads=8,
        num_kv_heads=8,
        head_size=_HEAD_SIZE,
        block_size=_BLOCK_SIZE,
        max_seqlen_q=4,
        max_seqlen_k=64,
        dtype="bf16",
        use_additive_bias=use_additive_bias,
    )


def _launch_with_bias(problem, additive_bias, **kwargs):
    """Call the public entry with placeholder tensors; validation fires first."""
    run_unified_attention_torch(
        problem=problem,
        q=None,
        k=None,
        v=None,
        out=None,
        cu_seqlens_q=None,
        seqused_k=None,
        softmax_scale=1.0,
        block_table=None,
        softcap=0.0,
        additive_bias=additive_bias,
        **kwargs,
    )


_BIAS = dict(dtype=torch.float32) if _HAS_TORCH else {}


@pytest.mark.parametrize(
    "use_bias, make_bias, sq_stride, match",
    [
        (True, lambda: None, 64, "requires an additive_bias tensor"),
        (False, lambda: torch.zeros(4, 64, **_BIAS), 64, "use_additive_bias is False"),
        (True, lambda: torch.zeros(4, 64, dtype=torch.float16), 64, "float32"),
        (True, lambda: torch.zeros(64, 4, **_BIAS).t(), 64, "contiguous along Sk"),
        (True, lambda: torch.zeros(4, 64, **_BIAS), None, "requires additive_bias_sq"),
        (True, lambda: torch.zeros(4, 64, **_BIAS), 32, "or >= max_seqlen_k=64"),
    ],
    ids=["missing", "flag_off", "fp16", "sk_strided", "no_sq_stride", "short_sq"],
)
@requires_torch
def test_additive_bias_bad_args_rejected(use_bias, make_bias, sq_stride, match):
    with pytest.raises(ValueError, match=match):
        _launch_with_bias(
            _bias_problem(use_bias),
            make_bias(),
            additive_bias_sq_stride=sq_stride,
        )


@requires_torch
@pytest.mark.parametrize("sq_stride", [0, 64, 71])
def test_additive_bias_sq_stride_accepted(sq_stride):
    from kernels.common.attention_unified import _validate_additive_bias

    bias = torch.zeros(4, 71, **_BIAS)
    assert (
        _validate_additive_bias(_bias_problem(), bias, 0, 0, sq_stride, bias.device)
        == sq_stride
    )


def test_additive_bias_off_needs_no_sq_stride():
    from kernels.common.attention_unified import _validate_additive_bias

    assert _validate_additive_bias(_bias_problem(False), None, 0, 0, None, None) == 0


@requires_torch
def test_additive_bias_device_mismatch_rejected():
    from kernels.common.attention_unified import _validate_additive_bias

    bias = torch.zeros(4, 64, **_BIAS)  # cpu
    with pytest.raises(ValueError, match="same device as q"):
        _validate_additive_bias(_bias_problem(), bias, 0, 0, 64, torch.device("meta"))


# i32 boundary shape: H=16, Sq=Sk=8192 -> a full [B,H,Sq,Sk] bias is 2**30
# elements per sequence, so B=2 tops out at index 2**31 - 1 and B=3 wraps i32.
_LONG = 8192
_LONG_HEAD = _LONG * _LONG
_LONG_BATCH = 16 * _LONG_HEAD


def _long_bias_problem(num_seqs):
    return UnifiedAttentionProblem(
        total_q=num_seqs * _LONG,
        num_seqs=num_seqs,
        num_query_heads=16,
        num_kv_heads=16,
        head_size=_HEAD_SIZE,
        block_size=_BLOCK_SIZE,
        max_seqlen_q=_LONG,
        max_seqlen_k=_LONG,
        dtype="bf16",
        use_additive_bias=True,
    )


def _validate_long(num_seqs, batch_stride, numel=1):
    from kernels.common.attention_unified import _validate_additive_bias

    # A meta tensor reports its storage size without allocating it.
    return _validate_additive_bias(
        _long_bias_problem(num_seqs),
        torch.empty(numel, device="meta", **_BIAS),
        batch_stride,
        _LONG_HEAD,
        _LONG,
        torch.device("meta"),
    )


@requires_torch
def test_additive_bias_i32_index_overflow_rejected():
    with pytest.raises(NotImplementedError, match="i32 bias addressing"):
        _validate_long(3, _LONG_BATCH)


@pytest.mark.parametrize(
    "num_seqs, batch_stride, numel",
    [(2, _LONG_BATCH, 2 * _LONG_BATCH), (3, 0, _LONG_BATCH)],
    ids=["max_index_fits", "batch_broadcast"],
)
@requires_torch
def test_additive_bias_i32_index_limit_accepted(num_seqs, batch_stride, numel):
    assert _validate_long(num_seqs, batch_stride, numel) == _LONG


@requires_torch
@pytest.mark.parametrize("batch_stride", [-1, 2**31], ids=["negative", "over_i32"])
def test_additive_bias_stride_range_rejected(batch_stride):
    from kernels.common.attention_unified import _validate_additive_bias

    bias = torch.zeros(4, 64, **_BIAS)
    with pytest.raises(ValueError, match="additive_bias_batch_stride"):
        _validate_additive_bias(_bias_problem(), bias, batch_stride, 0, 64, bias.device)


# _bias_problem reads index 3*sq_stride + 63 at most, so sq_stride=64 needs 256
# backed elements past the storage offset.
@pytest.mark.parametrize(
    "make_bias, ok",
    [
        (lambda: torch.zeros(256, **_BIAS), True),
        (lambda: torch.zeros(255, **_BIAS), False),
        (lambda: torch.zeros(320, **_BIAS)[64:], True),
        (lambda: torch.zeros(320, **_BIAS)[65:], False),
    ],
    ids=["exact", "short", "offset_exact", "offset_short"],
)
@requires_torch
def test_additive_bias_storage_bounds(make_bias, ok):
    from kernels.common.attention_unified import _validate_additive_bias

    bias = make_bias()
    args = (_bias_problem(), bias, 0, 0, 64, bias.device)
    if ok:
        assert _validate_additive_bias(*args) == 64
    else:
        with pytest.raises(ValueError, match="only backs"):
            _validate_additive_bias(*args)


def test_additive_bias_attn_values_rejects_null_ptr():
    """Direct ``_attn_values`` callers (benchmarks, parity builders) bypass the
    entry-point check; the packer must still refuse a null bias pointer."""
    from kernels.common.attention_unified import _attn_values

    with pytest.raises(ValueError, match="requires an additive_bias tensor"):
        _attn_values(
            problem=_bias_problem(),
            q=None,
            k=None,
            v=None,
            out=None,
            cu_seqlens_q=None,
            seqused_k=None,
            softmax_scale=1.0,
            block_table=None,
            softcap=0.0,
            sinks=None,
            bt_stride=0,
            include_bt_stride=True,
            additive_bias=None,
            include_additive_bias_strides=True,
            kv_addr_limit=None,
        )
