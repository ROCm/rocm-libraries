# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Scalar unified attention: a row whose leading keys are all masked must still
get the right O.

A sliding window masks every key before the window start. The online softmax
replaces a running max of -inf by 0 only as the exponent reference; carrying
that 0 to the next key pins m at 0, and with every visible score at about -110
(natural log) exp2(s - 0) underflows the fp32 P to 0, so O collapses (NaN or 0)
although the row sees keys.

Covered:
* scalar 2D (``backend="scalar"``), bottom-right and top-left, fp16 and bf16;
* scalar split-3D segment + reduce (launched directly: no dispatcher route
  reaches them), where a whole segment can be masked for a row and must not
  set the merge's overall max;
* rows that see no key at all (top-left, ``q_len > kv_len``): O must be 0.
* a NaN per-segment sum must make the reduce give NaN for that row, not be
  skipped as an empty segment.

Not covered: the default-mask scalar 3D body, which applies no window (and
whose prefill rows with fully masked trailing segments are a separate
question), sinks, softcap and multi-sequence batches.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from kernels.common.attention_unified import (  # noqa: E402
    UnifiedAttention3DSpec,
    UnifiedAttentionProblem,
    UnifiedAttentionReduceSpec,
    build_unified_attention_3d,
    build_unified_attention_reduce,
    run_unified_attention_torch,
)


def _arch():
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_properties(0).gcnArchName.lower()
    return next((a for a in ("gfx942", "gfx950") if a in name), None)


requires_cdna3_or_4 = pytest.mark.skipif(
    _arch() is None, reason="needs a gfx942 or gfx950 GPU with ROCm torch"
)

# Natural-log score of every visible key: exp2(-110 * log2(e)) = 2^-158.7 is
# below the smallest fp32 denormal, so P underflows to 0 when m is pinned at 0.
_SCORE = -110.0
_WINDOW = 32
_HQ, _HKV, _D, _BS = 8, 2, 64, 16
_NUM_SEGMENTS = 8
_DTYPES = ["fp16", "bf16"]


def _tdt(dtype):
    return torch.float16 if dtype == "fp16" else torch.bfloat16


def _problem(dtype, q_len, kv_len, num_blocks, top_left):
    return UnifiedAttentionProblem(
        total_q=q_len,
        num_seqs=1,
        num_query_heads=_HQ,
        num_kv_heads=_HKV,
        head_size=_D,
        block_size=_BS,
        max_seqlen_q=q_len,
        max_seqlen_k=kv_len,
        dtype=dtype,
        sliding_window=_WINDOW,
        num_kv_blocks=num_blocks,
        causal_top_left=top_left,
    )


def _inputs(dtype, q_len, kv_len):
    """Q, paged K/V and metadata where every visible score equals ``_SCORE``."""
    tdt = _tdt(dtype)
    g = torch.Generator(device="cuda").manual_seed(0)
    max_blocks = (kv_len + _BS - 1) // _BS
    num_blocks = max_blocks + 3
    scale = 1.0 / math.sqrt(_D)
    # q = 1, k = b with D * b * scale = _SCORE: b = -13.75, exact in fp16 and bf16.
    b = _SCORE / (_D * scale)
    q = torch.ones((q_len, _HQ, _D), device="cuda", dtype=tdt)
    k_cache = torch.full((num_blocks, _BS, _HKV, _D), b, device="cuda", dtype=tdt)
    v_cache = torch.randn(
        num_blocks, _BS, _HKV, _D, device="cuda", dtype=tdt, generator=g
    )
    perm = torch.randperm(num_blocks, device="cuda", generator=g)[:max_blocks]
    return dict(
        q=q,
        k=k_cache,
        v=v_cache,
        block_table=perm.view(1, max_blocks).to(torch.int32).contiguous(),
        cu=torch.tensor([0, q_len], dtype=torch.int32, device="cuda"),
        seqused=torch.tensor([kv_len], dtype=torch.int32, device="cuda"),
        scale=scale,
        num_blocks=num_blocks,
    )


def _reference(data, kv_len, top_left):
    """FP32 O for one sequence, causal + sliding window; keyless rows give 0."""
    q = data["q"]
    q_len = q.shape[0]
    nblk = (kv_len + _BS - 1) // _BS
    blocks = data["block_table"][0, :nblk].long()
    k = data["k"][blocks].reshape(-1, _HKV, _D)[:kv_len].float()
    v = data["v"][blocks].reshape(-1, _HKV, _D)[:kv_len].float()
    k = k.transpose(0, 1).repeat_interleave(_HQ // _HKV, dim=0)
    v = v.transpose(0, 1).repeat_interleave(_HQ // _HKV, dim=0)
    s = torch.matmul(q.float().transpose(0, 1), k.transpose(-1, -2)) * data["scale"]
    p = torch.arange(q_len, device=q.device).view(-1, 1)
    kk = torch.arange(kv_len, device=q.device).view(1, -1)
    diag = p + (0 if top_left else kv_len - q_len)
    visible = (kk <= diag) & (diag - kk < _WINDOW)
    s = s.masked_fill(~visible, float("-inf"))
    probs = torch.softmax(s, dim=-1).nan_to_num(0.0)
    return torch.matmul(probs, v).transpose(0, 1), visible.any(dim=-1)


def _run_scalar_2d(problem, data):
    out = torch.empty_like(data["q"])
    run_unified_attention_torch(
        problem=problem,
        q=data["q"],
        k=data["k"],
        v=data["v"],
        out=out,
        cu_seqlens_q=data["cu"],
        seqused_k=data["seqused"],
        softmax_scale=data["scale"],
        block_table=data["block_table"],
        softcap=0.0,
        backend="scalar",
    )
    return out


def _run_scalar_3d(problem, data, between=None):
    """Scalar split-3D segment kernel then the scalar reduce, launched directly.

    ``between(segm_max, segm_sum)``, if given, runs after the segment launch and
    before the reduce launch, so a test can edit the per-segment workspace.
    """
    from rocke.helpers.compile import compile_kernel
    from rocke.helpers.spec import SignatureBuilder
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    arch = _arch()
    io = "f16" if problem.dtype == "fp16" else "bf16"
    q_len = problem.total_q

    def launcher(kernel, signature):
        art = compile_kernel(kernel, arch=arch, capture_ir_text=False)
        return KernelLauncher(
            hsaco=art.hsaco, kernel_name=art.kernel_name, signature=signature
        )

    seg_sig = (
        SignatureBuilder()
        .ptr("segm_output_ptr", "f32")
        .ptr("segm_max_ptr", "f32")
        .ptr("segm_expsum_ptr", "f32")
        .ptr("query_ptr", io)
        .ptr("key_cache_ptr", io)
        .ptr("value_cache_ptr", io)
        .ptr("sink_ptr", io)
        .ptr("block_tables_ptr", "i32")
        .ptr("seq_lens_ptr", "i32")
        .ptr("alibi_slopes_ptr", "f32")
        .ptr("qq_bias_ptr", "f32")
        .ptr("query_start_len_ptr", "i32")
        .scalar("scale", "f32")
        .scalar("k_scale", "f32")
        .scalar("v_scale", "f32")
        .scalar("softcap", "f32")
        .scalar("num_seqs", "i32")
        .build()
    )
    red_sig = (
        SignatureBuilder()
        .ptr("output_ptr", io)
        .ptr("segm_output_ptr", "f32")
        .ptr("segm_max_ptr", "f32")
        .ptr("segm_expsum_ptr", "f32")
        .ptr("seq_lens_ptr", "i32")
        .ptr("query_start_len_ptr", "i32")
        .build()
    )
    seg = launcher(
        build_unified_attention_3d(
            UnifiedAttention3DSpec(problem, num_segments=_NUM_SEGMENTS), arch=arch
        ),
        seg_sig,
    )
    red = launcher(
        build_unified_attention_reduce(
            UnifiedAttentionReduceSpec(problem, num_segments=_NUM_SEGMENTS), arch=arch
        ),
        red_sig,
    )

    # The segm_ml descriptor spans max_seqlen_q + 1 tokens.
    rows = problem.max_seqlen_q + 1
    nan = float("nan")
    segm_out = torch.full(
        (rows, _HQ, _NUM_SEGMENTS, _D), nan, device="cuda", dtype=torch.float32
    )
    segm_max = torch.full((rows, _HQ, _NUM_SEGMENTS), nan, device="cuda")
    segm_sum = torch.full((rows, _HQ, _NUM_SEGMENTS), nan, device="cuda")
    out = torch.empty_like(data["q"])
    seg(
        {
            "segm_output_ptr": segm_out,
            "segm_max_ptr": segm_max,
            "segm_expsum_ptr": segm_sum,
            "query_ptr": data["q"],
            "key_cache_ptr": data["k"],
            "value_cache_ptr": data["v"],
            "sink_ptr": 0,
            "block_tables_ptr": data["block_table"],
            "seq_lens_ptr": data["seqused"],
            "alibi_slopes_ptr": 0,
            "qq_bias_ptr": 0,
            "query_start_len_ptr": data["cu"],
            "scale": float(data["scale"]),
            "k_scale": 1.0,
            "v_scale": 1.0,
            "softcap": 0.0,
            "num_seqs": 1,
        },
        config=LaunchConfig(grid=(q_len, _HQ, _NUM_SEGMENTS * _D), block=(64, 1, 1)),
    )
    if between is not None:
        torch.cuda.synchronize()
        between(segm_max, segm_sum)
    red(
        {
            "output_ptr": out,
            "segm_output_ptr": segm_out,
            "segm_max_ptr": segm_max,
            "segm_expsum_ptr": segm_sum,
            "seq_lens_ptr": data["seqused"],
            "query_start_len_ptr": data["cu"],
        },
        config=LaunchConfig(grid=(q_len, _HQ, _D), block=(64, 1, 1)),
    )
    return out


def _check(out, ref, has_key, dtype):
    assert torch.isfinite(out).all(), (
        f"{(~torch.isfinite(out)).any(dim=-1).any(dim=-1).sum().item()} rows "
        "with non-finite O"
    )
    err = (out.float() - ref).abs().amax(dim=(1, 2))
    # A row that sees keys has |O| ~ mean of V over its window (~0.2 for randn
    # V), so a collapsed row (O = 0) fails; a keyless row must be exactly 0.
    tol = 2e-2 if dtype == "fp16" else 4e-2
    bad = err >= tol
    assert not bad.any(), (
        f"{bad.sum().item()} of {out.shape[0]} query rows wrong; first bad row "
        f"{int(bad.nonzero()[0])} (sees keys: {bool(has_key[bad][0])})"
    )
    if (~has_key).any():
        assert (out[~has_key] == 0).all(), "keyless rows must give O = 0"


# (q_len, kv_len, top_left). Rows past the first window have masked leading
# keys; the top-left q_len > kv_len case also has rows that see no key.
_CASES = {
    "bottom_right": (128, 128, False),
    "top_left": (128, 128, True),
    "top_left_keyless": (160, 96, True),
}


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("case", list(_CASES))
def test_scalar_2d_masked_leading_keys(case, dtype):
    q_len, kv_len, top_left = _CASES[case]
    data = _inputs(dtype, q_len, kv_len)
    problem = _problem(dtype, q_len, kv_len, data["num_blocks"], top_left)
    out = _run_scalar_2d(problem, data)
    torch.cuda.synchronize()
    ref, has_key = _reference(data, kv_len, top_left)
    _check(out, ref, has_key, dtype)


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("case", ["top_left", "top_left_keyless"])
def test_scalar_3d_masked_leading_keys_and_segments(case, dtype):
    # The scalar 3D body applies the window only for non-default masks.
    q_len, kv_len, top_left = _CASES[case]
    data = _inputs(dtype, q_len, kv_len)
    problem = _problem(dtype, q_len, kv_len, data["num_blocks"], top_left)
    out = _run_scalar_3d(problem, data)
    torch.cuda.synchronize()
    ref, has_key = _reference(data, kv_len, top_left)
    _check(out, ref, has_key, dtype)


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", _DTYPES)
def test_scalar_3d_reduce_propagates_nan_segment_sum(dtype):
    """A NaN per-segment sum must reach O, not be skipped as an empty segment.

    The reduce treats only an exact-zero sum as an empty segment (``fcmp oeq``).
    A guard of the form ``sum > 0`` (``fcmp ogt``) is false for NaN, so it would
    drop the segment and leave the row finite, hiding the corruption.
    """
    q_len, kv_len, top_left = _CASES["top_left"]
    # Row 100 sees keys 69..100 (top-left causal, window 32). The workspace
    # segm_ml is [max_seqlen_q + 1, Hq, num_segments], indexed by the global
    # token (one sequence starting at 0), so [row, head, seg] is the slot the
    # segment kernel wrote and the reduce reads for (row, head).
    row, head = 100, 3
    picked = {}

    def poison(segm_max, segm_sum):
        sums = segm_sum[row, head]
        with_keys = (sums > 0).nonzero().flatten()
        assert with_keys.numel() > 0, f"row {row} head {head}: no segment has keys"
        seg = int(with_keys[0])
        picked["seg"] = seg
        segm_sum[row, head, seg] = float("nan")

    data = _inputs(dtype, q_len, kv_len)
    problem = _problem(dtype, q_len, kv_len, data["num_blocks"], top_left)
    out = _run_scalar_3d(problem, data, between=poison)
    torch.cuda.synchronize()
    ref, has_key = _reference(data, kv_len, top_left)
    assert bool(has_key[row])

    assert torch.isnan(out[row, head]).all(), (
        f"row {row} head {head}: NaN sum in segment {picked['seg']} did not "
        "reach O (reduce skipped it as empty)"
    )
    others = torch.ones(out.shape[:2], dtype=torch.bool, device=out.device)
    others[row, head] = False
    assert torch.isfinite(out[others]).all(), "NaN leaked into another (row, head)"
    tol = 2e-2 if dtype == "fp16" else 4e-2
    err = (out.float() - ref).abs().amax(dim=-1)[others]
    assert err.max().item() < tol, f"{(err >= tol).sum().item()} (row, head) wrong"
