# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tiled 2D attention: a row whose first KV tile is fully masked (sliding
window) must still get the right O.

The CTA's tile range starts at the window start of its first row, so later rows
can see a fully masked first tile. The running max must not be pinned at 0 by
that tile: with every visible score negative, exp2(s - 0) rounds P to 0 and O
collapses although the row sees keys. P rounds to 0 below about -17 natural for
fp16 and -93 for bf16 (P is cast to bf16 before PV); the scalar kernels keep P
in fp32, which rounds to 0 below about -104 (about -87 with denormals flushed). Each
epilogue branch is pinned so all three clamp sites are covered, at KV page
sizes 16 and 64 and head sizes 64, 128 and 256 (256 only where the variant fits
the LDS budget). One auto-route case per arch (no ``tuning_spec``, D64 bf16)
covers a multi-warp production kernel of the default builder.

Blind spots: gfx942 D128 sliding-window shapes normally route to the 4-warp
GQA kernel (finite -1e30 sentinel, not affected), so these cases pin tuning
variants of the default builder via ``tuning_spec``. Not covered here: the
scalar kernels (see test_attention_unified_scalar_masked_tile_numeric.py) and
multi-sequence batches.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from kernels.common.attention_unified import (  # noqa: E402
    UnifiedAttentionProblem,
    _gfx942_4warp_route,
    _tiled_spec_from_problem,
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

# One dispatcher geometry per epilogue branch of build_unified_attention_2d_tiled,
# keyed by (arch, KV page size). On gfx942 the 2x-block tiles of the 32-row
# variants exceed the 64 KiB LDS budget at page size 64, so the 1x-block
# variants of the same branches stand in there.
_BRANCH_VARIANT = {
    ("gfx942", 16): {
        "transposed32": "transposed_x8_nw1_mw32_t2xb_llvm",
        "mfma32": "wide32x8_nw1_mw32_t2xb_llvm",
        "striped16": "narrow_nw1_mw16_t1xb_llvm",
    },
    ("gfx942", 64): {
        "transposed32": "transposed_x8_nw1_mw32_t1xb_llvm",
        "mfma32": "wide32x8_nw1_mw32_t1xb_llvm",
        "striped16": "narrow_nw1_mw16_t1xb_llvm",
    },
    ("gfx950", 16): {
        "transposed32": "transposed32_nw1_mw32_t2xb_llvm",
        "mfma32": "wide32_nw1_mw32_t2xb_llvm",
        "striped16": "narrow_nw1_mw16_t1xb_llvm",
    },
    ("gfx950", 64): {
        "transposed32": "transposed32_nw1_mw32_t2xb_llvm",
        "mfma32": "wide32_nw1_mw32_t2xb_llvm",
        "striped16": "narrow_nw1_mw16_t1xb_llvm",
    },
}

# (dtype, natural-log score of every visible key), below the P underflow point
# when m is pinned at 0: about -17 for fp16 and -93 for bf16.
_SCORES = [("fp16", -20.0), ("bf16", -200.0)]
_WINDOW = 64
_HQ, _HKV = 32, 8
_HEAD_SIZES = [64, 128, 256]
_PAGE_SIZES = [16, 64]


def _problem(dtype, q_len, kv_len, num_blocks, bs, head_size):
    return UnifiedAttentionProblem(
        total_q=q_len,
        num_seqs=1,
        num_query_heads=_HQ,
        num_kv_heads=_HKV,
        head_size=head_size,
        block_size=bs,
        max_seqlen_q=q_len,
        max_seqlen_k=kv_len,
        dtype=dtype,
        sliding_window=_WINDOW,
        num_kv_blocks=num_blocks,
    )


def _branch_spec(problem, branch, bs):
    from dispatch.attention.unified_rules import canonicalize_tuning_spec

    arch = _arch()
    if arch == "gfx942":
        from dispatch.attention.gfx942_unified import (
            GFX942_TUNING_VARIANTS as variants,
        )
    else:
        from dispatch.attention.gfx950_unified import (
            GFX950_TUNING_VARIANTS as variants,
        )
    base = branch.removesuffix("_s1")
    spec_id = f"{arch}_u2d_{_BRANCH_VARIANT[(arch, bs)][base]}"
    variant = next(v for v in variants if v.spec_id == spec_id)
    knobs = {"use_transposed_scalar_state": True} if branch.endswith("_s1") else {}
    spec, why = canonicalize_tuning_spec(problem, variant, knobs)
    if spec is None and problem.head_size == 256:
        pytest.skip(f"{spec_id} {knobs} does not take D256 here: {why}")
    assert spec is not None, f"{spec_id} {knobs}: {why}"
    return spec


def _rows_with_masked_leading_tile(kernel_spec, q_len):
    """Query rows whose CTA's first KV tile lies wholly before their window.

    This recomputes the kernel's tile-start rule (the CTA's first tile is the one
    holding its first row's window start) instead of reading it from the kernel,
    so it would miss a change to that rule; the GPU failure on unfixed kernels
    is what confirms the two agree.
    """
    tile = kernel_spec.tile_size
    tokens = kernel_spec.block_m_per_warp * kernel_spec.num_warps // (_HQ // _HKV)
    rows = []
    for q0 in range(0, q_len, tokens):
        if q0 < _WINDOW - 1:
            continue
        first_tile_end = ((q0 - _WINDOW + 1) // tile + 1) * tile
        rows += [
            q0 + k
            for k in range(tokens)
            if q0 + k < q_len and q0 + k - _WINDOW + 1 >= first_tile_end
        ]
    return rows


def _reference(q, k_cache, v_cache, kv_len, block_table, scale, bs):
    """FP32 O for one sequence, bottom-right causal + sliding window."""
    q_len, _, d = q.shape
    nblk = (kv_len + bs - 1) // bs
    blocks = block_table[0, :nblk].long()
    k = k_cache[blocks].reshape(-1, _HKV, d)[:kv_len].float()
    v = v_cache[blocks].reshape(-1, _HKV, d)[:kv_len].float()
    k = k.transpose(0, 1).repeat_interleave(_HQ // _HKV, dim=0)
    v = v.transpose(0, 1).repeat_interleave(_HQ // _HKV, dim=0)
    s = torch.matmul(q.float().transpose(0, 1), k.transpose(-1, -2)) * scale
    p = torch.arange(q_len, device=q.device).view(-1, 1)
    kk = torch.arange(kv_len, device=q.device).view(1, -1)
    diag = p + (kv_len - q_len)
    visible = (kk <= diag) & (diag - kk < _WINDOW)
    s = s.masked_fill(~visible, float("-inf"))
    return torch.matmul(torch.softmax(s, dim=-1), v).transpose(0, 1)


def _run_and_check(dtype, score, bs, head_size, spec_for):
    """Run one sequence where every visible score is ``score`` and check O.

    ``spec_for(problem)`` returns the tuning spec to pin, or ``None`` for the
    production auto route.
    """
    q_len = kv_len = 256
    tdt = torch.float16 if dtype == "fp16" else torch.bfloat16
    g = torch.Generator(device="cuda").manual_seed(0)
    max_blocks = (kv_len + bs - 1) // bs
    num_blocks = max_blocks + 3
    scale = 1.0 / math.sqrt(head_size)
    # Every visible score equals `score`: q = a, k = b with a * b * D * scale = score.
    a = 1.0 if dtype == "fp16" else 4.0
    b = score / (a * head_size * scale)
    q = torch.full((q_len, _HQ, head_size), a, device="cuda", dtype=tdt)
    k_cache = torch.full((num_blocks, bs, _HKV, head_size), b, device="cuda", dtype=tdt)
    v_cache = torch.randn(
        num_blocks, bs, _HKV, head_size, device="cuda", dtype=tdt, generator=g
    )
    perm = torch.randperm(num_blocks, device="cuda", generator=g)[:max_blocks]
    block_table = perm.view(1, max_blocks).to(torch.int32).contiguous()
    cu = torch.tensor([0, q_len], dtype=torch.int32, device="cuda")
    seqused = torch.tensor([kv_len], dtype=torch.int32, device="cuda")

    problem = _problem(dtype, q_len, kv_len, num_blocks, bs, head_size)
    spec = spec_for(problem)
    kernel_spec = (
        spec.kernel_spec
        if spec is not None
        else _tiled_spec_from_problem(problem, _arch())
    )
    assert _rows_with_masked_leading_tile(
        kernel_spec, q_len
    ), "geometry no longer gives any row a fully masked leading tile"
    out = torch.empty_like(q)
    run_unified_attention_torch(
        problem=problem,
        q=q,
        k=k_cache,
        v=v_cache,
        out=out,
        cu_seqlens_q=cu,
        seqused_k=seqused,
        softmax_scale=scale,
        block_table=block_table,
        softcap=0.0,
        backend="tiled",
        tuning_spec=spec,
    )
    torch.cuda.synchronize()
    o_ref = _reference(q, k_cache, v_cache, kv_len, block_table, scale, bs)

    assert torch.isfinite(out).all(), "non-finite O"
    err = (out.float() - o_ref).abs()
    # Every row sees at least one key, so a collapsed row has |O| ~ 0 while the
    # reference is the mean of V over its window (|mean| ~ 0.1 for randn V).
    worst = err.amax(dim=(1, 2))
    tol = 2e-2 if dtype == "fp16" else 4e-2
    assert worst.max().item() < tol, (
        f"{(worst >= tol).sum().item()} of {q_len} query rows wrong; first bad "
        f"row {int((worst >= tol).nonzero()[0])}"
    )


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("head_size", _HEAD_SIZES)
@pytest.mark.parametrize("bs", _PAGE_SIZES)
@pytest.mark.parametrize("dtype,score", _SCORES)
@pytest.mark.parametrize(
    "branch", ["transposed32", "transposed32_s1", "mfma32", "striped16"]
)
def test_masked_leading_tile_keeps_rows_with_keys(branch, dtype, score, bs, head_size):
    _run_and_check(dtype, score, bs, head_size, lambda p: _branch_spec(p, branch, bs))


@requires_cdna3_or_4
@pytest.mark.gpu
def test_masked_leading_tile_auto_route_d64_bf16():
    # No tuning spec: the production route. At D64 bf16 with a sliding window it
    # is the default builder with several warps per CTA (gfx942: 4 warps, tile
    # 16; gfx950: 2 warps, tile 64), not the gfx942 4-warp GQA kernel.
    arch = _arch()
    problem = _problem("bf16", 256, 256, 256 // 16 + 3, 16, 64)
    assert _gfx942_4warp_route(problem, arch) is None
    assert _tiled_spec_from_problem(problem, arch).num_warps > 1
    _run_and_check("bf16", -200.0, 16, 64, lambda p: None)
