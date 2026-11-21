# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""ALiBi in the unified paged attention kernels (tiled 2D and split-KV 3D) on
gfx942 / gfx950, vs an FP32 paged reference.

The term is ``slope * (key - query)`` with the query at its diagonal position,
hipDNN's definition on the top-left diagonal; there the LSE is checked too. Slopes
are built in (hipDNN's per-head formula) or read from a caller tensor.
"""

import itertools
import math
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

import kernels.common.attention_unified as au  # noqa: E402
from dispatch.attention.common import AttentionTuningSpec  # noqa: E402
from kernels.common._attention_alibi import ALIBI_DESIGNS  # noqa: E402
from kernels.common.attention_unified import (  # noqa: E402
    UnifiedAttentionProblem,
    run_unified_attention_torch,
)
from kernels.gfx950.attention_tiled_2d import _alibi_design  # noqa: E402
from rocke.numeric.sdpa_reference import alibi_slopes  # noqa: E402


def _arch():
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_properties(0).gcnArchName.lower()
    return next((a for a in ("gfx942", "gfx950") if a in name), None)


_ARCH = _arch()
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(_ARCH is None, reason="needs a gfx942 or gfx950 GPU"),
]


def _tolerance(dtype):
    return 2e-2 if dtype == "fp16" else 4e-2


def _reference(
    q, k_cache, v_cache, q_lens, kv_lens, block_table, scale, slopes, tl, sinks=None
):
    """FP32 O and natural-log LSE; causal on the bottom-right diagonal, or the
    top-left one when ``tl``; ALiBi measured from the query's diagonal. A sink
    logit joins each row's denominator with no ALiBi term and no value (SEM-022)."""
    hq, d = q.shape[1], q.shape[2]
    hkv, bs = k_cache.shape[2], k_cache.shape[1]
    slope = slopes.float().view(hq, 1, 1)
    outs, lses, start = [], [], 0
    for i, (ql, kl) in enumerate(zip(q_lens, kv_lens)):
        qi = q[start : start + ql].float().transpose(0, 1)
        blocks = block_table[i, : (kl + bs - 1) // bs].long()
        k = k_cache[blocks].reshape(-1, hkv, d)[:kl].float()
        v = v_cache[blocks].reshape(-1, hkv, d)[:kl].float()
        k = k.transpose(0, 1).repeat_interleave(hq // hkv, dim=0)
        v = v.transpose(0, 1).repeat_interleave(hq // hkv, dim=0)
        s = torch.matmul(qi, k.transpose(-1, -2)) * scale
        p = torch.arange(ql, device=q.device).view(-1, 1)
        kk = torch.arange(kl, device=q.device).view(1, -1)
        diag = p if tl else p + (kl - ql)
        s = s + slope * (kk - diag)
        s = s.masked_fill(~(kk <= diag), float("-inf"))
        if sinks is not None:
            sink_col = sinks.float().view(hq, 1, 1).expand(hq, ql, 1)
            lse = torch.logsumexp(torch.cat([s, sink_col], dim=-1), dim=-1)
        else:
            lse = torch.logsumexp(s, dim=-1)
        has = torch.isfinite(lse)
        prob = torch.exp(s - torch.where(has, lse, 0.0).unsqueeze(-1))
        prob = torch.where(has.unsqueeze(-1), prob, 0.0)
        outs.append(torch.matmul(prob, v).transpose(0, 1))
        lses.append(lse.transpose(0, 1))
        start += ql
    return torch.cat(outs), torch.cat(lses).unsqueeze(-1)


def _run_case(
    q_lens,
    kv_lens,
    *,
    backend,
    hq=12,
    hkv=4,
    d=128,
    bs=16,
    dtype="bf16",
    top_left=False,
    lse=False,
    external=False,
    sinks=False,
    alibi_design=None,
    family=None,
    seed=0,
):
    """``family``: a 2D spec flag the dispatched kernel must set (or
    ``_PER_SCORE_ADD``), so the case runs the path it names; ``alibi_design``
    forces that arm on it."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    tdt = torch.float16 if dtype == "fp16" else torch.bfloat16
    total_q, num_seqs = sum(q_lens), len(q_lens)
    max_blocks = max(1, max((kl + bs - 1) // bs for kl in kv_lens))
    num_blocks = num_seqs * max_blocks + 3
    q = torch.randn(total_q, hq, d, device="cuda", dtype=tdt, generator=g)
    kc = torch.randn(num_blocks, bs, hkv, d, device="cuda", dtype=tdt, generator=g)
    vc = torch.randn(num_blocks, bs, hkv, d, device="cuda", dtype=tdt, generator=g)
    perm = torch.randperm(num_blocks, device="cuda", generator=g)[
        : num_seqs * max_blocks
    ]
    bt = perm.view(num_seqs, max_blocks).to(torch.int32).contiguous()
    cu = torch.tensor(
        [0, *itertools.accumulate(q_lens)], dtype=torch.int32, device="cuda"
    )
    seqused = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    slopes = torch.tensor(alibi_slopes(hq), dtype=torch.float32, device="cuda")
    if external:
        # Not the built-in vector, so a kernel ignoring the tensor fails.
        slopes = slopes.flip(0).contiguous()
    problem = UnifiedAttentionProblem(
        total_q=total_q,
        num_seqs=num_seqs,
        num_query_heads=hq,
        num_kv_heads=hkv,
        head_size=d,
        block_size=bs,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(kv_lens),
        dtype=dtype,
        num_kv_blocks=num_blocks,
        use_alibi=True,
        use_sinks=sinks,
        causal_top_left=top_left,
        right_bound=0,
        emit_lse=lse,
    )
    sink_t = torch.randn(hq, device="cuda", dtype=tdt, generator=g) if sinks else None
    tuning_spec = None
    if family is not None or alibi_design is not None:
        assert au.select_auto_path(problem, _ARCH) == "2d"
        spec = au._tiled_spec_from_problem(problem, _ARCH)
        if family == _PER_SCORE_ADD:
            assert _alibi_design(spec) is None, spec.kernel_name()
        elif family is not None:
            assert getattr(spec, family), f"{spec.kernel_name()} is not on {family}"
        if alibi_design is not None:
            tuning_spec = AttentionTuningSpec(
                path="2d",
                arch=_ARCH,
                builder_kind="tiled",
                compile_backend=au._select_2d_compile_backend(problem, _ARCH),
                candidate_name="alibi_design",
                tuning_id=alibi_design,
                kernel_spec=replace(spec, alibi_design=alibi_design),
                num_kv_blocks=num_blocks,
            )
    out = torch.empty_like(q)
    lse_t = torch.full((total_q, hq, 1), float("nan"), device="cuda") if lse else None
    scale = 1.0 / math.sqrt(d)
    run_unified_attention_torch(
        problem=problem,
        q=q,
        k=kc,
        v=vc,
        out=out,
        cu_seqlens_q=cu,
        seqused_k=seqused,
        softmax_scale=scale,
        block_table=bt,
        softcap=0.0,
        sinks=sink_t,
        backend=backend,
        alibi_slopes=slopes if external else None,
        lse=lse_t,
        tuning_spec=tuning_spec,
    )
    torch.cuda.synchronize()
    o_ref, lse_ref = _reference(
        q, kc, vc, q_lens, kv_lens, bt, scale, slopes, top_left, sink_t
    )
    tol = _tolerance(dtype)
    assert (out.float() - o_ref).abs().max().item() < tol
    if lse:
        finite = torch.isfinite(lse_ref)
        assert (lse_t[finite] - lse_ref[finite]).abs().max().item() < tol


_SHAPES = [
    ([1, 1, 1, 1], [4096, 777, 1500, 2049]),  # decode, ragged KV
    ([4, 2], [3000, 1029]),  # spec-decode
    ([128], [128]),  # prefill, s_q == s_kv
    ([96, 64], [300, 64]),  # chunked prefill
]


@pytest.mark.parametrize("external", [False, True], ids=["builtin", "external"])
@pytest.mark.parametrize("backend", ["tiled", "3d"])
@pytest.mark.parametrize("q_lens,kv_lens", _SHAPES)
def test_bottom_right(backend, q_lens, kv_lens, external):
    _run_case(q_lens, kv_lens, backend=backend, external=external)


@pytest.mark.parametrize("backend", ["tiled", "3d"])
@pytest.mark.parametrize(
    "q_lens,kv_lens",
    [([128], [128]), ([96, 64], [300, 64]), ([160, 64], [96, 64])],  # last: q > kv
)
def test_top_left_with_lse(backend, q_lens, kv_lens):
    if _ARCH != "gfx950":
        pytest.skip("top-left alignment is gfx950-only in the unified kernels")
    _run_case(q_lens, kv_lens, backend=backend, top_left=True, lse=True)


def test_bottom_right_lse_is_refused():
    with pytest.raises((ValueError, NotImplementedError), match="top-left"):
        _run_case([4], [64], backend="tiled", lse=True)


# The paths that kept the per-score add from before the design arms.
_PER_SCORE_ADD = "per-score add"

# gfx950 tiled 2D families ALiBi reaches through dispatch: (case kwargs, the spec
# flag of the family). Square prefill rows keep both diagonals equal.
_GFX950_FAMILIES = {
    "combo": (
        dict(q_lens=[512], kv_lens=[512], hq=64, hkv=8, d=64, bs=32),
        "use_transposed_mask_limit",
    ),
    "d128 single batch": (
        dict(q_lens=[1024], kv_lens=[1024], hq=32, hkv=8, bs=64),
        "use_transposed_mask_limit",
    ),
    "d128 multi batch": (
        dict(q_lens=[512, 384], kv_lens=[700, 384], hq=32, hkv=8, bs=64),
        "use_transposed_mask_limit",
    ),
    "d192 single batch": (
        dict(q_lens=[1024], kv_lens=[1024], hq=32, hkv=8, d=192, bs=64),
        "use_transposed_mask_limit",
    ),
    "d256 cohort": (
        dict(q_lens=[1024, 512], kv_lens=[1024, 600], hq=16, hkv=2, d=256),
        "use_mask_phase_split",
    ),
    "register-P": (
        dict(q_lens=[64, 48], kv_lens=[200, 48], hq=8, hkv=8, d=64, bs=32),
        "use_register_pv",
    ),
}


def _gfx950_only():
    if _ARCH != "gfx950":
        pytest.skip("the ALiBi design arms are on the gfx950 tiled 2D kernel")


@pytest.mark.parametrize("name", list(_GFX950_FAMILIES))
def test_gfx950_family_default_design(name):
    _gfx950_only()
    kw, family = _GFX950_FAMILIES[name]
    _run_case(backend="tiled", family=family, **kw)


@pytest.mark.parametrize("design", sorted(ALIBI_DESIGNS))
@pytest.mark.parametrize("name", ["d128 multi batch", "d256 cohort", "register-P"])
def test_gfx950_designs(name, design):
    _gfx950_only()
    kw, family = _GFX950_FAMILIES[name]
    _run_case(backend="tiled", family=family, alibi_design=design, **kw)


@pytest.mark.parametrize(
    "case,design",
    [
        ("combo", None),
        ("combo", "seed_split"),
        ("combo", "fma_split"),
        ("per-score add", None),
    ],
)
def test_sinks_on_the_top_left_diagonal(case, design):
    _gfx950_only()
    if case == "combo":
        kw, family = dict(_GFX950_FAMILIES["combo"][0]), "use_transposed_mask_limit"
        kw.update(q_lens=[320], kv_lens=[512])
    else:
        kw, family = dict(q_lens=[96, 64], kv_lens=[300, 64]), _PER_SCORE_ADD
    _run_case(
        backend="tiled",
        top_left=True,
        sinks=True,
        family=family,
        alibi_design=design,
        **kw,
    )


def test_sinks_on_the_top_left_diagonal_split_kv():
    _gfx950_only()
    _run_case([4, 2], [3000, 1029], backend="3d", top_left=True, sinks=True)


def test_sinks_off_the_top_left_diagonal_are_refused():
    with pytest.raises((ValueError, NotImplementedError), match="sinks"):
        _run_case([96], [300], backend="tiled", sinks=True)
