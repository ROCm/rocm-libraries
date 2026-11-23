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
from kernels.gfx942.attention_tiled_2d import (  # noqa: E402
    _alibi_design as _alibi_design_gfx942,
)
from kernels.gfx950.attention_tiled_2d import (  # noqa: E402
    _alibi_design as _alibi_design_gfx950,
)
from rocke.numeric.sdpa_reference import alibi_slopes  # noqa: E402


def _arch():
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_properties(0).gcnArchName.lower()
    return next((a for a in ("gfx942", "gfx950") if a in name), None)


_ARCH = _arch()
_alibi_design = {"gfx942": _alibi_design_gfx942, "gfx950": _alibi_design_gfx950}.get(
    _ARCH
)
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(_ARCH is None, reason="needs a gfx942 or gfx950 GPU"),
]


def _tolerance(dtype):
    return 2e-2 if dtype == "fp16" else 4e-2


def _reference(
    q,
    k_cache,
    v_cache,
    q_lens,
    kv_lens,
    block_table,
    scale,
    slopes,
    tl,
    sinks=None,
    window=0,
):
    """FP32 O and natural-log LSE; causal on the bottom-right diagonal, or the
    top-left one when ``tl``, limited to the last ``window`` keys when set; ALiBi
    measured from the query's diagonal. A sink logit joins each row's
    denominator with no ALiBi term and no value (SEM-022)."""
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
        visible = kk <= diag
        if window:
            visible = visible & (diag - kk < window)
        s = s.masked_fill(~visible, float("-inf"))
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


def _inputs(q_lens, kv_lens, hq, hkv, d, bs, dtype, external, seed):
    """Random paged Q/K/V with a shuffled block table, the slopes the reference
    uses and the bottom-right ALiBi problem (built-in slopes unless
    ``external``)."""
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
        right_bound=0,
    )
    return g, tdt, q, kc, vc, bt, cu, seqused, slopes, problem


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
    window=0,
    off_kernel=False,
    seed=0,
):
    """``family``: a 2D spec flag the dispatched kernel must set (or
    ``_PER_SCORE_ADD``), so the case runs the path it names; ``alibi_design``
    forces that arm on it. ``off_kernel``: run ALiBi on the kernel the problem
    without ALiBi dispatches to (a path dispatch keeps ALiBi off)."""
    g, tdt, q, kc, vc, bt, cu, seqused, slopes, problem = _inputs(
        q_lens, kv_lens, hq, hkv, d, bs, dtype, external, seed
    )
    problem = replace(
        problem,
        use_sinks=sinks,
        causal_top_left=top_left,
        emit_lse=lse,
        sliding_window=window,
    )
    sink_t = torch.randn(hq, device="cuda", dtype=tdt, generator=g) if sinks else None
    tuning_spec = None
    if family is not None or alibi_design is not None or off_kernel:
        assert au.select_auto_path(problem, _ARCH) == "2d"
        if off_kernel:
            spec = replace(
                au._tiled_spec_from_problem(replace(problem, use_alibi=False), _ARCH),
                use_alibi=True,
            )
        else:
            spec = au._tiled_spec_from_problem(problem, _ARCH)
        if family == _PER_SCORE_ADD:
            assert _alibi_design(spec) is None, spec.kernel_name()
        elif family == _4WARP:
            assert au._gfx942_4warp_fast(problem, _ARCH), spec.kernel_name()
        elif family is not None:
            assert getattr(spec, family), f"{spec.kernel_name()} is not on {family}"
        if alibi_design is not None or off_kernel:
            tuning_spec = AttentionTuningSpec(
                path="2d",
                arch=_ARCH,
                builder_kind="tiled",
                compile_backend=au._select_2d_compile_backend(problem, _ARCH),
                candidate_name="alibi_design",
                tuning_id=alibi_design,
                kernel_spec=(
                    spec
                    if alibi_design is None
                    else replace(spec, alibi_design=alibi_design)
                ),
                num_kv_blocks=problem.num_kv_blocks,
            )
    out = torch.empty_like(q)
    lse_t = (
        torch.full((problem.total_q, hq, 1), float("nan"), device="cuda")
        if lse
        else None
    )
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
        q, kc, vc, q_lens, kv_lens, bt, scale, slopes, top_left, sink_t, window
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
# The gfx942 4-warp GQA kernels (a dispatch route, not a spec flag).
_4WARP = "4-warp"

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


# gfx942 tiled 2D families ALiBi reaches through dispatch. Bottom-right causal, the
# only band the gfx942 kernels have; the reference anchors on the same diagonal.
_GFX942_FAMILIES = {
    "x8 single buffer": (
        dict(q_lens=[1024, 512], kv_lens=[1024, 700], hq=32, hkv=8, bs=64),
        "use_transposed_mask_limit",
    ),
    "x8 ring": (
        dict(q_lens=[1024], kv_lens=[1024], hq=32, hkv=8, bs=64, dtype="fp16"),
        "use_transposed_mask_limit",
    ),
    "x8 q direct": (
        dict(q_lens=[2048], kv_lens=[2048], hq=64, hkv=8, d=64, bs=32),
        "use_transposed_mask_limit",
    ),
    # Block size 64 keeps D192 off the 4-warp cohort.
    "x8 d192": (
        dict(q_lens=[1024], kv_lens=[1024], hq=32, hkv=8, d=192, bs=64),
        "use_mfma_32x32x8",
    ),
    "4-warp d256": (
        dict(q_lens=[1024, 512], kv_lens=[1024, 600], hq=16, hkv=2, d=256),
        _4WARP,
    ),
    "4-warp d192": (
        dict(q_lens=[700], kv_lens=[900], hq=32, hkv=8, d=192, bs=32),
        _4WARP,
    ),
    "4-warp window head fold": (
        dict(q_lens=[600, 300], kv_lens=[1500, 300], hq=32, hkv=8, window=512),
        _4WARP,
    ),
    "4-warp window": (
        dict(q_lens=[900], kv_lens=[1300], hq=32, hkv=4, window=256),
        _4WARP,
    ),
    "4-warp window bs64": (
        dict(q_lens=[900], kv_lens=[1300], hq=32, hkv=8, bs=64, window=256),
        _4WARP,
    ),
    "16x16 short": (
        dict(q_lens=[512], kv_lens=[512], hq=32, hkv=8, bs=64, dtype="fp16"),
        _PER_SCORE_ADD,
    ),
}


def _gfx942_only():
    if _ARCH != "gfx942":
        pytest.skip("gfx942 tiled 2D families")


@pytest.mark.parametrize("name", list(_GFX942_FAMILIES))
def test_gfx942_family_default_design(name):
    _gfx942_only()
    kw, family = _GFX942_FAMILIES[name]
    _run_case(backend="tiled", family=family, **kw)


@pytest.mark.parametrize("design", sorted(ALIBI_DESIGNS))
@pytest.mark.parametrize("name", ["x8 single buffer", "x8 d192", "16x16 short"])
def test_gfx942_designs(name, design):
    _gfx942_only()
    kw, family = _GFX942_FAMILIES[name]
    _run_case(backend="tiled", family=family, alibi_design=design, **kw)


@pytest.mark.parametrize("design", [None, "seed_split"])
def test_gfx942_register_p(design):
    # Dispatch keeps ALiBi off register-P on gfx942 (the 16x16 kernel without
    # it is faster with ALiBi); the kernel applies it all the same.
    _gfx942_only()
    _run_case(
        [512, 200],
        [512, 300],
        backend="tiled",
        hq=32,
        hkv=8,
        bs=64,
        off_kernel=True,
        family="use_register_pv",
        alibi_design=design,
    )


def test_sinks_on_the_top_left_diagonal_split_kv():
    _gfx950_only()
    _run_case([4, 2], [3000, 1029], backend="3d", top_left=True, sinks=True)


def test_sinks_off_the_top_left_diagonal_are_refused():
    with pytest.raises((ValueError, NotImplementedError), match="sinks"):
        _run_case([96], [300], backend="tiled", sinks=True)


# Shapes inside the launcher's internal graph replay: short prefill (2D) and
# decode (3D).
_GRAPH_CASES = {
    "2d short prefill": (
        "tiled",
        [512, 200],
        [512, 300],
        dict(hq=32, hkv=8, bs=64, dtype="fp16"),
    ),
    "3d decode": ("3d", [1, 1, 1, 1], [4096, 777, 1500, 2049], dict(hq=12, hkv=4)),
}


@pytest.fixture
def graph_replay(monkeypatch):
    """Enables the launcher's graph replay on this arch and counts the graphs
    it captures (new keys in its graph caches) and the calls it serves."""
    if _ARCH == "gfx942":
        monkeypatch.delenv("HIPDNN_GFX942_2D_GRAPH", raising=False)
        monkeypatch.delenv("HIPDNN_GFX942_3D_GRAPH", raising=False)
    else:
        monkeypatch.setenv("HIPDNN_GFX950_3D_GRAPH", "1")
    # Fresh caches: the process-wide ones are bounded and shared with other
    # tests, so an eviction there would look like a recapture.
    monkeypatch.setattr(au, "_2D_GRAPHS", {})
    monkeypatch.setattr(au, "_2D_GRAPH_REFS", {})
    monkeypatch.setattr(au, "_3D_GRAPHS", type(au._3D_GRAPHS)())
    calls = {"tiled": 0, "3d": 0}
    for name, backend in (("_run_2d_graphed", "tiled"), ("_run_3d_graph", "3d")):
        real = getattr(au, name)

        def spy(*args, real=real, backend=backend, **kwargs):
            calls[backend] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(au, name, spy)
    caches = {"tiled": au._2D_GRAPHS, "3d": au._3D_GRAPHS}
    state = {"calls": calls, "keys": {b: set(c) for b, c in caches.items()}}

    def captured(backend):
        """Graphs captured since the last call."""
        keys = set(caches[backend])
        new = keys - state["keys"][backend]
        state["keys"][backend] = keys
        return len(new)

    state["captured"] = captured
    yield state
    torch.cuda.synchronize()
    au.clear_attention_3d_graph_cache()  # drains and releases the fresh cache


def _graph_case(name):
    backend, q_lens, kv_lens, kw = _GRAPH_CASES[name]
    if backend == "tiled" and _ARCH != "gfx942":
        pytest.skip("the 2D graph replay is gfx942-only")
    return backend, q_lens, kv_lens, kw


def _graph_run(problem, backend, q, kc, vc, bt, cu, seqused, slopes, external, out):
    # The graph key holds every tensor's identity: one `out` across the calls.
    out.fill_(float("nan"))
    run_unified_attention_torch(
        problem=problem,
        q=q,
        k=kc,
        v=vc,
        out=out,
        cu_seqlens_q=cu,
        seqused_k=seqused,
        softmax_scale=1.0 / math.sqrt(problem.head_size),
        block_table=bt,
        softcap=0.0,
        backend=backend,
        alibi_slopes=slopes if external else None,
    )
    torch.cuda.synchronize()
    return out


@pytest.mark.parametrize("external", [False, True], ids=["builtin", "external"])
@pytest.mark.parametrize("name", list(_GRAPH_CASES))
def test_graph_replay_follows_rewritten_inputs(graph_replay, name, external):
    # A replay reads the captured buffers: Q (and caller slopes) rewritten in
    # place must reach the kernel without a new capture.
    backend, q_lens, kv_lens, kw = _graph_case(name)
    hq, hkv, d, bs = kw["hq"], kw["hkv"], kw.get("d", 128), kw.get("bs", 16)
    dtype = kw.get("dtype", "bf16")
    g, _, q, kc, vc, bt, cu, seqused, slopes, problem = _inputs(
        q_lens, kv_lens, hq, hkv, d, bs, dtype, external, seed=1
    )
    tol, scale = _tolerance(dtype), 1.0 / math.sqrt(d)
    out = torch.empty_like(q)
    for step in range(3):
        if step:
            q.copy_(torch.randn(q.shape, device="cuda", dtype=q.dtype, generator=g))
            if external:
                slopes.mul_(0.5)
        out = _graph_run(
            problem, backend, q, kc, vc, bt, cu, seqused, slopes, external, out
        )
        o_ref, _ = _reference(q, kc, vc, q_lens, kv_lens, bt, scale, slopes, False)
        assert (out.float() - o_ref).abs().max().item() < tol, f"call {step}"
        assert graph_replay["captured"](backend) == (0 if step else 1), f"call {step}"
    assert graph_replay["calls"][backend] == 3


@pytest.mark.parametrize("name", list(_GRAPH_CASES))
def test_graph_replay_keeps_alibi_on_and_off_apart(graph_replay, name):
    # ALiBi on and off on the same buffers are two graphs; neither replay
    # serves the other.
    backend, q_lens, kv_lens, kw = _graph_case(name)
    hq, hkv, d, bs = kw["hq"], kw["hkv"], kw.get("d", 128), kw.get("bs", 16)
    dtype = kw.get("dtype", "bf16")
    _, _, q, kc, vc, bt, cu, seqused, slopes, on = _inputs(
        q_lens, kv_lens, hq, hkv, d, bs, dtype, False, seed=2
    )
    off = replace(on, use_alibi=False)
    scale, tol = 1.0 / math.sqrt(d), _tolerance(dtype)
    refs = {
        True: _reference(q, kc, vc, q_lens, kv_lens, bt, scale, slopes, False)[0],
        False: _reference(
            q, kc, vc, q_lens, kv_lens, bt, scale, torch.zeros_like(slopes), False
        )[0],
    }
    out = torch.empty_like(q)
    for step, alibi in enumerate((True, False, True, False)):
        problem = on if alibi else off
        out = _graph_run(
            problem, backend, q, kc, vc, bt, cu, seqused, slopes, False, out
        )
        err = (out.float() - refs[alibi]).abs().max().item()
        assert err < tol, f"call {step} (alibi={alibi})"
        assert graph_replay["captured"](backend) == (1 if step < 2 else 0)
    assert graph_replay["calls"][backend] == 4
