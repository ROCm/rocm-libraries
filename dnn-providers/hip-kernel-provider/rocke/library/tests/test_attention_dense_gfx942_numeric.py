# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU numeric lane for the gfx942 dense flash-attn kernel.

Drives the PUBLIC entry point ``run_attention_dense_torch`` end-to-end on a real
gfx942 GPU and checks max_abs against an fp32 ``scaled_dot_product_attention``
oracle, for both the default and the P4 persistent grid. This is the committed,
CI-collectable form of the acceptance criterion's "functional GPU-numeric bench,
both variants, vs torch fp32 SDPA" -- previously only an out-of-tree verifier.

Specs come from the gfx942 DISPATCH factory, never hand-rolled (see :func:`_spec`),
so every row compiles the binary that actually ships rather than one that differs
from it by an untracked tuning default.

Every test is marked ``gpu`` and gated with a device skipif, so it is a graceful
skip on a CPU CI box and only executes on a gfx942 (MI300X) ROCm runner. Select
it with ``run_all.py --gpu`` (or ``pytest -m gpu``); the default CPU lane excludes
it via ``-m "not gpu"``. Run standalone:

    HIP_VISIBLE_DEVICES=0 python -m pytest tests/test_attention_dense_gfx942_numeric.py
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from kernels.gfx942.attention_dense import (
    _as_gfx942_spec,
    run_attention_dense_torch,
)


torch = pytest.importorskip("torch", reason="ROCm torch required")


def _gpu_ready():
    """True only on a gfx942 box with ROCm torch. Gate on ``gcnArchName`` (the ISA
    target), NOT the marketing name: the whole MI300 family is gfx942, but the
    marketing string varies (``MI300X``/``MI300A``/``MI308X``) and a substring check
    for ``"mi300"`` silently MISSES ``MI308X`` -- the exact skip that hid this lane on
    the first run. The arch string is stable across the family."""
    if not torch.cuda.is_available():
        return False
    arch = torch.cuda.get_device_properties(0).gcnArchName.lower()
    return "gfx942" in arch


requires_gfx942_gpu = pytest.mark.skipif(
    not _gpu_ready(), reason="needs a gfx942 (MI300X) GPU with ROCm torch"
)

_TORCH_DT = {"fp16": "float16", "bf16": "bfloat16"}

# (dtype, head_size, num_query_heads, num_kv_heads, persistent, causal) -- a compact
# cohort spanning both dtypes, D64/D128, GQA + MHA, causal + non-causal, and both grid
# variants. Sq is fixed at 512 (a 256 multiple so the persistent grid-stride has >1
# q-block of work). The last two rows cover the fp16-D128 swizzle default grid on the
# paths dispatch actually ships but the base rows miss: MHA, and non-causal.
_COHORT = [
    ("fp16", 128, 16, 4, False, True),  # flagship default (causal)
    ("fp16", 128, 16, 4, True, True),  # flagship persistent
    ("bf16", 128, 16, 4, True, True),  # bf16 D128 persistent (the VGPR-starved config)
    # bf16 D128 default: the plain-exp2 arm -- the one config _use_exp2_fast turns
    # off, and it does so on the grid, at every seqlen.
    ("bf16", 128, 16, 4, False, True),
    ("fp16", 64, 16, 16, False, True),  # D64 MHA default
    ("bf16", 64, 16, 4, True, True),  # D64 bf16 persistent (the wpe=4 config)
    ("fp16", 128, 16, 16, False, True),  # fp16 D128 MHA default -- swizzle path, MHA
    ("fp16", 128, 16, 4, False, False),  # fp16 D128 non-causal default -- swizzle path
]

# (dtype, head_size, num_query_heads, num_kv_heads, persistent, sliding_window, scale)
# -- STANDALONE sliding-window (no sinks; gfx942 dense has no sink support yet). All
# rows are causal (sliding_window > 0 requires causal). Window is a multiple of the
# shipped block_n (64): 128, 256. Covers both dtypes, D64/D128, and BOTH grid
# variants -- the persistent rows exercise the per-work-item start_tile prune that the
# default grid does not. ``scale`` None is the default 1/sqrt(D); the two scale-1.0
# rows run the window mask on fp32 scores scaled after the QK MFMA, one per grid.
_SWA_COHORT = [
    ("bf16", 128, 16, 4, False, 128, None),
    ("bf16", 128, 16, 4, True, 128, None),
    ("fp16", 128, 16, 4, False, 256, None),
    ("fp16", 128, 16, 4, True, 256, None),
    ("bf16", 64, 16, 4, False, 128, None),
    ("bf16", 64, 16, 4, True, 128, None),
    ("bf16", 128, 16, 4, False, 128, 1.0),
    ("bf16", 64, 16, 4, True, 128, 1.0),
]

# (head_size, dtype, persistent, scale) for test_dense_non_default_scale: every D128
# combination, plus bf16 D64 at the larger scale on both grids. D64 has its own K
# layout and, for bf16, its own waves-per-eu; the scaling line is shared.
_NON_DEFAULT_SCALE_ROWS = [
    (128, dtype, persistent, scale)
    for scale in (0.5, 1.0)
    for dtype in ("bf16", "fp16")
    for persistent in (False, True)
] + [(64, "bf16", False, 1.0), (64, "bf16", True, 1.0)]


def _spec(
    dtype,
    d,
    hq,
    hkv,
    persistent,
    *,
    causal=True,
    batch=1,
    sq=512,
    sliding_window=0,
    persist_decode="auto",
):
    """The SHIPPED gfx942 dense spec for a cohort row, built through the dispatch
    candidate (``gfx942_dense`` via ``dispatch.attention``) rather than hand-rolled.

    Hand-rolling the spec silently pins every tuned lever to the shared (gfx950)
    dataclass default, so the lane would assert on configs that do not ship. The
    concrete one this cohort hit: ``waves_per_eu``. Dispatch resolves it from the
    kernel's own policy (``_tuned_waves_per_eu``), which returns 4 for bf16/D64 --
    the row ``_COHORT`` above labels "the wpe=4 config" -- while the dataclass default
    is 2. That is not a cosmetic difference: ``waves_per_eu`` is emitted as the
    ``amdgpu-waves-per-eu`` attribute, changes register allocation, and is tagged into
    ``gfx942_kernel_name`` as ``wpe{N}``, so wpe2 and wpe4 are DIFFERENT binaries.
    ``num_persistent`` was likewise hard-coded to 304 beside a dispatch constant that
    already resolves to 304 -- left at the candidate default here so dispatch
    supplies the gfx942 CU count itself and the two cannot drift apart.

    Deriving the spec from the factory (the pattern
    ``test_attention_dense_gfx942_golden.py::mk_dispatch`` uses for its D64 cases)
    also means a future gfx942 tuning change is picked up here with no edit.

    Only the ``persistent`` knob is set rather than left at the default: the cohort
    asserts BOTH grid variants at one fixed Sq, where the default picks one. Every
    other lever -- block_n, the D64 K row-group pad, ragged -- is whatever the
    shipped path folds in. ``persist_decode`` stays "auto" unless a test pins one
    order to hold it fixed across shapes "auto" would resolve differently; a pinned
    order is set on the spec, since the knob refuses one that restates "auto".
    """
    # Imported lazily, mirroring the golden sibling: keeps module import (and hence
    # CPU collection of this gpu-marked file) independent of the dispatch package.
    from dispatch.attention import AttentionRequest, tuning_spec_with_knobs

    spec = tuning_spec_with_knobs(
        AttentionRequest(
            batch=batch,
            nhead_q=hq,
            nhead_k=hkv,
            seqlen_q=sq,
            seqlen_k=sq,
            hdim_q=d,
            hdim_v=d,
            arch="gfx942",
            mask_type=1 if causal else 0,
            dtype=dtype,
            sliding_window=sliding_window,
        ),
        "gfx942_dense",
        {"persistent": bool(persistent)},
    ).kernel_spec
    if persist_decode == "auto":
        return spec
    return dataclasses.replace(spec, persist_decode=persist_decode)


def _sdpa_reference(q, k, v, scale, *, causal=True, attn_mask=None):
    """fp32 SDPA oracle for ``[B,S,H,D]`` tensors, returned as ``[B,S,Hq,D]``.

    GQA is expanded HERE, by repeating each kv head to its query heads, rather than
    via the ``enable_gqa=`` kwarg: that kwarg is a recent addition to
    ``scaled_dot_product_attention``, and on an older ROCm torch passing it raises
    TypeError -- which ERRORS the whole gpu cohort instead of leaving it to the
    device gate. ``repeat_interleave`` along the head axis is exactly what
    ``enable_gqa`` does internally, and it is the mapping the kernel itself uses
    (hkv = hq // gqa). rep == 1 (MHA) makes it a plain copy. Pass ``attn_mask``
    (a boolean keep-mask) instead of ``causal`` for masks SDPA has no flag for.
    """
    import torch.nn.functional as F

    rep = q.shape[2] // k.shape[2]
    qf = q.transpose(1, 2).float()
    kf = k.transpose(1, 2).float().repeat_interleave(rep, dim=1)
    vf = v.transpose(1, 2).float().repeat_interleave(rep, dim=1)
    return F.scaled_dot_product_attention(
        qf,
        kf,
        vf,
        attn_mask=attn_mask,
        is_causal=causal and attn_mask is None,
        scale=scale,
    ).transpose(1, 2)


@requires_gfx942_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("dtype,d,hq,hkv,persistent,causal", _COHORT)
def test_dense_numeric_vs_fp32_sdpa(dtype, d, hq, hkv, persistent, causal):
    import torch

    tol = 2e-2 if dtype == "fp16" else 4e-2
    tdt = getattr(torch, _TORCH_DT[dtype])
    B, S = 1, 512
    scale = 1.0 / math.sqrt(d)
    torch.manual_seed(0)

    # run_attention_dense_torch ABI: q/out [B,S,Hq,D], k/v [B,S,Hkv,D], dense.
    q = torch.randn(B, S, hq, d, device="cuda", dtype=tdt)
    k = torch.randn(B, S, hkv, d, device="cuda", dtype=tdt)
    v = torch.randn(B, S, hkv, d, device="cuda", dtype=tdt)
    out = torch.empty(B, S, hq, d, device="cuda", dtype=tdt)

    spec = _spec(dtype, d, hq, hkv, persistent, causal=causal, batch=B, sq=S)
    run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)
    torch.cuda.synchronize()

    ref = _sdpa_reference(q, k, v, scale, causal=causal)

    max_abs = (ref - out.float()).abs().max().item()
    assert max_abs < tol, (
        f"{dtype} D{d} GQA{hq}/{hkv} {'causal' if causal else 'full'} "
        f"{'persist' if persistent else 'default'}: max_abs={max_abs:.3e} >= {tol}"
    )


@requires_gfx942_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("d,dtype,persistent,scale", _NON_DEFAULT_SCALE_ROWS)
def test_dense_non_default_scale(d, dtype, persistent, scale):
    """Softmax scales well above the default 1/sqrt(D).

    The score error a lossy scale step introduces grows with ``scale``, so the
    default-scale cohort above cannot see it. Rounding ``Q * scale * log2(e)``
    back to bf16/fp16 before the QK MFMA puts bf16 past its tolerance on both grids:
    D128 at both scales and D64 at 1.0. The kernel must apply the scale to the fp32
    scores. fp16 has three more mantissa bits and stays inside its tolerance either
    way, so the fp16 rows guard against regressions rather than catch this defect.
    The sliding-window paths are covered by the scale-1.0 rows of ``_SWA_COHORT``.
    Not covered: scales other than 0.5 and 1.0.
    """
    import torch

    hq, hkv = 16, 4
    tol = 2e-2 if dtype == "fp16" else 4e-2
    tdt = getattr(torch, _TORCH_DT[dtype])
    B, S = 1, 512
    torch.manual_seed(0)

    q = torch.randn(B, S, hq, d, device="cuda", dtype=tdt)
    k = torch.randn(B, S, hkv, d, device="cuda", dtype=tdt)
    v = torch.randn(B, S, hkv, d, device="cuda", dtype=tdt)
    out = torch.empty(B, S, hq, d, device="cuda", dtype=tdt)

    spec = _spec(dtype, d, hq, hkv, persistent, batch=B, sq=S)
    run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)
    torch.cuda.synchronize()

    max_abs = (_sdpa_reference(q, k, v, scale) - out.float()).abs().max().item()
    assert max_abs < tol, (
        f"{dtype} D{d} GQA16/4 scale={scale:g} "
        f"{'persist' if persistent else 'default'}: max_abs={max_abs:.3e} >= {tol}"
    )


@requires_gfx942_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("persistent", [False, True])
def test_dense_fp16_d128_large_equal_scores(persistent):
    """Every score in a row is equal and large (raw 128 * 427**2 = 23338112, at
    scale 16), so the softmax is uniform and the output is the causal running
    mean of V. Scaling the fp32 scores keeps the row max's exp2 argument exactly
    0 at any magnitude; it must stay finite and correct.

    A regression guard, not a reproducer: the pre-fix gfx942 kernel also passes
    it. It fails a kernel that takes the row max on unscaled scores and folds the
    scale into the exp2 argument (fp16 P overflows there), which is the form the
    gfx950 ordinary grid uses. Only fp16, D128 and this one magnitude are covered.
    """
    import torch

    d, hq, hkv, scale = 128, 16, 4, 16.0
    B, S = 1, 512
    torch.manual_seed(0)

    q = torch.full((B, S, hq, d), 427.0, device="cuda", dtype=torch.float16)
    k = torch.full((B, S, hkv, d), 427.0, device="cuda", dtype=torch.float16)
    v = torch.randn(B, S, hkv, d, device="cuda", dtype=torch.float16)
    out = torch.empty(B, S, hq, d, device="cuda", dtype=torch.float16)

    spec = _spec("fp16", d, hq, hkv, persistent, batch=B, sq=S)
    run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)
    torch.cuda.synchronize()

    assert torch.isfinite(out).all(), "non-finite output"
    max_abs = (_sdpa_reference(q, k, v, scale) - out.float()).abs().max().item()
    assert max_abs < 2e-2, (
        f"fp16 D128 equal scores scale=16 "
        f"{'persist' if persistent else 'default'}: max_abs={max_abs:.3e}"
    )


def _launcher_for(spec):
    """The cached ``KernelLauncher`` for ``spec``, or None if none is compiled.

    ``run_attention_dense_torch`` owns ``_DENSE_LAUNCHER_CACHE`` internally and
    exposes no accessor, so this reads the module-level dict through the very key
    function the production path uses -- a test-local reimplementation of the key
    would assert against itself rather than against the shipped one.
    """
    from kernels.common.attention_dense_spec import attention_dense_cache_key
    from kernels.gfx942.attention_dense import _DENSE_LAUNCHER_CACHE

    return _DENSE_LAUNCHER_CACHE.get(attention_dense_cache_key(spec, arch="gfx942"))


@requires_gfx942_gpu
@pytest.mark.gpu
def test_one_binary_serves_every_shape():
    """One compiled artifact, two shapes, correct numerics at both.

    The cohort above runs many shapes, but it stopped discriminating the moment
    batch/seqlen_q/seqlen_kv became runtime kernel params: it passes identically
    whether N shapes are served by N binaries or by one. The property that
    actually needs a guard now is *artifact reuse*.

    ``is`` on the launcher covers the whole path rather than just the key
    function. A key regression would land the two shapes in different cache
    slots; a lookup regression would overwrite the one slot with a freshly
    compiled launcher. Both yield a different object, and neither is visible to a
    numeric assertion -- recompiling per shape is *correct*, merely wasteful, so
    what regresses is the AOT instance count and first-call latency, which no
    accuracy check can see.

    Paired with the numeric check at both shapes so the test cannot pass by
    reusing one binary that happens to be wrong for the second shape.

    fp16 is the flagship default config; bf16 D128 would serve equally well now
    that no dtype forks the body on shape, but this is the path dispatch ships
    most of.
    """
    import torch

    from kernels.common.attention_dense_spec import attention_dense_cache_key
    from kernels.gfx942.attention_dense import _DENSE_LAUNCHER_CACHE

    dtype, d, hq, hkv = "fp16", 128, 16, 4  # flagship default, non-persistent
    tol = 2e-2
    tdt = getattr(torch, _TORCH_DT[dtype])
    scale = 1.0 / math.sqrt(d)

    shapes = ((1, 512), (4, 1024))
    specs = [
        _as_gfx942_spec(_spec(dtype, d, hq, hkv, False, batch=b, sq=s))
        for b, s in shapes
    ]

    # Preconditions: genuinely different shapes, on the runtime path, and sharing
    # one key -- otherwise the reuse assertion below is vacuous.
    assert specs[0].runtime_shape, "cohort row is not on the runtime-shape path"
    assert (specs[0].batch, specs[0].seqlen_q) != (specs[1].batch, specs[1].seqlen_q)
    keys = [attention_dense_cache_key(s, arch="gfx942") for s in specs]
    assert keys[0] == keys[1], f"shapes {shapes} did not share a cache key"

    # Own the cache state: evicting first makes the "exactly one new entry"
    # assertion independent of which tests ran before this one.
    _DENSE_LAUNCHER_CACHE.pop(keys[0], None)
    before = set(_DENSE_LAUNCHER_CACHE)

    launchers = []
    for (B, S), spec in zip(shapes, specs):
        torch.manual_seed(0)
        q = torch.randn(B, S, hq, d, device="cuda", dtype=tdt)
        k = torch.randn(B, S, hkv, d, device="cuda", dtype=tdt)
        v = torch.randn(B, S, hkv, d, device="cuda", dtype=tdt)
        out = torch.empty(B, S, hq, d, device="cuda", dtype=tdt)

        run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)
        torch.cuda.synchronize()
        launchers.append(_launcher_for(spec))

        ref = _sdpa_reference(q, k, v, scale)
        max_abs = (ref - out.float()).abs().max().item()
        assert max_abs < tol, f"B={B} S={S}: max_abs={max_abs:.3e} >= {tol}"

    assert launchers[0] is not None, (
        "no launcher cached after a successful run; _DENSE_LAUNCHER_CACHE is no "
        "longer keyed by attention_dense_cache_key and this test is blind"
    )
    assert launchers[0] is launchers[1], (
        f"shapes {shapes} share a cache key but were served by DIFFERENT launcher "
        "objects -- the runtime-shape kernel recompiled per shape, so the AOT "
        "instance count still scales with the shape space"
    )
    assert set(_DENSE_LAUNCHER_CACHE) - before == {keys[0]}, (
        "two shapes on the runtime path added more than one cache entry: "
        f"{sorted(set(_DENSE_LAUNCHER_CACHE) - before)}"
    )


@requires_gfx942_gpu
@pytest.mark.gpu
def test_one_binary_serves_every_head_config():
    """One compiled artifact, four head configs, correct numerics at each.

    The head-count twin of :func:`test_one_binary_serves_every_shape`: on the
    runtime-shape path ``num_query_heads`` / ``num_kv_heads`` are kernel params,
    so the GQA ratio and the token strides are computed at run time. A body that
    still baked one of them would compute wrong strides for every config but the
    first one compiled -- caught here by the numeric check, while the ``is`` /
    single-entry assertions catch a key or lookup regression that recompiles per
    head config. Covers GQA 16, non-pow2 GQA 5 and 7, and MHA.
    """
    import torch
    import torch.nn.functional as F

    from kernels.common.attention_dense_spec import attention_dense_cache_key
    from kernels.gfx942.attention_dense import _DENSE_LAUNCHER_CACHE

    dtype, d, B, S = "fp16", 128, 1, 512  # flagship default, non-persistent
    tol = 2e-2
    tdt = getattr(torch, _TORCH_DT[dtype])
    scale = 1.0 / math.sqrt(d)

    heads = ((128, 8), (40, 8), (28, 4), (32, 32))
    specs = [
        _as_gfx942_spec(_spec(dtype, d, hq, hkv, False, batch=B, sq=S))
        for hq, hkv in heads
    ]

    # Preconditions: all on the runtime path and sharing one key -- otherwise the
    # reuse assertion below is vacuous.
    assert all(s.runtime_shape for s in specs), "a head config left the runtime path"
    keys = [attention_dense_cache_key(s, arch="gfx942") for s in specs]
    assert len(set(keys)) == 1, f"head configs {heads} did not share a cache key"

    _DENSE_LAUNCHER_CACHE.pop(keys[0], None)
    before = set(_DENSE_LAUNCHER_CACHE)

    launchers = []
    for (hq, hkv), spec in zip(heads, specs):
        torch.manual_seed(0)
        q = torch.randn(B, S, hq, d, device="cuda", dtype=tdt)
        k = torch.randn(B, S, hkv, d, device="cuda", dtype=tdt)
        v = torch.randn(B, S, hkv, d, device="cuda", dtype=tdt)
        out = torch.empty(B, S, hq, d, device="cuda", dtype=tdt)

        run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)
        torch.cuda.synchronize()
        launchers.append(_launcher_for(spec))

        rep = hq // hkv
        ref = F.scaled_dot_product_attention(
            q.transpose(1, 2).float(),
            k.transpose(1, 2).float().repeat_interleave(rep, dim=1),
            v.transpose(1, 2).float().repeat_interleave(rep, dim=1),
            is_causal=True,
            scale=scale,
        ).transpose(1, 2)
        max_abs = (ref - out.float()).abs().max().item()
        assert max_abs < tol, f"Hq={hq} Hkv={hkv}: max_abs={max_abs:.3e} >= {tol}"

    assert launchers[0] is not None, (
        "no launcher cached after a successful run; _DENSE_LAUNCHER_CACHE is no "
        "longer keyed by attention_dense_cache_key and this test is blind"
    )
    assert all(lau is launchers[0] for lau in launchers), (
        f"head configs {heads} share a cache key but were served by DIFFERENT "
        "launcher objects -- the runtime-heads kernel recompiled per head config"
    )
    assert set(_DENSE_LAUNCHER_CACHE) - before == {keys[0]}, (
        "four head configs on the runtime path added more than one cache entry: "
        f"{sorted(set(_DENSE_LAUNCHER_CACHE) - before)}"
    )


@requires_gfx942_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("dtype,d,hq,hkv,persistent,sliding_window,scale", _SWA_COHORT)
def test_dense_swa_numeric_vs_fp32_sdpa(
    dtype, d, hq, hkv, persistent, sliding_window, scale
):
    """Sliding-window (SWA) numeric parity, standalone (no sinks), both grids.

    The band is the same one the gfx950 sibling masks (``_sink_reference``: causal
    ``ki > qi`` plus window ``ki <= qi - W`` masked out) and the kernel applies
    (``cmp_le(ktok, q)`` + ``cmp_gt(ktok, q - SW)``): keep key k for query q iff
    ``q - W < k <= q``. Expressed here as a boolean SDPA attn_mask -- matching this
    file's SDPA-based base oracle -- rather than gfx950's manual masked-softmax,
    which exists only because gfx950 also concatenates a sink column (gfx942 has no
    sinks yet). The diagonal k==q is always kept (W>0), so no row is fully masked."""
    import torch

    tol = 2e-2 if dtype == "fp16" else 4e-2
    tdt = getattr(torch, _TORCH_DT[dtype])
    B, S = 1, 512
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    torch.manual_seed(0)

    q = torch.randn(B, S, hq, d, device="cuda", dtype=tdt)
    k = torch.randn(B, S, hkv, d, device="cuda", dtype=tdt)
    v = torch.randn(B, S, hkv, d, device="cuda", dtype=tdt)
    out = torch.empty(B, S, hq, d, device="cuda", dtype=tdt)

    spec = _spec(
        dtype,
        d,
        hq,
        hkv,
        persistent,
        causal=True,
        batch=B,
        sq=S,
        sliding_window=sliding_window,
    )
    run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)
    torch.cuda.synchronize()

    qi = torch.arange(S, device="cuda").view(-1, 1)
    ki = torch.arange(S, device="cuda").view(1, -1)
    keep = (ki <= qi) & (ki > qi - sliding_window)  # [S, S] bool, True = attend
    ref = _sdpa_reference(q, k, v, scale, attn_mask=keep)

    max_abs = (ref - out.float()).abs().max().item()
    assert max_abs < tol, (
        f"{dtype} D{d} GQA{hq}/{hkv} swa{sliding_window} scale={scale:g} "
        f"{'persist' if persistent else 'default'}: max_abs={max_abs:.3e} >= {tol}"
    )


@requires_gfx942_gpu
@pytest.mark.gpu
def test_one_sliding_window_binary_serves_every_shape_and_head_config():
    """The sliding-window twin of :func:`test_one_binary_serves_every_head_config`.

    Non-persistent SWA is on the runtime-shape path: the window stays baked, while
    batch, both seqlens and both head counts are kernel params. Varying all of them
    at once under one window must reuse one launcher, and the band must still be
    right at every config -- a prune or mask that kept a baked shape term would pass
    the first config compiled and fail the rest.
    """
    import torch
    import torch.nn.functional as F

    from kernels.common.attention_dense_spec import attention_dense_cache_key
    from kernels.gfx942.attention_dense import _DENSE_LAUNCHER_CACHE

    dtype, d, window = "bf16", 128, 128
    tol = 4e-2
    tdt = getattr(torch, _TORCH_DT[dtype])
    scale = 1.0 / math.sqrt(d)

    # (batch, seqlen, Hq, Hkv): GQA 16, non-pow2 GQA 5 and 7, MHA.
    configs = ((1, 512, 128, 8), (2, 1024, 40, 8), (1, 2048, 28, 4), (4, 512, 32, 32))
    specs = [
        _as_gfx942_spec(
            _spec(dtype, d, hq, hkv, False, batch=b, sq=s, sliding_window=window)
        )
        for b, s, hq, hkv in configs
    ]

    assert all(s.runtime_shape for s in specs), "an SWA config left the runtime path"
    keys = [attention_dense_cache_key(s, arch="gfx942") for s in specs]
    assert len(set(keys)) == 1, f"SWA configs {configs} did not share a cache key"

    _DENSE_LAUNCHER_CACHE.pop(keys[0], None)
    before = set(_DENSE_LAUNCHER_CACHE)

    launchers = []
    for (B, S, hq, hkv), spec in zip(configs, specs):
        torch.manual_seed(0)
        q = torch.randn(B, S, hq, d, device="cuda", dtype=tdt)
        k = torch.randn(B, S, hkv, d, device="cuda", dtype=tdt)
        v = torch.randn(B, S, hkv, d, device="cuda", dtype=tdt)
        out = torch.empty(B, S, hq, d, device="cuda", dtype=tdt)

        run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)
        torch.cuda.synchronize()
        launchers.append(_launcher_for(spec))

        qi = torch.arange(S, device="cuda").view(-1, 1)
        ki = torch.arange(S, device="cuda").view(1, -1)
        keep = (ki <= qi) & (ki > qi - window)
        rep = hq // hkv
        ref = F.scaled_dot_product_attention(
            q.transpose(1, 2).float(),
            k.transpose(1, 2).float().repeat_interleave(rep, dim=1),
            v.transpose(1, 2).float().repeat_interleave(rep, dim=1),
            attn_mask=keep,
            scale=scale,
        ).transpose(1, 2)
        max_abs = (ref - out.float()).abs().max().item()
        assert max_abs < tol, (
            f"swa{window} B={B} S={S} Hq={hq} Hkv={hkv}: "
            f"max_abs={max_abs:.3e} >= {tol}"
        )

    assert launchers[0] is not None, (
        "no launcher cached after a successful run; _DENSE_LAUNCHER_CACHE is no "
        "longer keyed by attention_dense_cache_key and this test is blind"
    )
    assert all(lau is launchers[0] for lau in launchers), (
        f"SWA configs {configs} share a cache key but were served by DIFFERENT "
        "launcher objects -- the runtime-shape SWA kernel recompiled per config"
    )
    assert set(_DENSE_LAUNCHER_CACHE) - before == {keys[0]}, (
        "four SWA configs on the runtime path added more than one cache entry: "
        f"{sorted(set(_DENSE_LAUNCHER_CACHE) - before)}"
    )


# (batch, seqlen, Hq, Hkv) served by ONE persistent binary. W = NQB*Hq*B is the
# grid-stride work-item count (NQB = seqlen // 256); NP is the shipped 304:
#   W=40   < NP, NQB=1 (most CTAs idle; the interleave flip degenerates to qb=0)
#   W=320  in (NP, 2*NP), W % NP = 16
#   W=672  > 2*NP, W % NP = 64, odd B (non-pow2 batch magic)
#   W=384  MHA (gqa=1), W % NP = 80
#   W=608  == 2*NP exactly -- the tail-free control
#   W=120  odd Hq, even NQB=4: an interleave flip keyed on rem = qb0*Hq + hq
#          (instead of hq) maps qb0 {0,1,2,3} -> {0,2,2,0}, skipping tiles
# Non-pow2 GQA 5 and 7 put a real magic on the gqa divide.
_PERSIST_CONFIGS = (
    (1, 256, 40, 8),
    (2, 1024, 40, 8),
    (3, 2048, 28, 4),
    (2, 1536, 32, 32),
    (19, 512, 16, 4),
    (2, 1024, 15, 5),
)

# (dtype, persist_decode, interleave, sliding_window). Both decode orders pinned
# explicitly: "auto" resolves per shape, which would split the set across two keys.
# interleave only changes the qb_major body, so it rides that order alone.
_PERSIST_ONE_BINARY = [
    pytest.param("fp16", "hkv_major", False, 0, id="hkv_major"),
    pytest.param("fp16", "qb_major", False, 0, id="qb_major"),
    pytest.param("fp16", "qb_major", True, 0, id="qb_major-interleave"),
    pytest.param("bf16", "hkv_major", False, 128, id="hkv_major-swa128"),
    pytest.param("bf16", "qb_major", False, 128, id="qb_major-swa128"),
]


@requires_gfx942_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("dtype,persist_decode,interleave,window", _PERSIST_ONE_BINARY)
def test_one_persistent_binary_serves_every_shape_and_head_config(
    dtype, persist_decode, interleave, window
):
    """The persistent-grid twin of the one-binary tests above.

    The persistent grid takes batch, both seqlens and both head counts as kernel
    params too; only D, dtype, mask, window, ``num_persistent`` and the decode order
    stay baked. Its work-item count W is derived in the prologue and the grid-stride
    decode divides by runtime B / Hq / gqa / NQB through host-computed fast-division
    magics. A decode that kept a baked shape term, or a magic computed for the wrong
    divisor, would map work items to the wrong (qb, hq, bt) for every config but the
    first compiled -- duplicated or skipped tiles that only the numeric check sees.
    The configs straddle NP (W < NP, NP < W < 2*NP, W > 2*NP, W % NP != 0) so the
    grid-stride bound is exercised with idle CTAs, ragged tails and multiple trips.
    """
    import torch
    import torch.nn.functional as F

    from kernels.common.attention_dense_spec import attention_dense_cache_key
    from kernels.gfx942.attention_dense import _DENSE_LAUNCHER_CACHE

    d = 128
    tol = 2e-2 if dtype == "fp16" else 4e-2
    tdt = getattr(torch, _TORCH_DT[dtype])
    scale = 1.0 / math.sqrt(d)

    specs = [
        dataclasses.replace(
            _as_gfx942_spec(
                _spec(
                    dtype,
                    d,
                    hq,
                    hkv,
                    True,
                    batch=b,
                    sq=s,
                    sliding_window=window,
                    persist_decode=persist_decode,
                )
            ),
            interleave=interleave,
        )
        for b, s, hq, hkv in _PERSIST_CONFIGS
    ]

    # Preconditions: persistent, on the runtime path, the pinned order, one key --
    # and the configs really straddle NP, or the grid-stride claims are vacuous.
    assert all(
        s.persistent and s.runtime_shape for s in specs
    ), "a config left the persistent runtime-shape path"
    assert {s.resolved_persist_decode for s in specs} == {persist_decode}
    assert all(s.fastdiv_divisors for s in specs), "persistent decode has no fastdiv"
    np_ = specs[0].num_persistent
    work = [(s.seqlen_q // s.block_m) * s.num_query_heads * s.batch for s in specs]
    assert min(work) < np_ and max(work) > 2 * np_, (work, np_)
    assert any(w % np_ for w in work), (work, np_)
    keys = [attention_dense_cache_key(s, arch="gfx942") for s in specs]
    assert (
        len(set(keys)) == 1
    ), f"persistent configs {_PERSIST_CONFIGS} did not share a cache key"

    _DENSE_LAUNCHER_CACHE.pop(keys[0], None)
    before = set(_DENSE_LAUNCHER_CACHE)

    launchers = []
    for (B, S, hq, hkv), spec in zip(_PERSIST_CONFIGS, specs):
        torch.manual_seed(0)
        q = torch.randn(B, S, hq, d, device="cuda", dtype=tdt)
        k = torch.randn(B, S, hkv, d, device="cuda", dtype=tdt)
        v = torch.randn(B, S, hkv, d, device="cuda", dtype=tdt)
        # NaN-filled so a work item the decode skips cannot pass as stale zeros.
        out = torch.full((B, S, hq, d), float("nan"), device="cuda", dtype=tdt)

        run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)
        torch.cuda.synchronize()
        launchers.append(_launcher_for(spec))

        qi = torch.arange(S, device="cuda").view(-1, 1)
        ki = torch.arange(S, device="cuda").view(1, -1)
        keep = ki <= qi
        if window:
            keep &= ki > qi - window
        rep = hq // hkv
        ref = F.scaled_dot_product_attention(
            q.transpose(1, 2).float(),
            k.transpose(1, 2).float().repeat_interleave(rep, dim=1),
            v.transpose(1, 2).float().repeat_interleave(rep, dim=1),
            attn_mask=keep,
            scale=scale,
        ).transpose(1, 2)
        max_abs = (ref - out.float()).abs().max().item()
        assert max_abs < tol, (
            f"persist {persist_decode}{'+intl' if interleave else ''} "
            f"swa{window} B={B} S={S} Hq={hq} Hkv={hkv}: "
            f"max_abs={max_abs:.3e} >= {tol}"
        )

    assert launchers[0] is not None, (
        "no launcher cached after a successful run; _DENSE_LAUNCHER_CACHE is no "
        "longer keyed by attention_dense_cache_key and this test is blind"
    )
    assert all(lau is launchers[0] for lau in launchers), (
        f"persistent configs {_PERSIST_CONFIGS} share a cache key but were served "
        "by DIFFERENT launcher objects -- the persistent kernel recompiled per config"
    )
    assert set(_DENSE_LAUNCHER_CACHE) - before == {keys[0]}, (
        "persistent configs on the runtime path added more than one cache entry: "
        f"{sorted(set(_DENSE_LAUNCHER_CACHE) - before)}"
    )


@requires_gfx942_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("block_n", [32])
def test_dense_fp16_d128_numeric_correct_at_non_shipped_tile_width(block_n):
    """fp16-D128 stays numerically correct at a tile width dispatch never emits
    (``_DENSE_BLOCK_N = 64`` is a hard constant; block_n=32 is the small-tile double-K
    sweep direction). This does NOT prove the swizzle is engaged -- store and read
    apply the same permutation, so an off build is bit-identical here; the IR guard
    ``test_cfvst_swizzle_is_emitted_in_ir_with_matching_store_read_mask`` (CPU lane)
    covers that. This is the on-silicon correctness guard for the tile-width axis."""
    import torch

    d, hq, hkv, tol = 128, 16, 4, 2e-2
    B, S, scale = 1, 512, 1.0 / math.sqrt(128)
    torch.manual_seed(0)
    q = torch.randn(B, S, hq, d, device="cuda", dtype=torch.float16)
    k = torch.randn(B, S, hkv, d, device="cuda", dtype=torch.float16)
    v = torch.randn(B, S, hkv, d, device="cuda", dtype=torch.float16)
    out = torch.empty(B, S, hq, d, device="cuda", dtype=torch.float16)
    spec = dataclasses.replace(
        _spec("fp16", d, hq, hkv, False, batch=B, sq=S), block_n=block_n
    )
    run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)
    torch.cuda.synchronize()
    ref = _sdpa_reference(q, k, v, scale)
    max_abs = (ref - out.float()).abs().max().item()
    assert max_abs < tol, f"fp16 D128 block_n={block_n}: max_abs={max_abs:.3e} >= {tol}"


# bf16 D128 is the config this PR flips to exp2_fast, on BOTH grids. The oracle
# test above (tol=4e-2 vs fp32 SDPA) is far too loose to tell the two exp2 arms
# apart, so it cannot back the "results unchanged" claim. This does: a forced-off
# vs forced-on A/B on one identical spec must agree for every reachable softmax
# argument (both softmax args are always <= 0, exactly exp2_fast's precondition).
_EXP2_AB_COHORT = [
    ("bf16", 128, 16, 4, False),  # default grid (newly enabled here)
    ("bf16", 128, 16, 4, True),  # persistent grid
]


@requires_gfx942_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("dtype,d,hq,hkv,persistent", _EXP2_AB_COHORT)
def test_exp2_fast_matches_plain_exp2(dtype, d, hq, hkv, persistent):
    import torch

    tdt = getattr(torch, _TORCH_DT[dtype])
    B, S = 1, 512
    scale = 1.0 / math.sqrt(d)
    torch.manual_seed(0)

    q = torch.randn(B, S, hq, d, device="cuda", dtype=tdt)
    k = torch.randn(B, S, hkv, d, device="cuda", dtype=tdt)
    v = torch.randn(B, S, hkv, d, device="cuda", dtype=tdt)
    spec = _spec(dtype, d, hq, hkv, persistent, batch=B, sq=S)

    # Forced-off vs forced-on compile to distinct binaries (gfx942_kernel_name
    # tags the non-policy arm `e2f0`), so this really exercises both code paths.
    outs = {}
    for use_fast in (False, True):
        out = torch.empty(B, S, hq, d, device="cuda", dtype=tdt)
        # The dispatch factory hands back the SHARED spec; promote it to the gfx942
        # subclass (at shipped defaults for every other private knob) so only
        # use_exp2_fast moves between the two arms.
        run_attention_dense_torch(
            spec=dataclasses.replace(_as_gfx942_spec(spec), use_exp2_fast=use_fast),
            q=q,
            k=k,
            v=v,
            out=out,
            scale=scale,
        )
        outs[use_fast] = out
    torch.cuda.synchronize()

    max_abs = (outs[False].float() - outs[True].float()).abs().max().item()
    assert torch.equal(outs[False], outs[True]), (
        f"{dtype} D{d} {'persist' if persistent else 'default'}: exp2_fast "
        f"diverged from plain exp2 (max_abs={max_abs:.3e})"
    )


# Performance-only codegen knobs. None of them changes arithmetic (PV order keeps
# each output tile's key-step order, a width only splits the same stores, the
# diagonal split skips selects whose condition is always true), so every override
# must be BIT-identical to the shipped default. Sq=512 gives a second query block,
# whose below-diagonal tiles are what the split leaves unmasked.
_KNOB_AB_COHORT = [
    ("fp16", 128, False, dict(causal_diag_split=True)),
    ("fp16", 128, True, dict(causal_diag_split=True)),
    ("bf16", 64, False, dict(causal_diag_split=True)),
    ("fp16", 128, False, dict(pv_loop_order="k_major")),
    ("bf16", 128, True, dict(pv_loop_order="k_major")),
    ("bf16", 128, False, dict(o_store_width=1)),
    ("bf16", 64, True, dict(o_store_width=2)),
    ("fp16", 128, False, dict(pv_priority=2, pv_sched_fence_mask=0)),
    ("fp16", 128, False, dict(iglp=True, iglp_mode=1)),
]


@requires_gfx942_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("dtype,d,persistent,overrides", _KNOB_AB_COHORT)
def test_codegen_knob_is_bit_identical_to_default(dtype, d, persistent, overrides):
    import torch

    hq, hkv = 16, 4
    tdt = getattr(torch, _TORCH_DT[dtype])
    B, S = 1, 512
    scale = 1.0 / math.sqrt(d)
    torch.manual_seed(0)

    q = torch.randn(B, S, hq, d, device="cuda", dtype=tdt)
    k = torch.randn(B, S, hkv, d, device="cuda", dtype=tdt)
    v = torch.randn(B, S, hkv, d, device="cuda", dtype=tdt)
    base = _as_gfx942_spec(_spec(dtype, d, hq, hkv, persistent, batch=B, sq=S))

    outs = []
    for spec in (base, dataclasses.replace(base, **overrides)):
        out = torch.empty(B, S, hq, d, device="cuda", dtype=tdt)
        run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)
        outs.append(out)
    torch.cuda.synchronize()

    max_abs = (outs[0].float() - outs[1].float()).abs().max().item()
    assert torch.equal(outs[0], outs[1]), (
        f"{dtype} D{d} {'persist' if persistent else 'default'} {overrides}: "
        f"diverged from the shipped default (max_abs={max_abs:.3e})"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
