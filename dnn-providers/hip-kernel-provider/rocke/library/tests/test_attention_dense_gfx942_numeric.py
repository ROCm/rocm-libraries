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

# (dtype, head_size, num_query_heads, num_kv_heads, persistent, sliding_window) --
# STANDALONE sliding-window (no sinks; gfx942 dense has no sink support yet). All rows
# are causal (sliding_window > 0 requires causal). Window is a multiple of the shipped
# block_n (64): 128, 256. Covers both dtypes, D64/D128, and BOTH grid variants -- the
# persistent rows exercise the per-work-item start_tile prune that the default grid
# does not.
_SWA_COHORT = [
    ("bf16", 128, 16, 4, False, 128),
    ("bf16", 128, 16, 4, True, 128),
    ("fp16", 128, 16, 4, False, 256),
    ("fp16", 128, 16, 4, True, 256),
    ("bf16", 64, 16, 4, False, 128),
    ("bf16", 64, 16, 4, True, 128),
]

# Top-left causal at seqlen_q != seqlen_kv. gfx942 dense masks ``ktok <= query_tok``
# (top-left, torch ``is_causal=True``), so a cross-length request must match the
# SDPA ``is_causal=True`` oracle -- which is itself top-left (``tril`` with diagonal
# 0) at any Sq/Sk. Both directions: Sq<Sk (each row sees only its own prefix, not
# the Sk-Sq extra keys a bottom-right mask would add) and Sq>Sk (rows i >= Sk see
# every key; no row is fully masked under top-left). Sq % 256 == 0 and Sk % 64 == 0
# keep the shape inside ``supports_attention_dense`` (non-ragged cross-length).
# Rows: (dtype, head_size, num_query_heads, num_kv_heads, persistent, causal, sq, sk).
_TL_CROSS_COHORT = [
    (dt, d, 16, 4, persistent, True, sq, sk)
    for dt in ("bf16", "fp16")
    for d in (64, 128)
    for persistent in (False, True)
    for sq, sk in ((512, 1024), (1024, 512))
]

# Sliding window under top-left at Sq < Sk: keep k for query q iff q - W < k <= q.
# Sq > Sk (rows q >= Sk + W - 1 see no key and must output zeros) is covered by the
# band matrix (TestDenseBandNumeric) below, whose oracle checks the zero rows.
# Rows: (dtype, head_size, num_query_heads, num_kv_heads, persistent, sliding_window, sq, sk).
_SWA_TL_CROSS_COHORT = [
    ("bf16", 128, 16, 4, False, 128, 512, 1024),
    ("fp16", 128, 16, 4, True, 256, 512, 1024),
]


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
    sk=None,
    sliding_window=0,
):
    """The SHIPPED gfx942 dense spec for a cohort row, built through the dispatch
    factory (``dispatch.attention.gfx942._dense_spec``) rather than hand-rolled.

    Hand-rolling the spec silently pins every tuned lever to the shared (gfx950)
    dataclass default, so the lane would assert on configs that do not ship. The
    concrete one this cohort hit: ``waves_per_eu``. Dispatch resolves it from the
    kernel's own policy (``_tuned_waves_per_eu``), which returns 4 for bf16/D64 --
    the row ``_COHORT`` above labels "the wpe=4 config" -- while the dataclass default
    is 2. That is not a cosmetic difference: ``waves_per_eu`` is emitted as the
    ``amdgpu-waves-per-eu`` attribute, changes register allocation, and is tagged into
    ``gfx942_kernel_name`` as ``wpe{N}``, so wpe2 and wpe4 are DIFFERENT binaries.
    ``num_persistent`` was likewise hard-coded to 304 beside a dispatch constant that
    already resolves to 304 -- left at the request default here so ``_dense_spec``
    substitutes the gfx942 CU count itself and the two cannot drift apart.

    Deriving the spec from the factory (the pattern
    ``test_attention_dense_gfx942_golden.py::mk_dispatch`` uses for its D64 cases)
    also means a future gfx942 tuning change is picked up here with no edit.

    ``sk`` defaults to ``sq`` (self-attention); pass it for a cross-length row. The
    request is always top-left (``mask_type=1``), the only alignment gfx942 dense
    implements.

    Only ``dense_persistent`` is pinned rather than left on "auto": the cohort asserts
    BOTH grid variants at one fixed Sq, where "auto" would pick a single one. Every
    other lever -- block_n, the D64 K row-group pad, persist_decode, ragged -- is
    whatever the shipped path folds in.
    """
    # Imported lazily, mirroring the golden sibling: keeps module import (and hence
    # CPU collection of this gpu-marked file) independent of the dispatch package.
    from dispatch.attention import AttentionRequest
    from dispatch.attention.gfx942 import _dense_spec

    return _dense_spec(
        AttentionRequest(
            batch=batch,
            nhead_q=hq,
            nhead_k=hkv,
            seqlen_q=sq,
            seqlen_k=sq if sk is None else sk,
            hdim_q=d,
            hdim_v=d,
            arch="gfx942",
            mask_type=1 if causal else 0,
            dtype=dtype,
            sliding_window=sliding_window,
            algorithm="attention_dense",
            dense_persistent="on" if persistent else "off",
        )
    )


@requires_gfx942_gpu
@pytest.mark.gpu
@pytest.mark.parametrize(
    "dtype,d,hq,hkv,persistent,causal,sq,sk",
    [row + (512, 512) for row in _COHORT] + _TL_CROSS_COHORT,
)
def test_dense_numeric_vs_fp32_sdpa(dtype, d, hq, hkv, persistent, causal, sq, sk):
    import torch
    import torch.nn.functional as F

    tol = 2e-2 if dtype == "fp16" else 4e-2
    tdt = getattr(torch, _TORCH_DT[dtype])
    B, S, Sk = 1, sq, sk
    scale = 1.0 / math.sqrt(d)
    torch.manual_seed(0)

    # run_attention_dense_torch ABI: q/out [B,Sq,Hq,D], k/v [B,Sk,Hkv,D], dense.
    q = torch.randn(B, S, hq, d, device="cuda", dtype=tdt)
    k = torch.randn(B, Sk, hkv, d, device="cuda", dtype=tdt)
    v = torch.randn(B, Sk, hkv, d, device="cuda", dtype=tdt)
    out = torch.empty(B, S, hq, d, device="cuda", dtype=tdt)

    spec = _spec(dtype, d, hq, hkv, persistent, causal=causal, batch=B, sq=S, sk=Sk)
    run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)
    torch.cuda.synchronize()

    # fp32 SDPA oracle in [B,H,S,D] layout. GQA is expanded HERE, by repeating each
    # kv head to its query heads, rather than via the ``enable_gqa=`` kwarg: that
    # kwarg is a recent addition to ``scaled_dot_product_attention``, and on an older
    # ROCm torch passing it raises TypeError -- which ERRORS the whole gpu cohort
    # instead of leaving it to the device gate. ``repeat_interleave`` along the head
    # axis is exactly what ``enable_gqa`` does internally, and it is the mapping the
    # kernel itself uses (hkv = hq // gqa), so the asserted reference is unchanged.
    # rep == 1 (the MHA row) makes it a plain copy, matching the old
    # ``enable_gqa=(hkv != hq)`` no-op. ``is_causal=True`` is top-left (tril with
    # diagonal 0) at every Sq/Sk, which is the mask the kernel applies.
    rep = hq // hkv
    qf = q.transpose(1, 2).float()
    kf = k.transpose(1, 2).float().repeat_interleave(rep, dim=1)
    vf = v.transpose(1, 2).float().repeat_interleave(rep, dim=1)
    ref = F.scaled_dot_product_attention(
        qf, kf, vf, is_causal=causal, scale=scale
    ).transpose(
        1, 2
    )  # -> [B,S,Hq,D]

    max_abs = (ref - out.float()).abs().max().item()
    assert max_abs < tol, (
        f"{dtype} D{d} GQA{hq}/{hkv} Sq{S}/Sk{Sk} {'causal' if causal else 'full'} "
        f"{'persist' if persistent else 'default'}: max_abs={max_abs:.3e} >= {tol}"
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
    """One compiled artifact, three shapes (one cross-length), correct numerics at each.

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
    import torch.nn.functional as F

    from kernels.common.attention_dense_spec import attention_dense_cache_key
    from kernels.gfx942.attention_dense import _DENSE_LAUNCHER_CACHE

    dtype, d, hq, hkv = "fp16", 128, 16, 4  # flagship default, non-persistent
    tol = 2e-2
    tdt = getattr(torch, _TORCH_DT[dtype])
    scale = 1.0 / math.sqrt(d)

    # (batch, seqlen_q, seqlen_kv). The third shape is cross-length top-left
    # (Sq < Sk): seqlen_kv is a runtime param too, so it must reuse the same binary.
    shapes = ((1, 512, 512), (4, 1024, 1024), (2, 512, 1024))
    specs = [
        _as_gfx942_spec(_spec(dtype, d, hq, hkv, False, batch=b, sq=s, sk=sk))
        for b, s, sk in shapes
    ]

    # Preconditions: genuinely different shapes, on the runtime path, and sharing
    # one key -- otherwise the reuse assertion below is vacuous.
    assert specs[0].runtime_shape, "cohort row is not on the runtime-shape path"
    assert (specs[0].batch, specs[0].seqlen_q) != (specs[1].batch, specs[1].seqlen_q)
    assert specs[2].seqlen_q != specs[2].seqlen_kv
    keys = [attention_dense_cache_key(s, arch="gfx942") for s in specs]
    assert len(set(keys)) == 1, f"shapes {shapes} did not share a cache key"

    # Own the cache state: evicting first makes the "exactly one new entry"
    # assertion independent of which tests ran before this one.
    _DENSE_LAUNCHER_CACHE.pop(keys[0], None)
    before = set(_DENSE_LAUNCHER_CACHE)

    rep = hq // hkv
    launchers = []
    for (B, S, Sk), spec in zip(shapes, specs):
        torch.manual_seed(0)
        q = torch.randn(B, S, hq, d, device="cuda", dtype=tdt)
        k = torch.randn(B, Sk, hkv, d, device="cuda", dtype=tdt)
        v = torch.randn(B, Sk, hkv, d, device="cuda", dtype=tdt)
        out = torch.empty(B, S, hq, d, device="cuda", dtype=tdt)

        run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)
        torch.cuda.synchronize()
        launchers.append(_launcher_for(spec))

        ref = F.scaled_dot_product_attention(
            q.transpose(1, 2).float(),
            k.transpose(1, 2).float().repeat_interleave(rep, dim=1),
            v.transpose(1, 2).float().repeat_interleave(rep, dim=1),
            is_causal=True,
            scale=scale,
        ).transpose(1, 2)
        max_abs = (ref - out.float()).abs().max().item()
        assert max_abs < tol, f"B={B} Sq={S} Sk={Sk}: max_abs={max_abs:.3e} >= {tol}"

    assert launchers[0] is not None, (
        "no launcher cached after a successful run; _DENSE_LAUNCHER_CACHE is no "
        "longer keyed by attention_dense_cache_key and this test is blind"
    )
    assert all(lau is launchers[0] for lau in launchers), (
        f"shapes {shapes} share a cache key but were served by DIFFERENT launcher "
        "objects -- the runtime-shape kernel recompiled per shape, so the AOT "
        "instance count still scales with the shape space"
    )
    assert set(_DENSE_LAUNCHER_CACHE) - before == {keys[0]}, (
        "the shapes on the runtime path added more than one cache entry: "
        f"{sorted(set(_DENSE_LAUNCHER_CACHE) - before)}"
    )


@requires_gfx942_gpu
@pytest.mark.gpu
@pytest.mark.parametrize(
    "dtype,d,hq,hkv,persistent,sliding_window,sq,sk",
    [row + (512, 512) for row in _SWA_COHORT] + _SWA_TL_CROSS_COHORT,
)
def test_dense_swa_numeric_vs_fp32_sdpa(
    dtype, d, hq, hkv, persistent, sliding_window, sq, sk
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
    import torch.nn.functional as F

    tol = 2e-2 if dtype == "fp16" else 4e-2
    tdt = getattr(torch, _TORCH_DT[dtype])
    B, S, Sk = 1, sq, sk
    scale = 1.0 / math.sqrt(d)
    torch.manual_seed(0)

    q = torch.randn(B, S, hq, d, device="cuda", dtype=tdt)
    k = torch.randn(B, Sk, hkv, d, device="cuda", dtype=tdt)
    v = torch.randn(B, Sk, hkv, d, device="cuda", dtype=tdt)
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
        sk=Sk,
        sliding_window=sliding_window,
    )
    run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)
    torch.cuda.synchronize()

    qi = torch.arange(S, device="cuda").view(-1, 1)
    ki = torch.arange(Sk, device="cuda").view(1, -1)
    # [Sq, Sk] bool, True = attend; top-left diagonal (k <= q) at any Sq/Sk.
    keep = (ki <= qi) & (ki > qi - sliding_window)
    rep = hq // hkv
    qf = q.transpose(1, 2).float()
    kf = k.transpose(1, 2).float().repeat_interleave(rep, dim=1)
    vf = v.transpose(1, 2).float().repeat_interleave(rep, dim=1)
    ref = F.scaled_dot_product_attention(
        qf, kf, vf, attn_mask=keep, scale=scale
    ).transpose(
        1, 2
    )  # -> [B,S,Hq,D]

    max_abs = (ref - out.float()).abs().max().item()
    assert max_abs < tol, (
        f"{dtype} D{d} GQA{hq}/{hkv} Sq{S}/Sk{Sk} swa{sliding_window} "
        f"{'persist' if persistent else 'default'}: max_abs={max_abs:.3e} >= {tol}"
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
    import torch.nn.functional as F

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
    rep = hq // hkv
    qf = q.transpose(1, 2).float()
    kf = k.transpose(1, 2).float().repeat_interleave(rep, dim=1)
    vf = v.transpose(1, 2).float().repeat_interleave(rep, dim=1)
    ref = F.scaled_dot_product_attention(
        qf, kf, vf, is_causal=True, scale=scale
    ).transpose(1, 2)
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


# --------------------------------------------------------------------------- #
# The attention band, ragged lengths and cross-attention -- the gfx942 mirror of
# ``test_attention_dense_gfx950_numeric.py``'s TestDenseTopLeftCrossLengthNumeric /
# TestDenseBandNumeric, at the same shapes (gfx942 dense has no sinks, so the sinks
# row is dropped).
# --------------------------------------------------------------------------- #
_BAND_TOL = 4e-2  # bf16 vs the fp32 reference, as the cohorts above

# Cross-length top-left (unshifted diagonal), sq < sk and sq > sk, ragged and aligned.
# For sq > sk the ragged padded keys sit below the diagonal of every row with
# q >= Skv, so correctness depends on the key-pad mask.
#   (sq, skv, batch, ragged)
_TOP_LEFT_CASES = [
    pytest.param(1000, 1050, 1, True, id="ragged-1000x1050"),
    pytest.param(1050, 1000, 1, True, id="ragged-1050x1000"),
    pytest.param(197, 400, 1, True, id="ragged-197x400"),
    pytest.param(400, 197, 1, True, id="ragged-400x197"),
    pytest.param(1234, 300, 1, True, id="ragged-1234x300"),
    pytest.param(1050, 1000, 2, True, id="ragged-1050x1000-batch2"),
    pytest.param(1000, 1000, 2, True, id="ragged-self-1000-batch2"),
    pytest.param(512, 1024, 1, False, id="aligned-512x1024"),
    pytest.param(1024, 512, 1, False, id="aligned-1024x512"),
]

# The cuDNN band ``(left = sliding_window, right = right_bound)`` on the top-left
# diagonal: key ``k`` is visible to query ``q`` iff ``q - k < left`` (when set) and
# ``k <= q + right`` (right = -1 is unbounded, 0 causal, R > 0 lookahead). A row whose
# band holds no key must output zeros.
#   (sq, skv, window, right, persistent, ragged)
_BAND_CASES = [
    # causal window x cross-length, with empty rows
    pytest.param(1024, 512, 128, 0, False, False, id="causal-w128-1024x512-empty"),
    pytest.param(1024, 256, 128, 0, False, False, id="causal-w128-1024x256-empty"),
    pytest.param(
        1024, 512, 128, 0, True, False, id="persist-causal-w128-1024x512-empty"
    ),
    pytest.param(512, 1024, 128, 0, False, False, id="causal-w128-512x1024"),
    pytest.param(1024, 1024, 128, 0, False, False, id="causal-w128-1024x1024"),
    # non-causal (left-only) windows
    pytest.param(1024, 1024, 128, -1, False, False, id="left-only-w128-1024x1024"),
    pytest.param(512, 1024, 128, -1, False, False, id="left-only-w128-512x1024"),
    pytest.param(1024, 512, 128, -1, False, False, id="left-only-w128-1024x512-empty"),
    pytest.param(1024, 512, 128, -1, True, False, id="persist-left-only-w128-1024x512"),
    pytest.param(
        1024, 1024, 256, -1, True, False, id="persist-left-only-w256-1024x1024"
    ),
    # two-sided windows and lookahead
    pytest.param(1024, 1024, 128, 64, False, False, id="two-sided-w128-r64-1024x1024"),
    pytest.param(512, 1024, 192, 128, False, False, id="two-sided-w192-r128-512x1024"),
    pytest.param(
        1024, 512, 128, 64, True, False, id="persist-two-sided-w128-r64-1024x512"
    ),
    pytest.param(1024, 1024, 0, 100, False, False, id="lookahead-r100-1024x1024"),
    pytest.param(
        1024, 1024, 0, 100, True, False, id="persist-lookahead-r100-1024x1024"
    ),
    pytest.param(512, 1024, 0, 7, False, False, id="lookahead-r7-512x1024"),
    pytest.param(1024, 512, 0, 300, False, False, id="lookahead-r300-1024x512"),
    # full attention, aligned and ragged cross-length (cross-attention)
    pytest.param(512, 1024, 0, -1, False, False, id="full-512x1024"),
    pytest.param(1000, 1050, 0, -1, False, True, id="full-ragged-1000x1050"),
    pytest.param(1050, 1000, 0, -1, False, True, id="full-ragged-1050x1000"),
    pytest.param(300, 1234, 0, -1, True, True, id="persist-full-ragged-300x1234"),
    # lookahead on ragged cross-length (key padding can sit inside q + right)
    pytest.param(1000, 1050, 0, 64, False, True, id="lookahead-ragged-1000x1050"),
    pytest.param(1050, 1000, 0, 64, False, True, id="lookahead-ragged-1050x1000"),
    pytest.param(1050, 1000, 0, 64, True, True, id="persist-lookahead-ragged-1050x1000"),
]


def _band_direct_spec(
    sq, skv, window, right, persistent, ragged, *, batch=1, head_size=128
):
    from kernels.gfx942.attention_dense import Gfx942AttentionDenseSpec

    return Gfx942AttentionDenseSpec(
        batch=batch,
        seqlen_q=sq,
        seqlen_kv=skv,
        num_query_heads=4,
        num_kv_heads=1,
        head_size=head_size,
        causal=right >= 0,
        right_bound=max(right, 0),
        dtype="bf16",
        sliding_window=window,
        persistent=persistent,
        num_persistent=304,
        ragged=ragged,
    )


def _band_reference(q, k, v, scale, window, right):
    """fp32 reference for the band; also returns the rows that see no key."""
    import torch

    sq, skv = q.shape[1], k.shape[1]
    qi = torch.arange(sq, device=q.device)[:, None]
    ki = torch.arange(skv, device=q.device)[None, :]
    allowed = torch.ones(sq, skv, dtype=torch.bool, device=q.device)
    if right >= 0:
        allowed &= ki <= qi + right
    if window:
        allowed &= qi - ki < window
    scores = torch.einsum("bqhd,bkd->bhqk", q.float(), k[:, :, 0].float()) * scale
    scores = scores.masked_fill(~allowed[None, None], float("-inf"))
    empty = ~allowed.any(dim=1)
    probs = torch.softmax(scores, dim=-1)
    probs[:, :, empty, :] = 0.0
    ref = torch.einsum("bhqk,bkd->bqhd", probs, v[:, :, 0].float())
    return ref, empty


def _check_band(out, ref, empty, label):
    import torch

    assert not torch.isnan(out).any(), f"{label}: NaN in output"
    err = (out.float() - ref)[:, ~empty].abs().max().item()
    assert err < _BAND_TOL, f"{label}: max_abs={err:.3e} >= {_BAND_TOL}"
    if empty.any():
        worst = out.float()[:, empty].abs().max().item()
        assert worst == 0.0, f"{label}: {int(empty.sum())} empty rows not zero: {worst}"


def _run_band(spec, window, right, seed):
    import torch

    torch.manual_seed(seed)
    b, sq, skv, d = spec.batch, spec.seqlen_q, spec.seqlen_kv, spec.head_size
    q = torch.randn(b, sq, 4, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, skv, 1, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, skv, 1, d, device="cuda", dtype=torch.bfloat16)
    # Poison the output so a row the kernel never writes cannot pass as zero.
    out = torch.full_like(q, float("nan"))
    scale = 1.0 / math.sqrt(d)
    run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)
    torch.cuda.synchronize()
    return out, _band_reference(q, k, v, scale, window, right)


class TestDenseTopLeftCrossLengthNumeric:
    @requires_gfx942_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("persistent", [False, True], ids=["default", "persist"])
    @pytest.mark.parametrize("sq,skv,batch,ragged", _TOP_LEFT_CASES)
    def test_jit_matches_top_left(self, sq, skv, batch, ragged, persistent):
        spec = _band_direct_spec(sq, skv, 0, 0, persistent, ragged, batch=batch)
        out, (ref, empty) = _run_band(spec, 0, 0, seed=0)
        _check_band(out, ref, empty, spec.kernel_name())


class TestDenseBandNumeric:
    @requires_gfx942_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("sq,skv,window,right,persistent,ragged", _BAND_CASES)
    def test_jit_matches_band(self, sq, skv, window, right, persistent, ragged):
        spec = _band_direct_spec(sq, skv, window, right, persistent, ragged)
        out, (ref, empty) = _run_band(spec, window, right, seed=0)
        _check_band(out, ref, empty, spec.kernel_name())

    @requires_gfx942_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "sq,skv,window,right,persistent,ragged",
        [
            pytest.param(1024, 512, 128, -1, False, False, id="d64-left-only-empty"),
            pytest.param(1050, 1000, 0, 64, True, True, id="d64-persist-lookahead"),
            pytest.param(300, 1234, 0, -1, False, True, id="d64-full-ragged"),
        ],
    )
    def test_jit_matches_band_d64(self, sq, skv, window, right, persistent, ragged):
        """D64 takes the packed two-rows-per-DMA K/V path: the ragged tail and the
        band masks go through a different loader than D128."""
        spec = _band_direct_spec(
            sq, skv, window, right, persistent, ragged, head_size=64
        )
        out, (ref, empty) = _run_band(spec, window, right, seed=2)
        _check_band(out, ref, empty, spec.kernel_name())

    @requires_gfx942_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("sq,skv,window,right,persistent,ragged", _BAND_CASES)
    def test_dispatched_request_matches_band(
        self, sq, skv, window, right, persistent, ragged
    ):
        """The same bands through ``AttentionRequest`` -> dispatch -> dense spec."""
        from dispatch.attention import AttentionRequest
        from dispatch.attention.gfx942 import dense_spec_for_request

        req = AttentionRequest(
            batch=1,
            nhead_q=4,
            nhead_k=1,
            seqlen_q=sq,
            seqlen_k=skv,
            hdim_q=128,
            hdim_v=128,
            arch="gfx942",
            dtype="bf16",
            mask_type=0,  # NO_MASK: the band comes from (sliding_window, right_bound)
            sliding_window=window,
            right_bound=right,
            algorithm="attention_dense",
            dense_persistent="on" if persistent else "off",
        )
        spec = dense_spec_for_request(req)
        assert spec.causal == (right >= 0) and spec.right_bound == max(right, 0)
        assert spec.ragged == ragged and spec.persistent == persistent
        out, (ref, empty) = _run_band(spec, window, right, seed=1)
        _check_band(out, ref, empty, spec.kernel_name())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
