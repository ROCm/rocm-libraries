# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only tests for the gfx942 dense prefill kernel (AICK-1664).

Covers the SPEC / VALIDATION / PUBLIC-SURFACE layer and the IR build. No GPU and
no comgr: ``build_attention_dense`` is exercised for its ``KernelDef``, not
compiled or launched. The GPU numeric lane lives in the live benchmark
(``benchmarks/gfx942/attention/prefill/benchmark_dense_prefill_live.py``) --
mirroring the gfx950 precedent, where numeric correctness is a bench gate rather
than a CI pytest.

The central invariant asserted here is the **supports/build contract**:
``supports_attention_dense(spec)[0] is True`` must imply
``build_attention_dense(spec)`` succeeds. A predicate more permissive than the
builder lets dispatch select a spec it cannot build.

NOT WIRED INTO CI: like ``test_attention_dense_golden.py``, this file lives under
``library/tests/``, which ``platform/tests/run_all.py`` does NOT collect (it only
pytests ``platform/tests/``). Run it manually:

    cd rocke/library
    PYTHONPATH=../platform/python:. python -m pytest \
        tests/test_attention_dense_gfx942.py

A gfx942 golden-IR lane does not exist yet -- ``test_attention_dense_golden.py``
is gfx950-only and the fixture has no gfx942 SHAs. Tracked in the AICK-1664 plan.
"""

import pytest

from kernels.gfx942.attention_dense import (
    AttentionDenseSpec,
    attention_dense_block,
    attention_dense_grid,
    build_attention_dense,
    p0_kernel_name,
    supports_attention_dense,
    _p0_use_cfvst,
    _p0_use_exp2_fast,
)

# Query rows per CTA baked into the P0 body; block_n must divide it.
_BLOCK_M = 256
_EXPECTED_WORKGROUP_SIZE = (_BLOCK_M // 32) * 64  # 8 wave64s = 512 threads


def _spec(**kw) -> AttentionDenseSpec:
    base = dict(
        batch=1,
        seqlen_q=2048,
        seqlen_kv=2048,
        num_query_heads=128,
        num_kv_heads=8,
        head_size=128,
        causal=True,
        dtype="bf16",
        block_n=64,
    )
    base.update(kw)
    return AttentionDenseSpec(**base)


# --------------------------------------------------------------------------- #
# in-scope cohort: supports accepts, build emits
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("d", [64, 128])
def test_supports_accepts_in_scope_cohort(dtype, d):
    ok, why = supports_attention_dense(_spec(dtype=dtype, head_size=d), arch="gfx942")
    assert ok, why


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize(
    "hq,hkv", [(128, 8), (32, 32), (16, 4), (40, 8), (28, 4)]  # MHA, GQA, non-pow2
)
def test_build_emits_kernel_for_in_scope_cohort(dtype, d, causal, hq, hkv):
    """The P0 body builds for every shape the port claims, and names itself
    consistently with the batch-unique name dispatch/the launcher cache key on."""
    spec = _spec(
        dtype=dtype, head_size=d, causal=causal, num_query_heads=hq, num_kv_heads=hkv
    )
    kd = build_attention_dense(spec, arch="gfx942")
    assert kd.name == p0_kernel_name(spec)
    assert kd.attrs["max_workgroup_size"] == _EXPECTED_WORKGROUP_SIZE


def test_build_rejects_non_gfx942():
    with pytest.raises(NotImplementedError, match="gfx942-only"):
        build_attention_dense(_spec(), arch="gfx950")


# --------------------------------------------------------------------------- #
# kernel-name identity: batch is baked into the buffer extents but omitted from
# the shared kernel_name(), so a name-keyed cache would serve the B=1 binary for
# a B>1 launch and read out of bounds. p0_kernel_name() is the guard.
# --------------------------------------------------------------------------- #
def test_p0_kernel_name_covers_every_baked_parameter():
    """batch and waves_per_eu are both baked into the artifact but omitted from the
    shared kernel_name(): batch sizes the buffer-resource extents, waves_per_eu is
    emitted as amdgpu-waves-per-eu and changes register allocation. Either one
    colliding in a name-keyed cache serves the wrong binary."""
    assert p0_kernel_name(_spec(waves_per_eu=2)) != p0_kernel_name(
        _spec(waves_per_eu=3)
    )
    a = build_attention_dense(_spec(waves_per_eu=2), arch="gfx942")
    c = build_attention_dense(_spec(waves_per_eu=3), arch="gfx942")
    assert a.attrs["waves_per_eu"] != c.attrs["waves_per_eu"]
    assert a.name != c.name


def test_p0_kernel_name_is_batch_unique():
    names = {p0_kernel_name(_spec(batch=b)) for b in (1, 2, 4, 8)}
    assert len(names) == 4, f"batch must disambiguate the kernel name, got {names}"
    assert "_b4_" in p0_kernel_name(_spec(batch=4))


def test_build_bakes_batch_into_the_emitted_symbol():
    assert build_attention_dense(_spec(batch=4), arch="gfx942").name != (
        build_attention_dense(_spec(batch=1), arch="gfx942").name
    )


# --------------------------------------------------------------------------- #
# scope rejections -- each of these once reached the builder and raised, or (worse)
# built a silently wrong kernel
# --------------------------------------------------------------------------- #
def test_supports_rejects_non_gfx942():
    ok, why = supports_attention_dense(_spec(), arch="gfx950")
    assert not ok and "gfx942-only" in why


@pytest.mark.parametrize(
    "kw,marker",
    [
        (dict(persistent=True, num_persistent=228), "persistent"),
        (dict(varlen=True), "varlen"),
        (dict(seqlen_q=1000, seqlen_kv=1000, ragged=True), "ragged"),
        (dict(sliding_window=64), "sliding_window"),
    ],
)
def test_supports_rejects_modes_deferred_to_later_phases(kw, marker):
    """P0 implements the default-grid uniform dense path only. These modes must be
    rejected by ``supports`` -- not merely by ``build`` -- or dispatch selects a spec
    it cannot build (``_dense_spec`` sets ragged=True for any non-256-multiple
    self-attention length, which is most real serving shapes)."""
    ok, why = supports_attention_dense(_spec(**kw), arch="gfx942")
    assert not ok, f"{marker} must be rejected at the supports layer"
    assert marker in why


@pytest.mark.parametrize("block_n", [96, 160, 224])
def test_supports_rejects_block_n_not_dividing_the_query_tile(block_n):
    """``n_per = 256 // block_n`` floors in the causal KV clamp, so a block_n that
    does not divide the query tile silently DROPS the keys past the last whole
    sub-tile -- wrong numbers, no error."""
    ok, why = supports_attention_dense(
        _spec(block_n=block_n, seqlen_kv=block_n * 16), arch="gfx942"
    )
    assert not ok and "block_n" in why


def test_supports_rejects_block_n_larger_than_the_query_tile():
    """block_n > 256 makes n_per 0 -> zero-trip KV loop -> l == 0 -> rcp(0) -> NaN."""
    ok, why = supports_attention_dense(
        _spec(block_n=512, seqlen_kv=2048), arch="gfx942"
    )
    assert not ok and "block_n" in why


def test_supports_rejects_over_budget_lds():
    """block_n=128 at D128 needs 2*128*(128+8)*2 = 69632 B > the 64 KB gfx942 LDS.
    Without this gate it reaches comgr and dies with an opaque CODEGEN abort."""
    ok, why = supports_attention_dense(_spec(block_n=128, head_size=128), arch="gfx942")
    assert not ok and "LDS" in why


def test_supports_accepts_lds_budget_that_fits():
    # Same block_n at D64 halves the footprint (unpadded rows) and must still pass.
    ok, why = supports_attention_dense(_spec(block_n=128, head_size=64), arch="gfx942")
    assert ok, why


@pytest.mark.parametrize(
    "kw,limit",
    [
        # K/V buffer-resource num_records is an i32 (bytes). Hq=Hkv=8 keeps
        # qo_elems at 2**30 so ONLY the K/V check can fire -- otherwise this
        # passes merely because K/V happens to be checked before Q/O.
        (
            dict(
                batch=64,
                seqlen_q=16384,
                seqlen_kv=16384,
                num_query_heads=8,
                num_kv_heads=8,
            ),
            "K/V",
        ),
        # Q/O use raw 32-bit element offsets with NO hardware bounds clamp.
        (dict(batch=16, seqlen_q=8192, seqlen_kv=8192, num_query_heads=128), "Q/O"),
    ],
)
def test_supports_rejects_extents_past_32_bit_addressing(kw, limit):
    ok, why = supports_attention_dense(_spec(**kw), arch="gfx942")
    assert not ok and limit in why


def test_dataclass_rejects_out_of_scope_headsize():
    """D256 is AICK-1495/1496. The dataclass is the stricter guard -- it rejects at
    construction, before ``supports_attention_dense`` is reachable."""
    with pytest.raises(ValueError, match=r"head_size must be 64 or 128"):
        _spec(head_size=256)


# --------------------------------------------------------------------------- #
# the contract: supports is the single gate
# --------------------------------------------------------------------------- #
_CONTRACT_GRID = [
    dict(),
    dict(dtype="fp16"),
    dict(head_size=64),
    dict(causal=False),
    dict(num_query_heads=40, num_kv_heads=8),
    dict(block_n=32),
    dict(block_n=96, seqlen_kv=1536),
    dict(block_n=128),
    dict(block_n=128, head_size=64),
    dict(block_n=512, seqlen_kv=2048),
    dict(persistent=True, num_persistent=228),
    dict(varlen=True),
    dict(seqlen_q=1000, seqlen_kv=1000, ragged=True),
    dict(sliding_window=64),
    dict(batch=4),
    dict(batch=64, seqlen_q=16384, seqlen_kv=16384, num_kv_heads=8),
]


def _grid_id(kw: dict) -> str:
    """Value-bearing id: keys alone collide (three block_n cases)."""
    return "-".join(f"{k}{v}" for k, v in sorted(kw.items())) or "base"


def test_contract_grid_exercises_both_sides():
    """Guard against vacuity: if ``supports`` regressed to all-False, the contract
    test below would still pass on every case."""
    verdicts = [
        supports_attention_dense(_spec(**kw), arch="gfx942")[0] for kw in _CONTRACT_GRID
    ]
    accepted = [_grid_id(k) for k, v in zip(_CONTRACT_GRID, verdicts) if v]
    rejected = [_grid_id(k) for k, v in zip(_CONTRACT_GRID, verdicts) if not v]
    assert len(accepted) >= 6, f"grid must exercise the accepted region, got {accepted}"
    assert len(rejected) >= 6, f"grid must exercise the rejected region, got {rejected}"


@pytest.mark.parametrize("kw", _CONTRACT_GRID, ids=_grid_id)
def test_supports_true_implies_build_succeeds(kw):
    """The load-bearing invariant. If ``supports`` says yes, ``build`` must not
    raise; if it says no, ``build`` must raise rather than emit a kernel."""
    spec = _spec(**kw)
    ok, why = supports_attention_dense(spec, arch="gfx942")
    if ok:
        assert build_attention_dense(spec, arch="gfx942") is not None
    else:
        with pytest.raises(ValueError, match="unsupported"):
            build_attention_dense(spec, arch="gfx942")
        assert why


# --------------------------------------------------------------------------- #
# launch geometry
# --------------------------------------------------------------------------- #
def test_grid_and_block_geometry():
    s = _spec(seqlen_q=2048, num_query_heads=128, batch=1)
    assert attention_dense_grid(s) == (2048 // _BLOCK_M, 128, 1)
    assert attention_dense_block(s) == (s.num_waves * 64, 1, 1)
    assert attention_dense_block(s) == (_EXPECTED_WORKGROUP_SIZE, 1, 1)


def test_grid_covers_every_query_row_and_head():
    for sq, hq, batch in ((2048, 128, 1), (512, 16, 4), (4096, 40, 2)):
        s = _spec(
            seqlen_q=sq, seqlen_kv=sq, num_query_heads=hq, num_kv_heads=8, batch=batch
        )
        nqb, ghq, gb = attention_dense_grid(s)
        assert (
            nqb * _BLOCK_M == sq
        ), "grid must tile seqlen_q exactly (ragged is rejected)"
        assert (ghq, gb) == (hq, batch)


def test_persistent_grid_geometry_is_available_but_unbuildable():
    """``attention_dense_grid`` still describes the persistent grid (P4 will use it),
    but the body cannot be built for it yet -- keep those two facts in sync."""
    sp = _spec(persistent=True, num_persistent=228)
    assert attention_dense_grid(sp) == (228, 1, 1)
    assert not supports_attention_dense(sp, arch="gfx942")[0]
    with pytest.raises(ValueError, match="persistent is P4"):
        build_attention_dense(sp, arch="gfx942")


def test_extents_just_under_the_32_bit_limit_are_accepted():
    """Paired with the rejections above: without this, tightening _INT32_LIMIT to
    any smaller value would go unnoticed. 16128 = 63*256 keeps the tile multiples
    legal; kv=2113929216 B and qo=1056964608 elems are both < 2**31."""
    ok, why = supports_attention_dense(
        _spec(
            batch=64,
            seqlen_q=16128,
            seqlen_kv=16128,
            num_query_heads=8,
            num_kv_heads=8,
        ),
        arch="gfx942",
    )
    assert ok, why


@pytest.mark.parametrize(
    "kw,marker",
    [
        (dict(batch=0), "batch"),
        (dict(batch=-1), "batch"),
        (dict(seqlen_q=-256, seqlen_kv=-256), "seqlen"),
        (dict(num_query_heads=0, num_kv_heads=8), "num_query_heads"),
        (dict(num_query_heads=8, num_kv_heads=-1), "num_kv_heads"),
    ],
)
def test_supports_rejects_non_positive_extents(kw, marker):
    """Every dataclass validator is a divisibility test and Python's `%` is
    sign-following (-256 % 256 == 0, 8 % -1 == 0), so zero/negative shapes pass all
    of them. num_query_heads=0 is the worst: gqa == 0 emits `sdiv i32 %hq, 0`."""
    ok, why = supports_attention_dense(_spec(**kw), arch="gfx942")
    assert not ok, f"{kw} must be rejected (supports said ok)"
    assert marker in why, why


def test_supports_returns_rather_than_raises_for_block_n_zero():
    """__post_init__ evaluates `seqlen_kv % block_n` before validating block_n > 0,
    so block_n=0 raises ZeroDivisionError -- which must not escape a (bool, str) API."""
    base = AttentionDenseSpec(
        batch=1,
        seqlen_q=2048,
        seqlen_kv=2048,
        num_query_heads=128,
        num_kv_heads=8,
        head_size=128,
        causal=True,
        dtype="bf16",
        block_n=64,
    )
    object.__setattr__(base, "block_n", 0)  # frozen dataclass; bypass the ctor
    ok, why = supports_attention_dense(base, arch="gfx942")
    assert not ok and why


def test_tile_end_barrier_drains_lds_before_the_barrier():
    """C1 regression guard (AICK-1664), both V-feed paths.

    NBUF=1: the next iteration refills the SAME K/V LDS buffer, so the tile-END
    rendezvous must drain lgkmcnt BEFORE s_barrier. A bare s_barrier is NOT enough on
    gfx942 -- FeatureBackOffBarrier makes SIInsertWaitcnts skip the conservative
    pre-barrier drain, so the do_pv ds_reads stay in flight across it and another
    wave overwrites V_lds underneath them. Tested for BOTH:
      * naive V (D64 / bf16): V lands via async DMA, so the tile-START barrier can be
        bare + vmcnt(0) (making the DMA writes visible; draining lgkm would be dead);
      * conflict-free V / cfvst (D128 fp16, P1): V is published by an in-loop
        ds_write, so the tile-START rendezvous ALSO must be sync_lds_only -- a bare
        barrier there would race the perm_b32 store the same way the tile-end raced.
    """
    for spec, cfvst in (
        (_spec(head_size=64, dtype="fp16"), False),
        (_spec(head_size=128, dtype="fp16"), True),
    ):
        kernel = build_attention_dense(spec, arch="gfx942")
        loops = [o for o in kernel.body.ops if o.name == "scf.for"]
        assert len(loops) == 1, "expected exactly one KV loop"
        loop = loops[0]
        body = [o.name for o in loop.regions[0].ops]
        assert body[-1] == "scf.yield"
        assert body[-2] == "tile.sync_lds_only", (
            "tile-end rendezvous must be sync_lds_only (s_waitcnt lgkmcnt(0) + "
            f"s_barrier), not {body[-2]!r} -- a bare s_barrier races V_lds"
        )
        if cfvst:
            # cfvst tile-START: V is an in-loop ds_write, so publication is
            # sync_lds_only (lgkm drain + barrier), NOT a bare s_barrier, and it
            # follows a vmcnt(0) that drained the K DMA + V register loads.
            assert "tile.s_barrier_bare" not in body, (
                "cfvst tile-start must NOT use a bare s_barrier -- the V perm_b32 "
                "store needs an lgkm drain before publication (sync_lds_only)"
            )
            # >= 2 sync_lds_only: the store-publication one and the tile-end one.
            assert body.count("tile.sync_lds_only") >= 2
            assert "tile.s_waitcnt" in body
        else:
            # naive tile-START stays bare + vmcnt(0).
            i = body.index("tile.s_barrier_bare")
            assert body[i - 1] == "tile.s_waitcnt"
        # And the tile-end barrier must not be elidable: the elide pass targets
        # body_ops[-2].
        assert loop.attrs["elide_trailing_barrier"] is False


# --------------------------------------------------------------------------- #
# P2 exp2_fast gate (spill-driven) + fused/lazy rescale
# --------------------------------------------------------------------------- #
def _walk_op_names(op):
    """Yield every op name in the op tree (op + all nested region ops)."""
    yield op.name
    for region in getattr(op, "regions", ()):
        for child in region.ops:
            yield from _walk_op_names(child)


@pytest.mark.parametrize(
    "head_size, dtype, expected",
    [
        (64, "fp16", True),
        (128, "fp16", True),
        (64, "bf16", True),  # fused rescale gave the headroom (P2)
        (128, "bf16", False),  # spills on the .1k schedule even post-fused-rescale
    ],
)
def test_exp2_fast_gate_matches_the_spill_measured_matrix(head_size, dtype, expected):
    """The exp2_fast decision is a spill fact, not a preference.

    exp2_fast is numerically safe for every config (both softmax args <= 0), so the
    gate exists ONLY to keep occupancy: its sooner-available result raises register
    pressure, and bf16 D128's `.1k` MFMA schedule spills over the waves-per-eu=2 cap
    (measured 175->256 VGPR / 22 spill) even after the P2 fused rescale freed ~28
    VGPR. Every other config has the headroom. This pins the exact enabled set so a
    future edit that flips one arm has to update this matrix on purpose.
    """
    assert _p0_use_exp2_fast(head_size, dtype) is expected


@pytest.mark.parametrize(
    "head_size, dtype",
    [(128, "fp16"), (64, "fp16"), (64, "bf16"), (128, "bf16")],
)
def test_softmax_emits_the_gated_exp2_intrinsic(head_size, dtype):
    """The gate actually selects the intrinsic in the emitted IR.

    exp2_fast lowers to ``math.exp2_fast`` (llvm.amdgcn.exp2.f32 -> one v_exp_f32);
    plain exp2 lowers to ``math.exp2`` (llvm.exp2.f32, guarded range reduction). The
    softmax path must emit exactly one family, matching :func:`_p0_use_exp2_fast`, so
    the bf16-D128 spill guard is not silently defeated by an IR-level fallback.
    Parametrized (not a plain loop) so each config's failure is isolated -- the gate
    boundary case bf16 D128 must be reported even if an earlier config regresses.
    """
    spec = _spec(head_size=head_size, dtype=dtype)
    kernel = build_attention_dense(spec, arch="gfx942")
    names = [n for op in kernel.body.ops for n in _walk_op_names(op)]
    has_fast = "math.exp2_fast" in names
    has_plain = "math.exp2" in names
    if _p0_use_exp2_fast(head_size, dtype):
        assert has_fast and not has_plain, (
            f"{dtype} D{head_size}: gate says exp2_fast but IR has "
            f"fast={has_fast} plain={has_plain}"
        )
    else:
        assert has_plain and not has_fast, (
            f"{dtype} D{head_size}: gate says plain exp2 but IR has "
            f"fast={has_fast} plain={has_plain}"
        )


@pytest.mark.parametrize(
    "head_size, dtype", [(128, "fp16"), (64, "fp16"), (64, "bf16")]
)
def test_fused_rescale_casts_each_p_exactly_once(head_size, dtype):
    """P2 fused/lazy rescale: exp2 -> l_local accumulate -> cast -> pack in one pass.

    The pre-P2 code built a full f32 ``p_vals`` matrix (N_SUB*16 values), reduced it
    into ``l_local``, THEN cast+packed it in a separate ``relayout_p`` pass -- holding
    all those f32 regs live across both. The fused rescale casts each P exp result to
    ``dtype`` inline instead, so in the loop there is exactly ONE ``arith.cast_f32_to``
    per P element and the exp count is that plus one (alpha). More casts than P
    elements would mean a second materialization pass (the live-range regression this
    change removed) survived. The P-element count is ``N_SUB*16 = (block_n//32)*16``,
    which is head-size-independent, so it is derived from the spec here rather than
    hardcoded -- the assertion stays honest if block_n ever changes. Covers both the
    cfvst path (D128 fp16) and the naive-V path (D64, bf16) so neither can regress.

    NOTE: this pins the cast/exp COUNTS, not the ``l_local`` accumulation ORDER. The
    bit-identical-order claim is a numeric property verified by the GPU cohort (this
    file is not a numeric lane -- see the module docstring), not by op counting.
    """
    spec = _spec(head_size=head_size, dtype=dtype)
    n_p = spec.block_n // 32 * 16  # N_SUB * 16 -- P elements the softmax exps produce
    kernel = build_attention_dense(spec, arch="gfx942")
    loops = [o for o in kernel.body.ops if o.name == "scf.for"]
    assert len(loops) == 1
    names = list(_walk_op_names(loops[0]))
    n_exp = names.count("math.exp2_fast") + names.count("math.exp2")
    n_cast = names.count("arith.cast_f32_to")
    # exactly one P cast per P element -- no second pass (o_acc rescale uses
    # fmul/vec_pack, not cast; the final output cast lives outside the loop).
    assert n_cast == n_p, (
        f"{dtype} D{head_size}: expected {n_p} P casts (one per element, fused), "
        f"got {n_cast} -- a standalone relayout/materialization pass may have survived"
    )
    # one exp per P element plus alpha's exp.
    assert (
        n_exp == n_p + 1
    ), f"{dtype} D{head_size}: expected {n_p + 1} exps (P + alpha), got {n_exp}"
