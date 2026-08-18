# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only tests for the gfx942 dense prefill kernel.

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

The gfx942 golden-IR lane is a SEPARATE file, ``test_attention_dense_gfx942_golden.py``
with its own fixture (``tests/golden/attention_dense_gfx942_ir_sha256.json``). The
gfx950 ``test_attention_dense_golden.py`` and its fixture stay gfx942-free, which is
what keeps the gfx950 goldens untouched by anything in this file.
"""

import dataclasses
import hashlib

import pytest

from kernels.gfx942.attention_dense import (
    AttentionDenseSpec,
    Gfx942DenseTuning,
    attention_dense_block,
    attention_dense_grid,
    build_attention_dense,
    gfx942_kernel_name,
    run_attention_dense_torch,
    supports_attention_dense,
    _BLOCK_M,
    _k_group_pad_active,
    _k_group_stride,
    _use_exp2_fast,
    _tuned_waves_per_eu,
)

# Query rows per CTA baked into the P0 body; block_n must divide it. IMPORTED, not
# restated: a local literal here would silently stop testing the shipped geometry the
# moment the kernel's tile changed.
_EXPECTED_WORKGROUP_SIZE = (_BLOCK_M // 32) * 64  # 8 wave64s = 512 threads

# The shipped tuning. Constructed from the class default rather than restated field
# by field, so it cannot drift from what dispatch actually builds.
_DEFAULT_TUNING = Gfx942DenseTuning()


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


def _lower(kd) -> str:
    """Lower a built ``KernelDef`` to LLVM IR text. CPU-only -- no comgr, no GPU."""
    from rocke.core.lower_llvm import (
        _lower_kernel_to_llvm_python,
        _resolve_llvm_flavor,
    )

    return _lower_kernel_to_llvm_python(
        kd, arch="gfx942", llvm_flavor=_resolve_llvm_flavor()
    )


def _ir_body_sha(spec, tuning=_DEFAULT_TUNING) -> str:
    """SHA of the lowered IR with the kernel SYMBOL normalised out.

    The symbol appears in the ``define``, in the ``@smem_pool.<name>`` global and in
    every reference to either, so WITHOUT this normalisation any name change would
    make the IR text trivially differ and the injectivity property below would be
    vacuous -- it would only ever be observing the name change it is trying to
    justify. Same idiom as the out-of-tree ``ir_body_probe`` that proved the P0->gfx942
    identifier rename codegen-neutral.
    """
    kd = build_attention_dense(spec, arch="gfx942", tuning=tuning)
    body = _lower(kd).replace(kd.name, "KERNEL")
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


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
    assert kd.name == gfx942_kernel_name(spec)
    assert kd.attrs["max_workgroup_size"] == _EXPECTED_WORKGROUP_SIZE


def test_build_rejects_non_gfx942():
    with pytest.raises(NotImplementedError, match="gfx942-only"):
        build_attention_dense(_spec(), arch="gfx950")


# --------------------------------------------------------------------------- #
# kernel-name identity: batch is baked into the buffer extents but omitted from
# the shared kernel_name(), so a name-keyed cache would serve the B=1 binary for
# a B>1 launch and read out of bounds. gfx942_kernel_name() is the guard.
# --------------------------------------------------------------------------- #
def test_kernel_name_pins_the_two_shipped_cache_collisions():
    """Regression pin for the two collisions this kernel has ACTUALLY shipped.

    batch and waves_per_eu are both baked into the artifact but omitted from the
    shared ``kernel_name()``: batch sizes the buffer-resource extents, waves_per_eu is
    emitted as ``amdgpu-waves-per-eu`` and changes register allocation. Either one
    colliding in a name-keyed cache serves the wrong binary.

    This is the narrow historical pin. The GENERAL property -- no field of either
    dataclass may move the IR without moving the name -- is
    :func:`test_kernel_name_covers_every_baked_parameter` below, which discovers the
    fields by reflection and therefore cannot go stale when one is added.
    """
    assert gfx942_kernel_name(_spec(waves_per_eu=2)) != gfx942_kernel_name(
        _spec(waves_per_eu=3)
    )
    a = build_attention_dense(_spec(waves_per_eu=2), arch="gfx942")
    c = build_attention_dense(_spec(waves_per_eu=3), arch="gfx942")
    assert a.attrs["waves_per_eu"] != c.attrs["waves_per_eu"]
    assert a.name != c.name


# --------------------------------------------------------------------------- #
# kernel-name injectivity, PROPERTY-BASED over both dataclasses
#
# The bug class: `_DENSE_LAUNCHER_CACHE` is keyed on the kernel name and its
# `assert art.kernel_name == key` PASSES on a collision, so two IR-distinct configs
# that share a name silently serve the FIRST-compiled binary. This kernel has
# shipped that bug twice (batch, then waves_per_eu) because the guard test
# hand-enumerated the fields it checked -- so every field added afterwards was
# uncovered by construction. The tables below are driven by reflection over
# ``__dataclass_fields__`` and a completeness test makes a new field FAIL until it
# is given a perturbation, which is the whole point.
# --------------------------------------------------------------------------- #

# Fields the builder rejects outright in every legal combination: there is no second
# legal value to perturb to, so the property is untestable (and uninteresting -- an
# unbuildable config cannot collide with anything). Rejection itself is covered by
# test_supports_rejects_modes_deferred_to_later_phases.
_UNBUILDABLE_SPEC_FIELDS = frozenset({"varlen", "ragged", "sliding_window"})

# Fields that move the NAME but never the gfx942 IR. Over-naming is SAFE -- it costs
# a duplicate compile and can never serve a wrong binary -- so these are recorded,
# not asserted against. lazy_rescale is a gfx950-only lever (this builder never reads
# spec.lazy_rescale) that the SHARED kernel_name() nevertheless tags with `lazyrs`.
_NAME_ONLY_SPEC_FIELDS = frozenset({"lazy_rescale"})

# Second LEGAL values per field. Every candidate is filtered through
# supports_attention_dense before use, so a candidate that is illegal for a given
# base is skipped rather than asserted on; listing several per field keeps coverage
# when one base rejects one of them (e.g. use_cfvst=True is legal only at fp16 D128).
_SPEC_PERTURBATIONS = {
    "batch": (2, 4),
    "seqlen_q": (4096,),
    "seqlen_kv": (4096,),
    "num_query_heads": (32, 8),
    "num_kv_heads": (8, 2),
    "head_size": (64, 128),
    "causal": (False, True),
    "dtype": ("bf16", "fp16"),
    "sliding_window": (),  # unbuildable -- see _UNBUILDABLE_SPEC_FIELDS
    "ragged": (),  # unbuildable
    "varlen": (),  # unbuildable
    "block_n": (32, 128),
    "waves_per_eu": (3, 4),
    "lds_k_group_pad": (0, 16),
    "persistent": (True, False),
    "num_persistent": (152, 304),
    "interleave": (True, False),
    "persist_decode": ("qb_major", "hkv_major"),
    "lazy_rescale": (False, True),
}

_TUNING_PERTURBATIONS = {
    "block_m": (128, 512),
    "lds_row_pad": (0, 16),
    "v_row_pad": (0, 16),
    "use_cfvst": (False, True),
    "use_exp2_fast": (False, True),
    "waves_per_eu": (4, 3),
    "iglp": (True, False),
}

# Base configurations the perturbations are applied to, one field at a time. Chosen
# so every structurally distinct arm of the builder is a base: cfvst on (fp16 D128),
# cfvst off at D128 (bf16, exp2_fast), both D64 dtypes (packed 2-rows-per-DMA + the
# K row-group pad + the wpe=4 tune), and the P4 persistent grid -- without which
# num_persistent / interleave / persist_decode are inert and their coverage vacuous.
_INJECTIVITY_BASES = {
    "d128_fp16_cfvst": dict(head_size=128, dtype="fp16"),
    "d128_bf16_naive": dict(head_size=128, dtype="bf16"),
    "d64_fp16": dict(head_size=64, dtype="fp16"),
    "d64_bf16": dict(head_size=64, dtype="bf16"),
    "persistent_d128_fp16": dict(
        head_size=128,
        dtype="fp16",
        persistent=True,
        num_persistent=228,
        seqlen_q=4096,
        seqlen_kv=4096,
        num_query_heads=32,
    ),
}


def _injectivity_field_ids():
    """(kind, field) for every field of BOTH dataclasses, by reflection.

    Reflection at COLLECTION time is what makes this suite self-extending: a field
    added to either dataclass immediately becomes its own pytest case, and (having no
    perturbation entry) fails the completeness test below until it is given one.
    """
    return [
        pytest.param("spec", f, id=f"spec.{f}")
        for f in AttentionDenseSpec.__dataclass_fields__
    ] + [
        pytest.param("tuning", f, id=f"tuning.{f}")
        for f in Gfx942DenseTuning.__dataclass_fields__
    ]


def _perturbation_table(kind: str) -> dict:
    return _SPEC_PERTURBATIONS if kind == "spec" else _TUNING_PERTURBATIONS


def test_perturbation_table_covers_every_dataclass_field():
    """Anti-staleness gate. Reflection finds the fields; this asserts the tables know
    them all, so ADDING a field to either dataclass fails here with a clear message
    instead of silently going uncovered -- which is exactly how ``batch`` and then
    ``waves_per_eu`` each shipped a launcher-cache collision."""
    for kind, cls in (
        ("spec", AttentionDenseSpec),
        ("tuning", Gfx942DenseTuning),
    ):
        declared = set(cls.__dataclass_fields__)
        tabled = set(_perturbation_table(kind))
        assert declared == tabled, (
            f"{cls.__name__}: perturbation table is out of sync with the dataclass "
            f"(missing {sorted(declared - tabled)}, stale {sorted(tabled - declared)})"
            " -- every field must be given a second LEGAL value (or an empty tuple "
            "plus an entry in _UNBUILDABLE_SPEC_FIELDS) so the name-injectivity "
            "property below actually covers it"
        )


@pytest.mark.parametrize("kind, field", _injectivity_field_ids())
def test_kernel_name_covers_every_baked_parameter(kind, field):
    """IR body differs => kernel name differs, for EVERY field of both dataclasses.

    One-directional on purpose. The dangerous direction is IR-differs / name-identical:
    ``_DENSE_LAUNCHER_CACHE`` is keyed on the name and would serve the first-compiled
    binary for the second config -- a silent wrong answer (or an out-of-bounds read,
    which is what the ``batch`` collision was). The other direction, name-differs /
    IR-identical, is merely a duplicate compile and is therefore ALLOWED: it is the
    normal state for ``lazy_rescale`` (a gfx950 lever this builder ignores, tagged by
    the shared ``kernel_name()``) and for ``lds_row_pad`` / ``v_row_pad`` at D64,
    where the tag is emitted on "differs from default" rather than on "is live in
    this build" -- deliberately, because over-tagging is free and under-tagging is
    the bug.

    ``lds_k_group_pad`` is likewise correctly inert at D128 (it pads per row there,
    via ``_lds_row_stride``), so it must move neither the name nor the IR at the D128
    bases and both at the D64 ones. The test tolerates that by asserting the property
    per (base, value) rather than requiring a difference everywhere.

    The IR is compared with the symbol normalised out (:func:`_ir_body_sha`) --
    otherwise the name change alone would make the IR differ and the implication
    would hold vacuously.
    """
    candidates = _perturbation_table(kind)[field]
    tried = 0
    ir_moved = 0
    violations = []
    for base_id, base_kw in _INJECTIVITY_BASES.items():
        base_spec = _spec(**base_kw)
        base_tuning = _DEFAULT_TUNING
        ok, why = supports_attention_dense(base_spec, arch="gfx942", tuning=base_tuning)
        assert ok, f"base {base_id} must be in scope: {why}"
        base_name = gfx942_kernel_name(base_spec, base_tuning)
        base_sha = _ir_body_sha(base_spec, base_tuning)
        current = getattr(base_spec if kind == "spec" else base_tuning, field)
        for value in candidates:
            if value == current:
                continue
            try:
                if kind == "spec":
                    alt_spec = dataclasses.replace(base_spec, **{field: value})
                    alt_tuning = base_tuning
                else:
                    alt_spec = base_spec
                    alt_tuning = dataclasses.replace(base_tuning, **{field: value})
            except (ValueError, ZeroDivisionError):
                continue  # the dataclass validator rejected it: not a legal value
            ok, _ = supports_attention_dense(alt_spec, arch="gfx942", tuning=alt_tuning)
            if not ok:
                continue  # out of scope for this base; another base may accept it
            tried += 1
            alt_name = gfx942_kernel_name(alt_spec, alt_tuning)
            alt_sha = _ir_body_sha(alt_spec, alt_tuning)
            if alt_sha != base_sha:
                ir_moved += 1
                if alt_name == base_name:
                    violations.append(f"{base_id}: {kind}.{field}={value!r}")
    assert not violations, (
        f"{kind}.{field} changes the emitted IR but NOT the kernel name at "
        f"{violations} -- two distinct binaries would share one _DENSE_LAUNCHER_CACHE "
        f"key and the second launch would be served the first one's HSACO. Tag the "
        f"field in gfx942_kernel_name (spec fields: via _tuning_name_tags' sibling "
        f"logic or the shared kernel_name; tuning fields: in _tuning_name_tags)"
    )
    # Anti-vacuity, per field: the property above is trivially true for a field that
    # was never legally perturbed, so require that each field is either declared
    # unbuildable or actually exercised somewhere.
    if field in _UNBUILDABLE_SPEC_FIELDS and kind == "spec":
        assert tried == 0, (
            f"spec.{field} is listed in _UNBUILDABLE_SPEC_FIELDS but a perturbation "
            f"was accepted -- if it is now supported, give it real candidates and "
            f"drop it from that set"
        )
        return
    assert tried, (
        f"{kind}.{field}: no candidate in {candidates!r} was accepted by "
        f"supports_attention_dense at ANY base, so the injectivity property was not "
        f"exercised for this field at all. Add a legal second value."
    )
    # ... and that it actually reaches codegen somewhere, unless it is a documented
    # name-only field. A field that never moves IR anywhere is either dead in this
    # builder (say so here) or the perturbations are too weak to reach it.
    if kind == "spec" and field in _NAME_ONLY_SPEC_FIELDS:
        assert ir_moved == 0, (
            f"spec.{field} is documented in _NAME_ONLY_SPEC_FIELDS as never reaching "
            f"gfx942 codegen, but it moved the IR in {ir_moved} case(s) -- it is now "
            f"a live lever; remove it from that set"
        )
        return
    assert ir_moved, (
        f"{kind}.{field}: {tried} legal perturbation(s) were built and NONE changed "
        f"the emitted IR. Either the field is dead in this builder (add it to "
        f"_NAME_ONLY_SPEC_FIELDS with a reason) or the candidates are too weak"
    )


def test_kernel_name_is_batch_unique():
    names = {gfx942_kernel_name(_spec(batch=b)) for b in (1, 2, 4, 8)}
    assert len(names) == 4, f"batch must disambiguate the kernel name, got {names}"
    assert "_b4_" in gfx942_kernel_name(_spec(batch=4))


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
        # persistent is NOT here anymore -- it is supported (P4). See the persistent
        # build/decode tests below.
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
    """block_n=128 at D128/bf16 needs 2*128*(128+8)*2 = 69632 B > the 64 KB gfx942 LDS.
    Without this gate it reaches comgr and dies with an opaque CODEGEN abort.

    The pad is passed EXPLICITLY rather than inherited from the default tuning: the
    arithmetic above is what makes this spec over-budget, and at ``lds_row_pad=0`` the
    same spec lands on exactly 65536 B -- which is NOT ``> capacity``, so the gate
    flips to ACCEPT and this test's premise would silently invert into a tautology.
    The pad=0 boundary is pinned deliberately in the test below.
    """
    tuning = dataclasses.replace(_DEFAULT_TUNING, lds_row_pad=8)
    spec = _spec(block_n=128, head_size=128, dtype="bf16")
    ok, why = supports_attention_dense(spec, arch="gfx942", tuning=tuning)
    assert not ok and "LDS" in why
    # the arithmetic the rejection rests on, spelled out so a capacity/pad change is loud
    assert "69632" in why, why


def test_lds_budget_boundary_is_accepted_at_exactly_capacity():
    """The gate is ``> capacity``, so a footprint of EXACTLY 65536 B is legal.

    Same bn128/D128/bf16 spec as above with ``lds_row_pad=0``: K and V are each
    128*128*2 = 32768 B, totalling exactly the gfx942 LDS capacity. Determined from
    the code, not assumed -- and it is the CORRECT answer: LDS is allocated in whole,
    a workgroup that uses all 65536 B is legal (it just pins 1 WG/CU), and the emitted
    ``addrspace(3)`` pool really is 65536 B (asserted in the LDS-pool test file).
    Pinned here so the pad-value sweep cannot flip the gate's boundary semantics by
    accident, and so the over-budget test above keeps a real margin.
    """
    from kernels.common.attention_arch import attention_lds_capacity_bytes

    from kernels.gfx942.attention_dense import _lds_bytes

    tuning = dataclasses.replace(_DEFAULT_TUNING, lds_row_pad=0)
    spec = _spec(block_n=128, head_size=128, dtype="bf16")
    capacity = attention_lds_capacity_bytes("gfx942")
    assert _lds_bytes(spec, tuning) == capacity, "premise: exactly at capacity"
    ok, why = supports_attention_dense(spec, arch="gfx942", tuning=tuning)
    assert ok, f"a footprint of exactly {capacity} B must be accepted: {why}"
    # supports() => build() must hold at the boundary too.
    assert build_attention_dense(spec, arch="gfx942", tuning=tuning) is not None


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
    """D256 is served by its own wide-atom candidates. The dataclass is the stricter
    guard -- it rejects at construction, before ``supports_attention_dense`` is
    reachable."""
    with pytest.raises(ValueError, match=r"head_size must be 64 or 128"):
        _spec(head_size=256)


# --------------------------------------------------------------------------- #
# the contract: supports is the single gate
# --------------------------------------------------------------------------- #

# Tuning fields deliberately left ungated. Recorded rather than assumed so that
# adding a gate (or forgetting one) is a visible, reviewed decision.
#   use_exp2_fast: numerically safe in both directions here (both softmax args are
#     always <= 0), so it is a perf A/B, not a correctness or tile-exactness
#     hazard. Gating it would make the config unsweepable.
#   iglp: a compile-time scheduler directive (llvm.amdgcn.iglp.opt) that leaves no
#     runtime instruction and is legal on every config.
_TUNING_FIELDS_WITHOUT_A_REJECTED_REGION = frozenset({"use_exp2_fast", "iglp"})

# Rows are ``(spec kwargs, tuning kwargs)``. The tuning half is NOT optional
# decoration: supports_attention_dense validates the Gfx942DenseTuning struct too, so
# the contract "supports => build" has to be exercised on the knobs a sweep harness
# will actually turn. Every tuning field gets at least two rows -- its default (or a
# second accepted value) and one that must land in the REJECTED region -- so a new
# gate, or a gate that silently stops firing, shows up here. The two fields with no
# rejected region by design, and why, are recorded once at
# _TUNING_FIELDS_WITHOUT_A_REJECTED_REGION above.
_CONTRACT_GRID = [
    (dict(), dict()),
    (dict(dtype="fp16"), dict()),
    (dict(head_size=64), dict()),
    (dict(causal=False), dict()),
    (dict(num_query_heads=40, num_kv_heads=8), dict()),
    (dict(block_n=32), dict()),
    (dict(block_n=96, seqlen_kv=1536), dict()),
    (dict(block_n=128), dict()),
    (dict(block_n=128, head_size=64), dict()),
    (dict(block_n=512, seqlen_kv=2048), dict()),
    (dict(persistent=True, num_persistent=228), dict()),
    (dict(varlen=True), dict()),
    (dict(seqlen_q=1000, seqlen_kv=1000, ragged=True), dict()),
    (dict(sliding_window=64), dict()),
    (dict(batch=4), dict()),
    (dict(batch=64, seqlen_q=16384, seqlen_kv=16384, num_kv_heads=8), dict()),
    # --- tuning: block_m ---
    (dict(), dict(block_m=_BLOCK_M)),  # the default, spelled out
    (dict(), dict(block_m=128)),  # accepted: halves the query tile
    (dict(), dict(block_m=48)),  # REJECTED: not a multiple of 32 (wave floor)
    (dict(), dict(block_m=1024)),  # REJECTED: 2048-thread CTA > the 1024 max
    # REJECTED: 2048 % 320 != 0, so the last query tile runs past seqlen_q and Q/O
    # (plain global_load_vN, no bounds check) read and write out of bounds. 320 is a
    # multiple of 32 and of block_n and fits a 640-thread CTA, so this row reaches
    # the seqlen check rather than tripping an earlier one.
    (dict(seqlen_q=2048), dict(block_m=320)),
    # --- tuning: lds_row_pad / v_row_pad (the pad-value sweep surface) ---
    (dict(), dict(lds_row_pad=0)),  # accepted: the unpadded A/B arm
    (dict(), dict(lds_row_pad=2)),  # REJECTED: not a multiple of 4 elements
    (dict(), dict(v_row_pad=16)),  # accepted
    (dict(), dict(v_row_pad=-4)),  # REJECTED: negative
    # --- tuning: use_cfvst (the one tri-state with a rejected direction) ---
    (dict(dtype="fp16"), dict(use_cfvst=False)),  # accepted: OFF is always legal
    (dict(head_size=64), dict(use_cfvst=True)),  # REJECTED: policy says off at D64
    (dict(dtype="bf16", head_size=128), dict(use_cfvst=True)),  # REJECTED: spills
    # --- tuning: use_exp2_fast (no rejected region -- see the comment above) ---
    (dict(), dict(use_exp2_fast=False)),
    (dict(), dict(use_exp2_fast=True)),  # accepted: policy is True for every config
    # --- tuning: waves_per_eu ---
    (dict(), dict(waves_per_eu=4)),  # accepted
    (dict(), dict(waves_per_eu=0)),  # REJECTED: must resolve positive
    # --- tuning: iglp (no rejected region -- see the comment above) ---
    (dict(), dict(iglp=False)),
    (dict(dtype="fp16"), dict(iglp=True)),
]


def _grid_id(row) -> str:
    """Value-bearing id: keys alone collide (several block_n / block_m cases)."""
    spec_kw, tuning_kw = row
    parts = [f"{k}{v}" for k, v in sorted(spec_kw.items())]
    parts += [f"t.{k}{v}" for k, v in sorted(tuning_kw.items())]
    return "-".join(parts) or "base"


def _grid_row(row):
    """Materialise one grid row into ``(spec, tuning)``."""
    spec_kw, tuning_kw = row
    return _spec(**spec_kw), dataclasses.replace(_DEFAULT_TUNING, **tuning_kw)


def test_contract_grid_exercises_both_sides():
    """Guard against vacuity: if ``supports`` regressed to all-False, the contract
    test below would still pass on every case.

    The thresholds are expressed as a FRACTION of the grid, not as the frozen
    ``>= 6`` this started as: the grid grows every time a spec or tuning field is
    added, and a fixed floor stops being a real vacuity check the moment it is a
    small fraction of the rows. Absolute floors are kept underneath so a shrinking
    grid cannot weaken the guard either.
    """
    verdicts = [
        supports_attention_dense(spec, arch="gfx942", tuning=tuning)[0]
        for spec, tuning in map(_grid_row, _CONTRACT_GRID)
    ]
    accepted = [_grid_id(r) for r, v in zip(_CONTRACT_GRID, verdicts) if v]
    rejected = [_grid_id(r) for r, v in zip(_CONTRACT_GRID, verdicts) if not v]
    floor = max(8, len(_CONTRACT_GRID) // 4)
    assert len(accepted) >= floor, (
        f"grid must exercise the accepted region with >= {floor} of "
        f"{len(_CONTRACT_GRID)} rows, got {len(accepted)}: {accepted}"
    )
    assert len(rejected) >= floor, (
        f"grid must exercise the rejected region with >= {floor} of "
        f"{len(_CONTRACT_GRID)} rows, got {len(rejected)}: {rejected}"
    )


def test_contract_grid_covers_every_tuning_field_on_both_sides():
    """Every :class:`Gfx942DenseTuning` field must appear in the grid, and -- unless
    it is documented as having no rejected region -- on BOTH sides of the gate.

    Reflection over ``__dataclass_fields__`` again, for the same reason as the
    injectivity table: a knob added to the struct without a supports() gate (or with
    a gate nobody exercised) is exactly the dispatch fall-through hole that function
    exists to close, and a hand-written list of field names would not notice.
    """
    accepted_fields, rejected_fields = set(), set()
    for row in _CONTRACT_GRID:
        _, tuning_kw = row
        spec, tuning = _grid_row(row)
        ok, _ = supports_attention_dense(spec, arch="gfx942", tuning=tuning)
        (accepted_fields if ok else rejected_fields).update(tuning_kw)
    declared = set(Gfx942DenseTuning.__dataclass_fields__)
    missing = declared - (accepted_fields | rejected_fields)
    assert not missing, (
        f"Gfx942DenseTuning fields absent from _CONTRACT_GRID: {sorted(missing)} -- "
        f"add at least two rows per field (the default and one value that must be "
        f"REJECTED), or document it in _TUNING_FIELDS_WITHOUT_A_REJECTED_REGION"
    )
    assert declared - accepted_fields == set(), (
        f"no ACCEPTED grid row for {sorted(declared - accepted_fields)}: the "
        f"supports => build contract is never exercised for those knobs"
    )
    unexercised = declared - rejected_fields - _TUNING_FIELDS_WITHOUT_A_REJECTED_REGION
    assert not unexercised, (
        f"no REJECTED grid row for {sorted(unexercised)}: either add a value that "
        f"supports_attention_dense must reject, or -- if the field genuinely has no "
        f"illegal value -- record it in _TUNING_FIELDS_WITHOUT_A_REJECTED_REGION with "
        f"the reason"
    )
    strays = _TUNING_FIELDS_WITHOUT_A_REJECTED_REGION & rejected_fields
    assert not strays, (
        f"{sorted(strays)} are declared to have no rejected region but a grid row "
        f"was rejected -- they are gated now; drop them from that set"
    )


@pytest.mark.parametrize("row", _CONTRACT_GRID, ids=_grid_id)
def test_supports_true_implies_build_succeeds(row):
    """The load-bearing invariant. If ``supports`` says yes, ``build`` must not
    raise; if it says no, ``build`` must raise rather than emit a kernel.

    Exercised over the tuning struct as well as the spec: ``supports`` validates both,
    so a knob that only the BUILDER rejects reopens the dispatch fall-through hole
    this contract closes."""
    spec, tuning = _grid_row(row)
    ok, why = supports_attention_dense(spec, arch="gfx942", tuning=tuning)
    if ok:
        kd = build_attention_dense(spec, arch="gfx942", tuning=tuning)
        assert kd is not None
        # ...and the emitted symbol must be the name the launcher cache will key on
        # for this (spec, tuning) -- otherwise the sweep point compiles under one
        # identity and is looked up under another.
        assert kd.name == gfx942_kernel_name(spec, tuning)
    else:
        with pytest.raises(ValueError, match="unsupported"):
            build_attention_dense(spec, arch="gfx942", tuning=tuning)
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


def test_persistent_grid_geometry_and_build():
    """P4: ``attention_dense_grid`` is the 1-D ``num_persistent`` grid, and the
    persistent body now BUILDS (was rejected as P4 in P0-P3). Keep those in sync."""
    sp = _spec(persistent=True, num_persistent=228)
    assert attention_dense_grid(sp) == (228, 1, 1)
    ok, _ = supports_attention_dense(sp, arch="gfx942")
    assert ok
    assert build_attention_dense(sp, arch="gfx942") is not None


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
    """C1 regression guard, both V-feed paths.

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
# P2 exp2_fast policy + fused rescale
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
        (128, "bf16", True),  # re-measured: no spill -> last holdout enabled
    ],
)
def test_exp2_fast_is_enabled_for_every_config(head_size, dtype, expected):
    """exp2_fast is enabled for EVERY config. It is numerically safe everywhere (both
    softmax args -- alpha's m_i - m_new and p's s - m_new -- are <= 0, exactly
    exp2_fast's precondition, independent of head_size/dtype) and a strict VALU win.
    bf16 D128 was the last holdout on a measurement that no longer reproduces on the
    current kernel (re-measured: 0 scratch and lower VGPR than plain exp2 on both
    grids, numerically identical), so it is removed. This pins the enabled set (now
    all configs) so a future edit that re-introduces a rejected arm has to update
    this matrix on purpose.
    """
    assert _use_exp2_fast(head_size, dtype) is expected


@pytest.mark.parametrize(
    "head_size, dtype", [(128, "fp16"), (64, "fp16"), (64, "bf16"), (128, "bf16")]
)
@pytest.mark.parametrize("use_exp2_fast", [False, True])
def test_softmax_emits_the_gated_exp2_intrinsic(head_size, dtype, use_exp2_fast):
    """The resolved exp2_fast decision actually selects the intrinsic in the IR.

    exp2_fast lowers to ``math.exp2_fast`` (llvm.amdgcn.exp2.f32 -> one v_exp_f32);
    plain exp2 lowers to ``math.exp2`` (llvm.exp2.f32, guarded range reduction). The
    softmax path must emit exactly one family, matching the resolved tuning. This is
    parametrized on the tuning rather than the policy: the policy is now True for
    every config, so branching on it would make the plain-exp2 arm dead and leave
    ``b.exp2`` untested -- yet ``use_exp2_fast=False`` is still an accepted, swept
    override. Forcing both values keeps both codegen arms pinned. Each config is
    isolated, so a failure is reported even if an earlier one regresses.
    """
    spec = _spec(head_size=head_size, dtype=dtype)
    tuning = Gfx942DenseTuning(use_exp2_fast=use_exp2_fast)
    kernel = build_attention_dense(spec, arch="gfx942", tuning=tuning)
    names = [n for op in kernel.body.ops for n in _walk_op_names(op)]
    has_fast = "math.exp2_fast" in names
    has_plain = "math.exp2" in names
    if tuning.resolved_use_exp2_fast(spec):
        assert has_fast and not has_plain, (
            f"{dtype} D{head_size}: tuning says exp2_fast but IR has "
            f"fast={has_fast} plain={has_plain}"
        )
    else:
        assert has_plain and not has_fast, (
            f"{dtype} D{head_size}: tuning says plain exp2 but IR has "
            f"fast={has_fast} plain={has_plain}"
        )


@pytest.mark.parametrize(
    "head_size, dtype", [(128, "fp16"), (64, "fp16"), (64, "bf16"), (128, "bf16")]
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


# --------------------------------------------------------------------------- #
# P3 waves-per-eu occupancy tune
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "head_size, dtype, expected",
    [
        (64, "bf16", 4),  # 2 WG/CU (215->117 VGPR, 0 spill) -- +~50% at long seq
        (64, "fp16", 2),  # wpe=3 reaches 2 WG/CU but loses more ILP than it buys
        (128, "fp16", 2),  # LDS-bound at ~35 KB: no wpe reaches a 2nd WG/CU
        (128, "bf16", 2),  # LDS-bound: same
    ],
)
def test_waves_per_eu_selector_matches_measured_matrix(head_size, dtype, expected):
    """The per-config waves-per-eu is a measured occupancy fact, not a preference.

    Only bf16 D64 is overridden (to 4); every other config keeps the default 2. This
    pins the exact matrix so a future edit that flips one arm must update it on purpose
    (see :func:`_tuned_waves_per_eu` for the per-config measurement rationale).
    """
    assert _tuned_waves_per_eu(head_size, dtype) == expected


def test_build_bakes_the_tuned_waves_per_eu_attribute():
    """The tuned value reaches the emitted ``amdgpu-waves-per-eu`` kernel attribute.

    waves_per_eu changes register allocation and is baked into both the kernel_name
    (``wpe{N}``) and the attribute, so a spec built at waves_per_eu=4 must emit the 4
    attribute -- otherwise the name and the binary disagree (the cache-collision class
    of bug guarded elsewhere by :func:`gfx942_kernel_name`).
    """
    spec = _spec(head_size=64, dtype="bf16", waves_per_eu=4)
    kernel = build_attention_dense(spec, arch="gfx942")
    assert kernel.attrs.get("waves_per_eu") == 4
    # anchored on the full baked suffix, not a bare "_wpe4" (which "_wpe14" would
    # also match): batch + arch + wpe are all part of the identity.
    assert gfx942_kernel_name(spec).endswith("_gfx942_b1_wpe4")


def test_dispatch_applies_gfx942_waves_per_eu_tuning_and_leaves_gfx950_alone():
    """The gfx942 dispatch spec factory applies the tune; gfx950 stays at the default.

    The tune lives in gfx942's OWN ``_dense_spec`` (``dispatch/attention/gfx942.py``),
    so the kernel_name ``wpe`` tag and the emitted attribute agree on the dispatched
    path (``dense_spec_for_request`` -> ``run_attention_dense_torch``). gfx950 has a
    separate factory in its own arch module which MUST keep the spec default
    (waves_per_eu=2) -- this is the do-not-touch-gfx950 guard as an executable
    assertion. Both are exercised here precisely because they are now two functions:
    the guard is that they stayed different in the intended direction only.
    """
    from dispatch.attention import AttentionRequest
    from dispatch.attention.gfx942 import _dense_spec
    from dispatch.attention.gfx950 import _dense_spec as _dense_spec_gfx950

    # The gfx942 tune is an OVERRIDE relative to the shared spec's default; if that
    # default (owned by the gfx950 file) ever shifts, the "== 2" baseline below would
    # be silently wrong. Pin it so the assumption is explicit and fails loudly.
    assert (
        AttentionDenseSpec(
            batch=1,
            seqlen_q=2048,
            seqlen_kv=2048,
            num_query_heads=16,
            num_kv_heads=4,
            head_size=64,
            dtype="bf16",
            causal=True,
            block_n=64,
        ).waves_per_eu
        == 2
    )

    def _req(dtype, d, arch):
        return AttentionRequest(
            batch=1,
            nhead_q=16,
            nhead_k=4,
            seqlen_q=2048,
            seqlen_k=2048,
            hdim_q=d,
            hdim_v=d,
            arch=arch,
            mask_type=1,
            dtype=dtype,
            algorithm="attention_dense",
            dense_persistent="off",
        )

    # gfx942: only bf16 D64 is bumped to 4.
    assert _dense_spec(_req("bf16", 64, "gfx942")).waves_per_eu == 4
    assert _dense_spec(_req("fp16", 64, "gfx942")).waves_per_eu == 2
    assert _dense_spec(_req("bf16", 128, "gfx942")).waves_per_eu == 2
    assert _dense_spec(_req("fp16", 128, "gfx942")).waves_per_eu == 2
    # gfx950: untouched, spec default preserved even for the bf16-D64 shape.
    assert _dense_spec_gfx950(_req("bf16", 64, "gfx950")).waves_per_eu == 2

    # End-to-end: the dispatched (tuned) spec's wpe actually reaches the emitted
    # attribute -- not just the spec field. Guards against a builder that ignores
    # spec.waves_per_eu (which would keep the name/binary from agreeing).
    tuned = _dense_spec(_req("bf16", 64, "gfx942"))
    kernel = build_attention_dense(tuned, arch="gfx942")
    assert kernel.attrs.get("waves_per_eu") == 4
    # The K row-group pad needs no gfx942 tag: the SHARED name already carries
    # kpad{N} for it, so the gfx942 suffix ends at the waves-per-eu bump.
    assert gfx942_kernel_name(tuned).endswith("_wpe4")
    assert "_kpad8_" in gfx942_kernel_name(tuned)


# --------------------------------------------------------------------------- #
# P3 D64 K-LDS bank-conflict pad (Hypothesis #3) wiring
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("head_size, expected", [(64, True), (128, False)])
def test_k_group_pad_active_only_on_the_packed_path(head_size, expected):
    """The pad applies on the PACKED path only (rows_per_instr > 1, i.e. D64 here).
    D128 packs one row per DMA instruction and already carries a per-row K pad, so
    re-padding it would double-count. Same gate gfx950 spells `128 // head_size > 1`."""
    assert _k_group_pad_active(_spec(head_size=head_size)) is expected


def test_k_group_stride_matches_the_gfx950_formula():
    """One lever, one formula. gfx950's builder computes
    ``LDROW = K_GROUP * D + spec.lds_k_group_pad``; gfx942 must agree exactly, or the
    two arches have silently forked a shared spec field."""
    for d, rpi in ((64, 2), (128, 1)):
        spec = _spec(head_size=d)
        assert _k_group_stride(spec) == rpi * d + spec.lds_k_group_pad
    # the shipped D64 value, spelled out so a default change is loud
    assert _k_group_stride(_spec(head_size=64)) == 136  # 2*64 + 8


def test_k_group_pad_zero_reproduces_the_unpadded_layout():
    """``lds_k_group_pad=0`` IS the A/B probe -- it is what prices the pad's ~2x, and
    it replaced the old gfx942-private module override. It must disable the padded
    layout and name itself distinguishably, or an A/B run silently compares a kernel
    against itself through a name-keyed cache."""
    on = _spec(head_size=64, dtype="fp16")
    off = dataclasses.replace(on, lds_k_group_pad=0)
    assert _k_group_pad_active(on) is True
    assert _k_group_pad_active(off) is False
    assert gfx942_kernel_name(on) != gfx942_kernel_name(off)
    assert "_kpad8_" in gfx942_kernel_name(on)
    assert "_kpad0_" in gfx942_kernel_name(off)
    for spec in (on, off):
        built = build_attention_dense(spec, arch="gfx942")
        assert built.name == gfx942_kernel_name(spec)


def test_k_group_pad_is_inert_at_d128():
    """D128 must ignore the field entirely -- no layout change, no name change --
    since its K pad is per-row via _lds_row_stride."""
    d128 = _spec(head_size=128, dtype="fp16")
    for pad in (0, 8, 24):
        alt = dataclasses.replace(d128, lds_k_group_pad=pad)
        assert _k_group_pad_active(alt) is False
        assert gfx942_kernel_name(alt) == gfx942_kernel_name(d128)


def test_dispatch_ships_the_padded_d64_path_without_restating_the_pad():
    """The gfx942 dispatch spec must SHIP the pad at D64 -- but by inheriting the
    shared default, not by restating it. Restating would recreate the per-arch
    duplicate that collapsing ``d64_kpad`` into ``lds_k_group_pad`` removed, and would
    let the two drift. gfx950's own factory is exercised alongside to pin that this
    branch changed nothing for it."""
    from dispatch.attention import AttentionRequest
    from dispatch.attention.gfx942 import _dense_spec
    from dispatch.attention.gfx950 import _dense_spec as _dense_spec_gfx950

    # Pin the shared default: every assertion below is relative to it, so a silent
    # upstream change to the pad amount must fail loudly here rather than downstream.
    assert (
        AttentionDenseSpec(
            batch=1,
            seqlen_q=2048,
            seqlen_kv=2048,
            num_query_heads=16,
            num_kv_heads=4,
            head_size=64,
            dtype="fp16",
            causal=True,
            block_n=64,
        ).lds_k_group_pad
        == 8
    )

    def _req(dtype, d, arch):
        return AttentionRequest(
            batch=1,
            nhead_q=16,
            nhead_k=4,
            seqlen_q=2048,
            seqlen_k=2048,
            hdim_q=d,
            hdim_v=d,
            arch=arch,
            mask_type=1,
            dtype=dtype,
            algorithm="attention_dense",
            dense_persistent="off",
        )

    # D64 ships padded on both dtypes; D128 is inert.
    for dtype in ("fp16", "bf16"):
        assert _k_group_pad_active(_dense_spec(_req(dtype, 64, "gfx942"))) is True
        assert _k_group_pad_active(_dense_spec(_req(dtype, 128, "gfx942"))) is False
    # gfx950's factory also leaves the field at the shared default -- neither arm
    # restates it, which is the point of unifying them.
    assert _dense_spec_gfx950(_req("fp16", 64, "gfx950")).lds_k_group_pad == 8

    # End-to-end: the dispatched D64 spec's pad reaches the emitted symbol + layout.
    tuned = _dense_spec(_req("fp16", 64, "gfx942"))
    kernel = build_attention_dense(tuned, arch="gfx942")
    assert kernel.name == gfx942_kernel_name(tuned)
    assert "_kpad8_" in kernel.name
    assert _k_group_stride(tuned) == 136


# --------------------------------------------------------------------------- #
# Gfx942DenseTuning.iglp -- the never-before-executed codegen branch
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype, d", [("fp16", 128), ("bf16", 128), ("bf16", 64)])
def test_iglp_true_builds_lowers_and_emits_the_intrinsic(dtype, d):
    """The ``iglp=True`` arm must actually BUILD, LOWER and emit the intrinsic.

    Until this test existed the True arm of ``if tuning.iglp:`` had never been
    executed by anything -- it was a module constant nobody flipped. It is now a
    user-settable knob, and shipping a never-executed codegen branch as a knob is the
    highest silent-wrong-answer risk in the struct: a knob that quietly emits nothing
    turns a sweep point into a duplicate of the baseline that is reported as a
    measurement of iglp.

    Asserted, in order: the intrinsic reaches the lowered IR as a real ``call``
    (a ``declare`` alone would mean it was emitted and then dropped), it is placed
    exactly once (the runbook §8.4 placement is one directive at the top of the main
    loop, not one per unrolled step), it is ABSENT at the default, and the kernel name
    differs -- without which the name-keyed ``_DENSE_LAUNCHER_CACHE`` would hand the
    iglp build the non-iglp HSACO and the A/B would compare a kernel with itself.
    Run on the cfvst path (fp16 D128, the only one with in-loop ds_write traffic to
    interleave) and on two naive-V configs, since the emission is unconditional.
    """
    spec = _spec(head_size=d, dtype=dtype)
    off = dataclasses.replace(_DEFAULT_TUNING, iglp=False)
    on = dataclasses.replace(_DEFAULT_TUNING, iglp=True)

    ok, why = supports_attention_dense(spec, arch="gfx942", tuning=on)
    assert ok, f"iglp=True must be in scope: {why}"

    kd_on = build_attention_dense(spec, arch="gfx942", tuning=on)
    kd_off = build_attention_dense(spec, arch="gfx942", tuning=off)
    ir_on = _lower(kd_on)
    ir_off = _lower(kd_off)

    assert ir_on.count("call void @llvm.amdgcn.iglp.opt") == 1, (
        "iglp=True must emit exactly one llvm.amdgcn.iglp.opt CALL (one scheduler "
        "directive at the top of the main-loop body, per optimization_runbook §8.4)"
    )
    assert "iglp" not in ir_off, "the shipped default must emit no iglp directive"

    # identity: name AND body must both move, together.
    assert kd_on.name != kd_off.name
    assert kd_on.name == gfx942_kernel_name(spec, on)
    assert kd_on.name.endswith("_iglp1")
    assert _ir_body_sha(spec, on) != _ir_body_sha(spec, off)


# --------------------------------------------------------------------------- #
# P4 persistent grid-stride variant
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("decode", ["qb_major", "hkv_major"])
def test_persistent_builds_for_both_decodes(dtype, d, decode):
    """P4: the persistent grid-stride body builds for every in-scope config and both
    work-decode orders. The decode is a build-time (Python) branch, so both must emit;
    GPU numeric correctness (the decode bijection) is verified in the live cohort."""
    spec = _spec(
        head_size=d,
        dtype=dtype,
        persistent=True,
        num_persistent=228,
        persist_decode=decode,
    )
    ok, why = supports_attention_dense(spec, arch="gfx942")
    assert ok, why
    kernel = build_attention_dense(spec, arch="gfx942")
    assert kernel is not None


def test_persistent_kernel_name_carries_the_persist_tag():
    """The persistent identity must be in the name: a persistent and a default spec
    that agree on every other field compile to different binaries (different grid +
    body), so a shared name would collide in the launcher/HSACO cache."""
    default = _spec(head_size=128, dtype="fp16")
    persist = _spec(head_size=128, dtype="fp16", persistent=True, num_persistent=304)
    assert "persist304" in gfx942_kernel_name(persist)
    assert "persist" not in gfx942_kernel_name(default)
    assert gfx942_kernel_name(persist) != gfx942_kernel_name(default)


def test_persistent_and_default_share_one_inner_body():
    """The refactor factored the per-work-item compute into a single ``_run_work_item``
    used by both grids, so the two bodies must contain the SAME per-tile op mix (the
    32x32x8 QK/PV MFMAs and the softmax exps). Persistent only adds the outer
    grid-stride ``scf.for`` and the work decode; it must not drop or duplicate the
    inner MFMA/exp work relative to the default grid."""
    default = _spec(head_size=128, dtype="fp16")
    persist = _spec(head_size=128, dtype="fp16", persistent=True, num_persistent=304)
    kd = build_attention_dense(default, arch="gfx942")
    kp = build_attention_dense(persist, arch="gfx942")
    nd = [n for op in kd.body.ops for n in _walk_op_names(op)]
    npp = [n for op in kp.body.ops for n in _walk_op_names(op)]
    # same count of the heavy inner ops (per-tile MFMA + softmax exp), since the inner
    # loop body is shared verbatim.
    for op in ("math.exp2_fast", "arith.cast_f32_to"):
        assert nd.count(op) == npp.count(op), (
            f"{op}: default={nd.count(op)} persistent={npp.count(op)} -- the shared "
            "inner body diverged"
        )
    # persistent has exactly one MORE scf.for (the outer grid-stride loop).
    assert npp.count("scf.for") == nd.count("scf.for") + 1


def test_dispatch_persistent_auto_turns_on_for_large_sq_only():
    """P4 dispatch: ``dense_persistent='auto'`` resolves to persistent once the work
    (nqb*Hq*B) fills the gfx942 persistent grid (num_persistent defaulted to 304), and
    stays off for small Sq. Explicit on/off are honored; gfx950 keeps its 256 default
    and is otherwise untouched."""
    from dispatch.attention import AttentionRequest
    from dispatch.attention.gfx942 import _dense_spec
    from dispatch.attention.gfx950 import _dense_spec as _dense_spec_gfx950

    def _req(sq, arch, persist="auto"):
        return AttentionRequest(
            batch=1,
            nhead_q=16,
            nhead_k=4,
            seqlen_q=sq,
            seqlen_k=sq,
            hdim_q=128,
            hdim_v=128,
            arch=arch,
            mask_type=1,
            dtype="fp16",
            algorithm="attention_dense",
            dense_persistent=persist,
        )

    # gfx942 num_persistent defaulted to the 304-CU part's CU count.
    assert _dense_spec(_req(8192, "gfx942")).num_persistent == 304
    # auto: on for large Sq (nqb*Hq = 32*16 = 512 >= 304), off for small.
    assert _dense_spec(_req(8192, "gfx942")).persistent is True
    assert _dense_spec(_req(2048, "gfx942")).persistent is False  # 8*16 = 128 < 304
    # explicit modes honored.
    assert _dense_spec(_req(8192, "gfx942", "off")).persistent is False
    assert _dense_spec(_req(256, "gfx942", "on")).persistent is True
    # gfx950 untouched: keeps the 256 default (not the gfx942 304 override).
    assert _dense_spec_gfx950(_req(8192, "gfx950")).num_persistent == 256


# --------------------------------------------------------------------------- #
# run_attention_dense_torch entry point (guard logic; numeric lane is on-GPU)
# --------------------------------------------------------------------------- #
def test_run_attention_dense_torch_rejects_unsupported_spec():
    """The framework entry raises (not silently no-ops) for a supported-by-dataclass
    but out-of-scope spec. varlen is dataclass-valid but rejected by
    supports_attention_dense, so the entry must raise NotImplementedError before any
    compile/launch is attempted (rather than launching a kernel that does not exist)."""
    spec = _spec(head_size=128, dtype="fp16", varlen=True)
    with pytest.raises(NotImplementedError, match="unsupported|varlen"):
        run_attention_dense_torch(
            spec=spec, q=None, k=None, v=None, out=None, scale=0.1
        )


def test_run_attention_dense_torch_rejects_cu_seqlens():
    """gfx942 attention_dense is dense-only (varlen rejected), so the ABI has no
    cu_seqlens args; passing them is a caller error, not a silently-ignored kwarg."""
    spec = _spec(head_size=128, dtype="fp16")
    with pytest.raises(ValueError, match="cu_seqlens"):
        run_attention_dense_torch(
            spec=spec,
            q=None,
            k=None,
            v=None,
            out=None,
            scale=0.1,
            cu_seqlens_q=[0, 128],
        )
