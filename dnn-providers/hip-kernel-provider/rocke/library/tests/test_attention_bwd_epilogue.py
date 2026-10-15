# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU / IR tests of the dK / dV epilogue and LDS layout knobs of the backward.

Knobs: ``dkv_epilogue`` ("lds_staged" default, "lds_staged_pair", "direct"),
``dkv_store_vec`` (8 default, 4, 2, 1), ``dkv_atomic_guard`` ("branch"
default, "select"), ``lds_order`` ("resident_low" default, "phase_low") and
the row pads of ``lds_swizzle`` ("padN" on the row-major buffers, now
including the epilogue buffer ``dkv_out``).

* value sets and couplings, and the :data:`INERT_KNOBS` predicates;
* every value on every target and correctness configuration (plain, THD
  head-packed, atomic and narrow instances) builds where it is legal (and
  raises ``ValueError`` where it is not: LDS accumulators, LDS capacity),
  keeps the cross-wave LDS barrier model clean, ends every q step with a
  barrier, and gives IR that differs from the default instance apart from
  the kernel name exactly where the value is not inert;
* epilogue structure: ``"direct"`` has no LDS access and no barrier after the
  q loops and stores each accumulator cell once, with the same rounding ops as
  the staged path; ``"lds_staged_pair"`` has half the epilogue barriers;
  ``dkv_store_vec`` sets the dK / dV store and the staging read width;
  ``dkv_atomic_guard = "select"`` issues the same atomics without row
  branches, each value selected against ``+0.0``;
* LDS views: no access claims more alignment than its (padded) view has.

Device numerics (bit for bit equal dK / dV in direct mode) are in
``test_attention_bwd_numeric.py`` (``test_runtime_variants_agree``).
"""

from __future__ import annotations

import dataclasses
from functools import cache

import pytest
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower

from kernels.common.attention_bwd import (
    DKV_ATOMIC_GUARDS,
    DKV_EPILOGUES,
    DKV_STORE_VECS,
    INERT_KNOBS,
    LDS_ORDERS,
    AttnBwdSpec,
    _emit_main,
    attn_bwd_arch_facts,
    build_attn_bwd_main,
    lds_plan,
    validate_attn_bwd_spec,
)
from kernels.common._attention_bwd_refusals import resource_refusal

from ._attention_bwd_test_utils import distinct_configs
from .test_attention_bwd_ir import (
    ARCHS,
    _derived,
    _user_accesses,
    _walk,
    lds_hazards,
    q_loops,
    spec_of,
    step_bodies,
    traced_spec,
)

_PADDABLE = ("q", "do", "ds", "pt", "k_stage", "v_stage", "k_res", "v_res", "dkv_out")

KNOBS = {
    "direct": dict(dkv_epilogue="direct"),
    "pair": dict(dkv_epilogue="lds_staged_pair"),
    "vec4": dict(dkv_store_vec=4),
    "vec2": dict(dkv_store_vec=2),
    "vec1": dict(dkv_store_vec=1),
    "pair_vec2": dict(dkv_epilogue="lds_staged_pair", dkv_store_vec=2),
    "phase_low": dict(lds_order="phase_low"),
    "select": dict(dkv_atomic_guard="select"),
    "out_pad8": dict(lds_swizzle="dkv_out=pad8"),
    "out_pad16": dict(lds_swizzle="dkv_out=pad16"),
    "out_pad32": dict(lds_swizzle="dkv_out=pad32"),
    **{
        f"all_pad{n}": dict(lds_swizzle=",".join(f"{b}=pad{n}" for b in _PADDABLE))
        for n in (8, 16, 32)
    },
}
VARIANTS = ("band", "thd_none_pack", "band_atomic", "sv1_band")
ROWS = [
    (arch, cfg, d, var)
    for arch in ARCHS
    for cfg in distinct_configs(arch)
    for d in (64, 128)
    for var in VARIANTS
]


def _ids(row):
    return "-".join(map(str, row))


# These tests check the emitted structure (barriers, stores, widths, LDS
# spans), not the compiled code object: they resolve and build without the
# main kernel's recorded resource refusals (every static rule still applies),
# and a value the gate refuses is checked to raise from the shipped validator.
def _structural(spec, arch):
    return validate_attn_bwd_spec(spec, arch, main_kernel_refusals=False)


def _build(spec, arch):
    return _emit_main(spec, arch, main_kernel_refusals=False)


def _check_refusal(r, arch) -> None:
    if resource_refusal(r, arch) is not None:
        with pytest.raises(ValueError, match="resource gate"):
            validate_attn_bwd_spec(r, arch)


def _spec(row, knob=None) -> AttnBwdSpec:
    arch, cfg, d, var = row
    spec = spec_of(arch, cfg, d, "fp16", var)
    return dataclasses.replace(spec, **KNOBS[knob]) if knob else spec


def _expected_error(row, knob):
    """The ``ValueError`` text a knob value must raise on a row, or None."""
    arch = row[0]
    base = _structural(_spec(row), arch)
    kw = KNOBS[knob]
    if (
        base.acc_in_lds
        and base.dkv_mode == "direct"
        and kw.get("dkv_epilogue", "lds_staged") != "lds_staged"
    ):
        return "needs register accumulators"
    spec = dataclasses.replace(base, **kw)
    if not lds_plan(spec, arch).fits:
        return "LDS plan"
    return None


def _swizzle_inert(spec, arch) -> bool:
    """A ``lds_swizzle`` changes nothing when it names no buffer the kernel
    accesses (absent from the plan, or the epilogue buffer that atomic dK /
    dV reserve but never touch) and leaves the footprint unchanged."""
    if not spec.lds_swizzle:
        return True
    plan = lds_plan(spec, arch)
    unused = {"dkv_out"} if spec.dkv_mode == "atomic" else set()
    names = [item.split("=")[0] for item in spec.lds_swizzle.split(",")]
    if any(plan.has(n) and n not in unused for n in names):
        return False
    plain = lds_plan(dataclasses.replace(spec, lds_swizzle=None), arch)
    return plan.total_bytes == plain.total_bytes


def _inert(r, arch) -> bool:
    for k in ("dkv_epilogue", "dkv_store_vec", "dkv_atomic_guard", "lds_order"):
        default = AttnBwdSpec.__dataclass_fields__[k].default
        if getattr(r, k) != default and not INERT_KNOBS[k](r, arch):
            return False
    return _swizzle_inert(r, arch)


@cache
def _ir(spec, arch) -> str:
    r = _structural(spec, arch)
    text = lower(_build(spec, arch), arch=arch, llvm_flavor="llvm22")
    return text.replace(r.kernel_name("main"), "@KERNEL")


def _epilogue_ops(kernel):
    """Ops after the last q loop, in program order."""
    walk = [op for op, _ in _walk(kernel.body.ops)]
    in_q = {id(op) for lp in q_loops(kernel) for op, _ in _walk([lp])}
    last = max(i for i, op in enumerate(walk) if id(op) in in_q)
    return walk[last + 1 :]


# ---------------------------------------------------------------------------
# value sets, couplings, inert predicates
# ---------------------------------------------------------------------------


def test_defaults_and_value_sets():
    s = AttnBwdSpec(head_size=64)
    assert (s.dkv_epilogue, s.dkv_store_vec, s.lds_order) == (
        "lds_staged",
        8,
        "resident_low",
    )
    assert DKV_EPILOGUES == ("lds_staged", "lds_staged_pair", "direct")
    assert DKV_STORE_VECS == (8, 4, 2, 1)
    assert LDS_ORDERS == ("resident_low", "phase_low")
    assert DKV_ATOMIC_GUARDS == ("branch", "select") and s.dkv_atomic_guard == "branch"
    # the defaults add no salt to the kernel name
    assert (
        s.kernel_name()
        == AttnBwdSpec(head_size=64, name="rocke_attn_bwd").kernel_name()
    )
    assert "_" not in s.kernel_name().rsplit("sv8", 1)[1]
    for bad in (
        dict(dkv_epilogue="lds"),
        dict(dkv_store_vec=16),
        dict(dkv_store_vec=3),
        dict(lds_order="reverse"),
        dict(dkv_atomic_guard="none"),
        # the direct epilogue has no row store width to choose
        dict(dkv_epilogue="direct", dkv_store_vec=4),
    ):
        with pytest.raises(ValueError):
            AttnBwdSpec(head_size=64, **bad)
    # in atomic mode the store width is inert, not illegal
    AttnBwdSpec(head_size=64, dkv_mode="atomic", dkv_epilogue="direct", dkv_store_vec=4)
    for kw in KNOBS.values():
        name = AttnBwdSpec(head_size=64, **kw).kernel_name()
        assert name != s.kernel_name()


@pytest.mark.parametrize("arch", ("gfx1151", "gfx1201"))
def test_lds_accumulators_reject_the_register_epilogues(arch):
    for epi in ("direct", "lds_staged_pair"):
        with pytest.raises(ValueError, match="register accumulators"):
            validate_attn_bwd_spec(AttnBwdSpec(head_size=64, dkv_epilogue=epi), arch)
        # atomic mode: the epilogue is fp32 atomics either way (inert)
        r = validate_attn_bwd_spec(
            AttnBwdSpec(head_size=64, dkv_mode="atomic", dkv_epilogue=epi), arch
        )
        assert INERT_KNOBS["dkv_epilogue"](r, arch)
    for vec in DKV_STORE_VECS:
        validate_attn_bwd_spec(AttnBwdSpec(head_size=64, dkv_store_vec=vec), arch)


@pytest.mark.parametrize("arch", ARCHS)
def test_inert_predicates(arch):
    r = validate_attn_bwd_spec(AttnBwdSpec(head_size=64), arch)
    for k in ("dkv_epilogue", "dkv_store_vec"):
        assert not INERT_KNOBS[k](r, arch)
    assert INERT_KNOBS["dkv_atomic_guard"](r, arch)  # direct mode has no atomics
    atomic = validate_attn_bwd_spec(
        AttnBwdSpec(head_size=64, dkv_mode="atomic", dkv_store_vec=2), arch
    )
    assert INERT_KNOBS["dkv_store_vec"](atomic, arch)
    assert not INERT_KNOBS["dkv_atomic_guard"](atomic, arch)
    pair = dataclasses.replace(atomic, dkv_epilogue="lds_staged_pair")
    assert INERT_KNOBS["dkv_epilogue"](pair, arch)
    # placement order: inert exactly when nothing is resident
    resident = r.kv_residency == "lds" or r.kt_source == "lds" or r.acc_in_lds
    low = dataclasses.replace(r, lds_order="phase_low")
    assert INERT_KNOBS["lds_order"](low, arch) is (not resident)


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("cfg", ("default", "debug"))
def test_plan_of_each_value(arch, cfg):
    row = (arch, cfg, 128, "band")
    if cfg not in distinct_configs(arch):
        pytest.skip(f"{cfg} is not a distinct configuration on {arch}")
    base = validate_attn_bwd_spec(_spec(row), arch)
    plan = lds_plan(base, arch)
    eb, d, n0 = 2, base.head_size, base.block_n
    if base.acc_in_lds:
        assert not plan.has("dkv_out")
    else:
        out = plan.buffer("dkv_out")
        assert (out.nbytes, out.layout) == (n0 * d * eb, "rowmajor:none")
        assert out.offset == plan.resident_bytes
        direct = lds_plan(dataclasses.replace(base, dkv_epilogue="direct"), arch)
        assert not direct.has("dkv_out")
        assert dict(direct.phase_bytes)["epilogue"] == 0
        pair = lds_plan(dataclasses.replace(base, dkv_epilogue="lds_staged_pair"), arch)
        assert pair.ring_offsets("dkv_out") == (
            pair.resident_bytes,
            pair.resident_bytes + n0 * d * eb,
        )
        assert dict(pair.phase_bytes)["epilogue"] == 2 * n0 * d * eb
        pair.check()
        for pad in (8, 16, 32):
            padded = dataclasses.replace(base, lds_swizzle=f"dkv_out=pad{pad}")
            buf = lds_plan(padded, arch).buffer("dkv_out")
            assert (buf.nbytes, buf.layout) == (
                n0 * (d * eb + pad),
                f"rowmajor:pad{pad}",
            )
    # the store width does not change the plan
    vec1 = lds_plan(dataclasses.replace(base, dkv_store_vec=1), arch)
    assert vec1.buffers == plan.buffers
    # placement order: same footprint and buffers, residents above the phases
    low = lds_plan(dataclasses.replace(base, lds_order="phase_low"), arch)
    low.check()
    assert low.total_bytes == plan.total_bytes
    assert low.resident_bytes == plan.resident_bytes
    assert sorted((b.name, b.slot, b.nbytes) for b in low.buffers) == sorted(
        (b.name, b.slot, b.nbytes) for b in plan.buffers
    )
    top = max(s for _, s in plan.phase_bytes)
    assert low.phase_offset == 0 and plan.phase_offset == plan.resident_bytes
    for b in low.buffers:
        if b.resident:
            assert b.offset >= top
        else:
            assert b.end <= top
    shift = {(b.name, b.slot): b.offset for b in low.buffers}
    for b in plan.buffers:
        if not b.resident:
            assert shift[(b.name, b.slot)] == b.offset - plan.resident_bytes


# ---------------------------------------------------------------------------
# every value on every target: builds, race-free, differs where effective
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("knob", list(KNOBS))
@pytest.mark.parametrize("row", ROWS, ids=_ids)
def test_value_builds_race_free_and_differs_where_effective(row, knob):
    arch = row[0]
    spec = _spec(row, knob)
    err = _expected_error(row, knob)
    if err is not None:
        with pytest.raises(ValueError, match=err):
            validate_attn_bwd_spec(spec, arch)
        return
    r = _structural(spec, arch)
    _check_refusal(r, arch)
    kernel, tags = traced_spec(spec, arch, structural=True)
    assert not lds_hazards(kernel, tags)[:4]
    for ops in step_bodies(kernel):
        assert [op for op in ops if op.name != "scf.yield"][-1].name == "tile.sync"
    same = _ir(spec, arch) == _ir(_spec(row), arch)
    assert same is _inert(r, arch), (knob, same)


# ---------------------------------------------------------------------------
# epilogue structure
# ---------------------------------------------------------------------------

DIRECT_ROWS = [r for r in ROWS if "atomic" not in r[3]]


def _register_acc(row) -> bool:
    return not _structural(_spec(row), row[0]).acc_in_lds


def _out_stores(kernel, ops):
    users = _derived(kernel, ("dK", "dV"))
    return [
        op
        for op in ops
        if op.name.startswith("memref.global_store")
        and op.operands
        and op.operands[0].name in users
    ]


@pytest.mark.parametrize("row", [r for r in DIRECT_ROWS if _register_acc(r)], ids=_ids)
def test_direct_epilogue_stores_each_cell_once_without_lds(row):
    arch = row[0]
    r = _structural(_spec(row, "direct"), arch)
    _check_refusal(r, arch)
    kernel = _build(r, arch)
    epi = _epilogue_ops(kernel)
    assert not [op for op in epi if op.name.startswith(("tile.smem", "tile.ds_read"))]
    assert not [op for op in epi if op.name == "tile.sync"]
    wave = attn_bwd_arch_facts(arch).wave_size
    stores = _out_stores(kernel, epi)
    assert len(stores) == 2 * r.block_n * r.head_size // (r.waves * wave)
    assert all(op.name == "memref.global_store_typed" for op in stores)
    # every stored value is selected against zero on the row's liveness
    # (batched padding rows get exact zeros)
    by_result = {id(res): op for op in epi for res in op.results}
    assert all(by_result[id(op.operands[2])].name == "arith.select" for op in stores)
    # the same rounding ops as the staged path (one per fragment per output)
    staged = _build(_spec(row), arch)
    for name in ("vector.trunc_f32_to", "arith.fmul"):
        count = [
            sum(op.name == name for op in _epilogue_ops(k)) for k in (kernel, staged)
        ]
        assert count[0] == count[1], (name, count)
    # the per-row predicate: one branch per (fragment row, slot) and output
    g_rows = 2 * r.block_n // (r.waves * 16) * 4
    assert sum(op.name == "scf.if" for op in epi) == g_rows


@pytest.mark.parametrize("row", [r for r in DIRECT_ROWS if _register_acc(r)], ids=_ids)
def test_pair_epilogue_halves_the_barriers(row):
    """``"lds_staged_pair"``: two epilogue barriers instead of four, dK and dV
    staged in two disjoint buffers, each read back from the one it was
    written to (one buffer, written twice, in the default)."""
    arch = row[0]
    syncs, spans = {}, {}
    for knob in (None, "pair"):
        kernel, tags = traced_spec(_spec(row, knob), arch, structural=True)
        epi = _epilogue_ops(kernel)
        syncs[knob] = sum(op.name == "tile.sync" for op in epi)
        lds = [tags[id(op)] for op in epi if id(op) in tags]
        stores = [(lo, hi) for lo, hi, st, _p in lds if st]
        loads = [(lo, hi) for lo, hi, st, _p in lds if not st]
        # program order: the staging stores of each output, then its reads
        spans[knob] = (list(dict.fromkeys(stores)), list(dict.fromkeys(loads)))
    assert (syncs[None], syncs["pair"]) == (4, 2)
    assert len(spans[None][0]) == 1 and spans[None][0] == spans[None][1]
    (s0, s1), reads = spans["pair"]
    assert s0[1] <= s1[0] and reads == [s0, s1]


@pytest.mark.parametrize("vec", DKV_STORE_VECS)
@pytest.mark.parametrize("arch", ARCHS)
def test_store_width(arch, vec):
    elem = "half"
    for cfg in distinct_configs(arch):
        spec = dataclasses.replace(
            spec_of(arch, cfg, 128, "fp16", "band"), dkv_store_vec=vec
        )
        r = _structural(spec, arch)
        _check_refusal(r, arch)
        ir = _ir(spec, arch)
        acc = [x for x in _user_accesses(ir, ("%dK", "%dV")) if x[0] == "store"]
        want = elem if vec == 1 else f"<{vec} x {elem}>"
        assert acc and all(t == want and al == 2 * vec for _k, t, al in acc), acc[:4]
        kernel = _build(r, arch)
        epi = _epilogue_ops(kernel)
        threads = r.waves * attn_bwd_arch_facts(arch).wave_size
        per_pass = -(-(r.block_n * r.head_size // vec) // threads)
        expect = 2 * per_pass * (vec if vec == 1 else 1)
        assert len(_out_stores(kernel, epi)) == expect
        loads = [op for op in epi if op.name == "tile.smem_load_vN"]
        widths = {op.attrs["vec"] for op in loads}
        assert widths == ({min(vec, 4)} if r.acc_in_lds else {vec})


# ---------------------------------------------------------------------------
# LDS alignment of padded views
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pad", (8, 16, 32))
@pytest.mark.parametrize("arch", ARCHS)
def test_no_lds_access_claims_more_than_its_view(arch, pad):
    from kernels.common._attention_bwd_frag import LdsView

    seen = []
    orig = {n: getattr(LdsView, n) for n in ("load", "store")}

    def wrap(name):
        def f(self, b, *args, **kw):
            ops = b._region_stack[-1].ops
            n0 = len(ops)
            out = orig[name](self, b, *args, **kw)
            for op in ops[n0:]:
                if op.name in ("tile.smem_load_vN", "tile.smem_store_vN"):
                    seen.append((op.attrs["vec"], self.align_elems))
            return out

        return f

    spec = dataclasses.replace(
        spec_of(arch, distinct_configs(arch)[0], 64, "fp16", "band"),
        **KNOBS[f"all_pad{pad}"],
    )
    try:
        for n in orig:
            setattr(LdsView, n, wrap(n))
        build_attn_bwd_main(spec, arch=arch)
    finally:
        for n, f in orig.items():
            setattr(LdsView, n, f)
    assert seen
    assert all(vec <= align for vec, align in seen), sorted(set(seen))
    if pad == 8:  # a 4-element pad: 16-byte vectors become two 8-byte halves
        assert {align for _v, align in seen} >= {4}


@pytest.mark.parametrize("row", [r for r in ROWS if "atomic" in r[3]], ids=_ids)
def test_select_guard_issues_the_same_atomics_without_row_branches(row):
    """``dkv_atomic_guard = "select"``: the same number of epilogue atomics
    as the row-branch guard, none inside a per-row branch (only the CTA's
    liveness branch), each added value a select against ``+0.0``."""
    arch = row[0]
    kernels = {
        knob: build_attn_bwd_main(_spec(row, knob), arch=arch)
        for knob in (None, "select")
    }
    counts = {}
    for knob, kernel in kernels.items():
        in_q = {id(op) for lp in q_loops(kernel) for op, _ in _walk([lp])}
        epi = [
            (op, anc)
            for op, anc in _walk(kernel.body.ops)
            if op.name == "memref.global_atomic_add_f32" and id(op) not in in_q
        ]
        counts[knob] = len(epi)
        depth = {sum(a.name == "scf.if" for a in anc) for _op, anc in epi}
        if knob == "select":
            assert depth == {1}  # the liveness branch only
            by_result = {
                id(r): op for op, _ in _walk(kernel.body.ops) for r in op.results
            }
            for op, _anc in epi:
                val = by_result[id(op.operands[2])]
                assert val.name == "arith.select"
        else:
            assert depth == {2}  # liveness, then one branch per key row
    assert counts[None] == counts["select"] > 0
