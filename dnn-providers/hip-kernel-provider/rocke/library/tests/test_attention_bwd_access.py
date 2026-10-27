# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Memory-access width and bounds levers of the backward main kernel (and
the narrow-class width of every kernel of the plan).

Each lever moves the same bytes in fewer or branch-free accesses, so every
value is bit for bit the default's (``numerics = "bitwise"`` in the tuning
registry):

* ``ws_bounds = "buffer"``: buffer-descriptor bounds for the scalar
  workspace stats loads and the slice sink's stores (no address clamps, no
  row branches);
* ``dkv_cell_vec = 4``: the per-slot dK / dV epilogue stores 4 consecutive
  columns after an in-quad 4 x 4 transpose (DPP quad permutes);
* ``narrow_vec``: ``stage_vec = 1`` user-tensor accesses of 2 or 4 elements
  where every stride and base allows them (the plan narrows the cap);
* ``slab_carry``: Q / dO slab offsets carried across q steps (the synchronous
  step; refused with a ring or the counted waits).

(The vector ``lse2`` / ``Dsum`` loads are ``stats_vec``, tested with the
softmax-gradient levers.)

CPU: value sets, validator and inert rules (each inert predicate checked
against the IR over the staging bodies), the IR forms (buffer loads, DPP,
vector widths, carried offsets), default kernels unchanged, plan narrowing,
the resource gate of every new value (a value that fails it is refused
exactly, RESOURCE_REFUSED). Device: each lever's dQ / dK / dV bit for bit equal to the
default run (dQ through the split dQ kernel on gfx942 / gfx950 so every
output is deterministic; with atomic dQ, dK / dV in direct mode) and the
oracle.
"""

from __future__ import annotations

import dataclasses
import os
import re

import numpy as np
import pytest
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower

from dispatch.attention_bwd import tuning_common as tc
from kernels.common.attention_bwd import (
    DKV_CELL_VECS,
    INERT_KNOBS,
    NARROW_VECS,
    SPLIT_DQ_ARCHS,
    WS_BOUNDS,
    AttnBwdSpec,
    build_attn_bwd_main,
    validate_attn_bwd_spec,
)
from kernels.common._attention_bwd_frag import c_quad_transposable
from kernels.common._attention_bwd_sched import SCHEDS
from kernels.common._attention_bwd_body import BwdTileGeometry
from kernels.common.attention_bwd_plan import (
    FEATURE_GROUPS,
    AttnBwdPolicy,
    AttnBwdRequest,
    attn_bwd_narrow_width,
    attn_bwd_plan,
)
from kernels.common.attention_bwd_run import _BUILDERS
from ._attention_bwd_test_utils import find_bwd_case

ALL_ARCHS = ("gfx942", "gfx950", "gfx1151", "gfx1201")


def _main_ir(arch, **knobs) -> str:
    """The IR of the d64 bf16 main kernel with ``knobs`` (the emission: a
    resource refusal, RESOURCE_REFUSED, is bypassed; the gate test below
    covers the refusals)."""
    from unittest import mock

    from kernels.common import attention_bwd as AB

    spec = AttnBwdSpec(head_size=64, dtype="bf16", **knobs)
    with mock.patch.object(AB, "resource_refusal", lambda s, a: None):
        return lower(build_attn_bwd_main(spec, arch=arch), arch=arch)


def _body(ir: str, name: str) -> str:
    return ir.replace(name, "K")


def _policy(knobs) -> AttnBwdPolicy:
    return AttnBwdPolicy(
        declared_groups=frozenset(FEATURE_GROUPS),
        waive_unverified=True,
        tuning=lambda *a: dict(knobs),
    )


# ---------------------------------------------------------------------------
# value sets, legality, inert rules
# ---------------------------------------------------------------------------


def test_value_sets_and_defaults():
    assert WS_BOUNDS == ("clamp", "buffer")
    assert DKV_CELL_VECS == (1, 4) and NARROW_VECS == (1, 2, 4)
    s = AttnBwdSpec(head_size=64)
    assert (s.ws_bounds, s.dkv_cell_vec, s.narrow_vec) == ("clamp", 1, 1)
    assert s.slab_carry is False and s.non_default_fields() == ()
    for bad in (
        dict(ws_bounds="none"),
        dict(dkv_cell_vec=2),
        dict(narrow_vec=8),
        dict(slab_carry=1),
        dict(narrow_vec=2),  # stage_vec = 8
        dict(ws_bounds="buffer", ws_layout="token_major"),
    ):
        with pytest.raises(ValueError):
            AttnBwdSpec(head_size=64, **bad)
    for a in ("ws_bounds", "dkv_cell_vec", "narrow_vec", "slab_carry"):
        assert tc.backward_axis(a).numerics == "bitwise"


@pytest.mark.parametrize("arch", ALL_ARCHS)
def test_layout_facts_decide_legality(arch):
    s = validate_attn_bwd_spec(AttnBwdSpec(head_size=64), arch)
    geom = BwdTileGeometry.from_spec(s, arch)
    quad = c_quad_transposable(geom.g1.op)
    # the MFMA 16 x 16 C layouts transpose in lane quads; WMMA does not
    assert quad == (arch in ("gfx942", "gfx950"))
    knobs = {"dkv_cell_vec": 4}
    if quad:
        validate_attn_bwd_spec(AttnBwdSpec(head_size=64, **knobs), arch)
    else:
        with pytest.raises(ValueError, match="dkv_cell_vec"):
            validate_attn_bwd_spec(AttnBwdSpec(head_size=64, **knobs), arch)
    with pytest.raises(ValueError, match="q_rotate"):
        validate_attn_bwd_spec(
            AttnBwdSpec(head_size=64, slab_carry=True, q_rotate=True), arch
        )


# The staging bodies slab_carry is not built with (the steps ahead are formed
# from their index there), as validator refusals.
_RING_BODIES = {
    "vgpr2": dict(global_path="vgpr", ring_depth=2),
    "vgpr3": dict(global_path="vgpr", ring_depth=3),
    "merge": dict(global_path="vgpr", ring_depth=2, barrier_merge="on"),
    "counted": dict(loop_sync="counted"),
    "dma1": dict(global_path="dma", ring_depth=1),
    "dma2": dict(global_path="dma", ring_depth=2),
}


@pytest.mark.parametrize("arch", ("gfx942", "gfx950"))
@pytest.mark.parametrize("body", list(_RING_BODIES))
def test_slab_carry_is_refused_on_the_staging_bodies(arch, body):
    kw = _RING_BODIES[body]
    try:
        validate_attn_bwd_spec(AttnBwdSpec(head_size=64, **kw), arch)
    except ValueError:
        pytest.skip(f"{body} is not a body of {arch}")
    with pytest.raises(ValueError, match="slab_carry is built for the synchronous"):
        validate_attn_bwd_spec(AttnBwdSpec(head_size=64, slab_carry=True, **kw), arch)
    # the registry agrees (never enumerated there)
    assert not tc.backward_axis("slab_carry").applies(True, kw)


def test_inert_rules():
    arch = "gfx942"

    def v(**k):
        return validate_attn_bwd_spec(AttnBwdSpec(head_size=64, **k), arch)

    assert INERT_KNOBS["stats_vec"](v(stats_vec=4, head_pack=True), arch)
    assert not INERT_KNOBS["stats_vec"](v(stats_vec=4), arch)
    assert INERT_KNOBS["slab_carry"](v(slab_carry=True, head_pack=True), arch)
    # buffer bounds: the scalar workspace stats or the slice stores
    assert not INERT_KNOBS["ws_bounds"](v(ws_bounds="buffer"), arch)
    assert INERT_KNOBS["ws_bounds"](v(ws_bounds="buffer", stats_vec=4), arch)
    assert INERT_KNOBS["ws_bounds"](v(ws_bounds="buffer", loop_sync="counted"), arch)
    assert not INERT_KNOBS["ws_bounds"](
        v(ws_bounds="buffer", stats_vec=4, dkv_mode="atomic", dkv_sink="slices"),
        arch,
    )
    # the quad transpose: the per-slot epilogue of direct mode or the slices
    assert INERT_KNOBS["dkv_cell_vec"](v(dkv_cell_vec=4), arch)
    assert not INERT_KNOBS["dkv_cell_vec"](
        v(dkv_cell_vec=4, dkv_epilogue="direct"), arch
    )
    assert INERT_KNOBS["dkv_cell_vec"](
        v(dkv_cell_vec=4, dkv_epilogue="direct", dkv_mode="atomic"), arch
    )
    assert not INERT_KNOBS["dkv_cell_vec"](
        v(dkv_cell_vec=4, dkv_epilogue="direct", dkv_mode="atomic",
          dkv_sink="slices"),
        arch,
    )  # fmt: skip


# Bodies x lever values whose inert predicate is checked against the IR: an
# inert value gives the default's IR (apart from the kernel name), a value
# that is not inert changes it.
_INERT_BODIES = {
    "sync": {},
    "vgpr2": dict(global_path="vgpr", ring_depth=2),
    "merge": dict(global_path="vgpr", ring_depth=2, barrier_merge="on"),
    "counted": dict(loop_sync="counted"),
    "dma2": dict(global_path="dma", ring_depth=2),
    "pack": dict(head_pack=True),
    "stats_vec2": dict(stats_vec=2),
    "atomic": dict(dkv_mode="atomic"),
    "slices": dict(dkv_mode="atomic", dkv_sink="slices"),
    "direct_epi": dict(dkv_epilogue="direct"),
    "slices_direct": dict(dkv_mode="atomic", dkv_sink="slices", dkv_epilogue="direct"),
    "grouped": dict(step_order="head_outer", edge_tiles=True),
}  # fmt: skip
_INERT_VALUES = {
    "ws_bounds": "buffer",
    "dkv_cell_vec": 4,
    "slab_carry": True,
    "dkv_sink": "slices",
}


@pytest.mark.parametrize("arch", ("gfx942", "gfx950"))
@pytest.mark.parametrize("knob", list(_INERT_VALUES))
def test_inert_rules_match_the_ir(arch, knob):
    """For every body and lever value legal there: ``INERT_KNOBS`` says inert
    exactly when the kernel's IR equals the kernel without the value."""
    from unittest import mock

    from kernels.common import attention_bwd as AB

    value = _INERT_VALUES[knob]
    checked = 0
    for body, kw in _INERT_BODIES.items():
        if kw.get(knob) is not None:
            continue
        try:
            with mock.patch.object(AB, "resource_refusal", lambda s, a: None):
                ref = validate_attn_bwd_spec(
                    AttnBwdSpec(head_size=64, dtype="bf16", **kw), arch
                )
                spec = validate_attn_bwd_spec(
                    AttnBwdSpec(head_size=64, dtype="bf16", **{**kw, knob: value}),
                    arch,
                )
        except ValueError:
            continue  # not a legal pair on this target
        with mock.patch.object(AB, "resource_refusal", lambda s, a: None):
            a = lower(build_attn_bwd_main(spec, arch=arch), arch=arch)
            b = lower(build_attn_bwd_main(ref, arch=arch), arch=arch)
        same = _body(a, spec.kernel_name()) == _body(b, ref.kernel_name())
        assert INERT_KNOBS[knob](spec, arch) == same, (body, knob)
        checked += 1
    assert checked >= 4, checked


# ---------------------------------------------------------------------------
# IR forms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ALL_ARCHS)
def test_ws_bounds_stats_are_buffer_loads(arch):
    """``ws_bounds = "buffer"``: every clamped scalar stats load of the step
    (lse2 and Dsum per C row of the lane) becomes one buffer load."""
    base = _main_ir(arch)
    buf = _main_ir(arch, ws_bounds="buffer")
    assert "buffer.load" not in base
    loads = r"= load float, ptr addrspace\(1\)"
    one = r"= call (i32|float) @llvm\.amdgcn\.raw\.ptr\.buffer\.load\.(i32|f32)\("
    n = len(re.findall(one, buf))
    assert n >= 2 and n % 2 == 0  # lse2 and Dsum
    assert len(re.findall(loads, buf)) == len(re.findall(loads, base)) - n


@pytest.mark.parametrize("arch", ("gfx942", "gfx950"))
def test_cell_vec_transposes_in_quads(arch):
    knobs = dict(dkv_epilogue="direct")
    base = _main_ir(arch, **knobs)
    quad = _main_ir(arch, dkv_cell_vec=4, **knobs)
    # 4 DPP quad permutes per fragment (2 per transpose stage)
    dpp = r"= call i32 @llvm\.amdgcn\.(update|mov)\.dpp"
    frags = len(re.findall(r"store <4 x bfloat>", quad))
    assert len(re.findall(dpp, quad)) - len(re.findall(dpp, base)) == 4 * frags
    # 4 bf16 per store (8 bytes) instead of 2-byte stores
    assert re.search(r"store <4 x bfloat>", quad)
    slices = _main_ir(
        arch, dkv_mode="atomic", dkv_sink="slices", dkv_cell_vec=4, **knobs
    )
    assert re.search(r"store <4 x float>", slices)


@pytest.mark.parametrize("arch", ("gfx942", "gfx950"))
def test_slice_buffer_bounds_have_no_row_branch_stores(arch):
    k = dict(dkv_mode="atomic", dkv_sink="slices")
    clamp = _main_ir(arch, **k)
    buf = _main_ir(arch, ws_bounds="buffer", **k)
    assert "buffer.store" not in clamp
    assert len(re.findall(r"raw\.ptr\.buffer\.store", buf)) >= 4
    direct = _main_ir(arch, ws_bounds="buffer", dkv_epilogue="direct", **k)
    assert len(re.findall(r"raw\.ptr\.buffer\.store", direct)) >= 8


@pytest.mark.parametrize("arch", ALL_ARCHS)
def test_slab_carry_drops_the_per_step_products(arch):
    base = _main_ir(arch)
    carry = _main_ir(arch, slab_carry=True)

    def loop(ir):  # the q loop: from its first carried phi to its latch
        i = ir.index("%bwd_qqt = phi")
        latch = re.search(r"%(for\.latch\.\d+) \]", ir[i:]).group(1)
        return ir[i : ir.index(f"\n{latch}:", i)]

    def muls(ir):
        return len(re.findall(r"= mul (nsw |nuw )*i64", ir))

    assert muls(loop(carry)) + 4 <= muls(loop(base))  # 2 per tensor per step
    assert "%bwd_qoq = phi i64" in carry and "%bwd_qod = phi i64" in carry
    # every loop shape that carries: head_outer (q tiles inner) and edge tiles
    for kw in ({"step_order": "head_outer"}, {"edge_tiles": True}):
        assert "phi i64" in _main_ir(arch, slab_carry=True, **kw)
    # the grouped head_outer loops of edge tiles keep their slabs (inert); the
    # emission is compared (the d64 grouped kernel is resource-refused on
    # some targets: RESOURCE_REFUSED, bypassed here)
    from unittest import mock

    from kernels.common import attention_bwd as AB

    grouped = {"step_order": "head_outer", "edge_tiles": True}
    with mock.patch.object(AB, "resource_refusal", lambda s, a: None):
        spec = validate_attn_bwd_spec(
            AttnBwdSpec(head_size=64, dtype="bf16", slab_carry=True, **grouped), arch
        )
        assert INERT_KNOBS["slab_carry"](spec, arch)
        a = _main_ir(arch, slab_carry=True, **grouped)
        b = _main_ir(arch, **grouped)
        ref = validate_attn_bwd_spec(
            AttnBwdSpec(head_size=64, dtype="bf16", **grouped), arch
        )
    assert _body(a, spec.kernel_name()) == _body(b, ref.kernel_name())


@pytest.mark.parametrize("sched", SCHEDS)
@pytest.mark.parametrize("arch", ALL_ARCHS)
def test_slab_carry_builds_with_every_schedule(arch, sched):
    """slab_carry with every q-loop schedule the validator admits it with,
    on every head size and both sequence modes: the main kernel builds. The
    q loop carries the two slab offsets before the deferred dQ partial of
    ``stage_table_atomics`` (its partial is read past the offsets: the
    carried row and head of the partial are i32, the offsets i64). Emission
    only: a resource refusal (RESOURCE_REFUSED) is bypassed."""
    from unittest import mock

    from kernels.common import attention_bwd as AB

    built = 0
    with mock.patch.object(AB, "resource_refusal", lambda s, a: None):
        for d, dt in ((32, "fp16"), (64, "bf16"), (128, "bf16")):
            for seq in ("batched", "thd"):
                spec = AttnBwdSpec(
                    head_size=d, dtype=dt, seq_mode=seq, slab_carry=True, sched=sched
                )
                try:
                    validate_attn_bwd_spec(spec, arch)
                except ValueError as exc:
                    assert "slab_carry" not in str(exc), (d, seq, str(exc))
                    continue
                kernel = build_attn_bwd_main(spec, arch=arch)
                built += 1
                if d != 64:
                    continue
                ir = lower(kernel, arch=arch)
                assert "%bwd_qoq = phi i64" in ir and "%bwd_qod = phi i64" in ir
                deferred = "%bwd_qdqrow = phi i32" in ir
                assert deferred == (sched == "stage_table_atomics"), (seq, sched)
                if deferred:
                    assert "%bwd_qdqhead = phi i32" in ir
    if sched in ("none", "stage_table", "stage_table_atomics") and arch in (
        "gfx942",
        "gfx950",
    ):
        assert built == 6, built


def test_narrow_width_follows_the_request():
    base = dict(b=2, h_q=4, h_k=4, h_v=4, s_q=40, s_kv=40, d_qk=64, d_v=64)

    def req(unit, align):
        st = {}
        for n in ("q", "k", "v", "o", "do", "dq", "dk", "dv"):
            ss = 64 + unit
            sh = 40 * ss + unit
            st[n] = (4 * sh + unit, sh, ss, 1)
        return AttnBwdRequest(**base, strides=st, tensor_alignment=align)

    assert attn_bwd_narrow_width(req(4, 8)) == 4
    assert attn_bwd_narrow_width(req(4, 4)) == 2
    assert attn_bwd_narrow_width(req(2, 8)) == 2
    assert attn_bwd_narrow_width(req(1, 8)) == 1
    for unit, align, cap, want in (
        (4, 8, 4, 4), (4, 8, 2, 2), (4, 4, 4, 2), (1, 8, 4, 1), (4, 8, 1, 1),
    ):  # fmt: skip
        plan = attn_bwd_plan(
            req(unit, align), "gfx942", policy=_policy({"narrow_vec": cap})
        )
        for stage, spec in plan.specs.items():
            if hasattr(spec, "narrow_vec"):
                assert spec.narrow_vec == want, (stage, unit, align, cap)
        if want == 1:
            default = attn_bwd_plan(req(unit, align), "gfx942", policy=_policy({}))
            assert [s.kernel for s in plan.launches] == [
                s.kernel for s in default.launches
            ]
    # the wide class ignores the cap (no narrow access)
    wide = AttnBwdRequest(**base)
    p = attn_bwd_plan(wide, "gfx942", policy=_policy({"narrow_vec": 4}))
    assert p.specs["main"].narrow_vec == 1


@pytest.mark.parametrize("arch", ("gfx942", "gfx1151"))
def test_narrow_vec_widens_every_kernel(arch):
    st = {}
    for n in ("q", "k", "v", "o", "do", "dq", "dk", "dv"):
        st[n] = (4 * (40 * 68 + 4) + 4, 40 * 68 + 4, 68, 1)
    req = AttnBwdRequest(
        b=2, h_q=4, h_k=4, h_v=4, s_q=40, s_kv=40, d_qk=64, d_v=64, dtype="bf16",
        strides=st, tensor_alignment=8,
    )  # fmt: skip
    plan = attn_bwd_plan(req, arch, policy=_policy({"narrow_vec": 4}))
    for step in plan.launches:
        ir = lower(_BUILDERS[step.stage](plan.specs[step.stage], arch=arch), arch=arch)
        if step.stage in ("main", "prep", "convert"):
            assert re.search(r"<4 x bfloat>", ir), step.stage


def test_default_kernels_keep_their_names():
    """No lever value set: every kernel name and IR of a plan is the default's
    (the IR byte identity against the pre-lever tree is checked separately)."""
    req = AttnBwdRequest(
        b=2, h_q=8, h_k=2, h_v=2, s_q=100, s_kv=100, d_qk=64, d_v=64, causal=True
    )
    explicit = {
        "ws_bounds": "clamp", "dkv_cell_vec": 1, "narrow_vec": 1,
        "slab_carry": False, "dkv_sink": "atomic",
    }  # fmt: skip
    for arch in ALL_ARCHS:
        a = attn_bwd_plan(req, arch, policy=_policy({}))
        b = attn_bwd_plan(req, arch, policy=_policy(explicit))
        assert a == b


# ---------------------------------------------------------------------------
# resource gate of every new value (and the slice reduce)
# ---------------------------------------------------------------------------

_ATOMIC = dict(dkv_mode="atomic")
_SLICES = dict(dkv_mode="atomic", dkv_sink="slices")
# name -> (spec knobs, archs (None: all), problem fields, reference knobs): the
# reference is the same configuration without the lever (the atomic sink for
# the slice sink), so a gate finding is charged to the lever only when the
# reference does not have it
GATE_VARIANTS = {
    "slices": (_SLICES, None, {}, _ATOMIC),
    "slices_pair": (
        {**_SLICES, "dkv_epilogue": "lds_staged_pair"}, None, {}, _ATOMIC,
    ),
    "slices_direct": ({**_SLICES, "dkv_epilogue": "direct"}, None, {}, _ATOMIC),
    "slices_vec4": ({**_SLICES, "dkv_store_vec": 4}, None, {}, _ATOMIC),
    "slices_buffer": ({**_SLICES, "ws_bounds": "buffer"}, None, {}, _ATOMIC),
    "slices_cell4": (
        {**_SLICES, "dkv_epilogue": "direct", "dkv_cell_vec": 4},
        ("gfx942", "gfx950"), {}, _ATOMIC,
    ),
    "slices_split": (
        {**_SLICES, "dq_mode": "split"}, ("gfx942", "gfx950"), {},
        {**_ATOMIC, "dq_mode": "split"},
    ),
    "slices_thd": (_SLICES, None, {"seq_mode": "thd"}, _ATOMIC),
    "ws_buffer": ({"ws_bounds": "buffer"}, None, {}, {}),
    "cell4": (
        {"dkv_epilogue": "direct", "dkv_cell_vec": 4}, ("gfx942", "gfx950"), {},
        {"dkv_epilogue": "direct"},
    ),
    "slab_carry": ({"slab_carry": True}, None, {}, {}),
    "slab_carry_thd": ({"slab_carry": True}, None, {"seq_mode": "thd"}, {}),
    "slab_carry_hout": (
        {"slab_carry": True, "step_order": "head_outer"}, None, {},
        {"step_order": "head_outer"},
    ),
    "slab_carry_edge": (
        {"slab_carry": True, "edge_tiles": True}, None, {}, {"edge_tiles": True},
    ),
    "slab_carry_sta": (
        {"slab_carry": True, "sched": "stage_table_atomics"}, None, {},
        {"sched": "stage_table_atomics"},
    ),
    "narrow4": ({"narrow_vec": 4}, None, {"stage_vec": 1}, {}),
}  # fmt: skip
GATE_HEADS = (64, 128)
GATE_DTYPES = ("fp16", "bf16")


def gate_spec(variant: str, d: int, dtype: str, *, reference: bool = False):
    """The spec of a gate row (``reference``: the same row without the lever);
    the refusal test pins RESOURCE_REFUSED rows ``("access", arch, variant, d,
    dtype)`` with it."""
    knobs, _archs, problem, ref_knobs = GATE_VARIANTS[variant]
    return AttnBwdSpec(
        head_size=d, dtype=dtype, **problem, **(ref_knobs if reference else knobs)
    )


def _gate_params():
    return [
        pytest.param(arch, name, id=f"{arch}-{name}")
        for arch in ALL_ARCHS
        for name, (_k, archs, _p, _r) in GATE_VARIANTS.items()
        if archs is None or arch in archs
    ]


@pytest.mark.parametrize("arch,variant", _gate_params())
def test_resource_gate_of_the_new_values(arch, variant):
    """The resource gate (``evaluate_resource_gate``: zero scratch, zero
    spills, the register / LDS budgets, the q-loop AGPR-copy rule; plus LDS
    within the plan) of every new value on the default tiles (d64 / d128, both
    dtypes, band mask; production comgr on llvm22, else skipped). Every row
    the validator admits passes outright (a row that fails is refused
    exactly, RESOURCE_REFUSED rows ``access`` of the refusal test, and its
    refusal is the lever's own: the reference row without the lever is
    legal, or refused itself: a schedule value's own refusal). The slice
    reduce kernel passes the gate."""
    pytest.importorskip("rocke.runtime.comgr")
    from rocke.helpers.compile import compile_kernel

    from kernels.common.attention_bwd import attn_bwd_arch_facts, lds_plan

    from ._attention_bwd_resources import (
        GATE_FLAVOR,
        comgr_is_production,
        comgr_version,
        current_flavor,
        evaluate_resource_gate,
        find_objdump,
        pin_production_comgr,
    )

    pin_production_comgr()  # torch's bundled comgr, by path; torch not imported
    if find_objdump() is None:
        pytest.skip("llvm-objdump not found (needed for the q-loop AGPR copy rule)")
    flavor, comgr = current_flavor(), comgr_version()
    if flavor != GATE_FLAVOR or not comgr_is_production(comgr):
        pytest.skip(f"gate valid on {GATE_FLAVOR} with the production comgr only")
    _knobs, _archs, problem, _ref = GATE_VARIANTS[variant]
    facts = attn_bwd_arch_facts(arch)

    def findings(spec) -> set:
        r = validate_attn_bwd_spec(spec, arch)
        art = compile_kernel(
            build_attn_bwd_main(spec, arch=arch), arch=arch,
            capture_ir_text=False, backend="python",
        )  # fmt: skip
        res = evaluate_resource_gate(
            art.hsaco, arch=arch, lds_capacity_bytes=facts.lds_capacity_bytes,
            spec=r,
        )  # fmt: skip
        if not res.valid:
            pytest.skip(f"gate not valid here ({res.flavor}, {res.comgr})")
        out = set(res.failures) | {f"warn:{w}" for w in res.warnings}
        if res.resources.lds_bytes > lds_plan(r, arch).total_bytes:
            out.add("lds_above_plan")
        return out

    got, refused = {}, []
    for d in GATE_HEADS:
        for dt in GATE_DTYPES:
            ref = gate_spec(variant, d, dt, reference=True)
            try:
                validate_attn_bwd_spec(ref, arch)  # the reference row is legal
            except ValueError as exc:  # or refused for its own resources
                if "RESOURCE_REFUSED" not in str(exc):
                    raise
            try:
                validate_attn_bwd_spec(gate_spec(variant, d, dt), arch)
            except ValueError as exc:
                if "RESOURCE_REFUSED" not in str(exc):
                    raise
                refused.append((d, dt))
                continue
            lever = findings(gate_spec(variant, d, dt))
            if lever:
                got[(d, dt)] = tuple(sorted(lever))
    reduce_fail = ()
    if variant.startswith("slices"):
        aux = AttnBwdSpec(head_size=64, dtype="bf16", **problem).aux_spec("convert")
        art = compile_kernel(
            _BUILDERS["reduce"](aux, arch=arch), arch=arch,
            capture_ir_text=False, backend="python",
        )  # fmt: skip
        res = evaluate_resource_gate(
            art.hsaco, arch=arch, lds_capacity_bytes=facts.lds_capacity_bytes
        )
        reduce_fail = tuple(res.failures)
    assert not reduce_fail, reduce_fail
    assert not got, f"gate findings {got} (refused rows: {refused})"


# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------

try:
    from ._attention_bwd_harness import ARCH, DEVICE_ARCHS, narrow_inputs, run_case
    from .sdpa.bwd_cases import compare_bwd
    from .sdpa.bwd_kernel_cases import selected_configs
except ImportError:  # pragma: no cover
    ARCH, DEVICE_ARCHS = None, ()

CASES = (
    "bwd_seqlen_33x72_br",
    "bwd_gqa_gqa4_causal_tl",
    "bwd_kern_hkv_842_tl_fp16_d64",
)
# name -> (knobs, archs (None: every device target))
LEVERS = {
    "ws_buffer": ({"ws_bounds": "buffer"}, None),
    "ws_buffer_vgpr2": ({"ws_bounds": "buffer", "global_path": "vgpr", "ring_depth": 2}, None),
    "slab_carry": ({"slab_carry": True}, None),
    "slab_carry_edge": ({"slab_carry": True, "edge_tiles": True}, None),
    "slab_carry_head_outer": (
        {"slab_carry": True, "step_order": "head_outer"},
        None,
    ),
    "slab_carry_head_outer_edge": (
        {"slab_carry": True, "step_order": "head_outer", "edge_tiles": True},
        None,
    ),
    # the deferred dQ partial of stage_table_atomics carried with the slab
    # offsets (atomic dQ: the schedule refuses split dQ)
    "slab_carry_sta": (
        {"slab_carry": True, "sched": "stage_table_atomics"},
        None,
    ),
    "cell_vec4": (
        {"dkv_epilogue": "direct", "dkv_cell_vec": 4},
        ("gfx942", "gfx950"),
    ),
    "slices_buffer": (
        {"dkv_sink": "slices", "ws_bounds": "buffer"},
        None,
    ),
    "slices_direct_buffer": (
        {"dkv_sink": "slices", "dkv_epilogue": "direct", "ws_bounds": "buffer"},
        ("gfx942", "gfx950"),
    ),
    "slices_cell_vec4": (
        {"dkv_sink": "slices", "dkv_epilogue": "direct", "dkv_cell_vec": 4},
        ("gfx942", "gfx950"),
    ),
    "slices_cell_vec4_buffer": (
        {"dkv_sink": "slices", "dkv_epilogue": "direct", "dkv_cell_vec": 4,
         "ws_bounds": "buffer"},
        ("gfx942", "gfx950"),
    ),
}  # fmt: skip
# the reference run of a lever (the same knobs without it)
_LEVER_KEYS = {"ws_bounds", "slab_carry", "dkv_cell_vec"}


def _passes(run):
    probs = compare_bwd(run.case, run.dq, run.dk, run.dv, run.ref)
    assert not probs, probs
    assert not run.integrity, run.integrity


def _params():
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, None, id="no-device")]
    i, n = (int(x) for x in os.environ.get("ROCKE_BWD_SHARD", "0/1").split("/"))
    out = []
    for k, cid in enumerate(CASES):
        for name, (_knobs, archs) in LEVERS.items():
            if archs is not None and ARCH not in archs:
                continue
            if (k * len(LEVERS) + list(LEVERS).index(name)) % n != i:
                continue
            for cfg in selected_configs(ARCH):
                out.append(pytest.param(cid, name, cfg, id=f"{cid}-{name}-{cfg}"))
    return out


_REF: dict = {}


@pytest.mark.gpu
@pytest.mark.parametrize("case_id,lever,config", _params())
def test_lever_is_bitwise(case_id, lever, config):
    if ARCH not in DEVICE_ARCHS:
        pytest.skip("no supported device")
    from ._attention_bwd_harness import plan_case

    case = find_bwd_case(case_id)
    knobs, _ = LEVERS[lever]
    split = ARCH in SPLIT_DQ_ARCHS and knobs.get("sched") != "stage_table_atomics"
    base_knobs = {k: v for k, v in knobs.items() if k not in _LEVER_KEYS}
    if split:
        base_knobs["dq_mode"] = "split"
        knobs = {**knobs, "dq_mode": "split"}
    try:
        plan = plan_case(case, arch=ARCH, config=config, overrides=base_knobs)
    except ValueError as exc:  # the reference itself (e.g. a schedule value)
        if "RESOURCE_REFUSED" not in str(exc):
            raise
        pytest.skip(f"the reference {base_knobs} is refused: {exc}")
    spec = plan.specs["main"]
    try:
        validate_attn_bwd_spec(
            dataclasses.replace(
                spec, **{k: v for k, v in knobs.items() if hasattr(spec, k)}
            ),
            ARCH,
        )
    except ValueError as exc:
        pytest.skip(f"{knobs} not legal on {spec.kernel_name()}: {exc}")
    key = (case_id, config, repr(sorted(base_knobs.items())))
    if key not in _REF:
        ref = run_case(case, arch=ARCH, config=config, overrides=base_knobs)
        _passes(ref)
        _REF[key] = ref
    ref = _REF[key]
    run = run_case(case, arch=ARCH, config=config, overrides=knobs)
    _passes(run)
    main = run.plan.specs["main"]
    for k, v in knobs.items():
        if hasattr(main, k) and k != "dkv_sink":
            assert getattr(main, k) == v, k
    if split or main.dkv_mode == "direct" or main.dkv_sink == "slices":
        assert np.array_equal(run.dk, ref.dk), f"dK differs under {lever}"
        assert np.array_equal(run.dv, ref.dv), f"dV differs under {lever}"
    if split:
        assert np.array_equal(run.dq, ref.dq), f"dQ differs under {lever}"


def _narrow_params():
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, None, None, id="no-device")]
    out = []
    for cid in ("bwd_seqlen_33x72_br", "bwd_gqa_gqa4_causal_tl"):
        for view in ("bhsd", "bshd"):
            for unit in (2, 4):
                out.append(pytest.param(cid, view, unit, "default",
                                        id=f"{cid}-{view}-u{unit}"))  # fmt: skip
    return out


@pytest.mark.gpu
@pytest.mark.parametrize("case_id,view,unit,config", _narrow_params())
def test_narrow_vec_is_bitwise(case_id, view, unit, config):
    """stage_vec = 1 views whose strides and bases allow 2 / 4 elements: the
    capped widths give the element-wise run's bits (oracle and integrity:
    the NaN stride gaps stay unread)."""
    if ARCH not in DEVICE_ARCHS:
        pytest.skip("no supported device")
    case = find_bwd_case(case_id)
    inputs, request = narrow_inputs(case, view, unit, unit=unit)
    split = {"dq_mode": "split"} if ARCH in SPLIT_DQ_ARCHS else {}
    ref = run_case(
        case, arch=ARCH, config=config, inputs=inputs, request=request,
        overrides=dict(split),
    )  # fmt: skip
    assert ref.plan.specs["main"].stage_vec == 1
    _passes(ref)
    run = run_case(
        case, arch=ARCH, config=config, inputs=inputs, request=request,
        overrides={**split, "narrow_vec": 4},
    )  # fmt: skip
    _passes(run)
    assert run.plan.specs["main"].narrow_vec == unit
    assert run.plan.specs["prep"].narrow_vec == unit
    assert np.array_equal(run.dk, ref.dk) and np.array_equal(run.dv, ref.dv)
    if split:
        assert run.plan.specs["dq"].narrow_vec == unit
        assert np.array_equal(run.dq, ref.dq)
