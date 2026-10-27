# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``loop_sync = "counted"``: next-step loads before the dQ atomics, counted waits.

CPU / IR tests (every MFMA configuration, head size, dtype and the variants
below, both ``counted_issue`` values):

* the value sets, the couplings (CDNA only, register staging without a ring
  or stage table, default staging / stats order, not with ``barrier_merge``,
  the vmcnt field) and that the defaults are unsalted and emit the default IR;
* :func:`_attention_bwd_counted_model.counted_ir_problems` on every q loop:
  the counted barriers' ``N`` equal the dQ atomics between the last global load
  and the barrier (through the back-edge), no full / LDS-only barrier in the
  loop, the next-step loads before the step's first atomic, the step-top LDS
  stores fed by the carried values; the per-barrier counts are those of the
  model table (``expected_waits``); a 0 / 1-trip guard loop loads the first
  step of every q loop;
* the counted step is the full step's math: the same matrix ops in the same
  order and the same dQ atomics per step;
* mutations of the count (``N + 1``, ``N = 0``, a lost load event) are caught.

Compile tests (comgr + ``llvm-objdump``; production comgr for the resource
gate): :func:`_attention_bwd_counted_model.counted_isa_problems` on the
emitted code of every counted variant (no load in flight at a barrier, no
wait that drains an atomic issued after the youngest load, three barriers per
step, no ``vmcnt(0)`` in a ``late`` loop), the ISA catches the mutated counts,
and the resource gate passes for every counted kernel and its full twin (the
specs that fail it are refused by the validator).
"""

from __future__ import annotations

import dataclasses
import os
from functools import cache
from types import SimpleNamespace

import pytest
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower

from kernels.common import _attention_bwd_counted as counted_mod
from kernels.common._attention_bwd_body import BwdTileGeometry
from kernels.common._attention_bwd_counted import (
    COUNTED_ISSUES,
    COUNTED_VMCNT_MAX,
    CountedWaits,
)
from kernels.common.attention_bwd import (
    LOOP_SYNCS,
    AttnBwdSpec,
    attn_bwd_arch_facts,
    build_attn_bwd_main,
    check_counted_sync,
    register_estimates,
    validate_attn_bwd_spec,
)

from ._attention_bwd_test_utils import distinct_configs
from ._attention_bwd_counted_model import (
    counted_ir_problems,
    counted_isa_problems,
    expected_waits,
    ir_events,
)
from .sdpa.bwd_kernel_cases import config_knobs

MFMA_ARCHS = ("gfx942", "gfx950")
RDNA_ARCHS = ("gfx1151", "gfx1201")
HEADS = (32, 64, 128)
DTYPES = ("fp16", "bf16")

# Variants of the counted step: masks, edge tiles, head packing, atomic dK /
# dV, THD, narrow access, the step orders, q rotation, the LDS P / dS route,
# a non-default stage order, per-fragment G4, LDS accumulators, the THD slab
# hoist, the token-major workspace and mirror pairing.
VARIANTS = {
    "band": dict(mask_class="band"),
    "none": dict(mask_class="none"),
    "band_edge": dict(mask_class="band", edge_tiles=True),
    "band_pack": dict(mask_class="band", head_pack=True),
    "band_atomic_pack": dict(mask_class="band", dkv_mode="atomic", head_pack=True),
    "thd_band_atomic": dict(mask_class="band", seq_mode="thd", dkv_mode="atomic"),
    "sv1_band": dict(mask_class="band", stage_vec=1),
    # the flat head-outer loop (the grouped form with edge tiles is refused)
    "band_hout": dict(mask_class="band", step_order="head_outer"),
    "band_pack_hout": dict(mask_class="band", head_pack=True, step_order="head_outer"),
    "band_qrot_edge": dict(mask_class="band", q_rotate=True, edge_tiles=True),
    "band_qrot_hout": dict(mask_class="band", q_rotate=True, step_order="head_outer"),
    "band_ptlds": dict(mask_class="band", pt_route="lds"),
    "band_g4g1g3": dict(mask_class="band", stage_order="g4g1g3"),
    "band_dqfrag": dict(mask_class="band", dq_issue="fragment_regs"),
    "band_accl": dict(mask_class="band", acc_in_lds=True, agpr_alloc=(0, 0)),
    "thd_band_slab": dict(mask_class="band", seq_mode="thd", dkv_slab_hoist=True),
    "band_tokmajor": dict(mask_class="band", ws_layout="token_major"),
    "band_mirror": dict(mask_class="band", lb_mirror=True),
    # refused on every row (resource gate): see test_mirror_edge_rows_are_refused
    "band_mirror_edge": dict(mask_class="band", lb_mirror=True, edge_tiles=True),
}


def spec_of(arch, cfg, d, dt, var, issue) -> AttnBwdSpec:
    kw = dict(config_knobs(cfg, arch), **VARIANTS[var])
    return AttnBwdSpec(
        head_size=d, dtype=dt, loop_sync="counted", counted_issue=issue, **kw
    )


def _legal(arch, cfg, d, dt, var, issue) -> bool:
    try:
        validate_attn_bwd_spec(spec_of(arch, cfg, d, dt, var, issue), arch)
    except ValueError:
        return False
    return True


ROWS = [
    row
    for arch in MFMA_ARCHS
    for cfg in distinct_configs(arch)
    for d in HEADS
    for dt in DTYPES
    for var in VARIANTS
    for issue in COUNTED_ISSUES
    for row in [(arch, cfg, d, dt, var, issue)]
    if _legal(*row)
]


def _ids(row):
    return "-".join(map(str, row))


@cache
def kernel_of(arch, cfg, d, dt, var, issue):
    return build_attn_bwd_main(spec_of(arch, cfg, d, dt, var, issue), arch=arch)


def _walk(ops, anc=()):
    for op in ops:
        yield op, anc
        for r in op.regions:
            yield from _walk(r.ops, anc + (op,))


def _holds(op, name):
    return any(o.name == name for o, _ in _walk(op.regions[0].ops))


def q_loops(kernel):
    """The q loops (scf.for ops holding matrix ops and no matrix loop)."""
    mloops = [
        op for op, _ in _walk(kernel.body.ops)
        if op.name == "scf.for" and _holds(op, "tile.mma")
    ]  # fmt: skip
    ids = {id(x) for x in mloops}
    return [
        lp for lp in mloops
        if not any(id(o) in ids for o, _ in _walk(lp.regions[0].ops))
    ]  # fmt: skip


def _predecessors(kernel) -> dict:
    """``{id(op): [ops before it in its region]}`` for every op."""
    out = {}

    def visit(ops):
        for i, op in enumerate(ops):
            out[id(op)] = ops[:i]
            for r in op.regions:
                visit(r.ops)

    visit(kernel.body.ops)
    return out


# --------------------------------------------------------------------------- spec
def test_values_defaults_and_couplings():
    assert LOOP_SYNCS == ("full", "lds_only", "counted")
    assert COUNTED_ISSUES == ("early", "late")
    base = AttnBwdSpec(head_size=64)
    assert (base.loop_sync, base.counted_issue, base.lb_mirror) == (
        "full", "early", False,
    )  # fmt: skip
    spelled = AttnBwdSpec(head_size=64, loop_sync="full", counted_issue="early")
    assert spelled.kernel_name("main") == base.kernel_name("main")
    with pytest.raises(ValueError, match="counted_issue"):
        AttnBwdSpec(head_size=64, counted_issue="late")
    with pytest.raises(ValueError, match="counted_issue"):
        AttnBwdSpec(head_size=64, loop_sync="counted", counted_issue="mid")
    for issue in COUNTED_ISSUES:
        s = AttnBwdSpec(head_size=64, loop_sync="counted", counted_issue=issue)
        assert s.kernel_name("main") != base.kernel_name("main")
        for arch in MFMA_ARCHS:
            validate_attn_bwd_spec(s, arch)
        for arch in RDNA_ARCHS:
            with pytest.raises(ValueError, match="CDNA"):
                validate_attn_bwd_spec(s, arch)
    c = AttnBwdSpec(head_size=64, loop_sync="counted")
    for kw, match in (
        (dict(ring_depth=2), "ring_depth"),
        (dict(qdo_stage="loads_first"), "qdo_stage"),
        (dict(stats_load="step_top"), "stats_load"),
        (dict(sched="stage_table_atomics", counted_issue="late"), "early"),
        (dict(step_order="head_outer", edge_tiles=True), "grouped head-outer"),
    ):
        for arch in MFMA_ARCHS:
            with pytest.raises(ValueError, match=match):
                validate_attn_bwd_spec(dataclasses.replace(c, **kw), arch)
    # the flat head-outer loop, edge tiles alone and the schedule values (the
    # deferred atomics with the early issue) stay legal
    for kw in (dict(step_order="head_outer"), dict(edge_tiles=True),
               dict(sched="stage_table"), dict(sched="stage_table_atomics")):  # fmt: skip
        for arch in MFMA_ARCHS:
            validate_attn_bwd_spec(dataclasses.replace(c, **kw), arch)
    with pytest.raises(ValueError, match="global_path"):
        AttnBwdSpec(head_size=64, loop_sync="counted", global_path="dma")


def test_counted_excludes_barrier_merge():
    merged = AttnBwdSpec(head_size=64, loop_sync="counted", barrier_merge="on")
    with pytest.raises(ValueError, match="barrier_merge"):
        check_counted_sync(merged.resolved("gfx942"), "gfx942")
    for arch in MFMA_ARCHS:
        with pytest.raises(ValueError, match="barrier_merge"):
            validate_attn_bwd_spec(merged, arch)
    off = AttnBwdSpec(head_size=64, loop_sync="counted", barrier_merge="off")
    check_counted_sync(off.resolved("gfx942"), "gfx942")
    full = AttnBwdSpec(head_size=64, barrier_merge="on").resolved("gfx942")
    check_counted_sync(full, "gfx942")  # no coupling without counted


def test_counted_excludes_the_split_dq_main_kernel():
    """The counted waits keep the step's dQ atomics in flight; the split-dQ
    main kernel issues none, so the coupling refuses it."""
    for arch in MFMA_ARCHS:
        with pytest.raises(ValueError, match="dq_mode='atomic'"):
            validate_attn_bwd_spec(
                AttnBwdSpec(head_size=64, loop_sync="counted", dq_mode="split"), arch
            )
    from dispatch.attention_bwd.tuning_common import backward_axis

    assert not backward_axis("loop_sync").applies("counted", {"dq_mode": "split"})


def test_registry_axes():
    from dispatch.attention_bwd.tuning_common import backward_axis

    ls = backward_axis("loop_sync")
    for arch in MFMA_ARCHS:
        assert ls.values(arch) == ("full", "lds_only", "counted")
    for arch in RDNA_ARCHS:
        assert ls.values(arch) == ("full", "lds_only")
    assert ls.applies("counted", {})
    assert not ls.applies("counted", {"ring_depth": 2})
    assert not ls.applies("counted", {"qdo_stage": "joint"})
    assert ls.applies("counted", {"sched": "stage_table"})
    assert not ls.applies("counted", {"barrier_merge": "on"})
    assert not ls.applies("counted", {"step_order": "head_outer", "edge_tiles": True})
    assert ls.applies("counted", {"step_order": "head_outer"})
    ci = backward_axis("counted_issue")
    assert ci.choices == COUNTED_ISSUES and ci.family == 5
    assert ci.values("gfx1151") == ("early",)
    assert ci.applies("late", {"loop_sync": "counted"})
    sta = {"loop_sync": "counted", "sched": "stage_table_atomics"}
    assert ci.applies("early", sta) and not ci.applies("late", sta)
    assert not ci.applies("late", {"loop_sync": "full"})
    for name in ("qdo_stage", "stats_load"):
        ax = backward_axis(name)
        other = [v for v in ax.choices if v != ax.default][0]
        assert ax.applies(other, {}) and not ax.applies(other, {"loop_sync": "counted"})


def test_vmcnt_field_bounds_the_atomics():
    CountedWaits(COUNTED_VMCNT_MAX)
    with pytest.raises(ValueError, match="vmcnt"):
        CountedWaits(COUNTED_VMCNT_MAX + 1)
    # every legal counted row stays inside the field
    for row in ROWS:
        r = validate_attn_bwd_spec(spec_of(*row), row[0])
        assert BwdTileGeometry.from_spec(r, row[0]).counted_atomics <= 63


def test_counted_estimate_carries_the_next_step():
    for arch in MFMA_ARCHS:
        for d in HEADS:
            full = validate_attn_bwd_spec(
                AttnBwdSpec(head_size=d, agpr_alloc=(0, 0)), arch
            )
            cnt = dataclasses.replace(full, loop_sync="counted")
            wave = attn_bwd_arch_facts(arch).wave_size
            extra = (
                full.block_m * d // (wave * full.waves) + 2 * full.block_m * 16 // wave
            )
            assert (
                register_estimates(cnt, arch)["arch_vgpr"]
                == register_estimates(full, arch)["arch_vgpr"] + extra
            )


# --------------------------------------------------------------------------- IR
@pytest.mark.parametrize("row", ROWS, ids=_ids)
def test_counted_ir_model(row):
    arch, issue = row[0], row[-1]
    r = validate_attn_bwd_spec(spec_of(*row), arch)
    n = BwdTileGeometry.from_spec(r, arch).counted_atomics
    kernel = kernel_of(*row)
    loops = q_loops(kernel)
    assert loops
    prev = _predecessors(kernel)
    for lp in loops:
        probs, waits = counted_ir_problems(lp)
        assert not probs, probs
        assert waits == expected_waits(n, issue), (waits, n, issue)
        # no load pending at the loop entry: vmcnt(0) right after the guard
        before = prev[id(lp)]
        assert before[-1].name == "tile.s_waitcnt", [o.name for o in before[-3:]]
        assert before[-1].attrs["vmcnt"] == 0
        assert before[-2].name == "scf.for"
        # the staged values enter from a 0 / 1-trip guard loop of loads
        names = [m["name"] for m in lp.attrs["iter_args"]]
        pf = [i for i, nm in enumerate(names) if "pf" in nm.rsplit("_", 1)[-1]]
        assert pf, names
        inits = lp.operands[3:]
        srcs = {inits[i].op.name if inits[i].op is not None else None for i in pf}
        assert srcs == {"scf.for"}, srcs
        guard = inits[pf[0]].op
        assert any(k == "L" for k, _ in ir_events(guard.regions[0].ops))
        assert guard.operands[1].op.name == "arith.select"


@pytest.mark.parametrize("row", ROWS, ids=_ids)
def test_counted_lds_barriers_separate_cross_wave_accesses(row, monkeypatch):
    """The LDS hazard model of ``test_attention_bwd_ir`` on the counted
    kernel: every counted barrier drains LDS (``lgkmcnt(0)``), so it counts
    as a CTA barrier, and no two LDS accesses race across waves."""
    from . import test_attention_bwd_ir as T

    kernel, tags = T.traced_spec(spec_of(*row), row[0])
    prev = _predecessors(kernel)
    bare = [op for op, _ in _walk(kernel.body.ops) if op.name == "tile.s_barrier_bare"]
    assert bare
    for op in bare:
        w = prev[id(op)][-1]
        assert w.name == "tile.s_waitcnt" and w.attrs["lgkmcnt"] == 0
    monkeypatch.setattr(T, "_BARRIERS", T._BARRIERS + ("tile.s_barrier_bare",))
    # A mirror kernel's second kv tile is a CTA-uniform branch that starts
    # with a full barrier; the model does not count barriers in a branch, so
    # each tile's code is checked as its own program.
    top = kernel.body.ops
    k = next(
        (
            i
            for i, op in enumerate(top)
            if op.name == "scf.if"
            and op.regions[0].ops
            and op.regions[0].ops[0].name == "tile.sync"
        ),
        None,
    )
    parts = [top] if k is None else [top[:k], top[k].regions[0].ops]
    for ops in parts:
        part = SimpleNamespace(body=SimpleNamespace(ops=ops))
        assert not T.lds_hazards(part, tags)[:4]


@pytest.mark.parametrize("arch", MFMA_ARCHS)
@pytest.mark.parametrize("var", ("band", "band_hout", "band_pack", "band_ptlds"))
def test_counted_step_is_the_full_step(arch, var):
    """Same matrix ops in the same order and the same dQ atomics per q loop as
    the full twin (the math and its order are unchanged)."""
    for cfg in distinct_configs(arch):
        for issue in COUNTED_ISSUES:
            row = (arch, cfg, 64, "fp16", var, issue)
            if not _legal(*row):
                continue
            spec = spec_of(*row)
            full = dataclasses.replace(spec, loop_sync="full", counted_issue="early")
            ka, kb = kernel_of(*row), build_attn_bwd_main(full, arch=arch)
            for la, lb in zip(q_loops(ka), q_loops(kb), strict=True):
                ma = [o.attrs["op_id"] for o, _ in _walk(la.regions[0].ops)
                      if o.name == "tile.mma"]  # fmt: skip
                mb = [o.attrs["op_id"] for o, _ in _walk(lb.regions[0].ops)
                      if o.name == "tile.mma"]  # fmt: skip
                assert ma == mb
                aa = [k for k, _ in ir_events(la.regions[0].ops) if k == "A"]
                ab = [k for k, _ in ir_events(lb.regions[0].ops) if k == "A"]
                assert len(aa) == len(ab) > 0


def test_counted_ir_lowers_on_every_mfma_target():
    for arch in MFMA_ARCHS:
        for issue in COUNTED_ISSUES:
            k = kernel_of(arch, "default", 128, "bf16", "band_edge", issue)
            ir = lower(k, arch=arch, llvm_flavor="llvm22")
            assert "@llvm.amdgcn.s.barrier()" in ir


# --------------------------------------------------------------------------- mutations
def _mutated(monkeypatch, mode):
    orig = CountedWaits.barrier

    def barrier(self, b):
        n = self._since_load
        if mode == "plus1":
            self._since_load = n + 1
        elif mode == "zero":
            self._since_load = 0
        try:
            orig(self, b)
        finally:
            self._since_load = n

    if mode == "lost_load":
        monkeypatch.setattr(CountedWaits, "loads", lambda self: None)
    else:
        monkeypatch.setattr(CountedWaits, "barrier", barrier)


@pytest.mark.parametrize("mode", ("plus1", "zero", "lost_load"))
@pytest.mark.parametrize("issue", COUNTED_ISSUES)
def test_ir_model_catches_wrong_counts(monkeypatch, mode, issue):
    _mutated(monkeypatch, mode)
    spec = spec_of("gfx942", "default", 64, "fp16", "band_edge", issue)
    kernel = build_attn_bwd_main(spec, arch="gfx942")
    found = [p for lp in q_loops(kernel) for p in counted_ir_problems(lp)[0]]
    if mode == "lost_load" and issue == "late":
        # late: the loads sit right before G4 and the entry count is n, so a
        # lost reset still gives n at the middle barrier but 2 n at the end
        assert any("barrier 2" in p for p in found), found
    else:
        assert found, (mode, issue)


# --------------------------------------------------------------------------- ISA
def _isa_text(body: list[str], *, rotate: bool = False) -> str:
    """A tiny ``llvm-objdump`` style kernel: a prologue, then ``body`` as a
    loop closed by a back-edge to its first instruction (``rotate``: the
    loop's exit test laid out before its header, as a backend may)."""
    lines = ["0000000000001000 <k>:"]
    addr = 0x1000
    out = []

    def emit(text):
        nonlocal addr
        out.append((addr, text))
        addr += 4

    emit("s_nop 0")
    if rotate:
        emit("s_branch BRANCH_TO_HEADER")
        emit("s_endpgm")  # an exit block laid out before the header
    header = addr
    for t in body:
        emit(t)
    emit("s_cbranch_scc1 BACK")
    emit("s_endpgm")
    for a, t in out:
        t = t.replace("BACK", f"0 <k+0x{header - 0x1000:x}>")
        t = t.replace("BRANCH_TO_HEADER", f"0 <k+0x{header - 0x1000:x}>")
        lines.append(f"\t{t} // {a:012X}: 00000000")
    return "\n".join(lines) + "\n"


_GOOD = [
    "s_waitcnt vmcnt(1) lgkmcnt(0)",
    "s_barrier",
    "global_load_dwordx4 v[0:3], v[4:5], off",
    "v_mfma_f32_16x16x16_f16 v[8:11], v[0:1], v[2:3], v[8:11]",
    "s_waitcnt vmcnt(0) lgkmcnt(0)",
    "s_barrier",
    "global_atomic_add_f32 v[4:5], v6, off",
    "s_waitcnt vmcnt(1) lgkmcnt(0)",
    "s_barrier",
]


@pytest.mark.parametrize("rotate", (False, True))
def test_isa_model_on_a_synthetic_loop(rotate):
    """The ISA model on a hand-written counted loop (early issue, one atomic):
    clean as written; a barrier wait one too large leaves the load in flight;
    a step-end wait of 0 drains the atomic; a wait right after the load waits
    for it before the barrier; a missing barrier breaks the per-step count."""
    res = counted_isa_problems(_isa_text(_GOOD, rotate=rotate))
    assert not res["problems"], res
    assert [r["waits"] for r in res["loops"]] == [[1, 0, 1]]
    plus1 = list(_GOOD)
    plus1[4] = "s_waitcnt vmcnt(1) lgkmcnt(0)"
    assert any(
        "in flight" in p for p in counted_isa_problems(_isa_text(plus1))["problems"]
    )
    zero = list(_GOOD)
    zero[7] = "s_waitcnt vmcnt(0) lgkmcnt(0)"
    assert any("drains" in p for p in counted_isa_problems(_isa_text(zero))["problems"])
    early = _GOOD[:3] + ["s_waitcnt vmcnt(0)"] + _GOOD[3:]
    res = counted_isa_problems(_isa_text(early))
    assert any("waits for a load of this trip" in p for p in res["problems"]), res
    nobar = [x for i, x in enumerate(_GOOD) if i != 5]
    res = counted_isa_problems(_isa_text(nobar))
    assert any("barriers per trip" in p for p in res["problems"]), res
    # a single-wave workgroup whose barriers the backend dropped
    dropped = [x for x in _GOOD if x != "s_barrier"]
    assert not counted_isa_problems(_isa_text(dropped), single_wave=True)["problems"]
    res = counted_isa_problems(_isa_text(dropped))
    assert any("barriers per trip" in p for p in res["problems"]), res


ISA_VARIANTS = (
    "band", "band_edge", "band_pack", "band_atomic_pack", "thd_band_atomic",
    "band_hout", "band_ptlds", "band_dqfrag", "band_mirror",
)  # fmt: skip
ISA_ROWS = [
    r for r in ROWS if r[4] in ISA_VARIANTS and (r[3] == "fp16" or r[4] == "band")
]
ISA_ARCH_ENV = "ROCKE_BWD_ISA_ARCH"  # restrict the compile rows to one target


# Counted waits with mirror pairing fail the resource gate on the production
# compiler on most tiles (SGPR spills; with edge tiles on every tile, and
# scratch and VGPR spills on the one-wave d128 tile), and so do some of their
# full-barrier twins: those specs are refused by the validator
# (RESOURCE_REFUSED, pinned by test_attention_bwd_refusals.py), so ROWS holds
# none of them.
def _isa_rows():
    only = os.environ.get(ISA_ARCH_ENV)
    return [r for r in ISA_ROWS if not only or r[0] == only]


def test_mirror_edge_rows_are_refused():
    """Every counted mirror + edge row of the suite is refused (none is left
    to a test exception)."""
    from kernels.common._attention_bwd_refusals import resource_refusal

    for arch in MFMA_ARCHS:
        for cfg in distinct_configs(arch):
            for d in HEADS:
                for dt in DTYPES:
                    for issue in COUNTED_ISSUES:
                        row = (arch, cfg, d, dt, "band_mirror_edge", issue)
                        spec = spec_of(*row)
                        reason = resource_refusal(spec.resolved(arch), arch)
                        assert reason is not None and not _legal(*row), row


def _compile(kernel, arch):
    pytest.importorskip("rocke.runtime.comgr")
    from ._attention_bwd_resources import (
        compile_or_skip,
        find_objdump,
        pin_production_comgr,
    )

    pin_production_comgr()  # torch's bundled comgr, by path; torch not imported
    if find_objdump() is None:
        pytest.skip("llvm-objdump not found")
    return compile_or_skip(kernel, arch=arch, capture_ir_text=False, backend="python")


def _isa_model(kernel, arch):
    """:func:`counted_isa_problems` of the compiled kernel (a single-wave
    workgroup may have its barriers dropped by the backend)."""
    from ._attention_bwd_resources import disassemble, read_code_object_resources

    hsaco = _compile(kernel, arch).hsaco
    res = read_code_object_resources(hsaco, arch=arch)
    single = res.max_flat_workgroup_size <= res.wavefront_size
    return counted_isa_problems(disassemble(hsaco, arch=arch), single_wave=single)


@pytest.mark.parametrize("row", _isa_rows(), ids=_ids)
def test_counted_isa_model(row):
    arch, issue = row[0], row[-1]
    res = _isa_model(kernel_of(*row), arch)
    assert not res["problems"], res
    if issue == "late":
        assert not any(rec["vmcnt0"] for rec in res["loops"]), res["loops"]


@pytest.mark.parametrize("mode", ("plus1", "zero"))
@pytest.mark.parametrize("issue", COUNTED_ISSUES)
@pytest.mark.parametrize("arch", MFMA_ARCHS)
def test_isa_model_catches_wrong_counts(monkeypatch, arch, issue, mode):
    if os.environ.get(ISA_ARCH_ENV) not in (None, arch):
        pytest.skip(f"{ISA_ARCH_ENV} selects another target")
    _mutated(monkeypatch, mode)
    spec = spec_of(arch, "default", 64, "fp16", "band", issue)
    kernel = build_attn_bwd_main(spec, arch=arch)
    res = _isa_model(kernel, arch)
    want = "in flight" if mode == "plus1" else "drains"
    assert any(want in p for p in res["problems"]), res


@pytest.mark.parametrize("row", _isa_rows(), ids=_ids)
def test_counted_resource_gate_matches_the_full_twin(row):
    """The resource gate (production comgr) passes for a counted kernel and
    for its full twin: no failure and no crossed SGPR watch level."""
    from ._attention_bwd_resources import disassemble, evaluate_resource_gate

    arch = row[0]
    spec = spec_of(*row)
    full = dataclasses.replace(spec, loop_sync="full", counted_issue="early")
    facts = attn_bwd_arch_facts(arch)
    for s in (spec, full):
        r = validate_attn_bwd_spec(s, arch)
        art = _compile(build_attn_bwd_main(s, arch=arch), arch)
        isa = disassemble(art.hsaco, arch=arch)
        res = evaluate_resource_gate(
            art.hsaco, arch=arch, lds_capacity_bytes=facts.lds_capacity_bytes, spec=r,
            isa_text=isa,
        )  # fmt: skip
        if not res.valid:
            pytest.skip(
                f"not the production comgr / flavor ({res.flavor}, {res.comgr})"
            )
        assert not res.failures and not res.warnings, (s.loop_sync, res.record())


def test_model_module_constants():
    assert counted_mod.COUNTED_VMCNT_MAX == 63
