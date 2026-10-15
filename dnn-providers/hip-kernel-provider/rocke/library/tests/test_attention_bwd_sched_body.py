# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Schedule values (``sched``, ``setprio``) in the backward main kernel.

CPU / IR part (no comgr):

* the default (``sched = "none"``, ``setprio = False``) carries no schedule
  op, and the spelled-out default is the default kernel;
* ``iglp0`` / ``iglp1``: one ``iglp_opt`` per q step, the first op after the
  step's staging barrier;
* every value emits, per q step, exactly the hint sequence of its stage
  table (priority raises, ``sched_group_barrier`` groups, priority resets,
  ``sched_barrier(0)`` fences after A, B and C);
* the synchronous count model equals the body: per stage, the IR census of
  matrix ops, VMEM reads / writes and DS reads / writes (and the reads that
  lead the stage's first matrix op) equals ``StageTable.stages()``, over the
  shipped instance variants and the knob values that change the counts;
* fenced placement: the lse2 / Dsum loads sit in stage A, the read-ahead
  operands one stage before their GEMM;
* ``stage_table_atomics``: each step's atomics sit in stage A (the previous
  step's partial), none in D, the last partial is flushed once after the q
  loops, no branch encloses an atomic, and the first step starts from zeros;
* scheduling only: the matrix ops (count, atom, order), the step's atomic
  count and the resolved spec (apart from the two knobs) are the default's,
  the IR differs from it, and the read-ahead keeps every cross-wave LDS
  access behind a barrier.

Compile part (comgr; skips without it): LLVM call counts of the hints equal
the model per step body, the code object's matrix-instruction count equals
the default's and its text differs, and the resource gate (zero scratch, no
spills, register budgets, the hot-loop AGPR copy ceiling) passes for every
value on the shipped CDNA / RDNA instances, except that the RDNA d128 tiles
at the register-file edge may spill (reported; the default passes there).
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from functools import cache

import pytest
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower
from rocke.helpers.schedule import DS_READ, DS_WRITE, MFMA, VMEM_READ, VMEM_WRITE

from kernels.common._attention_bwd_body import BwdTileGeometry
from kernels.common._attention_bwd_refusals import resource_refusal
from kernels.common._attention_bwd_sched import (
    FENCED_SCHEDS,
    IGLP_SCHEDS,
    STAGE_TABLE_SCHEDS,
    STAGES,
)
from kernels.common.attention_bwd import (
    AttnBwdSpec,
    _not_built,
    attn_bwd_arch_facts,
    build_attn_bwd_main,
    lds_plan,
    register_estimates,
    validate_attn_bwd_spec,
)

from ._attention_bwd_test_utils import distinct_configs
from .test_attention_bwd_ir import (
    ARCHS,
    HEADS,
    VARIANTS,
    _walk,
    group_loops,
    lds_hazards,
    q_loops,
    step_bodies,
    traced_spec,
)
from .sdpa.bwd_kernel_cases import config_knobs

SCHED_PAIRS = (
    ("iglp0", False),
    ("iglp1", False),
    ("fences_only", False),
    ("stage_table", False),
    ("stage_table", True),
    ("stage_table_atomics", False),
    ("stage_table_atomics", True),
)
PAIR_IDS = [f"{s}{'-prio' if p else ''}" for s, p in SCHED_PAIRS]
_HINTS = (
    "tile.iglp_opt",
    "tile.sched_barrier",
    "tile.sched_group_barrier",
    "tile.s_setprio",
)
_BARRIERS = ("tile.sync", "tile.sync_lds_only")
_CLASS_OF = (
    ("tile.mma", MFMA),
    ("memref.global_load", VMEM_READ),
    ("memref.global_atomic", VMEM_WRITE),
    ("memref.global_store", VMEM_WRITE),
    ("tile.smem_load", DS_READ),
    ("tile.ds_read_tr", DS_READ),
    ("tile.smem_store", DS_WRITE),
)

# Instance variants of the census (the shipped ones that change the step).
BODY_VARIANTS = (
    "none",
    "band",
    "band_pack",
    "band_edge",
    "band_atomic_pack",
    "sv1_none",
    "thd_band_atomic",
    "band_ldssync",
    "band_edge_hout",
    "band_edge_atomic_pack_hout_ldssync",
)
# Knob values that change what a stage reads or writes (on the band variant).
KNOBS = {
    "": {},
    "ptlds": dict(pt_route="lds"),
    "grid2": dict(pt_route="lds", warp_grid_g02="2", warp_grid_g13="2"),
    "acclds": dict(acc_in_lds=True, agpr_alloc=(0, 0)),
    "kvlds": dict(kv_residency="lds", kt_source="lds"),
    "ktlds": dict(kt_source="lds"),
    "frag": dict(dq_issue="fragment"),
    "fregs": dict(dq_issue="fragment_regs"),
    "top": dict(stats_load="step_top"),
    "lf": dict(qdo_stage="loads_first"),
    "pad8": dict(lds_swizzle="q=pad8,do=pad8,ds=pad8,pt=pad8,k_res=pad8,v_res=pad8"),
    "pad16": dict(lds_swizzle="q=pad16,do=pad16,ds=pad16,k_res=pad16"),
    "fence": dict(agpr_alloc=(128, 128), agpr_route="agpr_fence"),
    "vform": dict(agpr_alloc=(128, 128), agpr_route="vgpr_form"),
    "token": dict(ws_layout="token_major", dkv_mode="atomic"),
    "rot": dict(q_rotate=True),
    "scale": dict(static_scale="dq_before_atomic"),
}


def _ids(row):
    return "-".join(str(x) for x in row)


def _spec(arch, cfg, d, var, knob, sched, prio, dt="fp16"):
    kw = dict(config_knobs(cfg, arch), **VARIANTS[var])
    kw.pop("sched", None)
    kw.pop("setprio", None)
    extra = dict(KNOBS[knob])
    if extra.get("warp_grid_g02") == "2":
        w = AttnBwdSpec(head_size=d, dtype=dt, **kw).resolved(arch).waves
        if w < 2:
            return None
        extra["warp_grid_g02"] = extra["warp_grid_g13"] = (2, w // 2)
    kw.update(extra)
    try:
        spec = AttnBwdSpec(head_size=d, dtype=dt, sched=sched, setprio=prio, **kw)
        r = validate_attn_bwd_spec(spec, arch)
    except ValueError:
        return None
    if _not_built(r):
        return None
    return spec


def _rows():
    out = []
    for arch in ARCHS:
        for cfg in distinct_configs(arch):
            for d in HEADS:
                for var in BODY_VARIANTS:
                    out.append((arch, cfg, d, var, ""))
        for d in HEADS:
            for knob in KNOBS:
                if knob:
                    out.append((arch, "default", d, "band", knob))
    return out


ROWS = _rows()


@cache
def _kernel(arch, cfg, d, var, knob, sched, prio):
    spec = _spec(arch, cfg, d, var, knob, sched, prio)
    if spec is None:
        return None, None
    return spec, build_attn_bwd_main(spec, arch=arch)


def _cls(name):
    for prefix, c in _CLASS_OF:
        if name.startswith(prefix):
            return c
    return None


def _split(ops):
    """``(staging, {stage: ops})`` of a q-step body: the staging up to the
    first top-level barrier, then A / B / C / D at the ``sched_barrier``
    fences (D runs to the end of the step)."""
    i = next(i for i, op in enumerate(ops) if op.name in _BARRIERS)
    regions, cur = [], []
    for op in ops[i + 1 :]:
        if op.name == "tile.sched_barrier":
            regions.append(cur)
            cur = []
        else:
            cur.append(op)
    regions.append(cur)
    assert len(regions) == 4, len(regions)
    return ops[:i], dict(zip(STAGES, regions))


def _census(ops):
    """``(Counter by class, DS reads before the first matrix op)``."""
    cnt, lead, seen = Counter(), 0, False
    for op, _anc in _walk(ops):
        c = _cls(op.name)
        if c is None:
            continue
        cnt[c] += 1
        if c == MFMA:
            seen = True
        elif c == DS_READ and not seen:
            lead += 1
    return cnt, lead


def _model(spec, arch):
    r = validate_attn_bwd_spec(spec, arch)
    return BwdTileGeometry.from_spec(r, arch).stage_table()


def _hint_seq(ops):
    out = []
    for op in ops:
        if op.name == "tile.iglp_opt":
            out.append(("iglp", op.attrs["level"]))
        elif op.name == "tile.s_setprio":
            out.append(("prio", op.attrs["level"]))
        elif op.name == "tile.sched_barrier":
            out.append(("fence", op.attrs["mask"]))
        elif op.name == "tile.sched_group_barrier":
            out.append(("group", op.attrs["mask"], op.attrs["count"]))
            assert op.attrs["group"] == 0
    return out


def _want_seq(t):
    out = []
    if t.sched in IGLP_SCHEDS:
        out.append(("iglp", int(t.sched[-1])))
    for st in t.stages():
        if t.setprio:
            out.append(("prio", 1))
        if t.sched in STAGE_TABLE_SCHEDS:
            out += [("group", m, c) for m, c in st.groups()]
        if t.setprio:
            out.append(("prio", 0))
        if t.sched in FENCED_SCHEDS and st.stage != "D":
            out.append(("fence", 0))
    return out


# ---------------------------------------------------------------------------
# default
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ARCHS)
def test_default_has_no_schedule_ops_and_the_explicit_default_is_it(arch):
    for cfg in distinct_configs(arch):
        for d in HEADS:
            for var in ("band", "band_edge_atomic_pack_hout_ldssync", "thd_sv1_none"):
                kw = dict(config_knobs(cfg, arch), **VARIANTS[var])
                spec = AttnBwdSpec(head_size=d, **kw)
                if resource_refusal(spec.resolved(arch), arch):
                    continue  # a recorded resource-gate failure: no kernel
                k = build_attn_bwd_main(spec, arch=arch)
                names = Counter(op.name for op, _ in _walk(k.body.ops))
                assert not any(names[h] for h in _HINTS), (cfg, d, var)
                explicit = dataclasses.replace(spec, sched="none", setprio=False)
                assert (
                    validate_attn_bwd_spec(explicit, arch).kernel_name()
                    == validate_attn_bwd_spec(spec, arch).kernel_name()
                )
                assert lower(k, arch=arch, llvm_flavor="llvm22") == lower(
                    build_attn_bwd_main(explicit, arch=arch),
                    arch=arch,
                    llvm_flavor="llvm22",
                )


def test_no_schedule_value_is_refused():
    for sched, prio in SCHED_PAIRS:
        spec = validate_attn_bwd_spec(
            AttnBwdSpec(head_size=64, sched=sched, setprio=prio), "gfx942"
        )
        assert not _not_built(spec)


# ---------------------------------------------------------------------------
# emission
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pair", SCHED_PAIRS, ids=PAIR_IDS)
@pytest.mark.parametrize("row", ROWS, ids=_ids)
def test_hint_sequence_and_stage_counts_equal_the_model(row, pair):
    """Per q step: the hint ops are the stage table's sequence, and (fenced
    values) each stage's IR census and leading reads equal the count model."""
    arch = row[0]
    spec, kernel = _kernel(*row, *pair)
    if spec is None:
        pytest.skip("not legal for this configuration")
    t = _model(spec, arch)
    assert t.pipeline == "sync" and t.sched == pair[0] and t.setprio == pair[1]
    want = _want_seq(t)
    for ops in step_bodies(kernel):
        assert _hint_seq(ops) == want
        if pair[0] in IGLP_SCHEDS:
            i = next(i for i, op in enumerate(ops) if op.name in _BARRIERS)
            assert ops[i + 1].name == "tile.iglp_opt"
            assert not any(op.name == "tile.iglp_opt" for op in ops[:i])
            continue
        if pair[0] not in FENCED_SCHEDS:
            continue
        _staging, regions = _split(ops)
        for sc in t.stages():
            got, lead = _census(regions[sc.stage])
            assert got[MFMA] == sc.mfma, (sc.stage, "mfma")
            assert got[VMEM_READ] == sc.vmem_read, (sc.stage, "vmem_read")
            assert got[VMEM_WRITE] == sc.vmem_write, (sc.stage, "vmem_write")
            assert got[DS_READ] == sc.ds_read, (sc.stage, "ds_read")
            assert got[DS_WRITE] == sc.ds_write, (sc.stage, "ds_write")
            assert lead == dict(sc.lead).get(DS_READ, 0), (sc.stage, "lead")
            if sc.mfma:
                # every VMEM read of a stage leads it
                assert dict(sc.lead).get(VMEM_READ, 0) == sc.vmem_read


@pytest.mark.parametrize("arch", ARCHS)
def test_fenced_placement(arch):
    """Fenced values: lse2 / Dsum loads in A (none in B), the dO^T reads of
    G1 in A (so A reads more than G0's operands), the Q^T reads of G3 in B
    (C then reads nothing for G3 under the relabel route) and an LDS K^T in
    C (D reads only dS); ``none`` keeps the stats with the softmax."""
    cfg = distinct_configs(arch)[0]
    for d in HEADS:
        spec, k = _kernel(arch, "default", d, "band", "ktlds", "fences_only", False)
        if spec is None:
            spec, k = _kernel(arch, cfg, d, "band", "", "fences_only", False)
        t = _model(spec, arch)
        r = validate_attn_bwd_spec(spec, arch)
        for ops in step_bodies(k):
            _st, reg = _split(ops)
            ca, _ = _census(reg["A"])
            cb, _ = _census(reg["B"])
            cc, _ = _census(reg["C"])
            cd, _ = _census(reg["D"])
            assert ca[VMEM_READ] == t.stats_loads() and cb[VMEM_READ] == 0
            g0_reads = t._frags("g0")[2] * dict(t.stage("A").lead)[DS_READ]
            assert ca[DS_READ] > g0_reads  # G0's operands plus dO^T
            if r.pt_route == "relabel" and not r.acc_in_lds:
                assert cc[DS_READ] == t.stage("C").ds_read
            if r.kt_source == "lds":
                assert cc[DS_READ] > 0
                # D reads dS only; the K^T reads moved into C
                assert cd[DS_READ] == t._g4_reads(kt_hoisted=True)[0]
                assert cd[DS_READ] < t._g4_reads(kt_hoisted=False)[0]


def _atomics(ops):
    return [
        (op, anc) for op, anc in _walk(ops) if op.name == "memref.global_atomic_add_f32"
    ]


@pytest.mark.parametrize("prio", [False, True])
@pytest.mark.parametrize("row", ROWS, ids=_ids)
def test_deferred_dq_atomics(row, prio):
    """``stage_table_atomics``: per step the dQ atomics sit in stage A and
    D has none; the partial is loop-carried (named ``...dq<i>``) and its
    first value is +0.0; one flush batch follows the q loops; no branch
    encloses any dQ atomic."""
    arch = row[0]
    spec, kernel = _kernel(*row, "stage_table_atomics", prio)
    if spec is None:
        pytest.skip("not legal for this configuration")
    t = _model(spec, arch)
    atoms = t.dq_atomics_per_lane
    r = validate_attn_bwd_spec(spec, arch)
    loops = q_loops(kernel)
    for ops in step_bodies(kernel):
        _st, reg = _split(ops)
        assert len(_atomics(reg["A"])) == atoms
        assert not _atomics(reg["B"]) and not _atomics(reg["C"])
        assert not _atomics(reg["D"])
    for lp in loops:
        names = [m["name"] for m in lp.attrs["iter_args"]]
        dq = [n for n in names if "dq" in n and n[-1].isdigit() and "dqh" not in n]
        assert len(dq) == atoms
    first = (group_loops(kernel) or loops)[0]  # the outermost carrier
    metas, inits = first.attrs["iter_args"], first.operands[3:]
    assert any(m["name"].endswith("dq0") for m in metas)
    for m, v in zip(metas, inits):
        if m["name"].endswith("dq0"):
            assert v.op.name == "arith.constant" and v.op.attrs["value"] == 0.0
    in_loop = {id(op) for lp in loops for op, _ in _walk(lp.regions[0].ops)}
    ws_dq = 0
    for op, anc in _walk(kernel.body.ops):
        if op.name != "memref.global_atomic_add_f32":
            continue
        if id(op) in in_loop:
            assert not any(a.name in ("scf.if", "scf.if_else") for a in anc)
            continue
        if r.dkv_mode == "atomic" and any(a.name == "scf.if" for a in anc):
            continue  # the dK / dV epilogue
        assert not any(a.name in ("scf.if", "scf.if_else") for a in anc)
        ws_dq += 1
    assert ws_dq == atoms  # the flush after the loops


@pytest.mark.parametrize("pair", SCHED_PAIRS, ids=PAIR_IDS)
@pytest.mark.parametrize("arch", ARCHS)
def test_scheduling_only(arch, pair):
    """Every value: the resolved spec equals the default's apart from the two
    knobs (agpr_alloc and the register estimate included), the kernel name is
    salted, the IR differs, and the matrix ops (count, atom, order) and the
    per-step dQ atomics of each q step are the default's."""
    for cfg in distinct_configs(arch):
        for d in HEADS:
            base = _spec(arch, cfg, d, "band_edge", "", "none", False)
            spec = _spec(arch, cfg, d, "band_edge", "", *pair)
            if spec is None:
                continue
            r0, r = (validate_attn_bwd_spec(s, arch) for s in (base, spec))
            assert dataclasses.replace(r, sched="none", setprio=False) == r0
            assert register_estimates(r, arch) == register_estimates(r0, arch)
            assert r.kernel_name() != r0.kernel_name()
            assert lds_plan(r, arch) == lds_plan(r0, arch)
            k0 = build_attn_bwd_main(base, arch=arch)
            k = build_attn_bwd_main(spec, arch=arch)
            ir0 = lower(k0, arch=arch, llvm_flavor="llvm22")
            ir = lower(k, arch=arch, llvm_flavor="llvm22")
            assert ir.replace(r.kernel_name(), "K") != ir0.replace(
                r0.kernel_name(), "K"
            )
            for b0, b1 in zip(step_bodies(k0), step_bodies(k), strict=True):
                m0 = [op.attrs["op_id"] for op, _ in _walk(b0) if op.name == "tile.mma"]
                m1 = [op.attrs["op_id"] for op, _ in _walk(b1) if op.name == "tile.mma"]
                assert m0 == m1
                assert len(_atomics(b0)) == len(_atomics(b1))


@pytest.mark.parametrize("pair", SCHED_PAIRS, ids=PAIR_IDS)
@pytest.mark.parametrize("arch", ARCHS)
def test_read_ahead_keeps_lds_accesses_behind_barriers(arch, pair):
    for cfg in distinct_configs(arch):
        for d in HEADS:
            for var, knob in (("band_edge", ""), ("band", "ptlds"), ("band", "kvlds")):
                spec = _spec(arch, cfg, d, var, knob, *pair)
                if spec is None:
                    continue
                kernel, tags = traced_spec(spec, arch)
                assert not lds_hazards(kernel, tags)[:4], (cfg, d, var, knob)


# ---------------------------------------------------------------------------
# compile: LLVM calls, code objects, resource gate
# ---------------------------------------------------------------------------

_LLVM = {
    "tile.sched_group_barrier": "call void @llvm.amdgcn.sched.group.barrier(",
    "tile.sched_barrier": "call void @llvm.amdgcn.sched.barrier(",
    "tile.s_setprio": "call void @llvm.amdgcn.s.setprio(",
    "tile.iglp_opt": "call void @llvm.amdgcn.iglp.opt(",
}


@pytest.mark.parametrize("pair", SCHED_PAIRS, ids=PAIR_IDS)
@pytest.mark.parametrize("arch", ARCHS)
def test_llvm_hint_calls_equal_the_model(arch, pair):
    for cfg in distinct_configs(arch):
        for d in HEADS:
            for var in ("band", "band_edge"):
                spec, k = _kernel(arch, cfg, d, var, "", *pair)
                if spec is None:
                    continue
                t = _model(spec, arch)
                model = t.ir_count_model()
                n = len(step_bodies(k))
                ir = lower(k, arch=arch, llvm_flavor="llvm22")
                for key, call in _LLVM.items():
                    assert ir.count(call) == n * model[key], (key, cfg, d, var)


def _compile(kernel, arch):
    pytest.importorskip("rocke.runtime.comgr")
    from ._attention_bwd_resources import compile_or_skip, pin_production_comgr

    assert kernel is not None, "an illegal instance is skipped before it compiles"
    pin_production_comgr()  # torch's bundled comgr, by path; torch not imported
    return compile_or_skip(kernel, arch=arch, capture_ir_text=False, backend="python")


@pytest.mark.parametrize("d", HEADS)
@pytest.mark.parametrize("pair", SCHED_PAIRS, ids=PAIR_IDS)
@pytest.mark.parametrize("arch", ARCHS)
def test_code_objects_differ_and_keep_the_matrix_instructions(arch, pair, d):
    """The arch's first configuration (plain band instance): the code
    object of every value has the default's matrix-instruction count, and
    its text differs from the default's (iglp may be applied without effect;
    the sweep records that, so it is not failed here)."""
    from .test_attention_bwd_sched import _disassemble, _MMA_RE

    cfg = distinct_configs(arch)[0]
    base, k0 = _kernel(arch, cfg, d, "band", "", "none", False)
    spec, k = _kernel(arch, cfg, d, "band", "", *pair)
    assert base is not None, (arch, cfg, d)  # the plain band instance is legal
    if spec is None:
        # a refused value (a coupling or RESOURCE_REFUSED) has no code object
        pytest.skip(f"{pair} is not legal on {arch} {cfg} d{d}")
    a0, a1 = _compile(k0, arch), _compile(k, arch)
    t0 = _disassemble(a0.hsaco, arch)
    t1 = _disassemble(a1.hsaco, arch)
    assert len(_MMA_RE.findall(t0)) == len(_MMA_RE.findall(t1))
    body = [ln.split("//")[0].strip() for ln in t1.splitlines() if "\t" in ln]
    body0 = [ln.split("//")[0].strip() for ln in t0.splitlines() if "\t" in ln]
    if pair[0] not in IGLP_SCHEDS:
        assert body != body0, (cfg, d)


# Resource-gate instances per value: every config of the arch, every head
# size, the plain, packed, edge-tile, atomic THD and narrow variants. A value
# whose kernel fails the gate on the production compiler at a point (the
# one- and two-wave WMMA d128 tiles sit at the edge of the 256-VGPR file) is
# refused there by the validator (RESOURCE_REFUSED,
# test_attention_bwd_refusals.py), so the point is not legal for it.
GATE_VARIANTS = ("band", "band_pack", "band_edge", "thd_band_atomic", "sv1_band")


GATE_POINTS = [
    (arch, cfg, d) for arch in ARCHS for cfg in distinct_configs(arch) for d in HEADS
]


@pytest.mark.parametrize("var", GATE_VARIANTS)
@pytest.mark.parametrize("pair", SCHED_PAIRS, ids=PAIR_IDS)
@pytest.mark.parametrize("point", GATE_POINTS, ids=_ids)
def test_resource_gate_per_value(point, pair, var):
    """Zero scratch, no spills, register budgets, LDS within the plan and the
    hot-loop AGPR copy ceiling, for every legal value on every configuration,
    head size and gate variant of every arch."""
    arch, cfg, d = point
    pytest.importorskip("rocke.runtime.comgr")
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

    spec, k = _kernel(arch, cfg, d, var, "", *pair)
    if spec is None:
        pytest.skip(f"{var} with {pair} is not legal on {point}")
    if find_objdump() is None:
        pytest.skip("llvm-objdump not found (needed for the q-loop AGPR copy rule)")
    flavor, comgr = current_flavor(), comgr_version()
    if flavor != GATE_FLAVOR or not comgr_is_production(comgr):
        pytest.skip(f"resource gate needs {GATE_FLAVOR} + production comgr")
    facts = attn_bwd_arch_facts(arch)

    def gate(spec, k):
        r = validate_attn_bwd_spec(spec, arch)
        res = evaluate_resource_gate(
            _compile(k, arch).hsaco, arch=arch,
            lds_capacity_bytes=facts.lds_capacity_bytes, spec=r,
        )  # fmt: skip
        if not res.valid:
            pytest.skip(f"gate not valid here ({res.flavor}, {res.comgr})")
        if res.resources.lds_bytes > lds_plan(r, arch).total_bytes:
            return res, ("LDS above the plan",)
        return res, tuple(res.failures) + tuple(res.warnings)

    res, bad = gate(spec, k)
    assert not bad, (bad, res.resources.arch_vgpr, res.resources.sgpr)
