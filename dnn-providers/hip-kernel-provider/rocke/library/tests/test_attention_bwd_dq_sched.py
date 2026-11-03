# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Schedule values of the split dQ kernel's key step (``dq_sched``).

The key step of the Q-parallel dQ kernel runs as stages A (G0' ``S^T = K
Q^T``), B (G2' ``dP^T = V dO^T`` and the softmax gradient) and C (G4' ``dQ +=
dS K``); :class:`DqStageTable` is its count model and its hooks. Values:
``none`` (no hint), ``iglp0`` / ``iglp1`` (one ``iglp_opt`` ahead of A),
``fences_only`` (a ``sched_barrier(0)`` after A and B), ``stage_table`` (the
fences and per-stage ``sched_group_barrier`` groups) and
``stage_table_setprio`` (with the wave priority raised around each stage).

CPU / IR part (no comgr):

* the value set, the spec field, the kernel-name salt, the plan / registry
  knob ``dq_sched`` (split dQ only);
* the default emits no hint and is the spelled-out default's IR;
* per key step, over the dQ kernel's instance variants and the knobs that
  change the counts (K / V row pads, the prefetch, the transpose source, the
  atoms, the tiles, ``stage_vec``): the hint ops are the model's sequence and,
  under a fenced value, every stage region's IR census (matrix ops, VMEM
  reads, DS reads / writes) and leading reads equal the model;
* scheduling only: every non-hint op of the kernel is the twin's in the same
  order (no op moves), the resolved spec apart from ``sched``, the LDS plan
  and the register estimate are the twin's, the IR differs;
* LDS: the barrier replay of every LDS access finds no cross-wave race, and
  finds them once the key step's barriers are removed (not vacuous).

Compile part (comgr; skips without it): the LLVM hint calls equal the model,
the code object keeps the twin's matrix-instruction count, and the dQ
kernel's resource gate per value.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from functools import cache

import pytest
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower
from rocke.helpers.schedule import DS_READ, DS_WRITE, MFMA, VMEM_READ, VMEM_WRITE

from kernels.common._attention_bwd_dqbody import BwdDqGeometry
from kernels.common._attention_bwd_sched import (
    DQ_FENCED_SCHEDS,
    DQ_SCHEDS,
    DQ_STAGES,
    IGLP_SCHEDS,
    DqStageTable,
)
from kernels.common.attention_bwd import AttnBwdSpec, attn_bwd_arch_facts
from kernels.common.attention_bwd_dq import (
    DQ_TUNING_FIELDS,
    AttnBwdDqSpec,
    build_attn_bwd_dq,
    dq_lds_plan,
    dq_register_estimates,
    validate_attn_bwd_dq_spec,
)

from .test_attention_bwd_dq import ARCHS, HEADS, _key_loops
from .test_attention_bwd_dq import VARIANTS as DQ_VARIANTS
from .test_attention_bwd_dq import _walk
from .test_attention_bwd_sched_body import _census, _hint_seq

VALUES = DQ_SCHEDS[1:]
_HINTS = (
    "tile.sched_barrier",
    "tile.sched_group_barrier",
    "tile.s_setprio",
    "tile.iglp_opt",
)
# Instance variants: the dQ suite's, plus the knobs that change a stage's
# counts (row pads that split vector accesses, the transpose source, the
# atoms, other tiles).
VARIANTS = {
    **DQ_VARIANTS,
    "kvpad8": {"kv_pad": 8},
    "kvpad32": {"kv_pad": 32},
    "prefetch_pad8": {"kv_prefetch": True, "kv_pad": 8},
    "n32": {"block_n": 32},
    "n128": {"block_n": 128},
    "w8_m128": {"waves": 8, "block_m": 128},
    "plain": {"transpose_source": "lds_plain"},
    "a02_16": {"atom_g02": "16x16x16"},
    "a4_32": {"atom_g4": "16x16x32", "transpose_source": "lds_plain"},
    "prefetch_plain": {"kv_prefetch": True, "transpose_source": "lds_plain"},
}
CLASSES = (("fp16", "batched", "band", 8), ("bf16", "thd", "none", 1))


def _spec(arch, d, dt, sm, mc, sv, var, sched="none"):
    return AttnBwdDqSpec(
        head_size=d, dtype=dt, seq_mode=sm, mask_class=mc, stage_vec=sv,
        sched=sched, **VARIANTS[var],
    )  # fmt: skip


def _legal(row, sched="none") -> bool:
    try:
        validate_attn_bwd_dq_spec(_spec(*row, sched), row[0])
    except ValueError:
        return False
    return True


# Rows whose default dQ kernel is legal (the twins of the scheduling-only
# comparisons), and the rows where at least one value is legal (the per-value
# checks; a superset: a row whose default kernel is refused for its resources,
# DQ_RESOURCE_REFUSED, still carries its legal values).
ROWS = [
    (arch, d, *cls, var)
    for arch in ARCHS
    for d in HEADS
    for cls in CLASSES
    for var in VARIANTS
    if _legal((arch, d, *cls, var))
]
VALUE_ROWS = [
    (arch, d, *cls, var)
    for arch in ARCHS
    for d in HEADS
    for cls in CLASSES
    for var in VARIANTS
    if any(_legal((arch, d, *cls, var), v) for v in DQ_SCHEDS)
]


@cache
def _kernel(row, sched="none"):
    return build_attn_bwd_dq(_spec(*row, sched), arch=row[0])


def _ids(row):
    return "-".join(map(str, row))


def _model(row, sched) -> DqStageTable:
    r = validate_attn_bwd_dq_spec(_spec(*row, sched), row[0])
    return BwdDqGeometry.from_spec(r, row[0]).stage_table()


def _regions(ops, t):
    """``{stage: ops}`` of a key-step body under a fenced value: the plain
    step's staging (up to its first barrier) dropped, then the parts between
    the ``sched_barrier(0)`` fences."""
    if not t.kv_prefetch:
        i = next(i for i, op in enumerate(ops) if op.name == "tile.sync")
        ops = ops[i + 1 :]
    parts, cur = [], []
    for op in ops:
        if op.name == "tile.sched_barrier":
            assert op.attrs["mask"] == 0
            parts.append(cur)
            cur = []
        else:
            cur.append(op)
    parts.append(cur)
    assert len(parts) == len(DQ_STAGES), len(parts)
    return dict(zip(DQ_STAGES, parts))


# ---------------------------------------------------------------------------
# values, spec, plan, registry
# ---------------------------------------------------------------------------


def test_value_set_spec_field_and_name_salt():
    assert DQ_SCHEDS[0] == "none"
    assert DQ_FENCED_SCHEDS == ("fences_only", "stage_table", "stage_table_setprio")
    assert "sched" in DQ_TUNING_FIELDS
    assert AttnBwdDqSpec(head_size=64).sched == "none"
    with pytest.raises(ValueError, match="sched"):
        AttnBwdDqSpec(head_size=64, sched="stage_table_atomics")
    base = validate_attn_bwd_dq_spec(AttnBwdDqSpec(head_size=64), "gfx950")
    names = {base.kernel_name()}
    for v in VALUES:
        r = validate_attn_bwd_dq_spec(AttnBwdDqSpec(head_size=64, sched=v), "gfx950")
        assert r.sched == v
        names.add(r.kernel_name())
    assert len(names) == len(DQ_SCHEDS)
    spelled = validate_attn_bwd_dq_spec(
        AttnBwdDqSpec(head_size=64, sched="none"), "gfx950"
    )
    assert spelled.kernel_name() == base.kernel_name()


@pytest.mark.parametrize("arch", ARCHS)
def test_iglp1_marks_the_dq_kernel_for_a_fresh_compile_process(arch):
    """``dq_sched = "iglp1"`` emits ``iglp_opt(1)``, whose LLVM strategy keeps
    state across the compiles of a process: the dQ kernel carries the
    fresh-process mark, the other values do not, and the mark leaves the LLVM
    IR unchanged."""
    from rocke.core.lower_llvm import lower_kernel_to_llvm
    from rocke.helpers.compile import COMGR_FRESH_PROCESS_ATTR

    row = (arch, 64, "fp16", "batched", "band", 8, "default")
    for sched in DQ_SCHEDS:
        k = _kernel(row, sched)
        assert k.attrs.get(COMGR_FRESH_PROCESS_ATTR, False) is (sched == "iglp1"), sched
    k = build_attn_bwd_dq(_spec(*row, "iglp1"), arch=arch)  # not the cached one
    ll = lower_kernel_to_llvm(k, arch=arch)
    assert "llvm.amdgcn.iglp.opt(i32 1)" in ll
    k.attrs.pop(COMGR_FRESH_PROCESS_ATTR)
    assert lower_kernel_to_llvm(k, arch=arch) == ll


def test_plan_and_registry_carry_dq_sched():
    from dispatch.attention_bwd.tuning_common import KIND_DQ, backward_axis
    from kernels.common.attention_bwd_plan import DQ_KNOBS

    assert "dq_sched" in DQ_KNOBS
    ax = backward_axis("dq_sched")
    assert ax.choices == DQ_SCHEDS and ax.default == "none"
    assert ax.kind == KIND_DQ and ax.family == 8 and ax.numerics == "bitwise"
    for v in VALUES:
        assert ax.applies(v, {"dq_mode": "split"})
        assert not ax.applies(v, {"dq_mode": "atomic"}) and not ax.applies(v, {})
    from kernels.common.attention_bwd import validate_attn_bwd_spec

    main = validate_attn_bwd_spec(AttnBwdSpec(head_size=64, dq_mode="split"), "gfx950")
    for v in VALUES:
        dq = validate_attn_bwd_dq_spec(main.dq_spec(sched=v), "gfx950")
        assert dq.sched == v


def test_sweep_enumerates_dq_sched_on_split_points():
    """The sweep's dQ values (stage T, one dQ field at a time) hold every
    ``dq_sched`` value on a split candidate once family 8 is released."""
    from benchmarks.common.attention_bwd_sweep import (
        DQ_EXTRA,
        Candidate,
        Release,
        default_knobs,
        dq_values,
        primary_slice,
    )

    assert "dq_sched" in DQ_EXTRA
    sl = primary_slice("gfx950", "general", 64)
    c = Candidate.make(sl, dict(default_knobs(sl), dq_mode="split"), {})
    vals = dq_values(c, Release(frozenset(range(1, 9))))
    got = {v["dq_sched"] for v in vals if set(v) == {"dq_sched"}}
    assert got == set(VALUES)
    assert dq_values(c, Release(frozenset(range(1, 8)))) == [{}]


# ---------------------------------------------------------------------------
# default
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ARCHS)
def test_default_emits_no_hint_and_is_the_spelled_default(arch):
    for d in HEADS:
        for var in ("default", "prefetch", "group_edge", "kvpad16"):
            row = (arch, d, "fp16", "batched", "band", 8, var)
            if not _legal(row):
                continue
            k = _kernel(row)
            names = Counter(op.name for op, _ in _walk(k.body.ops))
            assert not any(names[h] for h in _HINTS), (d, var)
            plain = build_attn_bwd_dq(
                dataclasses.replace(_spec(*row), sched="none"), arch=arch
            )
            assert lower(k, arch=arch, llvm_flavor="llvm22") == lower(
                plain, arch=arch, llvm_flavor="llvm22"
            )


# ---------------------------------------------------------------------------
# the count model against the IR
# ---------------------------------------------------------------------------


def test_rows_cover_the_variants():
    assert {r[-1] for r in ROWS} == set(VARIANTS)
    assert {r[0] for r in ROWS} == set(ARCHS)
    assert set(ROWS) <= set(VALUE_ROWS)
    # the rows only the values carry: their default kernel is refused for
    # its resources, and every value is legal there
    for row in set(VALUE_ROWS) - set(ROWS):
        with pytest.raises(ValueError, match="DQ_RESOURCE_REFUSED"):
            validate_attn_bwd_dq_spec(_spec(*row), row[0])
        assert all(_legal(row, v) for v in VALUES), row


@pytest.mark.parametrize("sched", DQ_SCHEDS)
@pytest.mark.parametrize("row", VALUE_ROWS, ids=_ids)
def test_hint_sequence_and_stage_counts_equal_the_model(row, sched):
    """Per key step (every key loop): the hint ops are the model's sequence;
    under a fenced value each stage region's census and leading reads equal
    the model (the prefetch loads in A, the next tile's stores in C)."""
    if not _legal(row, sched):
        pytest.skip("not legal for this configuration")
    t = _model(row, sched)
    assert t.sched == sched
    kernel = _kernel(row, sched)
    loops = _key_loops(kernel)
    assert loops
    model = t.ir_count_model()
    for lp in loops:
        ops = lp.regions[0].ops
        assert _hint_seq(ops) == t.hint_sequence()
        flat = [op for op, _ in _walk(ops)]
        assert sum(op.name == "tile.mma" for op in flat) == sum(t.mfma.values())
        for key in _HINTS:
            assert sum(op.name == key for op in flat) == model[key], key
        if sched in IGLP_SCHEDS:
            first = next(i for i, op in enumerate(ops) if op.name == "tile.iglp_opt")
            assert not any(op.name == "tile.mma" for op, _ in _walk(ops[:first]))
        if t.setprio and t.kv_prefetch:
            # s_setprio is a scheduling boundary: A's raise precedes every
            # global load of the prefetch step, so the loads share the
            # scheduling region of G0' and its groups (the plain step's
            # staging copy precedes A by design)
            raise_at = next(
                i for i, op in enumerate(flat) if op.name == "tile.s_setprio"
            )
            loads = [
                i
                for i, op in enumerate(flat)
                if op.name.startswith("memref.global_load")
            ]
            assert all(i > raise_at for i in loads), (raise_at, loads[:3])
        if sched not in DQ_FENCED_SCHEDS:
            continue
        parts = _regions(ops, t)
        for name, want, lead_want in t.regions():
            got, lead = _census(parts[name])
            for cls, cname in ((MFMA, "mfma"), (VMEM_READ, "vmem_read"),
                               (VMEM_WRITE, "vmem_write"), (DS_READ, "ds_read"),
                               (DS_WRITE, "ds_write")):  # fmt: skip
                assert got[cls] == want.get(cls, 0), (name, cname, got[cls], want)
            assert lead == lead_want.get(DS_READ, 0), (name, "lead", lead)
        if t.kv_prefetch:
            assert t.stage("A").vmem_read == t.prefetch()["vmem_read"] > 0
            assert dict(t.stage("C").fixed)[DS_WRITE] == t.prefetch()["ds_write"]


def test_model_counts_by_hand():
    """Hand-derived counts: gfx950 d128, kM0 64, kN0 64, 4 waves, G0' atom
    16x16x32 (8-element fragments), G4' 16x16x16 through tr_read."""
    t = DqStageTable(arch="gfx950", head_size=128, k_m0=64, k_n0=64, waves=4,
                     atom_g02=32, atom_g4=16)  # fmt: skip
    # G0' / G2': 4 key frags x 1 column frag x 4 K steps; G4': 1 x 8 x 4
    assert t.mfma == {"g0": 16, "g2": 16, "g4": 32}
    a, b, c = t.stages()
    assert (a.ds_read, dict(a.lead)[DS_READ]) == (16, 4)  # one 16-byte read
    assert (b.ds_read, c.ds_read, dict(c.lead)[DS_READ]) == (16, 32, 8)
    assert a.vmem_read == 0 and not c.fixed
    pad8 = dataclasses.replace(t, kv_pad=8)  # 4-element rows: two reads each
    assert pad8.stages()[0].ds_read == 32 and pad8.stages()[2].ds_read == 32
    plain = dataclasses.replace(t, transpose_source="lds_plain")
    assert plain.stages()[2].ds_read == 4 * 32  # one element per read
    pf = dataclasses.replace(t, kv_prefetch=True)
    # 64 x 128 / 8 = 1024 chunks over 256 threads: 4 per tile, K and V
    assert pf.prefetch() == {"vmem_read": 8, "ds_write": 8}
    assert dataclasses.replace(pf, stage_vec=1).prefetch()["vmem_read"] == 64
    assert dataclasses.replace(pf, kv_pad=8).prefetch()["ds_write"] == 16
    assert pf.stages()[0].vmem_read == 8 and dict(pf.stages()[2].fixed) == {DS_WRITE: 8}
    st = DqStageTable(arch="gfx950", head_size=64, k_m0=64, k_n0=64, waves=4,
                      sched="stage_table_setprio")  # fmt: skip
    m = st.ir_count_model()
    assert m["tile.s_setprio"] == 6 and m["tile.sched_barrier"] == 2
    assert m["group_counts"][MFMA] == sum(st.mfma.values())
    seq = st.hint_sequence()
    assert seq[0] == ("prio", 1) and seq[-1] == ("prio", 0)
    assert seq.count(("fence", 0)) == 2
    for bad in (dict(sched="stage_table_atomics"), dict(kv_pad=4),
                dict(transpose_source="xt_lds"), dict(k_m0=48)):  # fmt: skip
        with pytest.raises(ValueError):
            dataclasses.replace(st, **bad)
    with pytest.raises(ValueError, match="fenced"):
        DqStageTable(arch="gfx950", head_size=64, k_m0=64, k_n0=64, waves=4,
                     sched="iglp1").regions()  # fmt: skip


# ---------------------------------------------------------------------------
# scheduling only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sched", VALUES)
@pytest.mark.parametrize("arch", ARCHS)
def test_scheduling_only(arch, sched):
    """Against the twin without a schedule: every non-hint op of the kernel
    in the same order (no op moves), the resolved spec apart from ``sched``,
    the LDS plan and the register estimate; the kernel name is salted and
    the IR differs."""
    for row in ROWS:
        if row[0] != arch or not _legal(row, sched):
            continue
        r0 = validate_attn_bwd_dq_spec(_spec(*row), arch)
        r = validate_attn_bwd_dq_spec(_spec(*row, sched), arch)
        assert dataclasses.replace(r, sched="none") == r0
        assert dq_lds_plan(r, arch) == dq_lds_plan(r0, arch)
        assert dq_register_estimates(r, arch) == dq_register_estimates(r0, arch)
        assert r.kernel_name() != r0.kernel_name()
        k0, k = _kernel(row), _kernel(row, sched)
        n0 = [(op.name, op.attrs.get("op_id")) for op, _ in _walk(k0.body.ops)]
        n1 = [
            (op.name, op.attrs.get("op_id"))
            for op, _ in _walk(k.body.ops)
            if op.name not in _HINTS
        ]
        assert n1 == n0, row
        ir0 = lower(k0, arch=arch, llvm_flavor="llvm22")
        ir = lower(k, arch=arch, llvm_flavor="llvm22")
        assert ir.replace(r.kernel_name(), "K") != ir0.replace(r0.kernel_name(), "K")


# ---------------------------------------------------------------------------
# LDS: barrier replay of every access
# ---------------------------------------------------------------------------


def traced_dq(spec, arch):
    """Build the dQ kernel with every LDS op tagged ``(lo, hi, store,
    private)`` (the view's span in allocation elements)."""
    from kernels.common._attention_bwd_frag import LdsView

    from .test_attention_bwd_ir import _is_lds

    tags = {}
    orig = {n: getattr(LdsView, n) for n in ("load", "store", "tr16_load")}

    def tagged(name):
        def f(self, b, *args, **kw):
            ops = b._region_stack[-1].ops
            n0 = len(ops)
            out = orig[name](self, b, *args, **kw)
            new = [op for op in ops[n0:] if _is_lds(op)]
            assert new
            hi = self.base + self.rows * self.pitch * self.scale
            for op in new:
                tags[id(op)] = (self.base, hi, name == "store", False)
            return out

        return f

    try:
        for n in orig:
            setattr(LdsView, n, tagged(n))
        kernel = build_attn_bwd_dq(spec, arch=arch)
    finally:
        for n, f in orig.items():
            setattr(LdsView, n, f)
    return kernel, tags


_HAZARD_VARS = ("default", "prefetch", "group_edge", "prefetch_group_edge",
                "kvpad8", "plain")  # fmt: skip


@pytest.mark.parametrize("sched", DQ_SCHEDS)
@pytest.mark.parametrize("arch", ARCHS)
def test_lds_accesses_do_not_race(arch, sched):
    from .test_attention_bwd_ir import lds_hazards

    n = 0
    for d in HEADS:
        for var in _HAZARD_VARS:
            row = (arch, d, "fp16", "batched", "band", 8, var)
            if not _legal(row, sched):
                continue
            kernel, tags = traced_dq(_spec(*row, sched), arch)
            assert not lds_hazards(kernel, tags)[:4], row
            loop = [op for lp in _key_loops(kernel)
                    for op, _ in _walk(lp.regions[0].ops) if id(op) in tags]  # fmt: skip
            # the step's K / V writes (copies or prefetch stores) and reads
            assert any(tags[id(op)][2] for op in loop)
            assert any(not tags[id(op)][2] for op in loop)
            n += 1
    assert n


def test_hazard_replay_sees_the_key_step_barriers():
    """Mutants: without the key step's barriers the next tile's K / V writes
    race the step's reads (the replay is not vacuous)."""
    from .test_attention_bwd_ir import lds_hazards

    for var in ("default", "prefetch"):
        row = ("gfx950", 64, "fp16", "batched", "band", 8, var)
        kernel, tags = traced_dq(_spec(*row, "stage_table"), "gfx950")
        assert not lds_hazards(kernel, tags)
        for lp in _key_loops(kernel):
            body = lp.regions[0].ops
            body[:] = [op for op in body if op.name != "tile.sync"]
        assert lds_hazards(kernel, tags), var


# ---------------------------------------------------------------------------
# compile: LLVM calls, code objects, resource gate
# ---------------------------------------------------------------------------

_LLVM = {
    "tile.sched_group_barrier": "call void @llvm.amdgcn.sched.group.barrier(",
    "tile.sched_barrier": "call void @llvm.amdgcn.sched.barrier(",
    "tile.s_setprio": "call void @llvm.amdgcn.s.setprio(",
    "tile.iglp_opt": "call void @llvm.amdgcn.iglp.opt(",
}


@pytest.mark.parametrize("sched", DQ_SCHEDS)
@pytest.mark.parametrize("arch", ARCHS)
def test_llvm_hint_calls_equal_the_model(arch, sched):
    for row in VALUE_ROWS:
        if row[0] != arch or not _legal(row, sched):
            continue
        model = _model(row, sched).ir_count_model()
        k = _kernel(row, sched)
        n = len(_key_loops(k))
        ir = lower(k, arch=arch, llvm_flavor="llvm22")
        for key, call in _LLVM.items():
            assert ir.count(call) == n * model[key], (key, row)


def _compile(kernel, arch):
    pytest.importorskip("rocke.runtime.comgr")
    from ._attention_bwd_resources import compile_or_skip, pin_production_comgr

    pin_production_comgr()
    return compile_or_skip(kernel, arch=arch, capture_ir_text=True, backend="python")


@pytest.mark.parametrize("d", HEADS)
@pytest.mark.parametrize("sched", VALUES)
@pytest.mark.parametrize("arch", ARCHS)
def test_code_objects_keep_the_matrix_instructions(arch, sched, d):
    from .test_attention_bwd_sched import _MMA_RE, _disassemble

    n = 0
    for var in ("default", "prefetch"):
        row = (arch, d, "fp16", "batched", "band", 8, var)
        if not (_legal(row) and _legal(row, sched)):
            continue  # a refused pair (DQ_SCHED_SPILLING): no code object
        n += 1
        t0 = _disassemble(_compile(_kernel(row), arch).hsaco, arch)
        t1 = _disassemble(_compile(_kernel(row, sched), arch).hsaco, arch)
        assert len(_MMA_RE.findall(t0)) == len(_MMA_RE.findall(t1)) > 0
        b0 = [ln.split("//")[0].strip() for ln in t0.splitlines() if "\t" in ln]
        b1 = [ln.split("//")[0].strip() for ln in t1.splitlines() if "\t" in ln]
        if sched not in IGLP_SCHEDS:
            assert b1 != b0, (row, sched)
    assert n, (arch, sched, d)  # the default tile always takes the value


# Gate rows: the dQ suite's variants (both problem classes) and the count
# variants, every value and its twin.
GATE_VARS = tuple(VARIANTS)
# Default dQ kernels (dq_sched none) that fail the gate on the production
# compiler at these rows (the 4-wave 64 x 128 tile at d128 on gfx950 and at
# d64 on gfx942, both problem classes) are refused exactly
# (DQ_RESOURCE_REFUSED, pinned by test_attention_bwd_dq.DQ_SPILL_ROWS); every
# schedule value passes there. No failing row is exempted.


def test_sched_spilling_refusal_is_exact():
    """``iglp0`` with the K / V prefetch at d128 on gfx942 (the start tile)
    spills to scratch on the production compiler where the default key step
    does not: the validator refuses exactly that (DQ_SCHED_SPILLING), on both
    problem classes; one field less or another value, arch or head size is
    built."""
    from kernels.common.attention_bwd_dq import DQ_SCHED_SPILLING

    assert DQ_SCHED_SPILLING == {
        ("gfx942", 128, 4, 64, 64, True, "single", False, 0, "lds_plain", "iglp0")
    }
    for cls in CLASSES:
        for var in ("prefetch", "prefetch_plain"):
            row = ("gfx942", 128, *cls, var)
            with pytest.raises(ValueError, match="DQ_SCHED_SPILLING"):
                validate_attn_bwd_dq_spec(_spec(*row, "iglp0"), "gfx942")
            for other in ("none", "iglp1", "fences_only", "stage_table"):
                assert _legal(row, other), (row, other)
        assert _legal(("gfx950", 128, *cls, "prefetch"), "iglp0")
        assert _legal(("gfx942", 64, *cls, "prefetch"), "iglp0")
        assert _legal(("gfx942", 128, *cls, "prefetch_pad8"), "iglp0")
        assert _legal(("gfx942", 128, *cls, "default"), "iglp0")
        assert _legal(("gfx942", 128, *cls, "prefetch_group_edge"), "iglp0")


@pytest.mark.parametrize("var", GATE_VARS)
@pytest.mark.parametrize("sched", DQ_SCHEDS)
@pytest.mark.parametrize("arch", ARCHS)
def test_resource_gate_per_value(arch, sched, var):
    """The dQ kernel's resource gate (as
    ``test_attention_bwd_dq.test_zero_scratch_no_spills_and_no_key_loop_agpr_copies``:
    no scratch, no spills, the register budgets, LDS within the plan, no
    SGPR-margin warning, no AGPR copy in the key loops except under the AGPR
    reservation lever) for every legal value, head size and problem class."""
    pytest.importorskip("rocke.runtime.comgr")
    from ._attention_bwd_resources import (
        GATE_FLAVOR,
        comgr_is_production,
        comgr_version,
        current_flavor,
        evaluate_resource_gate,
        find_objdump,
    )

    if find_objdump() is None:
        pytest.skip("llvm-objdump not found (needed for the key-loop AGPR copy rule)")
    flavor, comgr = current_flavor(), comgr_version()
    facts = attn_bwd_arch_facts(arch)
    failures, n = [], 0
    for row in VALUE_ROWS:
        if row[0] != arch or row[-1] != var or not _legal(row, sched):
            continue
        art = _compile(_kernel(row, sched), arch)
        if flavor != GATE_FLAVOR or not comgr_is_production(comgr_version()):
            pytest.skip(f"resource gate needs {GATE_FLAVOR} + production comgr")
        r = validate_attn_bwd_dq_spec(_spec(*row, sched), arch)
        tile = {}
        if r.agpr_alloc == (0, 0):
            tile = dict(block_m=r.block_m, block_n=r.block_n, waves=r.waves)
        res = evaluate_resource_gate(
            art.hsaco, arch=arch, lds_capacity_bytes=facts.lds_capacity_bytes,
            ir_text=art.llvm_text, **tile,
        )  # fmt: skip
        if not res.valid:
            pytest.skip(f"gate not valid here ({res.flavor}, {comgr})")
        n += 1
        if res.failures or res.warnings:
            failures.append((row, res.failures, res.warnings))
        if res.resources.lds_bytes > dq_lds_plan(r, arch).total_bytes:
            failures.append((row, "LDS above the plan"))
        if tile:
            if not (res.loops and all(x["reads"] == 0 for x in res.loops)):
                failures.append((row, "key-loop AGPR copies", res.loops))
    if not n:
        pytest.skip(f"{var} is not legal with {sched} on {arch}")
    assert not failures, failures
