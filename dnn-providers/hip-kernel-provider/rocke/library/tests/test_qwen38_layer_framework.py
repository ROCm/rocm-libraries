# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Qwen3.8 layer-kernel framework (``kernels.gfx950.qwen38_mono``: framework,
stamps, atom_layout, toy_chain).

CPU: the ATOM-compatible ABI (kernarg offsets), the ATOM schedule / scratch
mirror and stamp format (against the ATOM tree when ``Q38_ATOM_DIR`` names
it), validation of ATOM's real schedules (single, multi-layer, heterogeneous)
and of an overlap lever that breaks the implicit cross-layer barrier, the toy
chain's knob grid (builds, mirrored ops only, default knobs byte-identical,
deviation-only names), LDS phase barriers, stage_kernel, the host models.

GPU (gfx950; ``harness.env_problem`` must be clear): toy-chain numerics over
the knob grid at a normal and a near-wrap step, graph replays with an
advancing step counter (``Q38_REPLAYS``, default 10**4; the T9 gate), and the
ISA check that the framework's LDS-phase barrier does not drain ``vmcnt``.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest

from kernels.gfx950.qwen38_mono import atom_layout as A
from kernels.gfx950.qwen38_mono import stamps as stp
from kernels.gfx950.qwen38_mono import toy_chain as T
from kernels.gfx950.qwen38_mono.framework import (
    ATOM_LAYER_ARGS,
    FrameworkKnobs,
    LayerShape,
    ScratchLayout,
    StageSpec,
    build_layer,
    build_layer_kernel,
    kernarg_offset,
    stage_kernel,
)
from rocke.core.ir import uses_python_only_ops
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.helpers import mailbox as mbx
from rocke.helpers.lds_arena import LdsEntry
from rocke.helpers.task_schedule import Dep, Stage, TaskTable, validate_stack

FLAVOR = "llvm22"


def replace_spec(spec, **kw):
    from dataclasses import replace

    return replace(spec, **kw)


def _ll(k) -> str:
    return lower_kernel_to_llvm(k, arch="gfx950", llvm_flavor=FLAVOR)


def _ops(k, name: str) -> list:
    out = []

    def walk(region):
        for op in region.ops:
            if op.name == name:
                out.append(op)
            for r in op.regions:
                walk(r)

    walk(k.body)
    return out


# ------------------------------------------------------------- ATOM tree


def _atom_dir() -> Path | None:
    d = os.environ.get("Q38_ATOM_DIR")
    if not d:
        return None
    p = Path(d) / "atom" / "models" / "qwen4_exp_mono" / "kernels"
    return p if p.is_dir() else None


def _load_atom(name: str):
    """Load ATOM's ``kernels/<name>.py`` without importing the ``atom``
    package (stub parents; layout.py imports only config.py)."""
    root = _atom_dir()
    if root is None:
        pytest.skip("Q38_ATOM_DIR not set (ATOM tree for the contract checks)")
    pkgs = ("atom", "atom.models", "atom.models.qwen4_exp_mono",
            "atom.models.qwen4_exp_mono.kernels")  # fmt: skip
    for pkg in pkgs:
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = []
            sys.modules[pkg] = m
    full = f"atom.models.qwen4_exp_mono.kernels.{name}"
    if full in sys.modules and hasattr(sys.modules[full], "__file__"):
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, root / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


def test_atom_layout_mirror_matches_atom_tree():
    _load_atom("config")
    lay = _load_atom("layout")
    assert A.TASKS == lay.TASKS
    assert A.BASE == lay.BASE
    assert A.ORDERS == lay.ORDERS
    assert A.REGIONS == lay.REGIONS
    assert A.SC == lay.SC
    assert A.SCRATCH_BYTES == lay.SCRATCH_BYTES
    assert A.MAX_ROWS == lay.MAX_ROWS
    sl = A.atom_layout()
    assert sl.offsets == lay.SC and sl.row_bytes == lay.SCRATCH_BYTES


def test_layer_args_and_stamp_format_match_atom_tree():
    root = _atom_dir()
    if root is None:
        pytest.skip("Q38_ATOM_DIR not set")
    src = (root / "kernel.py").read_text()
    block = re.search(r"LAYER_ARGS = \((.*?)\)\n", src, re.S).group(1)
    names = tuple(re.findall(r'"([a-z0-9_]+)"', block))
    assert names == ATOM_LAYER_ARGS
    st = (root / "stamps.py").read_text()
    for const, val in (("LAYER_SLOTS", stp.LAYER_SLOTS), ("POINTS", stp.POINTS),
                       ("MAX_STAGES", stp.MAX_STAGES)):  # fmt: skip
        assert re.search(rf"^{const} = {val}\b", st, re.M), const
    assert "PER_STAGE = 2 + WAVES" in st
    assert "(layer * GRID + fx.block_idx.x) * POINTS + tid" in st


def test_kernarg_offsets_atom_abi():
    assert kernarg_offset("scratch") == 16
    assert kernarg_offset("h_out") == 16 + 8 * 40
    k = T.build_toy_chain(T.ToyChainSpec(depth=1))
    names = [p.name.lstrip("%") for p in k.params]
    assert names == ["step_ptr", "layer", *ATOM_LAYER_ARGS]


# ------------------------------------------------------- ATOM schedules


@pytest.mark.parametrize("kind", ["delta", "qsa", "delta_ple"])
def test_atom_schedules_validate(kind):
    assert A.atom_table(kind).validate() == []
    assert A.atom_table(kind, stack=True).validate(layers=3) == []


def test_atom_heterogeneous_stack_validates():
    kinds = ("delta", "delta_ple", "delta", "qsa", "delta", "delta", "qsa")
    assert validate_stack(A.atom_stack(kinds)) == []


def test_overlap_lever_needs_parity():
    """Roadmap 3.2: A1(L+1) waiting on only part of M5(L) (an overlap lever)
    breaks the implicit all-to-all barrier; parity slots for the regions the
    next layer writes early repair the adjacent-layer case."""
    t = A.atom_table("delta", stack=True)
    early = []
    for s in t.stages:
        if s.name == "a1":
            s = Stage("a1", s.tasks, base=s.base, deps=(Dep("m5", full=False, layer=-1),),
                      reads=s.reads, writes=s.writes)  # fmt: skip
        early.append(s)
    lever = t.with_stages(early)
    errs = lever.validate(layers=2, strict=False)
    assert errs and all("cross-layer WAR" in e for e in errs)
    flagged = {re.search(r"writes '(\w+)'", e).group(1) for e in errs}
    # what M5(L) reads is no longer known to be read before layer L+1's
    # C1 / M1 / M3 / M4 rewrite it; earlier regions stay ordered through
    # the full chain A2 -> ... -> M5 -> A1(L+1)
    assert flagged == {"w", "minj", "lg", "sg", "dn"}
    from dataclasses import replace

    fixed = replace(lever, parity=tuple(A.REGIONS))
    assert fixed.validate(layers=2) == []


# ------------------------------------------------------------ toy chain


def _grid():
    for hop in T.HOPS:
        for consume in T.CONSUMES:
            for work in (0, 8):
                for layers in (1, 2):
                    yield T.ToyChainSpec(depth=3, hop=hop, consume=consume, work=work,
                                         layers=layers)  # fmt: skip


@pytest.mark.parametrize("spec", list(_grid()), ids=lambda s: s.name())
def test_toy_builds_mirrored_and_validates(spec):
    T.toy_table(spec).validate(layers=spec.layers)
    k = T.build_toy_chain(spec)
    assert not uses_python_only_ops(k)
    ll = _ll(k)
    assert "define amdgpu_kernel" in ll


_PARITY = [
    T.ToyChainSpec(depth=2, work=8),
    T.ToyChainSpec(depth=2, hop="last_arriver", consume="waves", reduce="dpp"),
    T.ToyChainSpec(depth=2, hop="reduce_task", layers=2,
                   knobs=FrameworkKnobs(stamps=True, head_sleep=64)),
    T.ToyChainSpec(depth=2, work=8, prefetch="l2_touch", placement="xcd_blocked"),
    T.ToyChainSpec(depth=2, layers=2, knobs=FrameworkKnobs(
        mailbox=mbx.MailboxKnobs(numa_copies=2), numa_flip_arg="s_down")),
    T.ToyChainSpec(depth=2, work=8, early="entry", knobs=FrameworkKnobs(step_first=True)),
    T.ToyChainSpec(depth=2, knobs=FrameworkKnobs(stamps=True, stamps_lds="own", lds_touch=True)),
    T.ToyChainSpec(depth=2, work=8, early="entry", knobs=FrameworkKnobs(step_load="smem")),
]  # fmt: skip


@pytest.mark.parametrize("spec", _PARITY, ids=lambda s: s.name())
def test_toy_cpp_engine_byte_identical(spec):
    """The framework + toy emit mirrored ops only: with the C++ engine
    binding, ``backend="both"`` lowers every lever's build to the same bytes."""
    if importlib.util.find_spec("rocke_engine") is None:
        pytest.skip("cpp engine needs the rocke_engine binding")
    os.environ.setdefault("ROCKE_CPP_QUIET_FALLBACK", "1")
    from rocke.helpers.compile import _lower_llvm_via_backend

    assert _lower_llvm_via_backend(T.build_toy_chain(spec), arch="gfx950",
                                   backend="both", spec=None)  # fmt: skip


def test_toy_default_knobs_byte_identical_and_names():
    a = T.ToyChainSpec(depth=2, work=8)
    b = T.ToyChainSpec(depth=2, work=8, knobs=FrameworkKnobs(), reduce="swizzle",
                       placement="atom", prefetch="off")  # fmt: skip
    assert _ll(T.build_toy_chain(a)) == _ll(T.build_toy_chain(b))
    assert a.name() == "rk_toy_d2_p160_w8_a2a_wave0"
    names = {T.build_toy_chain(T.ToyChainSpec(depth=2, work=8, **kw)).name for kw in (
        {}, {"reduce": "dpp"}, {"prefetch": "l2_touch"}, {"placement": "sequential"},
        {"placement": "xcd_blocked"}, {"layers": 2}, {"layers": 2, "parity": False},
        {"knobs": FrameworkKnobs(stamps=True)}, {"knobs": FrameworkKnobs(fresh_ids=False)},
        {"knobs": FrameworkKnobs(mailbox=mbx.MailboxKnobs(pause="s_sleep1"))},
    )}  # fmt: skip
    assert len(names) == 10
    k = T.build_toy_chain(T.ToyChainSpec(depth=2, knobs=FrameworkKnobs(stamps=True)))
    assert k.name.endswith("_tl")


def test_toy_lever_ir_shapes():
    base = _ll(T.build_toy_chain(T.ToyChainSpec(depth=2, work=8)))
    dpp = _ll(T.build_toy_chain(T.ToyChainSpec(depth=2, work=8, reduce="dpp")))
    assert "permlane32" in dpp and "update.dpp" in dpp and "update.dpp" not in base
    la = _ll(T.build_toy_chain(T.ToyChainSpec(depth=2, hop="last_arriver")))
    assert "global_atomic_add" in la
    pf = _ll(T.build_toy_chain(T.ToyChainSpec(depth=2, work=8, prefetch="l2_touch")))
    assert "global_load_dwordx4" in pf and "global_load_dwordx4" not in base
    nk = FrameworkKnobs(mailbox=mbx.MailboxKnobs(numa_copies=2), numa_flip_arg="s_down")
    nu = T.build_toy_chain(T.ToyChainSpec(depth=2, knobs=nk))
    assert nu.name.endswith("mbnuma2_cal")
    lay = T.toy_layout(T.ToyChainSpec(depth=2, knobs=nk))
    assert lay.numa_copies == 2 and all(o % 16384 == 0 for o in lay.offsets.values())
    nl = _ll(nu)
    one = _ll(T.build_toy_chain(T.ToyChainSpec(depth=2)))
    assert nl.count("global_store_dwordx2") == 2 * one.count("global_store_dwordx2")
    assert "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID)" in nl
    with pytest.raises(ValueError, match="numa_copies"):
        build_layer_kernel("x", [], TaskTable(()), ScratchLayout.of({}),
                           knobs=FrameworkKnobs(mailbox=mbx.MailboxKnobs(numa_copies=2)))
    st = _ll(T.build_toy_chain(T.ToyChainSpec(depth=2, knobs=FrameworkKnobs(stamps=True))))
    assert st.count("s_memrealtime") > base.count("s_memrealtime")


def test_toy_prefetch_keeps_destinations_until_drained():
    """The idle-CTA prefetch's asm loads: every destination is an operand of
    a later ``s_waitcnt vmcnt(0)`` asm (else the allocator reuses it while
    the load is in flight: a memory fault on gfx950)."""
    for depth, layers in ((2, 1), (8, 1), (3, 2)):
        spec = T.ToyChainSpec(depth=depth, work=8, prefetch="l2_touch", layers=layers)
        ll = _ll(T.build_toy_chain(spec))
        loads = re.findall(r'(%\S+) = call <4 x i32> asm sideeffect "global_load_dwordx4', ll)
        drains = re.findall(r'call void asm sideeffect "s_waitcnt vmcnt\(0\)", "([v,]+)"\(([^)]*)\)', ll)
        idle, per = (256 - 160) // 8, 160 // 8
        expect = (layers * depth - 1) * -(-per // idle) * -(-(8 * 512 // 4) // 512)
        assert len(loads) == expect
        assert len(drains) == -(-expect // T.PREFETCH_BATCH)
        used = " ".join(d[1] for d in drains)
        assert all(re.search(re.escape(v) + r"\b", used) for v in loads)
        assert sum(c.count("v") for c, _ in drains) == expect


def test_toy_early_entry_and_step_first():
    """``early="entry"``: g0's W / X loads move to the framework entry hook
    (before the stage loop); ``step_first``: the step load precedes them.
    Both default off (names and bytes unchanged)."""
    base = T.ToyChainSpec(depth=2, work=8)
    assert _ll(T.build_toy_chain(base)) == _ll(T.build_toy_chain(
        T.ToyChainSpec(depth=2, work=8, early="off", knobs=FrameworkKnobs(step_first=False))))
    lines = {}
    for sf in (False, True):
        spec = T.ToyChainSpec(depth=2, work=8, early="entry", knobs=FrameworkKnobs(step_first=sf))
        k = T.build_toy_chain(spec)
        assert k.name.endswith("_eaentry" + ("_sf" if sf else ""))
        ll = _ll(k).splitlines()
        step = next(i for i, ln in enumerate(ll) if "(i64 %step_ptr)" in ln)
        loads = [i for i, ln in enumerate(ll) if re.search(r"= load i32, ptr addrspace\(1\)", ln)]
        lines[sf] = (step, loads)
    s0, l0 = lines[False]
    s1, l1 = lines[True]
    assert len([i for i in l0 if i < s0]) == 2 * 8  # all 8 W + 8 X loads before the step
    assert len([i for i in l1 if i < s1]) == 0 and s1 < l1[0]
    # the same count of loads either way: g0 does not reload what entry issued
    assert len(l0) == len(l1)
    plain = [ln for ln in _ll(T.build_toy_chain(base)).splitlines()
             if re.search(r"= load i32, ptr addrspace\(1\)", ln)]  # fmt: skip
    assert len(plain) == len(l0)
    with pytest.raises(ValueError, match="early"):
        T.ToyChainSpec(early="bogus")


def test_stamps_own_lds_and_touch_knobs():
    """``stamps_lds="own"``: the stamps in their own (first) allocation;
    ``lds_touch``: one entry store keeps a padded arena allocated. Off by
    default (names and bytes unchanged); ``tlown`` only names stamped builds."""
    a = T.ToyChainSpec(depth=2, knobs=FrameworkKnobs(stamps=True))
    b = T.ToyChainSpec(depth=2, knobs=FrameworkKnobs(stamps=True, stamps_lds="arena", lds_touch=False))
    assert _ll(T.build_toy_chain(a)) == _ll(T.build_toy_chain(b))
    own = T.build_toy_chain(T.ToyChainSpec(depth=2, knobs=FrameworkKnobs(stamps=True, stamps_lds="own")))
    assert own.name.endswith("_tl_tlown")
    assert T.build_toy_chain(T.ToyChainSpec(depth=2, knobs=FrameworkKnobs(stamps_lds="own"))).name         == T.build_toy_chain(T.ToyChainSpec(depth=2)).name
    allocs = _ops(own, "tile.smem_alloc")
    assert len(allocs) == 2 and allocs[0].results[0].name.lstrip("%").startswith("tl")
    t = T.build_toy_chain(T.ToyChainSpec(depth=2, knobs=FrameworkKnobs(lds_touch=True)))
    assert t.name.endswith("_lt")
    n0 = len(_ops(T.build_toy_chain(T.ToyChainSpec(depth=2)), "tile.smem_store"))
    assert len(_ops(t, "tile.smem_store")) == n0 + 1
    with pytest.raises(ValueError, match="stamps_lds"):
        FrameworkKnobs(stamps_lds="x")


def test_step_load_smem_knob():
    """``step_load="smem"``: the step counter through one scalar asm load
    (glc) with its own ``lgkmcnt(0)``; default unchanged."""
    d = _ll(T.build_toy_chain(T.ToyChainSpec(depth=2)))
    assert d == _ll(T.build_toy_chain(T.ToyChainSpec(depth=2, knobs=FrameworkKnobs(step_load="vmem"))))
    k = T.build_toy_chain(T.ToyChainSpec(depth=2, knobs=FrameworkKnobs(step_load="smem")))
    assert k.name.endswith("_stsmem")
    ll = _ll(k)
    assert ll.count("s_load_dword $0, $1, 0x0 glc") == 1 and "s_load_dword" not in d
    with pytest.raises(ValueError, match="step_load"):
        FrameworkKnobs(step_load="x")


def test_kernarg_preload_knob():
    """``kernarg_preload=N``: the first N parameters ``inreg`` (Python engine
    only, reported by ``uses_python_only_ops``); default unchanged."""
    base = T.build_toy_chain(T.ToyChainSpec(depth=2, work=8))
    assert not uses_python_only_ops(base)
    assert "inreg" not in _ll(base)
    for n in (1, 3, 8):
        k = T.build_toy_chain(T.ToyChainSpec(depth=2, work=8, knobs=FrameworkKnobs(kernarg_preload=n)))
        assert k.name.endswith(f"_kp{n}")
        assert uses_python_only_ops(k) == ["param.inreg"]
        head = next(ln for ln in _ll(k).splitlines() if ln.startswith("define amdgpu_kernel"))
        params = head[head.index("(") + 1:].split(", ")
        assert [("inreg" in p) for p in params[:n + 1]] == [True] * n + [False]
    with pytest.raises(ValueError, match="kernarg_preload"):
        FrameworkKnobs(kernarg_preload=9)


def test_best_spec_is_the_gate_best():
    s = T.best_spec(4, 8, layers=2)
    k = T.build_toy_chain(s)
    assert k.name == "rk_toy_d4_p160_w8_a2a_wave0_l2_pfl2_touch_eaentry_rddpp_mbir_mbs_sleep1_mbnuma2_cal_kp8"
    # per shape: hop-only / shallow chains keep the plain levers
    assert T.build_toy_chain(T.best_spec(4, 0, layers=2)).name.endswith("l2_rddpp_mbir_mbnuma2_cal_kp8")
    assert T.build_toy_chain(T.best_spec(4, 8)).name.endswith("wave0_rddpp_mbir_mbnuma2_cal_kp8")
    assert T.build_toy_chain(T.best_spec(2, 8, layers=2)).name.endswith("l2_rddpp_mbir_mbnuma2_cal_kp8")
    assert "pfl2_touch_eaentry_rddpp_mbir_mbnuma2" in T.build_toy_chain(T.best_spec(8, 8)).name
    assert uses_python_only_ops(k) == [
        "param.inreg", "tile.buffer_store_cp", "tile.int_to_ptr", "tile.tagged_poll",
    ]
    assert T.toy_layout(s).numa_copies == 2


def test_toy_stack_seeds_from_last_layer_word():
    """A multi-layer launch seeds from ``h_in[4 (layers - 1)]`` (the previous
    launch's last layer); one layer keeps word 0 (default bytes unchanged)."""
    assert T.seed_word(T.ToyChainSpec()) == 0
    assert T.seed_word(T.ToyChainSpec(layers=2)) == 4
    from benchmarks.gfx950.qwen38_mono import toy_chain_bench as B

    h = B.seed_words(T.ToyChainSpec(layers=2), 77)
    assert h[4] == 77 and h.sum() == 77
    model = T.toy_reference(
        T.ToyChainSpec(depth=1, layers=2), 5, np.zeros(1, np.int32),
        np.zeros((2, 1, 160, 1), np.int32), [1, 2],
    )  # fmt: skip
    assert model["r"][1][0] == model["out"][1]


def _count_ops(region, name):
    n = 0
    for op in region.ops:
        n += op.name == name
        for r in op.regions:
            n += _count_ops(r, name)
    return n


def test_stacked_stamps_barrier_before_rebegin():
    """A stacked stamped launch barriers between a layer's stamp flush (LDS
    reads by waves 0..2) and the next layer's ``begin`` (zeroing stores from
    thread 0 / each wave's lane 0); single-layer and unstamped builds are
    unchanged."""
    def n(spec):
        return _count_ops(T.build_toy_chain(spec).body, "tile.sync_lds_only")

    st = FrameworkKnobs(stamps=True)
    one, two = n(T.ToyChainSpec(depth=2, knobs=st)), n(T.ToyChainSpec(depth=2, layers=2, knobs=st))
    plain2 = n(T.ToyChainSpec(depth=2, layers=2))
    plain1 = n(T.ToyChainSpec(depth=2))
    assert two - plain2 == 2 * (one - plain1) + 1  # 2 flushes + 1 re-begin barrier


def _fake_stamps(table, t_entry, starts, length):
    """Every CTA's points of one layer: entry ``t_entry[c]``, stage ``i``
    entered at ``starts[i]`` and left ``length`` ticks later."""
    st = np.zeros((stp.GRID, stp.POINTS), np.int64)
    for c in range(stp.GRID):
        st[c, 0] = t_entry[c]
        for i, _ in enumerate(table.order):
            st[c, stp.enter_ix(i)] = starts[i]
            st[c, stp.exit_ix(i)] = starts[i] + length
    return st


def test_stacked_stamps_decode_from_previous_layer_end():
    """``decode_launch``: a stacked launch's later layers are measured from
    the previous layer's end (their point 0 is each CTA's flush end, near the
    kernel entry on idle CTAs); ramps / skews only from launch entries,
    boundaries only into a launch's first layer."""
    from benchmarks.gfx950.qwen38_mono import toy_chain_bench as B

    spec = T.ToyChainSpec(depth=2, layers=2)
    table = T.toy_table(replace_spec(spec, layers=1))
    names = table.order
    l0 = _fake_stamps(table, [1000 + c for c in range(stp.GRID)], [1100, 1200, 1300], 50)
    # layer 1: idle CTAs re-begin right after the entry (1010), busy ones late
    l1 = _fake_stamps(table, [1010 if c > 200 else 1360 for c in range(stp.GRID)],
                      [1340, 1450, 1550], 50)  # fmt: skip
    d0, d1 = B.decode_launch([l0, l1], names, table)
    assert d0["li"] == 0 and d0["ramp"] == 100 and d0["total"] == 350
    assert d1["li"] == 1 and d1["ramp"] is None and d1["entry_skew"] == 0
    assert d1["t0"] == d0["t_end"] == 1350 and d1["total"] == 1600 - 1350
    raw = B.decode_stamps(l1, names, table)
    assert raw["t0"] == 1010  # the meaningless pre-fix decode
    # rebased points: the harness decoder sees the same t0
    rb = B.rebase_stamps(l1, d0["t_end"])
    assert (rb[:, 0] == 1350).all() and (l1[:, 0] != 1350).any()
    # second launch, 2000 ticks later: one boundary (into its layer 0)
    l0b, l1b = l0.copy(), l1.copy()
    for a in (l0b, l1b):
        a[a != 0] += 2000
    per = B.decode_launch([l0, l1], names, table) + B.decode_launch([l0b, l1b], names, table)
    sm = B.summarize({"stamps": [per]})
    assert sm["boundary"] == 3000 - 1600 and sm["ramp"] == 100 and sm["entry_skew"] == 255
    assert sm["total"] == statistics.median([350, 250, 350, 250])
    assert B.decode_launch([l0, l1 * 0], names, table)[1] is None
    assert B.decode_launch([l0 * 0, l1], names, table) == [None, None]


@pytest.mark.parametrize("placement", ["atom", "xcd_blocked", "sequential"])
@pytest.mark.parametrize("hop", list(T.HOPS))
@pytest.mark.parametrize("layers", [1, 2])
def test_toy_lds_parity_alternates_or_is_guarded(placement, hop, layers):
    """Every CTA's consecutive toy stages (launch order, across the layers of
    a stack) use different LDS exchange copies, or the later one starts with
    a CTA barrier. (The table-index parity repeated a copy on every GEMV
    level of reduce_task and at a2a stack layer boundaries: an intermittent
    wrong ``y`` once other waves' next-level ``dot`` stores got fast.)
    ``atom`` placement needs no barrier."""
    for depth in (1, 2, 3, 4):
        spec = T.ToyChainSpec(depth=depth, work=8, hop=hop, placement=placement, layers=layers)
        table = T.toy_table(spec)
        guards = T._parity_guards(spec, table)
        if placement == "atom":
            assert not guards, (depth, sorted(guards))
        last = {}
        for li in range(layers):
            for n in table.order:
                par = T._stage_counter(n, li, depth) & 1
                st = table.stage(n)
                for t in range(min(st.tasks, table.grid)):
                    c = table.cta_of(n, t)
                    if c in last:
                        assert last[c] != par or (li, n) in guards, (depth, li, n, c)
                    last[c] = par
        k = T.build_toy_chain(spec)
        base = _count_ops(T.build_toy_chain(replace_spec(spec, placement="atom")).body,
                          "tile.sync_lds_only")  # fmt: skip
        if placement == "atom" or not guards:
            continue
        assert _count_ops(k.body, "tile.sync_lds_only") > base


def test_toy_tables_shapes():
    s = T.ToyChainSpec(depth=3, hop="reduce_task")
    assert T.toy_stage_names(s) == ("g0", "r0", "g1", "r1", "g2", "r2")
    t = T.toy_table(s)
    assert t.stage("r1").base == 255 and t.stage("g1").tasks == 160
    assert T.toy_stage_names(T.ToyChainSpec(depth=2)) == ("g0", "g1", "out")
    lay = T.toy_layout(T.ToyChainSpec(depth=2, layers=2))
    assert lay.parity_slots == 2
    assert lay.offset("y1", 0, 1) == lay.offset("y1") + lay.row_bytes


def test_framework_phase_barriers_and_stage_kernel():
    def emit(b, ctx, p, t):
        ctx.lds.st_i32(b, ctx.stage_name, 0, t)

    stages = (
        Stage("s0", 8, deps=(Dep("s2", layer=-1),), writes=("r0",), lds_phase="a"),
        Stage("s1", 8, deps=("s0",), reads=("r0",), writes=("r1",), lds_phase="a"),
        Stage("s2", 8, deps=("s1",), reads=("r1",), lds_phase="b"),
    )
    t = TaskTable(stages)
    specs = [StageSpec(s, emit, lds=(LdsEntry(s.name, 4),)) for s in stages]
    lay = ScratchLayout.of({"r0": 8, "r1": 8})
    k = build_layer_kernel("fw_bar", specs, t, lay)
    bars = _ops(k, "tile.s_barrier_bare")
    assert len(bars) == 1  # only the a -> b phase change
    k2 = build_layer_kernel("fw_bar2", specs, t, lay, layers=2, parity=False)
    bars2 = _ops(k2, "tile.s_barrier_bare")
    assert len(bars2) == 3  # + layer boundary b -> a, + the second a -> b
    sk = stage_kernel(specs[1], t, lay)
    assert [p.name.lstrip("%") for p in sk.params][:2] == ["step_ptr", "layer"]
    assert not uses_python_only_ops(sk)
    _ll(sk)


def _dummy_factory(kind, shape, st):
    """Stand-in emitter: polls one pair of every region it reads (wave 0),
    puts one pair into every region it writes."""

    def emit(b, ctx, p, t):
        with b.scf_if(b.cmp_eq(ctx.wave, b.const_i32(0))):
            specs = [(ctx.mb(r, local=True), 0, 1) for r in st.reads if r in A.REGIONS]
            if specs:
                ctx.poll(specs)
            with b.scf_if(b.cmp_eq(ctx.lane, b.const_i32(0))):
                for r in st.writes:
                    if r in A.REGIONS:
                        ctx.put(ctx.mb(r), t, t)

    return StageSpec(st, emit, lds=(LdsEntry(f"lds_{st.name}", 64),))


@pytest.mark.parametrize("kind", ["delta", "qsa", "delta_ple"])
def test_build_layer_per_kind(kind):
    facts = {n: _dummy_factory for n in A.ORDERS[kind]}
    k = build_layer(kind, LayerShape(), factories=facts)
    assert k.name.startswith(f"rk_q38_{kind}_")
    assert [p.name.lstrip("%") for p in k.params] == ["step_ptr", "layer", *ATOM_LAYER_ARGS]
    phases = [A.PHASES[n] for n in A.ORDERS[kind]]
    changes = sum(1 for a, b in zip(phases, phases[1:]) if a != b)
    assert len(_ops(k, "tile.s_barrier_bare")) >= changes
    assert not uses_python_only_ops(k)
    _ll(k)
    nopb = build_layer(kind, factories=facts, knobs=FrameworkKnobs(phase_barriers=False))
    assert nopb.name.endswith("_nopb")
    # a suffix run (the FlyDSL prefix fills the rest)
    sk = build_layer(kind, stages=("m3", "m4", "m5"), factories=facts)
    assert sk.name == f"rk_q38_{kind}_m3_m4_m5"
    with pytest.raises(ValueError, match="contiguous"):
        build_layer(kind, stages=("m3", "m5"), factories=facts)
    with pytest.raises(KeyError, match="no emitter"):
        build_layer(kind, stages=("m3",), factories={})


def test_gdn_b3_track_emitter_plugs_in_unchanged():
    """The gdn-b3 track's ``b3_stage_spec`` (its own module, unchanged) as
    the B3 stage of a framework delta layer."""
    try:
        from kernels.gfx950.qwen38_mono import stage_b3 as S
    except Exception as e:  # noqa: BLE001 - another track's module, may be mid-edit
        pytest.skip(f"stage_b3 not importable: {type(e).__name__}: {str(e)[:100]}")

    def b3(kind, shape, st):
        return S.b3_stage_spec(S.B3Shape(shape.rows, shape.seq, shape.group),
                               base=st.base, phase=st.lds_phase)  # fmt: skip

    for shape in (LayerShape(), LayerShape(rows=4, seq=True)):
        k = build_layer("delta", shape, stages=("b3",), factories={"b3": b3})
        assert "b3" in k.name
        _ll(k)


def test_framework_rejects_order_mismatch_and_bad_stack():
    stages = (Stage("s0", 8), Stage("s1", 8, deps=("s0",)))
    t = TaskTable(stages)
    specs = [StageSpec(s, lambda b, c, p, tk: None) for s in reversed(stages)]
    with pytest.raises(ValueError, match="table order"):
        build_layer_kernel("x", specs, t, ScratchLayout.of({}))
    specs = [StageSpec(s, lambda b, c, p, tk: None) for s in stages]
    with pytest.raises(ValueError, match="parity_slots"):
        build_layer_kernel("x", specs, t, ScratchLayout.of({}), layers=2)


def test_host_models_agree():
    """toy_reference (full state) and the bench's closed-form checksum model
    give the same layer outputs."""
    from benchmarks.gfx950.qwen38_mono import toy_chain_bench as B

    spec = T.ToyChainSpec(depth=3, work=1, layers=2)
    rng = np.random.default_rng(3)
    w = rng.integers(-1000, 1000, size=(2, 3, 160, 512), dtype=np.int32)
    x = rng.integers(-1000, 1000, size=(512,), dtype=np.int32)
    bufs = B.ToyBufs(spec, 0, 0, 0, [0, 0], [0], 0, w, x, 99)
    step = (1 << 26) - 1
    ref = T.toy_reference(spec, 99, x, w, [step * 64 + 1, step * 64 + 2])
    cs = B.checksum_model(bufs, step, 1, [0])
    assert cs[(0, 0)] == B.wrap32(ref["out"][0]) and cs[(0, 1)] == B.wrap32(ref["out"][1])


# ------------------------------------------------------------------ GPU


def _gpu():
    from rocke.examples.gfx950.persistent_mailbox import harness as H

    prob = H.env_problem()
    if prob:
        pytest.skip(prob)
    from rocke.runtime.hip_module import get_device_arch

    if (get_device_arch() or "").split(":")[0] != "gfx950":
        pytest.skip("needs gfx950")
    return H


@pytest.fixture(scope="module")
def dev():
    H = _gpu()
    d = H.Dev(watchdog_s=120)
    yield d
    d.close()


def _check_specs():
    from benchmarks.gfx950.qwen38_mono import toy_chain_bench as B

    return B.check_grid()


@pytest.mark.gpu
@pytest.mark.parametrize("spec", _check_specs(), ids=lambda s: s.name())
def test_gpu_toy_numerics(dev, spec):
    from benchmarks.gfx950.qwen38_mono import toy_chain_bench as B

    bufs = B.alloc(dev, spec, chain=2)
    art, used = B.compile_art(spec)
    assert used in ("python", "cpp")
    for step in (5, (1 << 26) - 1):
        B.run_chain_once(dev, bufs, art, step=step, layers0=[0, spec.layers])
        errs = B.check_state(dev, bufs, step, [0, spec.layers])
        assert not errs, errs[:5]
    if spec.knobs.stamps:
        st = B.read_stamps(dev, bufs, spec.layers)
        d = B.decode_stamps(st, T.toy_table(spec).order, T.toy_table(spec))
        assert d and d["total"] > 0
        ends = [d["stages"][n]["end"] for n in T.toy_table(spec).order]
        assert ends == sorted(ends)  # stamps monotone along the chain


@pytest.mark.gpu
@pytest.mark.parametrize("which", [0, 1, 2, 3, 4, 5, 6])
def test_gpu_toy_graph_replays(dev, which):
    from benchmarks.gfx950.qwen38_mono import toy_chain_bench as B

    spec = [
        T.ToyChainSpec(depth=4, work=8),
        T.ToyChainSpec(depth=4, hop="last_arriver"),
        T.ToyChainSpec(depth=3, work=8, layers=2),
        T.best_spec(4, 8, layers=2),  # the T9 gate's current best
        T.best_spec(4, 0),
        T.best_spec(8, 8),  # single-layer deep chain (prefetch + early entry)
        # idle-CTA prefetch x 2-layer launch (the LDS-parity regression)
        replace_spec(T.best_spec(8, 8, layers=2), prefetch="l2_touch", consume="waves"),
    ][which]
    n = int(os.environ.get("Q38_REPLAYS", "10000"))
    r = B.replay_check(dev, spec, n)
    assert r["ok"], r["errors"]


@pytest.mark.gpu
@pytest.mark.parametrize("best", [False, True])
def test_gpu_stacked_stamps_complete(dev, best):
    """Stacked stamped launches: every CTA's points of every layer survive
    (the barrier between a layer's stamp flush and the next layer's begin),
    and :func:`decode_launch` decodes every layer (stamps monotone along the
    chain, later layers from the previous layer's end)."""
    from benchmarks.gfx950.qwen38_mono import toy_chain_bench as B

    spec = T.best_spec(3, 8, layers=2) if best else T.ToyChainSpec(depth=3, work=8, layers=2)
    spec = replace_spec(spec, knobs=replace_spec(spec.knobs, stamps=True))
    table = T.toy_table(replace_spec(spec, layers=1))
    bufs = B.alloc(dev, spec, chain=2)
    art, _ = B.compile_art(spec)
    for step in (5, 6, 7):
        B.run_chain_once(dev, bufs, art, step=step, layers0=[0, 2])
        assert not B.check_state(dev, bufs, step, [0, 2])
        for k in range(2):
            sts = [B.read_stamps(dev, bufs, 2 * k + li) for li in range(2)]
            for li, st in enumerate(sts):
                for i in range(len(table.order)):
                    e, x = st[:, stp.enter_ix(i)], st[:, stp.exit_ix(i)]
                    assert (e > 0).all() and (x >= e).all(), (step, k, li, i)
                assert (st[:, 0] > 0).all()
            d0, d1 = B.decode_launch(sts, table.order, table)
            assert d1["t0"] == d0["t_end"] and d1["total"] > 0 and d1["ramp"] is None
            for d in (d0, d1):
                ends = [d["stages"][n]["end"] for n in table.order]
                assert ends == sorted(ends)


def _objdump():
    for c in (os.environ.get("LLVM_OBJDUMP"),
              os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "llvm/bin/llvm-objdump"),
              shutil.which("llvm-objdump")):  # fmt: skip
        if c and os.path.isfile(c):
            return c
    return None


@pytest.mark.gpu
def test_isa_phase_barrier_does_not_drain_vmcnt():
    """Stage s0 issues weight loads that s1 (another LDS phase) consumes:
    the framework's phase barrier sits between them and must not wait
    ``vmcnt(0)``."""
    _gpu()
    objdump = _objdump()
    if objdump is None:
        pytest.skip("llvm-objdump not found")
    from rocke.helpers.compile import compile_kernel

    keep = {}

    def entry(b, ctx, p):
        # weight loads at the kernel entry (they dominate every stage)
        wp = p.ptr(b, "w_down")
        keep["w"] = [b.global_load_i32(wp, b.add(ctx.tid, b.const_i32(512 * j))) for j in range(4)]

    def s0(b, ctx, p, t):
        ctx.lds.st_i32(b, "e0", ctx.tid, t)

    def s1(b, ctx, p, t):
        acc = ctx.lds.ld_i32(b, "e1", ctx.tid)
        for v in keep["w"]:
            acc = b.add(acc, v)
        b.global_store(p.ptr(b, "h_out"), ctx.tid, acc)

    stages = (Stage("s0", 256, lds_phase="a"), Stage("s1", 256, lds_phase="b"))
    specs = [StageSpec(stages[0], s0, lds=(LdsEntry("e0", 512),)),
             StageSpec(stages[1], s1, lds=(LdsEntry("e1", 512),))]  # fmt: skip
    k = build_layer_kernel("fw_isa", specs, TaskTable(stages), ScratchLayout.of({}),
                           entry=entry)  # fmt: skip
    hsaco = compile_kernel(k, arch="gfx950", backend="python").hsaco
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "k.hsaco"
        p.write_bytes(hsaco)
        isa = subprocess.run([objdump, "-d", "--mcpu=gfx950", str(p)], check=True,
                             capture_output=True, text=True).stdout  # fmt: skip
    insns = [ln.split("//")[0].strip() for ln in isa.splitlines()]
    insns = [t for t in insns if t and not t.endswith(":") and not t.startswith(("Disassembly", ";"))]
    bars = [i for i, t in enumerate(insns) if t.split()[0] == "s_barrier"]
    assert bars
    for i in bars:
        prev = insns[i - 1]
        assert not (prev.startswith("s_waitcnt") and "vmcnt(0)" in prev), insns[i - 3:i + 1]

