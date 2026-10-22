# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``rocke.helpers.task_schedule`` and ``rocke.helpers.lds_arena`` (CPU).

Task tables of persistent layer kernels: placement maps (bijections, the
device emitter agrees with the host model), validation (deadlock freedom,
dataflow coverage, the cross-layer write-after-read check with and without
parity slots, heterogeneous stacks), LDS phase barriers, placement policies;
the LDS arena's offsets, row padding, pinned entries, limits, and that both
emit only mirrored ops (no Python-only opcode).
"""

from __future__ import annotations

import pytest

from rocke.core.ir import I32, IRBuilder, PtrType, uses_python_only_ops
from rocke.helpers.lds_arena import (
    LDS_LIMIT_WORDS,
    ONE_CTA_PER_CU_WORDS,
    LdsArena,
    LdsEntry,
)
from rocke.helpers.task_schedule import (
    Dep,
    Stage,
    TaskTable,
    ValidationError,
    for_my_tasks,
    ordered_after_all,
    task_index,
    validate_stack,
)


def _chain(stack: bool = False, partial_head: bool = False) -> TaskTable:
    """a (14 tasks) -> b (80, full) -> c (160, full) -> d (160, covers c)."""
    head = ()
    if stack:
        head = (Dep("d", full=not partial_head, covers=False, layer=-1),)
    return TaskTable(
        (
            Stage("a", 14, deps=head, reads=("h",) + (("out",) if stack else ()),
                  writes=("ra",), lds_phase="x"),
            Stage("b", 80, base=176, deps=("a",), reads=("ra",), writes=("rb",),
                  lds_phase="x"),
            Stage("c", 160, base=192, deps=("b",), reads=("rb",), writes=("rc",),
                  lds_phase="y"),
            Stage("d", 160, deps=(Dep("c", full=False, covers=True),), reads=("rc",),
                  writes=("out",), lds_phase="y"),
        ),
        inputs=("h",) + (() if stack else ("out",)),
    )  # fmt: skip


def test_affine_map_is_atom_rule_and_bijective():
    t = _chain()
    for name in t.order:
        s = t.stage(name)
        ctas = [t.cta_of(name, k) for k in range(s.tasks)]
        assert len(set(ctas)) == s.tasks
        for k, c in enumerate(ctas):
            assert c == (s.base + k) % 256
            assert t.task_of(name, c) == k
    assert t.tasks == {"a": 14, "b": 80, "c": 160, "d": 160}
    assert t.base == {"a": 0, "b": 176, "c": 192, "d": 0}


def test_xcd_blocked_map_inverse_and_xcd_grouping():
    t = _chain().place("xcd_blocked")
    s = t.stage("c")
    for k in range(s.tasks):
        assert t.task_of("c", t.cta_of("c", k)) == k
    # consecutive tasks share an XCD (round-robin dispatch: cta % 8)
    xcd = [t.cta_of("c", k) % 8 for k in range(32)]
    assert xcd == [(s.base % 8)] * 32 or len(set(xcd)) == 1


def test_sequential_placement_and_histogram():
    t = _chain().place("sequential", start=3)
    assert t.base == {"a": 3, "b": 17, "c": 97, "d": 1}
    h = t.histogram()
    assert sum(h) == 14 + 80 + 160 + 160
    assert t.cta_tasks(3)[0] == ("a", 0)


def test_rounds_beyond_grid():
    t = TaskTable((Stage("m4", 300),))
    assert t.rounds("m4") == 2
    assert t.task_of("m4", 10, 1) == 266
    assert t.task_of("m4", 100, 1) is None
    assert sum(t.histogram()) == 300


def test_validate_ok_and_lds_barriers():
    t = _chain()
    assert t.validate() == []
    assert t.lds_barriers() == (2,)


def test_validate_rejects_forward_dependency_and_unknown():
    bad = TaskTable((Stage("a", 4, deps=("b",)), Stage("b", 4)))
    with pytest.raises(ValidationError, match="not earlier"):
        bad.validate()
    with pytest.raises(ValidationError, match="unknown dependency"):
        TaskTable((Stage("a", 4, deps=("zz",)),)).validate()
    with pytest.raises(ValidationError, match="duplicate"):
        TaskTable((Stage("a", 4), Stage("a", 4))).validate()


def test_validate_rejects_unwritten_read():
    t = TaskTable((Stage("a", 4, writes=("r1",)), Stage("b", 4, deps=("a",), reads=("r2",))))
    with pytest.raises(ValidationError, match="r2"):
        t.validate()
    assert TaskTable(t.stages, inputs=("r2",)).validate() == []
    # an intra-stage hand-off (siblings / a final task) reads its own writes
    own = TaskTable((Stage("q4", 9, reads=("fc",), writes=("fc", "sel")),))
    assert own.validate() == []


def test_ordered_after_all_semantics():
    t = _chain(stack=True)
    tabs = [t, t]
    # d covers c, then nothing full after d inside the layer: c is ordered
    # before the next layer's a only through the full previous-layer edge
    after_c = ordered_after_all(tabs, (0, "c"))
    assert (1, "a") in after_c and (1, "d") in after_c
    after_a = ordered_after_all(tabs, (0, "a"))
    assert (0, "b") in after_a and (1, "a") in after_a
    # a partial (non-covering) edge orders nothing for all producer tasks
    tp = _chain(stack=True, partial_head=True)
    after_d = ordered_after_all([tp, tp], (0, "d"))
    assert (1, "a") not in after_d


def test_cross_layer_war_detected_and_fixed_by_parity():
    t = _chain(stack=True)
    assert t.validate(layers=3) == []  # full head edge: implicit barrier
    tp = _chain(stack=True, partial_head=True)  # an "overlap lever"
    errs = tp.validate(layers=2, strict=False)
    assert errs and all("cross-layer WAR" in e for e in errs)
    assert any("'rc'" in e and "layer 0 d" in e for e in errs)
    from dataclasses import replace

    fixed = replace(tp, parity=("ra", "rb", "rc", "out"))
    assert fixed.validate(layers=2) == []
    # parity only separates adjacent layers: layer 2 reuses layer 0's slot
    errs3 = fixed.validate(layers=3, strict=False)
    assert errs3 and all("layer 2" in e and "layer 0" in e for e in errs3)


def test_validate_stack_heterogeneous_and_prev_stage_names():
    t = _chain(stack=True)
    other = TaskTable(
        (
            Stage("p", 8, deps=(Dep("d", layer=-1),), reads=("out",), writes=("rp",)),
            Stage("q", 8, deps=("p",), reads=("rp",), writes=("out",)),
        ),
    )
    assert validate_stack([t, other]) == []
    bad = TaskTable((Stage("p", 8, deps=(Dep("zz", layer=-1),)),))
    with pytest.raises(ValidationError, match="previous-layer stage"):
        validate_stack([t, bad])


def test_dep_flags():
    assert Dep("x").covers  # full implies covers
    assert not Dep("x", full=False).covers
    with pytest.raises(ValueError):
        Dep("x", layer=1)


@pytest.mark.parametrize("policy", ["atom", "sequential", "xcd_blocked"])
def test_device_task_index_matches_host_model(policy):
    """Evaluate the emitted index math symbolically over every CTA."""
    t = _chain().place(policy)
    for name in t.order:
        b = IRBuilder("ti")
        bid = b.param("bid", I32)
        out = task_index(b, t, name, bid)
        ops = list(b.kernel.body.ops)
        for cta in range(256):
            env = {bid.name: cta}
            for op in ops:
                vals = [env.get(v.name, None) for v in op.operands]
                n = op.name
                if n == "arith.constant":
                    env[op.result.name] = op.attrs["value"]
                elif n == "arith.add":
                    env[op.result.name] = vals[0] + vals[1]
                elif n == "arith.and":
                    env[op.result.name] = vals[0] & vals[1]
                elif n == "arith.mul":
                    env[op.result.name] = vals[0] * vals[1]
                elif n == "arith.lshr":
                    env[op.result.name] = vals[0] >> vals[1]
                else:
                    raise AssertionError(f"unexpected op {n}")
            got = env[out.name]
            want = t.task_of(name, cta)
            assert (got if got < t.stage(name).tasks else None) == want, (policy, name, cta)


def test_for_my_tasks_emits_mirrored_ops_only():
    t = TaskTable((Stage("m4", 300),))
    b = IRBuilder("fmt")
    b.kernel.attrs["max_workgroup_size"] = 64
    out = b.param("out", PtrType(I32, "global"))
    bid = b.readfirstlane(b.block_id_x())
    for_my_tasks(b, t, "m4", bid, lambda tk: b.global_store(out, tk, tk))
    b.ret()
    assert not uses_python_only_ops(b.kernel)
    names = [op.name for op in b.kernel.body.ops]
    assert names.count("scf.if") == 2  # two rounds


# ------------------------------------------------------------------ arena


def test_arena_offsets_overlay_and_row_padding():
    a = LdsArena(
        {"attn": [("x8", 640), ("red", 128, True)], "moe": [("rw", 1600)]},
        rows=4,
        pinned=[("tl", 320)],
    )
    assert a.offset["x8"] == 0 and a.offset["red"] == 640
    st = a.stride["red"]
    assert st >= 128 and st % 64 == 16
    assert a.offset["rw"] == 0  # phases overlay
    top = max(640 + 4 * st, 1600)
    assert a.offset["tl"] == top
    assert a.words == (top + 320 + 3) // 4 * 4
    assert a.at("red", 2) == 640 + 2 * st
    assert a.phase_of["tl"] is None and a.phase_of["rw"] == "moe"


def test_arena_matches_atom_place_rule_rows1():
    # ATOM layout._place: (words + 3) // 4 * 4, no row padding at rows 1
    a = LdsArena({"p": [("a", 5), ("b", 7, True), ("c", 1)]})
    assert (a.offset["a"], a.offset["b"], a.offset["c"]) == (0, 8, 16)
    assert a.used_words == 20


def test_arena_limits_and_one_cta_per_cu():
    with pytest.raises(ValueError, match="exceeds"):
        LdsArena({"p": [("a", LDS_LIMIT_WORDS + 4)]})
    a = LdsArena({"p": [("a", 8)]}, min_words=ONE_CTA_PER_CU_WORDS)
    assert a.one_cta_per_cu() and a.words >= ONE_CTA_PER_CU_WORDS
    assert not LdsArena({"p": [("a", 8)]}).one_cta_per_cu()
    with pytest.raises(ValueError, match="duplicate"):
        LdsArena({"p": [("a", 8)], "q": [("a", 8)]})
    with pytest.raises(ValueError):
        LdsEntry("z", 0)


def test_arena_view_emits_one_alloc_mirrored_ops():
    a = LdsArena({"p": [("a", 16), ("b", 16, True)]}, rows=2)
    b = IRBuilder("lv")
    b.kernel.attrs["max_workgroup_size"] = 64
    v = a.emit(b)
    tid = b.thread_id_x()
    v.st_i32(b, "a", tid, tid)
    v.st_f32(b, "b", 3, b.const_f32(1.0), row=1)
    x = v.ld_i32(b, "a", tid)
    y = v.ld_f32(b, "b", 3, row=b.const_i32(1))
    _ = (x, y)
    b.ret()
    allocs = [op for op in b.kernel.body.ops if op.name == "tile.smem_alloc"]
    assert len(allocs) == 1
    assert not uses_python_only_ops(b.kernel)
