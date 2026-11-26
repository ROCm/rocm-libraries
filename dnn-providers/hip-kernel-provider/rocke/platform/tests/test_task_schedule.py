# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``rocke.helpers.task_schedule`` (CPU, host model only).

Task tables of persistent kernels: placement maps (bijections, inverse maps),
validation (deadlock freedom, dataflow coverage, the cross-layer
write-after-read check with and without parity slots, heterogeneous stacks),
LDS phase barriers and placement policies.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from rocke.helpers.task_schedule import (
    Dep,
    Stage,
    TaskTable,
    ValidationError,
    ordered_after_all,
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


def test_affine_map_is_base_offset_and_bijective():
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
    assert len(set(xcd)) == 1


def test_sequential_placement_and_histogram():
    t = _chain().place("sequential", start=3)
    assert t.base == {"a": 3, "b": 17, "c": 97, "d": 1}
    h = t.histogram()
    assert sum(h) == 14 + 80 + 160 + 160
    assert t.cta_tasks(3)[0] == ("a", 0)
    assert _chain().place("given") == _chain()
    with pytest.raises(ValueError):
        _chain().place("nope")


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
    t = TaskTable(
        (Stage("a", 4, writes=("r1",)), Stage("b", 4, deps=("a",), reads=("r2",)))
    )
    with pytest.raises(ValidationError, match="r2"):
        t.validate()
    assert TaskTable(t.stages, inputs=("r2",)).validate() == []
    # an intra-stage hand-off (siblings / a final task) reads its own writes
    own = TaskTable((Stage("q", 9, reads=("fc",), writes=("fc", "sel")),))
    assert own.validate() == []


def test_table_and_stage_argument_checks():
    with pytest.raises(ValueError, match="power of two"):
        TaskTable((Stage("a", 4),), grid=96)
    with pytest.raises(ValueError, match="xcds"):
        TaskTable((Stage("a", 4),), grid=4, xcds=8)
    with pytest.raises(ValueError):
        Stage("not a name", 4)
    with pytest.raises(ValueError):
        Stage("a", 0)
    with pytest.raises(ValueError):
        Stage("a", 4, map="spiral")


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
    tp = _chain(stack=True, partial_head=True)  # a partial head edge overlaps layers
    errs = tp.validate(layers=2, strict=False)
    assert errs and all("cross-layer WAR" in e for e in errs)
    assert any("'rc'" in e and "layer 0 d" in e for e in errs)

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
