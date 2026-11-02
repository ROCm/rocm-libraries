# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU contract of ``dispatch.gdn.gate_rules``, with no torch: the rule that
decides whether a GDN/KDA decode launch passes the fp32 gate (so a wrong, NaN
or write-spilling configuration is never ranked) and the arithmetic that sizes
a cold-memory rotation (so a "cold" time is not a warm one).

The tensor code that produces the errors and the byte counts is in
``builders/gfx950/gdn/gdn_decode.py`` and needs torch; its own tests
(``test_gdn_decode_timing.py``) skip where torch is missing. These tests run
everywhere, so a later edit to either rule cannot pass CI unnoticed. The last
tests pin that the driver delegates to this module instead of keeping a copy.
"""

from __future__ import annotations

import ast
import builtins
import dataclasses
import dis
import importlib.util
import inspect
import math
import types
from pathlib import Path

import pytest

from dispatch.gdn import gate_rules
from dispatch.gdn.gate_rules import (
    TOL,
    Gate,
    cold_copy_count,
    touched_bytes_of,
)

ERRORS = ("out_err", "state_err", "conv_err")


def _gate(spill=False, **errors):
    return Gate(**dict(dict.fromkeys(ERRORS, 0.0), **errors), spill=spill)


def _above(x):
    return math.nextafter(x, math.inf)


# ------------------------------------------------------------------ the gate
def test_a_launch_within_tolerance_with_no_spill_passes():
    assert _gate().passed
    assert _gate(out_err=TOL / 2, state_err=TOL / 3, conv_err=TOL / 4).passed
    assert _gate(out_err=TOL).passed  # the bound is inclusive


@pytest.mark.parametrize("slot", ERRORS)
def test_an_error_over_tolerance_fails_in_any_slot(slot):
    verdict = _gate(**{slot: _above(TOL)})
    assert not verdict.passed and not verdict.spill
    assert verdict.worst == getattr(verdict, slot)


@pytest.mark.parametrize("slot", ERRORS)
def test_a_nan_error_fails_however_the_other_errors_look(slot):
    """A NaN cannot hide behind a finite error: not behind zeros, not behind a
    larger finite error, wherever it sits in the order ``worst`` reduces."""
    for others in (0.0, TOL / 2, 1.0, 1e30):
        errors = dict.fromkeys(ERRORS, others)
        verdict = _gate(**dict(errors, **{slot: math.nan}))
        assert math.isnan(verdict.worst), (slot, others)
        assert not verdict.passed, (slot, others)


@pytest.mark.parametrize("slot", ERRORS)
def test_an_infinite_error_fails_in_any_slot(slot):
    verdict = _gate(**{slot: math.inf})
    assert verdict.worst == math.inf and not verdict.passed


@pytest.mark.parametrize("errors", [{}, {"out_err": TOL}, {"state_err": 1.0}])
def test_a_spill_fails_even_when_every_error_is_zero_or_large(errors):
    """A write into a slot the kernel was not told to write is a failure by
    itself: the errors only score the slots it was told to write."""
    verdict = _gate(spill=True, **errors)
    assert not verdict.passed and verdict.spill


def test_a_spill_below_every_tolerance_still_fails():
    verdict = _gate(spill=True)
    assert verdict.worst == 0.0 <= TOL and not verdict.passed


def test_worst_is_the_largest_error_and_names_its_slot():
    assert _gate(out_err=0.1, state_err=0.5, conv_err=0.2).worst == 0.5
    assert _gate(out_err=0.1, state_err=0.05, conv_err=0.2).worst == 0.2
    assert _gate().worst == 0.0


def test_the_tolerance_is_not_loosened_past_a_real_bug():
    """``TOL`` is sized against bf16 rounding noise; far above it, an indexing
    or reduction bug would be ranked as correct."""
    assert 0.0 < TOL <= 1e-2


def test_a_verdict_cannot_be_edited_after_it_is_scored():
    with pytest.raises(dataclasses.FrozenInstanceError):
        _gate().spill = True


# ------------------------------------------------------- the rotation sizing
def _must_not_measure():
    raise AssertionError("touched bytes were measured when they were not needed")


def test_warm_needs_one_set_and_never_measures_the_launch():
    assert cold_copy_count(0, _must_not_measure) == 1


@pytest.mark.parametrize("bad", [-1, -1024])
def test_a_negative_rotation_is_refused_before_anything_is_measured(bad):
    with pytest.raises(ValueError, match="rotate_mb"):
        cold_copy_count(bad, _must_not_measure)


def test_a_cold_rotation_needs_at_least_two_sets():
    assert cold_copy_count(1, lambda: 2**40) == 2  # one set already exceeds it
    assert cold_copy_count(1, lambda: 2**20) == 2  # exactly one set: still two


@pytest.mark.parametrize(
    "touched", [1, 4096, 2**20 - 1, 2**20, 2**20 + 1, 1056768, 7 * 2**20 + 13, 2**30]
)
@pytest.mark.parametrize("rotate_mb", [1, 16, 1024, 4096])
def test_one_cycle_reaches_the_target_and_not_grossly_more(touched, rotate_mb):
    """The claims the timing label makes: one cycle of all ``k`` sets touches
    at least ``rotate_mb`` MiB, a set is reused after at least that less one
    set, and ``k`` is the least such count (2 is the floor)."""
    target = rotate_mb * 2**20
    copies = cold_copy_count(rotate_mb, lambda: touched)
    assert copies >= 2
    assert copies * touched >= target
    assert (copies - 1) * touched >= target - touched
    assert copies == 2 or (copies - 1) * touched < target
    assert copies == max(2, -(-target // touched))  # integer ceiling


def test_the_launch_is_measured_once_per_count():
    calls = []
    cold_copy_count(8, lambda: calls.append(1) or 2**20)
    assert calls == [1]


# --------------------------------------------------- the touched-byte count
def test_touched_bytes_counts_the_state_slots_and_the_output_only():
    assert touched_bytes_of(3, 1000, 50) == 3 * 1000 + 50
    assert touched_bytes_of(3, 1000, 50, conv_slot_bytes=200) == 3 * (1000 + 200) + 50
    assert touched_bytes_of(1, 7, 0) == 7


def test_a_shipped_shape_gives_the_copy_count_the_label_prints():
    """Hv=32, Dk=Dv=128, bf16 state, one sequence updated in place: a 1 MiB
    state slot plus 8 KiB of output per launch."""
    state_slot = 32 * 128 * 128 * 2
    out = 1 * 32 * 128 * 2
    touched = touched_bytes_of(1, state_slot, out)
    assert touched == 2**20 + 8192
    assert cold_copy_count(1024, lambda: touched) == 1017
    # Disjoint read and write slots: two distinct slots per sequence.
    assert touched_bytes_of(2, state_slot, out) == 2 * 2**20 + 8192


# ------------------------------------------------------------ layering / wiring
def test_the_module_needs_no_torch_and_no_upper_layer():
    tree = ast.parse(inspect.getsource(gate_rules))
    absolute = set()
    relative = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            absolute.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level:
            relative.add((node.level, node.module))
        elif isinstance(node, ast.ImportFrom):
            absolute.add((node.module or "").split(".")[0])
    assert absolute <= {"__future__", "dataclasses", "math", "typing"}, absolute
    assert relative == {(1, "cli_args")}, relative


def _driver_tree():
    spec = importlib.util.find_spec("builders.gfx950.gdn.gdn_decode")
    assert spec is not None and spec.origin
    return ast.parse(Path(spec.origin).read_text())


def _function(tree, name):
    (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    return fn


def _called_names(node):
    return {
        n.func.id if isinstance(n.func, ast.Name) else n.func.attr
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))
    }


def test_the_driver_takes_the_rules_from_here_and_keeps_no_copy():
    """The driver (which needs torch, so this reads its source rather than
    importing it) imports the pass test and the rotation arithmetic instead of
    defining them, and its tensor code only feeds them."""
    tree = _driver_tree()
    imported = {
        alias.name
        for n in tree.body
        if isinstance(n, ast.ImportFrom) and n.module == "dispatch.gdn.gate_rules"
        for alias in n.names
    }
    assert imported == {"TOL", "Gate", "cold_copy_count", "touched_bytes_of"}
    defined = {
        t.id
        for n in tree.body
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    } | {n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    assert not defined & {"TOL", "Gate"}, "the driver redefines the gate's rule"

    (verdict,) = [
        n
        for n in ast.walk(_function(tree, "gate"))
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "Gate"
    ]
    # The verdict's fields are the computed errors and spill, never constants
    # (the tensor reductions that compute them are tested where torch is).
    assert not verdict.keywords
    assert [getattr(a, "id", None) for a in verdict.args] == [
        "out_err",
        "state_err",
        "conv_err",
        "spill",
    ]
    copies = _function(tree, "cold_copies")
    assert "cold_copy_count" in _called_names(copies)
    assert not _called_names(copies) & {"ceil", "max", "isfinite"}
    touched = _function(tree, "touched_bytes")
    assert "touched_bytes_of" in _called_names(touched)
    for name in ("gate", "touched_bytes", "cold_copies"):
        divisions = [
            n
            for n in ast.walk(_function(tree, name))
            if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Div, ast.FloorDiv))
        ]
        assert not divisions, f"{name} does its own division"


def _module_bindings(tree):
    """Names bound at module level: imports, assignments, defs and classes
    (not the locals of a function or class body)."""
    names = set()

    def visit(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(child.name)
                continue
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                names.update((a.asname or a.name).split(".")[0] for a in child.names)
            elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                names.add(child.id)
            visit(child)

    visit(tree)
    return names


def _name_loads(code, module_level=True):
    """Global names read by ``code`` and the functions nested in it, and, for
    the module's own code, the names it reads while executing (``LOAD_NAME``).
    The body of a class is not looked into."""
    ops = {"LOAD_GLOBAL", "LOAD_NAME"} if module_level else {"LOAD_GLOBAL"}
    loads = {ins.argval for ins in dis.get_instructions(code) if ins.opname in ops}
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            loads |= _name_loads(const, module_level=False)
    return loads


def test_the_driver_uses_no_name_it_neither_defines_nor_imports():
    """Moving the rules out of the driver leaves its tensor code calling what
    it now imports. Without torch the driver cannot be imported here, so a
    dropped import would only fail where torch is installed; this finds it
    from the source (what pyflakes calls an undefined name). It does not see a
    misspelt attribute, a name bound only on some path, or a name read in a
    class body."""
    spec = importlib.util.find_spec("builders.gfx950.gdn.gdn_decode")
    source = Path(spec.origin).read_text()
    allowed = _module_bindings(ast.parse(source)) | set(dir(builtins))
    allowed.add("__annotations__")  # read by a module-level annotated assignment
    undefined = _name_loads(compile(source, spec.origin, "exec")) - allowed
    assert not undefined, sorted(undefined)
