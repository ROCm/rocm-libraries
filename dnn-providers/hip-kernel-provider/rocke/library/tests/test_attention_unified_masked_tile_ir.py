# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Running-max carry and keyless guards of unified attention, on the IR (no GPU).

The GPU tests in ``test_attention_unified_2d_masked_tile_numeric.py`` and
``test_attention_unified_scalar_masked_tile_numeric.py`` check the numbers.
These tests check the two properties behind them on the kernel IR the Python
builders emit, so they run in every lane, GPU or not:

* Running max. Every online-softmax loop starts its running max at -inf and
  must carry the raw ``fmax`` to the next iteration. The kernels replace a
  -inf max by 0 only as the exponent reference (a ``select`` on
  ``fcmp ogt max, -inf``); carrying that 0 pins the max at 0 after fully
  masked leading keys, and P underflows for very negative scores.
* Keyless guard. A row whose softmax sum ``l`` is exactly 0 saw no key and
  stores O = 0. The test must be ``fcmp oeq l, 0``: ``fcmp ogt l, 0`` is false
  for NaN too, so it turned a NaN row into a finite result. The reduce weights
  an empty segment (max = -inf) by ``exp2(-inf - ref)`` = 0, with one clamped
  exponent reference ``ref`` instead of a per-segment test.

Covered: every tiled 2D epilogue branch on gfx942 and gfx950 at KV page sizes
16 and 64, the scalar 2D kernel under the default mask and under windowed
masks, the scalar 3D segment kernel under the default and a top-left windowed
mask, and the scalar reduce.

The checks follow the IR's def-use edges (loop iteration arguments, yields,
compares, selects), not value names.
"""

import math
from functools import lru_cache

import pytest

from kernels.common.attention_unified import (
    UnifiedAttention2DSpec,
    UnifiedAttention3DSpec,
    UnifiedAttentionReduceSpec,
    build_unified_attention_2d,
    build_unified_attention_3d,
    build_unified_attention_reduce,
)

from .test_attention_unified_2d_masked_tile_numeric import (
    _BRANCHES,
    _branch_spec,
)
from .test_attention_unified_2d_masked_tile_numeric import _problem as _tiled_problem
from .test_attention_unified_scalar_masked_tile_numeric import _NUM_SEGMENTS
from .test_attention_unified_scalar_masked_tile_numeric import (
    _problem as _scalar_problem,
)

_TILED_ARCHS = ("gfx942", "gfx950")
_PAGE_SIZES = (16, 64)
# The scalar builders emit the same IR for every arch.
_SCALAR_ARCH = "gfx942"

# (window, top_left) per scalar mask. Bottom-right causal is the default mask,
# with or without a window.
_SCALAR_MASKS = {
    "bottom_right": (0, False),
    "bottom_right_window": (32, False),
    "top_left_window": (32, True),
}


def _walk(region):
    for op in region.ops:
        yield op
        for inner in op.regions:
            yield from _walk(inner)


def _is_const(value, expected):
    op = value.op
    if op is None or op.name != "arith.constant":
        return False
    got = op.attrs.get("value")
    return isinstance(got, float) and not math.isnan(got) and got == expected


def _users(kernel):
    """``id(value) -> ops that read it``, over the whole kernel."""
    users = {}
    for op in _walk(kernel.body):
        for value in op.operands:
            users.setdefault(id(value), []).append(op)
    return users


def _reads_running_max(value, loop, slots):
    """Whether ``value`` is the loop's iteration argument for one of ``slots``."""
    names = {loop.attrs["iter_args"][slot]["name"] for slot in slots}
    return value.op is loop and value.name in names


def _running_max_slots(kernel):
    """``(loop, slot, yielded value, the loop's -inf slots)`` for every
    loop-carried value that starts at -inf: the online-softmax running maxima."""
    slots = []
    for loop in _walk(kernel.body):
        if loop.name != "scf.for":
            continue
        inits = loop.operands[3:]
        maxima = [i for i, init in enumerate(inits) if _is_const(init, -math.inf)]
        if not maxima:
            continue
        yields = [op for op in loop.regions[0].ops if op.name == "scf.yield"]
        assert len(yields) == 1, "a loop with a -inf carry has no single yield"
        slots += [(loop, i, yields[0].operands[i], maxima) for i in maxima]
    return slots


def _check_running_max(kernel, *, exponent_reference):
    """Each running max is carried as ``fmax(carried max, ...)``; with
    ``exponent_reference``, that raw max also feeds the clamped exponent
    reference ``select(fcmp ogt max -inf, max, 0)``, which an ``fsub`` reads."""
    slots = _running_max_slots(kernel)
    assert slots, "no loop carries a running max that starts at -inf"
    users = _users(kernel) if exponent_reference else None
    for loop, slot, carried, maxima in slots:
        where = f"loop {loop.results[0].name} slot {slot}"
        producer = carried.op.name if carried.op is not None else "a loop argument"
        assert carried.op is not None and carried.op.name == "arith.fmax", (
            f"{where} carries {producer} {carried.name}, not the raw fmax: a "
            "clamped running max pins m at 0 after fully masked leading keys"
        )
        # Rows that share softmax state (the transposed 32x32 epilogue) yield
        # one max for every slot, so any of the loop's running maxima will do.
        assert any(
            _reads_running_max(v, loop, maxima) for v in carried.op.operands
        ), f"{where}: the carried fmax does not read the loop's running max"
        if not exponent_reference:
            continue
        references = [
            op
            for op in users.get(id(carried), [])
            if op.name == "arith.select"
            and op.operands[1] is carried
            and _is_const(op.operands[2], 0.0)
            and op.operands[0].op is not None
            and op.operands[0].op.name == "arith.fcmp"
            and op.operands[0].op.attrs.get("pred") == "ogt"
            and op.operands[0].op.operands[0] is carried
            and _is_const(op.operands[0].op.operands[1], -math.inf)
        ]
        assert references, (
            f"{where}: no select(fcmp ogt max, -inf; max; 0) exponent reference; "
            "exp2(-inf - -inf) is NaN for a fully masked prefix"
        )
        assert any(
            op.name == "arith.fsub"
            for ref in references
            for op in users.get(id(ref.results[0]), [])
        ), f"{where}: the clamped exponent reference is never subtracted"


def _check_keyless_guard(kernel):
    """Every compare on a softmax sum that O is divided by (an ``rcp``
    operand) is ``fcmp oeq sum, 0``, and one selects O = 0 for it."""
    sums = {id(op.operands[0]) for op in _walk(kernel.body) if op.name == "math.rcp"}
    assert sums, "no rcp of a softmax sum"
    guards = [
        op
        for op in _walk(kernel.body)
        if op.name == "arith.fcmp" and id(op.operands[0]) in sums
    ]
    assert guards, "no keyless-row guard on the softmax sum"
    for op in guards:
        pred = op.attrs.get("pred")
        assert pred == "oeq" and _is_const(op.operands[1], 0.0), (
            f"keyless guard is fcmp {pred} {op.operands[0].name}, "
            f"{op.operands[1].name}, not fcmp oeq sum, 0: a NaN sum must reach O"
        )
    _check_selects_zero(kernel, guards)


def _check_selects_zero(kernel, guards):
    users = _users(kernel)
    assert any(
        user.name == "arith.select"
        and user.operands[0] is guard.results[0]
        and _is_const(user.operands[1], 0.0)
        for guard in guards
        for user in users.get(id(guard.results[0]), [])
    ), "no select(guard, 0, x) stores 0 for an empty row or segment"


@lru_cache(maxsize=None)
def _tiled_kernel(arch, branch, bs):
    problem = _tiled_problem("fp16", 256, 256, 256 // bs + 3, bs, 128)
    return _branch_spec(problem, branch, bs, arch).build(arch)


@lru_cache(maxsize=None)
def _scalar_kernel(kind, mask):
    window, top_left = _SCALAR_MASKS[mask]
    problem = _scalar_problem("fp16", 128, 128, 128 // 16 + 3, top_left, window=window)
    assert problem.default_mask == (not top_left)
    if kind == "2d":
        return build_unified_attention_2d(
            UnifiedAttention2DSpec(problem), arch=_SCALAR_ARCH
        )
    if kind == "3d":
        spec = UnifiedAttention3DSpec(problem, num_segments=_NUM_SEGMENTS)
        return build_unified_attention_3d(spec, arch=_SCALAR_ARCH)
    spec = UnifiedAttentionReduceSpec(problem, num_segments=_NUM_SEGMENTS)
    return build_unified_attention_reduce(spec, arch=_SCALAR_ARCH)


@pytest.mark.parametrize("bs", _PAGE_SIZES)
@pytest.mark.parametrize("branch", _BRANCHES)
@pytest.mark.parametrize("arch", _TILED_ARCHS)
def test_tiled_2d_carries_raw_running_max(arch, branch, bs):
    _check_running_max(_tiled_kernel(arch, branch, bs), exponent_reference=True)


@pytest.mark.parametrize("bs", _PAGE_SIZES)
@pytest.mark.parametrize("branch", _BRANCHES)
@pytest.mark.parametrize("arch", _TILED_ARCHS)
def test_tiled_2d_keyless_guard_is_exact_zero(arch, branch, bs):
    _check_keyless_guard(_tiled_kernel(arch, branch, bs))


@pytest.mark.parametrize("mask", list(_SCALAR_MASKS))
@pytest.mark.parametrize("kind", ["2d", "3d"])
def test_scalar_carries_raw_running_max(kind, mask):
    _check_running_max(_scalar_kernel(kind, mask), exponent_reference=True)


@pytest.mark.parametrize("mask", list(_SCALAR_MASKS))
def test_scalar_2d_keyless_guard_is_exact_zero(mask):
    # Every mask: bottom-right causal with Sq > Skv leaves rows keyless too.
    _check_keyless_guard(_scalar_kernel("2d", mask))


def test_scalar_reduce_carries_raw_running_max():
    # The reduce's max pass reads the per-segment maxima; an empty segment
    # stores -inf, so the raw fmax ignores it and needs no exponent clamp.
    _check_running_max(
        _scalar_kernel("reduce", "top_left_window"), exponent_reference=False
    )


def test_scalar_reduce_guards_are_exact_zero():
    """The combine pass uses one exponent reference: the overall max, or 0 when
    every segment is empty (``select(fcmp ogt overall, -inf; overall; 0)``,
    once, outside the loops). An empty segment stores max = -inf, so its factor
    ``exp2(-inf - ref)`` is 0 with no per-segment test. The row stores O = 0
    only for an exact-zero total."""
    kernel = _scalar_kernel("reduce", "top_left_window")
    _check_keyless_guard(kernel)
    loops = [op for op in kernel.body.ops if op.name == "scf.for"]
    max_loops = [loop for loop in loops if _is_const(loop.operands[3], -math.inf)]
    assert len(loops) == 2 and len(max_loops) == 1, "expected a max and a combine pass"
    overall = max_loops[0].results[0]
    combine = next(loop for loop in loops if loop is not max_loops[0])
    in_loops = [
        op
        for loop in loops
        for op in _walk(loop.regions[0])
        if op.name in ("arith.fcmp", "arith.select")
    ]
    assert not in_loops, (
        f"the reduce loops test or select per segment ({in_loops[0].name}); "
        "one exponent reference outside the loops covers an empty segment"
    )
    references = [
        op
        for op in kernel.body.ops
        if op.name == "arith.select"
        and op.operands[1] is overall
        and _is_const(op.operands[2], 0.0)
        and op.operands[0].op is not None
        and op.operands[0].op.name == "arith.fcmp"
        and op.operands[0].op.attrs.get("pred") == "ogt"
        and op.operands[0].op.operands[0] is overall
        and _is_const(op.operands[0].op.operands[1], -math.inf)
    ]
    assert len(references) == 1, (
        "no single select(fcmp ogt overall, -inf; overall; 0) exponent reference "
        "outside the loops; exp2(-inf - -inf) is NaN when every segment is empty"
    )
    ref = references[0].results[0]
    assert any(
        op.name == "arith.fsub" and op.operands[1] is ref
        for op in _walk(combine.regions[0])
    ), "the combine pass does not subtract the exponent reference"
