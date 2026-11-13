# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Structural assertions shared by counted-sync unit and exhaustive gates."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from rocke.core.ir import KernelDef, Op

from kernels.common._attention_bwd_body import BwdTileGeometry
from kernels.common.attention_bwd import AttnBwdSpec, validate_attn_bwd_spec

from ._attention_bwd_counted_model import counted_ir_problems, expected_waits, ir_events

__all__ = ["assert_counted_ir_model", "predecessors", "q_loops", "walk_ops"]


def walk_ops(
    ops: Sequence[Op], ancestors: tuple[Op, ...] = ()
) -> Iterator[tuple[Op, tuple[Op, ...]]]:
    """Yield every operation and its ancestor tuple in program order."""
    for op in ops:
        yield op, ancestors
        for region in op.regions:
            yield from walk_ops(region.ops, ancestors + (op,))


def _holds(op: Op, name: str) -> bool:
    return any(child.name == name for child, _ in walk_ops(op.regions[0].ops))


def q_loops(kernel: KernelDef) -> list[Op]:
    """Return the innermost matrix loops of a counted backward kernel."""
    matrix_loops = [
        op
        for op, _ in walk_ops(kernel.body.ops)
        if op.name == "scf.for" and _holds(op, "tile.mma")
    ]
    matrix_ids = {id(loop) for loop in matrix_loops}
    return [
        loop
        for loop in matrix_loops
        if not any(id(op) in matrix_ids for op, _ in walk_ops(loop.regions[0].ops))
    ]


def predecessors(kernel: KernelDef) -> dict[int, list[Op]]:
    """Map each operation identity to earlier operations in its region."""
    result = {}

    def visit(ops: Sequence[Op]) -> None:
        for index, op in enumerate(ops):
            result[id(op)] = ops[:index]
            for region in op.regions:
                visit(region.ops)

    visit(kernel.body.ops)
    return result


def assert_counted_ir_model(
    kernel: KernelDef, spec: AttnBwdSpec, arch: str, issue: str
) -> None:
    """Assert counted waits, load ordering, and loop-carried prefetch structure."""
    resolved = validate_attn_bwd_spec(spec, arch)
    atomics = BwdTileGeometry.from_spec(resolved, arch).counted_atomics
    loops = q_loops(kernel)
    assert loops
    previous = predecessors(kernel)
    for loop in loops:
        problems, waits = counted_ir_problems(loop)
        assert not problems, problems
        assert waits == expected_waits(atomics, issue), (waits, atomics, issue)

        before = previous[id(loop)]
        assert before[-1].name == "tile.s_waitcnt", [op.name for op in before[-3:]]
        assert before[-1].attrs["vmcnt"] == 0
        assert before[-2].name == "scf.for"

        names = [metadata["name"] for metadata in loop.attrs["iter_args"]]
        prefetch = [
            index for index, name in enumerate(names) if "pf" in name.rsplit("_", 1)[-1]
        ]
        assert prefetch, names
        initial = loop.operands[3:]
        sources = {
            initial[index].op.name if initial[index].op is not None else None
            for index in prefetch
        }
        assert sources == {"scf.for"}, sources
        guard = initial[prefetch[0]].op
        assert any(kind == "L" for kind, _ in ir_events(guard.regions[0].ops))
        assert guard.operands[1].op.name == "arith.select"
