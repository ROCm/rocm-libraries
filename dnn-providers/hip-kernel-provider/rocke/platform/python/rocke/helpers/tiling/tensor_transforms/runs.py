# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Run queries: whether a vector access stays wide.

A *run* (``VectorRun(axis, width)``) is one access standing for ``width`` consecutive values of
graph input ``axis``. Three questions, answered at build time from the access's first-element
coordinates (split by the IR range walk; read only, no IR):

- ``address_contiguous_run`` -- the run's elements are consecutive in memory, so one wide
  load / store addresses them all;
- ``validity_uniform_over_run`` -- every bound check has one answer for the whole run, so the
  first element's checks decide it;
- ``checks_monotone_over_run`` -- every checked coordinate moves by a constant step along the
  run, so the first and the last element decide every check, and the valid elements form one
  contiguous span.

Each coordinate carries ``(step along the run, start residue)``. The vector's input moves by 1,
every other input by 0; each step of the graph maps them on, by kind:

- ``embed`` / ``merge``: the weighted sum of the read steps; the residues scaled and added.
- ``offset`` / ``slice_window``: the step unchanged; the residue shifted.
- ``freeze``: step 0, the frozen value. ``broadcast`` writes nothing.
- ``unmerge``: a read of step 0 gives every piece step 0. A read of step 1 keeps the run inside
  the inner piece only when it starts at a multiple of ``width`` and ``width`` divides the inner
  length: the inner piece then has step 1 and the others step 0. Anything else varies.
- ``xor``: its second write has step 1 only when the first read has step 0 and a residue that is
  a multiple of ``width`` (so the xor changes no bit inside the run), the second read has step 1
  and starts at a multiple of ``width``, and ``width`` divides the xor's length. Otherwise it
  varies (``reads[0] = 1``, width 2 maps 0, 1 to 1, 0: same block, not ascending).

A coordinate's start residue is the one its split coordinate states (``AffineCoord.congruence``:
the constant, every per-wave part and the per-lane part combined). A coordinate that is a plain
constant is congruent to itself modulo every modulus the graph divides by.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

from .graph import TransformGraph, TransformType
from .ranges import AffineCoord, BoundCheck, Congruence, RuntimeValue, VectorRun

if TYPE_CHECKING:
    from ..descriptors import TensorDesc

__all__ = [
    "address_contiguous_run",
    "checks_monotone_over_run",
    "validity_uniform_over_run",
]


@dataclass(frozen=True)
class _RunStep:
    """How one coordinate moves along a run: by ``step`` per element (``None`` when that is not
    one constant), starting at ``congruence``. A coordinate that varies names the coordinate
    where the variation started and why, for the vector errors."""

    step: int | None
    congruence: Congruence
    origin: str | None = None
    reason: str | None = None


# --- the three queries --------------------------------------------------------------------------


def address_contiguous_run(
    graph: TransformGraph,
    tensor: TensorDesc,
    run: VectorRun,
    coords: tuple[AffineCoord, ...],
) -> bool:
    """The run's elements are consecutive in memory (the element step is exactly 1)."""
    steps = _run_steps(graph, run, coords)
    return _contiguity_blocker(graph, tensor, run, steps) is None


def validity_uniform_over_run(
    graph: TransformGraph,
    checks: tuple[BoundCheck, ...],
    run: VectorRun,
    coords: tuple[AffineCoord, ...],
) -> bool:
    """Every check in ``checks`` has one answer for every element of the run."""
    steps = _run_steps(graph, run, coords)
    return all(
        _check_uniform(check, steps[check.coordinate], run.width) for check in checks
    )


def checks_monotone_over_run(
    graph: TransformGraph,
    checks: tuple[BoundCheck, ...],
    run: VectorRun,
    coords: tuple[AffineCoord, ...],
) -> bool:
    """Every coordinate in ``checks`` moves by a constant step along the run."""
    return _first_non_monotone(checks, _run_steps(graph, run, coords)) is None


# --- shared by the queries and the access entries -----------------------------------------------


def _run_steps(
    graph: TransformGraph, run: VectorRun, coords: tuple[AffineCoord, ...]
) -> dict[str, _RunStep]:
    """Every named coordinate's step along the run and start residue."""
    exact = _exact_modulus(graph, run.width)
    steps: dict[str, _RunStep] = {}
    for i, (name, coord) in enumerate(zip(graph.inputs, coords)):
        steps[name] = _RunStep(1 if i == run.axis else 0, _start(coord, exact))
    for node in graph.nodes:
        reads = [steps[r] for r in node.read]
        kind, params = node.kind, node.params
        if kind is TransformType.EMBED:
            steps[node.write[0]] = _linear(reads, params.coeffs, exact)
        elif kind is TransformType.MERGE:
            weights = [
                prod(params.lengths[i + 1 :]) for i in range(len(params.lengths))
            ]
            steps[node.write[0]] = _linear(reads, weights, exact)
        elif kind is TransformType.UNMERGE:
            steps.update(
                _unmerge(reads[0], node.read[0], node.write, params.lengths, run)
            )
        elif kind in (TransformType.OFFSET, TransformType.SLICE):
            shift = params.shift if kind is TransformType.OFFSET else params.begin
            read = reads[0]
            steps[node.write[0]] = _RunStep(
                read.step, read.congruence.shifted(shift), read.origin, read.reason
            )
        elif kind is TransformType.FREEZE:
            steps[node.write[0]] = _RunStep(0, Congruence(exact, params.value))
        elif kind is TransformType.XOR:
            length = graph.edge(node.write[1]).domain
            steps[node.write[0]] = reads[0]
            steps[node.write[1]] = _xor(reads, node.write[1], length, run.width)
    return steps


def _contiguity_blocker(
    graph: TransformGraph,
    tensor: TensorDesc,
    run: VectorRun,
    steps: dict[str, _RunStep],
) -> tuple[str, str] | None:
    """``None`` when the run (its coordinates' ``steps``) is contiguous in memory; else the
    coordinate that breaks it and why, in plain words."""
    outputs = [steps[name] for name in graph.outputs]
    for step in outputs:
        if step.step is None:
            return step.origin, step.reason
    total, first_moving = 0, None
    for name, step, stride in zip(graph.outputs, outputs, tensor.strides):
        if step.step == 0:
            continue
        if isinstance(stride, RuntimeValue):
            return name, "runtime stride"
        total += step.step * stride
        first_moving = first_moving or name
    if total == 1:
        return None
    blocker = first_moving or tuple(graph.inputs)[run.axis]
    return blocker, f"its elements are {total} apart in memory"


def _first_non_monotone(
    checks: tuple[BoundCheck, ...], steps: dict[str, _RunStep]
) -> str | None:
    """The first checked coordinate whose step along the run is not one constant."""
    for check in checks:
        if steps[check.coordinate].step is None:
            return check.coordinate
    return None


def _check_uniform(check: BoundCheck, step: _RunStep, width: int) -> bool:
    """One answer for the whole run: the coordinate does not move, or it moves by 1 from a
    start whose distance to every checked end is a multiple of ``width``."""
    if step.step == 0:
        return True
    if step.step != 1 or step.congruence.modulus % width:
        return False
    for end in (check.lo, check.hi):
        if end is None:
            continue
        residue = _end_residue(end, width)
        if residue is None or (step.congruence.residue - residue) % width:
            return False
    return True


# --- rules ------------------------------------------------------------------------------------


def _exact_modulus(graph: TransformGraph, width: int) -> int:
    """A modulus that ``width`` and every ``unmerge`` divisor divide, so a plain constant keeps
    its exact residue through every step (only an ``unmerge`` divides)."""
    modulus = width
    for node in graph.nodes:
        if node.kind is TransformType.UNMERGE:
            modulus *= prod(node.params.lengths)
    return modulus


def _start(coord: AffineCoord, exact: int) -> Congruence:
    if not coord.uniform and coord.lane is None:
        return Congruence(exact, coord.const)
    return coord.congruence


def _aligned(congruence: Congruence, width: int) -> bool:
    """Known to be a multiple of ``width``."""
    return congruence.modulus % width == 0 and congruence.residue % width == 0


def _end_residue(end: int | RuntimeValue, width: int) -> int | None:
    """``end mod width``, from the int itself or from a runtime end's stated multiple-of fact."""
    if isinstance(end, int):
        return end % width
    if end._congruence.modulus % width:
        return None
    return (end._congruence.residue + end._plus) % width


def _linear(reads: list[_RunStep], coeffs: Sequence[int], exact: int) -> _RunStep:
    """``sum(read_i * coeff_i)``: steps and residues weighted and added."""
    step, congruence = 0, Congruence(exact, 0)
    for read, k in zip(reads, coeffs):
        if k == 0:
            continue
        if read.step is None:
            return _RunStep(None, Congruence(1, 0), read.origin, read.reason)
        step += read.step * k
        congruence = congruence.plus(read.congruence.scaled(k))
    return _RunStep(step, congruence)


def _unmerge(
    read: _RunStep,
    read_name: str,
    names: tuple[str, ...],
    lengths: tuple[int, ...],
    run: VectorRun,
) -> dict[str, _RunStep]:
    congruences = [read.congruence.floor_div(prod(lengths[1:]))]
    for i in range(1, len(lengths)):
        congruences.append(
            read.congruence.floor_div(prod(lengths[i + 1 :])).floor_mod(lengths[i])
        )
    width, inner = run.width, lengths[-1]
    if read.step == 0:
        return {n: _RunStep(0, c) for n, c in zip(names, congruences)}
    if read.step is None:
        origin, reason = read.origin, read.reason
    elif read.step != 1:
        origin = read_name
        reason = f"it moves by {read.step} per element, and the unmerge into {names} splits it"
    elif not _aligned(read.congruence, width):
        origin = read_name
        reason = (
            f"it may start at {read.congruence}, so the unmerge into {names} can split the "
            "vector"
        )
    elif inner % width:
        origin = read_name
        reason = (
            f"the unmerge into {names} has an inner length {inner} that {width} does not "
            "divide"
        )
    else:
        pieces = {n: _RunStep(0, c) for n, c in zip(names, congruences)}
        pieces[names[-1]] = _RunStep(1, congruences[-1])
        return pieces
    return {n: _RunStep(None, Congruence(1, 0), origin, reason) for n in names}


def _xor(reads: list[_RunStep], into: str, length: int, width: int) -> _RunStep:
    first, second = reads
    if first.step == 0 and second.step == 0:
        return _RunStep(0, Congruence(1, 0))
    for read in reads:
        if read.step is None:
            return _RunStep(None, Congruence(1, 0), read.origin, read.reason)
    if (
        first.step == 0
        and second.step == 1
        and _aligned(first.congruence, width)
        and _aligned(second.congruence, width)
        and length % width == 0
    ):
        return _RunStep(1, Congruence(width, 0))
    return _RunStep(
        None,
        Congruence(1, 0),
        into,
        f"the xor into '{into}' can reorder the vector's elements",
    )
