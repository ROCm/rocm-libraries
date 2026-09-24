# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared GEOMETRY helpers for the analysis + visualization layers -- PURE calc, no matplotlib.

The symbolic-origin resolver, the lane-span counter, and the recording's (arch, wave_size) accessor.
Imported by both the gates (``analysis/``) and the renderers (``visualization/``), so it has ONE home
here and neither layer re-rolls it.

A captured ``window.origin`` entry is an SSA ``Value`` whose ``.op`` back-refs form a walkable DAG that
bottoms out at named ROOTS -- the ``scf.for`` induction variable (the K-loop iteration), ``gpu.thread_id``
(the wave/lane), ``gpu.block_id`` (the macro-tile block), and ``arith.constant`` leaves -- combined by
``arith.add/sub/mul/div/mod``. Resolving an origin = pinning those roots (``k`` / ``tid`` / ``block_id``)
and evaluating the DAG to a concrete int.
"""

from __future__ import annotations

from typing import Any

_BINARY = {
    "arith.add": lambda a, b: a + b,
    "arith.sub": lambda a, b: a - b,
    "arith.mul": lambda a, b: a * b,
    "arith.div": lambda a, b: a // b,   # emit uses integer division
    "arith.mod": lambda a, b: a % b,
}


class OriginResolutionError(RuntimeError):
    """Raised when an origin DAG hits an op/root the resolver was not told how to pin."""


def resolve_value(value: Any, bindings: dict) -> int:
    """Resolve an SSA ``Value`` (or a plain int) to a concrete int by walking ``Value.op`` to the
    pinned roots.

    ``bindings`` supplies the substitution roots:
      - ``k``:        the ``scf.for`` induction value (e.g. the K-tile offset ``kb * tile_k``),
      - ``tid``:      the thread id (the wave = ``tid // wave_size``, lane = ``tid % wave_size``),
      - ``block_id``: optional ``{axis: int}`` for the macro-tile block (global origins).

    The loop IV is a substitution ROOT, not a constant leaf -- any value produced by the ``scf.for``
    op resolves to ``bindings['k']``.
    """
    if isinstance(value, int):
        return value
    op = getattr(value, "op", None)
    if op is None:
        raise OriginResolutionError(
            f"value {getattr(value, 'name', value)!r} has no producing op and is not an int"
        )
    name = op.name
    if name == "arith.constant":
        return int(op.attrs["value"])
    if name == "scf.for":
        if "k" not in bindings:
            raise OriginResolutionError("scf.for induction variable hit but no 'k' binding supplied")
        return int(bindings["k"])
    if name == "gpu.thread_id":
        if "tid" not in bindings:
            raise OriginResolutionError("gpu.thread_id hit but no 'tid' binding supplied")
        return int(bindings["tid"])
    if name == "gpu.block_id":
        axis = op.attrs.get("axis")
        block = bindings.get("block_id", {})
        if axis not in block:
            raise OriginResolutionError(f"gpu.block_id[{axis}] hit but no 'block_id' binding supplied")
        return int(block[axis])
    fn = _BINARY.get(name)
    if fn is None:
        raise OriginResolutionError(f"unhandled op in origin DAG: {name!r}")
    return fn(resolve_value(op.operands[0], bindings), resolve_value(op.operands[1], bindings))


def resolve_origin(origin: tuple, bindings: dict) -> tuple[int, ...]:
    """Resolve every axis of a captured ``window.origin`` to a concrete int tuple."""
    return tuple(resolve_value(o, bindings) for o in origin)


def _lane_span(encoding: Any) -> int:
    """Total threads an encoding spans = product over ALL lane-partition levels (wave outer + lane
    inner). `RegisterMapper.num_lanes` reads only the first level, so for a cooperative NDimP=2
    encoding it undercounts -- this is the count `emit_tensor_coordinates` actually decomposes."""
    span = 1
    for majors, minors in zip(encoding.lane_to_rh_major, encoding.lane_to_rh_minor):
        for major, minor in zip(majors, minors):
            span *= encoding.bucket_length(major, minor)
    return span


def _arch_wave(pipeline: Any) -> tuple[str, int]:
    """The (arch, wave_size) DERIVED from the recording -- either DECLARED by the caller via
    ``record_build(arch=, wave_size=)`` or captured from a recorded ``TileMma``. Fails loud if neither is
    present -- NO silent 'gfx90a'/64 fallback."""
    if pipeline.arch is None or pipeline.wave_size is None:
        raise ValueError(
            "pipeline has no arch/wave_size: no TileMma was recorded and none was declared. A kernel with "
            "no matrix instruction (reduction, scan, elementwise, LDS-combining epilogue) must pass the "
            "target it already resolved: record_build(build_fn, ..., arch='gfxNNN', wave_size=N).")
    return pipeline.arch, pipeline.wave_size


def _elem_bytes(dtype_name: str) -> int:
    from ..emit import _BYTE_WIDTH

    return _BYTE_WIDTH[dtype_name]
