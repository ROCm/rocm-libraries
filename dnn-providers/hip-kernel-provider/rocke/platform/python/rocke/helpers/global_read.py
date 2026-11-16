# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Prepared, complete in-bounds global reads with uniform whole-body selection.

Describe the *actual* extents of existing global TensorViews, then call
``emit_global_read_paths(b, specs, body)``. Bind invariant coordinates in the
body preheader and load at descriptor-coordinate deltas inside its loop.
This is not a masking API: every scalar/vector must be fully in bounds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from ..core.ir import I32, I64, IRBuilder, Value
from .tensor_view import BufferResource, TensorView, make_buffer_resource

__all__ = [
    "GlobalReadSpec",
    "GlobalReadPlan",
    "GlobalReadCursor",
    "emit_global_read_paths",
]

_MAX_INPUTS = 8
_BUFFER_BYTES = {"f16": 2, "bf16": 2, "f32": 4, "i32": 4}
_ELEM_BYTES = {
    **_BUFFER_BYTES,
    "f64": 8,
    "i64": 8,
    "i16": 2,
    "i8": 1,
    "fp8e4m3": 1,
    "bf8e5m2": 1,
    "tf32": 4,
}
_UNIFORM_OPS = frozenset(
    (
        "arith.add",
        "arith.sub",
        "arith.mul",
        "arith.div",
        "arith.mod",
        "arith.sext",
        "arith.zext",
        "arith.trunc",
        "arith.smin",
        "arith.smax",
        "arith.select",
        "arith.cmp",
        "arith.and",
        "arith.or",
        "arith.not",
        "tile.global_ptr_offset",
        "tile.global_ptr_add",
    )
)


def _uniform(
    b: IRBuilder, value: Value, depth: int = 0, visits: list[int] | None = None
) -> bool:
    # Proof only, never rewriting IR or silently broadcasting a lane's metadata.
    # Unknown producers (including loads, loop args and readfirstlane) fail closed.
    # A shared visit limit also bounds repeated subexpressions in a large DAG.
    if visits is None:
        visits = [256]
    if depth >= 64 or visits[0] == 0:
        return False
    visits[0] -= 1
    if value.op is None:
        return any(value is b.get_param(p.name) for p in b.kernel.params)
    op = value.op
    if op.name in ("arith.constant", "gpu.block_id"):
        return True
    return op.name in _UNIFORM_OPS and all(
        _uniform(b, v, depth + 1, visits) for v in op.operands
    )


def _wide(b: IRBuilder, value: int | Value) -> Value:
    if isinstance(value, int):
        return b.const_i64(value)
    if value.type == I64:
        return value
    if value.type == I32:
        return b.sext(value, I64)
    raise TypeError("global reads require signed i32 or i64 coordinates and metadata")


# select+cmp, not b.smin/b.smax: those lower as i32 only.
def _smin(b: IRBuilder, x: Value, y: Value) -> Value:
    return b.select(b.cmp_lt(x, y), x, y)


def _smax(b: IRBuilder, x: Value, y: Value) -> Value:
    return b.select(b.cmp_gt(x, y), x, y)


@dataclass(frozen=True)
class GlobalReadSpec:
    view: TensorView
    extents: tuple[int | Value, ...]

    def __post_init__(self) -> None:
        if self.view.addr_space != "global":
            raise ValueError("GlobalReadSpec requires a global TensorView")
        if len(self.extents) != self.view.rank:
            raise ValueError("global read extent rank mismatch")


@dataclass(frozen=True)
class GlobalReadPlan:
    view: TensorView
    addressing: str
    buffer: BufferResource | None = None

    @property
    def desc(self):
        return self.view.desc

    def bind(self, b: IRBuilder, indices: Sequence[Value]) -> GlobalReadCursor:
        off = self.desc.offset_i64(b, indices)
        byte_off = b.mul(off, b.const_i64(_ELEM_BYTES[self.view.dtype.name]))
        if self.addressing == "buffer":
            return GlobalReadCursor(self, b.trunc(byte_off, I32))
        return GlobalReadCursor(self, b.global_ptr_offset(self.view.base, byte_off))


@dataclass(frozen=True)
class GlobalReadCursor:
    plan: GlobalReadPlan
    base: Value

    def _offset(self, b: IRBuilder, deltas: Sequence[Value]) -> Value:
        if self.plan.addressing == "buffer":
            # Complete in-bounds accesses + the checked span prove the final
            # byte address fits i32. Modular delta arithmetic also permits a
            # negative delta from a bound coordinate inside that same span.
            if len(deltas) != self.plan.desc.rank:
                raise ValueError("global read delta rank mismatch")

            def narrow(value: Value) -> Value:
                if value.type == I32:
                    return value
                if value.type == I64:
                    return b.trunc(value, I32)
                raise TypeError("global read deltas require signed i32 or i64")

            off = None
            for delta, stride in zip(deltas, self.plan.desc.strides):
                delta = narrow(delta)
                if isinstance(stride, Value):
                    term = b.mul(delta, narrow(stride))
                elif stride == 1:
                    term = delta
                else:
                    # i32 arithmetic is modular; no signed host narrowing.
                    imm = ((int(stride) + (1 << 31)) % (1 << 32)) - (1 << 31)
                    term = b.mul(delta, b.const_i32(imm))
                off = term if off is None else b.add(off, term)
            byte_off = b.mul(off, b.const_i32(_ELEM_BYTES[self.plan.view.dtype.name]))
            return b.add(self.base, byte_off)
        return self.plan.desc.offset_i64(b, deltas)

    def load_vec(self, b: IRBuilder, deltas: Sequence[Value], n: int) -> Value:
        off = self._offset(b, deltas)
        dtype = self.plan.view.dtype
        if self.plan.addressing == "buffer":
            resource = self.plan.buffer
            assert resource is not None
            return b.buffer_load_vN(resource.rsrc, off, resource.soffset, dtype, n)
        return b.global_load_vN(self.base, off, dtype, n, align=1)

    def load_scalar(self, b: IRBuilder, deltas: Sequence[Value]) -> Value:
        off = self._offset(b, deltas)
        dtype = self.plan.view.dtype
        if self.plan.addressing == "buffer":
            resource = self.plan.buffer
            assert resource is not None
            return b.buffer_load(resource.rsrc, off, resource.soffset, dtype)
        return b.global_load(self.base, off, dtype=dtype)


def _can_buffer(b: IRBuilder, spec: GlobalReadSpec) -> bool:
    if spec.view.dtype.name not in _BUFFER_BYTES or not _uniform(b, spec.view.base):
        return False
    for value in (*spec.extents, *spec.view.desc.strides):
        if isinstance(value, int):
            if value < 0 or value > (1 << 63) - 1:
                return False
        elif (
            not isinstance(value, Value)
            or value.type not in (I32, I64)
            or not _uniform(b, value)
        ):
            return False
    return True


def _bounded(
    b: IRBuilder, value: int | Value, cap_int: int, cap: Value, minus: Value
) -> Value:
    # Only i64 metadata can overflow a 64-bit product; i32 metadata is used as is.
    if isinstance(value, int):
        return b.const_i64(min(value, cap_int))
    wide = _wide(b, value)
    if value.type == I64:
        return _smax(b, _smin(b, wide, cap), minus)
    return wide


def _span(b: IRBuilder, spec: GlobalReadSpec) -> tuple[Value, Value]:
    # span = 1 + sum((extent - 1) * stride). Operands are i32 or clamped to
    # [-1, cap], so each product fits i64. A term above `limit` clears `valid`;
    # when every term is within it, the sum of at most 8 terms cannot wrap.
    # Empty or negative metadata clears `valid` (the full-width path).
    elem_bytes = _BUFFER_BYTES[spec.view.dtype.name]
    limit_int = ((1 << 31) - 1) // elem_bytes
    limit = b.const_i64(limit_int)
    cap = b.const_i64(limit_int + 1)
    minus = b.const_i64(-1)
    zero = b.const_i64(0)
    one = b.const_i64(1)
    span = one
    valid = b.cmp_eq(zero, zero)
    for extent, stride in zip(spec.extents, spec.view.desc.strides):
        extent = _bounded(b, extent, limit_int + 1, cap, minus)
        stride = _bounded(b, stride, limit_int + 1, cap, minus)
        valid = b.land(valid, b.cmp_ge(extent, one))
        valid = b.land(valid, b.cmp_ge(stride, zero))
        term = b.mul(b.sub(extent, one), stride)
        valid = b.land(valid, b.cmp_le(term, limit))
        span = b.add(span, term)
    valid = b.land(valid, b.cmp_le(span, limit))
    return valid, b.mul(span, b.const_i64(elem_bytes))


def emit_global_read_paths(
    b: IRBuilder,
    specs: Sequence[GlobalReadSpec],
    emit_body: Callable[[tuple[GlobalReadPlan, ...]], None],
    *,
    addressing: str = "auto",
) -> None:
    """Emit the complete body once per safe static path (never per load).

    ``auto`` selects buffer reads only when all spans fit INT32_MAX bytes
    and all base/extent/stride metadata is provably workgroup-uniform. Unknown
    metadata or unsupported buffer dtypes select only global64. At most eight
    simultaneous inputs are supported, matching the allocation-free C helper.
    ``global64`` bypasses selection. There is deliberately no force-buffer mode.
    Callback-local cursors/plans must not escape their emitted body's lifetime.
    """
    specs = tuple(specs)
    if addressing not in ("auto", "global64"):
        raise ValueError("global read addressing must be 'auto' or 'global64'")
    if not 1 <= len(specs) <= _MAX_INPUTS:
        raise ValueError("global reads require between 1 and 8 inputs")
    for spec in specs:
        if spec.view.dtype.name not in _ELEM_BYTES:
            raise NotImplementedError(
                f"global reads do not support {spec.view.dtype.name}"
            )
    wide = tuple(GlobalReadPlan(spec.view, "global64") for spec in specs)
    if addressing == "global64" or not all(_can_buffer(b, spec) for spec in specs):
        emit_body(wide)
        return
    spans = [_span(b, spec) for spec in specs]
    valid = spans[0][0]
    for cond, _ in spans[1:]:
        valid = b.land(valid, cond)
    with b.scf_if_else(valid) as (then, otherwise):
        with then:
            plans = []
            for spec, (_, byte_span) in zip(specs, spans):
                size = b.trunc(byte_span, I32)
                resource = make_buffer_resource(b, spec.view.base, num_bytes=size)
                plans.append(GlobalReadPlan(spec.view, "buffer", resource))
            emit_body(tuple(plans))
        with otherwise:
            emit_body(wide)
