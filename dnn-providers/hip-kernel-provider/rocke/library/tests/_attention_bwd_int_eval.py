# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Typed host evaluator of the backward integer / address emitters.

A duck-typed stand-in for ``IRBuilder`` that executes the ops the backward
band, address and sink emitters use, on Python values, with the IR's typing
rules enforced:

* every value carries its type (``i1``, ``i32``, ``i64``, ``f32``, ``ptr``,
  ``vec``); binary ops require equal types (as the IR does);
* ``i32`` / ``i64`` arithmetic that leaves the type's range raises
  ``OverflowError`` (the lowering emits ``nsw``, so an overflow would be
  undefined behaviour on the device);
* ``smax`` / ``smin`` accept ``i32`` only (their lowering is ``llvm.smax.i32``);
* pointers are ``(buffer name, byte offset)``; loads, stores and atomics go
  to numpy buffers and raise on a null pointer, a misaligned or an
  out-of-bounds element;
* ``scf_for_iter`` supports the 0/1-trip guard used by the emitters: a dead
  body is evaluated in "dead" mode (memory is not touched, results are the
  initial values);
* the f32 -> bf16 rounding of a store or a cast follows ``bf16_cast``:
  ``"generic"`` is LLVM's ``fptrunc`` (round to nearest even; a NaN stays a
  quiet NaN of the same sign, ``(u | 0x400000) >> 16``), ``"rne_noguard"``
  the same rounding without the NaN guard (``(u + 0x7fff + lsb) >> 16`` on
  the wrapped 32-bit word). They differ on NaN inputs only;
* the i32 vector ops (``vector_add`` / ``vector_and`` / ``vector_lshr``) wrap
  modulo ``2**32`` (their lowering has no ``nsw``), unlike scalar ``add``.

``Value`` objects of ``rocke.core.ir`` are not used, so a value built here can
be tested with ``isinstance(x, int)`` exactly like a runtime IR value.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

_RANGE = {"i32": (-(1 << 31), (1 << 31) - 1), "i64": (-(1 << 63), (1 << 63) - 1)}
_NP = {
    "i32": np.int32,
    "i64": np.int64,
    "f32": np.float32,
    "f16": np.float16,
    "bf16": None,
    "i16": np.int16,
}
_ESIZE = {"i32": 4, "i64": 8, "f32": 4, "f16": 2, "bf16": 2, "i16": 2}


@dataclass(frozen=True)
class TV:
    """A typed value."""

    v: Any
    ty: str

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.ty}:{self.v}"

    @property
    def type(self):
        return _TypeView(self.ty)


@dataclass(frozen=True)
class _TypeView:
    name: str

    @property
    def count(self):  # vectors only
        return int(self.name.split("x")[-1].rstrip(">"))

    @property
    def elem(self):
        return _TypeView(self.name[4:].split("x")[0])

    def __eq__(self, other):
        return getattr(other, "name", None) == self.name

    def __hash__(self):
        return hash(self.name)


def _ty(t) -> str:
    return t if isinstance(t, str) else t.name


def _bf16_bits_to_f32(bits: int) -> float:
    return float(np.array([bits << 16], dtype=np.uint32).view(np.float32)[0])


BF16_ROUNDINGS = ("generic", "rne_noguard")


def _f32_bits(x: float) -> int:
    return int(np.array([x], dtype=np.float32).view(np.uint32)[0])


def _f32_of_bits(u: int) -> float:
    return float(np.array([u & 0xFFFFFFFF], dtype=np.uint32).view(np.float32)[0])


def _f32_to_bf16_bits(x: float, mode: str = "generic") -> int:
    """bf16 bits of ``x`` under rounding ``mode`` (see the module docstring)."""
    if mode not in BF16_ROUNDINGS:
        raise ValueError(f"bf16 rounding {mode!r} not in {BF16_ROUNDINGS}")
    u = _f32_bits(x)
    if mode == "generic" and (u & 0x7FFFFFFF) > 0x7F800000:
        return ((u | 0x400000) >> 16) & 0xFFFF
    rounding = 0x7FFF + ((u >> 16) & 1)
    return ((u + rounding) >> 16) & 0xFFFF


def _f32_obj(u: int) -> np.float32:
    """An ``np.float32`` holding the bits ``u`` exactly (a Python float would
    quiet a signalling NaN on the way through double)."""
    return np.array([u & 0xFFFFFFFF], dtype=np.uint32).view(np.float32)[0]


def _wrap32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= 1 << 31 else v


class VirtualTensor:
    """A huge read-mostly buffer: element ``e`` reads ``fn(e)``; writes are kept.

    Lets the address tests touch element offsets past ``2**31`` without
    allocating them.
    """

    def __init__(self, size: int, fn) -> None:
        self.size = int(size)
        self.fn = fn
        self.writes: dict[int, Any] = {}

    def reshape(self, *_a):
        return self

    def __getitem__(self, e: int):
        return self.writes.get(e, self.fn(e))

    def __setitem__(self, e: int, v) -> None:
        self.writes[e] = v


class IntEvalBuilder:
    """Executes builder calls on typed host values (see module docstring)."""

    def __init__(
        self, memory: dict | None = None, *, bf16_cast: str = "generic"
    ) -> None:
        if bf16_cast not in BF16_ROUNDINGS:
            raise ValueError(f"bf16_cast {bf16_cast!r} not in {BF16_ROUNDINGS}")
        self.bf16_cast = bf16_cast
        # name -> (numpy array, element type name)
        self.memory: dict[str, tuple[np.ndarray, str]] = {}
        for name, arr in (memory or {}).items():
            self.add_buffer(name, arr)
        self.dead = 0
        self.log: list[tuple] = []  # (kind, buffer, element index, value)
        self._yield = None

    # ----- memory -----

    def add_buffer(self, name: str, arr: np.ndarray, ty: str | None = None) -> TV:
        if ty is None:
            ty = {
                np.dtype(np.int32): "i32",
                np.dtype(np.int64): "i64",
                np.dtype(np.float32): "f32",
                np.dtype(np.float16): "f16",
                np.dtype(np.uint16): "bf16",
                np.dtype(np.int16): "i16",
            }[arr.dtype]
        self.memory[name] = (arr, ty)
        return self.ptr(name)

    def ptr(self, name: str | None, byte_off: int = 0) -> TV:
        return TV((name, byte_off), "ptr")

    def _elem(self, ptr: TV, idx: TV, ty: str) -> tuple[np.ndarray, int]:
        assert ptr.ty == "ptr", ptr
        assert idx.ty in ("i32", "i64"), idx
        name, off = ptr.v
        if name is None:
            raise MemoryError("null pointer dereferenced")
        arr, aty = self.memory[name]
        esize = _ESIZE[ty]
        if _ESIZE[aty] != esize:
            raise TypeError(f"access as {ty} into a {aty} buffer")
        byte = off + idx.v * esize
        if byte % esize:
            raise MemoryError(f"misaligned {ty} access at byte {byte} of {name}")
        e = byte // esize
        if not 0 <= e < arr.size:
            raise IndexError(f"{name}[{e}] out of bounds (size {arr.size})")
        return arr.reshape(-1), e

    def _read(self, ptr, idx, ty):
        if self.dead:
            return TV(0 if ty in ("i32", "i64") else 0.0, ty)
        flat, e = self._elem(ptr, idx, ty)
        raw = flat[e]
        self.log.append(("load", ptr.v[0], e, raw))
        if ty in ("i32", "i64"):
            return TV(int(raw), ty)
        if ty == "bf16":
            return TV(_bf16_bits_to_f32(int(raw)), "bf16")
        return TV(float(raw), ty)

    def _write(self, ptr, idx, value: TV, ty: str):
        if self.dead:
            return
        flat, e = self._elem(ptr, idx, ty)
        v = value.v
        if ty == "bf16":
            flat[e] = _f32_to_bf16_bits(v, self.bf16_cast)
        else:
            flat[e] = v
        self.log.append(("store", ptr.v[0], e, v))

    # ----- constants / arithmetic -----

    def const_i32(self, v):
        return self._chk(TV(int(v), "i32"))

    def const_i64(self, v):
        return self._chk(TV(int(v), "i64"))

    def const_f32(self, v):
        return TV(float(np.float32(v)), "f32")

    def _chk(self, x: TV) -> TV:
        if x.ty in _RANGE:
            lo, hi = _RANGE[x.ty]
            if not lo <= x.v <= hi:
                raise OverflowError(f"{x.ty} overflow: {x.v}")
        return x

    def _same(self, a: TV, b: TV) -> str:
        if a.ty != b.ty:
            raise TypeError(f"operand types differ: {a.ty} vs {b.ty}")
        return a.ty

    def add(self, a, b):
        t = self._same(a, b)
        if t == "f32":
            return self.fadd(a, b)
        return self._chk(TV(a.v + b.v, t))

    def sub(self, a, b):
        t = self._same(a, b)
        return self._chk(TV(a.v - b.v, t))

    def mul(self, a, b):
        t = self._same(a, b)
        return self._chk(TV(a.v * b.v, t))

    def div(self, a, b):
        t = self._same(a, b)
        if b.v == 0:
            raise ZeroDivisionError("sdiv by zero")
        q = abs(a.v) // abs(b.v)
        return self._chk(TV(q if (a.v >= 0) == (b.v > 0) else -q, t))

    def mod(self, a, b):
        t = self._same(a, b)
        q = self.div(a, b).v
        return TV(a.v - q * b.v, t)

    def smax(self, a, b):
        if self._same(a, b) != "i32":
            raise TypeError("smax lowers for i32 only")
        return TV(max(a.v, b.v), "i32")

    def smin(self, a, b):
        if self._same(a, b) != "i32":
            raise TypeError("smin lowers for i32 only")
        return TV(min(a.v, b.v), "i32")

    def shl(self, a, b):
        t = self._same(a, b)
        bits = 32 if t == "i32" else 64
        r = (a.v << b.v) & ((1 << bits) - 1)
        if r >= 1 << (bits - 1):
            r -= 1 << bits
        return TV(r, t)

    def lshr(self, a, b):
        t = self._same(a, b)
        bits = 32 if t == "i32" else 64
        return TV((a.v & ((1 << bits) - 1)) >> b.v, t)

    def xor(self, a, b):
        t = self._same(a, b)
        return TV(a.v ^ b.v, t)

    def zext(self, v, target):
        tt = _ty(target)
        if v.ty == "i1":
            return TV(int(bool(v.v)), tt)
        bits = 32 if v.ty == "i32" else 64
        return TV(v.v & ((1 << bits) - 1), tt)

    def sext(self, v, target):
        tt = _ty(target)
        if v.ty == "i1":
            return TV(-1 if v.v else 0, tt)
        return TV(v.v, tt)

    def trunc(self, v, target):
        tt = _ty(target)
        bits = 32 if tt == "i32" else 64
        r = v.v & ((1 << bits) - 1)
        if r >= 1 << (bits - 1):
            r -= 1 << bits
        return TV(r, tt)

    # ----- predicates -----

    def _cmp(self, a, b, fn):
        self._same(a, b)
        return TV(bool(fn(a.v, b.v)), "i1")

    def cmp_lt(self, a, b):
        return self._cmp(a, b, lambda x, y: x < y)

    def cmp_le(self, a, b):
        return self._cmp(a, b, lambda x, y: x <= y)

    def cmp_gt(self, a, b):
        return self._cmp(a, b, lambda x, y: x > y)

    def cmp_ge(self, a, b):
        return self._cmp(a, b, lambda x, y: x >= y)

    def cmp_eq(self, a, b):
        return self._cmp(a, b, lambda x, y: x == y)

    def cmp_ne(self, a, b):
        return self._cmp(a, b, lambda x, y: x != y)

    def fcmp(self, pred, a, b):
        x, y = a.v, b.v
        unordered = np.isnan(x) or np.isnan(y)
        ops = {
            "olt": lambda: not unordered and x < y,
            "ole": lambda: not unordered and x <= y,
            "ogt": lambda: not unordered and x > y,
            "oge": lambda: not unordered and x >= y,
            "oeq": lambda: not unordered and x == y,
            "one": lambda: not unordered and x != y,
        }
        return TV(bool(ops[pred]()), "i1")

    def land(self, a, b):
        if a.ty == "i1":
            return TV(bool(a.v and b.v), "i1")
        return TV(a.v & b.v, self._same(a, b))

    def lor(self, a, b):
        if a.ty == "i1":
            return TV(bool(a.v or b.v), "i1")
        return TV(a.v | b.v, self._same(a, b))

    def lnot(self, a):
        if a.ty == "i1":
            return TV(not a.v, "i1")
        return TV(~a.v, a.ty)

    def select(self, c, x, y):
        assert c.ty == "i1", c
        self._same(x, y)
        return x if c.v else y

    def readfirstlane(self, v):
        assert v.ty in ("i32", "i64")
        return v

    # ----- float -----

    def fadd(self, a, b):
        self._same(a, b)
        return TV(float(np.float32(a.v) + np.float32(b.v)), "f32")

    def fsub(self, a, b):
        self._same(a, b)
        return TV(float(np.float32(a.v) - np.float32(b.v)), "f32")

    def fmul(self, a, b):
        self._same(a, b)
        return TV(float(np.float32(a.v) * np.float32(b.v)), "f32")

    def fneg(self, a):
        assert a.ty == "f32", a
        return TV(_f32_of_bits(_f32_bits(a.v) ^ 0x80000000), "f32")

    def cast_to_f32(self, v):
        return TV(float(v.v), "f32")

    def cast_f32_to(self, v, target):
        tt = _ty(target)
        if tt == "f16":
            return TV(float(np.float16(v.v)), "f16")
        if tt == "bf16":
            bits = _f32_to_bf16_bits(v.v, self.bf16_cast)
            return TV(_bf16_bits_to_f32(bits), "bf16")
        return v

    def vec_cast_f32_to(self, v, target):
        """Element-wise ``fptrunc`` (the generic cast; ``bf16_cast`` rounding)."""
        tt = _ty(target)
        if v.type.elem.name != "f32":
            raise TypeError(f"vec_cast_f32_to expects f32 elements, got {v.ty}")
        out = [self.cast_f32_to(TV(x, "f32"), tt).v for x in v.v]
        return TV(tuple(out), f"vec<{tt}x{len(out)}>")

    # ----- bit-level ops -----

    def bitcast(self, v, target):
        tt = _ty(target)
        if (v.ty, tt) == ("f32", "i32"):
            return TV(_wrap32(_f32_bits(v.v)), "i32")
        if (v.ty, tt) == ("i32", "f32"):
            return TV(_f32_of_bits(v.v), "f32")
        raise TypeError(f"bitcast {v.ty} -> {tt} is not modelled")

    def byte_perm(self, a, b, sel):
        """``llvm.amdgcn.perm(a, b, sel)``: byte k = source byte sel[k] of the
        eight bytes ``b`` (0..3, LSB first) then ``a`` (4..7)."""
        assert a.ty == "i32" and b.ty == "i32", (a, b)
        src = (b.v & 0xFFFFFFFF) | ((a.v & 0xFFFFFFFF) << 32)
        out = 0
        for k in range(4):
            s = (int(sel) >> (8 * k)) & 0xFF
            if s > 7:
                raise NotImplementedError("perm selectors above 7 are not modelled")
            out |= ((src >> (8 * s)) & 0xFF) << (8 * k)
        return TV(_wrap32(out), "i32")

    def vec_bitcast(self, v, target):
        """Bit reinterpretation between ``<n x f32>``, ``<n x i32>`` and
        ``<2n x bf16>`` vectors (equal total size). Float elements come back
        as ``np.float32`` scalars, which keep every NaN payload bit."""
        tname = target.name if hasattr(target, "name") else str(target)
        src_t = v.type.elem.name
        dst_t = target.elem.name
        dst_n = target.count
        if src_t == "f32":
            words = [_f32_bits(x) for x in v.v]
        elif src_t == "i32":
            words = [x & 0xFFFFFFFF for x in v.v]
        elif src_t == "bf16":
            halves = [_f32_bits(x) >> 16 for x in v.v]
            words = [
                halves[2 * i] | (halves[2 * i + 1] << 16) for i in range(len(v.v) // 2)
            ]
        else:
            raise TypeError(f"vec_bitcast from {v.ty} is not modelled")
        if dst_t == "f32":
            out = tuple(_f32_obj(w) for w in words)
        elif dst_t == "i32":
            out = tuple(_wrap32(w) for w in words)
        elif dst_t == "bf16":
            out = tuple(
                _f32_obj(((w >> (16 * j)) & 0xFFFF) << 16)
                for w in words
                for j in (0, 1)
            )
        else:
            raise TypeError(f"vec_bitcast to {tname} is not modelled")
        if len(out) != dst_n:
            raise TypeError(f"vec_bitcast size mismatch: {v.ty} -> {tname}")
        return TV(out, f"vec<{dst_t}x{dst_n}>")

    def vector_splat(self, scalar, n):
        return TV(tuple([scalar.v] * n), f"vec<{scalar.ty}x{n}>")

    def _vec_i32(self, a, b, fn):
        if a.ty != b.ty or a.type.elem.name != "i32":
            raise TypeError(f"i32 vector op on {a.ty}, {b.ty}")
        return TV(
            tuple(
                _wrap32(fn(x & 0xFFFFFFFF, y & 0xFFFFFFFF)) for x, y in zip(a.v, b.v)
            ),
            a.ty,
        )

    def vector_add(self, a, b):
        return self._vec_i32(a, b, lambda x, y: x + y)

    def vector_and(self, a, b):
        return self._vec_i32(a, b, lambda x, y: x & y)

    def vector_lshr(self, a, b):
        return self._vec_i32(a, b, lambda x, y: x >> y)

    # ----- pointers and memory -----

    def global_ptr_add(self, ptr, off):
        assert ptr.ty == "ptr"
        if off.ty != "i64":
            raise TypeError("global_ptr_add offsets are formed in i64 here")
        name, base = ptr.v
        return TV((name, base + off.v), "ptr")

    def global_load_i32(self, ptr, idx, *, align=4):
        return self._read(ptr, idx, "i32")

    def global_load_f32(self, ptr, idx, *, align=4):
        return self._read(ptr, idx, "f32")

    def global_load(self, ptr, idx, dtype, *, align=1):
        return self._read(ptr, idx, _ty(dtype))

    def global_load_vN(self, ptr, idx, dtype, n, *, align=None):
        t = _ty(dtype)
        if not self.dead:
            byte = ptr.v[1] + idx.v * _ESIZE[t]
            if align and byte % align:
                raise MemoryError(f"vector load misaligned: byte {byte} align {align}")
        elems = [self._read(ptr, TV(idx.v + i, idx.ty), t) for i in range(n)]
        self.log.append(("vload", ptr.v[0], n, align))
        return TV(tuple(e.v for e in elems), f"vec<{t}x{n}>")

    def global_store(self, ptr, idx, value, *, align=1):
        self._write(ptr, idx, value, value.ty)

    def global_store_vN(self, ptr, idx, value, n, *, align=None):
        t = value.type.elem.name
        if not self.dead:
            byte = ptr.v[1] + idx.v * _ESIZE[t]
            if align and byte % align:
                raise MemoryError(f"vector store misaligned: byte {byte} align {align}")
        for i in range(n):
            self._write(ptr, TV(idx.v + i, idx.ty), TV(value.v[i], t), t)
        self.log.append(("vstore", ptr.v[0], n, align))

    def global_atomic_add_f32(self, ptr, idx, value):
        assert value.ty == "f32"
        if self.dead:
            return
        flat, e = self._elem(ptr, idx, "f32")
        flat[e] = np.float32(flat[e]) + np.float32(value.v)
        self.log.append(("atomic", ptr.v[0], e, value.v))

    # ----- vectors -----

    def vec_pack(self, comps, elem):
        t = _ty(elem)
        for c in comps:
            if c.ty != t:
                raise TypeError(f"vec_pack expected {t}, got {c.ty}")
        return TV(tuple(c.v for c in comps), f"vec<{t}x{len(comps)}>")

    def vec_extract(self, v, i):
        t = v.type.elem.name
        return TV(v.v[i], t)

    def vec_concat(self, a, b):
        t = a.type.elem.name
        return TV(a.v + b.v, f"vec<{t}x{len(a.v) + len(b.v)}>")

    # ----- control flow -----

    def scf_for_iter(self, lower, upper, step, iter_args, iv_name="k0", **_kw):
        trip = (upper.v - lower.v + step.v - 1) // step.v if upper.v > lower.v else 0
        if trip > 1:
            raise NotImplementedError("only 0/1-trip guard loops are evaluated")
        return _Loop(self, lower, [init for _n, init in iter_args], trip)

    def scf_yield(self, *values):
        self._yield = list(values)

    def scf_if(self, cond):
        return _If(self, bool(cond.v))

    def ret(self):
        return None


class _Loop:
    def __init__(self, b: IntEvalBuilder, iv, inits, trip) -> None:
        self.b = b
        self.iv = iv
        self.inits = inits
        self.trip = trip
        self.results = list(inits)

    def __enter__(self):
        if self.trip == 0:
            self.b.dead += 1
        self.b._yield = None
        return self.iv, list(self.inits)

    def __exit__(self, *exc):
        if self.trip == 0:
            self.b.dead -= 1
            self.results = list(self.inits)
        else:
            assert self.b._yield is not None, "loop body did not yield"
            self.results = self.b._yield
        self.b._yield = None
        return False


class _If:
    def __init__(self, b: IntEvalBuilder, taken: bool) -> None:
        self.b = b
        self.taken = taken

    def __enter__(self):
        if not self.taken:
            self.b.dead += 1
        return self

    def __exit__(self, *exc):
        if not self.taken:
            self.b.dead -= 1
        return False


@contextmanager
def dead_mode(b: IntEvalBuilder):
    b.dead += 1
    try:
        yield
    finally:
        b.dead -= 1


def i32(b: IntEvalBuilder, v: int) -> TV:
    """A runtime-like i32 value (not a Python int)."""
    return b.const_i32(v)
