# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WP-L1: the index-width lint (DESIGN-LB 8.2): no i32 address computation of an LB build may overflow.

An interval analysis over the rocKE IR of a kernel. Ranges start from the declared bounds of the i32 kernel arguments
(``arg_bounds``), the ranges of the read-only tables a kernel loads indices from (``table_bounds``: param name ->
(lo, hi) of every word), the hardware ids (block / thread / lane id) and constants, and propagate through ``arith``
ops (no wrap: the exact mathematical interval), ``select``, ``readfirstlane`` / ``readlane`` and ``scf.for`` induction
variables (``[lb, ub - 1]``). The **sinks** are every operand that is an address or a size: buffer offsets and buffer
resource sizes, ``global_load`` / ``global_store`` element indices, every i32 ``zext`` (an i32 offset widened into a
64-bit address), and the address operand of an inline-asm VMEM instruction (``global_*`` / ``buffer_*``: the vaddr
position of the template's VMEM line). Every i32 op in the backward slice of a sink whose interval leaves
``[-2^31, 2^31 - 1]`` is a finding; a slice value whose range is unknown (an asm result, a loop-carried value, a load
from an unannotated buffer) is reported as ``unbounded`` (a finding with ``strict``).

The host-side formulas of :func:`.lbf_plan.index_bounds` give the same numbers for the framework kernel; this lint
checks the IR itself, so a new address expression cannot escape the check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["Finding", "index_lint", "ranges"]

LO, HI = -(1 << 31), (1 << 31) - 1
FULL = (float("-inf"), float("inf"))
_VMEM = re.compile(r"^\s*(global_|buffer_|flat_|scratch_)(\w+)\s+(.*)$")


@dataclass
class Finding:
    kind: str  # "overflow" | "unbounded"
    op: str
    value: str
    lo: float
    hi: float
    sink: str

    def __str__(self) -> str:
        return f"{self.kind}: {self.op} {self.value} in [{self.lo}, {self.hi}] (feeds {self.sink})"


def _ops(region):
    for op in region.ops:
        yield op
        for r in op.regions:
            yield from _ops(r)


def _asm_vaddr(op) -> list[int]:
    """Operand indices of the vaddr of every VMEM line of an inline-asm template (``$k`` counts outputs first)."""
    t = op.attrs.get("template", "")
    cons = [c.strip() for c in op.attrs.get("constraints", "").split(",") if c.strip()]
    nout = sum(1 for c in cons if c.startswith("="))
    out = []
    for ln in t.splitlines():
        m = _VMEM.match(ln)
        if not m:
            continue
        kind, rest = m.group(2), m.group(3)
        args = [a.strip() for a in rest.split(",")]
        ret = kind.startswith("load") or kind.startswith("atomic") and (" sc0" in rest or " glc" in rest)
        pos = 1 if (ret or m.group(1) == "buffer_") else 0
        if pos < len(args):
            mm = re.match(r"\$(\d+)$", args[pos].split()[0])
            if mm and int(mm.group(1)) >= nout:
                out.append(int(mm.group(1)) - nout)
    return out


def _column(idx, defs, cols: int):
    """The table column of an index ``mul(u, cols) [+ c]`` (None when the pattern does not match)."""
    op = defs.get(idx.name)
    if op is None:
        return None
    if op.name == "arith.add":
        for a, c in ((op.operands[0], op.operands[1]), (op.operands[1], op.operands[0])):
            co = defs.get(c.name)
            if co is not None and co.name == "arith.constant" and isinstance(co.attrs.get("value"), int):
                ao = defs.get(a.name)
                if ao is not None and ao.name == "arith.mul" and _is_const(ao.operands[1], defs, cols):
                    return co.attrs["value"]
        return None
    if op.name == "arith.mul" and _is_const(op.operands[1], defs, cols):
        return 0
    if op.name == "arith.constant" and isinstance(op.attrs.get("value"), int):
        return op.attrs["value"] % cols
    return None


def _is_const(v, defs, val: int) -> bool:
    op = defs.get(v.name)
    return op is not None and op.name == "arith.constant" and op.attrs.get("value") == val


def ranges(kd, *, arg_bounds: dict, table_bounds: dict | None = None, grid: int = 512, threads: int = 512):
    """The interval of every value of ``kd`` (``{value name: (lo, hi)}``; unknown = (-inf, inf)), with the def map and
    the param aliases. ``arg_bounds``: i32 param name -> (lo, hi); ``table_bounds``: i64 param name (a table base) ->
    (lo, hi) of every i32 word loaded from it, or ``{"cols": C, col: (lo, hi), ...}`` per column."""
    table_bounds = table_bounds or {}
    rng: dict[str, tuple] = {}
    defs: dict[str, object] = {}
    ptr_of: dict[str, str] = {}  # int_to_ptr result -> param name (through readfirstlane)
    alias: dict[str, str] = {}  # value -> param name it is a copy of
    for p in kd.params:
        nm = "%" + p.name
        alias[nm] = p.name
        if p.name in arg_bounds:
            rng[nm] = tuple(arg_bounds[p.name])

    def r(v):
        return rng.get(v.name, FULL)

    for op in _ops(kd.body):
        res = op.results[0].name if op.results else None
        for x in op.results:
            defs[x.name] = op
        o = op.operands
        n = op.name
        val = None
        if n == "arith.constant":
            v = op.attrs.get("value")
            val = (v, v) if isinstance(v, int) else FULL
        elif n == "gpu.block_id":
            val = (0, grid - 1)
        elif n == "gpu.thread_id":
            val = (0, threads - 1)
        elif n == "tile.lane_id":
            val = (0, 63)
        elif n in ("tile.readfirstlane", "tile.readlane"):
            val = r(o[0])
            if o[0].name in alias:
                alias[res] = alias[o[0].name]
        elif n == "tile.int_to_ptr":
            if o[0].name in alias:
                ptr_of[res] = alias[o[0].name]
        elif n == "memref.global_load_typed":
            tb = table_bounds.get(ptr_of.get(o[0].name, ""))
            if isinstance(tb, dict):  # per-column bounds: {"cols": C, col: (lo, hi), ...}; column = the constant addend
                col = _column(o[1], defs, tb["cols"])
                cs = [v_ for k_, v_ in tb.items() if k_ != "cols"]
                val = tuple(tb[col]) if col in tb else (min(c_[0] for c_ in cs), max(c_[1] for c_ in cs))
            else:
                val = tuple(tb) if tb else FULL
        elif n in ("arith.add", "arith.sub", "arith.mul", "arith.div", "arith.mod", "arith.and", "arith.lshr", "arith.shl",
                   "arith.smax", "arith.smin", "arith.umin", "arith.select", "arith.zext", "arith.trunc", "arith.xor",
                   "arith.or"):
            a = r(o[-2]) if n == "arith.select" else r(o[0])
            c = r(o[-1]) if len(o) > 1 else None
            if n == "arith.add":
                val = (a[0] + c[0], a[1] + c[1])
            elif n == "arith.sub":
                val = (a[0] - c[1], a[1] - c[0])
            elif n == "arith.mul":
                ps = [a[0] * c[0], a[0] * c[1], a[1] * c[0], a[1] * c[1]] if FULL not in (a, c) else []
                ps = [x for x in ps if x == x]
                val = (min(ps), max(ps)) if ps else FULL
            elif n == "arith.div":
                if c[0] >= 1 and a != FULL:
                    val = (min(0, a[0]), max(0, int(a[1]) // int(c[0])))
                else:
                    val = FULL
            elif n == "arith.mod":
                val = (0, c[1] - 1) if c[0] >= 1 and a[0] >= 0 else FULL
            elif n == "arith.and":
                bs = [x[1] for x in (a, c) if x[0] >= 0 and x[1] != float("inf")]
                val = (0, min(bs)) if bs else FULL
            elif n == "arith.lshr":
                val = (0, int(a[1]) >> int(c[0])) if a[0] >= 0 and a[1] != float("inf") and c[0] == c[1] else (0, HI)
            elif n == "arith.shl":
                val = (a[0] * (1 << int(c[0])), a[1] * (1 << int(c[0]))) if c[0] == c[1] and FULL != a else FULL
            elif n == "arith.smax":
                val = (max(a[0], c[0]), max(a[1], c[1]))
            elif n in ("arith.smin", "arith.umin"):
                val = (min(a[0], c[0]), min(a[1], c[1]))
            elif n == "arith.select":
                val = (min(a[0], c[0]), max(a[1], c[1]))
            elif n == "arith.zext":
                val = a
                if o[0].name in alias:
                    alias[res] = alias[o[0].name]
            elif n == "arith.trunc":
                val = FULL
            else:
                val = FULL
        elif n == "scf.for":
            lb, ub = r(o[0]), r(o[1])
            iv = op.attrs.get("iv")
            if iv:
                rng[iv] = (lb[0], ub[1] - 1) if FULL not in (lb, ub) else FULL
        if res is not None and val is not None:
            rng[res] = val
    return rng, defs, alias


def index_lint(kd, *, arg_bounds: dict, table_bounds: dict | None = None, grid: int = 512, threads: int = 512
               ) -> list[Finding]:
    """Findings of kernel ``kd`` (see the module docstring): overflows first, then unbounded slice values."""
    rng, defs, alias = ranges(kd, arg_bounds=arg_bounds, table_bounds=table_bounds, grid=grid, threads=threads)
    # users of i64 values (a zext is a sink only when its result reaches an address: int_to_ptr, a buffer resource
    # base, or an inline-asm operand, through i64 arithmetic)
    users: dict[str, list] = {}
    for op in _ops(kd.body):
        for x in op.operands:
            users.setdefault(x.name, []).append(op)

    def reaches_addr(v, depth=0) -> bool:
        for u in users.get(v.name, []):
            if u.name in ("tile.int_to_ptr", "tile.inline_asm", "tile.buffer_rsrc"):
                return True
            if u.name in ("arith.add", "arith.mul", "arith.sub", "tile.readfirstlane") and depth < 8 and u.results:
                if reaches_addr(u.results[0], depth + 1):
                    return True
        return False

    # sinks
    sinks: list[tuple[str, object]] = []
    for op in _ops(kd.body):
        n, o = op.name, op.operands
        if n == "tile.buffer_load_cp":
            sinks += [(n, o[1]), (n, o[2])]
        elif n == "tile.buffer_store_cp":
            sinks += [(n, o[2]), (n, o[3])]
        elif n == "tile.buffer_rsrc":
            sinks.append((n + ".size", o[1]))
        elif n in ("memref.global_load_typed", "memref.global_store_typed"):
            sinks.append((n, o[1]))
        elif n == "arith.zext" and str(o[0].type) == "i32" and reaches_addr(op.results[0]):
            sinks.append((n, o[0]))
        elif n == "tile.inline_asm":
            for k_ in _asm_vaddr(op):
                if k_ < len(o) and str(o[k_].type) == "i32":
                    sinks.append((n + ":" + op.attrs.get("template", "").splitlines()[-1][:40], o[k_]))
    # masking ops: a bounded result of an and / mod / umin / smin does not depend on the range of its unbounded inputs
    masking = ("arith.and", "arith.mod", "arith.umin", "arith.smin", "arith.lshr", "arith.cmp")
    out, seen = [], set()
    for sname, v in sinks:
        stack = [(v, False)]
        while stack:
            x, masked = stack.pop()
            if (x.name, masked) in seen:
                continue
            seen.add((x.name, masked))
            op = defs.get(x.name)
            lo, hi = rng.get(x.name, FULL)
            unb = (lo, hi) == FULL and str(x.type) == "i32"
            if op is None:
                if unb and not masked:
                    out.append(Finding("unbounded", "param" if x.name in alias else "loop-carried", x.name, lo, hi, sname))
                continue
            if str(x.type) == "i32" and op.name.startswith("arith.") and not unb and (lo < LO or hi > HI):
                out.append(Finding("overflow", op.name, x.name, lo, hi, sname))
            if unb and not op.name.startswith("arith."):
                if not masked:
                    out.append(Finding("unbounded", op.name, x.name, lo, hi, sname))
                continue
            if op.name.startswith("arith.") or op.name in ("tile.readfirstlane", "tile.readlane"):
                m2 = masked or (op.name in masking and not unb)
                stack.extend((y, m2) for y in op.operands)
    return [f for f in out if f.kind == "overflow"] + [f for f in out if f.kind == "unbounded"]
