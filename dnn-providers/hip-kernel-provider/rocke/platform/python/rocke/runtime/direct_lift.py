# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Inline-asm idioms lifted to rocKE IR ops for the direct code-object compiler.

The direct compiler (``platform/rust/rocke-codeobj``) does not assemble inline
asm: it refuses a ``tile.inline_asm`` it does not know, and ``compile_kernel``
then compiles the kernel with COMGR. Persistent kernels (the Qwen3.8
monokernel family) build their device ops, mailbox polls and lane scans from
a small set of fixed asm templates (``rocke.helpers.device_ops``,
``rocke.helpers.mailbox``, the stage modules). :func:`lift` rewrites each
recognized template into the rocKE IR ops the direct compiler lowers natively,
so the same ``KernelDef`` compiles directly without editing its emitters.

It only runs for ``code_object_backend="direct"`` (``helpers.compile``), on a
copy: the kernel the caller holds, and its COMGR build, are untouched. An
unrecognized template stays as it is (the direct compiler then refuses it,
and the COMGR fallback compiles the kernel as before). ``ROCKE_DIRECT_LIFT=0``
turns the lifting off.

The lifted forms keep each template's semantics. A template clobbering
memory stays a compiler barrier: its lifted ops sit between compiler fences
and keep the waits the template ends with (they order the accesses after it:
a flag load waited before the data loads issue). Only the ``s_nop`` hazard
pads go (the direct compiler's hazard pass places its own). Python engine only.
"""

from __future__ import annotations

import os
import re
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Sequence

from ..core.ir import F32, I32, I64, KernelDef, Op, PtrType, Region, Value, VectorType

__all__ = ["LIFT_ENV", "lift", "lift_enabled", "lifted_templates"]

LIFT_ENV = "ROCKE_DIRECT_LIFT"

CPOL_SC0, CPOL_NT, CPOL_SC1 = 1, 2, 16


def lift_enabled() -> bool:
    return os.environ.get(LIFT_ENV, "1") != "0"


class _Ctx:
    """Fresh names, constants and op emission for one :func:`lift`."""

    def __init__(self, kernel: KernelDef) -> None:
        self.names = set()
        self.uses: Dict[int, int] = {}
        self._walk(kernel.body)
        self.n = 0
        self.out: List[Op] = []
        self.lifted: Dict[str, int] = {}

    def _walk(self, region: Region) -> None:
        for op in region.ops:
            for v in op.results:
                self.names.add(v.name)
            for v in op.operands:
                self.uses[id(v)] = self.uses.get(id(v), 0) + 1
            for r in op.regions:
                self._walk(r)

    def value(self, ty) -> Value:
        while True:
            self.n += 1
            name = f"%lft{self.n}"
            if name not in self.names:
                self.names.add(name)
                return Value(name, ty)

    def emit(self, name: str, operands: Sequence[Value], results: Sequence[Value] = (), **attrs) -> Op:
        op = Op(name=name, operands=list(operands), results=list(results), attrs=dict(attrs))
        self.out.append(op)
        return op

    def op1(self, name: str, operands: Sequence[Value], ty, **attrs) -> Value:
        v = self.value(ty)
        self.emit(name, operands, [v], **attrs)
        return v

    def const_i32(self, x: int) -> Value:
        return self.op1("arith.constant", [], I32, value=int(x), ity="i32")


# ----------------------------------------------------------------- templates


def _norm(template: str) -> str:
    return "\n".join(line.strip() for line in template.strip().split("\n") if line.strip())


_CPOL = re.compile(r"^(?P<bits>(?:\s+(?:sc0|sc1|nt))*)$")


def _cpol(bits: str) -> Optional[int]:
    aux = 0
    for b in bits.split():
        aux |= {"sc0": CPOL_SC0, "sc1": CPOL_SC1, "nt": CPOL_NT}[b]
    return aux


Rule = Callable[[_Ctx, Op, str, str], Optional[List[Value]]]
_RULES: List[Rule] = []


def _rule(fn: Rule) -> Rule:
    _RULES.append(fn)
    return fn


def _results(op: Op) -> List[Value]:
    return list(op.results)


@_rule
def _ptr(c: _Ctx, op: Op, t: str, cons: str):
    # device_ops.ptr_from_i64: s_mov_b64 / v_mov_b64 of an i64 into a pointer.
    if t in ("s_mov_b64 $0, $1", "v_mov_b64 $0, $1") and cons in ("=s,s", "=v,v") and len(op.results) == 1:
        r = op.results[0]
        if isinstance(r.type, PtrType) and r.type.space == "global":
            c.emit("tile.int_to_ptr", op.operands, [r])
            return []
    return None


@_rule
def _getreg(c: _Ctx, op: Op, t: str, cons: str):
    m = re.fullmatch(r"s_getreg_b32 \$0, hwreg\(HW_REG_(XCC_ID|HW_ID)\)", t)
    if m and cons == "=s" and len(op.results) == 1:
        c.emit("tile.s_getreg", [], op.results, hwreg=m.group(1))
        return []
    return None


@_rule
def _clock(c: _Ctx, op: Op, t: str, cons: str):
    m = re.fullmatch(r"s_mem(realtime|time) \$0\ns_waitcnt lgkmcnt\(0\)", t)
    if m and cons == "=s" and len(op.results) == 1 and op.results[0].type == I64:
        c.emit("tile.memrealtime" if m.group(1) == "realtime" else "tile.memtime", [], op.results)
        return []
    return None


@_rule
def _sleep_nop(c: _Ctx, op: Op, t: str, cons: str):
    if op.results or op.operands or cons not in ("", "~{memory}"):
        return None
    if t == "":
        # An empty side-effecting statement (a poll pause "none", a compiler
        # fence): the direct compiler hoists no load and moves no memory
        # access across a fence; ~{memory} becomes one.
        if cons == "~{memory}":
            c.emit("tile.compiler_fence", [])
        return []
    lines = t.split("\n")
    if all(re.fullmatch(r"s_sleep \d+", ln) for ln in lines):
        units = sum(int(ln.split()[1]) for ln in lines)
        if units > 0:
            c.emit("tile.s_sleep", [], units=units)
        return []
    if len(lines) == 1 and re.fullmatch(r"s_nop \d+", t):
        c.emit("tile.s_nop", [], n=int(t.split()[1]))
        return []
    m = re.fullmatch(r"s_waitcnt vmcnt\((\d+)\)", t)
    if m:
        c.emit("tile.s_waitcnt", [], vmcnt=int(m.group(1)))
        return []
    m = re.fullmatch(r"(?:s_waitcnt vmcnt\(0\)\n)?buffer_(wbl2|inv)((?: sc[01])+)(?:\ns_waitcnt vmcnt\(0\))?", t)
    if m and cons == "~{memory}":
        bits = m.group(2).split()
        scope = "system" if "sc0" in bits else "agent"
        if bits not in (["sc1"], ["sc0", "sc1"]):
            return None
        c.emit("memref.fence", [], scope=scope, ordering="release" if m.group(1) == "wbl2" else "acquire")
        return []
    return None


@_rule
def _store(c: _Ctx, op: Op, t: str, cons: str):
    # mailbox / device_ops puts: global_store_dword[xN] $0, $1, $2|off [offset:N] <cpol>
    if op.results:
        return None
    lines = t.split("\n")
    if lines and re.fullmatch(r"s_nop \d+", lines[-1]):
        lines = lines[:-1]  # a hazard pad after a wide asm store
    if not lines:
        return None
    stores = []
    for ln in lines:
        m = re.fullmatch(
            r"global_store_dword(?:x(2|4))? \$(\d+), \$(\d+), (\$\d+|off)(?: offset:(\d+))?((?: (?:sc0|sc1|nt))*)", ln
        )
        if not m:
            return None
        stores.append(m)
    if "=" in cons:
        return None
    for m in stores:
        n = int(m.group(1) or 1)
        addr = op.operands[int(m.group(2))]
        data = op.operands[int(m.group(3))]
        dwords = 1 if data.type == I32 else getattr(data.type, "count", 0)
        if dwords != n:
            return None
        base = m.group(4)
        aux = _cpol(m.group(6))
        off = int(m.group(5) or 0)
        if base == "off":
            if addr.type != I64:
                return None
            c.emit("tile.global_store_cp", [data, addr], aux=aux, offset=off)
        else:
            sbase = op.operands[int(base[1:])]
            if addr.type != I32 or sbase.type != I64:
                return None
            c.emit("tile.global_store_cp", [data, addr, sbase], aux=aux, offset=off)
    return []


@_rule
def _load(c: _Ctx, op: Op, t: str, cons: str):
    # global_load_dword[xN] $0, $k, $j|off [offset] <cpol> (+ s_waitcnt vmcnt(0)); a load
    # with no cache bits and no wait whose value is never read is the toy chain's L2 touch.
    lines = t.split("\n")
    wait = lines[-1] == "s_waitcnt vmcnt(0)"
    if wait:
        lines = lines[:-1]
    if len(lines) != len(op.results) or not lines:
        return None
    loads = []
    for k, ln in enumerate(lines):
        m = re.fullmatch(
            r"global_load_dword(?:x(2|4))? \$(\d+), \$(\d+), (\$\d+|off)(?: offset:(\d+))?((?: (?:sc0|sc1|nt))*)", ln
        )
        if not m or int(m.group(2)) != k:
            return None
        loads.append(m)
    for k, m in enumerate(loads):
        r = op.results[k]
        n = int(m.group(1) or 1)
        dwords = 1 if r.type in (I32, F32) else (2 if r.type == I64 else getattr(r.type, "count", 0))
        if dwords != n:
            return None
        nres = len(op.results)
        addr = op.operands[int(m.group(3)) - nres]
        aux, off = _cpol(m.group(6)), int(m.group(5) or 0)
        touch = not wait and aux == 0 and off == 0 and n == 4 and m.group(4) != "off" and c.uses.get(id(r), 0) <= 1
        if m.group(4) == "off":
            operands = [addr]
        else:
            operands = [addr, op.operands[int(m.group(4)[1:]) - nres]]
        if touch and len(operands) == 2:
            c.emit("tile.l2_touch_load", operands, [r])
            continue
        if not wait:
            return None
        c.emit("tile.global_load_cp", operands, [r], aux=aux, offset=off, dwords=n)
    return []


@_rule
def _hold(c: _Ctx, op: Op, t: str, cons: str):
    # The toy chain's L2-touch drain: s_waitcnt vmcnt(0) holding the touch loads' destinations.
    if t == "s_waitcnt vmcnt(0)" and not op.results and op.operands and cons == ",".join(["v"] * len(op.operands)):
        if all(v.type == VectorType(I32, 4) for v in op.operands):
            c.emit("tile.l2_touch_drain", op.operands)
            return []
    return None


@_rule
def _atomic(c: _Ctx, op: Op, t: str, cons: str):
    m = re.fullmatch(r"global_atomic_add(_x2)? \$0, \$1, \$2((?: sc[01])*)", t)
    if m and not op.results and cons == "v,v,s,~{memory}":
        bits = m.group(2).split()
        if "sc0" in bits:
            return None
        c.emit("tile.global_atomic_add_scoped", op.operands, scope="system" if "sc1" in bits else "agent", returns=False)
        return []
    m = re.fullmatch(r"global_atomic_add(_x2)? \$0, \$1, \$2, \$3((?: sc[01])*)\ns_waitcnt vmcnt\(0\)", t)
    if m and len(op.results) == 1 and cons == "=&v,v,v,s,~{memory}":
        bits = m.group(2).split()
        if bits not in (["sc0"], ["sc0", "sc1"]):
            return None
        c.emit("tile.global_atomic_add_scoped", op.operands, op.results, scope="system" if "sc1" in bits else "agent", returns=True)
        return []
    return None


_DPP_MOV2 = re.compile(
    r"s_nop 1\nv_mov_b32_dpp \$0, \$1 (row_sh[lr]:\d+) row_mask:(0x[0-9a-f]+) bank_mask:(0x[0-9a-f]+)"
    r"\nv_mov_b32_dpp \$0, \$1 (row_sh[lr]:\d+) row_mask:(0x[0-9a-f]+) bank_mask:(0x[0-9a-f]+)"
)


def _dpp_ctrl(text: str) -> Optional[int]:
    m = re.fullmatch(r"row_(shl|shr|ror):(\d+)", text)
    if m:
        k = int(m.group(2))
        return None if not 1 <= k <= 15 else {"shl": 0x100, "shr": 0x110, "ror": 0x120}[m.group(1)] + k
    m = re.fullmatch(r"row_bcast:(15|31)", text)
    if m:
        return 0x142 if m.group(1) == "15" else 0x143
    return None


@_rule
def _dpp(c: _Ctx, op: Op, t: str, cons: str):
    # stage_b3's xor-4 row exchange: two bank-masked row shifts into one register.
    m = _DPP_MOV2.fullmatch(t)
    if m and cons == "=&v,v" and len(op.results) == 1 and op.results[0].type == I32:
        src = op.operands[0]
        r1 = c.op1(
            "tile.dpp_op", [src, src], I32, op="mov", ctrl=_dpp_ctrl(m.group(1)),
            row_mask=int(m.group(2), 16), bank_mask=int(m.group(3), 16), bound_ctrl=False,
        )  # fmt: skip
        if (int(m.group(3), 16) | int(m.group(6), 16)) != 0xF or int(m.group(2), 16) != 0xF or int(m.group(5), 16) != 0xF:
            return None
        c.emit(
            "tile.dpp_op", [r1, src], op.results, op="mov", ctrl=_dpp_ctrl(m.group(4)),
            row_mask=int(m.group(5), 16), bank_mask=int(m.group(6), 16), bound_ctrl=False,
        )  # fmt: skip
        return []
    # device_ops.wave_umax_dpp: v_max_u32_dpp $0, $0, $0 steps in one register.
    if cons == "=v,0" and len(op.results) == 1 and op.results[0].type == I32:
        steps = [ln for ln in t.split("\n") if not re.fullmatch(r"s_nop \d+", ln)]
        parsed = []
        for ln in steps:
            mm = re.fullmatch(
                r"v_max_u32_dpp \$0, \$0, \$0 (row_\w+:\d+)(?: row_mask:(0x[0-9a-f]+))?(?: bank_mask:(0x[0-9a-f]+))?( bound_ctrl:[01])?", ln
            )
            if not mm or _dpp_ctrl(mm.group(1)) is None:
                return None
            parsed.append(mm)
        if not parsed:
            return None
        v = op.operands[0]
        for k, mm in enumerate(parsed):
            attrs = dict(
                op="max_u32", ctrl=_dpp_ctrl(mm.group(1)), row_mask=int(mm.group(2) or "0xf", 16),
                bank_mask=int(mm.group(3) or "0xf", 16), bound_ctrl=mm.group(4) is not None,
            )  # fmt: skip
            if k == len(parsed) - 1:
                c.emit("tile.dpp_op", [v, v, v], op.results, **attrs)
            else:
                v = c.op1("tile.dpp_op", [v, v, v], I32, **attrs)
        return []
    return None


@_rule
def _lds(c: _Ctx, op: Op, t: str, cons: str):
    # LDS at a raw byte address (an smem_addr_of value): ds_or / ds_write / ds_read_b32 and their lgkmcnt(0).
    m = re.fullmatch(r"ds_(or|write)_b32 \$0, \$1(\ns_waitcnt lgkmcnt\(0\))?", t)
    if m and not op.results and cons in ("v,v,~{memory}", "v,v"):
        if op.operands[0].type != I32 or op.operands[1].type != I32:
            return None
        c.emit(f"tile.ds_{m.group(1)}_b32", op.operands)
        if m.group(2):
            c.emit("tile.s_waitcnt", [], lgkmcnt=0)
        return []
    m = re.fullmatch(r"ds_read_b32 \$0, \$1(\ns_waitcnt lgkmcnt\(0\))?", t)
    if m and len(op.results) == 1 and op.results[0].type == I32 and cons in ("=v,v,~{memory}", "=v,v"):
        c.emit("tile.ds_read_b32", op.operands, op.results)
        return []
    return None


@_rule
def _alu(c: _Ctx, op: Op, t: str, cons: str):
    if t == "v_maximum3_f32 $0, |$1|, |$2|, |$3|" and cons == "=v,v,v,v" and op.results[0].type == F32:
        c.emit("arith.maximum3_f32", op.operands, op.results, abs=7)
        return []
    if t == "v_bcnt_u32_b32 $0, $1, 0" and cons == "=v,v":
        c.emit("tile.bcnt", op.operands, op.results)
        return []
    m = re.fullmatch(r"s_ff1_i32_b(32|64) \$0, \$1", t)
    if m and cons == "=s,s":
        c.emit("tile.s_ff1", op.operands, op.results)
        return []
    if t == "s_add_u32 $0, $1, $2\ns_addc_u32 $0, $0, 0" and cons == "=&s,s,s,~{scc}":
        return None  # a 64-bit add written as 32-bit halves: not lifted (its result type differs per site)
    return None


@_rule
def _cvt(c: _Ctx, op: Op, t: str, cons: str):
    # helpers.mx_quant cvt="asm": v_cvt_scalef32_pk_{fp8,bf8}_f32 into a fresh word or one half of $1.
    m = re.fullmatch(r"v_cvt_scalef32_pk_(fp8|bf8)_f32 \$0, \$1, \$2, \$3", t)
    if m and cons == "=v,v,v,v":
        zero = c.const_i32(0)
        c.emit(f"arith.cvt_scalef32_pk_{m.group(1)}_f32", [zero, *op.operands], op.results, hi=False)
        return []
    m = re.fullmatch(r"v_cvt_scalef32_pk_(fp8|bf8)_f32 \$0, \$2, \$3, \$4 op_sel:\[0,0,0,1\]", t)
    if m and cons == "=v,0,v,v,v":
        c.emit(f"arith.cvt_scalef32_pk_{m.group(1)}_f32", op.operands, op.results, hi=True)
        return []
    return None


@_rule
def _sload(c: _Ctx, op: Op, t: str, cons: str):
    # stage_b3's state table: s_load_dwordx{2,4} $k, $base, offset ... + one lgkmcnt(0).
    lines = t.split("\n")
    if lines[-1] != "s_waitcnt lgkmcnt(0)" or len(lines) - 1 != len(op.results) or len(op.operands) != 1:
        return None
    base = op.operands[0]
    if base.type != I64:
        return None
    ptrs: Dict[str, Value] = {}
    for k, ln in enumerate(lines[:-1]):
        m = re.fullmatch(rf"s_load_dword(x2|x4)? \${k}, \${len(op.results)}, (\d+)", ln)
        if not m:
            return None
        r = op.results[k]
        off = int(m.group(2))
        if r.type == I64:
            elem, n = I64, 1
        elif r.type == I32:
            elem, n = I32, 1
        elif isinstance(r.type, VectorType) and r.type.elem == I32 and r.type.count in (2, 4):
            elem, n = I32, r.type.count
        else:
            return None
        if off % elem_bytes(elem):
            return None
        key = elem.name
        if key not in ptrs:
            ptrs[key] = c.op1("tile.int_to_ptr", [base], PtrType(elem, "constant"))
        idx = c.const_i32(off // elem_bytes(elem))
        if n == 1:
            c.emit("memref.global_load_typed", [ptrs[key], idx], [r], elem_type=elem.name, align=elem_bytes(elem))
        else:
            c.emit("memref.global_load_vN", [ptrs[key], idx], [r], elem_type=elem.name, vec=n, align=4)
    return []


def elem_bytes(t) -> int:
    return 8 if t == I64 else 4


# ---------------------------------------------------------------- polls

_POLL_TOP = re.compile(r"\$\{:private\}poll\$\{:uid\}_top:")


def _poll(c: _Ctx, op: Op, t: str, cons: str):
    """``mailbox.asm_poll_template``: ``P`` (payload, tag) pairs over ``K``
    specs, a VGPR temp and an SGPR mask out; per-spec byte offsets, the
    bases and the zero-extended tag in. Lifted to ``tile.tagged_poll`` on a
    buffer resource of each base (one load per spec: 2 or 4 dwords)."""
    lines = t.split("\n")
    if not lines or not _POLL_TOP.fullmatch(lines[0]):
        return None
    loads, pause, i = [], "", 1
    while i < len(lines) and lines[i].startswith("global_load_dwordx2 "):
        m = re.fullmatch(r"global_load_dwordx2 \$(\d+), \$(\d+), \$(\d+)(?: offset:(\d+))? (sc1|sc0 sc1)", lines[i])
        if not m:
            return None
        loads.append(m)
        i += 1
    p_total = len(loads)
    if p_total == 0 or len(op.results) != p_total + 2:
        return None
    # Group the pairs into specs: consecutive loads with the same offset operand, offsets 0 then 8.
    specs: List[List[re.Match]] = []
    for m in loads:
        if m.group(4) and specs and specs[-1][0].group(2) == m.group(2) and int(m.group(4)) == 8 * len(specs[-1]):
            specs[-1].append(m)
        elif not m.group(4):
            specs.append([m])
        else:
            return None
    cpol = loads[0].group(5)
    if any(m.group(5) != cpol for m in loads):
        return None
    tail = lines[i:]
    pause_line = [ln for ln in tail if ln in ("s_nop 0", "s_sleep 1")]
    pause = pause_line[0] if pause_line else ""
    # Rebuild the exact template from what was parsed: anything else is not this idiom.
    from ..helpers.mailbox import asm_poll_template

    first_in = p_total + 2
    base_idx = sorted({int(m.group(3)) for m in loads})
    nbases = len(base_idx)
    base_of = [base_idx.index(int(s[0].group(3))) for s in specs]
    want, want_cons = asm_poll_template([len(s) for s in specs], nbases, base_of, cpol, pause)
    if _norm(want) != t or want_cons != cons:
        return None
    if any(c.uses.get(id(op.results[k]), 0) for k in (p_total, p_total + 1)):
        return None
    voffs = op.operands[: len(specs)]
    bases = op.operands[len(specs) : len(specs) + nbases]
    tag64 = op.operands[len(specs) + nbases]
    zop = getattr(tag64, "op", None)
    if zop is not None and zop.name == "arith.zext" and zop.operands and zop.operands[0].type == I32:
        tag = zop.operands[0]
    else:
        tag = c.op1("arith.trunc", [tag64], I32)
    size = c.const_i32(-1)
    rsrcs = []
    for b in bases:
        ptr = c.op1("tile.int_to_ptr", [b], PtrType(I32, "global"))
        rsrcs.append(c.op1("tile.buffer_rsrc", [ptr, size], VectorType(I32, 4)))
    operands: List[Value] = [tag]
    widths = []
    for k, s in enumerate(specs):
        operands += [rsrcs[base_of[k]], voffs[k]]
        widths.append(2 * len(s))
    vecs = [c.value(VectorType(I32, w)) for w in widths]
    c.emit("tile.tagged_poll", operands, vecs, dwords=widths, aux=CPOL_SC1 | (CPOL_SC0 if cpol == "sc0 sc1" else 0), pause=pause)
    p = 0
    for k, s in enumerate(specs):
        for q in range(len(s)):
            r = op.results[p]
            p += 1
            # The pair as the asm gave it: (payload, tag) words of the spec's load.
            lo = c.op1("vector.extract", [vecs[k]], I32, index=2 * q)
            hi = c.op1("vector.extract", [vecs[k]], I32, index=2 * q + 1)
            c.emit("vector.pack", [lo, hi], [r], elem="i32", vec=2)
    # The temp and mask outputs have no use (checked above), so they need no definition.
    return []


_RULES.append(_poll)


# ---------------------------------------------------------------- driver


_FENCES = ("tile.compiler_fence", "memref.fence", "tile.s_waitcnt")
_WAIT = re.compile(r"s_waitcnt((?: (?:vmcnt|lgkmcnt|expcnt)\(\d+\))+)")


def _fenced(c: _Ctx, t: str, cons: str, ops: List[Op]) -> List[Op]:
    """An asm statement clobbering memory is a compiler barrier for memory accesses, and the waits it ends
    with order what follows after its own accesses (a flag load waited before the data loads issue). The
    lifted ops keep both: a compiler fence before and after, and the template's trailing ``s_waitcnt``
    (``ROCKE_DIRECT_LIFT_FENCE=0`` drops them, for experiments only: protocols then race)."""
    if "~{memory}" not in cons or not ops or all(o.name in _FENCES for o in ops):
        return ops
    if os.environ.get("ROCKE_DIRECT_LIFT_FENCE", "1") == "0":
        return ops
    tail: List[Op] = []
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    while lines and _WAIT.fullmatch(lines[-1]):
        counts = dict(re.findall(r"(vmcnt|lgkmcnt|expcnt)\((\d+)\)", lines.pop()))
        tail.insert(0, Op(name="tile.s_waitcnt", operands=[], results=[], attrs={k: int(v) for k, v in counts.items()}))
    fence = lambda: Op(name="tile.compiler_fence", operands=[], results=[], attrs={})
    return [fence(), *ops, *tail, fence()]


def _lift_region(c: _Ctx, region: Region) -> Region:
    new_ops: List[Op] = []
    for op in region.ops:
        if op.name == "tile.inline_asm":
            t, cons = _norm(str(op.attrs.get("template", ""))), str(op.attrs.get("constraints", ""))
            done = False
            for rule in _RULES:
                c.out = []
                try:
                    got = rule(c, op, t, cons)
                except (IndexError, KeyError, ValueError, AttributeError):
                    got = None
                if got is not None:
                    lifted = _fenced(c, t, cons, c.out)
                    scope = op.attrs.get("dlevers")
                    if scope is not None:
                        # The lifted ops keep the statement's lever scope (``IRBuilder.lever_scope``).
                        lifted = [o if "dlevers" in o.attrs else replace(o, attrs={**o.attrs, "dlevers": scope}) for o in lifted]
                    new_ops.extend(lifted)
                    c.lifted[rule.__name__] = c.lifted.get(rule.__name__, 0) + 1
                    done = True
                    break
            if done:
                continue
        if op.regions:
            op = replace(op, regions=[_lift_region(c, r) for r in op.regions])
        new_ops.append(op)
    return Region(label=region.label, ops=new_ops)


def lift(kernel: KernelDef) -> KernelDef:
    """A copy of ``kernel`` with every recognized inline-asm idiom replaced
    by rocKE IR ops (see the module docstring); ``kernel`` is not changed.
    The copy's ``lifted`` attribute (a Python attribute, not a kernel
    attribute) counts the lifted statements per rule."""
    c = _Ctx(kernel)
    body = _lift_region(c, kernel.body)
    out = replace(kernel, body=body, attrs=dict(kernel.attrs), params=list(kernel.params))
    out.lifted = dict(c.lifted)  # type: ignore[attr-defined]
    return out


def lifted_templates(kernel: KernelDef) -> Dict[str, int]:
    """Rule -> count of the statements :func:`lift` would replace."""
    return dict(getattr(lift(kernel), "lifted", {}))
