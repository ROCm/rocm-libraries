# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Device-function lowering for the QSA stage functions (Python engine only).

A subclass of the family's device-function lowerer that adds what the FlyDSL
mono's Q3-Q7 code compiles to and the rocKE IR does not express yet. The ops
are this package's own (``q38.*``); no other rocKE build sees them, so every
existing kernel's IR is unchanged.

- ``q38.intr``: one LLVM intrinsic call with immediates (``update.dpp`` with
  row / bank masks, ``maximum.f32``, ``fma.f64``, ...): the FlyDSL helpers'
  exact intrinsic, so equal IR gives equal bits (and the same instruction).
- ``q38.ifr`` / ``q38.yield``: a runtime ``if`` / ``else`` whose regions yield
  values (an LLVM ``phi`` at the join), as FlyDSL's ``scf.if`` with results.
- ``arith.fma``: ``llvm.fma.<ty>`` (FlyDSL's ``fma`` / ``_fma64``) unless the
  ``fma`` knob asks for ``fmuladd`` (the rocKE default lowering, f32 only).
- ``arith.fmax`` on f32: ``llvm.maximum.f32`` (FlyDSL ``fx.max``), unless the
  ``fmax`` knob asks for ``maxnum``.
"""

from __future__ import annotations

from rocke.core.ir import F32, F64, I1, I32, IRBuilder, Op, Region, Type, Value

from ..device_function import (
    DeviceFunction,
    DeviceFunctionKnobs,
    _DeviceFunctionLowerer,
    _SYMBOL_RE,
)

_LL = {"i1": "i1", "i32": "i32", "i64": "i64", "f32": "float", "f64": "double"}


def _ll(t: Type) -> str:
    try:
        return _LL[t.name]
    except KeyError:
        raise TypeError(f"q38 ops: unsupported type {t.name}") from None


# ------------------------------------------------------------------ builders


def intr(b: IRBuilder, callee: str, ret: Type, args, *, name: str = "intr") -> Value:
    """``call <ret> @<callee>(args)``; ``args`` are Values or ``("i32", n)`` /
    ``("i1", bool)`` immediates, in order."""
    ops, spec = [], []
    for a in args:
        if isinstance(a, Value):
            spec.append(("v", len(ops)))
            ops.append(a)
        else:
            kind, val = a
            spec.append((kind, int(val)))
    return b._op(
        "q38.intr", ops, [ret], attrs={"callee": callee, "args": spec, "ret": ret.name},
        result_name_hint=name,
    ).result  # fmt: skip


def update_dpp(b, old: Value, src: Value, ctrl: int, row_mask=0xF, bank_mask=0xF, bound=False):
    """``llvm.amdgcn.update.dpp.i32`` (FlyDSL ``dpp_utils.update_dpp_i32``)."""
    return intr(
        b, "llvm.amdgcn.update.dpp.i32", I32,
        [old, src, ("i32", ctrl), ("i32", row_mask), ("i32", bank_mask), ("i1", bound)],
        name="dpp",
    )  # fmt: skip


def poll_fly(b: IRBuilder, tag: Value, specs, *, aux: int = 16, pause: str = "s_nop 0"):
    """FlyDSL ``mailbox._poll_spin``: ``v = load(specs)``; while any tag differs: ``pause``, reload. ``specs``:
    [(rsrc, byte offset, dwords in 2 / 4)]; returns one ``<dwords x i32>`` per spec."""
    ops, widths = [tag], []
    for rs, off, w in specs:
        if w not in (2, 4):
            raise ValueError(w)
        ops += [rs, off]
        widths.append(w)
    from rocke.core.ir import VectorType

    return b._op("q38.poll", ops, [VectorType(I32, w) for w in widths],
                 attrs={"dwords": widths, "aux": int(aux), "pause": pause}, result_name_hint="fpoll").results


def binop(b: IRBuilder, opcode: str, x: Value, y: Value) -> Value:
    """``%r = <opcode> <ty> x, y`` (``ashr``, ...)."""
    if opcode not in ("ashr", "udiv", "urem", "sdiv", "srem"):
        raise ValueError(opcode)
    return b._op("q38.bin", [x, y], [x.type], attrs={"op": opcode}, result_name_hint=opcode).result


def ashr(b: IRBuilder, x: Value, y: Value) -> Value:
    return binop(b, "ashr", x, y)


def umax_i32(b, x: Value, y: Value) -> Value:
    return intr(b, "llvm.umax.i32", I32, [x, y], name="umax")


def maximum_f32(b, x: Value, y: Value) -> Value:
    return intr(b, "llvm.maximum.f32", F32, [x, y], name="fmaxm")


def fma_f64(b, x: Value, y: Value, z: Value) -> Value:
    return intr(b, "llvm.fma.f64", F64, [x, y, z], name="fma64")


class _IfR:
    """``with ifr(b, cond, types) as (then_, else_, results)``: open each
    branch with ``with then_:`` / ``with else_:`` and end it with
    ``yield_(b, *values)``; ``results`` are the joined values."""

    def __init__(self, b: IRBuilder, cond: Value, types):
        self.b = b
        self.then_r, self.else_r = Region("then"), Region("else")
        self.op = b._op("q38.ifr", [cond], list(types), regions=[self.then_r, self.else_r],
                        result_name_hint="ifr")  # fmt: skip

    def __enter__(self):
        return _Branch(self.b, self.then_r), _Branch(self.b, self.else_r), list(self.op.results)

    def __exit__(self, *a):
        return False


class _Branch:
    def __init__(self, b, region):
        self.b, self.region = b, region

    def __enter__(self):
        self.b.push_region(self.region)
        return self

    def __exit__(self, *a):
        self.b.pop_region()
        return False


def ifr(b: IRBuilder, cond: Value, types) -> _IfR:
    return _IfR(b, cond, types)


def yield_(b: IRBuilder, *values: Value) -> None:
    b._op("q38.yield", list(values), [])


# ------------------------------------------------------------------ lowering


class QsaLowerer(_DeviceFunctionLowerer):
    def __init__(self, kernel, *, symbol, knobs, llvm_flavor, arch, fma="fma", fmax="maximum"):
        super().__init__(kernel, symbol=symbol, knobs=knobs, llvm_flavor=llvm_flavor, arch=arch)
        if fma not in ("fma", "fmuladd") or fmax not in ("maximum", "maxnum"):
            raise ValueError(f"fma {fma!r} / fmax {fmax!r}")
        self._q38_fma, self._q38_fmax = fma, fmax
        self._q38_yields: list = []

    def _declare(self, key: str, decl: str) -> None:
        self._decls[key] = decl
        self._need(key)

    def _op_q38_intr(self, op: Op) -> None:
        callee = op.attrs["callee"]
        ret = op.attrs["ret"]
        rty = _LL[ret]
        parts, types = [], []
        for kind, val in op.attrs["args"]:
            if kind == "v":
                v = op.operands[val]
                t = _ll(v.type)
                parts.append(f"{t} {self._operand(v)}")
                types.append(t)
            elif kind == "i32":
                parts.append(f"i32 {val}")
                types.append("i32")
            elif kind == "i1":
                parts.append(f"i1 {'true' if val else 'false'}")
                types.append("i1")
            else:
                raise ValueError(kind)
        self._declare(callee, f"declare {rty} @{callee}({', '.join(types)})")
        self._current().emit(f"  {op.result.name} = call {rty} @{callee}({', '.join(parts)})")

    def _op_q38_poll(self, op: Op) -> None:
        """FlyDSL's poll shape: the loads and the compare in the current block, then
        ``br bad, retry, done``; ``retry``: the pause asm, the loads again, the compare, ``br bad, retry, done``;
        ``done``: a phi per result."""
        tag = self._operand(op.operands[0])
        rest = op.operands[1:]
        widths = [int(w) for w in op.attrs["dwords"]]
        aux = int(op.attrs["aux"])
        pause = op.attrs.get("pause", "s_nop 0")

        def emit_loads(blk, suffix):
            names, bad = [], None
            for k, w in enumerate(widths):
                key, ty = self._cp_buffer_intrinsic("load", w)
                self._need(key)
                nm = self._fresh(f"fp{suffix}")
                blk.emit(f"  {nm} = call {ty} @llvm.amdgcn.{key}(ptr addrspace(8) {self._operand(rest[2 * k])}, "
                         f"i32 {self._operand(rest[2 * k + 1])}, i32 0, i32 {aux})")
                names.append((nm, ty))
                for lane in range(1, w, 2):
                    got = self._fresh("fp.tag")
                    ne = self._fresh("fp.ne")
                    blk.emit(f"  {got} = extractelement {ty} {nm}, i32 {lane}")
                    blk.emit(f"  {ne} = icmp ne i32 {got}, {tag}")
                    if bad is None:
                        bad = ne
                    else:
                        acc = self._fresh("fp.any")
                        blk.emit(f"  {acc} = or i1 {bad}, {ne}")
                        bad = acc
            return names, bad

        cur = self._current()
        first, bad0 = emit_loads(cur, "0")
        retry = self._new_block("fpoll.retry")
        done = self._new_block("fpoll.done")
        cur.emit(f"  br i1 {bad0}, label %{retry.label}, label %{done.label}")
        cur.terminated = True
        retry.emit(f'  call void asm sideeffect "{pause}", ""()')
        again, bad1 = emit_loads(retry, "1")
        retry.emit(f"  br i1 {bad1}, label %{retry.label}, label %{done.label}")
        retry.terminated = True
        for (n0, ty), (n1, _), r in zip(first, again, op.results):
            done.emit(f"  {r.name} = phi {ty} [ {n0}, %{cur.label} ], [ {n1}, %{retry.label} ]")

    def _op_q38_bin(self, op: Op) -> None:
        x, y = op.operands
        t = _ll(x.type)
        self._current().emit(f"  {op.result.name} = {op.attrs['op']} {t} {self._operand(x)}, {self._operand(y)}")

    def _op_arith_fma(self, op: Op) -> None:
        a = op.operands[0]
        if a.type.name == "f32" and self._q38_fma == "fmuladd":
            return super()._op_arith_fma(op)
        if a.type.name not in ("f32", "f64"):
            return super()._op_arith_fma(op)
        t = _ll(a.type)
        key = f"llvm.fma.{a.type.name}"
        self._declare(key, f"declare {t} @{key}({t}, {t}, {t})")
        ops = ", ".join(f"{t} {self._operand(v)}" for v in op.operands)
        self._current().emit(f"  {op.result.name} = call {t} @{key}({ops})")

    def _op_arith_fmax(self, op: Op) -> None:
        a = op.operands[0]
        if a.type.name != "f32" or self._q38_fmax == "maxnum":
            return super()._op_arith_fmax(op)
        self._declare("llvm.maximum.f32", "declare float @llvm.maximum.f32(float, float)")
        x, y = (self._operand(v) for v in op.operands)
        self._current().emit(f"  {op.result.name} = call float @llvm.maximum.f32(float {x}, float {y})")

    def _op_q38_yield(self, op: Op) -> None:
        if not self._q38_yields:
            raise RuntimeError("q38.yield outside q38.ifr")
        self._q38_yields[-1].append([self._operand(v) for v in op.operands])

    def _op_q38_ifr(self, op: Op) -> None:
        (cond,) = op.operands
        then_r, else_r = op.regions
        cur = self._current()
        then_blk = self._new_block("ifr.then")
        cur.emit(f"  br i1 {self._operand(cond)}, label %{then_blk.label}, label %IFR_ELSE")
        cur.terminated = True
        self._q38_yields.append([])
        self.lower_region(then_r)
        then_last = self._current()
        then_last.emit("  br label %IFR_END")
        then_last.terminated = True
        else_blk = self._new_block("ifr.else")
        self.lower_region(else_r)
        else_last = self._current()
        ys = self._q38_yields.pop()
        if len(ys) != 2 or any(len(y) != len(op.results) for y in ys):
            raise RuntimeError("q38.ifr: each branch must yield its results once, at its end")
        end_blk = self._new_block("ifr.end")
        else_last.emit(f"  br label %{end_blk.label}")
        else_last.terminated = True
        for blk in (cur, then_last):
            blk.lines = [
                ln.replace("%IFR_ELSE", f"%{else_blk.label}").replace("%IFR_END", f"%{end_blk.label}")
                for ln in blk.lines
            ]
        for k, r in enumerate(op.results):
            t = _ll(r.type)
            end_blk.emit(
                f"  {r.name} = phi {t} [ {ys[0][k]}, %{then_last.label} ], [ {ys[1][k]}, %{else_last.label} ]"
            )


def lower_qsa_function(
    kernel,
    *,
    arch: str,
    symbol: str,
    llvm_flavor: str | None = None,
    knobs: DeviceFunctionKnobs = DeviceFunctionKnobs(lds="arg", inline="always"),  # noqa: B008
    fma: str = "fma",
    fmax: str = "maximum",
) -> DeviceFunction:
    """``device_function.lower_device_function`` with :class:`QsaLowerer`."""
    if not _SYMBOL_RE.match(symbol):
        raise ValueError(f"bad device-function symbol {symbol!r}")
    low = QsaLowerer(kernel, symbol=symbol, knobs=knobs, llvm_flavor=llvm_flavor, arch=arch,
                     fma=fma, fmax=fmax)  # fmt: skip
    low._collect_smem(kernel.body)
    low._compute_smem_layout()
    low.lower_region(kernel.body)
    text = low.finalize()
    return DeviceFunction(
        symbol=symbol,
        ir=text,
        abi=tuple(low.abi),
        lds_bytes=int(low._smem_pool_size if low._smem_globals else 0),
        arch=arch,
        flavor=low._flavor,
        knobs=knobs,
        max_workgroup_size=kernel.max_workgroup_size,
    )


__all__ = ["poll_fly", "intr", "update_dpp", "binop", "ashr", "umax_i32", "maximum_f32", "fma_f64", "ifr", "yield_", "QsaLowerer",
           "lower_qsa_function", "F32", "F64", "I1", "I32"]  # fmt: skip
