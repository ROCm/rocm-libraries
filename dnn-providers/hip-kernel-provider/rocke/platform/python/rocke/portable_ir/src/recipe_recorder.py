#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# recipe_recorder.py -- a Python recorder that emits a builder recipe
# (rocke.recipe/v1) automatically.
#
# Instead of hand-writing the recipe JSON (verbose, error-prone, dual-maintained
# with the Python builder), the kernel is *authored idiomatically* against this
# recorder. The recorder's surface mirrors the IRBuilder (param/const/add/fmul/
# load/store/...), so authoring reads almost identically; the only delta from a
# normal builder is that compile-time / runtime control flow uses recorder
# primitives (`scf_for`, `scf_if`, `static_if`) so the emitted recipe stays
# PARAMETRIC (the spec is bound later, in the C VM, at JIT time).
#
# Recording is an OFFLINE step (CPython is fine here); the resulting recipe is
# consumed at runtime by the pure-C recipe VM (no CPython in hipDNN).
#
#   recipe_recorder.py --emit recipe   # author mini_attn idiomatically -> recipe
import argparse
import json
import sys
from contextlib import contextmanager

PTR_F32 = {"kind": "ptr", "pointee": "f32", "space": "global"}


class V:
    """An IR-value handle (a recipe register name + its type)."""

    def __init__(self, reg, type_json):
        self.reg = reg
        self.type = type_json


class IExpr:
    """A compile-time integer expression handle (spec / loop var / literal)."""

    def __init__(self, node):
        self.node = node


def _i(v):
    return {"t": "i", "v": v}


def _f(v):
    return {"t": "f", "v": v}


def _s(v):
    return {"t": "s", "v": v}


def _to_intexpr(x):
    return x.node if isinstance(x, IExpr) else x


class RecipeRecorder:
    def __init__(self, kernel_name_fmt, spec, kattrs=None):
        self._fmt = kernel_name_fmt
        self._spec = spec
        self._kattrs = kattrs or {}
        self._stack = [[]]  # body stack; top is the current instruction list
        self._n = 0

    # -- internals --
    def _emit(self, instr):
        self._stack[-1].append(instr)

    def _fresh(self, p="v"):
        self._n += 1
        return f"{p}{self._n}"

    # -- spec / params --
    def spec(self, name):
        return IExpr({"spec": name})

    def idiv(self, a, b):
        return IExpr({"div": [_to_intexpr(a), _to_intexpr(b)]})

    def imul(self, a, b):
        return IExpr({"mul": [_to_intexpr(a), _to_intexpr(b)]})

    def param(self, name, type_json, **opts):
        self._emit(
            {
                "op": "param",
                "name": name,
                "type": type_json,
                "bind": name,
                "attrs": opts,
            }
        )
        return V(name, type_json)

    # -- constants / ids --
    def const_i32(self, val):
        r = self._fresh("c")
        self._emit(
            {
                "op": "emit",
                "opcode": "arith.constant",
                "out": {"bind": r, "type": "i32"},
                "attrs": {"ity": _s("i32"), "value": {"t": "i", "v": _to_intexpr(val)}},
            }
        )
        return V(r, "i32")

    def const_f32(self, val):
        r = self._fresh("c")
        self._emit(
            {
                "op": "emit",
                "opcode": "arith.constant",
                "out": {"bind": r, "type": "f32"},
                "attrs": {"ity": _s("f32"), "value": _f(val)},
            }
        )
        return V(r, "f32")

    def thread_id_x(self):
        r = self._fresh("tid")
        self._emit({"op": "thread_id_x", "bind": r})
        return V(r, "i32")

    # -- generic ops (result type inferred) --
    def _emit_op(self, opcode, ins, rtype, attrs=None, hint="v"):
        r = self._fresh(hint)
        node = {
            "op": "emit",
            "opcode": opcode,
            "in": [v.reg for v in ins],
            "out": {"bind": r, "type": rtype},
        }
        if attrs:
            node["attrs"] = attrs
        self._emit(node)
        return V(r, rtype)

    def add(self, a, b):
        return self._emit_op("arith.add", [a, b], a.type)

    def mul(self, a, b):
        return self._emit_op("arith.mul", [a, b], a.type)

    def cast_to_f32(self, a):
        return self._emit_op("arith.cast_to_f32", [a], "f32")

    def load_vN(self, ptr, idx, elem, vec, align=16):
        rtype = {"kind": "vector", "elem": elem, "count": vec}
        return self._emit_op(
            "memref.global_load_vN",
            [ptr, idx],
            rtype,
            {"align": _i(align), "elem_type": _s(elem), "vec": _i(vec)},
        )

    def vec_extract(self, v, i):
        elem = v.type["elem"] if isinstance(v.type, dict) else v.type
        return self._emit_op("vector.extract", [v], elem, {"index": _i(i)})

    def fadd(self, a, b):
        return self._emit_op("arith.fadd", [a, b], a.type)

    def fsub(self, a, b):
        return self._emit_op("arith.fsub", [a, b], a.type)

    def fmul(self, a, b):
        return self._emit_op("arith.fmul", [a, b], a.type)

    def fmax(self, a, b):
        return self._emit_op("arith.fmax", [a, b], a.type)

    def exp2(self, a):
        return self._emit_op("math.exp2", [a], a.type)

    def rcp(self, a):
        return self._emit_op("math.rcp", [a], a.type)

    def cmp_gt(self, a, b):
        return self._emit_op("arith.cmp", [a, b], "i1", {"pred": _s("gt")})

    def load(self, ptr, idx, elem="f32", align=4):
        return self._emit_op(
            "memref.global_load_typed",
            [ptr, idx],
            elem,
            {"align": _i(align), "elem_type": _s(elem)},
        )

    def store(self, ptr, idx, val, elem="f32", align=4):
        self._emit(
            {
                "op": "emit",
                "opcode": "memref.global_store_typed",
                "in": [ptr.reg, idx.reg, val.reg],
                "attrs": {"align": _i(align), "elem_type": _s(elem)},
            }
        )

    def yield_(self, *vals):
        self._emit(
            {
                "op": "emit",
                "opcode": "scf.yield",
                "in": [v.reg for v in vals],
                "attrs": {"num": _i(len(vals))},
            }
        )

    def ret(self):
        self._emit({"op": "ret"})

    # -- control flow primitives (keep the recipe parametric) --
    @contextmanager
    def static_for(self, var, lo, hi, step=1):
        body = []
        self._stack.append(body)
        try:
            yield IExpr({"var": var})
        finally:
            self._stack.pop()
            self._emit(
                {
                    "op": "static_for",
                    "var": var,
                    "lo": _to_intexpr(lo),
                    "hi": _to_intexpr(hi),
                    "step": _to_intexpr(step),
                    "body": body,
                }
            )

    def scf_for(self, lo, hi, step, iters, iv="k"):
        return _ForRec(self, lo, hi, step, iters, iv)

    def static_for_acc(self, var, lo, hi, carried, step=1):
        """Compile-time (rolled) loop with loop-carried accumulators. The carry
        rides a stable register name across the VM's expansion iterations:
        a leading alias binds the carry to its init before the loop, and a
        trailing alias (inside the body) rebinds it each iteration. Neither
        alias emits IR, so the VM-expanded ops match a Python-time unroll
        exactly (byte-identical HSACO)."""
        return _StaticForRec(self, var, lo, hi, step, carried)

    @contextmanager
    def scf_if(self, cond):
        body = []
        self._stack.append(body)
        try:
            yield
        finally:
            self._stack.pop()
            self._emit({"op": "scf_if", "cond": cond.reg, "then": body})

    def static_if(self, pred, then_fn, else_fn):
        """Compile-time branch; BOTH arms are recorded (the VM picks one per
        spec at JIT time). Each arm's result is aliased to a common output."""
        out = self._fresh("out")
        tbody = []
        self._stack.append(tbody)
        rt = then_fn()
        self._emit({"op": "alias", "bind": out, "from": rt.reg})
        self._stack.pop()
        ebody = []
        self._stack.append(ebody)
        re = else_fn()
        self._emit({"op": "alias", "bind": out, "from": re.reg})
        self._stack.pop()
        self._emit(
            {"op": "static_if", "pred": _to_intexpr(pred), "then": tbody, "else": ebody}
        )
        return V(out, rt.type)

    def recipe(self):
        return {
            "schema": "rocke.recipe/v1",
            "kernel_name_fmt": self._fmt,
            "spec": self._spec,
            "attrs": {k: _i(v) for k, v in self._kattrs.items()},
            "program": self._stack[0],
        }


class _StaticForRec:
    def __init__(self, rec, var, lo, hi, step, carried):
        self.rec, self.var, self.lo, self.hi, self.step = rec, var, lo, hi, step
        self.carried = carried  # list of (name, init V)
        self._finals = {}
        self.results = None

    def __enter__(self):
        # Establish the stable carry register names in the outer scope.
        for name, init in self.carried:
            self.rec._emit({"op": "alias", "bind": name, "from": init.reg})
        self._body = []
        self.rec._stack.append(self._body)
        ivar = IExpr({"var": self.var})
        handles = [V(name, init.type) for name, init in self.carried]
        return ivar, handles

    def set_carry(self, name, val):
        self._finals[name] = val

    def __exit__(self, *exc):
        # Trailing alias inside the body: rebind each carry for the next iter.
        for name, init in self.carried:
            fv = self._finals.get(name)
            if fv is not None:
                self.rec._emit({"op": "alias", "bind": name, "from": fv.reg})
        self.rec._stack.pop()
        self.rec._emit(
            {
                "op": "static_for",
                "var": self.var,
                "lo": _to_intexpr(self.lo),
                "hi": _to_intexpr(self.hi),
                "step": _to_intexpr(self.step),
                "body": self._body,
            }
        )
        self.results = [V(name, init.type) for name, init in self.carried]
        return False


class _ForRec:
    def __init__(self, rec, lo, hi, step, iters, iv):
        self.rec, self.lo, self.hi, self.step, self.iters, self.iv = (
            rec,
            lo,
            hi,
            step,
            iters,
            iv,
        )
        self.results = None

    def __enter__(self):
        self._body = []
        self.rec._stack.append(self._body)
        iv_h = V(self.iv, "i32")
        iter_hs = [V(name, init.type) for name, init in self.iters]
        return iv_h, iter_hs

    def __exit__(self, *exc):
        self.rec._stack.pop()
        res = [self.rec._fresh("for") for _ in self.iters]
        self.rec._emit(
            {
                "op": "scf_for",
                "iv": self.iv,
                "lo": self.lo.reg,
                "hi": self.hi.reg,
                "step": self.step.reg,
                "iter": [{"name": name, "init": init.reg} for name, init in self.iters],
                "results": res,
                "unroll": False,
                "elide_trailing_barrier": True,
                "body": self._body,
            }
        )
        self.results = [V(r, init.type) for r, (_, init) in zip(res, self.iters)]
        return False


# ---------------------------------------------------------------------------
# mini_attn authored idiomatically against the recorder. Compare this to the
# hand-written recipe in mini_attn.py::make_recipe -- the authoring delta vs a
# normal builder is only: rec.* methods, scf_for via `with ... as (k, iters)`,
# static_if via two arm callables, and explicit yield_.
# ---------------------------------------------------------------------------
def record_mini_attn():
    rec = RecipeRecorder(
        "rocke_mini_attn_norm{use_norm}_{dtype}",
        spec=[{"name": "use_norm", "kind": "int"}, {"name": "dtype", "kind": "str"}],
        kattrs={"max_workgroup_size": 64},
    )
    Q = rec.param("Q", PTR_F32, noalias=True, readonly=True, align=16)
    O = rec.param("O", PTR_F32, noalias=True, writeonly=True, align=16)
    n = rec.param("n", "i32")
    tid = rec.thread_id_x()
    m0 = rec.const_f32(-1e30)
    l0 = rec.const_f32(0.0)
    acc0 = rec.const_f32(0.0)
    lo = rec.const_i32(0)
    step = rec.const_i32(1)
    loop = rec.scf_for(lo, n, step, [("m", m0), ("l", l0), ("acc", acc0)], iv="k")
    with loop as (k, (m, l, acc)):
        idx = rec.add(tid, k)
        x = rec.load(Q, idx)
        mnew = rec.fmax(m, x)
        p = rec.exp2(rec.fsub(x, mnew))
        c = rec.exp2(rec.fsub(m, mnew))
        lnew = rec.fadd(rec.fmul(l, c), p)
        accnew = rec.fadd(rec.fmul(acc, c), rec.fmul(p, x))
        rec.yield_(mnew, lnew, accnew)
    accf = loop.results[2]
    lf = loop.results[1]
    out = rec.static_if(
        rec.spec("use_norm"),
        then_fn=lambda: rec.fmul(accf, rec.rcp(lf)),
        else_fn=lambda: accf,
    )
    z = rec.const_i32(0)
    cond = rec.cmp_gt(n, z)
    with rec.scf_if(cond):
        rec.store(O, tid, out)
    rec.ret()
    return rec.recipe()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", choices=["recipe"], default="recipe")
    ap.parse_args()
    sys.stdout.write(json.dumps(record_mini_attn(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
