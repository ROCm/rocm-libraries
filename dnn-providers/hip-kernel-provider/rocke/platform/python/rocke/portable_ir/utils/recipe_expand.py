# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# recipe_expand.py -- a pure-Python expander for builder recipes
# (schema rocke.recipe/v1), mirroring the C recipe VM (recipe_vm.c) semantics:
# it replays a recipe against concrete spec values to produce a FLAT concrete
# recipe (== portable IR for that shape).
#
# Why this exists: it is the device-free correctness oracle for rolling. The
# rolling pipeline records concrete traces, infers a parametric recipe, then must
# prove the parametric recipe reproduces each concrete shape. Expanding the
# parametric recipe at a spec point and checking it is structurally identical
# (modulo SSA renaming) to the independently-recorded concrete recipe at that
# point establishes byte-identity WITHOUT the C VM / comgr: concrete recipe ->
# byte-identical HSACO is already proven, and identical op streams (up to SSA
# alpha-renaming) lower to identical HSACO.
#
# Supported instrs (same set as recipe_vm.c): param, const_i32, const_f32,
# thread_id_x, alias, ret, static_for, static_if, scf_for, scf_if, emit.
# intexpr: number | {spec} | {var} | {spec_str_eq:[n,lit]} | {OP:[a,b]} for
# OP in add/sub/mul/div/mod/eq/ne/lt/le/gt/ge.

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from rocke.portable_ir.src import abi as _abi

RECIPE_SCHEMA = "rocke.recipe/v1"


# --------------------------------------------------------------------------
# intexpr evaluation (mirrors rv_int in recipe_vm.c)
# --------------------------------------------------------------------------
_BIN = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: (a // b if b else 0),
    "mod": lambda a, b: (a % b if b else 0),
    "eq": lambda a, b: int(a == b),
    "ne": lambda a, b: int(a != b),
    "lt": lambda a, b: int(a < b),
    "le": lambda a, b: int(a <= b),
    "gt": lambda a, b: int(a > b),
    "ge": lambda a, b: int(a >= b),
}


class ExpandError(RuntimeError):
    pass


class GuardRejected(ExpandError):
    """A binding of a rolled recipe's free axes that the recipe's guard refuses.

    Distinct from ExpandError because it is not a malformed recipe or a bug: it
    is the expected answer for a configuration the kernel never supported. A JIT
    caller should treat it as "this recipe does not serve that shape", not as an
    engine failure."""


def magic_division_constants(divisor: int) -> tuple:
    """`(multiplier_i32, shift)` for a strength-reduced unsigned `n // divisor`.

    The canonical mirror of `helpers/transforms.py::calculate_magic_numbers` plus
    `do_magic_division`'s two's-complement wrap (itself a port of CK Tile's
    `magic_division32_bit_range`). Kernels bake these two integers in as
    `const_i32` operands of `(umul_hi(n, M) + n) >> s`, so a recipe that wants to
    stay parametric in the divisor has to regenerate them rather than fit them --
    the shift is logarithmic and the multiplier depends on the divisor's odd part.

    `recipe_vm.cpp` implements the same two lines; `roller.py` imports this for
    recognition. A test pins all three against the DSL helper.
    """
    # The upper bound keeps this inside the int64 range the C VM computes in, so
    # both mirrors agree everywhere they are defined.
    if divisor < 1 or divisor > 0x7FFFFFFF:
        raise ExpandError(f"magic division needs 1 <= divisor < 2^31, got {divisor}")
    shift = 0
    while (1 << shift) < divisor:
        shift += 1
    mult = (((1 << shift) - divisor) << 32) // divisor + 1
    return (mult - (1 << 32) if mult >= (1 << 31) else mult), shift


_UN = {
    "magic_multiplier": lambda d: magic_division_constants(d)[0],
    "magic_shift": lambda d: magic_division_constants(d)[1],
}


def eval_intexpr(
    node: Any, ivars: Dict[str, int], spec_int: Dict[str, int], spec_str: Dict[str, str]
) -> int:
    if isinstance(node, bool):
        return int(node)
    if isinstance(node, (int,)):
        return node
    if isinstance(node, float):
        return int(node)
    if isinstance(node, dict):
        if "spec" in node:
            name = node["spec"]
            if name not in spec_int:
                raise ExpandError(f"unknown spec int '{name}'")
            return spec_int[name]
        if "var" in node:
            name = node["var"]
            if name not in ivars:
                raise ExpandError(f"unknown loop var '{name}'")
            return ivars[name]
        if "spec_str_eq" in node:
            n, lit = node["spec_str_eq"]
            return int(spec_str.get(n) == lit)
        # unary functions take the operand directly, not a 2-element array
        for k, ufn in _UN.items():
            if k in node:
                return ufn(eval_intexpr(node[k], ivars, spec_int, spec_str))
        for k, fn in _BIN.items():
            if k in node:
                a, b = node[k]
                return fn(
                    eval_intexpr(a, ivars, spec_int, spec_str),
                    eval_intexpr(b, ivars, spec_int, spec_str),
                )
    raise ExpandError(f"bad intexpr: {node!r}")


# --------------------------------------------------------------------------
# guard evaluation (mirrors rocke_guard_check in recipe_vm.cpp)
# --------------------------------------------------------------------------
GUARD_SCHEMA = "rocke.guard/v1"


def check_guard(
    guard: Optional[Dict[str, Any]],
    spec_int: Dict[str, int],
    spec_str: Dict[str, str],
    *,
    require_verified: bool = False,
) -> "tuple[bool, str]":
    """Is this binding of a rolled recipe's free axes one the kernel supports?

    Returns `(ok, reason)`; `reason` is '' when ok. A recipe with no guard is
    accepted -- guards are additive, and every recipe recorded before they
    existed replays exactly as it did before.

    Three checks, in this order, because each one makes the next one safe:

      1. Every axis named in `free` is bound. An unbound axis would otherwise
         make the rules below raise rather than decide, and "the caller forgot
         an axis" is a different answer from "this shape is unsupported".
      2. `rules` in order, first failure wins. The order is load-bearing, not
         cosmetic -- see below.
      3. `require_verified` (opt-in) additionally demands an exact match against
         a point the generator actually built and compared. That is the strict
         policy: it trades the whole rolled interior for "every accepted point
         was verified byte-for-byte at generation time".

    On negative inputs: `mod` and `div` do not agree between this evaluator and
    the C one for a negative left operand (Python floors, C truncates), which
    has never mattered because spec values are sizes -- and a guard is the first
    thing to be handed a hostile one. Two properties keep it from mattering
    here, and a new kind of rule has to preserve both. Guards only ever test
    `mod(x, k) == 0`, and whether k divides x is the same question in either
    convention; and the emitter puts a bounds rule ahead of the divisibility
    rule on an axis, which combined with stopping at the first failure means a
    negative never reaches the `mod` at all. A rule that used `div`, or compared
    `mod` against something other than zero, would have neither protection."""
    if not guard:
        return True, ""
    schema = guard.get("schema")
    if schema != GUARD_SCHEMA:
        return False, f"unknown guard schema {schema!r} (want {GUARD_SCHEMA})"

    for axis in guard.get("free", []):
        if axis not in spec_int and axis not in spec_str:
            return False, f"free axis '{axis}' not bound"

    for rule in guard.get("rules", []):
        try:
            ok = eval_intexpr(rule["pred"], {}, spec_int, spec_str)
        except ExpandError as e:
            return False, f"guard rule failed to evaluate: {e}"
        if not ok:
            return False, rule.get("reason", "guard rule rejected")

    if require_verified:
        pts = guard.get("verified", [])
        if not pts:
            return False, "require_verified set but guard carries no verified points"
        bound = {**spec_int, **spec_str}
        if not any(all(bound.get(k) == v for k, v in p.items()) for p in pts):
            return False, "binding is not one of the generator-verified points"
    return True, ""


# --------------------------------------------------------------------------
# expander
# --------------------------------------------------------------------------
class _Expander:
    def __init__(self, spec_int: Dict[str, int], spec_str: Dict[str, str]):
        self.spec_int = spec_int
        self.spec_str = spec_str
        self.env: Dict[str, str] = {}  # template reg name -> emitted reg name
        self.ivars: Dict[str, int] = {}  # loop var -> value (flat; static_for scopes)
        self.out: List[Dict[str, Any]] = []
        self._n = 0

    def _fresh(self, hint: str = "e") -> str:
        self._n += 1
        return f"{hint}{self._n}"

    def _subst(self, name: str) -> str:
        """Substitute {var} loop-index tokens in a register name (parametric
        names, e.g. 'acc_m{lane}_n0' -> 'acc_m2_n0' when lane==2)."""
        if isinstance(name, str) and "{" in name:
            return name.format(**self.ivars)
        return name

    def _resolve(self, name: str) -> str:
        name = self._subst(name)
        if name not in self.env:
            raise ExpandError(f"unresolved register '{name}'")
        return self.env[name]

    def _expand_name_list(self, items: List[Any]) -> List[str]:
        """Flatten a list whose entries are register names (str) or rolled groups
        {"for": {var,lo,hi,step}, "name": "r{var}"} -> list of substituted names."""
        out: List[str] = []
        for it in items:
            if isinstance(it, dict) and "for" in it:
                fr = it["for"]
                var = fr["var"]
                saved = self.ivars.get(var)
                v = self._eval(fr["lo"])
                hi = self._eval(fr["hi"])
                step = self._eval(fr.get("step", 1)) or 1
                while v < hi:
                    self.ivars[var] = v
                    out.append(self._subst(it["name"]))
                    v += step
                if saved is None:
                    self.ivars.pop(var, None)
                else:
                    self.ivars[var] = saved
            else:
                out.append(self._subst(it))
        return out

    def _expand_iter_list(self, items: List[Any]) -> List[Dict[str, str]]:
        """Like _expand_name_list but each entry also carries an init register;
        returns [{"name": substituted, "init": substituted}, ...]."""
        out: List[Dict[str, str]] = []
        for it in items:
            if "for" in it:
                fr = it["for"]
                var = fr["var"]
                saved = self.ivars.get(var)
                v = self._eval(fr["lo"])
                hi = self._eval(fr["hi"])
                step = self._eval(fr.get("step", 1)) or 1
                while v < hi:
                    self.ivars[var] = v
                    out.append({"name": self._subst(it["name"]), "init": it["init"]})
                    v += step
                if saved is None:
                    self.ivars.pop(var, None)
                else:
                    self.ivars[var] = saved
            else:
                out.append({"name": it["name"], "init": it["init"]})
        return out

    def _eval(self, node: Any) -> int:
        return eval_intexpr(node, self.ivars, self.spec_int, self.spec_str)

    def _eval_type(self, t: Any) -> Any:
        """Materialize a result type, evaluating any intexpr in integer fields
        (e.g. a parametric smem buffer `shape: [{spec:TN}, 16]`)."""
        if isinstance(t, dict):
            # An intexpr embedded as a type field (spec/var/arithmetic) -> int.
            if any(
                k in t
                for k in (
                    "spec",
                    "var",
                    "spec_str_eq",
                    "add",
                    "sub",
                    "mul",
                    "div",
                    "mod",
                )
            ):
                return self._eval(t)
            return {k: self._eval_type(v) for k, v in t.items()}
        if isinstance(t, list):
            return [self._eval_type(v) for v in t]
        return t

    def _eval_attrs(self, attrs: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not attrs:
            return attrs
        out: Dict[str, Any] = {}
        for k, tv in attrs.items():
            if isinstance(tv, dict) and tv.get("t") == "i":
                out[k] = {"t": "i", "v": self._eval(tv["v"])}
            else:
                out[k] = copy.deepcopy(tv)
        return out

    def run(self, program: List[Dict[str, Any]]) -> None:
        for instr in program:
            self._instr(instr)

    def _instr(self, instr: Dict[str, Any]) -> None:
        op = instr["op"]
        if op == "param":
            name = instr["name"]
            self.env[instr.get("bind", name)] = name  # params keep their name
            self.out.append(
                {
                    "op": "param",
                    "name": name,
                    "type": instr["type"],
                    "bind": name,
                    "attrs": dict(instr.get("attrs", {})),
                }
            )
            return
        if op == "const_i32":
            r = self._fresh("c")
            self.env[instr["bind"]] = r
            self.out.append(
                {
                    "op": "emit",
                    "opcode": "arith.constant",
                    "out": {"bind": r, "type": "i32"},
                    "attrs": {
                        "ity": {"t": "s", "v": "i32"},
                        "value": {"t": "i", "v": self._eval(instr["val"])},
                    },
                }
            )
            return
        if op == "const_f32":
            r = self._fresh("c")
            self.env[instr["bind"]] = r
            self.out.append(
                {
                    "op": "emit",
                    "opcode": "arith.constant",
                    "out": {"bind": r, "type": "f32"},
                    "attrs": {
                        "ity": {"t": "s", "v": "f32"},
                        "value": {"t": "f", "v": instr["fval"]},
                    },
                }
            )
            return
        if op == "thread_id_x":
            r = self._fresh("tid")
            self.env[instr["bind"]] = r
            self.out.append({"op": "thread_id_x", "bind": r})
            return
        if op == "alias":
            self.env[self._subst(instr["bind"])] = self._resolve(instr["from"])
            return
        if op == "ret":
            self.out.append({"op": "ret"})
            return
        if op == "static_for":
            var = instr["var"]
            lo, hi = self._eval(instr["lo"]), self._eval(instr["hi"])
            step = self._eval(instr.get("step", 1)) or 1
            saved = self.ivars.get(var, None)
            iv = lo
            while iv < hi:
                self.ivars[var] = iv
                self.run(instr["body"])
                iv += step
            if saved is None:
                self.ivars.pop(var, None)
            else:
                self.ivars[var] = saved
            return
        if op == "static_if":
            pred = self._eval(instr["pred"])
            arm = instr.get("then") if pred else instr.get("else")
            if arm:
                self.run(arm)
            return
        if op == "scf_for":
            iv = self._fresh("iv")
            lo, hi, step = (
                self._resolve(instr["lo"]),
                self._resolve(instr["hi"]),
                self._resolve(instr["step"]),
            )
            # iter-args and results may be parametric (rolled groups + format
            # names) -> a spec-derived NUMBER of loop-carries (the "fan").
            iters = []
            for m in self._expand_iter_list(instr.get("iter", [])):
                nm = self._fresh("it")
                iters.append(
                    {"name": nm, "init": self._resolve(m["init"]), "_tmpl": m["name"]}
                )
            result_tmpls = self._expand_name_list(instr.get("results", []))
            results = [self._fresh("for") for _ in result_tmpls]
            # Bind iv + iter names inside the loop scope (VM uses flat regs).
            self.env[instr["iv"]] = iv
            for m in iters:
                self.env[m["_tmpl"]] = m["name"]
            body_expander = _Expander(self.spec_int, self.spec_str)
            body_expander.env = self.env
            body_expander.ivars = self.ivars
            body_expander._n = self._n
            body_expander.out = []
            body_expander.run(instr["body"])
            self._n = body_expander._n
            for tmpl, r in zip(result_tmpls, results):
                self.env[tmpl] = r
            self.out.append(
                {
                    "op": "scf_for",
                    "iv": iv,
                    "lo": lo,
                    "hi": hi,
                    "step": step,
                    "iter": [{"name": m["name"], "init": m["init"]} for m in iters],
                    "results": results,
                    "unroll": bool(instr.get("unroll", False)),
                    "elide_trailing_barrier": bool(
                        instr.get("elide_trailing_barrier", True)
                    ),
                    "body": body_expander.out,
                }
            )
            return
        if op == "scf_if":
            cond = self._resolve(instr["cond"])
            body_expander = _Expander(self.spec_int, self.spec_str)
            body_expander.env = self.env
            body_expander.ivars = self.ivars
            body_expander._n = self._n
            body_expander.out = []
            body_expander.run(instr.get("then", []))
            self._n = body_expander._n
            self.out.append({"op": "scf_if", "cond": cond, "then": body_expander.out})
            return
        if op == "emit":
            # Operands may be a rolled list (e.g. scf.yield carrying a fan of
            # loop-carries) and/or format names.
            ins = [
                self._resolve(n) for n in self._expand_name_list(instr.get("in", []))
            ]
            node: Dict[str, Any] = {"op": "emit", "opcode": instr["opcode"], "in": ins}
            if "out" in instr:
                r = self._fresh("v")
                self.env[self._subst(instr["out"]["bind"])] = r
                node["out"] = {"bind": r, "type": self._eval_type(instr["out"]["type"])}
            elif "outs" in instr:
                outs = []
                for o in instr["outs"]:
                    r = self._fresh("v")
                    self.env[self._subst(o["bind"])] = r
                    outs.append({"bind": r, "type": self._eval_type(o["type"])})
                node["outs"] = outs
            attrs = self._eval_attrs(instr.get("attrs"))
            if attrs:
                node["attrs"] = attrs
            self.out.append(node)
            return
        raise ExpandError(f"unknown instr op '{op}'")


def expand_recipe(
    recipe: Dict[str, Any],
    spec: Dict[str, Any],
    *,
    enforce_guard: bool = True,
    require_verified: bool = False,
) -> Dict[str, Any]:
    """Replay `recipe` at concrete spec values -> a flat concrete recipe.

    `spec` maps spec-axis name -> int or str value. Result mirrors what the C VM
    would build (one shape's portable IR), with freshly generated SSA names.

    If the recipe carries a guard it is checked first, before any op is emitted,
    and a refused binding raises GuardRejected -- the same decision the C VM
    makes at the same point, so the two paths agree on which shapes are servable
    as well as on the IR they produce. `enforce_guard=False` is for the
    generator, which has to replay candidate points in order to find out what
    the guard should say.

    `require_verified` narrows acceptance to the points the generator built and
    compared; see check_guard.

    Admission mirrors the C VM's, in the same order and for the same reasons:
    the wire ABI level first (so an artifact from a newer generator says so
    rather than looking corrupt), then the schema, then the guard. Until this
    was added the two engines disagreed about what they would accept -- the C VM
    checked the schema and this expander checked nothing, so it would happily
    replay a recipe the engine it is supposed to mirror would refuse, which is
    the wrong way round for an oracle."""
    _abi.check(recipe)
    schema = recipe.get("schema")
    if schema != RECIPE_SCHEMA:
        raise ExpandError(f"bad/missing schema {schema!r} (want {RECIPE_SCHEMA})")
    if enforce_guard:
        ok, why = check_guard(
            recipe.get("guard"),
            {k: int(v) for k, v in spec.items() if not isinstance(v, str)},
            {k: v for k, v in spec.items() if isinstance(v, str)},
            require_verified=require_verified,
        )
        if not ok:
            raise GuardRejected(why)
    spec_int = {k: int(v) for k, v in spec.items() if not isinstance(v, str)}
    spec_str = {k: v for k, v in spec.items() if isinstance(v, str)}
    ex = _Expander(spec_int, spec_str)
    ex.run(recipe["program"])
    return {
        "schema": RECIPE_SCHEMA,
        "kernel_name_fmt": recipe.get("kernel_name_fmt", ""),
        "spec": [],
        "attrs": recipe.get("attrs", {}),
        "program": ex.out,
    }


# --------------------------------------------------------------------------
# structural equivalence modulo SSA renaming (the oracle comparison)
# --------------------------------------------------------------------------
class _Equiv:
    def __init__(self) -> None:
        self.fwd: Dict[str, str] = {}  # a-name -> b-name
        self.rev: Dict[str, str] = {}  # b-name -> a-name
        self.why = ""

    def _bind(self, a: str, b: str) -> bool:
        if a in self.fwd:
            if self.fwd[a] != b:
                self.why = f"def {a} bound to {self.fwd[a]}, now {b}"
                return False
        if b in self.rev and self.rev[b] != a:
            self.why = f"def {b} already maps from {self.rev[b]}, now {a}"
            return False
        self.fwd[a] = b
        self.rev[b] = a
        return True

    def _use(self, a: str, b: str) -> bool:
        if self.fwd.get(a) != b:
            self.why = f"use {a}->{self.fwd.get(a)} != {b}"
            return False
        return True

    def lists(self, la: List[Dict[str, Any]], lb: List[Dict[str, Any]]) -> bool:
        if len(la) != len(lb):
            self.why = f"length {len(la)} != {len(lb)}"
            return False
        return all(self.instr(a, b) for a, b in zip(la, lb))

    def instr(self, a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        if a.get("op") != b.get("op"):
            self.why = f"op {a.get('op')} != {b.get('op')}"
            return False
        op = a["op"]
        if op == "param":
            if (
                a["name"] != b["name"]
                or a["type"] != b["type"]
                or a.get("attrs", {}) != b.get("attrs", {})
            ):
                self.why = f"param mismatch {a['name']}"
                return False
            return self._bind(a.get("bind", a["name"]), b.get("bind", b["name"]))
        if op == "ret":
            return True
        if op == "thread_id_x":
            return self._bind(a["bind"], b["bind"])
        if op == "emit":
            if a["opcode"] != b["opcode"]:
                self.why = f"opcode {a['opcode']} != {b['opcode']}"
                return False
            ai, bi = a.get("in", []), b.get("in", [])
            if len(ai) != len(bi) or not all(self._use(x, y) for x, y in zip(ai, bi)):
                self.why = self.why or f"operands of {a['opcode']}"
                return False
            if a.get("attrs", {}) != b.get("attrs", {}):
                self.why = (
                    f"attrs of {a['opcode']}: {a.get('attrs')} != {b.get('attrs')}"
                )
                return False
            if ("out" in a) != ("out" in b) or ("outs" in a) != ("outs" in b):
                self.why = f"result arity of {a['opcode']}"
                return False
            if "out" in a:
                if a["out"]["type"] != b["out"]["type"]:
                    self.why = f"result type of {a['opcode']}"
                    return False
                return self._bind(a["out"]["bind"], b["out"]["bind"])
            if "outs" in a:
                if len(a["outs"]) != len(b["outs"]):
                    return False
                for oa, ob in zip(a["outs"], b["outs"]):
                    if oa["type"] != ob["type"] or not self._bind(
                        oa["bind"], ob["bind"]
                    ):
                        return False
            return True
        if op == "scf_for":
            for key in ("lo", "hi", "step"):
                if not self._use(a[key], b[key]):
                    return False
            ia, ib = a.get("iter", []), b.get("iter", [])
            if len(ia) != len(ib):
                self.why = "scf_for iter arity"
                return False
            for ma, mb in zip(ia, ib):
                if not self._use(ma["init"], mb["init"]) or not self._bind(
                    ma["name"], mb["name"]
                ):
                    return False
            if not self._bind(a["iv"], b["iv"]):
                return False
            ra, rb = a.get("results", []), b.get("results", [])
            if len(ra) != len(rb) or not all(self._bind(x, y) for x, y in zip(ra, rb)):
                self.why = self.why or "scf_for results"
                return False
            return self.lists(a["body"], b["body"])
        if op == "scf_if":
            if not self._use(a["cond"], b["cond"]):
                return False
            return self.lists(a.get("then", []), b.get("then", []))
        self.why = f"unhandled op {op}"
        return False


def recipes_equiv(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """True iff the two flat concrete recipe programs are identical up to a
    consistent renaming of SSA registers (param names are NOT renamed). This is
    the byte-identity proxy for lowered HSACO."""
    return _Equiv().lists(a["program"], b["program"])


def equiv_reason(a: Dict[str, Any], b: Dict[str, Any]) -> str:
    """'' if equivalent, else a short reason for the first mismatch."""
    eq = _Equiv()
    return "" if eq.lists(a["program"], b["program"]) else (eq.why or "mismatch")
