# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# roller.py -- a general, multi-trace recipe roller.
#
# Input: several CONCRETE recipes (the same kernel recorded at different values
# of ONE structural spec axis, e.g. head_size D in {64,128,256}).
# Output: ONE PARAMETRIC recipe (using static_for + spec/var intexprs + the
# carry-alias trick) that, when expanded by recipe_expand at any D, reproduces
# the concrete recipe for that D -- byte-identically (modulo SSA names).
#
# It generalizes the bespoke attention head_size roller (roll_recipe.py) to:
#   - any single structural axis (not just head_size),
#   - nested rolling (the run may live inside an scf.for / scf.if body),
#   - multiple index-ladder constants (v0 + i*delta) per block,
#   - multiple loop-carried values (cross-block def->use), threaded via aliases,
#   - spec-scaled constants inside and outside the run (c = m*axis + k).
#
# SAFE BY CONSTRUCTION: every candidate roll is verified with the recipe_expand
# oracle against *all* input traces (and the caller adds held-out points). If a
# region can't be rolled or a roll fails verification, it FALLS BACK to the
# concrete sub-program -- correct, just not compressed. The roller never emits a
# wrong recipe; at worst it fails to compress.

from __future__ import annotations

import copy
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

from ..utils.recipe_expand import magic_division_constants


# --------------------------------------------------------------------------
# instruction helpers (operate on recipe instr dicts)
# --------------------------------------------------------------------------
def _defs(instr: Dict[str, Any]) -> List[str]:
    op = instr["op"]
    if op == "emit":
        if "out" in instr:
            return [instr["out"]["bind"]]
        if "outs" in instr:
            return [o["bind"] for o in instr["outs"]]
        return []
    if op in ("const_i32", "const_f32", "thread_id_x", "alias"):
        return [instr["bind"]]
    if op == "param":
        return [instr.get("bind", instr["name"])]
    if op == "scf_for":
        return list(instr.get("results", []))
    return []


def _use_slots(instr: Dict[str, Any]):
    """Yield (getter, setter) for each operand-name slot, so we can read/rewrite
    operands uniformly across instruction kinds."""
    op = instr["op"]
    if op == "emit":
        for i in range(len(instr.get("in", []))):
            yield (
                lambda i=i: instr["in"][i],
                lambda v, i=i: instr["in"].__setitem__(i, v),
            )
    elif op == "scf_for":
        for key in ("lo", "hi", "step"):
            yield (
                lambda key=key: instr[key],
                lambda v, key=key: instr.__setitem__(key, v),
            )
        for m in instr.get("iter", []):
            yield (lambda m=m: m["init"], lambda v, m=m: m.__setitem__("init", v))
    elif op == "scf_if":
        yield (lambda: instr["cond"], lambda v: instr.__setitem__("cond", v))
    elif op == "alias":
        yield (lambda: instr["from"], lambda v: instr.__setitem__("from", v))


def _uses(instr: Dict[str, Any]) -> List[str]:
    return [g() for g, _ in _use_slots(instr)]


def _rewrite_uses(instr: Dict[str, Any], mapping: Dict[str, str]) -> None:
    for get, setv in _use_slots(instr):
        v = get()
        if isinstance(v, str) and v in mapping:  # skip rolled-group dicts
            setv(mapping[v])


def _is_const_int(instr: Dict[str, Any]) -> bool:
    return (
        instr.get("op") == "emit"
        and instr.get("opcode") == "arith.constant"
        and isinstance(instr.get("attrs", {}).get("value", {}).get("v"), int)
    )


def _const_int(instr: Dict[str, Any]) -> int:
    return instr["attrs"]["value"]["v"]


def _set_const_int(instr: Dict[str, Any], expr: Any) -> None:
    instr["attrs"]["value"] = {"t": "i", "v": expr}


def _sig(instr: Dict[str, Any]) -> Tuple:
    """Structural signature ignoring SSA names and int constant *values*."""
    op = instr["op"]
    if op == "emit":
        akeys = tuple(sorted(instr.get("attrs", {}).keys()))
        return (
            op,
            instr["opcode"],
            len(instr.get("in", [])),
            ("out" in instr),
            len(instr.get("outs", [])),
            akeys,
        )
    if op == "scf_for":
        return (op, len(instr.get("iter", [])), len(instr.get("results", [])))
    return (op,)


# --------------------------------------------------------------------------
# linear inference (single axis): value = m*axis + k from two (axis,val) points
# --------------------------------------------------------------------------
def _linear_expr(
    axis: str, a0: int, v0: int, a1: int, v1: int, var: bool = False
) -> Optional[Any]:
    """Return an intexpr in {spec:axis} (or {var:axis} when var=True) that yields
    v0 at a0 and v1 at a1.

    Linear first; failing that, the candidate models in `fit_slot` get a look, so
    the structural roller can also express a block COUNT that is reciprocal in the
    axis (`512 div block_n`) or an operand of a strength-reduced division. Loop
    variables (`var=True`) stay linear-only: those are index ladders, where the
    richer forms have no meaning."""
    expr = _linear_only(axis, a0, v0, a1, v1, var)
    if expr is not None or var:
        return expr
    for cand in (_fit_parameter_free, _fit_reciprocal):
        got = cand([axis], [(a0,), (a1,)], [v0, v1])
        if got is not None:
            return got
    return None


def _linear_only(
    axis: str, a0: int, v0: int, a1: int, v1: int, var: bool = False
) -> Optional[Any]:
    """The strictly linear core: v = m*axis + k, or None."""
    ref = {"var": axis} if var else {"spec": axis}
    if v0 == v1:
        return v0
    if a0 == a1:
        return None
    p = Fraction(v1 - v0, a1 - a0)
    k = Fraction(v0) - p * a0
    if k.denominator != 1:
        return None
    k = int(k)
    if p.denominator == 1:
        term = {"mul": [ref, int(p)]} if int(p) != 1 else ref
    elif p.numerator == 1:
        # v = axis / d + k  (e.g. count = D // 8, or a padded tile origin). `div`
        # floors, so with two samples this is a guess about what happens between
        # them either way -- the `k == 0` case is no safer than `k != 0`, and both
        # stand or fall on the held-out points. The multi-axis renderer has always
        # allowed a non-zero intercept here; this makes the two agree.
        term = {"div": [ref, p.denominator]}
    else:
        return None
    if k == 0:
        return term
    return {"add": [term, k]} if k > 0 else {"sub": [term, -k]}


# --------------------------------------------------------------------------
# affine inference over N axes (the multi-axis generalization of _linear_expr)
# --------------------------------------------------------------------------
# `_linear_expr` fits one axis from two points. The functions below fit SEVERAL
# axes at once -- v = c0 + sum_j m_j * x_j -- from a set of sample points, which
# is what lets ONE recipe cover a cross product of non-reduction (shape / outer
# tile) axes instead of one recipe per axis. The reduction axis is deliberately
# not a target: it drives the hot loop's structure, not just its constants.
#
# The solve is EXACT (Fractions, no least squares) and is rejected unless it
# reproduces every sample point, so an axis that enters non-linearly (a spatial
# product, a clamped vector width) is refused rather than approximated.
def affine_solve(
    points: List[Tuple[int, ...]], vals: List[int]
) -> Optional[List[Fraction]]:
    """Exact affine fit v = c0 + sum_j m_j*x_j over ALL given points.

    Returns [c0, m_0, ...] as Fractions, or None when the system is
    inconsistent (no affine model reproduces every sample). An underdetermined
    system is resolved to the minimum-support solution (unconstrained
    coefficients come back 0), so an axis that never varies gets m_j = 0."""
    if not points:
        return None
    naxes = len(points[0])
    if any(len(p) != naxes for p in points) or len(vals) != len(points):
        return None
    ncol = naxes + 1
    rows = [
        [Fraction(1)] + [Fraction(x) for x in p] + [Fraction(v)]
        for p, v in zip(points, vals)
    ]
    piv: List[int] = []
    r = 0
    for c in range(ncol):
        pr = next((k for k in range(r, len(rows)) if rows[k][c] != 0), None)
        if pr is None:
            continue
        rows[r], rows[pr] = rows[pr], rows[r]
        pv = rows[r][c]
        rows[r] = [x / pv for x in rows[r]]
        for k in range(len(rows)):
            if k != r and rows[k][c] != 0:
                f = rows[k][c]
                rows[k] = [x - f * y for x, y in zip(rows[k], rows[r])]
        piv.append(c)
        r += 1
        if r == len(rows):
            break
    # An all-zero coefficient row with a non-zero rhs means no affine model fits.
    for k in range(r, len(rows)):
        if all(x == 0 for x in rows[k][:ncol]) and rows[k][ncol] != 0:
            return None
    sol = [Fraction(0)] * ncol
    for i, c in enumerate(piv):
        sol[c] = rows[i][ncol]
    for p, v in zip(points, vals):
        if sol[0] + sum(m * x for m, x in zip(sol[1:], p)) != v:
            return None
    return sol


def affine_intexpr(axes: List[str], sol: List[Fraction]) -> Optional[Any]:
    """Render an `affine_solve` result as an intexpr over {spec:axis} terms."""
    return render_affine([{"spec": a} for a in axes], sol)


def render_affine(refs: List[Any], sol: List[Fraction]) -> Optional[Any]:
    """Render `c0 + sum m_j*ref_j` as an intexpr, where each ref is itself an
    intexpr (a bare `{spec: axis}` for an affine fit, or a product of two for a
    cross term).

    Mirrors `_linear_expr`'s expressibility rules per term: an integer
    coefficient becomes a `mul` (elided at 1), and a unit fraction 1/d becomes a
    `div` -- the shape of constant that shows up as `count = axis // vec`. Any
    other fractional coefficient is refused, because `div` is floor division and
    a non-unit fraction would only agree with the builder by luck."""
    if len(sol) != len(refs) + 1:
        return None
    c0, coeffs = sol[0], sol[1:]
    if c0.denominator != 1:
        return None
    terms: List[Any] = []
    for ref, m in zip(refs, coeffs):
        if m == 0:
            continue
        if m.denominator == 1:
            terms.append(ref if m == 1 else {"mul": [ref, int(m)]})
        elif m.numerator == 1:
            terms.append({"div": [ref, m.denominator]})
        else:
            return None
    k = int(c0)
    if not terms:
        return k
    expr = terms[0]
    for t in terms[1:]:
        expr = {"add": [expr, t]}
    if k == 0:
        return expr
    return {"add": [expr, k]} if k > 0 else {"sub": [expr, -k]}


# --------------------------------------------------------------------------
# candidate models
#
# Affine covers most integers a kernel computes from a shape, because address
# arithmetic IS affine. The rest of these exist because a few constants are not
# values the kernel derived from the shape at all -- they are decisions a code
# generator made (which multiplier strength-reduces this division, how many tiles
# cover this extent), and those follow their own generating rule.
#
# Every candidate must fit EXACTLY at every recorded point, and they are tried
# simplest-first so the least presumptuous rule that explains the data wins. That
# ordering matters: searching a wider hypothesis class spends evidence, so the
# cheap protection is that anything found here still has to survive verification
# at points it was never fitted on.
# --------------------------------------------------------------------------
# Offsets applied to the axis before the magic operands are tried. Kept to two
# because each one is a hypothesis the fit does not pay for in samples, and the
# second earns its place: `ceil(log2(n+1))` is `n.bit_length()`, the trip count of
# a binary search over `n` items, which is how all three tiled attention kernels
# size their sequence-lookup loop. `0` is tried first so a true shift never gets
# explained as an offset one.
_MAGIC_OFFSETS = (0, 1)


def _fit_parameter_free(
    axes: List[str], points: List[Tuple[int, ...]], vals: List[int]
) -> Optional[Any]:
    """Functions with NO free parameters, so they cannot overfit: either they
    reproduce every point or they are out. Today that means the two operands of a
    strength-reduced division, whose generating formula lives in the DSL, and the
    logarithm underneath the shift -- which also turns up on its own as a
    binary-search trip count."""
    for j, axis in enumerate(axes):
        for off in _MAGIC_OFFSETS:
            for name, which in (("magic_multiplier", 0), ("magic_shift", 1)):
                try:
                    ok = all(
                        v == magic_division_constants(p[j] + off)[which]
                        for p, v in zip(points, vals)
                    )
                except Exception:
                    ok = False
                if ok:
                    operand: Any = {"spec": axis}
                    if off:
                        operand = {"add": [operand, off]}
                    return {name: operand}
    return None


def _fit_reciprocal(
    axes: List[str], points: List[Tuple[int, ...]], vals: List[int]
) -> Optional[Any]:
    """`k div x` and `ceil(k/x)` -- how a tile/block COUNT depends on a tile size.
    Not affine (it is a reciprocal), but long expressible as an intexpr; only the
    solver had never hypothesised it.

    `k` is pinned exactly when the division comes out even at every sample (the
    usual case: `k` is the problem extent). Otherwise the flooring leaves an
    interval of admissible `k`, and the boundary that corresponds to exact
    division is taken -- underdetermined, hence left to verification to confirm."""
    for j, axis in enumerate(axes):
        xs = [p[j] for p in points]
        if any(x <= 0 for x in xs) or len(set(xs)) < 2:
            continue
        ref = {"spec": axis}
        exact = {v * x for v, x in zip(vals, xs)}
        if len(exact) == 1:
            k = exact.pop()
            if k > 0 and all(v == k // x for v, x in zip(vals, xs)):
                return {"div": [k, ref]}
        # floor: v == k div x  ->  k in [v*x, v*x + x - 1]
        lo = max(v * x for v, x in zip(vals, xs))
        hi = min(v * x + x - 1 for v, x in zip(vals, xs))
        if lo <= hi and all(v == lo // x for v, x in zip(vals, xs)):
            return {"div": [lo, ref]}
        # ceil: v == (k + x - 1) div x  ->  k in [(v-1)*x + 1, v*x]
        clo = max((v - 1) * x + 1 for v, x in zip(vals, xs))
        chi = min(v * x for v, x in zip(vals, xs))
        if clo <= chi and all(v == (chi + x - 1) // x for v, x in zip(vals, xs)):
            return {"div": [{"add": [chi, {"sub": [ref, 1]}]}, ref]}
    return None


def _fit_cross(
    axes: List[str], points: List[Tuple[int, ...]], vals: List[int]
) -> Optional[Any]:
    """Affine PLUS pairwise products, for a constant that genuinely scales with
    two axes at once (`m*N*K`).

    A product is expressible (`mul` of two specs); the reason it is last is that
    it costs evidence. One-axis probes cannot see it -- along any single axis with
    the others fixed, a product looks exactly like a straight line -- so fitting it
    needs a point where two axes move together, which is why `roll_nd` only
    reaches this after recording extra interior points."""
    if len(axes) < 2:
        return None
    refs: List[Any] = [{"spec": a} for a in axes]
    idx: List[Tuple[int, ...]] = [(j,) for j in range(len(axes))]
    for j in range(len(axes)):
        for k in range(j + 1, len(axes)):
            refs.append({"mul": [{"spec": axes[j]}, {"spec": axes[k]}]})
            idx.append((j, k))
    basis: List[Tuple[int, ...]] = []
    for p in points:
        row = []
        for t in idx:
            prod = 1
            for j in t:
                prod *= p[j]
            row.append(prod)
        basis.append(tuple(row))
    sol = affine_solve(basis, vals)
    if sol is None:
        return None
    return render_affine(refs, sol)


def fit_slot(
    axes: List[str],
    points: List[Tuple[int, ...]],
    vals: List[int],
    allow_cross: bool = False,
) -> Tuple[Optional[Any], str]:
    """Fit ONE integer slot as a function of the axes, trying candidate models
    simplest-first. Returns (intexpr, "") or (None, reason)."""
    inexpressible = ""
    sol = affine_solve(points, vals)
    if sol is not None:
        expr = affine_intexpr(axes, sol)
        if expr is not None:
            return expr, ""
        # An affine model fits but cannot be written down. That does NOT settle the
        # slot: a reciprocal can pass through the same points and IS expressible
        # (two samples of `512 div b` are also fit by `24 - b/4`). Keep the reason
        # in case nothing else works, and carry on down the ladder.
        inexpressible = (
            f"affine fit {[str(c) for c in sol]} is not expressible as an intexpr "
            f"(a non-unit fractional coefficient would only agree with floor "
            f"division by luck)"
        )
    for cand in (_fit_parameter_free, _fit_reciprocal):
        expr = cand(axes, points, vals)
        if expr is not None:
            return expr, ""
    if allow_cross:
        expr = _fit_cross(axes, points, vals)
        if expr is not None:
            return expr, ""
    if inexpressible:
        return None, inexpressible
    return None, (
        f"integer {vals} over {points} fits no candidate model "
        f"(affine, magic-division operand, reciprocal"
        + (", cross term" if allow_cross else "")
        + ")"
    )


def merge_intexpr(ea: Any, eb: Any, axis: str, a0: int, a1: int) -> Optional[Any]:
    """Merge two intexpr trees that differ only at integer leaves, fitting each
    differing leaf linearly in `axis`.

    This is what lets a constant depend on BOTH a structural axis and a
    multi-axis constant model: the constant-axis annotation has already turned
    the value into a tree, so the structural merge has to reconcile trees rather
    than plain ints. Returns None if the trees differ in shape."""
    if ea == eb:
        return copy.deepcopy(eb)
    if isinstance(ea, bool) or isinstance(eb, bool):
        return None
    if isinstance(ea, int) and isinstance(eb, int):
        return _linear_expr(axis, a0, ea, a1, eb)
    if isinstance(ea, list) and isinstance(eb, list) and len(ea) == len(eb):
        out = []
        for x, y in zip(ea, eb):
            m = merge_intexpr(x, y, axis, a0, a1)
            if m is None:
                return None
            out.append(m)
        return out
    if isinstance(ea, dict) and isinstance(eb, dict) and set(ea) == set(eb):
        out = {}
        for k in ea:
            m = merge_intexpr(ea[k], eb[k], axis, a0, a1)
            if m is None:
                return None
            out[k] = m
        return out
    return None


# --------------------------------------------------------------------------
# merge two structurally-identical instructions, parameterizing spec-scaled ints
# --------------------------------------------------------------------------
def _merge_type(ta: Any, tb: Any, axis: str, a0: int, a1: int) -> Any:
    """Merge two result types, parameterizing integer fields that differ (e.g. an
    smem buffer `shape` that scales with the axis) as linear intexprs. Returns the
    merged type, or None if irreconcilable. A sentinel ``("__nolinear__",)`` is
    returned for an integer pair that isn't linearly expressible."""
    if ta == tb:
        return copy.deepcopy(tb)
    if isinstance(ta, bool) or isinstance(tb, bool):
        return None
    if isinstance(ta, int) and isinstance(tb, int):
        return _linear_expr(axis, a0, ta, a1, tb) or ("__nolinear__",)
    if isinstance(ta, list) and isinstance(tb, list) and len(ta) == len(tb):
        out = []
        for x, y in zip(ta, tb):
            m = _merge_type(x, y, axis, a0, a1)
            if m is None or m == ("__nolinear__",):
                return None
            out.append(m)
        return out
    if isinstance(ta, dict) and isinstance(tb, dict) and set(ta) == set(tb):
        out = {}
        for k in ta:
            m = _merge_type(ta[k], tb[k], axis, a0, a1)
            if m is None or m == ("__nolinear__",):
                return None
            out[k] = m
        return out
    return None


def _merge_instr(
    a: Dict[str, Any], b: Dict[str, Any], axis: str, a0: int, a1: int, align
):
    """Return a parametric instr that expands to `a` at a0 and `b` at a1, or None
    if they cannot be merged structurally."""
    if _sig(a) != _sig(b):
        return None
    op = a["op"]
    out = copy.deepcopy(b)
    if op == "emit":
        # Reconcile attrs: any INTEGER attr that differs (a constant's value, or
        # e.g. a sched_group_barrier's instruction count) is parameterized as a
        # linear function of the axis; non-integer differences can't be merged.
        aa, ba = a.get("attrs", {}), b.get("attrs", {})
        if aa != ba:
            if set(aa) != set(ba):
                return None
            for key in aa:
                av, bv = aa[key], ba[key]
                if av == bv:
                    continue
                if av.get("t") == "i" and bv.get("t") == "i":
                    # Plain ints fit directly; already-parametric trees (a
                    # multi-axis constant annotation) are merged leaf-wise so a
                    # constant may depend on the structural axis AND others.
                    expr = merge_intexpr(av["v"], bv["v"], axis, a0, a1)
                    if expr is None:
                        return None
                    out["attrs"][key] = {"t": "i", "v": expr}
                else:
                    return None
        # Reconcile result TYPES (e.g. an smem_alloc buffer shape that scales).
        for okey in ("out",):
            if okey in a and a[okey].get("type") != b[okey].get("type"):
                mt = _merge_type(a[okey]["type"], b[okey]["type"], axis, a0, a1)
                if mt is None:
                    return None
                out[okey]["type"] = mt
        if "outs" in a:
            for i, (oa, ob) in enumerate(zip(a["outs"], b["outs"])):
                if oa.get("type") != ob.get("type"):
                    mt = _merge_type(oa["type"], ob["type"], axis, a0, a1)
                    if mt is None:
                        return None
                    out["outs"][i]["type"] = mt
        return out
    if op == "scf_for":
        merged_body = align(a["body"], b["body"])
        if merged_body is None:
            return None
        out["body"] = merged_body
        return out
    if op == "scf_if":
        merged = align(a.get("then", []), b.get("then", []))
        if merged is None:
            return None
        out["then"] = merged
        return out
    # param / ret / thread_id_x / const_* : must already be identical-by-sig
    return out


# --------------------------------------------------------------------------
# run detection + rolling
# --------------------------------------------------------------------------
def _smallest_period(sigs: List[Tuple]) -> Optional[int]:
    n = len(sigs)
    for L in range(1, n // 2 + 1):
        if n % L:
            continue
        if all(sigs[i] == sigs[i % L] for i in range(n)):
            return L
    return None


def _deep_rewrite_uses(prog: List[Dict[str, Any]], mapping: Dict[str, str]) -> None:
    for instr in prog:
        _rewrite_uses(instr, mapping)
        for key in ("body", "then", "else"):
            if key in instr:
                _deep_rewrite_uses(instr[key], mapping)


def _mem_access(instr: Dict[str, Any]):
    """If `instr` is a scratchpad/global store or load, return
    ('store'|'load', buffer_name, address_operand_names); else None.
    Convention: in=[buffer, addr..., value] for stores; [buffer, addr...] for
    loads (the common shape for memref.global_* and tile.smem_*)."""
    op = instr.get("opcode", "")
    ins = instr.get("in", [])
    if instr.get("op") != "emit" or len(ins) < 2:
        return None
    if "store" in op:
        return ("store", ins[0], tuple(ins[1:-1]))
    if "load" in op:
        return ("load", ins[0], tuple(ins[1:]))
    return None


def _addr_keyer(body: List[Dict[str, Any]]):
    """Return a function name -> canonical key of its computation tree, so two
    addresses that compute the same location ('where on the bench') compare
    equal regardless of SSA names. Leaves are params/iv (by name) and constant
    values; intermediate SSA is expanded away."""
    defpos = {}
    for j, instr in enumerate(body):
        for d in _defs(instr):
            defpos[d] = j

    def key(name, depth=0):
        j = defpos.get(name)
        if j is None or depth > 24:
            return ("leaf", name)  # param / iv / loop-carry
        instr = body[j]
        if _is_const_int(instr):
            return ("c", _const_int(instr))
        return (
            instr.get("opcode", instr["op"]),
            tuple(key(x, depth + 1) for x in _uses(instr)),
        )

    return key


def scratchpad_edges(body: List[Dict[str, Any]]) -> Dict[int, int]:
    """Match each scratchpad LOAD to the STORE that wrote the location it reads
    ('drop on the bench' -> 'pick up from the bench'), keyed by buffer + address
    equivalence, with buffer+program-order as a fallback when address formulas
    differ (e.g. store/load swizzles). Returns {load_index: store_index}."""
    key = _addr_keyer(body)
    stores, loads = [], []
    for j, instr in enumerate(body):
        acc = _mem_access(instr)
        if not acc:
            continue
        kind, buf, addr = acc
        rec = (j, buf, tuple(key(a) for a in addr))
        (stores if kind == "store" else loads).append(rec)

    out: Dict[int, int] = {}
    used = set()
    # 1) exact buffer + address-tree match.
    for lj, lbuf, lkey in loads:
        for sj, sbuf, skey in stores:
            if sj in used or sj > lj or sbuf != lbuf:
                continue
            if skey == lkey:
                out[lj] = sj
                used.add(sj)
                break
    # 2) fallback: k-th unmatched store to a buffer -> k-th unmatched load.
    for buf in {b for _, b, _ in loads}:
        bstores = [sj for (sj, sb, _) in stores if sb == buf and sj not in used]
        bloads = [lj for (lj, lb, _) in loads if lb == buf and lj not in out]
        for sj, lj in zip(bstores, bloads):
            out[lj] = sj
            used.add(sj)
    return out


def lane_label_body(
    body: List[Dict[str, Any]], yields: List[str], n_lanes: int
) -> Dict[int, Any]:
    """Label each body op by MEANING, not appearance: which lane's result it
    ultimately feeds. Backward data-flow from the per-lane yield operands, plus
    scratchpad store->load memory edges so a side-effecting store inherits the
    lane of the load that consumes it.

    Returns {op_index: lane_int | 'S' (shared, feeds >1 lane) | None (feeds none,
    e.g. a barrier)}."""
    defpos = {}
    consumers: Dict[str, List[int]] = {}
    for j, instr in enumerate(body):
        for d in _defs(instr):
            defpos[d] = j
        for u in _uses(instr):
            consumers.setdefault(u, []).append(j)
    mem = scratchpad_edges(body)  # load_idx -> store_idx
    store_to_load = {s: l for l, s in mem.items()}
    seed = {defpos[y]: i for i, y in enumerate(yields) if y in defpos}

    label: Dict[int, Any] = {}

    def combine(acc, lane):
        if lane is None:
            return acc
        if acc is None:
            return lane
        return acc if acc == lane else "S"

    for j in range(len(body) - 1, -1, -1):
        lane = None
        if j in seed:
            lane = seed[j]
        # SSA consumers (downstream ops, already labeled in this reverse walk).
        for d in _defs(body[j]):
            for c in consumers.get(d, []):
                if c > j:
                    lane = combine(lane, label.get(c))
        # memory consumer: a store's value is "used" by the matching load.
        if j in store_to_load:
            lane = combine(lane, label.get(store_to_load[j]))
        label[j] = lane
    return label


def _deep_uses(prog: List[Dict[str, Any]]) -> set:
    """All operand names referenced anywhere in `prog` (recursing into regions)."""
    s = set()
    for instr in prog:
        for u in _uses(instr):
            if isinstance(u, str):
                s.add(u)
        for key in ("body", "then", "else"):
            if key in instr:
                s |= _deep_uses(instr[key])
    return s


def _appearing_block(
    la: List[Dict[str, Any]], lb: List[Dict[str, Any]], window: int = 24
) -> Optional[int]:
    """Length of a block present in `lb` but ABSENT from `la` at the divergence,
    or None. Detected by resynchronisation: deleting L ops from lb makes the two
    line up again for a decent stretch.

    This is worth naming because it is not a detector weakness that more effort
    could fix. A run seen ONCE carries no evidence of how one iteration feeds the
    next, and a run seen ZERO times carries none at all -- `_roll_run` learns the
    loop-carried values by diffing block 0 against block 1. So a count that goes
    0 -> 1 across the two samples cannot be rolled from those samples by any
    method; it needs sample points where the count reaches 2."""
    sa = [_sig(i) for i in la]
    sb = [_sig(i) for i in lb]
    m = min(len(sa), len(sb))
    f = 0
    while f < m and sa[f] == sb[f]:
        f += 1
    if f >= m:
        return None
    # Score each L by how far the two resynchronise. A fixed long window is no
    # good: the next growing run usually starts a few ops later and breaks the
    # match again, so take the best L rather than the first that clears a bar.
    best, best_run = None, 0
    for L in range(1, len(sb) - len(sa) + 1):
        lim = min(window, len(sa) - f, len(sb) - f - L)
        if lim <= 0:
            break
        k = 0
        while k < lim and sa[f + k] == sb[f + L + k]:
            k += 1
        if k > best_run:
            best, best_run = L, k
    return best if best_run >= 4 else None


def _sample_advice(a0: int, a1: int) -> str:
    """Sample points that would give the absent run >= 1 and >= 2 copies.

    If a block first appears between a0 and a1, the count is 0 at a0 and 1 at a1,
    so it is linear with step (a1 - a0) and reaches 1 at a1 and 2 at 2*a1 - a0."""
    return f"try sample_points=[{a1}, {2 * a1 - a0}] (or larger) instead"


def _divergence_reason(la: List[Dict[str, Any]], lb: List[Dict[str, Any]]) -> str:
    """A precise reason why no repeated-block run was found at the first
    signature divergence (informs which roller generalization is still needed)."""
    sa = [_sig(i) for i in la]
    sb = [_sig(i) for i in lb]
    m = min(len(sa), len(sb))
    f = 0
    while f < m and sa[f] == sb[f]:
        f += 1
    base = f"no run at level (|la|={len(la)} |lb|={len(lb)}, first diverge @ {f})"
    if f >= m:
        return base + ": one trace is a prefix of the other (pure append)"
    a, b = la[f], lb[f]
    if a.get("op") == "scf_for" and b.get("op") == "scf_for":
        na, nb = len(a.get("iter", [])), len(b.get("iter", []))
        if na != nb:
            return (
                base + f": runtime scf.for iter-arg arity scales with axis "
                f"({na}->{nb}) -- needs parametric scf_for iter-args "
                f"(variable loop-carry fan); roller+VM extension"
            )
    L = _appearing_block(la, lb)
    if L is not None:
        return (
            base + f": a {L}-op block is ABSENT at the smaller axis value and "
            f"present at the larger, so its iteration count goes 0 -> 1. A run "
            f"has to appear at least TWICE somewhere for its loop-carried values "
            f"to be readable (block 0 vs block 1), so no sampling-independent fix "
            f"exists -- this is a SAMPLE CHOICE problem"
        )
    if a.get("op") != b.get("op"):
        return base + f": op changes {a.get('op')} -> {b.get('op')} (peeling?)"
    return base + ": non-uniform/peeled blocks (boundary iterations differ?)"


def _run_candidates(la: List[Dict[str, Any]], lb: List[Dict[str, Any]]):
    """Candidate repeated-block runs at the FIRST signature divergence, ordered
    best-first (largest block period). Each is (run_start, a_end, run_end_b, L,
    n_a, n_b): la's run is la[run_start:a_end] (n_a blocks of period L), lb's is
    lb[run_start:run_end_b] (n_b blocks).

    Multiple periods can fit locally (a coincidental small period vs the true
    unroll-body period); the caller tries candidates until one rolls cleanly and
    the remainder aligns. Anchoring at the divergence + block-boundary expansion
    avoids tandem-repeat over-eating and supports MULTIPLE runs per level."""
    if len(lb) <= len(la):
        return []
    sa = [_sig(i) for i in la]
    sb = [_sig(i) for i in lb]
    m = min(len(la), len(lb))
    f = 0
    while f < m and sa[f] == sb[f]:
        f += 1
    # f == m means la is a prefix of lb (the run is a pure append at the end);
    # anchor at f = len(la) and detect the periodic tail below.
    cands = []
    seen = set()
    for L in range(1, max(f, len(lb) - f) + 1):
        back = f >= L and f + L <= len(lb) and sb[f - L : f] == sb[f : f + L]
        fwd = f + 2 * L <= len(lb) and sb[f : f + L] == sb[f + L : f + 2 * L]
        if not (back or fwd):
            continue
        blockpat = sb[f - L : f] if back else sb[f : f + L]
        run_start = f
        while (
            run_start - L >= 0
            and sa[run_start - L : run_start] == blockpat
            and sb[run_start - L : run_start] == blockpat
        ):
            run_start -= L
        run_end_b = f
        while run_end_b + L <= len(lb) and sb[run_end_b : run_end_b + L] == blockpat:
            run_end_b += L
        if (f - run_start) % L or (run_end_b - run_start) % L:
            continue
        n_a = (f - run_start) // L
        n_b = (run_end_b - run_start) // L
        if n_b < 2 or n_b <= n_a or sa[run_start:f] != blockpat * n_a:
            continue
        key = (run_start, run_end_b, L)
        if key in seen:
            continue
        seen.add(key)
        cands.append((run_start, f, run_end_b, L, n_a, n_b))
    # Smallest block first: the true unroll body is the PRIMITIVE repeating unit;
    # a 2x "superblock" is a non-primitive multiple that overfits the two sampled
    # traces. (L=1 coincidences are rejected by roll_run's clean-count check.)
    cands.sort(key=lambda c: c[3])
    return cands


def _segment_by_lane(body: List[Dict[str, Any]], label: Dict[int, Any], n: int):
    """Split a (yield-less) fan body into segments using lane labels:
    ('shared', ops, start, end) for lane-independent regions, and
    ('phase', [lane0_ops, lane1_ops, ...], start, end) for a contiguous per-lane
    run. Returns None if a per-lane region is not a clean lanes-0..n-1 partition."""
    segs = []
    i, N = 0, len(body)
    while i < N:
        if isinstance(label.get(i), int):
            start = i
            blocks, lanes, cur, blk = [], [], label[i], []
            while i < N and isinstance(label.get(i), int):
                lab = label[i]
                if lab != cur:
                    if lab < cur:  # lane reset -> a new phase starts here
                        break
                    blocks.append(blk)
                    lanes.append(cur)
                    blk, cur = [], lab
                blk.append(body[i])
                i += 1
            blocks.append(blk)
            lanes.append(cur)
            if lanes != list(range(n)) or any(len(b) != len(blocks[0]) for b in blocks):
                return None
            segs.append(("phase", blocks, start, i))
        else:
            start = i
            seg = []
            while i < N and not isinstance(label.get(i), int):
                seg.append(body[i])
                i += 1
            segs.append(("shared", seg, start, i))
    return segs


def _roll_run(
    mid_a: List[Dict[str, Any]],
    mid_b: List[Dict[str, Any]],
    axis: str,
    a0: int,
    a1: int,
    varname: str,
    L: int,
    n_a: int,
    n_b: int,
    lane_refs: Optional[Dict[str, Tuple[int, str, str]]] = None,
):
    """Roll a repeated run that appears n_a times in mid_a and n_b times in mid_b
    (period L) into a static_for. Returns (prologue_aliases + [static_for],
    suffix_rewrite) or None to fall back. suffix_rewrite maps last-block
    carry-out names -> carry register names (the caller applies it to the
    suffix).

    `lane_refs` (name -> (lane, fmt, fan_var)) lets a run index into a fan's
    per-lane values (e.g. a CShuffle/reduction epilogue over the loop results):
    operands referencing lane `start+k` in block k are rewritten to ``fmt{var}``
    and the static_for is shifted to iterate over the actual lane range."""
    count_expr = _linear_expr(axis, a0, n_a, a1, n_b)
    if count_expr is None:
        return None

    block0 = mid_b[0:L]
    block1 = mid_b[L : 2 * L]
    last = mid_b[(n_b - 1) * L : n_b * L]

    # def-position maps (instr index j, def index k) for block0 / block1.
    def defpos(block):
        m = {}
        for j, instr in enumerate(block):
            for k, d in enumerate(_defs(instr)):
                m[d] = (j, k)
        return m

    b0_pos = defpos(block0)
    b1_def_at = {
        (j, k): d for j, instr in enumerate(block1) for k, d in enumerate(_defs(instr))
    }
    last_def_at = {
        (j, k): d for j, instr in enumerate(last) for k, d in enumerate(_defs(instr))
    }

    template = copy.deepcopy(block1)

    # --- carries: operands in block1 referencing a block0 def ---
    carries: Dict[Tuple[int, int], Dict[str, Any]] = {}  # producing pos -> info
    for j, instr in enumerate(block1):
        for o, name in enumerate(_uses(instr)):
            if name in b0_pos:
                prod = b0_pos[name]  # carry-out producing position
                C = f"%__carry_{varname}_{prod[0]}_{prod[1]}"
                info = carries.setdefault(prod, {"name": C, "uses": [], "init": None})
                info["uses"].append((j, o))
                if info["init"] is None:
                    # carry-in for iteration 0 = block0's operand at the same site
                    info["init"] = _uses(block0[j])[o]

    # rewrite template carry use-sites -> carry register
    for prod, info in carries.items():
        for j, o in info["uses"]:
            slots = list(_use_slots(template[j]))
            slots[o][1](info["name"])

    # trailing aliases: carry <- this block's produced value (template def)
    for prod, info in carries.items():
        produced = b1_def_at[prod]
        template.append({"op": "alias", "bind": info["name"], "from": produced})

    # expose per-lane live-out defs (this block's defs that belong to a lane
    # family consumed per-lane downstream, e.g. GEMM B-load partials feeding the
    # mma run, or the mma outputs feeding scf.yield): alias to fmt{var}.
    if lane_refs:
        for (j, k), name in b1_def_at.items():
            if name in lane_refs:
                template.append(
                    {
                        "op": "alias",
                        "bind": lane_refs[name][1] + "{" + varname + "}",
                        "from": name,
                    }
                )

    # --- lane-refs: operands that index into a fan's per-lane values. block k
    # references lane (start+k); rewrite to fmt{var} and shift the loop to the
    # lane range. `start` (default 0) is also the var-base for index ladders.
    start = 0
    if lane_refs:
        starts = set()
        for j, instr in enumerate(block1):
            for o, name in enumerate(_uses(instr)):
                if name not in lane_refs:
                    continue
                lane1, fmt, _fv = lane_refs[name]
                name0 = _uses(block0[j])[o]
                if name0 not in lane_refs or lane1 - lane_refs[name0][0] != 1:
                    return None
                starts.add(lane_refs[name0][0])
                list(_use_slots(template[j]))[o][1](fmt + "{" + varname + "}")
        if len(starts) > 1:
            return None
        if starts:
            start = starts.pop()

    # --- index-ladder + spec-scaled constants inside the block ---
    a_block0 = mid_a[0:L] if mid_a else None
    for j, instr in enumerate(template):
        if not _is_const_int(instr):
            continue
        v_b0 = _const_int(block0[j])
        v_b1 = _const_int(block1[j])
        if v_b1 != v_b0:  # ladder: v0 + i*delta
            expr = _linear_expr(varname, start, v_b0, start + 1, v_b1, var=True)
            if expr is None:
                return None
            _set_const_int(instr, expr)
        elif a_block0 is not None:
            v_a0 = _const_int(a_block0[j])
            if v_a0 != v_b0:  # spec-scaled in-run const
                expr = _linear_expr(axis, a0, v_a0, a1, v_b0)
                if expr is None:
                    return None
                _set_const_int(instr, expr)

    prologue = [
        {"op": "alias", "bind": info["name"], "from": info["init"]}
        for info in carries.values()
    ]
    hi = count_expr if start == 0 else {"add": [count_expr, start]}
    static_for = {
        "op": "static_for",
        "var": varname,
        "lo": start,
        "hi": hi,
        "step": 1,
        "body": template,
    }

    suffix_rewrite = {last_def_at[prod]: info["name"] for prod, info in carries.items()}
    return prologue + [static_for], suffix_rewrite


# --------------------------------------------------------------------------
# align two programs (recursive); roll divergent runs
# --------------------------------------------------------------------------
class _Roller:
    def __init__(self, axis: str, a0: int, a1: int):
        self.axis, self.a0, self.a1 = axis, a0, a1
        self._run = 0
        self._fan = 0
        # original scf.for result name -> (lane, fmt_base, fan_var), so the parent
        # (which consumes the loop's results) can be re-pointed at the parametric
        # per-lane result names.
        self.lane_refs: Dict[str, Tuple[int, str, str]] = {}
        # Subset of lane_refs that are scf.for RESULTS: only these may have direct
        # (non-rolled) uses left in the parent, so only these feed the direct-map.
        self.result_refs: Dict[str, Tuple[int, str, str]] = {}
        # lane count of the fan whose body is currently being rolled; coupled runs
        # inside it should prefer this many blocks (lane-count consistency).
        self._fan_lanes: Optional[int] = None
        self.reason = ""

    def _fail(self, why: str) -> None:
        if not self.reason:
            self.reason = why

    def align(
        self, la: List[Dict[str, Any]], lb: List[Dict[str, Any]]
    ) -> Optional[List[Dict[str, Any]]]:
        if len(la) == len(lb):
            out = []
            for a, b in zip(la, lb):
                # A runtime scf.for whose iter-arg arity scales with the axis is
                # a variable loop-carry FAN -> roll its iter-args/results/body.
                if (
                    a.get("op") == "scf_for"
                    and b.get("op") == "scf_for"
                    and _sig(a) != _sig(b)
                ):
                    m = self._roll_fan(a, b)
                    if m is None:
                        self._fail(_divergence_reason([a], [b]))
                        return None
                    out.append(m)
                    continue
                m = _merge_instr(a, b, self.axis, self.a0, self.a1, self.align)
                if m is None:
                    if (
                        _is_const_int(a)
                        and _is_const_int(b)
                        and _const_int(a) != _const_int(b)
                        and _linear_expr(
                            self.axis, self.a0, _const_int(a), self.a1, _const_int(b)
                        )
                        is None
                    ):
                        self._fail(
                            f"non-affine constant {_const_int(a)} vs {_const_int(b)} "
                            f"over {self.axis} ({self.axis}={self.a0}/{self.a1}) -- the "
                            f"axis enters this constant non-linearly (e.g. a spatial "
                            f"product); needs multi-axis/polynomial constant inference"
                        )
                    else:
                        self._fail(
                            f"merge {a.get('op')}/{a.get('opcode','')} vs "
                            f"{b.get('op')}/{b.get('opcode','')}"
                        )
                    return None
                out.append(m)
            return out
        # Lengths differ -> one or more repeated runs live at this level. Try the
        # run candidates (largest block first); for each, roll it and RECURSE on
        # the remainder (handles MULTIPLE independent runs per level, e.g. GEMM
        # pipeline-nest + CShuffle). Accept the first that fully aligns.
        if len(lb) < len(la):
            self._fail(
                f"shorter-at-larger-axis (|la|={len(la)} > |lb|={len(lb)}); "
                f"order sample_points so the larger axis records more ops"
            )
            return None
        # If the first divergence is a variable-fan scf.for, roll it and recurse
        # on the remainder (the rest of the level -- e.g. an epilogue -- may have
        # its own runs/fans).
        f = 0
        m = min(len(la), len(lb))
        while f < m and _sig(la[f]) == _sig(lb[f]):
            f += 1
        if (
            f < m
            and la[f].get("op") == "scf_for"
            and lb[f].get("op") == "scf_for"
            and _sig(la[f]) != _sig(lb[f])
        ):
            fan = self._roll_fan(la[f], lb[f])
            if fan is not None:
                prefix = self.align(la[:f], lb[:f])
                suffix = self.align(la[f + 1 :], lb[f + 1 :])
                if prefix is not None and suffix is not None:
                    return prefix + [fan] + suffix
        cands = _run_candidates(la, lb)
        if not cands:
            self._fail(_divergence_reason(la, lb))
            return None
        # Lane-count consistency: inside a fan body, coupled per-lane runs must
        # use the SAME lane count as the fan. Prefer candidates whose block count
        # matches (still falling back to others if none roll).
        if self._fan_lanes is not None:
            cands.sort(key=lambda c: (c[5] != self._fan_lanes, c[3]))
        for run_start, a_end, run_end_b, L, n_a, n_b in cands:
            snap = (dict(self.lane_refs), dict(self.result_refs), self._run, self._fan)

            def _restore():
                self.lane_refs, self.result_refs, self._run, self._fan = (
                    dict(snap[0]),
                    dict(snap[1]),
                    snap[2],
                    snap[3],
                )

            prefix = self.align(la[:run_start], lb[:run_start])  # equal length
            if prefix is None:
                _restore()
                continue
            varname = f"_r{self._run}"
            self._run += 1
            mid_b = lb[run_start:run_end_b]
            self._lane_families(mid_b, lb[run_end_b:], L, n_b, varname)
            rolled = _roll_run(
                la[run_start:a_end],
                mid_b,
                self.axis,
                self.a0,
                self.a1,
                varname,
                L,
                n_a,
                n_b,
                lane_refs=self.lane_refs,
            )
            if rolled is None:
                _restore()
                continue
            run_instrs, suffix_rewrite = rolled
            suffix = self.align(la[a_end:], lb[run_end_b:])
            if suffix is None:
                _restore()
                continue
            suffix = copy.deepcopy(suffix)
            _deep_rewrite_uses(suffix, suffix_rewrite)
            return prefix + run_instrs + suffix
        extra = ""
        L = _appearing_block(la, lb)
        if L is not None:
            extra = (
                f"; a {L}-op block is absent at {self.axis}={self.a0} and present "
                f"at {self.a1} (count 0 -> 1), which no detector can roll from "
                f"these two samples -- {_sample_advice(self.a0, self.a1)}"
            )
        self._fail(
            f"no run candidate rolled at level: |la|={len(la)} |lb|={len(lb)} "
            f"({len(cands)} candidates tried); first ops="
            + ",".join(i.get("opcode", i["op"]) for i in lb[:4])
            + extra
        )
        return None

    def _lane_families(
        self,
        mid_b: List[Dict[str, Any]],
        suffix_b: List[Dict[str, Any]],
        L: int,
        n_b: int,
        varname: str,
    ) -> Dict[Tuple[int, int], str]:
        """Per-block def positions whose value is consumed downstream for EVERY
        lane -> a per-lane family (e.g. GEMM B-load partials feeding the mma run).
        Registers each member in self.lane_refs and returns {(j,k): out_fmt} so
        _roll_run aliases them to out_fmt{var}."""
        suffix_uses = _deep_uses(suffix_b)
        pos_names: Dict[Tuple[int, int], List[Optional[str]]] = {}
        for bi in range(n_b):
            for j, instr in enumerate(mid_b[bi * L : (bi + 1) * L]):
                for k, d in enumerate(_defs(instr)):
                    pos_names.setdefault((j, k), [None] * n_b)[bi] = d
        expose: Dict[Tuple[int, int], str] = {}
        for (j, k), names in pos_names.items():
            # Skip positions already registered (e.g. the pre-registered yield
            # family) so we don't shadow them with a fresh fmt.
            if any(nm in self.lane_refs for nm in names if nm):
                continue
            if all(names) and all(nm in suffix_uses for nm in names):
                out_fmt = f"__lr{self._fan}_"
                self._fan += 1
                for i, nm in enumerate(names):
                    self.lane_refs[nm] = (i, out_fmt, varname)
                expose[(j, k)] = out_fmt
        return expose

    def _roll_fan_body(
        self,
        full_a: List[Dict[str, Any]],
        full_b: List[Dict[str, Any]],
        n_a: int,
        n_b: int,
    ) -> Optional[List[Dict[str, Any]]]:
        """Roll a fan's loop body using lane labels (meaning, not appearance):
        shared regions are merged 1:1; per-lane PHASES are rolled with the fan's
        lane count, with inter-phase per-lane values threaded as families. This
        is what separates a shared A-tile from per-lane B-tiles in GEMM and
        survives the store->LDS->load memory hop."""
        ya, yb = full_a[-1].get("in", []), full_b[-1].get("in", [])
        lab_a = lane_label_body(full_a, ya, n_a)
        lab_b = lane_label_body(full_b, yb, n_b)
        segs_a = _segment_by_lane(full_a[:-1], lab_a, n_a)
        segs_b = _segment_by_lane(full_b[:-1], lab_b, n_b)
        if segs_a is None or segs_b is None or len(segs_a) != len(segs_b):
            return None
        out: List[Dict[str, Any]] = []
        for (ka, pa, _sa, _ea), (kb, pb, _sb, eb) in zip(segs_a, segs_b):
            if ka != kb:
                return None
            if kb == "shared":
                if len(pa) != len(pb):
                    return None
                for ia, ib in zip(pa, pb):
                    m = _merge_instr(ia, ib, self.axis, self.a0, self.a1, self.align)
                    if m is None:
                        return None
                    out.append(m)
            else:  # per-lane phase
                L = len(pb[0])
                if len(pa[0]) != L:
                    return None
                mid_a = [op for blk in pa for op in blk]
                mid_b = [op for blk in pb for op in blk]
                varname = f"_r{self._run}"
                self._run += 1
                # register inter-phase families (this phase's per-lane defs that a
                # LATER phase / the yield consumes).
                self._lane_families(mid_b, full_b[eb:], L, n_b, varname)
                rolled = _roll_run(
                    mid_a,
                    mid_b,
                    self.axis,
                    self.a0,
                    self.a1,
                    varname,
                    L,
                    n_a,
                    n_b,
                    lane_refs=self.lane_refs,
                )
                if rolled is None:
                    return None
                run_instrs, suffix_rewrite = rolled
                if suffix_rewrite:  # no cross-phase carries expected
                    return None
                out.extend(run_instrs)
        return out

    @staticmethod
    def _lane_prefix(*name_lists: List[str]) -> Optional[str]:
        """Python's own naming for a loop-carry fan, as a lane-indexed prefix.

        Builders name a fan's iter-args positionally -- ``acc_m0_n0``,
        ``acc_m0_n1``, ... -- so when every lane `i` is ``<prefix><i>`` the
        roller can carry Python's names through instead of minting synthetic
        ``__faN_`` ones. The prefix extends to lane counts never sampled, which
        is what makes it usable on a parametric fan. Returns None when the names
        are not simply the lane index (a 2-D ``acc_m{m}_n{n}`` fan, say), which
        leaves the synthetic naming in place: still correct, just
        alpha-equivalent rather than byte-identical."""
        prefix: Optional[str] = None
        for names in name_lists:
            if not names:
                return None
            for i, nm in enumerate(names):
                tail = str(i)
                if not nm.endswith(tail) or len(nm) == len(tail):
                    return None
                p = nm[: -len(tail)]
                if prefix is None:
                    prefix = p
                elif p != prefix:
                    return None
        return prefix

    def _roll_fan(
        self, a: Dict[str, Any], b: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Roll a runtime scf.for whose loop-carry count scales with the axis
        (variable loop-carry fan): parameterize iter-args + results over a lane
        axis and roll the per-lane body run + the scf.yield. `a`/`b` are the two
        concrete scf_for instrs (n_a < n_b lanes)."""
        iter_a, iter_b = a.get("iter", []), b.get("iter", [])
        res_a, res_b = a.get("results", []), b.get("results", [])
        n_a, n_b = len(iter_a), len(iter_b)
        if n_a == n_b or n_a < 1 or n_b < 2:
            return None
        if len(res_a) != n_a or len(res_b) != n_b:
            return None
        count = _linear_expr(self.axis, self.a0, n_a, self.a1, n_b)
        if count is None:
            return None
        init_b = iter_b[0]["init"]
        if any(m["init"] != init_b for m in iter_b):
            return None  # only uniform-init fans for now
        body_a, body_b = a["body"], b["body"]
        if not body_a or not body_b:
            return None
        if (
            body_a[-1].get("opcode") != "scf.yield"
            or body_b[-1].get("opcode") != "scf.yield"
        ):
            return None
        yield_b = body_b[-1]
        if len(yield_b.get("in", [])) != n_b:
            return None

        saved = dict(self.lane_refs)
        saved_run, saved_fan = self._run, self._fan
        fid = self._fan
        self._fan += 1
        fv = f"fa{fid}"
        res_fmt, new_fmt = f"__fr{fid}_", f"__fn{fid}_"
        # The iter-arg names reach the emitted .ll (the loop-carry phis), so
        # prefer Python's own lane naming over a synthetic one. res_/new_ are
        # internal threading the backend renames, and stay synthetic.
        acc_fmt = (
            self._lane_prefix([m["name"] for m in iter_a], [m["name"] for m in iter_b])
            or f"__fa{fid}_"
        )
        lanes = {"for": {"var": fv, "lo": 0, "hi": count, "step": 1}}

        saved_results = dict(self.result_refs)
        saved_lanes = self._fan_lanes

        def restore():
            self.lane_refs, self.result_refs = saved, saved_results
            self._run, self._fan, self._fan_lanes = saved_run, saved_fan, saved_lanes

        # Input lane-refs: the per-lane iter-args (read by the body's per-lane run).
        for i, m in enumerate(iter_b):
            self.lane_refs[m["name"]] = (i, acc_fmt, fv)
        # Output family: the yield's per-lane operands (the body run that produces
        # them will be exposed to new_fmt{lane} by _roll_run's family scan).
        for i, yv in enumerate(yield_b.get("in", [])):
            self.lane_refs[yv] = (i, new_fmt, fv)

        # Roll the loop body using LANE LABELS (meaning, not appearance): shared
        # regions merged 1:1, per-lane phases rolled at the fan's lane count, with
        # inter-phase per-lane values threaded as families. Falls back to the
        # general aligner for bodies that don't decompose into labeled phases.
        self._fan_lanes = n_b
        rolled_body = self._roll_fan_body(body_a, body_b, n_a, n_b)
        if rolled_body is None:
            rolled_body = self.align(body_a[:-1], body_b[:-1])
        self._fan_lanes = saved_lanes
        if rolled_body is None:
            restore()
            return None

        new_yield = {
            "op": "emit",
            "opcode": "scf.yield",
            "in": [{**lanes, "name": new_fmt + "{" + fv + "}"}],
            "attrs": {"num": {"t": "i", "v": count}},
        }

        # Result lane-refs so the parent (a CShuffle / reduction epilogue over the
        # results) sees res_b[i] as lane i under res_fmt.
        for i, rn in enumerate(res_b):
            self.lane_refs[rn] = (i, res_fmt, fv)
            self.result_refs[rn] = (i, res_fmt, fv)

        out = copy.deepcopy(b)
        out["iter"] = [{**lanes, "name": acc_fmt + "{" + fv + "}", "init": init_b}]
        out["results"] = [{**lanes, "name": res_fmt + "{" + fv + "}"}]
        out["body"] = rolled_body + [new_yield]
        return out


def _parameterize_name(na: str, nb: str, axis: str, a0: int, a1: int) -> str:
    """Derive a parametric kernel_name_fmt by tokenizing the two concrete kernel
    names and replacing digit runs equal to the axis values with '{axis}'
    (kernel names often encode the tile shape, e.g. g32_..._t16x32x16). Falls
    back to nb if the names don't cleanly correspond."""
    import re

    ta = re.findall(r"\d+|\D+", na)
    tb = re.findall(r"\d+|\D+", nb)
    if len(ta) != len(tb):
        return nb
    out = []
    for xa, xb in zip(ta, tb):
        if xa == xb:
            out.append(xb)
        elif xa.isdigit() and xb.isdigit() and int(xa) == a0 and int(xb) == a1:
            out.append("{" + axis + "}")
        else:
            return nb
    return "".join(out)


def roll_two(
    recipe_a: Dict[str, Any],
    recipe_b: Dict[str, Any],
    axis: str,
    a0: int,
    a1: int,
    spec_decl: List[Dict[str, str]],
    name_fmt: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Roll two concrete traces over `axis` into a parametric recipe, or None."""
    r = _Roller(axis, a0, a1)
    program = r.align(recipe_a["program"], recipe_b["program"])
    if program is None:
        roll_two.last_reason = r.reason
        return None
    # Re-point the parent's DIRECT references to a fan's results at the
    # parametric per-lane result names (e.g. for5 -> __fr lane_0). Per-lane runs
    # over the results (reduction/epilogue) are not yet supported -> verification
    # catches and falls back.
    if r.result_refs:
        direct = {rn: f"{fmt}{lane}" for rn, (lane, fmt, _fv) in r.result_refs.items()}
        _deep_rewrite_uses(program, direct)
    roll_two.last_reason = ""
    return {
        "schema": "rocke.recipe/v1",
        "kernel_name_fmt": name_fmt
        or _parameterize_name(
            recipe_a.get("kernel_name_fmt", ""),
            recipe_b.get("kernel_name_fmt", ""),
            axis,
            a0,
            a1,
        ),
        "spec": spec_decl,
        "attrs": recipe_b.get("attrs", {}),
        "program": program,
    }
