# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# roll_nd.py -- multi-axis rolling: ONE recipe covering a CROSS PRODUCT of axes.
#
#   roll_nd(build_at, axes={"N": [8, 16], "K": [64, 128]},
#           holdout_points=[{"N": 32, "K": 256}])
#
# `roll` (roll.py) parameterizes ONE axis, so covering k axes costs k recipes and
# each one only moves along its own axis. This module covers the cross product
# with a single recipe by fitting every integer constant as an AFFINE function of
# several axes at once.
#
# Why the non-reduction axes first
# --------------------------------
# A kernel's axes split by what they do to the trace:
#
#   * The REDUCTION axis (GEMM tile_k, attention head_size) drives the hot loop.
#     Changing it changes the loop's STRUCTURE -- op counts, vector widths, LDS
#     sizing -- so it needs structural inference (`roll`), and sometimes more
#     than that.
#   * The OUTER axes -- inter-tile (grid/problem shape) and intra-tile (per-warp
#     geometry) -- often leave the instruction sequence completely alone and only
#     move CONSTANTS. Measured on the gated families: five of seven axes are
#     "constants only", emitting an identical op count at every value.
#
# The second group is what this module targets. Those axes need no structural
# inference at all, only a constant model, and a constant model composes across
# axes for free -- which is exactly why the cross product is reachable.
#
# How it works
# ------------
#   1. Record the BASE point, then one PROBE per axis that moves that axis alone.
#   2. Check the probes are structurally identical to the base (same op at every
#      position). An axis that fails this is structural, and belongs to `roll`
#      via `structural_axis=` instead.
#   3. For every integer-bearing field, solve v = c0 + sum_j m_j*x_j exactly
#      (`roller.affine_solve`) and rewrite it as an intexpr over `{spec: axis}`.
#   4. Optionally roll a structural axis on top, over the annotated traces.
#   5. VERIFY with the recipe_expand oracle at every cross-product point and
#      every held-out point.
#
# Step 5 is the safety net and it is deliberately much wider than step 1: fitting
# reads 1 + sum(n_j - 1) traces, verification checks prod(n_j) + holdouts. A
# cross term (a constant that scales with N*K, say) fits the one-axis-at-a-time
# probes perfectly and then fails at the first interior point, so it is caught
# and refused rather than shipped.

from __future__ import annotations

import copy
import itertools
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from rocke.portable_ir.src.recording_builder import record_kernel
from rocke.portable_ir.src.roller import fit_slot, roll_two, _sig
from rocke.portable_ir.utils.recipe_expand import (
    equiv_reason,
    expand_recipe,
    recipes_equiv,
)

Point = Dict[str, int]


class NdRollResult:
    """Outcome of a multi-axis roll. `recipe` is None on refusal, with `reason`
    naming the axis/field that could not be modeled; the caller then keeps the
    concrete per-point recipes (graceful degradation, never a wrong roll)."""

    def __init__(
        self,
        recipe: Optional[Dict[str, Any]],
        reason: str,
        points: List[Point],
        traces: Dict[Tuple[int, ...], Dict[str, Any]],
        n_recorded: int = 0,
    ):
        self.recipe = recipe
        self.reason = reason  # "" on success
        self.points = points  # every point the recipe was verified at
        self.traces = traces  # point tuple -> concrete recipe
        self.n_recorded = n_recorded  # traces used for INFERENCE (not verify)

    @property
    def ok(self) -> bool:
        return self.recipe is not None


class _StructMismatch(Exception):
    pass


# --------------------------------------------------------------------------
# integer-bearing slots
# --------------------------------------------------------------------------
def _tree_slots(node: Any, setter):
    """(value, setter) for every integer leaf of a JSON-ish tree, in a
    deterministic order so the same walk lines up across traces."""
    if isinstance(node, bool):
        return
    if isinstance(node, int):
        yield node, setter
    elif isinstance(node, list):
        for i, x in enumerate(node):
            yield from _tree_slots(x, lambda v, i=i, n=node: n.__setitem__(i, v))
    elif isinstance(node, dict):
        for k in sorted(node):
            yield from _tree_slots(node[k], lambda v, k=k, n=node: n.__setitem__(k, v))


def _int_slots(instr: Dict[str, Any]) -> List[Tuple[int, Any]]:
    """Every integer an axis could plausibly move: typed-int attr values (a
    constant's value, a barrier's instruction count) and integers inside result
    types (an smem buffer shape). Mirrors what `_merge_instr` parameterizes for a
    single axis. Operand slots are SSA names, not integers, so they are absent."""
    out: List[Tuple[int, Any]] = []
    attrs = instr.get("attrs") or {}
    for k in sorted(attrs):
        a = attrs[k]
        if isinstance(a, dict) and a.get("t") == "i":
            out.extend(_tree_slots(a.get("v"), lambda v, a=a: a.__setitem__("v", v)))
    if isinstance(instr.get("out"), dict):
        o = instr["out"]
        out.extend(_tree_slots(o.get("type"), lambda v, o=o: o.__setitem__("type", v)))
    for o in instr.get("outs", []) or []:
        if isinstance(o, dict):
            out.extend(
                _tree_slots(o.get("type"), lambda v, o=o: o.__setitem__("type", v))
            )
    return out


def _shape_key(node: Any) -> Any:
    """A tree with every integer blanked, so two trees compare equal iff they
    differ ONLY in integers. Guards against a non-integer difference (a dtype
    string that changes with the axis) slipping through as if it were affine."""
    if isinstance(node, bool):
        return ("b", node)
    if isinstance(node, int):
        return ("i",)
    if isinstance(node, list):
        return [_shape_key(x) for x in node]
    if isinstance(node, dict):
        return {k: _shape_key(node[k]) for k in sorted(node)}
    return node


def _nonint_key(instr: Dict[str, Any]) -> Any:
    return (
        instr.get("op"),
        instr.get("opcode"),
        _shape_key(instr.get("attrs") or {}),
        _shape_key((instr.get("out") or {}).get("type")),
        [_shape_key((o or {}).get("type")) for o in (instr.get("outs") or [])],
        [p for p in ([instr.get("name")] if instr.get("op") == "param" else [])],
    )


def _zip_progs(progs: Sequence[List[Dict[str, Any]]], path: str = ""):
    """Walk several structurally-identical programs in lockstep, yielding the
    corresponding instruction from each. Raises _StructMismatch (naming the
    position and probe) as soon as the structures diverge."""
    n = len(progs[0])
    for pi, p in enumerate(progs[1:], 1):
        if len(p) != n:
            raise _StructMismatch(
                f"{path or 'program'}: {n} instructions at base vs {len(p)} at probe {pi}"
            )
    for j in range(n):
        group = [p[j] for p in progs]
        s0, k0 = _sig(group[0]), _nonint_key(group[0])
        for pi, ins in enumerate(group[1:], 1):
            what = group[0].get("opcode", group[0].get("op"))
            if _sig(ins) != s0:
                raise _StructMismatch(
                    f"{path}[{j}] {what}: op signature differs at probe {pi}"
                )
            if _nonint_key(ins) != k0:
                raise _StructMismatch(
                    f"{path}[{j}] {what}: a non-integer field differs at probe {pi} "
                    f"(type or string attr moves with the axis, not just constants)"
                )
        yield group
        for key in ("body", "then", "else"):
            if key in group[0]:
                if any(key not in ins for ins in group):
                    raise _StructMismatch(f"{path}[{j}]: region '{key}' missing")
                yield from _zip_progs([ins[key] for ins in group], f"{path}[{j}].{key}")


# --------------------------------------------------------------------------
# annotation: concrete integers -> affine intexprs over the axes
# --------------------------------------------------------------------------
def annotate_axes(
    base_prog: List[Dict[str, Any]],
    probe_progs: Sequence[List[Dict[str, Any]]],
    axis_names: List[str],
    points: List[Tuple[int, ...]],
    allow_cross: bool = False,
) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """Return a copy of `base_prog` with every axis-dependent integer replaced by
    an intexpr. `points[0]` is the base point; the rest line up with
    `probe_progs`. Returns (None, reason) if any field resists every candidate
    model (see `roller.fit_slot`)."""
    prog = copy.deepcopy(base_prog)
    progs: List[List[Dict[str, Any]]] = [prog] + list(probe_progs)
    try:
        groups = list(_zip_progs(progs))
    except _StructMismatch as e:
        return None, f"not constants-only: {e}"
    for group in groups:
        slots = [_int_slots(ins) for ins in group]
        if len({len(s) for s in slots}) != 1:
            what = group[0].get("opcode", group[0].get("op"))
            return None, f"{what}: differing integer-field counts across probes"
        for si, (v0, setter) in enumerate(slots[0]):
            vals = [s[si][0] for s in slots]
            if len(set(vals)) == 1:
                continue
            expr, why = fit_slot(axis_names, points, vals, allow_cross=allow_cross)
            if expr is None:
                what = group[0].get("opcode", group[0].get("op"))
                return None, f"{what}: {why}"
            setter(expr)
    return prog, ""


# --------------------------------------------------------------------------
# multi-axis kernel name
# --------------------------------------------------------------------------
def parameterize_name_nd(
    base_name: str,
    probe_names: Dict[str, str],
    base_point: Point,
    probe_points: Dict[str, Point],
) -> Tuple[Optional[str], str]:
    """Derive a `kernel_name_fmt` like ``conv_K{K}_N{N}`` by tokenizing the
    concrete names and replacing the digit runs that move with an axis.

    A digit run is attributed to an axis only when it changes exactly when that
    axis changes AND its two values are that axis's two values, which keeps a
    coincidentally-equal token (a tile width that happens to equal the shape)
    from being captured. Returns (None, reason) when a token moves for some other
    reason -- a name encoding a DERIVED quantity (a grid size) cannot be
    reconstructed from the axes, and guessing would emit the wrong symbol."""
    toks = re.findall(r"\d+|\D+", base_name)
    out = list(toks)
    for axis, pname in probe_names.items():
        ptoks = re.findall(r"\d+|\D+", pname)
        if len(ptoks) != len(toks):
            return None, (
                f"kernel name shape changes with {axis}: "
                f"{base_name!r} vs {pname!r}; pass name_fmt= explicitly"
            )
        for i, (a, b) in enumerate(zip(toks, ptoks)):
            if a == b:
                continue
            want_a, want_b = str(base_point[axis]), str(probe_points[axis][axis])
            if a == want_a and b == want_b:
                out[i] = "{" + axis + "}"
            else:
                return None, (
                    f"kernel name token {a!r}->{b!r} moves with {axis} but is not "
                    f"{want_a!r}->{want_b!r} (a derived quantity?); "
                    f"pass name_fmt= explicitly"
                )
    return "".join(out), ""


def _format_name(fmt: str, spec: Dict[str, Any]) -> str:
    """Substitute {name} from spec values, mirroring rv_format_name in the VM."""
    out = fmt
    for k, v in spec.items():
        out = out.replace("{" + str(k) + "}", str(v))
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def _record(build_at: Callable[..., Any], point: Point) -> Dict[str, Any]:
    _, recipe = record_kernel(lambda: build_at(**point))
    return recipe


class _Annotated:
    """One structural point's annotated program, plus what the probes revealed."""

    def __init__(
        self,
        prog: Optional[List[Dict[str, Any]]],
        reason: str,
        base_recipe: Dict[str, Any],
        probe_names: Dict[str, str],
        probe_points: Dict[str, Point],
        n_traces: int,
    ):
        self.prog = prog
        self.reason = reason
        self.base_recipe = base_recipe
        self.probe_names = probe_names
        self.probe_points = probe_points
        self.n_traces = n_traces


def _annotated_at(
    build_at: Callable[..., Any],
    base: Point,
    axes: Dict[str, List[int]],
    const_axes: List[str],
    axis_names: List[str],
    traces: Dict[Tuple[int, ...], Dict[str, Any]],
    cross: bool = False,
) -> _Annotated:
    """Record the base point plus one probe per constant axis per extra sample,
    then annotate every axis-dependent integer.

    One-axis probes cannot see an interaction between two axes, so `cross=True`
    additionally records the diagonal point of each axis pair -- the cheapest
    evidence that distinguishes `m*N + m*K` from `m*N*K`."""

    def key(p: Point) -> Tuple[int, ...]:
        return tuple(p[a] for a in axis_names)

    base_recipe = traces.setdefault(key(base), _record(build_at, base))
    probe_progs: List[List[Dict[str, Any]]] = []
    points: List[Tuple[int, ...]] = [key(base)]
    probe_names: Dict[str, str] = {}
    probe_points: Dict[str, Point] = {}
    for axis in const_axes:
        for v in axes[axis][1:]:
            p = dict(base, **{axis: v})
            r = traces.setdefault(key(p), _record(build_at, p))
            probe_progs.append(r["program"])
            points.append(key(p))
            if axis not in probe_names:
                probe_names[axis] = r.get("kernel_name_fmt", "")
                probe_points[axis] = p
    if cross:
        for a, b in itertools.combinations(const_axes, 2):
            p = dict(base, **{a: axes[a][1], b: axes[b][1]})
            if key(p) in {tuple(x) for x in points}:
                continue
            r = traces.setdefault(key(p), _record(build_at, p))
            probe_progs.append(r["program"])
            points.append(key(p))
    prog, reason = annotate_axes(
        base_recipe["program"], probe_progs, axis_names, points, allow_cross=cross
    )
    return _Annotated(
        prog, reason, base_recipe, probe_names, probe_points, 1 + len(probe_progs)
    )


def roll_nd(
    build_at: Callable[..., Any],
    *,
    axes: Dict[str, List[int]],
    structural_axis: Optional[str] = None,
    holdout_points: Optional[List[Point]] = None,
    spec_decl: Optional[List[Dict[str, str]]] = None,
    name_fmt: Optional[str] = None,
    extra_spec: Optional[Dict[str, Any]] = None,
    verify_points: Optional[List[Point]] = None,
    fit_cross_terms: bool = True,
) -> NdRollResult:
    """Roll `build_at` over SEVERAL axes at once into one parametric recipe.

    `axes` maps axis name -> sample values, first value being the base point;
    `build_at(**point)` must build and return a KernelDef. Every axis is fitted
    as an affine constant model except `structural_axis`, which additionally goes
    through the structural roller (runs -> static_for) on its first two samples.

    Verification covers the full cross product of `axes` plus `holdout_points`
    (override with `verify_points`), so separability is tested at points that
    were never fitted."""
    axis_names = list(axes)
    if not axis_names:
        raise ValueError("need at least one axis")
    for a, vals in axes.items():
        if len(vals) < 2:
            raise ValueError(f"axis {a!r} needs >= 2 sample values, got {vals!r}")
    if structural_axis is not None and structural_axis not in axes:
        raise ValueError(f"structural_axis {structural_axis!r} is not in axes")
    holdout_points = list(holdout_points or [])
    # `roll` takes bare axis values; here a point is a dict over ALL axes. Catch
    # the mix-up with a message instead of an IndexError deep in verification.
    for label, pts in (
        ("holdout_points", holdout_points),
        ("verify_points", verify_points or []),
    ):
        for p in pts:
            if not isinstance(p, dict):
                raise ValueError(
                    f"{label} takes one dict per point (e.g. "
                    f"{{{axis_names[0]!r}: 128}}), not a bare value {p!r} -- that is "
                    f"`roll`'s single-axis form"
                )
            missing = [a for a in axis_names if a not in p]
            if missing:
                raise ValueError(
                    f"{label} entry {p!r} is missing axes {missing}; every point "
                    f"must give a value for all of {axis_names}"
                )
    extra_spec = dict(extra_spec or {})
    const_axes = [a for a in axis_names if a != structural_axis]
    if spec_decl is None:
        spec_decl = [{"name": a, "kind": "int"} for a in axis_names]
        spec_decl += [
            {"name": k, "kind": "str" if isinstance(v, str) else "int"}
            for k, v in extra_spec.items()
        ]

    base: Point = {a: axes[a][0] for a in axis_names}
    traces: Dict[Tuple[int, ...], Dict[str, Any]] = {}

    def attempt(cross: bool) -> NdRollResult:
        return _roll_nd_once(
            build_at,
            base=base,
            axes=axes,
            const_axes=const_axes,
            axis_names=axis_names,
            traces=traces,
            structural_axis=structural_axis,
            holdout_points=holdout_points,
            spec_decl=spec_decl,
            name_fmt=name_fmt,
            extra_spec=extra_spec,
            verify_points=verify_points,
            cross=cross,
        )

    res = attempt(cross=False)
    # A cross term is invisible to one-axis probes -- along any single axis with
    # the others fixed, a product looks exactly like a straight line -- so it
    # surfaces at verification, not at inference. Retry with the product basis
    # only for the two failures it could explain.
    retryable = "verify failed" in res.reason or "fits no candidate model" in res.reason
    if not res.ok and fit_cross_terms and len(const_axes) > 1 and retryable:
        if not holdout_points:
            # Fitting products consumes grid points that verification relies on,
            # and a model means nothing on points it was fitted to. Say so rather
            # than quietly shrinking the evidence.
            return NdRollResult(
                None,
                f"{res.reason}; a cross term would explain this, but fitting one "
                f"consumes grid points that verification relies on -- pass "
                f"holdout_points= to enable it",
                res.points,
                traces,
                res.n_recorded,
            )
        res2 = attempt(cross=True)
        if res2.ok:
            return res2
        # Report whichever attempt got further, so the reason stays actionable.
        return res2 if len(res2.points) >= len(res.points) else res
    return res


def _roll_nd_once(
    build_at: Callable[..., Any],
    *,
    base: Point,
    axes: Dict[str, List[int]],
    const_axes: List[str],
    axis_names: List[str],
    traces: Dict[Tuple[int, ...], Dict[str, Any]],
    structural_axis: Optional[str],
    holdout_points: List[Point],
    spec_decl: List[Dict[str, str]],
    name_fmt: Optional[str],
    extra_spec: Dict[str, Any],
    verify_points: Optional[List[Point]],
    cross: bool,
) -> NdRollResult:
    """One inference+verification pass. `cross` widens the constant model to
    include pairwise products (and records the extra points that needs)."""
    a0 = _annotated_at(
        build_at, base, axes, const_axes, axis_names, traces, cross=cross
    )
    prog0, rec0 = a0.prog, a0.base_recipe
    n_recorded = a0.n_traces
    if prog0 is None:
        return NdRollResult(None, a0.reason, [], traces, n_recorded)

    # The structural axis needs its own second point, which doubles as that
    # axis's name probe -- so the name is derived only once both are in hand.
    a1: Optional[_Annotated] = None
    if structural_axis is not None:
        s1 = axes[structural_axis][1]
        base1 = dict(base, **{structural_axis: s1})
        a1 = _annotated_at(
            build_at, base1, axes, const_axes, axis_names, traces, cross=cross
        )
        n_recorded += a1.n_traces
        if a1.prog is None:
            return NdRollResult(
                None, f"at {structural_axis}={s1}: {a1.reason}", [], traces, n_recorded
            )

    if name_fmt is None:
        probe_names = dict(a0.probe_names)
        probe_points = dict(a0.probe_points)
        if structural_axis is not None and a1 is not None:
            probe_names[structural_axis] = a1.base_recipe.get("kernel_name_fmt", "")
            probe_points[structural_axis] = dict(
                base, **{structural_axis: axes[structural_axis][1]}
            )
        name_fmt, nreason = parameterize_name_nd(
            rec0.get("kernel_name_fmt", ""), probe_names, base, probe_points
        )
        if name_fmt is None:
            return NdRollResult(None, nreason, [], traces, n_recorded)
    assert name_fmt is not None

    if structural_axis is None:
        param = {
            "schema": "rocke.recipe/v1",
            "kernel_name_fmt": name_fmt,
            "spec": spec_decl,
            "attrs": rec0.get("attrs", {}),
            "program": prog0,
        }
    else:
        assert a1 is not None
        s0, s1 = axes[structural_axis][0], axes[structural_axis][1]
        param = roll_two(
            {"program": prog0, "kernel_name_fmt": rec0.get("kernel_name_fmt", "")},
            {
                "program": a1.prog,
                "kernel_name_fmt": a1.base_recipe.get("kernel_name_fmt", ""),
                "attrs": a1.base_recipe.get("attrs", {}),
            },
            structural_axis,
            s0,
            s1,
            spec_decl,
            name_fmt,
        )
        if param is None:
            return NdRollResult(
                None,
                f"structural roll on {structural_axis} failed: {roll_two.last_reason}",
                [],
                traces,
                n_recorded,
            )

    # ---- verify: full cross product + holdouts, against fresh recordings ----
    if verify_points is None:
        verify_points = [
            dict(zip(axis_names, combo))
            for combo in itertools.product(*(axes[a] for a in axis_names))
        ]
        verify_points += holdout_points
    verified: List[Point] = []
    for point in verify_points:
        pk = tuple(point[a] for a in axis_names)
        concrete = traces.get(pk) or _record(build_at, point)
        traces[pk] = concrete
        spec = {**point, **extra_spec}
        try:
            exp = expand_recipe(param, spec)
        except Exception as e:  # ExpandError and friends -> refuse, never raise
            return NdRollResult(
                None, f"expand failed at {point}: {e}", verified, traces, n_recorded
            )
        if not recipes_equiv(exp, concrete):
            return NdRollResult(
                None,
                f"verify failed at {point}: {equiv_reason(exp, concrete)}",
                verified,
                traces,
                n_recorded,
            )
        want = concrete.get("kernel_name_fmt", "")
        got = _format_name(param["kernel_name_fmt"], spec)
        if want and got != want:
            return NdRollResult(
                None,
                f"kernel name at {point}: {got!r} != {want!r}",
                verified,
                traces,
                n_recorded,
            )
        verified.append(point)
    return NdRollResult(param, "", verified, traces, n_recorded)


def roll_nd_report(result: NdRollResult) -> str:
    """One line: how many points one recipe covers, and at what recording cost."""

    def deep(prog: List[Dict[str, Any]]) -> int:
        n = 0
        for i in prog:
            if i.get("op") != "param":
                n += 1
            for k in ("body", "then", "else"):
                if k in i:
                    n += deep(i[k])
        return n

    if not result.ok:
        return f"NOT ROLLED: {result.reason}"
    pn = deep(result.recipe["program"])
    concrete = sum(deep(result.traces[t]["program"]) for t in result.traces)
    return (
        f"rolled: 1 recipe ({pn} ops) covers {len(result.points)} points "
        f"from {result.n_recorded} recorded traces "
        f"(concrete total={concrete} ops; {concrete / pn:.1f}x)"
    )
