# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# guard.py -- derive a per-recipe admission guard over a ROLLED recipe's free
# axes, so a JIT caller can be told "this recipe does not serve that shape"
# without compiling anything.
#
# Why this exists
# ---------------
# A concrete recipe needs no guard: it was recorded from one build that the
# kernel accepted, so its mere presence in the bundle is the validity statement,
# and a lookup miss is the rejection. Rolling breaks that. A rolled recipe
# generalizes the EMISSION over an axis -- it says nothing about which values of
# that axis the kernel would have agreed to build. The generator verified a
# handful of points; the recipe will happily replay at any value the caller
# supplies, including ones whose kernel would never have compiled. Rolling is
# what creates the need for a JIT-time check, not what removes it.
#
# The honest source of truth for "would the kernel have agreed" is the family's
# own Python gate (supports_tiled_2d, is_valid_spec, ...). Porting those to C++
# is a large, permanently-drifting job. This module takes the other route:
# MEASURE the gate at generation time, with the recipe's baked values already
# fixed, and compile what is left into a few intexpr predicates the existing C
# VM can already evaluate. No new evaluator, no port, and the guard cannot drift
# from the gate any faster than the bundle itself does.
#
# The contract is one-way and that asymmetry is the whole design:
#
#     check(b) accepts  ==>  gate(baked u b, arch) accepts
#
# A false ACCEPT is a bug -- it hands hipDNN a configuration the kernel never
# supported, which is exactly the failure this feature exists to prevent. A
# false REJECT is a coverage loss -- a shape that would have worked is refused,
# and hipDNN falls back to another provider. So every fallback below degrades
# toward the strict side, and derivation refuses to ship a guard it could not
# verify (see verify_guard).
#
#   python3 -m rocke.portable_ir.src.guard    # worked example on a toy gate

from __future__ import annotations

import itertools
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from rocke.portable_ir.utils.recipe_expand import GUARD_SCHEMA, check_guard

# A gate takes a complete binding of the free axes (the baked values are already
# captured in the closure) and answers whether the kernel supports it.
GateFn = Callable[[Dict[str, Any]], bool]

Point = Dict[str, Any]


class GuardDerivationError(RuntimeError):
    """Derivation could not produce a guard it is willing to stand behind."""


# --------------------------------------------------------------------------
# intexpr construction
# --------------------------------------------------------------------------
# Guards reuse the recipe intexpr language exactly as it is -- no new node kinds,
# so recipe_expand.eval_intexpr and rv_int in recipe_vm.cpp both already evaluate
# them, and the existing CI gate that pins those two against each other covers
# guards for free. Booleans are 0/1 ints, so `and` is multiplication and `or` is
# "sum is positive".
def _spec(name: str) -> Dict[str, Any]:
    return {"spec": name}


def _and(a: Any, b: Any) -> Dict[str, Any]:
    return {"mul": [a, b]}


def _or(a: Any, b: Any) -> Dict[str, Any]:
    return {"gt": [{"add": [a, b]}, 0]}


def _all(xs: Sequence[Any]) -> Any:
    if not xs:
        return 1
    out = xs[0]
    for x in xs[1:]:
        out = _and(out, x)
    return out


def _any_of(xs: Sequence[Any]) -> Any:
    if not xs:
        return 0
    out = xs[0]
    for x in xs[1:]:
        out = _or(out, x)
    return out


def _eq_val(axis: str, v: Any) -> Dict[str, Any]:
    """Equality against one value, for either an int or a string axis."""
    if isinstance(v, str):
        return {"spec_str_eq": [axis, v]}
    return {"eq": [_spec(axis), int(v)]}


def _point_eq(point: Point) -> Any:
    return _all([_eq_val(a, v) for a, v in sorted(point.items())])


# --------------------------------------------------------------------------
# per-axis compression
# --------------------------------------------------------------------------
def _arithmetic_stride(vals: Sequence[int]) -> Optional[int]:
    """The common stride if `vals` is exactly an arithmetic progression."""
    if len(vals) < 3:
        return None
    stride = vals[1] - vals[0]
    if stride <= 0:
        return None
    if any(b - a != stride for a, b in zip(vals, vals[1:])):
        return None
    return stride


def axis_rules(axis: str, legal: Sequence[Any]) -> List[Dict[str, Any]]:
    """Compile one axis's measured legal set into rules that accept exactly it.

    Two forms, both exact -- compression never widens what is accepted, so it
    can never be the source of a false accept:

      stride  three or more values in an arithmetic progression become a bounds
              rule plus a divisibility rule. This is the common case (head sizes
              16..256 by 16) and it stays two small rules however long the run.
      enum    anything else becomes an or-chain of equalities.

    The stride form emits the bounds rule FIRST. Python's `%` floors where C's
    truncates, so the two guard evaluators do not agree about `(x - lo) % stride`
    for a negative x; ordering bounds first means evaluation stops before the
    `mod` is reached. That is belt and braces rather than the only defence --
    the rule asks whether the remainder is ZERO, and divisibility is the same
    question under either convention -- but a future rule that divides, or that
    compares a remainder against something other than zero, would depend on the
    ordering alone. Keep bounds first."""
    vals = list(legal)
    if not vals:
        raise GuardDerivationError(f"axis '{axis}' has no legal values")
    if all(isinstance(v, str) for v in vals):
        return [
            {
                "reason": f"{axis} must be one of {', '.join(sorted(vals))}",
                "pred": _any_of([_eq_val(axis, v) for v in sorted(vals)]),
            }
        ]

    ints = sorted(int(v) for v in vals)
    stride = _arithmetic_stride(ints)
    lo, hi = ints[0], ints[-1]
    if stride is None:
        return [
            {
                "reason": f"{axis} must be one of {{{', '.join(str(v) for v in ints)}}}",
                "pred": _any_of([_eq_val(axis, v) for v in ints]),
            }
        ]

    rules = [
        {
            "reason": f"{axis} must be in [{lo}, {hi}]",
            "pred": _and({"ge": [_spec(axis), lo]}, {"le": [_spec(axis), hi]}),
        }
    ]
    if stride > 1:
        rules.append(
            {
                "reason": f"{axis} must be {lo} plus a multiple of {stride}",
                "pred": {"eq": [{"mod": [{"sub": [_spec(axis), lo]}, stride]}, 0]},
            }
        )
    return rules


# --------------------------------------------------------------------------
# coupling between axes
# --------------------------------------------------------------------------
# Per-axis rules cannot express legality that lives in a PAIR of axes -- that
# num_kv_heads has to divide num_query_heads, that a tile has to fit a block.
# Each candidate below is a cheap closed form that such a constraint usually
# takes; derivation keeps only the ones the measured data supports.
def _coupling_candidates(
    axes: Sequence[str], accepted: Sequence[Point]
) -> List[Tuple[str, Callable[[Point], bool], Any]]:
    ints = [a for a in axes if accepted and isinstance(accepted[0].get(a), int)]
    out: List[Tuple[str, Callable[[Point], bool], Any]] = []
    for a, b in itertools.permutations(ints, 2):
        out.append(
            (
                f"{a} must divide {b}",
                lambda p, a=a, b=b: p[a] != 0 and p[b] % p[a] == 0,
                {"eq": [{"mod": [_spec(b), _spec(a)]}, 0]},
            )
        )
        out.append(
            (
                f"{a} must be at most {b}",
                lambda p, a=a, b=b: p[a] <= p[b],
                {"le": [_spec(a), _spec(b)]},
            )
        )
    for a, b in itertools.combinations(ints, 2):
        if not accepted:
            continue
        cap = max(p[a] * p[b] for p in accepted)
        out.append(
            (
                f"{a} * {b} must be at most {cap}",
                lambda p, a=a, b=b, cap=cap: p[a] * p[b] <= cap,
                {"le": [{"mul": [_spec(a), _spec(b)]}, cap]},
            )
        )
    return out


def _fit_coupling(
    axes: Sequence[str], accepted: Sequence[Point], rejected: Sequence[Point]
) -> Tuple[List[Dict[str, Any]], List[Point]]:
    """Greedily pick coupling rules that explain rejects without costing accepts.

    A candidate is admissible only if it accepts every point the gate accepted:
    a rule that trims a real accept is a coverage loss we took without being
    asked. Among the admissible ones we repeatedly take whichever explains the
    most still-unexplained rejects, which keeps the rule count near-minimal
    without paying for an exact set cover.

    Returns (rules, still-unexplained rejects)."""
    remaining = list(rejected)
    rules: List[Dict[str, Any]] = []
    pool = [
        c
        for c in _coupling_candidates(axes, accepted)
        if all(c[1](p) for p in accepted)
    ]
    while remaining and pool:
        best, best_hit = None, 0
        for cand in pool:
            hit = sum(1 for p in remaining if not cand[1](p))
            if hit > best_hit:
                best, best_hit = cand, hit
        if best is None:
            break
        rules.append({"reason": best[0], "pred": best[2]})
        remaining = [p for p in remaining if best[1](p)]
        pool.remove(best)
    return rules, remaining


# --------------------------------------------------------------------------
# gate adapters
# --------------------------------------------------------------------------
def gate_from_spec(
    make_spec: Callable[..., Any],
    *,
    admits: Optional[Callable[[Any], Any]] = None,
    probe: Optional[Callable[..., Any]] = None,
    coherent: Optional[Callable[[Point], bool]] = None,
) -> GateFn:
    """Wrap a kernel family's own gating into a single point predicate.

    Mirrors the layering roll_regimes.legal_values documents, because a guard
    derived from a weaker gate than the one rolling used would disagree with the
    recipe it is attached to:

      make_spec  the spec dataclass __post_init__, which rejects per-field
      admits     the family's supports_* / is_valid_spec, where cross-field
                 constraints live; accepts a bool or the (ok, reason) pair
      probe      an actual build, the slowest and most truthful layer -- the only
                 one that catches constraints asserted deep inside the builder
      coherent   constraints the kernel depends on but does not itself check;
                 passing one here is a stopgap and the real fix belongs in the
                 kernel, since a guard can only be as honest as its gate"""

    def gate(point: Point) -> bool:
        if coherent is not None and not coherent(point):
            return False
        try:
            spec = make_spec(**point)
        except Exception:
            return False
        if admits is not None:
            try:
                verdict = admits(spec)
            except Exception:
                return False
            if not (verdict[0] if isinstance(verdict, tuple) else verdict):
                return False
        if probe is not None:
            try:
                probe(**point)
            except Exception:
                return False
        return True

    return gate


# --------------------------------------------------------------------------
# derivation
# --------------------------------------------------------------------------
def _find_reference(
    gate: GateFn, candidates: Dict[str, Sequence[Any]], rng: random.Random, tries: int
) -> Point:
    """A point the gate accepts, to hold the other axes at while measuring one.

    Midpoints first because kernel domains tend to be legal in the middle and
    ragged at the ends, then random draws. Everything downstream is measured
    relative to this point, so failing to find one is a hard stop rather than a
    reason to guess."""
    axes = sorted(candidates)
    mid = {a: list(candidates[a])[len(candidates[a]) // 2] for a in axes}
    if gate(mid):
        return mid
    first = {a: list(candidates[a])[0] for a in axes}
    if gate(first):
        return first
    for _ in range(tries):
        p = {a: rng.choice(list(candidates[a])) for a in axes}
        if gate(p):
            return p
    raise GuardDerivationError(
        "no reference point: the gate rejected every sampled combination of "
        f"{axes}. Either the candidate domains are wrong or the recipe's baked "
        "values are not supported at all."
    )


def _measure(
    gate: GateFn, candidates: Dict[str, Sequence[Any]], pool: Sequence[Point]
) -> Dict[str, List[Any]]:
    """Legal values per axis: those the gate accepts against ANY reference in
    `pool`, holding the other axes at that reference."""
    legal: Dict[str, List[Any]] = {}
    for axis in sorted(candidates):
        legal[axis] = [
            v for v in candidates[axis] if any(gate({**r, axis: v}) for r in pool)
        ]
    return legal


def _marginals(
    gate: GateFn,
    candidates: Dict[str, Sequence[Any]],
    ref: Point,
    rng: random.Random,
    pool_cap: int,
    pool_scan: int,
) -> Tuple[Dict[str, List[Any]], List[Point]]:
    """Per-axis domains measured against several reference points, not one.

    Measuring an axis while the others sit at a single reference reports what is
    legal ALONGSIDE THAT REFERENCE, which for coupled axes is a slice of the
    real domain rather than the domain. On a gate where the block size has to
    divide the head size, one reference with block 32 reports head sizes in
    steps of 32 and hides every legal head size only reachable with block 16 --
    so the guard refuses shapes the kernel supports. It stays SOUND, which is
    why this is a coverage question and not a correctness one, but a guard that
    throws away most of its axis is not earning the rolled recipe it protects.

    Two rounds fix it. The first finds a rough domain from one reference; a pool
    of accepted points then serves as references for the second, and a value
    survives if ANY of them accepts it. That is the marginal domain, which is
    what per-axis rules are supposed to describe -- the joint constraint is the
    coupling step's job, not theirs.

    The pool is seeded from the WHOLE candidate space, not just from the first
    round's domain, and that matters for tightly coupled axes. Where two axes
    are near-equality coupled (grouped-query attention, where the KV head count
    divides the query head count), the first round can collapse both to the
    single value the reference happened to use; a pool drawn from inside that
    result cannot escape it, and every later step inherits the collapse. Points
    drawn from outside it can.

    Costs at most |pool| gate calls per candidate plus `pool_scan` for the seed
    search, though the `any` stops at the first reference that accepts, so the
    common case is one. Lower both when the gate has to build."""
    legal = _measure(gate, candidates, [ref])
    if any(not v for v in legal.values()) or pool_cap <= 1:
        return legal, [ref]

    near, _ = _sample_cross(legal, max(pool_cap * 8, 64), rng)
    wide, _ = _sample_cross(candidates, pool_scan, rng)
    pool = [ref]
    seen = {a: {ref[a]} for a in ref}
    for p in itertools.chain(near, wide):
        if len(pool) >= pool_cap:
            break
        # A reference earns its place by bringing an axis value the pool does
        # not have yet; near-identical references just measure the same slice
        # again at full price. The novelty test is free, so it runs first.
        if any(p[a] not in seen[a] for a in p) and gate(p):
            pool.append(p)
            for a in p:
                seen[a].add(p[a])
    return _measure(gate, candidates, pool), pool


def _sample_cross(
    candidates: Dict[str, Sequence[Any]], cap: int, rng: random.Random
) -> Tuple[List[Point], bool]:
    """Up to `cap` points of a cross product; second value says whether the
    enumeration was exhaustive. Exhaustive matters: only then may derivation fall
    back to a blocklist, because only then is "nothing else exists in here" true
    rather than merely unobserved."""
    axes = sorted(candidates)
    total = 1
    for a in axes:
        total *= len(candidates[a])
    if total <= cap:
        return [
            dict(zip(axes, combo))
            for combo in itertools.product(*[list(candidates[a]) for a in axes])
        ], True
    seen, pts = set(), []
    for _ in range(cap * 4):
        if len(pts) >= cap:
            break
        p = {a: rng.choice(list(candidates[a])) for a in axes}
        k = tuple(sorted(p.items()))
        if k in seen:
            continue
        seen.add(k)
        pts.append(p)
    return pts, False


def derive_guard(
    gate: GateFn,
    candidates: Dict[str, Sequence[Any]],
    *,
    reference: Optional[Point] = None,
    arch: str = "",
    gate_name: str = "",
    verified: Optional[Sequence[Point]] = None,
    max_cross: int = 4096,
    oracle_samples: int = 512,
    pool_cap: int = 32,
    pool_scan: int = 512,
    seed: int = 0,
) -> Dict[str, Any]:
    """Measure `gate` over the recipe's free axes and compile a guard.

    `candidates` maps each FREE axis to the values worth considering -- the same
    domains the sweep drivers use. Anything outside them is not merely rejected,
    it was never asked about, so the domains should be generous.

    Six steps:

      1 reference   find one point the gate accepts
      2 measure     the marginal domain of each axis, over a pool of references
      3 compress    per-axis legal sets -> exact rules
      4 factorize   ask the gate across the cross product: does per-axis
                    legality imply joint legality?
      5 couple      if not, fit coupling rules, then degrade if they do not fit
      6 verify      check the result against the gate out of sample and refuse
                    to return an unsound one

    Raises GuardDerivationError rather than returning a guard it could not
    verify. A build that cannot prove its guard sound should fail loudly at
    generation time, which is the cheap place, instead of shipping a guard that
    waves through configurations the kernel cannot compile.

    `pool_cap` and `pool_scan` buy coverage, not correctness -- the result is
    sound at any setting, and larger values just refuse fewer shapes the kernel
    would have accepted. The default suits a declarative gate (spec constructor
    plus supports_*), where the whole derivation is a tenth of a second. Lower
    both when the gate includes a build probe, where each call costs a kernel
    build rather than a few comparisons."""
    rng = random.Random(seed)
    axes = sorted(candidates)
    if not axes:
        raise GuardDerivationError("no free axes to guard")

    ref = dict(reference) if reference else _find_reference(gate, candidates, rng, 256)
    if not gate(ref):
        raise GuardDerivationError(f"supplied reference point {ref} is not legal")

    legal, pool = _marginals(gate, candidates, ref, rng, pool_cap, pool_scan)
    for axis in axes:
        if not legal[axis]:
            raise GuardDerivationError(
                f"axis '{axis}' has no legal value at reference {ref}"
            )

    rules: List[Dict[str, Any]] = []
    for axis in axes:
        rules += axis_rules(axis, legal[axis])

    probes, exhaustive = _sample_cross(legal, max_cross, rng)
    verdicts = [(p, gate(p)) for p in probes]
    accepted = [p for p, ok in verdicts if ok]
    rejected = [p for p, ok in verdicts if not ok]
    if not accepted:
        raise GuardDerivationError(
            f"the gate rejected all {len(probes)} probed combinations even though "
            f"every axis is legal on its own at {ref}; a guard here would accept "
            "nothing"
        )

    method = "factored"
    if rejected:
        coupling, unexplained = _fit_coupling(axes, accepted, rejected)
        rules += coupling
        method = "coupled" if coupling else "factored"
        if unexplained and exhaustive:
            # Every point inside the per-axis rules was tested, so naming the bad
            # ones is complete rather than a guess at what else might be out
            # there. Cheaper than an allowlist and it keeps the interior open.
            method = "blocklist"
            rules.append(
                {
                    "reason": "combination is on the generator's reject list",
                    "pred": _all(
                        [
                            {"eq": [_point_eq(p), 0]}
                            for p in sorted(
                                unexplained, key=lambda d: sorted(d.items())
                            )
                        ]
                    ),
                }
            )
        elif unexplained:
            # Not exhaustive, so an unseen bad point may exist between the ones
            # we sampled. The only sound answer left is to accept nothing we did
            # not personally confirm. Strict, and it says so in `method`.
            method = "allowlist"
            rules = [
                {
                    "reason": "combination is not one of the generator-confirmed points",
                    "pred": _any_of(
                        [
                            _point_eq(p)
                            for p in sorted(accepted, key=lambda d: sorted(d.items()))
                        ]
                    ),
                }
            ]

    guard = {
        "schema": GUARD_SCHEMA,
        "free": axes,
        "rules": rules,
        "verified": [
            dict(sorted(p.items()))
            for p in (verified if verified is not None else accepted[:64])
        ],
        "derivation": {
            "method": method,
            "gate": gate_name,
            "arch": arch,
            "reference": dict(sorted(ref.items())),
            "references": len(pool),
            "measured": {a: len(legal[a]) for a in axes},
            "probed": len(probes),
            "exhaustive": exhaustive,
        },
    }

    report = verify_guard(
        guard, gate, candidates, samples=oracle_samples, seed=seed + 1
    )
    guard["derivation"]["oracle"] = report.summary()
    if not report.sound:
        raise GuardDerivationError(
            f"derived guard is UNSOUND: it accepts {len(report.unsound)} point(s) "
            f"the gate rejects, e.g. {report.unsound[0]}. Shipping it would let "
            "hipDNN compile a configuration the kernel does not support."
        )
    return guard


# --------------------------------------------------------------------------
# oracle
# --------------------------------------------------------------------------
class OracleReport:
    """What the guard and the gate disagreed about, out of sample.

    The two disagreements are not symmetric and are never reported as one
    number. `unsound` is a build break: the guard admitted something the kernel
    does not support, which is the exact failure the guard exists to prevent.
    `strict` is a coverage loss: real configurations that will now be refused and
    fall back to another provider. It is worth watching -- a guard that rejects
    most of its own axis is not earning the rolled recipe -- but it is safe."""

    def __init__(self) -> None:
        self.checked = 0
        self.agreed = 0
        self.unsound: List[Point] = []
        self.strict: List[Point] = []

    @property
    def sound(self) -> bool:
        return not self.unsound

    def summary(self) -> Dict[str, Any]:
        return {
            "checked": self.checked,
            "agreed": self.agreed,
            "unsound": len(self.unsound),
            "strict": len(self.strict),
        }

    def __str__(self) -> str:
        s = (
            f"oracle: {self.agreed}/{self.checked} agreed, "
            f"{len(self.unsound)} unsound, {len(self.strict)} over-strict"
        )
        if self.unsound:
            s += f"\n  UNSOUND example: {self.unsound[0]}"
        if self.strict:
            s += f"\n  over-strict example: {self.strict[0]}"
        return s


def guard_accepts(guard: Dict[str, Any], point: Point) -> bool:
    ok, _ = check_guard(
        guard,
        {k: int(v) for k, v in point.items() if not isinstance(v, str)},
        {k: v for k, v in point.items() if isinstance(v, str)},
    )
    return ok


def verify_guard(
    guard: Dict[str, Any],
    gate: GateFn,
    candidates: Dict[str, Sequence[Any]],
    *,
    samples: int = 512,
    seed: int = 1,
) -> OracleReport:
    """Compare a derived guard against the real gate on points it was not fitted to.

    Derivation measured the gate along a cross through one reference point and
    across the cross product of what that found. Both are shapes in the space,
    and a guard can fit its own measurements perfectly while being wrong just
    outside them. So the oracle draws from the FULL candidate space -- including
    values that never entered any legal set, which is where a too-permissive
    guard shows itself -- and asks both sides.

    This is what separates "derived from the gate" from "known to agree with the
    gate", and it is the reason derive_guard is willing to emit a machine-fitted
    predicate at all."""
    rng = random.Random(seed)
    report = OracleReport()
    pts, _ = _sample_cross(candidates, samples, rng)
    for p in pts:
        g, k = guard_accepts(guard, p), gate(p)
        report.checked += 1
        if g == k:
            report.agreed += 1
        elif g:
            report.unsound.append(p)
        else:
            report.strict.append(p)
    return report


# --------------------------------------------------------------------------
# attaching to a recipe
# --------------------------------------------------------------------------
def recipe_axes(recipe: Dict[str, Any]) -> List[str]:
    """The free axis names of a recipe. Entries of "spec" are either bare names
    or {"name":..., "kind":...} records depending on which authoring path
    produced the recipe; both forms are in the tree."""
    out = []
    for entry in recipe.get("spec") or []:
        out.append(entry["name"] if isinstance(entry, dict) else entry)
    return out


def attach_guard(recipe: Dict[str, Any], guard: Dict[str, Any]) -> Dict[str, Any]:
    """Put `guard` on `recipe`, checking the two agree about the free axes.

    A guard naming an axis the recipe does not take, or missing one it does, is a
    wiring mistake that would otherwise surface as a confusing runtime rejection
    (or as no protection at all on the unguarded axis)."""
    spec_axes = set(recipe_axes(recipe))
    guard_axes = set(guard.get("free") or [])
    if spec_axes != guard_axes:
        raise GuardDerivationError(
            f"guard covers {sorted(guard_axes)} but the recipe's free axes are "
            f"{sorted(spec_axes)}"
        )
    out = dict(recipe)
    out["guard"] = guard
    return out


if __name__ == "__main__":
    # A toy gate with one per-axis constraint and one coupling, so the derived
    # guard has to find both: head_size is 16..256 by 16, and block_size has to
    # divide it.
    def toy(p: Dict[str, Any]) -> bool:
        h, b = p["head_size"], p["block_size"]
        return 16 <= h <= 256 and h % 16 == 0 and b in (16, 32, 64) and h % b == 0

    cands = {"head_size": [8 * i for i in range(1, 40)], "block_size": [16, 32, 64, 96]}
    g = derive_guard(toy, cands, gate_name="toy", arch="gfx950")
    print(f"method={g['derivation']['method']}  rules={len(g['rules'])}")
    for r in g["rules"]:
        print(f"  - {r['reason']}")
    print(verify_guard(g, toy, cands, samples=400))
