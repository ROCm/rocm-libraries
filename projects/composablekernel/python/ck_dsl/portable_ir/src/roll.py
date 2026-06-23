# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# roll.py -- the productized rolling driver.
#
#   roll(build_at, axis="D", sample_points=[64,128], holdout_points=[256], ...)
#
# It records concrete traces from an UNMODIFIED production builder (via the
# build-time interception recorder), infers ONE parametric recipe over the given
# structural axis (roller.roll_two), then VERIFIES the parametric recipe with the
# recipe_expand oracle against every sample AND held-out point. The held-out
# points guard against two-point overfitting. On any failure it returns
# (None, reason) -- the caller keeps the concrete per-shape recipes (graceful
# degradation; never a wrong roll).

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from ck_dsl.portable_ir.utils.recipe_expand import equiv_reason, expand_recipe, recipes_equiv
from ck_dsl.portable_ir.src.recording_builder import kernel_to_recipe, record_kernel
from ck_dsl.portable_ir.src.roller import roll_two


class RollResult:
    def __init__(self, recipe: Optional[Dict[str, Any]], reason: str,
                 traces: Dict[int, Dict[str, Any]]):
        self.recipe = recipe
        self.reason = reason          # "" on success
        self.traces = traces          # axis value -> concrete recipe (validated)

    @property
    def ok(self) -> bool:
        return self.recipe is not None


def _record(build_at: Callable[[int], Any], v: int) -> Dict[str, Any]:
    _, recipe = record_kernel(lambda: build_at(v))
    return recipe


def roll(build_at: Callable[[int], Any], *, axis: str,
         sample_points: List[int], holdout_points: Optional[List[int]] = None,
         spec_decl: Optional[List[Dict[str, str]]] = None,
         name_fmt: Optional[str] = None,
         extra_spec: Optional[Dict[str, Any]] = None) -> RollResult:
    """Record `build_at(value)` at the given axis values and roll over `axis`.

    `build_at(value)` must build and return a KernelDef (it may construct its
    IRBuilder internally; recording is automatic). `extra_spec` supplies any
    additional spec values (e.g. {"dtype": "fp16"}) needed when expanding."""
    if len(sample_points) < 2:
        raise ValueError("need >= 2 sample_points to infer a roll")
    holdout_points = holdout_points or []
    extra_spec = extra_spec or {}
    spec_decl = spec_decl or [{"name": axis, "kind": "int"}]

    traces: Dict[int, Dict[str, Any]] = {v: _record(build_at, v) for v in sample_points}

    a0, a1 = sample_points[0], sample_points[1]
    param = roll_two(traces[a0], traces[a1], axis, a0, a1, spec_decl, name_fmt)
    if param is None:
        return RollResult(None, f"roll inference failed: {roll_two.last_reason}", traces)

    # Verify: expand the parametric recipe and compare to an independent concrete
    # recording at every sample AND held-out point.
    for v in list(sample_points) + list(holdout_points):
        concrete = traces.get(v) or _record(build_at, v)
        traces[v] = concrete
        spec = {axis: v, **extra_spec}
        exp = expand_recipe(param, spec)
        if not recipes_equiv(exp, concrete):
            return RollResult(None, f"verify failed at {axis}={v}: "
                              f"{equiv_reason(exp, concrete)}", traces)
    return RollResult(param, "", traces)


def roll_report(result: RollResult) -> str:
    """One-line storage summary: parametric size vs the concrete traces it covers."""
    def deep(prog):
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
    cov = sorted(result.traces)
    concrete_total = sum(deep(result.traces[v]["program"]) for v in cov)
    return (f"rolled: parametric={pn} ops covers {len(cov)} shapes "
            f"{cov} (concrete total={concrete_total} ops; "
            f"{concrete_total / pn:.1f}x)")


def _demo() -> int:
    """Roll two real kernels over head_size and report compression. Each roll
    is verified by the recipe_expand oracle at sampled AND held-out shapes."""
    from ck_dsl.portable_ir.examples import export_mha, qk_block

    cases = [
        ("unified-attention-2d fp16",
         lambda D: export_mha.build("fp16", D, 2048, 1, 32, 1),
         {"dtype": "fp16"}),
        ("qk_block f16",
         lambda D: qk_block.build_qk_block(D, "f16"),
         {"dtype": "f16"}),
    ]
    spec_decl = [{"name": "D", "kind": "int"}, {"name": "dtype", "kind": "str"}]
    rc = 0
    for label, build_at, extra in cases:
        r = roll(build_at, axis="D", sample_points=[64, 128],
                 holdout_points=[256, 192, 96, 512], spec_decl=spec_decl,
                 extra_spec=extra)
        print(f"{label:<28} {roll_report(r)}")
        rc |= 0 if r.ok else 1
    return rc


if __name__ == "__main__":
    raise SystemExit(_demo())
