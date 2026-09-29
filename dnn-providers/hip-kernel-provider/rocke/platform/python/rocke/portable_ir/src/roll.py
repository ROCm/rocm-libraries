# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# roll.py -- the single-axis rolling driver.
#
#   roll(build_at, axis="D", sample_points=[64,128], holdout_points=[256], ...)
#
# It records concrete traces from an UNMODIFIED production builder (via the
# build-time interception recorder), infers ONE parametric recipe over the given
# structural axis, then VERIFIES it with the recipe_expand oracle against every
# sample AND held-out point. The held-out points guard against two-point
# overfitting. On any failure it returns (None, reason) -- the caller keeps the
# concrete per-shape recipes (graceful degradation; never a wrong roll).
#
# This is a THIN WRAPPER over roll_nd with a single axis, kept because a
# one-axis roll is common and reads better without wrapping every value in a
# dict. It used to be a parallel implementation, which was a slow leak: the two
# had drifted into different builder calling conventions, and only this one
# skipped the kernel-name check, so the more convenient entry point was also the
# less safe one. Delegating fixes that by construction. Verified byte-identical
# to the old implementation on all seven gated axes before the switch.

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from rocke.portable_ir.src.roll_nd import roll_nd


class RollResult:
    def __init__(
        self,
        recipe: Optional[Dict[str, Any]],
        reason: str,
        traces: Dict[int, Dict[str, Any]],
    ):
        self.recipe = recipe
        self.reason = reason  # "" on success
        self.traces = traces  # axis value -> concrete recipe (validated)

    @property
    def ok(self) -> bool:
        return self.recipe is not None


def roll(
    build_at: Callable[[int], Any],
    *,
    axis: str,
    sample_points: List[int],
    holdout_points: Optional[List[int]] = None,
    spec_decl: Optional[List[Dict[str, str]]] = None,
    name_fmt: Optional[str] = None,
    extra_spec: Optional[Dict[str, Any]] = None,
) -> RollResult:
    """Record `build_at(value)` at the given axis values and roll over `axis`.

    `build_at(value)` must build and return a KernelDef (it may construct its
    IRBuilder internally; recording is automatic). `extra_spec` supplies any
    additional spec values (e.g. {"dtype": "fp16"}) needed when expanding.

    Note the calling convention: `build_at` takes the value POSITIONALLY here,
    while `roll_nd` passes every axis by keyword."""
    if len(sample_points) < 2:
        raise ValueError("need >= 2 sample_points to infer a roll")
    nd = roll_nd(
        lambda **point: build_at(point[axis]),
        axes={axis: list(sample_points)},
        structural_axis=axis,
        holdout_points=[{axis: v} for v in (holdout_points or [])],
        spec_decl=spec_decl,
        name_fmt=name_fmt,
        extra_spec=extra_spec,
    )
    # roll_nd keys traces by point tuple; a single axis makes those 1-tuples.
    return RollResult(
        nd.recipe, nd.reason, {pt[0]: rec for pt, rec in nd.traces.items()}
    )


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
    return (
        f"rolled: parametric={pn} ops covers {len(cov)} shapes "
        f"{cov} (concrete total={concrete_total} ops; "
        f"{concrete_total / pn:.1f}x)"
    )


def _demo() -> int:
    """Roll two real kernels over head_size and report compression. Each roll
    is verified by the recipe_expand oracle at sampled AND held-out shapes."""
    from rocke.portable_ir.examples import export_mha, qk_block

    cases = [
        (
            "unified-attention-2d fp16",
            lambda D: export_mha.build("fp16", D, 2048, 1, 32, 1),
            {"dtype": "fp16"},
        ),
        ("qk_block f16", lambda D: qk_block.build_qk_block(D, "f16"), {"dtype": "f16"}),
    ]
    spec_decl = [{"name": "D", "kind": "int"}, {"name": "dtype", "kind": "str"}]
    rc = 0
    for label, build_at, extra in cases:
        r = roll(
            build_at,
            axis="D",
            sample_points=[64, 128],
            holdout_points=[256, 192, 96, 512],
            spec_decl=spec_decl,
            extra_spec=extra,
        )
        print(f"{label:<28} {roll_report(r)}")
        rc |= 0 if r.ok else 1
    return rc


if __name__ == "__main__":
    raise SystemExit(_demo())
