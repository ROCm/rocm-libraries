# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# roll_regimes.py -- specialized recipes when ONE recipe cannot span an axis.
#
# `roll` and `roll_nd` assume a family is uniform along an axis: same structure
# everywhere, only constants and trip counts moving. Plenty of axes are not like
# that. A kernel picks a different code path past a threshold, pads its LDS
# differently at one head size, or re-vectorizes its loads -- and then no single
# parametric recipe can be right, because the thing being described is not one
# program.
#
# Forcing it produces a refusal, and a refusal costs the whole axis: every value
# falls back to its own concrete recipe. Specializing gets most of the win back.
# If an axis has 12 legal values in 2 regimes, 2 recipes cover all 12; that is
# 6x, just not 12x. The compression is worse and the correctness is identical --
# every regime is still verified byte-for-byte at every value it claims.
#
# Two ideas do the work here:
#
#   1. ASK THE KERNEL WHICH VALUES EXIST. A spec's __post_init__ already encodes
#      what it accepts ("head_size must be 64 or 128", "seqlen_kv must be a
#      multiple of block_n"). `legal_values` runs candidates through that
#      validation instead of guessing, so regimes are built over the domain the
#      kernel actually has -- and an axis whose legal set is tiny gets found out
#      before anyone spends effort rolling it.
#
#   2. LET THE ROLLER DRAW THE BOUNDARIES. Rather than predicting where the
#      structure changes, extend a regime one value at a time and let
#      verification say when to stop. The first value that fails to verify starts
#      the next regime. Boundaries are therefore discovered, never assumed.

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from rocke.portable_ir.src.recording_builder import record_kernel
from rocke.portable_ir.src.roll import roll
from rocke.portable_ir.utils.recipe_expand import expand_recipe, recipes_equiv


class Regime:
    """One recipe plus the axis values it is verified to cover."""

    def __init__(
        self,
        values: List[int],
        recipe: Optional[Dict[str, Any]],
        reason: str,
        sampled: List[int],
    ):
        self.values = values
        self.recipe = recipe
        self.reason = reason  # "" when rolled
        self.sampled = sampled  # points recorded to infer it

    @property
    def rolled(self) -> bool:
        return self.recipe is not None

    def __repr__(self) -> str:
        how = (
            f"rolled from {self.sampled}"
            if self.rolled
            else f"concrete: {self.reason[:40]}"
        )
        return f"<Regime {self.values} {how}>"


class RegimeRollResult:
    def __init__(self, regimes: List[Regime], axis: str, values: List[int]):
        self.regimes = regimes
        self.axis = axis
        self.values = values

    @property
    def n_recipes(self) -> int:
        return len(self.regimes)

    @property
    def n_rolled(self) -> int:
        return sum(1 for r in self.regimes if r.rolled)

    def recipe_for(self, value: int) -> Optional[Dict[str, Any]]:
        for r in self.regimes:
            if value in r.values:
                return r.recipe
        return None


def legal_values(
    axis: str,
    candidates: Sequence[int],
    make_spec: Callable[..., Any],
    *,
    admits: Optional[Callable[..., Any]] = None,
    probe: Optional[Callable[..., Any]] = None,
) -> List[int]:
    """Candidates the kernel's OWN validation accepts.

    `make_spec(**{axis: v})` constructs the kernel's spec, whose `__post_init__`
    raises for anything unsupported. Using it beats hardcoding a range in a driver:
    the constraint stays in one place and stays right when the kernel changes. It
    also answers a question worth asking before any rolling work starts -- how many
    values does this axis even have? An axis with two legal values cannot repay a
    parametric recipe.

    A spec constructor is often not the whole gate, though, and trusting it alone
    overstates a domain. Two more layers are optional and worth using when they
    exist:

    `admits` is an admission predicate such as `supports_tiled_2d`, returning
    either a bool or the `(ok, reason)` pair those functions use. This is where
    kernels put the constraints that span fields -- `num_query_heads %
    num_kv_heads == 0`, tile budgets -- and a spec that validates each field alone
    will happily accept a combination the kernel rejects.

    `probe` actually builds the kernel and treats any exception as illegal. It is
    the slowest and the most truthful, and it is the only thing that catches
    constraints asserted deep in the builder rather than declared up front (the
    tiled kernels assert on head-size stripe alignment there, well after the spec
    is constructed)."""
    out = []
    for v in candidates:
        try:
            spec = make_spec(**{axis: v})
        except Exception:
            continue
        if admits is not None:
            try:
                verdict = admits(spec)
            except Exception:
                continue
            ok = verdict[0] if isinstance(verdict, tuple) else verdict
            if not ok:
                continue
        if probe is not None:
            try:
                probe(**{axis: v})
            except Exception:
                continue
        out.append(v)
    return out


def _verify_at(
    build_at: Callable[[int], Any],
    recipe: Dict[str, Any],
    value: int,
    axis: str,
    extra_spec: Dict[str, Any],
) -> bool:
    """Does `recipe` reproduce the real recording at `value`, byte-for-byte?"""
    try:
        _, concrete = record_kernel(lambda: build_at(value))
        exp = expand_recipe(recipe, {axis: value, **extra_spec})
    except Exception:
        return False
    if not recipes_equiv(exp, concrete):
        return False
    want = concrete.get("kernel_name_fmt", "")
    got = recipe.get("kernel_name_fmt", "")
    if want and "{" not in got:
        return got == want
    return True


def roll_regimes(
    build_at: Callable[[int], Any],
    *,
    axis: str,
    values: Sequence[int],
    extra_spec: Optional[Dict[str, Any]] = None,
    max_regimes: int = 16,
    **roll_kwargs: Any,
) -> RegimeRollResult:
    """Cover `values` with as FEW recipes as the kernel's structure allows.

    Walks the sorted values, rolling from the first two of a regime and then
    extending it by verification one value at a time. The first value that does
    not verify closes the regime and opens the next, so a boundary is a measured
    fact rather than a guess. A value that cannot join any regime is returned as
    its own concrete entry -- the same graceful degradation `roll` gives, applied
    per regime instead of to the whole axis.
    """
    extra_spec = dict(extra_spec or {})
    vals = sorted(set(values))
    if len(vals) < 2:
        raise ValueError(f"need >= 2 values to form a regime, got {vals}")
    regimes: List[Regime] = []
    i = 0
    while i < len(vals):
        if len(regimes) >= max_regimes:
            regimes.append(Regime(vals[i:], None, "max_regimes reached", []))
            break
        if i + 1 >= len(vals):
            # A lone trailing value cannot be inferred from; keep it concrete.
            regimes.append(Regime([vals[i]], None, "single value left over", []))
            break
        sampled = [vals[i], vals[i + 1]]
        r = roll(
            build_at,
            axis=axis,
            sample_points=sampled,
            extra_spec=extra_spec,
            **roll_kwargs,
        )
        if not r.ok:
            regimes.append(Regime([vals[i]], None, r.reason, sampled))
            i += 1
            continue
        covered = list(sampled)
        j = i + 2
        while j < len(vals) and _verify_at(
            build_at, r.recipe, vals[j], axis, extra_spec
        ):
            covered.append(vals[j])
            j += 1
        regimes.append(Regime(covered, r.recipe, "", sampled))
        i = j
    return RegimeRollResult(regimes, axis, vals)


def regime_report(result: RegimeRollResult) -> str:
    """How many recipes the axis costs, and where the structure changes."""
    lines = [
        f"{result.axis}: {len(result.values)} legal values -> "
        f"{result.n_recipes} recipe(s), {result.n_rolled} rolled"
    ]
    for r in result.regimes:
        if r.rolled:
            lines.append(
                f"    {r.values[0]}..{r.values[-1]} ({len(r.values)} values) "
                f"from {len(r.sampled)} traces"
            )
        else:
            lines.append(f"    {r.values} concrete -- {r.reason[:70]}")
    return "\n".join(lines)
