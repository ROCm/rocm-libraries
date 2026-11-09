"""The hybrid's EXISTING knobs as one search space per layer cell (truemono-quick 3a; CPU-importable).

A cell is (kind, variant): ``delta.r4s`` (K=3 verify, 35 of 48 target layers), ``delta.r1`` (K=0 decode),
``qsa.r4s``, ``qsa.r1``, ``delta_ple.*``. The space joins (``CompositeKnobs``):

  moe    ``FusedKnobs`` of the fused rocKE MoE (``tune.catalog.fused_moe``), REBASED on the hybrid's served set
         (``stage_moe_fused.BEST_KNOBS``): level 0 of every categorical knob is the served value, so a config's
         canonical knobs are its changes against the hybrid;
  b3     ``B3Knobs`` (delta kinds only; ``tune.catalog.b3``), rebased on ``stage_b3.TUNED_SPECS_NUMA``;
  sched  ATOM's ``ATOM_Q38_SCHED`` levers of this kind / variant (``tune.catalog.sched``), based on ATOM's
         ``sched.BEST`` group for the cell.

Fixed (documented, not searched): ``numa`` (the hybrid's structural base: numa=1 functions + the numa=1 lever on
every narrow build; a plain base is another family, ``best_plain``), and the tolerance-class knob ``b3.exp``
(``hw`` exp is not bitwise: it needs DESIGN 12.4's T0-T5 ladder, not this bitwise gate). Every other knob and level
of the three owners is in the pool, solo losers included (PLAN decision 13).
"""

from __future__ import annotations

import dataclasses
import re

from kernels.gfx950.qwen38_mono.tune import catalog as C
from kernels.gfx950.qwen38_mono.tune.knobs import CATEGORICAL, ORDINAL, CompositeKnobs, Knob, DataclassKnobs, LeverStringKnobs, flatten_instance

FIXED = {
    "numa": "fixed at the hybrid's structural base (numa=1 functions + the numa=1 lever)",
    "exp": "tolerance class (hw exp): needs the T0-T5 ladder, not the bitwise gate",
    "act": "tolerance class (activation format): needs the T0-T5 ladder, not the bitwise gate",
}

# ATOM sched.BEST (ATOM-truemono/atom/models/qwen4_exp_mono/kernels/sched.py), per cell; numa=1 global
SCHED_BEST = {
    "delta.r1": {"hold.b1": 32, "hold.m4": 64, "hold.m3": 32},
    "delta.r4s": {"hold.m3": 128, "head_sleep": 128},
    "qsa.r4s": {"hold.m3": 64, "head_sleep": 128, "hold.m1": 32},
    "qsa.r1": {"hold.q1": 32},
    "delta_ple.r4s": {"hold.a2": 32, "hold.m3": 128},
    "delta_ple.r1": {"hold.b1": 32, "hold.m4": 64, "hold.a2": 32},
}
# ATOM's own lever defaults (a lever at its default is left out of the string)
ATOM_DEFAULT = {"head_sleep": 64}


def parse_cell(cell: str):
    kind, variant = cell.split(".")
    m = re.fullmatch(r"r(\d+)(s?)(?:g(\d+))?", variant)
    if not m or kind not in ("delta", "delta_ple", "qsa") or (m.group(2) and m.group(3)):
        raise ValueError(f"cell {cell!r}: <delta|delta_ple|qsa>.r<rows>[s|g<S>]")
    return kind, variant, int(m.group(1)), bool(m.group(2))


def group_of(variant: str) -> int:
    """S of a grouped verify variant ``r<rows>g<S>`` (0: not grouped)."""
    m = re.fullmatch(r"r\d+g(\d+)", variant)
    return int(m.group(1)) if m else 0


def grouped_sched_adapter(kind: str, variant: str, validate=None) -> LeverStringKnobs:
    """The schedule levers a GROUPED build takes (ATOM sched.resolve: wide builds drop numa / place / dyn, grouped
    builds drop the spread; ``b3pf`` is a narrow-delta lever): holds, head sleep, pause, pipe, prio. Base = the hybrid's
    (``best`` gives a grouped build numa only, dropped above 4 rows): ATOM's defaults."""
    cat = C.sched(kind, variant)
    knobs = [k for k in cat.knobs if k.path != "numa" and not k.path.startswith(("dyn.", "place.", "spread")) and k.path != "b3pf"]
    base = {k.path: ATOM_DEFAULT.get(k.path, 0) for k in knobs}
    knobs = [dataclasses.replace(k, values=(base[k.path],) + tuple(v for v in k.values if v != base[k.path]))
             if k.kind != ORDINAL else k for k in knobs]
    return LeverStringKnobs("sched", knobs, style="kv", base=base, validate=validate, variant=f"{kind}.{variant}",
                            excluded={**cat.excluded, "numa": "dropped by ATOM above 4 rows", "dyn.*": "narrow only",
                                      "b3pf": "narrow only", "spread": "ungrouped only"})  # fmt: skip


def rebase(ad: DataclassKnobs, base_obj, drop=("numa", "exp")) -> DataclassKnobs:
    """``ad`` over ``base_obj``: categorical / bool levels reordered so level 0 is the base's value, ordinal levels
    extended by it; knobs in ``drop`` left out (``FIXED``)."""
    vals = flatten_instance(base_obj, [k.path for k in ad.knobs])
    knobs = []
    for k in ad.knobs:
        if k.path in drop:
            continue
        v0 = vals[k.path]
        if k.kind == ORDINAL:
            values = k.values if v0 in k.values else tuple(sorted(set(k.values) | {v0}))
        else:
            values = (v0,) + tuple(v for v in k.values if v != v0)
        knobs.append(dataclasses.replace(k, values=values))
    excluded = {**ad.excluded, **{d: FIXED[d] for d in drop if any(k.path == d for k in ad.knobs)}}
    return DataclassKnobs(ad.name, base_obj, knobs, variant=ad.variant, rules=ad.rules, normalize=ad._normalize,
                          excluded=excluded)  # fmt: skip


def moe_adapter(rows: int, seq: bool) -> DataclassKnobs:
    from kernels.gfx950.qwen38_mono import stage_moe_fused as SMF

    return rebase(C.fused_moe(rows, seq), SMF.best_knobs(rows, "best"))


def b3_adapter(variant: str, rows: int, seq: bool) -> DataclassKnobs:
    from kernels.gfx950.qwen38_mono import stage_b3 as SB

    return rebase(C.b3(variant, numa=True), SB.tuned_knobs(SB.B3Shape(rows, seq, 0), numa=True))


def sched_adapter(kind: str, variant: str, validate=None) -> LeverStringKnobs:
    cat = C.sched(kind, variant)
    best = SCHED_BEST.get(f"{kind}.{variant}", {})
    knobs = []
    for k in cat.knobs:
        if k.path == "numa":
            continue
        v0 = best.get(k.path, ATOM_DEFAULT.get(k.path, 0))
        if k.kind == ORDINAL:
            values = k.values if v0 in k.values else tuple(sorted(set(k.values) | {v0}))
        else:
            values = (v0,) + tuple(v for v in k.values if v != v0)
        knobs.append(dataclasses.replace(k, values=values))
    if re.fullmatch(r"r\d+s?", variant):
        # the HC down-projection spread (W2 probe P3; ATOM-truemono sched.py: narrow ungrouped builds only): 0 off,
        # 1 / 2 / 3 / 4 the spread forms, 10 = 1 with the quarter pack (the test mutants 5-9, 11 are not levers)
        for st in ("a1", "m1"):
            knobs.append(Knob(f"spread.{st}", (0, 1, 2, 3, 4, 10), CATEGORICAL))
    base = {k.path: best.get(k.path, ATOM_DEFAULT.get(k.path, 0)) for k in knobs}
    return LeverStringKnobs("sched", knobs, style="kv", base=base, validate=validate, variant=f"{kind}.{variant}",
                            excluded={**cat.excluded, "numa": FIXED["numa"]})  # fmt: skip


def sched_string(kind: str, variant: str, levers: dict, extra: dict | None = None) -> str:
    """The full ``ATOM_Q38_SCHED`` value of a cell's levers: ``numa=1`` globally, then the cell's group with every
    lever off ATOM's default; ``extra`` levers (test mutants, the recompiled control's ``pause=0``) are appended
    as given."""
    items = [f"{k}={v}" for k, v in sorted(levers.items()) if v != ATOM_DEFAULT.get(k, 0)]
    items += [f"{k}={v}" for k, v in (extra or {}).items()]
    body = ",".join(items)
    return "numa=1" + (f";{kind}.{variant}:{body}" if body else "")


def space_for(cell: str, validate=None) -> CompositeKnobs:
    kind, variant, rows, seq = parse_cell(cell)
    if group_of(variant):
        # the grouped / wide builds keep the FlyDSL MoE and B3 (the hybrid has no rocKE functions there): schedule only
        return CompositeKnobs("hybrid_layer", [("sched", grouped_sched_adapter(kind, variant, validate))], variant=cell)
    parts = [("moe", moe_adapter(rows, seq))]
    if kind != "qsa":
        parts.append(("b3", b3_adapter(variant, rows, seq)))
    parts.append(("sched", sched_adapter(kind, variant, validate)))
    return CompositeKnobs("hybrid_layer", parts, variant=cell)


def full_config(ad: CompositeKnobs, config) -> dict:
    """``{part: full knob dict}``: the base values with ``config``'s changes applied."""
    canon = ad.canonical_knobs(config)
    out = {}
    for part, sub in ad.parts:
        vals = dict(sub.base_values())
        for k, v in canon.items():
            p, _, rest = k.partition(".")
            if p == part:
                vals[rest] = v
        out[part] = vals
    return out
