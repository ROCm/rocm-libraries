# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Execution contexts the pipeline benches GEMMs under.

A context is what the engine's `ExecutionContext` describes: the CUs the
kernels may use (`cu_budget`, 0 = every CU) and the tile schedule. The bench
reproduces one with

  * `--sm_count_target <cu_budget>`: hipBLASLt sizes kernel selection and
    persistent grids for that many CUs (process wide);
  * `--hybrid_assignment_policy DynamicWorkQueue|Auto` for the dynamic and
    auto schedules;
  * with `cotenant`, `hipblaslt-cotenant --cus <device CUs - cu_budget>`
    around the bench: a resident busy kernel holds the other CUs, so the
    kernels really run on `cu_budget` CUs.

The cotenant pins its kernel to the device named by `--device` through
HIP_VISIBLE_DEVICES, so the bench it starts sees that device as device 0;
`bench_command` resolves the bench's device index through the
HIP_VISIBLE_DEVICES the stage runs with, so both stay on the stage's devices.

The exclusive context (every CU, default schedule, no cotenant) is the
round's own stage03 bench; the contexts here are the others. A
`holdout` context is benched and enriched separately and never trained on:
stage04b reports the model on it (interpolation between trained budgets).

A config entry gives its budget as CUs (`cu_budget`) or as a share of the
device's CUs (`cu_fraction`), which `resolve_fractions` turns into CUs
when a run starts, from the CU count of the bench device. Inside a compute
partition that is the partition's: each partition is a device of its own,
with its own KFD node and HIP device, so the shares are sized for it. The
budget is the multiple of the budget granule (`budget_granule`) nearest the
share, at least one granule, and for a trained entry not below
`train.min_cu_fraction` where the device allows. The granule is the
budgets' own, apart from the rounding of the SM target in hipBLASLt's
launch rule (`hipblaslt_launch.SM_TARGET_GRANULE`), and a multiple of it, so
that hipBLASLt takes every budget as its SM target as it is and the kernels
size their grids for exactly the CUs the cotenant leaves them. An entry is
left out (`left_out` names them) when it resolves to the budget and
schedule of another (one with a budget of its own, or an earlier one:
trained entries come before held-out ones), or when the device has no
budget to give (no multiple of the granule below its CU count), so a device
too small for every share trains the exclusive context alone. An entry with
`enabled: false` is off (`off_entries`): no stage benches or trains it and
a run's snapshot leaves it out, so a config can carry a share that costs
too much to bench by default. The run's snapshot keeps each fraction beside
the budget it became.
"""
from __future__ import annotations

import copy
import json
import math
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set
from typing import Tuple

from .hipblaslt_launch import SM_TARGET_GRANULE

# The CU budgets a cu_fraction resolves to are whole multiples of this many
# CUs (WGPs on gfx1250), and at least one, so the smallest budget a run
# trains is one granule however small its share: one XCD's worth of CUs on
# gfx950. It must stay a multiple of the launch rule's rounding of the SM
# target (SM_TARGET_GRANULE), so that hipBLASLt sizes the kernels for exactly
# the CUs the cotenant leaves them; `contexts.granule` sets another for a
# config.
CU_BUDGET_GRANULE = 32
# The granule of the budgets of a run or audit directory pinned before
# `contexts.granule` existed, whose budgets were whole multiples of 32 CUs:
# what a pin without the key means, whatever CU_BUDGET_GRANULE becomes
# (run_orchestrator.py PINNED_UNLESS_NEUTRAL).
PREDATING_GRANULE = 32

SCHEDULES: Tuple[str, ...] = ("default", "dynamic", "auto")
HYBRID_POLICY = {"dynamic": "DynamicWorkQueue", "auto": "Auto"}
SPEC_FILE = "context.json"
WRAPPER_NAME = "hipblaslt-bench"
SECTIONS: Tuple[str, ...] = ("list", "holdout")
ShapeKey = Tuple[int, int, int, int]


@dataclass(frozen=True)
class BenchContext:
    cu_budget: int = 0
    schedule: str = "default"
    cotenant: bool = False
    fraction: float = 1.0
    holdout: bool = False
    name: Optional[str] = None

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        base = f"cu{self.cu_budget}" if self.cu_budget else "allcu"
        if self.cotenant:
            base += "_cot"
        if self.schedule != "default":
            base += f"_{self.schedule}"
        return base

    def is_exclusive(self) -> bool:
        return self.cu_budget == 0 and self.schedule == "default" and not self.cotenant

    def cotenant_cus(self, n_cu: int) -> int:
        return max(0, int(n_cu) - self.cu_budget) if self.cotenant else 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["name"] = self.label
        return d

    @classmethod
    def from_mapping(
        cls, m: Mapping[str, Any], holdout: bool = False
    ) -> "BenchContext":
        if unresolved(m):
            raise ValueError(
                f"context entry {dict(m)} gives a cu_fraction: its CU budget "
                f"needs the device's CU count (resolve_fractions)"
            )
        return cls(
            cu_budget=int(m.get("cu_budget", 0) or 0),
            schedule=str(m.get("schedule", "default") or "default"),
            cotenant=bool(m.get("cotenant", False)),
            fraction=float(m.get("fraction", 1.0)),
            holdout=bool(m.get("holdout", holdout)),
            name=m.get("name") or None,
        )


def unresolved(m: Mapping[str, Any]) -> bool:
    """True for an entry that gives a cu_fraction and no cu_budget."""
    return m.get("cu_fraction") is not None and m.get("cu_budget") is None


def enabled(m: Mapping[str, Any]) -> bool:
    """False for an entry the config turns off (`enabled: false`)."""
    return m.get("enabled", True) is not False


def has_unresolved(cfg: Mapping[str, Any]) -> bool:
    """True when an entry the config turns on gives a cu_fraction and no
    cu_budget."""
    return any(unresolved(m) for _, _, m in _entries(cfg))


def _all_entries(cfg: Mapping[str, Any]) -> List[Tuple[str, int, Mapping[str, Any]]]:
    """(section key, index, entry) of every mapping in `contexts.list` and
    `contexts.holdout`."""
    sec = cfg.get("contexts") or {}
    return [
        (key, i, m)
        for key in SECTIONS
        for i, m in enumerate(sec.get(key) or [])
        if isinstance(m, Mapping)
    ]


def _entries(cfg: Mapping[str, Any]) -> List[Tuple[str, int, Mapping[str, Any]]]:
    """`_all_entries` the config turns on, with their indices in the
    config."""
    return [(key, i, m) for key, i, m in _all_entries(cfg) if enabled(m)]


def off_entries(cfg: Mapping[str, Any]) -> List[Tuple[str, int, Mapping[str, Any]]]:
    """(section key, index, entry) of the entries the config turns off."""
    return [(key, i, m) for key, i, m in _all_entries(cfg) if not enabled(m)]


def describe(m: Mapping[str, Any]) -> str:
    """An entry's budget as the config gives it: `cu_fraction <share>` or
    `cu_budget <CUs>`, with a schedule other than the default."""
    if m.get("cu_fraction") is not None:
        what = f"cu_fraction {float(m['cu_fraction']):g}"
    else:
        what = f"cu_budget {int(m.get('cu_budget') or 0)}"
    return what if _schedule(m) == "default" else f"{what} / schedule {_schedule(m)}"


def trained_entries(cfg: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """The entries of `contexts.list` the config turns on other than the
    exclusive one, as the config gives them (an entry with a cu_fraction is
    never exclusive)."""
    return [
        m
        for key, _, m in _entries(cfg)
        if key == "list"
        and (unresolved(m) or not BenchContext.from_mapping(m).is_exclusive())
    ]


def holdout_entries(cfg: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """The entries of `contexts.holdout` the config turns on."""
    return [m for key, _, m in _entries(cfg) if key == "holdout"]


def reduces_cus(m: Mapping[str, Any]) -> bool:
    """True for an entry that leaves the kernels fewer than every CU."""
    return m.get("cu_fraction") is not None or int(m.get("cu_budget") or 0) > 0


def budget_granule(cfg: Mapping[str, Any]) -> int:
    """The granule of the CU budgets the cu_fraction entries of `cfg`
    resolve to: `contexts.granule`, else CU_BUDGET_GRANULE."""
    sec = cfg.get("contexts")
    granule = sec.get("granule") if isinstance(sec, Mapping) else None
    return CU_BUDGET_GRANULE if granule is None else int(granule)


def granule_errors(cfg: Mapping[str, Any], where: str = "contexts") -> List[str]:
    """The budget granule of `cfg` (`budget_granule`) as an error unless it
    is a whole multiple of SM_TARGET_GRANULE, so that hipBLASLt takes every
    budget as its SM target as it is."""
    granule = budget_granule(cfg)
    if granule >= SM_TARGET_GRANULE and granule % SM_TARGET_GRANULE == 0:
        return []
    return [
        f"{where}.granule {granule}: must be a multiple of {SM_TARGET_GRANULE}, "
        f"the SM-target granule of hipBLASLt's launch rule, so that hipBLASLt "
        f"sizes the kernels for exactly the CUs of each budget"
    ]


def fraction_budget(
    fraction: float,
    n_cu: int,
    floor: Optional[float] = None,
    granule: Optional[int] = None,
) -> int:
    """The CU budget `fraction` of a device with `n_cu` CUs stands for: the
    multiple of `granule` (default CU_BUDGET_GRANULE) nearest fraction *
    n_cu, at least one granule and below n_cu, and raised to at least
    `floor` * n_cu where a multiple below n_cu allows. ValueError when no
    multiple of the granule lies below n_cu."""
    granule = CU_BUDGET_GRANULE if granule is None else int(granule)
    if granule < 1:
        raise ValueError(f"a CU budget granule must be positive, got {granule}")
    top = (int(n_cu) - 1) // granule * granule
    if top < granule:
        raise ValueError(
            f"a device of {n_cu} CUs has no CU budget below its CU count (a "
            f"budget is a multiple of {granule} CUs)"
        )
    budget = int(math.floor(float(fraction) * n_cu / granule + 0.5)) * granule
    budget = min(max(budget, granule), top)
    while floor is not None and budget < float(floor) * n_cu and budget < top:
        budget += granule
    return budget


def _schedule(m: Mapping[str, Any]) -> str:
    return str(m.get("schedule") or "default")


def _resolution(
    cfg: Mapping[str, Any], n_cu: int
) -> Tuple[Dict[Tuple[str, int], int], Dict[Tuple[str, int], Optional[str]]]:
    """({(section key, index): budget} of the entries the config turns on
    that give a cu_fraction and no cu_budget, on a device of `n_cu` CUs, the
    floor of a trained entry being `train.min_cu_fraction`; {(section key,
    index): the entry it repeats, or None} of those left out: whose budget
    and schedule are those of an entry with a cu_budget of its own or of an
    earlier one, or, None, that the device has no budget for). Budgets are
    multiples of `budget_granule`; ValueError when that is not positive."""
    floor = (cfg.get("train") or {}).get("min_cu_fraction")
    granule = budget_granule(cfg)
    if granule < 1:
        raise ValueError(f"contexts.granule {granule}: must be positive")
    entries = _entries(cfg)
    taken: Dict[Tuple[int, str], str] = {}
    for key, i, m in entries:
        if not unresolved(m) and int(m.get("cu_budget") or 0) > 0:
            ident = (int(m["cu_budget"]), _schedule(m))
            taken.setdefault(ident, f"contexts.{key}[{i}]")
    budgets: Dict[Tuple[str, int], int] = {}
    left: Dict[Tuple[str, int], Optional[str]] = {}
    for key, i, m in entries:
        if not unresolved(m):
            continue
        try:
            budget = fraction_budget(
                float(m["cu_fraction"]),
                n_cu,
                floor if key == "list" else None,
                granule,
            )
        except ValueError:
            left[(key, i)] = None
            continue
        budgets[(key, i)] = budget
        ident = (budget, _schedule(m))
        if ident in taken:
            left[(key, i)] = taken[ident]
        else:
            taken[ident] = f"contexts.{key}[{i}]"
    return budgets, left


def resolve_fractions(
    cfg: Mapping[str, Any], n_cu: Optional[int]
) -> Tuple[Dict[str, Any], List[str]]:
    """(`cfg` with a cu_budget in each entry that gives a cu_fraction and no
    cu_budget, for a device of `n_cu` CUs, without the entries `_resolution`
    leaves out and those the config turns off (`off_entries`); one line per
    entry resolved, left out or off, in config order). `n_cu` may be None
    when no entry the config turns on gives only a cu_fraction (ValueError
    otherwise)."""
    out = copy.deepcopy(dict(cfg))
    if n_cu is None:
        if has_unresolved(out):
            raise ValueError("a cu_fraction needs the device's CU count")
        budgets: Dict[Tuple[str, int], int] = {}
        left: Dict[Tuple[str, int], Optional[str]] = {}
    else:
        budgets, left = _resolution(out, n_cu)
    said: Dict[Tuple[str, int], str] = {}
    for (key, i), budget in budgets.items():
        m = out["contexts"][key][i]
        what = f"contexts.{key}[{i}]: cu_fraction {float(m['cu_fraction']):g} is"
        if left.get((key, i)):
            said[(key, i)] = (
                f"{what} {budget} of {n_cu} CUs, the budget of "
                f"{left[(key, i)]}; left out"
            )
        else:
            m["cu_budget"] = budget
            said[(key, i)] = f"{what} {budget} of {n_cu} CUs"
    for key, i in left:
        if (key, i) not in budgets:
            m = out["contexts"][key][i]
            said[(key, i)] = (
                f"contexts.{key}[{i}]: cu_fraction {float(m['cu_fraction']):g} "
                f"has no budget on a device of {n_cu} CUs (a budget is a "
                f"multiple of {budget_granule(out)} CUs below its CU count); "
                f"left out"
            )
    off = off_entries(out)
    for key, i, m in off:
        said[(key, i)] = f"contexts.{key}[{i}]: {describe(m)} is off (enabled: false)"
    gone = set(left) | {(key, i) for key, i, _ in off}
    for key in SECTIONS:
        drop = {i for k, i in gone if k == key}
        if drop:
            out["contexts"][key] = [
                m for j, m in enumerate(out["contexts"][key]) if j not in drop
            ]
    order = sorted(said, key=lambda ki: (SECTIONS.index(ki[0]), ki[1]))
    return out, [said[ki] for ki in order]


def left_out(
    source: Mapping[str, Any], resolved: Mapping[str, Any]
) -> List[Tuple[str, int, Mapping[str, Any]]]:
    """(section key, index, entry) of the entries of `source` that give a
    cu_fraction and no cu_budget and that `resolved`, the config
    `resolve_fractions` made of it, left out."""
    kept = {
        (key, float(m["cu_fraction"]), _schedule(m))
        for key, _, m in _entries(resolved)
        if m.get("cu_fraction") is not None
    }
    return [
        (key, i, m)
        for key, i, m in _entries(source)
        if unresolved(m) and (key, float(m["cu_fraction"]), _schedule(m)) not in kept
    ]


def contexts_from_config(
    cfg: Mapping[str, Any], n_cu: Optional[int] = None
) -> List[BenchContext]:
    """The non-exclusive contexts of `contexts.list` and every
    `contexts.holdout` entry the config turns on, in config order; entries
    that give a cu_fraction are resolved for a device of `n_cu` CUs
    (ValueError without it)."""
    if n_cu is not None:
        cfg = resolve_fractions(cfg, n_cu)[0]
    sec = cfg.get("contexts") or {}
    out = [BenchContext.from_mapping(m) for m in sec.get("list") or [] if enabled(m)]
    out = [c for c in out if not c.is_exclusive()]
    out += [
        BenchContext.from_mapping(m, holdout=True)
        for m in sec.get("holdout") or []
        if enabled(m)
    ]
    return out


def context_errors(
    cfg: Mapping[str, Any], n_cu: Optional[int] = None, where: str = "contexts"
) -> List[str]:
    """Rules the schema cannot state. `n_cu` (when known) bounds budgets and
    resolves fractions; without it an entry that gives only a cu_fraction
    is checked for what needs no budget. Of the entries the config turns
    off only the exclusive one is an error: it is the round's own bench. A
    budget granule that is no multiple of the SM-target granule is the only
    error reported, since every budget follows from it."""
    bad = granule_errors(cfg, where)
    if bad:
        return bad
    granule = budget_granule(cfg)
    budgets: Dict[Tuple[str, int], int] = {}
    repeats: Dict[Tuple[str, int], Optional[str]] = {}
    if n_cu is not None:
        budgets, repeats = _resolution(cfg, n_cu)
    floor = (cfg.get("train") or {}).get("min_cu_fraction")
    errors: List[str] = []
    for key, i, m in off_entries(cfg):
        if not unresolved(m) and BenchContext.from_mapping(m).is_exclusive():
            errors.append(
                f"{where}.{key}[{i}]: the exclusive context cannot be turned off "
                f"(it is the round's own bench)"
            )
    seen: Dict[Any, str] = {}
    labels: Dict[str, str] = {}
    for key, i, m in _entries(cfg):
        at = f"{where}.{key}[{i}]"
        f = m.get("cu_fraction")
        if (
            key == "list"
            and f is not None
            and floor is not None
            and float(f) < float(floor)
        ):
            errors.append(
                f"{at}: cu_fraction {f:g} is below train.min_cu_fraction "
                f"{floor:g}, the smallest share the models may declare; lower "
                f"train.min_cu_fraction, leave it unset, or turn the entry off "
                f"(enabled: false)"
            )
        if (key, i) in repeats:
            continue
        if f is not None and m.get("cu_budget") is not None and n_cu is not None:
            try:
                want: Any = fraction_budget(
                    float(f), n_cu, floor if key == "list" else None, granule
                )
            except ValueError:
                want = "the device has none"
            if int(m["cu_budget"]) != want:
                errors.append(
                    f"{at}: cu_budget {m['cu_budget']} is not cu_fraction {f:g} "
                    f"of the device's {n_cu} CUs ({want})"
                )
        if unresolved(m) and (key, i) not in budgets:
            ident = ("fraction", float(f), _schedule(m))
            if ident in seen:
                errors.append(
                    f"{at}: cu_fraction {f:g} / schedule {_schedule(m)} is also "
                    f"{seen[ident]}"
                )
            seen[ident] = at
            if m.get("name"):
                if m["name"] in labels:
                    errors.append(
                        f"{at}: name {m['name']!r} is also {labels[m['name']]}"
                    )
                labels[str(m["name"])] = at
            continue
        if (key, i) in budgets:
            m = dict(m, cu_budget=budgets[(key, i)])
        c = BenchContext.from_mapping(m, holdout=key == "holdout")
        if c.cotenant and c.cu_budget <= 0:
            errors.append(f"{at}: cotenant needs a cu_budget > 0")
        if c.is_exclusive():
            if key == "holdout":
                errors.append(f"{at}: the exclusive context cannot be held out")
            elif c.fraction != 1.0:
                errors.append(
                    f"{at}: the exclusive context benches every shape "
                    f"(fraction must be 1)"
                )
            continue
        if n_cu is not None and c.cu_budget >= n_cu:
            errors.append(
                f"{at}: cu_budget {c.cu_budget} must be below the device's "
                f"{n_cu} CUs (0 means every CU)"
            )
        ident = (c.cu_budget, c.schedule)
        if ident in seen:
            errors.append(
                f"{at}: cu_budget {c.cu_budget} / schedule {c.schedule} is "
                f"also {seen[ident]}; rows carry only budget and schedule"
            )
        seen[ident] = at
        if c.label in labels:
            errors.append(f"{at}: name {c.label!r} is also {labels[c.label]}")
        labels[c.label] = at
    return errors


def sm_target(cu_budget: int, n_cu: int) -> int:
    """The CUs hipBLASLt sizes kernels for under `cu_budget` on a device of
    `n_cu` CUs: the budget rounded down to a multiple of SM_TARGET_GRANULE,
    and the whole device when that is 0 or not below the device's CUs
    (lib/hipblaslt_launch.py)."""
    target = int(cu_budget) // SM_TARGET_GRANULE * SM_TARGET_GRANULE
    return target if 0 < target < int(n_cu) else int(n_cu)


def collapsed_holdouts(
    contexts: Sequence[BenchContext], n_cu: int
) -> List[Tuple[BenchContext, str]]:
    """(held-out context, the trained one it collapses onto) for every
    held-out context of `contexts` whose kernels hipBLASLt sizes as a
    trained context's (`sm_target`, same schedule; the exclusive context,
    every CU and the default schedule, is a trained one): it holds out no
    launch the run does not train on."""
    trained: Dict[Tuple[int, str], str] = {(int(n_cu), "default"): "exclusive"}
    for c in contexts:
        if not c.holdout:
            trained.setdefault((sm_target(c.cu_budget, n_cu), c.schedule), c.label)
    out: List[Tuple[BenchContext, str]] = []
    for c in contexts:
        key = (sm_target(c.cu_budget, n_cu), c.schedule)
        if c.holdout and key in trained:
            out.append((c, trained[key]))
    return out


def trained_fractions(contexts: Sequence[BenchContext], n_cu: int) -> List[float]:
    """Usable-CU fractions the trained (non-holdout) contexts cover, the
    exclusive 1.0 included."""
    out = {1.0}
    for c in contexts:
        if not c.holdout and 0 < c.cu_budget < n_cu:
            out.add(c.cu_budget / float(n_cu))
    return sorted(out)


# ── shape selection ──────────────────────────────────────────────────────────


def select_shapes(
    shapes: Sequence[ShapeKey],
    ctx: BenchContext,
    seed: int,
    forced: Iterable[ShapeKey] = (),
) -> List[ShapeKey]:
    """The shapes `ctx` benches: all of them at fraction 1, else
    ceil(fraction * n) per cell (at least one): the shapes of `forced` it
    holds (`targeted_for`), then as many more as that leaves, drawn with
    `seed`; in the order of `shapes`. Without `forced` a cell's draw is the
    one it always was."""
    from .planner import shape_cell, stable_rng

    distinct = list(dict.fromkeys(shapes))
    if ctx.fraction >= 1.0:
        return distinct
    must = set(forced) & set(distinct)
    by_cell: Dict[str, List[ShapeKey]] = {}
    for s in distinct:
        by_cell.setdefault(shape_cell(s), []).append(s)
    chosen = set(must)
    for cell, members in sorted(by_cell.items()):
        k = min(len(members), max(1, math.ceil(ctx.fraction * len(members))))
        rng = stable_rng(seed, "context", ctx.label, cell)
        pool = sorted(s for s in members if s not in must)
        rest = max(0, k - (len(members) - len(pool)))
        chosen.update(rng.sample(pool, min(rest, len(pool))))
    return [s for s in distinct if s in chosen]


def targeted_for(path: Optional[Path], ctx: BenchContext) -> Set[ShapeKey]:
    """The shapes stage01 drew near tail GEMMs that ran in the context
    `ctx` (its targeted_shapes.json at `path`), which `select_shapes`
    benches in that context beside its own draw: none for the exclusive
    context, which benches every shape, or without the file."""
    if path is None or ctx.is_exclusive() or not Path(path).is_file():
        return set()
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return set()
    out: Set[ShapeKey] = set()
    for entries in (doc.get("leaves") or {}).values():
        for e in entries or []:
            t = e.get("tail") or {}
            if int(t.get("cu_budget") or 0) != ctx.cu_budget:
                continue
            if str(t.get("schedule") or "default") != ctx.schedule:
                continue
            out.update(
                (int(s[0]), int(s[1]), int(s[2]), int(s[3]))
                for s in e.get("shapes") or []
            )
    return out


# ── bench wrapper ────────────────────────────────────────────────────────────


def bench_spec(
    ctx: BenchContext,
    n_cu: int,
    bench: str,
    cotenant: Optional[str],
    log_dir: str,
    library_path: Optional[str] = None,
    path_append: Sequence[str] = (),
) -> Dict[str, Any]:
    """What the wrapper of `ctx` runs (`context.json`)."""
    cus = ctx.cotenant_cus(n_cu)
    if cus and not cotenant:
        raise ValueError(f"context {ctx.label} needs hipblaslt-cotenant")
    if cus >= n_cu:
        raise ValueError(f"context {ctx.label} would leave the bench no CU")
    return {
        "context": ctx.to_dict(),
        "n_cu": int(n_cu),
        "bench": str(bench),
        "cotenant": str(cotenant) if cus else None,
        "cotenant_cus": cus,
        "sm_count_target": int(ctx.cu_budget),
        "hybrid_assignment_policy": HYBRID_POLICY.get(ctx.schedule),
        "log_dir": str(log_dir),
        "library_path": library_path,
        "path_append": [str(p) for p in path_append],
    }


def _device_of(argv: Sequence[str]) -> Tuple[Optional[str], List[str]]:
    """(--device value, argv without it)."""
    out: List[str] = []
    device = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--device" and i + 1 < len(argv):
            device = argv[i + 1]
            i += 2
            continue
        if a.startswith("--device="):
            device = a.split("=", 1)[1]
            i += 1
            continue
        out.append(a)
        i += 1
    return device, out


def cotenant_device(device: str, environ: Mapping[str, str]) -> str:
    """hipblaslt-cotenant's `--device` for the bench's `--device device` in
    a process with `environ`. The cotenant replaces HIP_VISIBLE_DEVICES by
    its `--device`, so an index into a HIP_VISIBLE_DEVICES (or
    CUDA_VISIBLE_DEVICES) list the process already has is resolved through
    that list; ValueError when it cannot be."""
    index = int(device)
    for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        raw = str(environ.get(var, "")).strip()
        if not raw:
            continue
        entries = [e.strip() for e in raw.split(",")]
        if not 0 <= index < len(entries) or not entries[index].isdigit():
            raise ValueError(
                f"--device {device} does not name a device of {var}={raw}; "
                f"the cotenant would run elsewhere"
            )
        return entries[index]
    return str(index)


def bench_command(
    spec: Mapping[str, Any],
    argv: Sequence[str],
    stamp: str,
    environ: Optional[Mapping[str, str]] = None,
) -> Tuple[List[str], Dict[str, str]]:
    """(command, environment changes) of one bench call `argv` (as stage03
    passes it: `--device D --yaml F`) under `spec`, from a process with
    `environ` (default: this one's)."""
    environ = os.environ if environ is None else environ
    device, rest = _device_of(list(argv))
    bench_args = list(rest)
    if int(spec.get("sm_count_target") or 0) > 0:
        bench_args += ["--sm_count_target", str(int(spec["sm_count_target"]))]
    if spec.get("hybrid_assignment_policy"):
        bench_args += [
            "--hybrid_assignment_policy",
            str(spec["hybrid_assignment_policy"]),
        ]
    env: Dict[str, str] = {}
    if spec.get("library_path"):
        env["HIPBLASLT_TENSILE_LIBPATH"] = str(spec["library_path"])
    if spec.get("path_append"):
        env["PATH"] = os.pathsep.join(
            [environ.get("PATH", "")] + list(spec["path_append"])
        ).lstrip(os.pathsep)
    cus = int(spec.get("cotenant_cus") or 0)
    if cus <= 0:
        dev = [] if device is None else ["--device", device]
        return [str(spec["bench"]), *dev, *bench_args], env
    log = Path(spec["log_dir"]) / f"cotenant_{cus}_{stamp}.log"
    cmd = [str(spec["cotenant"]), "--cus", str(cus)]
    if device is not None:
        cmd += ["--device", cotenant_device(device, environ)]
    cmd += ["--log", str(log), "--", str(spec["bench"]), "--device", "0", *bench_args]
    return cmd, env


def write_wrapper(
    directory: Path, spec: Mapping[str, Any], python: str, script: Path
) -> Path:
    """`<directory>/hipblaslt-bench`: runs `script` (lib/context_bench.py)
    with `<directory>/context.json`, for stage03's --bench-binary."""
    directory.mkdir(parents=True, exist_ok=True)
    Path(spec["log_dir"]).mkdir(parents=True, exist_ok=True)
    spec_path = directory / SPEC_FILE
    spec_path.write_text(json.dumps(dict(spec), indent=2) + "\n")
    wrapper = directory / WRAPPER_NAME
    wrapper.write_text("#!/bin/sh\n" f'exec "{python}" "{script}" "{spec_path}" "$@"\n')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return wrapper


def wrapped_bench(binary: Path) -> Path:
    """The hipblaslt-bench a context's wrapper (`write_wrapper`) runs, or
    `binary` itself when it is not one."""
    binary = Path(binary)
    spec = binary.parent / SPEC_FILE
    if binary.name != WRAPPER_NAME or not spec.is_file():
        return binary
    try:
        bench = json.loads(spec.read_text()).get("bench")
    except (OSError, ValueError, AttributeError):
        return binary
    return Path(bench) if bench else binary
