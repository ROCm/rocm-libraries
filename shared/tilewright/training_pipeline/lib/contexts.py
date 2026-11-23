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
budgets' own: hipBLASLt takes every budget as its SM target as it is
(`hipblaslt_launch.SM_TARGET_GRANULE`), so the kernels size their grids for
exactly the CUs the cotenant leaves them, whatever the granule. An entry is
left out (`left_out` names them) when it resolves to the budget and
schedule of another (one with a budget of its own, or an earlier one:
trained entries come before held-out ones), or when the device has no
budget to give (no multiple of the granule below its CU count), so a device
too small for every share trains the exclusive context alone. An entry with
`enabled: false` is off (`off_entries`): no stage benches or trains it and
a run's snapshot leaves it out, so a config can carry a share that costs
too much to bench by default. The run's snapshot keeps each fraction beside
the budget it became.

Under launch rule version 3 a context also names the uniform summation
order its calls run with (`uniform_summation_order: on|off`) and the
workspace each launch is given (`workspace_bytes`), by default the round's
own (`round_launch`): hipBLASLt's default order on the arch (off on RDNA,
on elsewhere) unless `contexts.round_uniform_summation_order` sets it, and
the full workspace, `contexts.full_workspace_bytes` (default
FULL_WORKSPACE_BYTES, at or above every kernel's requirement). Every line
of a context carries both (`line_knobs`), with the bench's
`workspace_per_process`: one buffer per bench process, each launch given
the context's workspace whole. A run adds a launch-keyed context of every
CU, the default schedule and the round's order per size of
`contexts.workspaces` (default [0]): it benches only the lines on which some
pool kernel's launch differs from its launch in the full workspace
(`launch_changing_shapes`), since a kernel launched alike times alike.
Rules before version 3 plan neither, and their lines leave both to the
bench as before.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import stat
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence
from typing import Set, Tuple

# The CU budgets a cu_fraction resolves to are whole multiples of this many
# CUs, as the device reports them, and at least one, so the smallest budget
# a run trains is one granule however small its share: one XCD's worth of
# CUs on gfx950. `contexts.granule` sets another for a config.
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
# The largest workspace a launch can be given: hipBLASLt clamps a call's
# workspace to UINT32_MAX before TensileLite sizes its launch. It is the full
# workspace, at or above the requirement of every kernel the library can
# serve, so a launch in it is the launch of an unbounded workspace.
MAX_WORKSPACE_BYTES = (1 << 32) - 1
FULL_WORKSPACE_BYTES = MAX_WORKSPACE_BYTES
# The sizes of the launch-keyed workspace contexts a run adds when its
# config names none (`contexts.workspaces`).
DEFAULT_WORKSPACES: Tuple[int, ...] = (0,)
_MIB = 1 << 20


def workspace_label(workspace: int) -> str:
    """A workspace as a context label spells it: MiB with `m`, or bytes with
    `b` when it is no whole number of MiB."""
    w = int(workspace)
    return f"{w // _MIB}m" if w % _MIB == 0 else f"{w}b"


def summation_order_value(value: Any) -> bool:
    """A config's `uniform_summation_order` as a bool: on or off (YAML reads
    the bare words as booleans), 1 or 0; None means on. ValueError
    otherwise."""
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("on", "off"):
        return value.strip().lower() == "on"
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(f"uniform_summation_order {value!r}: give on or off")


def workspace_value(value: Any) -> int:
    """A config's `workspace_bytes` as bytes, from 0 to MAX_WORKSPACE_BYTES;
    None means FULL_WORKSPACE_BYTES. ValueError otherwise."""
    if value is None:
        return FULL_WORKSPACE_BYTES
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"workspace_bytes {value!r}: give a whole number of bytes")
    if not 0 <= value <= MAX_WORKSPACE_BYTES:
        raise ValueError(
            f"workspace_bytes {value}: must be from 0 to {MAX_WORKSPACE_BYTES} "
            f"(hipBLASLt clamps a workspace to UINT32_MAX)"
        )
    return value


RoundLaunch = Tuple[bool, int]
# The round's own summation order and workspace where no config is at hand
# (`round_launch` gives a run's).
DEFAULT_ROUND_LAUNCH: RoundLaunch = (True, FULL_WORKSPACE_BYTES)


def full_workspace(cfg: Mapping[str, Any]) -> int:
    """The full workspace of a run of `cfg`: `contexts.full_workspace_bytes`,
    else FULL_WORKSPACE_BYTES; ValueError for one that is no workspace or
    none at all (0)."""
    sec = cfg.get("contexts")
    value = sec.get("full_workspace_bytes") if isinstance(sec, Mapping) else None
    if value is None:
        return FULL_WORKSPACE_BYTES
    full = workspace_value(value)
    if full == 0:
        raise ValueError(
            "full_workspace_bytes 0: the full workspace holds partial tiles"
        )
    return full


# What a hipBLASLt handle defaults the uniform summation order to off on: an
# architecture whose name holds one of these (RDNA), as its handle constructor
# tests them.
SUMMATION_ORDER_OFF_ARCHS = ("gfx11", "gfx120")
# The config key that sets the round's own summation order instead.
ROUND_ORDER_KEY = "round_uniform_summation_order"


def production_summation_order(arch: Optional[str]) -> bool:
    """The uniform summation order a hipBLASLt handle defaults to on `arch`:
    off on SUMMATION_ORDER_OFF_ARCHS, on elsewhere."""
    return not any(a in str(arch or "") for a in SUMMATION_ORDER_OFF_ARCHS)


def round_order(cfg: Mapping[str, Any]) -> bool:
    """The uniform summation order of the round's own context of a run of
    `cfg`: `contexts.round_uniform_summation_order`, else the production
    default of its arch (`production_summation_order`); ValueError for a
    value that is none."""
    sec = cfg.get("contexts")
    value = sec.get(ROUND_ORDER_KEY) if isinstance(sec, Mapping) else None
    if value is None:
        return production_summation_order(cfg.get("arch"))
    return summation_order_value(value)


def round_launch(cfg: Mapping[str, Any]) -> RoundLaunch:
    """(summation order, workspace) of the round's own context of a run of
    `cfg` (`round_order`, `full_workspace`), which a context entry that names
    neither has."""
    return round_order(cfg), full_workspace(cfg)


@dataclass(frozen=True)
class BenchContext:
    cu_budget: int = 0
    schedule: str = "default"
    cotenant: bool = False
    fraction: float = 1.0
    holdout: bool = False
    name: Optional[str] = None
    uniform_summation_order: bool = True
    workspace_bytes: int = FULL_WORKSPACE_BYTES
    launch_keyed: bool = False
    # The round's own order and workspace, against which the label and
    # `is_exclusive` read the context's.
    round_order: bool = field(default=True, compare=False, repr=False)
    round_workspace: int = field(
        default=FULL_WORKSPACE_BYTES, compare=False, repr=False
    )

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        base = f"cu{self.cu_budget}" if self.cu_budget else "allcu"
        if self.cotenant:
            base += "_cot"
        if self.schedule != "default":
            base += f"_{self.schedule}"
        if self.uniform_summation_order != self.round_order:
            base += "_usoon" if self.uniform_summation_order else "_usooff"
        if self.workspace_bytes != self.round_workspace:
            base += f"_ws{workspace_label(self.workspace_bytes)}"
        return base

    def own_launch(self) -> bool:
        """Whether the context runs with the round's own order and
        workspace."""
        return (
            self.uniform_summation_order == self.round_order
            and self.workspace_bytes == self.round_workspace
        )

    def is_exclusive(self) -> bool:
        return (
            self.cu_budget == 0
            and self.schedule == "default"
            and not self.cotenant
            and self.own_launch()
        )

    def cotenant_cus(self, n_cu: int) -> int:
        return max(0, int(n_cu) - self.cu_budget) if self.cotenant else 0

    def execution_context(self) -> Any:
        """The engine's ExecutionContext of the calls this context benches
        (lib/features.py)."""
        from .features import ExecutionContext

        return ExecutionContext(
            cu_budget=self.cu_budget,
            schedule=self.schedule,
            uniform_summation_order=self.uniform_summation_order,
            workspace_bytes=self.workspace_bytes,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        del d["round_order"], d["round_workspace"]
        d["name"] = self.label
        return d

    @classmethod
    def from_mapping(
        cls,
        m: Mapping[str, Any],
        holdout: bool = False,
        launch: RoundLaunch = DEFAULT_ROUND_LAUNCH,
    ) -> "BenchContext":
        """The context of config entry `m`, in a run whose round's own order
        and workspace are `launch` (`round_launch`)."""
        if unresolved(m):
            raise ValueError(
                f"context entry {dict(m)} gives a cu_fraction: its CU budget "
                f"needs the device's CU count (resolve_fractions)"
            )
        order, workspace = _launch_of(m, launch)
        return cls(
            cu_budget=int(m.get("cu_budget", 0) or 0),
            schedule=str(m.get("schedule", "default") or "default"),
            cotenant=bool(m.get("cotenant", False)),
            fraction=float(m.get("fraction", 1.0)),
            holdout=bool(m.get("holdout", holdout)),
            name=m.get("name") or None,
            uniform_summation_order=order,
            workspace_bytes=workspace,
            launch_keyed=bool(m.get("launch_keyed", False)),
            round_order=launch[0],
            round_workspace=launch[1],
        )


def round_context(cfg: Mapping[str, Any]) -> BenchContext:
    """The round's own context of a run of `cfg`: every CU, the default
    schedule, and the round's order and workspace (`round_launch`)."""
    order, workspace = round_launch(cfg)
    return BenchContext(
        uniform_summation_order=order,
        workspace_bytes=workspace,
        round_order=order,
        round_workspace=workspace,
    )


def _launch_of(m: Mapping[str, Any], launch: RoundLaunch) -> RoundLaunch:
    """(summation order, workspace) of entry `m`, `launch`'s for what it
    leaves out; ValueError for a value that is none."""
    order = m.get("uniform_summation_order")
    workspace = m.get("workspace_bytes")
    return (
        launch[0] if order is None else summation_order_value(order),
        launch[1] if workspace is None else workspace_value(workspace),
    )


def line_knobs(ctx: BenchContext, version: int) -> Dict[str, Any]:
    """The knobs a bench line of `ctx` carries under launch rule `version`:
    from version 3 on, the context's uniform summation order and workspace,
    which the bench gives every launch whole from one buffer per process
    (`workspace_per_process`); before, none, so that the line leaves both
    to the bench."""
    if int(version) < 3:
        return {}
    return {
        "uniform_summation_order": int(ctx.uniform_summation_order),
        "user_allocated_workspace": int(ctx.workspace_bytes),
        "workspace_per_process": "true",
    }


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


def describe(m: Mapping[str, Any], launch: RoundLaunch = DEFAULT_ROUND_LAUNCH) -> str:
    """An entry's budget as the config gives it: `cu_fraction <share>` or
    `cu_budget <CUs>`, with a schedule other than the default and a
    summation order and workspace other than the round's (`launch`)."""
    if m.get("cu_fraction") is not None:
        what = f"cu_fraction {float(m['cu_fraction']):g}"
    else:
        what = f"cu_budget {int(m.get('cu_budget') or 0)}"
    schedule, uso, workspace = _variant(m, launch)
    if schedule != "default":
        what += f" / schedule {schedule}"
    if uso != launch[0]:
        what += f" / uniform_summation_order {_order_text(uso)}"
    if workspace != launch[1]:
        what += f" / workspace_bytes {workspace}"
    return what


def _order_text(uso: Any) -> str:
    return (
        {True: "on", False: "off"}.get(uso, str(uso))
        if isinstance(uso, bool)
        else str(uso)
    )


def _variant(
    m: Mapping[str, Any], launch: RoundLaunch = DEFAULT_ROUND_LAUNCH
) -> Tuple[str, Any, Any]:
    """(schedule, summation order, workspace) of an entry, `launch`'s for
    what it leaves out, the two last as given when they are invalid
    (`value_errors` names them)."""
    raw = m.get("uniform_summation_order")
    try:
        uso: Any = launch[0] if raw is None else summation_order_value(raw)
    except ValueError:
        uso = repr(raw)
    raw = m.get("workspace_bytes")
    try:
        workspace: Any = launch[1] if raw is None else workspace_value(raw)
    except ValueError:
        workspace = repr(raw)
    return _schedule(m), uso, workspace


def _launch(cfg: Mapping[str, Any]) -> RoundLaunch:
    """`round_launch` of `cfg`, the default where its full workspace is no
    workspace (`context_errors` names that)."""
    try:
        return round_launch(cfg)
    except ValueError:
        return DEFAULT_ROUND_LAUNCH


def value_errors(m: Mapping[str, Any]) -> List[str]:
    """What is wrong with an entry's summation order and workspace."""
    errors = []
    for parse, key in (
        (summation_order_value, "uniform_summation_order"),
        (workspace_value, "workspace_bytes"),
    ):
        try:
            parse(m.get(key))
        except ValueError as e:
            errors.append(str(e))
    return errors


def _is_exclusive_entry(
    m: Mapping[str, Any], launch: RoundLaunch = DEFAULT_ROUND_LAUNCH
) -> bool:
    """True for an entry with a budget of its own that is the exclusive
    context of a run whose round's order and workspace are `launch`; an
    entry with invalid values is not."""
    if unresolved(m) or value_errors(m):
        return False
    return BenchContext.from_mapping(m, launch=launch).is_exclusive()


def trained_entries(cfg: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """The entries of `contexts.list` the config turns on other than the
    exclusive one, as the config gives them (an entry with a cu_fraction is
    never exclusive)."""
    launch = _launch(cfg)
    return [
        m
        for key, _, m in _entries(cfg)
        if key == "list" and not _is_exclusive_entry(m, launch)
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
    is at least one CU."""
    granule = budget_granule(cfg)
    if granule >= 1:
        return []
    return [f"{where}.granule {granule}: must be at least one CU"]


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
    launch = _launch(cfg)
    taken: Dict[Tuple[Any, ...], str] = {}
    for key, i, m in entries:
        if not unresolved(m) and int(m.get("cu_budget") or 0) > 0:
            ident = (int(m["cu_budget"]), *_variant(m, launch))
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
        ident = (budget, *_variant(m, launch))
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
    launch = _launch(source)
    kept = {
        (key, float(m["cu_fraction"]), *_variant(m, launch))
        for key, _, m in _entries(resolved)
        if m.get("cu_fraction") is not None
    }
    return [
        (key, i, m)
        for key, i, m in _entries(source)
        if unresolved(m)
        and (key, float(m["cu_fraction"]), *_variant(m, launch)) not in kept
    ]


def rule_version_of(cfg: Mapping[str, Any]) -> int:
    """The launch rule version a run of `cfg` trains on (its `arch`)."""
    from .hipblaslt_launch import rule_version

    return rule_version(str(cfg.get("arch") or ""))


def workspace_sizes(cfg: Mapping[str, Any]) -> Any:
    """The sizes of the launch-keyed workspace contexts `cfg` asks for:
    `contexts.workspaces` (as given when it is no list), else
    DEFAULT_WORKSPACES."""
    sec = cfg.get("contexts")
    sizes = sec.get("workspaces") if isinstance(sec, Mapping) else None
    if sizes is None:
        return list(DEFAULT_WORKSPACES)
    return list(sizes) if isinstance(sizes, (list, tuple)) else sizes


def workspace_errors(cfg: Mapping[str, Any], where: str = "contexts") -> List[str]:
    """What is wrong with the full workspace (`full_workspace`) and with
    `contexts.workspaces`: a size that is no workspace (`workspace_value`),
    the full workspace (the round's own bench) or a repeat."""
    try:
        full = full_workspace(cfg)
    except ValueError as e:
        return [f"{where}.full_workspace_bytes: {e}"]
    sizes = workspace_sizes(cfg)
    if not isinstance(sizes, list):
        return [f"{where}.workspaces: give a list of workspace sizes in bytes"]
    errors: List[str] = []
    seen: Set[int] = set()
    for i, size in enumerate(sizes):
        try:
            w = workspace_value(size)
        except ValueError as e:
            errors.append(f"{where}.workspaces[{i}]: {e}")
            continue
        if size is None or w == full:
            errors.append(
                f"{where}.workspaces[{i}]: the full workspace is the round's own bench"
            )
        elif w in seen:
            errors.append(f"{where}.workspaces[{i}]: {w} is listed before")
        seen.add(w)
    return errors


def workspace_contexts(cfg: Mapping[str, Any]) -> List[BenchContext]:
    """The launch-keyed workspace contexts of a run of `cfg`: every CU, the
    default schedule and the round's summation order, one per size of
    `workspace_sizes` that a trained entry does not give already; none
    before launch rule version 3, which plans no workspace."""
    if rule_version_of(cfg) < 3 or workspace_errors(cfg):
        return []
    launch = round_launch(cfg)
    taken = {
        (c.cu_budget, c.schedule, c.uniform_summation_order, c.workspace_bytes)
        for c in (
            BenchContext.from_mapping(m, launch=launch)
            for key, _, m in _entries(cfg)
            if key == "list" and not unresolved(m) and not value_errors(m)
        )
    }
    out: List[BenchContext] = []
    for size in workspace_sizes(cfg):
        ctx = BenchContext(
            uniform_summation_order=launch[0],
            workspace_bytes=workspace_value(size),
            launch_keyed=True,
            round_order=launch[0],
            round_workspace=launch[1],
        )
        if (0, "default", launch[0], ctx.workspace_bytes) not in taken:
            out.append(ctx)
    return out


def contexts_from_config(
    cfg: Mapping[str, Any], n_cu: Optional[int] = None
) -> List[BenchContext]:
    """The non-exclusive contexts of `contexts.list`, the launch-keyed
    workspace contexts (`workspace_contexts`) and every `contexts.holdout`
    entry the config turns on, in that order; entries that give a
    cu_fraction are resolved for a device of `n_cu` CUs (ValueError without
    it)."""
    if n_cu is not None:
        cfg = resolve_fractions(cfg, n_cu)[0]
    sec = cfg.get("contexts") or {}
    launch = round_launch(cfg)
    out = [
        BenchContext.from_mapping(m, launch=launch)
        for m in sec.get("list") or []
        if enabled(m)
    ]
    out = [c for c in out if not c.is_exclusive()]
    out += workspace_contexts(cfg)
    out += [
        BenchContext.from_mapping(m, holdout=True, launch=launch)
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
    budget granule of no CU is the only error reported, since every budget
    follows from it."""
    bad = granule_errors(cfg, where)
    if bad:
        return bad
    granule = budget_granule(cfg)
    budgets: Dict[Tuple[str, int], int] = {}
    repeats: Dict[Tuple[str, int], Optional[str]] = {}
    if n_cu is not None:
        budgets, repeats = _resolution(cfg, n_cu)
    floor = (cfg.get("train") or {}).get("min_cu_fraction")
    errors: List[str] = workspace_errors(cfg, where)
    try:
        round_order(cfg)
    except ValueError as e:
        errors.append(f"{where}.{ROUND_ORDER_KEY}: {e}")
    launch = _launch(cfg)
    for key, i, m in off_entries(cfg):
        if _is_exclusive_entry(m, launch):
            errors.append(
                f"{where}.{key}[{i}]: the exclusive context cannot be turned off "
                f"(it is the round's own bench)"
            )
    seen: Dict[Any, str] = {}
    labels: Dict[str, str] = {}
    for key, i, m in _entries(cfg):
        at = f"{where}.{key}[{i}]"
        invalid = value_errors(m)
        if invalid:
            errors += [f"{at}: {e}" for e in invalid]
            continue
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
            ident = ("fraction", float(f), *_variant(m, launch))
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
        c = BenchContext.from_mapping(m, holdout=key == "holdout", launch=launch)
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
        ident = (c.cu_budget, c.schedule, c.uniform_summation_order, c.workspace_bytes)
        if ident in seen:
            errors.append(
                f"{at}: cu_budget {c.cu_budget} / schedule {c.schedule} is "
                f"also {seen[ident]}; rows carry only budget, schedule, "
                f"summation order and workspace"
            )
        seen[ident] = at
        if c.label in labels:
            errors.append(f"{at}: name {c.label!r} is also {labels[c.label]}")
        labels[c.label] = at
    return errors


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


def launch_changing_shapes(
    shapes: Sequence[ShapeKey],
    ctx: BenchContext,
    kernels: Sequence[Mapping[str, Any]],
    launch: Callable[..., Any],
    n_cu: int,
    mi_dtype: str,
) -> List[ShapeKey]:
    """The shapes of `shapes` on which some of `kernels` (lib/dat.py
    `kernel_dat_info` entries) launches otherwise in `ctx` than in the full
    workspace of its budget, schedule and summation order, by `launch`
    (lib/hipblaslt_launch.py `HipblasltLaunch`) on a device of `n_cu` CUs:
    the lines a launch-keyed context benches, a kernel launched alike timing
    alike."""
    here = ctx.execution_context()
    full = replace(ctx, workspace_bytes=ctx.round_workspace).execution_context()
    out = []
    for shape in shapes:
        problem = {
            "m": shape[0],
            "n": shape[1],
            "k": shape[2],
            "batch": shape[3],
            "mi_dtype": mi_dtype,
        }
        if any(
            launch(problem, kernel, kernel.get("attributes"), here, n_cu)
            != launch(problem, kernel, kernel.get("attributes"), full, n_cu)
            for kernel in kernels
        ):
            out.append(shape)
    return out


# ── canaries of the workspace contexts ──────────────────────────────────────
#
# After a launch-keyed workspace context benched, stage03c checks its
# results under the context (`contexts.canaries`, off by default, since the
# bench may lack what they need), every canary line run with
# `--verify_strict`, which compares D with the bench's host reference and
# exits non-zero on a failed check: `fast_check_samples` (line, kernel)
# pairs, each line restamped to run its one kernel on integer-exact
# operands, so that the comparison is exact, drawn only from lines the
# bench compares so (`integer_exact_checkable`; hipblaslt-bench's own
# `fast_check` refuses timing runs, which its runs always are); and
# `--post_bench_verify_iters` on a few lines, each kernel relaunched and its
# D compared with the first launch's. A canary set that performs no check
# is not verified, which stops the stage as a failed check does.
CANARY_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "fast_check_samples": 8,
    "verify_lines": 2,
    "verify_iters": 3,
}
_CANARY_FAILED = re.compile(
    r"post_bench_verify_iters failed for|fast_check.*\bfail|check\(s\) failed",
    re.IGNORECASE,
)
_CANARY_UNSUPPORTED = re.compile(
    r"fast_check (requires|does not support|supports\b)|Skipping integer_exact|"
    r"unrecognised option|unrecognized option|post_bench_verify_iters is not supported",
    re.IGNORECASE,
)
# hipblaslt-bench compares the D of an integer-exact line with its host
# reference only while that reference is exact (testing_matmul.hpp): for a
# 16-bit A (f16, the bench's default, or bf16) up to K 512, or 256 with alpha
# 2 and beta 0 or -2, and never on gfx11; other A types have no such limit.
INTEGER_EXACT_16BIT = ("f16_r", "bf16_r")
INTEGER_EXACT_K_LIMIT = 512
INTEGER_EXACT_K_LIMIT_ALPHA2 = 256
CANARY_LOG_GLOB = "block_*.log"
VERIFIED, NOT_VERIFIED, FAILED = "verified", "not verified", "failed"


def _arch_major(arch: str) -> Optional[int]:
    """The major version of a gfx target (gfx950: 9, gfx1151: 11)."""
    m = re.match(r"gfx([0-9a-f]+)", str(arch or "").lower())
    if m is None or len(m.group(1)) < 3:
        return None
    try:
        return int(m.group(1)[:-2])
    except ValueError:
        return None


def _line_float(line: str, key: str, default: float) -> float:
    from .bench_yaml import line_knob

    try:
        return float(line_knob(line, key) or default)
    except ValueError:
        return default


def integer_exact_checkable(line: str, arch: str) -> bool:
    """Whether hipblaslt-bench on `arch` compares `line`, run on integer-exact
    operands, with its host reference (see INTEGER_EXACT_16BIT)."""
    from .bench_yaml import line_knob, shape_of_line

    shape = shape_of_line(line)
    if shape is None:
        return False
    if (line_knob(line, "a_type") or INTEGER_EXACT_16BIT[0]) not in INTEGER_EXACT_16BIT:
        return True
    if _arch_major(arch) == 11:
        return False
    alpha, beta = _line_float(line, "alpha", 1.0), _line_float(line, "beta", 0.0)
    limit = (
        INTEGER_EXACT_K_LIMIT_ALPHA2
        if alpha == 2 and beta in (0, -2)
        else INTEGER_EXACT_K_LIMIT
    )
    return shape[2] <= limit


def checkable_pairs(
    lines: Sequence[str], kernels: Sequence[int], arch: str
) -> List[Tuple[int, int]]:
    """The (line, kernel) pairs of `lines` whose integer-exact canary the
    bench checks on `arch`: a line's kernels are those it lists, else
    `kernels`."""
    from .bench_yaml import line_solution_list

    return [
        (i, int(k))
        for i, line in enumerate(lines)
        if integer_exact_checkable(line, arch)
        for k in (line_solution_list(line) or kernels)
    ]


def canary_settings(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """`contexts.canaries` over CANARY_DEFAULTS."""
    sec = cfg.get("contexts")
    given = sec.get("canaries") if isinstance(sec, Mapping) else None
    return {**CANARY_DEFAULTS, **(dict(given) if isinstance(given, Mapping) else {})}


def canary_lines(
    lines: Sequence[str],
    kernels: Sequence[int],
    settings: Mapping[str, Any],
    seed: int,
    label: str,
    arch: str = "",
) -> Tuple[List[str], List[str]]:
    """(integer-exact lines, post-bench verification lines) of a context
    that benched `lines` on `arch`, drawn with `seed`: `fast_check_samples`
    of its `checkable_pairs`, each line restamped to run its kernel alone on
    integer-exact operands; and `verify_lines` of `lines` as they are."""
    from .bench_yaml import canary_line, set_line_knob
    from .planner import stable_rng

    rng = stable_rng(seed, "context canary", label)
    pairs = checkable_pairs(lines, kernels, arch)
    n_exact = min(int(settings["fast_check_samples"]), len(pairs))
    exact = []
    for i, k in sorted(rng.sample(pairs, n_exact)):
        line = canary_line(lines[i], k)
        if line is not None:
            exact.append(set_line_knob(line, "initialization", "integer_exact"))
    n_verify = min(int(settings["verify_lines"]), len(lines))
    verify = [lines[i] for i in sorted(rng.sample(range(len(lines)), n_verify))]
    return exact, verify


def _row_shape(fields: Mapping[str, str]) -> Optional[Tuple[int, int, int, int]]:
    """(M, N, K, batch_count) of a result row, as `shape_of_line` reads a line."""
    try:
        return (
            int(fields["m"]),
            int(fields["n"]),
            int(fields["k"]),
            int(fields.get("batch_count") or 1),
        )
    except (KeyError, ValueError):
        return None


def _compared(fields: Mapping[str, str]) -> bool:
    """Whether a result row shows the comparison of its D with the bench's
    host reference (its `norm_error`, which the bench prints only then)."""
    try:
        return math.isfinite(float(fields.get("norm_error", "")))
    except ValueError:
        return False


def canary_verdict(
    texts: Iterable[str],
    exact: Sequence[str] = (),
    verify: Sequence[str] = (),
    counted: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """What the logs `texts` of a context's canaries say: `failed`, the
    checks that failed (and their first lines, `examples`); `unsupported`,
    the lines saying the bench cannot run a check (no verdict, not a
    failure); and the checks performed, counted in the logs `counted` (all
    of `texts` when None): `exact_checks`, the results of the integer-exact
    lines `exact` (each its line's shape and one kernel) whose D the bench
    compared with its host reference, `verify_checks`, those of the lines
    `verify`, each also relaunched and compared with its first launch, and
    `checks`, both. `status`: failed, not verified (no check performed) or
    verified."""
    from . import bench_log
    from .bench_yaml import line_solution_list, shape_of_line

    texts = list(texts)
    failed: List[str] = []
    unsupported: List[str] = []
    for text in texts:
        for line in text.splitlines():
            if _CANARY_UNSUPPORTED.search(line):
                unsupported.append(line.strip())
            elif _CANARY_FAILED.search(line):
                failed.append(line.strip())
    wanted: Dict[Tuple[Any, Optional[int]], int] = {}
    for line in exact:
        listed = line_solution_list(line) or [None]
        key = (shape_of_line(line), listed[0])
        wanted[key] = wanted.get(key, 0) + 1
    verify_shapes = {shape_of_line(line) for line in verify}
    exact_checks = verify_checks = 0
    for text in texts if counted is None else counted:
        for problem in bench_log.iter_problems(text.splitlines()):
            for row in problem.tested:
                if not _compared(row.fields):
                    continue
                shape = _row_shape(row.fields)
                key = (shape, row.sol_index)
                if wanted.get(key):
                    wanted[key] -= 1
                    exact_checks += 1
                elif shape in verify_shapes:
                    verify_checks += 1
    checks = exact_checks + verify_checks
    return {
        "failed": len(failed),
        "examples": failed[:5],
        "unsupported": len(unsupported),
        "unsupported_examples": unsupported[:5],
        "exact_checks": exact_checks,
        "verify_checks": verify_checks,
        "checks": checks,
        "status": FAILED if failed else (VERIFIED if checks else NOT_VERIFIED),
    }


def canary_refusal(canaries: Mapping[str, Any]) -> Optional[str]:
    """Why a context's canary record (stage03c `canaries`) stops the stage:
    a failed check, or a canary set that performed none; None when it
    verified the context."""
    if canaries.get("failed"):
        return (
            f"{canaries['failed']} canary check(s) failed, e.g. "
            f"{canaries['examples'][0]}: the context's results are suspect"
        )
    if canaries.get("status") != VERIFIED:
        return (
            f"the canaries performed no check ({canaries.get('exact_lines', 0)} "
            f"integer-exact line(s) of {canaries.get('checkable_pairs', 0)} "
            f"checkable (line, kernel) pair(s), {canaries.get('verify_lines', 0)} "
            f"relaunched line(s), {canaries.get('unsupported', 0)} the bench "
            f"could not check): the context's results are not verified"
        )
    return None


def targeted_for(path: Optional[Path], ctx: BenchContext) -> Set[ShapeKey]:
    """The shapes stage01 drew near tail GEMMs that ran in the context
    `ctx`, and near seeded anchors that name its label among their
    `bench_contexts` (its targeted_shapes.json at `path`), which
    `select_shapes` benches in that context beside its own draw: none for
    the exclusive context, which benches every shape, or without the
    file."""
    if path is None or ctx.is_exclusive() or not Path(path).is_file():
        return set()
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return set()
    out: Set[ShapeKey] = set()
    for entries in (doc.get("leaves") or {}).values():
        for e in entries or []:
            if "anchor" in e:
                if ctx.label not in (e["anchor"].get("bench_contexts") or []):
                    continue
            else:
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
        "uniform_summation_order": int(ctx.uniform_summation_order),
        "workspace_bytes": int(ctx.workspace_bytes),
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
    bench_args = list(rest) + [str(a) for a in spec.get("bench_args") or ()]
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


def cotenant_kernels(
    cfg: Mapping[str, Any], library_dir: Optional[Path] = None
) -> Tuple[Set[int], List[int]]:
    """(the clustered kernels, the others in index order) of the config's
    pool in `library_dir` (by default the build's library of the config's
    arch: `runtime.build_dir`, else the release build under
    `runtime.rocm_libraries_root`, as the orchestrator finds it); a kernel is
    clustered when its ClusterDim spans more than one workgroup. A co-tenant
    context benches none of the clustered ones (`cotenant_line`): next to a
    co-tenant that holds part of the device for the whole bench, a kernel
    launched in clusters can wait forever for room to place a whole cluster,
    and the co-tenant waits for the bench."""
    from .library_pool import config_pool

    if library_dir is None:
        arch = str(cfg["arch"]).split(":")[0]
        runtime = cfg["runtime"]
        build = runtime.get("build_dir") or (
            Path(runtime["rocm_libraries_root"])
            / "projects"
            / "hipblaslt"
            / "build"
            / "release"
        )
        library_dir = Path(build) / "Tensile" / "library" / arch
    pool = config_pool(cfg, library_dir, cfg["hipblaslt"]["library_stem"])
    clustered: Set[int] = set()
    plain: List[int] = []
    for k in pool.kernels:
        attrs = k.get("attributes") or {}
        size = 1
        for axis in ("cluster_dim_x", "cluster_dim_y", "cluster_dim_z"):
            size *= max(1, int(attrs.get(axis) or 1))
        if size > 1:
            clustered.add(int(k["sol_idx_global"]))
        else:
            plain.append(int(k["sol_idx_global"]))
    return clustered, sorted(plain)


def cotenant_line(line: str, clustered: Set[int], plain: Sequence[int]) -> str:
    """`line` for a co-tenant context (`cotenant_kernels`): one that lists
    kernels without the clustered ones, which can leave it an empty list to
    drop; one that benches the whole library listing `plain` instead."""
    from .bench_yaml import (
        drop_line_knob,
        format_solution_list,
        line_solution_list,
        set_line_knob,
        without_kernels,
    )

    if line_solution_list(line) is not None:
        return without_kernels(line, clustered)
    line = drop_line_knob(line, "algo_method")
    line = set_line_knob(line, "requested_solution_num", -1)
    return set_line_knob(line, "solution_index_list", format_solution_list(plain))
