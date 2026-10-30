# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Evaluate a trained bundle exactly as it is deployed.

The bundle is written as MLREC_v2 at the deploy weight dtype, loaded with
`tilewright.load_model_from_memory`, and every GEMM is ranked by a
`tilewright.CandidateSet` holding the library's kernel pool, the way
TensileLite ranks at runtime: pool-order tie-break and the GEMM's execution
context (`cu_budget`, `schedule`; absent means exclusive use of the device,
the only context v2 models are trained for). MLREC_v3 bundles are written as
MLREC_v3 and ranked with hipBLASLt's launch plans (`lib.hipblaslt_launch`) as
the engine's launch model, the plans stage05 trains with and the deployed
TensileLite adapter must give. The tilewright Python module is required
(`pip install shared/tilewright/python`).

A kernel is identified by its position in the pool; solutions with identical
`tilewright::Config` fields are separate kernels that score alike, and the
engine ranks the first of them in pool order first.

Deployed pick of one GEMM (top-1 request, `min_scored` = 1). The runtime runs
the first kernel of the engine's ranking whose library predicates pass. A
bench without a kernel list times or skip-screens every kernel whose
predicates pass, so a kernel without a row is one the runtime skips too. A
GEMM benched from a plan has no rows for the kernels off its list (`listed`,
see `attach_plan`), which the runtime may run.
  * `pick_measurement="timed"`: walk the
    scored configs in rank order to the first one the runtime would run. When it has a timed row the
    pick costs that latency. When it has only a skip row (one cold warm-up
    call) or is off the plan's list, the GEMM is a miss (`miss` says which):
    it counts at the slowest latency measured for it;
  * `pick_measurement="warmup"` (what the validating stages use by default):
    as "timed", but a pick with only a skip row costs its warm-up latency, a
    close and slightly pessimistic stand-in for its timed latency; a pick
    off the plan's list stays a miss, since nothing measured it;
  * `pick_measurement="rebench"`: as "timed", but a miss whose GEMM carries
    a re-bench of that kernel (`rebench`: {solution: winner / pick latency
    measured together, lib/rebench.py `rebench_picks`}) costs the GEMM's
    winner latency over that ratio, capped at the winner's; a miss without
    one counts as under "warmup" (keeping its `miss` reason);
  * `pick_measurement="any"` (the default here): the first scored config
    with any row, at that row's latency (a skip row's warm-up); kernels off
    a plan's list are passed over like kernels without a row;
  * when the runtime would run no scored config (no trained cell, nothing
    feasible, or only configs without rows scored) it falls back to the
    Origami ranking; the fallback pick is the row with `is_origami_pick == 1`
    (the first tested row of a bench run without tilewright; CSVs without
    that column use the tested row with the lowest bench rank);
  * a GEMM with neither is not evaluable and is only counted.

Per-GEMM selection efficiency is winner_us / pick_us with winner_us the
fastest tested solution; aggregates are geometric means. Paired statistics
compare the deployed pick with the Origami pick on the GEMMs that have both;
those are the only numbers that compare the two.

Library pool: one CandidateSet per (layout, dtypes) key holding each kernel
(`sol_idx_global`) once, with its attributes when the kernel list carries
them. The pool order decides exact score ties, so a pool read from the
library file (`pool_from_kernels`) keeps the runtime order; without the
library file the pool is the union of the kernels measured in the evaluated
GEMMs (`build_pools`), in ascending solution index.

Solution indices name kernels of one hipBLASLt build only
(lib/kernel_identity.py). Plans and timing CSVs that record kernel ids are
read with their indices mapped to the library's (`attach_plans`,
`load_timings` with `identity`); stage05's loader does the same for GEMM
rows.

The engine reads TILEWRIGHT_FORCE_CELL, TILEWRIGHT_PICK_LOG and
TILEWRIGHT_DIAG once per process. `tilewright_module` removes them from the
environment before the module is imported, so an evaluation routes and logs
like the deployed runtime; `Evaluation.n_routing_mismatch` counts GEMMs the
engine served from another cell than `lib.subcells` routes them to, and
`require_consistent_routing` turns any into `RoutingMismatchError`.
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from . import features as fs
from . import hipblaslt_launch, mlrec, ui
from . import kernel_identity as ki
from . import repeats
from .grid import cell_key
from .hardware import DATATYPE_NAME, DATATYPE_VALUE, ArchConstants, DeviceHardware
from .subcells import (
    assign_subcell,
    resolve_model_cell,
    split_tree_from_labels,
    v3_serving_cell,
)

CONFIG_FIELDS: Tuple[str, ...] = (
    "mt_m",
    "mt_n",
    "mt_k",
    "mi_m",
    "mi_n",
    "mi_k",
    "occupancy",
    "cache_hints_a",
    "cache_hints_b",
    "grvw_a",
    "grvw_b",
    "gwvw_d",
)
POOL_KEY_FIELDS: Tuple[str, ...] = (
    "transA",
    "transB",
    "a_type",
    "b_type",
    "c_type",
    "d_type",
    "compute_type",
)


ENGINE_DEBUG_ENV: Tuple[str, ...] = (
    "TILEWRIGHT_FORCE_CELL",
    "TILEWRIGHT_PICK_LOG",
    "TILEWRIGHT_DIAG",
)
PICK_MEASUREMENTS: Tuple[str, ...] = ("warmup", "timed", "any")
# Rules that charge a pick with only a skip-screen row its warm-up latency.
WARMUP_COUNTS: Tuple[str, ...] = ("warmup", "any")
# "rebench" needs re-bench results attached to the GEMMs (lib/miss_rebench.py,
# validate.rebench_misses); PICK_MEASUREMENTS are the rules a stage can apply
# from its data alone.
REBENCH_PICK_MEASUREMENT = "rebench"
EVALUATION_PICK_MEASUREMENTS: Tuple[str, ...] = (
    *PICK_MEASUREMENTS,
    REBENCH_PICK_MEASUREMENT,
)
DEFAULT_PICK_MEASUREMENT = "any"
MISS_REASONS: Tuple[str, ...] = ("screened", "not_benched")


class RoutingMismatchError(Exception):
    """The engine served GEMMs from another cell than lib/subcells routes
    them to."""


def drop_engine_debug_env() -> List[str]:
    """Remove ENGINE_DEBUG_ENV from this process's environment; returns the
    names that were set."""
    dropped = [k for k in ENGINE_DEBUG_ENV if k in os.environ]
    for k in dropped:
        del os.environ[k]
    return dropped


def tilewright_module() -> Any:
    """The tilewright Python module, or RuntimeError with install advice."""
    dropped = drop_engine_debug_env()
    if dropped:
        ui.warn(
            "engine",
            f"removed {', '.join(dropped)} from the environment: evaluation "
            f"routes and ranks like the deployed runtime",
        )
    try:
        import tilewright  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "deployed evaluation needs the tilewright Python module "
            "(pip install shared/tilewright/python): " + str(e)
        ) from e
    return tilewright


# ── engine objects ───────────────────────────────────────────────────────────


def _enum_member(enum_type: Any, value: int) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError):
        for member in getattr(enum_type, "__members__", {}).values():
            if int(member) == value:
                return member
    raise ValueError(f"{enum_type!r} has no member {value}")


def _dim3(tw: Any, m: int, n: int, k: int) -> Any:
    d = tw.Dim3()
    d.m, d.n, d.k = int(m), int(n), int(k)
    return d


def make_problem(tw: Any, problem: Mapping[str, Any]) -> Any:
    """tilewright.Problem of `lib.features` problem kwargs."""
    p = tw.Problem()
    p.size = _dim3(tw, problem["m"], problem["n"], problem["k"])
    p.batch = int(problem["batch"])
    t, nn = tw.Transpose.T, tw.Transpose.N
    p.a_transpose = t if fs._transpose_bit(problem["a_transpose"]) else nn
    p.b_transpose = t if fs._transpose_bit(problem["b_transpose"]) else nn
    for field_name in ("a_dtype", "b_dtype", "c_dtype", "d_dtype", "mi_dtype"):
        name = fs.normalize_dtype_name(problem[field_name])
        if name not in DATATYPE_VALUE:
            raise ValueError(f"{field_name} {problem[field_name]!r} has no DataType")
        setattr(p, field_name, _enum_member(tw.DataType, DATATYPE_VALUE[name]))
    return p


_TAKES_ATTRIBUTES: Dict[int, Tuple[Any, bool]] = {}


def module_takes_attributes(tw: Any) -> bool:
    """True when the module's Config carries kernel attributes. An older
    module gets configs without them, with a warning (once): v2 models do not
    read attributes, but such a module also predates the engine's other
    ranking rules."""
    known = _TAKES_ATTRIBUTES.get(id(tw))
    if known is not None and known[0] is tw:
        return known[1]
    try:
        tw.Config(attributes={})
        takes = True
    except TypeError:
        takes = False
        ui.warn(
            "engine",
            "the tilewright module is older than this checkout's engine (its "
            "Config takes no kernel attributes), so evaluations can differ from "
            "the deployed runtime; reinstall it: pip install shared/tilewright/python",
        )
    _TAKES_ATTRIBUTES[id(tw)] = (tw, takes)
    return takes


def make_config(tw: Any, config: Mapping[str, Any], index: int) -> Any:
    """tilewright.Config of `lib.features` config kwargs and the kernel's
    `attributes` (`lib.dat.kernel_attributes`; none when absent)."""
    kwargs: Dict[str, Any] = dict(
        mt=_dim3(tw, config["mt_m"], config["mt_n"], config["mt_k"]),
        mi=_dim3(tw, config["mi_m"], config["mi_n"], config["mi_k"]),
        occupancy=int(config["occupancy"]),
        cache_hints_a=int(config["cache_hints_a"]),
        cache_hints_b=int(config["cache_hints_b"]),
        grvw_a=int(config["grvw_a"]),
        grvw_b=int(config["grvw_b"]),
        gwvw_d=int(config["gwvw_d"]),
        index=int(index),
    )
    if module_takes_attributes(tw):
        kwargs["attributes"] = {
            str(k): int(v) for k, v in (config.get("attributes") or {}).items()
        }
    return tw.Config(**kwargs)


_PARTIAL_REDUCTION_MEMBERS = {
    "none": "None_",
    "tree": "Tree",
    "parallel": "Parallel",
    "atomic": "Atomic",
}


def make_plan(tw: Any, plan: fs.LaunchPlan) -> Any:
    """tilewright.LaunchPlan of a `features.LaunchPlan`."""
    return tw.LaunchPlan(
        dynamic_queue=plan.dynamic_queue,
        grid_wgs=plan.grid_wgs,
        partial_reduction=getattr(
            tw.PartialReduction, _PARTIAL_REDUCTION_MEMBERS[plan.partial_reduction]
        ),
    )


class EngineLaunch:
    """A tilewright launch model giving the engine `launch`'s plan of pool
    kernel `config.index` on a device with `n_cu` CUs, its launch key
    (`launch.launch_key(kernel, attributes)`; 0 when `launch` has none) and
    `launch`'s `id` (empty when it has none)."""

    def __init__(
        self,
        tw: Any,
        pool: Sequence[Mapping[str, Any]],
        launch: fs.LaunchSource,
        n_cu: int,
    ) -> None:
        self.tw = tw
        self.pool = pool
        self.launch = launch
        self.n_cu = int(n_cu)
        self._schedules = {
            int(tw.Schedule.Default): "default",
            int(tw.Schedule.Dynamic): "dynamic",
            int(tw.Schedule.Auto): "auto",
        }
        self._key_of = getattr(launch, "launch_key", None)
        self.id = str(getattr(launch, "id", "") or "")

    def plan(self, config: Any, problem: Any, context: Any) -> Any:
        kernel = self.pool[int(config.index)]
        prob = {
            "m": int(problem.size.m),
            "n": int(problem.size.n),
            "k": int(problem.size.k),
            "batch": int(problem.batch),
            "mi_dtype": DATATYPE_NAME.get(int(problem.mi_dtype), ""),
        }
        ctx = fs.ExecutionContext(
            cu_budget=int(context.cu_budget),
            schedule=self._schedules[int(context.schedule)],
        )
        return make_plan(
            self.tw,
            self.launch(prob, kernel, kernel.get("attributes"), ctx, self.n_cu),
        )

    def launch_key(self, config: Any) -> int:
        if self._key_of is None:
            return 0
        kernel = self.pool[int(config.index)]
        return int(self._key_of(kernel, kernel.get("attributes"))) & 0xFFFFFFFFFFFFFFFF


def engine_launch(
    tw: Any,
    pool: Sequence[Mapping[str, Any]],
    launch: fs.LaunchSource,
    n_cu: int,
) -> EngineLaunch:
    """The launch model `candidate_set` hands the engine."""
    return EngineLaunch(tw, pool, launch, n_cu)


def candidate_set(
    tw: Any,
    model: Any,
    pool: Sequence[Mapping[str, Any]],
    hardware: Optional[DeviceHardware] = None,
) -> Any:
    """tilewright.CandidateSet of a kernel pool (`make_config` entries in pool
    order, `index` = pool position) with the engine's default pool-order
    tie-break, the one TensileLite uses. An MLREC_v3 model ranks with
    hipBLASLt's launch plans (`lib.hipblaslt_launch`) for its arch on
    `hardware` (its CUs and XCDs), the plans stage05 trains with;
    ValueError without `hardware`."""
    configs = [make_config(tw, k, i) for i, k in enumerate(pool)]
    info = tw.describe(model)
    if int(info.format_version) < 3:
        return tw.CandidateSet(model, configs)
    if hardware is None:
        raise ValueError("an MLREC_v3 candidate set needs the device hardware")
    launch = hipblaslt_launch.HipblasltLaunch(str(info.arch), hardware.num_xcd)
    return tw.CandidateSet(
        model, configs, launch=engine_launch(tw, pool, launch, int(hardware.n_cu))
    )


def make_hardware(tw: Any, hardware: DeviceHardware) -> Any:
    h = tw.Hardware()
    h.N_CU = int(hardware.n_cu)
    h.lds_capacity = int(hardware.lds_bytes)
    h.L2_capacity = int(hardware.l2_bytes)
    return h


def context_of(gemm: Mapping[str, Any]) -> fs.ExecutionContext:
    """Execution context of a GEMM dict or enriched row (`cu_budget`,
    `schedule`; empty or absent means exclusive)."""
    budget = gemm.get("cu_budget")
    schedule = gemm.get("schedule")
    return fs.ExecutionContext(
        cu_budget=int(float(budget)) if budget not in (None, "") else 0,
        schedule=str(schedule).lower() if schedule not in (None, "") else "default",
    )


def make_context(tw: Any, context: fs.ExecutionContext) -> Any:
    schedule = {
        "default": tw.Schedule.Default,
        "dynamic": tw.Schedule.Dynamic,
        "auto": tw.Schedule.Auto,
    }[context.schedule]
    return tw.ExecutionContext(cu_budget=int(context.cu_budget), schedule=schedule)


def load_model_bytes(data: bytes, tw: Any = None) -> Any:
    """Load an in-memory model file; RuntimeError when the engine rejects
    it."""
    tw = tw or tilewright_module()
    try:
        return tw.load_model_from_memory(bytes(data))
    except Exception as e:
        raise RuntimeError(f"the engine rejected the model: {e}") from e


def model_cell_label(tw: Any, model: Any, problem_obj: Any) -> Optional[str]:
    idx = int(tw.route(model, problem_obj))
    if idx < 0:
        return None
    return str(tw.cell_label(model, idx)) or None


# ── kernel pools ─────────────────────────────────────────────────────────────


def pool_key(gemm: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(str(gemm.get(f, "")) for f in POOL_KEY_FIELDS)


def gemm_key(gemm: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Identity of a GEMM problem (sizes, layout, dtypes), plus its execution
    context when that is not exclusive."""
    key = (
        int(gemm["m"]),
        int(gemm["n"]),
        int(gemm["k"]),
        int(gemm["batch_count"]),
    ) + pool_key(gemm)
    ctx = context_of(gemm)
    if not ctx.is_exclusive():
        key += (ctx.cu_budget, ctx.schedule)
    return key


def _kernel(c: Mapping[str, Any]) -> Dict[str, Any]:
    kw: Dict[str, Any] = dict(fs.config_kwargs_from_row(c))
    kw["sol_idx_global"] = int(c["sol_idx_global"])
    if c.get("attributes") is not None:
        kw["attributes"] = dict(c["attributes"])
    if c.get("kernel_name"):
        kw["kernel_id"] = ki.kernel_id(c)
    return kw


def build_pools(
    gemms: Iterable[Mapping[str, Any]],
) -> Dict[Tuple[str, ...], List[Dict[str, Any]]]:
    """Library pools from the candidates measured in `gemms`: per pool key,
    every kernel with a solution index once, ascending by index."""
    pools: Dict[Tuple[str, ...], Dict[int, Dict[str, Any]]] = {}
    for g in gemms:
        bucket = pools.setdefault(pool_key(g), {})
        for c in g.get("candidates", ()):
            sid = int(c.get("sol_idx_global", -1))
            if sid >= 0 and sid not in bucket:
                bucket[sid] = _kernel(c)
    return {k: [v[s] for s in sorted(v)] for k, v in pools.items()}


def pool_from_kernels(kernels: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Pool of one library from its kernel list in runtime order
    (`lib.dat.load_library_kernels`), each kernel with its `kernel_id`; a
    repeated solution index keeps its first entry."""
    pool: List[Dict[str, Any]] = []
    seen = set()
    for info in kernels:
        sid = info.get("sol_idx_global")
        if sid is None or int(sid) in seen:
            continue
        seen.add(int(sid))
        pool.append(_kernel(info))
    return pool


def order_candidates_by_pool(
    gemms: Iterable[Dict[str, Any]],
    pools: Mapping[Tuple[str, ...], Sequence[Mapping[str, Any]]],
) -> None:
    """Sort every GEMM's candidates in the pool order of its pool key, so
    that a first-wins tie-break over them picks what the engine's pool-order
    tie-break picks. Candidates outside the pool follow, by solution index."""
    positions = {
        key: {int(k["sol_idx_global"]): i for i, k in enumerate(pool)}
        for key, pool in pools.items()
    }
    for g in gemms:
        pos = positions.get(pool_key(g), {})
        end = len(pos)
        g["candidates"].sort(
            key=lambda c: (
                pos.get(int(c.get("sol_idx_global", -1)), end),
                int(c.get("sol_idx_global", -1)),
            )
        )


def library_pool(
    library_dir: Any,
    library_stem: str,
    cfg: Optional[Mapping[str, Any]] = None,
    device_index: int = 0,
    blocklist: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Pool of the library `<library_dir>/<library_stem>.dat[.zlib]`, with
    the bank the pipeline config `cfg` asks for on device `device_index`
    (`lib.bank_pool.config_pool`), without the kernels of `blocklist` (a
    `lib.correctness.Blocklist`) when given, else of the config's."""
    from .bank_pool import config_pool, without_blocked

    if blocklist is None:
        pool = config_pool(cfg, library_dir, library_stem, device_index)
    else:
        whole = config_pool(
            cfg, library_dir, library_stem, device_index, apply_blocklist=False
        )
        pool = without_blocked(whole, blocklist)
    return pool_from_kernels(pool.kernels)


def library_identity(
    library_dir: Any,
    library_stem: str,
    cfg: Optional[Mapping[str, Any]] = None,
    device_index: int = 0,
) -> ki.LibraryIdentity:
    """The kernel identity (lib/kernel_identity.py) of the files
    `library_pool` reads."""
    from .bank_pool import config_identity

    return config_identity(cfg, library_dir, library_stem, device_index)


def pools_for(
    gemms: Iterable[Mapping[str, Any]],
    library: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[Tuple[str, ...], List[Dict[str, Any]]]:
    """`build_pools(gemms)`, or every GEMM's key mapped to `library`."""
    gemms = list(gemms)
    if library is None:
        return build_pools(gemms)
    lib_pool = [dict(k) for k in library]
    return {key: lib_pool for key in {pool_key(g) for g in gemms}}


def restrict_to_pool(
    gemms: Iterable[Dict[str, Any]], pool: Sequence[Mapping[str, Any]]
) -> Tuple[List[Dict[str, Any]], int, int]:
    """GEMMs whose candidates are limited to the kernels of `pool` (winner
    recomputed over them), plus the numbers of candidate rows and of GEMMs
    dropped."""
    ids = {int(k["sol_idx_global"]) for k in pool}
    out: List[Dict[str, Any]] = []
    n_rows = n_gemms = 0
    for g in gemms:
        kept = [c for c in g["candidates"] if int(c.get("sol_idx_global", -1)) in ids]
        n_rows += len(g["candidates"]) - len(kept)
        if not kept:
            n_gemms += 1
            continue
        tested = [c["us"] for c in kept if not c.get("is_skip")]
        ng = dict(g)
        ng["candidates"] = kept
        ng["winner_us"] = min(tested) if tested else min(c["us"] for c in kept)
        out.append(ng)
    return out, n_rows, n_gemms


class PoolSet:
    """CandidateSets of one loaded model, one per pool key."""

    def __init__(
        self,
        tw: Any,
        model: Any,
        pools: Mapping[Tuple[str, ...], Sequence[Mapping]],
        hardware: Optional[DeviceHardware] = None,
    ) -> None:
        self.tw, self.model, self.hardware = tw, model, hardware
        self.pools = {k: list(v) for k, v in pools.items()}
        self._sets: Dict[Tuple[str, ...], Any] = {}

    def candidate_set(self, key: Tuple[str, ...]) -> Optional[Any]:
        pool = self.pools.get(key)
        if not pool:
            return None
        cs = self._sets.get(key)
        if cs is None:
            cs = self._sets[key] = candidate_set(
                self.tw, self.model, pool, self.hardware
            )
        return cs


# ── per-GEMM evaluation ──────────────────────────────────────────────────────


def attach_plan(
    gemms: Iterable[Dict[str, Any]],
    plan: Mapping[str, Any],
    identity: Optional[ki.LibraryIdentity] = None,
) -> int:
    """Set `listed` (the solution indices of the shape's kernel list) on every
    GEMM that `plan` (stage02's plan.json) benched from a list; returns how
    many. Shapes benched exhaustively keep no `listed`. With `identity` the
    listed indices are those of `identity`'s library
    (`lib.kernel_identity.remap_of`)."""
    return attach_plans(gemms, [plan], identity=identity)


def read_plan(path: Any) -> Dict[str, Any]:
    """A stage02 plan.json, with its `path` for messages."""
    import json

    doc = json.loads(Path(path).read_text())
    doc.setdefault("path", str(path))
    return doc


def remapped_plan_lookup(
    doc: Mapping[str, Any], identity: Optional[ki.LibraryIdentity], what: str
) -> Dict[Any, Dict[str, Any]]:
    """`lib.planner.plan_lookup` of a plan document, its kernel lists as
    indices of `identity`'s library (`lib.kernel_identity.remap_of`; listed
    kernels it does not hold under one index left out)."""
    from .planner import plan_lookup

    lookup = plan_lookup(doc)
    remap = ki.remap_of(doc, identity, f"plan {doc.get('path') or what}")
    if remap.same:
        return lookup
    return {
        shape: (
            dict(e, kernels=remap.indices(e["kernels"]))
            if e.get("kernels") is not None
            else e
        )
        for shape, e in lookup.items()
    }


def attach_plans(
    gemms: Iterable[Dict[str, Any]],
    plans: Sequence[Mapping[str, Any]],
    context_plans: Optional[
        Mapping[Tuple[int, str], Sequence[Mapping[str, Any]]]
    ] = None,
    identity: Optional[ki.LibraryIdentity] = None,
) -> int:
    """`attach_plan` over several plans (a training set's rounds and seed
    benches): a GEMM's `listed` is the union of its shape's lists, so a
    kernel any of its benches listed is not off-list. A GEMM of a context in
    `context_plans` ({(cu_budget, schedule): plans}, stage02's context
    re-plans) takes its lists from those instead. A shape some plan benched
    exhaustively keeps no `listed`. With `identity` the listed indices are
    those of `identity`'s library (`lib.kernel_identity.remap_of`), with the
    aliases of the alias groups they list (`ki.with_aliases`). Returns the
    number of GEMMs listed."""
    aliases = identity.aliases if identity is not None else {}
    general = [
        remapped_plan_lookup(p, identity, f"{i + 1} of {len(plans)}")
        for i, p in enumerate(plans)
    ]
    by_context = {
        key: [
            remapped_plan_lookup(p, identity, f"of context {key[0]}/{key[1]}")
            for p in ps
        ]
        for key, ps in (context_plans or {}).items()
    }
    n = 0
    for g in gemms:
        shape = (int(g["m"]), int(g["n"]), int(g["k"]), int(g.get("batch_count", 1)))
        c = context_of(g)
        lookups = by_context.get((c.cu_budget, c.schedule), general)
        entries = [lk[shape] for lk in lookups if shape in lk]
        if not entries or any(e.get("kernels") is None for e in entries):
            g.pop("listed", None)
            continue
        g["listed"] = ki.with_aliases(
            frozenset(int(s) for e in entries for s in e["kernels"]), aliases
        )
        n += 1
    return n


def measurements(
    gemm: Mapping[str, Any],
    aggregation: str = repeats.DEFAULT_AGGREGATION,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """({solution: timed latency}, {solution: warm-up latency}) of one
    GEMM's candidates, several rows of a solution merged by `aggregation`
    (lib/repeats.py); a solution with a timed row has no warm-up entry."""
    rows = []
    timed: Dict[int, float] = {}
    screened: Dict[int, float] = {}
    merge = False
    for c in gemm.get("candidates", ()):
        sid = int(c.get("sol_idx_global", -1))
        us = float(c.get("us", 0) or 0)
        if sid < 0 or not us > 0 or not math.isfinite(us):
            continue
        rows.append(c)
        merge = merge or sid in timed or sid in screened or bool(c.get("sustained"))
        (screened if c.get("is_skip") else timed)[sid] = us
    if not merge:
        return timed, screened
    timed, screened = {}, {}
    for c in repeats.merge_candidates(rows, aggregation):
        table = screened if c.get("is_skip") else timed
        table[int(c["sol_idx_global"])] = float(c["us"])
    return timed, screened


ProblemKey = Tuple[int, int, int, int, str, str]


_TIMING_COLUMNS = (
    "m",
    "n",
    "k",
    "batch_count",
    "transA",
    "transB",
    "sol_idx_global",
    "us",
    "is_skip",
    "cu_budget",
    "schedule",
)


def kernels_json(csv_dir: Any) -> Optional[Dict[str, Any]]:
    """The kernels.json of an enriched directory, which holds its kernel
    identity record (lib/kernel_identity.py); None without one."""
    import json

    from .dat import KERNELS_FILE

    path = Path(csv_dir) / KERNELS_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise ValueError(f"{path}: {e}") from None


def load_timings(
    csv_dirs: Iterable[Any],
    wanted: Optional[Iterable[ProblemKey]] = None,
    identity: Optional[ki.LibraryIdentity] = None,
    aggregation: str = repeats.DEFAULT_AGGREGATION,
    blocklist: Optional[Any] = None,
) -> Dict[ProblemKey, Dict[str, Any]]:
    """{(m, n, k, batch, transA, transB): GEMM dict with `candidates` and
    `winner_us` (the fastest timed solution, its rows merged by
    `aggregation`)} of the exclusive-context rows of the chunk_*.csv files
    in `csv_dirs`; rows of one problem in several files merge. `wanted`
    limits the problems read. With `identity` each
    directory's solution indices are those of `identity`'s library
    (`lib.kernel_identity.folder_remap` of its kernels.json): rows of kernels
    it does not hold under one index, or that differ from its kernel at
    their index, are left out. Rows of kernels `blocklist`
    (`lib.correctness.Blocklist`) names, by their names in `identity`'s
    library or else by their kernel ids in the directory's record, are left
    out too."""
    keep = None
    if wanted is not None:
        keep = {tuple(str(v) for v in key) for key in wanted}
    out: Dict[ProblemKey, Dict[str, Any]] = {}
    in_library = (
        blocklist.blocked_indices(identity.ids, identity.kernels)
        if blocklist and identity is not None
        else set()
    )
    for d in csv_dirs:
        doc = None
        if identity is not None or blocklist:
            try:
                doc = kernels_json(d)
            except ValueError as e:
                ui.warn("timings", f"{e}; its kernels are checked by their rows alone")
        remap = ki.folder_remap(doc, identity, f"{d}")
        blocked = in_library
        if blocklist and identity is None:
            try:
                blocked = blocklist.blocked_indices(
                    ki.recorded_ids(ki.read_record(doc)) or {}
                )
            except ValueError:
                blocked = set()
        for path in sorted(Path(d).glob("chunk_*.csv")):
            with path.open(newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None) or []
                col = {c: header.index(c) for c in _TIMING_COLUMNS if c in header}
                if not all(c in col for c in _TIMING_COLUMNS[:8]):
                    continue
                config_cols = [
                    (c, header.index(c)) for c in CONFIG_FIELDS if c in header
                ]
                at = [col[c] for c in _TIMING_COLUMNS[:6]]
                i_sid, i_us = col["sol_idx_global"], col["us"]
                i_skip = col.get("is_skip")
                i_sus = (
                    header.index("is_sustained") if "is_sustained" in header else None
                )
                i_cu, i_sched = col.get("cu_budget"), col.get("schedule")
                for row in reader:
                    try:
                        raw = tuple(row[i] for i in at)
                        if keep is not None and raw not in keep:
                            continue
                        context = context_of(
                            {
                                "cu_budget": row[i_cu] if i_cu is not None else "",
                                "schedule": row[i_sched] if i_sched is not None else "",
                            }
                        )
                        if not context.is_exclusive():
                            continue
                        key = (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3] or 1))
                        sid = int(row[i_sid])
                        fields = (
                            {c: row[i] for c, i in config_cols}
                            if remap.needs_row(sid)
                            else {}
                        )
                        sid = remap.row_index(sid, fields)
                        cand = {
                            "sol_idx_global": sid,
                            "us": float(row[i_us]),
                            "is_skip": i_skip is not None and row[i_skip] == "True",
                        }
                        if i_sus is not None and row[i_sus] == "True":
                            cand["sustained"] = True
                    except (IndexError, TypeError, ValueError):
                        continue
                    if sid is None or sid in blocked:
                        continue
                    full = key + (raw[4], raw[5])
                    out.setdefault(full, {"candidates": []})["candidates"].append(cand)
        remap.report()
    for g in out.values():
        timed, _ = measurements(g, aggregation)
        g["winner_us"] = min(timed.values()) if timed else 0.0
    return out


def pick_cost(
    sid: int,
    timed: Mapping[int, float],
    screened: Mapping[int, float],
    listed: Optional[Iterable[int]],
    pick_measurement: str = DEFAULT_PICK_MEASUREMENT,
    rebench: Optional[Mapping[int, float]] = None,
    off_list_sel_eff: Optional[float] = None,
) -> Tuple[bool, Optional[float], Optional[str]]:
    """(the runtime runs `sid`, its cost, miss reason) for a ranked kernel of
    one GEMM (see the module docstring); the cost of a miss is the slowest
    latency the GEMM measured, or with "rebench" its re-bench ratio (else a
    screened pick's warm-up latency)
    (`rebench`) applied to the winner. `off_list_sel_eff` charges an
    off-list pick without a re-bench the winner's latency over it instead
    of the slowest."""
    if pick_measurement not in EVALUATION_PICK_MEASUREMENTS:
        raise ValueError(
            f"pick_measurement {pick_measurement!r} is not one of "
            f"{EVALUATION_PICK_MEASUREMENTS}"
        )
    if sid in timed:
        return True, timed[sid], None
    if sid in screened:
        if pick_measurement in WARMUP_COUNTS:
            return True, screened[sid], None
        reason: Optional[str] = "screened"
    elif pick_measurement != "any" and listed is not None and sid not in listed:
        reason = "not_benched"
    else:
        return False, None, None
    ratio = (rebench or {}).get(sid)
    if pick_measurement == REBENCH_PICK_MEASUREMENT:
        if ratio and timed:
            return True, min(timed.values()) / min(1.0, float(ratio)), reason
        if reason == "screened":
            return True, screened[sid], reason
    if reason == "not_benched" and off_list_sel_eff and timed:
        return True, min(timed.values()) / float(off_list_sel_eff), reason
    worst = max([*timed.values(), *screened.values()], default=None)
    return True, worst, reason


@dataclass(frozen=True)
class OffListCharge:
    """What an off-list pick without a re-bench costs, as a selection
    efficiency: `by_context` for a GEMM of a listed (cu_budget, schedule),
    else `sel_eff` (None: the slowest latency the GEMM measured)."""

    sel_eff: Optional[float] = None
    by_context: Mapping[Tuple[int, str], float] = field(default_factory=dict)

    def for_gemm(self, gemm: Mapping[str, Any]) -> Optional[float]:
        c = context_of(gemm)
        return self.by_context.get((int(c.cu_budget), c.schedule), self.sel_eff)


def off_list_charge_for(
    charge: Union[None, float, OffListCharge], gemm: Mapping[str, Any]
) -> Optional[float]:
    """`pick_cost`'s `off_list_sel_eff` of `gemm` under `charge`."""
    if isinstance(charge, OffListCharge):
        return charge.for_gemm(gemm)
    return charge


@dataclass
class GemmResult:
    """Deployed result of one GEMM. `pick_index` / `origami_index` are pool
    positions (None outside the pool), `pick_rank` the pick's place in the
    engine's ranking, `miss` why the pick has no timing of its own."""

    index: int
    m: int
    n: int
    k: int
    batch: int
    leaf: str
    model_cell: Optional[str]
    served_by: Optional[str]
    winner_us: float
    pick_sol_idx: Optional[int] = None
    pick_us: Optional[float] = None
    origami_sol_idx: Optional[int] = None
    origami_us: Optional[float] = None
    n_scored: int = 0
    pick_rank: Optional[int] = None
    pick_index: Optional[int] = None
    origami_index: Optional[int] = None
    miss: Optional[str] = None
    # Whether the model's pick is a kernel whose view key the model's
    # trained-key list leaves out; None without such a list.
    pick_untrained: Optional[bool] = None

    @property
    def sel_eff(self) -> Optional[float]:
        if self.pick_us is None or self.pick_us <= 0 or self.winner_us <= 0:
            return None
        return self.winner_us / self.pick_us

    @property
    def origami_sel_eff(self) -> Optional[float]:
        if self.origami_us is None or self.origami_us <= 0 or self.winner_us <= 0:
            return None
        return self.winner_us / self.origami_us

    @property
    def log_ratio(self) -> float:
        se = self.sel_eff
        return float("nan") if se is None else -math.log(se)

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        d["sel_eff"] = self.sel_eff
        d["origami_sel_eff"] = self.origami_sel_eff
        return d


def origami_candidate(gemm: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """The candidate Origami picks at runtime (see the module docstring).
    `is_origami_pick` is None on candidates read from CSVs without the
    column."""
    cands = list(gemm.get("candidates", ()))
    flagged = [c for c in cands if c.get("is_origami_pick")]
    if flagged:
        return flagged[0]
    if any(c.get("is_origami_pick") is not None for c in cands):
        return None
    tested = [
        c
        for c in cands
        if not c.get("is_skip") and int(c.get("rank", -1)) >= 0 and c.get("us", 0) > 0
    ]
    return min(tested, key=lambda c: int(c["rank"])) if tested else None


def origami_summary(gemms: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Origami baseline of `gemms` from their rows alone (no engine)."""
    ratios, picks = [], []
    for g in gemms:
        oc = origami_candidate(g)
        w = float(g.get("winner_us", 0) or 0)
        if oc is not None and w > 0 and float(oc["us"]) > 0:
            ratios.append(w / float(oc["us"]))
            picks.append(float(oc["us"]))
    return {
        "sel_eff": geomean(ratios),
        "pick_us_geomean": geomean(picks),
        "n_eval": len(ratios),
    }


@dataclass
class Evaluation:
    results: List[GemmResult]
    weight_dtype: str
    min_scored: int
    n_pool_configs: Dict[str, int] = field(default_factory=dict)
    n_routing_mismatch: int = 0
    pick_measurement: str = DEFAULT_PICK_MEASUREMENT
    # Per pool, the kernels whose view key the model's trained-key list
    # leaves out (empty without such a list).
    n_untrained_configs: Dict[str, int] = field(default_factory=dict)

    def summary(
        self,
        results: Optional[Sequence[GemmResult]] = None,
        tie_tolerance: float = 0.01,
        bootstrap: int = 0,
    ) -> Dict[str, Any]:
        out = summarize(
            self.results if results is None else results,
            tie_tolerance=tie_tolerance,
            bootstrap=bootstrap,
        )
        out["pick_measurement"] = self.pick_measurement
        return out


def trained_mask(
    pool: Sequence[Mapping[str, Any]], header: Mapping[str, Any]
) -> Optional[List[bool]]:
    """Per kernel of `pool`, whether the model whose header is `header`
    (`mlrec.read_header`) lists its view key as trained; None when it lists
    none."""
    keys = header.get("trained_view_keys")
    if not keys:
        return None
    listed = {int(h) for h in keys}
    spec = header["attribute_spec"]
    return [
        fs.view_key_hash(fs.v3_view_key(k, k.get("attributes"), spec)) in listed
        for k in pool
    ]


def require_consistent_routing(evaluation: Evaluation, what: str) -> Evaluation:
    """`evaluation`, or RoutingMismatchError when the engine served any GEMM
    from another cell than lib/subcells routes it to."""
    if evaluation.n_routing_mismatch:
        raise RoutingMismatchError(
            f"{what}: the engine served {evaluation.n_routing_mismatch} of "
            f"{len(evaluation.results)} GEMMs from another cell than "
            f"lib/subcells routes them to; the measurements do not describe the "
            f"deployed model"
        )
    return evaluation


def evaluate_model(
    data: bytes,
    gemms: Sequence[Mapping[str, Any]],
    *,
    hardware: DeviceHardware,
    pools: Optional[Mapping[Tuple[str, ...], Sequence[Mapping]]] = None,
    min_scored: int = 1,
    labels: Optional[Iterable[str]] = None,
    tw: Any = None,
    pick_measurement: str = DEFAULT_PICK_MEASUREMENT,
    off_list_sel_eff: Union[None, float, OffListCharge] = None,
) -> Evaluation:
    """Deployed picks of `gemms` (stage05 GEMM dicts with `candidates` and
    `winner_us`, optionally `listed`) under the model file `data`. A GEMM's
    measurement of a solution is its timed rows of that solution index,
    else its skip rows, merged by the median (`measurements`; the loader
    merged them by the run's rule already); `off_list_sel_eff` as in
    `pick_cost`, or per GEMM context (`OffListCharge`)."""
    if pick_measurement not in EVALUATION_PICK_MEASUREMENTS:
        raise ValueError(
            f"pick_measurement {pick_measurement!r} is not one of "
            f"{EVALUATION_PICK_MEASUREMENTS}"
        )
    tw = tw or tilewright_module()
    header = mlrec.read_header(data)
    model = load_model_bytes(data, tw)
    if labels is None:
        labels = [c["label"] for c in mlrec.read_model(data)["cells"]]
    labels = list(labels)
    tree = split_tree_from_labels(labels)
    label_set = set(labels)
    pools = build_pools(gemms) if pools is None else pools
    poolset = PoolSet(tw, model, pools, hardware)
    positions = {
        key: {int(k["sol_idx_global"]): i for i, k in enumerate(pool)}
        for key, pool in poolset.pools.items()
    }
    hw = make_hardware(tw, hardware)
    trained = {key: trained_mask(pool, header) for key, pool in poolset.pools.items()}
    results: List[GemmResult] = []
    n_mismatch = 0
    for gi, g in enumerate(gemms):
        prob = fs.problem_kwargs_from_row(g)
        m, n, k, b = prob["m"], prob["n"], prob["k"], prob["batch"]
        leaf = assign_subcell(cell_key(m, n, k, b), m, n, k, b, tree)
        problem_obj = make_problem(tw, prob)
        cell = model_cell_label(tw, model, problem_obj)
        expected = (
            v3_serving_cell(m, n, k, b, tree, label_set)
            if header["version"] == 3
            else resolve_model_cell(leaf, label_set)
        )
        if cell != expected:
            n_mismatch += 1
        timed, screened = measurements(g)
        listed = g.get("listed")
        charge = off_list_charge_for(off_list_sel_eff, g)
        res = GemmResult(
            index=gi,
            m=m,
            n=n,
            k=k,
            batch=b,
            leaf=leaf,
            model_cell=cell,
            served_by=None,
            winner_us=float(g.get("winner_us", 0) or 0),
        )
        key = pool_key(g)
        pos_of = positions.get(key, {})
        oc = origami_candidate(g)
        if oc is not None:
            res.origami_sol_idx = int(oc.get("sol_idx_global", -1))
            res.origami_us = float(oc["us"])
            res.origami_index = pos_of.get(res.origami_sol_idx)
        cs = poolset.candidate_set(key)
        if cs is not None:
            pool = poolset.pools[key]
            ranked = cs.rank(
                problem_obj, hw, int(min_scored), make_context(tw, context_of(g))
            )
            for rank, r in enumerate(ranked):
                if not r.scored:
                    break
                res.n_scored += 1
                if res.served_by is not None:
                    continue
                sid = int(pool[int(r.config_index)]["sol_idx_global"])
                runs, cost, miss = pick_cost(
                    sid,
                    timed,
                    screened,
                    listed,
                    pick_measurement,
                    g.get("rebench"),
                    charge,
                )
                if runs:
                    res.pick_sol_idx, res.pick_us, res.miss = sid, cost, miss
                    res.pick_index = int(r.config_index)
                    res.pick_rank = rank
                    res.served_by = "model"
                    mask = trained.get(key)
                    if mask is not None:
                        res.pick_untrained = not mask[res.pick_index]
        if res.served_by is None and res.origami_us is not None:
            res.pick_sol_idx, res.pick_us = res.origami_sol_idx, res.origami_us
            res.pick_index = res.origami_index
            res.served_by = "origami_fallback"
        results.append(res)
    return Evaluation(
        results=results,
        weight_dtype=mlrec.WEIGHT_DTYPE_NAMES[header["weight_dtype"]],
        min_scored=int(min_scored),
        n_pool_configs={"|".join(k): len(v) for k, v in poolset.pools.items()},
        n_routing_mismatch=n_mismatch,
        pick_measurement=pick_measurement,
        n_untrained_configs={
            "|".join(k): m.count(False) for k, m in trained.items() if m is not None
        },
    )


def evaluate_bundle(
    bundle: Mapping[str, Any],
    gemms: Sequence[Mapping[str, Any]],
    *,
    arch: str,
    constants: ArchConstants,
    hardware: DeviceHardware,
    weight_dtype: str,
    pools: Optional[Mapping[Tuple[str, ...], Sequence[Mapping]]] = None,
    min_scored: int = 1,
    tw: Any = None,
    pick_measurement: str = DEFAULT_PICK_MEASUREMENT,
    off_list_sel_eff: Union[None, float, OffListCharge] = None,
) -> Evaluation:
    """`evaluate_model` of the bundle written at `weight_dtype`."""
    data = mlrec.write_model(bundle, None, arch, constants, weight_dtype)
    return evaluate_model(
        data,
        gemms,
        hardware=hardware,
        pools=pools,
        min_scored=min_scored,
        labels=list(bundle["models"].keys()),
        tw=tw,
        pick_measurement=pick_measurement,
        off_list_sel_eff=off_list_sel_eff,
    )


# ── statistics ───────────────────────────────────────────────────────────────


def geomean(values: Iterable[float]) -> float:
    logs = [math.log(v) for v in values if v is not None and v > 0 and math.isfinite(v)]
    return math.exp(sum(logs) / len(logs)) if logs else float("nan")


def bootstrap_ci(
    values: Sequence[float],
    statistic: Any = np.mean,
    resamples: int = 1000,
    seed: int = 0,
) -> Optional[List[float]]:
    """Percentile-bootstrap 95% interval of `statistic` over `values`; None
    for fewer than two values or no resamples."""
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if arr.size < 2 or resamples <= 0:
        return None
    rng = np.random.default_rng(seed)
    stats = np.empty(int(resamples), dtype=np.float64)
    for i in range(int(resamples)):
        stats[i] = statistic(arr[rng.integers(0, arr.size, arr.size)])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return [float(lo), float(hi)]


PERCENTILES: Tuple[float, ...] = (1.0, 5.0, 10.0, 25.0, 50.0)


def _percentiles(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {f"p{int(p)}": None for p in PERCENTILES}
    arr = np.asarray(values, dtype=np.float64)
    return {f"p{int(p)}": float(np.percentile(arr, p)) for p in PERCENTILES}


def summarize(
    results: Sequence[GemmResult],
    *,
    tie_tolerance: float = 0.01,
    bootstrap: int = 0,
    seed: int = 0,
) -> Dict[str, Any]:
    """Aggregate selection efficiency and model-vs-Origami paired statistics.

    `sel_eff` is the deployed pick against the fastest measured kernel over
    every evaluated GEMM; `paired` compares the deployed and the Origami pick
    on the GEMMs that have both. `tie_tolerance` is the relative latency
    difference treated as a tie; `bootstrap` > 0 adds a percentile-bootstrap
    95% interval of the paired geomean speedup (Origami pick latency /
    deployed pick latency)."""
    evaluated = [r for r in results if r.sel_eff is not None]
    paired = [r for r in evaluated if r.origami_sel_eff is not None]
    model_se = [r.sel_eff for r in evaluated]
    tol = math.log1p(max(float(tie_tolerance), 0.0))
    d = [math.log(r.origami_us / r.pick_us) for r in paired]
    wins = sum(1 for x in d if x > tol)
    losses = sum(1 for x in d if x < -tol)
    out: Dict[str, Any] = {
        "n_gemms": len(results),
        "n_evaluated": len(evaluated),
        "n_not_evaluable": len(results) - len(evaluated),
        "n_model_served": sum(1 for r in evaluated if r.served_by == "model"),
        "n_origami_fallback": sum(
            1 for r in evaluated if r.served_by == "origami_fallback"
        ),
        "n_miss": sum(1 for r in evaluated if r.miss is not None),
        "n_untrained_picks": sum(1 for r in evaluated if r.pick_untrained),
        "n_miss_by_reason": {
            why: sum(1 for r in evaluated if r.miss == why) for why in MISS_REASONS
        },
        "sel_eff": geomean(model_se),
        "pick_us_geomean": geomean(r.pick_us for r in evaluated),
        "winner_us_geomean": geomean(r.winner_us for r in evaluated),
        "model_sel_eff_percentiles": _percentiles(model_se),
        "n_origami": sum(1 for r in results if r.origami_sel_eff is not None),
        "origami_sel_eff": geomean(
            r.origami_sel_eff for r in results if r.origami_sel_eff is not None
        ),
        "origami_pick_us_geomean": geomean(
            r.origami_us for r in results if r.origami_sel_eff is not None
        ),
        "paired": {
            "n": len(paired),
            "model_sel_eff": geomean(r.sel_eff for r in paired),
            "origami_sel_eff": geomean(r.origami_sel_eff for r in paired),
            "speedup_geomean": math.exp(sum(d) / len(d)) if d else float("nan"),
            "tie_tolerance": float(tie_tolerance),
            "wins": wins,
            "ties": len(d) - wins - losses,
            "losses": losses,
            "same_kernel": sum(
                1 for r in paired if r.pick_sol_idx == r.origami_sol_idx
            ),
            "n_miss": sum(1 for r in paired if r.miss is not None),
            "model_sel_eff_percentiles": _percentiles([r.sel_eff for r in paired]),
            "origami_sel_eff_percentiles": _percentiles(
                [r.origami_sel_eff for r in paired]
            ),
        },
    }
    ci = bootstrap_ci(d, resamples=int(bootstrap), seed=seed)
    if ci is not None:
        out["paired"]["speedup_ci95"] = [math.exp(ci[0]), math.exp(ci[1])]
        out["paired"]["bootstrap_resamples"] = int(bootstrap)
    return out


# ── top-1 picks for parity checks ────────────────────────────────────────────


def top1_picks(
    data: bytes,
    problems: Sequence[Mapping[str, Any]],
    pool: Sequence[Mapping[str, Any]],
    *,
    hardware: DeviceHardware,
    min_scored: int = 1,
    tw: Any = None,
    with_ranking: bool = False,
) -> List[Dict[str, Any]]:
    """Top-1 config of each problem (dicts with m, n, k, batch_count, the
    hipBLASLt type strings and transA/transB) over one library pool, as the
    engine ranks it. `top1_index` is its pool position (`top1_index=` of the
    engine's pick log), `cell` the serving cell label (`leaf=`), `n_configs`
    the pool size; `top1_sig` and `sol_idx_global` describe the kernel. With
    `with_ranking`, `ranked` lists the pool positions of every scored config
    in rank order."""
    tw = tw or tilewright_module()
    model = load_model_bytes(data, tw)
    cs = candidate_set(tw, model, pool, hardware)
    hw = make_hardware(tw, hardware)
    out: List[Dict[str, Any]] = []
    for p in problems:
        row = dict(p)
        if "batch_count" not in row:
            row["batch_count"] = row.get("batch", 1)
        prob = fs.problem_kwargs_from_row(row)
        pobj = make_problem(tw, prob)
        rec: Dict[str, Any] = {
            "m": prob["m"],
            "n": prob["n"],
            "k": prob["k"],
            "batch": prob["batch"],
            "transA": row["transA"],
            "transB": row["transB"],
            "cell": model_cell_label(tw, model, pobj),
            "n_configs": len(pool),
        }
        ranked = cs.rank(pobj, hw, int(min_scored))
        scored = [r for r in ranked if r.scored]
        rec["n_scored"] = len(scored)
        if with_ranking:
            rec["ranked"] = [int(r.config_index) for r in scored]
        if rec["cell"] is None:
            rec["reason"] = "no_model"
        elif not scored:
            rec["reason"] = "no_survivors"
        else:
            rec["top1_index"] = int(scored[0].config_index)
            top = pool[rec["top1_index"]]
            rec["reason"] = "ok"
            rec["sol_idx_global"] = top.get("sol_idx_global")
            rec["top1_sig"] = tuple(
                int(top[f])
                for f in (
                    "mt_m",
                    "mt_n",
                    "mt_k",
                    "mi_m",
                    "mi_n",
                    "mi_k",
                    "cache_hints_a",
                    "cache_hints_b",
                )
            )
            rec["top1_score"] = float(scored[0].score)
        out.append(rec)
    return out
