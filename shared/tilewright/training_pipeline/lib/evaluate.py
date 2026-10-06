# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Evaluate a trained bundle exactly as it is deployed.

The bundle is written as MLREC_v2 at the deploy weight dtype, loaded with
`tilewright.load_model_from_memory`, and every GEMM is ranked by a
`tilewright.CandidateSet` holding the library's kernel pool, the way
TensileLite ranks at runtime: pool-order tie-break and the exclusive
execution context (the engine's defaults, the only context v2 models serve).
The tilewright Python module is required
(`pip install shared/tilewright/python`).

A kernel is identified by its position in the pool; solutions with identical
`tilewright::Config` fields are separate kernels that score alike, and the
engine ranks the first of them in pool order first.

Deployed pick of one GEMM (top-1 request, `min_scored` = 1):
  * walk the scored configs in rank order and take the first pool position
    this GEMM has a measurement for; the pick costs that solution's own
    latency. The bench measured every solution that passes the library's
    predicates for the problem, so this is the runtime's "first ranked
    solution whose predicates pass". A predicate-passing kernel the bench
    failed to measure is skipped here, where the runtime could pick it;
  * when no scored config is measured (no trained cell, nothing feasible, or
    only unmeasured configs scored) the runtime falls back to the Origami
    ranking; the fallback pick is the row with `is_origami_pick == 1` (the
    first tested row of a bench run without tilewright; CSVs without that
    column use the tested row with the lowest bench rank);
  * a GEMM with neither is not evaluable and is only counted.

Per-GEMM selection efficiency is winner_us / pick_us with winner_us the
fastest tested solution; aggregates are geometric means. Paired statistics
compare the deployed pick with the Origami pick on the GEMMs that have both.

Library pool: one CandidateSet per (layout, dtypes) key holding each kernel
(`sol_idx_global`) once, with its attributes when the kernel list carries
them. The pool order decides exact score ties, so a pool read from the
library file (`pool_from_kernels`) keeps the runtime order; without the
library file the pool is the union of the kernels measured in the evaluated
GEMMs (`build_pools`), in ascending solution index.

The engine reads TILEWRIGHT_FORCE_CELL, TILEWRIGHT_PICK_LOG and
TILEWRIGHT_DIAG once per process. `tilewright_module` removes them from the
environment before the module is imported, so an evaluation routes and logs
like the deployed runtime; `Evaluation.n_routing_mismatch` counts GEMMs the
engine served from another cell than `lib.subcells` routes them to, and
`require_consistent_routing` turns any into `RoutingMismatchError`.
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import features as fs
from . import mlrec, ui
from .grid import cell_key
from .hardware import DATATYPE_VALUE, ArchConstants, DeviceHardware
from .subcells import assign_subcell, resolve_model_cell, split_tree_from_labels

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


def candidate_set(tw: Any, model: Any, pool: Sequence[Mapping[str, Any]]) -> Any:
    """tilewright.CandidateSet of a kernel pool (`make_config` entries in pool
    order, `index` = pool position) with the engine's default pool-order
    tie-break, the one TensileLite uses."""
    return tw.CandidateSet(model, [make_config(tw, k, i) for i, k in enumerate(pool)])


def make_hardware(tw: Any, hardware: DeviceHardware) -> Any:
    h = tw.Hardware()
    h.N_CU = int(hardware.n_cu)
    h.lds_capacity = int(hardware.lds_bytes)
    h.L2_capacity = int(hardware.l2_bytes)
    return h


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
    """Identity of a GEMM problem (sizes, layout, dtypes)."""
    return (
        int(gemm["m"]),
        int(gemm["n"]),
        int(gemm["k"]),
        int(gemm["batch_count"]),
    ) + pool_key(gemm)


def _kernel(c: Mapping[str, Any]) -> Dict[str, Any]:
    kw: Dict[str, Any] = dict(fs.config_kwargs_from_row(c))
    kw["sol_idx_global"] = int(c["sol_idx_global"])
    if c.get("attributes") is not None:
        kw["attributes"] = dict(c["attributes"])
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
    (`lib.dat.load_library_kernels`); a repeated solution index keeps its
    first entry."""
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


def library_pool(library_dir: Any, library_stem: str) -> List[Dict[str, Any]]:
    """Pool of the library `<library_dir>/<library_stem>.dat[.zlib]`."""
    from .dat import load_library_kernels

    return pool_from_kernels(load_library_kernels(library_dir, library_stem))


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
        self, tw: Any, model: Any, pools: Mapping[Tuple[str, ...], Sequence[Mapping]]
    ) -> None:
        self.tw, self.model = tw, model
        self.pools = {k: list(v) for k, v in pools.items()}
        self._sets: Dict[Tuple[str, ...], Any] = {}

    def candidate_set(self, key: Tuple[str, ...]) -> Optional[Any]:
        pool = self.pools.get(key)
        if not pool:
            return None
        cs = self._sets.get(key)
        if cs is None:
            cs = self._sets[key] = candidate_set(self.tw, self.model, pool)
        return cs


# ── per-GEMM evaluation ──────────────────────────────────────────────────────


@dataclass
class GemmResult:
    """Deployed result of one GEMM. `pick_index` / `origami_index` are pool
    positions (None outside the pool), `pick_rank` the pick's place in the
    engine's ranking."""

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

    def summary(
        self,
        results: Optional[Sequence[GemmResult]] = None,
        tie_tolerance: float = 0.01,
        bootstrap: int = 0,
    ) -> Dict[str, Any]:
        return summarize(
            self.results if results is None else results,
            tie_tolerance=tie_tolerance,
            bootstrap=bootstrap,
        )


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
) -> Evaluation:
    """Deployed picks of `gemms` (stage05 GEMM dicts with `candidates` and
    `winner_us`) under the model file `data`. A GEMM's measurement of a
    solution is its fastest row of that solution index."""
    tw = tw or tilewright_module()
    header = mlrec.read_header(data)
    model = load_model_bytes(data, tw)
    if labels is None:
        labels = [c["label"] for c in mlrec.read_model(data)["cells"]]
    labels = list(labels)
    tree = split_tree_from_labels(labels)
    label_set = set(labels)
    pools = build_pools(gemms) if pools is None else pools
    poolset = PoolSet(tw, model, pools)
    positions = {
        key: {int(k["sol_idx_global"]): i for i, k in enumerate(pool)}
        for key, pool in poolset.pools.items()
    }
    hw = make_hardware(tw, hardware)
    results: List[GemmResult] = []
    n_mismatch = 0
    for gi, g in enumerate(gemms):
        prob = fs.problem_kwargs_from_row(g)
        m, n, k, b = prob["m"], prob["n"], prob["k"], prob["batch"]
        leaf = assign_subcell(cell_key(m, n, k, b), m, n, k, b, tree)
        problem_obj = make_problem(tw, prob)
        cell = model_cell_label(tw, model, problem_obj)
        if cell != resolve_model_cell(leaf, label_set):
            n_mismatch += 1
        measured: Dict[int, float] = {}
        for c in g.get("candidates", ()):
            sid = int(c.get("sol_idx_global", -1))
            us = float(c.get("us", 0) or 0)
            if sid >= 0 and us > 0 and (sid not in measured or us < measured[sid]):
                measured[sid] = us
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
            for rank, r in enumerate(cs.rank(problem_obj, hw, int(min_scored))):
                if not r.scored:
                    break
                res.n_scored += 1
                sid = pool[int(r.config_index)]["sol_idx_global"]
                if res.pick_us is None and sid in measured:
                    res.pick_sol_idx, res.pick_us = sid, measured[sid]
                    res.pick_index = int(r.config_index)
                    res.pick_rank = rank
                    res.served_by = "model"
        if res.pick_us is None and res.origami_us is not None:
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
    )


# ── statistics ───────────────────────────────────────────────────────────────


def geomean(values: Iterable[float]) -> float:
    logs = [math.log(v) for v in values if v is not None and v > 0 and math.isfinite(v)]
    return math.exp(sum(logs) / len(logs)) if logs else float("nan")


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

    `tie_tolerance` is the relative latency difference treated as a tie;
    `bootstrap` > 0 adds a percentile-bootstrap 95% interval of the paired
    geomean speedup (Origami pick latency / deployed pick latency)."""
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
            "model_sel_eff_percentiles": _percentiles([r.sel_eff for r in paired]),
            "origami_sel_eff_percentiles": _percentiles(
                [r.origami_sel_eff for r in paired]
            ),
        },
    }
    if bootstrap > 0 and len(d) > 1:
        rng = np.random.default_rng(seed)
        arr = np.asarray(d, dtype=np.float64)
        means = np.empty(int(bootstrap), dtype=np.float64)
        for i in range(int(bootstrap)):
            means[i] = arr[rng.integers(0, arr.size, arr.size)].mean()
        lo, hi = np.percentile(means, [2.5, 97.5])
        out["paired"]["speedup_ci95"] = [float(math.exp(lo)), float(math.exp(hi))]
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
) -> List[Dict[str, Any]]:
    """Top-1 config of each problem (dicts with m, n, k, batch_count, the
    hipBLASLt type strings and transA/transB) over one library pool, as the
    engine ranks it. `top1_index` is its pool position (`top1_index=` of the
    engine's pick log), `cell` the serving cell label (`leaf=`), `n_configs`
    the pool size; `top1_sig` and `sol_idx_global` describe the kernel."""
    tw = tw or tilewright_module()
    model = load_model_bytes(data, tw)
    cs = candidate_set(tw, model, pool)
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
