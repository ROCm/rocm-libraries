# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Benchmark planning: which kernels to time for each shape of a round.

The library heuristic is never consulted. A shape is either benched
exhaustively (every kernel the bench supports, in library order) or with an
explicit kernel list built from a ranker:

  1. the ranker's `top_k` kernels,
  2. every variant of the `families_top_f` best-ranked families (a family is
     the kernel signature: macro tile M, N, depthU, matrix instruction M, N,
     K and the A / B cache hints), at most `family_variants` per family
     (0: all),
  3. `explore_random_k` kernels drawn at random from the rest,
  4. per cell, kernels benched on fewer than `min_coverage` shapes of the
     cell are added to shapes that have room, until each reaches the quota.

Only kernels that pass the engine's gates for the shape (LDS fit and the
soft rules of `lib.feasibility` the run's models serve with) are planned.
Lists hold at most
`max_list` kernels (`MAX_BENCH_LIST`, the bench's limit for one problem, at
most).

Rankers order the candidate kernels of one shape:
  * NeighborsRanker: normalised latencies (winner_us / us) of exhaustively
    benched seed shapes, averaged over the nearest seeds of the same cell in
    log2 (M, N, K, batch) space,
  * ModelRanker: a trained model's ranking through the tilewright engine.

All randomness comes from `random.Random` seeded with the plan seed and the
shape or cell, so a plan is reproducible and does not depend on the order of
unrelated shapes.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import math
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence
from typing import Set, Tuple

from . import features as fs
from .feasibility import non_temporal_availability, passes_gates, soft_rules
from .grid import cell_key

ShapeKey = Tuple[int, int, int, int]
SOFT_RULES = (
    "non_temporal_rule",
    "small_batched_rule",
    "dot2_rule",
    "leading_free_rule",
)
FAMILY_FIELDS = (
    "mt_m",
    "mt_n",
    "mt_k",
    "mi_m",
    "mi_n",
    "mi_k",
    "cache_hints_a",
    "cache_hints_b",
)
# Bench limit on the length of one problem's solution_index_list.
MAX_BENCH_LIST = 4096
PLAN_FILE = "plan.json"
PLAN_VERSION = 1


@dataclass(frozen=True)
class PlanParams:
    top_k: int = 32
    families_top_f: int = 8
    family_variants: int = 0
    explore_random_k: int = 32
    min_coverage: int = 2
    max_list: int = 512
    probe_kernels: int = 3
    seed: int = 42

    def as_dict(self) -> Dict[str, int]:
        return asdict(self)


def family_of(kernel: Mapping[str, Any]) -> Tuple[int, ...]:
    return tuple(int(kernel.get(f, 0) or 0) for f in FAMILY_FIELDS)


def shape_cell(shape: ShapeKey) -> str:
    m, n, k, b = shape
    return cell_key(m, n, k, b)


def stable_rng(*parts: Any) -> random.Random:
    """Generator seeded from `parts` independent of PYTHONHASHSEED."""
    text = "\x1f".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "little"))


# ── kernel catalog ───────────────────────────────────────────────────────────


@dataclass
class Catalog:
    """Plannable kernels: the pool (the deployed library, in pool order) and
    optional bank kernels that only enter lists as family variants. A
    solution index appears once; the pool wins."""

    kernels: Dict[int, Dict[str, Any]]
    pool_ids: List[int]
    bank_ids: List[int]
    families: Dict[Tuple[int, ...], List[int]]
    nt_available: Tuple[bool, bool]

    @classmethod
    def build(
        cls,
        pool: Sequence[Mapping[str, Any]],
        bank: Sequence[Mapping[str, Any]] = (),
    ) -> "Catalog":
        kernels: Dict[int, Dict[str, Any]] = {}
        pool_ids: List[int] = []
        bank_ids: List[int] = []
        for src, ids in ((pool, pool_ids), (bank, bank_ids)):
            for info in src:
                sid = info.get("sol_idx_global")
                if sid is None or int(sid) in kernels:
                    continue
                kernels[int(sid)] = dict(info)
                ids.append(int(sid))
        families: Dict[Tuple[int, ...], List[int]] = {}
        for sid in pool_ids + bank_ids:
            families.setdefault(family_of(kernels[sid]), []).append(sid)
        return cls(
            kernels=kernels,
            pool_ids=pool_ids,
            bank_ids=bank_ids,
            families=families,
            nt_available=non_temporal_availability(kernels[s] for s in pool_ids),
        )


@dataclass(frozen=True)
class ProblemType:
    """The dtype / layout part of a problem, from the config's `hipblaslt`
    section."""

    trans_a: str
    trans_b: str
    a_type: str
    b_type: str
    c_type: str
    d_type: str
    compute_type: str

    @classmethod
    def from_config(cls, hl: Mapping[str, Any]) -> "ProblemType":
        return cls(
            trans_a=str(hl["transA"]),
            trans_b=str(hl["transB"]),
            a_type=str(hl["a_type"]),
            b_type=str(hl["b_type"]),
            c_type=str(hl["c_type"]),
            d_type=str(hl["d_type"]),
            compute_type=str(hl["compute_type"]),
        )

    def row(self, shape: ShapeKey) -> Dict[str, Any]:
        m, n, k, b = shape
        return {
            "m": m,
            "n": n,
            "k": k,
            "batch_count": b,
            "transA": self.trans_a,
            "transB": self.trans_b,
            "a_type": self.a_type,
            "b_type": self.b_type,
            "c_type": self.c_type,
            "d_type": self.d_type,
            "compute_type": self.compute_type,
        }


class Feasibility:
    """Kernels of a catalog that pass the engine's gates for a shape: the LDS
    gate and the `passes_gates` soft rules in `rules` (default: v2's, with
    `non_temporal_rule`). A model must be planned with the rules it serves
    with (`feasibility.soft_rules`), or deployment picks kernels no list
    has."""

    def __init__(
        self,
        catalog: Catalog,
        ptype: ProblemType,
        lds_bytes: int,
        non_temporal_rule: bool = True,
        *,
        rules: Optional[Mapping[str, bool]] = None,
    ) -> None:
        self.catalog = catalog
        self.ptype = ptype
        self.lds_bytes = int(lds_bytes)
        self.rules: Dict[str, bool] = (
            {k: bool(v) for k, v in rules.items()}
            if rules is not None
            else soft_rules({"non_temporal_rule": non_temporal_rule}, 2)
        )
        unknown = sorted(set(self.rules) - set(SOFT_RULES))
        if unknown:
            raise ValueError(f"unknown soft rules {unknown}; allowed: {SOFT_RULES}")
        self.non_temporal_rule = self.rules.get("non_temporal_rule", True)
        self._configs = {
            sid: fs.config_kwargs_from_row(k) for sid, k in catalog.kernels.items()
        }

    def ok(self, shape: ShapeKey, sid: int) -> bool:
        problem = fs.problem_kwargs_from_row(self.ptype.row(shape))
        return self._ok(problem, sid)

    def _ok(self, problem: Mapping[str, Any], sid: int) -> bool:
        nt_a, nt_b = self.catalog.nt_available
        return passes_gates(
            problem,
            self._configs[sid],
            lds_bytes=self.lds_bytes,
            nt_a_available=nt_a,
            nt_b_available=nt_b,
            **self.rules,
        )

    def feasible(self, shape: ShapeKey, ids: Iterable[int]) -> List[int]:
        problem = fs.problem_kwargs_from_row(self.ptype.row(shape))
        return [sid for sid in ids if self._ok(problem, sid)]


# ── rankers ──────────────────────────────────────────────────────────────────


class Ranker:
    name = "none"

    def rank(self, shape: ShapeKey) -> List[int]:
        """Solution indices, best first; may be empty or partial."""
        raise NotImplementedError


class NeighborsRanker(Ranker):
    """Ranking from exhaustively benched seed shapes (see module docstring).
    `seeds` maps a shape to {solution index: us}; a kernel missing from a
    neighbour scores 0 there."""

    name = "neighbors"

    def __init__(self, seeds: Mapping[ShapeKey, Mapping[int, float]], k: int = 4):
        self.k = max(1, int(k))
        self.perf: Dict[ShapeKey, Dict[int, float]] = {}
        for shape, lat in seeds.items():
            good = {int(s): float(u) for s, u in lat.items() if u and u > 0}
            if not good:
                continue
            best = min(good.values())
            self.perf[shape] = {s: best / u for s, u in good.items()}
        self.by_cell: Dict[str, List[ShapeKey]] = {}
        for shape in sorted(self.perf):
            self.by_cell.setdefault(shape_cell(shape), []).append(shape)

    @staticmethod
    def _distance(a: ShapeKey, b: ShapeKey) -> float:
        return math.sqrt(
            sum(
                (math.log2(max(x, 1)) - math.log2(max(y, 1))) ** 2 for x, y in zip(a, b)
            )
        )

    def neighbours(self, shape: ShapeKey) -> List[ShapeKey]:
        pool = self.by_cell.get(shape_cell(shape)) or sorted(self.perf)
        return sorted(pool, key=lambda s: (self._distance(shape, s), s))[: self.k]

    def rank(self, shape: ShapeKey) -> List[int]:
        near = self.neighbours(shape)
        if not near:
            return []
        weights = [1.0 / (self._distance(shape, s) + 0.25) for s in near]
        total = sum(weights)
        score: Dict[int, float] = {}
        for s, w in zip(near, weights):
            for sid, p in self.perf[s].items():
                score[sid] = score.get(sid, 0.0) + w * p / total
        return [sid for sid, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]


class ModelRanker(Ranker):
    """Ranking by a model file (MLREC bytes) through the tilewright engine,
    over every catalog kernel, in `context` (a `features.ExecutionContext`;
    None: exclusive): the cascade's survivors best-first, then every other
    kernel the model's serving rules admit, best-first. Falls back to
    `fallback` for shapes the model scores nothing for. `rules` are the
    soft rules the model serves with (`Feasibility(rules=...)`)."""

    name = "model"

    def __init__(
        self,
        data: bytes,
        catalog: Catalog,
        ptype: ProblemType,
        hardware: Any,
        *,
        fallback: Optional[Ranker] = None,
        tw: Any = None,
        context: Optional[fs.ExecutionContext] = None,
    ) -> None:
        from . import evaluate as ev
        from . import mlrec

        self.ev = ev
        self.tw = tw or ev.tilewright_module()
        header = mlrec.read_header(data)
        self.rules = soft_rules(header.get("serving") or {}, int(header["version"]))
        self.model = ev.load_model_bytes(data, self.tw)
        self.pool = ev.pool_from_kernels(
            catalog.kernels[s] for s in catalog.pool_ids + catalog.bank_ids
        )
        self.cs = ev.candidate_set(self.tw, self.model, self.pool, hardware)
        self.hw = ev.make_hardware(self.tw, hardware)
        self.ptype = ptype
        self.fallback = fallback
        self.context = ev.make_context(self.tw, context or fs.ExecutionContext())

    def rank(self, shape: ShapeKey) -> List[int]:
        prob = fs.problem_kwargs_from_row(self.ptype.row(shape))
        pobj = self.ev.make_problem(self.tw, prob)
        out = [
            int(self.pool[int(r.config_index)]["sol_idx_global"])
            for r in self.cs.rank(pobj, self.hw, len(self.pool), self.context)
            if r.scored
        ]
        if not out and self.fallback is not None:
            return self.fallback.rank(shape)
        return out


def seed_latencies(csv_dir: str) -> Dict[ShapeKey, Dict[int, float]]:
    """{shape: {solution index: fastest us}} of the enriched chunk CSVs in
    `csv_dir`. Tested rows count with their timed latency, skip rows with
    their warm-up latency (an upper bound on how slow they are)."""
    out: Dict[ShapeKey, Dict[int, float]] = {}
    for path in sorted(glob.glob(os.path.join(csv_dir, "chunk_*.csv"))):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    shape = (
                        int(row["m"]),
                        int(row["n"]),
                        int(row["k"]),
                        int(row.get("batch_count") or 1),
                    )
                    sid = int(row["sol_idx_global"])
                    us = float(row["us"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not (math.isfinite(us) and us > 0):
                    continue
                lat = out.setdefault(shape, {})
                if sid not in lat or us < lat[sid]:
                    lat[sid] = us
    return out


# ── planning ─────────────────────────────────────────────────────────────────


@dataclass
class ShapePlan:
    shape: ShapeKey
    role: str  # "seed" | "planned" | "exhaustive"
    kernels: Optional[List[int]] = None  # None: every supported kernel
    probe: List[int] = field(default_factory=list)
    n_feasible: int = 0
    n_truncated: int = 0
    sources: Dict[str, int] = field(default_factory=dict)

    @property
    def cell(self) -> str:
        return shape_cell(self.shape)

    def to_json(self) -> Dict[str, Any]:
        m, n, k, b = self.shape
        return {
            "m": m,
            "n": n,
            "k": k,
            "batch": b,
            "cell": self.cell,
            "role": self.role,
            "kernels": self.kernels,
            "probe": self.probe,
            "n_feasible": self.n_feasible,
            "n_truncated": self.n_truncated,
            "sources": self.sources,
        }


def select_seeds(
    shapes: Sequence[ShapeKey], per_cell: int, seed: int
) -> List[ShapeKey]:
    """Up to `per_cell` distinct shapes of each cell, drawn at random per
    cell; returned in input order."""
    by_cell: Dict[str, List[ShapeKey]] = {}
    for s in dict.fromkeys(shapes):
        by_cell.setdefault(shape_cell(s), []).append(s)
    chosen: Set[ShapeKey] = set()
    for cell, members in sorted(by_cell.items()):
        n = min(max(0, int(per_cell)), len(members))
        chosen.update(stable_rng(seed, "seed", cell).sample(members, n))
    return [s for s in dict.fromkeys(shapes) if s in chosen]


def _probe(
    shape: ShapeKey, ordered: Sequence[int], first: Optional[int], p: PlanParams
) -> List[int]:
    """Calibration kernels: the ranker's top kernel when there is one, then
    random picks from `ordered`."""
    out: List[int] = [first] if first is not None and first in ordered else []
    rest = [s for s in ordered if s not in out]
    n = max(0, int(p.probe_kernels) - len(out))
    out += stable_rng(p.seed, "probe", *shape).sample(rest, min(n, len(rest)))
    return out


def plan_exhaustive(
    shape: ShapeKey, feas: Feasibility, p: PlanParams, role: str = "exhaustive"
) -> ShapePlan:
    ok = feas.feasible(shape, feas.catalog.pool_ids)
    return ShapePlan(
        shape=shape,
        role=role,
        kernels=None,
        probe=_probe(shape, ok, None, p),
        n_feasible=len(ok),
    )


def plan_shape(
    shape: ShapeKey, ranked: Sequence[int], feas: Feasibility, p: PlanParams
) -> ShapePlan:
    """Kernel list of one shape before coverage quotas (steps 1-3 of the
    module docstring)."""
    cat = feas.catalog
    pool_ok = feas.feasible(shape, cat.pool_ids)
    pool_ok_set = set(pool_ok)
    ranked_ok = [s for s in dict.fromkeys(ranked) if s in cat.kernels]
    ranked_ok = feas.feasible(shape, ranked_ok)
    chosen: Dict[int, None] = {}
    sources = {"top_k": 0, "family": 0, "random": 0, "coverage": 0}
    truncated = 0

    def add(sid: int, src: str) -> None:
        nonlocal truncated
        if sid in chosen:
            return
        if len(chosen) >= p.max_list:
            truncated += 1
            return
        chosen[sid] = None
        sources[src] += 1

    for sid in ranked_ok[: max(0, p.top_k)]:
        add(sid, "top_k")
    fams: List[Tuple[int, ...]] = []
    for sid in ranked_ok:
        if len(fams) >= max(0, p.families_top_f):
            break
        fam = family_of(cat.kernels[sid])
        if fam not in fams:
            fams.append(fam)
    for fam in fams:
        variants = feas.feasible(shape, cat.families.get(fam, []))
        if p.family_variants > 0:
            variants = variants[: p.family_variants]
        for sid in variants:
            add(sid, "family")
    rest = [s for s in pool_ok if s not in chosen]
    n_rand = min(max(0, p.explore_random_k), len(rest))
    for sid in stable_rng(p.seed, "explore", *shape).sample(rest, n_rand):
        add(sid, "random")
    kernels = list(chosen)
    return ShapePlan(
        shape=shape,
        role="planned",
        kernels=kernels,
        probe=_probe(shape, kernels, ranked_ok[0] if ranked_ok else None, p),
        n_feasible=len(pool_ok_set),
        n_truncated=truncated,
        sources=sources,
    )


def apply_coverage(
    plans: Sequence[ShapePlan], feas: Feasibility, p: PlanParams
) -> Dict[str, int]:
    """Add pool kernels to planned shapes until every kernel is benched on at
    least `min_coverage` shapes of each cell where it is feasible (or on
    every such shape when the cell has fewer). Seed and exhaustive shapes
    count for every kernel feasible there. Returns {cell: kernels added}."""
    added: Dict[str, int] = {}
    if p.min_coverage <= 0:
        return added
    by_cell: Dict[str, List[ShapePlan]] = {}
    for sp in plans:
        by_cell.setdefault(sp.cell, []).append(sp)
    pool = feas.catalog.pool_ids
    for cell, members in sorted(by_cell.items()):
        feasible_on: Dict[int, List[ShapePlan]] = {sid: [] for sid in pool}
        count: Dict[int, int] = {sid: 0 for sid in pool}
        for sp in members:
            ok = feas.feasible(sp.shape, pool)
            listed = None if sp.kernels is None else set(sp.kernels)
            for sid in ok:
                feasible_on[sid].append(sp)
                if listed is None or sid in listed:
                    count[sid] += 1
        n_added = 0
        for sid in pool:
            need = min(p.min_coverage, len(feasible_on[sid])) - count[sid]
            if need <= 0:
                continue
            open_shapes = [
                sp
                for sp in feasible_on[sid]
                if sp.kernels is not None
                and sid not in sp.kernels
                and len(sp.kernels) < p.max_list
            ]
            rng = stable_rng(p.seed, "coverage", cell, sid)
            for sp in rng.sample(open_shapes, min(need, len(open_shapes))):
                assert sp.kernels is not None
                sp.kernels.append(sid)
                sp.sources["coverage"] = sp.sources.get("coverage", 0) + 1
                n_added += 1
        added[cell] = n_added
    return added


def build_plan(
    shapes: Sequence[ShapeKey],
    feas: Feasibility,
    p: PlanParams,
    *,
    ranker: Optional[Ranker],
    seeds: Iterable[ShapeKey] = (),
    exhaustive: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[List[ShapePlan], Dict[str, int]]:
    """Plan of every distinct shape: seeds and, with `exhaustive` or without a
    ranker, every shape are benched exhaustively. Returns the plans in input
    order and the coverage additions per cell."""
    seed_set = set(seeds)
    plans: List[ShapePlan] = []
    uniq = list(dict.fromkeys(shapes))
    for i, shape in enumerate(uniq):
        if shape in seed_set:
            plans.append(plan_exhaustive(shape, feas, p, role="seed"))
        elif exhaustive or ranker is None:
            plans.append(plan_exhaustive(shape, feas, p))
        else:
            plans.append(plan_shape(shape, ranker.rank(shape), feas, p))
        if progress is not None:
            progress(i + 1, len(uniq))
    added = {} if exhaustive or ranker is None else apply_coverage(plans, feas, p)
    return plans, added


def plan_document(
    plans: Sequence[ShapePlan], meta: Mapping[str, Any]
) -> Dict[str, Any]:
    return {"version": PLAN_VERSION, **meta, "shapes": [sp.to_json() for sp in plans]}


def plan_lookup(doc: Mapping[str, Any]) -> Dict[ShapeKey, Dict[str, Any]]:
    """{(m, n, k, batch): shape entry} of a plan document."""
    out: Dict[ShapeKey, Dict[str, Any]] = {}
    for e in doc.get("shapes") or []:
        out[(int(e["m"]), int(e["n"]), int(e["k"]), int(e.get("batch", 1)))] = e
    return out
