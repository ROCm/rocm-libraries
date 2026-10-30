# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Benchmark planning: which kernels to time for each shape of a round.

The library heuristic is never consulted. A shape is either benched
exhaustively (every kernel the bench supports, in library order) or with an
explicit kernel list built from a ranker:

  1. the ranker's `top_k` kernels,
  2. every variant of the `families_top_f` best-ranked families (a family is
     the kernel signature: macro tile M, N, depthU, matrix instruction M, N,
     K and the A / B cache hints, 0 for an operand the tensor data mover
     loads), at most `family_variants` per family (0: all),
  3. `explore_random_k` kernels drawn at random from the rest,
  4. per cell, kernels benched on fewer than `min_coverage` shapes of the
     cell are added to shapes that have room, until each reaches the quota,
  5. with `family_coverage` (the run's models serve the family-wise
     non-temporal rule), every kernel only that rule admits for the shape:
     within its region, the member it keeps of each family the
     non-temporal rule rejects whole, past `max_list`. A model trained
     without measurements of those members cannot rank them.

Only kernels that pass the engine's gates for the shape (LDS fit and the
soft rules of `lib.feasibility` the run's models serve with) are planned,
and of an alias group (lib/kernel_identity.py) only its first member in
pool order, the one the runtime serves (`Catalog`). Lists hold at most
`max_list` kernels (`MAX_BENCH_LIST`, the bench's limit for one problem, at
most).

Rankers order the candidate kernels of one shape:
  * NeighborsRanker: normalised latencies (winner_us / us) of exhaustively
    benched seed shapes, averaged over the nearest seeds of the same cell in
    log2 (M, N, K, batch) space,
  * ModelRanker: a trained model's ranking through the tilewright engine.

`seed_recall` scores the lists a NeighborsRanker plans on the seeds
themselves: it ranks each seed from the other seeds of its cell and compares
the list built from that ranking with the seed's own exhaustive bench.

All randomness comes from `random.Random` seeded with the plan seed and the
shape or cell, so a plan is reproducible and does not depend on the order of
unrelated shapes. A plan keeps each list in the order above; the bench line
lists it in the order of `shuffled` (stage02, `bench.kernel_order`).
"""
from __future__ import annotations

import csv
import glob
import hashlib
import math
import os
import random
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Collection, Dict, Iterable, List, Mapping, Optional
from typing import Sequence, Set, Tuple

from . import features as fs
from . import kernel_identity as ki
from .feasibility import (
    in_family_region,
    non_temporal_availability,
    non_temporal_families,
    passes_gates,
    soft_rules,
)
from .grid import cell_key

ShapeKey = Tuple[int, int, int, int]
SOFT_RULES = (
    "non_temporal_rule",
    "non_temporal_family_rule",
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
# `seed_recall`: a kernel within this fraction of a seed's fastest latency
# counts as near its winner, and the top_k values it scores besides the
# plan's own (those below the longest list a seed can get).
NEAR_TOLERANCE = 0.05
RECALL_TOP_KS = (16, 32, 48, 64, 96, 128, 192, 256)
# `seed_recall`'s default warning threshold for the lists' ceiling
# (stage02: `planner.recall_min_ceiling`).
RECALL_MIN_CEILING = 0.97


@dataclass(frozen=True)
class PlanParams:
    top_k: int = 32
    families_top_f: int = 8
    family_variants: int = 0
    explore_random_k: int = 32
    min_coverage: int = 2
    max_list: int = 512
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


def shuffled(ids: Sequence[int], seed: int, *key: Any) -> List[int]:
    """`ids` in the order a bench line lists them: shuffled with `seed` and
    `key` (what the line benches, e.g. its shape)."""
    out = list(ids)
    stable_rng(seed, "order", *key).shuffle(out)
    return out


# ── kernel catalog ───────────────────────────────────────────────────────────


@dataclass
class Catalog:
    """Plannable kernels: the pool (the deployed library, in pool order) and
    optional bank kernels that only enter lists as family variants. A
    solution index appears once; the pool wins. Of an alias group
    (lib/kernel_identity.py) only the first member in that order is
    plannable; `aliases` maps the others to it."""

    kernels: Dict[int, Dict[str, Any]]
    pool_ids: List[int]
    bank_ids: List[int]
    families: Dict[Tuple[int, ...], List[int]]
    nt_available: Tuple[bool, bool]
    aliases: Dict[int, int] = field(default_factory=dict)

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
        aliases = ki.alias_map(kernels[s] for s in pool_ids + bank_ids)
        pool_ids = [s for s in pool_ids if s not in aliases]
        bank_ids = [s for s in bank_ids if s not in aliases]
        for sid in aliases:
            del kernels[sid]
        families: Dict[Tuple[int, ...], List[int]] = {}
        for sid in pool_ids + bank_ids:
            families.setdefault(family_of(kernels[sid]), []).append(sid)
        return cls(
            kernels=kernels,
            pool_ids=pool_ids,
            bank_ids=bank_ids,
            families=families,
            nt_available=non_temporal_availability(kernels[s] for s in pool_ids),
            aliases=aliases,
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
        self._families = (
            non_temporal_families(
                self._configs[s] for s in catalog.pool_ids + catalog.bank_ids
            )
            if self.rules.get("non_temporal_family_rule")
            else None
        )

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
            nt_families=self._families,
            **self.rules,
        )

    def feasible(self, shape: ShapeKey, ids: Iterable[int]) -> List[int]:
        problem = fs.problem_kwargs_from_row(self.ptype.row(shape))
        return [sid for sid in ids if self._ok(problem, sid)]

    def family_rule_only(self, shape: ShapeKey) -> List[int]:
        """Catalog kernels (pool, then bank) that pass for `shape` only
        through the family-wise non-temporal rule: the members it keeps of
        the families the non-temporal rule rejects whole. Empty without
        that rule or outside its region."""
        if self._families is None or not in_family_region(*shape):
            return []
        problem = fs.problem_kwargs_from_row(self.ptype.row(shape))
        plain = dict(self.rules, non_temporal_family_rule=False)
        nt_a, nt_b = self.catalog.nt_available
        return [
            sid
            for sid in self.catalog.pool_ids + self.catalog.bank_ids
            if self._ok(problem, sid)
            and not passes_gates(
                problem,
                self._configs[sid],
                lds_bytes=self.lds_bytes,
                nt_a_available=nt_a,
                nt_b_available=nt_b,
                **plain,
            )
        ]


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

    def neighbours(
        self, shape: ShapeKey, exclude: Collection[ShapeKey] = ()
    ) -> List[ShapeKey]:
        """The `k` seeds nearest `shape` of its cell, or of every cell when
        its cell has none; never those of `exclude`."""
        skip = set(exclude)
        pool = [s for s in self.by_cell.get(shape_cell(shape), ()) if s not in skip]
        if not pool:
            pool = [s for s in sorted(self.perf) if s not in skip]
        return sorted(pool, key=lambda s: (self._distance(shape, s), s))[: self.k]

    def rank(self, shape: ShapeKey, exclude: Collection[ShapeKey] = ()) -> List[int]:
        """Solution indices best first, from the neighbours of `shape`
        without the seeds of `exclude` (`seed_recall` leaves a seed out of
        its own ranking)."""
        near = self.neighbours(shape, exclude)
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
    `fallback` for shapes the model scores nothing for. `rules` (soft rules
    as `Feasibility` takes them) override the model's serving rules, which
    only an MLREC_v3 model allows; `self.rules` are the ones it ranks with.
    A model ranks kernels its own rules exclude without having trained on
    them for such shapes. An MLREC_v3 model ranks without its alignment
    veto, a serving rule no plan applies."""

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
        rules: Optional[Mapping[str, bool]] = None,
    ) -> None:
        from . import evaluate as ev
        from . import mlrec

        self.ev = ev
        self.tw = tw or ev.tilewright_module()
        header = mlrec.read_header(data)
        version = int(header["version"])
        if version == 3:
            data = mlrec.without_alignment_veto(data)
            header = mlrec.read_header(data)
        own = soft_rules(header.get("serving") or {}, version)
        if rules is not None and dict(rules) != own:
            if version != 3:
                raise ValueError(
                    f"an MLREC_v{version} model serves with {own}; only v3 "
                    f"models rank with other soft rules"
                )
            family = bool(rules.get("non_temporal_family_rule"))
            data = mlrec.with_soft_rules(
                data, {**rules, "non_temporal_family_rule": family}
            )
            header = mlrec.read_header(data)
        self.rules = soft_rules(header.get("serving") or {}, version)
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
    n_feasible: int = 0
    n_truncated: int = 0
    sources: Dict[str, int] = field(default_factory=dict)
    n_family_over_limit: int = 0

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
            "n_feasible": self.n_feasible,
            "n_truncated": self.n_truncated,
            "n_family_over_limit": self.n_family_over_limit,
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


def plan_exhaustive(
    shape: ShapeKey, feas: Feasibility, role: str = "exhaustive"
) -> ShapePlan:
    ok = feas.feasible(shape, feas.catalog.pool_ids)
    return ShapePlan(shape=shape, role=role, kernels=None, n_feasible=len(ok))


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
    return ShapePlan(
        shape=shape,
        role="planned",
        kernels=list(chosen),
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


def add_family_rule_members(plans: Sequence[ShapePlan], coverage: Feasibility) -> None:
    """Append to every listed plan the kernels `coverage` admits for its
    shape only through the family-wise non-temporal rule
    (`Feasibility.family_rule_only`) that it does not list yet, beyond
    `max_list` up to the bench's limit (source `family_rule`); those past
    that limit are counted in `n_family_over_limit`."""
    for sp in plans:
        if sp.kernels is None:
            continue
        listed = set(sp.kernels)
        new = [s for s in coverage.family_rule_only(sp.shape) if s not in listed]
        if not new:
            continue
        room = max(0, MAX_BENCH_LIST - len(sp.kernels))
        sp.kernels.extend(new[:room])
        sp.n_family_over_limit += max(0, len(new) - room)
        sp.sources["family_rule"] = min(len(new), room)


def build_plan(
    shapes: Sequence[ShapeKey],
    feas: Feasibility,
    p: PlanParams,
    *,
    ranker: Optional[Ranker],
    seeds: Iterable[ShapeKey] = (),
    exhaustive: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
    family_coverage: Optional[Feasibility] = None,
) -> Tuple[List[ShapePlan], Dict[str, int]]:
    """Plan of every distinct shape: seeds and, with `exhaustive` or without a
    ranker, every shape are benched exhaustively. With `family_coverage` (a
    Feasibility with the family-wise non-temporal rule) every listed shape
    also gets the kernels only that rule admits (`add_family_rule_members`).
    Returns the plans in input order and the coverage additions per cell."""
    seed_set = set(seeds)
    plans: List[ShapePlan] = []
    uniq = list(dict.fromkeys(shapes))
    for i, shape in enumerate(uniq):
        if shape in seed_set:
            plans.append(plan_exhaustive(shape, feas, role="seed"))
        elif exhaustive or ranker is None:
            plans.append(plan_exhaustive(shape, feas))
        else:
            plans.append(plan_shape(shape, ranker.rank(shape), feas, p))
        if progress is not None:
            progress(i + 1, len(uniq))
    added = {} if exhaustive or ranker is None else apply_coverage(plans, feas, p)
    if family_coverage is not None:
        add_family_rule_members(plans, family_coverage)
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


# ── seed recall ──────────────────────────────────────────────────────────────


class _OneShape:
    """The gates of one shape as `plan_shape` asks them, the catalog kernels
    that pass found once: `seed_recall` plans each seed at several top_k."""

    def __init__(self, feas: Feasibility, shape: ShapeKey) -> None:
        self.catalog = feas.catalog
        self.shape = shape
        ids = feas.catalog.pool_ids + feas.catalog.bank_ids
        self.passing = frozenset(feas.feasible(shape, ids))

    def feasible(self, shape: ShapeKey, ids: Iterable[int]) -> List[int]:
        if shape != self.shape:
            raise ValueError(f"the gates of {self.shape} asked for {shape}")
        return [sid for sid in ids if sid in self.passing]


def _recall_scores(rows: Sequence[Tuple[bool, bool, float]]) -> Dict[str, Any]:
    """`seed_recall`'s scores of (winner listed, a near kernel listed,
    ceiling) rows, one per seed."""
    n = len(rows)
    found = [c for _, _, c in rows if c > 0]
    return {
        "n_seeds": n,
        "winner_recall": sum(1 for w, _, _ in rows if w) / n if n else None,
        "near_recall": sum(1 for _, x, _ in rows if x) / n if n else None,
        "ceiling": (
            math.exp(sum(math.log(c) for c in found) / len(found)) if found else None
        ),
        "n_unmeasured": n - len(found),
    }


def seed_recall(
    ranker: NeighborsRanker,
    feas: Feasibility,
    p: PlanParams,
    *,
    cells: Optional[Collection[str]] = None,
    top_ks: Iterable[int] = RECALL_TOP_KS,
    min_ceiling: float = RECALL_MIN_CEILING,
    near: float = NEAR_TOLERANCE,
) -> Dict[str, Any]:
    """Leave-one-out scores of the lists `ranker` plans, on its own seeds.

    Each seed with another measured seed in its cell (of `cells`; None:
    every cell) is ranked from the other seeds of its cell, as `ranker`
    ranks the cell's planned shapes but from one seed fewer, and gets the
    list `plan_shape` builds from that ranking with `p`: its top_k, family
    and random kernels, not the cell's coverage additions. Against the
    seed's own exhaustive bench, among the kernels `feas` lets the plan
    list for it:

      winner_recall  share of the seeds whose fastest kernel is listed,
      near_recall    share with a listed kernel within `near` of it,
      ceiling        geometric mean of the fastest kernel's latency over the
                     fastest listed one's: the selection efficiency a
                     perfect pick from the lists reaches. Over the seeds
                     whose list holds a kernel their bench measured; the
                     others (`n_unmeasured`) count as misses above.

    The scores are given at `p.top_k`, overall and per cell (`cells`), and
    in `by_top_k` at the plan's top_k, at each of `top_ks` below the longest
    list a scored seed can get (as many kernels as pass its gates, at most
    `p.max_list`) and at that length. `top_k_reaching` is the smallest of
    those from which on every ceiling is at least `min_ceiling` (None when
    the longest list's is not; random kernels can move a ceiling either way
    between two top_k). `n_alone` counts the seeds left out for want of
    another measured seed in their cell, `n_unscored` those whose bench
    measured no kernel the plan may list."""
    scored = []
    n_alone = n_unscored = 0
    for shape in sorted(ranker.perf):
        cell = shape_cell(shape)
        if cells is not None and cell not in cells:
            continue
        if len(ranker.by_cell.get(cell, ())) < 2:
            n_alone += 1
            continue
        gates = _OneShape(feas, shape)
        perf = {s: r for s, r in ranker.perf[shape].items() if s in gates.passing}
        if not perf:
            n_unscored += 1
            continue
        ranked = ranker.rank(shape, exclude=(shape,))
        scored.append((shape, cell, gates, perf, max(perf.values()), ranked))
    passing = max((len(gates.passing) for _, _, gates, *_ in scored), default=0)
    longest = min(passing, int(p.max_list))
    top_k = int(p.top_k)
    ks = sorted({top_k, *(int(k) for k in top_ks if int(k) < longest)})
    if longest > top_k:
        ks = sorted({*ks, longest})
    rows: Dict[int, List[Tuple[bool, bool, float]]] = {k: [] for k in ks}
    by_cell: Dict[str, List[Tuple[bool, bool, float]]] = {}
    for shape, cell, gates, perf, best, ranked in scored:
        for k in ks:
            listed = plan_shape(shape, ranked, gates, replace(p, top_k=k)).kernels
            got = max((perf[s] for s in listed or () if s in perf), default=0.0)
            row = (got >= best, got * (1.0 + near) >= best, got / best)
            rows[k].append(row)
            if k == top_k:
                by_cell.setdefault(cell, []).append(row)
    by_top_k = [{"top_k": k, **_recall_scores(rows[k])} for k in ks]
    reaching = None
    for e in reversed(by_top_k):
        if e["ceiling"] is None or e["ceiling"] < min_ceiling:
            break
        reaching = e["top_k"]
    return {
        "ranker": ranker.name,
        "neighbors_k": ranker.k,
        "top_k": top_k,
        "near_tolerance": near,
        "min_ceiling": min_ceiling,
        **_recall_scores(rows[top_k]),
        "n_cells": len(by_cell),
        "n_alone": n_alone,
        "n_unscored": n_unscored,
        "top_k_reaching": reaching,
        "by_top_k": by_top_k,
        "cells": {c: _recall_scores(r) for c, r in sorted(by_cell.items())},
    }
