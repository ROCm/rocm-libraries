# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Bench-level analysis of a run for the report: where the shipped rules and
model lose time against the fastest measured kernel, which kernels and kernel
parameters win and get picked, and which kernels of the pool never do.

Everything is relative to the oracle, the fastest timed solution of each
GEMM, and uses only tilewright's own selection steps. For one GEMM the levels
are the cheapest cost among

  oracle   every measured solution;
  pool     the kernels of the serving library pool (rows of other libraries,
           such as planner banks, can win but cannot be picked);
  hard     the kernels that pass the LDS gate, the leading-free rule (a
           contiguous free dimension holds one global-read vector:
           TensileLite's predicate, so a kernel failing it cannot run,
           whether the engine applies the rule, as MLREC_v2 does, or leaves
           it to the integration, as MLREC_v3 does) and the small-batched
           (small batched problems fit one tile) and Dot2 (only for M < 3)
           rules when the bundle serves with them (`serving_rules`: always
           for MLREC_v2, by its serving flags for MLREC_v3);
  soft     those that also pass the non-temporal rule, when the bundle serves
           with it (else the same as `hard`);
  tier     those of tier 1: the kernels the engine admits under the bundle's
           rules that are on the serving cell's whitelist, or every one it
           admits when the whitelist keeps none of the pool, less those that
           cannot run;
  scored   those the engine scores in the GEMM's execution context (a cascade
           model scores only the best view keys of a tier without a
           whitelist);
  pick     the deployed pick: the first scored kernel in rank order the
           runtime would run.

A kernel's cost is `lib.evaluate.pick_cost` under the run's
`validate.pick_measurement`, as in stages 4.5, 6, 7 and 8: a timed row costs
its latency; a skip row (screened, one warm-up run) costs its warm-up
latency under `any` and `warmup` and is a miss at the slowest latency the
GEMM measured under `timed`. The pick is the kernel the runtime serves
(`resolve_pick`): the first scored kernel that runs. A kernel without a row
and on the GEMM's planned kernel list (stage02 plan.json), or without a
list, is one the bench did not run, so the runtime does not either; one
that fails the leading-free rule cannot run (`leading_free_mask`); the
runtime passes over both. A kernel off the list that passes the rule may
run but was never measured: under `any` it is passed over too, under the
other measurements it is the pick, a miss (`not_benched`) whose cost is
unknown. The pick level then holds the next-ranked kernel that has a reading
the measurement accepts (the slowest latency the GEMM measured when none
does), and `decomposition` bounds the selection efficiency between that
pick at the slowest latency the GEMM measured and at the winner's
(`sel_eff_range`); levels before it charge such a kernel the slowest
latency.

Each level is clamped to be no faster than the previous one and no slower
than the pick, so the per-step losses log(level / previous level) add up to
-log(selection efficiency) and the pick is charged its own cost: a GEMM
whose engine scores a kernel the levels before `scored` leave out (the
report's rules disagree with the engine's; `n_inconsistent` counts them)
loses no more than its pick costs. A GEMM whose pick does not exist (no
serving cell, nothing feasible, or no scored kernel runs) is counted as
unserved; its loss is not decomposed, and `deployed_sel_eff` counts it with
the runtime's fallback pick when that was measured, as the stages do. `scored` and `pick` need the tilewright module
and the model file.

A kernel is its 8-field signature (SIG_FIELDS: macro tile M / N, depthU,
matrix instruction M / N / K, cache hints A / B); the solutions of the
library that share one are its members. A GEMM's measurement of a solution
merges its rows of that solution index as stage05 merges them
(`lib/repeats.py`: the run's `train.repeat_aggregation`, sustained rows
first, the GEMM's other kernels scaled to them).

Solution indices name kernels of one hipBLASLt build only
(lib/kernel_identity.py). With the library's identity (`library_identity`)
every bench folder and plan is read with its indices mapped to the
library's, as stage05 reads them; without it, folders whose records name
different kernels under one index are never merged, nor is a folder without
a record whose rows or attributes name other kernels than the records do
under the same indices (`load_bench` raises BuildChangedError, and the
report says why the analysis is missing).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional
from typing import Sequence, Set, Tuple, Union

import numpy as np

from . import evaluate as ev
from . import feasibility
from . import features as fs
from . import kernel_identity as ki
from . import mlrec
from . import repeats
from . import run_bundles
from . import run_layout
from .bundle import bundle_file
from .cell_evidence import round_dirs
from .contexts import BenchContext
from .correctness import config_blocklist
from .dat import KERNELS_FILE, kernel_attributes_of_doc, read_kernel_attributes
from .grid import cell_key
from .hardware import DeviceHardware
from .subcells import SplitRule, assign_subcell, load_cumulative_split_tree
from .subcells import resolve_model_cell

LEVELS = ("oracle", "pool", "hard", "soft", "tier", "scored", "pick")
STEPS = ("library", "hard_rules", "soft_rules", "whitelist", "cascade", "ranking")
ENGINE_STEPS = ("cascade", "ranking")
STEP_TITLES = {
    "library": "library pool",
    "hard_rules": "hard validity rules",
    "soft_rules": "non-temporal rule",
    "whitelist": "tier 1 (whitelist)",
    "cascade": "cascade",
    "ranking": "ranking",
}
SIG_FIELDS = fs.SIGNATURE_FIELDS
NEAR = (0.01, 0.02, 0.05)
PROBLEM_FIELDS = (
    "m",
    "n",
    "k",
    "batch_count",
    "transA",
    "transB",
    "a_type",
    "b_type",
    "c_type",
    "d_type",
    "compute_type",
)
CONFIG_PICK_MEASUREMENT = "warmup"
# Must equal run_orchestrator.STAGE_MARKER.
STAGE_MARKER = ".stage_complete.json"

Key = Tuple[Any, ...]
ShapeKey = Tuple[int, int, int, int]
PlanLookup = Mapping[ShapeKey, Optional[FrozenSet[int]]]
# (us, skip row, rank, runtime fallback flag or None, sustained row)
Row = Tuple[float, bool, int, Optional[bool], bool]
# One solution's row, or its rows when the data repeat it.
Rows = Union[Row, List[Row]]


def signature_of(kernel: Mapping[str, Any]) -> Tuple[int, ...]:
    return tuple(int(kernel[f]) for f in SIG_FIELDS)


def signature_label(sig: Sequence[int]) -> str:
    """How the report names a kernel: its signature, e.g. `MT256x256x64
    MI16x16x32 NTA0 NTB4` (macro tile M x N x depthU, matrix instruction,
    cache hints A and B)."""
    mt_m, mt_n, du, mi_m, mi_n, mi_k, cha, chb = sig
    return f"MT{mt_m}x{mt_n}x{du} MI{mi_m}x{mi_n}x{mi_k} NTA{cha} NTB{chb}"


def context_label(ctx: fs.ExecutionContext) -> str:
    if ctx.is_exclusive():
        return "exclusive"
    return BenchContext(cu_budget=ctx.cu_budget, schedule=ctx.schedule).label


# ── bench rows ───────────────────────────────────────────────────────────────


def chunk_files(dirs: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    for d in dirs:
        d = Path(d)
        if d.is_dir():
            out += sorted(
                p
                for p in d.iterdir()
                if p.name.startswith("chunk_") and p.name.endswith(".csv")
            )
    return out


def _has_chunks(d: Path) -> bool:
    return bool(chunk_files([d]))


def training_data_dir(rd: Path) -> Path:
    """The round's own rows its stage05 trained on: stage04b's train/ when
    it kept a gate slice out, else stage04."""
    train = rd / "stage04b" / "train"
    return train if train.is_dir() else rd / "stage04"


def gate_dir(rd: Path) -> Optional[Path]:
    gate = rd / "stage04b" / "gate"
    return gate if _has_chunks(gate) else None


def holdout_context_dir(rd: Path) -> Path:
    return rd / "stage03c" / "holdout" / "stage04"


def _plan_file(
    path: Path, identity: Optional[ki.LibraryIdentity]
) -> Tuple[
    Optional[Dict[str, Any]], Optional[Dict[ShapeKey, Optional[FrozenSet[int]]]]
]:
    """(the plan.json document, {shape: its kernel list, None when benched
    exhaustively}) of one stage02 plan, its indices those of `identity`'s
    library when given; (None, None) when it cannot be read."""
    try:
        doc = dict(json.loads(Path(path).read_text()), path=str(path))
    except (OSError, ValueError, TypeError):
        return None, None
    try:
        lookup = ev.remapped_plan_lookup(doc, identity, str(path))
    except (KeyError, TypeError, ValueError):
        return doc, None
    return doc, {
        shape: (
            frozenset(int(s) for s in e["kernels"])
            if e.get("kernels") is not None
            else None
        )
        for shape, e in lookup.items()
    }


def round_plan(
    rd: Path, identity: Optional[ki.LibraryIdentity] = None
) -> Optional[Dict[ShapeKey, Optional[FrozenSet[int]]]]:
    """{shape: its kernel list, None when benched exhaustively} of the
    round's plan.json (the planner's, lib/run_layout.py), its indices those
    of `identity`'s library when given; None without a plan."""
    try:
        path = run_layout.plan_dir(rd) / "plan.json"
    except ValueError:
        return None
    return _plan_file(path, identity)[1]


def round_context_plans(
    rd: Path, identity: Optional[ki.LibraryIdentity] = None
) -> Dict[Tuple[int, str], PlanLookup]:
    """{(cu_budget, schedule): `round_plan`-like lookup} of the round's
    execution-context re-plans (stage02's contexts/<label>/plan.json, each
    naming its context). stage03c benches a context's shape from its
    re-plan's list when the re-plan holds the shape, else from the round's
    own list."""
    try:
        root = run_layout.plan_dir(rd) / "contexts"
    except ValueError:
        return {}
    out: Dict[Tuple[int, str], PlanLookup] = {}
    for path in sorted(root.glob("*/plan.json")) if root.is_dir() else []:
        doc, lookup = _plan_file(path, identity)
        ctx = (doc or {}).get("context")
        if lookup is None or not isinstance(ctx, dict):
            continue
        try:
            key = (int(ctx["cu_budget"]), str(ctx.get("schedule") or "default"))
        except (KeyError, TypeError, ValueError):
            continue
        out[key] = lookup
    return out


def bench_dirs(run_dir: Path) -> List[Path]:
    """Every round's stage04 folder with enriched chunks."""
    return [rd / "stage04" for rd in round_dirs(run_dir) if _has_chunks(rd / "stage04")]


def _solution_id(sid: str, rank: str) -> Optional[int]:
    try:
        return int(sid)
    except (TypeError, ValueError):
        pass
    try:
        r = int(rank or -1)
    except ValueError:
        return None
    return -(r + 1) if r >= 0 else None


def _flag(v: Any) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes")


def parse_chunk(
    path: Path, remap: Optional[ki.Remap] = None
) -> Tuple[Dict[Key, Dict[int, Rows]], Dict[int, Dict[str, int]], Dict[str, int]]:
    """({GEMM key: {solution: its Row, or its Rows when repeated}},
    {solution: config kwargs}, counters) of one enriched chunk, with
    stage05's row rules. A GEMM key is the problem plus its execution
    context when that is not exclusive. With `remap`
    (`lib.kernel_identity.folder_remap` of its folder, which keeps what the
    rows show) each row's solution index is the library's, and rows it
    leaves out count as `rows_kernel_not_in_library`."""
    gemms: Dict[Key, Dict[int, Rows]] = {}
    configs: Dict[int, Dict[str, int]] = {}
    stats: Dict[str, int] = defaultdict(int)
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None) or []
        col = {name: i for i, name in enumerate(header)}
        missing = [c for c in PROBLEM_FIELDS + ("us",) if c not in col]
        if missing:
            raise ValueError(f"{path}: no column(s) {missing}")
        i_us, i_mt, i_mi = col["us"], col.get("mt_m"), col.get("mi_m")
        i_sid, i_rank, i_skip = (
            col.get("sol_idx_global"),
            col.get("rank"),
            col.get("is_skip"),
        )
        i_cu, i_sched = col.get("cu_budget"), col.get("schedule")
        i_ori = col.get("is_origami_pick")
        i_sus = col.get("is_sustained")
        i_key = [col[c] for c in PROBLEM_FIELDS]
        i_cfg = [(c, col[c]) for c in ev.CONFIG_FIELDS if c in col]
        for row in reader:
            try:
                us = float(row[i_us])
                mt_ok = i_mt is not None and int(row[i_mt] or 0) != 0
                mi_ok = i_mi is not None and int(row[i_mi] or 0) != 0
                key = tuple(row[i] for i in i_key)
            except (IndexError, ValueError):
                stats["bad_rows"] += 1
                continue
            if not (us > 0) or math.isnan(us) or not (mt_ok and mi_ok):
                stats["bad_rows"] += 1
                continue
            rank_s = row[i_rank] if i_rank is not None else ""
            sid = _solution_id(row[i_sid] if i_sid is not None else "", rank_s)
            if sid is None:
                stats["bad_rows"] += 1
                continue
            if remap is not None:
                fields = {c: row[i] for c, i in i_cfg} if remap.needs_row(sid) else {}
                sid = remap.row_index(sid, fields)
                if sid is None:
                    stats["rows_kernel_not_in_library"] += 1
                    continue
            try:
                key = (int(key[0]), int(key[1]), int(key[2]), int(key[3])) + key[4:]
                ctx = ev.context_of(
                    {
                        "cu_budget": row[i_cu] if i_cu is not None else "",
                        "schedule": row[i_sched] if i_sched is not None else "",
                    }
                )
                rank = int(rank_s or -1)
            except (IndexError, ValueError):
                stats["bad_rows"] += 1
                continue
            if not ctx.is_exclusive():
                key += (ctx.cu_budget, ctx.schedule)
            skip = i_skip is not None and row[i_skip] == "True"
            flag = _flag(row[i_ori]) if i_ori is not None else None
            sustained = i_sus is not None and row[i_sus] == "True"
            stats["rows_skip" if skip else "rows_tested"] += 1
            per = gemms.setdefault(key, {})
            per[sid] = _merge_row(per.get(sid), (us, skip, rank, flag, sustained))
            if sid >= 0 and sid not in configs:
                configs[sid] = fs.config_kwargs_from_row(dict(zip(header, row)))
    return gemms, configs, dict(stats)


# A load_bench worker's copies of the folder remaps, set once per worker
# process by the executor's initializer: their tables are larger than a
# chunk's rows.
_WORKER_REMAPS: List[Optional[ki.Remap]] = []


def _init_worker(remaps: Sequence[Optional[ki.Remap]]) -> None:
    global _WORKER_REMAPS
    _WORKER_REMAPS = list(remaps)


def _parse_folder_chunk(path: Path, folder: int) -> Tuple[
    Dict[Key, Dict[int, Rows]],
    Dict[int, Dict[str, int]],
    Dict[str, int],
    Optional[ki.Found],
]:
    """`parse_chunk` of a chunk of load_bench folder `folder` in a worker,
    and what its rows showed of the folder's kernels beyond the worker's
    earlier chunks (`ki.Found`)."""
    base = _WORKER_REMAPS[folder]
    if base is None:
        return (*parse_chunk(path), None)
    part = base.fresh()
    out = parse_chunk(path, part)
    found = part.found(base)
    base.absorb(found)
    return (*out, found)


def _merge_row(prev: Optional[Rows], new: Rows) -> Rows:
    """`new` joined to the rows of one solution so far."""
    if prev is None:
        return new
    return [
        *(prev if isinstance(prev, list) else [prev]),
        *(new if isinstance(new, list) else [new]),
    ]


def _labelled(per: Dict[int, Rows], aggregation: str) -> Dict[int, Row]:
    """One Row per solution of a GEMM, its rows merged as stage05 merges
    them (`lib.repeats.merge_candidates`)."""
    if not any(isinstance(r, list) or r[4] for r in per.values()):
        return per  # type: ignore[return-value]
    cands = [
        {
            "sol_idx_global": sid,
            "us": r[0],
            "is_skip": r[1],
            "rank": r[2],
            "is_origami_pick": r[3],
            "sustained": r[4],
        }
        for sid, rows in per.items()
        for r in (rows if isinstance(rows, list) else [rows])
    ]
    return {
        int(c["sol_idx_global"]): (
            float(c["us"]),
            bool(c["is_skip"]),
            int(c["rank"]),
            c["is_origami_pick"],
            bool(c["sustained"]),
        )
        for c in repeats.merge_candidates(cands, aggregation)
    }


@dataclass
class Gemm:
    key: Key
    leaf: str
    sids: np.ndarray
    us: np.ndarray
    skip: np.ndarray
    winner_us: float
    listed: Optional[FrozenSet[int]] = None
    fallback: Optional[int] = None

    @property
    def context(self) -> fs.ExecutionContext:
        extra = self.key[len(PROBLEM_FIELDS) :]
        if not extra:
            return fs.EXCLUSIVE
        return fs.ExecutionContext(cu_budget=int(extra[0]), schedule=str(extra[1]))

    @property
    def row(self) -> Dict[str, Any]:
        out = dict(zip(PROBLEM_FIELDS, self.key))
        ctx = self.context
        if not ctx.is_exclusive():
            out["cu_budget"], out["schedule"] = ctx.cu_budget, ctx.schedule
        return out


@dataclass
class Bench:
    gemms: List[Gemm]
    configs: Dict[int, Dict[str, Any]]
    stats: Dict[str, int]
    sources: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def _fallback(per: Mapping[int, Row]) -> Optional[int]:
    """The runtime's fallback pick (`lib.evaluate.origami_candidate`)."""
    flagged = [sid for sid, r in per.items() if r[3]]
    if flagged:
        return flagged[0]
    if any(r[3] is not None for r in per.values()):
        return None
    tested = [(r[2], sid) for sid, r in per.items() if not r[1] and r[2] >= 0]
    return min(tested)[1] if tested else None


def _listed(
    shape: ShapeKey,
    plans: Sequence[Optional[PlanLookup]],
    own: Optional[Sequence[Optional[PlanLookup]]] = None,
) -> Optional[FrozenSet[int]]:
    """The union of the kernel lists the plans of a GEMM's bench folders give
    its shape; None when any of them benched it exhaustively or has no plan.
    `own` (aligned with `plans`) holds each folder's re-plan of the GEMM's
    execution context, which takes the place of the folder's plan for the
    shapes it holds."""
    lists = []
    for j, lk in enumerate(plans):
        mine = own[j] if own is not None else None
        if mine is not None and shape in mine:
            lk = mine
        if lk is None or shape not in lk or lk[shape] is None:
            return None
        lists.append(lk[shape])
    return frozenset().union(*lists) if lists else None


def _recorded(d: Path) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[int, str]]]:
    """(kernels.json, {solution index: kernel id} of its record) of an
    enriched folder; None for what it lacks or cannot be read."""
    try:
        doc = ev.kernels_json(d)
    except ValueError:
        return None, None
    try:
        return doc, ki.recorded_ids(ki.read_record(doc))
    except ValueError:
        return doc, None


def same_build(
    dirs: Sequence[Path], recorded: Sequence[Optional[Mapping[int, str]]]
) -> Dict[int, Tuple[str, str]]:
    """{solution index: (kernel id, folder)} of the kernels the kernels.json
    records of `dirs` name (`recorded`, aligned with `dirs`: the `_recorded`
    ids of each); BuildChangedError when two of them name different kernels
    under one solution index: their rows come from different hipBLASLt
    builds and cannot be merged by index. Folders without a readable record
    are not compared here."""
    seen: Dict[int, Tuple[str, str]] = {}
    for d, ids in zip(dirs, recorded):
        for sid, kid in (ids or {}).items():
            prev = seen.setdefault(sid, (kid, str(d)))
            if prev[0] != kid:
                raise ki.BuildChangedError(
                    f"{prev[1]} and {d} name different kernels under solution "
                    f"{sid} ({prev[0]}, {kid}): they were enriched with different "
                    f"hipBLASLt builds, and without the library's kernel identities "
                    f"there is nothing to map them to"
                )
    return seen


def _attributes(doc: Optional[Mapping[str, Any]]) -> Dict[int, Dict[str, int]]:
    try:
        return kernel_attributes_of_doc(doc) if doc is not None else {}
    except ValueError:
        return {}


def folder_remaps(
    dirs: Sequence[Path], identity: Optional[ki.LibraryIdentity]
) -> List[Optional[ki.Remap]]:
    """`lib.kernel_identity.folder_remap` of each of `dirs`. Without
    `identity` the folders' records must agree (`same_build`), and a folder
    without a record is checked against the kernels they name
    (`ki.LegacyCheck.of_kernel_ids`: its rows by signature, its kernels.json
    by attributes), which `load_bench` refuses to merge it with when they
    differ; None for the other folders. An index no record names is not
    checked, so folders without a record are not compared with each
    other."""
    if identity is None:
        folders = [_recorded(d) for d in dirs]
        named = same_build(dirs, [ids for _, ids in folders])
        if not named:
            return [None] * len(dirs)
        then: Dict[int, Dict[str, int]] = {}
        for doc, recorded in folders:
            if recorded is not None:
                for sid, attrs in _attributes(doc).items():
                    then.setdefault(sid, attrs)
        ids = {sid: kid for sid, (kid, _) in named.items()}
        return [
            (
                None
                if recorded is not None
                else ki.Remap(
                    str(d),
                    "",
                    legacy=ki.LegacyCheck.of_kernel_ids(ids, then, _attributes(doc)),
                )
            )
            for d, (doc, recorded) in zip(dirs, folders)
        ]
    out: List[Optional[ki.Remap]] = []
    for d in dirs:
        try:
            doc = ev.kernels_json(d)
        except ValueError:
            doc = None
        out.append(ki.folder_remap(doc, identity, str(d)))
    return out


def _unlike_the_records(d: Path, r: ki.Remap) -> str:
    assert r.legacy is not None
    differ = r.legacy.differ
    s = min(differ)
    n = len(r.legacy.agree) + len(differ)
    return (
        f"{d} records no kernel identities, and {len(differ)} of the {n} kernels "
        f"it names differ from those the other folders' records name under the "
        f"same solution indices (e.g. solution {s}: {differ[s]} there, "
        f"{r.legacy.ids.get(s)} recorded): they were enriched with different "
        f"hipBLASLt builds, and without the library's kernel identities there is "
        f"nothing to map them to"
    )


def load_bench(
    dirs: Sequence[Path],
    *,
    tree: Optional[Mapping[str, SplitRule]] = None,
    jobs: int = 1,
    keep: Optional[Callable[[Mapping[str, Any]], bool]] = None,
    plans: Optional[Sequence[Optional[PlanLookup]]] = None,
    identity: Optional[ki.LibraryIdentity] = None,
    strict: bool = False,
    aggregation: str = repeats.DEFAULT_AGGREGATION,
    blocklist: Optional[Any] = None,
    pool_indices: Optional[Iterable[int]] = None,
    context_plans: Optional[
        Sequence[Optional[Mapping[Tuple[int, str], PlanLookup]]]
    ] = None,
) -> Bench:
    """GEMMs of the enriched chunks of `dirs` (their union; repeated
    measurements of a solution merged by `aggregation`, lib/repeats.py),
    routed to leaves with `tree`. `keep` filters GEMMs by their problem fields; `plans` (aligned
    with `dirs`, see `round_plan`) set each GEMM's planned kernel list, a
    GEMM of an execution context taking its shape's list from its
    context's re-plan of the folder (`context_plans`, aligned with `dirs`,
    see `round_context_plans`) when that holds the shape.
    With `identity` each folder's solution indices are the library's
    (`folder_remaps`; with `strict`, BuildChangedError unless every folder's
    indices still name their kernels) and a planned list holds the aliases
    of the alias groups it lists; without it, BuildChangedError when
    the folders record different builds, or a folder without a record names
    other kernels under their indices than the records do. Rows of kernels
    `blocklist` (`lib.correctness.Blocklist`) names, by their kernel ids in
    `identity`'s library or else in the folders' records, are left out
    (`rows_blocked`). With `pool_indices` (the solution indices of the
    kernel pool, in `identity`'s library when given) the rows of every other
    kernel are left out before a GEMM's rows merge, as stage05 leaves them
    out (`rows_outside_pool`: those of kernels not blocked, of
    `kernels_outside_pool` kernels; `gemms_outside_pool` counts the GEMMs
    left without a row)."""
    remaps = folder_remaps(dirs, identity)
    in_pool = None if pool_indices is None else frozenset(int(s) for s in pool_indices)
    blocked: Set[int] = set()
    if blocklist:
        if identity is not None:
            blocked = blocklist.blocked_indices(identity.ids)
        else:
            named: Dict[int, str] = {}
            for d in dirs:
                named.update(_recorded(Path(d))[1] or {})
            blocked = blocklist.blocked_indices(named)
    if strict:
        for r in remaps:
            if r is not None and r.legacy is None:
                r.require_same()
    files: List[Tuple[int, Path]] = [
        (i, p) for i, d in enumerate(dirs) for p in chunk_files([d])
    ]
    merged: Dict[Key, Dict[int, Rows]] = {}
    origin: Dict[Key, set] = defaultdict(set)
    configs: Dict[int, Dict[str, Any]] = {}
    stats: Dict[str, int] = defaultdict(int)
    outside_kernels: Set[int] = set()
    outside_gemms: Set[Key] = set()

    def merge(
        di: int,
        gemms: Mapping[Key, Dict[int, Rows]],
        cfgs: Mapping[int, Dict[str, int]],
        st: Mapping[str, int],
    ) -> None:
        for k, v in st.items():
            stats[k] += v
        for sid, c in cfgs.items():
            if in_pool is None or sid in in_pool:
                configs.setdefault(sid, c)
        for key, per in gemms.items():
            gone = (
                {s for s in per if s not in in_pool and s not in blocked}
                if in_pool is not None and not in_pool.issuperset(per)
                else None
            )
            if gone:
                stats["rows_outside_pool"] += sum(
                    len(per[s]) if isinstance(per[s], list) else 1 for s in gone
                )
                outside_kernels.update(gone)
                per = {s: v for s, v in per.items() if s not in gone}
                if not per:
                    outside_gemms.add(key)
                    continue
            origin[key].add(di)
            tgt = merged.get(key)
            if tgt is None:
                merged[key] = per
                continue
            for sid, val in per.items():
                tgt[sid] = _merge_row(tgt.get(sid), val)

    if jobs > 1 and len(files) > 1:
        workers = [r.fresh() if r is not None else None for r in remaps]
        with ProcessPoolExecutor(
            max_workers=min(jobs, len(files)),
            initializer=_init_worker,
            initargs=(workers,),
        ) as pool:
            parts = pool.map(
                _parse_folder_chunk, [p for _, p in files], [di for di, _ in files]
            )
            for (di, _), (gemms, cfgs, st, found) in zip(files, parts):
                folder = remaps[di]
                if folder is not None and found is not None:
                    folder.absorb(found)
                merge(di, gemms, cfgs, st)
    else:
        for di, p in files:
            merge(di, *parse_chunk(p, remaps[di]))
    if outside_kernels:
        stats["kernels_outside_pool"] = len(outside_kernels)
        n_gone = len(outside_gemms.difference(merged))
        if n_gone:
            stats["gemms_outside_pool"] = n_gone
    notes: List[str] = []
    for d, r in zip(dirs, remaps):
        if r is None:
            continue
        if identity is None:
            if not r.same:
                raise ki.BuildChangedError(_unlike_the_records(d, r))
            continue
        if strict:
            r.require_same()
        if not r.same:
            notes.append(r.explain())
    attributes: Dict[int, Dict[str, int]] = {}
    for d, r in zip(dirs, remaps):
        path = Path(d) / KERNELS_FILE
        if path.is_file():
            try:
                for old, attrs in read_kernel_attributes(path).items():
                    sid = r.index(old) if r is not None else old
                    if sid is not None:
                        attributes.setdefault(sid, attrs)
            except ValueError:
                stats["unreadable_kernel_files"] += 1
    for sid in blocked:
        configs.pop(sid, None)
    for sid, c in configs.items():
        if sid in attributes:
            c["attributes"] = attributes[sid]
    tree = dict(tree or {})
    out: List[Gemm] = []
    for key in sorted(merged, key=str):
        if keep is not None and not keep(dict(zip(PROBLEM_FIELDS, key))):
            stats["gemms_filtered"] += 1
            continue
        rows = merged[key]
        if blocked and not blocked.isdisjoint(rows):
            rows = {s: v for s, v in rows.items() if s not in blocked}
            stats["rows_blocked"] += len(merged[key]) - len(rows)
            if not rows:
                stats["gemms_blocked"] += 1
                continue
        per = _labelled(rows, aggregation)
        sids = np.fromiter(per.keys(), dtype=np.int64, count=len(per))
        us = np.fromiter((v[0] for v in per.values()), dtype=np.float64, count=len(per))
        skip = np.fromiter((v[1] for v in per.values()), dtype=bool, count=len(per))
        tested = us[~skip]
        winner = float(tested.min()) if tested.size else float(us.min())
        m, n, k, b = key[:4]
        folders = sorted(origin[key])
        ctx = tuple(key[len(PROBLEM_FIELDS) :]) or None
        listed = (
            _listed(
                (m, n, k, b),
                [plans[i] for i in folders],
                (
                    [(context_plans[i] or {}).get(ctx) for i in folders]
                    if context_plans is not None and ctx is not None
                    else None
                ),
            )
            if plans is not None
            else None
        )
        if listed is not None:
            if identity is not None:
                listed = ki.with_aliases(listed, identity.aliases)
            stats["gemms_listed"] += 1
        out.append(
            Gemm(
                key=key,
                leaf=assign_subcell(cell_key(m, n, k, b), m, n, k, b, tree),
                sids=sids,
                us=us,
                skip=skip,
                winner_us=winner,
                listed=listed,
                fallback=_fallback(per),
            )
        )
    stats["gemms"] = len(out)
    return Bench(
        gemms=out,
        configs=configs,
        stats=dict(stats),
        sources=[str(d) for d in dirs],
        notes=notes,
    )


# ── kernel pool ──────────────────────────────────────────────────────────────


class Pool:
    """A kernel pool in runtime order with per-field arrays for the gates
    and the signature of every kernel."""

    def __init__(self, kernels: Sequence[Mapping[str, Any]], source: str) -> None:
        self.kernels = [dict(k) for k in kernels]
        self.source = source
        self.sids = np.array(
            [int(k["sol_idx_global"]) for k in self.kernels], dtype=np.int64
        )
        self.pos = {int(s): i for i, s in enumerate(self.sids)}
        self.f = {
            name: np.array([int(k[name]) for k in self.kernels], dtype=np.int64)
            for name in ev.CONFIG_FIELDS
        }
        self.sigs = [signature_of(k) for k in self.kernels]
        self.signatures = sorted(set(self.sigs))
        index = {s: i for i, s in enumerate(self.signatures)}
        self.sig_index = np.array([index[s] for s in self.sigs], dtype=np.int64)
        self.nt_a, self.nt_b = feasibility.non_temporal_availability(self.kernels)
        families = feasibility.non_temporal_families(self.kernels)
        fam = [feasibility.family_key(k) for k in self.kernels]
        family_index = {f: i for i, f in enumerate(sorted(set(fam)))}
        self.family = np.array([family_index[f] for f in fam], dtype=np.int64)
        self.family_kept = np.array(
            [
                (int(k["cache_hints_a"]), int(k["cache_hints_b"])) == families[f].kept
                for k, f in zip(self.kernels, fam)
            ],
            dtype=bool,
        )
        self._view_hashes: Dict[Tuple[Any, ...], List[int]] = {}

    def __len__(self) -> int:
        return len(self.kernels)

    def whitelist_mask(
        self,
        signatures: Iterable[Sequence[int]],
        view_keys: Iterable[int] = (),
        spec: fs.AttributeSpec = (),
    ) -> np.ndarray:
        """Kernels whose signature is one of `signatures` or whose v3 view
        key (under attribute spec `spec`) hashes to one of `view_keys`."""
        wanted = {tuple(int(v) for v in s) for s in signatures}
        mask = np.array([s in wanted for s in self.sigs], dtype=bool)
        keys = {int(h) & 0xFFFFFFFFFFFFFFFF for h in view_keys}
        if keys:
            hashes = self._view_hashes.get(tuple(spec))
            if hashes is None:
                hashes = self._view_hashes[tuple(spec)] = [
                    fs.view_key_hash(
                        fs.v3_view_key(
                            fs.config_kwargs_from_row(k), k.get("attributes"), spec
                        )
                    )
                    for k in self.kernels
                ]
            mask |= np.array([h in keys for h in hashes], dtype=bool)
        return mask


def pool_for(bench: Bench, library: Optional[Sequence[Mapping[str, Any]]]) -> Pool:
    """The library's pool, else the measured kernels in ascending index."""
    if library is not None:
        return Pool(library, "library")
    kernels = []
    for sid in sorted(s for s in bench.configs if s >= 0):
        k = dict(bench.configs[sid])
        k["sol_idx_global"] = sid
        kernels.append(k)
    return Pool(kernels, "bench")


def leading_free_mask(pool: Pool, problem: Mapping[str, Any]) -> np.ndarray:
    """Kernels of `pool` that pass the leading-free rule for one problem
    (`lib.features` problem kwargs): a contiguous free dimension holds one
    global-read vector (M >= grvw_a unless A is transposed, N >= grvw_b when
    B is; not checked for Int8x4). It is TensileLite's
    LeadingFree0/1SizesGreaterOrEqual predicate, so a kernel failing it
    cannot run on the problem, whether or not the model's engine applies the
    rule; no bench row has such a kernel."""
    f = pool.f
    M, N = int(problem["m"]), int(problem["n"])
    ok = np.ones(len(pool), dtype=bool)
    if fs.normalize_dtype_name(problem["a_dtype"]) == "int8x4":
        return ok
    if not feasibility._is_t(problem["a_transpose"]):
        ok &= ~(M < f["grvw_a"])
    if feasibility._is_t(problem["b_transpose"]):
        ok &= ~(N < f["grvw_b"])
    return ok


def gate_masks(
    pool: Pool,
    problem: Mapping[str, Any],
    *,
    lds_bytes: int,
    rules: Mapping[str, bool],
) -> Tuple[np.ndarray, np.ndarray]:
    """(hard, soft) pass masks over `pool` for one problem (`lib.features`
    problem kwargs) under the rules a bundle serves with (`rules`:
    `lib.feasibility.passes_gates` keywords, `serving_rules`): `hard` = the
    LDS gate and the small-batched, Dot2 and leading-free rules where
    `rules` applies them (an absent keyword applies the rule, as
    `passes_gates` does), `soft` = `hard` and the non-temporal rule (refined
    family-wise) where `rules` applies it; `soft` is
    `lib.feasibility.passes_gates`, the kernels the engine admits."""
    f = pool.f
    M, N, K, B = (int(problem[x]) for x in ("m", "n", "k", "batch"))
    a_dt, b_dt = problem["a_dtype"], problem["b_dtype"]
    b_t = feasibility._is_t(problem["b_transpose"])
    la, lb = feasibility.lds_bits(a_dt), feasibility.lds_bits(b_dt)
    if la <= 0 or lb <= 0:
        none = np.zeros(len(pool), dtype=bool)
        return none, none.copy()
    usage = f["mt_m"] * f["mt_k"] * (la / 8.0) + f["mt_n"] * f["mt_k"] * (lb / 8.0)
    ok = usage <= float(lds_bytes)
    if rules.get("small_batched_rule", True):
        if M <= 256 and N <= 256 and K < 1024 and B != 1:
            ok &= ~((f["mt_m"] < M) | (f["mt_n"] < N))
    if rules.get("dot2_rule", True) and M > 2:
        ok &= ~((f["mi_m"] == 1) & (f["mi_n"] == 1) & (f["mi_k"] == 64))
    if rules.get("leading_free_rule", True):
        ok &= leading_free_mask(pool, problem)
    if not rules.get("non_temporal_rule", True):
        return ok, ok.copy()
    ra, rb = feasibility.rule_bits(a_dt), feasibility.rule_bits(b_dt)
    a_t = feasibility._is_t(problem["a_transpose"])
    cha, chb = f["cache_hints_a"], f["cache_hints_b"]
    hinted = (cha != 0) | (chb != 0)
    nt = feasibility.NON_TEMPORAL
    if (K * ra) % 1024 == 0:
        aligned = (f["mt_k"] * ra) % 1024 == 0
        to_b = (M <= f["mt_m"] * 2) & (not b_t) & ((N * rb) // max(M * ra, 1) > 5)
        to_a = ~to_b & (N <= f["mt_n"] * 2) & a_t & ((M * ra) // max(N * rb, 1) > 5)
        rest = ~to_b & ~to_a
        fail_aligned = (
            (to_b & pool.nt_b & (chb != nt))
            | (to_a & pool.nt_a & (cha != nt))
            | (rest & hinted)
        )
        fail = np.where(aligned, fail_aligned, hinted)
    else:
        fail = hinted
    if rules.get("non_temporal_family_rule", False) and feasibility.in_family_region(
        M, N, K, B
    ):
        passing = np.zeros(int(pool.family.max()) + 1 if len(pool) else 0, dtype=bool)
        np.logical_or.at(passing, pool.family, ~fail)
        fail = fail & ~(pool.family_kept & ~passing[pool.family])
    return ok, ok & ~fail


# ── the shipped model ────────────────────────────────────────────────────────


@dataclass
class ModelInfo:
    labels: List[str]
    whitelists: Dict[str, List[Tuple[int, ...]]]
    # Per cell, the view-key hashes of a v3 model's keyed whitelist, read
    # only when the model serves a whitelist tier.
    view_keys: Dict[str, List[int]] = field(default_factory=dict)
    non_temporal_rule: bool = True
    non_temporal_family_rule: bool = False
    cascade_keys: int = 0
    launch_keys: bool = False
    data: Optional[bytes] = None
    hardware: Optional[DeviceHardware] = None
    weight_dtype: Optional[str] = None
    round: Optional[str] = None
    source: str = "cells.json"
    library: Optional[Tuple[str, str]] = None
    # The config and device stage05 ranked the library with (for its bank).
    library_cfg: Optional[Dict[str, Any]] = None
    library_device: int = 0
    notes: List[str] = field(default_factory=list)
    train_round: Optional[str] = None
    deploy_round: Optional[str] = None
    deployed: Optional[bool] = None
    arch: Optional[str] = None
    attribute_spec: fs.AttributeSpec = ()
    # The rounds whose every GEMM a bundle trained on at once (the final
    # retrain's); None for a round's bundle.
    trained_rounds: Optional[List[str]] = None
    # The model file's format version and the rules it serves with
    # (`lib.feasibility.soft_rules` of its serving flags); None when the
    # file was not read.
    version: Optional[int] = None
    rules: Optional[Dict[str, bool]] = None


def serving_rules(model: Optional[ModelInfo]) -> Dict[str, bool]:
    """`lib.feasibility.passes_gates` rule keywords of the rules `model`
    serves with: those of its model file's serving flags when it was read
    (`ModelInfo.rules`), else MLREC_v2's: the small-batched, Dot2 and
    leading-free rules, and the non-temporal rule (refined family-wise) as
    the model records it."""
    if model is not None and model.rules is not None:
        return dict(model.rules)
    nt = model.non_temporal_rule if model is not None else True
    out = {
        "small_batched_rule": True,
        "dot2_rule": True,
        "leading_free_rule": True,
        "non_temporal_rule": bool(nt),
    }
    if nt and model is not None and model.non_temporal_family_rule:
        out["non_temporal_family_rule"] = True
    return out


def _read(path: Path) -> Optional[Dict[str, Any]]:
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def _read_yaml(path: Any) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    import yaml

    try:
        doc = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError):
        return None
    return doc if isinstance(doc, dict) else None


def deploy_records(run_dir: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    """(bundle dir, stage06 deploy_record.json) of every round that has one
    and of the final retrain while it is a bundle of the run, in the order
    the run deploys them (lib/run_bundles.py)."""
    out = []
    for rd in run_bundles.bundles(run_dir, round_dirs(run_dir)):
        rec = _read(rd / "stage06" / "deploy_record.json")
        if rec is not None:
            out.append((rd, rec))
    return out


def shipped_record(run_dir: Path) -> Optional[Tuple[Path, Dict[str, Any]]]:
    """The latest deploy record the deploy gate let through (`deployed` not
    false)."""
    for rd, rec in reversed(deploy_records(run_dir)):
        if rec.get("deployed", True) is not False:
            return rd, rec
    return None


def staged_record(run_dir: Path) -> Optional[Tuple[Path, Dict[str, Any]]]:
    """The latest deploy record of a completed stage06, deployed or not."""
    for rd, rec in reversed(deploy_records(run_dir)):
        if (rd / "stage06" / STAGE_MARKER).is_file():
            return rd, rec
    return None


def _round_of(stage05: Path, run_dir: Path) -> Optional[Path]:
    """The round folder of this run a stage05 folder belongs to."""
    by_name = {rd.name: rd for rd in round_dirs(run_dir)}
    for p in (stage05, *stage05.parents):
        if p.name in by_name:
            return by_name[p.name]
    return None


def final_model(
    run_dir: Path, *, want_bytes: bool = True, staged: bool = False
) -> Optional[ModelInfo]:
    """The bundle the run shipped: the train_dir of its latest deploy record
    not marked `deployed: false` (stage06's staged copy of the model file
    when it is the same bundle, else models.pt written at the metrics'
    weight dtype, else cells.json), a round's or the final retrain's
    (`deploy_records`). Without one, `staged` takes the latest staged
    bundle (`staged_record`, `deployed` False); otherwise, and in runs
    without deploy records, the newest round with stage05/models.pt or
    cells.json."""
    run_dir = Path(run_dir)
    shipped = shipped_record(run_dir)
    is_staged = False
    if shipped is None and staged:
        shipped = staged_record(run_dir)
        is_staged = shipped is not None
    candidates: List[Tuple[Path, Path, Optional[Dict[str, Any]]]] = []
    if shipped is not None:
        rd, rec = shipped
        s5 = Path(str(rec.get("train_dir") or rd / "stage05"))
        candidates.append((_round_of(s5, run_dir) or rd, s5, rec))
    else:
        candidates += [
            (rd, rd / "stage05", None) for rd in reversed(round_dirs(run_dir))
        ]
    for rd, s5, rec in candidates:
        cj = _read(s5 / "cells.json")
        if cj is None and not (s5 / "models.pt").is_file():
            continue
        met = _read(bundle_file(s5, "metrics.json")) or {}
        args = met.get("args") if isinstance(met.get("args"), dict) else {}
        info = ModelInfo(labels=[], whitelists={}, round=rd.name, train_round=rd.name)
        if rec is not None:
            info.deploy_round, info.deployed = shipped[0].name, not is_staged
            if is_staged:
                info.notes.append(
                    f"no bundle of this run passed the deploy gate: the analysis "
                    f"uses the latest staged bundle ({shipped[0].name}, not deployed)"
                )
        else:
            info.notes.append(
                (
                    "no bundle of this run passed the deploy gate"
                    if deploy_records(run_dir)
                    else "this run has no deploy record"
                )
                + ": the analysis uses its newest trained model"
            )
        info.weight_dtype = met.get("weight_dtype") or args.get("weight_dtype")
        info.non_temporal_rule = not bool(args.get("no_non_temporal_rule"))
        info.non_temporal_family_rule = bool(args.get("non_temporal_family_rule"))
        info.arch = args.get("arch") or met.get("arch")
        info.launch_keys = bool(args.get("launch_keys"))
        options = _read(bundle_file(s5, "model_options.json")) or {}
        try:
            info.attribute_spec = fs.parse_attribute_spec(
                options.get("attribute_spec") or []
            )
        except ValueError:
            info.attribute_spec = ()
        hw = met.get("hardware") if isinstance(met.get("hardware"), dict) else None
        try:
            info.hardware = (
                DeviceHardware(
                    **{k: int(hw[k]) for k in ("n_cu", "lds_bytes", "l2_bytes")},
                    num_xcd=(None if hw.get("num_xcd") is None else int(hw["num_xcd"])),
                )
                if hw
                else None
            )
        except (KeyError, TypeError, ValueError):
            info.hardware = None
        if args.get("library_dir") and args.get("library_stem"):
            info.library = (str(args["library_dir"]), str(args["library_stem"]))
            info.library_cfg = _read_yaml(args.get("config_yaml"))
            info.library_device = int(args.get("hardware_device") or 0)
        if want_bytes:
            info.data, info.source = _model_bytes(s5, rec, args, info, met)
        if info.data is not None:
            parsed = mlrec.read_model(info.data)
            info.labels = [str(c["label"]) for c in parsed["cells"]]
            info.whitelists = {
                str(c["label"]): [
                    tuple(int(v) for v in s) for s in c.get("smart_k_signatures") or []
                ]
                for c in parsed["cells"]
            }
            serving = parsed.get("serving") or {}
            info.version = int(parsed.get("version") or 2)
            info.rules = feasibility.soft_rules(serving, info.version)
            if parsed.get("version") == 3 and serving.get("whitelist_tier"):
                info.view_keys = {
                    str(c["label"]): [int(h) for h in c.get("view_key_hashes") or []]
                    for c in parsed["cells"]
                }
            info.non_temporal_rule = bool(serving.get("non_temporal_rule", True))
            info.non_temporal_family_rule = bool(
                serving.get("non_temporal_family_rule", False)
            )
            info.cascade_keys = int(serving.get("cascade_keys", 0) or 0)
            info.launch_keys = bool(serving.get("launch_keys", False))
            info.attribute_spec = tuple(parsed.get("attribute_spec") or ())
            info.arch = str(parsed.get("arch") or info.arch or "") or None
        elif cj is not None:
            labels = cj.get("model_labels")
            cells = [
                c
                for c in cj.get("cells") or []
                if isinstance(c, dict) and c.get("label")
            ]
            info.labels = (
                [str(x) for x in labels]
                if isinstance(labels, list)
                else [str(c["label"]) for c in cells]
            )
            info.whitelists = {
                str(c["label"]): [
                    tuple(int(v) for v in s) for s in c.get("smart_k_signatures") or []
                ]
                for c in cells
            }
            info.notes.append(
                "the model file could not be read: tier 1 uses cells.json whitelists, "
                "the ranking steps are not decomposed"
            )
        if run_bundles.round_index(rd) is None:
            info.trained_rounds = list(run_bundles.trained_rounds(rd))
        return info
    return None


def _model_bytes(
    s5: Path,
    rec: Optional[Mapping[str, Any]],
    args: Mapping[str, Any],
    info: ModelInfo,
    met: Mapping[str, Any],
) -> Tuple[Optional[bytes], str]:
    if rec is None:
        rec = _read(s5.parent / "stage06" / "deploy_record.json")
    if rec and rec.get("staged_model") and rec.get("train_dir"):
        try:
            same = Path(rec["train_dir"]).resolve() == s5.resolve()
            data = Path(rec["staged_model"]).read_bytes() if same else None
        except OSError:
            data = None
        if data is not None and hashlib.sha256(data).hexdigest() == rec.get("sha256"):
            info.weight_dtype = rec.get("weight_dtype") or info.weight_dtype
            return data, "stage06 staged model"
    pt = s5 / "models.pt"
    arch = args.get("arch") or met.get("arch")
    if not pt.is_file() or not arch or not info.weight_dtype:
        return None, "cells.json"
    try:
        import torch

        from .hardware import arch_constants

        bundle = torch.load(str(pt), map_location="cpu", weights_only=True)
        data = mlrec.write_model(
            bundle, None, str(arch), arch_constants(str(arch)), str(info.weight_dtype)
        )
    except Exception as e:  # noqa: BLE001
        info.notes.append(f"models.pt could not be written as a model file: {e}")
        return None, "cells.json"
    return data, "models.pt"


@dataclass
class BenchSet:
    """Folders of one group of benches, the plan of each and its execution
    contexts' re-plans (`round_context_plans`)."""

    dirs: List[Path]
    plans: List[Optional[PlanLookup]]
    context_plans: List[Optional[Dict[Tuple[int, str], PlanLookup]]] = field(
        default_factory=list
    )

    def __bool__(self) -> bool:
        return bool(self.dirs)


def model_benches(
    run_dir: Path,
    model: Optional[ModelInfo],
    *,
    with_plans: bool = True,
    identity: Optional[ki.LibraryIdentity] = None,
) -> Dict[str, BenchSet]:
    """The bench folders of the run relative to `model`'s training round T:

    training   rounds before T (stage04) and T's own training rows;
    held_out   T's deploy-gate slice and the stage04 of later rounds,
               GEMMs the model never trained on;
    gate       the deploy-gate slice of the round that shipped the model
               (stage06/gate.json measured on it);
    holdout_contexts  every round's held-out execution contexts
               (stage03c/holdout), never trained on and benched without
               a plan.

    A bundle that trained on every GEMM of its rounds at once (the final
    retrain: `ModelInfo.trained_rounds`) has their stage04 as training and
    that of any other round as held_out. With `identity` the plans' kernel
    lists are its library's indices. Each folder's GEMMs of an execution
    context are listed by the round's re-plan of that context
    (`round_context_plans`), as they were benched.
    """
    rounds = round_dirs(run_dir)
    names = [rd.name for rd in rounds]
    t = model.train_round if model is not None else None
    ti = names.index(t) if t in names else len(rounds) - 1
    every = (
        set(model.trained_rounds)
        if model is not None and model.trained_rounds is not None
        else None
    )
    plans = {rd.name: round_plan(rd, identity) if with_plans else None for rd in rounds}
    replans = {
        rd.name: round_context_plans(rd, identity) if with_plans else {}
        for rd in rounds
    }
    out = {k: BenchSet([], []) for k in ("training", "held_out", "gate")}
    out["holdout_contexts"] = BenchSet([], [])

    def add(kind: str, d: Optional[Path], rd: Path, plan: bool = True) -> None:
        if d is not None and _has_chunks(d):
            out[kind].dirs.append(d)
            out[kind].plans.append(plans[rd.name] if plan else None)
            out[kind].context_plans.append(replans[rd.name] if plan else None)

    for i, rd in enumerate(rounds):
        if every is not None:
            add("training" if rd.name in every else "held_out", rd / "stage04", rd)
        elif i < ti:
            add("training", rd / "stage04", rd)
        elif i == ti:
            add("training", training_data_dir(rd), rd)
            add("held_out", gate_dir(rd), rd)
        else:
            add("held_out", rd / "stage04", rd)
        add("holdout_contexts", holdout_context_dir(rd), rd, plan=False)
    dep = model.deploy_round if model is not None else None
    if dep in names:
        add("gate", gate_dir(rounds[names.index(dep)]), rounds[names.index(dep)])
    return out


# ── per-GEMM evaluation ──────────────────────────────────────────────────────


@dataclass
class Evaluated:
    """Per-GEMM results of `evaluate`, aligned with `gemms`. `levels` holds
    the costs of LEVELS (NaN where not computed); `served` marks GEMMs with
    a pick, `blocked` names the first empty level of the others; `miss` is
    the pick's miss reason and `fallback_us` the cost of the runtime's
    fallback pick of an unserved GEMM."""

    gemms: List[Gemm]
    pool: Pool
    model: Optional[ModelInfo]
    levels: np.ndarray
    served: np.ndarray
    blocked: List[Optional[str]]
    model_cell: List[Optional[str]]
    pos: List[np.ndarray]
    pick_pos: np.ndarray
    winner_pos: np.ndarray
    engine: bool
    pick_measurement: str = CONFIG_PICK_MEASUREMENT
    miss: List[Optional[str]] = field(default_factory=list)
    fallback_us: Optional[np.ndarray] = None
    # Per GEMM: the pick can run but was never measured (its level holds an
    # estimate, `resolve_pick`), the slowest latency the GEMM measured (its
    # worst case), and the scored kernels ranked above the pick that cannot
    # run (the leading-free rule).
    unmeasured: Optional[np.ndarray] = None
    worst_us: Optional[np.ndarray] = None
    n_unrunnable: Optional[np.ndarray] = None
    n_inconsistent: int = 0
    n_out_of_pool_rows: int = 0
    n_without_pool_kernel: int = 0
    n_winner_outside_pool: int = 0

    @property
    def contexts(self) -> List[str]:
        return [context_label(g.context) for g in self.gemms]


def _restricted(g: Gemm, inside: np.ndarray) -> Gemm:
    us, skip = g.us[inside], g.skip[inside]
    tested = us[~skip]
    return Gemm(
        g.key,
        g.leaf,
        g.sids[inside],
        us,
        skip,
        float(tested.min()) if tested.size else float(us.min()),
        g.listed,
        g.fallback if g.fallback in set(g.sids[inside].tolist()) else None,
    )


@dataclass
class Pick:
    """The kernel the runtime serves from one GEMM's scored ranking
    (`resolve_pick`): its pool position (-1: none runs), cost and miss
    reason; `unmeasured` when it can run but has no reading, `cost` then
    being the next-ranked kernel's with a reading the measurement accepts
    (else the slowest latency the GEMM measured); `n_unrunnable` counts the
    scored kernels ranked above it that cannot run."""

    pos: int = -1
    cost: float = math.nan
    miss: Optional[str] = None
    unmeasured: bool = False
    n_unrunnable: int = 0


def resolve_pick(
    scored: Sequence[int],
    sids: np.ndarray,
    runnable: np.ndarray,
    timed: Mapping[int, float],
    screened: Mapping[int, float],
    listed: Optional[Iterable[int]],
    pick_measurement: str,
) -> Pick:
    """The kernel the runtime serves from `scored` (the engine's scored
    pool positions, best first; `sids` the pool's solution indices) for one
    GEMM with rows `timed` / `screened` ({solution: latency}) and planned
    list `listed`: the first one that runs. A kernel with a row runs. One
    without a row that fails the leading-free rule (`runnable` False:
    TensileLite's predicate) cannot run, and one on the list (or without a
    list) is one the bench did not run (`lib.evaluate.pick_cost`): the
    runtime passes over both. One off the list that passes the rule may
    run: under `pick_measurement` "any" it is passed over like a kernel
    without a row, as the stages do; otherwise it is the pick, a
    `not_benched` miss without a reading, costed at the next-ranked kernel
    with a reading the measurement accepts, else the slowest latency the
    GEMM measured."""
    out = Pick()
    for j, p in enumerate(scored):
        sid = int(sids[p])
        if sid not in timed and sid not in screened and not runnable[p]:
            out.n_unrunnable += 1
            continue
        runs, cost, miss = ev.pick_cost(sid, timed, screened, listed, pick_measurement)
        if not runs:
            continue
        out.pos, out.miss = int(p), miss
        if miss != "not_benched":
            out.cost = float(cost)
            return out
        out.unmeasured = True
        out.cost = float(cost)
        for q in scored[j + 1 :]:
            qs = int(sids[q])
            if qs not in timed and qs not in screened:
                continue
            ok, c, m = ev.pick_cost(qs, timed, screened, listed, pick_measurement)
            if ok and m is None:
                out.cost = float(c)
                break
        return out
    return out


def evaluate(
    bench: Bench,
    pool: Pool,
    model: Optional[ModelInfo],
    *,
    tw: Any = None,
    pick_measurement: str = CONFIG_PICK_MEASUREMENT,
    restrict_to_pool: bool = False,
) -> Evaluated:
    """Decompose every GEMM of `bench` (see the module docstring). Rows of
    kernels outside `pool` stay oracle candidates (stages 4.5 and 6) unless
    `restrict_to_pool` drops them (stage 8). Without `tw` or the model file
    the `scored` and `pick` levels stay NaN."""
    if pick_measurement not in ev.PICK_MEASUREMENTS:
        raise ValueError(
            f"pick_measurement {pick_measurement!r} is not one of "
            f"{ev.PICK_MEASUREMENTS}"
        )
    screened_miss = pick_measurement not in ev.WARMUP_COUNTS
    off_list_miss = pick_measurement != "any"
    engine = tw is not None and model is not None and model.data is not None
    labels = set(model.labels) if model else set()
    lds = model.hardware.lds_bytes if model and model.hardware else None
    rules = serving_rules(model)
    wl_masks: Dict[str, np.ndarray] = {}
    if engine:
        loaded = ev.load_model_bytes(model.data, tw)
        cs = ev.candidate_set(tw, loaded, pool.kernels, model.hardware)
        hw = ev.make_hardware(tw, model.hardware)
        contexts: Dict[fs.ExecutionContext, Any] = {}
    kept: List[Gemm] = []
    rows = []
    n_out = n_empty = 0
    for g in bench.gemms:
        pos = np.array([pool.pos.get(int(s), -1) for s in g.sids], dtype=np.int64)
        inside = pos >= 0
        n_out += int((~inside).sum())
        if not inside.any():
            n_empty += 1
            continue
        if restrict_to_pool and not inside.all():
            g = _restricted(g, inside)
            pos = pos[inside]
        kept.append(g)
        rows.append(pos)
    n = len(kept)
    P = len(pool)
    levels = np.full((n, len(LEVELS)), np.nan)
    served = np.zeros(n, dtype=bool)
    blocked: List[Optional[str]] = [None] * n
    cells: List[Optional[str]] = [None] * n
    miss: List[Optional[str]] = [None] * n
    fallback_us = np.full(n, np.nan)
    unmeasured = np.zeros(n, dtype=bool)
    worst_us = np.full(n, np.nan)
    n_unrunnable = np.zeros(n, dtype=np.int64)
    pick_pos = np.full(n, -1, dtype=np.int64)
    winner_pos = np.full(n, -1, dtype=np.int64)
    n_bad = n_win_out = 0
    for i, (g, pos) in enumerate(zip(kept, rows)):
        prob = fs.problem_kwargs_from_row(g.row)
        w = g.winner_us
        tested = ~g.skip if (~g.skip).any() else np.ones_like(g.skip)
        wp = int(pos[np.flatnonzero(tested & (g.us == w))[0]])
        winner_pos[i] = wp
        n_win_out += wp < 0
        if g.fallback is not None:
            hit = np.flatnonzero(g.sids == g.fallback)
            if hit.size:
                fallback_us[i] = float(g.us[hit[0]])
        worst = float(g.us.max())
        worst_us[i] = worst
        row_cost = np.where(g.skip & screened_miss, worst, g.us)
        inside = pos >= 0
        cost = np.full(P, np.inf)
        cost[pos[inside]] = row_cost[inside]
        if off_list_miss and g.listed is not None:
            off = ~np.isin(pool.sids, np.fromiter(g.listed, dtype=np.int64))
            off[pos[inside]] = False
            cost[off] = worst
        lv = [w, max(w, float(cost.min()))]
        if engine:
            pobj = ev.make_problem(tw, prob)
            cell = ev.model_cell_label(tw, loaded, pobj)
        else:
            cell = resolve_model_cell(g.leaf, labels) if labels else None
        cells[i] = cell
        if lds is None:
            levels[i, :2] = lv
            blocked[i] = "hard"
            continue
        hard, soft = gate_masks(pool, prob, lds_bytes=lds, rules=rules)
        runnable = leading_free_mask(pool, prob)
        masks = [hard & runnable, soft & runnable]
        admitted = soft
        if cell is not None:
            wl = wl_masks.get(cell)
            if wl is None:
                wl = wl_masks[cell] = pool.whitelist_mask(
                    model.whitelists.get(cell) or [],
                    model.view_keys.get(cell) or [],
                    model.attribute_spec,
                )
            admitted = soft & wl
            if not admitted.any():
                admitted = soft
            masks.append(admitted & runnable)
        stop = None
        for name, m in zip(LEVELS[2:], masks):
            best = float(cost[m].min()) if m.any() else math.inf
            if not math.isfinite(best):
                stop = name
                break
            lv.append(max(lv[-1], best))
        if stop is None and cell is None:
            stop = "tier"
        if stop is None and engine:
            ctx = g.context
            if ctx not in contexts:
                contexts[ctx] = ev.make_context(tw, ctx)
            ranked = cs.rank(pobj, hw, 1, contexts[ctx])
            scored = [int(r.config_index) for r in ranked if r.scored]
            timed = {int(s): float(u) for s, u, k in zip(g.sids, g.us, g.skip) if not k}
            screened = {int(s): float(u) for s, u, k in zip(g.sids, g.us, g.skip) if k}
            pk = resolve_pick(
                scored, pool.sids, runnable, timed, screened, g.listed, pick_measurement
            )
            n_unrunnable[i] = pk.n_unrunnable
            runs = [p for p in scored if math.isfinite(cost[p])]
            if pk.pos < 0:
                stop = "scored"
            else:
                if not admitted[runs].all():
                    n_bad += 1
                # When the rule fails every scored kernel with a cost (the
                # pool's vector widths disagree with the build benched), the
                # pick has a row: it ran, so it is costed among `runs`.
                costed = [p for p in runs if runnable[p]] or runs
                lv.append(max(lv[-1], float(cost[costed].min())))
                lv.append(max(w, pk.cost))
                lv[1:-1] = [min(v, lv[-1]) for v in lv[1:-1]]
                pick_pos[i] = pk.pos
                served[i] = True
                miss[i] = pk.miss
                unmeasured[i] = pk.unmeasured
        levels[i, : len(lv)] = lv
        blocked[i] = stop
    return Evaluated(
        gemms=kept,
        pool=pool,
        model=model,
        levels=levels,
        served=served,
        blocked=blocked,
        model_cell=cells,
        pos=rows,
        pick_pos=pick_pos,
        winner_pos=winner_pos,
        engine=engine,
        pick_measurement=pick_measurement,
        miss=miss,
        fallback_us=fallback_us,
        unmeasured=unmeasured,
        worst_us=worst_us,
        n_unrunnable=n_unrunnable,
        n_inconsistent=n_bad,
        n_out_of_pool_rows=n_out,
        n_without_pool_kernel=n_empty,
        n_winner_outside_pool=n_win_out,
    )


# ── aggregates ───────────────────────────────────────────────────────────────


def _geo_eff(logs: np.ndarray) -> Optional[float]:
    logs = logs[np.isfinite(logs)]
    return float(math.exp(-logs.mean())) if logs.size else None


def _group_keys(e: Evaluated, by: str) -> List[str]:
    if by == "leaf":
        return [g.leaf for g in e.gemms]
    if by == "model_cell":
        return [c or "(none)" for c in e.model_cell]
    if by == "context":
        return e.contexts
    raise ValueError(f"no grouping {by!r}")


def decomposition(
    e: Evaluated, by: str = "leaf", where: Optional[Sequence[int]] = None
) -> Dict[str, Dict[str, Any]]:
    """Per cell (`leaf` or `model_cell`) or execution context (`context`) and
    overall ("*") over the GEMMs `where` (default all): the loss of each
    selection step as an efficiency factor (product = selection efficiency)
    over the served GEMMs, the efficiency ceilings of each level over every
    GEMM that reaches it, why the unserved GEMMs have no pick, the misses
    among the picks, and `deployed_sel_eff`, the served picks and the
    runtime's fallback picks of the unserved GEMMs together, as stages 4.5,
    6 and 8 count them. It is not the stages' number where they charge a
    pick differently: a pick that can run but was never measured counts here
    at its estimate (`resolve_pick`), where stages 4.5, 6 and 8 charge it
    the slowest latency its GEMM measured unless a re-bench measured it
    (`validate.rebench_misses`) and stage05 the winner's latency over
    `train.off_list_sel_eff` (the slowest latency when that gives no
    efficiency: unset, or `auto` with fewer re-benched misses than
    `train.off_list_min_misses` and no `train.off_list_fallback`); a kernel
    off the planned list that cannot run is passed over here, as the runtime
    passes over it, where stages 4.5, 5, 6 and 8 charge it as their pick
    under every measurement but `any`. Stage 7 costs the runtime's top-1
    without the planned list: it leaves out of its selection efficiency a
    problem whose top-1 kernel has no row, whether the kernel was never
    measured or cannot run. `n_unmeasured` counts the
    picks that can run but were never measured; `sel_eff_range` and
    `deployed_sel_eff_range` bound the efficiencies with each of them at the
    slowest latency its GEMM measured and at the winner's ([low, high]; None
    without such picks). `n_top_unrunnable` counts the served GEMMs with at
    least one scored kernel ranked above the pick that cannot run (the
    leading-free rule; `Pick.n_unrunnable`), the top-ranked one or another,
    which the runtime passes over."""
    keys = _group_keys(e, by)
    groups: Dict[str, List[int]] = defaultdict(list)
    for i in range(len(e.gemms)) if where is None else where:
        groups[keys[i]].append(i)
        groups["*"].append(i)
    n = len(e.gemms)
    unmeasured = e.unmeasured if e.unmeasured is not None else np.zeros(n, bool)
    unrunnable = e.n_unrunnable if e.n_unrunnable is not None else np.zeros(n, int)
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log(e.levels)
        fb = np.log(e.fallback_us) if e.fallback_us is not None else None
        worst = np.log(e.worst_us) if e.worst_us is not None else logs[:, -1]
    pick_lo = np.where(unmeasured, worst, logs[:, -1])
    pick_hi = np.where(unmeasured, logs[:, 0], logs[:, -1])
    out: Dict[str, Dict[str, Any]] = {}
    for k, idx in groups.items():
        idx_a = np.asarray(idx, dtype=np.int64)
        lg = logs[idx_a]
        sv = e.served[idx_a]
        rec: Dict[str, Any] = {"n_gemms": len(idx), "n_served": int(sv.sum())}
        rec["blocked"] = dict(
            Counter(e.blocked[i] for i in idx if not e.served[i] and e.blocked[i])
        )
        misses = Counter(e.miss[i] for i in idx if e.served[i] and e.miss[i])
        rec["n_miss"] = sum(misses.values())
        rec["n_miss_by_reason"] = dict(misses)
        um = unmeasured[idx_a] & sv
        rec["n_unmeasured"] = int(um.sum())
        rec["n_top_unrunnable"] = int(((unrunnable[idx_a] > 0) & sv).sum())
        rec["sel_eff_range"] = rec["deployed_sel_eff_range"] = None
        steps = {}
        if sv.any():
            d = np.diff(lg[sv], axis=1)
            for j, s in enumerate(STEPS):
                steps[s] = _geo_eff(d[:, j])
            rec["sel_eff"] = _geo_eff(lg[sv, -1] - lg[sv, 0])
            if um.any():
                rec["sel_eff_range"] = [
                    _geo_eff(pick_lo[idx_a][sv] - lg[sv, 0]),
                    _geo_eff(pick_hi[idx_a][sv] - lg[sv, 0]),
                ]
        else:
            rec["sel_eff"] = None
            for j, s in enumerate(STEPS):
                col = lg[:, j + 1] - lg[:, j]
                steps[s] = _geo_eff(col) if e.engine or s not in ENGINE_STEPS else None
        rec["steps"] = steps
        rec["ceiling"] = {
            name: _geo_eff(lg[:, j] - lg[:, 0])
            for j, name in enumerate(LEVELS)
            if j > 0
        }
        if e.engine:
            fbk = fb[idx_a] if fb is not None else np.nan
            dep = np.where(sv, lg[:, -1], fbk)
            ok = np.isfinite(dep)
            rec["n_fallback"] = int((ok & ~sv).sum())
            rec["n_deployed"] = int(ok.sum())
            rec["deployed_sel_eff"] = _geo_eff(dep[ok] - lg[ok, 0])
            if um.any():
                lo = np.where(sv, pick_lo[idx_a], fbk)
                hi = np.where(sv, pick_hi[idx_a], fbk)
                rec["deployed_sel_eff_range"] = [
                    _geo_eff(lo[ok] - lg[ok, 0]),
                    _geo_eff(hi[ok] - lg[ok, 0]),
                ]
        losses = {s: 1.0 - v for s, v in steps.items() if v is not None}
        rec["dominant"] = (
            max(losses, key=losses.get) if losses and max(losses.values()) > 0 else None
        )
        out[k] = rec
    return out


def context_decomposition(e: Evaluated) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """{context label: per-leaf decomposition of its GEMMs}."""
    idx: Dict[str, List[int]] = defaultdict(list)
    for i, c in enumerate(e.contexts):
        idx[c].append(i)
    return {c: decomposition(e, where=v) for c, v in sorted(idx.items())}


COMMON_PARAMS: Dict[str, Callable[[Mapping[str, Any]], str]] = {
    "macro tile (MT_M x MT_N)": lambda k: f"{k['mt_m']}x{k['mt_n']}",
    "depthU": lambda k: str(k["mt_k"]),
    "macro tile and depthU": lambda k: f"{k['mt_m']}x{k['mt_n']}x{k['mt_k']}",
    "matrix instruction (MI_M x MI_N x MI_K)": lambda k: (
        f"{k['mi_m']}x{k['mi_n']}x{k['mi_k']}"
    ),
    "cache hints (A,B)": lambda k: f"{k['cache_hints_a']},{k['cache_hints_b']}",
}


def diagnostic_params(pool: Pool) -> Dict[str, Callable[[Mapping[str, Any]], str]]:
    """Framework-specific kernel parameters: occupancy, vector widths and
    every kernel attribute that varies across the pool (workgroup mapping,
    stagger, ...)."""
    fns: Dict[str, Callable[[Mapping[str, Any]], str]] = {
        "occupancy": lambda k: str(k["occupancy"]),
        "GRVW (A,B)": lambda k: f"{k['grvw_a']},{k['grvw_b']}",
        "GWVW (D)": lambda k: str(k["gwvw_d"]),
    }
    names = sorted({n for k in pool.kernels for n in (k.get("attributes") or {})})
    for name in names:
        vals = {(k.get("attributes") or {}).get(name) for k in pool.kernels}
        if len(vals) > 1:
            fns[name] = lambda k, _n=name: str(
                (k.get("attributes") or {}).get(_n, "n/a")
            )
    return fns


def _in_pool(g: Gemm, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(pool positions, latencies, tested) of a GEMM's rows of pool kernels."""
    inside = pos >= 0
    return pos[inside], g.us[inside], ~g.skip[inside]


def kernel_stats(e: Evaluated) -> Dict[str, Any]:
    """Per kernel signature of the pool: GEMMs where a member was tested /
    only screened, wins, picks, GEMMs with a tested member within 1/2/5% of
    the oracle, the best ratio to the oracle, the oracle loss if every
    member were dropped, the cells it wins / is picked in, the whitelists
    that hold it, and per member solution its wins and picks. GEMMs whose
    oracle is outside the pool count in `n_winner_outside_pool`."""
    pool = e.pool
    S, P = len(pool.signatures), len(pool)
    si = pool.sig_index
    tested = np.zeros(S, dtype=np.int64)
    screened = np.zeros(S, dtype=np.int64)
    wins = np.zeros(S, dtype=np.int64)
    picks = np.zeros(S, dtype=np.int64)
    member_wins = np.zeros(P, dtype=np.int64)
    member_picks = np.zeros(P, dtype=np.int64)
    near = np.zeros((len(NEAR), S), dtype=np.int64)
    best = np.full(S, np.inf)
    unique_loss = np.zeros(S)
    orphaned = np.zeros(S, dtype=np.int64)
    win_cells: Dict[int, Counter] = defaultdict(Counter)
    pick_cells: Dict[int, Counter] = defaultdict(Counter)
    for i, (g, pos) in enumerate(zip(e.gemms, e.pos)):
        p, us, t = _in_pool(g, pos)
        s = si[p]
        ts = np.unique(s[t])
        tested[ts] += 1
        screened[np.setdiff1d(np.unique(s[~t]), ts)] += 1
        ratio = us / g.winner_us
        for j, x in enumerate(NEAR):
            near[j, np.unique(s[t & (ratio <= 1.0 + x)])] += 1
        if t.any():
            np.minimum.at(best, s[t], ratio[t])
        wp = int(e.winner_pos[i])
        if wp >= 0:
            ws = int(si[wp])
            wins[ws] += 1
            member_wins[wp] += 1
            win_cells[ws][g.leaf] += 1
            others = us[t & (s != ws)]
            if others.size:
                unique_loss[ws] += math.log(float(others.min()) / g.winner_us)
            else:
                orphaned[ws] += 1
        if e.served[i]:
            pp = int(e.pick_pos[i])
            picks[si[pp]] += 1
            member_picks[pp] += 1
            pick_cells[int(si[pp])][g.leaf] += 1
    whitelisted = Counter()
    if e.model:
        for _cell, sigs in e.model.whitelists.items():
            for sg in set(sigs):
                whitelisted[tuple(sg)] += 1
    members: Dict[int, List[int]] = defaultdict(list)
    for p in range(P):
        members[int(si[p])].append(p)
    G = max(len(e.gemms), 1)
    rows = []
    for j, sig in enumerate(pool.signatures):
        k = pool.kernels[members[j][0]]
        mem = sorted(
            members[j],
            key=lambda p: (-member_wins[p], -member_picks[p], int(pool.sids[p])),
        )
        rows.append(
            {
                "sig": signature_label(sig),
                "signature": list(sig),
                "mt": f"{k['mt_m']}x{k['mt_n']}",
                "depth_u": int(k["mt_k"]),
                "mi": f"{k['mi_m']}x{k['mi_n']}x{k['mi_k']}",
                "ch": f"{k['cache_hints_a']},{k['cache_hints_b']}",
                "n_members": len(mem),
                "members": [
                    {
                        "sid": int(pool.sids[p]),
                        "wins": int(member_wins[p]),
                        "picks": int(member_picks[p]) if e.engine else None,
                    }
                    for p in mem
                ],
                "tested": int(tested[j]),
                "screened": int(screened[j]),
                "wins": int(wins[j]),
                "picks": int(picks[j]) if e.engine else None,
                "near": [int(near[x, j]) for x in range(len(NEAR))],
                "best_ratio": float(best[j]) if np.isfinite(best[j]) else None,
                "drop_loss": 1.0 - math.exp(-unique_loss[j] / G),
                "orphaned": int(orphaned[j]),
                "whitelisted_cells": whitelisted.get(sig, 0),
                "win_cells": win_cells[j].most_common(3),
                "pick_cells": pick_cells[j].most_common(3),
            }
        )
    return {
        "kernels": rows,
        "n_gemms": len(e.gemms),
        "n_served": int(e.served.sum()),
        "n_solutions": P,
        "n_winner_outside_pool": e.n_winner_outside_pool,
    }


def pruning_candidates(stats: Mapping[str, Any]) -> Dict[str, Any]:
    """Kernel signatures that never win and are never picked, safest first
    (fewest tested near-ties to the oracle), and the signatures without any
    measurement (no evidence either way)."""
    rows = stats["kernels"]
    never = [
        r
        for r in rows
        if r["wins"] == 0 and not r["picks"] and (r["tested"] or r["screened"])
    ]
    never.sort(
        key=lambda r: (
            r["near"][2],
            r["near"][1],
            r["near"][0],
            r["best_ratio"] is None,
            -(r["best_ratio"] or 0.0),
            -r["tested"],
            r["signature"],
        )
    )
    unmeasured = [r for r in rows if not r["tested"] and not r["screened"]]
    return {
        "never": never,
        "unmeasured": unmeasured,
        "n_pool": len(rows),
        "n_solutions": stats.get("n_solutions"),
        "picks_known": any(r["picks"] is not None for r in rows),
    }


def _param_table(
    e: Evaluated, fns: Mapping[str, Callable[[Mapping[str, Any]], str]], per: str
) -> Dict[str, Any]:
    """Per parameter and value: kernels with it (`per` = "signature" counts
    signatures, else solutions), wins, picks, and the oracle loss over all
    GEMMs if every kernel with that value were removed (GEMMs left without a
    tested pool kernel are counted, not scored). `by_cell` holds wins /
    picks per leaf."""
    pool = e.pool
    G = max(len(e.gemms), 1)
    out: Dict[str, Any] = {"params": {}, "by_cell": {}, "counted": per}
    for name, fn in fns.items():
        vals = [fn(k) for k in pool.kernels]
        arr = np.array(vals, dtype=object)
        if per == "signature":
            n_kern = Counter(v for _, v in set(zip(pool.sigs, vals)))
        else:
            n_kern = Counter(vals)
        wins: Counter = Counter()
        picks: Counter = Counter()
        loss: Dict[str, float] = defaultdict(float)
        stranded: Counter = Counter()
        cell_w: Dict[str, Counter] = defaultdict(Counter)
        cell_p: Dict[str, Counter] = defaultdict(Counter)
        for i, (g, pos) in enumerate(zip(e.gemms, e.pos)):
            wp = int(e.winner_pos[i])
            if wp >= 0:
                wv = vals[wp]
                wins[wv] += 1
                cell_w[g.leaf][wv] += 1
                p, us, t = _in_pool(g, pos)
                other = us[t & (arr[p] != wv)]
                if other.size:
                    loss[wv] += math.log(float(other.min()) / g.winner_us)
                else:
                    stranded[wv] += 1
            if e.served[i]:
                pv = vals[int(e.pick_pos[i])]
                picks[pv] += 1
                cell_p[g.leaf][pv] += 1
        rows = [
            {
                "value": v,
                "kernels": n_kern[v],
                "wins": wins.get(v, 0),
                "picks": picks.get(v, 0) if e.engine else None,
                "drop_loss": 1.0 - math.exp(-loss.get(v, 0.0) / G),
                "stranded": stranded.get(v, 0),
            }
            for v in n_kern
        ]
        rows.sort(key=lambda r: (-r["wins"], -(r["picks"] or 0), r["value"]))
        out["params"][name] = rows
        out["by_cell"][name] = {
            c: {
                v: [cell_w[c].get(v, 0), cell_p[c].get(v, 0)]
                for v in set(cell_w[c]) | set(cell_p[c])
            }
            for c in set(cell_w) | set(cell_p)
        }
    return out


def parameter_stats(e: Evaluated, *, diagnostics: bool = False) -> Dict[str, Any]:
    """`_param_table` of the common GPU / GEMM parameters (macro tile,
    depthU, matrix instruction, cache hints; kernels counted as signatures),
    and with `diagnostics` of the framework-specific ones
    (`diagnostic_params`; counted as solutions) under "diagnostics"."""
    out = _param_table(e, COMMON_PARAMS, "signature")
    out["diagnostics"] = (
        _param_table(e, diagnostic_params(e.pool), "solution") if diagnostics else None
    )
    return out


def cell_kernel_usage(e: Evaluated, top: int = 5) -> Dict[str, Dict[str, Any]]:
    """Per leaf: distinct winning and picked signatures and the most frequent
    of each, and how often the pick is the oracle's solution / signature."""
    w: Dict[str, Counter] = defaultdict(Counter)
    p: Dict[str, Counter] = defaultdict(Counter)
    same_sid: Counter = Counter()
    same_sig: Counter = Counter()
    si = e.pool.sig_index
    for i, g in enumerate(e.gemms):
        wp = int(e.winner_pos[i])
        if wp >= 0:
            w[g.leaf][int(si[wp])] += 1
        if e.served[i]:
            pp = int(e.pick_pos[i])
            p[g.leaf][int(si[pp])] += 1
            same_sid[g.leaf] += int(pp == wp)
            same_sig[g.leaf] += int(wp >= 0 and bool(si[pp] == si[wp]))
    label = [signature_label(s) for s in e.pool.signatures]
    out = {}
    for c in sorted(set(w) | set(p)):
        out[c] = {
            "n_winners": len(w[c]),
            "n_picks": len(p[c]) if e.engine else None,
            "top_winners": [(label[k], n) for k, n in w[c].most_common(top)],
            "top_picks": [(label[k], n) for k, n in p[c].most_common(top)],
            "pick_is_winner": same_sid[c] if e.engine else None,
            "pick_has_winner_signature": same_sig[c] if e.engine else None,
        }
    return out


# ── whole run ────────────────────────────────────────────────────────────────


def model_blocklist(model: Optional[ModelInfo]) -> Optional[Any]:
    """The blocklist of the model's config (lib/correctness.py). One that
    cannot be read becomes a note, and the model's config then reads the
    pool whole."""
    if model is None or model.library_cfg is None:
        return None
    try:
        return config_blocklist(model.library_cfg)
    except ValueError as e:
        model.notes.append(
            f"the run's blocklist cannot be read ({e}); no kernel is left out of "
            f"the analysis"
        )
        model.library_cfg = {
            k: v for k, v in model.library_cfg.items() if k != "correctness"
        }
        return None


def library_pool(model: Optional[ModelInfo]) -> Optional[List[Dict[str, Any]]]:
    if model is None or model.library is None:
        return None
    lib_dir, stem = model.library
    try:
        return ev.library_pool(lib_dir, stem, model.library_cfg, model.library_device)
    except (OSError, ValueError, RuntimeError) as e:
        model.notes.append(
            f"library pool unreadable ({type(e).__name__}); using the measured kernels"
        )
        return None


def pool_composition(
    model: Optional[ModelInfo],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """(`lib.library_pool.summary` of the pool the model ranks, without the
    kernels of its blocklist; None and why when the run names no library or
    the pool cannot be read)."""
    from . import library_pool

    if model is None or model.library is None:
        return None, "the run's bundle names no library"
    lib_dir, stem = model.library
    try:
        whole = library_pool.config_pool(
            model.library_cfg,
            lib_dir,
            stem,
            model.library_device,
            apply_blocklist=False,
        )
    except (OSError, ValueError, RuntimeError) as e:
        return None, f"the kernel pool cannot be read ({type(e).__name__}: {e})"
    pool = library_pool.without_blocked(whole, model_blocklist(model))
    return library_pool.summary(pool), None


def library_identity(model: Optional[ModelInfo]) -> Optional[ki.LibraryIdentity]:
    """The kernel identity (lib/kernel_identity.py) of the model's library
    files, which bench folders and plans are mapped to; None when the run
    names no library or it cannot be read."""
    if model is None or model.library is None:
        return None
    lib_dir, stem = model.library
    try:
        return ev.library_identity(
            lib_dir, stem, model.library_cfg, model.library_device
        )
    except (OSError, ValueError, RuntimeError) as e:
        model.notes.append(
            f"library kernel identities unreadable ({type(e).__name__}); bench "
            f"folders are read by their solution indices"
        )
        return None


def engine_module(model: Optional[ModelInfo]) -> Any:
    if model is None or model.data is None or model.hardware is None:
        return None
    try:
        return ev.tilewright_module()
    except RuntimeError as e:
        model.notes.append(str(e))
        return None


def bundle_tree(run_dir: Path, model: Optional[ModelInfo]) -> Dict[str, SplitRule]:
    """The split tree of `model`'s bundle: the splits of the rounds up to its
    training round (a later round's splits belong to that round's bundle),
    or of the rounds it trained on (`ModelInfo.trained_rounds`)."""
    rounds = round_dirs(run_dir)
    if model is not None and model.trained_rounds is not None:
        every = set(model.trained_rounds)
        return load_cumulative_split_tree([rd for rd in rounds if rd.name in every])
    names = [rd.name for rd in rounds]
    t = model.train_round if model is not None else None
    return load_cumulative_split_tree(
        rounds[: names.index(t) + 1] if t in names else rounds
    )


def analyze_run(
    run_dir: Path,
    *,
    jobs: int = 1,
    use_engine: bool = True,
    pick_measurement: str = CONFIG_PICK_MEASUREMENT,
    diagnostics: bool = False,
    staged: bool = False,
    identity: bool = False,
    audit_dir: Optional[Path] = None,
    aggregation: str = repeats.DEFAULT_AGGREGATION,
) -> Optional[Dict[str, Any]]:
    """Every bench-level section of the report under the bundle the run
    shipped (`final_model`; with `staged`, the latest staged bundle when none
    was), its GEMMs routed to the leaves of its own split tree: the training
    benches it learned from (the top level of the result), and, when
    present, the benches it never trained on (`held_out`, its deploy-gate
    slice alone in `gate`, the held-out execution contexts in
    `holdout_contexts`), each with its decomposition per leaf and per
    execution context; with `identity`, `identity_summary` under
    "identity". None without training benches."""
    run_dir = Path(run_dir)
    model = final_model(run_dir, want_bytes=use_engine, staged=staged)
    blocklist = model_blocklist(model)
    ident = library_identity(model)
    sets = model_benches(run_dir, model, identity=ident)
    if not sets["training"]:
        return None
    tree = bundle_tree(run_dir, model)
    lib = library_pool(model)
    tw = engine_module(model) if use_engine else None

    def run(bs: BenchSet) -> Tuple[Evaluated, Bench]:
        bench = load_bench(
            bs.dirs,
            tree=tree,
            jobs=jobs,
            plans=bs.plans,
            identity=ident,
            aggregation=aggregation,
            blocklist=blocklist,
            context_plans=bs.context_plans,
        )
        if model is not None:
            model.notes.extend(n for n in bench.notes if n not in model.notes)
        e = evaluate(
            bench,
            pool_for(bench, lib),
            model,
            tw=tw,
            pick_measurement=pick_measurement,
        )
        return e, bench

    e, bench = run(sets["training"])
    out = summarize_evaluation(e, bench, diagnostics=diagnostics)
    out["sources"] = {k: [str(d) for d in bs.dirs] for k, bs in sets.items() if bs.dirs}
    for kind in ("held_out", "gate", "holdout_contexts"):
        if not sets[kind]:
            continue
        ek, bk = run(sets[kind])
        out[kind] = {
            "n_gemms": len(ek.gemms),
            "n_served": int(ek.served.sum()),
            "bench": dict(bk.stats, n_sources=len(bk.sources)),
            "decomposition": decomposition(ek),
            "contexts": context_decomposition(ek),
        }
    if identity:
        out["identity"] = identity_summary(
            run_dir,
            model,
            lib if lib is not None else e.pool.kernels,
            pool_source="library" if lib is not None else "bench",
            audit_dir=audit_dir,
            jobs=jobs,
            ident=ident,
        )
    return out


# ── kernel identity ──────────────────────────────────────────────────────────


def key_members(
    kernels: Sequence[Mapping[str, Any]], grouping: Callable[[Mapping[str, Any]], Any]
) -> Dict[str, Any]:
    """Solutions per key of `grouping` over `kernels`: keys, the largest and
    mean number of members, and the keys (and their solutions) with more
    than one member."""
    counts = Counter(grouping(k) for k in kernels)
    shared = [v for v in counts.values() if v > 1]
    return {
        "n_solutions": len(kernels),
        "n_keys": len(counts),
        "max_members": max(counts.values(), default=0),
        "mean_members": len(kernels) / len(counts) if counts else None,
        "n_shared_keys": len(shared),
        "n_shared_solutions": sum(shared),
    }


def exhaustive_benches(
    run_dir: Path,
    audit_dir: Optional[Path] = None,
    current: Optional[Mapping[int, str]] = None,
) -> Tuple[Optional[str], List[Tuple[Path, Optional[Any]]], Optional[str]]:
    """(name, [(enriched folder, the `lib.audit.GenerationMap` of its
    generation or None)], why the audit set is not used) of the
    exhaustively benched GEMMs of a run: the audit set (`lib.audit`) when it
    is complete and covers the pool `current` ({solution: digest}), its
    generations mapped to that pool's indices by kernel digest, else round
    0's seed bench (the planner's exhaustive bootstrap)."""
    from . import audit

    why = None
    if audit_dir is None:
        why = "no audit set is configured (audit.dir)"
    else:
        st, why = audit.status(Path(audit_dir), current)
        if st == "ok":
            try:
                gens = audit.generation_dirs(
                    Path(audit_dir), audit.manifest(Path(audit_dir)) or {}, current
                )
            except ValueError as e:
                gens, why = [], str(e)
            if gens:
                dirs = [(d, gmap) for gdirs, gmap in gens for d in gdirs]
                return "the audit set", dirs, None
    try:
        seed = run_layout.plan_dir(Path(run_dir) / "round_0") / "seed" / "stage04"
    except ValueError:
        return None, [], why
    if _has_chunks(seed):
        return "round 0's exhaustive seed bench", [(seed, None)], why
    return None, [], why


def gemm_dicts(
    dirs: Sequence[Tuple[Path, Optional[Any]]],
    *,
    jobs: int = 1,
    identity: Optional[ki.LibraryIdentity] = None,
) -> List[Dict[str, Any]]:
    """GEMM dicts with candidates (as stage05 loads them) of the folders of
    `exhaustive_benches`: an audit generation's rows at the indices its
    `GenerationMap` gives them (a kernel benched again by a later
    generation keeps only that one's rows), another folder's mapped with
    `identity` (`load_bench`), and each solution its timed row before a
    warm-up row."""
    merged: Dict[Key, Dict[str, Any]] = {}
    for d, gmap in dirs:
        bench = load_bench([d], jobs=jobs, identity=identity if gmap is None else None)
        for g in bench.gemms:
            cands = []
            for s, u, k in zip(g.sids, g.us, g.skip):
                sid = int(s) if gmap is None else gmap.index(int(s))
                if sid is not None:
                    cands.append(
                        {"sol_idx_global": sid, "us": float(u), "is_skip": bool(k)}
                    )
            row = merged.get(g.key)
            if row is None:
                merged[g.key] = dict(g.row, candidates=cands)
            else:
                row["candidates"].extend(cands)
    out = []
    for row in merged.values():
        best: Dict[int, Dict[str, Any]] = {}
        for c in row["candidates"]:
            prev = best.get(c["sol_idx_global"])
            if prev is None or (prev["is_skip"], prev["us"]) > (c["is_skip"], c["us"]):
                best[c["sol_idx_global"]] = c
        if not best:
            continue
        row["candidates"] = list(best.values())
        tested = [c["us"] for c in row["candidates"] if not c["is_skip"]]
        row["winner_us"] = min(tested or [c["us"] for c in row["candidates"]])
        out.append(row)
    return out


def identity_summary(
    run_dir: Path,
    model: Optional[ModelInfo],
    kernels: Sequence[Mapping[str, Any]],
    *,
    pool_source: str = "library",
    audit_dir: Optional[Path] = None,
    jobs: int = 1,
    ident: Optional[ki.LibraryIdentity] = None,
) -> Dict[str, Any]:
    """What a kernel's identity costs in this run. `kernels` is the pool in
    runtime order (library table, then bank). The engine scores a key of
    the v3 grouping (signature, the attribute spec's reads, and the launch
    key when the model serves launch keys) once and serves its first member
    in pool order that the problem admits. Per grouping, `signature` (no
    launch keys) and `launch` (`train.launch_keys`), the solutions per key
    over the pool and the identity ceiling (`lib.identity_ceiling`) on the
    exhaustively benched GEMMs (`exhaustive_benches`, their indices those
    of `ident`'s library), restricted to the pool; `served` names the
    grouping the model serves."""
    from . import audit
    from . import hipblaslt_launch
    from . import identity_ceiling as ic

    spec = tuple(model.attribute_spec) if model is not None else ()
    launch_keys = bool(model.launch_keys) if model is not None else False
    launch = None
    if model is not None and model.library:
        launch = hipblaslt_launch.HipblasltLaunch.for_library(model.library[1])
    elif model is not None and model.arch:
        launch = hipblaslt_launch.HipblasltLaunch(model.arch)
    groupings = {"signature": ic.engine_grouping(spec, None)}
    if launch is not None:
        groupings["launch"] = ic.engine_grouping(spec, launch)
    out: Dict[str, Any] = {
        "fields": list(fs.SIGNATURE_FIELDS),
        "attribute_reads": [n for n, _ in fs.attribute_reads(spec)],
        "launch_keys": launch_keys,
        "served": "launch" if launch_keys else "signature",
        "pool_source": pool_source,
        "members": {k: key_members(kernels, g) for k, g in groupings.items()},
        "ceiling": {},
        "gemms": None,
        "n_gemms": 0,
        "not_audit": None,
    }
    current = None
    if audit_dir is not None and model is not None and model.library:
        try:
            from .bank_pool import config_pool

            current = audit.pool_digests(
                config_pool(
                    model.library_cfg, *model.library, model.library_device
                ).kernels
            )
        except (OSError, ValueError, RuntimeError):
            current = None
    name, dirs, why = exhaustive_benches(run_dir, audit_dir, current)
    out["gemms"], out["not_audit"] = name, why
    if not dirs:
        return out
    library = ev.pool_from_kernels(kernels)
    gemms, _, _ = ev.restrict_to_pool(
        gemm_dicts(dirs, jobs=jobs, identity=ident), library
    )
    out["n_gemms"] = len(gemms)
    if not gemms:
        return out
    pools = ev.pools_for(gemms, library)
    for k, g in groupings.items():
        out["ceiling"][k] = ic.ceiling(gemms, g, k, pools).to_dict()
    return out


def summarize_evaluation(
    e: Evaluated, bench: Bench, *, diagnostics: bool = False
) -> Dict[str, Any]:
    stats = kernel_stats(e)
    m = e.model
    return {
        "model": (
            {
                "round": m.round,
                "train_round": m.train_round,
                "deploy_round": m.deploy_round,
                "deployed": m.deployed,
                "source": m.source,
                "n_cells": len(m.labels),
                "non_temporal_rule": m.non_temporal_rule,
                "version": m.version,
                "rules": serving_rules(m),
                "cascade_keys": m.cascade_keys,
                "launch_keys": m.launch_keys,
                "weight_dtype": m.weight_dtype,
                "notes": list(m.notes),
            }
            if m
            else None
        ),
        "engine": e.engine,
        "pick_measurement": e.pick_measurement,
        "pool": {
            "source": e.pool.source,
            "size": len(e.pool),
            "signatures": len(e.pool.signatures),
        },
        "bench": dict(bench.stats, n_sources=len(bench.sources)),
        "n_gemms": len(e.gemms),
        "n_served": int(e.served.sum()),
        "n_inconsistent": e.n_inconsistent,
        "n_out_of_pool_rows": e.n_out_of_pool_rows,
        "n_without_pool_kernel": e.n_without_pool_kernel,
        "n_winner_outside_pool": e.n_winner_outside_pool,
        "decomposition": decomposition(e),
        "contexts": context_decomposition(e),
        "kernels": stats,
        "pruning": pruning_candidates(stats),
        "parameters": parameter_stats(e, diagnostics=diagnostics),
        "cell_usage": cell_kernel_usage(e),
    }


def workload_keep(
    wf: Mapping[str, Any],
) -> Optional[Callable[[Mapping[str, Any]], bool]]:
    """GEMM filter of a stage08 `workload_filter` block (normalized values)."""
    want = {k: str(v) for k, v in (wf or {}).items() if v not in (None, "")}
    if not want:
        return None

    def norm(k: str, v: Any) -> str:
        s = str(v).strip()
        return s.upper() if k in ("transA", "transB") else s.lower()

    want = {k: norm(k, v) for k, v in want.items()}
    return lambda row: all(norm(k, row.get(k, "")) == v for k, v in want.items())


def analyze_dataset(
    enriched: Path,
    *,
    workload_filter: Mapping[str, Any],
    pool: Pool,
    model: Optional[ModelInfo],
    tree: Mapping[str, SplitRule],
    tw: Any,
    jobs: int = 1,
    pick_measurement: str = CONFIG_PICK_MEASUREMENT,
    aggregation: str = repeats.DEFAULT_AGGREGATION,
) -> Optional[Dict[str, Any]]:
    """Decomposition of one validation dataset (report only), with stage 8's
    restriction of the rows to the library pool."""
    bench = load_bench(
        [enriched],
        tree=tree,
        jobs=jobs,
        keep=workload_keep(workload_filter),
        aggregation=aggregation,
    )
    if not bench.gemms:
        return None
    e = evaluate(
        bench,
        pool,
        model,
        tw=tw,
        pick_measurement=pick_measurement,
        restrict_to_pool=True,
    )
    return {
        "n_gemms": len(e.gemms),
        "n_served": int(e.served.sum()),
        "engine": e.engine,
        "decomposition": decomposition(e),
    }


def read_per_gemm(path: Path) -> Dict[str, Dict[str, Any]]:
    """Per leaf of a stage08 per_gemm.csv: GEMMs, those the model served,
    its misses, and the selection efficiency of the model's own picks."""
    acc: Dict[str, List[float]] = defaultdict(list)
    count: Counter = Counter()
    misses: Counter = Counter()
    try:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                leaf = r.get("leaf") or "(none)"
                count[leaf] += 1
                if r.get("served_by") != "model":
                    continue
                try:
                    w, p = float(r["winner_us"]), float(r["pick_us"])
                except (KeyError, TypeError, ValueError):
                    continue
                if w > 0 and p > 0:
                    acc[leaf].append(math.log(p / w))
                    misses[leaf] += bool(r.get("miss"))
    except OSError:
        return {}
    out = {}
    for leaf in count:
        xs = acc.get(leaf, [])
        out[leaf] = {
            "n_gemms": count[leaf],
            "n_served": len(xs),
            "n_miss": misses.get(leaf, 0),
            "sel_eff": math.exp(-sum(xs) / len(xs)) if xs else None,
            "_logs": xs,
        }
    return out


def stage08_enriched_dir(stage08: Path, ds: Mapping[str, Any]) -> Optional[Path]:
    """The enriched chunks a stage08 dataset was evaluated on."""
    p = Path(str(ds.get("path") or ""))
    if chunk_files([p]):
        return p
    q = stage08 / str(ds.get("name") or "") / "stage04_enriched"
    return q if chunk_files([q]) else None
