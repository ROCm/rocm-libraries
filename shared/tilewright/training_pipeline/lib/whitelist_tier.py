# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The whitelist tier of an MLREC_v3 bundle (`train.whitelist_tier_k`).

After training, every cell of the bundle gets K view keys that the engine
scores first (tier 1 with the model's `whitelist_tier` serving flag), chosen
on the training GEMMs that route to the cell:

  GEMMs    the enriched data stage05 trained from, read as stage05 reads it
           (`run_analysis.load_bench` with the run's library identity,
           repeat aggregation and blocklist, keeping the rows of the pool's
           kernels alone: `universe` below), every leaf's training side as
           stage05's `split_validation` splits it (`training_gemms`), each
           routed by the bundle itself;
  scores   the deployed model's: the bundle packed at the deploy weight
           dtype with cascade 0 and no tier (`scoring_bundle`), so that the
           engine gives every key it serves its full score under the
           bundle's serving rules and hipBLASLt's launch plans, over the pool
           the runtime ranks: the library's table, then the bank, without
           the kernels the config's blocklist names (`universe`);
  cost     a key's latency in a GEMM is that of its first pool kernel with a
           row (warm-up rows count); a key without a row, or that the engine
           does not score (infeasible under the serving rules), does not
           cover the GEMM;
  keys     per cell, stage05's smart_K greedy (`select_smart_k_signatures`
           over a `ScoreTable`: the key covering the most GEMMs not yet
           covered, then the key that most improves the mean log efficiency
           of the model's picks) takes K keys; a cell no training GEMM
           reaches gets none.

The execution contexts of a cell's GEMMs (exclusive use, CU budgets,
schedules) count as `train.whitelist_tier_contexts` says (`CONTEXT_MODES`):

  pooled   every GEMM alike, whatever its context;
  balanced each context the cell's GEMMs run in weighs the same in all:
           the greedy covers the most uncovered GEMMs of every context in
           proportion to that context's GEMMs, then improves the sum over
           the contexts of their mean log efficiency;
  reserve  the cell's K keys are taken on its exclusive GEMMs (on all of
           them when it has none), then up to RESERVE_KEYS more on its other
           GEMMs, each covering one of them or improving their mean log
           efficiency.

A cell whose GEMMs share one context gets the pooled tier in every mode.

The greedy serves a cell's mean, so a key that only a few of its GEMMs
need, such as the winners of a context its GEMMs rarely run in, can drop
out while the model's pick from the rest takes more than twice the
winner's time there. The tail guard (`guard`) then keeps such a key: on a
GEMM of the cell's own data whose pick from the tier falls below
GUARD_FLOOR of its winner, the fastest key within GUARD_NEAR of the winner
that the model scores above that pick joins the tier, when on the cell's
GEMMs, its own and those stage05 held out of its training alike, the tier
with it and the keys kept before it adds no new pick below the floor to
the tier the greedy chose, undoes no rescue of those keys and keeps
GUARD_MEAN_FLOOR of the held-out picks' geomean (`guard_verdict`); at most
`guard_keys` per cell, each GEMM and context counting on its own, and only
for collapses no key kept before it lifted.

The weights do not change, so the tier can be built again for another pool
without training (scripts/build_whitelist_tier.py). A tier names view keys,
never solution indices: the engine scores the tier keys a pool holds and
serves a problem none of whose tier keys is in its pool or feasible through
the cascade over every feasible key, as the model without the tier would.
"""
from __future__ import annotations

import hashlib
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence
from typing import Set, Tuple

import numpy as np

from . import bank_pool
from . import evaluate as ev
from . import features as fs
from . import hardware as hwlib
from . import mlrec
from . import procs
from . import shipped_subset as ss
from .stage05_split import stage05_module

TIER_FILE = "whitelist_tier.json"
# models.pt key of the tier's record; a bundle without it holds no tier.
RECORD = "whitelist_tier"
# How the tier counts the execution contexts of a cell's GEMMs; a record
# without `contexts` predates the choice and is pooled.
CONTEXT_MODES: Tuple[str, ...] = ("pooled", "balanced", "reserve")
DEFAULT_CONTEXTS = "pooled"
# The keys `reserve` may add to a cell's K for its GEMMs outside exclusive use.
RESERVE_KEYS = 8
_CHUNK = 256


def has_tier(bundle: Mapping[str, Any]) -> bool:
    return bool(bundle.get(RECORD))


def smart_k_cells(bundle: Mapping[str, Any]) -> List[str]:
    """Cells of `bundle` with a smart_K whitelist (`--whitelist signature`
    or `view-key`), which a tier would replace."""
    keyed = not has_tier(bundle)
    return sorted(
        label
        for label, e in (bundle.get("models") or {}).items()
        if e.get("smart_k_signatures") or (keyed and e.get("view_key_hashes"))
    )


def refusal(bundle: Mapping[str, Any]) -> Optional[str]:
    """Why `bundle` cannot get a tier, else None."""
    if str(bundle.get("catalog") or "v2") != "v3":
        return "the whitelist tier needs an MLREC_v3 bundle (train.catalog v3)"
    if (bundle.get("serving") or {}).get("launch_keys"):
        return (
            "the whitelist tier lists view keys; a model with launch keys "
            "scores each launch class of a view key on its own"
        )
    cells = smart_k_cells(bundle)
    if cells:
        return (
            f"{len(cells)} cells have a smart_K whitelist (train.whitelist "
            f"signature or view-key), e.g. {cells[0]}; the tier replaces it: "
            f"train with train.whitelist none"
        )
    return None


def _without_keys(entry: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(entry)
    out.pop("view_key_hashes", None)
    return out


def strip(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    """`bundle` without its tier (unchanged when it has none)."""
    if not has_tier(bundle):
        return dict(bundle)
    out = {k: v for k, v in bundle.items() if k != RECORD}
    out["models"] = {
        label: _without_keys(e) for label, e in (bundle.get("models") or {}).items()
    }
    out["serving"] = dict(bundle.get("serving") or {}, whitelist_tier=False)
    return out


def scoring_bundle(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    """`bundle` served with cascade 0 and no whitelist: the engine gives every
    key it serves its full score."""
    out = strip(bundle)
    out["models"] = {
        label: _without_keys(e) for label, e in (out.get("models") or {}).items()
    }
    out["serving"] = dict(
        out.get("serving") or {}, cascade_keys=0, whitelist_tier=False
    )
    return out


def view_key_hashes(
    kernels: Iterable[Mapping[str, Any]], spec: fs.AttributeSpec
) -> List[int]:
    """The view-key hash of every kernel of a pool."""
    return [
        fs.view_key_hash(
            fs.v3_view_key(fs.config_kwargs_from_row(k), k.get("attributes"), spec)
        )
        for k in kernels
    ]


def keys_digest(hashes: Iterable[int]) -> str:
    """Digest of a pool's distinct view keys: equal pools of keys, equal
    digests, whatever their solution indices."""
    h = hashlib.sha256()
    for v in sorted({int(x) & 0xFFFFFFFFFFFFFFFF for x in hashes}):
        h.update(v.to_bytes(8, "little"))
    return h.hexdigest()[:16]


def universe(
    cfg: Optional[Mapping[str, Any]],
    library_dir: Any,
    library_stem: str,
    device_index: int,
    spec: fs.AttributeSpec,
) -> ss.Universe:
    """The pool the runtime ranks with the model: the library's table, then
    the bank the config asks for, without the kernels its blocklist names
    (`lib.bank_pool.config_pool`), grouped by view key."""
    lp = bank_pool.config_pool(cfg, library_dir, library_stem, device_index)
    return ss.build_universe(
        ev.pool_from_kernels(lp.table), ev.pool_from_kernels(lp.bank.kernels), spec
    )


def split_gemms(
    gemms: Sequence[Any],
    tree: Optional[Mapping[str, Any]],
    validation_frac: float,
    seed: int,
    s5: Any = None,
) -> Tuple[List[Any], List[Any]]:
    """(training side, held-out side) of `gemms` (`run_analysis.Gemm`): per
    leaf of the split tree `tree` (stage05's routing), the GEMMs stage05's
    `split_validation` keeps for training and those it holds out (its
    validation GEMMs, which it never trains on), the leaves in the order of
    their first GEMM and each leaf's GEMMs in their order. Leaves stage05
    did not train count too: the bundle serves their GEMMs with a model of
    another cell."""
    s5 = s5 or stage05_module()
    by_leaf: Dict[str, List[Any]] = {}
    for g in gemms:
        m, n, k, b = (int(v) for v in g.key[:4])
        by_leaf.setdefault(s5._leaf_of(m, n, k, b, dict(tree or {})), []).append(g)
    train_out: List[Any] = []
    held_out: List[Any] = []
    for group in by_leaf.values():
        rows = [g.row for g in group]
        train, _val = s5.split_validation(rows, validation_frac, seed)
        kept = {id(r) for r in train}
        for g, r in zip(group, rows):
            (train_out if id(r) in kept else held_out).append(g)
    return train_out, held_out


def training_gemms(
    gemms: Sequence[Any],
    tree: Optional[Mapping[str, Any]],
    validation_frac: float,
    seed: int,
    s5: Any = None,
) -> List[Any]:
    """The training side of `gemms` (`split_gemms`)."""
    return split_gemms(gemms, tree, validation_frac, seed, s5)[0]


def leaf_counts(
    gemms: Sequence[Any],
    tree: Optional[Mapping[str, Any]],
    validation_frac: float,
    seed: int,
    s5: Any = None,
) -> Dict[str, Tuple[int, int]]:
    """{leaf: (training, validation) GEMMs} of `gemms` as stage05 splits
    them, to compare with the counts its cells.json records."""
    s5 = s5 or stage05_module()
    by_leaf: Dict[str, List[Any]] = {}
    for g in gemms:
        m, n, k, b = (int(v) for v in g.key[:4])
        by_leaf.setdefault(s5._leaf_of(m, n, k, b, dict(tree or {})), []).append(g)
    out: Dict[str, Tuple[int, int]] = {}
    for leaf, group in by_leaf.items():
        tr, va = s5.split_validation([g.row for g in group], validation_frac, seed)
        out[leaf] = (len(tr), len(va))
    return out


def served_context(context: fs.ExecutionContext, n_cu: int) -> fs.ExecutionContext:
    """`context` as the engine serves it on a device of `n_cu` CUs: a budget
    of at least every CU is no budget."""
    if 0 < n_cu <= context.cu_budget:
        return fs.ExecutionContext(cu_budget=0, schedule=context.schedule)
    return context


def context_name(context: fs.ExecutionContext) -> str:
    """How the tier's records name an execution context: "exclusive", else
    "cu<budget>" ("allcu" for every CU) with a schedule other than the
    default appended."""
    if context.is_exclusive():
        return "exclusive"
    name = f"cu{context.cu_budget}" if context.cu_budget else "allcu"
    return name if context.schedule == "default" else f"{name}_{context.schedule}"


# ── scoring ──────────────────────────────────────────────────────────────────

_W: Dict[str, Any] = {}


def _init_worker(
    parent: Optional[int],
    data: bytes,
    kernels: Sequence[Mapping[str, Any]],
    hardware: hwlib.DeviceHardware,
    key_of_pos: np.ndarray,
    pos_of_sid: Mapping[int, int],
) -> None:
    if parent is not None:
        procs.die_with(parent)
    tw = ev.tilewright_module()
    model = ev.load_model_bytes(data, tw)
    _W.clear()
    _W.update(
        tw=tw,
        model=model,
        cs=ev.candidate_set(tw, model, kernels, hardware),
        hw=ev.make_hardware(tw, hardware),
        key_of_pos=key_of_pos,
        pos_of_sid=pos_of_sid,
        contexts={},
    )


def _score_chunk(
    jobs: Sequence[Tuple[Dict[str, Any], fs.ExecutionContext, np.ndarray, np.ndarray]],
) -> List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Per GEMM: (serving cell index, the keys it measured that the engine
    scores, their costs, their full scores)."""
    w = _W
    tw, model, cs, hw = w["tw"], w["model"], w["cs"], w["hw"]
    key_of_pos, pos_of_sid = w["key_of_pos"], w["pos_of_sid"]
    out = []
    for prob, context, sids, us in jobs:
        ctx = w["contexts"].get(context)
        if ctx is None:
            ctx = w["contexts"][context] = ev.make_context(tw, context)
        p = ev.make_problem(tw, prob)
        cell = int(tw.route(model, p))
        score: Dict[int, float] = {}
        for r in cs.rank(p, hw, 1, ctx):
            if not r.scored:
                break
            score.setdefault(int(key_of_pos[int(r.config_index)]), float(r.score))
        first: Dict[int, int] = {}
        cost: Dict[int, float] = {}
        for sid, u in zip(sids.tolist(), us.tolist()):
            pos = pos_of_sid.get(int(sid))
            if pos is None:
                continue
            key = int(key_of_pos[pos])
            if key not in first or pos < first[key]:
                first[key], cost[key] = pos, float(u)
        keys = [k for k in cost if k in score and math.isfinite(score[k])]
        out.append(
            (
                cell,
                np.array(keys, dtype=np.int64),
                np.array([cost[k] for k in keys], dtype=np.float64),
                np.array([score[k] for k in keys], dtype=np.float64),
            )
        )
    return out


def engine_scores(
    data: bytes,
    u: ss.Universe,
    hardware: hwlib.DeviceHardware,
    gemms: Sequence[Any],
    workers: int = 1,
) -> List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """`_score_chunk` of every GEMM of `gemms` under model file `data` over
    the universe `u`, in `workers` spawned processes that end with this
    one."""
    pos_of_sid = {int(s): i for i, s in enumerate(u.sids)}
    jobs = [(fs.problem_kwargs_from_row(g.row), g.context, g.sids, g.us) for g in gemms]
    chunks = [jobs[i : i + _CHUNK] for i in range(0, len(jobs), _CHUNK)]
    args = (data, u.kernels, hardware, u.key, pos_of_sid)
    if workers > 1 and len(chunks) > 1:
        with ProcessPoolExecutor(
            max_workers=min(int(workers), len(chunks)),
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
            initargs=(os.getpid(), *args),
        ) as pool:
            parts = list(pool.map(_score_chunk, chunks))
    else:
        _init_worker(None, *args)
        parts = [_score_chunk(c) for c in chunks]
    return [r for part in parts for r in part]


# ── the tier ─────────────────────────────────────────────────────────────────


def _number(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None


@dataclass
class Tier:
    """`keys`: per cell label, the view-key hashes of its tier in the order
    the greedy took them; `cells`: per cell label, what the tier was chosen
    on and its in-sample efficiency; `record`: what models.pt keeps."""

    k: int
    keys: Dict[str, List[int]] = field(default_factory=dict)
    cells: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    record: Dict[str, Any] = field(default_factory=dict)

    def doc(self) -> Dict[str, Any]:
        """The whitelist_tier.json document."""
        return {
            **self.record,
            "cells": {
                label: {**self.cells[label], "keys": self.keys.get(label, [])}
                for label in sorted(self.cells)
            },
        }


def _table(
    items: Sequence[Tuple[float, Tuple[Any, ...], fs.ExecutionContext]], s5: Any
) -> Tuple[List[int], Any]:
    """(the universe indices of the keys measured in `items`, ascending;
    their stage05 `ScoreTable` over the items' GEMMs in order). An item is a
    GEMM's winner latency, `_score_chunk` result and execution context."""
    keys = sorted({int(x) for _, r, _c in items for x in r[1].tolist()})
    row_of = {x: j for j, x in enumerate(keys)}
    us = np.full((len(keys), len(items)), np.inf)
    score = np.full((len(keys), len(items)), -np.inf)
    for gi, (_w, r, _c) in enumerate(items):
        rows = np.array([row_of[int(x)] for x in r[1].tolist()], dtype=np.int64)
        us[rows, gi] = r[2]
        score[rows, gi] = r[3]
    win = np.array([w for w, _r, _c in items])
    table = s5.ScoreTable([(x,) for x in keys], list(range(len(items))), us, score, win)
    return keys, table


def _smart_k(table: Any, k: int, s5: Any) -> List[int]:
    """The keys stage05's smart_K greedy takes from `table`, k at most."""
    picked = s5.select_smart_k_signatures([], int(k), table=table, with_metrics=False)
    return [int(t[0]) for t in picked["sigs"]]


def _context_weights(contexts: Sequence[fs.ExecutionContext]) -> np.ndarray:
    """Per GEMM, a weight that gives every context of `contexts` (one per
    GEMM) the same total: the least common multiple of the contexts' GEMM
    counts over the GEMM's context's count, so that coverage counts stay
    exact integers; the reciprocal of that count where the multiple outgrows
    float precision."""
    counts: Dict[fs.ExecutionContext, int] = {}
    for c in contexts:
        counts[c] = counts.get(c, 0) + 1
    total = math.lcm(*counts.values())
    if total * len(counts) >= 1 << 53:
        return np.array([1.0 / counts[c] for c in contexts])
    return np.array([total // counts[c] for c in contexts], dtype=np.int64)


def _greedy(
    table: Any,
    k: int,
    weight: Optional[np.ndarray] = None,
    start: Sequence[int] = (),
    gain_only: bool = False,
) -> List[int]:
    """Rows of up to `k` keys of `table` (a stage05 `ScoreTable`) taken by
    the greedy of stage05's `_greedy_table` with every GEMM counting by its
    `weight` (all 1, and no `start`: `_greedy_table` itself) after the rows
    `start`: the key covering the most uncovered weight (ties by weighted
    regret gain), then the key that most improves the weighted mean log
    efficiency of the model's picks. With `gain_only` the second step stops
    where that key would not make the picks it changes faster in all (by
    their weighted log latency)."""
    if table.empty:
        return []
    n_keys, n_gemms = table.us.shape
    has, score, log_us, log_win = table.has, table.score, table.log_us, table.log_win
    w = np.ones(n_gemms, dtype=np.int64) if weight is None else np.asarray(weight)
    wf = w.astype(np.float64)
    chosen: List[int] = []
    free = np.ones(n_keys, dtype=bool)
    best = np.full(n_gemms, -np.inf)
    log_pick = np.zeros(n_gemms)
    covered = np.zeros(n_gemms, dtype=bool)

    def take(j: int) -> None:
        upd = has[j] & (score[j] > best)
        best[upd] = score[j][upd]
        log_pick[upd] = log_us[j][upd]
        covered[upd] = True
        free[j] = False

    for j in start:
        take(int(j))
    while len(chosen) < int(k) and not covered.all():
        new = has & ~covered
        better = has & covered & (score > best)
        gain = np.where(new, log_win - log_us, 0.0) + np.where(
            better, log_pick - log_us, 0.0
        )
        n_new = (new * w).sum(axis=1)
        ok = free & (n_new > 0)
        if not ok.any():
            break
        top = ok & (n_new == n_new[ok].max())
        j = int(np.argmax(np.where(top, (gain * wf).sum(axis=1), -np.inf)))
        take(j)
        chosen.append(j)
    while len(chosen) < int(k):
        better = has & (score > best)
        valid = better | covered
        gain = np.where(valid, log_win - np.where(better, log_us, log_pick), 0.0)
        count = (valid * w).sum(axis=1)
        ok = free & (count > 0)
        if not ok.any():
            break
        avg = (gain * wf).sum(axis=1) / np.where(count > 0, count, 1)
        j = int(np.argmax(np.where(ok, avg, -np.inf)))
        if gain_only and not (wf * (log_pick - log_us[j]))[better[j]].sum() > 0:
            break
        take(j)
        chosen.append(j)
    return chosen


def _cell_keys(
    items: Sequence[Tuple[float, Tuple[Any, ...], fs.ExecutionContext]],
    keys: Sequence[int],
    table: Any,
    k: int,
    mode: str,
    s5: Any,
) -> List[int]:
    """The universe indices of the keys of one cell's tier, in the order
    taken, from its GEMMs `items` and their `_table` (`keys`, `table`)."""
    contexts = [c for _w, _r, c in items]
    if mode == "pooled" or len(set(contexts)) < 2:
        return _smart_k(table, k, s5)
    if mode == "balanced":
        return [keys[j] for j in _greedy(table, k, _context_weights(contexts))]
    exclusive = [it for it in items if it[2].is_exclusive()]
    others = [it for it in items if not it[2].is_exclusive()]
    base = _smart_k(_table(exclusive, s5)[1] if exclusive else table, k, s5)
    other_keys, other_table = _table(others, s5)
    row = {x: j for j, x in enumerate(other_keys)}
    extra = _greedy(
        other_table,
        RESERVE_KEYS,
        start=[row[x] for x in base if x in row],
        gain_only=True,
    )
    return base + [other_keys[j] for j in extra]


def _context_order(c: fs.ExecutionContext) -> Tuple[Any, ...]:
    return (not c.is_exclusive(), c.cu_budget == 0, -c.cu_budget, c.schedule)


def _per_context(
    items: Sequence[Tuple[float, Tuple[Any, ...], fs.ExecutionContext]],
    table: Any,
    chosen: Sequence[int],
    s5: Any,
) -> Dict[str, Dict[str, Any]]:
    """Per execution context of one cell's GEMMs `items` (`context_name`):
    their number, those the tier `chosen` covers, and the in-sample
    efficiency of the tier and of every measured key on them."""
    cols: Dict[fs.ExecutionContext, List[int]] = {}
    for gi, (_w, _r, c) in enumerate(items):
        cols.setdefault(c, []).append(gi)
    out: Dict[str, Dict[str, Any]] = {}
    for c in sorted(cols, key=_context_order):
        idx = np.array(cols[c], dtype=np.int64)
        sub = s5.ScoreTable(
            table.keys,
            idx.tolist(),
            table.us[:, idx],
            table.score[:, idx],
            table.winner_us[idx],
        )
        every = s5._evaluate_keys(sub, table.keys)
        on_tier = s5._evaluate_keys(sub, [(x,) for x in chosen])
        out[context_name(c)] = {
            "n_gemms": len(idx),
            "covered_gemms": int(on_tier["n_eval"]),
            "in_sample_sel_eff_all_keys": _number(every["sel_eff"]),
            "in_sample_sel_eff": _number(on_tier["sel_eff"]),
        }
    return out


def choose(
    labels: Sequence[str],
    scored: Sequence[Tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    winner_us: Sequence[float],
    k: int,
    hashes: np.ndarray,
    s5: Any = None,
    contexts: Optional[Sequence[fs.ExecutionContext]] = None,
    mode: str = DEFAULT_CONTEXTS,
) -> Tier:
    """The tier of a model with cells `labels` from the scored training
    GEMMs (`_score_chunk` results, in the GEMMs' order, with their winner
    latencies and execution contexts as served, `served_context`), its keys
    chosen as `mode` (`CONTEXT_MODES`) says; without `contexts`, which only
    pooled may do, every GEMM counts as exclusive. A cell whose GEMMs run in
    more than one context records its in-sample efficiency per context.
    ValueError for an unknown mode, or contexts that are not one per GEMM."""
    if mode not in CONTEXT_MODES:
        raise ValueError(
            f"whitelist tier contexts {mode!r} is not one of {list(CONTEXT_MODES)}"
        )
    if contexts is None and mode != DEFAULT_CONTEXTS:
        raise ValueError(f"a {mode} whitelist tier needs the GEMMs' contexts")
    if contexts is not None and len(contexts) != len(scored):
        raise ValueError(
            f"{len(contexts)} execution contexts for {len(scored)} scored GEMMs"
        )
    s5 = s5 or stage05_module()
    by_cell: Dict[int, List[Tuple[float, Tuple[Any, ...], fs.ExecutionContext]]] = {}
    every_context = contexts if contexts is not None else [fs.EXCLUSIVE] * len(scored)
    for w, r, c in zip(winner_us, scored, every_context):
        if r[0] >= 0:
            by_cell.setdefault(int(r[0]), []).append((float(w), r, c))
    tier = Tier(k=int(k))
    for ci, label in enumerate(labels):
        items = [it for it in by_cell.get(ci, []) if it[0] > 0 and it[1][1].size]
        if not items:
            tier.cells[label] = {"n_gemms": 0}
            continue
        keys, table = _table(items, s5)
        every = s5._evaluate_keys(table, sorted(table.keys, key=lambda t: t[0]))
        chosen = _cell_keys(items, keys, table, int(k), mode, s5)
        on_tier = s5._evaluate_keys(table, [(x,) for x in chosen])
        tier.keys[label] = [int(hashes[x]) for x in chosen]
        tier.cells[label] = {
            "n_gemms": len(items),
            "n_keys_measured": len(keys),
            "n_keys": len(chosen),
            "covered_gemms": int(on_tier["n_eval"]),
            "in_sample_sel_eff_all_keys": _number(every["sel_eff"]),
            "in_sample_sel_eff": _number(on_tier["sel_eff"]),
        }
        by_context = _per_context(items, table, chosen, s5)
        if len(by_context) > 1:
            tier.cells[label]["contexts"] = by_context
    return tier


# ── tail guard ───────────────────────────────────────────────────────────────

# A pick below this fraction of its GEMM's winner is a collapse: the per-cell
# gate's default tail floor (lib/audit.py CELL_GATE_DEFAULTS).
GUARD_FLOOR = 0.5
# A guard key is a near-winner of a GEMM it rescues: within this fraction of
# the GEMM's winner.
GUARD_NEAR = 0.9
# The keys the guard may add to a cell's tier by default (stage05
# --whitelist-tier-guard-keys).
GUARD_KEYS = RESERVE_KEYS
# The guard keys of a cell may cost its held-out picks at most this factor
# of their geomean: the per-cell gate's default tolerance.
GUARD_MEAN_FLOOR = 0.99
# Refused candidates a cell's guard record names.
_GUARD_SHOWN = 4
Item = Tuple[float, Tuple[Any, ...]]


def _pick(r: Tuple[Any, ...], keys: np.ndarray) -> int:
    """The position in the scored GEMM `r` (`_score_chunk`) of the model's
    pick among its measured keys in `keys` (universe indices), the first
    on equal scores; -1 when none of them is measured."""
    idx = np.flatnonzero(np.isin(r[1], keys))
    if not idx.size:
        return -1
    return int(idx[np.argmax(r[3][idx])])


def _efficiency(w: float, cost: float) -> float:
    return min(1.0, float(w) / float(cost)) if cost > 0 else 0.0


def tail_change(
    items: Sequence[Item],
    before: Iterable[int],
    after: Iterable[int],
    floor: float = GUARD_FLOOR,
    current: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    """How the model's picks among each GEMM's measured keys (the tier's
    in-sample convention, `_evaluate_keys`) change on `items` ((winner
    latency, `_score_chunk` result) of one cell's GEMMs) when the cell's
    tier `before` (universe key indices) becomes `after`: the GEMMs both
    tiers cover, the picks that change, the positions in `items` of the new
    picks below `floor` (from at or above it) and of the picks lifted from
    below it to at or above it, and the geomean ratio after / before over
    the covered GEMMs (None without any). With `current`, a tier between
    the two (`before` with keys added since), a pick `current` holds at or
    above the floor that falls below it under `after` is new too, so that a
    key cannot undo what the keys before it lifted. A GEMM `before` does
    not cover is served by the cascade, which this does not model, and is
    left out."""
    b = np.array(sorted({int(x) for x in before}), dtype=np.int64)
    a = np.array(sorted({int(x) for x in after}), dtype=np.int64)
    c = (
        None
        if current is None
        else np.array(sorted({int(x) for x in current}), dtype=np.int64)
    )
    n = changed = 0
    new: List[int] = []
    healed: List[int] = []
    logs = 0.0
    for i, (w, r) in enumerate(items):
        pb = _pick(r, b)
        if pb < 0:
            continue
        pa = _pick(r, a)
        if pa < 0:
            continue
        n += 1
        eb, ea = _efficiency(w, r[2][pb]), _efficiency(w, r[2][pa])
        ec = eb
        if c is not None:
            pc = _pick(r, c)
            ec = _efficiency(w, r[2][pc]) if pc >= 0 else eb
        if pa != pb:
            changed += 1
        if ea < floor and (eb >= floor or ec >= floor):
            new.append(i)
        elif eb < floor <= ea:
            healed.append(i)
        logs += math.log(max(ea, 1e-12)) - math.log(max(eb, 1e-12))
    return {
        "n_gemms": n,
        "n_changed": changed,
        "new": new,
        "healed": healed,
        "geomean_ratio": math.exp(logs / n) if n else None,
    }


def guard_verdict(
    train: Sequence[Item],
    held: Sequence[Item],
    before: Iterable[int],
    after: Iterable[int],
    *,
    floor: float = GUARD_FLOOR,
    mean_floor: float = GUARD_MEAN_FLOOR,
    current: Optional[Iterable[int]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """(keep, record) of keys a cell's tier `before` would gain to become
    `after`, on the cell's own GEMMs (`train`, the side its tier was chosen
    on) and its held-out ones (`held`, never trained on): kept only when no
    pick of either side falls below `floor` from at or above it under
    `before`, or under `current` when given (`tail_change`), and the
    held-out picks keep at least `mean_floor` of their geomean under
    `before` (the training side's when the cell holds out none). Keys added
    one at a time pass `before` as the tier before any of them and
    `current` as the tier with those kept so far, so that the keys together
    keep both bounds and none undoes the rescue of one before it."""
    t = tail_change(train, before, after, floor, current)
    h = tail_change(held, before, after, floor, current)
    mean_side = h if h["n_gemms"] else t
    ratio = mean_side["geomean_ratio"]
    why = []
    if t["new"] or h["new"]:
        why.append(
            f"{len(t['new'])} new pick(s) below {floor:g} on its own GEMMs, "
            f"{len(h['new'])} on its held-out GEMMs"
        )
    if ratio is not None and ratio < mean_floor:
        side = "held-out" if h["n_gemms"] else "own"
        why.append(f"its {side} picks' geomean x{ratio:.4f} < x{mean_floor:g}")

    def brief(c: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "n_gemms": c["n_gemms"],
            "n_changed": c["n_changed"],
            "n_new_below_floor": len(c["new"]),
            "n_healed": len(c["healed"]),
            "geomean_ratio": c["geomean_ratio"],
        }

    rec = {"own": brief(t), "held_out": brief(h), "why_not": "; ".join(why) or None}
    return not why, rec


def _rescues(
    train: Sequence[Item], tier: Sequence[int], floor: float, near: float
) -> Tuple[int, Dict[int, List[int]]]:
    """(the collapses of `train` under the tier `tier`, {rescue key: the
    positions in `train` of the collapses it rescues}): a GEMM whose pick
    from the tier is below `floor` of its winner is a collapse, and its
    rescue is its fastest measured key outside the tier that is within
    `near` of the winner and that the model scores above the tier's pick
    (`guard_cell`); a collapse without one is counted only."""
    in_tier = {int(x) for x in tier}
    t_arr = np.array(sorted(in_tier), dtype=np.int64)
    n = 0
    rescues: Dict[int, List[int]] = {}
    for i, (w, r) in enumerate(train):
        p = _pick(r, t_arr)
        if p < 0 or _efficiency(w, r[2][p]) >= floor:
            continue
        n += 1
        best: Optional[Tuple[float, float, int]] = None
        for j in range(r[1].size):
            key = int(r[1][j])
            if key in in_tier or not r[3][j] > r[3][p]:
                continue
            if _efficiency(w, r[2][j]) < near:
                continue
            cand = (float(r[2][j]), -float(r[3][j]), key)
            if best is None or cand < best:
                best = cand
        if best is not None:
            rescues.setdefault(best[2], []).append(i)
    return n, rescues


def guard_cell(
    train: Sequence[Item],
    held: Sequence[Item],
    tier_keys: Sequence[int],
    *,
    max_keys: int = GUARD_KEYS,
    floor: float = GUARD_FLOOR,
    near: float = GUARD_NEAR,
    mean_floor: float = GUARD_MEAN_FLOOR,
) -> Tuple[List[int], Dict[str, Any]]:
    """(the universe indices of the keys the tail guard adds to a cell's
    tier `tier_keys`, what it found) from the cell's own GEMMs `train` and
    its held-out ones `held` ((winner latency, `_score_chunk` result)).

    A GEMM of `train` whose pick from the tier is below `floor` of its
    winner is a collapse. Its rescue is its fastest measured key outside
    the tier that is within `near` of the winner and that the model scores
    above the tier's pick, so that keeping it moves the pick there: the
    GEMM's winner when the model ranks the winner above the pick, else the
    near-winner it does. Before each key is tried, the collapses and their
    rescues are found again under the tier with the keys kept so far, so
    that no key joins for a collapse an earlier one lifted; the rescue
    lifting the most of those still open is tried next, each key once.
    It joins only when `guard_verdict` keeps the tier with it and the keys
    kept before it against `tier_keys`: no new pick below `floor` on either
    side of the cell's GEMMs, none of the picks the keys kept before it
    lifted falling back below it, and the held-out geomean kept; at most
    `max_keys` join."""
    tier = [int(x) for x in tier_keys]
    rec: Dict[str, Any] = {"n_collapses": 0}
    if max_keys <= 0 or not tier:
        return [], rec
    rec["n_collapses"], rescues = _rescues(train, tier, floor, near)
    rec["n_rescuable"] = sum(len(v) for v in rescues.values())
    if not rescues:
        return [], rec
    added: List[int] = []
    tried: Set[int] = set()
    refused: List[Dict[str, Any]] = []
    verdict: Optional[Dict[str, Any]] = None
    while len(added) < int(max_keys):
        left = [k for k in rescues if k not in tried]
        if not left:
            break
        key = min(left, key=lambda x: (-len(rescues[x]), x))
        tried.add(key)
        ok, check = guard_verdict(
            train,
            held,
            tier,
            tier + added + [key],
            floor=floor,
            mean_floor=mean_floor,
            current=tier + added,
        )
        if not ok:
            refused.append({"key": key, "lifts": len(rescues[key]), **check})
            continue
        added.append(key)
        verdict = check
        _, rescues = _rescues(train, tier + added, floor, near)
    rec["added"] = added
    rec["n_lifted"] = (
        len(tail_change(train, tier, tier + added, floor)["healed"]) if added else 0
    )
    rec["check"] = verdict
    rec["n_refused"] = len(refused)
    rec["refused"] = refused[:_GUARD_SHOWN]
    return added, rec


def guard(
    tier: Tier,
    labels: Sequence[str],
    scored: Sequence[Tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    winner_us: Sequence[float],
    held_scored: Sequence[Tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    held_winner_us: Sequence[float],
    hashes: np.ndarray,
    *,
    max_keys: int = GUARD_KEYS,
    floor: float = GUARD_FLOOR,
    near: float = GUARD_NEAR,
    mean_floor: float = GUARD_MEAN_FLOOR,
) -> int:
    """The tail guard (`guard_cell`) on every cell of `tier`, from its
    training GEMMs (`scored` with `winner_us`, as `choose` read them) and
    its held-out GEMMs (`held_scored`, `held_winner_us`), each routed by
    the model itself: a cell's guard keys follow its greedy keys in
    `tier.keys`, and a cell with a collapse records what the guard found
    (`guard`, its keys named by view-key hash). Returns the keys added."""

    def by_cell(
        rows: Sequence[Tuple[int, np.ndarray, np.ndarray, np.ndarray]],
        wins: Sequence[float],
    ) -> Dict[int, List[Item]]:
        out: Dict[int, List[Item]] = {}
        for w, r in zip(wins, rows):
            if r[0] >= 0 and w > 0 and r[1].size:
                out.setdefault(int(r[0]), []).append((float(w), r))
        return out

    index = {int(h): i for i, h in enumerate(hashes.tolist())}
    train_of, held_of = by_cell(scored, winner_us), by_cell(held_scored, held_winner_us)
    n_added = 0
    for ci, label in enumerate(labels):
        keys = [index[int(h)] for h in tier.keys.get(label, []) if int(h) in index]
        added, rec = guard_cell(
            train_of.get(ci, []),
            held_of.get(ci, []),
            keys,
            max_keys=max_keys,
            floor=floor,
            near=near,
            mean_floor=mean_floor,
        )
        if not rec["n_collapses"]:
            continue
        rec["added"] = [int(hashes[x]) for x in rec.get("added", [])]
        for r in rec.get("refused", []):
            r["key"] = int(hashes[r["key"]])
        rec["n_held_out_gemms"] = len(held_of.get(ci, []))
        if label in tier.cells:
            tier.cells[label]["guard"] = rec
        if added:
            tier.keys[label] = list(tier.keys.get(label, [])) + [
                int(hashes[x]) for x in added
            ]
            if label in tier.cells:
                tier.cells[label]["n_keys"] = len(tier.keys[label])
            n_added += len(added)
    return n_added


def build(
    bundle: Mapping[str, Any],
    gemms: Sequence[Any],
    *,
    k: int,
    u: ss.Universe,
    hardware: hwlib.DeviceHardware,
    arch: str,
    weight_dtype: str,
    workers: int = 1,
    s5: Any = None,
    say: Optional[Callable[[str], None]] = None,
    mode: str = DEFAULT_CONTEXTS,
    held_out: Sequence[Any] = (),
    guard_keys: int = GUARD_KEYS,
) -> Tier:
    """The tier of `bundle` with `k` keys per cell, chosen on the training
    GEMMs `gemms` (`split_gemms`) over the universe `u` as `mode`
    (`CONTEXT_MODES`) counts their execution contexts on `hardware`, with
    the model file the bundle packs to at `weight_dtype` (`scoring_bundle`),
    then with up to `guard_keys` keys per cell that the tail guard (`guard`)
    adds from those GEMMs and the held-out ones `held_out` (0: no guard).
    ValueError for a bundle that cannot get one (`refusal`), k < 1 or an
    unknown mode."""
    why = refusal(bundle)
    if why:
        raise ValueError(why)
    if int(k) < 1:
        raise ValueError(f"a whitelist tier of {k} keys")
    if mode not in CONTEXT_MODES:
        raise ValueError(
            f"whitelist tier contexts {mode!r} is not one of {list(CONTEXT_MODES)}"
        )
    if int(guard_keys) < 0:
        raise ValueError(f"a tail guard of {guard_keys} keys")
    s5 = s5 or stage05_module()
    data = mlrec.write_model(
        scoring_bundle(bundle), None, arch, hwlib.arch_constants(arch), weight_dtype
    )
    tw = ev.tilewright_module()
    model = ev.load_model_bytes(data, tw)
    labels = [
        str(tw.cell_label(model, i)) for i in range(int(tw.describe(model).n_cells))
    ]
    held = list(held_out) if int(guard_keys) > 0 else []
    if say:
        say(
            f"scoring {len(gemms):,} training GEMMs"
            + (f" and {len(held):,} held-out ones" if held else "")
            + f" over {u.n:,} kernels ({u.n_keys:,} view keys) with "
            f"{max(1, int(workers))} workers"
        )
    scored = engine_scores(data, u, hardware, list(gemms) + held, workers)
    tier = choose(
        labels,
        scored[: len(gemms)],
        [g.winner_us for g in gemms],
        k,
        u.hashes,
        s5,
        contexts=[served_context(g.context, hardware.n_cu) for g in gemms],
        mode=mode,
    )
    n_guard = 0
    if int(guard_keys) > 0:
        n_guard = guard(
            tier,
            labels,
            scored[: len(gemms)],
            [g.winner_us for g in gemms],
            scored[len(gemms) :],
            [g.winner_us for g in held],
            u.hashes,
            max_keys=int(guard_keys),
        )
    reached = [c for c in tier.cells.values() if c.get("n_gemms")]
    guarded = [c["guard"] for c in tier.cells.values() if c.get("guard")]
    tier.record = {
        "k": int(k),
        "contexts": mode,
        **({"reserve_k": RESERVE_KEYS} if mode == "reserve" else {}),
        "n_cells": len(tier.keys),
        "n_keys": sum(len(v) for v in tier.keys.values()),
        "n_gemms": sum(c["n_gemms"] for c in reached),
        "weight_dtype": str(weight_dtype),
        "pool": {
            "n_kernels": int(u.n),
            "n_keys": int(u.n_keys),
            "keys_digest": keys_digest(u.hashes.tolist()),
        },
        "guard": {
            "keys": int(guard_keys),
            "floor": GUARD_FLOOR,
            "near": GUARD_NEAR,
            "mean_floor": GUARD_MEAN_FLOOR,
            "n_held_out_gemms": len(held),
            "n_cells_with_collapses": len(guarded),
            "n_collapses": sum(int(g["n_collapses"]) for g in guarded),
            "n_keys_added": n_guard,
            "n_lifted": sum(int(g.get("n_lifted") or 0) for g in guarded),
        },
    }
    return tier


def apply(bundle: Mapping[str, Any], tier: Tier) -> Dict[str, Any]:
    """`bundle` serving `tier`: whitelist tier on, each cell's view keys
    those of the tier (none for a cell without), and the tier's record."""
    out = strip(bundle)
    out["models"] = {
        label: dict(e, view_key_hashes=list(tier.keys.get(label, [])))
        for label, e in (out.get("models") or {}).items()
    }
    out["serving"] = dict(out.get("serving") or {}, whitelist_tier=True)
    out[RECORD] = {"k": int(tier.k), **tier.record}
    return out


def pool_check(
    bundle: Mapping[str, Any], kernels: Sequence[Mapping[str, Any]]
) -> Optional[Dict[str, Any]]:
    """How the tier of `bundle` meets the pool `kernels` the runtime ranks
    with it (None without a tier): the pool's keys and digest, the tier keys
    it lacks and the cells left with none of theirs, which the cascade
    serves."""
    if not has_tier(bundle):
        return None
    spec = fs.parse_attribute_spec(bundle.get("attribute_spec") or [])
    present = {int(h) for h in view_key_hashes(kernels, spec)}
    missing = 0
    bare: List[str] = []
    for label, e in (bundle.get("models") or {}).items():
        keys = [int(h) & 0xFFFFFFFFFFFFFFFF for h in e.get("view_key_hashes") or []]
        lost = sum(1 for h in keys if h not in present)
        missing += lost
        if keys and lost == len(keys):
            bare.append(label)
    return {
        "n_kernels": len(kernels),
        "n_keys": len(present),
        "keys_digest": keys_digest(present),
        "tier_keys_missing": missing,
        "cells_without_tier_keys": sorted(bare),
    }


def record_text(record: Mapping[str, Any]) -> str:
    detail = {
        "balanced": "; every execution context weighing the same",
        "reserve": (
            f"; up to {record.get('reserve_k')} more per cell for GEMMs outside "
            f"exclusive use"
        ),
    }.get(str(record.get("contexts") or DEFAULT_CONTEXTS), "")
    g = record.get("guard") or {}
    if g.get("n_keys_added"):
        detail += (
            f"; {g['n_keys_added']} added by the tail guard, lifting "
            f"{g.get('n_lifted')} of {g.get('n_collapses')} collapses on the "
            f"cells' own GEMMs"
        )
    return (
        f"{record.get('k')} view keys per cell in {record.get('n_cells')} cells "
        f"({record.get('n_keys')} in all{detail})"
    )


def write_doc(path: Path, tier: Tier) -> None:
    from .fs import write_json

    write_json(path, tier.doc())
