# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The whitelist tier of an MLREC_v3 bundle (`train.whitelist_tier_k`).

After training, every cell of the bundle gets K view keys that the engine
scores first (tier 1 with the model's `whitelist_tier` serving flag), chosen
on the training GEMMs that route to the cell:

  GEMMs    the enriched data stage05 trained from, read as stage05 reads it
           (`run_analysis.load_bench` with the run's library identity,
           repeat aggregation and blocklist), every leaf's training side as
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
from typing import Tuple

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


def training_gemms(
    gemms: Sequence[Any],
    tree: Optional[Mapping[str, Any]],
    validation_frac: float,
    seed: int,
    s5: Any = None,
) -> List[Any]:
    """The training side of `gemms` (`run_analysis.Gemm`): per leaf of the
    split tree `tree` (stage05's routing), the GEMMs stage05's
    `split_validation` keeps for training, the leaves in the order of their
    first GEMM and each leaf's GEMMs in their order. Leaves stage05 did not
    train count too: the bundle serves their GEMMs with a model of another
    cell."""
    s5 = s5 or stage05_module()
    by_leaf: Dict[str, List[Any]] = {}
    for g in gemms:
        m, n, k, b = (int(v) for v in g.key[:4])
        by_leaf.setdefault(s5._leaf_of(m, n, k, b, dict(tree or {})), []).append(g)
    out: List[Any] = []
    for group in by_leaf.values():
        rows = [g.row for g in group]
        train, _val = s5.split_validation(rows, validation_frac, seed)
        kept = {id(r) for r in train}
        out += [g for g, r in zip(group, rows) if id(r) in kept]
    return out


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


def choose(
    labels: Sequence[str],
    scored: Sequence[Tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    winner_us: Sequence[float],
    k: int,
    hashes: np.ndarray,
    s5: Any = None,
) -> Tier:
    """The tier of a model with cells `labels` from the scored training
    GEMMs (`_score_chunk` results, in the GEMMs' order, with their winner
    latencies)."""
    s5 = s5 or stage05_module()
    by_cell: Dict[int, List[Tuple[float, Tuple[Any, ...]]]] = {}
    for w, r in zip(winner_us, scored):
        if r[0] >= 0:
            by_cell.setdefault(int(r[0]), []).append((float(w), r))
    tier = Tier(k=int(k))
    for ci, label in enumerate(labels):
        items = [it for it in by_cell.get(ci, []) if it[0] > 0 and it[1][1].size]
        if not items:
            tier.cells[label] = {"n_gemms": 0}
            continue
        keys = sorted({int(x) for _, r in items for x in r[1].tolist()})
        row_of = {x: j for j, x in enumerate(keys)}
        us = np.full((len(keys), len(items)), np.inf)
        score = np.full((len(keys), len(items)), -np.inf)
        for gi, (_w, r) in enumerate(items):
            rows = np.array([row_of[int(x)] for x in r[1].tolist()], dtype=np.int64)
            us[rows, gi] = r[2]
            score[rows, gi] = r[3]
        win = np.array([w for w, _ in items])
        table = s5.ScoreTable(
            [(x,) for x in keys], list(range(len(items))), us, score, win
        )
        every = s5._evaluate_keys(table, sorted(table.keys, key=lambda t: t[0]))
        picked = s5.select_smart_k_signatures(
            [], int(k), table=table, with_metrics=False
        )
        chosen = [int(t[0]) for t in picked["sigs"]]
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
    return tier


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
) -> Tier:
    """The tier of `bundle` with `k` keys per cell, chosen on the training
    GEMMs `gemms` (`training_gemms`) over the universe `u`, with the model
    file the bundle packs to at `weight_dtype` (`scoring_bundle`).
    ValueError for a bundle that cannot get one (`refusal`) or k < 1."""
    why = refusal(bundle)
    if why:
        raise ValueError(why)
    if int(k) < 1:
        raise ValueError(f"a whitelist tier of {k} keys")
    s5 = s5 or stage05_module()
    data = mlrec.write_model(
        scoring_bundle(bundle), None, arch, hwlib.arch_constants(arch), weight_dtype
    )
    tw = ev.tilewright_module()
    model = ev.load_model_bytes(data, tw)
    labels = [
        str(tw.cell_label(model, i)) for i in range(int(tw.describe(model).n_cells))
    ]
    if say:
        say(
            f"scoring {len(gemms):,} training GEMMs over {u.n:,} kernels "
            f"({u.n_keys:,} view keys) with {max(1, int(workers))} workers"
        )
    scored = engine_scores(data, u, hardware, gemms, workers)
    tier = choose(labels, scored, [g.winner_us for g in gemms], k, u.hashes, s5)
    reached = [c for c in tier.cells.values() if c.get("n_gemms")]
    tier.record = {
        "k": int(k),
        "n_cells": len(tier.keys),
        "n_keys": sum(len(v) for v in tier.keys.values()),
        "n_gemms": sum(c["n_gemms"] for c in reached),
        "weight_dtype": str(weight_dtype),
        "pool": {
            "n_kernels": int(u.n),
            "n_keys": int(u.n_keys),
            "keys_digest": keys_digest(u.hashes.tolist()),
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
    return (
        f"{record.get('k')} view keys per cell in {record.get('n_cells')} cells "
        f"({record.get('n_keys')} in all)"
    )


def write_doc(path: Path, tier: Tier) -> None:
    from .fs import write_json

    write_json(path, tier.doc())
