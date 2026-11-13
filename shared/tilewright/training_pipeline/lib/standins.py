# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Stand-ins for the whitelisted view keys of the whitelist tier.

A model whose cells list a whitelist tier (lib/whitelist_tier.py) serves a
problem from the cell's whitelisted view keys. It may also list, per
whitelisted key, up to three other trained keys that stand in for it
(serving flag bit 12, lib/mlrec.py): when the pool holds no kernel of a
whitelisted key, the first listed stand-in the pool holds joins the cell's
whitelist, at most one per missing key, so the call stays a whitelist call
(the engine's README, How a problem is ranked).

Choice (`choose`), from the tier builder's `ScoreTable` of a cell's training
GEMMs (the training side of stage05's split only): per GEMM and measured key,
the latency of the key's first pool kernel and the deployed model's full
score. For each whitelisted key s, the tier without s is evaluated as the
tier's greedy (`whitelist_tier._greedy`) evaluates a tier: per GEMM the model
picks the best-scored measured key, the earlier key on equal scores. The
candidates are the trained keys outside the tier, taken in two phases:

  1. keys that cover training GEMMs only s covered, by the number of such
     GEMMs, then by the summed gain (the log efficiency on the GEMMs the key
     newly covers plus the log improvement of the covered picks it changes),
     then table order; no gain test applies;
  2. keys that make the covered picks they change faster in sum (a positive
     summed log improvement), by the mean log efficiency of the model's
     picks after adding them, then table order.

The first `depth` distinct keys of phase 1 then phase 2 are listed, best
first. A key whose removal changes no training pick has no phase 1, so it
lists only keys that improve the whole tier's training picks, and none when
no such key exists. The rule is a function of the table and the tier as a
set: the tier's keys enter in ascending view-key hash. `RULE` names it in
the records that hold its lists.

Serving (`missing`, `active`, `active_standins`, `summary`), as the engine
fixes it per candidate set: for the cell's whitelisted keys the model lists
as trained that the pool lacks, in ascending order of the keys, the first
listed stand-in the pool holds that no earlier missing key took.

Deployment (`deployed_depth`, `refusal`, `deployed_records`): stage05
chooses the lists with its tier only for a run that sets `deploy.standins`,
and stage06 packs the lists of the tier's record, each cut to the depth the
key names, into the model it stages, and only into a bundle the engine can
serve them for whose record holds lists of this rule; lib/audit.py packs
them alike.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from . import features as fs

# The rule `choose` codes, as the records of its lists name it.
RULE = "per-key.1"
# The stand-ins a record keeps per whitelisted key (the most a model lists).
DEPTH = 3
_U64 = (1 << 64) - 1


# ── choice ───────────────────────────────────────────────────────────────────


def _state(
    table: Any, rows: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(best score, log latency of the pick, covered) of each GEMM of
    `table` when the model picks from `rows` (taken in order; an earlier
    row keeps a GEMM on equal scores), as the tier's greedy takes keys."""
    n_gemms = table.us.shape[1]
    best = np.full(n_gemms, -np.inf)
    log_pick = np.zeros(n_gemms)
    covered = np.zeros(n_gemms, dtype=bool)
    for j in rows:
        upd = table.has[j] & (table.score[j] > best)
        best[upd] = table.score[j][upd]
        log_pick[upd] = table.log_us[j][upd]
        covered[upd] = True
    return best, log_pick, covered


def _step(
    table: Any,
    best: np.ndarray,
    log_pick: np.ndarray,
    covered: np.ndarray,
    lost: np.ndarray,
    cand: np.ndarray,
) -> Tuple[List[int], List[int]]:
    """The two phases from a state: (rows that cover lost GEMMs, by most
    covered, then their gain; rows that improve the covered picks they
    change in sum, by the mean log efficiency after them), over the
    candidate rows `cand` (a mask)."""
    has, score, log_us, log_win = table.has, table.score, table.log_us, table.log_win
    better = has & (score > best)
    imp = np.where(better & covered, log_pick - log_us, 0.0).sum(axis=1)
    n_lost = (has & lost).sum(axis=1)
    phase1: List[int] = []
    if lost.any():
        new = has & ~covered
        gain = np.where(new, log_win - log_us, 0.0) + np.where(
            better & covered, log_pick - log_us, 0.0
        )
        g = gain.sum(axis=1)
        ok = cand & (n_lost > 0)
        phase1 = sorted(
            np.flatnonzero(ok).tolist(),
            key=lambda j: (-int(n_lost[j]), -float(g[j]), j),
        )
    valid = better | covered
    eff = np.where(valid, log_win - np.where(better, log_us, log_pick), 0.0)
    count = valid.sum(axis=1)
    avg = eff.sum(axis=1) / np.maximum(count, 1)
    ok2 = cand & (count > 0) & (imp > 0)
    phase2 = sorted(np.flatnonzero(ok2).tolist(), key=lambda j: (-float(avg[j]), j))
    return phase1, phase2


def choose(
    table: Any, tier_rows: Sequence[int], cand: np.ndarray, depth: int = DEPTH
) -> Dict[int, List[int]]:
    """{tier row: up to `depth` stand-in rows, best first} of a cell's
    `ScoreTable` (the module docstring) for the tier `tier_rows` (rows of
    `table`, in ascending view-key hash); `cand` masks the rows a stand-in
    may be (trained keys), and the tier's rows never are."""
    tier = [int(x) for x in tier_rows]
    cand = np.asarray(cand, dtype=bool) & ~np.isin(np.arange(table.us.shape[0]), tier)
    _b, _l, covered_full = _state(table, tier)
    out: Dict[int, List[int]] = {}
    for s in tier:
        start = [t for t in tier if t != s]
        best, log_pick, covered = _state(table, start)
        lost = covered_full & ~covered
        phase1, phase2 = _step(table, best, log_pick, covered, lost, cand)
        picked: List[int] = []
        for j in phase1 + phase2:
            if j not in picked:
                picked.append(j)
            if len(picked) >= int(depth):
                break
        out[s] = picked
    return out


def records(
    lists: Mapping[str, Mapping[Any, Sequence[int]]],
    labels: Sequence[str],
    depth: int,
) -> List[Tuple[int, int, List[int]]]:
    """The records a model with cells `labels` (file order) lists for the
    per-cell lists `lists` ({cell label: {whitelisted hash: stand-ins}}),
    each truncated to its first `depth`: (cell index, key, stand-ins),
    ascending; a key without a stand-in has no record."""
    out = []
    index = {label: i for i, label in enumerate(labels)}
    for label, per_key in lists.items():
        for key, keys in per_key.items():
            kept = [int(h) & _U64 for h in list(keys)[: int(depth)]]
            if kept:
                out.append((index[label], int(key) & _U64, kept))
    out.sort(key=lambda r: (r[0], r[1]))
    return out


def text(recs: Iterable[Tuple[int, int, Sequence[int]]]) -> str:
    """A format-free text of records: one line per record, `cell<TAB>key<TAB>`
    and the stand-ins comma-separated, ascending."""
    return "".join(
        f"{c}\t{k}\t{','.join(str(int(x)) for x in s)}\n"
        for c, k, s in sorted(recs, key=lambda r: (r[0], r[1]))
    )


# ── serving ──────────────────────────────────────────────────────────────────


def pool_hashes(
    kernels: Iterable[Mapping[str, Any]], spec: fs.AttributeSpec
) -> Set[int]:
    """The view-key hashes of a pool's kernels (pool dicts with the config
    fields and `attributes`) under the attribute spec `spec`."""
    return {
        fs.view_key_hash(
            fs.v3_view_key(fs.config_kwargs_from_row(k), k.get("attributes"), spec)
        )
        & _U64
        for k in kernels
    }


def missing(
    shortlist: Iterable[int], trained: Optional[Set[int]], pool: Set[int]
) -> List[int]:
    """The whitelisted view keys `shortlist` of a cell the model lists as
    trained (every one without the gate) of which the pool holds no kernel,
    ascending."""
    return sorted(
        {
            int(h) & _U64
            for h in shortlist
            if (trained is None or int(h) & _U64 in trained)
            and int(h) & _U64 not in pool
        }
    )


def active(
    lost: Sequence[int],
    lists: Mapping[int, Sequence[int]],
    pool: Set[int],
) -> List[int]:
    """The stand-ins that join a cell's whitelist for its missing keys `lost`
    (ascending), from its records `lists` ({key: stand-ins}): per missing
    key, the first listed stand-in the pool holds that no earlier key took;
    ascending."""
    out: List[int] = []
    for s in lost:
        for r in lists.get(int(s), []):
            r = int(r) & _U64
            if r in pool and r not in out:
                out.append(r)
                break
    return sorted(out)


def _trained(parsed: Mapping[str, Any]) -> Optional[Set[int]]:
    keys = parsed.get("trained_view_keys")
    if not (parsed.get("serving") or {}).get("trained_key_gate") or keys is None:
        return None
    return {int(h) & _U64 for h in keys}


def active_standins(
    parsed: Mapping[str, Any], pool: Iterable[int]
) -> Dict[int, List[int]]:
    """Per cell index of a parsed model (lib/mlrec.py read_model, file
    order) with any: the stand-ins that join its whitelist on the pool whose
    view-key hashes are `pool`, as the engine's
    CandidateSet::standins(cell) gives them."""
    held = {int(h) & _U64 for h in pool}
    trained = _trained(parsed)
    by_cell: Dict[int, Dict[int, List[int]]] = {}
    for cell, key, keys in (parsed.get("serving") or {}).get("standins") or []:
        by_cell.setdefault(int(cell), {})[int(key) & _U64] = [
            int(h) & _U64 for h in keys
        ]
    out: Dict[int, List[int]] = {}
    for i, cell in enumerate(parsed["cells"]):
        if i not in by_cell:
            continue
        lost = missing(cell.get("view_key_hashes") or [], trained, held)
        got = active(lost, by_cell[i], held)
        if got:
            out[i] = got
    return out


def summary(parsed: Mapping[str, Any], pool: Iterable[int]) -> Dict[str, Any]:
    """What the engine's `[TILEWRIGHT_DIAG STANDINS]` line reports for the
    pool whose view-key hashes are `pool`: the cells whose whitelist
    stand-ins join, the cells that lack a trained whitelisted key, the
    stand-ins that join and the trained whitelisted keys the pool lacks, over
    the cells with a whitelist."""
    held = {int(h) & _U64 for h in pool}
    trained = _trained(parsed)
    joined = active_standins(parsed, held)
    lacking, missing_keys = 0, 0
    for cell in parsed["cells"]:
        lost = missing(cell.get("view_key_hashes") or [], trained, held)
        lacking += 1 if lost else 0
        missing_keys += len(lost)
    return {
        "cells": sorted(joined),
        "cells_lacking_keys": lacking,
        "standins": sum(len(v) for v in joined.values()),
        "missing": missing_keys,
    }


# ── deployment ───────────────────────────────────────────────────────────────


def deployed_depth(cfg: Optional[Mapping[str, Any]]) -> Optional[int]:
    """The stand-ins per tier key stage06 packs into every MLREC_v3 model it
    stages under config `cfg` (`deploy.standins`: {depth: J}; absent or
    null: None, no stand-ins, which stage05 then does not choose either);
    ValueError for a depth that is not a whole number from 1 to DEPTH."""
    section = ((cfg or {}).get("deploy") or {}).get("standins")
    if section is None:
        return None
    depth = section.get("depth") if isinstance(section, Mapping) else None
    if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= DEPTH:
        raise ValueError(
            f"the depth of the stand-ins must be a whole number from 1 to {DEPTH}, "
            f"got {depth!r}"
        )
    return int(depth)


def refusal(bundle: Mapping[str, Any], serving: Mapping[str, Any]) -> Optional[str]:
    """Why the models.pt bundle `bundle`, deployed with the serving options
    `serving`, cannot carry stand-ins, or None. The engine serves them for
    the view keys of a whitelist tier under the trained-key gate; a bundle
    trained on the bank never carries them; its tier's record holds lists of
    this rule (`RULE`)."""
    from . import whitelist_tier as wt

    cells = list((bundle.get("models") or {}).values())
    if str(bundle.get("catalog") or "v2") != "v3":
        return "it is not an MLREC_v3 bundle"
    if not serving.get("whitelist_tier", True) or not wt.has_tier(bundle):
        return "it has no whitelist tier"
    if serving.get("launch_keys"):
        return "it serves launch keys"
    if any(m.get("smart_k_signatures") for m in cells):
        return "a cell lists signatures"
    if not serving.get("trained_key_gate") or not bundle.get("trained_view_keys"):
        return "it is not served with the trained-key gate"
    if serving.get("bank"):
        return "it was trained on the bank as well as the table"
    rule = (bundle.get(wt.RECORD) or {}).get("standins_rule")
    if not rule:
        return (
            "its whitelist tier's record holds no stand-ins; give it them with "
            "scripts/build_whitelist_tier.py --keep-tier --standins"
        )
    if rule != RULE:
        return f"its stand-ins follow the rule {rule!r}, not {RULE!r}"
    return None


def deployed_records(
    bundle: Mapping[str, Any], depth: int
) -> List[Tuple[int, int, List[int]]]:
    """The stand-in records of the lists `bundle`'s tier record keeps, each
    cut to its first `depth`, for the bundle's cells in file order (its
    labels sorted; `records`). ValueError for a damaged record: lists for a
    cell the bundle lacks or that a repair left without stand-ins
    (`standins_repaired`), for a key the cell does not whitelist or that is
    not trained, or with a stand-in that is not trained, that the cell
    whitelists or that the list repeats."""
    from . import whitelist_tier as wt

    rec = bundle.get(wt.RECORD) or {}
    models = bundle.get("models") or {}
    trained = {int(h) & _U64 for h in bundle.get("trained_view_keys") or []}
    repaired = {str(r.get("cell")) for r in rec.get("standins_repaired") or []}
    lists = wt.standins_of(rec)
    for label, per_key in lists.items():
        if label not in models or label in repaired:
            why = "a repair left it without them" if label in repaired else "no cell"
            raise ValueError(
                f"the tier's record lists stand-ins for {label!r}, {why} (a damaged "
                f"record)"
            )
        tier = {int(h) & _U64 for h in models[label].get("view_key_hashes") or []}
        for key, keys in per_key.items():
            if key not in tier or key not in trained:
                raise ValueError(
                    f"the tier's record lists stand-ins in {label!r} for {key}, not a "
                    f"trained key the cell whitelists (a damaged record)"
                )
            bad = [h for h in keys if h not in trained or h in tier]
            if bad or len(set(keys)) != len(keys) or not 1 <= len(keys) <= DEPTH:
                raise ValueError(
                    f"the tier's record lists stand-ins in {label!r} for {key} that "
                    f"are untrained, whitelisted by the cell or repeated (a damaged "
                    f"record)"
                )
    return records(lists, sorted(models), depth)


__all__: Sequence[str] = (
    "DEPTH",
    "RULE",
    "active",
    "active_standins",
    "choose",
    "deployed_depth",
    "deployed_records",
    "missing",
    "pool_hashes",
    "records",
    "refusal",
    "summary",
    "text",
)
