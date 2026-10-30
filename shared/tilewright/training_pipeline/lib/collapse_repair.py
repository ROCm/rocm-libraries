# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Repair of the collapses and flagged cells that keep a bundle out
(run_orchestrator.py `FinalRetrain`, lib/cell_merge.py `RoundRepair`,
scripts/repair_bundle.py).

A collapse is a pick of the candidate below the per-cell gate's tail floor
(lib/audit.py `cell_gate`, gate.tail_floor) on an audit GEMM of one group
(the exclusive context or a CU budget) where the bundle that ships picked
at or above it, as a repeated measurement confirms (`collapses_of`). The
gate refuses such a candidate. The repair changes the candidate only where
it collapses, and the gate then judges the repaired bundle as a whole
again:

  localize   each collapse to the leaf whose network served it (the
             engine's routing), its pick, and the pick the same network
             makes over every key it serves when the whitelist tier does
             not restrict it (`whitelist_tier.scoring_bundle`);
  tier add   when that unrestricted pick is another kernel, outside the
             leaf's tier, that reads at or above the floor on the GEMM
             (`AuditView.reading`), its view key joins the leaf's tier, at
             most `whitelist_tier.GUARD_KEYS` per leaf, and stays only when
             the tail guard keeps it on the leaf's own data
             (`whitelist_tier.guard_verdict`: the leaf's tier with it and
             the keys kept before it, against the tier the candidate holds,
             adds no new pick below the floor on the GEMMs the bundle
             trained the leaf on or held out of its training, undoes no
             rescue of those keys, and keeps the held-out picks' geomean)
             and it lifts the collapse;
  carry      a collapse no kept key lifts (the network itself ranks the
  over       collapsing kernel first, or the tail guard refuses the key)
             sends its base cell back to the bundle that ships: the
             candidate's networks and tier keys of that base cell, the cell
             and every leaf under it, are replaced by that bundle's, so its
             picks there are exactly that bundle's. A base cell the gate's
             cell rules flagged (`carry_also`: significantly worse, or fallen
             below its floor) is carried the same way: no tier key answers
             a cell's decline.

The audit only detects and verifies. Every key a repair adds is the
leaf's own network's choice, kept on the leaf's own data; every carried
network is one the run shipped. No audit GEMM trains anything, and the
repaired bundle's audit numbers come from the GEMMs that found its
collapses, so they carry the same small optimistic bias as the final
retrain's own decision: a fresh audit confirms them.
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence
from typing import Set, Tuple

from . import subcells as sc
from . import whitelist_tier as wt

ShapeKey = Tuple[int, int, int, int]
Key = Tuple[str, ShapeKey]
RECORD = "repair.json"
SCHEMA = "collapse_repair_v1"
# The flag of a carried cell's whitelist_tier.json entry.
CARRIED = "carried_by_repair"


@dataclass(frozen=True)
class Collapse:
    """A deciding new pick below the tail floor: its audit group and GEMM,
    its base cell, and the shipped bundle's and the candidate's
    efficiency there."""

    group: str
    shape: ShapeKey
    cell: str
    before: float
    after: float

    @property
    def key(self) -> Key:
        return (self.group, self.shape)

    def text(self) -> str:
        m, n, k, b = self.shape
        return f"{self.group} {m}x{n}x{k}" + (f" batch {b}" if b != 1 else "")


def collapses_of(cell_gate: Mapping[str, Any]) -> List[Collapse]:
    """The collapses that decide the per-cell gate result `cell_gate`
    (lib/audit.py `cell_gate`): every confirmed new pick below the tail
    floor of every group compared, in group then GEMM order."""
    out: List[Collapse] = []
    for label, g in (cell_gate.get("groups") or {}).items():
        if not isinstance(g, dict) or "reason" in g:
            continue
        for t in g.get("tail") or []:
            if not t.get("decides"):
                continue
            shape = tuple(int(v) for v in t["shape"][:4])
            out.append(
                Collapse(
                    str(label),
                    shape,  # type: ignore[arg-type]
                    str(t.get("cell") or ""),
                    float(t["before"]),
                    float(t["after"]),
                )
            )
    return out


def only_the_tail(cell_gate: Mapping[str, Any]) -> bool:
    """The per-cell gate result `cell_gate` refuses on its tail rule alone:
    its cell rules found no cell (regressed or below the floor) and it has
    a deciding collapse."""
    return (
        cell_gate.get("passed") is False
        and not cell_gate.get("n_regressed")
        and not cell_gate.get("n_floor")
        and bool(cell_gate.get("n_tail"))
    )


# ── bundles ──────────────────────────────────────────────────────────────────


def _signed_like(keys: Sequence[int], h: int) -> int:
    """`h` (an unsigned 64-bit view-key hash) as `keys` store theirs."""
    h = int(h) & 0xFFFFFFFFFFFFFFFF
    if any(int(x) < 0 for x in keys) and h >= 1 << 63:
        return h - (1 << 64)
    return h


def tier_hashes(entry: Mapping[str, Any]) -> List[int]:
    """The unsigned view-key hashes of a bundle entry's tier, in order."""
    return [int(h) & 0xFFFFFFFFFFFFFFFF for h in entry.get("view_key_hashes") or []]


def with_tier_keys(
    bundle: Mapping[str, Any], adds: Mapping[str, Sequence[int]]
) -> Dict[str, Any]:
    """`bundle` with the view-key hashes `adds` ({leaf: hashes}) appended to
    those leaves' tiers (a hash a tier holds already is not repeated), its
    tier record counting them."""
    out = dict(bundle)
    models = dict(bundle.get("models") or {})
    added = 0
    for leaf, hashes in adds.items():
        entry = dict(models[leaf])
        keys = list(entry.get("view_key_hashes") or [])
        have = set(tier_hashes(entry))
        for h in hashes:
            h = int(h) & 0xFFFFFFFFFFFFFFFF
            if h not in have:
                keys.append(_signed_like(keys, h))
                have.add(h)
                added += 1
        entry["view_key_hashes"] = keys
        models[leaf] = entry
    out["models"] = models
    rec = bundle.get(wt.RECORD)
    if isinstance(rec, dict):
        out[wt.RECORD] = dict(rec, n_keys=int(rec.get("n_keys") or 0) + added)
    return out


def base_cells_of(labels: Iterable[str]) -> Dict[str, List[str]]:
    """{base cell: its labels among `labels`}."""
    out: Dict[str, List[str]] = {}
    for label in labels:
        out.setdefault(sc.base_cell(str(label)), []).append(str(label))
    return out


def carry_cells(
    bundle: Mapping[str, Any], shipped: Mapping[str, Any], cells: Iterable[str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(`bundle` with the networks of the base cells `cells` replaced by
    those of the bundle that ships, `shipped`; what moved). A base cell's
    labels are the cell and every leaf under it, each with its tier keys
    and feature statistics; ValueError when `shipped` has no model of a
    base cell. The bundle's fallback parents and tier record follow."""
    cells = sorted(set(cells))
    mine = base_cells_of(bundle.get("models") or {})
    theirs = base_cells_of(shipped.get("models") or {})
    missing = [c for c in cells if not theirs.get(c)]
    if missing:
        raise ValueError(
            f"the bundle that ships has no model of {', '.join(missing)} to carry"
        )
    models = {
        k: v
        for k, v in (bundle.get("models") or {}).items()
        if sc.base_cell(k) not in cells
    }
    moved: Dict[str, Any] = {}
    for c in cells:
        for label in sorted(theirs[c]):
            models[label] = shipped["models"][label]
        moved[c] = {"removed": sorted(mine.get(c, [])), "carried": sorted(theirs[c])}
    out = dict(bundle)
    out["models"] = models
    fallback = [
        p for p in bundle.get("fallback_parents") or [] if sc.base_cell(p) not in cells
    ]
    fallback += [
        p for p in shipped.get("fallback_parents") or [] if sc.base_cell(p) in cells
    ]
    out["fallback_parents"] = sorted(fallback)
    rec = bundle.get(wt.RECORD)
    if isinstance(rec, dict):
        out[wt.RECORD] = dict(
            rec,
            n_keys=sum(len(tier_hashes(e)) for e in models.values()),
            n_cells=len(models),
        )
    return out, moved


# ── the audit ────────────────────────────────────────────────────────────────


@dataclass
class Pick:
    """A model's pick on one audit GEMM: the solution, the leaf whose
    network served it (None when the model served nothing there)."""

    sid: Optional[int]
    leaf: Optional[str]


class AuditView:
    """The audit GEMMs and repeated measurements a repair reads (lib/audit.py
    `evaluate`'s preparation of the audit set, every evaluable GEMM kept),
    with the picks of model files on any of them."""

    def __init__(
        self,
        spec: Any,
        cfg: Mapping[str, Any],
        *,
        hardware: Any,
        tw: Any = None,
    ) -> None:
        from . import audit as auditlib
        from . import audit_sustained as asus
        from . import audit_v2 as v2
        from . import evaluate as ev

        self.ev, self.auditlib = ev, auditlib
        self.tw = tw or ev.tilewright_module()
        self.spec = dataclasses.replace(spec, trained_rounds=[])
        self.hardware = hardware
        null, prep = auditlib._prepare(b"", self.spec, cfg, None)
        if prep is None:
            raise RuntimeError(
                f"the audit set cannot evaluate: {(null or {}).get('status_text')}"
            )
        self.prep = prep
        self.gemm: Dict[Key, Dict[str, Any]] = {
            (
                auditlib._label_of(prep.labels, ev.context_of(g)),
                auditlib._shape_of(g),
            ): g
            for g in prep.gemms
        }
        self.in_pool = {int(k["sol_idx_global"]) for k in prep.pool.kernels}
        self.kernel = {int(k["sol_idx_global"]): k for k in prep.pool.kernels}
        self.v2: Any = None
        self.sustained: Any = None
        if spec.protocol == "both":
            kernels = v2.AuditKernels.of(
                auditlib.manifest(spec.audit_dir) or {},
                auditlib.pool_digests(prep.pool.kernels),
            )
            d = Path(spec.v2_dir) if spec.v2_dir else v2.v2_dir(spec.audit_dir)
            self.v2 = auditlib._repeated(d, prep, kernels)
            sd = d / asus.DIRNAME
            if v2.manifest(sd) is not None:
                self.sustained = auditlib._repeated(sd, prep, kernels)

    def picks(self, data: bytes, keys: Iterable[Key]) -> Dict[Key, Pick]:
        """The picks of model file `data` on the audit GEMMs `keys` (those
        the audit set holds), as the audit evaluation makes them."""
        ev = self.ev
        found = [k for k in keys if k in self.gemm]
        gemms = [self.gemm[k] for k in found]
        if not gemms:
            return {}
        out = ev.evaluate_model(
            data,
            gemms,
            hardware=self.hardware,
            pools=ev.pools_for(gemms, self.prep.library),
            tw=self.tw,
            pick_measurement=self.spec.pick_measurement,
            allow_drift=True,
        )
        return {
            k: Pick(
                None if r.pick_sol_idx is None else int(r.pick_sol_idx),
                r.model_cell if r.served_by == "model" else None,
            )
            for k, r in zip(found, out.results)
        }

    def label(self, sid: Optional[int]) -> Optional[str]:
        """How records name the pool kernel `sid`: its signature
        (`kernel_identity.signature_label`), never its solution index."""
        from .kernel_identity import signature_label

        k = self.kernel.get(int(sid)) if sid is not None else None
        return None if k is None else signature_label(k)

    def reading(self, key: Key, sid: int) -> Tuple[Optional[float], Optional[str]]:
        """(the efficiency of kernel `sid` on the audit GEMM `key`, the
        protocol of the reading): the sustained composite's for an exclusive
        GEMM it measured with the kernel, else v2's (its winner the fastest
        median of the pool, so the pick and the winner come from one
        repeated bench), else the exhaustive v1 reading (a skip row at its
        warm-up); (None, None) without any."""
        auditlib = self.auditlib
        for name, rep in (("sustained", self.sustained), ("v2", self.v2)):
            if rep is None or (name == "sustained" and key[0] != auditlib.EXCLUSIVE):
                continue
            mm = auditlib.pool_medians(rep.meas.get(key), self.in_pool)
            if mm is None or getattr(mm, "unplaced", False) or not mm.median:
                continue
            if int(sid) in mm.median:
                return min(1.0, float(mm.winner_us) / mm.median[int(sid)]), name
        g = self.gemm.get(key)
        if g is None:
            return None, None
        timed, screened = self.ev.measurements(g)
        us = timed.get(int(sid), screened.get(int(sid)))
        w = float(g.get("winner_us") or 0)
        if not us or w <= 0:
            return None, None
        return min(1.0, w / us), "v1"


# ── the repair ───────────────────────────────────────────────────────────────

# {leaf: (its training GEMMs, its held-out GEMMs)} as `whitelist_tier.Item`
# lists scored over the universe, and {view-key hash: universe key index}.
LeafData = Tuple[Dict[str, Tuple[List[Any], List[Any]]], Dict[int, int]]


@dataclass
class Repair:
    """A repair's outcome: the repaired bundle (None when nothing could be
    changed) and its record."""

    bundle: Optional[Dict[str, Any]]
    record: Dict[str, Any] = field(default_factory=dict)


def _r(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(float(v), 6)


def repair(
    candidate: Mapping[str, Any],
    shipped: Mapping[str, Any],
    collapses: Sequence[Collapse],
    audit: AuditView,
    *,
    pack: Callable[[Mapping[str, Any]], bytes],
    leaf_data: Callable[[Set[str]], LeafData],
    view_key_hash: Callable[[Mapping[str, Any]], int],
    floor: float = wt.GUARD_FLOOR,
    max_keys: int = wt.GUARD_KEYS,
    mean_floor: float = wt.GUARD_MEAN_FLOOR,
    carry_also: Iterable[str] = (),
) -> Repair:
    """The repair of `candidate` (a models.pt bundle) against the bundle
    that ships, `shipped`, for its `collapses` on the audit set `audit`
    and the base cells `carry_also` its cell rules flagged (see the module
    docstring). `pack` writes a bundle as the audit reads it, `leaf_data`
    gives the scored GEMMs of leaves, `view_key_hash` the view-key hash of
    a pool kernel. The record names every collapse with what was found and
    done, the picks of the repaired bundle on the collapsing GEMMs, and the
    base cells carried for a cell rule (`carried_for_cell_rules`)."""
    also = sorted({sc.base_cell(str(c)) for c in carry_also})
    rec: Dict[str, Any] = {
        "schema": SCHEMA,
        "floor": floor,
        "max_keys_per_leaf": max_keys,
        "mean_floor": mean_floor,
        "collapses": [],
        "tier_adds": {},
        "carried_cells": {},
        "carried_for_cell_rules": also,
    }
    if not collapses and not also:
        rec["outcome"] = "no collapse to repair"
        return Repair(None, rec)
    if not collapses:
        try:
            repaired, moved = carry_cells(candidate, shipped, also)
        except ValueError as ex:
            rec["outcome"] = f"cannot repair: {ex}"
            return Repair(None, rec)
        rec.update(
            carried_cells=moved,
            n_collapses=0,
            n_lifted_on_the_audit=0,
            n_keys_added=0,
            n_cells_carried=len(moved),
            outcome=f"0 tier key(s) added, {len(moved)} base cell(s) carried from "
            f"the bundle that ships for a cell rule's finding",
        )
        return Repair(repaired, rec)
    keys = [c.key for c in collapses]
    cand_picks = audit.picks(pack(candidate), keys)
    free_picks = audit.picks(pack(wt.scoring_bundle(candidate)), keys)
    models = candidate.get("models") or {}
    wanted: Dict[str, List[Tuple[int, Collapse]]] = {}
    entries: Dict[Key, Dict[str, Any]] = {}
    for c in collapses:
        p, u = cand_picks.get(c.key), free_picks.get(c.key)
        e: Dict[str, Any] = {
            "gemm": c.text(),
            "group": c.group,
            "shape": list(c.shape),
            "cell": c.cell,
            "shipped": _r(c.before),
            "candidate": _r(c.after),
        }
        entries[c.key] = e
        rec["collapses"].append(e)
        if p is None or p.sid is None or p.leaf is None:
            e["found"] = "the candidate's model serves no pick there"
            continue
        e.update(leaf=p.leaf, pick=audit.label(p.sid))
        tier = tier_hashes(models.get(p.leaf) or {})
        if not tier:
            e["found"] = "the leaf has no whitelist tier"
            continue
        if u is None or u.sid is None:
            e["found"] = "the leaf's network without its tier serves no pick"
            continue
        e["unrestricted_pick"] = audit.label(u.sid)
        if u.sid == p.sid:
            e["found"] = (
                "a ranking error: the leaf's network ranks the collapsing kernel "
                "first without its tier too"
            )
            continue
        h = view_key_hash(audit.kernel[u.sid])
        if h in tier:
            e["found"] = "the unrestricted pick's key is in the tier already"
            continue
        reading, proto = audit.reading(c.key, u.sid)
        e.update(unrestricted_reading=_r(reading), unrestricted_protocol=proto)
        if reading is None or reading < floor:
            e["found"] = (
                "tier-induced, but the unrestricted pick reads below the floor too"
            )
            continue
        e["found"] = "tier-induced: the tier lacks the leaf network's own pick"
        wanted.setdefault(p.leaf, []).append((h, c))

    adds: Dict[str, List[int]] = {}
    if wanted:
        data, index = leaf_data(set(wanted))
        for leaf, tries in sorted(wanted.items()):
            train, held = data.get(leaf, ([], []))
            tier = [index[h] for h in tier_hashes(models[leaf]) if h in index]
            kept: List[int] = []
            leaf_rec: Dict[str, Any] = {
                "n_own_gemms": len(train),
                "n_held_out_gemms": len(held),
                "tried": [],
            }
            for h, c in tries:
                t: Dict[str, Any] = {"key": h, "for": c.text()}
                leaf_rec["tried"].append(t)
                if h in kept:
                    t["kept"] = True
                    t["why"] = "kept for another collapse of the leaf"
                    entries[c.key]["tier_add"] = h
                    continue
                if len(kept) >= max_keys:
                    t.update(kept=False, why=f"the leaf has {max_keys} keys added")
                    continue
                if h not in index:
                    t.update(kept=False, why="the key is not in the pool's universe")
                    continue
                now = tier + [index[x] for x in kept]
                ok, check = wt.guard_verdict(
                    train,
                    held,
                    tier,
                    now + [index[h]],
                    floor=floor,
                    mean_floor=mean_floor,
                    current=now,
                )
                t.update(kept=ok, check=check)
                if ok:
                    kept.append(h)
                    entries[c.key]["tier_add"] = h
                else:
                    t["why"] = check["why_not"]
            if kept:
                adds[leaf] = kept
            rec["tier_adds"][leaf] = dict(leaf_rec, added=kept)

    repaired: Dict[str, Any] = dict(candidate)
    if adds:
        repaired = with_tier_keys(candidate, adds)
        after = audit.picks(pack(repaired), keys)
        for c in collapses:
            e = entries[c.key]
            if "tier_add" not in e:
                continue
            p = after.get(c.key)
            reading = None
            if p is not None and p.sid is not None:
                reading, proto = audit.reading(c.key, p.sid)
                e.update(
                    after_tier_add=audit.label(p.sid),
                    after_tier_add_reading=_r(reading),
                )
            e["lifted_by_tier_add"] = reading is not None and reading >= floor

    carry = sorted(
        {
            sc.base_cell(entries[c.key].get("leaf") or c.cell)
            for c in collapses
            if not entries[c.key].get("lifted_by_tier_add")
        }
        | set(also)
    )
    if carry:
        try:
            repaired, moved = carry_cells(repaired, shipped, carry)
        except ValueError as ex:
            rec["outcome"] = f"cannot repair: {ex}"
            return Repair(None, rec)
        rec["carried_cells"] = moved
        dropped = {leaf: hs for leaf, hs in adds.items() if sc.base_cell(leaf) in carry}
        if dropped:
            rec["tier_adds_dropped_by_carry"] = {k: list(v) for k, v in dropped.items()}
            adds = {k: v for k, v in adds.items() if k not in dropped}
        for c in collapses:
            cell = sc.base_cell(entries[c.key].get("leaf") or c.cell)
            if cell in carry:
                entries[c.key]["carried"] = cell

    final = audit.picks(pack(repaired), keys)
    n_lifted = 0
    for c in collapses:
        e = entries[c.key]
        p = final.get(c.key)
        reading, proto = (
            audit.reading(c.key, p.sid)
            if p is not None and p.sid is not None
            else (None, None)
        )
        e.update(
            repaired_pick=None if p is None else audit.label(p.sid),
            repaired_reading=_r(reading),
            repaired_protocol=proto,
        )
        if reading is not None and reading >= floor:
            n_lifted += 1
    rec["n_collapses"] = len(collapses)
    rec["n_lifted_on_the_audit"] = n_lifted
    rec["n_keys_added"] = sum(len(v) for v in adds.values())
    rec["n_cells_carried"] = len(rec["carried_cells"])
    rec["outcome"] = (
        f"{sum(len(v) for v in adds.values())} tier key(s) added in {len(adds)} "
        f"leaf(s), {len(rec['carried_cells'])} base cell(s) carried from the "
        f"bundle that ships"
        + (f" ({len(also)} for a cell rule's finding)" if also else "")
        + f"; {n_lifted} of {len(collapses)} collapse(s) read at or above "
        f"{floor:g} on the audit set"
    )
    return Repair(repaired, rec)


# ── the repaired bundle's records ────────────────────────────────────────────


def kept_tier_adds(record: Mapping[str, Any]) -> Dict[str, List[int]]:
    """{leaf: the view-key hashes the repair of `record` added to its tier}
    over the leaves whose keys the bundle keeps: a carried base cell's
    leaves serve the bundle that ships's tiers, and their added keys are
    gone (`tier_adds_dropped_by_carry`)."""
    carried = set(record.get("carried_cells") or {})
    return {
        str(leaf): [int(h) for h in r.get("added") or []]
        for leaf, r in sorted((record.get("tier_adds") or {}).items())
        if isinstance(r, Mapping)
        and r.get("added")
        and sc.base_cell(str(leaf)) not in carried
    }


def summary(record: Mapping[str, Any]) -> Dict[str, Any]:
    """What the repair of `record` changed, as the repaired bundle's
    cells.json and metrics.json record it: its outcome, the tier keys it
    added per leaf (`kept_tier_adds`) and the base cells it carried from
    the bundle that ships, with the labels each removed and took, and, when
    the record holds them, the flagged cells with what was done in each
    (`cells`), the sha256 of the bundles their networks come from
    (`origins`) and the splits the carries revoked (`revoked_splits`,
    `barred_cells`)."""
    out = {
        "schema": SCHEMA,
        "outcome": record.get("outcome"),
        "tier_keys_added": kept_tier_adds(record),
        "carried_cells": {
            str(c): dict(m)
            for c, m in sorted((record.get("carried_cells") or {}).items())
        },
    }
    for key in ("cells", "origins", "revoked_splits", "barred_cells"):
        if record.get(key):
            out[key] = copy.deepcopy(record[key])
    return out


def tier_doc(
    candidate_doc: Optional[Mapping[str, Any]],
    shipped_doc: Optional[Mapping[str, Any]],
    record: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """The whitelist_tier.json of the repaired `bundle`: the candidate's,
    each leaf's keys as the bundle serves them, a leaf with added keys
    naming them (`repair_keys`; none for a leaf of a carried cell, whose
    added keys the carry dropped), and a carried cell's leaves described as
    the bundle that ships describes them (marked CARRIED); None when the
    candidate has none."""
    if not isinstance(candidate_doc, dict):
        return None
    doc = copy.deepcopy(dict(candidate_doc))
    cells = dict(doc.get("cells") or {})
    carried = set(record.get("carried_cells") or {})
    theirs = (shipped_doc or {}).get("cells") or {}
    for label in list(cells):
        if sc.base_cell(label) in carried:
            cells.pop(label)
    models = bundle.get("models") or {}
    for label, entry in models.items():
        if sc.base_cell(label) in carried:
            cells[label] = dict(theirs.get(label) or {}, **{CARRIED: True})
        if label in cells:
            cells[label]["keys"] = tier_hashes(entry)
    for leaf, keys in kept_tier_adds(record).items():
        if leaf in cells:
            cells[leaf]["repair_keys"] = keys
    doc["cells"] = {k: cells[k] for k in sorted(cells)}
    rec = bundle.get(wt.RECORD)
    if isinstance(rec, dict):
        doc.update({k: v for k, v in rec.items() if k != "cells"})
    return doc


def cells_doc(
    cells: Mapping[str, Any], bundle: Mapping[str, Any], record: Mapping[str, Any]
) -> Dict[str, Any]:
    """cells.json of the repaired `bundle` from the candidate's `cells`,
    after the repair of `record`. Its `model_labels` and `fallback_parents`
    are the bundle's. Its entries are those of the cells the candidate
    trained that the bundle still serves with the candidate's networks,
    each counting the tier keys the bundle serves it with
    (`n_whitelist_tier_keys`; `n_repair_tier_keys` of them the repair
    added, after the entry's metrics were taken), and the counts of
    trained cells follow them. A carried base cell's labels keep no entry,
    even those the candidate shares with the bundle that ships: their
    networks are that bundle's, and their origins name the bundle that
    trained them (run_orchestrator.py `record_cell_origins`). `repair`
    says what changed (`summary`)."""
    models = bundle.get("models") or {}
    carried = set(record.get("carried_cells") or {})
    added = kept_tier_adds(record)
    tiered = wt.has_tier(bundle)
    out = dict(cells)
    out["model_labels"] = sorted(models)
    if "fallback_parents" in cells or bundle.get("fallback_parents"):
        out["fallback_parents"] = sorted(bundle.get("fallback_parents") or [])
    entries: List[Any] = []
    for c in cells.get("cells") or []:
        if not isinstance(c, dict):
            continue
        label = str(c.get("label"))
        if label not in models or sc.base_cell(label) in carried:
            continue
        c = dict(c)
        if tiered:
            c["n_whitelist_tier_keys"] = len(tier_hashes(models[label]))
        if label in added:
            c["n_repair_tier_keys"] = len(added[label])
        entries.append(c)
    out["cells"] = entries
    if "n_cells_trained" in cells:
        out["n_cells_trained"] = len(entries)
    if "n_subcells_trained" in cells:
        out["n_subcells_trained"] = sum(
            1 for c in entries if sc.is_subcell(str(c.get("label")))
        )
    out["repair"] = summary(record)
    return out


def metrics_doc(
    metrics: Mapping[str, Any], bundle: Mapping[str, Any], record: Mapping[str, Any]
) -> Dict[str, Any]:
    """metrics.json of the repaired `bundle` from the candidate's
    `metrics`, after the repair of `record`: what stage05 trained and
    measured stays as it recorded it, before the repair; the tier record
    (`whitelist_tier`, its counts the bundle's), the fallback parents and
    the serving switches are the bundle's, and `repair` says what changed
    (`summary`)."""
    out = dict(metrics)
    rec = bundle.get(wt.RECORD)
    if isinstance(rec, dict):
        out["whitelist_tier"] = {**(metrics.get("whitelist_tier") or {}), **rec}
    if "fallback_parents" in metrics or bundle.get("fallback_parents"):
        out["fallback_parents"] = sorted(bundle.get("fallback_parents") or [])
    if isinstance(bundle.get("serving"), dict):
        out["serving"] = dict(bundle["serving"])
    out["repair"] = summary(record)
    return out
