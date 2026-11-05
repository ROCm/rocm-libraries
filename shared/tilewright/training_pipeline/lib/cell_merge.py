# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Repair on refusal: a bundle the per-cell gate keeps out (lib/audit.py
`cell_gate`) for findings in a few base cells ships with those cells
repaired, instead of being refused whole.

The new bundle stays the default: only base cells the gate itself flags
change, never cells that merely read lower on the audit's point estimates
(a per-cell best-of picks noise and loses out of sample). A refusal is
repairable (`repairable`) when the gate refused, every deciding finding
names a base cell (a cell of a deciding group significantly worse beyond
gate.cell_tolerance or confidently fallen below gate.cell_floor, or a
confirmed new collapse below gate.tail_floor in any group), and at most
gate.repair_max_cells base cells are flagged; above that the bundle is
refused whole. Per flagged base cell, in this order:

  tier add   a collapse whose leaf network picks another kernel without its
             whitelist tier gets that kernel's key, kept only by the tail
             guard on the leaf's own data (lib/collapse_repair.py); the new
             network stays;
  carry      a finding still open (a collapse no kept key lifts, or a cell
             rule's finding) carries the whole base-cell subtree, every
             leaf's network, tier keys and feature statistics, from the
             incumbent: the bundle the run ships, which the gate compared
             with, found by the sha256 of its model files (`find_incumbent`)
             and checked again before its networks are read.

A carried subtree may lack splits the new bundle's run tree holds: they are
revoked (`revoked_rules`; lib/subcells.py `REVOKED_SPLITS` in the repaired
bundle's stage05/), so that the next round, the final retrain and every
reader of the run's split tree build on the incumbent's structure there,
and stage04b never splits that base cell again.

The repaired bundle is judged again (`judge`) by the same per-cell gate
against the same incumbent, each group under the protocol the refusal used
(a group measured only under a weaker one waits for its v2 bench), and by
an all-group check (`group_check`): no audit group (the exclusive context
or a CU budget) significantly worse than the incumbent beyond
gate.cell_tolerance, family-wise at gate.alpha over the groups with the
gate's sign flips. A final retrain must also stay not worse on the mean
(lib/audit.py `not_worse`); a round's bundle must also pass the round's
deploy gate, which its stage06 measures when it deploys the repaired bundle
(`RoundRepair`). The same audit GEMMs flagged the cells and judge the
repair, so the repaired bundle's audit numbers are selected on them
(`SELECTION_NOTE`): fresh held-out audits confirm them. A
round's repair that cannot be judged yet (the audit set did not measure the
repaired bundle, or a group waits for its v2 bench) stops the round for the
same command to try again, at most gate.repair_max_attempts times for one
bundle and settings; then the round is refused whole (`RoundRepair.wait`).

Installing a repaired bundle (`install`) keeps the originals in
`<owner>/repair/unrepaired/`, names the installed models.pt by its sha256
(`INSTALLED`, written before the copy) and replaces stage05's models.pt
last, so that an interrupted install is finished (`installed_repair`)
before anything reads the bundle; a round's install that its stage06 then
refuses is recorded as refused first and withdrawn the same way
(`RoundRepair.refuse`, `withdraw`), so that the repaired bundle is never
measured again. The refusal stage06 writes into its deploy record is the
decision (`kept_out`), whether or not the orchestrator marked that stage06
complete, and the stage06 that measures an installed repair is not
attempted twice (`RoundRepair.measure`).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence
from typing import Tuple

import numpy as np

from . import audit as auditlib
from . import run_bundles as rb
from . import subcells as sc
from . import ui

SCHEMA = "cell_merge_v1"
REPAIR_DIR = "repair"
UNREPAIRED = "unrepaired"
INSTALLED = "installed.json"
# A round's repair decision (<round>/repair/), and the repair script's record.
DECISION = "decision.json"
REPAIR_RECORD = "repair.json"
# The bundle files a repair replaces in stage05/, models.pt last.
REPAIRED_FILES = ("cells.json", "whitelist_tier.json", "metrics.json", "models.pt")
DEFAULT_MAX_CELLS = 3
# gate.repair_max_attempts: the attempts of a round's repair whose repaired
# bundle waits for its measurement before the round is refused whole.
DEFAULT_MAX_ATTEMPTS = 3
# The outcomes of a round's repair that stand for its bundle under the same
# settings (`RoundRepair.decided`). "repair failed" (scripts/repair_bundle.py
# stopped on an error, or could not start) is tried again.
FINAL_OUTCOMES = ("refused", "not repairable", "not repaired")
# scripts/repair_bundle.py's exit status when no repair is possible.
NOT_REPAIRED_EXIT = 3
# What a round's decision keeps of the deploy record of the stage06 that
# measured its repaired bundle.
STAGE06_KEYS = ("deployed", "deploy_gate", "cell_gate", "sha256", "models_sha256")
SELECTION_NOTE = (
    "the same audit GEMMs flagged these cells and judged the repaired bundle "
    "again, so its audit numbers are selected on them (selected_by_audit); "
    "fresh held-out audits confirm them"
)
FINDING_KINDS = ("regressed", "floor", "collapse")
ShapeKey = Tuple[int, int, int, int]


# ── the gate's findings ─────────────────────────────────────────────────────


def flagged_cells(cell_gate: Mapping[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """{base cell: its deciding findings} of a `lib.audit.cell_gate` result:
    the regressed and fallen cells of the groups the cell rules judge
    (`deciding`) and the confirmed new collapses that decide, in any group.
    A finding that names no cell is listed under ""."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for label, g in (cell_gate.get("groups") or {}).items():
        if not isinstance(g, dict) or "reason" in g:
            continue
        if g.get("deciding"):
            for kind in ("regressed", "floor"):
                for c in g.get(kind) or []:
                    entry = {"kind": kind, "group": str(label)}
                    entry.update(
                        {
                            k: c.get(k)
                            for k in ("n", "ratio", "ratio_ci", "p", "before", "after")
                        }
                    )
                    cell = sc.base_cell(str(c.get("cell") or ""))
                    out.setdefault(cell, []).append(entry)
        for t in g.get("tail") or []:
            if not t.get("decides"):
                continue
            cell = sc.base_cell(str(t.get("cell") or ""))
            out.setdefault(cell, []).append(
                {
                    "kind": "collapse",
                    "group": str(label),
                    "shape": [int(v) for v in t.get("shape") or []],
                    "before": t.get("before"),
                    "after": t.get("after"),
                }
            )
    return out


def repairable(
    cell_gate: Optional[Mapping[str, Any]], max_cells: int
) -> Tuple[bool, str, Dict[str, List[Dict[str, Any]]]]:
    """(the refusal of `cell_gate` is repaired, why, the flagged cells): the
    gate refused, every deciding finding names a base cell, and at most
    `max_cells` base cells are flagged."""
    if not isinstance(cell_gate, Mapping) or cell_gate.get("passed") is not False:
        return False, "the per-cell gate did not keep it out", {}
    flagged = flagged_cells(cell_gate)
    if not flagged:
        return False, "no finding of the per-cell gate decides", {}
    if "" in flagged:
        return False, "a deciding finding of the per-cell gate names no cell", flagged
    if len(flagged) > int(max_cells):
        return (
            False,
            f"{len(flagged)} base cells flagged, more than gate.repair_max_cells "
            f"{int(max_cells)}: refused whole",
            flagged,
        )
    return True, f"{len(flagged)} base cell(s) flagged: {', '.join(flagged)}", flagged


def cell_rule_cells(flagged: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[str]:
    """The flagged base cells with a cell rule's finding (regressed or
    fallen): no tier add answers one, so they are carried."""
    return sorted(
        c
        for c, findings in flagged.items()
        if any(f.get("kind") in ("regressed", "floor") for f in findings)
    )


def finding_text(f: Mapping[str, Any]) -> str:
    """One flagged finding (`flagged_cells`) as the run's log shows it."""
    if f.get("kind") == "collapse":
        m, n, k, b = (list(f.get("shape") or []) + [0, 0, 0, 1])[:4]
        return (
            f"{f.get('group')} {m}x{n}x{k}"
            + (f" batch {b}" if b != 1 else "")
            + f" collapsed {float(f.get('after') or 0) * 100:.2f}% (was "
            f"{float(f.get('before') or 0) * 100:.2f}%)"
        )
    what = "regressed" if f.get("kind") == "regressed" else "fell below the floor"
    p = f.get("p")
    return f"{f.get('group')} {what} x{float(f.get('ratio') or 0):.4f}" + (
        f" p {float(p):.2g}" if isinstance(p, (int, float)) else ""
    )


def flagged_lines(flagged: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[str]:
    """One line per flagged base cell with its findings."""
    return [
        f"{cell}: " + "; ".join(finding_text(f) for f in findings)
        for cell, findings in sorted(flagged.items())
    ]


def cell_actions(
    flagged: Mapping[str, Sequence[Mapping[str, Any]]],
    record: Mapping[str, Any],
    candidate_sha256: Optional[str],
    incumbent_sha256: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    """{flagged base cell: its findings, what the repair of `record` did
    there (`tier add`: the new networks kept, with keys added; `subtree
    carry`: the cell and every leaf from the incumbent; `none`) and the
    sha256 of the bundle whose networks now serve it}."""
    from .collapse_repair import kept_tier_adds

    carried = set(record.get("carried_cells") or {})
    added = kept_tier_adds(record)
    out: Dict[str, Dict[str, Any]] = {}
    for cell, findings in sorted(flagged.items()):
        keys = {leaf: v for leaf, v in added.items() if sc.base_cell(leaf) == cell}
        if cell in carried:
            action, origin = "subtree carry", incumbent_sha256
        elif keys:
            action, origin = "tier add", candidate_sha256
        else:
            action, origin = "none", candidate_sha256
        out[cell] = {
            "findings": [dict(f) for f in findings],
            "action": action,
            "tier_keys_added": keys,
            "origin_sha256": origin,
        }
    return out


# ── split revocation ─────────────────────────────────────────────────────────


def revoked_rules(
    tree: Mapping[str, sc.SplitRule],
    incumbent_labels: Iterable[str],
    cells: Iterable[str],
) -> List[sc.SplitRule]:
    """The rules of the run's split tree `tree` inside the base cells
    `cells` that the incumbent's own tree (its labels', which its model
    serves with) lacks: carrying those cells' subtrees from the incumbent
    revokes them. ValueError when the incumbent splits a node of those cells
    that `tree` does not split, or splits it otherwise: the repaired bundle
    would then serve a structure no round trains."""
    theirs = sc.split_tree_from_labels(str(x) for x in incumbent_labels)
    wanted = set(cells)
    out: List[sc.SplitRule] = []
    for parent, rule in sorted(theirs.items()):
        if sc.base_cell(parent) not in wanted:
            continue
        mine = tree.get(parent)
        if mine is None or (mine.axis, int(mine.threshold)) != (
            rule.axis,
            int(rule.threshold),
        ):
            have = "no split" if mine is None else f"{mine.axis}{mine.threshold}"
            raise ValueError(
                f"the incumbent splits {parent} at {rule.axis}{rule.threshold}, "
                f"the run's split tree has {have} there"
            )
    for parent in sorted(tree):
        if sc.base_cell(parent) in wanted and parent not in theirs:
            out.append(tree[parent])
    return out


def _revoked_rules_of(doc: Mapping[str, Any]) -> Dict[str, sc.SplitRule]:
    return {
        str(r["cell"]): sc.SplitRule.from_json(r)
        for r in doc.get("revoked") or []
        if isinstance(r, dict) and r.get("cell")
    }


def inherit_revocations(
    round_dir: Path, base: Optional[Path], priors: Sequence[Path]
) -> bool:
    """Copy the revocations of the bundle `base` a round built on into the
    round's stage05/ when `base` is none of the round's `priors` (a final
    retrain): later rounds read the run's tree through their prior rounds
    only. The round's own record keeps what it holds and is written only
    when it lacks some of them; whether it was."""
    if base is None or Path(base) in [Path(p) for p in priors]:
        return False
    theirs = sc.revocations(base)
    mine = sc.revocations(round_dir)
    rules, barred = _revoked_rules_of(mine), set(mine.get("barred_cells") or [])
    new = {k: r for k, r in _revoked_rules_of(theirs).items() if k not in rules}
    more = set(theirs.get("barred_cells") or []) - barred
    if not new and not more:
        return False
    extra = {
        k: v
        for k, v in mine.items()
        if k not in ("schema", "revoked", "barred_cells", "inherited_from")
    }
    sc.write_revocations(
        Path(round_dir) / "stage05" / sc.REVOKED_SPLITS,
        [*rules.values(), *new.values()],
        barred | more,
        **extra,
        inherited_from=str(Path(base).name),
    )
    return True


# ── the all-group check ──────────────────────────────────────────────────────


def group_check(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    settings: Optional[Mapping[str, Any]] = None,
    bootstrap: int = auditlib.DEFAULTS["bootstrap"],
    official: Optional[str] = None,
) -> Dict[str, Any]:
    """Whether any audit group of `after` (the exclusive context and every
    CU budget either record evaluated) is significantly worse than
    `before`: each group paired GEMM by GEMM as the per-cell gate pairs it
    (`lib.audit.paired_change`), the groups tested as one family with the
    gate's sign flips (`lib.audit.decline_p`, `bootstrap` flips), and a
    group flagged when its p is below gate.alpha and its equal-weight cell
    mean ratio below 1 - gate.cell_tolerance. `passed` is None when no group
    pairs or the flips cannot reach alpha."""
    st = {**auditlib.CELL_GATE_DEFAULTS, **dict(settings or {})}
    alpha, tol = float(st["alpha"]), float(st["cell_tolerance"])
    labels = list(
        dict.fromkeys(
            [
                auditlib.EXCLUSIVE,
                *auditlib._group_labels(before),
                *auditlib._group_labels(after),
            ]
        )
    )
    groups: Dict[str, Dict[str, Any]] = {}
    family: Dict[str, Tuple[List[ShapeKey], np.ndarray]] = {}
    for label in labels:
        change, why, proto = auditlib.paired_change(
            before, after, label, bootstrap, official
        )
        if change is None:
            groups[label] = {"protocol": proto, "reason": why}
            continue
        a = auditlib.per_gemm_of(auditlib.protocol_record(before, proto), label)
        b = auditlib.per_gemm_of(auditlib.protocol_record(after, proto), label)
        shapes = sorted(set(a) & set(b))
        family[label] = (
            shapes,
            np.log([b[s] for s in shapes]) - np.log([a[s] for s in shapes]),
        )
        groups[label] = {
            "protocol": proto,
            "n_gemms": change["n_gemms"],
            "n_cells": change["n_cells"],
            "ratio": change["cell_mean_ratio"],
            "ratio_ci": change["cell_mean_ratio_ci"],
            "gemm_geomean_ratio": change["gemm_geomean_ratio"],
        }
    flips = max(0, int(bootstrap))
    testable = bool(family) and flips >= auditlib.min_flips(alpha)
    p = auditlib.decline_p(family, flips, seed=0) if testable else {}
    worse = []
    for label, g in groups.items():
        if "reason" in g:
            continue
        g["p"] = p.get(label)
        g["worse"] = bool(
            g["p"] is not None and g["p"] < alpha and g["ratio"] < 1.0 - tol
        )
        if g["worse"]:
            worse.append(label)
    passed: Optional[bool] = (not worse) if testable else None
    shown = [
        f"{k} x{groups[k]['ratio']:.4f} p {groups[k]['p']:.2g}" for k in worse
    ] or [f"{k} x{g['ratio']:.4f}" for k, g in groups.items() if "reason" not in g]
    if passed is None:
        text = (
            "no group pairs the two bundles"
            if not family
            else f"{flips} sign flips cannot reach alpha {alpha:g}"
        )
    elif worse:
        text = (
            f"{len(worse)} group(s) significantly worse beyond "
            f"{tol * 100:.1f}% (family-wise alpha {alpha:g} over {len(family)} "
            f"groups): " + ", ".join(shown)
        )
    else:
        text = (
            f"no group significantly worse beyond {tol * 100:.1f}% (family-wise "
            f"alpha {alpha:g} over {len(family)} groups): " + ", ".join(shown)
        )
    return {
        "passed": passed,
        "alpha": alpha,
        "tolerance": tol,
        "worse": worse,
        "groups": groups,
        "reason": text,
    }


# ── the judgement of a repaired bundle ──────────────────────────────────────


def weaker_groups(
    refusal: Mapping[str, Any], after: Optional[Mapping[str, Any]], official: str
) -> List[str]:
    """The groups the refusal `refusal` judged that the gate result `after`
    judged under a weaker protocol than it, or not at all."""

    def rank(label: str, proto: Any) -> int:
        order = auditlib.preference(official, label)
        return order.index(proto) if proto in order else len(order)

    now = (after or {}).get("groups") or {}
    return sorted(
        k
        for k, g in (refusal.get("groups") or {}).items()
        if isinstance(g, dict)
        and "reason" not in g
        and rank(k, (now.get(k) or {}).get("protocol")) > rank(k, g.get("protocol"))
    )


@dataclass
class Verdict:
    """The judgement of a repaired bundle: `status` pass, fail or wait (a
    group lacks the protocol the refusal used; the same command tries
    again), with why and what each rule found."""

    status: str
    reason: str
    cell_gate: Optional[Dict[str, Any]] = None
    groups: Optional[Dict[str, Any]] = None
    not_worse: Optional[Tuple[Any, str, Any, str]] = None
    weaker: List[str] = field(default_factory=list)

    def record(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "status": self.status,
            "reason": self.reason,
            "cell_gate": (self.cell_gate or {}).get("reason"),
            "n_tail_new": (self.cell_gate or {}).get("n_tail_new"),
            "n_tail_healed": (self.cell_gate or {}).get("n_tail_healed"),
            "all_groups": self.groups,
        }
        if self.not_worse is not None:
            out.update(not_worse=self.not_worse[0], change=self.not_worse[1])
        if self.weaker:
            out["weaker_groups"] = list(self.weaker)
        return out


def judge(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    refusal: Mapping[str, Any],
    settings: Mapping[str, Any],
    bootstrap: int,
    official: str,
    *,
    mean_rule: bool,
) -> Verdict:
    """The repaired bundle's audit record `after` judged against the
    incumbent's `before`: the per-cell gate with `settings`, under
    protocols no weaker than the refusal `refusal` used, the all-group
    check (`group_check`), and with `mean_rule` the equal-weight cell mean
    not below the incumbent's (a final retrain's rule)."""
    gate = auditlib.cell_gate(before, after, settings, bootstrap, official=official)
    weaker = weaker_groups(refusal, gate, official)
    if weaker:
        return Verdict(
            "wait",
            f"the audit set measured the repaired bundle under other protocols "
            f"than the gate that kept it out ({', '.join(weaker)}): its picks "
            f"wait for their v2 measurement",
            gate,
            weaker=weaker,
        )
    groups = group_check(before, after, settings, bootstrap, official)
    nw = auditlib.not_worse(before, after, official=official) if mean_rule else None
    why = []
    if nw is not None and nw[0] is not True:
        why.append(
            f"{'worse' if nw[0] is False else 'not comparable'} on the audit set: "
            f"{nw[1]}"
        )
    if gate.get("passed") is not True:
        why.append(f"the per-cell gate keeps it out: {gate.get('reason')}")
    if groups.get("passed") is False:
        why.append(f"the all-group check keeps it out: {groups.get('reason')}")
    if why:
        return Verdict("fail", "; ".join(why), gate, groups, nw)
    text = f"per-cell gate: {gate.get('reason')}; all groups: {groups.get('reason')}"
    if nw is not None:
        text = f"not worse on the audit set: {nw[1]}; " + text
    return Verdict("pass", text, gate, groups, nw)


# ── the incumbent, by content ────────────────────────────────────────────────


def sha256_file(path: Path) -> Optional[str]:
    h = hashlib.sha256()
    try:
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def _read(path: Path) -> Optional[Any]:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def bundle_dirs(run_dir: Path) -> List[Path]:
    """The directories of a run that hold or held one of its bundles: its
    rounds, its final retrain and the one trained beside it, and the final
    retrains a later one moved to stale/."""
    run = Path(run_dir)
    out = sorted(
        (p for p in run.glob("round_*") if rb.round_index(p) is not None),
        key=lambda p: rb.round_index(p) or 0,
    )
    out += [run / rb.FINAL_RETRAIN_DIR, run / rb.FINAL_RETRAIN_NEXT_DIR]
    out += sorted(run.glob(f"stale/*/{rb.FINAL_RETRAIN_DIR}*"))
    return [p for p in out if p.is_dir()]


@dataclass
class Incumbent:
    """The bundle a repair carries cells from: its directory where it was
    found, the sha256 of the model its gate scored (`staged_sha256`, its
    stage06's staged file) and of its stage05/models.pt, and how the two
    were tied together (`verified`)."""

    dir: Path
    staged_sha256: Optional[str]
    models_sha256: str
    verified: str

    def to_json(self, run_dir: Path) -> Dict[str, Any]:
        try:
            where = str(Path(self.dir).relative_to(Path(run_dir)))
        except ValueError:
            where = str(self.dir)
        return {
            "dir": where,
            "staged_sha256": self.staged_sha256,
            "models_sha256": self.models_sha256,
            "verified": self.verified,
        }


def find_incumbent(
    run_dir: Path,
    *,
    staged_sha256: Optional[str] = None,
    models_sha256: Optional[str] = None,
    hint: Optional[Path] = None,
) -> Optional[Incumbent]:
    """The bundle directory of the run (`hint` first, then `bundle_dirs`)
    whose stage06 deployed the model of sha256 `staged_sha256` (the staged
    file's, or that file's without its alignment veto, `veto_free_sha256`,
    which a round's gates score) and whose stage05/models.pt is the one that
    deploy packed (its deploy record's `models_sha256`, when it records one)
    and has sha256 `models_sha256` when given; None when none has. Never a
    directory by its name: a final retrain a later one replaced moves to
    stale/, and its place holds another bundle."""
    if not staged_sha256 and not models_sha256:
        return None
    seen = set()
    for d in ([Path(hint)] if hint else []) + bundle_dirs(run_dir):
        key = str(Path(d).resolve()) if Path(d).exists() else str(d)
        if key in seen or not Path(d).is_dir():
            continue
        seen.add(key)
        rec = _read(Path(d) / "stage06" / rb.DEPLOY_RECORD) or {}
        if staged_sha256 and staged_sha256 not in (
            rec.get("sha256"),
            rec.get("veto_free_sha256"),
        ):
            continue
        m = sha256_file(Path(d) / "stage05" / "models.pt")
        if m is None or (models_sha256 and m != models_sha256):
            continue
        packed = rec.get("models_sha256")
        if packed and packed != m:
            continue
        how = (
            "deploy record: staged model and models.pt"
            if packed
            else ("deploy record: staged model" if staged_sha256 else "models.pt")
        )
        if staged_sha256 and staged_sha256 != rec.get("sha256"):
            how = how.replace("staged model", "staged model without its alignment veto")
        return Incumbent(Path(d), staged_sha256 or rec.get("sha256"), m, how)
    return None


# ── installing a repaired bundle ─────────────────────────────────────────────


def _put(src: Path, dst: Path) -> None:
    part = dst.parent / f".{dst.name}.repair"
    shutil.copy2(src, part)
    os.replace(part, dst)


def _write(path: Path, rec: Mapping[str, Any]) -> None:
    from .fs import atomic_write_text

    atomic_write_text(Path(path), json.dumps(_safe(rec), indent=2) + "\n")


def _safe(obj: Any) -> Any:
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {str(k): _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def carried_labels(record: Mapping[str, Any]) -> List[str]:
    """The labels a repair of `record` took from the incumbent."""
    return [
        str(label)
        for moved in (record.get("carried_cells") or {}).values()
        for label in (moved or {}).get("carried") or []
    ]


Origins = Callable[[Path, Path, List[str]], None]


def copy_repair(
    owner: Path,
    d: Path,
    record: Mapping[str, Any],
    base: Optional[Path],
    *,
    files: Sequence[str] = REPAIRED_FILES,
    audit: Optional[str] = None,
    origins: Optional[Origins] = None,
) -> None:
    """The files of the repair in `d` into `<owner>/stage05` (the revoked
    splits too) and, with `audit`, the owner's audit record; the origins of
    the cells it carried from `base` (`origins(stage05, base, labels)`);
    models.pt last, so that until the install is complete stage05/models.pt
    is the unrepaired one."""
    s5 = Path(owner) / "stage05"
    for name in (*files, sc.REVOKED_SPLITS):
        if name != "models.pt" and (d / name).is_file():
            _put(d / name, s5 / name)
    if audit and (d / audit).is_file():
        _put(d / audit, Path(owner) / audit)
    carried = carried_labels(record)
    if carried and base is not None and origins is not None:
        origins(s5, Path(base), carried)
    _put(d / "models.pt", s5 / "models.pt")


def install(
    owner: Path,
    d: Path,
    record: Dict[str, Any],
    base: Path,
    *,
    copy: Callable[[Path, Dict[str, Any], Path], None],
    files: Sequence[str] = REPAIRED_FILES,
    audit: Optional[str] = None,
    installed: str = INSTALLED,
    record_name: str = REPAIR_RECORD,
    keep_stage06: bool = False,
) -> Path:
    """Make the repaired bundle in `d` the bundle of `owner` (a round or a
    final retrain directory): the originals of `files`, stage05's
    cell_origins.json and revoked splits and, with `audit`, the owner's
    audit record (with `keep_stage06`, its whole stage06/) are kept once in
    `<d>/unrepaired`; `installed` names the installed models.pt by its
    sha256, the unrepaired one and the incumbent `base` by theirs, written
    before `copy` replaces models.pt last. The kept directory."""
    owner, d = Path(owner), Path(d)
    s5 = owner / "stage05"
    keep = d / UNREPAIRED
    if not keep.is_dir():
        tmp = d / f"{UNREPAIRED}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        for name in (*files, rb.CELL_ORIGINS, sc.REVOKED_SPLITS):
            if (s5 / name).is_file():
                shutil.copy2(s5 / name, tmp / name)
        if audit and (owner / audit).is_file():
            shutil.copy2(owner / audit, tmp / audit)
        if keep_stage06 and (owner / "stage06").is_dir():
            shutil.copytree(owner / "stage06", tmp / "stage06", symlinks=True)
        os.rename(tmp, keep)
    inc = record.get("incumbent") if isinstance(record.get("incumbent"), dict) else {}
    record["installed"] = {
        "models_sha256": sha256_file(d / "models.pt"),
        "unrepaired": str(keep.relative_to(owner)),
        "unrepaired_sha256": sha256_file(keep / "models.pt"),
        "carried_from": Path(base).name,
        "incumbent_models_sha256": inc.get("models_sha256")
        or sha256_file(Path(base) / "stage05" / "models.pt"),
        "state": "installed",
    }
    _write(d / installed, record)
    _write(d / record_name, record)
    copy(d, record, Path(base))
    return keep


def resolve_base(record: Mapping[str, Any], run_dir: Path) -> Optional[Path]:
    """The incumbent of an installed repair's `record`: found by the sha256
    it recorded, its directory only for a record without one."""
    inst = record.get("installed") or {}
    hint = Path(run_dir) / str(inst.get("carried_from") or "")
    sha = inst.get("incumbent_models_sha256")
    if not sha:
        return hint if inst.get("carried_from") else None
    found = find_incumbent(run_dir, models_sha256=sha, hint=hint)
    if found is None:
        ui.warn(
            "repair",
            f"the bundle the repair carried cells from (models.pt sha256 "
            f"{sha[:12]}) is no longer in {run_dir}: the carried cells keep no "
            f"origin",
        )
    return found.dir if found is not None else None


def installed_repair(
    owner: Path,
    d: Path,
    *,
    finish: Callable[[Dict[str, Any]], None],
    installed: str = INSTALLED,
) -> Optional[Dict[str, Any]]:
    """The record of the repair installed into `<owner>/stage05` while its
    models.pt is the installed one; an install an interruption left
    unfinished (models.pt, which the copy replaces last, still the
    unrepaired one) is finished first (`finish(record)` copies it again).
    None otherwise, and for a repair a round withdrew."""
    rec = _read(Path(d) / installed)
    if not isinstance(rec, dict):
        return None
    inst = rec.get("installed") or {}
    sha = inst.get("models_sha256")
    if not sha or inst.get("state", "installed") != "installed":
        return None
    now = sha256_file(Path(owner) / "stage05" / "models.pt")
    if now != sha:
        kept = sha256_file(Path(owner) / str(inst.get("unrepaired")) / "models.pt")
        if sha256_file(Path(d) / "models.pt") != sha or now != kept:
            return None
        finish(rec)
    return rec


def withdraw(owner: Path, d: Path, *, installed: str = INSTALLED) -> bool:
    """Put the unrepaired bundle and its stage06 back in place of a round's
    installed repair (`install` with `keep_stage06`): the record says so
    first, the kept files replace the repaired ones (a file the unrepaired
    bundle lacked is removed), the repaired bundle's stage06 moves to
    `<d>/stage06.repaired`, and models.pt goes last, so that
    `finish_withdrawal` completes an interrupted one. False when nothing
    was kept."""
    owner, d = Path(owner), Path(d)
    rec = _read(d / installed)
    keep = d / UNREPAIRED
    if not isinstance(rec, dict) or not (keep / "models.pt").is_file():
        return False
    rec.setdefault("installed", {})["state"] = "withdrawn"
    _write(d / installed, rec)
    _restore(owner, d)
    return True


def _restore(owner: Path, d: Path) -> None:
    s5, keep = owner / "stage05", d / UNREPAIRED
    for name in (*REPAIRED_FILES, rb.CELL_ORIGINS, sc.REVOKED_SPLITS):
        if name == "models.pt":
            continue
        if (keep / name).is_file():
            _put(keep / name, s5 / name)
        else:
            (s5 / name).unlink(missing_ok=True)
    kept06, s06 = keep / "stage06", owner / "stage06"
    if kept06.is_dir() and _read(s06 / rb.DEPLOY_RECORD) != _read(
        kept06 / rb.DEPLOY_RECORD
    ):
        repaired = d / "stage06.repaired"
        if s06.exists():
            shutil.rmtree(repaired, ignore_errors=True)
            os.rename(s06, repaired)
        tmp = owner / ".stage06.restore"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(kept06, tmp, symlinks=True)
        os.rename(tmp, s06)
    _put(keep / "models.pt", s5 / "models.pt")


def kept_out(rec: Mapping[str, Any], sha: Optional[str]) -> bool:
    """The deploy record `rec` of a stage06 kept the bundle whose models.pt
    has sha256 `sha` out: deployed false, with the reason stage06's refusal
    writes as its last act (`not_deployed`) and no error of a failed
    install. That record is the refusal whether or not the orchestrator
    marked the stage06 complete: a code change found as it ended, or a kill
    after it wrote the record, leaves it unmarked."""
    return bool(
        sha
        and rec.get("models_sha256") == sha
        and rec.get("deployed") is False
        and rec.get("not_deployed")
        and not rec.get("error")
    )


def finish_withdrawal(owner: Path, d: Path, *, installed: str = INSTALLED) -> bool:
    """Complete a withdrawal an interruption left unfinished; whether the
    owner's stage05 holds the unrepaired bundle after a withdrawal."""
    rec = _read(Path(d) / installed)
    inst = (rec or {}).get("installed") or {}
    if inst.get("state") != "withdrawn":
        return False
    now = sha256_file(Path(owner) / "stage05" / "models.pt")
    if now == inst.get("models_sha256"):
        _restore(Path(owner), Path(d))
        return True
    return now == inst.get("unrepaired_sha256")


# ── a round's repair ─────────────────────────────────────────────────────────


class RoundRepair:
    """gate.repair for one round: after its stage06 refused the round's
    bundle on the per-cell gate alone, the repair of the flagged base cells
    (scripts/repair_bundle.py into `<round>/repair/`), protocol v2 of the
    repaired bundle with audit.v2 rounds, its judgement on the audit set
    against the incumbent (`judge`: the per-cell gate under the refusal's
    protocols and the all-group check), then its install into
    `<round>/stage05` and stage06 again, which measures it on the round's
    deploy gate and deploys it, or refuses it and the install is withdrawn
    (`refuse`). The round's own records stay: the unrepaired bundle and its
    stage06 in `<round>/repair/unrepaired/`. `<round>/repair/decision.json`
    records the flagged cells, what was done in each, the judgement, the
    cells' origins by sha256, the revoked splits and the outcome. A decision
    made for this bundle with these settings is never made again
    (`decided`), and a repaired bundle a stage06 kept out is never measured
    again (`resume`, `measure`), so that measuring again cannot turn a
    refusal into a pass. `stage06(**kw)` runs the round's stage06, with
    `retries` overriding runtime.stage_retries."""

    def __init__(
        self,
        pipeline: Any,
        rd: Path,
        index: int,
        stage06: Callable[..., int],
        origins: Origins,
        repair_script: Path,
        audit_eval_script: Path,
        cells: Any,
    ) -> None:
        self.p = pipeline
        self.rd = Path(rd)
        self.index = index
        self.stage06 = stage06
        self.origins = origins
        self.repair_script = Path(repair_script)
        self.audit_eval_script = Path(audit_eval_script)
        self.cells = cells
        self.d = self.rd / REPAIR_DIR

    # the install, as the shared helpers do it
    def copy(self, d: Path, record: Mapping[str, Any], base: Optional[Path]) -> None:
        copy_repair(self.rd, d, record, base, origins=self.origins)

    def resume(self) -> int:
        """Finish what an interruption left before stage06 reads the round's
        bundle: a withdrawal, an install, or the refusal of the installed
        repaired bundle, recorded in decision.json or only in the deploy
        record of a stage06 that measured it and kept it out (`kept_out`,
        complete or not), which is then recorded and withdrawn (`refuse`).
        1 when such a bundle cannot be withdrawn."""
        if finish_withdrawal(self.rd, self.d):
            return 0
        rec = installed_repair(
            self.rd,
            self.d,
            finish=lambda r: self.copy(self.d, r, resolve_base(r, self.p.run_dir)),
        )
        if rec is None:
            return 0
        dec = _read(self.d / DECISION) or {}
        if dec.get("outcome") == "refused":
            return self.refuse(dec, {})
        done = _read(self.rd / "stage06" / rb.DEPLOY_RECORD) or {}
        if kept_out(done, (rec.get("installed") or {}).get("models_sha256")):
            return self.refuse(dec, done)
        return 0

    def measure(self) -> int:
        """stage06 of the round's bundle; of an installed repair in one
        attempt only, so that a stage06 that wrote its refusal and then died
        is never followed by another measurement of the bundle: the next
        invocation takes that refusal (`resume`)."""
        inst = ((_read(self.d / INSTALLED) or {}).get("installed")) or {}
        if inst.get("state") == "installed" and inst.get(
            "models_sha256"
        ) == sha256_file(self.rd / "stage05" / "models.pt"):
            return self.stage06(retries=0)
        return self.stage06()

    def refuse(self, dec: Dict[str, Any], rec: Mapping[str, Any]) -> int:
        """Record that stage06 keeps the installed repaired bundle out (its
        deploy record `rec`), then withdraw the install (`withdraw`): the
        decision first, so that an interruption anywhere leaves a refusal to
        finish, never a repair to make again. 1, with an error, when nothing
        was kept to withdraw to."""
        if dec.get("outcome") != "refused":
            why = rec.get("not_deployed") or "stage06 kept it out"
            dec["stage06"] = {k: rec.get(k) for k in STAGE06_KEYS}
            dec.update(
                outcome="refused",
                deployed=False,
                decided=_now(),
                reason=f"the repaired bundle passed the audit set, but its stage06 "
                f"keeps it out: {why}; the unrepaired bundle stays the round's, "
                f"refused",
            )
            _write(self.d / DECISION, dec)
            ui.warn("repair", f"{self.rd.name}: {dec['reason']}")
        if withdraw(self.rd, self.d):
            return 0
        ui.err(
            "repair",
            f"{self.rd.name}: the repaired bundle its stage06 keeps out cannot be "
            f"withdrawn: {self.d / UNREPAIRED} holds no unrepaired bundle to put "
            f"back; restore it and run the same command",
        )
        return 1

    def settings(self) -> Dict[str, Any]:
        cfg = self.p.cfg
        return {
            "max_cells": int(self.p.get("gate.repair_max_cells")),
            "cell_gate": auditlib.cell_gate_settings(cfg),
            "official": str(self.p.get("audit.official")),
            "bootstrap": int(auditlib.settings(cfg)["bootstrap"]),
        }

    def decide(self, rec: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """(exit status, the round's deploy record) after the round's
        stage06 wrote `rec`."""
        s5 = self.rd / "stage05"
        now = sha256_file(s5 / "models.pt")
        inst = ((_read(self.d / INSTALLED) or {}).get("installed")) or {}
        dec = _read(self.d / DECISION) or {}
        if inst.get("state") == "installed" and inst.get("models_sha256") == now:
            if dec.get("outcome") == "refused" or rec.get("deployed", True) is False:
                rc = self.refuse(dec, rec)
                return rc, _read(self.rd / "stage06" / rb.DEPLOY_RECORD) or rec
            dec["stage06"] = {k: rec.get(k) for k in STAGE06_KEYS}
            if dec.get("outcome") != "repaired":
                dec.update(outcome="repaired", deployed=True, decided=_now())
                dec["reason"] = (
                    f"{dec.get('reason') or 'repaired'}; its stage06 deployed "
                    f"it ({(rec.get('deploy_gate') or {}).get('reason')})"
                )
                _write(self.d / DECISION, dec)
            return 0, rec
        if rec.get("deployed", True) is not False or not self.p.get("gate.repair"):
            return 0, rec
        if self.cells is None or getattr(self.cells, "record_only", False):
            return 0, rec
        if (_read(self.rd / "stage06" / "cell_gate.json") or {}).get(
            "passed"
        ) is not False:
            return 0, rec
        st = self.settings()
        if self.decided(dec, inst, now, st):
            ui.warn(
                "repair",
                f"{self.rd.name}: this bundle's repair was decided before: "
                f"{dec.get('reason')}",
            )
            return 0, rec
        return self.repair(rec, now, st)

    def decided(
        self,
        dec: Dict[str, Any],
        inst: Mapping[str, Any],
        now: Optional[str],
        st: Mapping[str, Any],
    ) -> bool:
        """The decision `dec` stands for the round's bundle `now` under the
        settings `st`: one of FINAL_OUTCOMES for it, or a repair of it that
        its stage06 kept out and the round withdrew (`inst`, the install
        record), which `dec` is made to say when it does not."""
        if dec.get("settings") != st:
            return False
        if dec.get("candidate_sha256") == now and dec.get("outcome") in FINAL_OUTCOMES:
            return True
        if inst.get("state") != "withdrawn" or inst.get("unrepaired_sha256") != now:
            return False
        kept = _read(self.d / "stage06.repaired" / rb.DEPLOY_RECORD) or {}
        dec.update(
            outcome="refused",
            deployed=False,
            decided=_now(),
            reason=f"the repaired bundle was withdrawn after its stage06 kept it "
            f"out: {kept.get('not_deployed') or 'see ' + str(self.d)}",
        )
        _write(self.d / DECISION, dec)
        return True

    def _decision(self, now: Optional[str], st: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "schema": SCHEMA,
            "round": self.rd.name,
            "candidate_sha256": now,
            "settings": dict(st),
            "selected_by_audit": True,
            "selection_note": SELECTION_NOTE,
        }

    def _stop(self, dec: Dict[str, Any], outcome: str, why: str) -> None:
        dec.update(outcome=outcome, reason=why, deployed=False, decided=_now())
        self.d.mkdir(parents=True, exist_ok=True)
        _write(self.d / DECISION, dec)
        ui.warn("repair", f"{self.rd.name} stays out ({outcome}): {why}")

    def wait(self, dec: Dict[str, Any], attempts: int, why: str) -> int:
        """The repaired bundle could not be judged now (`why`) in the
        `attempts`-th attempt for this bundle and these settings: recorded
        (outcome `wait`), the round stops with 1 and the same command tries
        again; the gate.repair_max_attempts-th refuses the round whole
        (`not repaired`, which stands), and the round goes on with 0."""
        cap = int(self.p.get("gate.repair_max_attempts"))
        dec["attempts"] = attempts
        if attempts >= cap:
            self._stop(
                dec,
                "not repaired",
                f"{why}, in each of {attempts} attempt(s) "
                f"(gate.repair_max_attempts {cap}): refused whole",
            )
            return 0
        dec.update(outcome="wait", reason=why, deployed=False, decided=_now())
        _write(self.d / DECISION, dec)
        ui.err(
            "repair",
            f"{self.rd.name}: {why}; attempt {attempts} of {cap}: the same "
            f"command tries again",
        )
        return 1

    def repair(
        self, rec: Dict[str, Any], now: Optional[str], st: Mapping[str, Any]
    ) -> Tuple[int, Dict[str, Any]]:
        p, rd, d = self.p, self.rd, self.d
        cg = _read(rd / "stage06" / "cell_gate.json")
        dec = self._decision(now, st)
        # the attempts that waited (`wait`) for this bundle and these settings
        before = _read(d / DECISION) or {}
        same = before.get("candidate_sha256") == now
        same = same and before.get("settings") == dict(st)
        waited = int(before.get("attempts") or 0) if same else 0
        _set_aside(d)
        ok, why, flagged = repairable(cg, st["max_cells"])
        dec["flagged"] = flagged
        dg = rec.get("deploy_gate")
        if ok and not isinstance(dg, dict):
            ok, why = (
                False,
                "its deploy record does not say what its deploy gate decided",
            )
        elif ok and dg.get("passed") is not True:
            ok, why = False, f"its deploy gate keeps it out too: {dg.get('reason')}"
        if not ok:
            self._stop(dec, "not repairable", why)
            return 0, rec
        for line in flagged_lines(flagged):
            ui.info("repair", f"{rd.name} {line}")
        base = (cg or {}).get("baseline") or {}
        inc = find_incumbent(
            p.run_dir,
            staged_sha256=base.get("sha256"),
            hint=Path(str(base.get("dir"))) if base.get("dir") else None,
        )
        if inc is None:
            self._stop(
                dec,
                "not repaired",
                f"no bundle of the run is the one the per-cell gate compared with "
                f"(staged model sha256 {str(base.get('sha256'))[:12]})",
            )
            return 0, rec
        dec["incumbent"] = inc.to_json(p.run_dir)
        d.mkdir(parents=True, exist_ok=True)
        _write(d / "cell_gate.json", cg or {})
        record, code = self.build(inc, st)
        dec["repair"] = record
        dec["cells"] = record.get("cells") or cell_actions(
            flagged, record, now, inc.models_sha256
        )
        if not (d / "models.pt").is_file():
            self._stop(
                dec,
                "not repaired" if code == NOT_REPAIRED_EXIT else "repair failed",
                f"no repair: {record.get('outcome')}",
            )
            return 0, rec
        if p.get("audit.v2") == "rounds":
            p.audit_v2(
                d,
                [(p.v2_name(rd), d / "models.pt")],
                self.index,
                **_round_scoring(p),
            )
        recs = self.evaluate(inc)
        if recs is None:
            why = (
                f"the audit set did not measure the repaired bundle and the "
                f"incumbent (see {d / 'audit.json'})"
            )
            return self.wait(dec, waited + 1, why), rec
        other = scored_elsewhere(recs[0], inc)
        if other:
            self._stop(dec, "not repaired", other)
            return 0, rec
        v = judge(
            recs[0],
            recs[1],
            cg or {},
            st["cell_gate"],
            st["bootstrap"],
            st["official"],
            mean_rule=False,
        )
        dec["judged"] = v.record()
        record["judged"] = dec["judged"]
        _write(d / REPAIR_RECORD, record)
        if v.status == "wait":
            return self.wait(dec, waited + 1, v.reason), rec
        if v.status != "pass":
            self._stop(
                dec,
                "refused",
                f"its repair ({record.get('outcome')}) does not pass: {v.reason}",
            )
            return 0, rec
        dec.update(outcome="installed", decided=_now(), deployed=None)
        dec["reason"] = f"repaired ({record.get('outcome')}): {v.reason}"
        _write(d / DECISION, dec)
        install(rd, d, record, inc.dir, copy=self.copy, keep_stage06=True)
        _write(d / DECISION, dec)
        ui.ok(
            "repair",
            f"{rd.name} repaired ({record.get('outcome')}); stage06 measures it "
            f"on the round's gate GEMMs and deploys it when it passes",
        )
        rc = self.measure()
        if rc:
            return rc, rec
        return self.decide(_read(rd / "stage06" / rb.DEPLOY_RECORD) or {})

    def build(
        self, inc: Incumbent, st: Mapping[str, Any]
    ) -> Tuple[Dict[str, Any], int]:
        """scripts/repair_bundle.py of the round's bundle against the
        incumbent into `<round>/repair/`: (its record, its exit status)."""
        p, rd, d = self.p, self.rd, self.d
        cmd = [sys.executable, "-u", str(self.repair_script)]
        cmd += ["--train-dir", str(rd / "stage05"), "--baseline-dir", str(inc.dir)]
        cmd += ["--baseline-sha256", inc.models_sha256]
        cmd += ["--cell-gate", str(d / "cell_gate.json"), "--out-dir", str(d)]
        cmd += ["--audit-dir", str(auditlib.config_dir(p.cfg))]
        cmd += ["--config", str(p.snapshot)]
        cmd += ["--library-dir", str(p.paths.library_dir)]
        cmd += ["--hardware-device", str(p.devices[0])]
        cmd += ["--weight-dtype", p.weight_dtype, "--official", st["official"]]
        cmd += ["--max-cells", str(st["max_cells"])]
        cmd += ["--jobs", str(max(1, int(p.get("train.workers") or 1)))]
        ui.banner(f"{rd.name} -- repairing the base cells its gate flagged", ui.C.BLUE)
        rc = p.run_tool(rd, "repair", cmd, self.index)
        record = _read(d / REPAIR_RECORD)
        record = record if isinstance(record, dict) else {}
        if rc or not (d / "models.pt").is_file():
            (d / "models.pt").unlink(missing_ok=True)
            if rc != NOT_REPAIRED_EXIT:
                record["outcome"] = (
                    f"the repair failed (rc={rc}; see {rd / 'repair.log'})"
                )
        record["incumbent"] = inc.to_json(p.run_dir)
        return record, rc

    def evaluate(
        self, inc: Incumbent
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """(the incumbent's audit record, the repaired bundle's), scored by
        scripts/audit_eval.py as the round's stage06 scores them (without the
        alignment veto: `round_scoring_args`): the incumbent as its bundle
        directory, the repaired bundle leaving out the shapes of the rounds
        up to this one."""
        p, d = self.p, self.d
        out = d / "audit.json"
        cmd = [sys.executable, "-u", str(self.audit_eval_script)]
        cmd += ["--round-dir", str(inc.dir), "--model-file", str(d / "models.pt")]
        for r in auditlib.rounds_until(self.rd, True):
            cmd += ["--trained-round-dir", str(r)]
        cmd += ["--out", str(out), *p.audit_eval_common()]
        cmd += round_scoring_args(p)
        rc = p.run_tool(d, "audit_eval", cmd, self.index)
        doc = _read(out)
        recs = doc.get("models") if isinstance(doc, dict) else None
        if rc or not isinstance(recs, list) or len(recs) != 2:
            return None
        if not all(auditlib.is_measured(r) for r in recs):
            return None
        return recs[0], recs[1]


def round_scoring_args(pipeline: Any) -> List[str]:
    """The option of scripts/audit_eval.py that scores bundles as the run
    scores its rounds' (`Pipeline.round_scoring_args`: without the alignment
    veto); none from a pipeline without it."""
    fn = getattr(pipeline, "round_scoring_args", None)
    return list(fn()) if callable(fn) else []


def _round_scoring(pipeline: Any) -> Dict[str, Any]:
    """The keyword of `Pipeline.audit_v2` that registers bundles as the run
    scores its rounds' (without the alignment veto), when it sets the veto."""
    return {"without_veto": True} if round_scoring_args(pipeline) else {}


def scored_elsewhere(rec: Mapping[str, Any], inc: Incumbent) -> Optional[str]:
    """Why the audit record `rec`, scored as the incumbent's bundle
    directory, is not of the incumbent `inc`: the model its gate scored
    (`inc.staged_sha256`) is not what the directory's deploy record now
    points at (a final retrain a later one replaced keeps a path into
    final_retrain/). None when it is, or when the gate's model is unknown."""
    got = rec.get("model_sha256")
    if not inc.staged_sha256 or got == inc.staged_sha256:
        return None
    return (
        f"the audit set scored {rec.get('model')} (sha256 {str(got)[:12]}) for "
        f"the incumbent in {inc.dir}, not the model its gate compared with "
        f"(sha256 {inc.staged_sha256[:12]})"
    )


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _set_aside(d: Path) -> None:
    """Move an earlier repair directory into `<its parent>/stale/<stamp>/`."""
    if not d.exists():
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dest = d.parent / "stale" / stamp
    dest.mkdir(parents=True, exist_ok=True)
    shutil.move(str(d), str(dest / d.name))
