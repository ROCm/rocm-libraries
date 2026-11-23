#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Repair the base cells the per-cell gate flags in a trained bundle it
keeps out of a run (lib/collapse_repair.py, lib/cell_merge.py), without
training.

    repair_bundle.py --train-dir <stage05 dir> --baseline-dir <bundle dir>
        --cell-gate <cell gate json> --out-dir <dir> --audit-dir <audit>
        --config <run config> [--baseline-sha256 SHA] [--max-cells N]
        [--library-dir DIR] [--hardware-device N] [--weight-dtype D]
        [--official P] [--jobs N] [--quiet]

The bundle of <train-dir>/models.pt (a round's or a final retrain's
stage05) is kept out by the per-cell gate result --cell-gate (a
lib/audit.py `cell_gate` result, or a decision.json holding one as
`cell_gate`) against the bundle the run ships, whose directory (a round,
or a final retrain) is --baseline-dir and whose models.pt the carried cells
come from; --baseline-sha256 names that models.pt by its sha256, and a file
of other content is refused. The flagged base cells
(`cell_merge.flagged_cells`), at most --max-cells of them, are repaired:
each collapse by a tier key of its leaf's own network when the tail guard
keeps it, every finding still open, and every cell the cell rules flagged,
by the base cell's whole subtree from the baseline. The leaves' own GEMMs
are the enriched folders, split tree, validation split and repeat
aggregation stage05 recorded in <train-dir>/metrics.json, read over the
pool of the config's library and bank without its blocklist, as stage05
read them; the audit GEMMs and picks are lib/audit.py's (the config's
library stem, pick measurement and contexts, --official).

<out-dir> receives `repair.json` (lib/collapse_repair.py `RECORD`: every
collapse, what was found and done, the picks after, every flagged cell with
its findings, action and the sha256 of the bundle that now serves it, and
the splits the carries revoke) and, when the repair changed the bundle,
`models.pt`, `cells.json` (`collapse_repair.cells_doc`: the labels the
repaired bundle holds, no entry for a carried cell, whose origin names the
bundle that trained it, and each entry's tier keys counted as the bundle
serves them), `metrics.json` (`metrics_doc`: stage05's, with the bundle's
tier record), for a bundle with a whitelist tier `whitelist_tier.json`,
and, when a carried subtree lacks splits of the run's tree (or the bundle
revoked some before), `revoked_splits.json` (lib/subcells.py); the
per-cell gate then judges that bundle (run_orchestrator.py `FinalRetrain`,
lib/cell_merge.py `RoundRepair`). Exit 0 when it wrote a repaired bundle;
3 when no repair was possible (repair.json says why); 1 on errors; 2 when
an input is part of an audit set it must not train from.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from lib import audit as auditlib  # noqa: E402
from lib import cell_merge as cm  # noqa: E402
from lib import collapse_repair as cr  # noqa: E402
from lib import correctness as cx  # noqa: E402
from lib import evaluate as ev  # noqa: E402
from lib import features as fs  # noqa: E402
from lib import hardware as hwlib  # noqa: E402
from lib import mlrec  # noqa: E402
from lib import repeats  # noqa: E402
from lib import run_analysis as ra  # noqa: E402
from lib import subcells as sc  # noqa: E402
from lib import ui  # noqa: E402
from lib import whitelist_tier as wt  # noqa: E402
from lib.bundle import bundle_file  # noqa: E402
from lib.fs import write_json  # noqa: E402
from lib.grid import cell_key  # noqa: E402
from lib.stage05_split import stage05_module  # noqa: E402

EXIT_NOT_REPAIRED = 3


class Refused(Exception):
    """What keeps the repair from running; `code` is the exit status."""

    def __init__(self, message: str, code: int = 1) -> None:
        super().__init__(message)
        self.code = code


def _json(path: Path) -> Dict[str, Any]:
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise Refused(f"cannot read {path}: {e}") from None
    if not isinstance(doc, dict):
        raise Refused(f"{path} holds no JSON object")
    return doc


def _bundle(path: Path) -> Dict[str, Any]:
    import torch

    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001
        raise Refused(f"cannot read {path}: {e}") from None


def leaf_data_reader(
    bundle: Dict[str, Any],
    args: Dict[str, Any],
    cfg: Dict[str, Any],
    *,
    library_dir: Path,
    stem: str,
    device: int,
    hardware: Any,
    arch: str,
    dtype: str,
    jobs: int,
    say: Any,
) -> Any:
    """The `collapse_repair.repair` reader of leaves' GEMMs: the GEMMs of the
    leaves' base cells in the enriched folders stage05 recorded in `args`,
    read and split as it read and split them, scored by `bundle` without
    its tier (`whitelist_tier.scoring_bundle`), each kept for the leaf the
    engine serves it from."""
    s5 = stage05_module()
    dirs = [Path(d) for d in args.get("enriched_csv_dir") or []]
    if not dirs:
        raise Refused("the bundle's metrics.json records no enriched folders")
    leaks = auditlib.audit_inputs(auditlib.existing_paths(dirs))
    if leaks:
        raise Refused("; ".join(leaks), code=2)
    try:
        frac, seed = float(args["validation_frac"]), int(args["seed"])
    except (KeyError, TypeError, ValueError):
        raise Refused("the bundle's metrics.json records no validation_frac / seed")
    tree = sc.load_split_tree(
        args.get("prior_round_dir") or [], args.get("current_round_dir")
    )
    spec = fs.parse_attribute_spec(bundle.get("attribute_spec") or [])

    def read(leaves: Set[str]) -> cr.LeafData:
        t0 = time.time()
        u = wt.universe(cfg, library_dir, stem, device, spec)
        bases = {sc.base_cell(x) for x in leaves}

        def keep(p: Dict[str, Any]) -> bool:
            shape = (int(p["m"]), int(p["n"]), int(p["k"]), int(p["batch_count"]))
            return cell_key(*shape) in bases

        bench = ra.load_bench(
            dirs,
            jobs=jobs,
            keep=keep,
            identity=ev.library_identity(library_dir, stem, cfg, device),
            aggregation=str(
                args.get("repeat_aggregation") or repeats.PREDATING_AGGREGATION
            ),
            blocklist=cx.config_blocklist(cfg),
            pool_indices=u.sids.tolist(),
        )
        mine = bench.gemms
        train, held = wt.split_gemms(mine, tree, frac, seed, s5=s5)
        data = mlrec.write_model(
            wt.scoring_bundle(bundle), None, arch, hwlib.arch_constants(arch), dtype
        )
        tw = ev.tilewright_module()
        model = ev.load_model_bytes(data, tw)
        labels = [
            str(tw.cell_label(model, i)) for i in range(int(tw.describe(model).n_cells))
        ]
        scored = wt.engine_scores(data, u, hardware, train + held, jobs)
        out: Dict[str, Any] = {leaf: ([], []) for leaf in leaves}
        for i, (g, r) in enumerate(zip(train + held, scored)):
            if r[0] < 0 or not g.winner_us > 0 or not r[1].size:
                continue
            leaf = labels[int(r[0])]
            if leaf in out:
                out[leaf][0 if i < len(train) else 1].append((float(g.winner_us), r))
        say(
            f"{len(mine):,} GEMMs of {len(bases)} base cell(s) in {len(dirs)} "
            f"enriched folders scored for {len(leaves)} leaf(s) "
            f"({time.time() - t0:.0f}s)"
        )
        return out, {int(h): i for i, h in enumerate(u.hashes.tolist())}

    return read


def revocations(
    train_dir: Path,
    args: Dict[str, Any],
    shipped: Dict[str, Any],
    rec: Dict[str, Any],
) -> Optional[Tuple[List[Any], List[str]]]:
    """(the split rules the repaired bundle revokes, the base cells barred
    from splitting again) for the record `rec` of a repair: the rules of
    the split tree stage05 trained the bundle with (`args`, its
    metrics.json arguments) inside the carried base cells that the
    shipped bundle `shipped` lacks (`cell_merge.revoked_rules`), with the
    revocations the bundle held before; None when there are none. The
    record names them (`revoked_splits`, `barred_cells`)."""
    carried = sorted(rec.get("carried_cells") or {})
    before = sc.revocations(Path(train_dir).parent)
    rules = {
        str(r["cell"]): sc.SplitRule.from_json(r)
        for r in before.get("revoked") or []
        if isinstance(r, dict) and r.get("cell")
    }
    barred = set(before.get("barred_cells") or [])
    if carried:
        tree = sc.load_split_tree(
            args.get("prior_round_dir") or [], args.get("current_round_dir")
        )
        new = cm.revoked_rules(tree, shipped.get("models") or {}, carried)
        rules.update({r.cell: r for r in new})
        barred |= {sc.base_cell(r.cell) for r in new}
    if not rules and not barred:
        return None
    rec["revoked_splits"] = [
        dict(r.to_json(), base_cell=sc.base_cell(r.cell))
        for r in sorted(rules.values(), key=lambda r: r.cell)
    ]
    rec["barred_cells"] = sorted(barred)
    return list(rules.values()), sorted(barred)


def run(a: argparse.Namespace) -> int:
    import run_orchestrator as ro

    say = (lambda m: None) if a.quiet else (lambda m: ui.info("repair", m))
    try:
        cfg, _ = ro.load_config(a.config)
    except ro.ConfigError as e:
        raise Refused("; ".join(e.errors)) from None
    train_dir, out_dir = a.train_dir.resolve(), a.out_dir.resolve()
    if out_dir == train_dir:
        raise Refused("--out-dir must not be --train-dir: the bundle there stays")
    gate = _json(a.cell_gate)
    if isinstance(gate.get("cell_gate"), dict):
        gate = gate["cell_gate"]
    flagged = cm.flagged_cells(gate)
    if a.max_cells is not None and len(flagged) > a.max_cells:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in ("models.pt", "cells.json", "metrics.json", wt.TIER_FILE):
            (out_dir / name).unlink(missing_ok=True)
        (out_dir / sc.REVOKED_SPLITS).unlink(missing_ok=True)
        rec = {
            "schema": cr.SCHEMA,
            "flagged": flagged,
            "outcome": f"not repaired: {len(flagged)} base cells flagged, more "
            f"than --max-cells {a.max_cells}",
        }
        write_json(out_dir / cr.RECORD, rec)
        ui.warn("repair", rec["outcome"])
        return EXIT_NOT_REPAIRED
    collapses = cr.collapses_of(gate)
    cand_path = bundle_file(train_dir, "models.pt")
    base_path = Path(a.baseline_dir) / "stage05" / "models.pt"
    have = cm.sha256_file(base_path)
    if a.baseline_sha256 and have != a.baseline_sha256:
        raise Refused(
            f"{base_path} is not the bundle to carry cells from: its sha256 is "
            f"{have}, not {a.baseline_sha256}"
        )
    candidate = _bundle(cand_path)
    shipped = _bundle(base_path)
    met = _json(bundle_file(train_dir, "metrics.json"))
    args = met.get("args") or {}
    library_dir = a.library_dir or ro.derive_paths(cfg).library_dir
    device = (
        a.hardware_device
        if a.hardware_device is not None
        else int(cfg["runtime"]["devices"][0])
    )
    arch = str(cfg["arch"])
    dtype = a.weight_dtype or str(ro.cfg_get(cfg, "deploy.weight_dtype"))
    stem = str(cfg["hipblaslt"]["library_stem"])
    try:
        hardware = hwlib.device_hardware(cfg, device)
        cfg, _ = ro.resolve_contexts(cfg, device)
    except ro.ConfigError as e:
        raise Refused("; ".join(e.errors)) from None
    except (ValueError, RuntimeError, OSError) as e:
        raise Refused(str(e)) from None
    st = auditlib.settings(cfg)
    gate_st = auditlib.cell_gate_settings(cfg) or auditlib.CELL_GATE_DEFAULTS
    floor = float(gate_st["tail_floor"]) or wt.GUARD_FLOOR
    spec = auditlib.Spec(
        audit_dir=a.audit_dir,
        library_dir=library_dir,
        library_stem=stem,
        hardware_device=device,
        pick_measurement=str(ro.cfg_get(cfg, "validate.pick_measurement")),
        target=float(st["target"]),
        bootstrap=int(st["bootstrap"]),
        official=a.official or auditlib.official_protocol(cfg),
    )
    t0 = time.time()
    audit = cr.AuditView(spec, cfg, hardware=hardware)
    spec_attrs = fs.parse_attribute_spec(candidate.get("attribute_spec") or [])

    def pack(b: Dict[str, Any]) -> bytes:
        return mlrec.write_model(b, None, arch, hwlib.arch_constants(arch), dtype)

    def view_key_hash(kernel: Dict[str, Any]) -> int:
        return int(wt.view_key_hashes([kernel], spec_attrs)[0]) & 0xFFFFFFFFFFFFFFFF

    reader = leaf_data_reader(
        candidate,
        args,
        cfg,
        library_dir=Path(library_dir),
        stem=stem,
        device=device,
        hardware=hardware,
        arch=arch,
        dtype=dtype,
        jobs=max(1, int(a.jobs)),
        say=say,
    )
    rules = cm.cell_rule_cells(flagged)
    say(
        f"{len(collapses)} collapse(s) and {len(rules)} cell(s) flagged by the cell "
        f"rules to repair against {a.baseline_dir}"
    )
    out = cr.repair(
        candidate,
        shipped,
        collapses,
        audit,
        pack=pack,
        leaf_data=reader,
        view_key_hash=view_key_hash,
        floor=floor,
        max_keys=int(a.max_keys),
        carry_also=rules,
    )
    origins = {"candidate": cm.sha256_file(cand_path), "incumbent": have}
    rec = dict(
        out.record,
        candidate=str(cand_path),
        baseline=str(a.baseline_dir),
        flagged=flagged,
        origins=origins,
        elapsed_s=round(time.time() - t0, 1),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("models.pt", "cells.json", "metrics.json", wt.TIER_FILE):
        (out_dir / name).unlink(missing_ok=True)
    (out_dir / sc.REVOKED_SPLITS).unlink(missing_ok=True)
    revoked = None
    if out.bundle is not None:
        try:
            revoked = revocations(train_dir, args, shipped, rec)
        except ValueError as e:
            rec["outcome"] = f"cannot repair: {e}"
            out.bundle = None
    rec["cells"] = cm.cell_actions(
        flagged, rec, origins["candidate"], origins["incumbent"]
    )
    if out.bundle is None:
        write_json(out_dir / cr.RECORD, rec)
        ui.warn("repair", f"no repaired bundle: {rec.get('outcome')}")
        return EXIT_NOT_REPAIRED
    if revoked is not None:
        sc.write_revocations(
            out_dir / sc.REVOKED_SPLITS,
            revoked[0],
            revoked[1],
            candidate_sha256=origins["candidate"],
            incumbent_sha256=origins["incumbent"],
        )
    import torch

    torch.save(out.bundle, str(out_dir / "models.pt"))
    cells_path = bundle_file(train_dir, "cells.json")
    if cells_path.is_file():
        write_json(
            out_dir / "cells.json", cr.cells_doc(_json(cells_path), out.bundle, rec)
        )
    write_json(out_dir / "metrics.json", cr.metrics_doc(met, out.bundle, rec))
    cand_doc = bundle_file(train_dir, wt.TIER_FILE)
    ship_doc = Path(a.baseline_dir) / "stage05" / wt.TIER_FILE
    doc = cr.tier_doc(
        _json(cand_doc) if cand_doc.is_file() else None,
        _json(ship_doc) if ship_doc.is_file() else None,
        rec,
        out.bundle,
    )
    if doc is not None:
        write_json(out_dir / wt.TIER_FILE, doc)
    write_json(out_dir / cr.RECORD, rec)
    ui.ok("repair", f"{rec['outcome']}: {out_dir / 'models.pt'}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="repair the collapses that keep a trained bundle out "
        "(lib/collapse_repair.py)"
    )
    ap.add_argument("--train-dir", type=Path, required=True, help="stage05 output")
    ap.add_argument(
        "--baseline-dir",
        type=Path,
        required=True,
        help="the bundle the run ships: a round or a final retrain directory",
    )
    ap.add_argument(
        "--cell-gate",
        type=Path,
        required=True,
        help="the per-cell gate result that names the collapses (or a decision "
        "holding it)",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--audit-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True, help="the run's config")
    ap.add_argument("--library-dir", type=Path, default=None)
    ap.add_argument("--hardware-device", type=int, default=None)
    ap.add_argument("--weight-dtype", default=None)
    ap.add_argument(
        "--official",
        choices=auditlib.OFFICIAL_CHOICES,
        default=None,
        help="the protocol of the official exclusive number (default: the "
        "config's audit.official)",
    )
    ap.add_argument(
        "--baseline-sha256",
        default=None,
        help="the sha256 of the --baseline-dir bundle's models.pt the cells are "
        "carried from; a models.pt of other content is refused",
    )
    ap.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="repair nothing when the gate flags more base cells than this",
    )
    ap.add_argument(
        "--max-keys",
        type=int,
        default=wt.GUARD_KEYS,
        help="tier keys the repair may add to a leaf",
    )
    ap.add_argument("--jobs", type=int, default=1, help="reading and scoring")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.max_keys < 0:
        ap.error("--max-keys must be >= 0")
    if a.max_cells is not None and a.max_cells < 1:
        ap.error("--max-cells must be >= 1")
    try:
        return run(a)
    except Refused as e:
        ui.err("repair", str(e))
        return e.code


if __name__ == "__main__":
    sys.exit(main())
