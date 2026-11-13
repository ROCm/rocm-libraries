#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Build the whitelist tier of a trained bundle again, without training.

    build_whitelist_tier.py --train-dir <stage05 dir> --out-dir <dir>
        [--k K] [--contexts pooled|balanced|reserve] [--guard-keys G]
        [[--keep-tier] --standins]
        [--config-yaml CFG] [--library-dir DIR --library-stem STEM]
        [--hardware-device N] [--weight-dtype D] [--jobs N]

The tier (lib/whitelist_tier.py) of the bundle in <train-dir>/models.pt is
chosen again on the training GEMMs its stage05 recorded (metrics.json: the
enriched folders, the split tree, the validation split and the repeat
aggregation), over the pool of the library the options name, else of the one
stage05 ranked; the config (`--config-yaml`, else stage05's) gives the bank
and the blocklist. After a change of the kernel pool (another hipBLASLt
build, bank or blocklist) the rows of kernels the pool lacks are left out as
stage05 leaves them out, so they no longer count, not even as a GEMM's
winner; kernels it adds have no rows, and every tier key is a key of the
pool: the bench folders are read with the pool's kernel identity
(lib/kernel_identity.py), so a folder benched with another build counts its
kernels under their indices in this one. `--k 0` removes the tier; without
`--k` the bundle keeps the size its tier has. `--contexts` counts the
execution contexts of a cell's GEMMs as stage05's --whitelist-tier-contexts
does; without it the tier keeps the way its record names (pooled when it
names none). `--guard-keys` (default lib/whitelist_tier.py GUARD_KEYS)
bounds the keys the tail guard adds per cell, checked on the GEMMs stage05
trained each cell on and on those it held out (`whitelist_tier.guard`); 0
builds the tier without the guard. With `--standins` the tier built also
gets the stand-ins of its keys (`whitelist_tier.choose_standins`), chosen
on the training GEMMs alone; without it, none.

`--keep-tier --standins` keeps the bundle's tier and adds the stand-ins of
its keys, for a bundle whose tier holds none: the tier is chosen again as
its record says (its size, contexts and guard) and the bundle is refused
unless every cell's keys equal the ones chosen again, in order
(`whitelist_tier.kept_tier`). A cell whose tier record lists `repair_keys`
is accepted when its keys are the ones chosen again followed by exactly
those, which the bundle's repair record (`<train-dir>/../repair/repair.json`
of lib/cell_merge.py, its installed models.pt this bundle's) names for it;
it keeps no stand-ins and the record names it (`standins_repaired`, with the
repair record's sha256). The stand-ins are chosen against the bundle's own
keys; models.pt, whitelist_tier.json and metrics.json record them, the other
files are copied as they are.

Otherwise <out-dir> receives the stage05 files of <train-dir> with the new tier in
models.pt, `whitelist_tier.json`, and the size, the way it counts the
contexts and the guard's keys in model_options.json (`whitelist_tier_k`,
`whitelist_tier_contexts`, `whitelist_tier_guard_keys`), metrics.json
(`whitelist_tier`, with
`rebuilt_from` and how the training split compares with the counts
cells.json records) and cells.json; stage06 deploys it (`--train-dir
<out-dir>`). <train-dir> is left as it is. Exit 0; 1 on errors; 2 when an
input is part of an audit set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import yaml  # noqa: E402

from lib import audit as auditlib  # noqa: E402
from lib import cell_merge as cm  # noqa: E402
from lib import correctness as cx  # noqa: E402
from lib import evaluate as ev  # noqa: E402
from lib import features as fs  # noqa: E402
from lib import hardware as hwlib  # noqa: E402
from lib import kernel_identity as ki  # noqa: E402
from lib import mlrec  # noqa: E402
from lib import repeats  # noqa: E402
from lib import run_analysis as ra  # noqa: E402
from lib import run_bundles as rb  # noqa: E402
from lib import subcells as sc  # noqa: E402
from lib import ui  # noqa: E402
from lib import whitelist_tier as wt  # noqa: E402
from lib.bundle import bundle_file  # noqa: E402
from lib.fs import write_json  # noqa: E402
from lib.stage05_split import stage05_module  # noqa: E402

COPIED = ("training_log.csv", ki.MEMBER_ORDER_FILE, rb.CELL_ORIGINS)


class Refused(Exception):
    """What keeps the tier from being built; `code` is the exit status."""

    def __init__(self, message: str, code: int = 1) -> None:
        super().__init__(message)
        self.code = code


def _json(path: Path) -> Dict[str, Any]:
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise Refused(f"cannot read {path}: {e}") from None
    if not isinstance(doc, dict):
        raise Refused(f"{path} holds no JSON object")
    return doc


def split_check(
    gemms: List[Any], tree: Any, frac: float, seed: int, cells: Dict[str, Any]
) -> Dict[str, Any]:
    """How the training / validation counts of the cells stage05 trained,
    split again from the GEMMs read now, compare with its cells.json."""
    counts = wt.leaf_counts(gemms, tree, frac, seed)
    trained = {
        str(c["label"]): (
            int(c.get("n_train_gemms", -1)),
            int(c.get("n_val_gemms", -1)),
        )
        for c in cells.get("cells") or []
        if isinstance(c, dict) and "label" in c
    }
    differ = sorted(lab for lab, n in trained.items() if counts.get(lab) != n)
    return {
        "cells": len(trained),
        "cells_with_other_counts": len(differ),
        "examples": differ[:10],
        "exact": not differ,
    }


def repair_record(
    train_dir: Path, doc: Optional[Dict[str, Any]]
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """(the repair record of the bundle in `train_dir`, its sha256) when its
    tier record `doc` names repair keys, else (None, None): the record of the
    repair installed into it (lib/cell_merge.py, `<owner>/repair/repair.json`
    beside `<owner>/stage05`). Refused without one, for a record that names
    no models.pt it was installed into, and for the record of a repair
    installed into another models.pt."""
    named = [
        label
        for label, c in ((doc or {}).get("cells") or {}).items()
        if isinstance(c, dict) and c.get("repair_keys")
    ]
    if not named:
        return None, None
    path = train_dir.parent / cm.REPAIR_DIR / cm.REPAIR_RECORD
    try:
        raw = path.read_bytes()
        rec = json.loads(raw)
    except (OSError, ValueError) as e:
        raise Refused(
            f"cell {named[0]}'s tier record names repair keys, and {path} cannot be "
            f"read: {e}"
        ) from None
    if not isinstance(rec, dict):
        raise Refused(f"{path} holds no JSON object")
    installed = (rec.get("installed") or {}).get("models_sha256")
    if not installed:
        raise Refused(f"{path} names no models.pt its repair was installed into")
    if installed != cm.sha256_file(bundle_file(train_dir, "models.pt")):
        raise Refused(f"{path} records a repair installed into another models.pt")
    return rec, hashlib.sha256(raw).hexdigest()


def keep(
    a: argparse.Namespace,
    bundle: Dict[str, Any],
    met: Dict[str, Any],
    cells: Dict[str, Any],
    train_dir: Path,
    out_dir: Path,
    s5: Any,
) -> int:
    """--keep-tier: the bundle with its own tier and the stand-ins of its
    keys, into `out_dir`."""
    if not wt.has_tier(bundle):
        raise Refused(f"{train_dir}: its models.pt has no whitelist tier to keep")
    record = bundle.get(wt.RECORD) or {}
    k = int(record.get("k") or 0)
    if k < 1:
        raise Refused(f"{train_dir}: its tier record names no size")
    mode = str(record.get("contexts") or wt.DEFAULT_CONTEXTS)
    if mode not in wt.CONTEXT_MODES:
        raise Refused(f"{train_dir}: its tier counts contexts as {mode!r}")
    guard_keys = int((record.get("guard") or {}).get("keys") or 0)
    doc_path = bundle_file(train_dir, wt.TIER_FILE)
    doc = _json(doc_path) if doc_path.is_file() else None
    repair, repair_sha256 = repair_record(train_dir, doc)
    tier, summary = rebuild(
        a,
        k,
        mode,
        bundle,
        met.get("args") or {},
        cells,
        train_dir,
        s5,
        guard_keys,
        standins=True,
    )
    try:
        new, checked = wt.kept_tier(bundle, tier, doc, repair, repair_sha256)
    except ValueError as e:
        raise Refused(str(e)) from None
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in (*COPIED, "cells.json", s5.MODEL_OPTIONS_FILE):
        src = bundle_file(train_dir, name)
        if src.is_file():
            shutil.copy2(src, out_dir / name)
    import torch

    torch.save(new, str(out_dir / "models.pt"))
    fields = {key: v for key, v in new[wt.RECORD].items() if key != "standins"}
    kept = {
        **checked,
        "elapsed_s": summary["elapsed_s"],
        "rebuilt_from": summary["rebuilt_from"],
        "library": summary["library"],
        "split": summary["split"],
        "n_gemms": summary["n_gemms"],
    }
    write_json(
        out_dir / "metrics.json",
        dict(
            met,
            whitelist_tier={**(met.get("whitelist_tier") or {}), **fields},
            standins_kept_tier=kept,
        ),
    )
    write_json(out_dir / wt.TIER_FILE, {**(doc or {}), **new[wt.RECORD]})
    ui.ok(
        "tier",
        f"the bundle's tier kept ({checked['cells_equal']} of {checked['cells']} cells "
        f"chosen again alike, {len(checked['cells_repaired'])} repaired) with "
        f"stand-ins for {fields['n_standin_lists']} keys: {out_dir}",
    )
    return 0


def run(a: argparse.Namespace) -> int:
    s5 = stage05_module()
    train_dir, out_dir = a.train_dir.resolve(), a.out_dir.resolve()
    if out_dir == train_dir:
        raise Refused("--out-dir must not be --train-dir: the bundle there stays")
    models_pt = bundle_file(train_dir, "models.pt")
    met = _json(bundle_file(train_dir, "metrics.json"))
    cells = _json(bundle_file(train_dir, "cells.json"))
    args = met.get("args") or {}

    import torch

    try:
        bundle = torch.load(str(models_pt), map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001
        raise Refused(f"cannot read {models_pt}: {e}") from None
    if a.keep_tier:
        return keep(a, bundle, met, cells, train_dir, out_dir, s5)
    k = a.k if a.k is not None else (bundle.get(wt.RECORD) or {}).get("k")
    if k is None:
        raise Refused(f"{models_pt} has no whitelist tier: name its size with --k")
    k = int(k)
    mode = str(
        a.contexts
        or (bundle.get(wt.RECORD) or {}).get("contexts")
        or wt.DEFAULT_CONTEXTS
    )
    if mode not in wt.CONTEXT_MODES:
        raise Refused(f"{models_pt}: its tier counts contexts as {mode!r}")
    guard_keys = int(a.guard_keys if a.guard_keys is not None else wt.GUARD_KEYS)
    tier: Optional[wt.Tier] = None
    summary: Optional[Dict[str, Any]] = None
    if k == 0:
        new = wt.strip(bundle)
        mode = wt.DEFAULT_CONTEXTS
    else:
        tier, summary = rebuild(
            a, k, mode, bundle, args, cells, train_dir, s5, guard_keys, a.standins
        )
        new = wt.apply(bundle, tier)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in COPIED:
        src = bundle_file(train_dir, name)
        if src.is_file():
            shutil.copy2(src, out_dir / name)
    for entry in cells.get("cells") or []:
        if isinstance(entry, dict):
            entry.pop("n_whitelist_tier_keys", None)
            if tier is not None:
                n = len(tier.keys.get(str(entry.get("label")), []))
                entry["n_whitelist_tier_keys"] = n
    write_json(out_dir / "cells.json", cells)
    torch.save(new, str(out_dir / "models.pt"))
    options_path = bundle_file(train_dir, s5.MODEL_OPTIONS_FILE)
    options = _json(options_path) if options_path.is_file() else {}
    options = {
        **options,
        "whitelist_tier_k": k,
        "whitelist_tier_contexts": mode,
        "whitelist_tier_guard_keys": guard_keys if k else None,
    }
    write_json(out_dir / s5.MODEL_OPTIONS_FILE, options)
    write_json(
        out_dir / "metrics.json",
        dict(met, serving=new.get("serving"), whitelist_tier=summary),
    )
    tier_path = out_dir / wt.TIER_FILE
    if tier is None:
        tier_path.unlink(missing_ok=True)
        ui.ok("tier", f"the bundle without a whitelist tier: {out_dir}")
        return 0
    wt.write_doc(tier_path, tier)
    split = summary["split"] if summary else {}
    if split and not split["exact"]:
        ui.warn(
            "split",
            f"{split['cells_with_other_counts']} of {split['cells']} trained "
            f"cells split into other counts than cells.json records (GEMMs "
            f"whose every row belongs to a kernel this pool lacks are gone)",
        )
    ui.ok(
        "tier",
        f"{wt.record_text(tier.record)} over {tier.record['pool']['n_kernels']:,} "
        f"kernels, chosen on {tier.record['n_gemms']:,} training GEMMs: {out_dir}",
    )
    return 0


def rebuild(
    a: argparse.Namespace,
    k: int,
    mode: str,
    bundle: Dict[str, Any],
    args: Dict[str, Any],
    cells: Dict[str, Any],
    train_dir: Path,
    s5: Any,
    guard_keys: int = wt.GUARD_KEYS,
    standins: bool = False,
) -> Tuple[wt.Tier, Dict[str, Any]]:
    """(the tier, its metrics.json summary) of `bundle` with `k` keys per
    cell, chosen as `mode` (`whitelist_tier.CONTEXT_MODES`) counts the
    contexts, on the GEMMs stage05 recorded in `args`, with up to
    `guard_keys` tail-guard keys per cell and, with `standins`, the
    stand-ins of its keys."""
    why = wt.refusal(bundle)
    if why:
        raise Refused(why)
    cfg_path = a.config_yaml or args.get("config_yaml")
    cfg: Dict[str, Any] = {}
    if cfg_path:
        try:
            cfg = yaml.safe_load(Path(cfg_path).read_text()) or {}
        except (OSError, yaml.YAMLError) as e:
            raise Refused(f"cannot read the config {cfg_path}: {e}") from None
    lib_dir = a.library_dir or args.get("library_dir")
    stem = a.library_stem or args.get("library_stem")
    if not lib_dir or not stem:
        raise Refused(
            "no library: stage05 ranked none, so name it with --library-dir "
            "and --library-stem"
        )
    device = (
        a.hardware_device
        if a.hardware_device is not None
        else int(args.get("hardware_device") or 0)
    )
    dirs = [Path(d) for d in args.get("enriched_csv_dir") or []]
    if not dirs:
        raise Refused(f"{train_dir}: metrics.json records no enriched folders")
    leaks = auditlib.audit_inputs(
        auditlib.existing_paths([*dirs, Path(lib_dir), cfg_path])
    )
    if leaks:
        raise Refused("; ".join(leaks), code=2)
    try:
        frac, seed = float(args["validation_frac"]), int(args["seed"])
    except (KeyError, TypeError, ValueError):
        raise Refused(
            f"{train_dir}: metrics.json records no validation_frac / seed"
        ) from None
    arch = str(args.get("arch") or cfg.get("arch") or "")
    dtype = str(
        a.weight_dtype
        or args.get("weight_dtype")
        or (cfg.get("deploy") or {}).get("weight_dtype")
        or ""
    )
    if not arch or dtype not in mlrec.WEIGHT_DTYPES:
        raise Refused("name the arch and --weight-dtype the bundle deploys with")
    tree = sc.load_split_tree(
        args.get("prior_round_dir") or [], args.get("current_round_dir")
    )
    spec = fs.parse_attribute_spec(bundle.get("attribute_spec") or [])
    say = (lambda m: None) if a.quiet else (lambda m: ui.info("tier", m))
    t0 = time.time()
    try:
        hardware = hwlib.device_hardware(cfg, device)
        identity = ev.library_identity(lib_dir, stem, cfg, device)
        u = wt.universe(cfg, lib_dir, stem, device, spec)
        bench = ra.load_bench(
            dirs,
            jobs=a.jobs,
            identity=identity,
            aggregation=str(
                args.get("repeat_aggregation") or repeats.PREDATING_AGGREGATION
            ),
            blocklist=cx.config_blocklist(cfg),
            pool_indices=u.sids.tolist(),
        )
    except (ValueError, RuntimeError, OSError) as e:
        raise Refused(str(e)) from None
    for note in bench.notes:
        ui.warn("identity", note)
    say(f"{len(bench.gemms):,} GEMMs from {len(dirs)} enriched folders")
    gemms, held_out = wt.split_gemms(bench.gemms, tree, frac, seed, s5=s5)
    try:
        tier = wt.build(
            bundle,
            gemms,
            k=k,
            u=u,
            hardware=hardware,
            arch=arch,
            weight_dtype=dtype,
            workers=a.jobs,
            s5=s5,
            say=say,
            mode=mode,
            held_out=held_out,
            guard_keys=guard_keys,
            standins=standins,
        )
    except (ValueError, RuntimeError) as e:
        raise Refused(str(e)) from None
    return tier, {
        **tier.record,
        "elapsed_s": time.time() - t0,
        "rebuilt_from": str(train_dir),
        "library": {"dir": str(lib_dir), "stem": str(stem)},
        "split": split_check(bench.gemms, tree, frac, seed, cells),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="build the whitelist tier of a trained bundle again "
        "(lib/whitelist_tier.py)"
    )
    ap.add_argument("--train-dir", type=Path, required=True, help="stage05 output")
    ap.add_argument(
        "--out-dir", type=Path, required=True, help="where the new bundle goes"
    )
    ap.add_argument(
        "--k",
        type=int,
        default=None,
        help="view keys per cell (0: no tier; default: the bundle's tier's)",
    )
    ap.add_argument(
        "--contexts",
        choices=wt.CONTEXT_MODES,
        default=None,
        help="how a cell's tier counts the execution contexts of its GEMMs "
        "(stage05 --whitelist-tier-contexts; default: as the bundle's tier "
        "does, else pooled)",
    )
    ap.add_argument(
        "--guard-keys",
        type=int,
        default=None,
        help="keys the tail guard may add to a cell's tier (stage05 "
        f"--whitelist-tier-guard-keys; default {wt.GUARD_KEYS}; 0: no guard)",
    )
    ap.add_argument(
        "--config-yaml",
        type=Path,
        default=None,
        help="run config: hardware, bank and blocklist (default: stage05's)",
    )
    ap.add_argument(
        "--library-dir",
        type=Path,
        default=None,
        help="Tensile library dir of the pool (default: stage05's)",
    )
    ap.add_argument("--library-stem", default=None, help="default: stage05's")
    ap.add_argument(
        "--hardware-device", type=int, default=None, help="default: stage05's"
    )
    ap.add_argument(
        "--weight-dtype",
        choices=sorted(mlrec.WEIGHT_DTYPES),
        default=None,
        help="dtype the bundle deploys with (default: stage05's)",
    )
    ap.add_argument(
        "--keep-tier",
        action="store_true",
        help="keep the bundle's tier, chosen again only to check it, and add the "
        "stand-ins of its keys (with --standins)",
    )
    ap.add_argument(
        "--standins",
        action="store_true",
        help="give the tier's keys their stand-ins (lib/standins.py): the tier "
        "built here, or with --keep-tier the bundle's own",
    )
    ap.add_argument("--jobs", type=int, default=1, help="reading and scoring")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.keep_tier and not a.standins:
        ap.error("--keep-tier adds the stand-ins of the bundle's tier: give --standins")
    if a.keep_tier and (
        a.k is not None or a.contexts is not None or a.guard_keys is not None
    ):
        ap.error(
            "--keep-tier keeps the tier its record describes: no --k, --contexts "
            "or --guard-keys"
        )
    if a.k is not None and a.k < 0:
        ap.error("--k must be >= 0")
    if a.k == 0 and a.contexts is not None:
        ap.error("--contexts chooses the keys of a tier, and --k 0 removes it")
    if a.k == 0 and a.standins:
        ap.error("--standins chooses the stand-ins of a tier, and --k 0 removes it")
    if a.guard_keys is not None and a.guard_keys < 0:
        ap.error("--guard-keys must be >= 0")
    if (a.library_dir is None) != (a.library_stem is None):
        ap.error("--library-dir and --library-stem go together")
    try:
        return run(a)
    except Refused as e:
        ui.err("tier", str(e))
        return e.code


if __name__ == "__main__":
    sys.exit(main())
