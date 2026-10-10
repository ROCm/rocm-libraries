#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Build the whitelist tier of a trained bundle again, without training.

    build_whitelist_tier.py --train-dir <stage05 dir> --out-dir <dir>
        [--k K] [--config-yaml CFG] [--library-dir DIR --library-stem STEM]
        [--hardware-device N] [--weight-dtype D] [--jobs N]

The tier (lib/whitelist_tier.py) of the bundle in <train-dir>/models.pt is
chosen again on the training GEMMs its stage05 recorded (metrics.json: the
enriched folders, the split tree, the validation split and the repeat
aggregation), over the pool of the library the options name, else of the one
stage05 ranked; the config (`--config-yaml`, else stage05's) gives the bank
and the blocklist. After a change of the kernel pool (another hipBLASLt
build, bank or blocklist) the rows of kernels the pool lacks no longer count,
kernels it adds have no rows, and every tier key is a key of the pool: the
bench folders are read with the pool's kernel identity
(lib/kernel_identity.py), so a folder benched with another build counts its
kernels under their indices in this one. `--k 0` removes the tier; without
`--k` the bundle keeps the size its tier has.

<out-dir> receives the stage05 files of <train-dir> with the new tier in
models.pt, `whitelist_tier.json`, and the size in model_options.json
(`whitelist_tier_k`), metrics.json (`whitelist_tier`, with `rebuilt_from`
and how the training split compares with the counts cells.json records) and
cells.json; stage06 deploys it (`--train-dir <out-dir>`). <train-dir> is
left as it is. Exit 0; 1 on errors; 2 when an input is part of an audit set.
"""
from __future__ import annotations

import argparse
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
    k = a.k if a.k is not None else (bundle.get(wt.RECORD) or {}).get("k")
    if k is None:
        raise Refused(f"{models_pt} has no whitelist tier: name its size with --k")
    k = int(k)
    tier: Optional[wt.Tier] = None
    summary: Optional[Dict[str, Any]] = None
    if k == 0:
        new = wt.strip(bundle)
    else:
        tier, summary = rebuild(a, k, bundle, args, cells, train_dir, s5)
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
    write_json(out_dir / s5.MODEL_OPTIONS_FILE, {**options, "whitelist_tier_k": k})
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
    bundle: Dict[str, Any],
    args: Dict[str, Any],
    cells: Dict[str, Any],
    train_dir: Path,
    s5: Any,
) -> Tuple[wt.Tier, Dict[str, Any]]:
    """(the tier, its metrics.json summary) of `bundle` with `k` keys per
    cell, on the GEMMs stage05 recorded in `args`."""
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
    tree = sc.load_cumulative_split_tree(args.get("prior_round_dir") or [])
    if args.get("current_round_dir"):
        tree.update(sc.load_cumulative_split_tree([args["current_round_dir"]]))
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
        )
    except (ValueError, RuntimeError, OSError) as e:
        raise Refused(str(e)) from None
    for note in bench.notes:
        ui.warn("identity", note)
    say(f"{len(bench.gemms):,} GEMMs from {len(dirs)} enriched folders")
    gemms = wt.training_gemms(bench.gemms, tree, frac, seed, s5=s5)
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
    ap.add_argument("--jobs", type=int, default=1, help="reading and scoring")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.k is not None and a.k < 0:
        ap.error("--k must be >= 0")
    if (a.library_dir is None) != (a.library_stem is None):
        ap.error("--library-dir and --library-stem go together")
    try:
        return run(a)
    except Refused as e:
        ui.err("tier", str(e))
        return e.code


if __name__ == "__main__":
    sys.exit(main())
