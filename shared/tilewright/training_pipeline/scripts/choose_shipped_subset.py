#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Choose the planner-bank kernels to ship next to a run's library pool.

    choose_shipped_subset.py <run_dir> [--out-dir DIR] [--sizes 0,32,...,all]

The bundle is the one the run shipped (its latest deploy record the gate let
through), else its newest round's stage05 model (`run_analysis.final_model`);
`--model-file` replaces it. Bank kernels are chosen on the training-side
GEMMs of every round up to the bundle's training round (never deploy-gate
slices) and measured on the GEMMs the bundle never trained on: the gate
slice of its training round and the stage04 of later rounds, plus, apart,
the held-out execution contexts. A run without such GEMMs yet (round 0)
chooses on stage05's training GEMMs and evaluates on its validation
(early-stopping) GEMMs, with the split recovered from stage05's outputs
(`lib.stage05_split`); when it cannot be recovered exactly, a shape-hashed
share of the training-side shapes (`--held-out-split`) is kept out of the
selection instead, and the output says which. No validation dataset is read.

For every size B it reports the selection-side and held-out selection
efficiency of the simulated deployed picks (`lib.shipped_subset`), the
perfect-scoring ceiling, the identity ceiling (`lib.identity_ceiling`, under
the engine's grouping for the bundle) and, with bench_shipped_rank
(scripts/experiments, hipBLASLt's launch model when the bundle records it,
its plans and launch keys checked against lib.hipblaslt_launch), the p50 /
p99 time of one
cached ranking call on the pool plus the first B kernels of the greedy
order, single-threaded on one pinned core. <out-dir>/shipped_subset.json holds the orders, the numbers
and the smallest B within 0.002 of the best held-out selection efficiency
under each selection-time budget; the run report shows it.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from lib import bank_pool as bp  # noqa: E402
from lib import evaluate as ev  # noqa: E402
from lib import run_analysis as ra  # noqa: E402
from lib import shipped_subset as ss  # noqa: E402
from lib import stage05_split as s5split  # noqa: E402

OUT_NAME = "shipped_subset.json"
SCHEMA = 1
HELD_LABELS = {
    "held_out": "held-out (deploy-gate slice and later rounds)",
    "stage05_validation": "validation (early-stopping) GEMMs, not a deploy-gate slice",
    "selection_split": "hashed share of training-side shapes (in-sample)",
}


def run_config(run_dir: Path) -> Dict[str, Any]:
    """The run's newest expanded config (configs/<ts>_<mode>.yaml)."""
    paths = [
        p
        for p in sorted(glob.glob(str(run_dir / "configs" / "*.yaml")))
        if not p.endswith(".source.yaml")
    ]
    if not paths:
        return {}
    with open(paths[-1]) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg if isinstance(cfg, dict) else {}


def _get(cfg: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        if not isinstance(cfg, dict) or part not in cfg:
            return None
        cfg = cfg[part]
    return cfg


def bundle_pick_measurement(run_dir: Path, info: ra.ModelInfo) -> Optional[str]:
    """The pick measurement the bundle's stage05 was given."""
    if not info.train_round:
        return None
    try:
        met = json.loads(
            (run_dir / info.train_round / "stage05" / "metrics.json").read_text()
        )
    except (OSError, ValueError):
        return None
    v = (met.get("args") or {}).get("pick_measurement")
    return v if v in ev.PICK_MEASUREMENTS else None


def shape_split(gemms: List[Any], fraction: float, seed: int):
    """(kept, held out): GEMMs whose shape hashes below `fraction` are held
    out, every execution context of a shape on one side."""

    def held(g: Any) -> bool:
        m, n, k, b = g.key[:4]
        h = hashlib.sha256(f"{seed}:{m}:{n}:{k}:{b}".encode()).digest()
        return int.from_bytes(h[:8], "little") / 2.0**64 < fraction

    out = ([], [])
    for g in gemms:
        out[held(g)].append(g)
    return out


def _fmt(v: Optional[float], scale: float = 100.0, digits: int = 2) -> str:
    return "-" if v is None else f"{v * scale:.{digits}f}"


def table_text(doc: Dict[str, Any]) -> str:
    rows = doc["sizes"]

    def contexts(part: str) -> List[str]:
        seen: Dict[str, None] = {}
        for r in rows:
            seen.update(dict.fromkeys(((r.get(part) or {}).get("per_context") or {})))
        return list(seen)

    ev_ctx, hc_ctx = contexts("held_out"), contexts("holdout_contexts")
    head = (
        "| B | configs | selection | evaluation | "
        + "".join(f"eval {c} | " for c in ev_ctx)
        + "held-out contexts | "
        + "".join(f"{c} | " for c in hc_ctx)
        + "eval >1% loss | ceiling eval | identity ceiling eval | p50 us | p99 us "
        "| win-order eval |"
    )
    lines = [head, "|" + "---|" * (12 + len(ev_ctx) + len(hc_ctx))]
    for r in rows:
        h, s = r["held_out"], r["selection"]
        hc = r.get("holdout_contexts") or {}
        lat = r.get("latency_us") or {}
        wb = (r.get("wins_baseline") or {}).get("held_out") or {}

        def per(part: Dict[str, Any], c: str) -> str:
            return _fmt(((part.get("per_context") or {}).get(c) or {}).get("sel_eff"))

        lines.append(
            f"| {r['size']} | {r['n_configs']} | {_fmt(s.get('sel_eff'))} "
            f"| {_fmt(h.get('sel_eff'))} | "
            + "".join(f"{per(h, c)} | " for c in ev_ctx)
            + f"{_fmt(hc.get('sel_eff'))} | "
            + "".join(f"{per(hc, c)} | " for c in hc_ctx)
            + f"{h.get('n_loss_1pct', 0)}/{h.get('n', 0)} "
            f"| {_fmt(h.get('ceiling'))} | {_fmt(h.get('identity_ceiling'))} "
            f"| {_fmt(lat.get('p50'), 1, 1)} "
            f"| {_fmt(lat.get('p99'), 1, 1)} | {_fmt(wb.get('sel_eff'))} |"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None, help="default: <run_dir>")
    ap.add_argument("--model-file", type=Path, default=None, help="MLREC_v3 bundle")
    ap.add_argument("--library-dir", default=None, help="default: the bundle's")
    ap.add_argument("--library-stem", default=None, help="default: the bundle's")
    ap.add_argument(
        "--bank-stem",
        action="append",
        default=None,
        help="bank library file (repeatable; default: the run's bank, "
        "planner.bank / bank_library_stems as lib/bank_pool reads them)",
    )
    ap.add_argument(
        "--pci-chip-id",
        default=None,
        help="the bench device's PCI chip id for the bank's hardware predicates "
        "(default: the config's hardware.pci_chip_id, else the KFD topology)",
    )
    ap.add_argument(
        "--sizes",
        default=",".join(ss.size_label(b) for b in ss.DEFAULT_SIZES),
        help="bank-kernel counts B to report ('all': the whole bank)",
    )
    ap.add_argument(
        "--pick-measurement",
        choices=ev.PICK_MEASUREMENTS,
        default=None,
        help="default: the bundle's stage05, else validate.pick_measurement",
    )
    ap.add_argument("--budgets", default="10,20", help="selection-time budgets, us")
    ap.add_argument("--jobs", type=int, default=8, help="processes")
    ap.add_argument(
        "--held-out-split",
        type=float,
        default=0.5,
        help="share of training-side shapes held out when the run has no held-out GEMMs",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--parity", type=int, default=200, help="GEMMs checked against the engine"
    )
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--bench-binary", type=Path, default=None)
    ap.add_argument("--bench-shapes", type=int, default=1000)
    ap.add_argument(
        "--bench-core", type=int, default=None, help="default: the first allowed CPU"
    )
    ap.add_argument("--bench-iters", type=int, default=200)
    ap.add_argument("--bench-warmup", type=int, default=50)
    a = ap.parse_args(argv)

    run_dir = a.run_dir.resolve()
    out_dir = (a.out_dir or run_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = ss.parse_sizes(a.sizes)
    budgets = [float(x) for x in a.budgets.split(",") if x.strip()]
    t0 = time.time()

    def say(msg: str) -> None:
        print(f"[{time.time() - t0:7.1f}s] {msg}", file=sys.stderr, flush=True)

    info = ra.final_model(run_dir, want_bytes=a.model_file is None)
    if info is None:
        print(f"{run_dir}: no trained bundle", file=sys.stderr)
        return 2
    if a.model_file is not None:
        info.data, info.source = a.model_file.read_bytes(), str(a.model_file)
    if info.data is None or info.hardware is None:
        print(
            f"{run_dir}: the bundle's model file or hardware is unreadable",
            file=sys.stderr,
        )
        return 2
    cfg = run_config(run_dir)
    pm = (
        a.pick_measurement
        or bundle_pick_measurement(run_dir, info)
        or _get(cfg, "validate.pick_measurement")
        or ra.CONFIG_PICK_MEASUREMENT
    )
    lib_dir = a.library_dir or (info.library[0] if info.library else None)
    stem = a.library_stem or (info.library[1] if info.library else None)
    if not lib_dir or not stem:
        print("need the library dir and its stem", file=sys.stderr)
        return 2
    lib_cfg = dict(info.library_cfg or cfg)
    hw_block = dict(lib_cfg.get("hardware") or {})
    for k, v in asdict(info.hardware).items():
        hw_block.setdefault(k, v)
    if a.pci_chip_id is not None:
        hw_block["pci_chip_id"] = int(a.pci_chip_id, 0)
    lib_cfg["hardware"] = hw_block
    try:
        if a.bank_stem is not None:
            lp = bp.library_pool(
                lib_dir,
                stem,
                bp.bank_device(lib_cfg, info.library_device),
                a.bank_stem,
            )
        else:
            lp = bp.config_pool(lib_cfg, lib_dir, stem, info.library_device)
    except RuntimeError as e:
        print(f"bank device: {e} (or pass --pci-chip-id)", file=sys.stderr)
        return 2
    banks = lp.bank.stems
    if not lp.bank.kernels:
        print(
            "the run's config gives no bank (planner.bank / bank_library_stems); "
            "name its files with --bank-stem",
            file=sys.stderr,
        )
        return 2
    tw = ev.tilewright_module()
    mv = ss.model_view(info.data, info.hardware, tw)
    pool = ev.pool_from_kernels(lp.table)
    bank = ev.pool_from_kernels(lp.bank.kernels)
    u = ss.build_universe(pool, bank, mv.spec, mv.key_launch)
    n_bank = u.n - u.n_pool
    say(
        f"bundle {info.source} ({info.train_round}), {u.n_pool} pool + {n_bank} bank "
        f"kernels, {u.n_keys} candidates in {u.n_views} view keys, cascade "
        f"{mv.cascade}, pick {pm}"
    )

    sets = ra.model_benches(run_dir, info)

    def load(bs: ra.BenchSet) -> List[Any]:
        if not bs:
            return []
        return ra.load_bench(bs.dirs, plans=bs.plans, jobs=a.jobs).gemms

    sel_g = load(sets["training"])
    held_g = load(sets["held_out"])
    hc_g = load(sets["holdout_contexts"])
    held_source = "held_out"
    notes: List[str] = []
    split_doc: Optional[Dict[str, Any]] = None
    if not held_g:
        rec = (
            s5split.recover(
                run_dir / info.train_round / "stage05",
                sel_g,
                data_dirs=sets["training"].dirs,
            )
            if info.train_round
            else s5split.Split(reason="the bundle has no training round")
        )
        split_doc = rec.summary()
        if rec.exact:
            sel_g, held_g = rec.train + rec.untrained, rec.validation
            held_source = "stage05_validation"
            notes.append(
                "no GEMM of this run is held out from the bundle yet: the "
                "evaluation GEMMs are stage05's validation (early-stopping) GEMMs, "
                "not a deploy-gate slice; they chose the checkpoint but gave no "
                "gradients. The selection GEMMs are stage05's training GEMMs"
                + (
                    f" plus the {len(rec.untrained)} GEMMs of cells it did not train"
                    if rec.untrained
                    else ""
                )
            )
        else:
            sel_g, held_g = shape_split(sel_g, a.held_out_split, a.seed)
            held_source = "selection_split"
            notes.append(
                f"no GEMM of this run is held out from the bundle yet and stage05's "
                f"training / validation split could not be recovered exactly "
                f"({rec.reason}): {a.held_out_split:g} of the training-side shapes "
                f"(by hash) are kept out of the selection and measured instead; "
                f"the bundle trained on them"
            )
    say(
        f"GEMMs: {len(sel_g)} selection, {len(held_g)} held-out ({held_source}), "
        f"{len(hc_g)} held-out contexts"
    )

    def table(gemms: List[Any]) -> Optional[ss.Table]:
        if not gemms:
            return None
        return ss.build_table(gemms, u, mv, pick_measurement=pm, workers=a.jobs)

    sel = table(sel_g)
    say("selection scores done")
    held = table(held_g)
    hc = table(hc_g)
    say("held-out scores done")
    if sel is None:
        print("no training-side GEMMs", file=sys.stderr)
        return 2

    rng = np.random.default_rng(a.seed)
    par: Dict[str, Any] = {}
    if a.parity > 0:
        pt = held if held is not None else sel
        rows = np.sort(rng.choice(pt.n, size=min(a.parity, pt.n), replace=False))
        random_s = u.present(rng.choice(u.bank, size=min(128, n_bank), replace=False))
        for name, present in (
            ("full_set", u.present(u.bank)),
            ("random_subset", random_s),
        ):
            par[name] = ss.parity(
                pt, present, rows, pick_measurement=pm, data=mv.data,
                hardware=mv.hardware, tw=tw,
            )  # fmt: skip
            par[name + "_cascade"] = ss.cascade_parity(
                pt, present, rows, data=mv.data, hardware=mv.hardware, tw=tw
            )
        say(
            "parity " + ", ".join(f"{k} {v['n_same']}/{v['n']}" for k, v in par.items())
        )

    finite = [b for b in sizes if b is not None]
    max_b = min(max(finite, default=0), n_bank)
    wins = ss.win_order(sel)
    greedy = ss.greedy_order(
        sel,
        max_b,
        progress=lambda i, obj: (
            say(f"greedy {i}/{max_b}: geomean {math.exp(obj / sel.n) * 100:.3f}%")
            if i % 32 == 0 or i == max_b
            else None
        ),
    )
    say("greedy done")

    binary = model_file = None
    launch_id, launch_par = "", None
    core = a.bench_core
    if not a.no_latency:
        if core is None:
            core = min(os.sched_getaffinity(0))
        binary = a.bench_binary or ss.build_bench(
            PIPELINE_DIR / "scripts" / "experiments", out_dir / "bench_build", a.jobs
        )
        model_file = out_dir / "model.tilewright.bin"
        model_file.write_bytes(mv.data)
        shapes = ss.bench_shapes(sel, a.bench_shapes, a.seed)
        launch_id = str(
            tw.describe(ev.load_model_bytes(mv.data, tw)).provenance.launch_model or ""
        )
        if launch_id:
            every = out_dir / "bench_inputs_plans.txt"
            every.write_text(
                ss.bench_inputs(sel.problems[0], mv.hardware, shapes, u.kernels)
            )
            launch_par = ss.launch_parity(
                binary, every, sel.problems[0], shapes, u.kernels, mv.hardware,
                launch_id,
            )  # fmt: skip
            say(
                f"launch plans of {Path(binary).name} vs lib.hipblaslt_launch: "
                f"{launch_par['n_same']}/{launch_par['n']}"
            )
            if launch_par["n_same"] != launch_par["n"] or not launch_par["checked"]:
                notes.append(
                    f"the latency binary's launch plans differ from the pipeline's "
                    f"({launch_par['n_same']}/{launch_par['n']} equal): its ranking "
                    f"time is not that of the deployed launch model"
                )

    rows_out: List[Dict[str, Any]] = []
    seen = set()
    for b in sizes:
        nb = n_bank if b is None else min(int(b), n_bank)
        if nb in seen:
            continue
        seen.add(nb)
        chosen = list(u.bank) if b is None else greedy.order[:nb]
        present = u.present(chosen)
        row: Dict[str, Any] = {
            "size": ss.size_label(b) if b is None or nb == b else str(nb),
            "n_bank": nb,
            "n_configs": int(present.sum()),
            "bank_sol_idx": [int(u.sids[i]) for i in chosen],
            "selection": ss.set_metrics(sel, present),
            "held_out": ss.set_metrics(held, present, worst=20) if held else {"n": 0},
        }
        if hc is not None:
            row["holdout_contexts"] = ss.set_metrics(hc, present)
        if 0 < nb < n_bank:
            pw = u.present(wins[:nb])
            row["wins_baseline"] = {
                "bank_sol_idx": [int(u.sids[i]) for i in wins[:nb]],
                "selection": ss.set_metrics(sel, pw, with_ceiling=False),
                "held_out": (
                    ss.set_metrics(held, pw, with_ceiling=False) if held else {"n": 0}
                ),
            }
        if binary is not None:
            inputs = out_dir / f"bench_inputs_{nb}.txt"
            inputs.write_text(
                ss.bench_inputs(
                    sel.problems[0], mv.hardware, shapes, u.shipped(present)
                )
            )
            row["latency_us"] = ss.run_bench(
                binary, model_file, inputs, core=core,
                iters=a.bench_iters, warmup=a.bench_warmup,
            )  # fmt: skip
        rows_out.append(row)
        say(
            f"B={row['size']}: held-out {_fmt(row['held_out'].get('sel_eff'))}"
            + (
                f", p99 {row['latency_us']['p99']:.1f} us"
                if row.get("latency_us")
                else ""
            )
        )

    doc = {
        "schema": SCHEMA,
        "run": run_dir.name,
        "bundle": {
            "source": str(info.source),
            "train_round": info.train_round,
            "deploy_round": info.deploy_round,
            "weight_dtype": info.weight_dtype,
            "sha256": mv.sha256,
            "cascade_keys": mv.cascade,
        },
        "pick_measurement": pm,
        "pool": {"stem": stem, "n": u.n_pool},
        "bank": {"stems": banks, "n": n_bank},
        "n_view_keys": u.n_keys,
        "selection": {"n_gemms": sel.n, "source": "training"},
        "held_out": {
            "n_gemms": held.n if held else 0,
            "source": held_source,
            "label": HELD_LABELS[held_source],
            "stage05_split": split_doc,
        },
        "holdout_contexts": {"n_gemms": hc.n if hc else 0},
        "greedy": {
            "order_sol_idx": [int(u.sids[i]) for i in greedy.order],
            "objective_geomean": [math.exp(o / sel.n) for o in greedy.objective],
        },
        "wins_order_sol_idx": [int(u.sids[i]) for i in wins[:max_b]],
        "latency": (
            {
                "binary": Path(binary).name if binary is not None else None,
                "launch_model": launch_id or None,
                "launch_parity": launch_par,
                "pinned_cpus": 1,
                "iters": a.bench_iters,
                "warmup": a.bench_warmup,
                "n_shapes": len(shapes),
                "min_scored": 1,
            }
            if binary is not None
            else None
        ),
        "parity": par,
        "sizes": rows_out,
        "recommendation": ss.recommend(rows_out, budgets),
        "notes": notes,
    }
    out = out_dir / OUT_NAME
    out.write_text(json.dumps(doc, indent=1, allow_nan=False, default=_json_default))
    print(table_text(doc))
    for budget, v in doc["recommendation"].items():
        print(
            f"{budget} us: "
            + (
                "no measured size fits"
                if v is None
                else f"B={v['size']} (held-out {_fmt(v['held_out_sel_eff'])}%, "
                f"p99 {v['p99_us']:.1f} us)"
            )
        )
    say(f"wrote {out}")
    return 0


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(type(o))


if __name__ == "__main__":
    sys.exit(main())
