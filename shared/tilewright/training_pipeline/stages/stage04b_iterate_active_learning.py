# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage04b_iterate_active_learning -- validate the prior round's models on
this round's new GEMMs and decide per cell: pass / retrain / split.

Runs in active rounds only (R >= 1), between stage04 (enrich) and stage05
(retrain); the `b` encodes "stage 4.5". In round R it reads
round_(R-1)/stage05/models.pt and round_R/stage04, and writes

    decisions.json     per-cell measurements and decisions
    retrain_cells.txt  stage05 --only-cells list: retrained cells, leaves
                       without a model of their own that stage05 can train
                       now, and the children of this round's splits
    splits.json        this round's split rules (read by stage01 and stage05)
    gate/, train/      with --gate-frac: the rows of this round's gate GEMMs
                       and every other row of --enriched-csv-dir (chunk_*.csv
                       and kernels.json), each with an enrich_summary.json
                       keeping what the folder's own records of how its rows
                       were produced (the library, the blocklist, the launch
                       record), so that stage05 knows the labels it trains on

Deploy gate (--gate-frac F > 0): of each leaf's held-out GEMMs, round(F * n)
(the lowest values of a hash of --seed, --round-idx and the GEMM) are kept
out of this round's training: stage05 trains on train/ instead of
--enriched-csv-dir, and stage06 measures the bundle this round ships on
gate/, GEMMs no bundle has trained on. Later rounds train on them like on
any earlier round's data.

Only cells stage05 can train are listed: a leaf with at least
--train-min-cell-gemms GEMMs over every round's stage04 data (the data stage05
trains on, without this round's gate GEMMs). A leaf without its own model gets
decision `train` when it has that many and `no_model` otherwise; it stays
served by its nearest trained ancestor until it is trained. A retrain cell
below the bound is reported in `cells_not_trainable`. Split parents whose
model the bundle keeps for such leaves are listed in `cells_fallback_parents`
and get no decision.

Held-out data: this round's GEMMs minus any shape the evaluated models could
have trained on, i.e. shapes stage01 re-used from history
(`round_R/stage01/reused_shapes.yaml`, or --reused-shapes) and GEMMs present
in a prior round's stage04 data.

Measurement: the bundle is evaluated exactly as deployed (lib/evaluate.py):
MLREC_v2 at --weight-dtype, the engine's routing with ancestor fallback, the
per-cell whitelist, the library pool, and the Origami fallback for GEMMs the
model scores nothing for. A pick without a timed row of its own is a miss
under --pick-measurement timed (GEMMs benched from --plan know which kernels
were never benched). Cells are the leaves of the cumulative split tree of the
prior rounds; a leaf without its own model is served like at runtime.
With --rebench-misses every miss of the bundle (and of --prior-models) is
re-benched on --devices with --bench-binary and charged by its measured
ratio to the winner (lib/miss_rebench.py); the decisions use those numbers,
decisions.json keeps the configured rule's as `sel_eff_new_pre_rebench` per
cell and `aggregate.pre_rebench`, and rebench_misses.json lists the misses
(with the library digest of their solution indices, and each pick and
re-benched winner named by signature and kernel id).
Thresholds are absolute selection efficiencies. Origami's numbers are
recorded on the GEMMs that have an Origami pick, next to the model's on the
same GEMMs, and printed only with --origami-reference.

Decision per cell with a model and >= --min-cell-gemms held-out GEMMs:
  * sel_eff_new >= --sel-eff-threshold                         -> pass
  * below the threshold, after --split-after-attempts retrains, and
    either below --split-floor or not improving by
    --split-min-improvement over the improvement baseline      -> split
  * otherwise                                                  -> retrain
The improvement baseline is the round_(R-2) bundle (--prior-models) on the
same GEMMs, else the cell's previous sel_eff_new (--prior-decisions). Without
a baseline (round 1, a new child) only the split floor triggers a split.
--split-min-improvement <= 0 disables the improvement check (any cell below
the threshold splits once its attempts are used up).

A split needs --min-cell-gemms evaluated GEMMs on each side and, so that
stage05 can train both children, --train-min-cell-gemms GEMMs on each side
over every round's data; a leaf that misses either is retrained instead and
reported (`cells_split_blocked`), and so is a per-leaf shape budget
(--per-leaf-budget or round_R/stage01/per_cell_plan.json) that can never
supply the evaluated GEMMs.

The stage fails (exit 1, no outputs) when the engine serves any GEMM from
another cell than lib/subcells routes it to.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = THIS_DIR.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

try:
    import torch  # noqa: E402

    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

import stage05_train as _train  # noqa: E402
from lib import audit as auditlib  # noqa: E402
from lib import evaluate as ev  # noqa: E402
from lib import hardware as hwlib  # noqa: E402
from lib import miss_rebench as mr  # noqa: E402
from lib import mlrec  # noqa: E402
from lib import repeats  # noqa: E402
from lib import subcells as sc  # noqa: E402
from lib import ui  # noqa: E402
from lib.correctness import config_blocklist  # noqa: E402
from lib.fs import write_json  # noqa: E402
from lib.shapes import read_reused_shapes  # noqa: E402

LOW_SEL_EFF = 0.70


def _finite(text: str) -> float:
    v = float(text)
    if not math.isfinite(v):
        raise argparse.ArgumentTypeError(f"expected a finite number, got {text!r}")
    return v


_ROUND_RE = re.compile(r"^round_(\d+)$")


def _round_index(p: Path) -> Optional[int]:
    m = _ROUND_RE.match(p.name)
    return int(m.group(1)) if m else None


def _discover_prior_rounds(models_dir: Path, round_idx: int) -> List[Path]:
    """Round dirs of the run that holds `models_dir`, before `round_idx`."""
    run_dir = Path(models_dir).resolve().parent.parent
    rounds = []
    for d in run_dir.glob("round_*"):
        i = _round_index(d)
        if i is not None and i < round_idx:
            rounds.append((i, d))
    return [d for _i, d in sorted(rounds)]


def history_gemm_keys(round_dirs: Iterable[Path]) -> Set[Tuple[Any, ...]]:
    """GEMM keys of every row of the rounds' stage04 chunk_*.csv files."""
    keys: Set[Tuple[Any, ...]] = set()
    for rd in round_dirs:
        d = Path(rd) / "stage04"
        if not d.is_dir():
            continue
        for p in sorted(d.glob("chunk_*.csv")):
            with p.open() as f:
                for row in csv.DictReader(f):
                    try:
                        keys.add(ev.gemm_key(row))
                    except (KeyError, TypeError, ValueError):
                        continue
    return keys


GATE_DIR = "gate"
TRAIN_DIR = "train"
ENRICH_SUMMARY = "enrich_summary.json"
# What a stage04 enrich_summary.json says of every row of its folder; its
# counts describe the whole folder and stay out of a slice's summary.
SLICE_SUMMARY_KEYS = (
    "library_stem",
    "extra_library_stems",
    "plan",
    "library_files",
    "invalid_library_files",
    "n_library_solutions",
    "missing_size_mapping_keys",
    "context",
    "kernels_blocked",
    "blocklist",
    "library_digest",
    "launch",
)
_ROW_KEY_FIELDS = (
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
    "cu_budget",
    "schedule",
)


def _gate_unit(seed: int, round_idx: int, gemm: Dict[str, Any]) -> float:
    key = "|".join(str(v) for v in ev.gemm_key(gemm))
    h = hashlib.sha256(f"gate|{int(seed)}|{int(round_idx)}|{key}".encode()).digest()
    return int.from_bytes(h[:8], "little") / float(1 << 64)


def select_gate(
    held_out: Dict[str, List[dict]], frac: float, seed: int, round_idx: int
) -> Dict[str, List[dict]]:
    """{leaf: gate GEMMs}: round(frac * n) of each leaf's n held-out GEMMs,
    lowest `_gate_unit` first."""
    out: Dict[str, List[dict]] = {}
    if frac <= 0:
        return out
    for label, gs in held_out.items():
        n = int(math.floor(frac * len(gs) + 0.5))
        if n > 0:
            out[label] = sorted(gs, key=lambda g: _gate_unit(seed, round_idx, g))[:n]
    return out


def split_rows(
    src: Path, gate_keys: Set[Tuple[Any, ...]], out_dir: Path
) -> Dict[str, int]:
    """Copy the chunk_*.csv rows of `src` to `out_dir`/gate (rows of GEMMs in
    `gate_keys`) and `out_dir`/train (all other rows), chunk by chunk with
    each file's header; the kernel attribute file goes to both, and so does
    what `src`'s enrich_summary.json records of how its rows were produced
    (`SLICE_SUMMARY_KEYS`: the library, the blocklist that filtered them, how
    their benches launched the kernels), with the slice's own row count.
    Returns the row counts."""
    from lib.dat import KERNELS_FILE

    counts = {GATE_DIR: 0, TRAIN_DIR: 0}
    dirs = {name: out_dir / name for name in counts}
    for d in dirs.values():
        d.mkdir(parents=True)
        if (src / KERNELS_FILE).is_file():
            shutil.copy2(src / KERNELS_FILE, d / KERNELS_FILE)
    for path in sorted(src.glob("chunk_*.csv")):
        with path.open(newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                continue
            idx = [header.index(c) if c in header else -1 for c in _ROW_KEY_FIELDS]
            writers: Dict[str, Any] = {}
            files = []
            last: Optional[Tuple[str, ...]] = None
            to_gate = False
            try:
                for row in reader:
                    raw = tuple(row[i] if 0 <= i < len(row) else "" for i in idx)
                    if raw != last:
                        last = raw
                        try:
                            key = ev.gemm_key(dict(zip(_ROW_KEY_FIELDS, raw)))
                        except (KeyError, TypeError, ValueError):
                            key = None
                        to_gate = key in gate_keys
                    name = GATE_DIR if to_gate else TRAIN_DIR
                    w = writers.get(name)
                    if w is None:
                        fh = (dirs[name] / path.name).open("w", newline="")
                        files.append(fh)
                        w = writers[name] = csv.writer(fh, lineterminator="\n")
                        w.writerow(header)
                    w.writerow(row)
                    counts[name] += 1
            finally:
                for fh in files:
                    fh.close()
    try:
        summary = json.loads((src / ENRICH_SUMMARY).read_text())
    except (OSError, ValueError):
        summary = None
    if isinstance(summary, dict):
        for name, d in dirs.items():
            part = {k: summary[k] for k in SLICE_SUMMARY_KEYS if k in summary}
            part.update(slice=name, slice_of=str(src), rows=counts[name])
            write_json(d / ENRICH_SUMMARY, part)
    return counts


def _per_leaf_budget(arg: Optional[int], round_dir: Path) -> Optional[int]:
    if arg is not None:
        return int(arg)
    plan = round_dir / "stage01" / "per_cell_plan.json"
    if plan.is_file():
        try:
            v = json.loads(plan.read_text()).get("shapes_per_cell")
            return int(v) if v is not None else None
        except (ValueError, TypeError, OSError):
            return None
    return None


def _evaluate(
    bundle: Dict[str, Any],
    gemms: List[dict],
    pools: Dict[Tuple[str, ...], List[Dict[str, int]]],
    args: argparse.Namespace,
    constants: hwlib.ArchConstants,
    hardware: hwlib.DeviceHardware,
    tw: Any,
    what: str,
    bench: Optional[mr.MissBench] = None,
) -> Tuple[Dict[int, ev.GemmResult], Optional[mr.Measured]]:
    """({id(gemm): deployed result} of `bundle` on `gemms`, with `bench` the
    evaluation whose misses were re-benched); RoutingMismatchError when the
    engine routes any GEMM elsewhere than lib/subcells."""

    def evaluate(rule: str) -> ev.Evaluation:
        out = ev.evaluate_bundle(
            bundle,
            gemms,
            arch=args.arch,
            constants=constants,
            hardware=hardware,
            weight_dtype=args.weight_dtype,
            pools=pools,
            tw=tw,
            pick_measurement=rule,
        )
        return ev.require_consistent_routing(out, what)

    measured = mr.measured_evaluation(
        evaluate, gemms, bench, args.pick_measurement, what.replace(" ", "_")
    )
    by_id = {id(g): r for g, r in zip(gemms, measured.evaluation.results)}
    return by_id, (measured if bench is not None else None)


def _context_label(g: Mapping[str, Any]) -> str:
    c = ev.context_of(g)
    return f"{c.cu_budget}/{c.schedule}"


def _summaries_by_context(
    gemms: List[dict], results: Dict[int, ev.GemmResult]
) -> Dict[str, Any]:
    """{"cu_budget/schedule": summary} of the evaluated GEMMs."""
    groups: Dict[str, List[ev.GemmResult]] = {}
    for g in gemms:
        r = results.get(id(g))
        if r is not None:
            groups.setdefault(_context_label(g), []).append(r)
    return {k: ev.summarize(v) for k, v in sorted(groups.items())}


def _context_holdout(
    args,
    bundle,
    library,
    split_tree,
    constants,
    hardware,
    tw,
    identity,
    aggregation,
    blocklist=None,
):
    """Per-context summary of `bundle` on --context-holdout-dir (None when the
    directory has no rows)."""
    if not any(args.context_holdout_dir.glob("chunk_*.csv")):
        ui.warn("holdout", f"no held-out context rows in {args.context_holdout_dir}")
        return None
    cells, _ = _train.load_enriched_chunks(
        str(args.context_holdout_dir),
        args.quiet,
        split_tree=(split_tree or None),
        identity=identity,
        aggregation=aggregation,
        blocklist=blocklist,
    )
    gemms = [g for gs in cells.values() for g in gs]
    try:
        results, _ = _evaluate(
            bundle,
            gemms,
            ev.pools_for(gemms, library),
            args,
            constants,
            hardware,
            tw,
            "held-out contexts",
        )
    except (ev.RoutingMismatchError, RuntimeError, ValueError) as e:
        ui.warn("holdout", f"held-out context evaluation failed: {e}")
        return {"error": str(e)}
    out = _summaries_by_context(gemms, results)
    if not args.quiet:
        for label, summ in out.items():
            ui.info(
                "holdout",
                f"context {label}: sel_eff {summ['sel_eff']:.4f} over "
                f"{summ['n_evaluated']} GEMM(s)",
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="stage04b: validate the prior round's models on this "
        "round's new GEMMs and decide pass / retrain / split per cell."
    )
    ap.add_argument(
        "--models-dir",
        type=Path,
        required=True,
        help="prior round's stage05 dir (models.pt + cells.json)",
    )
    ap.add_argument(
        "--enriched-csv-dir",
        type=Path,
        required=True,
        help="this round's stage04 dir (chunk_*.csv)",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument(
        "--config-yaml",
        type=Path,
        default=None,
        help="run config; its `hardware:` block sets the device values",
    )
    ap.add_argument("--hardware-device", type=int, default=0)
    ap.add_argument(
        "--weight-dtype",
        choices=sorted(mlrec.WEIGHT_DTYPES),
        default="bf16",
        help="weight dtype of the deployed model",
    )
    ap.add_argument(
        "--library-dir",
        type=Path,
        default=None,
        help="Tensile library dir holding <library-stem>.dat[.zlib]; with "
        "--library-stem the candidate pool is that library (default: the "
        "kernels measured in this round's data)",
    )
    ap.add_argument("--library-stem", default=None)
    ap.add_argument(
        "--sel-eff-threshold",
        type=_finite,
        default=0.95,
        help="pass threshold: the held-out selection efficiency a cell needs",
    )
    ap.add_argument(
        "--split-floor",
        type=_finite,
        default=0.85,
        help="below this a cell whose attempts are used up splits without the "
        "improvement check; 0 disables",
    )
    ap.add_argument(
        "--pick-measurement",
        choices=ev.PICK_MEASUREMENTS,
        default="warmup",
        help="warmup: a skip row counts with its warm-up latency and a kernel off the plan's list is a miss; timed: both are misses; any: a skip row counts "
        "and an off-list kernel is passed over (lib/evaluate.py)",
    )
    ap.add_argument(
        "--plan",
        type=Path,
        default=None,
        help="stage02 plan.json of this round: GEMMs benched from a kernel "
        "list miss when the model picks a kernel off it",
    )
    ap.add_argument(
        "--origami-reference",
        action="store_true",
        help="also print Origami's numbers (an external reference; they never "
        "enter a decision)",
    )
    ap.add_argument(
        "--gate-frac",
        type=_finite,
        default=0.0,
        help="fraction of each leaf's held-out GEMMs kept out of this round's "
        "training for the deploy gate (writes gate/ and train/); 0 disables",
    )
    ap.add_argument(
        "--seed", type=int, default=42, help="run seed; picks the gate GEMMs"
    )
    ap.add_argument(
        "--split-min-improvement",
        type=float,
        default=0.02,
        help="a below-threshold cell that improved less than this over its "
        "improvement baseline splits (once its attempts are used up); <= 0 "
        "disables the check",
    )
    ap.add_argument(
        "--split-after-attempts",
        type=int,
        default=2,
        help="retrains a cell gets before it may split (0: split at once)",
    )
    ap.add_argument(
        "--max-split-depth",
        type=int,
        default=6,
        help="cells at this split depth are never split",
    )
    ap.add_argument(
        "--structural-split-min-octaves",
        type=float,
        default=0.0,
        help="split any cell whose widest splittable axis still spans more "
        "than this many octaves, whatever its sel_eff (0 disables)",
    )
    ap.add_argument(
        "--structural-split-max-depth",
        type=int,
        default=-1,
        help="depth limit of structural splits (-1: --max-split-depth)",
    )
    ap.add_argument(
        "--prior-decisions",
        type=Path,
        action="append",
        default=None,
        help="repeatable earlier `<round>/stage04b/decisions.json`, oldest "
        "first: retrain attempts and previous sel_eff per cell",
    )
    ap.add_argument(
        "--prior-round-dir",
        type=Path,
        action="append",
        default=None,
        help="repeatable `<run>/round_<i>/` dirs before this round (default: "
        "the run's rounds before --round-idx): split tree and history shapes",
    )
    ap.add_argument(
        "--prior-models",
        type=Path,
        default=None,
        help="round_(R-2) stage05/models.pt: improvement baseline on the same " "GEMMs",
    )
    ap.add_argument(
        "--reused-shapes",
        type=Path,
        action="append",
        default=None,
        help="reused_shapes.yaml file(s) or stage01 dir(s) listing history "
        "shapes re-benched this round (default: <round>/stage01)",
    )
    ap.add_argument(
        "--no-history-exclusion",
        action="store_true",
        help="keep GEMMs that also occur in a prior round's stage04 data",
    )
    ap.add_argument(
        "--context-holdout-dir",
        type=Path,
        default=None,
        help="enriched CSVs of held-out execution contexts (stage03c): "
        "reported per context in aggregate.context_holdout, never part of the "
        "decisions",
    )
    ap.add_argument(
        "--per-leaf-budget",
        type=int,
        default=None,
        help="new shapes stage01 gives each leaf per round (default: "
        "<round>/stage01/per_cell_plan.json shapes_per_cell)",
    )
    ap.add_argument("--round-idx", type=int, default=1)
    ap.add_argument(
        "--min-cell-gemms",
        type=int,
        default=20,
        help="fewer held-out GEMMs carry the previous decision forward; "
        "also the per-side minimum of a split",
    )
    ap.add_argument(
        "--train-min-cell-gemms",
        type=int,
        default=50,
        help="stage05 --min-cell-gemms: GEMMs over every round a cell needs to "
        "be listed for training, and each side of a split needs",
    )
    ap.add_argument(
        "--final-round",
        action="store_true",
        help="demote splits to retrains (no later round trains the children)",
    )
    ap.add_argument("--quiet", action="store_true")
    mr.add_arguments(ap)
    auditlib.add_arguments(ap)
    args = ap.parse_args()

    if not _TORCH_OK:
        ui.err("err", "torch is not importable; pip install -r requirements.txt")
        return 1
    try:
        tw = ev.tilewright_module()
    except RuntimeError as e:
        ui.err("engine", str(e))
        return 1
    cfg: Dict[str, Any] = {}
    if args.config_yaml is not None:
        with args.config_yaml.open() as f:
            cfg = yaml.safe_load(f) or {}
    try:
        hardware = hwlib.device_hardware(cfg, args.hardware_device)
        constants = hwlib.arch_constants(args.arch)
    except (ValueError, RuntimeError, OSError) as e:
        ui.err("hardware", str(e))
        return 1
    try:
        bench = mr.from_args(args, cfg, args.out_dir / "rebench", hardware.n_cu)
    except ValueError as e:
        ui.err("rebench", str(e))
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    round_dir = args.out_dir.resolve().parent
    t0 = time.time()
    if not args.quiet:
        ui.banner("stage04b  validate prior models on new GEMMs", ui.C.HEAD)
        ui.info("cfg", f"models_dir       : {args.models_dir}")
        ui.info("cfg", f"enriched_csv_dir : {args.enriched_csv_dir}")
        ui.info("cfg", f"out_dir          : {args.out_dir}")
        ui.info("cfg", f"arch / weights   : {args.arch} / {args.weight_dtype}")
        ui.info(
            "cfg",
            f"threshold / floor: {args.sel_eff_threshold} / {args.split_floor}",
        )

    models_pt = args.models_dir / "models.pt"
    if not models_pt.exists():
        ui.err("err", f"missing models.pt at {models_pt}")
        return 1
    bundle = torch.load(models_pt, map_location="cpu", weights_only=True)
    models = bundle["models"]
    if not args.quiet:
        ui.ok(
            "models",
            f"{len(models)} cells (trained in that round: "
            f"{bundle.get('n_models_trained_this_round', len(models))}, "
            f"carried: {bundle.get('n_models_carried_forward', 0)})",
        )
    prior_bundle: Optional[Dict[str, Any]] = None
    if args.prior_models is not None:
        try:
            prior_bundle = torch.load(
                str(args.prior_models), map_location="cpu", weights_only=True
            )
        except Exception as e:
            ui.warn("prior", f"cannot load --prior-models ({e}); not used")

    prior_round_dirs = list(args.prior_round_dir or []) or _discover_prior_rounds(
        args.models_dir, args.round_idx
    )
    split_tree = sc.load_cumulative_split_tree(prior_round_dirs)
    if split_tree and not args.quiet:
        ui.ok("splits", f"cumulative split tree: {len(split_tree)} split cells")

    if (args.library_dir is None) != (args.library_stem is None):
        ui.err("pool", "--library-dir and --library-stem go together")
        return 1
    identity = (
        ev.library_identity(
            args.library_dir, args.library_stem, cfg, args.hardware_device
        )
        if args.library_stem
        else None
    )
    blocklist = config_blocklist(cfg)
    new_cells, _load_stats = _train.load_enriched_chunks(
        str(args.enriched_csv_dir),
        args.quiet,
        split_tree=(split_tree or None),
        identity=identity,
        aggregation=repeats.config_aggregation(cfg),
        blocklist=blocklist,
    )

    reused: Set[Tuple[int, int, int, int]] = set()
    reused_files = list(args.reused_shapes or [round_dir / "stage01"])
    for p in reused_files:
        reused |= read_reused_shapes(p)
    history = (
        set() if args.no_history_exclusion else history_gemm_keys(prior_round_dirs)
    )
    excluded_by_cell: Dict[str, int] = {}
    all_new: List[dict] = []
    held_out: Dict[str, List[dict]] = {}
    for label, gs in new_cells.items():
        keep = []
        for g in gs:
            all_new.append(g)
            shape = (g["m"], g["n"], g["k"], g["batch_count"])
            if shape in reused or ev.gemm_key(g) in history:
                excluded_by_cell[label] = excluded_by_cell.get(label, 0) + 1
            else:
                keep.append(g)
        if keep:
            held_out[label] = keep
    n_excluded = sum(excluded_by_cell.values())
    if not args.quiet:
        ui.ok(
            "data",
            f"new GEMMs: {len(all_new)}  held out: {len(all_new) - n_excluded}  "
            f"excluded (re-used / seen before): {n_excluded}  "
            f"[{len(reused)} re-used shapes from {len(reused_files)} file(s), "
            f"{len(history)} history GEMMs]",
        )
    n_listed = 0
    if args.plan is not None:
        try:
            n_listed = ev.attach_plan(all_new, ev.read_plan(args.plan), identity)
        except (OSError, ValueError, KeyError, TypeError) as e:
            ui.err("plan", f"cannot read --plan {args.plan}: {e!r}")
            return 1
        if not args.quiet:
            ui.ok("plan", f"{n_listed} new GEMMs were benched from a kernel list")

    library = (
        ev.library_pool(args.library_dir, args.library_stem, cfg, args.hardware_device)
        if args.library_stem
        else None
    )
    pools = ev.pools_for(all_new, library)
    held_list = [g for gs in held_out.values() for g in gs]
    prior_results: Dict[int, ev.GemmResult] = {}
    measured: Dict[str, mr.Measured] = {}
    try:
        results: Dict[int, ev.GemmResult] = {}
        if held_list:
            results, m_held = _evaluate(
                bundle,
                held_list,
                pools,
                args,
                constants,
                hardware,
                tw,
                "held out",
                bench,
            )
            if m_held is not None:
                measured["held_out"] = m_held
        if prior_bundle is not None and held_list:
            try:
                prior_results, m_prior = _evaluate(
                    prior_bundle,
                    held_list,
                    pools,
                    args,
                    constants,
                    hardware,
                    tw,
                    "--prior-models",
                    bench,
                )
                if m_prior is not None:
                    measured["prior_models"] = m_prior
            except (RuntimeError, ValueError) as e:
                ui.warn("prior", f"--prior-models evaluation failed ({e}); not used")
    except ev.RoutingMismatchError as e:
        ui.err("eval", str(e))
        return 1
    prior_labels = set((prior_bundle or {}).get("models", {}))
    pre_results: Dict[int, ev.GemmResult] = {}
    if "held_out" in measured:
        pre_results = {
            id(g): r for g, r in zip(held_list, measured["held_out"].pre.results)
        }

    gate = select_gate(held_out, args.gate_frac, args.seed, args.round_idx)
    gate_record: Optional[Dict[str, Any]] = None
    training_dir = args.enriched_csv_dir
    for name in (GATE_DIR, TRAIN_DIR):
        if (args.out_dir / name).exists():
            shutil.rmtree(args.out_dir / name)
    if gate:
        gate_keys = {ev.gemm_key(g) for gs in gate.values() for g in gs}
        rows = split_rows(args.enriched_csv_dir, gate_keys, args.out_dir)
        training_dir = args.out_dir / TRAIN_DIR
        gate_record = {
            "frac": args.gate_frac,
            "seed": args.seed,
            "n_gemms": len(gate_keys),
            "n_cells": len(gate),
            "n_rows": rows[GATE_DIR],
            "n_training_rows": rows[TRAIN_DIR],
            "dir": str(args.out_dir / GATE_DIR),
            "training_dir": str(training_dir),
            "per_cell": {label: len(gs) for label, gs in sorted(gate.items())},
        }
        if not args.quiet:
            ui.ok(
                "gate",
                f"{len(gate_keys)} held-out GEMMs of {len(gate)} cells kept out of "
                f"this round's training for the deploy gate -> {args.out_dir / GATE_DIR}",
            )

    union = _train.gemm_shapes_by_leaf(
        [Path(rd) / "stage04" for rd in prior_round_dirs] + [training_dir],
        split_tree or None,
    )
    train_min = max(1, int(args.train_min_cell_gemms))
    fallback_parents = sorted(lbl for lbl in models if lbl in split_tree)

    min_split_count = max(1, int(args.min_cell_gemms))
    need_for_split = 2 * min_split_count
    budget = _per_leaf_budget(args.per_leaf_budget, round_dir)
    budget_check = {
        "per_leaf_budget": budget,
        "gemms_needed_for_split": need_for_split,
        "feasible": None if budget is None else budget >= need_for_split,
    }
    if budget is not None and budget < need_for_split:
        ui.warn(
            "budget",
            f"a split needs >= {min_split_count} evaluated GEMMs per side "
            f"({need_for_split} in the leaf) but each leaf gets {budget} new "
            f"shapes per round: no cell can split",
        )

    prior_sel_eff_by_cell: Dict[str, Optional[float]] = {}
    prior_low_by_cell: Dict[str, Optional[int]] = {}
    prior_low_n_by_cell: Dict[str, Optional[int]] = {}
    last_decision_by_cell: Dict[str, str] = {}
    retrain_attempts_by_cell: Dict[str, int] = {}
    for pth in args.prior_decisions or []:
        if not pth.exists():
            ui.warn("prior", f"missing prior decisions: {pth}")
            continue
        try:
            pj = json.loads(pth.read_text())
        except (OSError, ValueError) as e:
            ui.warn("prior", f"cannot read {pth}: {e}")
            continue
        for r in pj.get("per_cell", []):
            cell = r["cell"]
            if r.get("sel_eff_new") is not None:
                prior_sel_eff_by_cell[cell] = r.get("sel_eff_new")
            if int(r.get("n_eval") or 0) > 0:
                prior_low_by_cell[cell] = r.get("n_low_seleff_lt_0p70")
                prior_low_n_by_cell[cell] = int(r["n_eval"])
            dec = r.get("decision")
            if dec and dec != "no_change":
                last_decision_by_cell[cell] = dec
            if dec == "retrain":
                retrain_attempts_by_cell[cell] = (
                    retrain_attempts_by_cell.get(cell, 0) + 1
                )

    def _ancestor_value(table: Dict[str, Optional[float]], label: str):
        v = table.get(label)
        cur = label
        while v is None and "#" in cur:
            cur = sc.parent_of(cur) or ""
            if not cur:
                break
            v = table.get(cur)
        return v

    results_rows: List[Dict[str, Any]] = []
    lists: Dict[str, List[str]] = {
        k: []
        for k in (
            "pass",
            "pass_carry",
            "no_change",
            "retrain",
            "train",
            "split",
            "no_data",
            "no_model",
            "not_trainable",
            "split_blocked",
        )
    }
    split_rules: List[sc.SplitRule] = []
    tree_leaves = {lf for p in split_tree for lf in sc.leaves_under(p, split_tree)}
    all_labels = sorted((set(models) | set(new_cells) | tree_leaves) - set(split_tree))
    lbl_w = min(max(max((len(x) for x in all_labels), default=24), 24), 80)

    for label in all_labels:
        gemms = held_out.get(label, [])
        n_new = len(gemms)
        n_train = len(union.get(label, []))
        cres = [results[id(g)] for g in gemms]
        summ = ev.summarize(cres) if cres else None
        base_row: Dict[str, Any] = {
            "cell": label,
            "n_new_gemms": n_new,
            "n_excluded_not_held_out": excluded_by_cell.get(label, 0),
            "n_training_gemms": n_train,
            "n_gate_gemms": len(gate.get(label, [])),
        }
        if summ is not None and pre_results:
            pre = ev.summarize([pre_results[id(g)] for g in gemms])
            base_row["sel_eff_new_pre_rebench"] = pre["sel_eff"]
            base_row["n_miss_pre_rebench"] = pre["n_miss"]
        if summ is not None:
            base_row.update(
                {
                    "n_model_served": summ["n_model_served"],
                    "n_origami_fallback": summ["n_origami_fallback"],
                    "n_miss": summ["n_miss"],
                    "model_cells": sorted({r.model_cell or "" for r in cres} - {""}),
                    "paired_sel_eff_new": (
                        summ["paired"]["model_sel_eff"]
                        if summ["paired"]["n"] > 0
                        else None
                    ),
                }
            )
        if label not in models:
            lists["no_model"].append(label)
            if n_train >= train_min:
                decision, reason = "train", "no_prior_model"
                lists["train"].append(label)
            else:
                decision = "no_model"
                reason = (
                    f"no_prior_model__too_few_training_gemms({n_train}<{train_min})"
                )
            row = dict(base_row)
            row.update(
                {
                    "decision": decision,
                    "reason": reason,
                    "n_eval": summ["n_evaluated"] if summ else 0,
                    "sel_eff_new": summ["sel_eff"] if summ else None,
                    "origami_sel_eff_new": summ["origami_sel_eff"] if summ else None,
                    "origami_n_eval": summ["n_origami"] if summ else 0,
                }
            )
            results_rows.append(row)
            if not args.quiet:
                ui.warn(
                    "eval",
                    f"{label:{lbl_w}s}  decision={decision} ({reason}, "
                    f"{n_new} held-out gemms)",
                )
            continue

        if n_new < args.min_cell_gemms:
            lists["no_data"].append(label)
            prior_dec = last_decision_by_cell.get(label)
            if prior_dec == "pass":
                decision, reason = "pass", "carried_forward_no_new_data"
                lists["pass_carry"].append(label)
            else:
                decision, reason = "no_change", "no_new_data"
                lists["no_change"].append(label)
            row = dict(base_row)
            row.update(
                {
                    "decision": decision,
                    "reason": reason,
                    "n_eval": 0,
                    "sel_eff_new": prior_sel_eff_by_cell.get(label),
                    "n_low_seleff_lt_0p70": prior_low_by_cell.get(label),
                    "carried_low_n_eval": prior_low_n_by_cell.get(label),
                    "origami_sel_eff_new": None,
                    "prior_decision": prior_dec,
                }
            )
            results_rows.append(row)
            if not args.quiet:
                tag, color = (
                    ("CARRY", ui.C.GREEN)
                    if decision == "pass"
                    else ("NOCHG", ui.C.YELLOW)
                )
                pse = _ancestor_value(prior_sel_eff_by_cell, label)
                print(
                    f"  {color}[{tag}]{ui.C.ENDC} {label:{lbl_w}s}  "
                    f"sel_eff={_train.fmt_pct(pse)}(carried)  n_eval=  0/{n_new:>3d}",
                    flush=True,
                )
            continue

        assert summ is not None
        sel_eff_new = summ["sel_eff"]
        n_eval = summ["n_evaluated"]
        per_gemm_log_ratio = np.asarray([r.log_ratio for r in cres], dtype=np.float64)
        n_low = int(np.nansum(per_gemm_log_ratio > -math.log(LOW_SEL_EFF)))
        ori_sel = summ["origami_sel_eff"] if summ["n_origami"] > 0 else None

        same_data: Optional[float] = None
        if label in prior_labels and prior_results:
            pse_same = ev.summarize([prior_results[id(g)] for g in gemms])["sel_eff"]
            if math.isfinite(pse_same):
                same_data = float(pse_same)

        prior_se = prior_sel_eff_by_cell.get(label)
        attempts = retrain_attempts_by_cell.get(label, 0)
        threshold = float(args.sel_eff_threshold)
        floor = float(args.split_floor)

        structural_octaves, structural_axis = 0.0, None
        if args.structural_split_min_octaves > 0:
            oct_by_axis = sc.cell_axis_octaves(label)
            if oct_by_axis:
                structural_axis, structural_octaves = max(
                    oct_by_axis.items(), key=lambda kv: kv[1]
                )
        struct_depth = args.structural_split_max_depth
        if struct_depth is None or struct_depth < 0:
            struct_depth = args.max_split_depth
        force_structural = (
            structural_octaves >= args.structural_split_min_octaves > 0
            and sc.split_depth(label) < min(args.max_split_depth, int(struct_depth))
        )

        if same_data is not None:
            baseline, baseline_source = same_data, "same_data_prior_model"
        else:
            baseline, baseline_source = prior_se, "cross_round_prior_decisions"
        if math.isnan(sel_eff_new):
            decision, reason = "retrain", "sel_eff_nan_on_new_data"
        elif force_structural:
            decision = "split"
            reason = (
                f"structural_width({structural_axis}={structural_octaves:.1f}oct>="
                f"{args.structural_split_min_octaves:.1f}; sel_eff={sel_eff_new:.3f})"
            )
        elif sel_eff_new >= threshold:
            decision, reason = "pass", f"above_threshold(th={threshold:.3f})"
        else:
            attempts_ok = attempts >= args.split_after_attempts
            if args.split_min_improvement <= 0:
                not_improving = True
            else:
                not_improving = (
                    baseline is not None
                    and (sel_eff_new - baseline) < args.split_min_improvement
                )
            below_floor = floor > 0 and sel_eff_new < floor
            if attempts_ok and (not_improving or below_floor):
                decision = "split"
                base = (
                    f"below_floor({floor:.3f})"
                    if below_floor
                    else f"below_threshold({threshold:.3f})"
                )
                delta = (
                    "no_baseline"
                    if baseline is None
                    else f"delta={sel_eff_new - baseline:+.3f}[{baseline_source}]"
                )
                reason = (
                    f"{base}__attempts={attempts}>={args.split_after_attempts}__{delta}"
                )
            elif not attempts_ok:
                decision = "retrain"
                reason = (
                    f"below_threshold__augment_more_first("
                    f"{attempts + 1}/{args.split_after_attempts})"
                )
            else:
                decision = "retrain"
                reason = (
                    "below_threshold__no_split_trigger("
                    + (
                        "no_baseline"
                        if baseline is None
                        else f"delta={sel_eff_new - baseline:+.3f}"
                        f">={args.split_min_improvement:.3f}[{baseline_source}]"
                    )
                    + ")"
                )

        if args.final_round and decision == "split":
            decision, reason = "retrain", f"{reason}__final_round_no_split"

        split_axis = split_threshold = None
        if decision == "split":
            pairs = [
                (g, float(lr))
                for g, lr in zip(gemms, per_gemm_log_ratio)
                if not math.isnan(float(lr))
            ]
            if not sc.split_eligible_axes(label, args.max_split_depth):
                decision, reason = "retrain", f"{reason}__unsplittable"
            elif len(pairs) < need_for_split:
                decision = "retrain"
                reason = (
                    f"{reason}__split_needs_{need_for_split}_gemms_has_{len(pairs)}"
                )
                lists["split_blocked"].append(label)
            else:
                picked = sc.choose_split_for_cell(
                    label,
                    [g for g, _ in pairs],
                    log_ratios=[lr for _, lr in pairs],
                    min_split_count=min_split_count,
                    max_split_depth=args.max_split_depth,
                )
                if picked is None:
                    decision, reason = "retrain", f"{reason}__unsplittable"
                else:
                    values = [
                        sc.axis_value(picked[0], *s) for s in union.get(label, [])
                    ]
                    n_lo = sum(1 for v in values if v <= picked[1])
                    n_hi = len(values) - n_lo
                    if min(n_lo, n_hi) < train_min:
                        decision = "retrain"
                        reason = (
                            f"{reason}__children_need_{train_min}_training_gemms"
                            f"(lo={n_lo},hi={n_hi})"
                        )
                        lists["split_blocked"].append(label)
                    else:
                        split_axis, split_threshold = picked
                        split_rules.append(
                            sc.build_split_rule(
                                cell_label=label,
                                axis=split_axis,
                                threshold=split_threshold,
                                reason=reason,
                                prior_sel_eff=prior_se,
                                this_sel_eff=sel_eff_new,
                            )
                        )

        if decision == "retrain" and n_train < train_min:
            reason = f"{reason}__too_few_training_gemms({n_train}<{train_min})"
            lists["not_trainable"].append(label)

        row = dict(base_row)
        row.update(
            {
                "decision": decision,
                "reason": reason,
                "validation_kind": "held_out_new_data",
                "n_eval": n_eval,
                "sel_eff_new": sel_eff_new,
                "pick_us_geomean": summ["pick_us_geomean"],
                "winner_us_geomean": summ["winner_us_geomean"],
                "n_low_seleff_lt_0p70": n_low,
                "sel_eff_prior_model_same_data": same_data,
                "prior_sel_eff_new": prior_se,
                "prior_retrain_attempts": attempts,
                "origami_sel_eff_new": ori_sel,
                "origami_n_eval": summ["n_origami"],
                "origami_pick_us_geomean": (
                    summ["origami_pick_us_geomean"] if ori_sel is not None else None
                ),
                "paired": summ["paired"],
                "split_axis": split_axis,
                "split_threshold": split_threshold,
            }
        )
        results_rows.append(row)
        lists[decision].append(label)

        if not args.quiet:
            if same_data is not None:
                pse, where = same_data, "same GEMMs"
            else:
                pse, where = _ancestor_value(prior_sel_eff_by_cell, label), "other"
            prior_str = (
                f"  prior={_train.fmt_pct(pse)}"
                f"(d={(sel_eff_new - pse) * 100:+7.2f}, {where})"
                if pse is not None and not math.isnan(sel_eff_new)
                else "  prior=" + "--".rjust(18)
            )
            ori_str = ""
            if args.origami_reference and ori_sel is not None:
                ori_str = (
                    f"  [reference] origami={_train.fmt_pct(ori_sel)} vs "
                    f"{_train.fmt_pct(row['paired_sel_eff_new'])} on "
                    f"{summ['n_origami']} GEMMs"
                )
            split_str = (
                f"  SPLIT[{split_axis}={split_threshold}]  reason={reason}"
                if decision == "split"
                else ""
            )
            color = {"pass": ui.C.GREEN, "split": ui.C.RED}.get(decision, ui.C.YELLOW)
            tag = {"pass": "PASS ", "split": "SPLIT"}.get(decision, "RETRN")
            print(
                f"  {color}[{tag}]{ui.C.ENDC} {label:{lbl_w}s}  "
                f"sel_eff={_train.fmt_pct(sel_eff_new)}{prior_str}  "
                f"n_eval={n_eval:>3d}/{n_new:>3d}  miss={summ['n_miss']:>3d}  "
                f"low<70%={n_low:>3d}{ori_str}{split_str}",
                flush=True,
            )

    evaluated = [
        r
        for r in results_rows
        if (r.get("n_eval", 0) or 0) > 0
        and r.get("sel_eff_new") is not None
        and r.get("decision") not in ("no_change",)
        and r.get("reason") != "carried_forward_no_new_data"
    ]

    def _wlog(key: str, weight: str = "n_eval") -> Optional[float]:
        v = _train.wlog_mean([(r.get(key), int(r.get(weight) or 0)) for r in evaluated])
        return None if math.isnan(v) else v

    overall = ev.summarize(list(results.values()), bootstrap=1000) if results else None
    by_context = _summaries_by_context(held_list, results)
    context_holdout = None
    if args.context_holdout_dir is not None:
        context_holdout = _context_holdout(
            args,
            bundle,
            library,
            split_tree,
            constants,
            hardware,
            tw,
            identity,
            repeats.config_aggregation(cfg),
            blocklist,
        )
    aggregate = {
        "n_cells_evaluated": len(evaluated),
        "n_gemms_evaluated": sum(int(r["n_eval"]) for r in evaluated),
        "n_gemms_excluded_not_held_out": n_excluded,
        "n_gemms_listed": n_listed,
        "n_low_seleff_lt_0p70_total": sum(
            int(r.get("n_low_seleff_lt_0p70") or 0) for r in evaluated
        ),
        "n_miss_total": sum(int(r.get("n_miss") or 0) for r in evaluated),
        "pick_measurement": (
            ev.REBENCH_PICK_MEASUREMENT if bench is not None else args.pick_measurement
        ),
        "winner_us_geomean": _wlog("winner_us_geomean"),
        "global_sel_eff_new": _wlog("sel_eff_new"),
        "global_model_pick_us_geomean": _wlog("pick_us_geomean"),
        "n_gemms_paired": sum(int(r.get("origami_n_eval") or 0) for r in evaluated),
        "global_paired_sel_eff_new": _wlog("paired_sel_eff_new", "origami_n_eval"),
        "global_origami_sel_eff_new": _wlog("origami_sel_eff_new", "origami_n_eval"),
        "global_origami_pick_us_geomean": _wlog(
            "origami_pick_us_geomean", "origami_n_eval"
        ),
        "all_held_out": overall,
    }
    if "held_out" in measured:
        rec = measured["held_out"].record or {}
        aggregate["pre_rebench"] = dict(
            mr.pre_rebench(
                ev.summarize(list(pre_results.values())), args.pick_measurement
            ),
            global_sel_eff_new=_wlog("sel_eff_new_pre_rebench"),
        )
        for key in ("n_rebenched", "n_rebench_failed", "n_context_skipped"):
            aggregate[key] = rec.get(key)
        aggregate["rebench"] = {
            tag: mr.record_counts(m.record) for tag, m in measured.items()
        }
        (args.out_dir / mr.RECORD_FILE).write_text(
            json.dumps(
                _train.jsonable(
                    {
                        tag: mr.named_record(m.record, identity)
                        for tag, m in measured.items()
                    }
                ),
                indent=2,
            )
            + "\n"
        )
    if len(by_context) > 1:
        aggregate["by_context"] = by_context
    if context_holdout is not None:
        aggregate["context_holdout"] = context_holdout

    decisions_json = {
        "schema_version": "pipeline-v3-2026-10-03",
        "round_idx": int(args.round_idx),
        "models_dir": str(args.models_dir),
        "enriched_csv_dir": str(args.enriched_csv_dir),
        "validation_kind": "held_out_new_data",
        "weight_dtype": args.weight_dtype,
        "hardware": asdict(hardware),
        "sel_eff_threshold": args.sel_eff_threshold,
        "split_floor": args.split_floor,
        "split_min_improvement": args.split_min_improvement,
        "split_after_attempts": args.split_after_attempts,
        "train_min_cell_gemms": train_min,
        "split_budget_check": budget_check,
        "gate": gate_record,
        "elapsed_s": time.time() - t0,
        "summary": {
            "pass": len(lists["pass"]),
            "pass_carried_forward": len(lists["pass_carry"]),
            "no_change": len(lists["no_change"]),
            "retrain": len(lists["retrain"]),
            "train": len(lists["train"]),
            "split": len(lists["split"]),
            "split_blocked": len(lists["split_blocked"]),
            "no_model": len(lists["no_model"]),
            "not_trainable": len(lists["not_trainable"]),
            "no_data": len(lists["no_data"]),
            "fallback_parents": len(fallback_parents),
            "total_evaluated": len(all_labels),
        },
        "aggregate": aggregate,
        "cells_pass": lists["pass"],
        "cells_pass_carried_forward": lists["pass_carry"],
        "cells_no_change": lists["no_change"],
        "cells_retrain": lists["retrain"],
        "cells_train": lists["train"],
        "cells_split": lists["split"],
        "cells_split_blocked": lists["split_blocked"],
        "cells_no_model": lists["no_model"],
        "cells_not_trainable": lists["not_trainable"],
        "cells_no_data": lists["no_data"],
        "cells_fallback_parents": fallback_parents,
        "per_cell": results_rows,
    }
    out_path = args.out_dir / "decisions.json"
    out_path.write_text(json.dumps(_train.jsonable(decisions_json), indent=2))

    blocked = set(lists["not_trainable"])
    retrain_targets = [c for c in lists["retrain"] if c not in blocked]
    retrain_targets += lists["train"]
    for rule in split_rules:
        retrain_targets += [rule.lo_label, rule.hi_label]
    (args.out_dir / "retrain_cells.txt").write_text(",".join(retrain_targets))
    sc.write_splits_json(args.out_dir / "splits.json", args.round_idx, split_rules)
    if args.audit_dir is not None:
        auditlib.run_for_stage(
            args.out_dir / auditlib.RESULT_FILE,
            mlrec.write_model(bundle, None, args.arch, constants, args.weight_dtype),
            auditlib.Spec(
                audit_dir=args.audit_dir,
                library_dir=args.library_dir,
                library_stem=args.library_stem
                or str((cfg.get("hipblaslt") or {}).get("library_stem") or ""),
                hardware_device=args.hardware_device,
                pick_measurement=args.pick_measurement,
                target=args.audit_target,
                bootstrap=args.audit_bootstrap,
                trained_rounds=auditlib.trained_rounds(
                    args, auditlib.rounds_until(round_dir, inclusive=False)
                ),
                official=args.audit_official,
            ),
            cfg,
            hardware=hardware,
            tw=tw,
            what="stage04b: the model entering the round",
            source=str(models_pt),
            quiet=args.quiet,
        )

    if not args.quiet:
        ui.banner("Done", ui.C.GREEN)
        print(
            f"  {ui.C.BOLD}{'global_sel_eff_new (deployed)':32s}:{ui.C.ENDC} "
            f"{_train.fmt_pct(aggregate['global_sel_eff_new'])} over "
            f"{aggregate['n_gemms_evaluated']} GEMMs, "
            f"{aggregate['n_miss_total']} misses",
            flush=True,
        )
        if "pre_rebench" in aggregate:
            pre = aggregate["pre_rebench"]
            print(
                f"  {ui.C.BOLD}{'misses re-benched':32s}:{ui.C.ENDC} "
                f"{aggregate['n_rebenched']} measured, "
                f"{aggregate['n_rebench_failed']} failed, "
                f"{aggregate['n_context_skipped']} context GEMMs skipped; "
                f"under {pre['pick_measurement']} "
                f"{_train.fmt_pct(pre['global_sel_eff_new'])}",
                flush=True,
            )
        if args.origami_reference and overall and overall["paired"]["n"]:
            p = overall["paired"]
            print(
                f"  {ui.C.BOLD}{'[reference] Origami, same GEMMs':32s}:{ui.C.ENDC} "
                f"n={p['n']}  deployed={_train.fmt_pct(p['model_sel_eff'])}  "
                f"origami={_train.fmt_pct(p['origami_sel_eff'])}  "
                f"wins/ties/losses={p['wins']}/{p['ties']}/{p['losses']}",
                flush=True,
            )
        for k, what in (
            ("pass", "met the threshold on held-out GEMMs"),
            ("pass_carry", "passed before; too few held-out GEMMs now"),
            ("retrain", "below the threshold"),
            ("train", "no model of its own yet; enough GEMMs to train now"),
            ("split", "split this round"),
            ("split_blocked", "would split but too few GEMMs for two children"),
            ("no_change", "too few held-out GEMMs, no earlier pass"),
            ("no_model", "no model of its own in the prior bundle"),
            ("not_trainable", "to retrain, but too few GEMMs for stage05"),
        ):
            ui.ok("done", f"{k:14s}: {len(lists[k]):3d} cells ({what})")
        if lists["split_blocked"]:
            ui.warn(
                "split",
                f"split blocked by GEMM count: {', '.join(lists['split_blocked'])}",
            )
        if fallback_parents:
            ui.info(
                "done",
                f"split parents serving leaves without a model: "
                f"{', '.join(fallback_parents)}",
            )
        ui.ok("done", f"decisions.json -> {out_path}")
        if split_rules:
            for line in sc.format_split_tree(
                {r.cell: r for r in split_rules}
            ).splitlines():
                print(f"        {line}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
