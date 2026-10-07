#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage01b_plan_benchmarks -- decide which kernels stage03 benches for each
shape, without the library heuristic.

Reads stage01's shapes.yaml and the config's `planner:` section, and writes
the bench yaml stage02 / stage03 run plus the plan stage02 and stage04 read.
The kernel lists come from `lib.planner`; this stage chooses the mode and the
ranker:

  exhaustive  every shape benches every pool and bank kernel, in library
              order (an explicit list; `algo_method: 1`, every kernel of the
              problem type, when they exceed the bench's list limit).
  planned     every shape benches an explicit list (`solution_index_list`,
              the planner's order). The ranker is
                * the model of --model-file (a later round's current model,
                  or `planner.model_file`), else
                * a seed bootstrap: `planner.seed_per_cell` shapes per cell
                  are benched exhaustively first (stage02 / stage03 / stage04
                  run on them as given by --seed-commands, into seed/), and
                  `planner.ranker` ranks from their results: `neighbors`
                  (nearest seed shapes) or `stage05` (a model trained on the
                  seeds, with `neighbors` for shapes it scores nothing for).
  auto        planned when the pool and bank hold more than
              `planner.exhaustive_max_kernels` kernels, else exhaustive.

A cell whose shapes are all seeds needs no planning; when every shape of the
round would be a seed the round is benched exhaustively instead.

Every bench line gets `skip_screen_prepass: true`, so the skip screen warms
up every candidate before timing any and its result does not depend on the
bench order.

Outputs in --out-dir:
  shapes.yaml        bench lines of the shapes still to bench (seed shapes
                     are left out: their logs are in seed/stage03/logs)
  plan.json          every shape: role (seed / planned / exhaustive), kernel
                     list (null: exhaustive), probe kernels for stage02, the
                     number of feasible pool kernels, list sources
  plan_summary.json  counts per role, list lengths, coverage additions,
                     truncations, ranker
  seed/              the seed bootstrap: shapes.yaml, plan.json and one
                     directory and log per sub-stage
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = THIS_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import yaml  # noqa: E402

from lib import hardware as hwlib  # noqa: E402
from lib import ui  # noqa: E402
from lib.bench_yaml import (  # noqa: E402
    drop_line_knob,
    format_solution_list,
    read_bench_lines,
    set_line_knob,
    shape_of_line,
)
from lib.bank_pool import config_pool  # noqa: E402
from lib.feasibility import soft_rules  # noqa: E402
from lib.fs import atomic_write_text, write_json  # noqa: E402
from lib.planner import (  # noqa: E402
    MAX_BENCH_LIST,
    PLAN_FILE,
    Catalog,
    Feasibility,
    ModelRanker,
    NeighborsRanker,
    PlanParams,
    ProblemType,
    Ranker,
    ShapeKey,
    ShapePlan,
    build_plan,
    plan_document,
    seed_latencies,
    select_seeds,
)

MODES = ("exhaustive", "planned", "auto")
RANKERS = ("neighbors", "stage05")
V3_RULE_KEYS = ("small_batched_rule", "dot2_rule", "non_temporal_rule")
SEED_STAGES = ("stage02", "stage03", "stage04")
# stage05 exit code when it wrote models but left some cells untrained.
STAGE05_CELLS_NOT_TRAINED = 3
DEFAULTS: Dict[str, Any] = {
    "mode": "exhaustive",
    "exhaustive_max_kernels": 512,
    "seed_per_cell": 8,
    "ranker": "neighbors",
    "neighbors_k": 4,
    "model_file": None,
    "bank_library_stems": [],
    "bank": "none",
    **PlanParams().as_dict(),
}


def planner_settings(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (cfg.get("planner") or {}).items() if v is not None})
    return out


def plan_params(settings: Mapping[str, Any]) -> PlanParams:
    return PlanParams(**{k: int(settings[k]) for k in PlanParams().as_dict()})


def config_rules(cfg: Mapping[str, Any]) -> Dict[str, bool]:
    """Soft rules the run's models serve with, from the `train` section: v3
    applies only the rules switched on there (the non-temporal rule only with
    `non_temporal_rule: true`), v2 the non-temporal rule unless it is false."""
    train = cfg.get("train") or {}
    if str(train.get("catalog") or "v2") == "v3":
        serving = {k: bool(train.get(k)) for k in V3_RULE_KEYS}
        return soft_rules(serving, 3)
    nt = train.get("non_temporal_rule")
    return soft_rules({"non_temporal_rule": True if nt is None else bool(nt)}, 2)


# ── bench lines ──────────────────────────────────────────────────────────────


def exhaustive_list(catalog: Catalog) -> Optional[List[int]]:
    """The list an exhaustive shape benches: every pool and bank kernel, in
    library order; None when the bench cannot take that many."""
    ids = catalog.pool_ids + catalog.bank_ids
    return ids if len(ids) <= MAX_BENCH_LIST else None


def render_planned_line(
    line: str, plan: ShapePlan, everything: Optional[Sequence[int]] = None
) -> str:
    """`line` set up to bench `plan`: its kernel list; for an exhaustive
    shape `everything`, else every kernel the library has for the problem
    type (`algo_method: 1`, which includes libraries outside the pool and
    bank). Always with the order-independent skip screen."""
    kernels = plan.kernels if plan.kernels is not None else everything
    if kernels is None:
        line = drop_line_knob(line, "solution_index_list")
        line = set_line_knob(line, "algo_method", "1")
        line = set_line_knob(line, "requested_solution_num", "-1")
    else:
        line = set_line_knob(line, "solution_index_list", format_solution_list(kernels))
    return set_line_knob(line, "skip_screen_prepass", "true")


def write_lines(
    path: Path,
    lines: Sequence[str],
    plans: Mapping[ShapeKey, ShapePlan],
    everything: Optional[Sequence[int]],
) -> int:
    out = []
    for line in lines:
        shape = shape_of_line(line)
        if shape is None or shape not in plans:
            continue
        out.append(render_planned_line(line, plans[shape], everything))
    atomic_write_text(path, "".join(out))
    return len(out)


# ── seed bootstrap ───────────────────────────────────────────────────────────


def run_sub_stage(name: str, cmd: Sequence[str], log: Path, quiet: bool) -> int:
    if not quiet:
        ui.info("seed", f"{name}: {log}")
    t0 = time.time()
    with log.open("w") as f:
        rc = subprocess.run(
            [sys.executable, "-u", *[str(c) for c in cmd]],
            stdout=f,
            stderr=subprocess.STDOUT,
        ).returncode
    if not quiet:
        say = ui.ok if rc == 0 else ui.warn
        say("seed", f"{name}: rc={rc} ({ui.fmt_dur(time.time() - t0)})")
    return rc


def model_bytes(path: Path, arch: str, weight_dtype: str) -> bytes:
    """MLREC bytes of a model file: a stage05 `models.pt` bundle is written at
    `weight_dtype`, anything else is read as MLREC."""
    if path.suffix != ".pt":
        return path.read_bytes()
    import torch

    from lib import mlrec

    bundle = torch.load(path, map_location="cpu", weights_only=True)
    return mlrec.write_model(
        bundle, None, arch, hwlib.arch_constants(arch), weight_dtype
    )


# ── main ─────────────────────────────────────────────────────────────────────


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="stage01b -- plan the kernels benched per shape (no "
        "library heuristic)."
    )
    ap.add_argument("--in-dir", type=Path, required=True, help="stage01 dir.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--config-yaml", type=Path, required=True)
    ap.add_argument(
        "--dat-dir", type=Path, required=True, help="Tensile library directory."
    )
    ap.add_argument("--library-stem", required=True, help="the pool's library.")
    ap.add_argument("--arch", default=None, help="needed with a models.pt model.")
    ap.add_argument("--weight-dtype", default="bf16")
    ap.add_argument(
        "--hardware-device",
        type=int,
        default=0,
        help="device whose LDS size and CU count the gates and models use, "
        "unless the config's `hardware:` block sets them.",
    )
    ap.add_argument(
        "--model-file",
        type=Path,
        default=None,
        help="rank with this model (models.pt or MLREC) instead of a seed "
        "bootstrap.",
    )
    ap.add_argument(
        "--contexts",
        default=None,
        help="JSON list of execution contexts ({name, cu_budget, schedule}) to "
        "re-plan with --model-file: contexts/<name>/shapes.yaml and plan.json",
    )
    ap.add_argument(
        "--seed-commands",
        default=None,
        help="JSON object: stage name (stage02, stage03, stage04, stage05) -> "
        "argv (script path first) of the seed bootstrap's sub-stages; "
        "stage02 may be null (adaptive timing). Their paths must point "
        "into <out-dir>/seed, where the seed shapes and plan are written.",
    )
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    t0 = time.time()
    with args.config_yaml.open() as f:
        cfg = yaml.safe_load(f) or {}
    st = planner_settings(cfg)
    if st["mode"] not in MODES:
        ui.err("cfg", f"planner.mode must be one of {MODES}, got {st['mode']!r}")
        return 1
    if st["ranker"] not in RANKERS:
        ui.err("cfg", f"planner.ranker must be one of {RANKERS}")
        return 1
    try:
        params = plan_params(st)
    except (TypeError, ValueError) as e:
        ui.err("cfg", f"planner: {e}")
        return 1
    in_yaml = args.in_dir / "shapes.yaml"
    if not in_yaml.is_file():
        ui.err("err", f"missing {in_yaml}")
        return 1
    lines = read_bench_lines(in_yaml)
    shapes = [s for s in (shape_of_line(x) for x in lines) if s is not None]
    if not shapes:
        ui.err("err", f"no shape lines in {in_yaml}")
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        library = config_pool(
            {**cfg, "planner": {**st, "enabled": True}},
            args.dat_dir,
            args.library_stem,
            args.hardware_device,
        )
        hardware = hwlib.device_hardware(cfg, args.hardware_device)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as e:
        ui.err("load", str(e))
        return 1
    catalog = Catalog.build(library.table, library.bank.kernels)
    ptype = ProblemType.from_config(cfg["hipblaslt"])
    feas = Feasibility(catalog, ptype, hardware.lds_bytes, rules=config_rules(cfg))

    mode = st["mode"]
    if mode == "auto":
        n_cat = len(catalog.kernels)
        mode = "planned" if n_cat > int(st["exhaustive_max_kernels"]) else "exhaustive"
    if not args.quiet:
        ui.banner("stage01b plan benchmarks", ui.C.HEAD)
        ui.info(
            "cfg", f"shapes       : {len(shapes):,} ({len(set(shapes)):,} distinct)"
        )
        ui.info(
            "cfg",
            f"catalog      : {len(catalog.pool_ids)} pool + "
            f"{len(catalog.bank_ids)} bank kernels, {len(catalog.families)} families",
        )
        ui.info("cfg", f"mode         : {mode} (configured {st['mode']})")
        ui.info("cfg", f"params       : {params.as_dict()}")

    ranker: Optional[Ranker] = None
    seeds: List[ShapeKey] = []
    ranker_desc = "none"
    if mode == "planned":
        seeds = []
        if args.model_file is not None:
            try:
                data = model_bytes(args.model_file, args.arch or "", args.weight_dtype)
                ranker = ModelRanker(data, catalog, ptype, hardware)
            except Exception as e:
                ui.err("model", f"cannot rank with {args.model_file}: {e}")
                return 1
            ranker_desc = f"model {args.model_file}"
            feas = Feasibility(catalog, ptype, hardware.lds_bytes, rules=ranker.rules)
        else:
            seeds = select_seeds(shapes, int(st["seed_per_cell"]), params.seed)
            if len(seeds) == len(set(shapes)):
                ui.warn(
                    "plan",
                    "every shape would be a seed; benching the round exhaustively",
                )
                mode, seeds = "exhaustive", []
            else:
                rc, ranker = _seed_bootstrap(
                    args, st, params, lines, seeds, feas, catalog, ptype, hardware
                )
                if rc:
                    return rc
                ranker_desc = f"seed bootstrap ({st['ranker']})"
                if isinstance(ranker, ModelRanker):
                    feas = Feasibility(
                        catalog, ptype, hardware.lds_bytes, rules=ranker.rules
                    )

    def _progress(i: int, n: int) -> None:
        if not args.quiet and (i == n or i % 500 == 0):
            ui.grey("plan", f"{i:,}/{n:,} shapes")

    plans, added = build_plan(
        shapes,
        feas,
        params,
        ranker=ranker,
        seeds=seeds,
        exhaustive=mode == "exhaustive",
        progress=_progress,
    )
    by_shape = {sp.shape: sp for sp in plans}
    to_bench = {s: sp for s, sp in by_shape.items() if sp.role != "seed"}
    n_lines = write_lines(
        args.out_dir / "shapes.yaml", lines, to_bench, exhaustive_list(catalog)
    )
    meta = {
        "mode": mode,
        "configured_mode": st["mode"],
        "ranker": ranker_desc,
        "library_stem": args.library_stem,
        "bank_library_stems": list(st["bank_library_stems"] or []),
        "bank": st["bank"],
        "bank_stems_used": library.bank.stems,
        "params": params.as_dict(),
        "rules": feas.rules,
    }
    write_json(args.out_dir / PLAN_FILE, plan_document(plans, meta))
    lens = [len(sp.kernels) for sp in plans if sp.kernels is not None]
    summary = {
        **meta,
        "n_shapes": len(plans),
        "n_lines": n_lines,
        "roles": {
            r: sum(1 for sp in plans if sp.role == r)
            for r in ("seed", "planned", "exhaustive")
        },
        "list_len": {
            "min": min(lens) if lens else None,
            "median": statistics.median(lens) if lens else None,
            "max": max(lens) if lens else None,
        },
        "feasible_pool_median": (
            statistics.median(sp.n_feasible for sp in plans) if plans else None
        ),
        "planned_kernel_fraction": (
            sum(lens)
            / max(1, sum(sp.n_feasible for sp in plans if sp.kernels is not None))
            if lens
            else None
        ),
        "coverage_added": added,
        "n_truncated_shapes": sum(1 for sp in plans if sp.n_truncated),
        "elapsed_s": round(time.time() - t0, 1),
    }
    write_json(args.out_dir / "plan_summary.json", summary)
    if not args.quiet:
        ui.ok("plan", f"roles        : {summary['roles']}")
        if lens:
            ui.ok("plan", f"list length  : {summary['list_len']}")
        if summary["n_truncated_shapes"]:
            ui.warn(
                "plan",
                f"{summary['n_truncated_shapes']} shape(s) hit the {params.max_list}-kernel "
                f"list limit",
            )
        ui.ok("write", f"shapes.yaml  : {n_lines:,} lines")
    if args.contexts:
        if args.model_file is None or not isinstance(ranker, ModelRanker):
            ui.warn("contexts", "no model to re-plan with; contexts keep these lists")
        else:
            rc = _replan_contexts(
                args, data, lines, shapes, feas, params, catalog, ptype, hardware
            )
            if rc:
                return rc
    if not args.quiet:
        ui.banner("Done", ui.C.GREEN)
    return 0


def _replan_contexts(
    args, data, lines, shapes, feas, params, catalog, ptype, hardware
) -> int:
    """Plan every shape again with the model ranking in each context of
    --contexts; writes contexts/<name>/{shapes.yaml, plan.json}."""
    from lib import features as fs

    try:
        wanted = json.loads(args.contexts)
        contexts = [
            (
                str(c["name"]),
                fs.ExecutionContext(
                    cu_budget=int(c.get("cu_budget", 0)),
                    schedule=str(c.get("schedule", "default")),
                ),
            )
            for c in wanted
        ]
    except (ValueError, KeyError, TypeError) as e:
        ui.err("contexts", f"--contexts: {e}")
        return 1
    for name, ctx in contexts:
        ranker = ModelRanker(data, catalog, ptype, hardware, context=ctx)
        plans, _ = build_plan(shapes, feas, params, ranker=ranker)
        out = args.out_dir / "contexts" / name
        out.mkdir(parents=True, exist_ok=True)
        by_shape = {sp.shape: sp for sp in plans}
        n = write_lines(out / "shapes.yaml", lines, by_shape, exhaustive_list(catalog))
        meta = {
            "mode": "planned",
            "ranker": f"model {args.model_file} in context {name}",
            "context": {"cu_budget": ctx.cu_budget, "schedule": ctx.schedule},
            "library_stem": args.library_stem,
            "params": params.as_dict(),
        }
        write_json(out / PLAN_FILE, plan_document(plans, meta))
        if not args.quiet:
            ui.ok("contexts", f"{name}: {n:,} re-planned lines -> {out}")
    return 0


def _seed_bootstrap(
    args: argparse.Namespace,
    st: Mapping[str, Any],
    params: PlanParams,
    lines: Sequence[str],
    seeds: Sequence[ShapeKey],
    feas: Feasibility,
    catalog: Catalog,
    ptype: ProblemType,
    hardware: Any,
):
    """Bench `seeds` exhaustively and build the ranker; (rc, ranker)."""
    if not args.seed_commands:
        ui.err("seed", "the seed bootstrap needs --seed-commands (or --model-file)")
        return 1, None
    try:
        cmds = json.loads(args.seed_commands)
    except ValueError as e:
        ui.err("seed", f"--seed-commands is not JSON: {e}")
        return 1, None
    missing = [k for k in ("stage03", "stage04") if not cmds.get(k)]
    if st["ranker"] == "stage05" and not cmds.get("stage05"):
        missing.append("stage05")
    if missing:
        ui.err("seed", f"--seed-commands lacks {missing}")
        return 1, None
    seed_dir = args.out_dir / "seed"
    seed_dir.mkdir(parents=True, exist_ok=True)
    seed_plans, _ = build_plan(seeds, feas, params, ranker=None, exhaustive=True)
    for sp in seed_plans:
        sp.role = "seed"
    by_shape = {sp.shape: sp for sp in seed_plans}
    n = write_lines(seed_dir / "shapes.yaml", lines, by_shape, exhaustive_list(catalog))
    write_json(
        seed_dir / PLAN_FILE,
        plan_document(seed_plans, {"mode": "exhaustive", "ranker": "none"}),
    )
    if not args.quiet:
        ui.info("seed", f"{len(seeds):,} seed shapes ({n:,} lines) -> {seed_dir}")
    for key in SEED_STAGES:
        cmd = cmds.get(key)
        if not cmd:
            continue
        (seed_dir / key).mkdir(parents=True, exist_ok=True)
        rc = run_sub_stage(key, cmd, seed_dir / f"{key}.log", args.quiet)
        if rc:
            ui.err(
                "seed", f"seed {key} failed (rc={rc}); see {seed_dir / (key + '.log')}"
            )
            return 2, None
    lat = seed_latencies(str(seed_dir / "stage04"))
    if not lat:
        ui.err("seed", "the seed bench produced no measurements")
        return 2, None
    ranker: Ranker = NeighborsRanker(lat, k=int(st["neighbors_k"]))
    if st["ranker"] == "stage05":
        (seed_dir / "stage05").mkdir(parents=True, exist_ok=True)
        rc = run_sub_stage(
            "stage05", cmds["stage05"], seed_dir / "stage05.log", args.quiet
        )
        models = seed_dir / "stage05" / "models.pt"
        if rc not in (0, STAGE05_CELLS_NOT_TRAINED) or not models.is_file():
            ui.err("seed", f"seed stage05 failed (rc={rc})")
            return 2, None
        try:
            data = model_bytes(models, args.arch or "", args.weight_dtype)
            ranker = ModelRanker(data, catalog, ptype, hardware, fallback=ranker)
        except Exception as e:
            ui.err("seed", f"cannot rank with the seed model: {e}")
            return 2, None
    return 0, ranker


if __name__ == "__main__":
    sys.exit(main())
