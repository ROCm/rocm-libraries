#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage02_plan_benchmarks -- decide which kernels stage03 benches for each
shape, without the library heuristic.

Reads stage01's shapes.yaml and the config's `planner:` section, and writes
the bench yaml stage03 runs plus the plan stage04 reads. The kernel lists
come from `lib.planner`; this stage chooses the mode and the ranker:

  exhaustive  every shape benches every pool and bank kernel (an explicit
              list; `algo_method: 1`, every kernel of the problem type in
              the library's order, when they exceed the bench's list limit).
  planned     every shape benches an explicit list (`solution_index_list`).
              The ranker is
                * the model of --model-file (a later round's current model,
                  or `planner.model_file`), else
                * a seed bootstrap: `planner.seed_per_cell` shapes per cell
                  are benched exhaustively first (stage03 and stage04 run on
                  them as given by --seed-commands, into seed/), and
                  `planner.ranker` ranks from their results: `neighbors`
                  (nearest seed shapes) or `stage05` (a model trained on the
                  seeds, with `neighbors` for shapes it scores nothing for).
  auto        planned when the pool and bank hold more than
              `planner.exhaustive_max_kernels` kernels, else exhaustive.

Every mode benches an alias group (lib/kernel_identity.py: kernels whose
names differ only in the temporal hint of an operand the tensor data mover
loads) as its first member in pool order (`lib.planner.Catalog`).

Kernels are gated, and a model ranks them, with the soft rules of
`planner.rules`: `config` (default) the rules the run's next models serve
with (`train` section; a model serving with other ones ranks under these),
`model` the ranking model's own serving rules.

When the run's models serve the family-wise non-temporal rule
(`train.non_temporal_family_rule`), every planned list of a shape in the
rule's region, the round's and every context's, also lists each kernel only
that rule admits (`lib.planner.add_family_rule_members`), beyond
`planner.max_list` up to the bench's list limit (members past it are
counted apart from the truncations): models with the rule need those
kernels measured, and the rankers that plan the next round cannot rank
kernels no model measured.

A cell whose shapes are all seeds needs no planning; when every shape of the
round would be a seed the round is benched exhaustively instead.

The lists a seed bootstrap ranks take `planner.top_k_round0` as their top_k
when it is set: a later round ranks with a model, whose lists need not be as
long.

Those lists are scored on the seeds before they are written
(`lib.planner.seed_recall`): every seed of a planned cell is ranked from the
other seeds of its cell and given the list the planner would build from
that ranking, which its own exhaustive bench then scores. The stage warns
when the lists' ceiling, the geometric mean of the fastest kernel's latency
over the fastest listed one's, is below `planner.recall_min_ceiling`, and
names the top_k that reaches it. With `planner.ranker: stage05` the seeds
score the neighbors ranker the model falls back to: the model trained on
every seed.

Every bench line gets `skip_screen_prepass: true`, so the skip screen warms
up every candidate before timing any and its result does not depend on the
bench order. An explicit list is benched in a shuffle seeded with
`bench.order_seed` and the shape (`bench.kernel_order: shuffled`, the
default; lib/bench_yaml.py says why), not in the planner's or the library's
order; plan.json keeps each list in the planner's order and records the
seed, so the bench order can be derived again (`lib.planner.shuffled`).
With `bench.sacrificial_kernel` (the default) a line with a skip ratio
lists its first kernel twice: the copy at position 0 absorbs the screen's
first cold call, which reads too slow, and readers of the logs set its row
aside (lib/bench_log.py).

Outputs in --out-dir:
  shapes.yaml        bench lines of the shapes still to bench (seed shapes
                     are left out: their logs are in seed/stage03/logs)
  plan.json          every shape: role (seed / planned / exhaustive), kernel
                     list (null: exhaustive), the number of feasible pool
                     kernels, list sources; the kernel ids of the pool and
                     bank with the digest of their library files and the
                     aliases left unbenched (lib/kernel_identity.py), so
                     that later stages read its solution indices as the
                     kernels they named; and the
                     `kernel_order`, `order_seed` and `sacrificial_kernel`
                     of the bench lines
  plan_summary.json  counts per role, list lengths, coverage additions,
                     truncations, family-rule members over the bench's
                     limit, ranker; `seed_recall`, the seed bootstrap's
                     scores (null for a round without one)
  seed/              the seed bootstrap: shapes.yaml, plan.json and one
                     directory and log per sub-stage

The stage exits 2 when the seed bootstrap failed in a way another attempt
may cure (the seed bench exited 2, or a sub-stage was killed by SIGHUP,
SIGINT, SIGKILL or SIGTERM), and 1 on any other failure.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = THIS_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import yaml  # noqa: E402

from lib import hardware as hwlib  # noqa: E402
from lib import kernel_identity as ki  # noqa: E402
from lib import procs  # noqa: E402
from lib import ui  # noqa: E402
from lib.bench_yaml import (  # noqa: E402
    drop_line_knob,
    format_solution_list,
    kernel_order_seed,
    line_skip_ratio,
    read_bench_lines,
    sacrificial_kernel,
    set_line_knob,
    shape_of_line,
)
from lib.bank_pool import config_identity, config_pool  # noqa: E402
from lib.feasibility import soft_rules  # noqa: E402
from lib.fs import atomic_write_text, write_json  # noqa: E402
from lib.planner import (  # noqa: E402
    MAX_BENCH_LIST,
    NEAR_TOLERANCE,
    PLAN_FILE,
    RECALL_MIN_CEILING,
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
    seed_recall,
    select_seeds,
    shape_cell,
    shuffled,
)

MODES = ("exhaustive", "planned", "auto")
RANKERS = ("neighbors", "stage05")
RULE_SOURCES = ("config", "model")
V3_RULE_KEYS = (
    "small_batched_rule",
    "dot2_rule",
    "non_temporal_rule",
    "non_temporal_family_rule",
)
SEED_STAGES = ("stage03", "stage04")
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
    "rules": "config",
    **PlanParams().as_dict(),
    "top_k_round0": None,
    "recall_min_ceiling": RECALL_MIN_CEILING,
}


def planner_settings(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (cfg.get("planner") or {}).items() if v is not None})
    return out


def plan_params(settings: Mapping[str, Any]) -> PlanParams:
    return PlanParams(**{k: int(settings[k]) for k in PlanParams().as_dict()})


def seed_plan_params(settings: Mapping[str, Any], params: PlanParams) -> PlanParams:
    """`params` of the lists the seed bootstrap ranks: `top_k_round0`, when
    set, as their top_k."""
    top_k = settings.get("top_k_round0")
    if top_k is None:
        return params
    if int(top_k) < 0:
        raise ValueError(f"top_k_round0 must be at least 0, got {top_k}")
    return replace(params, top_k=int(top_k))


def recall_min_ceiling(settings: Mapping[str, Any]) -> float:
    value = float(settings["recall_min_ceiling"])
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"recall_min_ceiling must be in [0, 1], got {value:g}")
    return value


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
    line: str,
    plan: ShapePlan,
    everything: Optional[Sequence[int]] = None,
    order_seed: Optional[int] = None,
    sacrificial: bool = False,
) -> str:
    """`line` set up to bench `plan`: its kernel list; for an exhaustive
    shape `everything`, else every kernel the library has for the problem
    type (`algo_method: 1`, which includes libraries outside the pool and
    bank). A list is shuffled with `order_seed` and the shape
    (`planner.shuffled`) unless it is None, and with `sacrificial` a
    screened line (a skip ratio above 0) lists its first kernel twice, as
    long as the bench takes one more. Always with the order-independent skip
    screen."""
    kernels = plan.kernels if plan.kernels is not None else everything
    if kernels is None:
        line = drop_line_knob(line, "solution_index_list")
        line = set_line_knob(line, "algo_method", "1")
        line = set_line_knob(line, "requested_solution_num", "-1")
    else:
        if order_seed is not None:
            kernels = shuffled(kernels, order_seed, *plan.shape)
        if (
            sacrificial
            and kernels
            and len(kernels) < MAX_BENCH_LIST
            and line_skip_ratio(line) > 0
        ):
            kernels = [kernels[0], *kernels]
        line = set_line_knob(line, "solution_index_list", format_solution_list(kernels))
    return set_line_knob(line, "skip_screen_prepass", "true")


def write_lines(
    path: Path,
    lines: Sequence[str],
    plans: Mapping[ShapeKey, ShapePlan],
    everything: Optional[Sequence[int]],
    record: Mapping[str, Any],
) -> int:
    """Write the bench lines of the shapes of `plans` to `path`, as
    `line_record` `record` says (kernel order, sacrificial kernel)."""
    out = []
    for line in lines:
        shape = shape_of_line(line)
        if shape is None or shape not in plans:
            continue
        out.append(
            render_planned_line(
                line,
                plans[shape],
                everything,
                record["order_seed"],
                record["sacrificial_kernel"],
            )
        )
    atomic_write_text(path, "".join(out))
    return len(out)


def line_record(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """The plan.json entries saying how the bench lines list each shape's
    kernels: `kernel_order` with the `order_seed` of its shuffle (None for
    lists kept in plan order), and `sacrificial_kernel`, whether a screened
    line lists its first kernel twice."""
    seed = kernel_order_seed(cfg)
    return {
        "kernel_order": "listed" if seed is None else "shuffled",
        "order_seed": seed,
        "sacrificial_kernel": sacrificial_kernel(cfg),
    }


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
            preexec_fn=procs.die_with_parent(),
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


# ── seed recall ──────────────────────────────────────────────────────────────

# Cells the recall warning names, weakest first.
WEAK_CELLS_SHOWN = 3


def check_seed_recall(
    ranker: Optional[Ranker],
    feas: Feasibility,
    params: PlanParams,
    shapes: Sequence[ShapeKey],
    seeds: Sequence[ShapeKey],
    min_ceiling: float,
    quiet: bool,
) -> Optional[Dict[str, Any]]:
    """`lib.planner.seed_recall` of the lists the seed bootstrap's ranker
    plans with `params`, over the seeds of the cells the round plans (those
    with a shape that is not a seed); a warning when their ceiling is below
    `min_ceiling`. A stage05 seed model trained on every seed, so the
    neighbors ranker it falls back to is scored in its place. None without a
    neighbors ranker; a failure is a warning, never the stage's."""
    neighbors = ranker.fallback if isinstance(ranker, ModelRanker) else ranker
    if not isinstance(neighbors, NeighborsRanker):
        return None
    seed_set = set(seeds)
    cells = {shape_cell(s) for s in shapes if s not in seed_set}
    try:
        rec = seed_recall(neighbors, feas, params, cells=cells, min_ceiling=min_ceiling)
    except Exception as e:
        ui.warn("recall", f"cannot score the lists on the seeds: {e}")
        return {"error": str(e)}
    rec["plan_ranker"] = "stage05" if neighbors is not ranker else neighbors.name
    ceiling = rec["ceiling"]
    if not quiet:
        if not rec["n_seeds"]:
            ui.info(
                "recall",
                "no seed of a planned cell has another measured seed in its "
                "cell: nothing to score",
            )
        else:
            ui.info(
                "recall",
                f"{rec['n_seeds']:,} seed(s) of {rec['n_cells']:,} cell(s), each "
                f"ranked from the other seeds of its cell"
                + (
                    " by the neighbors ranker, the stage05 model's fallback"
                    if neighbors is not ranker
                    else ""
                ),
            )
            ui.info(
                "recall",
                f"top_k {params.top_k}: winner listed "
                f"{rec['winner_recall']:.1%}, a kernel within "
                f"{NEAR_TOLERANCE:.0%} listed {rec['near_recall']:.1%}, ceiling "
                + (f"{ceiling:.3f}" if ceiling is not None else "none")
                + " (winner / fastest listed, geometric mean)"
                + (
                    f"; {rec['n_unmeasured']} list(s) hold no kernel their "
                    f"seed's bench measured"
                    if rec["n_unmeasured"]
                    else ""
                ),
            )
            by_top_k = [
                f"{e['top_k']} {e['ceiling']:.3f}"
                for e in rec["by_top_k"]
                if e["ceiling"] is not None
            ]
            if by_top_k:
                ui.info("recall", "ceiling by top_k: " + ", ".join(by_top_k))
    if ceiling is not None and ceiling < min_ceiling:
        reach = rec["top_k_reaching"]
        weak = sorted(
            (s["ceiling"] or 0.0, c)
            for c, s in rec["cells"].items()
            if (s["ceiling"] or 0.0) < min_ceiling
        )
        ui.warn(
            "recall",
            f"the lists ranked from the seeds reach a ceiling of {ceiling:.3f} "
            f"on the seeds, below planner.recall_min_ceiling {min_ceiling:g}; "
            + (
                f"top_k {reach} and above reach it (planner.top_k_round0)"
                if reach is not None
                else f"no top_k up to {rec['by_top_k'][-1]['top_k']} reaches it"
            )
            + (
                "; weakest cells: "
                + ", ".join(f"{c} {v:.3f}" for v, c in weak[:WEAK_CELLS_SHOWN])
                if weak
                else ""
            ),
        )
    return rec


# ── main ─────────────────────────────────────────────────────────────────────


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="stage02 -- plan the kernels benched per shape (no "
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
        help="JSON list of execution contexts ({name, cu_budget, schedule, "
        "fraction}) to re-plan with --model-file the shapes stage03c benches "
        "in them: contexts/<name>/shapes.yaml and plan.json",
    )
    ap.add_argument(
        "--seed-commands",
        default=None,
        help="JSON object: stage name (stage03, stage04, stage05) -> argv "
        "(script path first) of the seed bootstrap's sub-stages. Their paths "
        "must point into <out-dir>/seed, where the seed shapes and plan are "
        "written.",
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
    if st["rules"] not in RULE_SOURCES:
        ui.err("cfg", f"planner.rules must be one of {RULE_SOURCES}")
        return 1
    try:
        params = plan_params(st)
        seed_params = seed_plan_params(st, params)
        min_ceiling = recall_min_ceiling(st)
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
        pool_cfg = {**cfg, "planner": {**(cfg.get("planner") or {}), "enabled": True}}
        library = config_pool(
            pool_cfg, args.dat_dir, args.library_stem, args.hardware_device
        )
        identity = config_identity(
            pool_cfg, args.dat_dir, args.library_stem, args.hardware_device
        )
        hardware = hwlib.device_hardware(cfg, args.hardware_device)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as e:
        ui.err("load", str(e))
        return 1
    catalog = Catalog.build(library.table, library.bank.kernels)
    lines_record = line_record(cfg)
    kernel_record = {
        ki.RECORD_KEY: ki.record(
            identity, catalog.pool_ids + catalog.bank_ids, catalog.aliases
        ),
        **lines_record,
    }
    ptype = ProblemType.from_config(cfg["hipblaslt"])
    target = config_rules(cfg)
    rank_rules = target if st["rules"] == "config" else None
    feas = Feasibility(catalog, ptype, hardware.lds_bytes, rules=target)
    family = feas if target.get("non_temporal_family_rule") else None

    mode = st["mode"]
    if mode == "auto":
        n_cat = len(catalog.kernels)
        mode = "planned" if n_cat > int(st["exhaustive_max_kernels"]) else "exhaustive"
    if not args.quiet:
        ui.banner("stage02 plan benchmarks", ui.C.HEAD)
        ui.info(
            "cfg", f"shapes       : {len(shapes):,} ({len(set(shapes)):,} distinct)"
        )
        ui.info(
            "cfg",
            f"catalog      : {len(catalog.pool_ids)} pool + "
            f"{len(catalog.bank_ids)} bank kernels, {len(catalog.families)} families"
            + (
                f"; {len(catalog.aliases)} aliases benched as their group's "
                f"first member"
                if catalog.aliases
                else ""
            ),
        )
        ui.info("cfg", f"mode         : {mode} (configured {st['mode']})")
        ui.info("cfg", f"params       : {params.as_dict()}")

    ranker: Optional[Ranker] = None
    seeds: List[ShapeKey] = []
    ranker_desc = "none"
    recall: Optional[Dict[str, Any]] = None
    if mode == "planned":
        seeds = []
        if args.model_file is not None:
            try:
                data = model_bytes(args.model_file, args.arch or "", args.weight_dtype)
                ranker = ModelRanker(data, catalog, ptype, hardware, rules=rank_rules)
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
                    args,
                    st,
                    params,
                    lines,
                    seeds,
                    feas,
                    catalog,
                    ptype,
                    hardware,
                    kernel_record,
                )
                if rc:
                    return rc
                ranker_desc = f"seed bootstrap ({st['ranker']})"
                if isinstance(ranker, ModelRanker):
                    feas = Feasibility(
                        catalog, ptype, hardware.lds_bytes, rules=ranker.rules
                    )
                params = seed_params
                if not args.quiet and st["top_k_round0"] is not None:
                    ui.info(
                        "cfg",
                        f"top_k        : {params.top_k} for the lists the seeds "
                        f"rank (planner.top_k_round0)",
                    )
                recall = check_seed_recall(
                    ranker, feas, params, shapes, seeds, min_ceiling, args.quiet
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
        family_coverage=family,
    )
    by_shape = {sp.shape: sp for sp in plans}
    to_bench = {s: sp for s, sp in by_shape.items() if sp.role != "seed"}
    n_lines = write_lines(
        args.out_dir / "shapes.yaml",
        lines,
        to_bench,
        exhaustive_list(catalog),
        lines_record,
    )
    meta = {
        "mode": mode,
        "configured_mode": st["mode"],
        "ranker": ranker_desc,
        "library_stem": args.library_stem,
        "bank_library_stems": list(st["bank_library_stems"] or []),
        "bank": st["bank"],
        "bank_stems_used": library.bank.stems,
        "library_kind": list(library.source.kinds) if library.source else None,
        "params": params.as_dict(),
        "rules": feas.rules,
        "family_rule_coverage": family is not None,
        "library_digest": identity.digest,
        "n_aliases": len(catalog.aliases),
        **lines_record,
    }
    write_json(
        args.out_dir / PLAN_FILE, plan_document(plans, {**meta, **kernel_record})
    )
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
        "family_rule_added": sum(sp.sources.get("family_rule", 0) for sp in plans),
        "family_rule_shapes": sum(1 for sp in plans if sp.sources.get("family_rule")),
        "n_truncated_shapes": sum(1 for sp in plans if sp.n_truncated),
        "family_rule_over_limit_shapes": sum(
            1 for sp in plans if sp.n_family_over_limit
        ),
        "seed_recall": recall,
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
        if summary["family_rule_over_limit_shapes"]:
            ui.warn(
                "plan",
                f"{summary['family_rule_over_limit_shapes']} shape(s) list fewer "
                f"members of the family-wise non-temporal rule than it keeps: "
                f"the bench takes {MAX_BENCH_LIST} kernels per list",
            )
        ui.ok("write", f"shapes.yaml  : {n_lines:,} lines")
    if args.contexts:
        if args.model_file is None or not isinstance(ranker, ModelRanker):
            ui.warn("contexts", "no model to re-plan with; contexts keep these lists")
        else:
            rc = _replan_contexts(
                args,
                data,
                lines,
                shapes,
                feas,
                params,
                catalog,
                ptype,
                hardware,
                kernel_record,
                int((cfg.get("contexts") or {}).get("seed", 42)),
                family,
            )
            if rc:
                return rc
    if not args.quiet:
        ui.banner("Done", ui.C.GREEN)
    return 0


def _replan_contexts(
    args,
    data,
    lines,
    shapes,
    feas,
    params,
    catalog,
    ptype,
    hardware,
    kernel_record,
    context_seed,
    family=None,
) -> int:
    """Plan the shapes each context of --contexts benches again with the
    model ranking in that context: the shapes stage03c draws for it
    (`lib.contexts.select_shapes`, its `fraction` per cell with
    `contexts.seed`), with the family-wise rule's kernels when `family`
    (`build_plan`'s `family_coverage`); writes contexts/<name>/{shapes.yaml,
    plan.json}."""
    from lib import features as fs
    from lib.contexts import BenchContext, select_shapes

    try:
        wanted = json.loads(args.contexts)
        contexts = [
            BenchContext(
                cu_budget=int(c.get("cu_budget", 0)),
                schedule=str(c.get("schedule", "default")),
                fraction=float(c.get("fraction", 1.0)),
                name=str(c["name"]),
            )
            for c in wanted
        ]
    except (ValueError, KeyError, TypeError) as e:
        ui.err("contexts", f"--contexts: {e}")
        return 1
    for bench_ctx in contexts:
        name = str(bench_ctx.name)
        ctx = fs.ExecutionContext(
            cu_budget=bench_ctx.cu_budget, schedule=bench_ctx.schedule
        )
        ranker = ModelRanker(
            data, catalog, ptype, hardware, context=ctx, rules=feas.rules
        )
        chosen = select_shapes(shapes, bench_ctx, context_seed)
        plans, _ = build_plan(
            chosen, feas, params, ranker=ranker, family_coverage=family
        )
        out = args.out_dir / "contexts" / name
        out.mkdir(parents=True, exist_ok=True)
        by_shape = {sp.shape: sp for sp in plans}
        n = write_lines(
            out / "shapes.yaml",
            lines,
            by_shape,
            exhaustive_list(catalog),
            kernel_record,
        )
        meta = {
            "mode": "planned",
            "ranker": f"model {args.model_file} in context {name}",
            "context": {"cu_budget": ctx.cu_budget, "schedule": ctx.schedule},
            "fraction": bench_ctx.fraction,
            "context_seed": context_seed,
            "library_stem": args.library_stem,
            "params": params.as_dict(),
            **kernel_record,
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
    kernel_record: Mapping[str, Any],
):
    """Bench `seeds` exhaustively and build the ranker; (rc, ranker). rc is
    2 when a retry may cure the failure (the seed bench's exit 2, or a
    sub-stage killed from outside), else 1."""
    if not args.seed_commands:
        ui.err("seed", "the seed bootstrap needs --seed-commands (or --model-file)")
        return 1, None
    try:
        cmds = json.loads(args.seed_commands)
    except ValueError as e:
        ui.err("seed", f"--seed-commands is not JSON: {e}")
        return 1, None
    missing = [k for k in SEED_STAGES if not cmds.get(k)]
    if st["ranker"] == "stage05" and not cmds.get("stage05"):
        missing.append("stage05")
    if missing:
        ui.err("seed", f"--seed-commands lacks {missing}")
        return 1, None
    unknown = sorted(set(cmds) - {*SEED_STAGES, "stage05"})
    if unknown:
        ui.err("seed", f"--seed-commands: unknown sub-stages {unknown}")
        return 1, None
    seed_dir = args.out_dir / "seed"
    seed_dir.mkdir(parents=True, exist_ok=True)
    seed_plans, _ = build_plan(seeds, feas, params, ranker=None, exhaustive=True)
    for sp in seed_plans:
        sp.role = "seed"
    by_shape = {sp.shape: sp for sp in seed_plans}
    n = write_lines(
        seed_dir / "shapes.yaml",
        lines,
        by_shape,
        exhaustive_list(catalog),
        kernel_record,
    )
    write_json(
        seed_dir / PLAN_FILE,
        plan_document(
            seed_plans, {"mode": "exhaustive", "ranker": "none", **kernel_record}
        ),
    )
    if not args.quiet:
        ui.info("seed", f"{len(seeds):,} seed shapes ({n:,} lines) -> {seed_dir}")
    for key in SEED_STAGES:
        cmd = cmds[key]
        (seed_dir / key).mkdir(parents=True, exist_ok=True)
        rc = run_sub_stage(key, cmd, seed_dir / f"{key}.log", args.quiet)
        if rc:
            ui.err(
                "seed", f"seed {key} failed (rc={rc}); see {seed_dir / (key + '.log')}"
            )
            return procs.retry_status(rc, bench=key == "stage03"), None
    lat = seed_latencies(str(seed_dir / "stage04"))
    if not lat:
        ui.err("seed", "the seed bench produced no measurements")
        return 1, None
    ranker: Ranker = NeighborsRanker(lat, k=int(st["neighbors_k"]))
    if st["ranker"] == "stage05":
        (seed_dir / "stage05").mkdir(parents=True, exist_ok=True)
        rc = run_sub_stage(
            "stage05", cmds["stage05"], seed_dir / "stage05.log", args.quiet
        )
        models = seed_dir / "stage05" / "models.pt"
        if rc not in (0, STAGE05_CELLS_NOT_TRAINED) or not models.is_file():
            ui.err("seed", f"seed stage05 failed (rc={rc})")
            return procs.retry_status(rc) or 1, None
        try:
            data = model_bytes(models, args.arch or "", args.weight_dtype)
            ranker = ModelRanker(
                data,
                catalog,
                ptype,
                hardware,
                fallback=ranker,
                rules=feas.rules if st["rules"] == "config" else None,
            )
        except Exception as e:
            ui.err("seed", f"cannot rank with the seed model: {e}")
            return 1, None
    return 0, ranker


if __name__ == "__main__":
    sys.exit(main())
