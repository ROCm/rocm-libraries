# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage08_optional_validate_selection_efficiency -- selection efficiency of
a trained bundle on external datasets, deployed model vs Origami.

A dataset is one of
  (1) a hipblaslt-bench YAML of problems: benched here (stage02 probe when the
      config has a `probe:` section and bench.adaptive is off, then stage03
      over every candidate) with --bench-binary, then enriched;
  (2) a directory of stage03 block_*.log files: enriched with stage04;
  (3) a directory of stage04 chunk_*.csv files: evaluated as is.

The bundle is evaluated exactly as deployed (lib/evaluate.py): MLREC_v2 at
--weight-dtype, the engine's routing over the split tree rebuilt from the
bundle's cell labels, the whitelists, and the library pool (--library-dir /
--library-stem; candidates outside the library are dropped). The pick is a
pool position and costs that solution's own latency. A GEMM the model
scores nothing for gets the runtime's fallback, the Origami pick, so every
GEMM with a measured Origami pick counts. Model and Origami are compared on
the same GEMMs (paired: geomeans, wins / ties / losses within
--tie-tolerance, low percentiles, bootstrap interval of the speedup).

Output per dataset in <out-dir>/<name>/: bench/ and stage04_enriched/ when
produced here, per_cell_sel_eff.json, per_gemm.csv (pick and Origami pick
as pool position and solution index); plus <out-dir>/summary.json over all
datasets.

A dataset fails (rc 1 in summary.json, the stage exits 1) when it cannot be
evaluated or when the engine serves any of its GEMMs from another cell than
lib/subcells routes it to; its outputs are still written.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = THIS_DIR.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import yaml  # noqa: E402

from lib import evaluate as ev  # noqa: E402
from lib import hardware as hwlib  # noqa: E402
from lib import mlrec  # noqa: E402
from lib import ui  # noqa: E402
from run_orchestrator import arch_base  # noqa: E402


def _detect_kind(path: Path) -> str:
    """ "yaml" / "logs" / "enriched" / "missing" / "unknown"."""
    if not path.exists():
        return "missing"
    if path.is_file():
        return "yaml" if path.suffix.lower() in (".yaml", ".yml") else "unknown"
    names = [p.name for p in path.iterdir()]
    if any(n.startswith("chunk_") and n.endswith(".csv") for n in names):
        return "enriched"
    if any(n.startswith("block_") and n.endswith(".log") for n in names):
        return "logs"
    return "unknown"


def _run_subprocess(cmd: List[str], log_path: Path, quiet: bool = False) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not quiet:
        ui.info("run", " ".join(str(c) for c in cmd))
    t0 = time.time()
    with log_path.open("w", buffering=1) as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            if not quiet:
                sys.stdout.write(line)
                sys.stdout.flush()
            logf.write(line)
        rc = proc.wait()
    if not quiet:
        (ui.ok if rc == 0 else ui.err)(
            "run", f"  completed in {ui.fmt_dur(time.time() - t0)} (rc={rc})"
        )
    return rc


def _fail(name: str, path: Path, reason: str, rc: int = 1) -> Dict[str, Any]:
    ui.err("dataset", f"{name}: {reason}")
    return {"name": name, "path": str(path), "rc": rc, "reason": reason}


def _bench_yaml(
    name: str, path: Path, out: Path, args: argparse.Namespace
) -> Tuple[Optional[Path], Optional[str]]:
    """Bench every candidate of the problems in `path`; (logs dir, error)."""
    if args.bench_binary is None or not Path(args.bench_binary).exists():
        return None, "a yaml dataset needs an existing --bench-binary"
    bench_in = out / "bench_input"
    bench_in.mkdir(parents=True, exist_ok=True)
    entries = yaml.safe_load(path.read_text()) or []
    if isinstance(entries, dict):
        entries = [entries]
    n = 0
    with (bench_in / "shapes.yaml").open("w") as f:
        for e in entries:
            if not isinstance(e, dict):
                continue
            e = dict(e)
            e["requested_solution_num"] = -1
            e["print_kernel_info"] = 1
            e.setdefault("iters", 2)
            e.setdefault("cold_iters", 2)
            f.write(
                "- "
                + yaml.safe_dump(
                    e, default_flow_style=True, sort_keys=False, width=10**9
                ).strip()
                + "\n"
            )
            n += 1
    if not args.quiet:
        ui.info("bench", f"{name}: {n} problems, every candidate benched")
    cfg = {}
    if args.config_yaml is not None:
        cfg = yaml.safe_load(args.config_yaml.read_text()) or {}
    probe = cfg.get("probe") or {}
    adaptive = bool(((cfg.get("bench") or {}).get("adaptive") or {}).get("enabled"))
    common = ["--bench-binary", str(args.bench_binary), "--devices", args.devices]
    if probe and not adaptive:
        rc = _run_subprocess(
            [
                sys.executable,
                "-u",
                str(THIS_DIR / "stage02_probe.py"),
                "--in-dir",
                str(bench_in),
                "--out-dir",
                str(bench_in),
                "--config-yaml",
                str(args.config_yaml),
                *common,
            ],
            out / "stage02.log",
            args.quiet,
        )
        if rc != 0:
            ui.warn("probe", f"{name}: probe failed (rc={rc}); fixed iters used")
    cmd = [
        sys.executable,
        "-u",
        str(THIS_DIR / "stage03_load_balance_offline_tuning.py"),
        "--in-dir",
        str(bench_in),
        "--out-dir",
        str(out / "bench"),
        *common,
        "--blocks-per-gpu",
        str(args.blocks_per_gpu),
    ]
    if args.config_yaml is not None:
        cmd += ["--config-yaml", str(args.config_yaml)]
    rc = _run_subprocess(cmd, out / "stage03.log", args.quiet)
    if rc != 0:
        return None, f"stage03 failed (rc={rc})"
    return out / "bench" / "logs", None


def _enrich(
    name: str, logs_dir: Path, out: Path, dat_dir: Path, args: argparse.Namespace
) -> Tuple[Optional[Path], Optional[str]]:
    enriched = out / "stage04_enriched"
    cmd = [
        sys.executable,
        "-u",
        str(THIS_DIR / "stage04_convert_to_enriched_dataset.py"),
        "--log-dir",
        str(logs_dir),
        "--out-dir",
        str(enriched),
        "--dat-dir",
        str(dat_dir),
        "--progress-every",
        "60",
    ]
    if args.library_stem:
        cmd += ["--library-stem", args.library_stem]
    rc = _run_subprocess(cmd, out / "stage04.log", args.quiet)
    if rc != 0:
        return None, f"stage04 failed (rc={rc})"
    return enriched, None


def _norm_field(field: str, val: Any) -> str:
    s = str(val).strip()
    if field == "compute_type" and s.lower().startswith("c_"):
        s = s[2:]
    return s


def _per_cell_rows(
    results: List[ev.GemmResult], min_cell_gemms: int, tie_tolerance: float
) -> List[Dict[str, Any]]:
    rows = []
    by_leaf: Dict[str, List[ev.GemmResult]] = {}
    for r in results:
        by_leaf.setdefault(r.leaf, []).append(r)
    for leaf in sorted(by_leaf):
        rs = by_leaf[leaf]
        s = ev.summarize(rs, tie_tolerance=tie_tolerance)
        cells = Counter(r.model_cell for r in rs)
        rows.append(
            {
                "cell": leaf,
                "model_cell": cells.most_common(1)[0][0],
                "model_cells": sorted(c for c in cells if c is not None),
                "n_gemms": len(rs),
                "low_n": len(rs) < int(min_cell_gemms),
                "n_evaluated": s["n_evaluated"],
                "n_model_served": s["n_model_served"],
                "n_origami_fallback": s["n_origami_fallback"],
                "model_sel_eff": s["sel_eff"],
                "model_pick_us_geomean": s["pick_us_geomean"],
                "winner_us_geomean": s["winner_us_geomean"],
                "origami_sel_eff": s["origami_sel_eff"],
                "origami_n_eval": s["n_origami"],
                "origami_pick_us_geomean": s["origami_pick_us_geomean"],
                "paired": s["paired"],
            }
        )
    return rows


def _write_per_gemm(path: Path, results: List[ev.GemmResult]) -> None:
    fields = [
        "m",
        "n",
        "k",
        "batch",
        "leaf",
        "model_cell",
        "served_by",
        "winner_us",
        "pick_index",
        "pick_sol_idx",
        "pick_us",
        "origami_index",
        "origami_sol_idx",
        "origami_us",
        "sel_eff",
        "origami_sel_eff",
        "n_scored",
        "pick_rank",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r.to_json())


def _geo(rows: List[Dict[str, Any]], key: str) -> float:
    return ev.geomean(r[key] for r in rows if r.get(key) is not None)


def _evaluate_dataset(
    name: str,
    path: Path,
    out: Path,
    enriched_dir: Path,
    ctx: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    import stage05_train as _train

    cells, _stats = _train.load_enriched_chunks(str(enriched_dir), args.quiet)
    gemms = [g for gs in cells.values() for g in gs]
    n_before = len(gemms)
    wf = {
        k: _norm_field(k, v)
        for k, v in ctx["workload_filter"].items()
        if v is not None and str(v) != ""
    }
    if wf:
        gemms = [
            g
            for g in gemms
            if all(_norm_field(k, g.get(k, "")) == v for k, v in wf.items())
        ]
    n_filtered = n_before - len(gemms)
    n_rows_dropped = n_gemms_dropped = 0
    if ctx["library"] is not None:
        gemms, n_rows_dropped, n_gemms_dropped = ev.restrict_to_pool(
            gemms, ctx["library"]
        )
    if not args.quiet:
        ui.info(
            "data",
            f"{name}: {n_before} GEMMs, {n_filtered} dropped by the workload "
            f"filter, {n_gemms_dropped} with no candidate of the library "
            f"({n_rows_dropped} out-of-library rows dropped); {len(gemms)} left",
        )
    if not gemms:
        return _fail(name, path, "no GEMM left to evaluate")

    evaluation = ev.evaluate_model(
        ctx["model_bytes"],
        gemms,
        hardware=ctx["hardware"],
        pools=ev.pools_for(gemms, ctx["library"]),
        labels=ctx["labels"],
        tw=ctx["tw"],
    )
    results = evaluation.results
    per_cell = _per_cell_rows(results, args.min_cell_gemms, args.tie_tolerance)
    overall = ev.summarize(
        results, tie_tolerance=args.tie_tolerance, bootstrap=args.bootstrap
    )
    summary = {
        "n_gemms_before_filter": n_before,
        "n_gemms_dropped_by_filter": n_filtered,
        "n_gemms_without_library_candidate": n_gemms_dropped,
        "n_out_of_library_rows": n_rows_dropped,
        "workload_filter": wf,
        "library_stem": args.library_stem,
        "weight_dtype": args.weight_dtype,
        "n_cells_evaluated": len(per_cell),
        "n_routing_mismatch": evaluation.n_routing_mismatch,
        "model_geomean_sel_eff": _geo(per_cell, "model_sel_eff"),
        "origami_geomean_sel_eff": _geo(per_cell, "origami_sel_eff"),
        "model_geomean_gemm_weighted": overall["sel_eff"],
        "origami_geomean_gemm_weighted": overall["origami_sel_eff"],
        "overall": overall,
    }
    per_cell_path = out / "per_cell_sel_eff.json"
    per_cell_path.write_text(
        json.dumps(
            _train.jsonable({"per_cell": per_cell, "summary": summary}), indent=2
        )
    )
    _write_per_gemm(out / "per_gemm.csv", results)

    if not args.quiet:
        w = min(max(max((len(r["cell"]) for r in per_cell), default=24), 24), 80)
        ui.info(
            "eval",
            f"{'cell':<{w}} {'n':>6} {'model':>8} {'origami':>8} "
            f"{'fallback':>8}  W/T/L",
        )
        for r in per_cell:
            p = r["paired"]
            ui.info(
                "eval",
                f"{r['cell']:<{w}} {r['n_gemms']:>6d} "
                f"{_train.fmt_pct(r['model_sel_eff'])} "
                f"{_train.fmt_pct(r['origami_sel_eff'])} "
                f"{r['n_origami_fallback']:>8d}  "
                f"{p['wins']}/{p['ties']}/{p['losses']}",
            )
        p = overall["paired"]
        ci = p.get("speedup_ci95")
        ui.ok(
            "eval",
            f"{name}: paired n={p['n']}  model={_train.fmt_pct(p['model_sel_eff'])}  "
            f"origami={_train.fmt_pct(p['origami_sel_eff'])}  "
            f"wins/ties/losses={p['wins']}/{p['ties']}/{p['losses']}"
            + (f"  speedup 95% CI=[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""),
        )
        if overall["n_not_evaluable"]:
            ui.warn(
                "eval",
                f"{overall['n_not_evaluable']} GEMMs had neither a measured "
                f"model pick nor a measured Origami pick",
            )
    result = {
        "name": name,
        "path": str(path),
        "rc": 0,
        "per_cell_json": str(per_cell_path),
        "n_evaluated_cells": len(per_cell),
        "model_geomean_sel_eff": summary["model_geomean_sel_eff"],
        "origami_geomean_sel_eff": summary["origami_geomean_sel_eff"],
        "summary": summary,
    }
    try:
        ev.require_consistent_routing(evaluation, name)
    except ev.RoutingMismatchError as e:
        ui.err("eval", str(e))
        result.update(rc=1, reason=str(e))
    return result


def _run_one_dataset(
    name: str,
    path: Path,
    dat_dir: Optional[Path],
    ctx: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    out = args.out_dir / name
    out.mkdir(parents=True, exist_ok=True)
    kind = _detect_kind(path)
    if not args.quiet:
        ui.banner(f"dataset :: {name}  kind={kind}", ui.C.BLUE)
    if kind in ("missing", "unknown"):
        return _fail(name, path, f"input kind {kind}")
    enriched_dir = path
    if kind in ("yaml", "logs"):
        logs_dir = path
        if kind == "yaml":
            logs_dir, err = _bench_yaml(name, path, out, args)
            if err:
                return _fail(name, path, err)
        if dat_dir is None or not dat_dir.is_dir():
            return _fail(
                name, path, f"enrichment needs a Tensile library dir (got {dat_dir})"
            )
        enriched, err = _enrich(name, logs_dir, out, dat_dir, args)
        if err:
            return _fail(name, path, err)
        enriched_dir = enriched
    return _evaluate_dataset(name, path, out, enriched_dir, ctx, args)


def dataset_library_dirs(specs: Optional[List[str]], arch: str) -> Dict[str, Path]:
    """{dataset name: Tensile library dir} of `--dataset-build-dir NAME=DIR`
    flags: `DIR/Tensile/library/<arch without target features>`, the
    directory a hipBLASLt build places the arch's libraries in."""
    out: Dict[str, Path] = {}
    for spec in specs or []:
        sep = "=" if "=" in spec else ":"
        if sep not in spec:
            raise ValueError(f"--dataset-build-dir expects NAME=DIR, got {spec!r}")
        nm, pth = spec.split(sep, 1)
        out[nm.strip()] = Path(pth.strip()) / "Tensile" / "library" / arch_base(arch)
    return out


def _parse_specs(
    spec_args: Optional[List[str]], yaml_file: Optional[Path]
) -> List[Tuple[str, Path]]:
    """`name=path` / `name:path` flags plus a YAML list of {name, path}."""
    out: List[Tuple[str, Path]] = []
    if yaml_file is not None:
        data = yaml.safe_load(yaml_file.read_text())
        if not isinstance(data, list):
            raise SystemExit(f"datasets yaml must be a list; got {type(data).__name__}")
        for entry in data:
            if isinstance(entry, list) and len(entry) == 2:
                nm, p = entry
            elif isinstance(entry, dict):
                nm, p = entry.get("name"), entry.get("path")
            else:
                raise SystemExit(f"bad dataset entry: {entry!r}")
            if not nm or not p:
                raise SystemExit(f"dataset entry missing name/path: {entry!r}")
            out.append((str(nm), Path(str(p))))
    for s in spec_args or []:
        sep = "=" if "=" in s else ":"
        if sep not in s:
            raise SystemExit(f"expected NAME=PATH, got {s!r}")
        nm, p = s.split(sep, 1)
        out.append((nm.strip(), Path(p.strip())))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="stage08: deployed selection efficiency of a trained "
        "bundle on external datasets (yaml / bench logs / enriched chunks)."
    )
    ap.add_argument(
        "--train-dir", type=Path, required=True, help="stage05 dir with models.pt"
    )
    ap.add_argument("--arch", required=True)
    ap.add_argument(
        "--config-yaml",
        type=Path,
        default=None,
        help="run config: `hardware:` block, and `probe:` for yaml datasets",
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
        help="Tensile library dir: the library pool and the enrichment of "
        "log / yaml datasets",
    )
    ap.add_argument(
        "--library-stem",
        default=None,
        help="library whose kernels form the candidate pool; other rows are " "dropped",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="NAME=PATH (bench yaml, logs dir or enriched dir); repeatable",
    )
    ap.add_argument(
        "--dataset-build-dir",
        action="append",
        default=None,
        help="NAME=DIR: hipBLASLt build directory a dataset's bench ran with; "
        "its Tensile/library/<arch> enriches that dataset (default: "
        "--library-dir)",
    )
    ap.add_argument(
        "--datasets-yaml",
        type=Path,
        default=None,
        help="YAML list of {name, path} (or [name, path]) entries",
    )
    ap.add_argument(
        "--bench-binary",
        type=Path,
        default=None,
        help="hipblaslt-bench (yaml datasets)",
    )
    ap.add_argument("--devices", default="0", help="GPU ids for yaml datasets")
    ap.add_argument("--blocks-per-gpu", type=int, default=20)
    ap.add_argument(
        "--min-cell-gemms",
        type=int,
        default=20,
        help="cells with fewer GEMMs are flagged low_n (still evaluated)",
    )
    ap.add_argument(
        "--tie-tolerance",
        type=float,
        default=0.01,
        help="relative latency difference counted as a tie in the paired stats",
    )
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="bootstrap resamples of the paired speedup interval (0: none)",
    )
    for fld in ("a-type", "b-type", "c-type", "d-type", "compute-type"):
        ap.add_argument(f"--filter-{fld}", default=None)
    ap.add_argument("--filter-transA", default=None)
    ap.add_argument("--filter-transB", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    datasets = _parse_specs(args.dataset, args.datasets_yaml)
    if not datasets:
        ui.err("err", "no datasets: pass --dataset NAME=PATH or --datasets-yaml")
        return 1
    if args.library_stem and args.library_dir is None:
        ui.err("err", "--library-stem needs --library-dir")
        return 1
    try:
        tw = ev.tilewright_module()
    except RuntimeError as e:
        ui.err("engine", str(e))
        return 1
    cfg: Dict[str, Any] = {}
    if args.config_yaml is not None:
        cfg = yaml.safe_load(args.config_yaml.read_text()) or {}
    try:
        hardware = hwlib.device_hardware(cfg, args.hardware_device)
        constants = hwlib.arch_constants(args.arch)
    except (ValueError, RuntimeError, OSError) as e:
        ui.err("hardware", str(e))
        return 1
    try:
        dat_by_name = dataset_library_dirs(args.dataset_build_dir, args.arch)
    except ValueError as e:
        ui.err("err", str(e))
        return 1
    models_pt = args.train_dir / "models.pt"
    if not models_pt.exists():
        ui.err("err", f"models bundle missing: {models_pt}")
        return 1
    import torch

    bundle = torch.load(str(models_pt), map_location="cpu", weights_only=True)
    model_bytes = mlrec.write_model(
        bundle, None, args.arch, constants, args.weight_dtype
    )
    library = (
        ev.library_pool(args.library_dir, args.library_stem)
        if args.library_stem
        else None
    )
    ctx = {
        "tw": tw,
        "hardware": hardware,
        "model_bytes": model_bytes,
        "labels": list(bundle["models"].keys()),
        "library": library,
        "workload_filter": {
            "a_type": args.filter_a_type,
            "b_type": args.filter_b_type,
            "c_type": args.filter_c_type,
            "d_type": args.filter_d_type,
            "compute_type": args.filter_compute_type,
            "transA": args.filter_transA,
            "transB": args.filter_transB,
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.quiet:
        ui.banner("stage08  deployed selection efficiency")
        ui.info("cfg", f"train_dir    : {args.train_dir}")
        ui.info("cfg", f"cells        : {len(bundle['models'])}")
        ui.info("cfg", f"weight dtype : {args.weight_dtype}")
        ui.info("cfg", f"hardware     : {asdict(hardware)}")
        ui.info(
            "cfg",
            f"library pool : {args.library_stem or 'kernels measured in each dataset'}"
            + (f" ({len(library)} kernels)" if library is not None else ""),
        )
        ui.info("cfg", f"datasets     : {[(n, str(p)) for n, p in datasets]}")

    summaries: List[Dict[str, Any]] = []
    for name, path in datasets:
        dat_dir = dat_by_name.get(name, args.library_dir)
        summaries.append(_run_one_dataset(name, path, dat_dir, ctx, args))
    out_json = args.out_dir / "summary.json"
    import stage05_train as _train

    out_json.write_text(json.dumps(_train.jsonable({"datasets": summaries}), indent=2))
    n_failed = sum(1 for s in summaries if s.get("rc") != 0)
    if not args.quiet:
        ui.banner("Done", ui.C.GREEN if n_failed == 0 else ui.C.YELLOW)
        for s in summaries:
            if s.get("rc") != 0:
                ui.warn("dataset", f"{s['name']:20s}  rc={s['rc']}  {s.get('reason')}")
                continue
            p = s["summary"]["overall"]["paired"]
            ui.ok(
                "dataset",
                f"{s['name']:20s}  n_cells={s['n_evaluated_cells']}  paired n={p['n']}  "
                f"model={_train.fmt_pct(p['model_sel_eff'])}  "
                f"origami={_train.fmt_pct(p['origami_sel_eff'])}",
            )
        ui.ok("done", f"summary.json -> {out_json}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
