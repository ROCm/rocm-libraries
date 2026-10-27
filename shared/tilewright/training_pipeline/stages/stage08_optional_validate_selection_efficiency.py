# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage08_optional_validate_selection_efficiency -- selection efficiency of
a trained bundle on external datasets. Report-only: nothing here feeds a
training or deploy decision.

A dataset is one of
  (1) a hipblaslt-bench YAML of problems: benched here like training benches
      (stage03 over every candidate) with --bench-binary, then enriched.
      Each entry's M, N, K and batch_count become a bench line of
      lib.bench_yaml.render_bench_line with the config's data types, layout
      and bench knobs (rotating buffer, flush, skip screen, adaptive
      timing); entries that name another data type or layout than the
      config are dropped, repeated problems benched once;
  (2) a directory of stage03 block_*.log files: enriched with stage04;
  (3) a directory of stage04 chunk_*.csv files: evaluated as is.

The bundle is evaluated exactly as deployed (lib/evaluate.py): MLREC_v2 at
--weight-dtype, the engine's routing over the split tree rebuilt from the
bundle's cell labels, the whitelists, and the library pool (--library-dir /
--library-stem, followed by the bank of the config's planner section,
lib/bank_pool.py; candidates outside the pool are dropped). Logs are
enriched with the rows of the library and of the bank's files (stage04
--extra-library-stem), and a dataset of enriched rows whose
enrich_summary.json shows the bank's rows dropped fails: a bank kernel
without rows would be passed over as if its predicates failed, and the
winner taken from the library alone. Kernels the bundle's blocklist names
(lib/correctness.py `bundle_blocklist`: the sidecar its deploy staged, else
the blocklist its stage05 applied; the config's when neither can be read)
are in no pool and their rows are dropped, as the runtime serving the bundle
leaves them out, so the winner is the fastest kernel the bundle counts as
computing correct results; a config whose blocklist names other kernels is
warned about. The pick is a pool position and costs that solution's own
latency; under --pick-measurement timed a pick without a timed row of its
own is a miss. With --rebench-misses every miss is re-benched on --devices
with --bench-binary and charged by its measured ratio to the winner
(lib/miss_rebench.py); the configured rule's numbers stay as `pre_rebench`.
A GEMM the model scores nothing for gets the runtime's fallback, the Origami
pick. The headline is the model against the fastest measured kernel over
every evaluated GEMM. Origami is an external reference, compared only on the
same GEMMs (paired: geomeans, wins / ties / losses within --tie-tolerance,
low percentiles, bootstrap interval of the speedup), and printed only with
--origami-reference.

A dataset's solution indices are read as the kernels they named: rows
enriched with another hipBLASLt build (--dataset-build-dir, or an enriched
directory) whose kernels.json records kernel ids are mapped to the library's
indices by kernel id (lib/kernel_identity.py), and rows of kernels the
library lacks are left out.

Output per dataset in <out-dir>/<name>/: bench/ and stage04_enriched/ when
produced here, per_cell_sel_eff.json (its summary with the library digest
and the bank's files), per_gemm.csv (pick and Origami pick as pool position,
solution index, kernel id and signature, and misses), with --rebench-misses
rebench/ and rebench_misses.json (with the library digest of its solution
indices, and each pick and re-benched winner named by signature and
kernel id); plus <out-dir>/summary.json over all datasets.

A dataset fails (rc 1 in summary.json, the stage exits 1) when it cannot be
evaluated or when the engine serves any of its GEMMs from another cell than
lib/subcells routes it to; its outputs are still written. A dataset whose
GEMMs are a registered never-train set (lib/external_sets.py) fails before
it is benched or evaluated.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = THIS_DIR.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import yaml  # noqa: E402

from lib import bank_pool  # noqa: E402
from lib import evaluate as ev  # noqa: E402
from lib import external_sets as xs  # noqa: E402
from lib import hardware as hwlib  # noqa: E402
from lib import kernel_identity as ki  # noqa: E402
from lib import miss_rebench as mr  # noqa: E402
from lib import mlrec  # noqa: E402
from lib import procs  # noqa: E402
from lib import repeats  # noqa: E402
from lib import ui  # noqa: E402
from lib.bench_yaml import write_bench_yaml  # noqa: E402
from lib.correctness import Blocklist, blocklist_key, bundle_blocklist  # noqa: E402
from lib.correctness import config_blocklist  # noqa: E402
from lib.shapes import Shape  # noqa: E402
from run_orchestrator import arch_base  # noqa: E402


ENRICH_SUMMARY = "enrich_summary.json"


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
            preexec_fn=procs.die_with_parent(),
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


TYPE_FIELDS = (
    "a_type",
    "b_type",
    "c_type",
    "d_type",
    "compute_type",
    "transA",
    "transB",
    "scaleA",
    "scaleB",
)


def dataset_shapes(
    entries: List[Any], hipblaslt: Dict[str, Any]
) -> Tuple[List[Shape], Dict[str, int]]:
    """(problems to bench, counts) of a bench yaml's entries: the M, N, K and
    batch_count of every entry whose data types and layout (the fields it
    names) are the config's, each problem once."""
    want = {
        f: _norm_field(f, hipblaslt[f])
        for f in TYPE_FIELDS
        if hipblaslt.get(f) is not None
    }
    seen: Dict[Tuple[int, int, int, int], Shape] = {}
    counts = {"n_entries": 0, "n_other_type": 0, "n_repeated": 0, "n_unreadable": 0}
    for e in entries:
        if not isinstance(e, dict):
            continue
        counts["n_entries"] += 1
        if any(f in e and _norm_field(f, e[f]) != v for f, v in want.items()):
            counts["n_other_type"] += 1
            continue
        try:
            key = (
                int(e["M"]),
                int(e["N"]),
                int(e["K"]),
                int(e.get("batch_count", 1) or 1),
            )
        except (KeyError, TypeError, ValueError):
            counts["n_unreadable"] += 1
            continue
        if key in seen:
            counts["n_repeated"] += 1
            continue
        seen[key] = Shape(m=key[0], n=key[1], k=key[2], batch=key[3])
    return list(seen.values()), counts


def _bench_yaml(
    name: str, path: Path, out: Path, args: argparse.Namespace
) -> Tuple[Optional[Path], Optional[str]]:
    """Bench every candidate of the problems in `path` like a training bench;
    (logs dir, error)."""
    if args.bench_binary is None or not Path(args.bench_binary).exists():
        return None, "a yaml dataset needs an existing --bench-binary"
    cfg = (
        yaml.safe_load(args.config_yaml.read_text()) or {}
        if args.config_yaml is not None
        else {}
    )
    if not isinstance(cfg.get("hipblaslt"), dict):
        return None, (
            "a yaml dataset is benched like the training data: it needs "
            "--config-yaml with the hipblaslt section"
        )
    entries = yaml.safe_load(path.read_text()) or []
    if isinstance(entries, dict):
        entries = [entries]
    why = xs.refusal(xs.entry_gemms(entries))
    if why is not None:
        return None, why
    shapes, counts = dataset_shapes(list(entries), cfg["hipblaslt"])
    if not shapes:
        return None, f"no problem of the config's data types and layout ({counts})"
    bench_in = out / "bench_input"
    bench_in.mkdir(parents=True, exist_ok=True)
    write_bench_yaml(shapes, cfg, bench_in / "shapes.yaml")
    (out / "bench_input.json").write_text(
        json.dumps(dict(counts, n_benched=len(shapes)), indent=2) + "\n"
    )
    if not args.quiet:
        ui.info(
            "bench",
            f"{name}: {len(shapes)} problems benched like training "
            f"({counts['n_other_type']} of other data types or layout, "
            f"{counts['n_repeated']} repeated, {counts['n_unreadable']} unreadable "
            f"of {counts['n_entries']} entries)",
        )
    cmd = [
        sys.executable,
        "-u",
        str(THIS_DIR / "stage03_load_balance_offline_tuning.py"),
        "--in-dir",
        str(bench_in),
        "--out-dir",
        str(out / "bench"),
        "--bench-binary",
        str(args.bench_binary),
        "--devices",
        args.devices,
        "--blocks-per-gpu",
        str(args.blocks_per_gpu),
    ]
    if args.config_yaml is not None:
        cmd += ["--config-yaml", str(args.config_yaml)]
    rc = _run_subprocess(cmd, out / "stage03.log", args.quiet)
    if rc != 0:
        return None, f"stage03 failed (rc={rc})"
    return out / "bench" / "logs", None


def bank_stems(
    cfg: Mapping[str, Any], dat_dir: Path, args: argparse.Namespace
) -> List[str]:
    """The config's bank files in `dat_dir` besides --library-stem
    (`lib.bank_pool.bank_library_stems`); none without --library-stem."""
    if not args.library_stem:
        return []
    return bank_pool.bank_library_stems(
        cfg, dat_dir, args.library_stem, args.hardware_device
    )


def _enrich(
    name: str,
    logs_dir: Path,
    out: Path,
    dat_dir: Path,
    args: argparse.Namespace,
    extra_stems: List[str],
) -> Tuple[Optional[Path], Optional[str]]:
    """stage04 over `logs_dir`, keeping the rows of --library-stem and of
    `extra_stems` (the bank, which the evaluation pool ranks too)."""
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
        for stem in extra_stems:
            cmd += ["--extra-library-stem", stem]
    rc = _run_subprocess(cmd, out / "stage04.log", args.quiet)
    if rc != 0:
        return None, f"stage04 failed (rc={rc})"
    return enriched, None


def unenriched_stems(enriched_dir: Path, stems: List[str]) -> List[str]:
    """The files of `stems` whose rows stage04 dropped from `enriched_dir`
    (its enrich_summary.json names a --library-stem without them as
    --extra-library-stem); none when the folder has no summary."""
    try:
        summary = json.loads((enriched_dir / ENRICH_SUMMARY).read_text())
    except (OSError, ValueError):
        return []
    if not isinstance(summary, dict) or not summary.get("library_stem"):
        return []
    kept = {summary["library_stem"], *(summary.get("extra_library_stems") or [])}
    return [s for s in stems if s not in kept]


def _norm_field(field: str, val: Any) -> str:
    s = str(val).strip()
    if field == "compute_type" and s.lower().startswith("c_"):
        s = s[2:]
    return s


def _bench_input_yamls(out: Path) -> List[Path]:
    """The bench lines of a yaml dataset benched here."""
    return [out / "bench_input" / "shapes.yaml"]


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
                "n_miss": s["n_miss"],
                "model_sel_eff": s["sel_eff"],
                "model_pick_us_geomean": s["pick_us_geomean"],
                "winner_us_geomean": s["winner_us_geomean"],
                "paired_model_sel_eff": s["paired"]["model_sel_eff"],
                "origami_sel_eff": s["origami_sel_eff"],
                "origami_n_eval": s["n_origami"],
                "origami_pick_us_geomean": s["origami_pick_us_geomean"],
                "paired": s["paired"],
            }
        )
    return rows


def _write_per_gemm(
    path: Path,
    results: List[ev.GemmResult],
    kernel_ids: Mapping[int, str],
    signatures: Mapping[int, str],
) -> None:
    """per_gemm.csv; `kernel_ids` and `signatures` name the kernel of each
    solution index."""
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
        "pick_kernel_id",
        "pick_signature",
        "pick_us",
        "origami_index",
        "origami_sol_idx",
        "origami_kernel_id",
        "origami_signature",
        "origami_us",
        "sel_eff",
        "origami_sel_eff",
        "n_scored",
        "pick_rank",
        "miss",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = r.to_json()
            for what in ("pick", "origami"):
                sid = row.get(f"{what}_sol_idx")
                sid = None if sid is None else int(sid)
                row[f"{what}_kernel_id"] = kernel_ids.get(sid, "")
                row[f"{what}_signature"] = signatures.get(sid, "")
            w.writerow(row)


def _signatures(
    pools: Mapping[Any, List[Dict[str, Any]]], gemms: List[Dict[str, Any]]
) -> Dict[int, str]:
    """{solution index: signature} of the pools' kernels and the GEMMs'
    measured ones."""
    out: Dict[int, str] = {}
    for kernels in [*pools.values(), *(g["candidates"] for g in gemms)]:
        for k in kernels:
            sid = int(k["sol_idx_global"])
            if sid >= 0 and sid not in out:
                out[sid] = ki.signature_label(k)
    return out


def _evaluate_dataset(
    name: str,
    path: Path,
    out: Path,
    enriched_dir: Path,
    ctx: Dict[str, Any],
    args: argparse.Namespace,
    bank: List[str],
) -> Dict[str, Any]:
    import stage05_train as _train

    cells, _stats = _train.load_enriched_chunks(
        str(enriched_dir),
        args.quiet,
        identity=ctx["identity"],
        aggregation=ctx["aggregation"],
        blocklist=ctx.get("blocklist"),
    )
    gemms = [g for gs in cells.values() for g in gs]
    why = xs.refusal((g["m"], g["n"], g["k"], g["batch_count"]) for g in gemms)
    if why is not None:
        return _fail(name, path, why)
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

    pools = ev.pools_for(gemms, ctx["library"])

    def evaluate(rule: str) -> ev.Evaluation:
        return ev.evaluate_model(
            ctx["model_bytes"],
            gemms,
            hardware=ctx["hardware"],
            pools=pools,
            labels=ctx["labels"],
            tw=ctx["tw"],
            pick_measurement=rule,
        )

    bench = ctx["rebench"]
    if bench is not None:
        bench = replace(
            bench,
            work_dir=out / "rebench",
            lines={**mr.read_lines(_bench_input_yamls(out)), **bench.lines},
        )
    measured = mr.measured_evaluation(
        evaluate, gemms, bench, args.pick_measurement, name
    )
    evaluation = measured.evaluation
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
        "bank_library_stems": bank,
        "library_digest": ctx["identity"].digest if ctx["identity"] else None,
        "n_rows_kernel_not_in_library": _stats.get("rows_kernel_not_in_library", 0),
        "weight_dtype": args.weight_dtype,
        "n_cells_evaluated": len(per_cell),
        "n_routing_mismatch": evaluation.n_routing_mismatch,
        "pick_measurement": evaluation.pick_measurement,
        "model_sel_eff": overall["sel_eff"],
        "n_evaluated": overall["n_evaluated"],
        "n_miss": overall["n_miss"],
        "paired": {
            k: overall["paired"].get(k)
            for k in (
                "n",
                "model_sel_eff",
                "origami_sel_eff",
                "wins",
                "ties",
                "losses",
                "speedup_ci95",
            )
        },
        "overall": overall,
    }
    if measured.record is not None:
        rec = measured.record
        pre = ev.summarize(measured.pre.results, tie_tolerance=args.tie_tolerance)
        summary["pre_rebench"] = mr.pre_rebench(pre, args.pick_measurement)
        summary["rebench"] = mr.record_counts(rec)
        for key in ("n_rebenched", "n_rebench_failed", "n_context_skipped"):
            summary[key] = rec[key]
        (out / mr.RECORD_FILE).write_text(
            json.dumps(_train.jsonable(mr.named_record(rec, ctx["identity"])), indent=2)
            + "\n"
        )
    per_cell_path = out / "per_cell_sel_eff.json"
    per_cell_path.write_text(
        json.dumps(
            _train.jsonable({"per_cell": per_cell, "summary": summary}), indent=2
        )
    )
    _write_per_gemm(
        out / "per_gemm.csv",
        results,
        ctx["identity"].ids if ctx["identity"] is not None else {},
        _signatures(pools, gemms),
    )

    if not args.quiet:
        ref = args.origami_reference
        w = min(max(max((len(r["cell"]) for r in per_cell), default=24), 24), 80)
        ui.info(
            "eval",
            f"{'cell':<{w}} {'n':>6} {'model':>8} {'misses':>6} {'fallback':>8}"
            + (f"  [reference] {'paired':>8} {'origami':>8}  W/T/L" if ref else ""),
        )
        for r in per_cell:
            p = r["paired"]
            ui.info(
                "eval",
                f"{r['cell']:<{w}} {r['n_gemms']:>6d} "
                f"{_train.fmt_pct(r['model_sel_eff'])} {r['n_miss']:>6d} "
                f"{r['n_origami_fallback']:>8d}"
                + (
                    f"  [reference] {_train.fmt_pct(p['model_sel_eff'])} "
                    f"{_train.fmt_pct(p['origami_sel_eff'])}  "
                    f"{p['wins']}/{p['ties']}/{p['losses']}"
                    if ref
                    else ""
                ),
            )
        ui.ok(
            "eval",
            f"{name}: model={_train.fmt_pct(overall['sel_eff'])} over "
            f"{overall['n_evaluated']} GEMMs, {overall['n_miss']} misses",
        )
        if "pre_rebench" in summary:
            pre = summary["pre_rebench"]
            ui.info(
                "eval",
                f"{name}: misses re-benched: {summary['n_rebenched']} measured, "
                f"{summary['n_rebench_failed']} failed, "
                f"{summary['n_context_skipped']} context GEMMs skipped; under "
                f"{pre['pick_measurement']} {_train.fmt_pct(pre['sel_eff'])}",
            )
        p = overall["paired"]
        ci = p.get("speedup_ci95")
        if ref and p["n"]:
            ui.info(
                "eval",
                f"{name}: [reference] Origami on the same {p['n']} GEMMs: "
                f"model={_train.fmt_pct(p['model_sel_eff'])}  "
                f"origami={_train.fmt_pct(p['origami_sel_eff'])}  "
                f"wins/ties/losses={p['wins']}/{p['ties']}/{p['losses']}"
                + (f"  speedup 95% CI=[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""),
            )
        if overall["n_not_evaluable"]:
            ui.warn(
                "eval",
                f"{overall['n_not_evaluable']} GEMMs not evaluated: neither the "
                f"model's pick nor the runtime's fallback pick was measured",
            )
    result = {
        "name": name,
        "path": str(path),
        "rc": 0,
        "per_cell_json": str(per_cell_path),
        "n_evaluated_cells": len(per_cell),
        "model_sel_eff": summary["model_sel_eff"],
        "n_evaluated": summary["n_evaluated"],
        "n_miss": summary["n_miss"],
        "paired_n": summary["paired"]["n"],
        "paired_model_sel_eff": summary["paired"]["model_sel_eff"],
        "paired_origami_sel_eff": summary["paired"]["origami_sel_eff"],
        "summary": summary,
    }
    if "pre_rebench" in summary:
        result["pre_rebench"] = summary["pre_rebench"]
        for key in ("n_rebenched", "n_rebench_failed", "n_context_skipped"):
            result[key] = summary[key]
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
    bank: List[str] = []
    if dat_dir is not None and dat_dir.is_dir():
        try:
            bank = bank_stems(ctx["cfg"], dat_dir, args)
        except (OSError, ValueError, RuntimeError) as e:
            return _fail(name, path, f"cannot read the bank's library files: {e}")
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
        enriched, err = _enrich(name, logs_dir, out, dat_dir, args, bank)
        if err:
            return _fail(name, path, err)
        enriched_dir = enriched
    dropped = unenriched_stems(enriched_dir, bank)
    if dropped:
        return _fail(
            name,
            path,
            f"{enriched_dir} was enriched without the rows of the bank's library "
            f"file(s) {', '.join(dropped)}, which the pool ranks: enrich its logs "
            f"again with them as stage04 --extra-library-stem",
        )
    return _evaluate_dataset(name, path, out, enriched_dir, ctx, args, bank)


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


def bundle_blocklist_of(
    args: argparse.Namespace, cfg: Mapping[str, Any]
) -> Optional[Blocklist]:
    """The blocklist the bundle of --train-dir is evaluated with: its own
    (`lib.correctness.bundle_blocklist`: the sidecar its deploy staged, else
    the blocklist its stage05 applied), as the runtime that serves it leaves
    those kernels out; the config's when the bundle's cannot be read, with
    a warning, as when the config's names other kernels. ValueError when the
    config's cannot be read."""
    invocation = config_blocklist(cfg)
    own, where = bundle_blocklist(args.train_dir, args.library_stem or "")
    mine = frozenset(
        blocklist_key(n) for n in (invocation.canonical if invocation else ())
    )
    if own is None:
        if invocation:
            ui.warn(
                "blocklist",
                f"the bundle's blocklist is unknown ({where}): evaluating with "
                f"the config's ({len(invocation)} kernel(s))",
            )
        return invocation
    if frozenset(blocklist_key(n) for n in own.canonical) != mine and not args.quiet:
        ui.warn(
            "blocklist",
            f"the bundle serves with its own blocklist, {where}: "
            f"{len(own)} kernel(s), the config's {len(mine)}; it is evaluated "
            f"with its own, as the runtime that serves it leaves those out",
        )
    return own


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
        help="run config: `hardware:` block, and `hipblaslt:` and `bench:` "
        "for yaml datasets",
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
    ap.add_argument(
        "--pick-measurement",
        choices=ev.PICK_MEASUREMENTS,
        default="warmup",
        help="warmup: a skip row counts with its warm-up latency and a kernel off the plan's list is a miss; timed: both are misses; any: a skip row counts "
        "and an off-list kernel is passed over (lib/evaluate.py)",
    )
    ap.add_argument(
        "--origami-reference",
        action="store_true",
        help="also print Origami's numbers on the same GEMMs (an external "
        "reference)",
    )
    ap.add_argument("--quiet", action="store_true")
    mr.add_arguments(ap, bench=False)
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
    try:
        blocklist = bundle_blocklist_of(args, cfg)
    except ValueError as e:
        ui.err("blocklist", str(e))
        return 1
    library = (
        ev.library_pool(
            args.library_dir,
            args.library_stem,
            cfg,
            args.hardware_device,
            blocklist=blocklist or Blocklist(),
        )
        if args.library_stem
        else None
    )
    identity = (
        ev.library_identity(
            args.library_dir, args.library_stem, cfg, args.hardware_device
        )
        if args.library_stem
        else None
    )
    try:
        rebench = mr.from_args(args, cfg, args.out_dir / "rebench", hardware.n_cu)
    except ValueError as e:
        ui.err("err", str(e))
        return 1
    ctx = {
        "cfg": cfg,
        "tw": tw,
        "rebench": rebench,
        "hardware": hardware,
        "model_bytes": model_bytes,
        "labels": list(bundle["models"].keys()),
        "library": library,
        "identity": identity,
        "blocklist": blocklist,
        "aggregation": repeats.config_aggregation(cfg),
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
            line = (
                f"{s['name']:20s}  n_cells={s['n_evaluated_cells']}  "
                f"model={_train.fmt_pct(s['model_sel_eff'])} over "
                f"{s['n_evaluated']} GEMMs, {s['n_miss']} misses"
            )
            if args.origami_reference and s["paired_n"]:
                line += (
                    f"  [reference] same {s['paired_n']} GEMMs: model="
                    f"{_train.fmt_pct(s['paired_model_sel_eff'])} origami="
                    f"{_train.fmt_pct(s['paired_origami_sel_eff'])}"
                )
            ui.ok("dataset", line)
        ui.ok("done", f"summary.json -> {out_json}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
