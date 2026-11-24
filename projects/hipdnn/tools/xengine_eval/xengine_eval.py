#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Cross-engine heuristic evaluation driver and report (schema "hipdnn.xengine_eval/1").

Subcommands
  run     drive the C++ harness (hipdnn_xengine_eval) over a graph corpus and append one
          JSON row per graph to <out>/rows.jsonl (resumable, shardable).
  report  summarise one or more rows.jsonl files (several arches allowed) into report.json
          and an optional markdown file.

Standard library only; Python 3.10+.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

SCHEMA = "hipdnn.xengine_eval/1"
RUN_SCHEMA = "hipdnn.xengine_eval.run/1"
REPORT_SCHEMA = "hipdnn.xengine_eval.report/1"

POLICY_ORDER = ["modeA", "static", "default"]
GRAPH_SUFFIXES = (".json", ".fb")
# Non-graph files that live beside corpus_gen / bundle graphs.
NON_GRAPH_NAMES = {"manifest.json", "eval_slice.json", "support.json", "sweep.json", "run.json"}
NON_GRAPH_SUFFIXES = (".meta.json", ".support.json", ".sweep.json", ".problems.json")
# Slack (seconds) added to the per-process timeout for process start-up and plugin loading.
TIMEOUT_SLACK_S = 120.0


# ----------------------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------------------


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _norm(path: str) -> str:
    return os.path.normpath(os.path.abspath(path))


def collect_graphs(source: str) -> list[str]:
    """Sorted absolute graph paths from a directory (*.json, *.fb) or a list file (one path per line)."""
    src = Path(source)
    if src.is_dir():
        found = []
        for root, _dirs, files in os.walk(src):
            for name in files:
                low = name.lower()
                if not low.endswith(GRAPH_SUFFIXES):
                    continue
                if low in NON_GRAPH_NAMES or low.endswith(NON_GRAPH_SUFFIXES):
                    continue
                found.append(_norm(os.path.join(root, name)))
        return sorted(found)
    if src.is_file():
        paths = []
        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                paths.append(_norm(line))
        return sorted(set(paths))
    raise SystemExit(f"--graphs: {source} is neither a directory nor a file")


def parse_shard(text: Optional[str]) -> tuple[int, int]:
    if text is None:
        return 0, 1
    try:
        k_text, n_text = text.split("/")
        k, n = int(k_text), int(n_text)
    except ValueError:
        raise SystemExit(f"--shard expects K/N (e.g. 0/4), got '{text}'")
    if n < 1 or not 0 <= k < n:
        raise SystemExit(f"--shard K/N needs N >= 1 and 0 <= K < N (K is 0-based), got '{text}'")
    return k, n


def load_done(rows_path: Path) -> set[str]:
    """graph_paths already in rows.jsonl; repairs a torn final line left by a killed driver."""
    done: set[str] = set()
    if not rows_path.exists():
        return done
    with rows_path.open("rb") as fp:
        data = fp.read()
    if data and not data.endswith(b"\n"):
        cut = data.rfind(b"\n") + 1
        print(f"warning: {rows_path} ends in a partial line; dropping it", file=sys.stderr)
        with rows_path.open("wb") as fp:
            fp.write(data[:cut])
        data = data[:cut]
    for raw in data.splitlines():
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("graph_path"):
            done.add(_norm(row["graph_path"]))
    return done


def _describe_exit(rc: int) -> str:
    if rc < 0:
        try:
            return f"killed by signal {signal.Signals(-rc).name}"
        except ValueError:
            return f"killed by signal {-rc}"
    return f"exit code {rc}"


class Runner:
    def __init__(self, args: argparse.Namespace, out_dir: Path):
        self.args = args
        self.out_dir = out_dir
        self.rows_fp = (out_dir / "rows.jsonl").open("a", encoding="utf-8")
        self.stderr_fp = (out_dir / "harness_stderr.log").open("a", encoding="utf-8")
        self.done = load_done(out_dir / "rows.jsonl")
        self.arch: Optional[str] = None
        self.device_name: Optional[str] = None
        self.ranking_metric: Optional[str] = args.ranking_metric
        self.n_rows_written = 0
        self.n_crashed = 0
        self.n_invalid = 0

    def close(self) -> None:
        self.rows_fp.close()
        self.stderr_fp.close()

    # -- harness invocation ---------------------------------------------------------------

    def harness_command(self, list_file: str) -> list[str]:
        a = self.args
        cmd = [a.bin, "--graph-list", list_file]
        for d in a.plugin_dir or []:
            cmd += ["--plugin-dir", d]
        for flag, value in (
            ("--policies", a.policies),
            ("--latency-repeats", a.latency_repeats),
            ("--warmup", a.warmup),
            ("--max-iterations", a.max_iterations),
            ("--stability", a.stability),
            ("--ranking-metric", a.ranking_metric),
            ("--graph-timeout-s", a.graph_timeout_s),
        ):
            if value is not None:
                cmd += [flag, str(value)]
        if a.no_autotune:
            cmd.append("--no-autotune")
        if a.no_immediate:
            cmd.append("--no-immediate")
        return cmd

    def harness_env(self, cache_dir: str) -> dict[str, str]:
        env = dict(os.environ)
        env["HIPDNN_DISABLE_EXACT_ENGINE_CACHE"] = "1"
        env["HIPDNN_CACHE_DIR"] = cache_dir
        # MIOpen's find results (user db) would carry one process's autotune into the next
        # process's immediate picks, so each process gets an empty one. Its compiled-kernel
        # cache only saves compile time: shared when the caller sets MIOPEN_CUSTOM_CACHE_DIR.
        # Both default under ~/.cache or /tmp/.cache, which may belong to another user.
        env["MIOPEN_USER_DB_PATH"] = os.path.join(cache_dir, "miopen-udb")
        env.setdefault("MIOPEN_CUSTOM_CACHE_DIR", os.path.join(cache_dir, "miopen-kcache"))
        # The harness forces the policy order per query; an inherited value must not leak in.
        env.pop("HIPDNN_HEUR_POLICY_ORDER", None)
        if self.args.ranking_metric is None:
            env.pop("HIPDNN_HEUR_RANKING_METRIC", None)
        return env

    def record(self, row: dict[str, Any]) -> None:
        self.rows_fp.write(json.dumps(row, separators=(",", ":")) + "\n")
        self.rows_fp.flush()
        os.fsync(self.rows_fp.fileno())
        self.n_rows_written += 1
        if row.get("arch"):
            self.arch = row["arch"]
        if row.get("device_name"):
            self.device_name = row["device_name"]
        if row.get("ranking_metric"):
            self.ranking_metric = row["ranking_metric"]
        if row.get("is_valid") is False:
            self.n_invalid += 1
        self.done.add(_norm(row["graph_path"]))

    def run_process(self, graphs: list[str]) -> tuple[int, bool]:
        """One harness process over `graphs`. Streams rows to rows.jsonl as they arrive.

        Returns (exit code, timed_out). A fresh, empty HIPDNN_CACHE_DIR is used per process.
        """
        cache_dir = tempfile.mkdtemp(prefix="hipdnn_xeval_cache_")
        fd, list_file = tempfile.mkstemp(prefix="hipdnn_xeval_graphs_", suffix=".txt")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fp:
                fp.write("\n".join(graphs) + "\n")
            cmd = self.harness_command(list_file)
            self.stderr_fp.write(f"\n=== {_now()} {' '.join(cmd)} ({len(graphs)} graphs)\n")
            self.stderr_fp.flush()
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=self.stderr_fp,
                env=self.harness_env(cache_dir),
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            timed_out = threading.Event()
            timer = None
            if self.args.graph_timeout_s:
                limit = float(self.args.graph_timeout_s) * len(graphs) + TIMEOUT_SLACK_S

                def kill() -> None:
                    timed_out.set()
                    proc.kill()

                timer = threading.Timer(limit, kill)
                timer.daemon = True
                timer.start()
            expected = {_norm(g) for g in graphs}
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        self.stderr_fp.write(f"[stdout] {line}\n")
                        continue
                    if not isinstance(row, dict) or not str(row.get("schema", "")).startswith(
                        "hipdnn.xengine_eval/"
                    ):
                        self.stderr_fp.write(f"[stdout] {line}\n")
                        continue
                    if not row.get("graph_path"):
                        self.stderr_fp.write(f"[stdout, no graph_path] {line}\n")
                        continue
                    if _norm(row["graph_path"]) not in expected:
                        print(
                            f"warning: harness returned unexpected graph {row['graph_path']}",
                            file=sys.stderr,
                        )
                    self.record(row)
                rc = proc.wait()
            except BaseException:
                proc.kill()
                proc.wait()
                raise
            finally:
                if timer is not None:
                    timer.cancel()
            return rc, timed_out.is_set()
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)
            try:
                os.unlink(list_file)
            except OSError:
                pass

    def crash_row(self, graph: str, reason: str) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "graph_path": graph,
            "graph_key": None,
            "graph_id": None,
            "arch": self.arch,
            "device_name": self.device_name,
            "ranking_metric": self.ranking_metric or "tflops",
            "is_valid": False,
            "skip_reason": reason,
            "engines": [],
            "selection": {},
            "autotune": None,
            "driver_generated": True,
        }

    def run_chunk(self, chunk: list[str]) -> None:
        rc, timed_out = self.run_process(chunk)
        remaining = [g for g in chunk if _norm(g) not in self.done]
        if not remaining:
            if rc != 0:
                print(f"warning: harness {_describe_exit(rc)} after emitting every row", file=sys.stderr)
            return
        why = "timed out" if timed_out else _describe_exit(rc)
        print(
            f"chunk process {why} with {len(remaining)}/{len(chunk)} graphs unreported; "
            "rerunning them one per process",
            file=sys.stderr,
        )
        for graph in remaining:
            rc, timed_out = self.run_process([graph])
            if _norm(graph) in self.done:
                continue
            reason = (
                f"harness timed out when run alone ({self.args.graph_timeout_s}s per-graph limit)"
                if timed_out
                else f"harness crashed when run alone ({_describe_exit(rc)})"
            )
            self.n_crashed += 1
            self.record(self.crash_row(graph, reason))


def write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def cmd_run(args: argparse.Namespace) -> int:
    if args.chunk < 1:
        raise SystemExit("--chunk must be >= 1")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    k, n = parse_shard(args.shard)

    all_graphs = collect_graphs(args.graphs)
    shard = all_graphs[k::n]

    run_json_path = out_dir / "run.json"
    if run_json_path.exists():
        run_meta = json.loads(run_json_path.read_text(encoding="utf-8"))
    else:
        run_meta = {
            "schema": RUN_SCHEMA,
            "host": platform.node(),
            "platform": platform.platform(),
            "arch": None,
            "device_name": None,
            "source_sha": args.source_sha,
            "invocations": [],
        }
    if args.source_sha:
        run_meta["source_sha"] = args.source_sha
    invocation: dict[str, Any] = {
        "argv": sys.argv,
        "start": _now(),
        "end": None,
        "status": "running",
        "bin": args.bin,
        "graphs": args.graphs,
        "shard": f"{k}/{n}",
        "chunk": args.chunk,
        "n_graphs_total": len(all_graphs),
        "n_graphs_in_shard": len(shard),
    }
    run_meta["invocations"].append(invocation)
    write_json_atomic(run_json_path, run_meta)

    runner = Runner(args, out_dir)
    pending = [g for g in shard if _norm(g) not in runner.done]
    invocation["n_graphs_skipped_resume"] = len(shard) - len(pending)
    print(
        f"{len(all_graphs)} graphs, shard {k}/{n} = {len(shard)}, "
        f"{len(shard) - len(pending)} already done, {len(pending)} to run",
        file=sys.stderr,
    )
    status = "completed"
    try:
        for start in range(0, len(pending), args.chunk):
            chunk = pending[start : start + args.chunk]
            runner.run_chunk(chunk)
            print(
                f"[{min(start + args.chunk, len(pending))}/{len(pending)}] rows written: "
                f"{runner.n_rows_written} (invalid {runner.n_invalid}, crashed {runner.n_crashed})",
                file=sys.stderr,
            )
    except KeyboardInterrupt:
        status = "interrupted"
    except BaseException:
        status = "error"
        raise
    finally:
        runner.close()
        invocation["end"] = _now()
        invocation["status"] = status
        invocation["n_rows_written"] = runner.n_rows_written
        invocation["n_invalid"] = runner.n_invalid
        invocation["n_crashed"] = runner.n_crashed
        run_meta["arch"] = runner.arch or run_meta.get("arch")
        run_meta["device_name"] = runner.device_name or run_meta.get("device_name")
        write_json_atomic(run_json_path, run_meta)
    return 0 if status == "completed" else 130


# ----------------------------------------------------------------------------------------
# report: helpers
# ----------------------------------------------------------------------------------------

NODE_FAMILY = {
    "ConvolutionFwdAttributes": "conv_fwd",
    "ConvolutionBwdAttributes": "conv_bwd_data",
    "ConvolutionWrwAttributes": "conv_bwd_weights",
    "SdpaAttributes": "sdpa_fwd",
    "SdpaBackwardAttributes": "sdpa_bwd",
    "BatchnormAttributes": "batchnorm_fwd_training",
    "BatchnormInferenceAttributes": "batchnorm_inference",
    "BatchnormInferenceAttributesVarianceExt": "batchnorm_inference",
    "BatchnormBackwardAttributes": "batchnorm_bwd",
    "LayernormAttributes": "layernorm_fwd",
    "LayernormBackwardAttributes": "layernorm_bwd",
    "RMSNormAttributes": "rmsnorm_fwd",
    "RMSNormBackwardAttributes": "rmsnorm_bwd",
    "MatmulAttributes": "matmul",
    "BlockScaleDequantizeAttributes": "block_scale_dequantize",
    "ResampleFwdAttributes": "resample_fwd",
    "ResampleBwdAttributes": "resample_bwd",
    "ReductionAttributes": "reduction",
    "PointwiseAttributes": "pointwise",
}


def family_from_graph_json(path: str) -> Optional[str]:
    """Operation family from the node types of a JSON graph; None if unreadable / no nodes."""
    if not path.lower().endswith(".json"):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fp:
            graph = json.load(fp)
    except (OSError, ValueError):
        return None
    nodes = graph.get("nodes") if isinstance(graph, dict) else None
    if not isinstance(nodes, list) or not nodes:
        return None
    anchors: list[str] = []
    has_pointwise = False
    for node in nodes:
        node_type = node.get("type") if isinstance(node, dict) else None
        if not isinstance(node_type, str):
            continue
        fam = NODE_FAMILY.get(node_type, node_type)
        if fam == "pointwise":
            has_pointwise = True
        elif fam not in anchors:
            anchors.append(fam)
    if not anchors:
        return "pointwise" if has_pointwise else None
    return "+".join(anchors) + ("+pointwise" if has_pointwise else "")


def family_from_path(path: str) -> str:
    """corpus_gen writes <op>/graphs/*.fb, so skip a 'graphs' directory; else the parent name."""
    parent = Path(path).parent
    if parent.name == "graphs" and parent.parent.name:
        return parent.parent.name
    return parent.name or "unknown"


_family_cache: dict[str, str] = {}


def operation_family(row: dict[str, Any]) -> str:
    explicit = row.get("operation")
    if isinstance(explicit, str) and explicit:
        return explicit
    path = row.get("graph_path") or ""
    if path not in _family_cache:
        _family_cache[path] = family_from_graph_json(path) or family_from_path(path)
    return _family_cache[path]


def positive(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value > 0 else None


def time_ms(d: Optional[dict[str, Any]], robust: str, avg: str = "avg_time_ms") -> Optional[float]:
    """Robust mean in ms, falling back to the average; None when absent or non-positive."""
    if not isinstance(d, dict):
        return None
    value = positive(d.get(robust))
    return value if value is not None else positive(d.get(avg))


def percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0,100]) of an ascending list."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def describe(values: Iterable[float], tol: Optional[float] = None, geomean_1p: bool = False) -> dict[str, Any]:
    vals = sorted(v for v in values if v is not None and math.isfinite(v))
    out: dict[str, Any] = {
        "n": len(vals),
        "mean": None,
        "p50": None,
        "p90": None,
        "p95": None,
        "max": None,
    }
    if tol is not None:
        out["frac_gt_tol"] = None
    if geomean_1p:
        out["geomean_1p"] = None
    if not vals:
        return out
    out["mean"] = sum(vals) / len(vals)
    out["p50"] = percentile(vals, 50)
    out["p90"] = percentile(vals, 90)
    out["p95"] = percentile(vals, 95)
    out["max"] = vals[-1]
    if tol is not None:
        out["frac_gt_tol"] = sum(1 for v in vals if v > tol) / len(vals)
    if geomean_1p:
        out["geomean_1p"] = math.exp(sum(math.log(1.0 + v) for v in vals) / len(vals))
    return out


def rate(hits: list[bool]) -> dict[str, Any]:
    return {"n": len(hits), "rate": (sum(hits) / len(hits)) if hits else None}


ENGINE_BUCKETS = [("1-2", 1, 2), ("3-5", 3, 5), ("6-10", 6, 10), (">10", 11, 10**9)]


def engine_bucket(n_engines: int) -> str:
    for label, lo, hi in ENGINE_BUCKETS:
        if lo <= n_engines <= hi:
            return label
    return "0"


# ----------------------------------------------------------------------------------------
# report: per-graph records
# ----------------------------------------------------------------------------------------


def build_record(row: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Reduce one valid harness row to the quantities the metrics need."""
    engines = []
    for e in row.get("engines") or []:
        imm = e.get("immediate") or {}
        imm_ms = None
        if imm.get("is_valid") is not False and not imm.get("error"):
            imm_ms = time_ms(imm, "robustMeanMs")
        pred = e.get("prediction") or {}
        pred_val = positive(pred.get("value")) if pred.get("status") == "available" else None
        engines.append(
            {
                "id": e.get("engine_id"),
                "name": e.get("engine_name") or str(e.get("engine_id")),
                "imm_ms": imm_ms,
                "pred_status": pred.get("status"),
                "pred": pred_val,
            }
        )
    timed = [e for e in engines if e["imm_ms"] is not None]
    imm_best = min(timed, key=lambda e: e["imm_ms"]) if timed else None
    imm_by_id = {e["id"]: e["imm_ms"] for e in timed}

    auto = row.get("autotune") or {}
    auto_ms = None
    auto_id = None
    auto_name = None
    if not auto.get("error"):
        best = auto.get("best")
        if isinstance(best, dict):
            auto_id = best.get("engine_id")
            auto_name = best.get("engine_name")
            auto_ms = time_ms(best, "robust_time_ms")
            if auto_ms is None:
                for r in auto.get("results") or []:
                    if r.get("engine_id") == auto_id and r.get("succeeded"):
                        auto_ms = time_ms(r, "robust_time_ms")
                        break
    if auto_ms is None:
        auto_id = auto_name = None

    per_policy: dict[str, dict[str, Any]] = {}
    for policy, sel in (row.get("selection") or {}).items():
        if not isinstance(sel, dict) or sel.get("error"):
            continue
        execution = sel.get("execution") if isinstance(sel.get("execution"), dict) else None
        exec_ms = None
        if execution and not execution.get("error"):
            exec_ms = time_ms(execution, "robustMeanMs")
        pick_id = sel.get("picked_engine_id")
        warm = sel.get("warm_us") if isinstance(sel.get("warm_us"), dict) else {}
        p: dict[str, Any] = {
            "pick_id": pick_id,
            "pick_name": sel.get("picked_engine_name"),
            "exec_id": execution.get("engine_id") if execution else None,
            "exec_ms": exec_ms,
            "cold_us": positive(sel.get("cold_us")),
            "warm_us": positive(warm.get("median")),
            "first_query": bool(sel.get("first_query_in_process")),
            "regret_auto": None,
            "sel_regret": None,
            "hit_argmin": None,
            "hit_auto": None,
        }
        if exec_ms is not None and auto_ms is not None:
            p["regret_auto"] = exec_ms / auto_ms - 1.0
        if pick_id is not None:
            pick_imm = imm_by_id.get(pick_id)
            if pick_imm is not None and imm_best is not None:
                p["sel_regret"] = pick_imm / imm_best["imm_ms"] - 1.0
            if imm_best is not None:
                p["hit_argmin"] = pick_id == imm_best["id"]
            if auto_id is not None:
                p["hit_auto"] = pick_id == auto_id
        per_policy[policy] = p

    return {
        "arch": row.get("arch") or "unknown",
        "family": operation_family(row),
        "path": row.get("graph_path"),
        "metric": row.get("ranking_metric") or "tflops",
        "flops": positive(row.get("flops")),
        "n_engines": len(engines),
        "engines": engines,
        "competing": len(timed) >= 2,
        "imm_best_ms": imm_best["imm_ms"] if imm_best else None,
        "imm_best_id": imm_best["id"] if imm_best else None,
        "imm_best_name": imm_best["name"] if imm_best else None,
        "auto_ms": auto_ms,
        "auto_id": auto_id,
        "auto_name": auto_name,
        "headroom": (imm_best["imm_ms"] / auto_ms - 1.0) if imm_best and auto_ms else None,
        "policies": per_policy,
    }


def calibration(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """Per engine, median predicted/measured (tflops metric) or measured/predicted (time metric).

    With the graph's flop count (optional row field "flops") the ratio is absolute. Without it
    the ratio is normalised within the graph so that its geometric mean over the engines that
    have both a prediction and a measurement is 1: it then says which engines the L1 favours
    relative to their peers (>1 over-predicts the engine, <1 under-predicts it).
    """
    absolute: dict[str, list[float]] = defaultdict(list)
    relative: dict[str, list[float]] = defaultdict(list)
    for rec in recs:
        both = [e for e in rec["engines"] if e["imm_ms"] is not None and e["pred"] is not None]
        if not both:
            continue
        tflops_metric = rec["metric"] != "time"
        if rec["flops"] is not None:
            for e in both:
                if tflops_metric:
                    measured = rec["flops"] / (e["imm_ms"] * 1e9)  # TFLOPS
                    absolute[e["name"]].append(e["pred"] / measured)
                else:
                    absolute[e["name"]].append(e["imm_ms"] / e["pred"])
        elif len(both) >= 2:
            raw = [
                (e["pred"] * e["imm_ms"]) if tflops_metric else (e["imm_ms"] / e["pred"])
                for e in both
            ]
            gm = math.exp(sum(math.log(r) for r in raw) / len(raw))
            for e, r in zip(both, raw):
                relative[e["name"]].append(r / gm)

    def stat(values: list[float]) -> dict[str, Any]:
        vals = sorted(values)
        return {
            "n": len(vals),
            "median": percentile(vals, 50),
            "p10": percentile(vals, 10),
            "p90": percentile(vals, 90),
        }

    out: dict[str, Any] = {}
    for name in sorted(set(absolute) | set(relative)):
        entry: dict[str, Any] = {}
        if name in absolute:
            entry["absolute"] = stat(absolute[name])
        if name in relative:
            entry["relative"] = stat(relative[name])
        out[name] = entry
    return out


def latency_block(recs: list[dict[str, Any]], policies: list[str]) -> dict[str, Any]:
    """Cold / warm latency per policy, ModeA - static overhead, and the engine-count split.

    The first query of a process includes one-off model loading; those are reported apart
    (`first_in_process_cold_us`) so they do not distort the cold distribution.
    """

    def per_policy(subset: list[dict[str, Any]]) -> dict[str, Any]:
        out = {}
        for policy in policies:
            cold, warm, first = [], [], []
            for rec in subset:
                p = rec["policies"].get(policy)
                if not p:
                    continue
                if p["cold_us"] is not None:
                    (first if p["first_query"] else cold).append(p["cold_us"])
                if p["warm_us"] is not None:
                    warm.append(p["warm_us"])
            out[policy] = {
                "cold_us": describe(cold),
                "warm_median_us": describe(warm),
                "first_in_process_cold_us": describe(first),
            }
        return out

    block: dict[str, Any] = per_policy(recs)
    overhead: dict[str, Any] = {}
    if "modeA" in policies and "static" in policies:
        cold_d, warm_d = [], []
        for rec in recs:
            a, s = rec["policies"].get("modeA"), rec["policies"].get("static")
            if not a or not s:
                continue
            if a["warm_us"] is not None and s["warm_us"] is not None:
                warm_d.append(a["warm_us"] - s["warm_us"])
            if a["cold_us"] is not None and s["cold_us"] is not None and not a["first_query"] and not s["first_query"]:
                cold_d.append(a["cold_us"] - s["cold_us"])
        overhead = {"warm_median_us": describe(warm_d), "cold_us": describe(cold_d)}
    by_bucket: dict[str, Any] = {}
    for label, _lo, _hi in ENGINE_BUCKETS:
        subset = [r for r in recs if engine_bucket(r["n_engines"]) == label]
        by_bucket[label] = {"n_graphs": len(subset), "policies": per_policy(subset)}
    return {"policies": block, "modeA_minus_static": overhead, "by_engine_count": by_bucket}


def summarize(recs: list[dict[str, Any]], tol: float, policies: list[str]) -> dict[str, Any]:
    """All metrics for one group of per-graph records (competing graphs unless noted)."""
    competing = [r for r in recs if r["competing"]]
    out: dict[str, Any] = {
        "n_graphs_valid": len(recs),
        "n_competing": len(competing),
        "autotune_headroom": describe(
            (r["headroom"] for r in competing if r["headroom"] is not None), tol, True
        ),
        "regret_vs_autotune": {},
        "selection_regret": {},
        "hit_rates": {},
        "pick_exec_mismatch": {},
    }
    for policy in policies:
        ps = [(r, r["policies"].get(policy)) for r in competing]
        ps = [(r, p) for r, p in ps if p]
        out["regret_vs_autotune"][policy] = describe(
            (p["regret_auto"] for _r, p in ps if p["regret_auto"] is not None), tol, True
        )
        out["selection_regret"][policy] = describe(
            (p["sel_regret"] for _r, p in ps if p["sel_regret"] is not None), tol, True
        )
        out["hit_rates"][policy] = {
            "pick_is_argmin_immediate": rate([p["hit_argmin"] for _r, p in ps if p["hit_argmin"] is not None]),
            "pick_is_autotune_best": rate([p["hit_auto"] for _r, p in ps if p["hit_auto"] is not None]),
        }
        out["pick_exec_mismatch"][policy] = sum(
            1 for _r, p in ps if p["exec_id"] is not None and p["pick_id"] is not None and p["exec_id"] != p["pick_id"]
        )
    out["latency_competing"] = latency_block(competing, policies)
    out["latency_all_valid"] = latency_block(recs, policies)
    out["l1_calibration"] = calibration(competing)

    full = [r for r in competing if all(e["pred"] is not None for e in r["engines"])]
    none_avail = [r for r in competing if all(e["pred"] is None for e in r["engines"])]
    n = len(competing)
    out["coverage"] = {
        "n_competing": n,
        "all_engines_have_prediction": len(full),
        "frac_all_engines_have_prediction": (len(full) / n) if n else None,
        "no_engine_has_prediction": len(none_avail),
        "frac_no_engine_has_prediction": (len(none_avail) / n) if n else None,
    }
    return out


# ----------------------------------------------------------------------------------------
# report: driver
# ----------------------------------------------------------------------------------------


def load_rows(paths: list[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """All rows of all files; later duplicates of (arch, graph_key|graph_path) replace earlier ones."""
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    stats = {"lines": 0, "bad_lines": 0, "duplicates": 0}
    for path in paths:
        with open(path, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                stats["lines"] += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    stats["bad_lines"] += 1
                    continue
                if not isinstance(row, dict) or not str(row.get("schema", "")).startswith("hipdnn.xengine_eval/"):
                    stats["bad_lines"] += 1
                    continue
                key = (str(row.get("arch")), str(row.get("graph_key") or row.get("graph_path")))
                if key in by_key:
                    stats["duplicates"] += 1
                by_key[key] = row
    return list(by_key.values()), stats


def worst_graphs(recs: list[dict[str, Any]], policies: list[str], limit: int = 20) -> list[dict[str, Any]]:
    scored = [
        r for r in recs if r["competing"] and r["policies"].get("modeA") and r["policies"]["modeA"]["regret_auto"] is not None
    ]
    scored.sort(key=lambda r: r["policies"]["modeA"]["regret_auto"], reverse=True)
    out = []
    for r in scored[:limit]:
        out.append(
            {
                "arch": r["arch"],
                "family": r["family"],
                "graph_path": r["path"],
                "regret_vs_autotune_modeA": r["policies"]["modeA"]["regret_auto"],
                "autotune_best": {"engine": r["auto_name"], "ms": r["auto_ms"]},
                "immediate_best": {"engine": r["imm_best_name"], "ms": r["imm_best_ms"]},
                "engines": [
                    {"name": e["name"], "immediate_ms": e["imm_ms"], "predicted": e["pred"], "prediction_status": e["pred_status"]}
                    for e in r["engines"]
                ],
                "picks": {
                    pol: {
                        "engine": p["pick_name"],
                        "exec_ms": p["exec_ms"],
                        "regret_vs_autotune": p["regret_auto"],
                    }
                    for pol, p in r["policies"].items()
                    if pol in policies
                },
            }
        )
    return out


def cmd_report(args: argparse.Namespace) -> int:
    tol = args.tolerance
    rows, load_stats = load_rows(args.rows)

    recs: list[dict[str, Any]] = []
    n_invalid = 0
    invalid_reasons: dict[str, int] = defaultdict(int)
    for row in rows:
        if row.get("is_valid") is False:
            n_invalid += 1
            invalid_reasons[str(row.get("skip_reason") or "unspecified")[:120]] += 1
            continue
        rec = build_record(row)
        if rec is not None:
            recs.append(rec)

    seen_policies = {p for r in recs for p in r["policies"]}
    policies = [p for p in POLICY_ORDER if p in seen_policies] + sorted(seen_policies - set(POLICY_ORDER))

    arches = sorted({r["arch"] for r in recs})
    families = sorted({r["family"] for r in recs})

    def groups_for(subset: list[dict[str, Any]]) -> dict[str, Any]:
        result = {"ALL": summarize(subset, tol, policies)}
        for fam in sorted({r["family"] for r in subset}):
            result[fam] = summarize([r for r in subset if r["family"] == fam], tol, policies)
        return result

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "generated": _now(),
        "inputs": [str(p) for p in args.rows],
        "tolerance": tol,
        "policies": policies,
        "definitions": {
            "competing_graph": "valid row with >= 2 engines that have a valid immediate time",
            "regret_vs_autotune": "exec_ms(policy) / autotune_best_ms - 1 (not clamped)",
            "selection_regret": "immediate_ms(pick) / min_e immediate_ms(e) - 1",
            "autotune_headroom": "min_e immediate_ms(e) / autotune_best_ms - 1",
            "stats": "mean, linear-interpolated percentiles, max; frac_gt_tol = fraction of values > tolerance; geomean_1p = geomean of (1 + value)",
            "timing": "robustMeanMs (immediate/execution) and robust_time_ms (autotune), falling back to the average",
        },
        "load": {**load_stats, "rows": len(rows), "invalid_rows": n_invalid, "invalid_reasons": dict(invalid_reasons)},
        "counts": {
            "valid_graphs": len(recs),
            "competing_graphs": sum(1 for r in recs if r["competing"]),
            "excluded_non_competing": sum(1 for r in recs if not r["competing"]),
        },
        "arches": arches,
        "families": families,
        "by_arch": {arch: groups_for([r for r in recs if r["arch"] == arch]) for arch in arches},
        "overall": groups_for(recs),
        "worst_by_regret_vs_autotune_modeA": worst_graphs(recs, policies),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(render_markdown(report), encoding="utf-8")
    c = report["counts"]
    print(
        f"{len(rows)} rows, {n_invalid} invalid, {c['valid_graphs']} valid, "
        f"{c['competing_graphs']} competing -> {out_path}",
        file=sys.stderr,
    )
    return 0


# ----------------------------------------------------------------------------------------
# report: markdown
# ----------------------------------------------------------------------------------------


def _pct(v: Optional[float]) -> str:
    return "-" if v is None else f"{100.0 * v:.1f}%"


def _us(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.1f}"


def _num(v: Optional[float], digits: int = 3) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def _md_table(header: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines) + "\n"


def summary_table(groups: dict[str, Any], policies: list[str], tol: float) -> str:
    header = [
        "op family", "policy", "n competing",
        "regret mean", "p50", "p95", "max", f">{int(tol * 100)}%",
        "sel.regret mean", "sel.regret p95",
        "hit argmin", "hit autotune",
        "cold p50 us", "cold p95 us", "warm p50 us", "warm p95 us",
    ]
    body = []
    names = ["ALL"] + [f for f in groups if f != "ALL"]
    for fam in names:
        g = groups[fam]
        if g["n_competing"] == 0:
            continue
        for pol in policies:
            reg = g["regret_vs_autotune"][pol]
            sel = g["selection_regret"][pol]
            hit = g["hit_rates"][pol]
            lat = g["latency_competing"]["policies"][pol]
            body.append(
                [
                    fam, pol, str(g["n_competing"]),
                    _pct(reg["mean"]), _pct(reg["p50"]), _pct(reg["p95"]), _pct(reg["max"]), _pct(reg["frac_gt_tol"]),
                    _pct(sel["mean"]), _pct(sel["p95"]),
                    _pct(hit["pick_is_argmin_immediate"]["rate"]), _pct(hit["pick_is_autotune_best"]["rate"]),
                    _us(lat["cold_us"]["p50"]), _us(lat["cold_us"]["p95"]),
                    _us(lat["warm_median_us"]["p50"]), _us(lat["warm_median_us"]["p95"]),
                ]
            )
    return _md_table(header, body) if body else "_no competing graphs_\n"


def headroom_table(groups: dict[str, Any], tol: float) -> str:
    body = []
    for fam in ["ALL"] + [f for f in groups if f != "ALL"]:
        g = groups[fam]
        h = g["autotune_headroom"]
        c = g["coverage"]
        if g["n_competing"] == 0:
            continue
        body.append(
            [
                fam, str(g["n_graphs_valid"]), str(g["n_competing"]),
                _pct(h["mean"]), _pct(h["p50"]), _pct(h["p95"]), _pct(h["max"]), _pct(h["frac_gt_tol"]),
                _pct(c["frac_all_engines_have_prediction"]), _pct(c["frac_no_engine_has_prediction"]),
            ]
        )
    if not body:
        return ""
    return _md_table(
        [
            "op family", "valid graphs", "competing", "headroom mean", "p50", "p95", "max", f">{int(tol * 100)}%",
            "all engines predicted", "no engine predicted",
        ],
        body,
    )


def latency_split_table(group: dict[str, Any], policies: list[str]) -> str:
    body = []
    for label, entry in group["latency_all_valid"]["by_engine_count"].items():
        if entry["n_graphs"] == 0:
            continue
        for pol in policies:
            lat = entry["policies"][pol]
            body.append(
                [
                    label, pol, str(entry["n_graphs"]),
                    _us(lat["cold_us"]["p50"]), _us(lat["cold_us"]["p95"]),
                    _us(lat["warm_median_us"]["p50"]), _us(lat["warm_median_us"]["p95"]), _us(lat["warm_median_us"]["max"]),
                ]
            )
    over = group["latency_all_valid"]["modeA_minus_static"]
    text = ""
    if body:
        text += _md_table(
            ["engines", "policy", "n graphs", "cold p50 us", "cold p95 us", "warm p50 us", "warm p95 us", "warm max us"],
            body,
        )
    if over:
        w = over["warm_median_us"]
        text += f"\nModeA - static warm overhead (us, n={w['n']}): mean {_us(w['mean'])}, p50 {_us(w['p50'])}, p95 {_us(w['p95'])}, max {_us(w['max'])}\n"
    return text


def calibration_table(group: dict[str, Any]) -> str:
    body = []
    for name, entry in group["l1_calibration"].items():
        for mode in ("absolute", "relative"):
            s = entry.get(mode)
            if s:
                body.append([name, mode, str(s["n"]), _num(s["median"]), _num(s["p10"]), _num(s["p90"])])
    return _md_table(["engine", "mode", "n", "median", "p10", "p90"], body) if body else ""


def render_markdown(report: dict[str, Any]) -> str:
    tol = report["tolerance"]
    policies = report["policies"]
    c = report["counts"]
    ld = report["load"]
    md = ["# Cross-engine heuristic evaluation\n"]
    md.append(
        f"Inputs: {', '.join(report['inputs'])}\n\n"
        f"Rows {ld['rows']} (invalid {ld['invalid_rows']}, duplicates replaced {ld['duplicates']}); "
        f"valid graphs {c['valid_graphs']}, competing graphs {c['competing_graphs']} "
        f"(>= 2 engines with a valid immediate time). Tolerance {tol:.0%}.\n\n"
        "regret = exec_ms(policy) / autotune_best_ms - 1 (negative = beat autotune); "
        "sel.regret = immediate_ms(pick) / min immediate - 1. "
        "Cold/warm = selection latency over competing graphs (microseconds; first query in each process excluded from cold).\n"
    )
    sections = [(f"Arch {arch}", report["by_arch"][arch]) for arch in report["arches"]]
    sections.append(("Overall (all arches)", report["overall"]))
    for title, groups in sections:
        md.append(f"\n## {title}\n")
        md.append(summary_table(groups, policies, tol))
        head = headroom_table(groups, tol)
        if head:
            md.append("\n### Autotune headroom and L1 coverage\n\n" + head)
        all_group = groups["ALL"]
        split = latency_split_table(all_group, policies)
        if split:
            md.append("\n### Selection latency by number of applicable engines (all valid graphs)\n\n" + split)
        cal = calibration_table(all_group)
        if cal:
            md.append("\n### L1 calibration (predicted / measured; see report.json for definitions)\n\n" + cal)
    worst = report["worst_by_regret_vs_autotune_modeA"]
    if worst:
        md.append("\n## Worst graphs by ModeA regret vs autotune\n")
        body = []
        for w in worst:
            engines = ", ".join(f"{e['name']}={_num(e['immediate_ms'], 4)}ms" for e in w["engines"])
            picks = ", ".join(
                f"{pol}:{p['engine']}({_pct(p['regret_vs_autotune'])})" for pol, p in w["picks"].items()
            )
            body.append(
                [
                    w["arch"], w["family"], os.path.basename(w["graph_path"] or ""),
                    _pct(w["regret_vs_autotune_modeA"]),
                    f"{w['autotune_best']['engine']} {_num(w['autotune_best']['ms'], 4)}ms",
                    engines, picks,
                ]
            )
        md.append(_md_table(["arch", "family", "graph", "modeA regret", "autotune best", "immediate times", "picks (regret)"], body))
    return "\n".join(md) + "\n"


# ----------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the harness over a graph corpus")
    run.add_argument("--bin", required=True, help="path to hipdnn_xengine_eval")
    run.add_argument("--graphs", required=True, help="directory (*.json, *.fb, recursive) or a file listing graph paths")
    run.add_argument("--out", required=True, help="output directory (rows.jsonl, run.json, harness_stderr.log)")
    run.add_argument("--plugin-dir", action="append", help="engine plugin directory (repeatable)")
    run.add_argument("--shard", help="K/N: run every N-th graph of the sorted list starting at K (0-based K)")
    run.add_argument("--chunk", type=int, default=25, help="graphs per harness process (default 25)")
    run.add_argument("--source-sha", help="git sha of the build under test, recorded in run.json")
    run.add_argument("--policies", help="harness: comma list of modeA,static,default")
    run.add_argument("--latency-repeats", type=int)
    run.add_argument("--no-autotune", action="store_true")
    run.add_argument("--no-immediate", action="store_true")
    run.add_argument("--warmup", type=int)
    run.add_argument("--max-iterations", type=int)
    run.add_argument("--stability", type=float)
    run.add_argument("--ranking-metric", choices=["tflops", "time"])
    run.add_argument(
        "--graph-timeout-s",
        type=float,
        help="per-graph limit: passed to the harness and used as the process timeout "
        "(limit * graphs in the process + slack)",
    )
    run.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="summarise rows.jsonl files")
    rep.add_argument("--rows", nargs="+", required=True, help="one or more rows.jsonl (several arches allowed)")
    rep.add_argument("--out", required=True, help="report.json")
    rep.add_argument("--md", help="markdown summary")
    rep.add_argument("--tolerance", type=float, default=0.05, help="regret tolerance for frac>tol (default 0.05)")
    rep.set_defaults(func=cmd_report)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
