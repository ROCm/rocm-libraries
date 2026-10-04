# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage02_probe -- measure each shape once, then calibrate the bench knobs
(iters / cold_iters / skip_slow_solution_ratio) of the full bench per shape.

hipblaslt-bench's measurement noise falls like 1 / sqrt(iters), so a constant
iteration count either wastes time on slow shapes or leaves fast ones noisy.
Per shape:

  iters = cold_iters = max(1, round(probe.duration_us / probe_us))
  skip_slow_solution_ratio:
    bucket mode   ratio_light if probe_us < fast_us_threshold else ratio_heavy
    sigmoid mode  logistic ramp in log(probe_us) between skip_ratio_min and
                  skip_ratio_max, centred at skip_ratio_center_us

All knobs live in the config's `probe:` section. The orchestrator skips this
stage when `bench.adaptive.enabled` is set, since the bench then times every
kernel adaptively.

Workflow:
  1. Distribute stage01's shapes.yaml round-robin over `--devices`; every probe
     line gets `requested_solution_num: 1` (one kernel is enough to measure the
     shape) and the probe iteration counts from the config.
  2. Run one hipblaslt-bench per device in parallel. A bench whose log stops
     growing for --stall-timeout-s (after --startup-grace-s until its first
     shape finishes) is killed.
  3. Read each shape's `us` back from the log, matched by the (m, n, k,
     batch_count) the bench prints with it. A shape the bench printed nothing
     for (no solution, crash, killed) keeps the default knobs.
  4. Write shapes_probed.yaml: the original lines in their original order with
     the calibrated knobs and `requested_solution_num: -1`.

Every bench process leads its own process group. SIGINT, SIGTERM or SIGHUP
stops all of them (`lib.procs`) and the stage exits with 128 + the signal
number.

Outputs in `--out-dir`:
  shapes_probed.yaml                    input of stage03
  logs/probe_NNNN_linesX-Y_gpuD.log     one log per device; X..Y is the
                                        range of source lines its shapes span
  tmp/                                  bench yamls of the running processes,
                                        removed when the stage ends
"""
from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = THIS_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import yaml  # noqa: E402

from lib import procs  # noqa: E402
from lib import ui  # noqa: E402
from lib.bench_env import bench_env_with  # noqa: E402
from lib.bench_log import LogProbe, parse_log_file  # noqa: E402
from lib.bench_yaml import read_bench_lines, shape_of_line  # noqa: E402
from lib.fs import atomic_write_text  # noqa: E402

ShapeTuple = Tuple[int, int, int, int]

# ── per-line knob rewriting ──────────────────────────────────────────────────

_ITERS_RE = re.compile(r"(?<!\w)iters:\s*\d+")
_COLD_RE = re.compile(r"(?<!\w)cold_iters:\s*\d+")
_SKIP_RE = re.compile(r"(?<!\w)skip_slow_solution_ratio:\s*[\d.eE+-]+")
_RSN_RE = re.compile(r"(?<!\w)requested_solution_num:\s*-?\d+")


def _set_knob(line: str, pattern: re.Pattern, key: str, value: str) -> str:
    if pattern.search(line):
        return pattern.sub(f"{key}: {value}", line, count=1)
    return line.rstrip("\n").rstrip("}") + f", {key}: {value}}}\n"


def _fmt_ratio(x: float) -> str:
    return f"{x:.4f}".rstrip("0").rstrip(".") or "0"


def _stamp_iters_in_line(
    line: str,
    iters: int,
    cold_iters: int,
    skip_ratio: Optional[float] = None,
    rsn: Optional[int] = None,
) -> str:
    """`line` with iters / cold_iters (and optionally the skip ratio and
    requested_solution_num) replaced, or appended when absent."""
    line = _set_knob(line, _COLD_RE, "cold_iters", str(cold_iters))
    line = _set_knob(line, _ITERS_RE, "iters", str(iters))
    if skip_ratio is not None:
        line = _set_knob(
            line, _SKIP_RE, "skip_slow_solution_ratio", _fmt_ratio(skip_ratio)
        )
    if rsn is not None:
        line = _set_knob(line, _RSN_RE, "requested_solution_num", str(rsn))
    return line if line.endswith("\n") else line + "\n"


# ── probe-pass split (multi-device) ──────────────────────────────────────────


def _build_probe_yamls(
    lines: List[str],
    devices: List[int],
    first_iters: int,
    first_cold: int,
    other_iters: int,
    other_cold: int,
) -> Tuple[List[List[str]], List[List[int]]]:
    """Round-robin `lines` over `devices`: (per-device stamped lines,
    per-device indices into `lines`). The first line of each device gets
    (first_iters, first_cold), the rest (other_iters, other_cold); every line
    gets `requested_solution_num: 1`."""
    n_dev = len(devices)
    buckets: List[List[int]] = [list(range(d, len(lines), n_dev)) for d in range(n_dev)]
    per_dev_lines: List[List[str]] = []
    for bucket in buckets:
        stamped = []
        for j, i in enumerate(bucket):
            it, cold = (
                (first_iters, first_cold) if j == 0 else (other_iters, other_cold)
            )
            stamped.append(_stamp_iters_in_line(lines[i], it, cold, rsn=1))
        per_dev_lines.append(stamped)
    return per_dev_lines, buckets


# ── reading the probe logs ───────────────────────────────────────────────────


def _positive_float(text: Optional[str]) -> Optional[float]:
    try:
        v = float(text)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def probe_us_by_shape(log_path: Path) -> Dict[ShapeTuple, float]:
    """Fastest `us` of every problem in one probe log, keyed by the (m, n, k,
    batch_count) the bench printed for it."""
    out: Dict[ShapeTuple, float] = {}
    if not log_path.exists():
        return out
    for prob in parse_log_file(log_path):
        shape = prob.shape()
        if shape is None:
            continue
        times = [_positive_float(row.fields.get("us")) for row in prob.tested]
        times = [t for t in times if t is not None]
        if not times:
            continue
        best = min(times)
        out[shape] = min(best, out.get(shape, best))
    return out


# ── parallel bench launcher ──────────────────────────────────────────────────


class _ProbeRunner(threading.Thread):
    """Probe one device: write `lines` to `yaml_path`, run hipblaslt-bench on
    it with `--device <D>` under the stall watchdog, output to `log_path`."""

    def __init__(
        self,
        device: int,
        lines: List[str],
        bench_binary: Path,
        log_path: Path,
        yaml_path: Path,
        children: procs.ChildGroups,
        *,
        stall_s: float = 300.0,
        startup_grace_s: float = 900.0,
    ) -> None:
        super().__init__(name=f"probe-d{device}", daemon=True)
        self.device = device
        self.lines = lines
        self.bench_binary = bench_binary
        self.log_path = log_path
        self.yaml_path = yaml_path
        self.children = children
        self.stall_s = float(stall_s)
        self.startup_grace_s = float(startup_grace_s)
        self.rc: Optional[int] = None
        self.killed: Optional[str] = None
        self.wall_s: float = 0.0

    def run(self) -> None:
        t0 = time.time()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.yaml_path.write_text("".join(self.lines))
            progress = LogProbe(self.log_path, 0)
            # --device rather than HIP_VISIBLE_DEVICES, so every process uses
            # the same global device numbering.
            with self.log_path.open("w", buffering=1) as logf:
                proc = self.children.popen(
                    [
                        str(self.bench_binary),
                        "--device",
                        str(self.device),
                        "--yaml",
                        str(self.yaml_path),
                    ],
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    env=bench_env_with(),
                )
                self.rc, self.killed = procs.wait_with_stall_watchdog(
                    proc,
                    lambda: procs.file_size(self.log_path),
                    self.stall_s,
                    poll_s=procs.poll_interval(self.stall_s),
                    startup_grace_s=self.startup_grace_s,
                    probe_progress=progress.progressed,
                    probe_fatal=progress.fatal_seen,
                )
            if self.killed is not None:
                what = (
                    "unrecoverable GPU queue abort in the log"
                    if self.killed == "fatal"
                    else f"no output for {self.stall_s:.0f}s"
                )
                with self.log_path.open("a") as logf:
                    logf.write(f"\n*** KILLED [{self.killed}]: {what} ***\n")
        except Exception:
            self.rc = -1
        finally:
            self.yaml_path.unlink(missing_ok=True)
        self.wall_s = time.time() - t0


# ── calibration ──────────────────────────────────────────────────────────────


def _calibrate(probe_us: Optional[float], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Bench knobs for a shape measured at `probe_us` (None: not measured).
    iters is only clamped at the bench's floor of 1."""
    duration_us = float(cfg["duration_us"])
    mode = str(cfg.get("skip_ratio_mode", "bucket")).lower()
    fast_threshold = float(cfg.get("fast_us_threshold", 30.0))
    ratio_light = float(cfg.get("ratio_light", 0.30))
    ratio_heavy = float(cfg.get("ratio_heavy", 0.80))
    if probe_us is None or probe_us <= 0:
        fail_ratio = (
            float(cfg.get("skip_ratio_min", 0.05)) if mode == "sigmoid" else ratio_light
        )
        return {
            "iters": 100,
            "cold_iters": 100,
            "skip_slow_solution_ratio": fail_ratio,
            "probe_us": None,
            "bucket": "fail",
        }
    iters = max(1, int(round(duration_us / probe_us)))
    if mode == "sigmoid":
        # Slower GEMMs separate their candidates more clearly and cost more to
        # bench, so they prune harder; fast, near-tied ones prune little.
        lo = float(cfg.get("skip_ratio_min", 0.05))
        hi = float(cfg.get("skip_ratio_max", 0.95))
        center = float(cfg.get("skip_ratio_center_us", fast_threshold))
        k = float(cfg.get("skip_ratio_k", 1.5))
        x = math.log(max(probe_us, 1e-9) / max(center, 1e-9))
        ratio = round(lo + (hi - lo) / (1.0 + math.exp(-k * x)), 4)
        bucket = "sig"
    else:
        bucket = "fast" if probe_us < fast_threshold else "slow"
        ratio = ratio_light if bucket == "fast" else ratio_heavy
    return {
        "iters": iters,
        "cold_iters": iters,
        "skip_slow_solution_ratio": ratio,
        "probe_us": probe_us,
        "bucket": bucket,
    }


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    args = _parse_args()
    work_dir = args.out_dir / "tmp"
    children = procs.ChildGroups()
    with procs.stop_children_on_exit(children):
        try:
            return _run(args, children, work_dir)
        finally:
            children.stop_all()
            shutil.rmtree(work_dir, ignore_errors=True)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="stage02 -- probe per-shape us, then calibrate iters / "
        "cold_iters / skip_slow_solution_ratio."
    )
    ap.add_argument(
        "--in-dir", type=Path, required=True, help="stage01 dir (shapes.yaml)."
    )
    ap.add_argument(
        "--out-dir", type=Path, required=True, help="writes shapes_probed.yaml + logs/."
    )
    ap.add_argument(
        "--config-yaml",
        type=Path,
        required=True,
        help="run config; its `probe:` section drives the calibration.",
    )
    ap.add_argument("--bench-binary", type=Path, required=True)
    ap.add_argument(
        "--devices",
        default="0,1,2,3,4,5,6,7",
        help="comma-separated GPU device IDs for the parallel probe.",
    )
    ap.add_argument(
        "--startup-grace-s",
        type=float,
        default=900.0,
        help="minimum lifetime of a probe process before the stall rule may "
        "kill it, until it finishes its first shape.",
    )
    ap.add_argument(
        "--stall-timeout-s",
        type=float,
        default=300.0,
        help="kill a probe process whose log has not grown for this many "
        "seconds; its unmeasured shapes keep the default knobs. 0 disables "
        "the watchdog.",
    )
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


def _run(args: argparse.Namespace, children: procs.ChildGroups, work_dir: Path) -> int:
    with args.config_yaml.open() as f:
        cfg = yaml.safe_load(f) or {}
    pcfg = cfg.get("probe", {}) or {}
    required_keys = [
        "first_line_iters",
        "first_line_cold_iters",
        "other_lines_iters",
        "other_lines_cold_iters",
        "duration_us",
    ]
    sigmoid = str(pcfg.get("skip_ratio_mode", "bucket")).lower() == "sigmoid"
    if not sigmoid:
        required_keys += ["fast_us_threshold", "ratio_light", "ratio_heavy"]
    missing = [k for k in required_keys if k not in pcfg]
    if missing:
        ui.err("cfg", f"config probe: section is missing {missing}")
        return 1

    devices = [int(d) for d in args.devices.split(",") if d.strip()]
    in_yaml = args.in_dir / "shapes.yaml"
    if not devices:
        ui.err("err", "--devices is empty")
        return 1
    if not in_yaml.exists():
        ui.err("err", f"missing shapes.yaml at {in_yaml}")
        return 1
    if not args.bench_binary.exists():
        ui.err("err", f"missing hipblaslt-bench: {args.bench_binary}")
        return 1

    if not args.quiet:
        ui.banner("stage02 probe -> calibrate iters")
        ui.info("cfg", f"in_yaml      : {in_yaml}")
        ui.info("cfg", f"devices      : {devices}")
        ui.info("cfg", f"duration_us  : {pcfg['duration_us']}")
        ui.info(
            "cfg",
            f"probe iters  : first={pcfg['first_line_iters']}/"
            f"{pcfg['first_line_cold_iters']}  "
            f"other={pcfg['other_lines_iters']}/{pcfg['other_lines_cold_iters']}",
        )
        if sigmoid:
            ui.info(
                "cfg",
                f"skip ratio   : sigmoid min={pcfg.get('skip_ratio_min', 0.05)} "
                f"max={pcfg.get('skip_ratio_max', 0.95)} "
                f"center={pcfg.get('skip_ratio_center_us', 30)}us "
                f"k={pcfg.get('skip_ratio_k', 1.5)}",
            )
        else:
            ui.info(
                "cfg",
                f"fast cutoff  : {pcfg['fast_us_threshold']}us  "
                f"light={pcfg['ratio_light']}  heavy={pcfg['ratio_heavy']}",
            )

    if not args.quiet:
        ui.banner("Step 1/4  Build per-device probe yamls", ui.C.BLUE)
    lines = read_bench_lines(in_yaml)
    n_total = len(lines)
    if n_total == 0:
        ui.err("err", "no shape lines in shapes.yaml")
        return 1
    devices = devices[: max(1, min(len(devices), n_total))]
    per_dev_lines, per_dev_indices = _build_probe_yamls(
        lines,
        devices,
        int(pcfg["first_line_iters"]),
        int(pcfg["first_line_cold_iters"]),
        int(pcfg["other_lines_iters"]),
        int(pcfg["other_lines_cold_iters"]),
    )
    if not args.quiet:
        for dev, idx in zip(devices, per_dev_indices):
            ui.ok("split", f"  device {dev}: {len(idx):>6,} shapes")

    if not args.quiet:
        ui.banner("Step 2/4  Run probe on all devices in parallel", ui.C.BLUE)
    logs_dir = args.out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    for stale in logs_dir.glob("probe_*.log"):
        stale.unlink()
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir()
    runners: List[_ProbeRunner] = []
    for d_idx, (dev, idx_list) in enumerate(zip(devices, per_dev_indices)):
        name = f"probe_{d_idx:04d}_lines{idx_list[0] + 1}-{idx_list[-1] + 1}_gpu{dev}"
        runners.append(
            _ProbeRunner(
                dev,
                per_dev_lines[d_idx],
                args.bench_binary,
                logs_dir / f"{name}.log",
                work_dir / f"{name}.yaml",
                children,
                stall_s=args.stall_timeout_s,
                startup_grace_s=args.startup_grace_s,
            )
        )
    t0 = time.time()
    for r in runners:
        r.start()
    last_print = 0.0
    while any(r.is_alive() for r in runners):
        time.sleep(1.0)
        if not args.quiet and time.time() - last_print >= 10.0:
            done = sum(1 for r in runners if not r.is_alive())
            alive = [r.device for r in runners if r.is_alive()]
            ui.grey(
                "probe",
                f"devices_done={done}/{len(runners)}  alive={alive}  "
                f"elapsed={ui.fmt_dur(time.time() - t0)}",
            )
            last_print = time.time()
    for r in runners:
        r.join()
    for r in runners:
        if r.rc == 0:
            if not args.quiet:
                ui.ok(
                    "probe", f"  device {r.device}: rc=0  wall={ui.fmt_dur(r.wall_s)}"
                )
        else:
            # Not fatal: the bench streams one record per shape, so a device
            # that died on shape N still logged the shapes before it.
            killed = f"  killed [{r.killed}]" if r.killed else ""
            ui.warn(
                "probe",
                f"  device {r.device}: rc={r.rc}{killed}  log={r.log_path}  (shapes "
                f"it did not measure get default knobs)",
            )

    if not args.quiet:
        ui.banner("Step 3/4  Parse per-shape probe_us", ui.C.BLUE)
    per_shape_us: List[Optional[float]] = [None] * n_total
    for r, idx_list in zip(runners, per_dev_indices):
        measured = probe_us_by_shape(r.log_path)
        n_hit = 0
        for i in idx_list:
            shape = shape_of_line(lines[i])
            us = measured.get(shape) if shape is not None else None
            if us is not None:
                per_shape_us[i] = us
                n_hit += 1
        if n_hit < len(idx_list):
            ui.warn(
                "parse",
                f"  device {r.device}: measured {n_hit}/{len(idx_list)} shapes "
                f"(rest -> default knobs)",
            )
    n_have = sum(1 for u in per_shape_us if u is not None)
    if n_have == 0:
        ui.err(
            "err",
            "no shape on any device produced a probe measurement; check the "
            "probe logs (a total failure usually means the bench cannot run "
            "this config's dtype/layout at all)",
        )
        return 2
    if not args.quiet:
        ui.ok("parse", f"probe_us populated for {n_have:,}/{n_total:,} shapes")

    if not args.quiet:
        ui.banner("Step 4/4  Calibrate + write shapes_probed.yaml", ui.C.BLUE)
    counts = {"fast": 0, "slow": 0, "sig": 0, "fail": 0}
    out_lines: List[str] = []
    for line, us in zip(lines, per_shape_us):
        knobs = _calibrate(us, pcfg)
        out_lines.append(
            _stamp_iters_in_line(
                line,
                knobs["iters"],
                knobs["cold_iters"],
                skip_ratio=knobs["skip_slow_solution_ratio"],
                rsn=-1,
            )
        )
        counts[knobs["bucket"]] += 1
    out_yaml = args.out_dir / "shapes_probed.yaml"
    atomic_write_text(out_yaml, "".join(out_lines))
    if not args.quiet:
        if sigmoid:
            ui.ok(
                "calib",
                f"sigmoid-calibrated={counts['sig']:,}  fail(default)={counts['fail']:,}",
            )
        else:
            ui.ok(
                "calib",
                f"fast={counts['fast']:,}  slow={counts['slow']:,}  "
                f"fail(default)={counts['fail']:,}",
            )
        ui.ok("write", f"shapes_probed.yaml: {n_total:,} lines -> {out_yaml}")
        ui.banner("Done", ui.C.GREEN)
    return 0


if __name__ == "__main__":
    sys.exit(main())
