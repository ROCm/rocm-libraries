# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Profiler harness primitive - profile a kernel and RETURN a measurement record.

Composes the primitives into one `rocke.bench.measurement/v1` record:
  counters  (PMU, rocprofv3)   +   resources (from the same rocprofv3 CSV)
  +  wall (a separate un-profiled run)   ->  one record.

Writes NOTHING (pure produce-side): a consumer decides where the record goes.
Reuses rocke's `probe_rocprof_single` pattern - wrap a kernel-launch command with
rocprofv3; PMU replay is separated from wall timing (replay perturbs time).

Stdlib only. Needs a GPU + rocprofv3 for `counters`; degrades to wall-only if the
profiler is unavailable.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import statistics
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

from . import counters as _counters
from . import schema as _schema

_HELPER_KERNEL_PREFIXES = ("__amd_", "__hip_", "rocclr")  # memset/fill etc - skip


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _median(vals: Sequence[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def _write_pmc_input(raws: Sequence[str], path: Path) -> None:
    # One counter per pass = always fits a single pass (robust; grouping is a
    # later optimization). rocprofv3 -i replays the command once per pmc line.
    path.write_text("".join(f"pmc: {r}\n" for r in raws))


def _run_rocprofv3(cmd: Sequence[str], pmc_input: Path, outdir: Path,
                   env: dict, timeout: int) -> bool:
    try:
        proc = subprocess.run(
            ["rocprofv3", "-i", str(pmc_input), "-d", str(outdir),
             "--output-format", "csv", "--", *cmd],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _read_counter_csvs(outdir: Path) -> list[dict]:
    rows: list[dict] = []
    for f in glob.glob(str(outdir / "**" / "*counter_collection.csv"), recursive=True):
        with open(f, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def _pick_target_kernel(rows: list[dict], match: Optional[str]) -> Optional[str]:
    """Busiest non-helper kernel whose name CONTAINS `match` (substring).

    Substring, not exact-equality: rocKE bakes tile/pad/vec into the dispatched
    symbol, so an exact name is brittle. `match=None` picks the overall busiest
    non-helper kernel. Returns None when nothing matches (caller should warn).
    """
    counts: dict[str, int] = {}
    for r in rows:
        name = r.get("Kernel_Name", "")
        if any(name.startswith(p) or p in name for p in _HELPER_KERNEL_PREFIXES):
            continue
        if match and match not in name:
            continue
        counts[name] = counts.get(name, 0) + 1
    return max(counts, key=counts.get) if counts else None


def _parse_perfjson(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("PerfJSON:"):
            try:
                return json.loads(line.removeprefix("PerfJSON:").strip())
            except Exception:
                return {}
    return {}


def _wall(cmd: Sequence[str], env: dict, timeout: int) -> dict:
    """Separate un-profiled run; parse the launcher's PerfJSON if it prints one."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, env=env)
    except (OSError, subprocess.SubprocessError):
        return {}
    p = _parse_perfjson(proc.stdout or "")
    wall: dict = {}
    if "ms" in p:
        wall["ms_median"] = float(p["ms"])
    for k in ("tflops", "gbps", "pct_peak"):
        if k in p:
            wall["gbs" if k == "gbps" else k] = float(p[k])
    return wall


def profile(cmd: Sequence[str], arch: str, *, match: Optional[str] = None,
            label: Optional[str] = None, op: str = "unknown",
            shape: Optional[dict] = None, env: Optional[dict] = None,
            timeout: int = 1800, warn: Optional[Callable[[str], None]] = None) -> dict:
    """Profile the kernel launched by `cmd` and return a measurement record.

    `cmd` is a kernel-launch command (list of argv). `arch` selects the counter map.

    Two independent knobs (kept separate on purpose):
      - `match`: substring of the *dispatched* kernel symbol to profile; else the
        busiest non-helper dispatch is used. This is only a profiler-side filter.
      - `label`: the *identity* name written to the record (what comparison pairs
        on). Set a stable `label` so an optimization that renames the dispatched
        symbol still pairs across runs. If omitted, the dispatched symbol is used.

    `warn(msg)` (optional) is called on each degradation (no counters selected,
    profiler failed, no matching dispatch, counters didn't populate) so a caller
    can surface it instead of the record silently degrading to wall-only.
    """
    def _warn(msg: str) -> None:
        if warn:
            warn(msg)

    env = {**os.environ, **(env or {})}
    sel = _counters.discover(arch)               # normalized -> raw
    raw_to_norm = {raw: norm for norm, raw in sel.items()}

    counters_out: dict = {}
    resources: dict = {}
    kmeta: dict = {}
    if not sel:
        _warn(f"no PMU counters available for {arch} "
              "(rocprofv3 missing/unsupported); producing a wall-only record")
    with tempfile.TemporaryDirectory(prefix="rocke_perf_prof_") as tmp:
        tmp = Path(tmp)
        outdir = tmp / "prof"
        ran = False
        if sel:
            pmc = tmp / "pmc.txt"
            _write_pmc_input(list(sel.values()), pmc)
            ran = _run_rocprofv3(cmd, pmc, outdir, env, timeout)
            if not ran:
                _warn("rocprofv3 failed to run the kernel; counters unavailable "
                      "(wall-only record)")
        if ran:
            rows = _read_counter_csvs(outdir)
            target = _pick_target_kernel(rows, match)
            if target is None:
                _warn("no matching kernel dispatch in profiler output"
                      + (f" for match={match!r}" if match else "")
                      + "; counters empty")
            trows = [r for r in rows if r.get("Kernel_Name") == target] if target else []
            # counters: median per raw counter across the target kernel dispatches
            by_raw: dict[str, list[float]] = {}
            for r in trows:
                cn = r.get("Counter_Name", "")
                if cn in raw_to_norm:
                    try:
                        by_raw.setdefault(cn, []).append(float(r["Counter_Value"]))
                    except (ValueError, KeyError):
                        pass
            for raw, vals in by_raw.items():
                m = _median(vals)
                if m is not None:
                    counters_out[raw_to_norm[raw]] = int(m) if m == int(m) else m
            if target and not counters_out:
                _warn(f"kernel {target!r} matched but no requested counters "
                      "populated (arch counter gap?)")
            # resources + kernel meta come free in the same CSV (static per kernel)
            if trows:
                r0 = trows[0]
                def _i(k):
                    try:
                        return int(float(r0.get(k, 0)))
                    except (ValueError, TypeError):
                        return 0
                resources = {"vgpr": _i("VGPR_Count"), "agpr": _i("Accum_VGPR_Count"),
                             "sgpr": _i("SGPR_Count"), "lds_bytes": _i("LDS_Block_Size"),
                             "source": "rocprofv3"}
                kmeta = {"kernel_name": r0.get("Kernel_Name", target or ""),
                         "workgroup_size": _i("Workgroup_Size"),
                         "grid_size": _i("Grid_Size")}

    wall = _wall(cmd, env, timeout)

    derived: dict = {}
    if counters_out.get("total_clocks"):
        derived["busy_fraction"] = counters_out.get("busy_cycles", 0) / counters_out["total_clocks"]
    if (counters_out.get("l2_hit", 0) + counters_out.get("l2_miss", 0)) > 0:
        derived["l2_hit_rate"] = counters_out["l2_hit"] / (counters_out["l2_hit"] + counters_out["l2_miss"])

    dispatched = kmeta.get("kernel_name", "") or (match or "")
    kernel_name = label or dispatched
    kernel: dict = {"kernel_name": kernel_name, "op": op, "shape": shape or {},
                    "grid": [kmeta.get("grid_size", 0)],
                    "block": [kmeta.get("workgroup_size", 0)]}
    if label and dispatched and dispatched != label:
        kernel["dispatch_symbol"] = dispatched   # keep the real symbol for debugging

    record = {
        "schema": _schema.SCHEMA_VERSION,
        "run": {"run_id": f"{_utc()}", "arch": arch, "timestamp": _utc(),
                "gpu_name": "", "rocm_version": ""},
        "kernel": kernel,
        "wall": wall,
        "counters": counters_out,
        "resources": resources,
        "derived": derived,
        "captured_counters": sorted(counters_out),
        "verify": {},
    }
    _schema.validate(record)
    return record
