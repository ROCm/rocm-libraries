# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Profiler harness primitive - profile a kernel and RETURN a measurement record.

Composes the primitives into one `rocke.bench.measurement/v1` record:
  counters  (PMU, rocprofv3)   +   resources (from the same rocprofv3 CSV)
  +  profiled (timing of the profiled run, correlates with counters)
  +  wall (a separate un-profiled run = real-world timing)   ->  one record.

Does not persist records: a consumer decides where the record goes. PMU replay is
separated from wall timing because replay perturbs time.

Stdlib only. Needs a GPU + rocprofv3 for `counters`; degrades to wall-only if the
profiler is unavailable.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import math
import os
import statistics
import subprocess
import tempfile
import uuid
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


def _write_pmc_input(groups: Sequence[Sequence[str]], path: Path) -> None:
    # One pmc line per group = one rocprofv3 pass = one kernel replay. Counters in a
    # group are collected in a single execution, so they are mutually consistent
    # (e.g. busy_cycles/wait_cycles from the same run). counters.group_counters packs
    # as many counters per group as each hardware block's slots allow.
    path.write_text("".join("pmc: " + " ".join(g) + "\n" for g in groups))


def _run_rocprofv3(
    cmd: Sequence[str], pmc_input: Path, outdir: Path, env: dict, timeout: int
) -> tuple[bool, str]:
    """Run the kernel under rocprofv3. Returns (ok, stdout) so the caller can also
    parse the profiled run's PerfJSON (its timing correlates with the counters)."""
    try:
        proc = subprocess.run(
            [
                "rocprofv3",
                "-i",
                str(pmc_input),
                "-d",
                str(outdir),
                "--output-format",
                "csv",
                "--",
                *cmd,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return False, ""
    return proc.returncode == 0, (proc.stdout or "")


def _count_passes(outdir: Path) -> int:
    """How many collection passes rocprofv3 actually ran (one `pmc_N` dir each)."""
    return len([p for p in glob.glob(str(outdir / "pmc_*")) if Path(p).is_dir()])


def _read_counter_csvs(outdir: Path) -> list[dict]:
    rows: list[dict] = []
    for f in glob.glob(str(outdir / "**" / "*counter_collection.csv"), recursive=True):
        path = Path(f)
        # rocprofv3 writes each replay pass under a 1-indexed `pmc_N` dir; the
        # "pmc_0" sentinel only applies to a CSV not under any such dir (e.g. a
        # single-pass layout), so it never collides with a real pass.
        counter_pass = next(
            (parent.name for parent in path.parents if parent.name.startswith("pmc_")),
            "pmc_0",
        )
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                row["counter_pass"] = counter_pass
                rows.append(row)
    return rows


def _counter_medians(trows: list[dict], raw_to_norm: dict, warmup: int) -> dict:
    """normalized -> median counter value across the target kernel's dispatches.

    Drops the first `warmup` dispatches independently for every counter pass so the
    counters exclude cold-cache launches and line up with timed-only wall/profiled
    measurements. Raises when warmup consumes an entire populated pass rather than
    silently treating warmup samples as measured data.
    """
    by_raw: dict[str, dict[tuple[str, int], float]] = {}
    for r in trows:
        cn = r.get("Counter_Name", "")
        if cn not in raw_to_norm:
            continue
        try:
            did = int(float(r.get("Dispatch_Id", 0)))
            val = float(r["Counter_Value"])
        except (ValueError, KeyError, TypeError):
            continue
        counter_pass = str(r.get("counter_pass", "pmc_0"))
        by_raw.setdefault(cn, {})[(counter_pass, did)] = val
    out: dict = {}
    for raw, by_key in by_raw.items():
        by_pass: dict[str, list[tuple[int, float]]] = {}
        for (counter_pass, dispatch_id), value in by_key.items():
            by_pass.setdefault(counter_pass, []).append((dispatch_id, value))
        kept: list[float] = []
        for counter_pass, pairs in sorted(by_pass.items()):
            ordered = [value for _, value in sorted(pairs)]
            measured = ordered[warmup:]
            if not measured:
                raise RuntimeError(
                    f"warmup={warmup} leaves no measured dispatches for "
                    f"counter {raw!r} in {counter_pass}"
                )
            kept.extend(measured)
        m = _median(kept)
        if m is not None:
            out[raw_to_norm[raw]] = int(m) if m == int(m) else m
    return out


def _counter_samples(trows: list[dict], raw_to_norm: dict) -> list[dict]:
    """Per-dispatch normalized counter values for the target kernel (raw signal).

    Returns one dict per dispatch and rocprofv3 counter pass, with `dispatch_id`,
    `counter_pass`, and normalized counter values. When rocprofv3 emits both
    timestamps, each sample also has `duration_ns`. Includes ALL dispatches (warmup
    NOT dropped) so downstream tooling can slice warmup vs steady-state itself. Opt-in (see
    `profile(per_dispatch=True)`) because it is much larger than the aggregate.
    """
    by_disp: dict[tuple[str, int], dict] = {}
    for r in trows:
        cn = r.get("Counter_Name", "")
        if cn not in raw_to_norm:
            continue
        try:
            did = int(float(r.get("Dispatch_Id", 0)))
            val = float(r["Counter_Value"])
        except (ValueError, KeyError, TypeError):
            continue
        counter_pass = str(r.get("counter_pass", "pmc_0"))
        d = by_disp.setdefault(
            (counter_pass, did),
            {"dispatch_id": did, "counter_pass": counter_pass},
        )
        d[raw_to_norm[cn]] = int(val) if val == int(val) else val
        try:
            start_ns = int(float(r["Start_Timestamp"]))
            end_ns = int(float(r["End_Timestamp"]))
        except (ValueError, KeyError, TypeError):
            continue
        if end_ns >= start_ns:
            d["duration_ns"] = end_ns - start_ns
    return [by_disp[k] for k in sorted(by_disp)]


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


def _perf_from_stdout(stdout: str) -> dict:
    """Timing metrics from a launcher's PerfJSON line (ms/tflops/gbs/pct_peak)."""
    p = _parse_perfjson(stdout)
    out: dict = {}
    for source, target in (
        ("ms", "ms_median"),
        ("tflops", "tflops"),
        ("gbps", "gbs"),
        ("pct_peak", "pct_peak"),
    ):
        try:
            value = float(p[source])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            out[target] = value
    return out


def _verification_from_stdout(stdout: str, *, verified: bool = False) -> dict:
    """Correctness fields emitted by rocke.run_manifest, when present."""
    if not verified:
        return {}
    p = _parse_perfjson(stdout)
    out: dict = {}
    try:
        value = float(p["max_abs_diff"])
        if math.isfinite(value):
            out["max_abs_diff"] = value
    except (KeyError, TypeError, ValueError):
        pass
    for key in ("bad_count", "total"):
        try:
            out[key] = int(p[key])
        except (KeyError, TypeError, ValueError):
            pass
    if "bad_count" in out:
        out["ok"] = out["bad_count"] == 0
    return out


def _wall(cmd: Sequence[str], env: dict, timeout: int) -> tuple[dict, dict]:
    """Separate un-profiled run -> real-world timing (no profiler overhead)."""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"kernel command failed: {exc}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(
            f"kernel command exited with status {proc.returncode}{suffix}"
        )
    stdout = proc.stdout or ""
    wall = _perf_from_stdout(stdout)
    if "ms_median" not in wall:
        raise RuntimeError(
            "kernel command did not emit valid PerfJSON timing (finite ms)"
        )
    return wall, _verification_from_stdout(stdout, verified="--verify" in cmd)


def profile(
    cmd: Sequence[str],
    arch: str,
    *,
    match: Optional[str] = None,
    label: Optional[str] = None,
    op: str = "unknown",
    shape: Optional[dict] = None,
    warmup: int = 0,
    per_dispatch: bool = False,
    env: Optional[dict] = None,
    timeout: int = 1800,
    warn: Optional[Callable[[str], None]] = None,
) -> dict:
    """Profile the kernel launched by `cmd` and return a measurement record.

    `cmd` is a kernel-launch command (list of argv). `arch` selects the counter map.

    Two independent knobs (kept separate on purpose):
      - `match`: substring of the *dispatched* kernel symbol to profile; else the
        busiest non-helper dispatch is used. This is only a profiler-side filter.
      - `label`: the *identity* name written to the record (what comparison pairs
        on). Set a stable `label` so an optimization that renames the dispatched
        symbol still pairs across runs. If omitted, the dispatched symbol is used.

    `warmup`: number of leading (warmup) dispatches to drop per counter, so the
    counter medians exclude cold-cache warmup launches (rocprofv3 records every
    dispatch and the CSV has no warmup flag, so the caller supplies the count -
    e.g. the launcher's warmup_iters). Default 0 = keep all dispatches.

    `per_dispatch`: also emit a `counter_samples` section - the raw per-dispatch
    counter values (all dispatches, keyed by Dispatch_Id) for downstream profiling.
    Opt-in (off by default) because it is much larger than the aggregate.

    `warn(msg)` (optional) is called on each degradation (no counters selected,
    profiler failed, no matching dispatch, counters didn't populate) so a caller
    can surface it instead of the record silently degrading to wall-only.
    """

    def _warn(msg: str) -> None:
        if warn:
            warn(msg)

    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    env = {**os.environ, **(env or {})}
    sel = _counters.discover(arch)  # normalized -> raw
    raw_to_norm = {raw: norm for norm, raw in sel.items()}

    counters_out: dict = {}
    resources: dict = {}
    kmeta: dict = {}
    samples: list = []
    prof_stdout = ""
    if not sel:
        _warn(
            f"no PMU counters available for {arch} "
            "(rocprofv3 missing/unsupported); producing a wall-only record"
        )
    with tempfile.TemporaryDirectory(prefix="rocke_perf_prof_") as tmp:
        tmp = Path(tmp)
        outdir = tmp / "prof"
        ran = False
        groups: list = []
        if sel:
            pmc = tmp / "pmc.txt"
            groups = _counters.group_counters(list(sel.values()))
            _write_pmc_input(groups, pmc)
            ran, prof_stdout = _run_rocprofv3(cmd, pmc, outdir, env, timeout)
            if not ran:
                _warn(
                    "rocprofv3 failed to run the kernel; counters unavailable "
                    "(wall-only record)"
                )
            else:
                npass = _count_passes(outdir)
                if npass > len(groups):
                    # rocprofv3 split a group across passes -> those counters came
                    # from different executions, so cross-counter ratios aren't
                    # coherent.
                    _warn(
                        f"rocprofv3 used more passes ({npass}) than the "
                        f"{len(groups)} counter group(s) requested; a group was "
                        "split, so counters in it are not from one execution "
                        "(check counters._BLOCK_SLOTS for this arch)"
                    )
        if ran:
            rows = _read_counter_csvs(outdir)
            target = _pick_target_kernel(rows, match)
            if target is None:
                _warn(
                    "no matching kernel dispatch in profiler output"
                    + (f" for match={match!r}" if match else "")
                    + "; counters empty"
                )
            trows = (
                [r for r in rows if r.get("Kernel_Name") == target] if target else []
            )
            # median per counter across the target kernel's dispatches, warmup dropped
            counters_out = _counter_medians(trows, raw_to_norm, warmup)
            if per_dispatch:
                samples = _counter_samples(trows, raw_to_norm)
            if target and not counters_out:
                _warn(
                    f"kernel {target!r} matched but no requested counters "
                    "populated (arch counter gap?)"
                )
            # resources + kernel meta come free in the same CSV (static per kernel)
            if trows:
                r0 = trows[0]

                def _i(k):
                    try:
                        return int(float(r0.get(k, 0)))
                    except (ValueError, TypeError):
                        return 0

                resources = {
                    "vgpr": _i("VGPR_Count"),
                    "agpr": _i("Accum_VGPR_Count"),
                    "sgpr": _i("SGPR_Count"),
                    "lds_bytes": _i("LDS_Block_Size"),
                    "source": "rocprofv3",
                }
                kmeta = {
                    "kernel_name": r0.get("Kernel_Name", target or ""),
                    "workgroup_size": _i("Workgroup_Size"),
                    "grid_size": _i("Grid_Size"),
                }

    # profiled: timing of the profiled run (same execution as the counters, so it
    # correlates with them). wall: a separate un-profiled run (real-world timing).
    profiled = _perf_from_stdout(prof_stdout)
    wall, verify = _wall(cmd, env, timeout)

    derived: dict = {}
    busy = counters_out.get("busy_cycles")
    total = counters_out.get("total_clocks")
    if busy is not None and total:
        derived["busy_fraction"] = busy / total
    hits = counters_out.get("l2_hit")
    misses = counters_out.get("l2_miss")
    if hits is not None and misses is not None and (hits + misses) > 0:
        derived["l2_hit_rate"] = hits / (hits + misses)
    if profiled.get("ms_median") and wall.get("ms_median"):
        derived["profiler_overhead_pct"] = (
            (profiled["ms_median"] - wall["ms_median"]) / wall["ms_median"] * 100.0
        )

    dispatched = kmeta.get("kernel_name", "") or (match or "")
    command_id = hashlib.sha256("\0".join(cmd).encode()).hexdigest()[:12]
    kernel_name = label or dispatched or f"command_{command_id}"
    kernel: dict = {
        "kernel_name": kernel_name,
        "op": op,
        "shape": shape or {},
        "grid": [kmeta.get("grid_size", 0)],
        "block": [kmeta.get("workgroup_size", 0)],
    }
    if label and dispatched and dispatched != label:
        kernel["dispatch_symbol"] = dispatched  # keep the real symbol for debugging

    now = _utc()
    record = {
        "schema": _schema.SCHEMA_VERSION,
        "run": {
            "run_id": f"{now}_{uuid.uuid4().hex[:12]}",
            "arch": arch,
            "timestamp": now,
            "gpu_name": "",
            "rocm_version": "",
        },
        "kernel": kernel,
        "wall": wall,
        "profiled": profiled,
        "counters": counters_out,
        "resources": resources,
        "derived": derived,
        "captured_counters": sorted(counters_out),
        "verify": verify,
    }
    if per_dispatch:
        record["counter_samples"] = samples  # raw per-dispatch values (opt-in)
    _schema.validate(record)
    return record
