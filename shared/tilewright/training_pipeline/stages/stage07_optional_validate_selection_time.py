# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage07: selection time of the deployed model and C++/Python pick parity.

Runs the hipblaslt-bench of the deploy checkout's build on every bench yaml
(filtered to the config's data types and layout) and checks the runtime
against stage06's deploy record.

Pick parity (one run per yaml, `requested_solution_num: 1`, with
`TILEWRIGHT_DIAG` and `TILEWRIGHT_PICK_LOG` on):
  * every `[TILEWRIGHT_DIAG FILE]` line of the deployed library must name the
    co-located copy of the deployed file, with the recorded cell and split
    counts, feature hash and weight type, and that file must hash to the
    recorded sha256;
  * the engine's top-1 pick and serving cell of every problem are compared
    with `lib.evaluate.top1_picks` on the same model bytes, the library's
    kernel pool read from its .dat and the device hardware. Both sides run
    the same engine, so a mismatch points at how hipBLASLt maps its problem,
    kernels or hardware to tilewright.

Selection time (separate runs, pick log off): for every request size in
`stage07.timing_request_sizes`, `stage07.timing_repetitions` repetitions of
an ML-on and an ML-off run over the identical problem list, the order
alternating between repetitions. Each problem's `Solution selection time`
is paired across the two runs; the report gives the per-problem ML-on minus
ML-off deltas (median over repetitions) as median and percentiles. The first
problem of every run, which carries the library load, is reported
separately. ML-on runs must show the deployed model loading; ML-off runs get
no tilewright variables at all.

Everything is written below --out-dir. The exit status is non-zero when a
bench fails, the deployed model is not the one the runtime loaded, or pick
parity is below `stage07.parity_min_match_rate`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = THIS_DIR.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

import yaml  # noqa: E402

from lib import ui  # noqa: E402
from run_orchestrator import arch_base, cfg_get  # noqa: E402

_RE_SEL_TIME = re.compile(r"Solution selection time:\s*(\S+)\s*us")
_RE_SUPPORTED = re.compile(r"^Is supported\s+(\d+)\s*/\s*Total solutions:\s*(\d+)")
_RE_HEADER = re.compile(r"^\[(\d+)\]:(.*)$")
_RE_DIAG_FILE = re.compile(r"\[TILEWRIGHT_DIAG FILE\]\s*(.*)$")
_RE_DIAG_FAIL = re.compile(r"\[TILEWRIGHT_DIAG FAIL\]\s*(.*)$")
_RE_PICK = re.compile(
    r"\[TILEWRIGHT_PICK\]\s+m=(\d+)\s+n=(\d+)\s+k=(\d+)\s+b=(\d+)\s+"
    r"tA=([NT])\s+tB=([NT])\s+leaf=(\S*)\s+"
    r"top1_sig=\(mt_m=(\d+),mt_n=(\d+),mt_k=(\d+),mi_m=(\d+),mi_n=(\d+),"
    r"mi_k=(\d+),cha=(-?\d+),chb=(-?\d+)\)\s+top1_score=(\S+)\s+n_configs=(\d+)"
)

FILTER_FIELDS = (
    "a_type",
    "b_type",
    "c_type",
    "d_type",
    "compute_type",
    "transA",
    "transB",
)
DROP_KEYS = (
    "adaptive",
    "warmup_time",
    "sample_time",
    "measure_time",
    "max_measure_time",
    "min_iters",
    "max_iters",
    "noise_threshold",
    "stability_threshold",
    "stability_window",
    "stability_interval",
    "skip_slow_solution_ratio",
    "print_kernel_info",
    "rotating",
)

Key = Tuple[int, int, int, int, str, str]


# ── bench yaml ──────────────────────────────────────────────────────────────


def _norm_field(name: str, value: Any) -> str:
    s = str(value).strip()
    if name == "compute_type" and s.lower().startswith("c_"):
        s = s[2:]
    return s


def workload_filter(cfg: Mapping[str, Any]) -> Dict[str, str]:
    hbl = cfg.get("hipblaslt") or {}
    wf = {f: _norm_field(f, hbl[f]) for f in FILTER_FIELDS if hbl.get(f) is not None}
    for f in ("scaleA", "scaleB"):
        if hbl.get(f) is not None:
            wf[f] = str(int(hbl[f]))
    return wf


def _entry_matches(entry: Mapping[str, Any], wf: Mapping[str, str]) -> bool:
    for name, want in wf.items():
        if name in ("scaleA", "scaleB"):
            got = str(int(entry.get(name, 0) or 0))
        else:
            got = _norm_field(name, entry.get(name, ""))
        if got != want:
            return False
    return True


def load_bench_entries(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: a bench yaml is a list of problem mappings")
    return [dict(e) for e in data if isinstance(e, dict)]


def normalized_entries(
    entries: Sequence[Mapping[str, Any]], requested_solutions: int
) -> List[Dict[str, Any]]:
    """Copies set to run one timed iteration of `requested_solutions`
    solutions, without the training-bench timing and pruning knobs."""
    out = []
    for e in entries:
        d = {k: v for k, v in e.items() if k not in DROP_KEYS}
        d["iters"] = 1
        d["cold_iters"] = 0
        d["requested_solution_num"] = int(requested_solutions)
        out.append(d)
    return out


def write_bench_yaml(entries: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for e in entries:
            line = yaml.safe_dump(
                dict(e), default_flow_style=True, width=1 << 20, sort_keys=False
            ).strip()
            f.write(f"- {line}\n")


def entry_key(e: Mapping[str, Any]) -> Key:
    return (
        int(e.get("M", e.get("m", 0))),
        int(e.get("N", e.get("n", 0))),
        int(e.get("K", e.get("k", 0))),
        int(e.get("batch_count", 1) or 1),
        str(e.get("transA", "N")),
        str(e.get("transB", "N")),
    )


# ── bench runs ──────────────────────────────────────────────────────────────


def run_bench(
    bench: Path,
    yaml_path: Path,
    log_path: Path,
    device: int,
    env: Dict[str, str],
    *,
    startup_grace_s: float,
    stall_timeout_s: float,
    quiet: bool,
) -> int:
    """Run one bench, output to `log_path`. A bench silent for longer than
    the budget is killed (startup gets its own, longer budget)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(bench), "--yaml", str(yaml_path), "--device", str(device)]
    if not quiet:
        ui.info("bench", " ".join(cmd))
    t0 = time.time()
    with log_path.open("w", buffering=1, errors="replace") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            errors="replace",
            bufsize=1,
            env=env,
            cwd=str(log_path.parent),
        )
        assert proc.stdout is not None
        last = [time.time(), False]

        def _drain() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                logf.write(line)
                last[0], last[1] = time.time(), True

        pump = threading.Thread(target=_drain, daemon=True)
        pump.start()
        while proc.poll() is None:
            time.sleep(0.5)
            budget = stall_timeout_s if last[1] else startup_grace_s
            if budget > 0 and time.time() - last[0] > budget:
                ui.err("bench", f"no output for {budget:.0f}s; killing {bench.name}")
                proc.kill()
                break
        rc = proc.wait()
        pump.join(timeout=30)
    if not quiet:
        (ui.ok if rc == 0 else ui.err)(
            "bench", f"rc={rc} after {ui.fmt_dur(time.time() - t0)}  -> {log_path}"
        )
    return rc


# ── log parsing ─────────────────────────────────────────────────────────────


@dataclass
class Call:
    sel_us: float
    n_lines: int
    supported: Optional[int] = None
    key: Optional[Key] = None


@dataclass
class BenchLog:
    calls: List[Call] = field(default_factory=list)
    diag_files: List[Dict[str, str]] = field(default_factory=list)
    diag_fails: List[str] = field(default_factory=list)
    picks: List[Dict[str, Any]] = field(default_factory=list)
    unattributed_sel_lines: int = 0


def _kv(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for tok in text.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def _float(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return float("nan")


def parse_bench_log(path: Path) -> BenchLog:
    """Selection-time calls (grouped per problem by the bench's `Is
    supported` line, identified by the first result row after it), tilewright
    diagnostics and pick-log lines of one bench log."""
    out = BenchLog()
    pending: List[float] = []
    awaiting: Optional[Tuple[Call, List[str]]] = None
    current: Optional[Call] = None
    if not path.is_file():
        return out
    with path.open(errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if awaiting is not None:
                call, names = awaiting
                awaiting = None
                values = [v.strip() for v in line.strip().split(",")]
                row = dict(zip(names, values))
                try:
                    call.key = (
                        int(row["m"]),
                        int(row["n"]),
                        int(row["k"]),
                        int(row.get("batch_count", "1")),
                        row["transA"],
                        row["transB"],
                    )
                except (KeyError, ValueError):
                    pass
                continue
            m = _RE_SEL_TIME.search(line)
            if m:
                pending.append(_float(m.group(1)))
                continue
            m = _RE_SUPPORTED.match(line.strip())
            if m:
                current = Call(
                    sel_us=float(sum(pending)) if pending else float("nan"),
                    n_lines=len(pending),
                    supported=int(m.group(1)),
                )
                out.calls.append(current)
                pending = []
                continue
            m = _RE_HEADER.match(line.strip())
            if m and current is not None and current.key is None:
                names = [n.strip() for n in m.group(2).split(",")]
                if names and names[0] == "function":
                    names = names[1:]
                awaiting = (current, names)
                continue
            m = _RE_PICK.search(line)
            if m:
                g = m.groups()
                out.picks.append(
                    {
                        "key": (int(g[0]), int(g[1]), int(g[2]), int(g[3]), g[4], g[5]),
                        "cell": g[6],
                        "top1_sig": tuple(int(v) for v in g[7:15]),
                        "top1_score": _float(g[15]),
                        "n_configs": int(g[16]),
                    }
                )
                continue
            m = _RE_DIAG_FILE.search(line)
            if m:
                out.diag_files.append(_kv(m.group(1)))
                continue
            m = _RE_DIAG_FAIL.search(line)
            if m:
                out.diag_fails.append(m.group(1))
    out.unattributed_sel_lines = len(pending)
    return out


# ── checks ──────────────────────────────────────────────────────────────────


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_diag(
    log: BenchLog, expected_path: Path, record: Mapping[str, Any]
) -> Tuple[bool, List[str]]:
    """(ok, problems): the deployed model loaded from `expected_path` with
    the recorded contents."""
    problems: List[str] = []
    want = os.path.realpath(expected_path)
    mine = [d for d in log.diag_files if os.path.realpath(d.get("path", "")) == want]
    for msg in log.diag_fails:
        problems.append(f"engine load failure: {msg}")
    if not mine:
        seen = sorted({d.get("path", "?") for d in log.diag_files})
        problems.append(
            f"the runtime never loaded {expected_path}"
            + (f" (it loaded {seen})" if seen else " (no tilewright model loaded)")
        )
        return False, problems
    d = mine[0]
    expect = {
        "n_cells": str(record.get("n_cells")),
        "n_splits": str(record.get("n_splits")),
        "qhash": str(record.get("feature_catalog_hash")),
        "weights": str(record.get("weight_dtype")),
        "arch": str(record.get("arch")),
    }
    for k, v in expect.items():
        if d.get(k) != v:
            problems.append(f"loaded model {k}={d.get(k)} but the deploy recorded {v}")
    return not problems, problems


def compare_picks(
    problems: Sequence[Mapping[str, Any]],
    cpp: Sequence[Mapping[str, Any]],
    py: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Join engine pick-log lines and offline picks by problem."""
    cpp_by: Dict[Key, Mapping[str, Any]] = {}
    inconsistent = 0
    for p in cpp:
        k = tuple(p["key"])
        if k in cpp_by:
            prev = cpp_by[k]
            if (prev["top1_sig"], prev["cell"]) != (p["top1_sig"], p["cell"]):
                inconsistent += 1
            continue
        cpp_by[k] = p
    py_by: Dict[Key, Mapping[str, Any]] = {}
    for p in py:
        k = (
            int(p["m"]),
            int(p["n"]),
            int(p["k"]),
            int(p["batch"]),
            str(p["transA"]),
            str(p["transB"]),
        )
        py_by.setdefault(k, p)
    rows: List[Dict[str, Any]] = []
    counts: Dict[str, int] = defaultdict(int)
    seen = set()
    for e in problems:
        key = entry_key(e)
        if key in seen:
            continue
        seen.add(key)
        c, p = cpp_by.get(key), py_by.get(key)
        row: Dict[str, Any] = {"m": key[0], "n": key[1], "k": key[2], "batch": key[3]}
        row["transA"], row["transB"] = key[4], key[5]
        py_ok = p is not None and p.get("reason") == "ok"
        if c is not None:
            row["cpp_sig"], row["cpp_cell"] = list(c["top1_sig"]), c["cell"]
            row["cpp_score"] = c["top1_score"]
            if p is not None and p.get("n_configs") is not None:
                if int(c["n_configs"]) != int(p["n_configs"]):
                    row["cpp_n_configs"] = c["n_configs"]
                    counts["pool_size_differs"] += 1
        if p is not None:
            row["py_reason"] = p.get("reason")
            row["py_cell"] = p.get("cell")
            if py_ok:
                row["py_sig"], row["py_score"] = list(p["top1_sig"]), p["top1_score"]
        if c is None and not py_ok:
            row["result"] = "agree_unscored"
        elif c is None:
            row["result"] = "no_cpp_pick"
        elif not py_ok:
            row["result"] = "no_py_pick"
        elif tuple(c["top1_sig"]) != tuple(p["top1_sig"]):
            row["result"] = "pick_mismatch"
        elif c["cell"] != p.get("cell"):
            row["result"] = "cell_mismatch"
        else:
            row["result"] = "match"
        counts[row["result"]] += 1
        rows.append(row)
    n = len(rows)
    agree = counts.get("match", 0) + counts.get("agree_unscored", 0)
    pool_sizes = sorted(
        {int(p["n_configs"]) for p in py if p.get("n_configs") is not None}
    )
    return {
        "n_problems": n,
        "n_cpp_picks": len(cpp_by),
        "n_py_picks": sum(1 for p in py_by.values() if p.get("reason") == "ok"),
        "n_agree": agree,
        "match_rate": agree / n if n else None,
        "counts": dict(counts),
        "library_pool_size": pool_sizes[0] if len(pool_sizes) == 1 else pool_sizes,
        "n_inconsistent_cpp_picks": inconsistent,
        "per_problem": rows,
    }


def json_clean(obj: Any) -> Any:
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {str(k): json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_clean(v) for v in obj]
    return obj


# ── statistics ──────────────────────────────────────────────────────────────


def percentile(xs: Sequence[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    pos = (len(s) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def describe(xs: Sequence[float]) -> Dict[str, Any]:
    xs = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    if not xs:
        return {"n": 0}
    out: Dict[str, Any] = {"n": len(xs), "mean": statistics.fmean(xs)}
    for q in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        out[f"p{q}"] = percentile(xs, q)
    out["min"], out["max"] = min(xs), max(xs)
    return out


def _occurrence_keys(calls: Sequence[Call]) -> List[Tuple[Any, ...]]:
    seen: Dict[Any, int] = defaultdict(int)
    out = []
    for pos, c in enumerate(calls):
        base: Any = c.key if c.key is not None else ("#pos", pos)
        out.append((base, seen[base]))
        seen[base] += 1
    return out


def paired_timing(reps: Sequence[Tuple[BenchLog, BenchLog]]) -> Dict[str, Any]:
    """Per-problem ML-on vs ML-off selection times over repetitions. The
    first call of every run (library load) is kept out of the pairing."""
    on_t: Dict[Any, List[float]] = defaultdict(list)
    off_t: Dict[Any, List[float]] = defaultdict(list)
    delta_t: Dict[Any, List[float]] = defaultdict(list)
    cold_on: List[float] = []
    cold_off: List[float] = []
    unpaired = 0
    for on, off in reps:
        if on.calls:
            cold_on.append(on.calls[0].sel_us)
        if off.calls:
            cold_off.append(off.calls[0].sel_us)
        on_map = dict(zip(_occurrence_keys(on.calls), on.calls))
        off_map = dict(zip(_occurrence_keys(off.calls), off.calls))
        cold = set()
        if on.calls:
            cold.add(_occurrence_keys(on.calls)[0])
        if off.calls:
            cold.add(_occurrence_keys(off.calls)[0])
        keys = (set(on_map) | set(off_map)) - cold
        for k in keys:
            a, b = on_map.get(k), off_map.get(k)
            if (
                a is None
                or b is None
                or not (math.isfinite(a.sel_us) and math.isfinite(b.sel_us))
            ):
                unpaired += 1
                continue
            on_t[k].append(a.sel_us)
            off_t[k].append(b.sel_us)
            delta_t[k].append(a.sel_us - b.sel_us)
    per_on = [statistics.median(v) for v in on_t.values()]
    per_off = [statistics.median(v) for v in off_t.values()]
    per_delta = [statistics.median(v) for v in delta_t.values()]
    ratios = [
        statistics.median(on_t[k]) / statistics.median(off_t[k])
        for k in on_t
        if statistics.median(off_t[k]) > 0
    ]
    return {
        "repetitions": len(reps),
        "n_problems_paired": len(per_delta),
        "n_unpaired": unpaired,
        "ml_on_us": describe(per_on),
        "ml_off_us": describe(per_off),
        "delta_us": describe(per_delta),
        "ratio_on_over_off": describe(ratios),
        "first_call_us": {"ml_on": cold_on, "ml_off": cold_off},
    }


# ── stage ───────────────────────────────────────────────────────────────────


@dataclass
class Context:
    cfg: Dict[str, Any]
    record: Dict[str, Any]
    bench: Path
    library_dir: Path
    stem: str
    weights_path: Path
    device: int
    out_dir: Path
    startup_grace_s: float
    stall_timeout_s: float
    quiet: bool
    failures: List[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.failures.append(msg)
        ui.err("check", msg)


def _env(
    ctx: Context, tilewright: Optional[Dict[str, str]], timing: bool
) -> Dict[str, str]:
    from lib.bench_env import bench_env_with

    extra = {"HIPBLASLT_TENSILE_LIBPATH": str(ctx.library_dir)}
    if timing:
        extra["TENSILE_DB"] = "0x10000"
    return bench_env_with(extra, tilewright=tilewright)


def _bench(
    ctx: Context, yaml_path: Path, log: Path, env: Dict[str, str]
) -> Optional[BenchLog]:
    rc = run_bench(
        ctx.bench,
        yaml_path,
        log,
        ctx.device,
        env,
        startup_grace_s=ctx.startup_grace_s,
        stall_timeout_s=ctx.stall_timeout_s,
        quiet=ctx.quiet,
    )
    if rc != 0:
        ctx.fail(f"hipblaslt-bench failed (rc={rc}); see {log}")
        return None
    return parse_bench_log(log)


def _offline_picks(
    ctx: Context, problems: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    from lib import dat, evaluate
    from lib.hardware import device_hardware

    data = Path(ctx.record["staged_model"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != ctx.record["sha256"]:
        raise RuntimeError(
            f"{ctx.record['staged_model']} differs from the deploy record"
        )
    pool = evaluate.pool_from_kernels(
        dat.load_library_kernels(ctx.library_dir, ctx.stem)
    )
    hw = device_hardware(ctx.cfg, ctx.device)
    rows = []
    for e in problems:
        m, n, k, b, ta, tb = entry_key(e)
        row = {"m": m, "n": n, "k": k, "batch_count": b, "transA": ta, "transB": tb}
        row.update(
            {f: e[f] for f in ("a_type", "b_type", "c_type", "d_type", "compute_type")}
        )
        rows.append(row)
    return evaluate.top1_picks(data, rows, pool, hardware=hw, min_scored=1)


def run_parity(
    ctx: Context, tag: str, entries: List[Dict[str, Any]], min_rate: float
) -> Dict[str, Any]:
    out = ctx.out_dir / tag / "parity"
    yaml_path = out / "bench.yaml"
    write_bench_yaml(normalized_entries(entries, 1), yaml_path)
    env = _env(
        ctx,
        {
            "TENSILE_USE_TILEWRIGHT": "1",
            "TILEWRIGHT_DIAG": "1",
            "TILEWRIGHT_PICK_LOG": "1",
        },
        timing=False,
    )
    log = _bench(ctx, yaml_path, out / "bench.log", env)
    if log is None:
        return {"ok": False, "error": "bench failed"}
    diag_ok, diag_problems = check_diag(log, ctx.weights_path, ctx.record)
    for p in diag_problems:
        ctx.fail(f"{tag}: {p}")
    try:
        py = _offline_picks(ctx, entries)
    except Exception as e:
        ctx.fail(f"{tag}: offline ranking failed: {e!r}")
        return {"ok": False, "diag_ok": diag_ok, "error": repr(e)}
    result = compare_picks(entries, log.picks, py)
    result["diag_ok"] = diag_ok
    result["min_match_rate"] = min_rate
    rate = result["match_rate"]
    result["ok"] = diag_ok and rate is not None and rate >= min_rate
    with (out / "parity.json").open("w") as f:
        json.dump(json_clean(result), f, indent=2)
    if result["n_inconsistent_cpp_picks"]:
        ui.warn(
            "parity",
            f"{tag}: the runtime logged different picks for the same problem "
            f"{result['n_inconsistent_cpp_picks']} times; the first one is compared",
        )
    if result["counts"].get("pool_size_differs"):
        ui.warn(
            "parity",
            f"{tag}: the runtime ranked a pool of a different size than the "
            f"library file holds ({result['library_pool_size']}) for "
            f"{result['counts']['pool_size_differs']} problems",
        )
    if rate is None:
        ctx.fail(f"{tag}: no problem to compare")
    elif rate < min_rate:
        ctx.fail(
            f"{tag}: pick parity {rate * 100:.2f}% < {min_rate * 100:.2f}% "
            f"({result['counts']}); see {out / 'parity.json'}"
        )
    else:
        ui.ok(
            "parity",
            f"{tag}: {result['n_agree']}/{result['n_problems']} problems agree",
        )
    return {k: v for k, v in result.items() if k != "per_problem"}


def run_timing(
    ctx: Context,
    tag: str,
    entries: List[Dict[str, Any]],
    sizes: Sequence[int],
    reps: int,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    on_env = _env(
        ctx, {"TENSILE_USE_TILEWRIGHT": "1", "TILEWRIGHT_DIAG": "1"}, timing=True
    )
    off_env = _env(ctx, None, timing=True)
    for rsn in sizes:
        out = ctx.out_dir / tag / "timing" / f"rsn{rsn}"
        yaml_path = out / "bench.yaml"
        write_bench_yaml(normalized_entries(entries, rsn), yaml_path)
        pairs: List[Tuple[BenchLog, BenchLog]] = []
        for rep in range(reps):
            order = ("on", "off") if rep % 2 == 0 else ("off", "on")
            logs: Dict[str, Optional[BenchLog]] = {}
            for mode in order:
                env = on_env if mode == "on" else off_env
                logs[mode] = _bench(
                    ctx, yaml_path, out / f"rep{rep}_ml_{mode}.log", env
                )
            on, off = logs["on"], logs["off"]
            if on is None or off is None:
                continue
            ok, problems = check_diag(on, ctx.weights_path, ctx.record)
            if not ok:
                for p in problems:
                    ctx.fail(f"{tag} rsn={rsn} rep {rep}: ML-on run: {p}")
                continue
            if off.diag_files:
                ctx.fail(
                    f"{tag} rsn={rsn} rep {rep}: tilewright loaded in the ML-off run"
                )
                continue
            for name, lg in (("ML-on", on), ("ML-off", off)):
                if len(lg.calls) != len(entries):
                    ui.warn(
                        "timing",
                        f"{tag} rsn={rsn} rep {rep}: {name} run has {len(lg.calls)} "
                        f"selections for {len(entries)} problems",
                    )
            pairs.append((on, off))
        stats = paired_timing(pairs)
        results[f"rsn{rsn}"] = stats
        d = stats["delta_us"]
        if d.get("n"):
            ui.ok(
                "timing",
                f"{tag} rsn={rsn}: ML-on minus ML-off per problem: median "
                f"{d['p50']:+.2f} us, p10 {d['p10']:+.2f}, p90 {d['p90']:+.2f}, "
                f"p99 {d['p99']:+.2f} over {d['n']} problems",
            )
        else:
            ctx.fail(f"{tag} rsn={rsn}: no paired selection times")
    return results


def _parse_sizes(text: Optional[str], default: Sequence[int]) -> List[int]:
    if not text:
        return [int(v) for v in default]
    out = [int(v) for v in text.split(",") if v.strip()]
    if not out or min(out) < 1:
        raise ValueError(f"bad request sizes {text!r}")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="stage07: selection time and C++/Python pick parity of the deployed model"
    )
    ap.add_argument("--config-yaml", type=Path, required=True)
    ap.add_argument(
        "--build-dir",
        type=Path,
        required=True,
        help="hipBLASLt build directory of the deploy checkout",
    )
    ap.add_argument(
        "--deploy-record", type=Path, required=True, help="stage06 deploy_record.json"
    )
    ap.add_argument(
        "--bench-yaml",
        "--yaml",
        type=Path,
        action="append",
        required=True,
        dest="bench_yaml",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--bench-binary", type=Path, default=None)
    ap.add_argument(
        "--library-dir",
        type=Path,
        default=None,
        help="default: <build-dir>/Tensile/library/<arch>",
    )
    ap.add_argument(
        "--request-sizes",
        default=None,
        help="comma list; default stage07.timing_request_sizes",
    )
    ap.add_argument("--repetitions", type=int, default=None)
    ap.add_argument("--parity-min-match-rate", type=float, default=None)
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--no-parity", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    with args.config_yaml.open() as f:
        cfg = yaml.safe_load(f) or {}
    with args.deploy_record.open() as f:
        record = json.load(f)
    arch_dir = arch_base(str(cfg.get("arch", "")))
    target = next(
        (t for t in record.get("targets", []) if t.get("dir") == arch_dir), None
    )
    if target is None:
        ui.err("cfg", f"the deploy record has no target for {arch_dir}")
        return 2
    library_dir = (
        args.library_dir or args.build_dir / "Tensile" / "library" / arch_dir
    ).resolve()
    bench = (
        args.bench_binary or args.build_dir / "clients" / "hipblaslt-bench"
    ).resolve()
    sizes = _parse_sizes(
        args.request_sizes, cfg_get(cfg, "stage07.timing_request_sizes")
    )
    reps = int(args.repetitions or cfg_get(cfg, "stage07.timing_repetitions"))
    min_rate = float(
        args.parity_min_match_rate
        if args.parity_min_match_rate is not None
        else cfg_get(cfg, "stage07.parity_min_match_rate")
    )
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = Context(
        cfg=cfg,
        record=record,
        bench=bench,
        library_dir=library_dir,
        stem=str(target["library_stem"]),
        weights_path=library_dir / str(target["weights_file"]),
        device=int(args.device),
        out_dir=out_dir,
        startup_grace_s=float(cfg_get(cfg, "bench.startup_grace_s")),
        stall_timeout_s=float(cfg_get(cfg, "bench.stall_timeout_s")),
        quiet=args.quiet,
    )
    ui.banner("stage07  selection time + pick parity")
    ui.info("cfg", f"bench        : {bench}")
    ui.info("cfg", f"library      : {library_dir}")
    ui.info("cfg", f"library stem : {ctx.stem}")
    ui.info(
        "cfg",
        f"model        : {ctx.weights_path}  (sha256 {str(record.get('sha256'))[:16]})",
    )
    ui.info(
        "cfg",
        f"timing       : {'off' if args.no_timing else f'rsn {sizes}, {reps} repetitions'}",
    )
    ui.info(
        "cfg",
        f"parity       : {'off' if args.no_parity else f'min match rate {min_rate}'}",
    )

    summary: Dict[str, Any] = {
        "deploy_record": str(args.deploy_record.resolve()),
        "model": str(ctx.weights_path),
        "library_stem": ctx.stem,
        "per_yaml": {},
    }
    if not bench.is_file():
        ctx.fail(f"hipblaslt-bench not found: {bench}")
    elif not ctx.weights_path.is_file():
        ctx.fail(
            f"{ctx.weights_path} does not exist: build the hipblaslt-tilewright-models target "
            f"of {args.build_dir}"
        )
    elif sha256_file(ctx.weights_path) != record.get("sha256"):
        ctx.fail(
            f"{ctx.weights_path} is not the deployed model (sha256 differs); build "
            f"the hipblaslt-tilewright-models target of {args.build_dir}"
        )
    else:
        wf = workload_filter(cfg)
        for yp in args.bench_yaml:
            tag = yp.stem
            try:
                entries = load_bench_entries(yp)
            except (OSError, ValueError, yaml.YAMLError) as e:
                ctx.fail(f"{yp}: {e}")
                continue
            kept = [e for e in entries if _entry_matches(e, wf)]
            ui.banner(
                f"{tag}: {len(kept)} of {len(entries)} problems match {wf}", ui.C.BLUE
            )
            res: Dict[str, Any] = {
                "yaml": str(yp),
                "n_entries": len(entries),
                "n_kept": len(kept),
            }
            if not kept:
                ctx.fail(
                    f"{tag}: no problem matches the config's data types and layout"
                )
            else:
                if not args.no_parity:
                    res["parity"] = run_parity(ctx, tag, kept, min_rate)
                if not args.no_timing:
                    res["timing"] = run_timing(ctx, tag, kept, sizes, reps)
            summary["per_yaml"][tag] = res
    summary["failures"] = ctx.failures
    summary["ok"] = not ctx.failures
    with (out_dir / "summary.json").open("w") as f:
        json.dump(json_clean(summary), f, indent=2, default=str)
    if ctx.failures:
        ui.err(
            "done",
            f"{len(ctx.failures)} check(s) failed; see {out_dir / 'summary.json'}",
        )
        return 1
    ui.ok("done", f"all checks passed; {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
