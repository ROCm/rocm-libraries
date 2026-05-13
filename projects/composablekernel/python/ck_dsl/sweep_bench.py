# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Run every built instance through the shared launcher and find the
best correct kernel per shape.

This implements the runbook's §2.3 hygiene:

  - Warm up each kernel (the launcher already does this).
  - Verify correctness at a small shape first (`max_abs_diff=0`); skip
    perf for any kernel that fails.
  - Run >=3 fresh-process attempts per (kernel, shape) and report
    median + spread.
  - Discard the first run of a fresh process if `cold_first_run=True`.
  - Print one TSV-like row per (shape, kernel) to stdout for easy
    grep / awk.

It deliberately spawns the launcher binary (not the kernel directly)
so the timing/verify code paths are exactly the ones tested elsewhere.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class RunResult:
    name: str
    M: int
    N: int
    K: int
    runs: List[float] = field(default_factory=list)  # per-attempt TFLOPS
    correct: bool = False
    error: str = ""

    @property
    def median_tflops(self) -> float:
        return statistics.median(self.runs) if self.runs else 0.0

    @property
    def spread_pct(self) -> float:
        if len(self.runs) < 2:
            return 0.0
        med = self.median_tflops
        if med == 0.0:
            return 0.0
        return (max(self.runs) - min(self.runs)) / med * 100


def _maybe_sudo(cmd: List[str]) -> List[str]:
    s = shutil.which("sudo")
    if s and os.environ.get("CK_DSL_USE_SUDO", "1") != "0":
        return [s, "-n"] + cmd
    return cmd


def _run_launcher(
    launcher: Optional[Path],
    hsaco: Path,
    manifest: Path,
    shape: Tuple[int, int, int],
    verify: bool,
    timeout: float = 120.0,
) -> Tuple[int, str]:
    if launcher is None:
        args = [
            sys.executable,
            "-m",
            "ck_dsl.run_manifest",
            str(hsaco),
            str(manifest),
            "--shape",
            ",".join(str(s) for s in shape),
        ]
        env = os.environ.copy()
        # Preserve the current package under sudo.
        env["PYTHONPATH"] = (
            str(Path(__file__).resolve().parents[1])
            + os.pathsep
            + env.get("PYTHONPATH", "")
        )
    else:
        args = [
            str(launcher),
            str(hsaco),
            str(manifest),
            "--shape",
            ",".join(str(s) for s in shape),
        ]
        env = None
    if verify:
        args.append("--verify")
    if (
        launcher is None
        and shutil.which("sudo")
        and os.environ.get("CK_DSL_USE_SUDO", "1") != "0"
    ):
        args = [
            shutil.which("sudo"),
            "-n",
            "env",
            f"PYTHONPATH={env['PYTHONPATH']}",
        ] + args
    else:
        args = _maybe_sudo(args) if launcher is not None else args
    r = subprocess.run(args, env=env, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def _parse_tflops(out: str) -> Optional[float]:
    for line in out.splitlines():
        if line.startswith("Perf:"):
            for tok in line.split(","):
                tok = tok.strip()
                if tok.endswith(" TFlops"):
                    return float(tok.removesuffix(" TFlops").strip())
    return None


def _is_correct(out: str) -> bool:
    return "max_abs_diff=0" in out


def _write_per_instance_manifest(inst: Dict[str, object], dst: Path) -> Path:
    """Each instance entry in the sweep manifest has every field the
    launcher needs. We splat it into a stand-alone JSON so the launcher
    (which expects a per-kernel manifest) can consume it as-is."""
    m = {
        "schema": "ck.dsl.example.manifest/v1",
        "kind": "gemm_fp16",
        "kernel_name": inst["name"],
        "hsaco": Path(str(inst["hsaco"])).name,
        "block_m": int(inst["block_m"]),
        "block_n": int(inst["block_n"]),
        "block_k": int(inst["block_k"]),
        "threads_per_block": int(inst["threads_per_block"]),
        "warmup_iters": 5,
        "timed_iters": 100,
        "default_shape": [3328, 4096, 4096],
        "args_signature": [
            {"name": "A", "type": "ptr<f16, global>", "size_bytes": 8},
            {"name": "B", "type": "ptr<f16, global>", "size_bytes": 8},
            {"name": "C", "type": "ptr<f16, global>", "size_bytes": 8},
            {"name": "M", "type": "i32", "size_bytes": 4},
            {"name": "N", "type": "i32", "size_bytes": 4},
            {"name": "K", "type": "i32", "size_bytes": 4},
        ],
        "timing_ms": {"build_ms": float(inst.get("build_ms", 0.0))},
        "hsaco_bytes": int(inst.get("hsaco_bytes", 0)) or 0,
        "ck_dependency": False,
        "ir_authored": True,
        "atoms": ["dispatcher-sweep-instance"],
        "notes": "instance manifest emitted by sweep_bench",
    }
    dst.write_text(json.dumps(m, indent=2, sort_keys=True))
    return dst


def sweep_run(
    sweep_manifest_path: Path,
    launcher: Optional[Path],
    *,
    shapes: Optional[Sequence[Tuple[int, int, int]]] = None,
    verify_shape: Tuple[int, int, int] = (256, 256, 256),
    attempts: int = 3,
    cold_first_run: bool = True,
    skip_uncorrect: bool = True,
    out_csv: Optional[Path] = None,
    work_dir: Optional[Path] = None,
) -> List[RunResult]:
    """Run every instance in the sweep manifest at every shape and
    return one `RunResult` per (instance, shape).

    `attempts` is the number of fresh-process invocations per
    (kernel, shape) — the runbook recommends >=3.
    `cold_first_run`: if True, run one extra invocation per kernel and
    discard it (we observed first-run drops of 2x; §2.3 of the runbook).
    """
    manifest = json.loads(sweep_manifest_path.read_text())
    instances = manifest["instances"]
    if shapes is None:
        shapes = [(int(s["M"]), int(s["N"]), int(s["K"])) for s in manifest["shapes"]]

    work_dir = work_dir or sweep_manifest_path.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    results: List[RunResult] = []
    csv_lines: List[str] = []
    if out_csv:
        csv_lines.append(
            "name,M,N,K,attempts,median_tflops,min_tflops,max_tflops,spread_pct,correct"
        )

    for inst in instances:
        if not inst.get("build_ok", True):
            continue
        # Emit a per-kernel manifest the launcher understands.
        kname = inst["name"]
        per_manifest = work_dir / f"{kname[:120]}.manifest.json"
        _write_per_instance_manifest(inst, per_manifest)
        hsaco = Path(str(inst["hsaco"]))

        # 1) verify at small shape first.
        rc, out = _run_launcher(
            launcher, hsaco, per_manifest, verify_shape, verify=True
        )
        if rc != 0 or not _is_correct(out):
            for sh in shapes:
                r = RunResult(name=kname, M=sh[0], N=sh[1], K=sh[2])
                r.error = f"verify rc={rc}"
                results.append(r)
                if out_csv:
                    csv_lines.append(f"{kname},{sh[0]},{sh[1]},{sh[2]},0,0,0,0,0,FALSE")
            if skip_uncorrect:
                continue
        # 2) perf attempts per shape.
        for sh in shapes:
            res = RunResult(name=kname, M=sh[0], N=sh[1], K=sh[2], correct=True)
            if cold_first_run:
                try:
                    _run_launcher(launcher, hsaco, per_manifest, sh, verify=False)
                except Exception as e:
                    res.error = f"cold-run: {e}"
            for _ in range(attempts):
                try:
                    rc, out = _run_launcher(
                        launcher, hsaco, per_manifest, sh, verify=False
                    )
                    if rc != 0:
                        res.error = f"perf rc={rc}"
                        continue
                    t = _parse_tflops(out)
                    if t is not None:
                        res.runs.append(t)
                except Exception as e:
                    res.error = f"perf: {e}"
            results.append(res)
            if out_csv:
                med = res.median_tflops
                lo = min(res.runs) if res.runs else 0.0
                hi = max(res.runs) if res.runs else 0.0
                csv_lines.append(
                    f"{kname},{sh[0]},{sh[1]},{sh[2]},{len(res.runs)},"
                    f"{med:.3f},{lo:.3f},{hi:.3f},{res.spread_pct:.2f},"
                    f"{'TRUE' if res.correct else 'FALSE'}"
                )

    if out_csv:
        out_csv.write_text("\n".join(csv_lines) + "\n")

    return results


def best_per_shape(
    results: Sequence[RunResult],
) -> Dict[Tuple[int, int, int], RunResult]:
    """Return the highest-median-TFLOPS correct kernel for each shape."""
    out: Dict[Tuple[int, int, int], RunResult] = {}
    for r in results:
        if not r.correct or not r.runs:
            continue
        key = (r.M, r.N, r.K)
        cur = out.get(key)
        if cur is None or r.median_tflops > cur.median_tflops:
            out[key] = r
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run sweep_manifest.json through the launcher and find the best kernel per shape."
    )
    parser.add_argument("sweep_manifest", type=Path)
    parser.add_argument(
        "--launcher",
        type=Path,
        default=None,
        help="legacy C++ launcher path; omit to use python -m ck_dsl.run_manifest",
    )
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--no-cold-discard", action="store_true")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument(
        "--shape",
        action="append",
        default=None,
        help="override shapes (M,N,K). May be repeated. "
        "If omitted, uses shapes from the sweep manifest.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="if >0, only run the first N instances (useful for smoke tests).",
    )
    args = parser.parse_args(argv)

    if args.shape:
        shapes = [tuple(int(x) for x in s.split(",")) for s in args.shape]
    else:
        shapes = None

    sm = args.sweep_manifest
    if args.limit > 0:
        # Cut down the manifest in-place into a temp file.
        m = json.loads(sm.read_text())
        m["instances"] = m["instances"][: args.limit]
        sm_lim = sm.parent / f"{sm.stem}.limit{args.limit}{sm.suffix}"
        sm_lim.write_text(json.dumps(m, indent=2, sort_keys=True))
        sm = sm_lim

    results = sweep_run(
        sm,
        args.launcher,
        shapes=shapes,
        attempts=args.attempts,
        cold_first_run=not args.no_cold_discard,
        out_csv=args.csv,
    )
    best = best_per_shape(results)

    print(f"\n=== best per shape ({len(best)} shapes, {len(results)} runs) ===")
    for sh, r in sorted(best.items()):
        print(
            f"  {sh[0]:>5}x{sh[1]:>5}x{sh[2]:>5}: "
            f"{r.median_tflops:>7.2f} TFLOPS  (spread {r.spread_pct:>4.1f}%)  "
            f"{r.name}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
