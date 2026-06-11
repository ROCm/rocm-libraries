#!/usr/bin/env python3
"""Performance bench harness for the SpargeAttn + SageAttn comparison chart.

Sweeps a sparsity range for the sparse kernels and records dense / sage
baselines, writing one CSV row per (curve, sparsity, seed). Feed the CSV to
plot.py to reproduce docs/pv_skip_mode_comparison.png.

Curves:
  - fmha_dense  : dense FMHA, fp16            (tile_example_fmha_fwd)
  - fmha_sage   : dense + SageAttn fp8 quant  (tile_example_sageattn_fwd)
  - sparge_fp16 : sparse + fp16               (tile_example_sparge)
  - sparge_sage : sparse + int8 BLOCKSCALE Q/K (tile_example_sparge -qscale=bs)

Timing comes from the binary's own in-program hipEvent GPU timer (the "X ms"
it prints to stdout), not rocprof. For the sparse curves this timer brackets
ALL THREE GPU kernels (kstats + blockmap + attention) via a single grouped
launch_kernel call, so sparse TOPS reflect the true end-to-end cost. No
rocprof dependency -- only an MI300-class GPU is needed.

Example (run from the example dir):
  python3 docs/run_bench.py --bin-dir build/bin --csv docs/sparge_bench.csv
  # behind a scheduler (e.g. SLURM):
  python3 docs/run_bench.py --bin-dir /path/build/bin --launcher "srun --jobid=123 --overlap"
"""
from __future__ import annotations

import argparse
import csv
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

SHAPE = dict(b=2, h=16, s=8192, d=128)
WARMUP, REPEAT = 5, 100

TOPK_SWEEP_FULL = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
TOPK_SWEEP_SMOKE = [0.50]

# The binary prints ", <avg_ms> ms," from its in-program hipEvent GPU timer.
MS_RE = re.compile(r",\s*([0-9.]+)\s*ms")
SPARSITY_RE = re.compile(r"sparsity=([0-9.eE+\-]+)\(([0-9]+)/([0-9]+)\)")

FINITE_THRESH = 20.0

CURVES = {
    "fmha_dense": dict(
        bin="tile_example_fmha_fwd", prec="fp16",
        extra_flags=["-mask=0", "-vlayout=r"],
        has_sparsity=False, has_topk=False, pv_mode=None, pv_threshold=None, qscale=None,
    ),
    "fmha_sage": dict(
        bin="tile_example_sageattn_fwd", prec="fp8bf16",
        extra_flags=["-mask=0", "-vlayout=r", "-qscale=bs"],
        has_sparsity=False, has_topk=False, pv_mode=None, pv_threshold=None, qscale=None,
    ),
    "sparge_fp16": dict(
        bin="tile_example_sparge", prec="fp16",
        extra_flags=["-pipeline=vsa", "-simthreshd1=0.001", "-cdfthreshd=-1"],
        has_sparsity=True, has_topk=True, pv_mode="warp", pv_threshold=FINITE_THRESH, qscale=None,
    ),
    "sparge_sage": dict(
        bin="tile_example_sparge", prec="fp16",
        extra_flags=["-pipeline=vsa", "-simthreshd1=0.001", "-cdfthreshd=-1"],
        has_sparsity=True, has_topk=True, pv_mode="warp", pv_threshold=FINITE_THRESH, qscale="bs",
    ),
}

CSV_COLS = [
    "curve_name", "pv_mode", "topk", "pv_threshold", "qscale",
    "measured_sparsity", "active_blocks", "total_blocks",
    "mean_ns", "tops", "seed", "run_id",
]


def tops_from(mean_ns: float) -> float:
    b, h, s, d = SHAPE["b"], SHAPE["h"], SHAPE["s"], SHAPE["d"]
    return 4.0 * b * h * s * s * d / mean_ns / 1e3


def build_cli(bin_dir: Path, curve: str, topk, seed: int) -> list[str]:
    cfg = CURVES[curve]
    cli = [
        str(bin_dir / cfg["bin"]),
        "-v=0",
        f"-b={SHAPE['b']}", f"-h={SHAPE['h']}", f"-s={SHAPE['s']}", f"-d={SHAPE['d']}",
        f"-prec={cfg['prec']}",
        "-iperm=1", "-operm=1",
        f"-warmup={WARMUP}", f"-repeat={REPEAT}",
        "-kname=1",
        f"-seed={seed}",
    ] + cfg["extra_flags"]
    if cfg["has_topk"]:
        cli += [f"-topk={topk}", f"-pv_mode={cfg['pv_mode']}", f"-pv_threshold={cfg['pv_threshold']}"]
    if cfg.get("qscale"):
        cli += [f"-qscale={cfg['qscale']}"]
    return cli


def parse_ms(stdout: str) -> float:
    m = MS_RE.search(stdout)
    if not m:
        raise RuntimeError("No '<avg_ms> ms' token in stdout")
    return float(m.group(1))


def parse_sparsity(stdout: str) -> tuple[float, int, int]:
    m = SPARSITY_RE.search(stdout)
    if not m:
        raise RuntimeError("No sparsity= line in stdout")
    return float(m.group(1)), int(m.group(2)), int(m.group(3))


def run_one(bin_dir: Path, launcher: list[str], curve: str, topk, seed: int, run_dir: Path) -> dict:
    cfg = CURVES[curve]
    run_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"{curve}__topk{topk:.2f}__seed{seed}" if topk is not None else f"{curve}__seed{seed}")
    stdout_path = run_dir / f"{tag}.stdout.txt"

    cmd = launcher + build_cli(bin_dir, curve, topk, seed)
    print(f"[run] {tag}\n      " + " ".join(shlex.quote(x) for x in cmd), flush=True)

    t0 = time.time()
    with stdout_path.open("w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"binary exit={proc.returncode} after {dt:.1f}s; see {stdout_path}")

    _ = list(run_dir.iterdir())  # force attr refresh on networked filesystems

    out = stdout_path.read_text()
    avg_ms = parse_ms(out)
    mean_ns = avg_ms * 1e6
    tops = tops_from(mean_ns)
    if cfg["has_sparsity"]:
        sparsity, active, total = parse_sparsity(out)
    else:
        sparsity, active, total = 0.0, 0, 0

    print(f"      avg_ms={avg_ms:.3f}  tops={tops:.2f}  "
          f"sparsity={sparsity:.4f} ({active}/{total})  wall={dt:.1f}s", flush=True)
    return dict(
        curve_name=curve,
        pv_mode=("" if cfg["pv_mode"] is None else cfg["pv_mode"]),
        topk=("" if topk is None else f"{topk:.4f}"),
        pv_threshold=("" if cfg["pv_threshold"] is None else f"{cfg['pv_threshold']:g}"),
        qscale=(cfg.get("qscale") or ""),
        measured_sparsity=f"{sparsity:.6f}", active_blocks=str(active), total_blocks=str(total),
        mean_ns=f"{mean_ns:.1f}", tops=f"{tops:.3f}", seed=str(seed), run_id=tag,
    )


def append_csv(rows: list[dict], path: Path):
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin-dir", type=Path, default=Path("build/bin"),
                    help="dir holding tile_example_* binaries (default: build/bin)")
    ap.add_argument("--launcher", default="",
                    help='command prefix per run, e.g. "srun --jobid=123 --overlap" (default: run directly)')
    ap.add_argument("--smoke", action="store_true", help="single sparsity point (quick check)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=42)
    ap.add_argument("--curves", nargs="+", default=list(CURVES.keys()), choices=list(CURVES.keys()))
    ap.add_argument("--csv", type=Path, default=Path("sparge_bench.csv"))
    ap.add_argument("--run-dir", type=Path, default=Path("bench_runs") / time.strftime("sweep_%Y%m%d_%H%M%S"))
    args = ap.parse_args()

    launcher = shlex.split(args.launcher)
    topks = TOPK_SWEEP_SMOKE if args.smoke else TOPK_SWEEP_FULL
    seeds = [args.seed_start + i for i in range(args.seeds)]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)

    plan = []
    for curve in args.curves:
        if CURVES[curve]["has_topk"]:
            plan += [(curve, tk, sd) for tk in topks for sd in seeds]
        else:
            plan += [(curve, None, sd) for sd in seeds]

    print(f"[plan] {len(plan)} runs -> {args.csv}  (run_dir={args.run_dir})", flush=True)
    failures = []
    t0 = time.time()
    for curve, tk, sd in plan:
        try:
            append_csv([run_one(args.bin_dir, launcher, curve, tk, sd, args.run_dir)], args.csv)
        except Exception as e:
            print(f"[FAIL] {curve} topk={tk} seed={sd}: {e}", file=sys.stderr)
            failures.append((curve, tk, sd, str(e)))

    print(f"\n[done] {len(plan)-len(failures)}/{len(plan)} ok, wall={time.time()-t0:.1f}s")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
