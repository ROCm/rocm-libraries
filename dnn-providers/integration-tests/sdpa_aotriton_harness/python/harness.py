"""End-to-end driver for the SDPA gpu_ref-vs-AOTriton numerical harness.

Framing: **AOTriton** is the oracle / reference of record; the **gpu_ref**
kernel is the candidate under test.

Pipeline:
  (a) gen_inputs   -> populate a run dir with Q/K/V/mask + manifests
  (b) C++ driver   -> for each case, run THIS branch's fp32 gpu_ref candidate,
                      writing gpuref_o.npy
  (c) run_torch    -> AOTriton oracle + math (HP/LP) references via PyTorch
  (d) compare      -> adaptive-tolerance pass/fail
  (e) summary      -> a table to stdout; non-zero exit if any case FAILs/ERRORs

The C++ driver is invoked per case with flags mapped from the manifest:
  plain : (no mode flags)
  causal: --right 0 --top-left
  window: --left/--right (those that are >= 0) and --top-left | --bottom-right
  mask  : --mask <path>
  always: --dtype <bf16|fp16>, and --scale <f> if scale is not null

torch is only needed transitively (run_torch imports it lazily); harness itself
needs only numpy and the standard library, so it passes ``py_compile`` cleanly.
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

import compare as compare_mod
import gen_inputs
import manifest as mf
import run_torch


def _default_run_dir() -> str:
    """A timestamped run directory under ./runs/."""
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.abspath(os.path.join("runs", stamp))


def _driver_argv(driver: str, man: Dict[str, Any]) -> List[str]:
    """Build the C++ driver command line for one case from its manifest."""
    files = man["files"]
    argv: List[str] = [
        driver,
        "--q",
        files["q"],
        "--k",
        files["k"],
        "--v",
        files["v"],
        "--o",
        files["gpuref_o"],
        "--dtype",
        man["dtype"],
    ]

    if man.get("has_mask"):
        argv += ["--mask", files["mask"]]

    if man["scale"] is not None:
        argv += ["--scale", repr(float(man["scale"]))]

    mode = man["mode"]
    if mode == "causal":
        # top-left causal on a square matrix.
        argv += ["--right", "0", "--top-left"]
    elif mode == "window":
        if man["left"] >= 0:
            argv += ["--left", str(man["left"])]
        if man["right"] >= 0:
            argv += ["--right", str(man["right"])]
        argv += ["--top-left"] if man["top_left"] else ["--bottom-right"]
    # plain / mask: no window/causal flags.

    return argv


def run_driver(driver: str, run_dir: str) -> int:
    """Invoke the C++ driver for every case. Mark errored cases in their manifest.

    Returns the number of cases that errored.
    """
    index = mf.read_index(run_dir)
    n_err = 0
    for name in index["cases"]:
        man = mf.read_manifest(run_dir, name)
        argv = _driver_argv(driver, man)
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            man.setdefault("status", {})
            man["status"]["state"] = "error"
            man["status"]["reason"] = f"driver launch failed: {exc}"
            mf.write_manifest(run_dir, man)
            n_err += 1
            continue

        if proc.returncode != 0:
            man.setdefault("status", {})
            man["status"]["state"] = "error"
            man["status"]["reason"] = (
                f"driver exit {proc.returncode}: "
                f"{proc.stderr.strip()[:500] or '(no stderr)'}"
            )
            mf.write_manifest(run_dir, man)
            n_err += 1
            continue

        # Leave status untouched on success; run_torch sets it to ok/skipped.
    return n_err


def _fmt(value: Optional[float]) -> str:
    """Format an optional float for the summary table."""
    if value is None:
        return "-"
    return f"{value:.3e}"


def print_summary(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Print the results table and return pass/fail/skip/error counts."""
    header = (
        f"{'name':<46} {'dtype':<5} {'shape (BxHqxHkv SqxSkvxD)':<28} "
        f"{'mode':<7} {'backend':<9} {'err':<10} {'budget':<10} "
        f"{'thresh':<10} {'ratio':<8} {'g_vs_fp32':<11} {'a_vs_lp':<11} "
        f"{'RESULT':<6}"
    )
    print(header)
    print("-" * len(header))

    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0, "ERROR": 0}
    for r in results:
        counts[r["result"]] = counts.get(r["result"], 0) + 1
        shape = f"{r['B']}x{r['Hq']}x{r['Hkv']} " f"{r['Sq']}x{r['Skv']}x{r['D']}"
        ratio = r.get("ratio")
        ratio_s = f"{ratio:.2f}" if isinstance(ratio, (int, float)) else "-"
        g_vs_fp32 = _fmt(r.get("gpuref_vs_fp32"))
        if r.get("gpuref_vs_fp32_warn"):
            g_vs_fp32 += "!"
        a_vs_lp = _fmt(r.get("aotriton_vs_lp"))
        if r.get("aotriton_vs_lp_warn"):
            a_vs_lp += "!"
        line = (
            f"{r['name']:<46} {r['dtype']:<5} {shape:<28} "
            f"{r['mode']:<7} {str(r.get('backend') or '-'):<9} "
            f"{_fmt(r.get('err')):<10} {_fmt(r.get('budget')):<10} "
            f"{_fmt(r.get('threshold')):<10} {ratio_s:<8} {g_vs_fp32:<11} "
            f"{a_vs_lp:<11} {r['result']:<6}"
        )
        print(line)
        if r["result"] in ("SKIP", "ERROR", "FAIL") and r.get("reason"):
            print(f"    -> {r['reason']}")

    print()
    print(
        f"Totals: {counts.get('PASS', 0)} pass, {counts.get('FAIL', 0)} fail, "
        f"{counts.get('SKIP', 0)} skip, {counts.get('ERROR', 0)} error."
    )
    return counts


def run_harness(
    driver: str,
    tier: str,
    out_dir: Optional[str] = None,
    fudge: float = 4.0,
    seed_base: int = 0,
) -> int:
    """Run the full pipeline; return a process exit code (0 = all good)."""
    run_dir = os.path.abspath(out_dir) if out_dir else _default_run_dir()
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    print(f"[1/4] Generating tier '{tier}' inputs ...")
    gen_inputs.generate(tier, run_dir, seed_base)

    print("[2/4] Running C++ gpu_ref driver ...")
    run_driver(driver, run_dir)

    print("[3/4] Running torch references (AOTriton + math) ...")
    run_torch.run(run_dir)

    print("[4/4] Comparing ...")
    results = compare_mod.compare(run_dir, fudge)

    print()
    counts = print_summary(results)

    # Non-zero exit if anything failed or errored. Skips are allowed.
    return 1 if (counts.get("FAIL", 0) or counts.get("ERROR", 0)) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--driver",
        required=True,
        help="path to the built sdpa_aotriton_ref_driver binary",
    )
    parser.add_argument(
        "--tier",
        choices=("quick", "medium", "large", "irregular"),
        default="quick",
        help="case tier (default: quick)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="run directory (default: a timestamped dir under ./runs/)",
    )
    parser.add_argument(
        "--fudge",
        type=float,
        default=4.0,
        help="multiplier on the low-precision budget (default: 4.0)",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help="base added to each per-case seed (default: 0)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="(retained for compatibility) keep the run directory; it is never "
        "auto-deleted, so this is currently a no-op kept for explicitness.",
    )
    args = parser.parse_args()

    driver = os.path.abspath(args.driver)
    if not os.path.exists(driver):
        print(f"error: driver not found: {driver}", file=sys.stderr)
        return 2

    return run_harness(
        driver=driver,
        tier=args.tier,
        out_dir=args.out,
        fudge=args.fudge,
        seed_base=args.seed_base,
    )


if __name__ == "__main__":
    raise SystemExit(main())
