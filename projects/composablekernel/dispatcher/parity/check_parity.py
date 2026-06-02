#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
End-to-end parity orchestrator -- parity deliverable (e/f).

Ties the four building blocks together to prove the dispatcher reproduces Tile
Engine, in three escalating stages:

  1. identifier   -- Python encode_identifier vs C++ KernelKey::encode_identifier
                     for every config translated from the TE JSON. Pure host C++
                     + python; ALWAYS runnable here (no GPU, no hipcc, no cmake).
                     This is the offline<->runtime registry-key guarantee.

  2. numerical    -- drive codegen for ONE config, build the single-kernel harness
                     (deliverable d), run it with -verify=1 over several problem
                     sizes, and require PASSED. If a Tile Engine build dir is
                     given, run the matching TE benchmark with -verify and require
                     it passes too -- so both producers agree against the same CPU
                     reference. GPU-gated (needs a ROCm device + hipcc).

  3. performance  -- compare dispatcher harness throughput (TFLOP/s) against the
                     Tile Engine benchmark's reported tflops within a relative
                     tolerance. GPU-gated, and needs --te-build-dir.

The user's ordering -- "numerical parity first, then performance parity" -- is
enforced: a numerical failure short-circuits before performance is judged.

This box has g++/hipcc/python but NO GPU and NO cmake, so stages 2-3 are gated:
without a GPU they report SKIPPED (not FAILED), and the TE comparison is skipped
unless --te-build-dir points at a real Tile Engine build. Use --dry-run anywhere
to print the exact command plan without executing anything.

Usage:
    # Always-on, CPU-only: the registry-key guarantee.
    python check_parity.py configs/single_fp16_rcr.json

    # Full plan without running (safe on this CPU-only box):
    python check_parity.py configs/single_fp16_rcr.json --dry-run

    # On a GPU node, dispatcher-only numerical+perf:
    python check_parity.py configs/single_fp16_rcr.json --sizes 512x512x512,1024x1024x1024

    # On a GPU node, full dispatcher-vs-TE numerical + performance parity:
    python check_parity.py configs/single_fp16_rcr.json \
        --te-build-dir /path/to/tile_engine/build --perf-tol 0.10
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from identifier import encode_identifier
from te_to_dispatcher import TranslationError, translate_file

_HERE = Path(__file__).resolve().parent
_DRIVE_CODEGEN = _HERE / "drive_codegen.py"
_BUILD_HARNESS = _HERE / "build_harness.sh"
_CHECK_IDENTIFIER = _HERE / "check_identifier_parity.py"


# --------------------------------------------------------------------------- #
# Naming: the joint key shared by the dispatcher header and the TE executable.
# --------------------------------------------------------------------------- #
_SEP = "=" * 72

# Sentinel used by run_harness() when dry_run=True. Named constant to make the
# contract explicit for _adjudicate_numerical().
_DRYRUN_VERDICT = "DRYRUN"


def _capitalize_bool(b: bool) -> str:
    """Return 'True' or 'False' matching Python str(bool)."""
    return str(bool(b)).capitalize()


def te_kernel_name(cfg: Dict[str, Any]) -> str:
    """Raw-TE kernel name used by BOTH codegen header and TE benchmark target.

    codegen (drive_codegen -> unified_gemm_codegen) names the header
    ``gemm_<name>.hpp`` and Tile Engine names its executable
    ``benchmark_gemm_universal_<name>``, where ``<name>`` is built from the *raw*
    TE trait strings (``compv3``/``intrawave``/``default``) -- NOT the canonical
    dispatcher form (where e.g. scheduler ``default`` -> ``auto``). So this is
    distinct from encode_identifier() and is the right key for locating files.

    For ``preshufflev2`` configs, ``unified_gemm_codegen.py``'s
    ``KernelNaming.generate()`` appends ``_preshuffle`` to the kernel name.
    We mirror that here so file-based lookups succeed.
    """
    te = cfg["_te"]
    alg = cfg["algorithm"]
    cap = _capitalize_bool
    name = (
        f"{te['datatype']}_{te['layout']}_"
        f"{te['pipeline']}_{te['epilogue']}_{te['scheduler']}_"
        f"{cap(alg['pad_m'])}_{cap(alg['pad_n'])}_{cap(alg['pad_k'])}_{cap(alg['persistent'])}_"
        f"{alg['tile_m']}x{alg['tile_n']}x{alg['tile_k']}_"
        f"{alg['warp_m']}x{alg['warp_n']}x{alg['warp_k']}_"
        f"{alg['warp_tile_m']}x{alg['warp_tile_n']}x{alg['warp_tile_k']}"
    )
    # KernelNaming.generate() appends _preshuffle for GemmVariant.PRESHUFFLE.
    if te["pipeline"] in ("preshufflev2",):
        name += "_preshuffle"
    return name


def dispatcher_header_path(output_dir: Path, kernel_set: str, cfg: Dict[str, Any]) -> Path:
    return output_dir / kernel_set / f"gemm_{te_kernel_name(cfg)}.hpp"


def te_benchmark_name(cfg: Dict[str, Any]) -> str:
    return f"benchmark_gemm_universal_{te_kernel_name(cfg)}"


# --------------------------------------------------------------------------- #
# Environment probes.
# --------------------------------------------------------------------------- #
def has_gpu() -> bool:
    """True if a ROCm GPU looks present.

    rocminfo is authoritative when available -- it enumerates HSA agents, so a
    real GPU shows up as a ``gfx<arch>`` agent name. We require that, because
    ``/dev/kfd`` alone is a false positive: the amdkfd driver node exists on
    nodes that have the driver loaded but no GPU bound (as on this CPU-only box).
    Only when rocminfo is entirely absent do we fall back to the device node.
    """
    rocminfo = shutil.which("rocminfo")
    if rocminfo:
        try:
            out = subprocess.run([rocminfo], capture_output=True, text=True, timeout=30)
            return out.returncode == 0 and re.search(r"\bgfx\d", out.stdout) is not None
        except (OSError, subprocess.SubprocessError):
            return False
    return Path("/dev/kfd").exists()


def have_hipcc() -> bool:
    return shutil.which("hipcc") is not None


# --------------------------------------------------------------------------- #
# Problem-size parsing.
# --------------------------------------------------------------------------- #
def parse_sizes(spec: str) -> List[Tuple[int, int, int]]:
    """'512x512x512,1024x1024x1024' -> [(512,512,512),(1024,1024,1024)]."""
    sizes: List[Tuple[int, int, int]] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        m = re.fullmatch(r"(\d+)x(\d+)x(\d+)", token)
        if not m:
            raise ValueError(f"bad size {token!r} (expected MxNxK)")
        sizes.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    if not sizes:
        raise ValueError("no problem sizes parsed")
    return sizes


# --------------------------------------------------------------------------- #
# Output parsing.
# --------------------------------------------------------------------------- #
_RE_GFLOPS = re.compile(r"\(([\d.]+)\s*GFLOP/s\)")


def parse_harness_output(text: str) -> Dict[str, Any]:
    """Pull verdict + throughput out of harness.cpp stdout."""
    res: Dict[str, Any] = {"verdict": "UNKNOWN", "tflops": None, "detail": ""}
    if "SKIPPED" in text:
        res["verdict"] = "SKIPPED"
        for line in text.splitlines():
            if line.startswith("SKIPPED"):
                res["detail"] = line.strip()
    elif "PASSED" in text:
        res["verdict"] = "PASSED"
    elif "FAILED" in text:
        res["verdict"] = "FAILED"
    m = _RE_GFLOPS.search(text)
    if m:
        res["tflops"] = float(m.group(1)) / 1000.0  # GFLOP/s -> TFLOP/s
    return res


def parse_te_csv(csv_path: Path) -> Optional[Dict[str, float]]:
    """Read the last data row of a TE benchmark CSV (latency/tflops/bandwidth).

    TE only writes a row when the kernel verifies (verify enabled), so a present
    row IS the numerical pass signal as well as the perf source.
    """
    if not csv_path.exists():
        return None
    lines = [ln for ln in csv_path.read_text().splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    header = lines[0].split(",")
    row = lines[-1].split(",")
    cols = dict(zip(header, row))

    def _column_float(key: str) -> Optional[float]:
        for k, v in cols.items():
            if k.startswith(key):
                try:
                    return float(v)
                except ValueError:
                    return None
        return None

    return {
        "latency_ms": _column_float("latency"),
        "tflops": _column_float("tflops"),
        "bandwidth": _column_float("bandwidth"),
    }


# --------------------------------------------------------------------------- #
# Stage 1: identifier parity (always runnable).
# --------------------------------------------------------------------------- #
def stage_identifier(config_path: Path, dry_run: bool) -> bool:
    print(_SEP)
    print("STAGE 1/3  identifier parity  (python encode_identifier vs C++ KernelKey)")
    print(_SEP)
    cmd = [sys.executable, str(_CHECK_IDENTIFIER), str(config_path)]
    print("  $ " + " ".join(cmd))
    if dry_run:
        print("  [dry-run] not executed")
        return True
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    ok = proc.returncode == 0
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- #
# Stage 2/3 helpers: codegen + build (shared setup for GPU stages).
# --------------------------------------------------------------------------- #
def drive_codegen(config_path: Path, index: int, output_dir: Path, kernel_set: str,
                  dry_run: bool) -> Tuple[bool, List[str]]:
    cmd = [
        sys.executable, str(_DRIVE_CODEGEN), str(config_path),
        "--index", str(index),
        "--output-dir", str(output_dir),
        "--kernel-set", kernel_set,
    ]
    plan = ["  $ " + " ".join(cmd)]
    print(plan[0])
    if dry_run:
        print("  [dry-run] not executed")
        return True, plan
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode == 0, plan


def build_harness(header: Path, arch: str, dry_run: bool) -> bool:
    cmd = ["bash", str(_BUILD_HARNESS), str(header), arch]
    print("  $ " + " ".join(cmd))
    if dry_run:
        print("  [dry-run] not executed")
        return True
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode == 0


def run_harness(sizes: List[Tuple[int, int, int]], dry_run: bool
                ) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
    harness = _HERE / "harness"
    results: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for (m, n, k) in sizes:
        cmd = [str(harness), f"-m={m}", f"-n={n}", f"-k={k}", "-verify=1"]
        print("  $ " + " ".join(cmd))
        if dry_run:
            print("  [dry-run] not executed")
            results[(m, n, k)] = {"verdict": _DRYRUN_VERDICT, "tflops": None, "detail": ""}
            continue
        proc = subprocess.run(cmd, capture_output=True, text=True)
        sys.stdout.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        results[(m, n, k)] = parse_harness_output(proc.stdout)
    return results


def find_te_executable(te_build_dir: Path, cfg: Dict[str, Any]) -> Optional[Path]:
    name = te_benchmark_name(cfg)
    direct = te_build_dir / name
    if direct.is_file():
        return direct
    matches = [p for p in te_build_dir.rglob(name) if p.is_file()]
    return matches[0] if matches else None


def run_te_benchmark(exe: Path, m: int, n: int, k: int, csv_stub: Path, dry_run: bool
                     ) -> Optional[Dict[str, float]]:
    csv_stub.with_suffix(".csv").unlink(missing_ok=True)
    # --warmup 3 --repeat 20 mirrors the harness's stream_config (cold_niters_=3,
    # nrepeat_=20) so the two stacks measure on comparable footing.
    cmd = [str(exe), f"-m={m}", f"-n={n}", f"-k={k}", "-verify=1",
           "-warmup=3", "-repeat=20",
           f"-csv_filename={csv_stub}"]
    print("  $ " + " ".join(cmd))
    if dry_run:
        print("  [dry-run] not executed")
        return None
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    return parse_te_csv(csv_stub.with_suffix(".csv"))


# --------------------------------------------------------------------------- #
# Orchestration helpers.
# --------------------------------------------------------------------------- #
def _fail(summary: Dict[str, str], **stages: str) -> int:
    """Merge extra stage results into summary, print it, return 1."""
    summary.update(stages)
    _print_summary(summary)
    return 1


# --------------------------------------------------------------------------- #
# Main orchestration.
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> int:
    configs = translate_file(args.config)
    if not configs:
        print(f"error: no valid dispatcher configs from {args.config}", file=sys.stderr)
        return 1
    if not (0 <= args.index < len(configs)):
        print(f"error: index {args.index} out of range (0..{len(configs)-1})",
              file=sys.stderr)
        return 1

    cfg = configs[args.index]
    identifier = encode_identifier(cfg)
    sizes = parse_sizes(args.sizes)

    print(f"config file : {args.config}")
    print(f"config #    : {args.index} of {len(configs)}")
    print(f"identifier  : {identifier}")
    print(f"kernel name : {te_kernel_name(cfg)}")
    print(f"arch        : {args.arch}")
    print(f"sizes       : {', '.join(f'{m}x{n}x{k}' for m, n, k in sizes)}")
    gpu = has_gpu()
    print(f"gpu present : {gpu}   hipcc: {have_hipcc()}")
    print(f"te build dir: {args.te_build_dir or '(none -- dispatcher-only)'}")
    print()

    summary: Dict[str, str] = {}

    # ---- Stage 1: identifier (always) ------------------------------------- #
    id_ok = stage_identifier(args.config, args.dry_run)
    summary["identifier"] = "PASS" if id_ok else "FAIL"
    if not id_ok and not args.dry_run:
        return _fail(summary)

    # ---- Gating for GPU stages -------------------------------------------- #
    gpu_runnable = args.dry_run or (gpu and have_hipcc())
    if not gpu_runnable:
        reason = "no GPU" if not gpu else "no hipcc"
        print()
        print(_SEP)
        print(f"STAGE 2/3  numerical + performance parity  -- SKIPPED ({reason})")
        print(_SEP)
        print("  Build/run requires a ROCm GPU node. Re-run there, or use --dry-run")
        print("  to see the full command plan.")
        summary["numerical"] = f"SKIPPED ({reason})"
        summary["performance"] = f"SKIPPED ({reason})"
        _print_summary(summary)
        return 0 if id_ok else 1

    # ---- Codegen + build (shared) ----------------------------------------- #
    print()
    print(_SEP)
    print("STAGE 2/3  numerical parity  (codegen -> build harness -> verify)")
    print(_SEP)
    cg_ok, _ = drive_codegen(args.config, args.index, args.output_dir,
                             args.kernel_set, args.dry_run)
    if not cg_ok:
        return _fail(summary,
                     numerical="FAIL (codegen)",
                     performance="SKIPPED (numerical failed)")

    header = dispatcher_header_path(args.output_dir, args.kernel_set, cfg)
    if not args.dry_run and not header.exists():
        print(f"error: expected generated header not found: {header}", file=sys.stderr)
        return _fail(summary,
                     numerical="FAIL (missing header)",
                     performance="SKIPPED (numerical failed)")

    if not build_harness(header, args.arch, args.dry_run):
        return _fail(summary,
                     numerical="FAIL (build)",
                     performance="SKIPPED (numerical failed)")

    # ---- Run dispatcher harness ------------------------------------------- #
    disp = run_harness(sizes, args.dry_run)

    # ---- Run TE benchmark (optional) -------------------------------------- #
    te_exe: Optional[Path] = None
    if args.te_build_dir:
        te_exe = find_te_executable(args.te_build_dir, cfg)
        if te_exe is None and not args.dry_run:
            print(f"  warning: TE executable {te_benchmark_name(cfg)} not found "
                  f"under {args.te_build_dir}; running dispatcher-only.")

    te_results: Dict[Tuple[int, int, int], Optional[Dict[str, float]]] = {}
    if te_exe is not None or args.dry_run:
        for (m, n, k) in sizes:
            if te_exe is None:  # dry-run with no real exe
                print(f"  $ {te_benchmark_name(cfg)} -m={m} -n={n} -k={k} "
                      f"-verify=1 -csv_filename=<stub>")
                print("  [dry-run] not executed")
                te_results[(m, n, k)] = None
                continue
            stub = _HERE / f"te_{m}x{n}x{k}"
            te_results[(m, n, k)] = run_te_benchmark(te_exe, m, n, k, stub, args.dry_run)

    # ---- Adjudicate numerical --------------------------------------------- #
    has_te_exe = bool(te_exe)
    num_ok = _adjudicate_numerical(sizes, disp, te_results, has_te_exe, args.dry_run)
    summary["numerical"] = "PASS" if num_ok else "FAIL"

    if not num_ok and not args.dry_run:
        return _fail(summary, performance="SKIPPED (numerical failed)")

    # ---- Stage 3: performance --------------------------------------------- #
    print()
    print(_SEP)
    print("STAGE 3/3  performance parity  (dispatcher TFLOP/s vs Tile Engine)")
    print(_SEP)
    if te_exe is None and not args.dry_run:
        print("  No Tile Engine build given -- reporting dispatcher throughput only.")
        for (m, n, k) in sizes:
            t = disp.get((m, n, k), {}).get("tflops")
            print(f"  {m}x{n}x{k}: dispatcher={_fmt(t)} TFLOP/s")
        summary["performance"] = "INFO (dispatcher-only, no TE baseline)"
    else:
        perf_ok = _adjudicate_performance(sizes, disp, te_results, args.perf_tol,
                                          args.dry_run)
        summary["performance"] = "PASS" if perf_ok else "FAIL"
        if not perf_ok and not args.dry_run:
            return _fail(summary)

    _print_summary(summary)
    return 0


def _adjudicate_numerical(
    sizes: List[Tuple[int, int, int]],
    disp: Dict[Tuple[int, int, int], Dict[str, Any]],
    te_results: Dict[Tuple[int, int, int], Optional[Dict[str, float]]],
    has_te_exe: bool,
    dry_run: bool,
) -> bool:
    print()
    print("  numerical verdict per size:")
    ok = True
    for sz in sizes:
        m, n, k = sz
        d = disp.get(sz, {})
        dv = d.get("verdict", "UNKNOWN")
        line = f"    {m}x{n}x{k}: dispatcher={dv}"
        if has_te_exe:
            te = te_results.get(sz)
            te_pass = te is not None and te.get("tflops") is not None
            line += f"  tile_engine={'PASSED' if te_pass else 'NO-ROW/FAILED'}"
            if not dry_run and not (dv == "PASSED" and te_pass):
                if dv != "SKIPPED":
                    ok = False
        else:
            if not dry_run and dv not in ("PASSED", "SKIPPED", _DRYRUN_VERDICT):
                ok = False
        print(line)
    return ok


def _adjudicate_performance(
    sizes: List[Tuple[int, int, int]],
    disp: Dict[Tuple[int, int, int], Dict[str, Any]],
    te_results: Dict[Tuple[int, int, int], Optional[Dict[str, float]]],
    perf_tol: float,
    dry_run: bool,
) -> bool:
    print(f"  relative tolerance: {perf_tol:.0%}")
    ok = True
    for sz in sizes:
        m, n, k = sz
        dt = disp.get(sz, {}).get("tflops")
        te = te_results.get(sz)
        tt = te.get("tflops") if te else None
        if dry_run:
            print(f"    {m}x{n}x{k}: dispatcher=<t> tile_engine=<t> (dry-run)")
            continue
        if dt is None or tt is None or tt == 0:
            print(f"    {m}x{n}x{k}: dispatcher={_fmt(dt)} tile_engine={_fmt(tt)} "
                  f"-> INSUFFICIENT DATA")
            ok = False
            continue
        rel = abs(dt - tt) / tt
        verdict = "OK" if rel <= perf_tol else "OUT-OF-TOL"
        if rel > perf_tol:
            ok = False
        print(f"    {m}x{n}x{k}: dispatcher={dt:.1f} tile_engine={tt:.1f} TFLOP/s "
              f"(rel {rel:.1%}) -> {verdict}")
    return ok


def _fmt(x: Optional[float]) -> str:
    return f"{x:.1f}" if isinstance(x, float) else "n/a"


def _print_summary(summary: Dict[str, str]) -> None:
    print()
    print(_SEP)
    print("PARITY SUMMARY")
    print(_SEP)
    for stage in ("identifier", "numerical", "performance"):
        if stage in summary:
            print(f"  {stage:<12}: {summary[stage]}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("config", type=Path, help="Tile Engine config JSON")
    ap.add_argument("--index", type=int, default=0,
                    help="Which translated config to check (default 0)")
    ap.add_argument("--sizes",
                    default="512x512x512,1024x1024x1024,2048x2048x2048,513x511x33",
                    help="Comma-separated MxNxK problem sizes. "
                         "513x511x33 is intentionally non-tile-aligned to exercise "
                         "the padding code path (pad_m/n/k=True configs).")
    ap.add_argument("--arch", default="gfx942", help="GPU arch for harness build")
    ap.add_argument("--output-dir", type=Path, default=_HERE / "generated",
                    help="Codegen output directory")
    ap.add_argument("--kernel-set", default="parity_single",
                    help="Kernel set subdirectory name")
    ap.add_argument("--te-build-dir", type=Path, default=None,
                    help="Tile Engine build dir containing benchmark_gemm_universal_* "
                         "executables (enables dispatcher-vs-TE comparison)")
    ap.add_argument("--perf-tol", type=float, default=0.10,
                    help="Relative throughput tolerance for performance parity "
                         "(default 0.10 = 10%%)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the full command plan without executing")
    args = ap.parse_args()

    try:
        return run(args)
    except (TranslationError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
