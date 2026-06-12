#!/usr/bin/env python3
"""Apples-to-apples GEMM A/B: bridge kernel vs old-TE kernel, ONE harness.

Why this exists
---------------
The earlier sweep (allsweep6144rcrfp16.py) compared the bridge's dispatcher
measurement against old TE's *standalone benchmark binary*
(benchmark_gemm_universal_<stem>). That comparison is NOT apples-to-apples:
the device kernel is byte-identical, yet old TE's standalone binary reports
~18-20% lower TFLOPS at e.g. 1024^3 / compv4. rocprof shows the identical
kernel genuinely runs longer in that process -- ~+8% cycles plus a lower
sustained SCLK -- a power/clock + execution-environment artifact of that
binary, NOT a bridge speedup, compiler difference, or kernel difference.
(See diagnose.md sec.4.)

This harness removes the artifact: it builds the OLD-TE kernel into a .so from
old TE's own generated header and runs BOTH the bridge kernel and the old-TE
kernel through the SAME worker (run_one_gemm_kernel.py). Measured this way the
gap collapses to ~1%, which is the honest result.

Usage:
  python3 ab_same_harness.py                 # default kernel list + shapes
  python3 ab_same_harness.py <stem> [<stem>...]
"""
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/AMD/muozturk/New_project/rocm-libraries/projects/composablekernel")
DISP = ROOT / "dispatcher"
GEN = DISP / "build" / "generated_kernels"
SRC = DISP / "bindings" / "ctypes" / "gemm_ctypes_lib.cpp"
STATIC = DISP / "build" / "libck_tile_dispatcher.a"
BR_SO_DIR = DISP / "build" / "examples"
WORKER = ROOT / "tile_engine/ops/gemm/run_one_gemm_kernel.py"
# old-TE generated single-kernel headers (develop-parity worktree)
OLD_GEN = Path(
    "/home/AMD/muozturk/New_project/rocm-libraries/.claude/worktrees/develop-parity"
    "/projects/composablekernel/build/tile_engine/ops/gemm/gemm_universal/fp16/rcr"
)
OUT = DISP / "parity_diag" / "regression" / "_ab_same_harness_build"
ARCH = os.environ.get("GFX_ARCH", "gfx942")
DEVICE = os.environ.get("PARITY_DEVICE", "0")
REPEATS = int(os.environ.get("AB_REPEATS", "3"))

SHAPES = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048),
          (1024, 512, 256), (4096, 4096, 4096)]

DEFAULT_STEMS = [
    "fp16_rcr_compv4_default_intrawave_False_False_False_False_64x128x64_2x2x1_32x32x16",
    "fp16_rcr_compv4_cshuffle_intrawave_False_False_False_False_64x128x64_1x4x1_32x32x16",
    "fp16_rcr_compv4_default_intrawave_False_False_False_False_128x128x64_4x1x1_32x32x16",
]

PYPATH = os.pathsep.join([str(DISP / "python"), str(ROOT / "tile_engine/ops/gemm")])


def build_old_so(stem: str) -> Path | None:
    """Compile old TE's generated kernel header into a bridge-loadable .so."""
    hdr = OLD_GEN / f"gemm_universal_single_{stem}.hpp"
    if not hdr.exists():
        return None
    OUT.mkdir(parents=True, exist_ok=True)
    obj = OUT / f"{stem}.o"
    lib = OUT / f"libold_{stem}.so"
    common = [
        "-fPIC", "-O3",
        f"-I{DISP / 'include'}", f"-I{ROOT / 'include'}", f"-I{ROOT}", f"-I{GEN}",
        "-DCK_TILE_SINGLE_KERNEL_INCLUDE", f"-include{hdr}", "-D__HIP_PLATFORM_AMD__",
        f"--offload-arch={ARCH}", f'-DGFX_ARCH="{ARCH}"',
        "-Wno-undefined-func-template", "-Wno-float-equal",
    ]
    cc = subprocess.run(["/opt/rocm/bin/hipcc", "-c", *common, str(SRC), "-o", str(obj)],
                        capture_output=True)
    if cc.returncode != 0:
        return None
    ln = subprocess.run(["/opt/rocm/bin/hipcc", "-shared", "-fPIC",
                         f"--offload-arch={ARCH}", "--hip-link",
                         str(obj), str(STATIC), "-o", str(lib)], capture_output=True)
    return lib if ln.returncode == 0 else None


def meas(so: Path, M: int, N: int, K: int) -> float | None:
    if not so or not Path(so).exists():
        return None
    payload = json.dumps({"so_path": str(so), "problem": {"M": M, "N": N, "K": K},
                          "kernel_name": "x"})
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = DEVICE
    env["GEMM_PYPATH"] = PYPATH
    best = None
    for _ in range(REPEATS):
        p = subprocess.run([sys.executable, str(WORKER)], input=payload.encode(),
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
        for line in p.stdout.decode().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("ok"):
                best = d["tflops"] if best is None else max(best, d["tflops"])
    return best


def main():
    stems = sys.argv[1:] or DEFAULT_STEMS
    print(f"{'shape':>14} {'bridge':>9} {'oldTE':>9} {'gap%':>7}  kernel")
    for stem in stems:
        old_so = build_old_so(stem)
        br_so = BR_SO_DIR / f"libgemm_{stem}.so"
        if old_so is None:
            print(f"  [skip: no old-TE header] {stem}")
            continue
        for (M, N, K) in SHAPES:
            b = meas(br_so, M, N, K)
            o = meas(old_so, M, N, K)
            gap = (b - o) / o * 100 if (b and o) else float("nan")
            print(f"{f'{M}x{N}x{K}':>14} {b or float('nan'):9.2f} "
                  f"{o or float('nan'):9.2f} {gap:7.2f}  {stem[:40]}")


if __name__ == "__main__":
    main()
