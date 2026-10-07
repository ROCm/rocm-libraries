#!/usr/bin/env python3
"""Insert a fixed delay at one point of both main-loop copies and rebuild the code object.

usage: inject_delay.py SRC_LOGDIR NAME POINT CYCLES

POINT is where the delay goes in each main-loop copy (even and odd waves):
  none     no delay (round-trip control)
  bar1     right after the first barrier (start of the TDM descriptor window)
  pretc    right before s_wait_tensorcnt (after the last TDM load)
  bar2     right after the second barrier (before the read-address swaps)
  phase1   before the 16th WMMA (middle of phase 1)
  phase2   before the 16th WMMA after the second barrier (middle of phase 2)
  tcnt     no delay; CYCLES replaces the count of the main-loop s_wait_tensorcnt instead.
           8 waits for the loads of two iterations back, which times a 3-deep TDM pipeline
           (results are wrong: the loads overwrite the buffer being read)

The delay is CYCLES/16 copies of `s_nop 15`. Writes logs/<stamp>-inj-NAME/ with the
patched .s, the code object and a ClientParameters.ini pointing at it, for alt_timing.py.
If the iteration grows by about CYCLES, that point is on the critical path; if not, the
wave would have waited there anyway.
"""
import glob, os, re, subprocess, sys, time

BENCH = os.path.dirname(os.path.abspath(__file__))
LLVM = "/opt/rocm-10.1.0a20260908+bkc.20260917/lib/llvm/bin"

src, name, point, cycles = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
asm = glob.glob(f"{BENCH}/{src}/**/assembly/*.s", recursive=True)[0]
ini = glob.glob(f"{BENCH}/{src}/**/ClientParameters.ini", recursive=True)[0]
lines = open(asm).read().split("\n")

starts = [i for i, l in enumerate(lines) if re.match(r"/\* MAINLOOP_C0(_PortB)? start \*/", l)]
ends = [i for i, l in enumerate(lines) if re.match(r"/\* MAINLOOP_C0(_PortB)? end \*/", l)]
assert len(starts) == len(ends) >= 1, "main-loop copies not found"


def site(a, b):
    body = range(a, b)
    bars = [i for i in body if lines[i].startswith("s_barrier_wait")]
    wmma = [i for i in body if lines[i].startswith("v_wmma")]
    if point == "bar1":
        return bars[0] + 1
    if point == "pretc":
        return next(i for i in body if lines[i].startswith("s_wait_tensorcnt"))
    if point == "bar2":
        return bars[1] + 1
    if point == "phase1":
        return wmma[15]
    if point == "phase2":
        return [i for i in wmma if i > bars[1]][15]
    raise SystemExit(f"unknown point {point}")


delay = [f"s_nop 15                                           // injected delay ({point})"] * (cycles // 16)
if point == "tcnt":
    for a, b in zip(starts, ends):
        at = next(i for i in range(a, b) if lines[i].startswith("s_wait_tensorcnt"))
        lines[at] = re.sub(r"s_wait_tensorcnt\s+\d+", f"s_wait_tensorcnt {cycles}", lines[at])
elif point != "none":
    for a, b in sorted(zip(starts, ends), reverse=True):
        at = site(a, b)
        lines[at:at] = delay

out = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-inj-{name}"
os.makedirs(out)
s = f"{out}/kernel.s"
open(s, "w").write("\n".join(lines))
subprocess.run([f"{LLVM}/clang", "-x", "assembler", "--target=amdgcn-amd-amdhsa",
                f"-mcode-object-version={os.environ.get('CO_VERSION', '4')}", "-c",
                "-Xclangas", "-target-feature", "-Xclangas", "+real-true16",
                "-mcpu=gfx1250", "-mno-wavefrontsize64", s, "-o", f"{out}/kernel.o"], check=True)
subprocess.run([f"{LLVM}/clang", "--target=amdgcn-amd-amdhsa", "-Xlinker", "--build-id=sha1",
                f"{out}/kernel.o", "-o", f"{out}/kernel.co"], check=True)
cfg = re.sub(r"^code-object=.*$", f"code-object={out}/kernel.co", open(ini).read(), flags=re.M)
open(f"{out}/ClientParameters.ini", "w").write(cfg)
print(out)
