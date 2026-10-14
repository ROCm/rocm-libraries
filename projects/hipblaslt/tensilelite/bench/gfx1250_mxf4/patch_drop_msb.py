#!/usr/bin/env python3
"""Timing-only floor probe (same convention as patch_tdm.py's drop= variants -- NOT
expected to produce correct results, only used to measure the cycle-cost ceiling):
strip every s_set_vgpr_msb in the recurring MAINLOOP_C0 region.

mxf4's main loop issues one s_set_vgpr_msb before almost every v_wmma_scale/ds_load
(247 of them per iteration, vs. fp8's reference kernel batching 2 VALU/LDS ops per
msb set). This probe measures the absolute best-case cycle ceiling if all 247/iter
were eliminated (real elimination would require reordering ops to group same-bank
operations together, which this probe does NOT do -- it just deletes the instruction
to see if its issue slot is on the critical path at all).

usage: patch_drop_msb.py SRC_LOGDIR NAME [--frac F]
  --frac F   drop only every Nth s_set_vgpr_msb (F=1 default: drop all; F=2: drop half, etc.)
Writes logs/<stamp>-dropmsb-NAME/.
"""
import argparse, glob, os, re, subprocess, sys, time

BENCH = os.path.dirname(os.path.abspath(__file__))
LLVM = os.environ.get("LLVM_BIN", "/opt/rocm/lib/llvm/bin")

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("name")
ap.add_argument("--every", type=int, default=1, help="drop every Nth occurrence (1=all)")
args = ap.parse_args()

asm = glob.glob(f"{BENCH}/{args.src}/**/assembly/*.s", recursive=True)[0]
ini = glob.glob(f"{BENCH}/{args.src}/**/ClientParameters.ini", recursive=True)[0]
lines = open(asm).read().split("\n")

out = []
inLoop = False
seen = 0
dropped = 0
msbRe = re.compile(r"^s_set_vgpr_msb ")
for l in lines:
    if re.match(r"/\* MAINLOOP_C0 start \*/", l):
        inLoop = True
    elif re.match(r"/\* MAINLOOP_C0 end \*/", l):
        inLoop = False

    if inLoop and msbRe.match(l):
        seen += 1
        if seen % args.every == 0:
            dropped += 1
            continue  # drop this instruction
    out.append(l)

print(f"{dropped}/{seen} s_set_vgpr_msb dropped from MAINLOOP_C0 (every={args.every})", file=sys.stderr)
assert seen > 200, f"expected 247 s_set_vgpr_msb in one MAINLOOP_C0 copy, saw {seen}"

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-dropmsb-{args.name}"
os.makedirs(dst)
s = f"{dst}/kernel.s"
open(s, "w").write("\n".join(out))
subprocess.run([f"{LLVM}/clang", "-x", "assembler", "--target=amdgcn-amd-amdhsa", "-mcode-object-version=4",
                "-c", "-Xclangas", "-target-feature", "-Xclangas", "+real-true16", "-mcpu=gfx1250",
                "-mno-wavefrontsize64", s, "-o", f"{dst}/kernel.o"], check=True)
subprocess.run([f"{LLVM}/clang", "--target=amdgcn-amd-amdhsa", "-Xlinker", "--build-id=sha1",
                f"{dst}/kernel.o", "-o", f"{dst}/kernel.co"], check=True)
cfg = re.sub(r"^code-object=.*$", f"code-object={dst}/kernel.co", open(ini).read(), flags=re.M)
open(f"{dst}/ClientParameters.ini", "w").write(cfg)
print(dst)
