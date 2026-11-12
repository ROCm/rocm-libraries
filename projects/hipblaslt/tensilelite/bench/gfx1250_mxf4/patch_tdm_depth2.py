#!/usr/bin/env python3
"""Drain A/B TDM before scale issues in the hoisted main loop.

Inserts `s_wait_tensorcnt 2` immediately before the MXSA descriptor rebuild in
MAINLOOP_C0, so MXSA/MXSB are issued against a 0-deep (or low) outstanding
count instead of as the 3rd/4th loads behind in-flight A/B.

Does not reduce issue count. Tests the hypothesis that MXSB's ~289-cycle issue
Idle is TDM queue-depth backpressure.

usage: patch_tdm_depth2.py SRC_LOGDIR NAME

Measured on logs/20261014-205500-hoist-sk3_24 (CD[2,4], GPU 3):
  validate 3/3 smoke PASSED
  alt_timing.py 6 rounds: 1.011x (202.2 -> 199.9 us), faster in 5/6.
"""
import glob, os, re, subprocess, sys, time

BENCH = os.path.dirname(os.path.abspath(__file__))
LLVM = os.environ.get("LLVM_BIN", "/opt/rocm/lib/llvm/bin")

src, name = sys.argv[1], sys.argv[2]
src = src if os.path.isabs(src) else f"{BENCH}/{src}"
asms = glob.glob(f"{src}/**/assembly/*.s", recursive=True) + glob.glob(f"{src}/kernel.s")
inis = glob.glob(f"{src}/**/ClientParameters.ini", recursive=True) + glob.glob(f"{src}/ClientParameters.ini")
asm, ini = asms[0], inis[0]
lines = open(asm).read().split("\n")

out, in_loop, n_ins = [], False, 0
for l in lines:
    if "MAINLOOP_C0 start" in l:
        in_loop = True
    if "MAINLOOP_C0 end" in l:
        in_loop = False
    if in_loop and l.startswith("s_mov_b32 s[sgprtdmMXSAGroup0+0], 1"):
        out.append("s_wait_tensorcnt 2                              "
                   "// patched: drain A/B before scale TDMs")
        n_ins += 1
    out.append(l)
assert n_ins == 1, n_ins

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-tdmdepth2-{name}"
os.makedirs(dst)
s = f"{dst}/kernel.s"
open(s, "w").write("\n".join(out))
subprocess.run([f"{LLVM}/clang", "-x", "assembler", "--target=amdgcn-amd-amdhsa",
                "-mcode-object-version=4", "-c", "-Xclangas", "-target-feature",
                "-Xclangas", "+real-true16", "-mcpu=gfx1250", "-mno-wavefrontsize64",
                s, "-o", f"{dst}/kernel.o"], check=True)
subprocess.run([f"{LLVM}/clang", "--target=amdgcn-amd-amdhsa", "-Xlinker",
                "--build-id=sha1", f"{dst}/kernel.o", "-o", f"{dst}/kernel.co"],
               check=True)
open(f"{dst}/ClientParameters.ini", "w").write(
    re.sub(r"^code-object=.*$", f"code-object={dst}/kernel.co",
           open(ini).read(), flags=re.M))
print(dst)
