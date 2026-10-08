#!/usr/bin/env python3
"""Change how much the main-loop TDM loads move, for timing only (results are wrong).

usage: patch_tdm.py SRC_LOGDIR NAME [rowsA=N] [rowsB=N] [rowsSA=N] [rowsSB=N] [drop=A,B,SA,SB]

rowsA/rowsB set tile1 (rows per wave, normally 64 for A and B, 2 for the scales) of the loads
inside both main-loop copies. drop removes those tensor_load_to_lds instructions; waits on
tensorcnt then simply pass earlier. Writes logs/<stamp>-tdm-NAME/ for alt_timing.py.
"""
import glob, os, re, subprocess, sys, time

BENCH = os.path.dirname(os.path.abspath(__file__))
LLVM = os.environ.get("LLVM_BIN", "/opt/rocm/lib/llvm/bin")
TC = {"A": "A", "B": "B", "SA": "MXSA", "SB": "MXSB"}

src, name = sys.argv[1], sys.argv[2]
opts = dict(a.split("=", 1) for a in sys.argv[3:])
asm = glob.glob(f"{BENCH}/{src}/**/assembly/*.s", recursive=True)[0]
ini = glob.glob(f"{BENCH}/{src}/**/ClientParameters.ini", recursive=True)[0]
lines = open(asm).read().split("\n")

inLoop, out, changed = False, [], 0
drop = {TC[t] for t in opts.get("drop", "").split(",") if t}
for l in lines:
    if re.match(r"/\* MAINLOOP_C0(_PortB)? start \*/", l):
        inLoop = True
    elif re.match(r"/\* MAINLOOP_C0(_PortB)? end \*/", l):
        inLoop = False
    if inLoop:
        m = re.match(r"s_mov_b32 s\[sgprtdm(\w+?)Group1\+4\], ", l)
        if m and f"rows{next(k for k, v in TC.items() if v == m.group(1))}" in opts:
            n = opts[f"rows{next(k for k, v in TC.items() if v == m.group(1))}"]
            l = f"s_mov_b32 s[sgprtdm{m.group(1)}Group1+4], {n}  // patched tile1"
            changed += 1
        m = re.match(r"tensor_load_to_lds s\[sgprtdm(\w+?)Group0", l)
        if m and m.group(1) in drop:
            changed += 1
            continue
    out.append(l)
print(f"{changed} lines changed", file=sys.stderr)

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-tdm-{name}"
os.makedirs(dst)
s = f"{dst}/kernel.s"
open(s, "w").write("\n".join(out))
subprocess.run([f"{LLVM}/clang", "-x", "assembler", "--target=amdgcn-amd-amdhsa", "-mcode-object-version=4",
                "-c", "-Xclangas", "-target-feature", "-Xclangas", "+real-true16", "-mcpu=gfx1250",
                "-mno-wavefrontsize64", s, "-o", f"{dst}/kernel.o"], check=True, stderr=subprocess.DEVNULL)
subprocess.run([f"{LLVM}/clang", "--target=amdgcn-amd-amdhsa", "-Xlinker", "--build-id=sha1",
                f"{dst}/kernel.o", "-o", f"{dst}/kernel.co"], check=True)
cfg = re.sub(r"^code-object=.*$", f"code-object={dst}/kernel.co", open(ini).read(), flags=re.M)
open(f"{dst}/ClientParameters.ini", "w").write(cfg)
print(dst)
