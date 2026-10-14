#!/usr/bin/env python3
"""Hoist the main-loop A/B TDM address update + descriptor rebuild above the FREE barrier.

usage: patch_tdm_hoist.py SRC_LOGDIR NAME [scales=1]

The loop issues A, B, MXSA, MXSB right after the FREE barrier (s_barrier_signal/wait -1),
each preceded by ~20 SALU of address update, LDS swap and descriptor rebuild. Only the
tensor_load_to_lds itself needs the barrier (it overwrites LDS other waves were reading),
so the A and B SALU move to just before the s_wait_dscnt that precedes the barrier, and
their loads issue back to back after it.

MXSA/MXSB descriptors alias the A/B SGPRs, so their rebuild must stay after the A/B loads.
scales=1 still hoists their address update and LDS swap (registers not aliased).
SCC and s64 are not live across the hoist point in this loop.
"""
import glob, os, re, subprocess, sys, time

BENCH = os.path.dirname(os.path.abspath(__file__))
LLVM = os.environ.get("LLVM_BIN", "/opt/rocm/lib/llvm/bin")

src, name = sys.argv[1], sys.argv[2]
opts = dict(a.split("=") for a in sys.argv[3:])
hoistScaleAddr = opts.get("scales", "1") == "1"
asms = glob.glob(f"{BENCH}/{src}/kernel.s") or glob.glob(f"{BENCH}/{src}/**/assembly/*.s", recursive=True)
ini = glob.glob(f"{BENCH}/{src}/**/ClientParameters.ini", recursive=True)[0]
lines = open(asms[0]).read().split("\n")

lo = next(i for i, l in enumerate(lines) if l.startswith("/* MAINLOOP_C0 start */"))
hi = next(i for i, l in enumerate(lines) if l.startswith("/* MAINLOOP_C0 end */"))

def blockRange(t):
    s = next(i for i in range(lo, hi) if lines[i].startswith(f"/* TDM addr update: {t} +="))
    e = next(i for i in range(s, hi) if lines[i].startswith(f"tensor_load_to_lds s[sgprtdm{t}Group0"))
    return s, e

ranges = {t: blockRange(t) for t in ("A", "B", "MXSA", "MXSB")}
firstBlock = min(s for s, _ in ranges.values())
bar = firstBlock - 1
assert lines[bar].startswith("s_barrier_wait -1") and lines[bar - 1].startswith("s_barrier_signal -1"), lines[bar]
dscnt = bar - 2
assert lines[dscnt].startswith("s_wait_dscnt"), lines[dscnt]

moved, drop = [], set()
for t in ("A", "B"):
    s, e = ranges[t]
    moved += lines[s:e]
    drop.update(range(s, e))
if hoistScaleAddr:
    for t in ("MXSA", "MXSB"):
        s, e = ranges[t]
        # address update + LDS swap: lines before the descriptor rebuild starts
        k = next(i for i in range(s, e) if lines[i].startswith(f"s_mov_b32 s[sgprtdm{t}Group0+0], 1"))
        moved += lines[s:k]
        drop.update(range(s, k))

out = []
for i, l in enumerate(lines):
    if i == dscnt:
        out += [f"// patched: hoisted TDM address/descriptor setup above FREE barrier"] + moved
    if i in drop:
        continue
    out.append(l)
print(f"hoisted {len(moved)} SALU lines above the FREE barrier", file=sys.stderr)

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-hoist-{name}"
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
