#!/usr/bin/env python3
"""Move the main-loop cluster barrier to guard the buffer the multicast TDM load overwrites.

usage: patch_cluster_safe.py SRC_LOGDIR NAME

Generated subtile loop (per iteration i): FREE barrier -1, TDM loads writing buffer i%2
(multicast into every peer's LDS), wave-0 cluster signal -3, ..., cluster wait -3 before the
back-branch. That wait only proves peers passed iteration i's signal, while iteration i+1's
multicast overwrites buffer (i+1)%2, which a lagging peer may still be reading.

This patch removes the loop's signal and wait and emits, right after the FREE barrier:
    wave 0: s_barrier_signal -3 ; all waves: s_barrier_wait -3
so no WG issues a multicast load before every peer has freed that buffer. Signal/wait
counts per iteration are unchanged (one each).
"""
import glob, os, re, subprocess, sys, time

BENCH = os.path.dirname(os.path.abspath(__file__))
LLVM = os.environ.get("LLVM_BIN", "/opt/rocm/lib/llvm/bin")

src, name = sys.argv[1], sys.argv[2]
asms = glob.glob(f"{BENCH}/{src}/kernel.s") or glob.glob(f"{BENCH}/{src}/**/assembly/*.s", recursive=True)
ini = glob.glob(f"{BENCH}/{src}/**/ClientParameters.ini", recursive=True)[0]
lines = open(asms[0]).read().split("\n")

lo = next(i for i, l in enumerate(lines) if l.startswith("label_LoopBeginL:"))
hi = next(i for i in range(lo, len(lines)) if lines[i].startswith("s_cbranch_scc0 label_LoopBeginL"))

drop = set()
sig = next(i for i in range(lo, hi) if lines[i].startswith("s_barrier_signal -3"))
skipLabel = next(i for i in range(sig, hi) if lines[i].startswith("label_skipCBPreSignal"))
br = next(i for i in range(sig, lo, -1) if lines[i].startswith("s_cbranch_scc0 label_skipCBPreSignal"))
cmp_ = next(i for i in range(br, lo, -1) if lines[i].startswith("s_cmp_eq_u32 s[sgprWaveIdx], 0"))
drop.update([cmp_, br, sig, skipLabel])
drop.update(i for i in range(sig, skipLabel) if lines[i].strip() == ".align 16")
wt = next(i for i in range(hi, lo, -1) if lines[i].startswith("s_barrier_wait -3"))
drop.add(wt)

firstTdm = next(i for i in range(lo, hi) if lines[i].startswith("tensor_load_to_lds"))
free = next(i for i in range(firstTdm, lo, -1) if lines[i].startswith("s_barrier_wait -1"))

out = []
for i, l in enumerate(lines):
    if i in drop:
        continue
    out.append(l)
    if i == free:
        out += ["s_cmp_eq_u32 s[sgprWaveIdx], 0                     // patched: wave 0 signals the cluster",
                "s_cbranch_scc0 .L_clusterSafeSkip",
                "s_barrier_signal -3                                // patched: cluster FREE signal",
                ".L_clusterSafeSkip:",
                "s_barrier_wait -3                                  // patched: all peers freed this buffer"]
print(f"moved cluster signal/wait to after the FREE barrier (dropped {len(drop)} lines)", file=sys.stderr)

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-csafe-{name}"
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
