#!/usr/bin/env python3
"""Defer the main-loop READY point (s_wait_tensorcnt + barrier) past N WMMAs (gemm_wt opt_v7 style).

usage: patch_late_ready.py SRC_LOGDIR NAME N [split=1]

In the baseline loop the READY block sits right after the TDM issue window:
    s_wait_tensorcnt k ; s_barrier_signal -1 ; s_barrier_wait -1
followed by the next iteration's K=0 LDS reads interleaved with the remaining K=3 WMMAs.
Those WMMAs only use K=3 registers (loaded before the FREE barrier), so the first N of them
are hoisted above the READY block, giving the in-flight TDM loads N more WMMAs to land.
split=1 makes it a split-phase barrier: tensorcnt wait + signal after N/2 hoisted WMMAs,
barrier wait after all N.

s_set_vgpr_msb prefixes are re-derived after reordering: every VGPR-using instruction keeps
the MSB value that was in effect for it originally.
"""
import glob, os, re, subprocess, sys, time

BENCH = os.path.dirname(os.path.abspath(__file__))
LLVM = os.environ.get("LLVM_BIN", "/opt/rocm/lib/llvm/bin")

src, name, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
opts = dict(a.split("=") for a in sys.argv[4:])
split = opts.get("split", "0") == "1"
asms = glob.glob(f"{BENCH}/{src}/kernel.s") or glob.glob(f"{BENCH}/{src}/**/assembly/*.s", recursive=True)
ini = glob.glob(f"{BENCH}/{src}/**/ClientParameters.ini", recursive=True)[0]
lines = open(asms[0]).read().split("\n")

MSB = re.compile(r"^s_set_vgpr_msb (\d+)")
def needsMsb(l):
    return re.match(r"^(v_|ds_|global_|buffer_|flat_|scratch_)", l) is not None

lo = next(i for i, l in enumerate(lines) if l.startswith("/* MAINLOOP_C0 start */"))
hi = next(i for i, l in enumerate(lines) if l.startswith("/* MAINLOOP_C0 end */"))
w = next(i for i in range(lo, hi) if lines[i].startswith("s_wait_tensorcnt"))
assert lines[w + 1].startswith("s_barrier_signal") and lines[w + 2].startswith("s_barrier_wait")

msb = None
for i in range(lo, w):
    m = MSB.match(lines[i])
    if m:
        msb = int(m.group(1))
items = []  # (line, requiredMsb or None)
for i in range(w + 3, hi):
    l = lines[i]
    m = MSB.match(l)
    if m:
        msb = int(m.group(1))
        continue
    items.append((l, msb if needsMsb(l) else None))
endMsb = msb

wmmaIdx = [k for k, (l, _) in enumerate(items) if l.startswith("v_wmma")]
assert len(wmmaIdx) >= N, f"only {len(wmmaIdx)} WMMAs after READY"
hoist = set(wmmaIdx[:N])
hoisted = [items[k] for k in sorted(hoist)]
rest = [items[k] for k in range(len(items)) if k not in hoist]

waitL, sigL, barL = lines[w], lines[w + 1], lines[w + 2]
if split:
    h = N // 2
    seq = hoisted[:h] + [(waitL, None), (sigL, None)] + hoisted[h:] + [(barL, None)] + rest
else:
    seq = hoisted + [(waitL, None), (sigL, None), (barL, None)] + rest

out, cur = [], None
for l, req in seq:
    if req is not None and req != cur:
        out.append(f"s_set_vgpr_msb {req}")
        cur = req
    out.append(l)
if endMsb is not None and cur != endMsb:
    out.append(f"s_set_vgpr_msb {endMsb}")

lines = lines[:w] + out + lines[hi:]
print(f"hoisted {N} WMMAs above READY (split={split})", file=sys.stderr)

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-lr-{name}"
os.makedirs(dst)
s = f"{dst}/kernel.s"
open(s, "w").write("\n".join(lines))
subprocess.run([f"{LLVM}/clang", "-x", "assembler", "--target=amdgcn-amd-amdhsa", "-mcode-object-version=4",
                "-c", "-Xclangas", "-target-feature", "-Xclangas", "+real-true16", "-mcpu=gfx1250",
                "-mno-wavefrontsize64", s, "-o", f"{dst}/kernel.o"], check=True, stderr=subprocess.DEVNULL)
subprocess.run([f"{LLVM}/clang", "--target=amdgcn-amd-amdhsa", "-Xlinker", "--build-id=sha1",
                f"{dst}/kernel.o", "-o", f"{dst}/kernel.co"], check=True)
cfg = re.sub(r"^code-object=.*$", f"code-object={dst}/kernel.co", open(ini).read(), flags=re.M)
open(f"{dst}/ClientParameters.ini", "w").write(cfg)
print(dst)
