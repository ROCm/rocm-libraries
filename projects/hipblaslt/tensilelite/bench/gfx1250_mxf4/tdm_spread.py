#!/usr/bin/env python3
"""Move main-loop TDM issue blocks to later points in the iteration and rebuild the code object.

usage: tdm_spread.py SRC_LOGDIR NAME A:N B:N MXSA:N MXSB:N

Each TENSOR:N moves that tensor's block (address update, LDS swap, descriptor setup and
tensor_load_to_lds) to right after the N-th v_wmma of the main loop (1-based). Blocks with
N <= the WMMA count at the first barrier stay where they are. A load issued after the
s_wait_tensorcnt still belongs to the buffer being filled for the next iteration, so the
wait count becomes the number of loads still issued before it. Results stay correct.
Writes logs/<stamp>-spread-NAME/ (kernel.s, kernel.co, ClientParameters.ini).
"""
import glob, os, re, subprocess, sys, time

BENCH = os.path.dirname(os.path.abspath(__file__))
LLVM = "/opt/rocm-10.1.0a20260908+bkc.20260917/lib/llvm/bin"

src, name = sys.argv[1], sys.argv[2]
want = {k: int(v) for k, v in (a.split(":") for a in sys.argv[3:])}
asm = glob.glob(f"{BENCH}/{src}/**/assembly/*.s", recursive=True)[0]
ini = glob.glob(f"{BENCH}/{src}/**/ClientParameters.ini", recursive=True)[0]
lines = open(asm).read().split("\n")

lo = next(i for i, l in enumerate(lines) if l.startswith("label_LoopBeginL:"))
hi = next(i for i, l in enumerate(lines) if i > lo and l.startswith("label_SkipMainloop:"))
loop = lines[lo:hi]

blocks = {}
for t in ("A", "B", "MXSA", "MXSB"):
    s = next(i for i, l in enumerate(loop) if l.startswith(f"/* TDM addr update: {t} +="))
    e = next(i for i in range(s, len(loop)) if loop[i].startswith(f"tensor_load_to_lds s[sgprtdm{t}Group0")) + 1
    blocks[t] = (s, e)

wmmaAt = []
for i, l in enumerate(loop):
    if l.startswith("v_wmma"):
        wmmaAt.append(i)
first = min(s for s, _ in blocks.values())
wmmaAtTdm = sum(1 for i in wmmaAt if i < first)
moved = {t: n for t, n in want.items() if n > wmmaAtTdm}

out, skip = [], set()
for t in moved:
    skip.update(range(*blocks[t]))
insertAfter = {}
for t, n in moved.items():
    insertAfter.setdefault(wmmaAt[n - 1], []).append(t)
waitIdx = next(i for i, l in enumerate(loop) if l.startswith("s_wait_tensorcnt"))
waitPos = sum(1 for i in wmmaAt if i < waitIdx)
before = sum(1 for t in blocks if want.get(t, 0) <= waitPos)
for i, l in enumerate(loop):
    if i in skip:
        continue
    if i == waitIdx:
        l = re.sub(r"s_wait_tensorcnt \d+", f"s_wait_tensorcnt {before}", l)
    out.append(l)
    for t in insertAfter.get(i, []):
        out += loop[blocks[t][0]:blocks[t][1]]
print(f"moved {moved}, wait count {before}", file=sys.stderr)
lines = lines[:lo] + out + lines[hi:]

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-spread-{name}"
os.makedirs(f"{dst}/assembly")
s = f"{dst}/assembly/kernel.s"
open(s, "w").write("\n".join(lines))
open(f"{dst}/kernel.s", "w").write("\n".join(lines))
subprocess.run([f"{LLVM}/clang", "-x", "assembler", "--target=amdgcn-amd-amdhsa",
                f"-mcode-object-version={os.environ.get('CO_VERSION', '4')}", "-c",
                "-Xclangas", "-target-feature", "-Xclangas", "+real-true16",
                "-mcpu=gfx1250", "-mno-wavefrontsize64", s, "-o", f"{dst}/kernel.o"], check=True,
               stderr=subprocess.DEVNULL)
subprocess.run([f"{LLVM}/clang", "--target=amdgcn-amd-amdhsa", "-Xlinker", "--build-id=sha1",
                f"{dst}/kernel.o", "-o", f"{dst}/kernel.co"], check=True)
cfg = re.sub(r"^code-object=.*$", f"code-object={dst}/kernel.co", open(ini).read(), flags=re.M)
open(f"{dst}/ClientParameters.ini", "w").write(cfg)
print(dst)
