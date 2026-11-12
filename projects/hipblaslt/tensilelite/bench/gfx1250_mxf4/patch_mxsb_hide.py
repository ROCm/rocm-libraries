#!/usr/bin/env python3
"""Hide main-loop MXSB TDM issue under WMMA work after A/B/MXSA issue.

ATT shows MXSB (4th TDM after the GR barrier) paying ~300 Idle cycles/iter.
This patch cuts the last N WMMAs from the K-phase above the TDM addr-update
block (safe once wait_dscnt+barrier has drained LDS) and replays them between
MXSA and MXSB issue so the TDM pipe can drain before MXSB is attempted.

usage: patch_mxsb_hide.py SRC_LOGDIR NAME [nwmma=12]
"""
import glob
import os
import re
import subprocess
import sys
import time

BENCH = os.path.dirname(os.path.abspath(__file__))
LLVM = os.environ.get("LLVM_BIN", "/opt/rocm/lib/llvm/bin")

src, name = sys.argv[1], sys.argv[2]
opts = dict(a.split("=", 1) for a in sys.argv[3:])
nwmma = int(opts.get("nwmma", "12"))
assert nwmma >= 1

asms = glob.glob(f"{BENCH}/{src}/kernel.s") or glob.glob(
    f"{BENCH}/{src}/**/assembly/*.s", recursive=True
)
inis = glob.glob(f"{BENCH}/{src}/**/ClientParameters.ini", recursive=True)
assert asms and inis, (asms, inis)
lines = open(asms[0]).read().splitlines()

lo = next(i for i, l in enumerate(lines) if l.startswith("label_LoopBeginL:"))
hi = next(
    i for i, l in enumerate(lines) if i > lo and l.startswith("label_SkipMainloop:")
)
loop = lines[lo:hi]

tdm_a = next(i for i, l in enumerate(loop) if l.startswith("/* TDM addr update: A +="))
# Collect (msb?, wmma) groups walking upward; stop at wait_dscnt / barrier / ds.
groups = []  # list of lists of line indices, each ending in a v_wmma
i = tdm_a - 1
while i >= 0 and len(groups) < nwmma:
    if loop[i].startswith("v_wmma"):
        grp = [i]
        j = i - 1
        if j >= 0 and loop[j].startswith("s_set_vgpr_msb"):
            grp.insert(0, j)
            j -= 1
        groups.append(grp)
        i = j
        continue
    if loop[i].startswith("s_set_vgpr_msb"):
        # orphan msb belonging to a later moved wmma already consumed
        i -= 1
        continue
    if loop[i].strip() == "" or loop[i].startswith("/*"):
        i -= 1
        continue
    # ds_load / wait / other: skip over ds_loads but do not take them
    if loop[i].startswith("ds_load"):
        i -= 1
        continue
    break
assert len(groups) == nwmma, (len(groups), nwmma)
# groups were collected bottom-up; restore program order
groups.reverse()
wmma_idxs = [k for g in groups for k in g]
wmma_tail = [loop[k] for k in wmma_idxs]

mxsa_load = next(
    i
    for i, l in enumerate(loop)
    if l.startswith("tensor_load_to_lds s[sgprtdmMXSAGroup0")
)
mxsb_start = next(
    i
    for i in range(mxsa_load + 1, len(loop))
    if loop[i].startswith("s_mov_b32 s[sgprtdmMXSBGroup0")
)
mxsb_end = (
    next(
        i
        for i in range(mxsb_start, len(loop))
        if loop[i].startswith("tensor_load_to_lds s[sgprtdmMXSBGroup0")
    )
    + 1
)
mxsb_block = loop[mxsb_start:mxsb_end]

drop = set(wmma_idxs) | set(range(mxsb_start, mxsb_end))
out = []
for i, l in enumerate(loop):
    if i in drop:
        continue
    out.append(l)
    if i == mxsa_load:
        out += [
            f"// patched: hide MXSB behind {nwmma} WMMAs after MXSA",
            *wmma_tail,
            *mxsb_block,
        ]

print(
    f"moved {nwmma} WMMAs ({len(wmma_idxs)} lines) + MXSB "
    f"({mxsb_end - mxsb_start} lines) after MXSA",
    file=sys.stderr,
)

lines = lines[:lo] + out + lines[hi:]
dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-mxsbhide-{name}"
os.makedirs(dst)
asm = f"{dst}/kernel.s"
open(asm, "w").write("\n".join(lines) + "\n")
subprocess.run(
    [
        f"{LLVM}/clang",
        "-x",
        "assembler",
        "--target=amdgcn-amd-amdhsa",
        "-mcode-object-version=4",
        "-c",
        "-Xclangas",
        "-target-feature",
        "-Xclangas",
        "+real-true16",
        "-mcpu=gfx1250",
        "-mno-wavefrontsize64",
        asm,
        "-o",
        f"{dst}/kernel.o",
    ],
    check=True,
)
subprocess.run(
    [
        f"{LLVM}/clang",
        "--target=amdgcn-amd-amdhsa",
        "-Xlinker",
        "--build-id=sha1",
        f"{dst}/kernel.o",
        "-o",
        f"{dst}/kernel.co",
    ],
    check=True,
)
cfg = re.sub(
    r"^code-object=.*$",
    f"code-object={dst}/kernel.co",
    open(inis[0]).read(),
    flags=re.M,
)
open(f"{dst}/ClientParameters.ini", "w").write(cfg)
print(dst)
