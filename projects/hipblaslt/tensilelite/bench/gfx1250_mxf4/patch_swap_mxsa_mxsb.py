#!/usr/bin/env python3
"""Diagnostic: swap the main-loop MXSA and MXSB tensor_load_to_lds issue order.

ATT shows the 4th-issued TDM load (MXSB, after A/B/MXSA) paying ~450 Idle cycles/iter
while the other three pay ~0-8. This swaps MXSA and MXSB's blocks (each: descriptor
setup + its own tensor_load_to_lds), so MXSB issues 3rd and MXSA issues 4th -- a pure
reorder, same registers, same total work -- to see whether the big stall is a queue-depth
effect (follows "whichever TDM load issues last") or specific to MXSB's identity/role.

usage: patch_swap_mxsa_mxsb.py SRC_LOGDIR NAME
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

asms = glob.glob(f"{BENCH}/{src}/kernel.s") or glob.glob(
    f"{BENCH}/{src}/**/assembly/*.s", recursive=True
)
inis = glob.glob(f"{BENCH}/{src}/**/ClientParameters.ini", recursive=True)
assert asms and inis, (asms, inis)
lines = open(asms[0]).read().splitlines()

lo = next(i for i, l in enumerate(lines) if l.startswith("label_LoopBeginL:"))
hi = next(i for i, l in enumerate(lines) if i > lo and l.startswith("label_SkipMainloop:"))
loop = lines[lo:hi]


def block(start_prefix, end_prefix, from_i=0):
    s = next(i for i in range(from_i, len(loop)) if loop[i].startswith(start_prefix))
    e = next(i for i in range(s, len(loop)) if loop[i].startswith(end_prefix)) + 1
    return s, e


mxsa_s, mxsa_e = block("s_mov_b32 s[sgprtdmMXSAGroup0", "tensor_load_to_lds s[sgprtdmMXSAGroup0")
mxsb_s, mxsb_e = block("s_mov_b32 s[sgprtdmMXSBGroup0", "tensor_load_to_lds s[sgprtdmMXSBGroup0", mxsa_e)
assert mxsa_e <= mxsb_s, (mxsa_e, mxsb_s)   # MXSA block must precede MXSB block, contiguous region between

mxsa_block = loop[mxsa_s:mxsa_e]
between = loop[mxsa_e:mxsb_s]     # whatever sits between the two blocks today (comments etc.), kept as-is
mxsb_block = loop[mxsb_s:mxsb_e]

out = loop[:mxsa_s] + ["// patched: MXSB issued before MXSA (order swap diagnostic)"] + \
      mxsb_block + between + mxsa_block + loop[mxsb_e:]

print(f"MXSA block {mxsa_e - mxsa_s} lines, MXSB block {mxsb_e - mxsb_s} lines, "
      f"{len(between)} lines kept between them -- swapped", file=sys.stderr)

lines = lines[:lo] + out + lines[hi:]
dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-swapmxsamxsb-{name}"
os.makedirs(dst)
asm = f"{dst}/kernel.s"
open(asm, "w").write("\n".join(lines) + "\n")
subprocess.run(
    [f"{LLVM}/clang", "-x", "assembler", "--target=amdgcn-amd-amdhsa",
     "-mcode-object-version=4", "-c", "-Xclangas", "-target-feature",
     "-Xclangas", "+real-true16", "-mcpu=gfx1250", "-mno-wavefrontsize64",
     asm, "-o", f"{dst}/kernel.o"],
    check=True,
)
subprocess.run(
    [f"{LLVM}/clang", "--target=amdgcn-amd-amdhsa", "-Xlinker", "--build-id=sha1",
     f"{dst}/kernel.o", "-o", f"{dst}/kernel.co"],
    check=True,
)
cfg = re.sub(r"^code-object=.*$", f"code-object={dst}/kernel.co", open(inis[0]).read(), flags=re.M)
open(f"{dst}/ClientParameters.ini", "w").write(cfg)
print(dst)
