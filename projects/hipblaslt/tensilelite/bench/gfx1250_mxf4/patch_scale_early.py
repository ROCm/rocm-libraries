#!/usr/bin/env python3
"""Move selected main-loop scale LDS reads to the start of the preceding K phase.

The generated schedule puts the 12 scale reads for K=n at the end of phase n-1,
immediately before s_wait_dscnt. Their destination scale VGPR set is not used by
phase n-1, so issuing them immediately after phase n-1's opening dscnt wait gives
the intervening WMMAs and data LDS reads time to hide their latency.

K0 is special: its source is the next TDM buffer, so it moves only to the
s_wait_alu immediately after that buffer's tensor wait and workgroup barrier.

usage: patch_scale_early.py SRC_LOGDIR NAME [ks=1,2]
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
ks = {int(x) for x in opts.get("ks", "1,2").split(",")}
delay = int(opts.get("delay", "0"))
assert ks <= {0, 1, 2, 3} and ks
assert delay >= 0

asms = glob.glob(f"{BENCH}/{src}/kernel.s") or glob.glob(
    f"{BENCH}/{src}/**/assembly/*.s", recursive=True
)
inis = glob.glob(f"{BENCH}/{src}/**/ClientParameters.ini", recursive=True)
lines = open(asms[0]).read().splitlines()

scale_re = re.compile(
    r"^ds_load_b32 v\d+, v(?:33|36) offset:\d+\s+"
    r"// scaleMXS[AB]\[group\d+,K=(\d+)\]"
)

start = next(i for i, line in enumerate(lines) if line == "/* MAINLOOP_C0 start */")
end = next(i for i, line in enumerate(lines[start:], start) if line == "/* MAINLOOP_C0 end */")

loads = {k: [] for k in ks}
drop = set()
for i in range(start, end):
    match = scale_re.match(lines[i])
    if match and int(match.group(1)) in ks:
        k = int(match.group(1))
        loads[k].append(lines[i])
        drop.add(i)

assert all(len(loads[k]) == 12 for k in ks), {k: len(v) for k, v in loads.items()}

# The opening wait of phase k-1 is the earliest point after the prior use of
# this ping-pong scale VGPR set. K1 follows the first wait, K2 the second, etc.
waits = [
    i for i in range(start, end)
    if lines[i].startswith("s_wait_dscnt 0") and "Wait for LR to complete" in lines[i]
]
assert len(waits) == 5, waits
insert_at = {}
for k in ks:
    if not k:
        continue
    target = waits[k - 1]
    if delay:
        seen = 0
        for i in range(target + 1, end):
            if lines[i].startswith("v_wmma_scale_f32_32x16x128_f4"):
                seen += 1
                if seen == delay:
                    target = i
                    break
        assert seen == delay, (k, seen, delay)
    insert_at[target] = k
if 0 in ks:
    ready = next(
        i for i in range(start, end)
        if lines[i].startswith("s_wait_alu depctr_va_vdst(0)")
        and lines[i - 1].startswith("s_barrier_wait -1")
        and lines[i - 2].startswith("s_barrier_signal -1")
        and lines[i - 3].startswith("s_wait_tensorcnt 4")
    )
    insert_at[ready] = 0

out = []
for i, line in enumerate(lines):
    if i in drop:
        continue
    out.append(line)
    if i in insert_at:
        k = insert_at[i]
        out += [
            f"// patched: issue K={k} scale reads at start of preceding phase",
            "s_set_vgpr_msb 0",
            *loads[k],
        ]

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-scaleearly-{name}"
os.makedirs(dst)
asm = f"{dst}/kernel.s"
open(asm, "w").write("\n".join(out) + "\n")
subprocess.run(
    [
        f"{LLVM}/clang", "-x", "assembler", "--target=amdgcn-amd-amdhsa",
        "-mcode-object-version=4", "-c", "-Xclangas", "-target-feature",
        "-Xclangas", "+real-true16", "-mcpu=gfx1250", "-mno-wavefrontsize64",
        asm, "-o", f"{dst}/kernel.o",
    ],
    check=True,
)
subprocess.run(
    [
        f"{LLVM}/clang", "--target=amdgcn-amd-amdhsa", "-Xlinker",
        "--build-id=sha1", f"{dst}/kernel.o", "-o", f"{dst}/kernel.co",
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
