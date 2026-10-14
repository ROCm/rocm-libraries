#!/usr/bin/env python3
"""Insert s_prefetch_inst_pc_rel for the main loop body at kernel entry (gemm_wt opt_v7 style).

usage: patch_iprefetch.py SRC_LOGDIR NAME [CHUNKS]
CHUNKS 4 KiB prefetches (default: enough to cover label_LoopBeginL .. the loop back-branch)
are issued as the first instructions of the kernel. The offset operand must be an immediate,
so the kernel is assembled once with placeholders, the real PC-relative offsets are read back
from the disassembly (target = prefetch instruction address + offset), and it is reassembled.
SRC_LOGDIR may be a Tensile logdir or a patched logdir with kernel.s.
"""
import glob, os, re, subprocess, sys, time

BENCH = os.path.dirname(os.path.abspath(__file__))
LLVM = os.environ.get("LLVM_BIN", "/opt/rocm/lib/llvm/bin")
KLEN = 31  # (31 + 1) * 128 B = 4 KiB, the maximum

src, name = sys.argv[1], sys.argv[2]
chunks = int(sys.argv[3]) if len(sys.argv) > 3 else None
asms = glob.glob(f"{BENCH}/{src}/kernel.s") or glob.glob(f"{BENCH}/{src}/**/assembly/*.s", recursive=True)
ini = glob.glob(f"{BENCH}/{src}/**/ClientParameters.ini", recursive=True)[0]
lines = open(asms[0]).read().split("\n")

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-ipf-{name}"
os.makedirs(dst)

def build(text):
    s = f"{dst}/kernel.s"
    open(s, "w").write("\n".join(text))
    subprocess.run([f"{LLVM}/clang", "-x", "assembler", "--target=amdgcn-amd-amdhsa", "-mcode-object-version=4",
                    "-c", "-Xclangas", "-target-feature", "-Xclangas", "+real-true16", "-mcpu=gfx1250",
                    "-mno-wavefrontsize64", s, "-o", f"{dst}/kernel.o"], check=True, stderr=subprocess.DEVNULL)
    return subprocess.run([f"{LLVM}/llvm-objdump", "-d", "--mcpu=gfx1250", f"{dst}/kernel.o"],
                          capture_output=True, text=True, check=True).stdout

inMacro, entry = False, None
start = next(i for i, l in enumerate(lines) if l.startswith("label_ASM_Start:"))
for i in range(start, len(lines)):
    l = lines[i].strip()
    if l.startswith(".macro"):
        inMacro = True
    elif l.startswith(".endm"):
        inMacro = False
    elif not inMacro and re.match(r"^[sv]_\w+", l):
        entry = i
        break

def withPrefetch(offsets):
    pf = [f"s_prefetch_inst_pc_rel 0x{o:x}, null, {KLEN}  // patched: prefetch main loop chunk {k}"
          for k, o in enumerate(offsets)]
    return lines[:entry] + pf + lines[entry:]

dis = build(withPrefetch([0] * (chunks or 2)))
loopAt = int(re.search(r"^([0-9a-f]+) <label_LoopBeginL>:", dis, re.M).group(1), 16)
backAt = int(re.findall(r"s_cbranch_scc\d label_LoopBeginL\s+// ([0-9A-F]+):", dis)[-1], 16)
pfAt = [int(a, 16) for a in re.findall(r"s_prefetch_inst_pc_rel.*// ([0-9A-F]+):", dis)]
need = (backAt + 4 - loopAt + 4095) // 4096
n = chunks or need
if len(pfAt) != n:
    dis = build(withPrefetch([0] * n))
    pfAt = [int(a, 16) for a in re.findall(r"s_prefetch_inst_pc_rel.*// ([0-9A-F]+):", dis)]
    loopAt = int(re.search(r"^([0-9a-f]+) <label_LoopBeginL>:", dis, re.M).group(1), 16)
offsets = [loopAt + 4096 * k - pfAt[k] for k in range(n)]
dis = build(withPrefetch(offsets))
got = [int(a, 16) for a in re.findall(r"s_prefetch_inst_pc_rel.*// ([0-9A-F]+):", dis)]
assert got == pfAt, "prefetch addresses moved after patching offsets"
print(f"loop 0x{loopAt:x}..0x{backAt:x} ({backAt + 4 - loopAt} B), {n} chunks, offsets {[hex(o) for o in offsets]}",
      file=sys.stderr)

subprocess.run([f"{LLVM}/clang", "--target=amdgcn-amd-amdhsa", "-Xlinker", "--build-id=sha1",
                f"{dst}/kernel.o", "-o", f"{dst}/kernel.co"], check=True)
cfg = re.sub(r"^code-object=.*$", f"code-object={dst}/kernel.co", open(ini).read(), flags=re.M)
open(f"{dst}/ClientParameters.ini", "w").write(cfg)
print(dst)
