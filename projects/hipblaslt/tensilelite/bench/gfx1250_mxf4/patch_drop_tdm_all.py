#!/usr/bin/env python3
"""Timing-only floor probe: strip ALL TDM data movement (A, B, MXSA, MXSB -- not just
scale) from the recurring MAINLOOP_C0 region. Keeps ds_load_b128/b32 (LDS->VGPR reads,
now reading stale/garbage data) and v_wmma_scale (the matmul itself) completely
unchanged. This isolates the floor for "global->LDS transfer were entirely free",
i.e. everything left over is LDS-read + WMMA-execute + loop/barrier/SALU overhead.

Same convention as patch_drop_scale.py / patch_drop_msb.py: NOT expected to be
correct, cycles-only comparison via cycles.sh (not validate.py).

usage: patch_drop_tdm_all.py SRC_LOGDIR NAME
Writes logs/<stamp>-droptdmall-NAME/.
"""
import glob, os, re, subprocess, sys, time

BENCH = os.path.dirname(os.path.abspath(__file__))
LLVM = os.environ.get("LLVM_BIN", "/opt/rocm/lib/llvm/bin")

src, name = sys.argv[1], sys.argv[2]
asm = glob.glob(f"{BENCH}/{src}/**/assembly/*.s", recursive=True)[0]
ini = glob.glob(f"{BENCH}/{src}/**/ClientParameters.ini", recursive=True)[0]
lines = open(asm).read().split("\n")

scaleLoadRe = re.compile(r"^ds_load_b32 .*scaleMXS[AB]\[")
tensors = ["A", "B", "MXSA", "MXSB"]
tdmBlockStartRe = {t: re.compile(rf"^s_mov_b32 s\[sgprtdm{t}Group0\+0\], 1$") for t in tensors}
tdmBlockEndMarker = {t: f"// TDM: global->LDS for {t}" for t in tensors}
addrUpdateRe = re.compile(r"/\* TDM addr update: (A|B|MXSA|MXSB) ")

out = []
inLoop = False
dropped_loads = 0
dropped_tdm_blocks = {t: 0 for t in tensors}
dropped_misc = 0
i, n = 0, len(lines)
while i < n:
    l = lines[i]
    if re.match(r"/\* MAINLOOP_C0 start \*/", l):
        inLoop = True
    elif re.match(r"/\* MAINLOOP_C0 end \*/", l):
        inLoop = False

    if inLoop:
        if scaleLoadRe.match(l):
            dropped_loads += 1
            i += 1
            continue
        if addrUpdateRe.match(l):
            dropped_misc += 1
            i += 1
            continue
        matched = None
        for t, rx in tdmBlockStartRe.items():
            if rx.match(l):
                matched = t
                break
        if matched:
            j = i
            while tdmBlockEndMarker[matched] not in lines[j]:
                j += 1
            dropped_tdm_blocks[matched] += 1
            i = j + 1
            continue

    out.append(l)
    i += 1

print(f"dropped {dropped_loads} scale ds_load_b32, tdm blocks: {dropped_tdm_blocks}, "
      f"{dropped_misc} addr-update/swap comment lines from MAINLOOP_C0", file=sys.stderr)
assert dropped_loads == 48
for t in tensors:
    assert dropped_tdm_blocks[t] == 1, f"expected 1 {t} TDM block per loop copy, got {dropped_tdm_blocks[t]}"

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-droptdmall-{name}"
os.makedirs(dst)
s = f"{dst}/kernel.s"
open(s, "w").write("\n".join(out))
subprocess.run([f"{LLVM}/clang", "-x", "assembler", "--target=amdgcn-amd-amdhsa", "-mcode-object-version=4",
                "-c", "-Xclangas", "-target-feature", "-Xclangas", "+real-true16", "-mcpu=gfx1250",
                "-mno-wavefrontsize64", s, "-o", f"{dst}/kernel.o"], check=True)
subprocess.run([f"{LLVM}/clang", "--target=amdgcn-amd-amdhsa", "-Xlinker", "--build-id=sha1",
                f"{dst}/kernel.o", "-o", f"{dst}/kernel.co"], check=True)
cfg = re.sub(r"^code-object=.*$", f"code-object={dst}/kernel.co", open(ini).read(), flags=re.M)
open(f"{dst}/ClientParameters.ini", "w").write(cfg)
print(dst)
