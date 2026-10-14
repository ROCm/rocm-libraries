#!/usr/bin/env python3
"""Timing-only floor probe (same convention as patch_tdm.py's drop= variants / the
earlier patch_drop_msb.py -- NOT expected to produce correct results): strip every
scale-related data-movement instruction from the recurring MAINLOOP_C0 region, while
leaving v_wmma_scale_f32_32x16x128_f4 itself untouched (it's the matmul instruction;
there's no way to "exclude the scale part" of it without swapping to a different,
non-scaled WMMA opcode entirely, which is a much bigger change than a timing probe).

Removes, per iteration:
  - the MXSA and MXSB tensor_load_to_lds descriptor-rebuild + issue blocks (the two
    extra TDM calls that only exist because of MX scaling)
  - all ds_load_b32 scaleMXSA[...]/scaleMXSB[...] LDS reads (48/iter)
  - the TDM addr-update / LDS-buffer-swap lines that only feed those two blocks

This isolates "what if moving the scale factors around (global->LDS, LDS->VGPR) were
completely free" -- i.e. the floor for everything in the current scale path *except*
the per-wmma scale-multiply itself, which is inseparable from v_wmma_scale.

usage: patch_drop_scale.py SRC_LOGDIR NAME
Writes logs/<stamp>-dropscale-NAME/.
"""
import glob, os, re, subprocess, sys, time

BENCH = os.path.dirname(os.path.abspath(__file__))
LLVM = os.environ.get("LLVM_BIN", "/opt/rocm/lib/llvm/bin")

src, name = sys.argv[1], sys.argv[2]
asm = glob.glob(f"{BENCH}/{src}/**/assembly/*.s", recursive=True)[0]
ini = glob.glob(f"{BENCH}/{src}/**/ClientParameters.ini", recursive=True)[0]
lines = open(asm).read().split("\n")

scaleLoadRe = re.compile(r"^ds_load_b32 .*scaleMXS[AB]\[")
tdmBlockStartRe = {
    "MXSA": re.compile(r"^s_mov_b32 s\[sgprtdmMXSAGroup0\+0\], 1$"),
    "MXSB": re.compile(r"^s_mov_b32 s\[sgprtdmMXSBGroup0\+0\], 1$"),
}
tdmBlockEndMarker = {"MXSA": "// TDM: global->LDS for MXSA", "MXSB": "// TDM: global->LDS for MXSB"}
# the addr-update / LDS-swap comment pairs that immediately precede each TDM block
addrUpdateRe = re.compile(r"/\* TDM addr update: MXS[AB] ")

out = []
inLoop = False
dropped_loads = 0
dropped_tdm_blocks = 0
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
            # this comment + the two lines that follow it (addr add, then the
            # "TDM: swap ... LDS buffer" comment + xor) all only exist for MXSA/MXSB
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
            dropped_tdm_blocks += 1
            i = j + 1  # drop lines[i..j] inclusive (descriptor rebuild + the TDM issue)
            continue

    out.append(l)
    i += 1

print(f"dropped {dropped_loads} scale ds_load_b32, {dropped_tdm_blocks} scale TDM blocks, "
      f"{dropped_misc} addr-update/swap comment lines from MAINLOOP_C0", file=sys.stderr)
assert dropped_loads == 48, f"expected 48 ds_load_b32 scale reads, got {dropped_loads}"
assert dropped_tdm_blocks == 2, f"expected 2 scale TDM blocks (MXSA, MXSB), got {dropped_tdm_blocks}"

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-dropscale-{name}"
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
