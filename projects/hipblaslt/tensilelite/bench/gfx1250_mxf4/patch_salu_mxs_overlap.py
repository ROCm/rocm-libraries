#!/usr/bin/env python3
"""Overlap post-FREE MXSA/MXSB descriptor SALU with a pending K=2 WMMA.

After A/B tensor_load the kernel emits ~30 SALU of MXSA/MXSB Group0/1 rebuild
+ iterate issue, then a leftover K=2 WMMA (C[3,0]) that only needs already-
buffered VGPRs. Hoist that WMMA to immediately after A/B loads so the WMMA
pipe stays busy through the scale-descriptor SALU.

Leaves the wave0 cluster-barrier signal in place (after scale issues) so SCC
and sync timing stay as before.

usage: patch_salu_mxs_overlap.py SRC_LOGDIR NAME
"""
import glob, os, re, subprocess, sys, time

BENCH = os.path.dirname(os.path.abspath(__file__))
LLVM = os.environ.get("LLVM_BIN", "/opt/rocm/lib/llvm/bin")

src, name = sys.argv[1], sys.argv[2]
src = src if os.path.isabs(src) else f"{BENCH}/{src}"
asms = glob.glob(f"{src}/**/assembly/*.s", recursive=True) + glob.glob(f"{src}/kernel.s")
inis = glob.glob(f"{src}/**/ClientParameters.ini", recursive=True) + glob.glob(f"{src}/ClientParameters.ini")
asm, ini = asms[0], inis[0]
lines = open(asm).read().split("\n")

# Find MAINLOOP post-FREE A/B loads followed by MXSA rebuild, and the K=2 WMMA
# that currently sits after .L_sciter_MXSB_*.
main_start = next(i for i, l in enumerate(lines) if "MAINLOOP_C0 start" in l)
main_end = next(i for i, l in enumerate(lines) if "MAINLOOP_C0 end" in l)

# A/B load pair after FREE barrier in MAINLOOP
ab_b = None
for i in range(main_start, main_end):
    if (lines[i].startswith("tensor_load_to_lds") and "for A" in lines[i] and
            i + 1 < main_end and lines[i + 1].startswith("tensor_load_to_lds") and
            "for B" in lines[i + 1] and
            i + 2 < main_end and lines[i + 2].startswith("s_mov_b32 s[sgprtdmMXSAGroup0+0], 1")):
        ab_b = i + 1  # last of A/B loads
        break
assert ab_b is not None, "MAINLOOP A/B loads + MXSA rebuild"

# K=2 leftover WMMA: first v_wmma after .L_sciter_MXSB_ in MAINLOOP that
# comments mention K=2
mxsb_label = next(i for i in range(ab_b, main_end)
                  if lines[i].startswith(".L_sciter_MXSB_"))
wmma_i = next(i for i in range(mxsb_label, main_end)
              if lines[i].startswith("v_wmma_scale_f32_32x16x128_f4") and "K=2" in lines[i])
# Include preceding s_set_vgpr_msb if present
msb_i = wmma_i - 1 if lines[wmma_i - 1].startswith("s_set_vgpr_msb") else wmma_i

# Do not move the s_cmp / cbranch / barrier_signal that may wrap this WMMA;
# only the msb+wmma pair. If s_cmp is immediately above msb, leave it.
assert "K=2" in lines[wmma_i]

hoist = lines[msb_i:wmma_i + 1]
hoist = [hoist[0] + " // patched: hoist K=2 WMMA over MXS descriptor SALU"
         if i == 0 else h for i, h in enumerate(hoist)]

out = []
i = 0
n = len(lines)
hoisted = False
skipped = False
while i < n:
    if i == ab_b and not hoisted:
        out.append(lines[i])
        out.append("// patched: K=2 WMMA overlapping MXSA/MXSB descriptor setup")
        out.extend(hoist)
        hoisted = True
        i += 1
        continue
    if msb_i <= i <= wmma_i and not skipped:
        # drop original location
        if i == wmma_i:
            skipped = True
        i += 1
        continue
    out.append(lines[i])
    i += 1

assert hoisted and skipped

text = "\n".join(out)
if not text.endswith("\n"):
    text += "\n"

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-salumxs-{name}"
os.makedirs(dst)
s = f"{dst}/kernel.s"
open(s, "w").write(text)
subprocess.run([f"{LLVM}/clang", "-x", "assembler", "--target=amdgcn-amd-amdhsa",
                "-mcode-object-version=4", "-c", "-Xclangas", "-target-feature",
                "-Xclangas", "+real-true16", "-mcpu=gfx1250", "-mno-wavefrontsize64",
                s, "-o", f"{dst}/kernel.o"], check=True)
subprocess.run([f"{LLVM}/clang", "--target=amdgcn-amd-amdhsa", "-Xlinker", "--build-id=sha1",
                f"{dst}/kernel.o", "-o", f"{dst}/kernel.co"], check=True)
cfg = re.sub(r"^code-object=.*$", f"code-object={dst}/kernel.co", open(ini).read(), flags=re.M)
open(f"{dst}/ClientParameters.ini", "w").write(cfg)
print(f"hoisted msb+wmma from {msb_i} after AB@{ab_b}", file=sys.stderr)
print(dst)
