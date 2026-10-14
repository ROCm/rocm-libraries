#!/usr/bin/env python3
"""Gate the main-loop A/B TDM descriptor rebuild + tensor_load_to_lds behind a wave-id
check, so only wave 0 issues A and only wave 1 issues B (mirroring the fp8 MXF8_MAF
kernel's producer/consumer wave split), instead of all 4 waves redundantly reissuing
identical tensor_load_to_lds for A and B every iteration.

Only patches the recurring MAINLOOP_C0 copy (the hot path, ~128 iterations); the
one-shot PRELOOP and TAILLOOP TDM issues are left untouched (negligible, <2% of
iterations, and TAILLOOP interacts with StreamK/tail logic we don't want to disturb).

The TDM address += / LDS-buffer-swap XOR immediately before each block are left
UNGATED and run in every wave: they feed sgprtdmLdsAddr{A,B}, which every wave needs
to compute its own local-read (ds_load) address from LDS, regardless of which wave
actually wrote the data there.

usage: patch_tdm_wavegate.py SRC_LOGDIR NAME
Writes logs/<stamp>-tdmwg-NAME/ (kernel.co + ClientParameters.ini) for cycles.sh /
run.sh-style consumption. Results are expected to be numerically correct (same bytes
land in the same LDS locations, just written by one wave instead of four).
"""
import glob, os, re, subprocess, sys, time

BENCH = os.path.dirname(os.path.abspath(__file__))
LLVM = os.environ.get("LLVM_BIN", "/opt/rocm/lib/llvm/bin")

src, name = sys.argv[1], sys.argv[2]
asm = glob.glob(f"{BENCH}/{src}/**/assembly/*.s", recursive=True)[0]
ini = glob.glob(f"{BENCH}/{src}/**/ClientParameters.ini", recursive=True)[0]
lines = open(asm).read().split("\n")

# WaveIdx = ttmp8[29:25] (bfe width=5 offset=25 -> packed imm 0x00050019), same bit
# field the hand-written fp8 kernel reads at kernel entry. ttmp8 is HW-written at
# wave launch and never touched by this kernel body, so it's valid at any point.
WAVEID_READ = "s_bfe_u32 s64, ttmp8, 0x00050019             // patched: wave id (ttmp8[29:25])"

out = []
inLoop = False
blockStartRe = {
    "A": re.compile(r"^s_mov_b32 s\[sgprtdmAGroup0\+0\], 1$"),
    "B": re.compile(r"^s_mov_b32 s\[sgprtdmBGroup0\+0\], 1$"),
}
blockEndMarker = {"A": "// TDM: global->LDS for A", "B": "// TDM: global->LDS for B"}
targetWave = {"A": 0, "B": 1}
counter = 0
changed = 0
i = 0
n = len(lines)
while i < n:
    l = lines[i]
    if re.match(r"/\* MAINLOOP_C0 start \*/", l):
        inLoop = True
    elif re.match(r"/\* MAINLOOP_C0 end \*/", l):
        inLoop = False

    matchedTensor = None
    if inLoop:
        for t, rx in blockStartRe.items():
            if rx.match(l):
                matchedTensor = t
                break

    if matchedTensor:
        t = matchedTensor
        # collect the full descriptor-rebuild block up to and including the
        # tensor_load_to_lds line that carries this tensor's marker comment.
        block = [l]
        j = i + 1
        while j < n and blockEndMarker[t] not in lines[j]:
            block.append(lines[j])
            j += 1
        block.append(lines[j])  # the tensor_load_to_lds line itself
        label = f".L_tdmwg_{t}_{counter}"
        counter += 1
        changed += 1
        out.append(WAVEID_READ)
        out.append(f"s_cmp_eq_u32 s64, {targetWave[t]}" + f"              // patched: keep TDM {t} only on wave {targetWave[t]}")
        out.append(f"s_cbranch_scc0 {label}               // patched: other waves skip it")
        out.extend(block)
        out.append(f"{label}:")
        i = j + 1
        continue

    out.append(l)
    i += 1

print(f"{changed} TDM blocks gated (expect 2: A on wave0, B on wave1)", file=sys.stderr)
assert changed == 2, "expected to gate exactly the main-loop A and B TDM blocks"

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-tdmwg-{name}"
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
