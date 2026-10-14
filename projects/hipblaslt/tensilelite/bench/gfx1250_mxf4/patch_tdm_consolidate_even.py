#!/usr/bin/env python3
"""fp8-style producer consolidation for A/B TDM, verified correctness-preserving (not
a timing-only drop like the earlier floor probes).

Topology (confirmed via static trace + the paired read-offsets {0,8704,17408,26112}):
  - A/B are fetched as 4 disjoint 64-row windows, one per wave (wave i's global address
    is shifted by i*64 rows; its LDS write lands at offset i*17408).
  - LDS reads are paired by waveIdM=waveId%2 (A) / waveIdN=waveId/2 (B): each pair's
    single base-offset read already spans BOTH partners' physical write slots
    (0+8704+17408+26112 bytes = exactly 2 slots = 128 rows = one M/N-band).
  - A's writer-pairs are {wave0,wave1} -> M-band-0, {wave2,wave3} -> M-band-1.
  - B's writer-pairs are {wave0,wave1} -> N-band-0, {wave2,wave3} -> N-band-1.
So wave0 and wave1 are *already* writing into the same consumed band; same for wave2/3.

This patch: for A and B, widen the EVEN wave's (0 and 2) row-count clamp from 64 to
128 rows (absorbing its odd partner's share, starting from the same global address it
already uses -- no address change needed), and skip A/B TDM issuance entirely for the
ODD waves (1 and 3). Total bytes fetched and their LDS destinations are unchanged;
only the instruction count (4 -> 2 per iteration) and per-wave issue distribution
change. Scale (MXSA/MXSB) is left untouched for this first POC.

usage: patch_tdm_consolidate_even.py SRC_LOGDIR NAME
Writes logs/<stamp>-tdmconsolidate-NAME/.
"""
import glob, os, re, subprocess, sys, time

BENCH = os.path.dirname(os.path.abspath(__file__))
LLVM = os.environ.get("LLVM_BIN", "/opt/rocm/lib/llvm/bin")

src, name = sys.argv[1], sys.argv[2]
asm = glob.glob(f"{BENCH}/{src}/**/assembly/*.s", recursive=True)[0]
ini = glob.glob(f"{BENCH}/{src}/**/ClientParameters.ini", recursive=True)[0]
lines = open(asm).read().split("\n")

WAVEID_READ = "s_bfe_u32 s64, ttmp8, 0x00050019             // patched: wave id (ttmp8[29:25])"

# --- step 1: widen the one-time clamp cap for A and B from 64 to 128 rows ---
clampCapRe = {
    "A": re.compile(r"^(s_min_u32 s\[sgprtdmClampA\], s\[sgprtdmClampA\]), 64(\s*//.*)?$"),
    "B": re.compile(r"^(s_min_u32 s\[sgprtdmClampB\], s\[sgprtdmClampB\]), 64(\s*//.*)?$"),
}
widened = {"A": 0, "B": 0}
for i, l in enumerate(lines):
    for t, rx in clampCapRe.items():
        m = rx.match(l)
        if m:
            lines[i] = f"{m.group(1)}, 128                                  // patched: widened 64->128 (even wave absorbs odd partner)"
            widened[t] += 1

assert widened["A"] == 1, f"expected to widen 1 A clamp line, got {widened['A']}"
assert widened["B"] == 1, f"expected to widen 1 B clamp line, got {widened['B']}"

# --- step 2: gate A/B TDM issuance in the main loop to even waves only ---
blockStartRe = {
    "A": re.compile(r"^s_mov_b32 s\[sgprtdmAGroup0\+0\], 1$"),
    "B": re.compile(r"^s_mov_b32 s\[sgprtdmBGroup0\+0\], 1$"),
}
blockEndMarker = {"A": "// TDM: global->LDS for A", "B": "// TDM: global->LDS for B"}

out = []
inLoop = False
counter = 0
gated = 0
i, n = 0, len(lines)
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
        block = [l]
        j = i + 1
        while j < n and blockEndMarker[t] not in lines[j]:
            block.append(lines[j])
            j += 1
        block.append(lines[j])
        label = f".L_tdmcons_{t}_{counter}"
        counter += 1
        gated += 1
        out.append(WAVEID_READ)
        out.append("s_and_b32 s64, s64, 1                          // patched: isOdd = waveId & 1")
        out.append(f"s_cmp_eq_u32 s64, 0                            // patched: keep {t} TDM only on even waves (0,2)")
        out.append(f"s_cbranch_scc0 {label}                         // patched: odd waves skip it")
        out.extend(block)
        out.append(f"{label}:")
        i = j + 1
        continue

    out.append(l)
    i += 1

print(f"widened clamps: {widened}, gated TDM blocks: {gated} (expect 2: A, B)", file=sys.stderr)
assert gated == 2, "expected to gate exactly the main-loop A and B TDM blocks"

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-tdmconsolidate-{name}"
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
