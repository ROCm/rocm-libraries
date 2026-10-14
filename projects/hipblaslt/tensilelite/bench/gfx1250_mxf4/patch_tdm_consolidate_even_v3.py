#!/usr/bin/env python3
"""fp8-style producer consolidation for A/B TDM -- v3.

v1: widened even wave's (0,2) row clamp 64->128, skip-branched A/B TDM issuance
for odd waves (1,3) -- but ONLY inside the main loop, and left the hardcoded
`s_wait_tensorcnt 4` unconditional. That caused a GPU page fault / full machine
hang (see patch_tdm_consolidate_even.py docstring for the postmortem).

v2: fixed the wait by making it wave-conditional (even=4, odd=2) -- but a
second static-trace pass (done BEFORE touching the GPU this time) found that
the clamp widening (step 1) is a single global one-time register consumed by
FOUR separate A/B issuance call sites, not just the main loop's:
  - two in the preloop prefetch fill (PGR2 double-buffer warm-up, each runs
    once, guarded by `label_SkipPreloopGR`)
  - the main loop steady-state copy (inside MAINLOOP_C0, runs once/iteration)
  - one in the tail/epilogue fill (runs once)
v2 only gated the main-loop copy, so in the other three sites odd waves would
still issue the now-widened (128-row) descriptor unconditionally, overlapping
with the even partner's territory. Confirmed via grep that all `s_wait_tensorcnt`
sites besides the single main-loop one use value 0 (full drain -- inherently
safe regardless of per-wave issue count), so only the issuance gate needs to be
extended to all four sites; the wait fix stays scoped to the one partial-drain
site.

v3: gate ALL FOUR A/B issuance sites (not just the one inside MAINLOOP_C0) to
even waves only, keep the single wave-conditional wait-count fix from v2.

usage: patch_tdm_consolidate_even_v3.py SRC_LOGDIR NAME
Writes logs/<stamp>-tdmconsolidatev3-NAME/.
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

# --- step 2: gate EVERY A/B TDM issuance site (preloop x2, main loop x1, tail x1)
#     to even waves only -- not just the main loop's copy (v2's bug). ---
blockStartRe = {
    "A": re.compile(r"^s_mov_b32 s\[sgprtdmAGroup0\+0\], 1$"),
    "B": re.compile(r"^s_mov_b32 s\[sgprtdmBGroup0\+0\], 1$"),
}
blockEndMarker = {"A": "// TDM: global->LDS for A", "B": "// TDM: global->LDS for B"}

out = []
counter = 0
gated = 0
waitFixed = 0
i, n = 0, len(lines)
while i < n:
    l = lines[i]

    matchedTensor = None
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

    # --- step 3: fix the hardcoded s_wait_tensorcnt 4 to be wave-conditional ---
    if re.match(r"^s_wait_tensorcnt 4(\s*//.*)?$", l):
        label_odd = ".L_tdmcons_wait_odd"
        label_end = ".L_tdmcons_wait_end"
        out.append(WAVEID_READ)
        out.append("s_and_b32 s64, s64, 1                          // patched: isOdd = waveId & 1")
        out.append("s_cmp_eq_u32 s64, 0                            // patched: even waves issued 4 TDM ops this iter (A,B,SA,SB)")
        out.append(f"s_cbranch_scc0 {label_odd}                     // patched: odd waves issued only 2 (SA,SB)")
        out.append("s_wait_tensorcnt 4                              // patched: even wave wait (was unconditional)")
        out.append(f"s_branch {label_end}")
        out.append(f"{label_odd}:")
        out.append("s_wait_tensorcnt 2                              // patched: odd wave wait (A/B TDM skipped this round)")
        out.append(f"{label_end}:")
        waitFixed += 1
        i += 1
        continue

    out.append(l)
    i += 1

print(f"widened clamps: {widened}, gated TDM blocks: {gated} (expect 8: A,B x {{preloop1,preloop2,mainloop,tail}}), wait fixed: {waitFixed} (expect 1)", file=sys.stderr)
assert gated == 8, f"expected to gate all 4 A/B issuance sites (8 blocks total), got {gated}"
assert waitFixed == 1, "expected to fix exactly the one hardcoded s_wait_tensorcnt 4"

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-tdmconsolidatev3-{name}"
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
