#!/usr/bin/env python3
"""fp8-style producer consolidation for A/B TDM -- v2, fixing the wait-count bug
found in v1 that caused a GPU page fault / full machine hang.

v1 recap: widened the EVEN wave's (0,2) row-count clamp 64->128 and skip-branched
A/B TDM issuance for ODD waves (1,3). This is address-safe (confirmed via static
trace of the TDM descriptor fields: dim1 = full-tensor SizeI bound, tile1 = our
clamp, global/LDS strides unaffected by tile1 width) but broke a DIFFERENT
invariant: the main loop has exactly one hardcoded synchronization point,

    s_wait_tensorcnt 4   // Wait TDM (tensor_load_to_lds): A=1 B=1 SA=1 SB=1

which assumes every wave issues exactly 4 async tensor_load_to_lds ops per
iteration (A, B, MXSA, MXSB) and waits for the outstanding-TDM counter to drain
back to <=4 (prior iteration's 4 complete, this iteration's 4 stay in flight)
before the barrier. Under v1, ODD waves only issue 2 ops/iter (MXSA+MXSB, since
A/B are skipped), so their counter never approaches 4 and this wait becomes a
no-op for them -- letting odd waves race past the barrier and have the *next*
iteration's descriptor setup overwrite the SGPR-based TDM descriptor fields
before hardware finished consuming them for the *current* iteration's in-flight
MXSA/MXSB transfer. That race explained both the wrong-but-bounded results on
small sizes and the escalation to a real page fault (corrupted descriptor ->
OOB global read) on a larger-K size, which combined with "GPU recovery
disabled" on this host caused an unrecoverable hang requiring a full reboot.

v2 fix: make the wait count itself wave-conditional, matching each wave's
actual per-iteration issuance count (even=4, odd=2), using the same waveId
read/branch technique as the issuance gate.

usage: patch_tdm_consolidate_even_v2.py SRC_LOGDIR NAME
Writes logs/<stamp>-tdmconsolidatev2-NAME/.
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
waitFixed = 0
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

    # --- step 3: fix the hardcoded s_wait_tensorcnt 4 to be wave-conditional ---
    if inLoop and re.match(r"^s_wait_tensorcnt 4(\s*//.*)?$", l):
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

print(f"widened clamps: {widened}, gated TDM blocks: {gated} (expect 2: A, B), wait fixed: {waitFixed} (expect 1)", file=sys.stderr)
assert gated == 2, "expected to gate exactly the main-loop A and B TDM blocks"
assert waitFixed == 1, "expected to fix exactly the one hardcoded s_wait_tensorcnt 4"

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-tdmconsolidatev2-{name}"
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
