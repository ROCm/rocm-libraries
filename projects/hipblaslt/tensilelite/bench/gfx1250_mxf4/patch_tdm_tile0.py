#!/usr/bin/env python3
"""Disambiguation experiment: force tile1 (the TDM row-count field) to 0 for the
non-owning waves' A/B load in the main loop, WITHOUT touching whether/how many
tensor_load_to_lds instructions get issued, so s_wait_tensorcnt bookkeeping stays
byte-for-byte identical to baseline (every wave still issues exactly 4 TDM calls per
iteration -- A, B, MXSA, MXSB -- same as unpatched).

This isolates one specific question from the (failed) full-skip wavegate experiment:
was that failure caused by (a) wait-count/tensorcnt accounting corruption (fewer
instructions issued per non-owning wave), or (b) each wave's A/B tile1 window genuinely
covering DISTINCT rows (cooperative 4-way partition via waveGlobalRowStart = wId*64),
so zeroing it for 3-of-4 waves starves 3/4 of the compute of real data either way.

If this variant PASSES validation: the wait-count bug (a) was the sole cause of the
earlier failure, and the "redundant tile1 data" premise holds -- zeroing non-owners'
transfer size is harmless, meaning real savings may be achievable with a wait-count-
aware full-skip patch.

If this variant FAILS: each wave's tile1 covers genuinely distinct data (cooperative
partition, hypothesis (b)), meaning the earlier "4x redundant TDM issuance" framing is
WRONG -- the per-wave issuance is necessary cooperative work, not redundant duplication.

usage: patch_tdm_tile0.py SRC_LOGDIR NAME
Writes logs/<stamp>-tile0-NAME/ (kernel.co + ClientParameters.ini).
"""
import glob, os, re, subprocess, sys, time

BENCH = os.path.dirname(os.path.abspath(__file__))
LLVM = os.environ.get("LLVM_BIN", "/opt/rocm/lib/llvm/bin")

src, name = sys.argv[1], sys.argv[2]
asm = glob.glob(f"{BENCH}/{src}/**/assembly/*.s", recursive=True)[0]
ini = glob.glob(f"{BENCH}/{src}/**/ClientParameters.ini", recursive=True)[0]
lines = open(asm).read().split("\n")

WAVEID_READ = "s_bfe_u32 s64, ttmp8, 0x00050019             // patched: wave id (ttmp8[29:25])"

# the one line per tensor, inside the recurring main loop, that sets tile1 from the
# (already wave-specific) clamp register.
targetLine = {
    "A": re.compile(r"^s_mov_b32 s\[sgprtdmAGroup1\+4\], s\[sgprtdmClampA\]"),
    "B": re.compile(r"^s_mov_b32 s\[sgprtdmBGroup1\+4\], s\[sgprtdmClampB\]"),
}
dstReg = {"A": "s[sgprtdmAGroup1+4]", "B": "s[sgprtdmBGroup1+4]"}
ownerWave = {"A": 0, "B": 1}

out = []
inLoop = False
changed = 0
for l in lines:
    if re.match(r"/\* MAINLOOP_C0 start \*/", l):
        inLoop = True
    elif re.match(r"/\* MAINLOOP_C0 end \*/", l):
        inLoop = False

    out.append(l)
    if inLoop:
        for t, rx in targetLine.items():
            if rx.match(l):
                changed += 1
                out.append(WAVEID_READ)
                out.append(f"s_cmp_lg_u32 s64, {ownerWave[t]}              // patched: scc=1 if NOT the owning wave for {t}")
                out.append(f"s_cmov_b32 {dstReg[t]}, 0               // patched: non-owners request 0 rows (same tensorcnt accounting)")
                break

print(f"{changed} tile1 fields patched (expect 2: A, B)", file=sys.stderr)
assert changed == 2, "expected to patch exactly the main-loop A and B tile1 lines"

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-tile0-{name}"
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
