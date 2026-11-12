#!/usr/bin/env python3
"""Even-wave A/B TDM consolidation for hoisted cluster kernels.

Widens per-wave A/B clamps 64->128 and issues A/B tensor_load_to_lds only on
even waves (0,2). Odd waves still issue MXSA/MXSB. Main-loop wait_tensorcnt 4
becomes wave-conditional (even=4, odd=2).

Unlike patch_tdm_consolidate_even_v3, this gates only the load sites (not the
descriptor blocks): the TDM-setup hoist puts A/B descriptor rebuild above the
FREE barrier and both loads after it, so the old block matcher merges them.

Uses s_bitcmp0 on sgprWaveIdx (no temp SGPR). Do NOT use bare s64 — on the
CD[2,4] DU512 kernel that is sgprtdmAGroup1.

usage: patch_tdm_cons_hoist.py SRC_LOGDIR NAME
SRC_LOGDIR may be logs/<stamp> with kernel.s at the top level or under **/assembly/.

Measured on logs/20261014-205500-hoist-sk3_24 (CD[2,4], GPU 3):
  validate.py 10/10 PASSED (mx-scale-jitter=4)
  alt_timing.py 8 rounds: 0.83x vs baseline (202.5 -> 244.3 us) — CORRECT BUT SLOWER.
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

widened = {"A": 0, "B": 0}
for i, l in enumerate(lines):
    for t, imm in (("A", "sgprtdmClampA"), ("B", "sgprtdmClampB")):
        m = re.match(rf"^(s_min_u32 s\[{imm}\], s\[{imm}\]), 64(\s*//.*)?$", l)
        if m:
            lines[i] = (f"{m.group(1)}, 128                                  "
                        f"// patched: widened 64->128 (even wave absorbs odd partner)")
            widened[t] += 1
assert widened == {"A": 1, "B": 1}, widened

loadRe = {
    "A": re.compile(r"^tensor_load_to_lds s\[sgprtdmAGroup0:sgprtdmAGroup0\+3\], "
                    r"s\[sgprtdmAGroup1:sgprtdmAGroup1\+7\].*$"),
    "B": re.compile(r"^tensor_load_to_lds s\[sgprtdmBGroup0:sgprtdmBGroup0\+3\], "
                    r"s\[sgprtdmBGroup1:sgprtdmBGroup1\+7\].*$"),
}
GATE = ["s_bitcmp0_b32 s[sgprWaveIdx], 0                   "
        "// patched: SCC=1 if even wave (bit0==0)"]

out, gated, waitFixed, counter = [], {"A": 0, "B": 0}, 0, 0
for l in lines:
    matched = next((t for t, rx in loadRe.items() if rx.match(l)), None)
    if matched:
        label = f".L_tdmcons_{matched}_{counter}"
        counter += 1
        gated[matched] += 1
        out += GATE + [
            f"s_cbranch_scc0 {label}                         "
            f"// patched: odd waves skip {matched} TDM",
            l, f"{label}:"]
        continue
    if re.match(r"^s_wait_tensorcnt 4(\s*//.*)?$", l):
        out += GATE + [
            "s_cbranch_scc0 .L_tdmcons_wait_odd              // patched: odd -> wait 2",
            "s_wait_tensorcnt 4                              // patched: even (A,B,SA,SB)",
            "s_branch .L_tdmcons_wait_end",
            ".L_tdmcons_wait_odd:",
            "s_wait_tensorcnt 2                              // patched: odd (SA,SB only)",
            ".L_tdmcons_wait_end:"]
        waitFixed += 1
        continue
    out.append(l)

print(f"widened={widened} gated={gated} waitFixed={waitFixed}", file=sys.stderr)
assert gated == {"A": 4, "B": 4} and waitFixed == 1

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-tdmcons-{name}"
os.makedirs(dst)
s = f"{dst}/kernel.s"
open(s, "w").write("\n".join(out))
subprocess.run([f"{LLVM}/clang", "-x", "assembler", "--target=amdgcn-amd-amdhsa",
                "-mcode-object-version=4", "-c", "-Xclangas", "-target-feature",
                "-Xclangas", "+real-true16", "-mcpu=gfx1250", "-mno-wavefrontsize64",
                s, "-o", f"{dst}/kernel.o"], check=True)
subprocess.run([f"{LLVM}/clang", "--target=amdgcn-amd-amdhsa", "-Xlinker",
                "--build-id=sha1", f"{dst}/kernel.o", "-o", f"{dst}/kernel.co"],
               check=True)
open(f"{dst}/ClientParameters.ini", "w").write(
    re.sub(r"^code-object=.*$", f"code-object={dst}/kernel.co",
           open(ini).read(), flags=re.M))
print(dst)
