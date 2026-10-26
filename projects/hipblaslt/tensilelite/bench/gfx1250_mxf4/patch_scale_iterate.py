#!/usr/bin/env python3
"""Scale TDM ITERATE: one issuer wave per scale tensor covers all K groups.

Baseline (DU512, WT 4x8, 4 waves): every wave issues MXSA and MXSB for its own
K group (spansPerKGroup=1) — 2 scale tensor_load_to_lds per wave, 4th in the
A,B,SA,SB burst. That 4th load is the ~289-cycle issue Idle (MXSB).

This patch (prototype mx_v7-style):
  - Wave 0 issues MXSA once with iterate_count=3 (4 K groups), lds_inc=128,
    global_inc=SizeI*4; waves 1-3 skip MXSA.
  - Wave 1 issues MXSB once with the same iterate pattern (SizeJ*4); others skip.
  - Keeps the existing pad word and tile1=8 so LDS layout/LR offsets stay put.
  - Uses s[100:103] as a shared Group2 (SRG2) scratch (4-aligned); bumps next_free_sgpr.
  - Fixes shadow s_wait_tensorcnt 4 for asymmetric issue counts (owners 3, others 2).

Per-wave issue count becomes: w0: A,B,SA (3); w1: A,B,SB (3); w2/w3: A,B (2).

usage: patch_scale_iterate.py SRC_LOGDIR NAME
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

# Scratch Group2 in high SGPRs (baseline next_free_sgpr is 98).
# Must be 4-aligned for tensor_load_to_lds SRG2.
G2 = 100  # s[100:103]

def bump_sgpr(lines, need):
    for i, l in enumerate(lines):
        m = re.match(r"^(\s*\.amdhsa_next_free_sgpr\s+)(\d+)(.*)$", l)
        if m and int(m.group(2)) < need:
            lines[i] = f"{m.group(1)}{need} // patched: Group2 scratch s[{G2}:{G2+3}]{m.group(3)}"
            return True
    return False

assert bump_sgpr(lines, G2 + 4), "failed to bump next_free_sgpr"

loadRe = {
    "MXSA": re.compile(
        r"^tensor_load_to_lds s\[sgprtdmMXSAGroup0:sgprtdmMXSAGroup0\+3\], "
        r"s\[sgprtdmMXSAGroup1:sgprtdmMXSAGroup1\+7\](.*)$"),
    "MXSB": re.compile(
        r"^tensor_load_to_lds s\[sgprtdmMXSBGroup0:sgprtdmMXSBGroup0\+3\], "
        r"s\[sgprtdmMXSBGroup1:sgprtdmMXSBGroup1\+7\](.*)$"),
}
clearIterRe = {
    "MXSA": re.compile(
        r"^s_and_b32 s\[sgprtdmMXSAGroup1\], s\[sgprtdmMXSAGroup1\], 0xfff7ffff(.*)$"),
    "MXSB": re.compile(
        r"^s_and_b32 s\[sgprtdmMXSBGroup1\], s\[sgprtdmMXSBGroup1\], 0xfff7ffff(.*)$"),
}
waitRe = re.compile(r"^s_wait_tensorcnt 4(\s+//.*)?$")
owner = {"MXSA": 0, "MXSB": 1}
sizeSgpr = {"MXSA": "sgprSizeI", "MXSB": "sgprSizeJ"}

out = []
stats = {"clear": 0, "load": 0, "wait": 0}
counter = 0
for l in lines:
    # 1) Keep pad, enable iterate instead of clearing it.
    matched_clear = next((t for t, rx in clearIterRe.items() if rx.match(l)), None)
    if matched_clear:
        out.append(
            f"s_or_b32 s[sgprtdm{matched_clear}Group1], s[sgprtdm{matched_clear}Group1], 0x80000 "
            f"// patched: set iterate_enable (keep pad)")
        stats["clear"] += 1
        continue

    # 2) Replace scale tensor_load with owner-gated iterate form.
    matched_load = next((t for t, rx in loadRe.items() if rx.match(l)), None)
    if matched_load:
        t = matched_load
        label = f".L_sciter_{t}_{counter}"
        counter += 1
        stats["load"] += 1
        out.append(f"s_cmp_eq_u32 s[sgprWaveIdx], {owner[t]}          "
                   f"// patched: {t} iterate issuer = wave {owner[t]}")
        out.append(f"s_cbranch_scc0 {label}                         "
                   f"// patched: other waves skip {t}")
        # Owner wave's Address/LDS were for its own K group. Rebase descriptor
        # to K group 0 so ITERATE walks 0..3 (no-op when owner wave is 0).
        out.append(f"s_lshl_b32 s{G2}, s[{sizeSgpr[t]}], 2              "
                   f"// patched: tmp = Size*4 (one K-group global row)")
        out.append(f"s_mul_i32 s{G2}, s{G2}, s[sgprWaveIdx]             "
                   f"// patched: * kGroup (= waveId)")
        out.append(f"s_sub_u32 s[sgprtdm{t}Group0+2], s[sgprtdm{t}Group0+2], s{G2} "
                   f"// patched: global addr back to K group 0")
        out.append(f"s_subb_u32 s[sgprtdm{t}Group0+3], s[sgprtdm{t}Group0+3], 0")
        out.append(f"s_mul_i32 s{G2}, s[sgprWaveIdx], 128               "
                   f"// patched: kGroup * chunk")
        out.append(f"s_sub_u32 s[sgprtdm{t}Group0+1], s[sgprtdm{t}Group0+1], s{G2} "
                   f"// patched: LDS back to K group 0 column")
        out.append(f"s_mov_b32 s{G2}, 1                               "
                   f"// patched: Group2+0 tensor_dim2 = 1")
        out.append(f"s_mov_b32 s{G2+1}, 128                           "
                   f"// patched: Group2+1 lds_inc = 128 (next K-group column)")
        out.append(f"s_lshl_b32 s{G2+2}, s[{sizeSgpr[t]}], 2            "
                   f"// patched: Group2+2 global_inc = Size*4 (K-group row)")
        out.append(f"s_mov_b32 s{G2+3}, 0x30000                       "
                   f"// patched: Group2+3 iter_count=3 << 16 (4 K groups)")
        out.append(
            f"tensor_load_to_lds s[sgprtdm{t}Group0:sgprtdm{t}Group0+3], "
            f"s[sgprtdm{t}Group1:sgprtdm{t}Group1+7], "
            f"s[{G2}:{G2+3}], null // patched: {t} ITERATE x4 K groups")
        out.append(f"{label}:")
        continue

    # 3) Shadow wait assumed 4 issues/wave. Owners issue 3, others 2.
    if waitRe.match(l):
        stats["wait"] += 1
        lab2 = f".L_sciter_wait2_{stats['wait']}"
        labd = f".L_sciter_waitdone_{stats['wait']}"
        out += [
            f"s_cmp_ge_u32 s[sgprWaveIdx], 2                       "
            f"// patched: waves 2/3 issue only A,B",
            f"s_cbranch_scc1 {lab2}",
            f"s_wait_tensorcnt 3                                   "
            f"// patched: waves 0/1: A,B,S* (was 4)",
            f"s_branch {labd}",
            f"{lab2}:",
            f"s_wait_tensorcnt 2                                   "
            f"// patched: waves 2/3: A,B only",
            f"{labd}:",
        ]
        continue

    out.append(l)

print(f"clear->iterate: {stats['clear']}  loads gated: {stats['load']}  "
      f"wait4->split: {stats['wait']}", file=sys.stderr)
assert stats["clear"] == 8, stats
assert stats["load"] == 8, stats
assert stats["wait"] >= 1, stats

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-sciter-{name}"
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
