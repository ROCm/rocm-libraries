#!/usr/bin/env python3
"""Issue scale TDMs from waves 2/3 with ITERATE (MXSB-idle lever #3).

Baseline: every wave issues A,B,MXSA,MXSB (4). MXSB as 4th sits ~289 cycles in
issue Idle; barriers make that everyone's problem, but WMMAs between issue and
wait_tensorcnt can hide it if the stalled wave is not the one executing those
early WMMAs.

sciter2 put MXSA on wave 0 and MXSB on wave 1 (still the waves that immediately
run post-issue WMMAs). This patch moves scale ownership to waves 2/3:

  - Wave 2 issues MXSA once with iterate_count=3 (all 4 K groups).
  - Wave 3 issues MXSB once with the same iterate pattern.
  - Waves 0/1 issue only A,B and can start WMMAs while 2/3 finish scales.
  - Per-wave issues: w0/w1=2 (A,B); w2=3 (A,B,SA); w3=3 (A,B,SB).
  - wait_tensorcnt split: 2 for w0/w1, 3 for w2/w3.

Group2 scratch at s[100:103]; bumps next_free_sgpr. Uses sgprWaveIdx (not s64).

usage: patch_tdm_scalewave.py SRC_LOGDIR NAME
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

G2 = 100  # s[100:103], 4-aligned


def bump_sgpr(lines, need):
    for i, l in enumerate(lines):
        m = re.match(r"^(\s*\.amdhsa_next_free_sgpr\s+)(\d+)(.*)$", l)
        if m and int(m.group(2)) < need:
            lines[i] = (f"{m.group(1)}{need} // patched: Group2 scratch "
                        f"s[{G2}:{G2+3}]{m.group(3)}")
            return True
        m = re.match(r"^(\s*\.sgpr_count:\s+)(\d+)(.*)$", l)
        if m and int(m.group(2)) < need:
            lines[i] = f"{m.group(1)}{need}{m.group(3)}"
    return False


assert bump_sgpr(lines, G2 + 4)

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
owner = {"MXSA": 2, "MXSB": 3}
sizeSgpr = {"MXSA": "sgprSizeI", "MXSB": "sgprSizeJ"}

out, stats, counter = [], {"clear": 0, "load": 0, "wait": 0}, 0
for l in lines:
    matched_clear = next((t for t, rx in clearIterRe.items() if rx.match(l)), None)
    if matched_clear:
        out.append(
            f"s_or_b32 s[sgprtdm{matched_clear}Group1], s[sgprtdm{matched_clear}Group1], "
            f"0x80000 // patched: set iterate_enable (keep pad)")
        stats["clear"] += 1
        continue

    matched_load = next((t for t, rx in loadRe.items() if rx.match(l)), None)
    if matched_load:
        t = matched_load
        label = f".L_scwave_{t}_{counter}"
        counter += 1
        stats["load"] += 1
        out.append(f"s_cmp_eq_u32 s[sgprWaveIdx], {owner[t]}          "
                   f"// patched: {t} iterate issuer = wave {owner[t]}")
        out.append(f"s_cbranch_scc0 {label}                         "
                   f"// patched: other waves skip {t}")
        # Owner's Address/LDS were for its own K group; rebase to group 0.
        out.append(f"s_lshl_b32 s{G2}, s[{sizeSgpr[t]}], 2              "
                   f"// patched: tmp = Size*4")
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
                   f"// patched: Group2+1 lds_inc = 128")
        out.append(f"s_lshl_b32 s{G2+2}, s[{sizeSgpr[t]}], 2            "
                   f"// patched: Group2+2 global_inc = Size*4")
        out.append(f"s_mov_b32 s{G2+3}, 0x30000                       "
                   f"// patched: Group2+3 iter_count=3 << 16")
        out.append(
            f"tensor_load_to_lds s[sgprtdm{t}Group0:sgprtdm{t}Group0+3], "
            f"s[sgprtdm{t}Group1:sgprtdm{t}Group1+7], "
            f"s[{G2}:{G2+3}], null // patched: {t} ITERATE x4 on wave {owner[t]}")
        out.append(f"{label}:")
        continue

    if waitRe.match(l):
        stats["wait"] += 1
        lab3 = f".L_scwave_wait3_{stats['wait']}"
        labd = f".L_scwave_waitdone_{stats['wait']}"
        out += [
            f"s_cmp_ge_u32 s[sgprWaveIdx], 2                       "
            f"// patched: waves 2/3 issue A,B,S*",
            f"s_cbranch_scc1 {lab3}",
            f"s_wait_tensorcnt 2                                   "
            f"// patched: waves 0/1: A,B only",
            f"s_branch {labd}",
            f"{lab3}:",
            f"s_wait_tensorcnt 3                                   "
            f"// patched: waves 2/3: A,B,S*",
            f"{labd}:",
        ]
        continue

    out.append(l)

print(f"clear->iterate: {stats['clear']}  loads gated: {stats['load']}  "
      f"wait4->split: {stats['wait']}", file=sys.stderr)
assert stats["clear"] == 8, stats
assert stats["load"] == 8, stats
assert stats["wait"] >= 1, stats

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-tdmscwave-{name}"
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
