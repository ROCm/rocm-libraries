#!/usr/bin/env python3
"""Slim A+B Group1 rebuilds via SGPR shadow (post-dsspread SALU lever).

A/B Group1[0:3] (control + dims) are invariant across K within a tile; only
Group0 (addr + LDS) and G1+4 (clamp) / G1+5 (stride) need per-iter work.
MXSA/MXSB alias those SGPRs, forcing full rebuilds. Unalias needs +24 SGPRs.

Spill Address*Base (s52-s59) to v[1004:1011] (MSB bank 768) once; rebase via
v_readfirstlane. Repurpose s52-s59 as A G1[0:3] (s52-s55) + B G1[0:3] (s56-s59).
Capture at the pre-MAINLOOP rebuild (always runs iff MAINLOOP runs); slim only
the MAINLOOP A/B rebuilds. PRELOOP/TAILLOOP stay full.

usage: patch_salu_shadow.py SRC_LOGDIR NAME
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

VGPR0 = 1004  # physical; encoded as v[N-768] with msb dst=3 (192)
SHA0, SHB0 = 52, 56  # A/B G1[0:3] shadows (8 SGPRs = former Address*Base)
stats = {"spill": 0, "rebase": 0, "capA": 0, "capB": 0, "slimA": 0, "slimB": 0}


def is_g1_ctrl(l, tc):
    imm = "0x7500000" if tc in ("A", "B") else None
    return l.startswith(f"s_mov_b32 s[sgprtdm{tc}Group1], {imm}")


def find_g1_end(lines, ctrl_i, tc):
    for j in range(ctrl_i, min(ctrl_i + 30, len(lines))):
        if f"s_mov_b64 s[sgprtdm{tc}Group1+6" in lines[j]:
            return j
    raise AssertionError(f"no {tc} Group1+6 at {ctrl_i}")


def find_rebuild_start(lines, ctrl_i, tc):
    for j in range(ctrl_i, max(ctrl_i - 5, -1), -1):
        if lines[j].startswith(f"s_mov_b32 s[sgprtdm{tc}Group0+0], 1"):
            return j
    raise AssertionError(f"no {tc} Group0+0 at {ctrl_i}")


def vgpr(phys):
    return f"v[{phys}-768]"


def spill_bases():
    out = ["s_set_vgpr_msb 192                                 "
           "// patched: dst bank 3 for Address*Base spill"]
    names = [
        "sgprAddressABase", "sgprAddressABase+1",
        "sgprAddressBBase", "sgprAddressBBase+1",
        "sgprAddressMXSABase", "sgprAddressMXSABase+1",
        "sgprAddressMXSBBase", "sgprAddressMXSBBase+1",
    ]
    for i, sg in enumerate(names):
        out.append(f"v_mov_b32 {vgpr(VGPR0+i)}, s[{sg}] "
                   f"// patched: spill {sg}")
    out.append("s_set_vgpr_msb 0                                   "
               "// patched: restore vgpr msb after base spill")
    return out


def rebase_from_vgpr():
    # v_readfirstlane: VGPR is src0 → bank 3; pack = dst[7:6]|src2[5:4]|src1[3:2]|src0[1:0]
    out = ["s_set_vgpr_msb 3                                   "
           "// patched: src0 bank 3 for Address*Base reload"]
    pairs = [
        ("sgprAddressA", 0), ("sgprAddressB", 2),
        ("sgprAddressMXSA", 4), ("sgprAddressMXSB", 6),
    ]
    for sg, off in pairs:
        out.append(f"v_readfirstlane_b32 s[{sg}+0], {vgpr(VGPR0+off)} "
                   f"// patched: re-base {sg} lo")
        out.append(f"v_readfirstlane_b32 s[{sg}+1], {vgpr(VGPR0+off+1)} "
                   f"// patched: re-base {sg} hi")
    out.append("s_set_vgpr_msb 0                                   "
               "// patched: restore vgpr msb after rebase")
    return out


def capture(tc, sh):
    return [
        f"s_mov_b64 s[{sh}:{sh}+1], s[sgprtdm{tc}Group1+0:sgprtdm{tc}Group1+0+1] "
        f"// patched: capture {tc} G1[0:1] shadow",
        f"s_mov_b64 s[{sh}+2:{sh}+3], s[sgprtdm{tc}Group1+2:sgprtdm{tc}Group1+2+1] "
        f"// patched: capture {tc} G1[2:3] shadow",
    ]


def slim(tc, sh):
    addr = f"sgprAddress{tc}"
    lds = f"sgprtdmLdsAddr{tc}"
    clamp = f"sgprtdmClamp{tc}"
    stride = "sgprStrideA0I" if tc == "A" else "sgprStrideB1J"
    return [
        f"s_mov_b32 s[sgprtdm{tc}Group0+0], 1                            "
        f"// patched: slim {tc} Group0/1 from shadow",
        f"s_mov_b64 s[sgprtdm{tc}Group0+2:sgprtdm{tc}Group0+2+1], s[{addr}:{addr}+1]",
        f"s_or_b32 s[sgprtdm{tc}Group0+3], s[sgprtdm{tc}Group0+3], 0x80000000 "
        f"// set type field to 2(image)",
        f"s_mov_b32 s[sgprtdm{tc}Group0+1], s[{lds}]",
        f"s_mov_b64 s[sgprtdm{tc}Group1+0:sgprtdm{tc}Group1+0+1], s[{sh}:{sh}+1] "
        f"// patched: {tc} G1 from shadow",
        f"s_mov_b64 s[sgprtdm{tc}Group1+2:sgprtdm{tc}Group1+2+1], s[{sh}+2:{sh}+3]",
        f"s_mov_b32 s[sgprtdm{tc}Group1+4], s[{clamp}]    // set tile1 = clamped validRows",
        f"s_lshr_b32 s[sgprtdm{tc}Group1+5], s[{stride}], 0x1",
        f"s_mov_b64 s[sgprtdm{tc}Group1+6:sgprtdm{tc}Group1+6+1], 0 "
        f"// Group1+6/+7 reserved",
    ]


# --- Pass 0: spill bases + rebase from VGPR ---
out = []
i = 0
n = len(lines)
while i < n:
    l = lines[i]
    if re.match(r"^\s*\.amdhsa_next_free_vgpr\s+\d+", l):
        out.append(f"  .amdhsa_next_free_vgpr {VGPR0 + 8} "
                   f"// patched: +8 for Address*Base spill")
        i += 1
        continue
    if ("s_mov_b64 s[sgprAddressMXSBBase:sgprAddressMXSBBase+1], "
            "s[sgprAddressMXSB:sgprAddressMXSB+1]" in l):
        out.append(l)
        out.extend(spill_bases())
        stats["spill"] = 1
        i += 1
        continue
    if (l.startswith("s_mov_b64 s[sgprAddressA:sgprAddressA+1], "
                     "s[sgprAddressABase:sgprAddressABase+1]")):
        out.extend(rebase_from_vgpr())
        stats["rebase"] += 1
        i += 1
        while i < n and (
                lines[i].startswith("s_mov_b64 s[sgprAddressB:sgprAddressB+1], s[sgprAddressBBase") or
                lines[i].startswith("s_mov_b64 s[sgprAddressMXSA:sgprAddressMXSA+1], s[sgprAddressMXSABase") or
                lines[i].startswith("s_mov_b64 s[sgprAddressMXSB:sgprAddressMXSB+1], s[sgprAddressMXSBBase")):
            i += 1
        continue
    out.append(l)
    i += 1
assert stats["spill"] == 1 and stats["rebase"] >= 1, stats
lines = out

# --- Pass 1: capture at pre-MAINLOOP A/B; slim MAINLOOP A/B ---
a_ctrls = [i for i, l in enumerate(lines) if is_g1_ctrl(l, "A")]
b_ctrls = [i for i, l in enumerate(lines) if is_g1_ctrl(l, "B")]
assert len(a_ctrls) >= 3 and len(b_ctrls) >= 3, (a_ctrls, b_ctrls)

main_start = next(i for i, l in enumerate(lines) if "MAINLOOP_C0 start" in l)
main_end = next(i for i, l in enumerate(lines) if "MAINLOOP_C0 end" in l)
# pre-MAINLOOP = last A/B ctrl before MAINLOOP_C0 start
pre_a = max(c for c in a_ctrls if c < main_start)
pre_b = max(c for c in b_ctrls if c < main_start)
main_a = [c for c in a_ctrls if main_start < c < main_end]
main_b = [c for c in b_ctrls if main_start < c < main_end]
assert len(main_a) == 1 and len(main_b) == 1, (main_a, main_b)

cap_a_end = find_g1_end(lines, pre_a, "A")
cap_b_end = find_g1_end(lines, pre_b, "B")
slim_a_start = find_rebuild_start(lines, main_a[0], "A")
slim_a_end = find_g1_end(lines, main_a[0], "A")
slim_b_start = find_rebuild_start(lines, main_b[0], "B")
slim_b_end = find_g1_end(lines, main_b[0], "B")

out = []
i = 0
n = len(lines)
while i < n:
    if i == cap_a_end:
        out.append(lines[i])
        out.extend(capture("A", SHA0))
        stats["capA"] += 1
        i += 1
        continue
    if i == cap_b_end:
        out.append(lines[i])
        out.extend(capture("B", SHB0))
        stats["capB"] += 1
        i += 1
        continue
    if i == slim_a_start:
        out.extend(slim("A", SHA0))
        stats["slimA"] += 1
        i = slim_a_end + 1
        continue
    if i == slim_b_start:
        out.extend(slim("B", SHB0))
        stats["slimB"] += 1
        i = slim_b_end + 1
        continue
    out.append(lines[i])
    i += 1

assert stats["capA"] == 1 and stats["capB"] == 1 and stats["slimA"] == 1 and stats["slimB"] == 1, stats

text = "\n".join(out)
if not text.endswith("\n"):
    text += "\n"

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-salushadow-{name}"
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
print(f"spill={stats['spill']} rebase={stats['rebase']} "
      f"capA={stats['capA']} capB={stats['capB']} "
      f"slimA={stats['slimA']} slimB={stats['slimB']}", file=sys.stderr)
print(dst)
