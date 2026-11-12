#!/usr/bin/env python3
"""Unalias MXSA/MXSB TDM descriptors from A/B (MXSB-idle lever #2).

Baseline aliases:
  sgprtdmMXSAGroup{0,1} -> A Group{0,1}
  sgprtdmMXSBGroup{0,1} -> B Group{0,1}

So after A/B issue the scale descriptor rebuild must overwrite those SGPRs,
forcing a serial rebuild+reissue window (A,B issue → rebuild SA → issue SA →
rebuild SB → issue SB). That is the SGPR-alias half of the 4th-issue Idle.

This patch:
  1. Gives MXSA/MXSB dedicated 4+8 SGPR pairs (100/104 and 112/116; next_free 124).
  2. In MAINLOOP_C0, hoists the scale descriptor rebuilds above the FREE barrier
     (safe once unaliased), so after the barrier the four tensor_load_to_lds
     issue back-to-back with no SALU between them.

usage: patch_tdm_unalias.py SRC_LOGDIR NAME
"""
import glob, os, re, subprocess, sys, time

BENCH = os.path.dirname(os.path.abspath(__file__))
LLVM = os.environ.get("LLVM_BIN", "/opt/rocm/lib/llvm/bin")

MXSA_G0, MXSA_G1 = 100, 104
MXSB_G0, MXSB_G1 = 112, 116
NEXT_FREE = 124

src, name = sys.argv[1], sys.argv[2]
src = src if os.path.isabs(src) else f"{BENCH}/{src}"
asms = glob.glob(f"{src}/**/assembly/*.s", recursive=True) + glob.glob(f"{src}/kernel.s")
inis = glob.glob(f"{src}/**/ClientParameters.ini", recursive=True) + glob.glob(f"{src}/ClientParameters.ini")
asm, ini = asms[0], inis[0]
lines = open(asm).read().split("\n")

# --- 1) Rematerialize .set aliases + bump SGPR count ---
out, sets = [], 0
for l in lines:
    if re.match(r"^\s*\.amdhsa_next_free_sgpr\s+\d+", l):
        out.append(f"  .amdhsa_next_free_sgpr {NEXT_FREE} // patched: unalias MXSA/MXSB descriptors")
        continue
    if re.match(r"^\s*\.sgpr_count:\s+\d+", l):
        out.append(f"    .sgpr_count:                 {NEXT_FREE}")
        continue
    repl = {
        ".set sgprtdmMXSAGroup0, sgprtdmAGroup0+0":
            f".set sgprtdmMXSAGroup0, {MXSA_G0}  // patched: dedicated (was alias A)",
        ".set sgprtdmMXSAGroup1, sgprtdmAGroup1+0":
            f".set sgprtdmMXSAGroup1, {MXSA_G1}  // patched: dedicated (was alias A)",
        ".set sgprtdmMXSBGroup0, sgprtdmBGroup0+0":
            f".set sgprtdmMXSBGroup0, {MXSB_G0}  // patched: dedicated (was alias B)",
        ".set sgprtdmMXSBGroup1, sgprtdmBGroup1+0":
            f".set sgprtdmMXSBGroup1, {MXSB_G1}  // patched: dedicated (was alias B)",
    }
    if l.strip() in repl:
        out.append(repl[l.strip()])
        sets += 1
        continue
    out.append(l)
assert sets == 4, sets
lines = out


def find_rebuild(lines, tc, start):
    """(rebuild_start, load_idx) for scale tensor tc; rebuild ends just before load."""
    load_re = re.compile(
        rf"^tensor_load_to_lds s\[sgprtdm{tc}Group0:sgprtdm{tc}Group0\+3\], "
        rf"s\[sgprtdm{tc}Group1:sgprtdm{tc}Group1\+7\]")
    load_idx = next(i for i in range(start, len(lines)) if load_re.match(lines[i]))
    rebuild_start = next(
        i for i in range(load_idx, start - 1, -1)
        if re.match(rf"^s_mov_b32 s\[sgprtdm{tc}Group0\+0\], 1$", lines[i]))
    return rebuild_start, load_idx


def desc_only(lines, rs, ld):
    block = lines[rs:ld]
    while block and "Scale GR" in block[-1]:
        block = block[:-1]
    return block


# --- 2) Hoist scale descriptor rebuild above FREE barrier in MAINLOOP_C0 ---
out, i, hoisted = [], 0, 0
n = len(lines)
in_main = False
while i < n:
    l = lines[i]
    if "MAINLOOP_C0 start" in l:
        in_main = True
    if "MAINLOOP_C0 end" in l:
        in_main = False

    if (in_main and "TDM: swap MXSB LDS buffer" in l and
            i + 1 < n and lines[i + 1].startswith("s_xor_b32 s[sgprtdmLdsAddrMXSB]")):
        out.append(l)
        out.append(lines[i + 1])
        # Next non-empty setup leads to barrier then A,B loads then scale rebuilds.
        j = i + 2
        while j < n and not (lines[j].startswith("tensor_load_to_lds") and "for A" in lines[j]):
            j += 1
        assert "for B" in lines[j + 1], lines[j + 1]
        sa_rs, sa_ld = find_rebuild(lines, "MXSA", j + 2)
        sb_rs, sb_ld = find_rebuild(lines, "MXSB", sa_ld + 1)

        out.append("// patched: hoisted MXSA/MXSB descriptor rebuild (unaliased)")
        out.extend(desc_only(lines, sa_rs, sa_ld))
        out.extend(desc_only(lines, sb_rs, sb_ld))
        # Pre-barrier remainder (wait_dscnt + barrier), then A/B/SA/SB loads only.
        for k in range(i + 2, j):
            out.append(lines[k])
        out.append(lines[j])       # A load
        out.append(lines[j + 1])   # B load
        for ld in (sa_ld, sb_ld):
            if "Scale GR" in lines[ld - 1]:
                out.append(lines[ld - 1])
            out.append(lines[ld])
        i = sb_ld + 1
        hoisted += 1
        continue

    out.append(l)
    i += 1

assert hoisted == 1, hoisted

text = "\n".join(out)
assert "sgprtdmMXSAGroup0, sgprtdmAGroup0" not in text
assert f".set sgprtdmMXSAGroup0, {MXSA_G0}" in text
assert f".set sgprtdmMXSBGroup0, {MXSB_G0}" in text
m = re.search(
    r"MAINLOOP_C0 start.*?tensor_load_to_lds s\[sgprtdmAGroup0.*?"
    r"tensor_load_to_lds s\[sgprtdmBGroup0.*?"
    r"tensor_load_to_lds s\[sgprtdmMXSAGroup0.*?"
    r"tensor_load_to_lds s\[sgprtdmMXSBGroup0",
    text, re.S)
assert m, "mainloop A,B,SA,SB issue burst not found"
b_to_sa = re.search(
    r"tensor_load_to_lds s\[sgprtdmBGroup0.*?tensor_load_to_lds s\[sgprtdmMXSAGroup0",
    m.group(0), re.S).group(0)
assert "s_mov_b32 s[sgprtdmMXSAGroup0+0], 1" not in b_to_sa

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-tdmunalias-{name}"
os.makedirs(dst)
s = f"{dst}/kernel.s"
open(s, "w").write(text if text.endswith("\n") else text + "\n")
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
print(f"unaliased MXSA@{MXSA_G0}/{MXSA_G1} MXSB@{MXSB_G0}/{MXSB_G1} "
      f"next_free={NEXT_FREE} hoist={hoisted}", file=sys.stderr)
print(dst)
