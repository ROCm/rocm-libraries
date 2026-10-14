#!/usr/bin/env python3
"""Per-wave TDM tensor ownership for the DU512 MXF4 subtile kernel (gemm_wt opt_v7 style).

The baseline main loop is cooperative: every wave loads a 64-row slice of A and B and
one of the 4 scale K groups for MXSA and MXSB (4 tensor_load_to_lds per wave per iter).
This patch makes one wave own each tensor and load its whole tile:
  wave 0: A    one 256-row load (wave 0's address / LDS addr are already the tile base)
  wave 1: B    one 256-row load, rebased by -1 wave slice (32*StrideB1J bytes, 17408 LDS)
  wave 2: MXSA 4 loads, one per K group (+-k*SizeI*4 global, +-k*128 LDS around wave 2)
  wave 3: MXSB 4 loads, same with SizeJ
The per-wave LDS slices are contiguous (64 rows * 272B padded = 17408), so the 256-row
load lands byte-identically and local-read addressing is unchanged.

The loop s_wait_tensorcnt becomes per-wave: 1 for the A/B owners, 4 for scale owners.
Prologue loads stay cooperative, so the first iteration over-waits once.

usage: patch_tdm_owner.py SRC_LOGDIR NAME [scales=0]   (scales=0: leave scales cooperative)
"""
import glob, os, re, subprocess, sys, time

BENCH = os.path.dirname(os.path.abspath(__file__))
LLVM = os.environ.get("LLVM_BIN", "/opt/rocm/lib/llvm/bin")

src, name = sys.argv[1], sys.argv[2]
opts = dict(a.split("=") for a in sys.argv[3:])
ownScales = opts.get("scales", "1") != "0"

asm = glob.glob(f"{BENCH}/{src}/**/assembly/*.s", recursive=True)[0]
ini = glob.glob(f"{BENCH}/{src}/**/ClientParameters.ini", recursive=True)[0]
lines = open(asm).read().split("\n")

WAVE_SLICE_LDS = 17408
WAVEID = "s_bfe_u32 s64, ttmp8, 0x00050019             // patched: wave id (ttmp8[29:25])"

def gate(wave, body, tag):
    label = f".L_tdmown_{tag}"
    return [WAVEID, f"s_cmp_eq_u32 s64, {wave}", f"s_cbranch_scc0 {label}"] + body + [f"{label}:"]

def ownAB(block, t):
    size, stride = ("SizeI", "StrideA0I") if t == "A" else ("SizeJ", "StrideB1J")
    owner = 0 if t == "A" else 1
    out = []
    for l in block:
        if l.startswith(f"s_mov_b32 s[sgprtdm{t}Group1+4], s[sgprtdmClamp{t}]"):
            out.append(f"s_min_u32 s[sgprtdm{t}Group1+4], s[sgpr{size}], 256 // patched: whole-tile rows")
            continue
        out.append(l)
        if owner and l.startswith(f"s_mov_b64 s[sgprtdm{t}Group0+2:"):
            out += [f"s_mul_i32 s64, s[sgpr{stride}], {32 * owner}         // patched: owner wave slice offset",
                    f"s_sub_u32 s[sgprtdm{t}Group0+2], s[sgprtdm{t}Group0+2], s64",
                    f"s_subb_u32 s[sgprtdm{t}Group0+3], s[sgprtdm{t}Group0+3], 0"]
        if owner and l.startswith(f"s_mov_b32 s[sgprtdm{t}Group0+1], s[sgprtdmLdsAddr{t}]"):
            out.append(f"s_sub_u32 s[sgprtdm{t}Group0+1], s[sgprtdm{t}Group0+1], {WAVE_SLICE_LDS * owner}")
    return out

def ownScale(block, t):
    size = "SizeI" if t == "MXSA" else "SizeJ"
    owner = 2 if t == "MXSA" else 3
    out = []
    for k in range(4):
        d = k - owner
        for l in block:
            out.append(l)
            if d and l.startswith(f"s_mov_b64 s[sgprtdm{t}Group0+2:"):
                op, opc = ("s_add_u32", "s_addc_u32") if d > 0 else ("s_sub_u32", "s_subb_u32")
                out += [f"s_mul_i32 s64, s[sgpr{size}], {4 * abs(d)}           // patched: K group {k} row offset",
                        f"{op} s[sgprtdm{t}Group0+2], s[sgprtdm{t}Group0+2], s64",
                        f"{opc} s[sgprtdm{t}Group0+3], s[sgprtdm{t}Group0+3], 0"]
            if d and l.startswith(f"s_mov_b32 s[sgprtdm{t}Group0+1], s[sgprtdmLdsAddr{t}]"):
                op = "s_add_u32" if d > 0 else "s_sub_u32"
                out.append(f"{op} s[sgprtdm{t}Group0+1], s[sgprtdm{t}Group0+1], {128 * abs(d)}")
    return out

blockStartRe = {t: re.compile(rf"^s_mov_b32 s\[sgprtdm{t}Group0\+0\], 1$") for t in ("A", "B", "MXSA", "MXSB")}
out, inLoop, done, waits = [], False, [], 0
i, n = 0, len(lines)
while i < n:
    l = lines[i]
    if l.startswith("/* MAINLOOP_C0 start */"):
        inLoop = True
    elif l.startswith("/* MAINLOOP_C0 end */"):
        inLoop = False
    t = next((t for t, rx in blockStartRe.items() if rx.match(l)), None) if inLoop else None
    if t and (t in ("A", "B") or ownScales):
        j = i
        while f"// TDM: global->LDS for {t}" not in lines[j]:
            j += 1
        block = lines[i:j + 1]
        if t in ("A", "B"):
            out += gate(0 if t == "A" else 1, ownAB(block, t), t)
        else:
            out += gate(2 if t == "MXSA" else 3, ownScale(block, t), t)
        done.append(t)
        i = j + 1
        continue
    if inLoop and re.match(r"^s_wait_tensorcnt \d+", l):
        # waves 0/1 issue 1 (+2 cooperative scales if not owned); waves 2/3 issue 4 (or 2)
        lo, hi = (1, 4) if ownScales else (3, 2)
        out += [WAVEID,
                "s_cmp_lt_u32 s64, 2",
                "s_cbranch_scc0 .L_tdmown_wait_hi",
                f"s_wait_tensorcnt {lo}                                 // patched: A/B owner waves",
                "s_branch .L_tdmown_wait_done",
                ".L_tdmown_wait_hi:",
                f"s_wait_tensorcnt {hi}                                 // patched: scale owner waves",
                ".L_tdmown_wait_done:"]
        waits += 1
        i += 1
        continue
    out.append(l)
    i += 1

print(f"owned {done}, rewrote {waits} loop waits", file=sys.stderr)
assert waits == 1 and len(done) == (4 if ownScales else 2)

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-tdmown-{name}"
os.makedirs(dst)
s = f"{dst}/kernel.s"
open(s, "w").write("\n".join(out))
subprocess.run([f"{LLVM}/clang", "-x", "assembler", "--target=amdgcn-amd-amdhsa", "-mcode-object-version=4",
                "-c", "-Xclangas", "-target-feature", "-Xclangas", "+real-true16", "-mcpu=gfx1250",
                "-mno-wavefrontsize64", s, "-o", f"{dst}/kernel.o"], check=True,
               stderr=subprocess.DEVNULL)
subprocess.run([f"{LLVM}/clang", "--target=amdgcn-amd-amdhsa", "-Xlinker", "--build-id=sha1",
                f"{dst}/kernel.o", "-o", f"{dst}/kernel.co"], check=True)
cfg = re.sub(r"^code-object=.*$", f"code-object={dst}/kernel.co", open(ini).read(), flags=re.M)
open(f"{dst}/ClientParameters.ini", "w").write(cfg)
print(dst)
