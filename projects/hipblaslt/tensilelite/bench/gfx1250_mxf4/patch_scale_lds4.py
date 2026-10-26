#!/usr/bin/env python3
"""Scale-only LDS ring depth 4 (prototype GEMM_MX_SBUF=4).

A/B stay double-buffered; MXSA/MXSB move to a contiguous 4-buffer ring at the
end of LDS so NumLdsBlk=4 for scales alone fits in 320 KiB:

  [A0|B0][A1|B1][S0|S1|S2|S3]
   147456*2      + 8192*4  = 327680

  AB_SIZE     = 147456   (sizeA+sizeB; A/B xor stride)
  SCALE_BASE  = 294912   (MXSA); MXSB = SCALE_BASE+4096
  SCALE_STRIDE= 8192     (MXSA+MXSB per ring slot)
  SCALE_N     = 4

Scale GR/LR swaps become +stride with wrap at ceiling (base+N*stride).
Main-loop wait for scale-issuing waves loosens 3 -> 5 so up to three scale
fills can stay in flight with A/B (FIFO tensorcnt; A/B still recycle every 2).

usage: patch_scale_lds4.py SRC_LOGDIR NAME
"""
import glob, os, re, subprocess, sys, time

BENCH = os.path.dirname(os.path.abspath(__file__))
LLVM = os.environ.get("LLVM_BIN", "/opt/rocm/lib/llvm/bin")

AB_SIZE = 147456
OLD_BLK = 155648
OLD_LDS = 311296
SCALE_BUF = 8192          # MXSA+MXSB bytes per ring slot
SCALE_N = 4
SCALE_SPAN = SCALE_N * SCALE_BUF  # 32768
MXSA0, MXSB0 = 147456, 151552
MXSA_NEW = 2 * AB_SIZE            # 294912
MXSB_NEW = MXSA_NEW + (MXSB0 - MXSA0)  # 299008
NEW_LDS = MXSA_NEW + SCALE_SPAN   # 327680
assert NEW_LDS == 327680

src, name = sys.argv[1], sys.argv[2]
src = src if os.path.isabs(src) else f"{BENCH}/{src}"
asms = glob.glob(f"{src}/**/assembly/*.s", recursive=True) + glob.glob(f"{src}/kernel.s")
inis = glob.glob(f"{src}/**/ClientParameters.ini", recursive=True) + glob.glob(f"{src}/ClientParameters.ini")
asm, ini = asms[0], inis[0]
lines = open(asm).read().split("\n")

ROTATE_GR = [
    "s_add_u32 s[sgprtdmLdsAddr{tc}], s[sgprtdmLdsAddr{tc}], {stride} // patched: scale LDS ring +{stride}",
    "s_sub_u32 s93, s[sgprtdmLdsAddr{tc}], {span} // patched: wrap candidate (addr - {span})",
    "s_cmp_ge_u32 s[sgprtdmLdsAddr{tc}], s[sgprtdmLdsSwapMask{tc}] // patched: past ring ceiling?",
    "s_cselect_b32 s[sgprtdmLdsAddr{tc}], s93, s[sgprtdmLdsAddr{tc}] // patched: wrap scale LDS write",
]


def rotate_gr(tc):
    return [s.format(tc=tc, stride=SCALE_BUF, span=SCALE_SPAN) for s in ROTATE_GR]


def rotate_lr(off_v, ceil_v):
    # v32 is store-only in this kernel; safe scratch in the main loop.
    return [
        f"s_mov_b32 s93, {SCALE_BUF}                              // patched: scale LR ring stride",
        f"v_add_nc_u32 {off_v}, {off_v}, s93                       // patched: scale LR -> next buffer",
        f"s_mov_b32 s93, {SCALE_SPAN}                             // patched: scale LR ring span",
        f"v_sub_nc_u32 v32, {off_v}, s93                           // patched: wrap candidate",
        f"v_cmp_ge_u32 s93, {off_v}, {ceil_v}                      // patched: past ring ceiling?",
        f"v_cndmask_b32 {off_v}, {off_v}, v32, s93                 // patched: wrap scale LR offset",
    ]


out = []
stats = {
    "lds": 0, "m0": 0, "ab_mask": 0, "scale_base": 0, "scale_ceil": 0,
    "ab_lr": 0, "scale_lr_init": 0, "gr_swap": 0, "lr_swap": 0, "wait": 0,
}

i = 0
while i < len(lines):
    l = lines[i]

    # LDS size
    if "group_segment_fixed_size" in l and str(OLD_LDS) in l:
        out.append(l.replace(str(OLD_LDS), str(NEW_LDS)))
        stats["lds"] += 1
        i += 1
        continue
    if l.startswith("s_mov_b32 m0, 0x4c000"):
        out.append(f"s_mov_b32 m0, {hex(NEW_LDS)}                              "
                   f"// patched: LDS clamp at {NEW_LDS} bytes")
        stats["m0"] += 1
        i += 1
        continue

    # A/B GR swap-mask init: stride AB_SIZE (xor still built below)
    m = re.match(
        r"^(s_add_u32 s\[sgprtdmLdsSwapMask([AB])\], s93, )155648(\s+//.*)?$", l)
    if m:
        out.append(f"{m.group(1)}{AB_SIZE}      // patched: addr + AB_SIZE({AB_SIZE})")
        stats["ab_mask"] += 1
        i += 1
        continue

    # Scale GR base
    m = re.match(
        r"^(s_add_u32 s94, s94, )(147456|151552)(\s+//.*subtile LDS offset for (MXSA|MXSB).*)$", l)
    if m:
        tc = m.group(4)
        new = MXSA_NEW if tc == "MXSA" else MXSB_NEW
        out.append(f"{m.group(1)}{new}                         "
                   f"// patched: ldsOffset = woffset + {new} (scale ring base {tc})")
        stats["scale_base"] += 1
        i += 1
        continue

    # After scale base add: keep mov addr, replace xor-mask init with ceiling
    m = re.match(
        r"^s_mov_b32 s\[sgprtdmLdsAddr(MXSA|MXSB)\], s94(.*)$", l)
    if m and out and "patched: ldsOffset" in out[-1]:
        tc = m.group(1)
        out.append(l)
        n1, n2 = lines[i + 1], lines[i + 2]
        assert f"sgprtdmLdsSwapMask{tc}" in n1 and str(OLD_BLK) in n1, n1
        assert "s_xor_b32" in n2 and f"sgprtdmLdsSwapMask{tc}" in n2, n2
        out.append(
            f"s_add_u32 s[sgprtdmLdsSwapMask{tc}], s94, {SCALE_SPAN}   "
            f"// patched: scale ring ceiling = base + {SCALE_N}*{SCALE_BUF}")
        out.append(
            f"// patched: no xor mask for {tc} (rotate uses ceiling)")
        stats["scale_ceil"] += 1
        i += 3
        continue

    # A/B LR span
    if re.match(r"^s_mov_b32 s93, 155648\s+// [AB] LR buffer span", l):
        out.append(f"s_mov_b32 s93, {AB_SIZE}                              "
                   f"// patched: A/B LR buffer span ({AB_SIZE})")
        stats["ab_lr"] += 1
        i += 1
        continue

    # Scale LR bases (hex immediates)
    if "s_mov_b32 s93, 0x24000" in l and "A scale" in l:
        out.append(f"s_mov_b32 s93, {hex(MXSA_NEW)}                             "
                   f"// patched: scale MXSA LDS base {MXSA_NEW}")
        stats["scale_base"] += 1
        i += 1
        continue
    if "s_mov_b32 s93, 0x25000" in l and "B scale" in l:
        out.append(f"s_mov_b32 s93, {hex(MXSB_NEW)}                             "
                   f"// patched: scale MXSB LDS base {MXSB_NEW}")
        stats["scale_base"] += 1
        i += 1
        continue

    # Scale LR swap init: span -> ceiling (no xor)
    m = re.match(
        r"^s_mov_b32 s93, 155648\s+// scale(MXSA|MXSB) LR buffer span", l)
    if m:
        tc = m.group(1)
        off_v = "v33" if tc == "MXSA" else "v36"
        ceil_v = "v34" if tc == "MXSA" else "v37"
        assert lines[i + 1].startswith(f"v_add_nc_u32 {ceil_v}, {off_v}"), lines[i + 1]
        assert lines[i + 2].startswith(f"v_xor_b32 {ceil_v}"), lines[i + 2]
        out.append(f"s_mov_b32 s93, {SCALE_SPAN}                              "
                   f"// patched: scale{tc} LR ring span ({SCALE_SPAN})")
        out.append(f"v_add_nc_u32 {ceil_v}, {off_v}, s93                         "
                   f"// patched: scale{tc} LR ceiling = offset + span")
        out.append(f"// patched: scale{tc} LR uses ceiling (no xor mask)")
        stats["scale_lr_init"] += 1
        i += 3
        continue

    # Scale GR swap (xor -> rotate). Comment line then xor.
    if "TDM: swap MXSA LDS buffer (XOR" in l or "TDM: swap MXSB LDS buffer (XOR" in l:
        tc = "MXSA" if "MXSA" in l else "MXSB"
        out.append(f"/* TDM: rotate {tc} LDS write buffer (scale ring depth {SCALE_N}) */")
        n1 = lines[i + 1]
        assert n1.startswith(f"s_xor_b32 s[sgprtdmLdsAddr{tc}]"), n1
        out.extend(rotate_gr(tc))
        stats["gr_swap"] += 1
        i += 2
        continue

    # Scale LR swap toggles
    if l.strip() == "v_xor_b32 v33, v33, v34                            // MXSA LR buffer toggle":
        out.append("/* Emit code to rotate MXSA LR vgpr offsets (scale ring) */")
        out.extend(rotate_lr("v33", "v34"))
        stats["lr_swap"] += 1
        i += 1
        continue
    if l.strip() == "v_xor_b32 v36, v36, v37                            // MXSB LR buffer toggle":
        out.append("/* Emit code to rotate MXSB LR vgpr offsets (scale ring) */")
        out.extend(rotate_lr("v36", "v37"))
        stats["lr_swap"] += 1
        i += 1
        continue

    # Loosen main-loop wait for scale issuers (keep asymmetric structure)
    if re.match(r"^s_wait_tensorcnt 3\s+// patched: waves 0/1", l):
        out.append("s_wait_tensorcnt 5                                   "
                   "// patched: waves 0/1: A,B + up to 3 scale fills (depth4)")
        stats["wait"] += 1
        i += 1
        continue

    out.append(l)
    i += 1

# Sanity
text = "\n".join(out)
assert stats["lds"] >= 2, stats
assert stats["m0"] == 1, stats
assert stats["ab_mask"] == 2, stats
assert stats["scale_ceil"] == 2, stats
assert stats["scale_lr_init"] == 2, stats
assert stats["ab_lr"] == 2, stats
assert stats["gr_swap"] >= 3, stats
assert stats["lr_swap"] >= 4, stats  # MXSA+MXSB × >=2 sites
assert stats["wait"] == 1, stats
assert str(OLD_LDS) not in text or text.count(str(NEW_LDS)) >= 2
assert "0x24000" not in text and "0x25000" not in text
# Old scale bases should be gone from ldsOffset lines
assert not re.search(r"ldsOffset = woffset \+ 147456", text)
assert not re.search(r"ldsOffset = woffset \+ 151552", text)

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-sclds4-{name}"
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
print(stats, file=sys.stderr)
print(dst)
