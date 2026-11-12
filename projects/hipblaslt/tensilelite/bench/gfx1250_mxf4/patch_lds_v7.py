#!/usr/bin/env python3
"""Rewrite LDS to gemm_wt-style A0/A1/B0/B1 segments + single-issuer TDM.

Base: dsspread (sclds4 + ds-spread). Layout from lds_v7_layout.py.

  - A0/A1/B0/B1 in distinct 64 KB segments; buf0/buf1 stacked per half
  - Wave 0 issues A ITERATE x2 (A0 -> A1, 128 rows each)
  - Wave 1 issues B ITERATE x2 (B0 -> B1)
  - MXSA -> wave 2, MXSB -> wave 3 (pf-wave ownership)
  - LR bases retargeted to half bases; tile immediates unchanged
  - Scale rings relocated after data (depth 2 x 8192, same buffer shape)

usage: patch_lds_v7.py SRC_LOGDIR NAME [wait=N|ab=N,sc=N]

READY s_wait_tensorcnt policy (default wait=1):
  wait=N     all waves use N
  ab=N,sc=M  waves 0/1 (A/B) use N; waves 2/3 (scale) use M

wait=0 was correct but ~0.61x (full drain). wait=1 validates clean and
is ~1.01x vs dsspread. wait=2 races on multi-WG / edge sizes.
"""
from __future__ import annotations

import glob, os, re, subprocess, sys, time

from lds_v7_layout import make_layout

BENCH = os.path.dirname(os.path.abspath(__file__))
LLVM = os.environ.get("LLVM_BIN", "/opt/rocm/lib/llvm/bin")

src, name = sys.argv[1], sys.argv[2]
opts = dict(a.split("=", 1) for a in sys.argv[3:] if "=" in a)
src = src if os.path.isabs(src) else f"{BENCH}/{src}"
asms = glob.glob(f"{src}/**/assembly/*.s", recursive=True) + glob.glob(f"{src}/kernel.s")
inis = glob.glob(f"{src}/**/ClientParameters.ini", recursive=True) + glob.glob(f"{src}/ClientParameters.ini")
asm, ini = asms[0], inis[0]
lines = open(asm).read().split("\n")
L = make_layout()
G2 = 100  # existing Group2 scratch for scale ITERATE; also used for A/B ITERATE

# READY wait policy
if "ab" in opts or "sc" in opts:
    WAIT_AB = int(opts.get("ab", "1"))
    WAIT_SC = int(opts.get("sc", "2"))
    WAIT_ALL = None
else:
    WAIT_ALL = int(opts.get("wait", "1"))
    WAIT_AB = WAIT_SC = WAIT_ALL


def repl_imm(text: str, old: int, new: int, why: str) -> tuple[str, int]:
    """Replace decimal or hex immediates equal to old with new (comment tagged)."""
    n = 0
    # hex 0x...
    hx = f"0x{old:x}"
    Hx = f"0x{old:X}"

    def sub_hex(m):
        nonlocal n
        n += 1
        return f"0x{new:x}"

    text2 = re.sub(rf"\b0x{old:x}\b", sub_hex, text, flags=re.I)
    # decimal as standalone token in asm immediates / comments after comma or space
    def sub_dec(m):
        nonlocal n
        n += 1
        return m.group(1) + str(new) + m.group(2)

    text2, nd = re.subn(rf"(?<![\w]){old}(?![\w])", str(new), text2)
    n += nd
    return text2, n


# ---------- pass over lines ----------
out: list[str] = []
stats = {
    "tdm_a_init": 0, "tdm_b_init": 0, "sc_init": 0,
    "lr_a": 0, "lr_b": 0, "lr_span": 0, "sc_lr": 0,
    "sc_wrap": 0, "gate_a": 0, "gate_b": 0,
    "sc_owner": 0, "wait": 0, "ab_swap": 0,
}
i = 0
n = len(lines)
in_main = False
gate_counter = 0

while i < n:
    l = lines[i]

    if "/* MAINLOOP_C0 start */" in l:
        in_main = True
    elif "/* MAINLOOP_C0 end */" in l:
        in_main = False

    # --- A TDM LDS addr init: wave-slice -> A0, swap distance = buf ---
    if re.match(r"^s_mul_i32 s93, s93, 17408\s+// woffset = wId", l):
        # Peek: next add decides A vs B
        nxt = lines[i + 1] if i + 1 < n else ""
        if "ldsOffset = woffset + 0" in nxt or re.search(r"woffset \+ 0 \(subtile LDS offset for A\)", nxt):
            out += [
                f"s_mov_b32 s93, {L.a0}                              "
                f"// patched: LDS A half0 base A0={L.a0}",
                f"s_mov_b32 s[sgprtdmLdsAddrA], s93                  "
                f"// patched: tdmLdsAddrA = A0 (single-issuer)",
                f"s_add_u32 s[sgprtdmLdsSwapMaskA], s93, {L.buf}    "
                f"// patched: A0+buf",
                f"s_xor_b32 s[sgprtdmLdsSwapMaskA], s93, s[sgprtdmLdsSwapMaskA] "
                f"// patched: swapMask = A0 XOR (A0+buf)",
            ]
            # skip original mul/add/mov/add/xor (5 lines typical)
            # consume until xor swapMask line
            while i < n and "swapMask = addr XOR" not in lines[i]:
                i += 1
            stats["tdm_a_init"] += 1
            i += 1
            continue
        if "73728" in nxt or "subtile LDS offset for B" in nxt:
            out += [
                f"s_mov_b32 s93, {L.b0}                              "
                f"// patched: LDS B half0 base B0={L.b0}",
                f"s_mov_b32 s[sgprtdmLdsAddrB], s93                  "
                f"// patched: tdmLdsAddrB = B0 (single-issuer)",
                f"s_add_u32 s[sgprtdmLdsSwapMaskB], s93, {L.buf}    "
                f"// patched: B0+buf",
                f"s_xor_b32 s[sgprtdmLdsSwapMaskB], s93, s[sgprtdmLdsSwapMaskB] "
                f"// patched: swapMask = B0 XOR (B0+buf)",
            ]
            while i < n and "swapMask = addr XOR" not in lines[i]:
                i += 1
            stats["tdm_b_init"] += 1
            i += 1
            continue

    # --- scale TDM LDS bases ---
    if "scale ring base MXSA" in l or re.search(r"ldsOffset = woffset \+ 294912", l):
        out.append(
            f"s_add_u32 s94, s94, {L.sc_mxsa}                         "
            f"// patched: scale ring base MXSA={L.sc_mxsa}")
        stats["sc_init"] += 1
        i += 1
        continue
    if "scale ring base MXSB" in l or re.search(r"ldsOffset = woffset \+ 299008", l):
        out.append(
            f"s_add_u32 s94, s94, {L.sc_mxsb}                         "
            f"// patched: scale ring base MXSB={L.sc_mxsb}")
        stats["sc_init"] += 1
        i += 1
        continue
    if "scale ring ceiling = base + 4*8192" in l:
        which = "MXSA" if "MXSA" in l else "MXSB"
        out.append(
            f"s_add_u32 s[sgprtdmLdsSwapMask{which}], s94, {L.sc_ring}   "
            f"// patched: scale ring ceiling = base + {L.sc_ring} (depth {L.sc_nbuf})")
        i += 1
        continue

    # --- LR A partition: waveIdM * 0x8800 -> select A0/A1 ---
    if re.match(r"^s_mov_b32 s93, 0x8800\s+// LDS bytes per wave for A", l):
        # v38 = waveIdM (0/1). base = A0 + waveIdM * (A1-A0)
        out += [
            f"s_mov_b32 s93, {L.inc_a}                           "
            f"// patched: A1-A0={L.inc_a}",
            f"v_mul_lo_u32 v38, s93, v38                         "
            f"// patched: waveIdM * (A1-A0)",
            f"v_add_nc_u32 v38, {L.a0}, v38                      "
            f"// patched: A LR group base A0/A1",
        ]
        i += 1
        if i < n and "v_mul_lo_u32 v38, s93, v38" in lines[i]:
            i += 1
        stats["lr_a"] += 1
        continue

    if re.match(r"^s_mov_b32 s93, 0x8800\s+// LDS bytes per wave for B", l):
        out += [
            f"s_mov_b32 s93, {L.inc_b}                           "
            f"// patched: B1-B0={L.inc_b}",
            f"v_mul_lo_u32 v38, s93, v38                         "
            f"// patched: waveIdN * (B1-B0)",
            f"v_add_nc_u32 v38, {L.b0}, v38                      "
            f"// patched: B LR group base B0/B1",
        ]
        i += 1
        if i < n and "v_mul_lo_u32 v38, s93, v38" in lines[i]:
            i += 1
        stats["lr_b"] += 1
        continue

    # Drop old B matrix offset 73728 (now folded into B0/B1)
    if re.match(r"^v_add_nc_u32 v(24|26|28|30), 73728, v\1", l):
        out.append(f"// patched: dropped B matrix +73728 (B0/B1 bases)")
        i += 1
        continue

    # LR buffer swap span 147456 -> buf
    if "A/B LR buffer span" in l:
        out.append(
            f"s_mov_b32 s93, {L.buf}                              "
            f"// patched: A/B LR buffer span ({L.buf})")
        stats["lr_span"] += 1
        i += 1
        continue

    # Scale LR bases + ring span
    if "scale MXSA LDS base" in l:
        out.append(
            f"s_mov_b32 s93, {L.sc_mxsa}                             "
            f"// patched: scale MXSA LDS base {L.sc_mxsa}")
        stats["sc_lr"] += 1
        i += 1
        continue
    if "scale MXSB LDS base" in l:
        out.append(
            f"s_mov_b32 s93, {L.sc_mxsb}                             "
            f"// patched: scale MXSB LDS base {L.sc_mxsb}")
        stats["sc_lr"] += 1
        i += 1
        continue
    if "scaleMXSA LR ring span" in l:
        out.append(
            f"s_mov_b32 s93, {L.sc_ring}                              "
            f"// patched: scaleMXSA LR ring span ({L.sc_ring})")
        i += 1
        continue
    if "scaleMXSB LR ring span" in l:
        out.append(
            f"s_mov_b32 s93, {L.sc_ring}                              "
            f"// patched: scaleMXSB LR ring span ({L.sc_ring})")
        i += 1
        continue

    # Scale rotate wrap distance
    if re.search(r"s_sub_u32 s93, s\[sgprtdmLdsAddrMXS[AB]\], 32768", l):
        out.append(l.replace("32768", str(L.sc_ring)).replace(
            "patched: wrap candidate (addr - 32768)",
            f"patched: wrap candidate (addr - {L.sc_ring})"))
        stats["sc_wrap"] += 1
        i += 1
        continue
    if re.search(r"v_sub_nc_u32 v32, v3[36], 32768", l):
        out.append(l.replace("32768", str(L.sc_ring)))
        stats["sc_wrap"] += 1
        i += 1
        continue

    # Scale iterate owners 0/1 -> 2/3
    if re.search(r"s_cmp_eq_u32 s\[sgprWaveIdx\], 0\s+// patched: MXSA iterate issuer = wave 0", l):
        out.append(
            "s_cmp_eq_u32 s[sgprWaveIdx], 2          "
            "// patched: MXSA iterate issuer = wave 2")
        stats["sc_owner"] += 1
        i += 1
        continue
    if re.search(r"s_cmp_eq_u32 s\[sgprWaveIdx\], 1\s+// patched: MXSB iterate issuer = wave 1", l):
        out.append(
            "s_cmp_eq_u32 s[sgprWaveIdx], 3          "
            "// patched: MXSB iterate issuer = wave 3")
        stats["sc_owner"] += 1
        i += 1
        continue

    # Gate A/B tensor_load in MAINLOOP + matching preloop-after-setup sites:
    # rewrite the descriptor clamp to 128 rows and add ITERATE, gate by wave.
    # Mainloop has bare loads after barrier; preloop has full rebuild then load.
    if (in_main or True) and l.startswith(
            "tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], "
            "s[sgprtdmAGroup1:sgprtdmAGroup1+7] // TDM: global->LDS for A"):
        lab = f".L_ldsv7_A_{gate_counter}"
        gate_counter += 1
        out += [
            f"s_cmp_eq_u32 s[sgprWaveIdx], 0                      "
            f"// patched: A issuer = wave 0",
            f"s_cbranch_scc0 {lab}",
            # Point LDS at current buf of A0 (sgprtdmLdsAddrA tracks buf toggle)
            f"s_mov_b32 s[sgprtdmAGroup0+1], s[sgprtdmLdsAddrA]   "
            f"// patched: A LDS = A0/buf",
            # 128 rows per iterate half
            f"s_mov_b32 s[sgprtdmAGroup1+4], 128                  "
            f"// patched: A tile1 = 128 rows (half)",
            f"s_or_b32 s[sgprtdmAGroup1], s[sgprtdmAGroup1], 0x80000 "
            f"// patched: A iterate_enable",
            f"s_mov_b32 s{G2}, 1                                 "
            f"// patched: A Group2 dim2=1",
            f"s_mov_b32 s{G2+1}, {L.inc_a}                       "
            f"// patched: A lds_inc = A1-A0",
            # fp4 bpe=1/2: 128 rows * byte_stride = 128 * (StrideA0I/2) = 64*StrideA0I
            f"s_mul_i32 s{G2+2}, s[sgprStrideA0I], 64             "
            f"// patched: A global_inc = 128 rows * bpe/2",
            f"s_mov_b32 s{G2+3}, 0x10000                         "
            f"// patched: A iterate_count=1",
            f"tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], "
            f"s[sgprtdmAGroup1:sgprtdmAGroup1+7], "
            f"s[{G2}:{G2+3}], null // patched: A ITERATE A0->A1 wave0",
            f"{lab}:",
        ]
        stats["gate_a"] += 1
        i += 1
        continue

    if l.startswith(
            "tensor_load_to_lds s[sgprtdmBGroup0:sgprtdmBGroup0+3], "
            "s[sgprtdmBGroup1:sgprtdmBGroup1+7] // TDM: global->LDS for B"):
        lab = f".L_ldsv7_B_{gate_counter}"
        gate_counter += 1
        out += [
            f"s_cmp_eq_u32 s[sgprWaveIdx], 1                      "
            f"// patched: B issuer = wave 1",
            f"s_cbranch_scc0 {lab}",
            # Rebase wave1's B addr back to N-tile base (wave1 starts +64 rows)
            f"s_mul_i32 s{G2}, s[sgprStrideB1J], 32               "
            f"// patched: 64 rows * bpe0.5 = 32*stride elems",
            f"s_sub_u32 s[sgprtdmBGroup0+2], s[sgprtdmBGroup0+2], s{G2}",
            f"s_subb_u32 s[sgprtdmBGroup0+3], s[sgprtdmBGroup0+3], 0",
            f"s_mov_b32 s[sgprtdmBGroup0+1], s[sgprtdmLdsAddrB]   "
            f"// patched: B LDS = B0/buf",
            f"s_mov_b32 s[sgprtdmBGroup1+4], 128                  "
            f"// patched: B tile1 = 128 rows",
            f"s_or_b32 s[sgprtdmBGroup1], s[sgprtdmBGroup1], 0x80000 "
            f"// patched: B iterate_enable",
            f"s_mov_b32 s{G2}, 1",
            f"s_mov_b32 s{G2+1}, {L.inc_b}",
            f"s_mul_i32 s{G2+2}, s[sgprStrideB1J], 64             "
            f"// patched: B global_inc = 128 rows * bpe/2",
            f"s_mov_b32 s{G2+3}, 0x10000",
            f"tensor_load_to_lds s[sgprtdmBGroup0:sgprtdmBGroup0+3], "
            f"s[sgprtdmBGroup1:sgprtdmBGroup1+7], "
            f"s[{G2}:{G2+3}], null // patched: B ITERATE B0->B1 wave1",
            f"{lab}:",
        ]
        stats["gate_b"] += 1
        i += 1
        continue

    # wait_tensorcnt split for new ownership
    if re.search(r"s_cmp_ge_u32 s\[sgprWaveIdx\], 2\s+// patched: waves 2/3 issue only A,B", l) or \
       re.search(r"s_cmp_ge_u32 s\[sgprWaveIdx\], 2\s+// patched: waves 2/3 issue A,B", l):
        # Replace whole wait block
        # Expect: cmp; cbranch; wait5; branch; lab; wait2; labdone
        block = [l]
        j = i + 1
        while j < n and not lines[j].startswith(".L_sciter_waitdone"):
            block.append(lines[j])
            j += 1
        if j < n:
            block.append(lines[j])  # waitdone label
        # Find labels
        lab2 = next(x.rstrip(":") for x in block if x.startswith(".L_sciter_wait2"))
        labd = next(x.rstrip(":") for x in block if x.startswith(".L_sciter_waitdone"))
        if WAIT_ALL is not None:
            out += [
                f"s_wait_tensorcnt {WAIT_ALL}                                   "
                f"// patched: lds_v7 READY wait={WAIT_ALL}",
            ]
        else:
            out += [
                f"s_cmp_le_u32 s[sgprWaveIdx], 1                      "
                f"// patched: waves 0/1 A/B vs 2/3 scale",
                f"s_cbranch_scc0 .L_ldsv7_waitSC_{stats['wait']}",
                f"s_wait_tensorcnt {WAIT_AB}                                   "
                f"// patched: A/B issuer READY wait={WAIT_AB}",
                f"s_branch .L_ldsv7_waitdone_{stats['wait']}",
                f".L_ldsv7_waitSC_{stats['wait']}:",
                f"s_wait_tensorcnt {WAIT_SC}                                   "
                f"// patched: scale issuer READY wait={WAIT_SC}",
                f".L_ldsv7_waitdone_{stats['wait']}:",
            ]
        stats["wait"] += 1
        i = j + 1
        continue

    # A/B LDS XOR swap in mainloop still uses sgprtdmLdsSwapMask — already
    # rebuilt at init. Scale +8192 / wrap uses sc_ring — handled above.

    out.append(l)
    i += 1

text = "\n".join(out)

# Fix A global_inc: use element stride * 128 * bpe.
# Re-read: for fp4 packed, global row bytes = SizeK/2 for K-major A? A is Alik
# K-major: row i at i*LDA. LDA ~ K. Byte address += 128 * LDA * (4/8) = 64 * LDA.
# Keep s_mul stride, 64 as emitted. Verify comment only.

print("layout:", L, file=sys.stderr)
print("stats:", stats, file=sys.stderr)
assert stats["tdm_a_init"] >= 1 and stats["tdm_b_init"] >= 1, stats
assert stats["lr_a"] >= 1 and stats["lr_b"] >= 1, stats
assert stats["gate_a"] >= 1 and stats["gate_b"] >= 1, stats
assert stats["sc_owner"] == 8, stats  # 4 MXSA + 4 MXSB sites
assert stats["wait"] >= 1, stats

print(f"wait policy: all={WAIT_ALL} ab={WAIT_AB} sc={WAIT_SC}", file=sys.stderr)
dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-ldsv7-{name}"
os.makedirs(dst)
s = f"{dst}/kernel.s"
open(s, "w").write(text)
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
open(f"{dst}/layout.txt", "w").write(str(L) + "\n" + str(stats) + "\n")
print(dst)
