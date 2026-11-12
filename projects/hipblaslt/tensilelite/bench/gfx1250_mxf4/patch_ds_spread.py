#!/usr/bin/env python3
"""Hide post-READY LDS-read burst behind K=3 WMMAs + simplify scale LR rotate SALU.

On sclds4, after the READY barrier the next iter's K=0 A/B ds_load_b128s issue as a
solid ~20-load burst before K=3 WMMAs resume -- nothing for the LDS pipe to hide
behind. Separately, scale LR rotate reloads stride/span into s93 every time.

Scale ds_load_b32 must NOT be moved earlier across WMMAs: K=1/2/3 reuse the same
scale VGPRs (v42-45, v54-61), so spreading those loads is a WAR on the in-flight
WMMA scale operands. Only the post-READY ds_load_b128 burst (disjoint high-bank
A/B prefetch regs) is redistributed.

Changes:
  1. After `s_wait_alu depctr_va_vdst(0)` (LR swap settled), take the leading
     consecutive ds_load_b128 run and interleave it 1:1 after the following K=3
     WMMAs (preserving load order, WMMA order, and per-instr s_set_vgpr_msb).
  2. Scale LR rotate: drop the two s_movs; use VALU immediates 8192 / 32768.

usage: patch_ds_spread.py SRC_LOGDIR NAME
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

MSB = re.compile(r"^s_set_vgpr_msb (\d+)\s*(//.*)?$")
LOAD128 = re.compile(r"^ds_load_b128 ")
LOAD32 = re.compile(r"^ds_load_b32 ")
WMMA = re.compile(r"^v_wmma_scale_f32_32x16x128_f4 ")
WAIT_ALU = re.compile(r"^s_wait_alu depctr_va_vdst\(0\)\s*(//.*)?")
LR_ROT = re.compile(r"^s_mov_b32 s93, 8192\s+// patched: scale LR ring stride$")
SCALE_BUF, SCALE_SPAN = 8192, 32768

stats = {"lr_imm": 0, "ready_spread": 0, "loads_moved": 0}


def emit_msb(items):
    """items: (msb, line) -> lines with s_set_vgpr_msb on change."""
    out, last = [], None
    for msb, line in items:
        if msb != last:
            out.append(f"s_set_vgpr_msb {msb}                                   "
                       f"// patched: ds-spread msb")
            last = msb
        out.append(line)
    return out, last


# Pass 1: LR rotate immediates
pass1 = []
i = 0
while i < len(lines):
    l = lines[i]
    if LR_ROT.match(l):
        add, mov2, sub, cmp_, cnd = lines[i + 1:i + 6]
        m_add = re.match(
            r"^v_add_nc_u32 (v\d+), (v\d+), s93\s+// patched: scale LR -> next buffer$", add)
        assert m_add and m_add.group(1) == m_add.group(2), add
        off = m_add.group(1)
        assert re.match(rf"^s_mov_b32 s93, {SCALE_SPAN}\s+// patched: scale LR ring span$", mov2), mov2
        assert re.match(rf"^v_sub_nc_u32 v32, {off}, s93\s+// patched: wrap candidate$", sub), sub
        m_cmp = re.match(
            rf"^v_cmp_ge_u32 s93, {off}, (v\d+)\s+// patched: past ring ceiling\?$", cmp_)
        assert m_cmp, cmp_
        ceil = m_cmp.group(1)
        assert re.match(
            rf"^v_cndmask_b32 {off}, {off}, v32, s93\s+// patched: wrap scale LR offset$", cnd), cnd
        pass1.append(f"v_add_nc_u32 {off}, {off}, {SCALE_BUF}                    "
                     f"// patched: scale LR +{SCALE_BUF} (imm, no s_mov)")
        pass1.append(f"v_sub_nc_u32 v32, {off}, {SCALE_SPAN}                     "
                     f"// patched: wrap candidate (imm)")
        pass1.append(f"v_cmp_ge_u32 s93, {off}, {ceil}                      "
                     f"// patched: past ring ceiling?")
        pass1.append(f"v_cndmask_b32 {off}, {off}, v32, s93                 "
                     f"// patched: wrap scale LR offset")
        stats["lr_imm"] += 1
        i += 6
        continue
    pass1.append(l)
    i += 1

assert stats["lr_imm"] >= 4, stats

# Pass 2: post-READY b128 burst -> interleave with following WMMAs
out = []
in_loop = False
i = 0
n = len(pass1)
current_msb = None
while i < n:
    l = pass1[i]
    # NGLL/tail copy has a similar burst but different live scale/tile
    # registers -- spreading it fails validate. Only touch MAINLOOP_C0.
    if l.startswith("/* MAINLOOP_C0 start */"):
        in_loop = True
    elif l.startswith("/* MAINLOOP_C0 end */"):
        in_loop = False

    out.append(l)
    m = MSB.match(l)
    if m:
        current_msb = int(m.group(1))
    i += 1

    if not (in_loop and WAIT_ALU.match(l) and "LR offset swap" in l):
        continue

    # Collect leading b128 burst (with msb), stopping at first wmma / b32 / foreign.
    j = i
    pending = None
    msb = current_msb
    burst = []  # (msb, line)
    while j < n:
        lj = pass1[j]
        mm = MSB.match(lj)
        if mm:
            pending = int(mm.group(1))
            j += 1
            continue
        if LOAD128.match(lj):
            if pending is not None:
                msb = pending
                pending = None
            assert msb is not None
            burst.append((msb, lj))
            j += 1
            continue
        break
    if len(burst) < 4:
        continue  # not a real burst
    # A trailing s_set_vgpr_msb before the first WMMA is normal (pending carries
    # into the rest scan). Only abort if the burst itself is empty of loads.

    # Collect the following pure wmma / b128 / b32 stream until foreign.
    # We only MOVE the leading b128 burst; b32 scales and already-interleaved
    # b128s stay in relative order with WMMAs.
    k = j
    # keep `pending` / `msb` from the burst scan -- msb before first WMMA
    rest = []  # (msb, line, kind) kind in {W, L128, L32}
    while k < n:
        lk = pass1[k]
        mm = MSB.match(lk)
        if mm:
            pending = int(mm.group(1))
            k += 1
            continue
        if WMMA.match(lk):
            if pending is not None:
                msb = pending
                pending = None
            rest.append((msb, lk, "W"))
            k += 1
            continue
        if LOAD128.match(lk):
            if pending is not None:
                msb = pending
                pending = None
            rest.append((msb, lk, "L128"))
            k += 1
            continue
        if LOAD32.match(lk):
            if pending is not None:
                msb = pending
                pending = None
            rest.append((msb, lk, "L32"))
            k += 1
            continue
        break
    if pending is not None or not any(r[2] == "W" for r in rest):
        continue

    wmmas = [(m, line) for m, line, kind in rest if kind == "W"]
    # Other loads in rest keep their position relative to WMMAs: rebuild as
    # sequence of (optional non-burst loads attached after each WMMA), then
    # distribute the burst loads into those WMMA slots.
    # Simpler: strip rest into wmma-anchored groups, then inject burst evenly.
    groups = []  # each: (wmma_or_None, [trailing non-wmma loads before next wmma])
    # First, any leading non-wmma in rest (shouldn't happen after burst, but)
    idx = 0
    preamble = []
    while idx < len(rest) and rest[idx][2] != "W":
        preamble.append((rest[idx][0], rest[idx][1]))
        idx += 1
    while idx < len(rest):
        assert rest[idx][2] == "W"
        w = (rest[idx][0], rest[idx][1])
        idx += 1
        trail = []
        while idx < len(rest) and rest[idx][2] != "W":
            trail.append((rest[idx][0], rest[idx][1]))
            idx += 1
        groups.append((w, trail))

    n_w = len(groups)
    slots = [[] for _ in range(n_w)]
    for bi, ld in enumerate(burst):
        slots[min(bi * n_w // len(burst), n_w - 1)].append(ld)

    emitted = list(preamble)
    for gi, (w, trail) in enumerate(groups):
        emitted.append(w)
        emitted.extend(slots[gi])
        emitted.extend(trail)

    text_lines, current_msb = emit_msb(emitted)
    # Restore msb that would be in effect at end of original region
    end_msb = rest[-1][0]
    if current_msb != end_msb:
        text_lines.append(f"s_set_vgpr_msb {end_msb}                                   "
                          f"// patched: ds-spread restore msb")
        current_msb = end_msb

    stats["ready_spread"] += 1
    stats["loads_moved"] += len(burst)
    out.extend(text_lines)
    i = k

assert stats["ready_spread"] >= 1, stats

text = "\n".join(out)
if not text.endswith("\n"):
    text += "\n"

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-dsspread-{name}"
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
print(f"lr_imm={stats['lr_imm']} ready_spread={stats['ready_spread']} "
      f"loads_moved={stats['loads_moved']}", file=sys.stderr)
print(dst)
