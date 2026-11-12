#!/usr/bin/env python3
"""Group same-bank ds_load_b128s so one s_set_vgpr_msb covers up to G loads.

In the main loop every WMMA/ds_load alternation costs two s_set_vgpr_msb (into the
ds_load bank and back to the WMMA bank).  This pass keeps the 1:1-ish interleave but
pulls a later b128 load with the *same* effective msb back so it directly follows an
earlier one (up to G in a row), then re-derives the msb sets.

Only ds_load_b128 are moved (scale ds_load_b32 keep their position: WAR on live WMMA
scale VGPRs).  A load only moves across WMMAs and other b128 loads inside a "pure"
window (only s_set_vgpr_msb / ds_load_b128 / ds_load_b32 / v_wmma_scale lines), never
across a b32, a barrier, a wait, or any foreign line.  b128 destinations are the
disjoint prefetch banks, so reordering them among themselves / past WMMAs is safe (same
argument as patch_msb_reorder.py).  The msb in effect at the end of every window is
restored, so code after the window sees the original state.

usage: patch_msb_group.py SRC_LOGDIR NAME [--group G] [--max-move N]
Writes logs/<stamp>-msbgroup-NAME/ (kernel.s/.o/.co + ClientParameters.ini).
"""
import argparse, glob, os, re, subprocess, sys, time

BENCH = os.path.dirname(os.path.abspath(__file__))
LLVM = os.environ.get("LLVM_BIN", "/opt/rocm/lib/llvm/bin")

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("name")
ap.add_argument("--group", type=int, default=2, help="max consecutive same-msb b128 loads")
ap.add_argument("--max-move", type=int, default=4,
                help="max WMMAs/loads a load may be pulled back across")
args = ap.parse_args()

src = args.src if os.path.isabs(args.src) else f"{BENCH}/{args.src}"
asm = glob.glob(f"{src}/**/assembly/*.s", recursive=True)[0]
ini = glob.glob(f"{src}/**/ClientParameters.ini", recursive=True)[0]
lines = open(asm).read().split("\n")

MSB = re.compile(r"^s_set_vgpr_msb (\d+)\s*(//.*)?$")
L128 = re.compile(r"^ds_load_b128 ")
L32 = re.compile(r"^ds_load_b32 ")
WMMA = re.compile(r"^v_wmma_scale_f32_32x16x128_f4 ")


def msb_line(v, why):
    return f"s_set_vgpr_msb {v}                                   // {why}"


out = []
n = len(lines)
i = 0
in_loop = False
cur_msb = None  # msb in effect as seen in the original stream
stats = dict(windows=0, sets_before=0, sets_after=0, moved=0)

while i < n:
    l = lines[i]
    if l.startswith("label_LoopBeginL:"):
        in_loop = True
    elif l.startswith("label_ExitMainloopBufEven:"):
        in_loop = False

    m = MSB.match(l.strip())
    if not in_loop or not (m or L128.match(l) or L32.match(l) or WMMA.match(l)):
        out.append(l)
        if m:
            cur_msb = int(m.group(1))
        i += 1
        continue

    # collect a pure window starting here
    j = i
    pending = None
    msb = cur_msb
    recs = []  # [effective msb, line, kind]
    n_sets_before = 0
    while j < n:
        lj = lines[j]
        mm = MSB.match(lj.strip())
        if mm:
            pending = int(mm.group(1))
            n_sets_before += 1
            j += 1
            continue
        kind = ("L128" if L128.match(lj) else "L32" if L32.match(lj)
                else "W" if WMMA.match(lj) else None)
        if kind is None:
            break
        if pending is not None:
            msb = pending
            pending = None
        if msb is None:
            break
        recs.append([msb, lj, kind])
        j += 1
    if pending is not None or not recs:
        # not a pure window (msb set directly before a foreign line): leave as is
        j = max(j, i + 1)
        for k in range(i, j):
            out.append(lines[k])
            mk = MSB.match(lines[k].strip())
            if mk:
                cur_msb = int(mk.group(1))
        i = j
        continue

    end_msb = recs[-1][0]

    # pull same-msb b128 loads back to follow the previous load of that msb
    order = list(range(len(recs)))
    pos = {k: k for k in order}
    seq = list(recs)
    k = 0
    while k < len(seq):
        if seq[k][2] == "L128":
            # extend a run of same-msb loads after seq[k]
            run = 1
            t = k + 1
            while run < args.group and t < len(seq):
                # skip over WMMAs / differing-msb b128 loads, stop at b32
                u = t
                found = None
                span = 0
                while u < len(seq) and span <= args.max_move:
                    if seq[u][2] == "L32":
                        break
                    if seq[u][2] == "L128" and seq[u][0] == seq[k][0]:
                        found = u
                        break
                    span += 1
                    u += 1
                if found is None:
                    break
                if found != t:
                    seq.insert(t, seq.pop(found))
                    stats["moved"] += 1
                run += 1
                t += 1
            k = t if run > 1 else k + 1
        else:
            k += 1

    # re-emit with msb only on change
    last = cur_msb
    emitted = []
    for e_msb, e_line, _ in seq:
        if e_msb != last:
            emitted.append(msb_line(e_msb, "patched: grouped"))
            stats["sets_after"] += 1
            last = e_msb
        emitted.append(e_line)
    stats["windows"] += 1
    stats["sets_before"] += n_sets_before
    out.extend(emitted)
    cur_msb = last  # == msb of last emitted instr
    if cur_msb != end_msb:
        # original state at window end differs (reordered tail): restore it
        out.append(msb_line(end_msb, "patched: restore"))
        stats["sets_after"] += 1
        cur_msb = end_msb
    i = j

print(f"windows={stats['windows']} sets {stats['sets_before']} -> {stats['sets_after']} "
      f"(loads moved {stats['moved']})  [group={args.group}, max_move={args.max_move}]",
      file=sys.stderr)

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-msbgroup-{args.name}"
os.makedirs(dst)
s = f"{dst}/kernel.s"
open(s, "w").write("\n".join(out))
subprocess.run([f"{LLVM}/clang", "-x", "assembler", "--target=amdgcn-amd-amdhsa",
                "-mcode-object-version=4", "-c", "-Xclangas", "-target-feature",
                "-Xclangas", "+real-true16", "-mcpu=gfx1250", "-mno-wavefrontsize64",
                s, "-o", f"{dst}/kernel.o"], check=True, stderr=subprocess.DEVNULL)
subprocess.run([f"{LLVM}/clang", "--target=amdgcn-amd-amdhsa", "-Xlinker", "--build-id=sha1",
                f"{dst}/kernel.o", "-o", f"{dst}/kernel.co"], check=True)
cfg = re.sub(r"^code-object=.*$", f"code-object={dst}/kernel.co", open(ini).read(), flags=re.M)
open(f"{dst}/ClientParameters.ini", "w").write(cfg)
print(dst)
