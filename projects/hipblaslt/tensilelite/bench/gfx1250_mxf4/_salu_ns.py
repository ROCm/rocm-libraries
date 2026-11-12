#!/usr/bin/env python3
"""Slim + redistribute main-loop TDM descriptor SALU (post-dsspread).

Cannot unalias MXSA/MXSB from A/B (needs +24 SGPRs; file at 104 / cap ~105).
Instead:

  1. Bake iterate_enable into scale Group1 control imm (drop per-iter s_or 0x80000).
  2. Hoist invariant Group2 words (tensor_dim2=1, lds_inc=128, iter_count) above
     the wave-gated scale issue so MXSA/MXSB only recompute global_inc (Size*4).
  3. Interleave the pre-FREE A/B/MXS addr+rebuild SALU blob into the preceding
     K=2 WMMA stream (descriptor SGPRs are not WMMA operands), so that window
     is not a solid SALU stall before the barrier.

usage: patch_salu_slim.py SRC_LOGDIR NAME
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

G1_WITH_ITER = 0xbf100000 | 0x80000  # 0xbf180000
G1_CTRL = re.compile(
    r"^s_mov_b32 s\[sgprtdmMXS(A|B)Group1\], 0xbf100000\s+// TDM Group1 control$")
ITER_OR = re.compile(
    r"^s_or_b32 s\[sgprtdmMXS(A|B)Group1\], s\[sgprtdmMXS\1Group1\], 0x80000 "
    r"// patched: set iterate_enable \(keep pad\)$")
HOIST_MARK = "// __no_spread__"
MSB = re.compile(r"^s_set_vgpr_msb (\d+)")
WMMA = re.compile(r"^v_wmma_scale_f32_32x16x128_f4 ")
LOAD = re.compile(r"^ds_load_b(128|32) ")

def glue_salu(blob):
    """Split pre-FREE blob into coarse atomic chunks.

    Fine-grained spreading is unsafe: A/B Group1 packing and scale-ring wrap
    both stage values in s93, so those sequences must stay contiguous.
    Chunks (in order): A-addr, A-swap+rebuild, B-addr, B-swap+rebuild,
    MXSA-addr+rotate, MXSB-addr+rotate.
    """
    def find(pred, start=0):
        for i in range(start, len(blob)):
            if pred(blob[i]):
                return i
        return None

    a_addr = find(lambda l: "TDM addr update: A" in l or
                  l.startswith("s_add_u64 s[sgprAddressA"))
    # Prefer comment if present; else the add itself
    a_swap = find(lambda l: "swap A LDS" in l or
                  l.startswith("s_xor_b32 s[sgprtdmLdsAddrA]"), a_addr or 0)
    b_addr = find(lambda l: "TDM addr update: B" in l or
                  l.startswith("s_add_u64 s[sgprAddressB"), a_swap or 0)
    b_swap = find(lambda l: "swap B LDS" in l or
                  l.startswith("s_xor_b32 s[sgprtdmLdsAddrB]"), b_addr or 0)
    sa = find(lambda l: "addr update: MXSA" in l or
              (l.startswith("s_lshl_b32 s93, s[sgprSizeI]") and "scale step" in l),
              b_swap or 0)
    sb = find(lambda l: "addr update: MXSB" in l or
              (l.startswith("s_lshl_b32 s93, s[sgprSizeJ]") and "scale step" in l),
              sa or 0)
    cuts = [i for i in (a_addr, a_swap, b_addr, b_swap, sa, sb) if i is not None]
    # Always start at 0
    if not cuts or cuts[0] != 0:
        cuts = [0] + cuts
    # Dedup sorted
    cuts = sorted(set(cuts))
    chunks = []
    for i, c in enumerate(cuts):
        end = cuts[i + 1] if i + 1 < len(cuts) else len(blob)
        chunk = [l for l in blob[c:end]
                 if not (l.strip().startswith("/*") or l.strip().startswith("//"))]
        if chunk:
            chunks.append(chunk)
    return chunks


stats = {"iter_bake": 0, "g2_hoist": 0, "salu_spread": 0, "salu_chunks": 0}

# --- Pass 1: iterate bake ---
p1 = []
for l in lines:
    m = G1_CTRL.match(l)
    if m:
        p1.append(f"s_mov_b32 s[sgprtdmMXS{m.group(1)}Group1], {hex(G1_WITH_ITER)}         "
                  f"// patched: Group1 control | iterate_enable")
        stats["iter_bake"] += 1
        continue
    if ITER_OR.match(l):
        continue
    p1.append(l)
assert stats["iter_bake"] >= 4, stats

# --- Pass 2: hoist Group2 invariants above each MXSA wave-gate in MAINLOOP ---
# Pattern:
#   /* Scale GR: MXSA ... */
#   s_cmp_eq_u32 s[sgprWaveIdx], 0
#   s_cbranch_scc0 .L_sciter_MXSA_N
#   ... s_mov s100,1 / s101,128 / s_lshl s102 / s_mov s103,0x30000 / tensor_load
#   .L_sciter_MXSA_N:
#   ... MXSB rebuild ...
#   /* Scale GR: MXSB ... */
#   s_cmp_eq ... 1
#   ... same Group2 with SizeJ ...
#
# Rewrite: emit s100/s101/s103 once before MXSA cmp; inside each gate only s102 + load.
p2 = []
i = 0
n = len(p1)
in_main = False
while i < n:
    l = p1[i]
    if "MAINLOOP_C0 start" in l:
        in_main = True
    if "MAINLOOP_C0 end" in l:
        in_main = False

    if (in_main and l.strip().startswith("/* Scale GR: MXSA") and
            i + 2 < n and "sgprWaveIdx], 0" in p1[i + 1]):
        # Find MXSA tensor_load and its Group2 movs
        j = i + 1
        while j < n and not (p1[j].startswith("tensor_load_to_lds") and "MXSA" in p1[j]):
            j += 1
        assert j < n, "MXSA load"
        # Collect Group2 lines just above the load
        g2 = []
        k = j - 1
        while k > i and ("patched: Group2" in p1[k] or p1[k].startswith("s_mov_b32 s10") or
                         p1[k].startswith("s_lshl_b32 s102")):
            g2.insert(0, p1[k])
            k -= 1
        # Expect 4 Group2 lines
        assert len(g2) == 4, g2
        # Find MXSB gate + load similarly
        sb_cmt = next(t for t in range(j + 1, n) if p1[t].strip().startswith("/* Scale GR: MXSB"))
        sb_ld = next(t for t in range(sb_cmt, n)
                     if p1[t].startswith("tensor_load_to_lds") and "MXSB" in p1[t])
        g2b = []
        k = sb_ld - 1
        while k > sb_cmt and ("patched: Group2" in p1[k] or p1[k].startswith("s_mov_b32 s10") or
                              p1[k].startswith("s_lshl_b32 s102")):
            g2b.insert(0, p1[k])
            k -= 1
        assert len(g2b) == 4, g2b

        # Emit: Scale GR MXSA comment, then invariant Group2, then from wave cmp through
        # MXSA load but with only s102 from g2, then through MXSB with only s102 from g2b.
        p2.append(l)  # MXSA Scale GR comment
        p2.append("s_mov_b32 s100, 1                               "
                  "// patched: Group2+0 tensor_dim2 = 1 (hoisted)")
        p2.append("s_mov_b32 s101, 128                           "
                  "// patched: Group2+1 lds_inc = 128 (hoisted)")
        p2.append("s_mov_b32 s103, 0x30000                       "
                  "// patched: Group2+3 iter_count (hoisted)")
        # MXSA path: skip comment (done), emit i+1..j inclusive but replace Group2 block
        # with only the s102 line
        s102_a = next(x for x in g2 if x.startswith("s_lshl_b32 s102"))
        g2_set = set(g2)
        emitted_s102 = False
        for t in range(i + 1, j + 1):
            if p1[t] in g2_set:
                if not emitted_s102:
                    p2.append(s102_a)
                    emitted_s102 = True
                continue
            p2.append(p1[t])
        # Between MXSA load and MXSB comment: keep as-is (MXSB rebuild)
        for t in range(j + 1, sb_cmt + 1):
            p2.append(p1[t])
        # MXSB path with only s102
        s102_b = next(x for x in g2b if x.startswith("s_lshl_b32 s102"))
        g2b_set = set(g2b)
        emitted_s102 = False
        for t in range(sb_cmt + 1, sb_ld + 1):
            if p1[t] in g2b_set:
                if not emitted_s102:
                    p2.append(s102_b)
                    emitted_s102 = True
                continue
            p2.append(p1[t])
        stats["g2_hoist"] += 1
        i = sb_ld + 1
        continue

    p2.append(l)
    i += 1

assert stats["g2_hoist"] >= 1, stats

# --- Pass 3: spread pre-FREE SALU blob into preceding WMMA stream (MAINLOOP only) ---
out = []
i = 0
n = len(p2)
in_main = False
while i < n:
    l = p2[i]
    if "MAINLOOP_C0 start" in l:
        in_main = True
    if "MAINLOOP_C0 end" in l:
        in_main = False

    if in_main and l.strip() == HOIST_MARK:
        # SALU blob: mark .. s_wait_dscnt before FREE barrier
        blob_start = i + 1
        wait = next(t for t in range(blob_start, n)
                    if p2[t].startswith("s_wait_dscnt 0") and t + 1 < n and
                    p2[t + 1].startswith("s_barrier_signal"))
        blob = p2[blob_start:wait]
        # Drop pure comment-only lines from chunks but keep them attached to next
        raw = []
        for x in blob:
            if x.strip().startswith("/*") or x.strip().startswith("//"):
                continue  # drop narrative comments in the spread blob
            raw.append(x)
        chunks = glue_salu(raw)
        stats["salu_chunks"] = len(chunks)

        # Walk backward from mark to collect the preceding wmma/ds/msb window
        # (stop at wait_dscnt / barrier / tensor_load / foreign salu that's not msb)
        pre = []  # (kind, msb, line) in forward order
        msb = None
        # Find msb state at mark by scanning forward from last known... scan back for items
        items_rev = []
        t = i - 1
        while t >= 0:
            lt = p2[t]
            if MSB.match(lt):
                items_rev.append(("msb", int(MSB.match(lt).group(1)), lt))
                t -= 1
                continue
            if WMMA.match(lt) or LOAD.match(lt):
                items_rev.append(("op", None, lt))
                t -= 1
                continue
            if not lt.strip() or lt.strip().startswith("/*") or lt.strip().startswith("//"):
                t -= 1
                continue
            break  # foreign
        items = list(reversed(items_rev))
        # Resolve effective msb per op
        cur = None
        ops = []  # (msb, line, is_wmma)
        for kind, val, line in items:
            if kind == "msb":
                cur = val
                continue
            assert cur is not None
            ops.append((cur, line, line.startswith("v_wmma")))

        wmma_idx = [k for k, o in enumerate(ops) if o[2]]
        if len(wmma_idx) < 4 or len(chunks) < 4:
            # Not enough cover — leave blob in place
            out.append(l)
            i += 1
            continue

        # Remove the preceding window from out (we'll re-emit). out currently ends
        # just before mark; strip the scanned items.
        # Count how many lines of p2[t+1:i] were window content.
        win_start = t + 1
        # out was built up to i (exclusive of mark). Truncate back to win_start lines
        # relative to how we appended — easier: rebuild out without last (i - win_start) lines
        n_back = i - win_start
        assert n_back > 0
        out = out[:-n_back]

        # Assign chunks evenly after WMMAs
        n_w = len(wmma_idx)
        slots = [[] for _ in range(n_w)]
        for ci, ch in enumerate(chunks):
            slots[min(ci * n_w // len(chunks), n_w - 1)].extend(ch)

        # Re-emit ops with msb, injecting SALU after each WMMA
        last_msb = None
        wi = 0
        for msb_v, line, is_w in ops:
            if msb_v != last_msb:
                out.append(f"s_set_vgpr_msb {msb_v}                                   "
                           f"// patched: salu-slim msb")
                last_msb = msb_v
            out.append(line)
            if is_w:
                out.extend(slots[wi])
                wi += 1

        out.append("// patched: pre-FREE TDM SALU interleaved into prior WMMAs")
        # Then wait_dscnt + rest
        out.append(p2[wait])
        stats["salu_spread"] += 1
        i = wait + 1
        continue

    out.append(l)
    i += 1

import sys; print("spread-disabled",stats,file=sys.stderr)

text = "\n".join(out)
if not text.endswith("\n"):
    text += "\n"

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-saluslim-{name}"
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
print(f"iter_bake={stats['iter_bake']} g2_hoist={stats['g2_hoist']} "
      f"salu_spread={stats['salu_spread']} chunks={stats['salu_chunks']}", file=sys.stderr)
print(dst)
