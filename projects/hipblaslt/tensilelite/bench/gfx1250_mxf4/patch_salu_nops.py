#!/usr/bin/env python3
"""Drop dead MXSA wave-0 iterate addr adjust + hoist Group2 invariants.

MXSA iterate issuer is hardcoded to wave 0, so the pre-issue
  s_lshl/s_mul/s_sub (global) + s_mul/s_sub (LDS)
rewind by WaveIdx is always a no-op on the taken path. Delete it in MAINLOOP.
MXSB issuer is wave 1 — rewind stays.

Also hoist s101/s103 (true Group2 invariants) above each MXSA wave-gate;
keep s_mov s100,1 and s102 in-gate after any addr adjust (s100 is scratch).

usage: patch_salu_nops.py SRC_LOGDIR NAME
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

stats = {"mxsa_adj_drop": 0, "g2_hoist": 0}

# --- Pass 1: drop MXSA wave-0 addr adjust in MAINLOOP ---
out = []
i = 0
n = len(lines)
in_main = False
while i < n:
    l = lines[i]
    if "MAINLOOP_C0 start" in l:
        in_main = True
    if "MAINLOOP_C0 end" in l:
        in_main = False

    # Match the 6-op adjust block after MXSA wave-0 gate, before Group2 movs
    if (in_main and
            l.startswith("s_lshl_b32 s100, s[sgprSizeI], 2") and
            "Size*4" in l and
            i >= 2 and "MXSA iterate issuer = wave 0" in lines[i - 2]):
        # Expect: lshl, mul WaveIdx, sub global, subb, mul 128, sub LDS
        block = lines[i:i + 6]
        assert "s_mul_i32 s100, s100, s[sgprWaveIdx]" in block[1], block[1]
        assert "global addr back to K group 0" in block[2], block[2]
        assert "subb_u32" in block[3], block[3]
        assert "kGroup * chunk" in block[4], block[4]
        assert "LDS back to K group 0" in block[5], block[5]
        out.append("// patched: MXSA wave0 addr adjust elided (WaveIdx==0)")
        stats["mxsa_adj_drop"] += 1
        i += 6
        continue

    out.append(l)
    i += 1

assert stats["mxsa_adj_drop"] >= 1, stats
lines = out

# --- Pass 2: g2 hoist s101/s103 (same as patch_salu_slim pass 2) ---
out = []
i = 0
n = len(lines)
in_main = False
while i < n:
    l = lines[i]
    if "MAINLOOP_C0 start" in l:
        in_main = True
    if "MAINLOOP_C0 end" in l:
        in_main = False

    if (in_main and l.strip().startswith("/* Scale GR: MXSA") and
            i + 2 < n and "sgprWaveIdx], 0" in lines[i + 1]):
        j = i + 1
        while j < n and not (lines[j].startswith("tensor_load_to_lds") and "MXSA" in lines[j]):
            j += 1
        assert j < n
        g2 = []
        k = j - 1
        while k > i and ("patched: Group2" in lines[k] or lines[k].startswith("s_mov_b32 s10") or
                         lines[k].startswith("s_lshl_b32 s102")):
            g2.insert(0, lines[k])
            k -= 1
        assert len(g2) == 4, g2
        sb_cmt = next(t for t in range(j + 1, n)
                      if lines[t].strip().startswith("/* Scale GR: MXSB"))
        sb_ld = next(t for t in range(sb_cmt, n)
                     if lines[t].startswith("tensor_load_to_lds") and "MXSB" in lines[t])
        g2b = []
        k = sb_ld - 1
        while k > sb_cmt and ("patched: Group2" in lines[k] or lines[k].startswith("s_mov_b32 s10") or
                              lines[k].startswith("s_lshl_b32 s102")):
            g2b.insert(0, lines[k])
            k -= 1
        assert len(g2b) == 4, g2b

        out.append(l)
        out.append("s_mov_b32 s101, 128                           "
                   "// patched: Group2+1 lds_inc = 128 (hoisted)")
        out.append("s_mov_b32 s103, 0x30000                       "
                   "// patched: Group2+3 iter_count (hoisted)")
        drop_a = {x for x in g2 if x.startswith("s_mov_b32 s101") or
                  x.startswith("s_mov_b32 s103")}
        drop_b = {x for x in g2b if x.startswith("s_mov_b32 s101") or
                  x.startswith("s_mov_b32 s103")}
        for t in range(i + 1, j + 1):
            if lines[t] in drop_a:
                continue
            out.append(lines[t])
        for t in range(j + 1, sb_cmt + 1):
            out.append(lines[t])
        for t in range(sb_cmt + 1, sb_ld + 1):
            if lines[t] in drop_b:
                continue
            out.append(lines[t])
        stats["g2_hoist"] += 1
        i = sb_ld + 1
        continue

    out.append(l)
    i += 1

assert stats["g2_hoist"] >= 1, stats

text = "\n".join(out)
if not text.endswith("\n"):
    text += "\n"

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-salunops-{name}"
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
print(f"mxsa_adj_drop={stats['mxsa_adj_drop']} g2_hoist={stats['g2_hoist']}", file=sys.stderr)
print(dst)
