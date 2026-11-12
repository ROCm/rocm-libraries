#!/usr/bin/env python3
"""Issue MXSA/MXSB before A/B in the main-loop burst (MXSB-idle lever #3a / #2 proxy).

True A↔MXSA / B↔MXSB unalias needs +24 general SGPRs; gfx1250 caps at s105 and
this kernel already uses ~98. With aliasing kept, the workable "issue MXSB
earlier" probe reorders the post-FREE burst from A,B,SA,SB to SA,SB,A,B:

  - Scale descriptor rebuilds still overwrite A/B SGPRs, so A/B rebuild moves
    from above the FREE barrier to just before the A/B issues (after scales).
  - MXSB becomes the 2nd tensor_load instead of the 4th.

usage: patch_tdm_scalefirst.py SRC_LOGDIR NAME
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


def find_rebuild(lines, tc, start):
    load_re = re.compile(
        rf"^tensor_load_to_lds s\[sgprtdm{tc}Group0:sgprtdm{tc}Group0\+3\], "
        rf"s\[sgprtdm{tc}Group1:sgprtdm{tc}Group1\+7\]")
    load_idx = next(i for i in range(start, len(lines)) if load_re.match(lines[i]))
    rebuild_start = next(
        i for i in range(load_idx, start - 1, -1)
        if re.match(rf"^s_mov_b32 s\[sgprtdm{tc}Group0\+0\], 1$", lines[i]))
    return rebuild_start, load_idx


def strip_scale_comment(block):
    while block and "Scale GR" in block[-1]:
        block = block[:-1]
    return block


# Locate MAINLOOP_C0 hoisted A rebuild (first s_mov AGroup0+0 after "hoisted TDM").
in_main = False
hoist_comment = None
a_rs = b_rs = a_load = None
for i, l in enumerate(lines):
    if "MAINLOOP_C0 start" in l:
        in_main = True
    if "MAINLOOP_C0 end" in l:
        in_main = False
    if in_main and "hoisted TDM address" in l:
        hoist_comment = i
    if (in_main and hoist_comment is not None and a_rs is None and
            re.match(r"^s_mov_b32 s\[sgprtdmAGroup0\+0\], 1$", l)):
        a_rs = i
    if a_rs is not None and b_rs is None and re.match(r"^s_mov_b32 s\[sgprtdmBGroup0\+0\], 1$", l):
        b_rs = i
    if (a_rs is not None and a_load is None and l.startswith("tensor_load_to_lds")
            and "for A" in l):
        a_load = i
        break
assert None not in (a_rs, b_rs, a_load), (a_rs, b_rs, a_load)
assert "for B" in lines[a_load + 1]

# B rebuild ends at MXSA addr update
b_end = next(i for i in range(b_rs, a_load) if "TDM addr update: MXSA" in lines[i])
a_re = lines[a_rs:b_rs]
b_re = lines[b_rs:b_end]
# Between A swap and A rebuild: nothing to keep special — a_rs follows A xor.
# pre_barrier = from MXSA addr update through line before A load
pre_barrier = lines[b_end:a_load]
sa_rs, sa_ld = find_rebuild(lines, "MXSA", a_load + 2)
sb_rs, sb_ld = find_rebuild(lines, "MXSB", sa_ld + 1)
sa_re = strip_scale_comment(lines[sa_rs:sa_ld])
sb_re = strip_scale_comment(lines[sb_rs:sb_ld])

# Labels right after each scale load (sciter skip targets) must move with the loads.
sa_lbl = sa_ld + 1 if lines[sa_ld + 1].startswith(".L_sciter_MXSA") else None
sb_lbl = sb_ld + 1 if lines[sb_ld + 1].startswith(".L_sciter_MXSB") else None
# a_re currently spans A-rebuild .. B-addr/xor (up to b_rs). Split so B addr/xor
# stay with B rebuild after A issue.
b_addr = next((i for i in range(a_rs, b_rs)
               if "TDM addr update: B" in lines[i] or
               lines[i].startswith("s_add_u64 s[sgprAddressB")), b_rs)
a_re_only = lines[a_rs:b_addr]
b_prefix = lines[b_addr:b_rs]  # B addr + LDS xor
b_re_full = b_prefix + lines[b_rs:b_end]

new_mid = []
new_mid.append("// patched: A/B descriptor rebuild deferred until after scale issues")
new_mid.extend(pre_barrier)
new_mid.append("// patched: issue MXSA/MXSB before A/B (MXSB earlier)")
new_mid.extend(sa_re)
if "Scale GR" in lines[sa_ld - 1] and "Scale GR" not in (sa_re[-1] if sa_re else ""):
    new_mid.append(lines[sa_ld - 1])
new_mid.append(lines[sa_ld])
if sa_lbl is not None:
    new_mid.append(lines[sa_lbl])
new_mid.extend(sb_re)
if "Scale GR" in lines[sb_ld - 1] and "Scale GR" not in (sb_re[-1] if sb_re else ""):
    new_mid.append(lines[sb_ld - 1])
new_mid.append(lines[sb_ld])
if sb_lbl is not None:
    new_mid.append(lines[sb_lbl])
new_mid.append("// patched: rebuild+issue A/B after scales (aliasing forces this)")
new_mid.extend(a_re_only)
new_mid.append(lines[a_load])
new_mid.extend(b_re_full)
new_mid.append(lines[a_load + 1])

# Replace [a_rs .. sb_ld] (and sciter labels if present) with new_mid
tail_start = (sb_lbl + 1) if sb_lbl is not None else (sb_ld + 1)
out = lines[:a_rs] + new_mid + lines[tail_start:]
text = "\n".join(out)

# Sanity: first tensor_load after the FREE barrier in mainloop is MXSA
body = text[text.find("MAINLOOP_C0 start"):text.find("MAINLOOP_C0 end")]
# Find barrier then next tensor_load
bm = re.search(r"s_barrier_wait -1[^\n]*\n(?:[^\n]*\n)*?tensor_load_to_lds s\[sgprtdm(\w+)Group0", body)
assert bm and bm.group(1) == "MXSA", bm.group(1) if bm else None

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-tdmsfirst-{name}"
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
print("scalefirst: SA,SB then A,B", file=sys.stderr)
print(dst)
