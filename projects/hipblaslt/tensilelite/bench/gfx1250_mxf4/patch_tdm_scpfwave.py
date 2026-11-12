#!/usr/bin/env python3
"""Move scale ITERATE issuers to non-TDM waves (gemm_wt mx_wave = pf_wave).

sclds4/sciter puts MXSA on wave 0 and MXSB on wave 1 — the same waves that also
issue A/B and run the early post-issue WMMAs. gemm_wt instead issues scale TDMs
from the operand's PREFETCH wave (the non-TDM waves), so data producers are not
also blocked on the scale iterate fill.

With tdm_wave_a=0, tdm_wave_b=1 the pf waves are 2 and 3:
  wave 2 owns MXSA ITERATE x4
  wave 3 owns MXSB ITERATE x4
  waves 0/1 keep A,B only and wait_tensorcnt 2
  waves 2/3 wait_tensorcnt 5 (A,B + up to 3 in-flight scale fills, depth4)

Expects an already-sciter'd (+ optionally dsspread) kernel.s.

usage: patch_tdm_scpfwave.py SRC_LOGDIR NAME
"""
import glob, os, re, subprocess, sys, time

BENCH = os.path.dirname(os.path.abspath(__file__))
LLVM = os.environ.get("LLVM_BIN", "/opt/rocm/lib/llvm/bin")

src, name = sys.argv[1], sys.argv[2]
src = src if os.path.isabs(src) else f"{BENCH}/{src}"
asms = glob.glob(f"{src}/**/assembly/*.s", recursive=True) + glob.glob(f"{src}/kernel.s")
inis = glob.glob(f"{src}/**/ClientParameters.ini", recursive=True) + glob.glob(f"{src}/ClientParameters.ini")
asm, ini = asms[0], inis[0]
text = open(asm).read()

n_sa = len(re.findall(
    r"s_cmp_eq_u32 s\[sgprWaveIdx\], 0\s+// patched: MXSA iterate issuer = wave 0", text))
n_sb = len(re.findall(
    r"s_cmp_eq_u32 s\[sgprWaveIdx\], 1\s+// patched: MXSB iterate issuer = wave 1", text))
assert n_sa == 4 and n_sb == 4, f"expected 4 MXSA + 4 MXSB sciter gates, got {n_sa}/{n_sb}"

text = re.sub(
    r"(s_cmp_eq_u32 s\[sgprWaveIdx\], )0(\s+// patched: MXSA iterate issuer = wave )0",
    r"\g<1>2\g<2>2", text)
text = re.sub(
    r"(s_cmp_eq_u32 s\[sgprWaveIdx\], )1(\s+// patched: MXSB iterate issuer = wave )1",
    r"\g<1>3\g<2>3", text)

# Flip wait split: scale issuers are now waves >= 2
wait_pat = re.compile(
    r"s_cmp_ge_u32 s\[sgprWaveIdx\], 2\s+// patched: waves 2/3 issue only A,B\n"
    r"s_cbranch_scc1 (\.L_sciter_wait2_\d+)\n"
    r"s_wait_tensorcnt 5\s+// patched: waves 0/1: A,B \+ up to 3 scale fills \(depth4\)\n"
    r"s_branch (\.L_sciter_waitdone_\d+)\n"
    r"(\.L_sciter_wait2_\d+):\n"
    r"s_wait_tensorcnt 2\s+// patched: waves 2/3: A,B only\n"
    r"(\.L_sciter_waitdone_\d+):",
    re.M)

def flip_wait(m):
    lab2, labd = m.group(1), m.group(2)
    assert m.group(3) == lab2 and m.group(4) == labd
    return (
        f"s_cmp_ge_u32 s[sgprWaveIdx], 2                       "
        f"// patched: waves 2/3 issue A,B + scale ITERATE\n"
        f"s_cbranch_scc1 {lab2}\n"
        f"s_wait_tensorcnt 2                                   "
        f"// patched: waves 0/1: A,B only (gemm_wt pf-wave scales)\n"
        f"s_branch {labd}\n"
        f"{lab2}:\n"
        f"s_wait_tensorcnt 5                                   "
        f"// patched: waves 2/3: A,B + up to 3 scale fills (depth4)\n"
        f"{labd}:")

text, nwait = wait_pat.subn(flip_wait, text)
assert nwait >= 1, f"wait flip matched {nwait}"
assert text.count("MXSA iterate issuer = wave 2") == 4
assert text.count("MXSB iterate issuer = wave 3") == 4
assert "waves 0/1: A,B only (gemm_wt pf-wave scales)" in text

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-tdmscpf-{name}"
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
print(f"remapped MXSA->w2 MXSB->w3, flipped {nwait} wait(s)", file=sys.stderr)
print(dst)
