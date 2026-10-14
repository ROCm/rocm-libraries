#!/usr/bin/env python3
"""Assign each main-loop TDM load (A, B, MXSA, MXSB) to a set of producer waves,
mirroring the gemm_wt opt_v7 per-wave TDM ownership. Generalises patch_tdm_wavegate.py.

usage: patch_tdm_roles.py SRC_LOGDIR NAME A=0 B=1 MXSA=2 MXSB=3 [WAIT=n]
  T=w[+w...]  waves that issue tensor T in MAINLOOP_C0 (unlisted tensors stay on all waves)
  WAIT=n      override the loop s_wait_tensorcnt threshold

Only the descriptor build + tensor_load_to_lds is gated; the address increment and LDS
buffer swap before each block stay ungated since every wave's local reads depend on them.

The loop wait must be <= the number of loads the waiting wave issued this iteration,
otherwise a previous-iteration load can still be in flight at the barrier. Prologue
loads are still issued by all waves, so a wave issuing k loads per iteration over-waits
on the first iteration only. Default WAIT = min over waves of loads issued per iteration.
"""
import glob, os, re, subprocess, sys, time

BENCH = os.path.dirname(os.path.abspath(__file__))
LLVM = os.environ.get("LLVM_BIN", "/opt/rocm/lib/llvm/bin")
NUM_WAVES = 4
TENSORS = ("A", "B", "MXSA", "MXSB")

src, name = sys.argv[1], sys.argv[2]
owners = {t: list(range(NUM_WAVES)) for t in TENSORS}
waitOverride = None
for a in sys.argv[3:]:
    k, v = a.split("=")
    if k == "WAIT":
        waitOverride = int(v)
    else:
        owners[k] = [int(w) for w in v.split("+")]

perWave = [sum(w in owners[t] for t in TENSORS) for w in range(NUM_WAVES)]
waitN = waitOverride if waitOverride is not None else min(perWave)
print(f"owners={owners} loads/wave/iter={perWave} loop wait tensorcnt {waitN}", file=sys.stderr)

asm = glob.glob(f"{BENCH}/{src}/**/assembly/*.s", recursive=True)[0]
ini = glob.glob(f"{BENCH}/{src}/**/ClientParameters.ini", recursive=True)[0]
lines = open(asm).read().split("\n")

WAVEID_READ = "s_bfe_u32 s64, ttmp8, 0x00050019             // patched: wave id (ttmp8[29:25])"
blockStartRe = {t: re.compile(rf"^s_mov_b32 s\[sgprtdm{t}Group0\+0\], 1$") for t in TENSORS}
blockEndMarker = {t: f"// TDM: global->LDS for {t}" for t in TENSORS}

out, inLoop, counter, gated, waits = [], False, 0, [], 0
i, n = 0, len(lines)
while i < n:
    l = lines[i]
    if l.startswith("/* MAINLOOP_C0 start */"):
        inLoop = True
    elif l.startswith("/* MAINLOOP_C0 end */"):
        inLoop = False

    t = next((t for t, rx in blockStartRe.items() if rx.match(l)), None) if inLoop else None
    if t and len(owners[t]) < NUM_WAVES:
        j = i
        while blockEndMarker[t] not in lines[j]:
            j += 1
        label = f".L_tdmrole_{t}_{counter}"
        counter += 1
        gated.append(t)
        mask = sum(1 << w for w in owners[t])
        out.append(WAVEID_READ)
        out.append(f"s_lshr_b32 s64, 0x{mask:x}, s64            // patched: owner mask >> wave id")
        out.append(f"s_bitcmp1_b32 s64, 0                   // patched: is this wave a {t} producer")
        out.append(f"s_cbranch_scc0 {label}")
        out.extend(lines[i:j + 1])
        out.append(f"{label}:")
        i = j + 1
        continue

    if inLoop and re.match(r"^s_wait_tensorcnt \d+", l):
        out.append(f"s_wait_tensorcnt {waitN}                                 // patched: per-wave producer roles")
        waits += 1
        i += 1
        continue

    out.append(l)
    i += 1

print(f"gated {gated}, rewrote {waits} loop tensorcnt waits", file=sys.stderr)
assert waits == 1, "expected exactly one s_wait_tensorcnt in MAINLOOP_C0"
assert sorted(gated) == sorted(t for t in TENSORS if len(owners[t]) < NUM_WAVES)

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-tdmrole-{name}"
os.makedirs(dst)
s = f"{dst}/kernel.s"
open(s, "w").write("\n".join(out))
subprocess.run([f"{LLVM}/clang", "-x", "assembler", "--target=amdgcn-amd-amdhsa", "-mcode-object-version=4",
                "-c", "-Xclangas", "-target-feature", "-Xclangas", "+real-true16", "-mcpu=gfx1250",
                "-mno-wavefrontsize64", s, "-o", f"{dst}/kernel.o"], check=True)
subprocess.run([f"{LLVM}/clang", "--target=amdgcn-amd-amdhsa", "-Xlinker", "--build-id=sha1",
                f"{dst}/kernel.o", "-o", f"{dst}/kernel.co"], check=True)
cfg = re.sub(r"^code-object=.*$", f"code-object={dst}/kernel.co", open(ini).read(), flags=re.M)
open(f"{dst}/ClientParameters.ini", "w").write(cfg)
print(dst)
