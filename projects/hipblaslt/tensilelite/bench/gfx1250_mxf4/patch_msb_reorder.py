#!/usr/bin/env python3
"""Reorder ds_load/wmma within each dependency-safe window of the main loop to cluster
instructions that need the same s_set_vgpr_msb value adjacently (mirroring what the
fp8 reference kernel's static ordering achieves), then dedupe redundant consecutive
s_set_vgpr_msb sets.

Why this is safe (no correctness risk from the reorder itself):
Each window is the span between one `s_wait_dscnt 0` (phase start) and the next
non-ds_load/wmma/msb instruction. Within such a window:
  - every ds_load prefetches data for the *next* subIterK phase (never consumed until
    a wait several phases later) -- moving it earlier/later within this window cannot
    race with its own consumer.
  - every wmma consumes data already guaranteed ready by the wait that just fired, and
    each wmma writes a distinct destination (no cross-wmma WAW within one phase).
  - ds_load destinations and wmma destinations are disjoint register ranges (verified:
    ds_load targets the v512+/v768+ prefetch-buffer banks, wmma targets accumulator
    vgprs below that), so there is no WAW/RAW hazard between the two groups either.
A window is only touched if EVERY non-blank line in it is exactly one of
s_set_vgpr_msb / ds_load_b128 / ds_load_b32 / v_wmma_scale_f32_32x16x128_f4 -- any
other instruction (barrier, TDM rebuild, LR-buffer-swap xor, loop-counter dec/cmp,
stray comment) stops the window at that point and that content is left untouched.

Algorithm per window:
  1. Resolve each ds_load/wmma to its *effective* msb value (tracking current state
     as in the original stream, so implicit/inherited values are materialized).
  2. Split into loads[] and wmmas[], each stable-sorted by effective msb value
     (preserves relative order among ties -- no need to reorder within a tie group).
  3. Re-emit loads sorted, then wmmas sorted, re-inserting s_set_vgpr_msb only when
     the value changes from the previous emitted instruction (across the whole
     window, including the loads->wmmas boundary).

usage: patch_msb_reorder.py SRC_LOGDIR NAME
Writes logs/<stamp>-msbreorder-NAME/.
"""
import glob, os, re, subprocess, sys, time

BENCH = os.path.dirname(os.path.abspath(__file__))
LLVM = os.environ.get("LLVM_BIN", "/opt/rocm/lib/llvm/bin")

src, name = sys.argv[1], sys.argv[2]
asm = glob.glob(f"{BENCH}/{src}/**/assembly/*.s", recursive=True)[0]
ini = glob.glob(f"{BENCH}/{src}/**/ClientParameters.ini", recursive=True)[0]
lines = open(asm).read().split("\n")

msbRe = re.compile(r"^s_set_vgpr_msb (\d+)\s*(//.*)?$")
loadRe = re.compile(r"^ds_load_b(128|32) ")
wmmaRe = re.compile(r"^v_wmma_scale_f32_32x16x128_f4 ")
waitRe = re.compile(r"^s_wait_dscnt 0\s*(//.*)?$")

out = []
inLoop = False
i, n = 0, len(lines)
windowsFound = 0
recordsTouched = 0
msbSaved = 0

while i < n:
    l = lines[i]
    if re.match(r"/\* MAINLOOP_C0 start \*/", l):
        inLoop = True
    elif re.match(r"/\* MAINLOOP_C0 end \*/", l):
        inLoop = False

    out.append(l)
    i += 1

    if inLoop and waitRe.match(l):
        # scan forward collecting the pure-content window
        j = i
        pending_msb = None  # explicit msb line seen but not yet attached to an instr
        currentMsb = None
        records = []  # (effectiveMsb, line, category)
        while j < n:
            lj = lines[j]
            m = msbRe.match(lj)
            if m:
                pending_msb = int(m.group(1))
                j += 1
                continue
            if loadRe.match(lj) or wmmaRe.match(lj):
                if pending_msb is not None:
                    currentMsb = pending_msb
                    pending_msb = None
                assert currentMsb is not None, "window started without an explicit s_set_vgpr_msb"
                cat = "LOAD" if loadRe.match(lj) else "WMMA"
                records.append([currentMsb, lj, cat])
                j += 1
                continue
            break  # first foreign line: window ends here
        # pending_msb with no instruction attached (shouldn't happen, but if it does,
        # the window wasn't pure after all -- bail out conservatively)
        if pending_msb is not None or not records:
            continue

        windowsFound += 1
        loads = [r for r in records if r[2] == "LOAD"]
        wmmas = [r for r in records if r[2] == "WMMA"]
        loads.sort(key=lambda r: r[0])
        wmmas.sort(key=lambda r: r[0])

        lastMsb = None
        emitted = []
        for grp in (loads, wmmas):
            for msbVal, instrLine, _ in grp:
                recordsTouched += 1
                if msbVal != lastMsb:
                    emitted.append(f"s_set_vgpr_msb {msbVal}                                   // patched: reordered")
                    lastMsb = msbVal
                else:
                    msbSaved += 1
                emitted.append(instrLine)

        out.extend(emitted)
        i = j  # resume scanning right where the pure window ended

print(f"{windowsFound} reorderable windows found, {recordsTouched} ds_load/wmma records "
      f"reordered, {msbSaved} s_set_vgpr_msb instructions eliminated by the reorder "
      f"(vs. the dedupe the generator already does in place)", file=sys.stderr)
assert windowsFound >= 3, "expected at least the 3 clean subIterK=0,1,2 windows per loop copy"

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-msbreorder-{name}"
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
