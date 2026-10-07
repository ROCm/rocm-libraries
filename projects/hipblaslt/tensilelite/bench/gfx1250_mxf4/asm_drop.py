#!/usr/bin/env python3
"""Comment out matching instructions in a range of a build's kernel and rebuild the code object.

usage: asm_drop.py SRC_LOGDIR NAME START_LABEL END_LABEL REGEX

Every line from START_LABEL: up to END_LABEL: that matches REGEX (re.search) becomes a comment.
START_LABEL "-" means the top of the file, END_LABEL "-" the end. For localizing faults and
costs; results are wrong. Writes logs/<stamp>-drop-NAME/ with kernel.s, kernel.co and a
ClientParameters.ini pointing at it (usable by alt_timing.py and validate.py).
"""
import glob, os, re, subprocess, sys, time

BENCH = os.path.dirname(os.path.abspath(__file__))
LLVM = "/opt/rocm-10.1.0a20260908+bkc.20260917/lib/llvm/bin"

src, name, start, end, pattern = sys.argv[1:6]
asm = glob.glob(f"{BENCH}/{src}/**/assembly/*.s", recursive=True)[0]
ini = glob.glob(f"{BENCH}/{src}/**/ClientParameters.ini", recursive=True)[0]
lines = open(asm).read().split("\n")

lo = 0 if start == "-" else next(i for i, l in enumerate(lines) if l.startswith(start + ":"))
hi = len(lines) if end == "-" else next(i for i, l in enumerate(lines) if i > lo and l.startswith(end + ":"))
rx = re.compile(pattern)
dropped = 0
for i in range(lo, hi):
    if rx.search(lines[i]) and not lines[i].lstrip().startswith("//"):
        lines[i] = "// dropped: " + lines[i]
        dropped += 1
print(f"dropped {dropped} lines in [{lo + 1}, {hi}]", file=sys.stderr)

out = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-drop-{name}"
os.makedirs(f"{out}/assembly")
s = f"{out}/assembly/kernel.s"
open(s, "w").write("\n".join(lines))
subprocess.run([f"{LLVM}/clang", "-x", "assembler", "--target=amdgcn-amd-amdhsa",
                f"-mcode-object-version={os.environ.get('CO_VERSION', '4')}", "-c",
                "-Xclangas", "-target-feature", "-Xclangas", "+real-true16",
                "-mcpu=gfx1250", "-mno-wavefrontsize64", s, "-o", f"{out}/kernel.o"], check=True)
subprocess.run([f"{LLVM}/clang", "--target=amdgcn-amd-amdhsa", "-Xlinker", "--build-id=sha1",
                f"{out}/kernel.o", "-o", f"{out}/kernel.co"], check=True)
cfg = re.sub(r"^code-object=.*$", f"code-object={out}/kernel.co", open(ini).read(), flags=re.M)
open(f"{out}/ClientParameters.ini", "w").write(cfg)
print(out)
