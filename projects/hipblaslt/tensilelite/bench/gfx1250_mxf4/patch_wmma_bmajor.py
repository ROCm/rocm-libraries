#!/usr/bin/env python3
"""Reorder 4x8 WMMA tiles to B-major snake (prototype-style LR slack).

Current emit_mfma zigzag sweeps A as the outer dim (4 rows x 8 B cols). Late B
tiles for the next K are loaded near the end of the previous K-step and first
used ~11 WMMA later (~88 clk) — just above the scaled-WMMA LDS floor (~76 clk).
B-major (8 B rows x 4 A cols) pushes first-use of those late B tiles later and
lifts modelled min lead to ~14 WMMA (~112 clk) with LR issue slots held fixed.

Moves each (s_set_vgpr_msb?, v_wmma_scale) bundle; leaves ds_load holes in place.
Rewrites matrix_*_reuse on the prev WMMA when the next shares that operand.

usage: patch_wmma_bmajor.py SRC_LOGDIR NAME
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

wmmaRe = re.compile(
    r"v_wmma_scale_f32_32x16x128_f4\b.*// MFMA C\[(\d+),(\d+)\] \+= A\[(\d+),K=(\d+)\] \* B\[(\d+),K=(\d+)\]"
)
msbRe = re.compile(r"^s_set_vgpr_msb\b")
reuseRe = re.compile(r"\s*matrix_[ab]_reuse")


def zigzag(rows, cols):
    out = []
    for r in range(rows):
        cols_i = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        out.extend((r, c) for c in cols_i)
    return out


# B-major: outer = B (8), inner = A (4) → pairs (a, b)
B_MAJOR = [(c, r) for r, c in zigzag(8, 4)]
assert len(B_MAJOR) == 32 and len(set(B_MAJOR)) == 32


def strip_reuse(line):
    return reuseRe.sub("", line)


def with_reuse(line, a_reuse, b_reuse):
    line = strip_reuse(line)
    flags = []
    if a_reuse:
        flags.append("matrix_a_reuse")
    if b_reuse:
        flags.append("matrix_b_reuse")
    if not flags:
        return line
    # Insert flags before the // comment.
    if " //" in line:
        head, comment = line.split(" //", 1)
        return f"{head.rstrip()} {' '.join(flags)} //{comment}"
    return f"{line.rstrip()} {' '.join(flags)}"


# Collect WMMA sites: line index, optional preceding s_set_vgpr_msb, (a,b,k).
sites = []
for i, l in enumerate(lines):
    m = wmmaRe.search(l)
    if not m:
        continue
    a, b, aa, k, bb, kk = map(int, m.groups())
    assert a == aa and b == bb and k == kk, (a, b, k, aa, bb, kk)
    msb_i = i - 1 if i > 0 and lines[i - 1].lstrip().startswith("s_set_vgpr_msb") else -1
    sites.append({"i": i, "msb": msb_i, "a": a, "b": b, "k": k})

# Group into contiguous runs of 32 covering a full 4x8 at one K.
groups = []
i = 0
while i < len(sites):
    chunk = sites[i:i + 32]
    if len(chunk) < 32:
        break
    ks = {s["k"] for s in chunk}
    pairs = {(s["a"], s["b"]) for s in chunk}
    if len(ks) == 1 and pairs == set(B_MAJOR):
        groups.append(chunk)
        i += 32
    else:
        i += 1

assert groups, "no 4x8 WMMA groups found"
print(f"WMMA sites: {len(sites)}  B-major groups: {len(groups)}", file=sys.stderr)

# Apply reorder: rewrite msb+wmma lines at the original slot positions.
new_lines = list(lines)
for g in groups:
    by_ab = {(s["a"], s["b"]): s for s in g}
    slots = g  # slot order = original emission order
    for slot, (a, b) in zip(slots, B_MAJOR):
        src = by_ab[(a, b)]
        # Place src's msb+wmma text into slot's locations
        if slot["msb"] >= 0 and src["msb"] >= 0:
            new_lines[slot["msb"]] = lines[src["msb"]]
        new_lines[slot["i"]] = strip_reuse(lines[src["i"]])

# Recompute reuse across all WMMA lines in file order (after reorder).
wmma_idxs = [s["i"] for s in sites]  # original indices still hold the WMMA slots
# After reorder, sites[i]["i"] still points to slot positions in order; content changed.
ordered_idx = []
for g in groups:
    for slot in g:
        ordered_idx.append(slot["i"])
# Also include any WMMA not in a group, in file order
grouped = {s["i"] for g in groups for s in g}
for s in sites:
    if s["i"] not in grouped:
        ordered_idx.append(s["i"])
ordered_idx = sorted(set(ordered_idx))

def ab_of(line):
    m = wmmaRe.search(line)
    return (int(m.group(1)), int(m.group(2))) if m else None

for prev_i, cur_i in zip(ordered_idx, ordered_idx[1:]):
    prev_ab = ab_of(new_lines[prev_i])
    cur_ab = ab_of(new_lines[cur_i])
    if not prev_ab or not cur_ab:
        continue
    a_reuse = prev_ab[0] == cur_ab[0]
    b_reuse = prev_ab[1] == cur_ab[1]
    new_lines[prev_i] = with_reuse(strip_reuse(new_lines[prev_i]), a_reuse, b_reuse)
# Clear reuse on last of each isolated / keep last without forward neighbour already handled

# Verify each group is now B-major
bad = 0
for g in groups:
    got = [ab_of(new_lines[s["i"]]) for s in g]
    if got != B_MAJOR:
        bad += 1
assert bad == 0, f"{bad} groups not B-major after patch"

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-wmma-{name}"
os.makedirs(dst)
s = f"{dst}/kernel.s"
open(s, "w").write("\n".join(new_lines))
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
print(dst)
