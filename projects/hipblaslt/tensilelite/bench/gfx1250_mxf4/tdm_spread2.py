#!/usr/bin/env python3
"""Delay main-loop TDM issues to later points in the iteration (hoist-aware, statically checked).

usage: tdm_spread2.py SRC_ASM_OR_DIR NAME [A:N] [B:N] [MXSA:N] [MXSB:N] [--ini INI]

Replaces tdm_spread.py, whose per-tensor "address update .. tensor_load" ranges overlap on a
build where the A/B setup is hoisted above the FREE barrier: moving A and B then duplicates the
FREE barrier and the A load. Here the movable units are:

  A, B       the single tensor_load_to_lds (their descriptors are built, hoisted, before the
             FREE barrier and stay there)
  MXSA, MXSB descriptor build + tensor_load_to_lds (starting at `s_mov_b32 sgprtdmMXS?Group0+0`)

Barriers (FREE, READY, cluster) and the address-update / LDS-swap SALU never move. N is the
1-based index of the v_wmma after which the unit is issued; unspecified units stay at the
original issue point (after WMMA 95 on cl24). Units can only move LATER: before the FREE barrier
the buffer is still being read.

Static checks (any failure aborts before a code object is produced):
  * MXSA's descriptor shares A's SGPRs and MXSB's shares B's, so A must issue before MXSA and
    B before MXSB.
  * SCC: the inserted SALU clobbers SCC; the insertion point may not sit between an SCC
    producer and its consumer (s_cmp -> s_cbranch_scc*, loop-counter s_sub_u32 -> branch).
  * No jumped-over line may write an SGPR a moved unit reads, or touch one it writes.
  * The patched loop is a permutation of the original except for the s_wait_tensorcnt count.
  * Cluster wait (-3) must stay after every moved unit; units may not pass the loop end.

The s_wait_tensorcnt count becomes the number of loads issued before it in program order
(loads complete in order, so this still forces the previous iteration's loads to land).
Writes logs/<stamp>-spread2-NAME/ (kernel.s, kernel.o, kernel.co, ClientParameters.ini).
`--check-only` runs the static checks and the identity test without assembling.
"""
import glob, os, re, subprocess, sys, time, collections

BENCH = os.path.dirname(os.path.abspath(__file__))
LLVM = os.environ.get("LLVM_BIN", "/opt/rocm/lib/llvm/bin")
ORDER = ("A", "B", "MXSA", "MXSB")
ALIAS_BEFORE = (("A", "MXSA"), ("B", "MXSB"))   # descriptor SGPR aliasing: first must issue first

args = [a for a in sys.argv[1:]]
check_only = "--check-only" in args
if check_only:
    args.remove("--check-only")
ini = None
if "--ini" in args:
    k = args.index("--ini")
    ini = args[k + 1]
    del args[k:k + 2]
src, name = args[0], args[1]
want = {k: int(v) for k, v in (a.split(":") for a in args[2:])}
for t in want:
    assert t in ORDER, t

asm = src if src.endswith(".s") else glob.glob(f"{BENCH}/{src}/**/assembly/*.s", recursive=True)[0]
lines = open(asm).read().split("\n")
lo = next(i for i, l in enumerate(lines) if l.startswith("label_LoopBeginL:"))
hi = next(i for i, l in enumerate(lines) if i > lo and l.startswith("label_SkipMainloop:"))
loop = lines[lo:hi]

# ---- locate units (indices into loop) ----
unit = {}
for t in ("A", "B"):
    idx = [i for i, l in enumerate(loop) if l.startswith(f"tensor_load_to_lds s[sgprtdm{t}Group0")]
    assert len(idx) == 1, (t, idx)
    unit[t] = (idx[0], idx[0] + 1)
for t in ("MXSA", "MXSB"):
    s = [i for i, l in enumerate(loop) if l.startswith(f"s_mov_b32 s[sgprtdm{t}Group0+0], 1")]
    e = [i for i, l in enumerate(loop) if l.startswith(f"tensor_load_to_lds s[sgprtdm{t}Group0")]
    assert len(s) == 1 and len(e) == 1 and s[0] < e[0], (t, s, e)
    unit[t] = (s[0], e[0] + 1)
# units must be a contiguous pure-SALU region (no barrier / wmma / label inside)
for t, (s, e) in unit.items():
    for l in loop[s:e]:
        assert not re.match(r"(v_|s_barrier|s_cbranch|s_branch|s_wait|label_|s_cmp)", l), (t, l)
ranges = sorted(unit.values())
for (s0, e0), (s1, e1) in zip(ranges, ranges[1:]):
    assert e0 <= s1, "overlapping units"

wmma = [i for i, l in enumerate(loop) if l.startswith("v_wmma")]
nW = len(wmma)
orig_pos = sum(1 for i in wmma if i < min(s for s, _ in unit.values()))   # WMMAs before the burst
free_idx = [i for i, l in enumerate(loop) if l.startswith("s_barrier_wait -1")]
cb_wait = [i for i, l in enumerate(loop) if l.startswith("s_barrier_wait -3")]
assert cb_wait, "no cluster wait in loop"
wait_idx = next(i for i, l in enumerate(loop) if l.startswith("s_wait_tensorcnt"))
wait_pos = sum(1 for i in wmma if i < wait_idx)
first_unit = min(s for s, _ in unit.values())
assert any(i < first_unit for i in free_idx), "FREE barrier must precede the burst"

pos = {t: want.get(t, orig_pos) for t in ORDER}
for t, n in want.items():
    assert n >= orig_pos, f"{t}:{n} is before the FREE barrier (WMMA {orig_pos}); loads cannot move earlier"
    assert n <= nW - 2, f"{t}:{n}: keep at least 2 WMMAs before the loop end"
rank = {t: (pos[t], ORDER.index(t)) for t in ORDER}
for a, b in ALIAS_BEFORE:
    assert rank[a] <= rank[b], f"{a} must issue before {b} (shared descriptor SGPRs)"

moved = [t for t in ORDER if t in want and want[t] != orig_pos]
skip = set()
for t in moved:
    skip.update(range(*unit[t]))
# the unmoved units keep their original text position; moved units go after WMMA N
insert_after = collections.defaultdict(list)
for t in moved:
    insert_after[wmma[pos[t] - 1]].append(t)
for k in insert_after:
    insert_after[k].sort(key=ORDER.index)

# ---- static checks ----
SCC_WRITERS = re.compile(r"s_(and|or|xor|lsh[lr]|ashr|add|sub|addc|subc|cmp|bitcmp|min|max|abs|not|nand|nor|xnor|andn2|orn2|bfe|bcnt|mul|cselect_never|absdiff)_?")
SCC_READERS = re.compile(r"(s_cbranch_scc|s_cselect|s_addc|s_subb|s_cmovk)")
def writes_scc(l):
    m = re.match(r"(s_\w+)", l)
    if not m: return False
    op = m.group(1)
    if op.startswith(("s_mov", "s_cmov", "s_wait", "s_set_vgpr", "s_barrier", "s_nop", "s_cbranch", "s_branch")): return False
    return bool(SCC_WRITERS.match(op)) or op.startswith("s_cmp")
for t in moved:
    n = pos[t]
    j = wmma[n - 1] + 1
    while j < len(loop):                     # first SCC-relevant instruction after the insertion point
        l = loop[j]
        if j in skip or not l or l.startswith(("//", "/*", "s_set_vgpr", "v_", "ds_", "s_wait", ".align", "buffer_", "global_", "tensor_")):
            j += 1; continue
        if SCC_READERS.match(l):
            sys.exit(f"SCC check: {t}:{n} would sit before an SCC consumer at loop line {j}: {l}")
        if writes_scc(l) or l.startswith("label_"):
            break
        j += 1
    # also the producer side: SCC producer immediately before with a consumer after the point is
    # caught by the forward scan above (the consumer is hit before any writer).

def sgprs(lines_):
    return set(re.findall(r"sgpr\w+", " ".join(lines_)))
def first_operand_sgprs(lines_):
    out = set()
    for l in lines_:
        m = re.match(r"s_\w+\s+([^,]+),", l)
        if m: out |= set(re.findall(r"sgpr\w+", m.group(1)))
    return out
ALIASES = {"sgprtdmMXSAGroup0": "sgprtdmAGroup0", "sgprtdmMXSAGroup1": "sgprtdmAGroup1",
           "sgprtdmMXSBGroup0": "sgprtdmBGroup0", "sgprtdmMXSBGroup1": "sgprtdmBGroup1"}
def canon(sset): return {ALIASES.get(x, x) for x in sset}
for t in moved:
    u = loop[slice(*unit[t])]
    uses, defs = canon(sgprs(u)), canon(first_operand_sgprs(u))
    a, b = (unit[t][1], wmma[pos[t] - 1]) if wmma[pos[t] - 1] >= unit[t][1] else (None, None)
    # lines jumped over: from the unit's old slot up to the new insertion point (excluding moved units)
    over = [loop[i] for i in range(unit[t][1], wmma[pos[t] - 1] + 1) if i not in skip]
    for l in over:
        if re.match(r"s_\w+\s+", l):
            ldef = canon(first_operand_sgprs([l]))
            bad = ldef & uses
            if bad: sys.exit(f"def/use: {t}:{pos[t]} jumps over a write of {bad}: {l}")
        lall = canon(sgprs([l]))
        if lall & defs and not l.startswith(("tensor_load_to_lds",)):
            sys.exit(f"alias: {t}:{pos[t]} jumps over a line touching its descriptor regs: {l}")
        if l.startswith("tensor_load_to_lds") and canon(sgprs([l])) & defs:
            # a different tensor's load reading regs this unit would overwrite: must stay before
            pass
    # a moved unit may not pass the load that consumes the regs it overwrites (checked by rank above),
    # and may not pass the cluster wait.
    assert wmma[pos[t] - 1] < min(cb_wait), f"{t}:{pos[t]} passes the cluster wait"

# ---- build ----
before = sum(1 for t in ORDER if (pos[t] <= wait_pos))
out = []
for i, l in enumerate(loop):
    if i in skip:
        continue
    if i == wait_idx:
        l = re.sub(r"s_wait_tensorcnt \d+", f"s_wait_tensorcnt {before}", l)
    out.append(l)
    for t in insert_after.get(i, []):
        out += loop[slice(*unit[t])]
# the A/B loads must be issued before the wait when they stay in place; moved ones may be after it
new_lines = lines[:lo] + out + lines[hi:]

# permutation check
def norm(ls): return collections.Counter(re.sub(r"s_wait_tensorcnt \d+", "s_wait_tensorcnt N", l) for l in ls)
assert norm(loop) == norm(out), "loop is not a permutation of the original"
if not moved:
    assert out == loop, "identity patch changed the loop"
    assert before == int(re.search(r"s_wait_tensorcnt (\d+)", loop[wait_idx]).group(1)), "identity wait count differs"
print(f"orig issue after WMMA {orig_pos}, wait after WMMA {wait_pos}; positions {pos}; moved {moved}; wait count {before}",
      file=sys.stderr)
# order in program text (for the record)
seq = [(next(i for i, l in enumerate(out) if l.startswith('tensor_load_to_lds') and f'sgprtdm{t}Group0' in l), t) for t in ORDER]
print("load order:", [t for _, t in sorted(seq)], file=sys.stderr)
if check_only:
    print("static checks OK", file=sys.stderr)
    sys.exit(0)

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-spread2-{name}"
os.makedirs(f"{dst}/assembly")
s = f"{dst}/assembly/kernel.s"
open(s, "w").write("\n".join(new_lines))
open(f"{dst}/kernel.s", "w").write("\n".join(new_lines))
subprocess.run([f"{LLVM}/clang", "-x", "assembler", "--target=amdgcn-amd-amdhsa",
                f"-mcode-object-version={os.environ.get('CO_VERSION', '4')}", "-c",
                "-Xclangas", "-target-feature", "-Xclangas", "+real-true16",
                "-mcpu=gfx1250", "-mno-wavefrontsize64", s, "-o", f"{dst}/kernel.o"], check=True,
               stderr=subprocess.DEVNULL)
subprocess.run([f"{LLVM}/clang", "--target=amdgcn-amd-amdhsa", "-Xlinker", "--build-id=sha1",
                f"{dst}/kernel.o", "-o", f"{dst}/kernel.co"], check=True)
ini = ini or glob.glob(f"{BENCH}/logs/20261022-baseline-cl24-gpu0/ClientParameters.ini")[0]
cfg = re.sub(r"^code-object=.*$", f"code-object={dst}/kernel.co", open(ini).read(), flags=re.M)
open(f"{dst}/ClientParameters.ini", "w").write(cfg)
print(dst)
