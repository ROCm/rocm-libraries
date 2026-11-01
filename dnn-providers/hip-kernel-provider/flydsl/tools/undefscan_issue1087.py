# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Verbatim copy of tooling/llvm-lostcopy/undefscan.py from the attachment to ROCm/FlyDSL#1087 (REPORT-gfx950-splitkit-lost-subrange.tar.gz), written by Xinya Zhang (AMD, AOTriton maintainer) for that report; used unmodified.
"""Find kernels where an undefined SGPR reaches a buffer descriptor that is used.

"Reads an undefined register" is not by itself a defect: LLVM knowingly emits reads
of registers it never defined on paths where the value is dead, so a bare scan for
them flags a third of the tree and means nothing.  What made the dkdv miscompile a
bug is narrower -- the undefined pair was the *base address* of a buffer resource
that a `buffer_load` then dereferenced, so bounds clamping could not save it.

Two analyses run together over a CFG built from the branch targets llvm-objdump
prints:

  defined   must-be-defined-on-all-paths (intersection over predecessors).  Seeded
            with the hardware-initialised registers, whose count is per-kernel:
            COMPUTE_PGM_RSRC2 holds USER_SGPR_COUNT, which grows with the
            kernarg-preload length, plus the workgroup-id enables after it.
  tainted   may-derive-from-an-undefined-read (union over predecessors).

Taint crosses the SGPR spill path, which is where the dkdv bug hides: spills go to
lanes of a VGPR via v_writelane_b32 / v_readlane_b32 rather than to scratch, so a
constant-lane pair is tracked as its own storage cell.

A finding is a buffer_* instruction whose 4-dword resource operand is tainted.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

BIN = os.environ.get("LLVM_BIN", "")  # e.g. /opt/rocm/llvm/bin; empty = use PATH
OBJDUMP = os.path.join(BIN, "llvm-objdump") if BIN else "llvm-objdump"
READELF = os.path.join(BIN, "llvm-readelf") if BIN else "llvm-readelf"

NSGPR = 102
ALL = frozenset(range(NSGPR))

RANGE = re.compile(r"s\[(\d+):(\d+)\]")
SINGLE = re.compile(r"(?<![\w\[:])s(\d+)(?![\d:])")
INSN = re.compile(r"\s*(\S+)\s+(.*?)\s*//\s*([0-9A-F]+):")
TARGET = re.compile(r"<[^>]*\+0x([0-9a-f]+)>")
LANE = re.compile(r"v(\d+),\s*(?:v(\d+)|s(\d+)),\s*(\d+)")
NO_DST = (
    "s_cmp",
    "s_store",
    "buffer_store",
    "ds_write",
    "global_store",
    "flat_store",
    "s_branch",
    "s_cbranch",
    "s_setpc",
    "s_barrier",
    "s_waitcnt",
    "s_nop",
    "s_endpgm",
    "s_set_gpr_idx",
    "scratch_store",
    "s_sleep",
    "s_sethalt",
)


def regs(text):
    out = set()
    for lo, hi in RANGE.findall(text):
        out |= set(range(int(lo), int(hi) + 1))
    for m in SINGLE.finditer(RANGE.sub("", text)):
        out.add(int(m.group(1)))
    return {r for r in out if r < NSGPR}


def preinit_count(path):
    out = subprocess.run(
        [READELF, "-x", ".rodata", str(path)], capture_output=True, text=True
    ).stdout
    words = []
    for line in out.splitlines():
        m = re.match(r"\s*0x[0-9a-f]+((?: [0-9a-f]{8})+)", line)
        if m:
            words += [int(w, 16) for w in m.group(1).split()]
    if len(words) < 16:
        return 19
    rsrc2 = int.from_bytes(words[13].to_bytes(4, "big"), "little")  # kd offset 52
    return ((rsrc2 >> 1) & 0x1F) + sum((rsrc2 >> b) & 1 for b in (7, 8, 9))


def disassemble(path):
    dis = subprocess.run(
        [OBJDUMP, "-d", "--mcpu=gfx950", str(path)], capture_output=True, text=True
    ).stdout
    insns, base = [], None
    for line in dis.splitlines():
        m = re.match(r"^([0-9a-f]{16}) <", line)
        if m:
            base = int(m.group(1), 16)
            continue
        m = INSN.match(line)
        if m and base is not None:
            op, ops, addr = m.groups()
            tgt = TARGET.search(line)
            insns.append(
                {
                    "pc": int(addr, 16),
                    "op": op,
                    "ops": ops,
                    "target": base + int(tgt.group(1), 16) if tgt else None,
                }
            )
    return insns


def build_blocks(insns):
    index = {ins["pc"]: i for i, ins in enumerate(insns)}
    leaders = {0}
    for i, ins in enumerate(insns):
        op = ins["op"]
        if op.startswith(("s_branch", "s_cbranch")):
            if ins["target"] in index:
                leaders.add(index[ins["target"]])
            if i + 1 < len(insns):
                leaders.add(i + 1)
        elif op.startswith(("s_endpgm", "s_setpc", "s_swappc")) and i + 1 < len(insns):
            leaders.add(i + 1)
    starts = sorted(leaders)
    blocks = list(zip(starts, starts[1:] + [len(insns)]))
    of_index = {s: b for b, (s, _) in enumerate(blocks)}

    succs = []
    for b, (_, e) in enumerate(blocks):
        last = insns[e - 1]
        op, out = last["op"], []
        if op.startswith("s_branch"):
            if last["target"] in index:
                out = [of_index[index[last["target"]]]]
        elif op.startswith("s_cbranch"):
            if last["target"] in index:
                out.append(of_index[index[last["target"]]])
            if b + 1 < len(blocks):
                out.append(b + 1)
        elif op.startswith(("s_endpgm", "s_setpc", "s_swappc")):
            out = []
        elif b + 1 < len(blocks):
            out = [b + 1]
        succs.append(out)
    return blocks, succs


def defs_of(ins):
    if ins["op"].startswith(NO_DST):
        return set()
    return regs(ins["ops"].split(",")[0])


def uses_of(ins):
    parts = [p.strip() for p in ins["ops"].split(",")]
    srcs = parts if ins["op"].startswith(NO_DST) else parts[1:]
    out = set()
    for s in srcs:
        out |= regs(s)
    return out


def srsrc_of(ins):
    """The 4-dword resource operand of a buffer_* instruction."""
    parts = [p.strip() for p in ins["ops"].split(",")]
    for p in parts[1:]:
        m = RANGE.fullmatch(p)
        if m and int(m.group(2)) - int(m.group(1)) == 3:
            return set(range(int(m.group(1)), int(m.group(2)) + 1))
    return set()


def transfer(ins, defined, tainted, lanes):
    """Apply one instruction to the running (defined, tainted, lanes) state."""
    op, ops = ins["op"], ins["ops"]
    if op.startswith("v_writelane_b32"):
        m = LANE.search(ops)
        if m:
            src = m.group(3)
            dirty = src is not None and (int(src) not in defined or int(src) in tainted)
            lanes[(int(m.group(1)), int(m.group(4)))] = dirty
        return
    if op.startswith("v_readlane_b32"):
        m = re.search(r"s(\d+),\s*v(\d+),\s*(\d+)", ops)
        if m:
            dst = int(m.group(1))
            defined.add(dst)
            tainted.discard(dst)
            if lanes.get((int(m.group(2)), int(m.group(3))), False):
                tainted.add(dst)
        return
    used = uses_of(ins)
    dirty = bool((used - defined) | (used & tainted))
    for r in defs_of(ins):
        defined.add(r)
        if dirty:
            tainted.add(r)
        else:
            tainted.discard(r)


def scan(path):
    insns = disassemble(path)
    if not insns:
        return []
    preinit = preinit_count(path)
    blocks, succs = build_blocks(insns)
    preds = [[] for _ in blocks]
    for b, out in enumerate(succs):
        for s in out:
            preds[s].append(b)

    entry = frozenset(range(preinit))
    in_def = [entry if b == 0 else ALL for b in range(len(blocks))]
    in_taint = [set() for _ in blocks]
    in_lane = [{} for _ in blocks]

    for _ in range(len(blocks) + 2):  # bounded fixpoint
        changed = False
        for b in range(len(blocks)):
            if b == 0:
                d, t, l = set(entry), set(), {}
            elif not preds[b]:
                d, t, l = set(ALL), set(), {}  # an edge this parser missed
            else:
                outs = [
                    run_block(insns, blocks[p], in_def[p], in_taint[p], in_lane[p])
                    for p in preds[b]
                ]
                d = ALL.intersection(*(o[0] for o in outs))
                t = set().union(*(o[1] for o in outs))
                l = {}
                for o in outs:
                    for k, v in o[2].items():
                        l[k] = l.get(k, False) or v
            if (d, t, l) != (in_def[b], in_taint[b], in_lane[b]):
                in_def[b], in_taint[b], in_lane[b] = d, t, l
                changed = True
        if not changed:
            break

    findings = []
    for b, (s, e) in enumerate(blocks):
        d, t, l = set(in_def[b]), set(in_taint[b]), dict(in_lane[b])
        for i in range(s, e):
            ins = insns[i]
            if ins["op"].startswith("buffer_"):
                rsrc = srsrc_of(ins)
                if rsrc & t:
                    findings.append(
                        (ins["pc"], ins["op"], ins["ops"], sorted(rsrc & t))
                    )
            transfer(ins, d, t, l)
    return findings


def run_block(insns, span, d0, t0, l0):
    d, t, l = set(d0), set(t0), dict(l0)
    for i in range(*span):
        transfer(insns[i], d, t, l)
    return d, t, l


def main(paths):
    flagged = 0
    for p in paths:
        try:
            f = scan(p)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"### {Path(p).name}  SCAN ERROR {exc}")
            continue
        if f:
            flagged += 1
            print(f"### {Path(p).name}")
            for pc, op, ops, bad in f[:6]:
                print(f"    0x{pc:08X}  {op} {ops}   tainted rsrc: {bad}")
            if len(f) > 6:
                print(f"    ... {len(f) - 6} more")
    print(
        f"\n{flagged} of {len(paths)} objects have a buffer access on a tainted descriptor"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
