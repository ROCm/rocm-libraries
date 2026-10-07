#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Compile-only ISA / resource report for Tile Engine GEMM kernels.

Reads either the AMDGPU assembly written by ``clang -save-temps`` (``*.s``) or
a device code object (``*.o`` / ``*.out`` / ``*.hsaco`` / ``*.co``, read with
``llvm-readelf --notes`` and ``llvm-objdump -d``) and reports, per kernel:

* resources from the AMDGPU metadata: VGPR / AGPR / SGPR counts, VGPR and SGPR
  spills, private (scratch) and group (LDS) segment sizes, and the share of the
  target's LDS capacity;
* optionally, the occupancy printed by ``-Rpass-analysis=kernel-resource-usage``
  (``--remarks build.log``);
* the instruction mix of every natural loop (one per CFG back edge, i.e. an
  edge to a dominating block, so out-of-line blocks that branch back are not
  mistaken for loops; all paths of the body are counted), with the
  exact opcodes of the memory / matrix ops (``v_wmma_*``, ``ds_load_b128``,
  ``ds_store_*``, ``tensor_load_to_lds``, ``global_load_async_to_lds_b128``,
  ``scratch_*``), every ``s_wait_*`` immediate and the number of workgroup
  barriers (``barrier_syncs``: ``s_barrier`` / ``s_barrier_wait``; the
  matching ``s_barrier_signal`` is only counted in the ``barrier`` class);
* the hot loop (the innermost MMA loop with the most matrix MACs), its K
  elements per iteration and per-K-tile normalised counts when ``--tile`` is
  given (K per iteration = MACs / per-wave C tile, from the MxNxK in the
  ``v_wmma`` / ``v_mfma`` names);
* prologue length (instructions before the first inner MMA loop) and the
  epilogue: store opcodes (widths), barriers and tail MMAs of the code after
  the hot loop that is not in an inner MMA loop, or of the enclosing tile loop
  minus its K loop for persistent kernels.

No GPU is required. Typical use, on the output of compile_one.py:

  python3 isa_report.py out/*gfx1250.s --remarks out/build.log \\
      --kernel GemmKernel --tile 256x256x64_2x4x1_16x16x32

``--ref other.s`` prints the hot-loop / resource delta against a second input
(e.g. a rocKE code object), ``--json`` emits machine-readable output and
``--check`` returns 1 if a kernel spills or exceeds the LDS capacity.
"""

import argparse
import collections
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# LDS bytes available to one workgroup (CU / WGP); used for lds_pct and --check.
LDS_CAPACITY = {
    "gfx1250": 327680,
    "gfx950": 163840,
}
DEFAULT_LDS_CAPACITY = 65536

# Instruction classes, first match wins. "mma" covers WMMA, SWMMAC and MFMA.
CLASSES = [
    ("mma", r"^v_(s?wmma|smfmac|mfma)"),
    ("tensor_load", r"^tensor_load_to_lds"),
    ("tensor_store", r"^tensor_store_from_lds"),
    ("async_to_lds", r"^(global|cluster|buffer)_load_async_to_lds"),
    ("ds_load", r"^ds_(load|read)"),
    ("ds_store", r"^ds_(store|write)"),
    ("global_load", r"^(global|buffer|flat)_load"),
    ("global_store", r"^(global|buffer|flat)_store"),
    ("scratch", r"^scratch_"),
    ("wait", r"^s_wait"),
    ("barrier", r"^s_barrier"),
    ("vgpr_msb", r"^s_set_vgpr_msb$"),
    ("branch", r"^s_(cbranch|branch)"),
    ("valu", r"^v_"),
    ("salu", r"^s_"),
]
_CLASS_RES = [(name, re.compile(pat)) for name, pat in CLASSES]

# Classes whose exact opcodes (and thus widths) are listed per region.
DETAIL_CLASSES = {
    "mma",
    "tensor_load",
    "tensor_store",
    "async_to_lds",
    "ds_load",
    "ds_store",
    "global_load",
    "global_store",
    "scratch",
}
STORE_CLASSES = {"global_store", "ds_store", "tensor_store", "scratch"}
_STORE_RE = re.compile(r"_(store|write)")

_MMA_SHAPE_RE = re.compile(r"_(\d+)x(\d+)x(\d+)")
_BRANCH_RE = re.compile(r"^s_(cbranch_\w+|branch)$")
_NO_FALLTHROUGH_RE = re.compile(r"^s_(branch|endpgm\w*|setpc_b64|trap)$")

METADATA_KEYS = [
    "vgpr_count",
    "agpr_count",
    "sgpr_count",
    "vgpr_spill_count",
    "sgpr_spill_count",
    "private_segment_fixed_size",
    "group_segment_fixed_size",
    "wavefront_size",
    "max_flat_workgroup_size",
]


def classify(opcode):
    for name, pattern in _CLASS_RES:
        if pattern.match(opcode):
            return name
    return "other"


def normalize_opcode(opcode):
    """Drop the _e32 / _e64 / _dpp / _sdwa encoding suffix."""
    return re.sub(r"_(e32|e64|e64_dpp|dpp|sdwa)$", "", opcode)


def mma_macs(opcode):
    """Multiply-accumulates of one matrix instruction, from its MxNxK name."""
    m = _MMA_SHAPE_RE.search(opcode)
    if not m:
        return 0
    return int(m.group(1)) * int(m.group(2)) * int(m.group(3))


class Function:
    """Linear instruction list of one function plus its control-flow edges."""

    def __init__(self, name):
        self.name = name
        self.insts = []  # (opcode, operands)
        self.branches = []  # (inst index, target inst index)

    def add(self, opcode, operands):
        self.insts.append((normalize_opcode(opcode), operands.strip()))


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------


def parse_metadata(text):
    """Parse the ``amdhsa.kernels`` YAML list written by the AMDGPU backend.

    Accepts the ``.amdgpu_metadata`` block of a ``.s`` file or the output of
    ``llvm-readelf --notes``. Only the scalar per-kernel keys are kept."""
    kernels = []
    current = None
    in_kernels = False
    indent = None
    for line in text.splitlines():
        if re.match(r"^\s*amdhsa\.kernels:\s*$", line):
            in_kernels = True
            indent = None
            continue
        if not in_kernels:
            continue
        m = re.match(r"^(\s*)- \.(\w+):\s*(.*)$", line)
        if m and (indent is None or len(m.group(1)) == indent):
            indent = len(m.group(1))
            current = {}
            kernels.append(current)
            line = " " * (indent + 2) + "." + m.group(2) + ": " + m.group(3)
        if indent is None:
            continue
        stripped = line.strip()
        lead = len(line) - len(line.lstrip())
        if stripped and lead <= indent and not stripped.startswith("-"):
            in_kernels = False
            current = None
            continue
        m = re.match(r"^\s*\.(\w+):\s*(\S.*)?$", line)
        if current is not None and m and lead == indent + 2 and m.group(2):
            value = m.group(2).strip().strip("'\"")
            current[m.group(1)] = int(value) if re.fullmatch(r"-?\d+", value) else value
    return {k["name"]: k for k in kernels if "name" in k}


def parse_asm(text):
    """Split clang ``.s`` output into functions."""
    functions = []
    current = None
    labels = {}
    pending = []
    # Function symbols are the ".type X,@function" names; a bare extracted
    # function body has none, then any non-local label starts a function.
    declared = set(re.findall(r"^\s*\.type\s+([^,\s]+),\s*@function", text, re.M))
    for line in text.splitlines():
        if current is None:
            m = re.match(r"^([A-Za-z_$][\w.$]*):", line)
            if (
                m
                and not line.startswith(".L")
                and (not declared or m.group(1) in declared)
            ):
                current = Function(m.group(1))
                labels, pending = {}, []
            continue
        if line.startswith(".Lfunc_end"):
            for idx, label in pending:
                if label in labels:
                    current.branches.append((idx, labels[label]))
            functions.append(current)
            current = None
            continue
        m = re.match(r"^([.\w$]+):", line)
        if m:
            labels[m.group(1)] = len(current.insts)
            continue
        body = line.split(";", 1)[0]
        m = re.match(r"^\s+([a-z_][a-z0-9_]*)\b(.*)$", body)
        if not m:
            continue
        opcode = normalize_opcode(m.group(1))
        operands = m.group(2)
        if _BRANCH_RE.match(opcode):
            target = operands.strip().split(",")[0].strip()
            pending.append((len(current.insts), target))
        current.add(opcode, operands)
    return functions


def parse_objdump(text):
    """Split ``llvm-objdump -d`` output into functions. Branch targets are
    decoded from the SOPP simm16 operand (dword offset from PC + 4)."""
    functions = []
    current = None
    addrs = {}
    pending = []

    def close():
        if current is None:
            return
        for idx, target in pending:
            if target in addrs:
                current.branches.append((idx, addrs[target]))
        functions.append(current)

    for line in text.splitlines():
        m = re.match(r"^([0-9a-fA-F]+) <(.+)>:\s*$", line)
        if m:
            close()
            current = Function(m.group(2))
            addrs, pending = {}, []
            continue
        if current is None:
            continue
        m = re.match(r"^\s+([a-z_][a-z0-9_]*)\s*(.*?)\s*//\s*([0-9A-Fa-f]+):", line)
        if not m:
            continue
        opcode = normalize_opcode(m.group(1))
        pc = int(m.group(3), 16)
        addrs[pc] = len(current.insts)
        if _BRANCH_RE.match(opcode):
            imm = re.match(r"^(-?\d+)", m.group(2))
            if imm:
                simm = int(imm.group(1)) & 0xFFFF
                if simm & 0x8000:
                    simm -= 0x10000
                pending.append((len(current.insts), pc + 4 + 4 * simm))
        current.add(opcode, m.group(2))
    close()
    return functions


def llvm_tool(name, llvm_bin):
    candidates = []
    if llvm_bin:
        candidates.append(Path(llvm_bin) / name)
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    candidates.append(Path(rocm) / "lib" / "llvm" / "bin" / name)
    for c in candidates:
        if c.is_file():
            return str(c)
    found = shutil.which(name)
    if found:
        return found
    sys.exit(f"isa_report: cannot find {name}; pass --llvm-bin or set ROCM_PATH")


def load(path, llvm_bin=None):
    """Return (functions, metadata, target) for a .s file or a code object."""
    with open(path, "rb") as f:
        is_elf = f.read(4) == b"\x7fELF"
    if is_elf:
        notes = subprocess.run(
            [llvm_tool("llvm-readelf", llvm_bin), "--notes", path],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        disasm = subprocess.run(
            [
                llvm_tool("llvm-objdump", llvm_bin),
                "-d",
                "--no-show-raw-insn",
                path,
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        text = notes + "\n" + disasm
        functions = parse_objdump(disasm)
    else:
        text = Path(path).read_text(errors="ignore")
        functions = parse_asm(text)
    m = re.search(r'\.amdgcn_target\s+"[^"]*?(gfx[0-9a-z]+)', text)
    target = m.group(1) if m else None
    return functions, parse_metadata(text), target


def parse_remarks(text):
    """Map kernel name -> {occupancy, ...} from -Rpass-analysis remarks."""
    out = {}
    name = None
    for line in text.splitlines():
        m = re.search(r"Function Name: (\S+)", line)
        if m:
            name = m.group(1)
            out[name] = {}
            continue
        m = re.search(r":\s+Occupancy \[waves/SIMD\]: (\d+)", line)
        if name and m:
            out[name]["occupancy"] = int(m.group(1))
    return out


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def basic_blocks(func):
    """Split ``func`` into basic blocks ``[start, end)`` and their successors."""
    n = len(func.insts)
    targets = dict(func.branches)
    starts = {0}
    starts.update(func.branches[i][1] for i in range(len(func.branches)))
    for i, (opcode, _) in enumerate(func.insts):
        if _BRANCH_RE.match(opcode) or _NO_FALLTHROUGH_RE.match(opcode):
            starts.add(i + 1)
    starts = sorted(s for s in starts if s < n)
    blocks = list(zip(starts, starts[1:] + [n]))
    block_of = {start: b for b, (start, _) in enumerate(blocks)}
    succs = []
    for b, (_, end) in enumerate(blocks):
        out = []
        if end - 1 in targets:
            out.append(block_of[targets[end - 1]])
        if not _NO_FALLTHROUGH_RE.match(func.insts[end - 1][0]) and end < n:
            out.append(b + 1)
        succs.append(out)
    return blocks, succs


def dominators(succs):
    """Immediate dominators (Cooper, Harvey, Kennedy) of blocks reachable from
    block 0; unreachable blocks map to None."""
    order, seen = [], set()
    stack = [(0, iter(succs[0]))] if succs else []
    seen.add(0)
    while stack:
        b, it = stack[-1]
        nxt = next((s for s in it if s not in seen), None)
        if nxt is None:
            order.append(b)
            stack.pop()
        else:
            seen.add(nxt)
            stack.append((nxt, iter(succs[nxt])))
    rpo = order[::-1]
    index = {b: i for i, b in enumerate(rpo)}
    preds = collections.defaultdict(list)
    for b in rpo:
        for s in succs[b]:
            preds[s].append(b)
    idom = {b: None for b in range(len(succs))}
    idom[0] = 0

    def intersect(a, b):
        while a != b:
            while index[a] > index[b]:
                a = idom[a]
            while index[b] > index[a]:
                b = idom[b]
        return a

    changed = True
    while changed:
        changed = False
        for b in rpo[1:]:
            done = [p for p in preds[b] if idom[p] is not None]
            new = done[0]
            for p in done[1:]:
                new = intersect(p, new)
            if idom[b] != new:
                idom[b] = new
                changed = True
    return idom


def dominates(idom, a, b):
    while True:
        if a == b:
            return True
        if b == 0 or idom[b] is None:
            return False
        b = idom[b]


def find_loops(func):
    """One natural loop per back edge (an edge u -> h where h dominates u):
    the header h plus every block that reaches u without passing through h.
    Out-of-line blocks that branch back to the code they were split from are
    not back edges and are ignored. Returns ``(blocks, [(header, body)])``."""
    if not func.insts:
        return [], []
    blocks, succs = basic_blocks(func)
    idom = dominators(succs)
    preds = [[] for _ in blocks]
    for b, out in enumerate(succs):
        for s in out:
            preds[s].append(b)
    loops = set()
    for latch, out in enumerate(succs):
        for header in out:
            if idom[latch] is None or not dominates(idom, header, latch):
                continue
            body = {header}
            stack = [latch]
            while stack:
                b = stack.pop()
                if b not in body:
                    body.add(b)
                    stack.extend(preds[b])
            loops.add((header, tuple(sorted(body))))
    return blocks, sorted(loops)


def region_mix(insts):
    classes = collections.Counter()
    opcodes = collections.Counter()
    waits = collections.Counter()
    macs = 0
    for opcode, operands in insts:
        cls = classify(opcode)
        classes[cls] += 1
        if cls in DETAIL_CLASSES:
            opcodes[opcode] += 1
        if cls == "mma":
            macs += mma_macs(opcode)
        if cls == "wait":
            # Keep every operand: gfx10/11 s_waitcnt_vscnt etc. carry the
            # count in the second one ("null, 0x0").
            waits[f"{opcode} {' '.join(operands.split())}".strip()] += 1
    barriers = sum(
        n
        for op, n in collections.Counter(o for o, _ in insts).items()
        if op in ("s_barrier", "s_barrier_wait")
    )
    return {
        "insts": len(insts),
        "classes": dict(sorted(classes.items())),
        "opcodes": dict(sorted(opcodes.items())),
        "waits": dict(sorted(waits.items())),
        "barrier_syncs": barriers,
        "mma_macs": macs,
    }


def parse_tile(spec):
    """Parse a Tile Engine tile string ``MxNxK_WMxWNxWK_wmxwnxwk``."""
    parts = spec.split("_")
    block = [int(x) for x in parts[0].split("x")]
    warps = [int(x) for x in parts[1].split("x")] if len(parts) > 1 else [1, 1, 1]
    if len(block) != 3 or len(warps) != 3:
        raise ValueError(f"bad --tile {spec!r}, expected MxNxK_WMxWNxWK[_...]")
    return block, warps


def k_per_iteration(loop_macs, tile):
    """Block-K elements one hot-loop iteration covers, from the MACs one wave
    issues and the per-wave C tile (block M/N divided by the warp layout)."""
    (m, n, _), (wm, wn, wk) = tile
    per_wave_mn = (m // wm) * (n // wn)
    if not per_wave_mn or not loop_macs:
        return None
    return loop_macs * wk / per_wave_mn


def scale(mix, factor):
    def div(d):
        return {k: round(v * factor, 3) for k, v in d.items()}

    return {
        "insts": round(mix["insts"] * factor, 3),
        "classes": div(mix["classes"]),
        "opcodes": div(mix["opcodes"]),
        "waits": div(mix["waits"]),
        "barrier_syncs": round(mix["barrier_syncs"] * factor, 3),
    }


def analyze(func, meta, target, remarks, tile, lds_capacity):
    blocks, loops = find_loops(func)
    loop_reports = []
    for header, body in loops:
        insts = [func.insts[i] for b in body for i in range(*blocks[b])]
        mix = region_mix(insts)
        mix["range"] = [blocks[header][0], max(blocks[b][1] for b in body) - 1]
        mix["blocks"] = len(body)
        loop_reports.append(mix)
    bodies = [set(body) for _, body in loops]

    report = {"name": func.name, "target": target}
    res = {k: meta[k] for k in METADATA_KEYS if k in meta}
    res["vgpr_msb_count"] = sum(1 for op, _ in func.insts if op == "s_set_vgpr_msb")
    cap = lds_capacity or LDS_CAPACITY.get(target, DEFAULT_LDS_CAPACITY)
    res["lds_capacity"] = cap
    if "group_segment_fixed_size" in res:
        lds = res["group_segment_fixed_size"]
        res["lds_pct"] = round(100.0 * lds / cap, 1)
        res["lds_blocks_per_cu"] = cap // lds if lds else None
    res.update(remarks.get(func.name, {}))
    report["resources"] = res
    report["whole"] = region_mix(func.insts)
    report["loops"] = loop_reports

    # Hot-loop candidates: MMA loops with no MMA loop nested inside them, so a
    # persistent outer tile loop is never picked over its K loop.
    mma = [i for i, lp in enumerate(loop_reports) if lp["mma_macs"]]
    inner = [i for i in mma if not any(bodies[j] < bodies[i] for j in mma if j != i)]
    if inner:
        # Most MACs per iteration; ties go to the first loop in the text.
        hot_i = max(inner, key=lambda i: (loop_reports[i]["mma_macs"], -i))
        hot = loop_reports[hot_i]
        hot_report = dict(hot)
        if tile:
            kpi = k_per_iteration(hot["mma_macs"], tile)
            hot_report["k_per_iter"] = kpi
            if kpi:
                hot_report["per_k_tile"] = scale(hot, tile[0][2] / kpi)
        report["hot_loop"] = hot_report
        report["prologue_insts"] = min(loop_reports[i]["range"][0] for i in inner)
        # Epilogue: for a persistent kernel the enclosing tile loop minus its
        # inner MMA loops; otherwise everything laid out after the hot loop
        # header that is not in an inner MMA loop (pipeline tail, epilogue and
        # out-of-line blocks).
        in_loop = set().union(*(bodies[i] for i in inner))
        outer = [j for j in mma if bodies[hot_i] < bodies[j]]
        if outer:
            region = min((bodies[j] for j in outer), key=len) - in_loop
        else:
            start = hot["range"][0]
            region = {
                b for b, (lo, _) in enumerate(blocks) if lo > start and b not in in_loop
            }
        tail = [func.insts[i] for b in sorted(region) for i in range(*blocks[b])]
        report["epilogue"] = region_mix(tail)
    else:
        report["hot_loop"] = None
        report["prologue_insts"] = None
        report["epilogue"] = region_mix(func.insts)
    report["epilogue"]["stores"] = {
        op: n
        for op, n in report["epilogue"]["opcodes"].items()
        if classify(op) in STORE_CLASSES and _STORE_RE.search(op)
    }
    return report


def build_reports(path, args, kernel, tile_spec):
    functions, metadata, target = load(path, args.llvm_bin)
    target = args.target or target
    remarks = parse_remarks(Path(args.remarks).read_text()) if args.remarks else {}
    tile = parse_tile(tile_spec) if tile_spec else None
    reports = []
    # Without any metadata (e.g. an extracted function body) report everything.
    all_functions = args.all_functions or not metadata
    for func in functions:
        if func.name not in metadata and not all_functions:
            continue
        if kernel and kernel not in func.name:
            continue
        meta = metadata.get(func.name, {})
        reports.append(analyze(func, meta, target, remarks, tile, args.lds_capacity))
    return reports


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _fmt(d):
    return " ".join(f"{k}={v}" for k, v in d.items()) if d else "-"


def short_name(name, width=100):
    return name if len(name) <= width else name[: width - 3] + "..."


def print_report(r, out=None):
    p = lambda s="": print(s, file=out or sys.stdout)  # noqa: E731
    p(f"== {short_name(r['name'])}")
    p(f"   resources: {_fmt(r['resources'])}")
    p(f"   whole: insts={r['whole']['insts']} {_fmt(r['whole']['classes'])}")
    if r["prologue_insts"] is not None:
        p(f"   prologue: {r['prologue_insts']} insts before the first inner MMA loop")
    for lp in r["loops"]:
        tag = (
            " (hot)" if r["hot_loop"] and lp["range"] == r["hot_loop"]["range"] else ""
        )
        p(
            f"   loop [{lp['range'][0]},{lp['range'][1]}]{tag} insts={lp['insts']} "
            f"barrier_syncs={lp['barrier_syncs']}: {_fmt(lp['classes'])}"
        )
        p(f"      ops: {_fmt(lp['opcodes'])}")
        p(f"      waits: {_fmt(lp['waits'])}")
    hot = r["hot_loop"]
    if hot and hot.get("k_per_iter"):
        k = hot["per_k_tile"]
        p(
            f"   hot loop: k_per_iter={hot['k_per_iter']:g}; per K-tile: "
            f"barrier_syncs={k['barrier_syncs']} {_fmt(k['opcodes'])}"
        )
        p(f"      waits per K-tile: {_fmt(k['waits'])}")
    ep = r["epilogue"]
    p(
        f"   epilogue: insts={ep['insts']} tail_mma={ep['classes'].get('mma', 0)} barrier_syncs={ep['barrier_syncs']} "
        f"stores: {_fmt(ep['stores'])}"
    )


def compare(cur, ref, out=None):
    """Print hot-loop / resource deltas of two single-kernel reports."""
    p = lambda s="": print(s, file=out or sys.stdout)  # noqa: E731
    p(f"== delta: {short_name(cur['name'], 60)} vs ref {short_name(ref['name'], 60)}")
    keys = sorted(set(cur["resources"]) | set(ref["resources"]))
    for k in keys:
        a, b = cur["resources"].get(k), ref["resources"].get(k)
        if a != b:
            p(f"   resource {k}: {a} (ref {b})")

    def norm(r):
        hot = r["hot_loop"]
        if not hot:
            return {}
        src = hot.get("per_k_tile") or hot
        merged = dict(src["opcodes"])
        merged.update(src["waits"])
        merged["barrier_syncs"] = src["barrier_syncs"]
        merged["insts"] = src["insts"]
        return merged

    a, b = norm(cur), norm(ref)
    for k in sorted(set(a) | set(b)):
        if a.get(k, 0) != b.get(k, 0):
            p(f"   hot {k}: {a.get(k, 0)} (ref {b.get(k, 0)})")


def check(r):
    """Static gates: no spills / scratch and LDS within capacity."""
    res = r["resources"]
    errors = []
    for k in ("vgpr_spill_count", "sgpr_spill_count", "private_segment_fixed_size"):
        if res.get(k):
            errors.append(f"{k}={res[k]}")
    if res.get("group_segment_fixed_size", 0) > res["lds_capacity"]:
        errors.append(
            f"group_segment_fixed_size={res['group_segment_fixed_size']} > "
            f"{res['lds_capacity']}"
        )
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("inputs", nargs="+", help=".s file(s) or code object(s)")
    parser.add_argument("--kernel", help="only report kernels containing SUBSTR")
    parser.add_argument(
        "--remarks", help="build log with kernel-resource-usage remarks"
    )
    parser.add_argument("--tile", help="tile string MxNxK_WMxWNxWK[_wmxwnxwk]")
    parser.add_argument("--target", help="override the detected gfx target")
    parser.add_argument("--lds-capacity", type=int, help="override LDS bytes per CU")
    parser.add_argument("--llvm-bin", help="directory with llvm-readelf/llvm-objdump")
    parser.add_argument(
        "--all-functions",
        action="store_true",
        help="also report non-kernel functions (no metadata)",
    )
    parser.add_argument("--ref", help="reference .s / code object to diff against")
    parser.add_argument("--ref-kernel", help="kernel SUBSTR in --ref (default: first)")
    parser.add_argument("--ref-tile", help="tile string of --ref (default: --tile)")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument(
        "--check", action="store_true", help="exit 1 on spills or LDS overflow"
    )
    args = parser.parse_args(argv)

    reports = []
    for path in args.inputs:
        for r in build_reports(path, args, args.kernel, args.tile):
            r["input"] = path
            reports.append(r)
    if not reports:
        print("isa_report: no matching kernel found", file=sys.stderr)
        return 2

    ref_reports = (
        build_reports(args.ref, args, args.ref_kernel, args.ref_tile or args.tile)
        if args.ref
        else []
    )

    if args.json:
        json.dump({"kernels": reports, "ref": ref_reports}, sys.stdout, indent=1)
        print()
    else:
        for r in reports:
            print_report(r)
        if ref_reports:
            compare(reports[0], ref_reports[0])

    rc = 0
    if args.check:
        for r in reports:
            errors = check(r)
            if errors:
                print(
                    f"CHECK FAILED {short_name(r['name'], 60)}: {', '.join(errors)}",
                    file=sys.stderr,
                )
                rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
