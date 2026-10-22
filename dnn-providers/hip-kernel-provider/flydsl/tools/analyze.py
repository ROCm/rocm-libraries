# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Per-object resources (AMDGPU metadata note + ISA) and the #1087 detector.

analyze(objdir) -> dict.   Metadata: msgpack in the NT_AMDGPU_METADATA ELF note
(same note hkp_pack._metadata_document reads).  ISA: llvm-objdump disassembly of the hsaco, with the kernel-descriptor
fields the CFG analysis needs (user/system SGPR counts, accum offset) decoded from .rodata (`isa_text`).

#1087 rule (see VERDICT.md): the issue's miscompile is an SGPR that is spilled to a VGPR
lane (v_writelane_b32 vX, sN, lane) while no definition of sN reaches the write -- the
`s_mov_b64` that should stage the value into the spill range is gone.  `pattern` =
forward must-define dataflow over the kernel's CFG finds >=1 v_writelane whose SGPR
source is not defined on every path from entry.  `exposed` = the object has SGPR spills
(sgpr_spill_count > 0): the same spill-staging chain exists and the bug could occur.
"""
import re, struct, json, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import msgpack


# ---------------- ELF metadata ----------------
def amdgpu_metadata(blob: bytes) -> dict:
    assert blob[:4] == b"\x7fELF"
    (shoff,) = struct.unpack_from("<Q", blob, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", blob, 0x3A)
    for i in range(shnum):
        off = shoff + i * shentsize
        name, typ, flags, addr, o, size = struct.unpack_from("<IIQQQQ", blob, off)
        if typ != 7:
            continue
        p, end = o, o + size
        while p + 12 <= end:
            nsz, dsz, nt = struct.unpack_from("<III", blob, p)
            p += 12
            nm = blob[p : p + nsz].rstrip(b"\0")
            p += (nsz + 3) & ~3
            desc = blob[p : p + dsz]
            p += (dsz + 3) & ~3
            if nm == b"AMDGPU" and nt == 32:
                return msgpack.unpackb(desc, raw=False)
    raise ValueError("no NT_AMDGPU_METADATA note")


# ---------------- ISA text from the hsaco ----------------
def _section(blob, want):
    (shoff,) = struct.unpack_from("<Q", blob, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", blob, 0x3A)
    sh = lambda i: struct.unpack_from("<IIQQQQ", blob, shoff + i * shentsize)
    stroff = sh(shstrndx)[4]
    for i in range(shnum):
        name, typ, flags, addr, o, size = sh(i)
        nm = blob[stroff + name : blob.index(b"\0", stroff + name)].decode()
        if nm == want:
            return blob[o : o + size]
    raise ValueError(f"no section {want}")


def isa_text(hsaco_path):
    """Assembly-like text of the object: `.amdhsa_*` directives for the descriptor fields `initial_defined` and the accum offset
    read, then the disassembly with branch targets turned into labels (parse_kernel's input).
    """
    hsaco_path = Path(hsaco_path)
    kd = _section(hsaco_path.read_bytes(), ".rodata")
    rsrc3, rsrc2 = (
        struct.unpack_from("<I", kd, 44)[0],
        struct.unpack_from("<I", kd, 52)[0],
    )
    head = [
        f".amdhsa_user_sgpr_count {(rsrc2 >> 1) & 0x1f}",
        f".amdhsa_system_sgpr_private_segment_wavefront_offset {rsrc2 & 1}",
    ]
    head += [
        f".amdhsa_system_sgpr_workgroup_id_{n} {(rsrc2 >> b) & 1}"
        for n, b in (("x", 7), ("y", 8), ("z", 9))
    ]
    head += [
        f".amdhsa_system_sgpr_workgroup_info {(rsrc2 >> 10) & 1}",
        f".amdhsa_accum_offset {((rsrc3 & 0x3f) + 1) * 4}",
    ]
    import subprocess

    llvm = os.environ.get("LLVM_BIN", "/opt/rocm/llvm/bin")
    dis = subprocess.run(
        [os.path.join(llvm, "llvm-objdump"), "-d", "--mcpu=gfx950", str(hsaco_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rows, base, targets = [], 0, set()
    for line in dis.splitlines():
        m = re.match(r"^([0-9a-f]{16}) <", line)
        if m:
            base = int(m.group(1), 16)
            continue
        m = re.match(r"^\s+(\S+)\s*(.*?)\s*//\s*([0-9A-F]+):", line)
        if not m:
            continue
        mn, ops, pc = m.group(1), m.group(2), int(m.group(3), 16)
        t = re.search(r"<[^>]*\+0x([0-9a-f]+)>\s*$", line)
        tgt = (
            base + int(t.group(1), 16)
            if t and mn.startswith(("s_branch", "s_cbranch"))
            else None
        )
        if tgt is not None:
            targets.add(tgt)
            ops = f".Lt{tgt:x}"
        rows.append((pc, mn, ops))
    out = head + ["kernel:"]
    for pc, mn, ops in rows:
        if pc in targets:
            out.append(f".Lt{pc:x}:")
        out.append(f"\t{mn} {ops}")
    return "\n".join(out) + "\n"


# ---------------- ISA parse ----------------
SREG = re.compile(r"^s(\d+)$")
SRNG = re.compile(r"^s\[(\d+):(\d+)\]$")


def sregs(op):
    op = op.strip()
    m = SREG.match(op)
    if m:
        return {int(m.group(1))}
    m = SRNG.match(op)
    if m:
        return set(range(int(m.group(1)), int(m.group(2)) + 1))
    return set()


NO_DST = re.compile(
    r"^(s_(cmp|bitcmp|cbranch|branch|waitcnt|nop|barrier|setprio|endpgm|sleep|setreg|set_gpr_idx|"
    r"store|buffer_store|dcache|icache|sendmsg|trap|wait|setkill|ttracedata|incperflevel|decperflevel|"
    r"cmovk|cbranch|setpc|swappc_dummy)\w*)$"
)
COUNT_STOP = re.compile(r"^(s_cbranch\w*|s_branch|s_endpgm)$")


def parse_ops(rest):
    rest = rest.split("//")[0].split(";")[0].strip()
    ops, depth, cur = [], 0, ""
    for ch in rest:
        if ch == "[":
            depth += 1
        if ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            ops.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        ops.append(cur.strip())
    return ops


def defs_of(mn, ops):
    """SGPR numbers defined by this instruction."""
    d = set()
    if not ops:
        return d
    if mn.startswith("s_") and not NO_DST.match(mn):
        d |= sregs(ops[0])
    elif mn in ("v_readlane_b32", "v_readfirstlane_b32"):
        d |= sregs(ops[0])
    elif mn.startswith("v_cmp") and not mn.startswith("v_cmpx"):
        d |= sregs(ops[0])
    elif (
        mn.startswith("v_")
        and len(ops) > 1
        and re.search(r"_co_|v_mad_[iu]64|v_div_scale", mn)
    ):
        d |= sregs(ops[1])
    return d


class Insn:
    __slots__ = ("mn", "ops", "line", "label", "defs", "wl")


def parse_kernel(text):
    """Instructions + labels of the (single) kernel's .text body."""
    insns, labels = [], {}
    body = False
    for ln, raw in enumerate(text.splitlines(), 1):
        s = raw.strip()
        if not s or s.startswith(";") or s.startswith("//"):
            continue
        if not body:
            if re.match(r"^[A-Za-z_.$][\w.$]*:", s) and not s.startswith("."):
                body = True
            else:
                continue
        m = re.match(r"^([A-Za-z_.$][\w.$]*):", s)
        if m:
            labels[m.group(1)] = len(insns)
            continue
        if s.startswith("."):
            if s.startswith((".Lfunc_end", ".section", ".amdhsa_kernel")):
                break
            continue
        mn, _, rest = s.partition(" ")
        i = Insn()
        i.mn = mn
        i.ops = parse_ops(rest)
        i.line = ln
        i.defs = defs_of(mn, i.ops)
        insns.append(i)
    return insns, labels


def initial_defined(text):
    n_user = int(re.search(r"\.amdhsa_user_sgpr_count (\d+)", text).group(1))
    n_sys = sum(
        int(m.group(1))
        for m in re.finditer(
            r"\.amdhsa_system_sgpr_workgroup_(?:id_[xyz]|info) (\d)", text
        )
    )
    n_sys += (
        int(
            re.search(
                r"\.amdhsa_system_sgpr_private_segment_wavefront_offset (\d)", text
            ).group(1)
        )
        if re.search(r"\.amdhsa_system_sgpr_private_segment_wavefront_offset", text)
        else 0
    )
    return set(range(n_user + n_sys + 2))  # +2 slack for flat-scratch/private-seg init


def undefined_writelanes(text):
    insns, labels = parse_kernel(text)
    if not insns:
        return None
    # basic blocks
    leaders = {0} | set(labels.values())
    for k, i in enumerate(insns):
        if COUNT_STOP.match(i.mn) and k + 1 < len(insns):
            leaders.add(k + 1)
    starts = sorted(l for l in leaders if l < len(insns))
    bidx = {s: b for b, s in enumerate(starts)}
    ends = [
        starts[b + 1] if b + 1 < len(starts) else len(insns) for b in range(len(starts))
    ]
    succ = [[] for _ in starts]
    for b, (s, e) in enumerate(zip(starts, ends)):
        last = insns[e - 1]
        if last.mn == "s_branch":
            t = last.ops[0]
            succ[b] = [bidx[labels[t]]] if t in labels else []
        elif last.mn.startswith("s_cbranch"):
            t = last.ops[0]
            if t in labels:
                succ[b].append(bidx[labels[t]])
            if b + 1 < len(starts):
                succ[b].append(b + 1)
        elif last.mn == "s_endpgm":
            pass
        elif b + 1 < len(starts):
            succ[b].append(b + 1)
    preds = [[] for _ in starts]
    for b, ss in enumerate(succ):
        for t in ss:
            preds[t].append(b)
    ALL = set(range(512))
    gen = []
    for s, e in zip(starts, ends):
        g = set()
        for i in insns[s:e]:
            g |= i.defs
        gen.append(g)
    entry = initial_defined(text)
    IN = [set(ALL) for _ in starts]
    IN[0] = set(entry)
    OUT = [IN[b] | gen[b] for b in range(len(starts))]
    changed = True
    while changed:
        changed = False
        for b in range(len(starts)):
            if b == 0:
                new_in = set(entry)
            elif preds[b]:
                new_in = set(ALL)
                for p in preds[b]:
                    new_in &= OUT[p]
            else:
                new_in = set(ALL)  # unreachable
            new_out = new_in | gen[b]
            if new_in != IN[b] or new_out != OUT[b]:
                IN[b], OUT[b] = new_in, new_out
                changed = True
    bad = []
    nwl = 0
    for b, (s, e) in enumerate(zip(starts, ends)):
        cur = set(IN[b])
        for i in insns[s:e]:
            if i.mn == "v_writelane_b32":
                nwl += 1
                src = sregs(i.ops[1]) if len(i.ops) > 1 else set()
                if src and not (src <= cur):
                    bad.append((i.line, i.ops))
            cur |= i.defs
    return {"writelanes": nwl, "undefined": bad}


# ---------------- resources ----------------
def occupancy(vgpr_total, lds, threads):
    """Waves/SIMD on gfx950: 512 VGPR/lane/SIMD (unified arch+acc, 8-granule), 160 KiB LDS/CU, 4 SIMD/CU, max 8."""
    alloc = max(8, (vgpr_total + 7) // 8 * 8)
    by_vgpr = min(8, 512 // alloc)
    waves_wg = (threads + 63) // 64
    wg_lds = (163840 // lds) if lds else 1 << 30
    by_lds = (wg_lds * waves_wg) // 4
    return max(0, min(by_vgpr, by_lds, 8)), by_vgpr, by_lds


def analyze(objdir):
    objdir = Path(objdir)
    blob = (objdir / "obj.hsaco").read_bytes()
    doc = amdgpu_metadata(blob)
    k = doc["amdhsa.kernels"][0]
    text = isa_text(objdir / "obj.hsaco")
    m = re.search(r"\.amdhsa_accum_offset (\d+)", text)
    accum_off = int(m.group(1)) if m else None
    vg, ag = k[".vgpr_count"], k.get(".agpr_count", 0)
    total = (
        (accum_off + ag)
        if (accum_off is not None and ag)
        else (accum_off if accum_off else vg)
    )
    threads = k.get(".max_flat_workgroup_size", 256)
    occ, ov, ol = occupancy(total, k[".group_segment_fixed_size"], threads)
    arch_v = accum_off if (accum_off is not None and ag) else vg
    r = dict(
        vgpr=arch_v,
        agpr=ag,
        vgpr_total=total,
        sgpr=k[".sgpr_count"],
        scratch_bytes=k[".private_segment_fixed_size"],
        sgpr_spill=k[".sgpr_spill_count"],
        vgpr_spill=k[".vgpr_spill_count"],
        lds_bytes=k[".group_segment_fixed_size"],
        occupancy=occ,
        occ_by_vgpr=ov,
        occ_by_lds=ol,
        threads=threads,
        target=doc.get("amdhsa.target"),
        kernarg=k[".kernarg_segment_size"],
    )
    ul = undefined_writelanes(text)
    r["writelanes"] = ul["writelanes"] if ul else 0
    r["undef_writelanes"] = len(ul["undefined"]) if ul else 0
    r["undef_detail"] = [(l, ",".join(o)) for l, o in (ul["undefined"] if ul else [])][
        :8
    ]
    import os

    os.environ.setdefault("LLVM_BIN", "/opt/rocm/llvm/bin")
    import undefscan_issue1087 as u

    r["tainted_buffer_accesses"] = len(u.scan(objdir / "obj.hsaco"))
    r["exposed_1087"] = (
        "pattern"
        if (r["undef_writelanes"] or r["tainted_buffer_accesses"])
        else ("exposed" if r["sgpr_spill"] > 0 else "no")
    )
    return r


if __name__ == "__main__":
    print(json.dumps(analyze(sys.argv[1]), indent=1))
