# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Code-object resource gate for the attention backward kernels.

Everything here reads a compiled HSACO (bytes) or its disassembly text; nothing
emits IR. Four parts:

* :func:`read_code_object_resources` parses the ELF directly (no external
  tool): the AMDGPU metadata note (msgpack) for VGPR/AGPR/SGPR counts, spill
  counts, LDS and scratch bytes, and the kernel descriptor for the
  arch-VGPR / AGPR split (``accum_offset`` on the unified-register-file CDNA
  targets). :func:`occupancy_waves_per_simd` turns that into an occupancy.
* :func:`find_loops` / :func:`select_q_loop` slice the disassembly into loop
  bodies (basic-block range from a back-edge target to the back-edge branch),
  and :func:`count_agpr_copies` counts ``v_accvgpr_read*`` /
  ``v_accvgpr_write*`` / ``v_accvgpr_mov*`` inside one body (the hot-loop
  AGPR-copy counter; ``probe_isa_inspect`` has no such bucket and counts the
  whole kernel).
* :func:`evaluate_resource_gate` applies the backward resource rules (scratch
  0, spills 0, arch VGPR and AGPR within their own budgets, occupancy >= 1,
  LDS within the per-arch capacity, q-loop AGPR copies within the model
  ceiling) and records the LLVM flavor and comgr version the result was
  produced with. A gate result is valid only on the ``llvm22`` flavor.
* :func:`lds_barrier_crossings` compares the LDS bytes between consecutive
  prologue barriers in the LLVM IR and in the ISA: an LDS access the compiler
  moved across a barrier that orders it against other waves shows up there.

Disassembly goes through ``disassemble_hsaco`` from the ``probe_isa_inspect``
probe (imported by relative path), which needs ``llvm-objdump``;
:func:`find_objdump` locates one and exports it through ``LLVM_OBJDUMP``.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import struct
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GATE_FLAVOR = "llvm22"
# The production comgr: the ROCm vintage of the library that compiled the gate's
# kernels must be this one (torch's bundled comgr on the cluster, pinned by path
# without importing torch: :func:`pin_production_comgr`). The flavor alone does
# not say it: a preset ``ROCKE_LLVM_FLAVOR=llvm22`` also reports llvm22 when a
# process loaded a newer system comgr, whose register allocation differs.
GATE_COMGR_ROCM = "7.2"

# Per-arch register and LDS budgets used by the gate. The CDNA register file is
# two budgets: at most 256 arch VGPRs and at most 256 AGPRs per wave.
ARCH_VGPR_BUDGET = 256
AGPR_BUDGET = 256
# SGPR trip-wire: the largest SGPR count among the shipped backward instances
# on the production comgr (the gfx942 d64 atomic-mode instance; the narrow THD
# tiles are at 102-104). A count above it is not a failure by itself (no
# spill), but it is recorded as the warning ``sgpr_margin`` so kernarg or ABI
# growth is looked at before it spills.
SGPR_WATCH = 106

_NT_AMDGPU_METADATA = 32
_SHT_SYMTAB = 2
_SHT_NOTE = 7


# --------------------------------------------------------------------------- msgpack
class _MsgpackReader:
    """Minimal msgpack decoder for the AMDGPU metadata note (maps, arrays, str, ints)."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def _take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise ValueError("truncated msgpack data")
        out = self.data[self.pos : self.pos + n]
        self.pos += n
        return out

    def _unpack(self, fmt: str) -> Any:
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self._take(size))[0]

    def read(self) -> Any:
        tag = self._take(1)[0]
        if tag <= 0x7F:
            return tag
        if tag >= 0xE0:
            return tag - 0x100
        if 0x80 <= tag <= 0x8F:
            return self._map(tag & 0x0F)
        if 0x90 <= tag <= 0x9F:
            return [self.read() for _ in range(tag & 0x0F)]
        if 0xA0 <= tag <= 0xBF:
            return self._take(tag & 0x1F).decode("utf-8", "replace")
        simple = {0xC0: None, 0xC2: False, 0xC3: True}
        if tag in simple:
            return simple[tag]
        fixed = {
            0xCA: ">f",
            0xCB: ">d",
            0xCC: ">B",
            0xCD: ">H",
            0xCE: ">I",
            0xCF: ">Q",
            0xD0: ">b",
            0xD1: ">h",
            0xD2: ">i",
            0xD3: ">q",
        }
        if tag in fixed:
            return self._unpack(fixed[tag])
        if tag in (0xD9, 0xDA, 0xDB):
            n = self._unpack({0xD9: ">B", 0xDA: ">H", 0xDB: ">I"}[tag])
            return self._take(n).decode("utf-8", "replace")
        if tag in (0xC4, 0xC5, 0xC6):
            n = self._unpack({0xC4: ">B", 0xC5: ">H", 0xC6: ">I"}[tag])
            return self._take(n)
        if tag in (0xDC, 0xDD):
            n = self._unpack(">H" if tag == 0xDC else ">I")
            return [self.read() for _ in range(n)]
        if tag in (0xDE, 0xDF):
            return self._map(self._unpack(">H" if tag == 0xDE else ">I"))
        raise ValueError(f"unsupported msgpack tag 0x{tag:02x}")

    def _map(self, n: int) -> dict:
        out = {}
        for _ in range(n):
            key = self.read()
            out[key] = self.read()
        return out


def decode_msgpack(data: bytes) -> Any:
    """Decode one msgpack document (the subset the AMDGPU metadata note uses)."""
    return _MsgpackReader(data).read()


# --------------------------------------------------------------------------- ELF
@dataclass(frozen=True)
class _Section:
    name: str
    sh_type: int
    addr: int
    offset: int
    size: int
    link: int
    entsize: int


def _elf_sections(blob: bytes) -> list[_Section]:
    if blob[:4] != b"\x7fELF" or blob[4] != 2 or blob[5] != 1:
        raise ValueError("not a little-endian ELF64 code object")
    shoff = struct.unpack_from("<Q", blob, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", blob, 0x3A)
    raw = []
    for i in range(shnum):
        base = shoff + i * shentsize
        name, sh_type, _flags, addr, offset, size, link, _info, _align, entsize = (
            struct.unpack_from("<IIQQQQIIQQ", blob, base)
        )
        raw.append((name, sh_type, addr, offset, size, link, entsize))
    strtab = raw[shstrndx]
    out = []
    for name, sh_type, addr, offset, size, link, entsize in raw:
        start = strtab[3] + name
        end = blob.index(b"\x00", start)
        out.append(
            _Section(
                blob[start:end].decode("ascii", "replace"),
                sh_type,
                addr,
                offset,
                size,
                link,
                entsize,
            )
        )
    return out


def _metadata_note(blob: bytes, sections: Sequence[_Section]) -> dict | None:
    for sec in sections:
        if sec.sh_type != _SHT_NOTE:
            continue
        pos, end = sec.offset, sec.offset + sec.size
        while pos + 12 <= end:
            namesz, descsz, ntype = struct.unpack_from("<III", blob, pos)
            name_start = pos + 12
            desc_start = name_start + ((namesz + 3) & ~3)
            name = blob[name_start : name_start + namesz].rstrip(b"\x00")
            if name == b"AMDGPU" and ntype == _NT_AMDGPU_METADATA:
                return decode_msgpack(blob[desc_start : desc_start + descsz])
            pos = desc_start + ((descsz + 3) & ~3)
    return None


def _kernel_descriptors(blob: bytes, sections: Sequence[_Section]) -> dict[str, bytes]:
    """Map kernel name -> its 64-byte kernel descriptor (symbols ``<name>.kd``)."""
    out: dict[str, bytes] = {}
    for sec in sections:
        if sec.sh_type != _SHT_SYMTAB or sec.entsize == 0:
            continue
        strtab = sections[sec.link]
        for i in range(sec.size // sec.entsize):
            base = sec.offset + i * sec.entsize
            st_name, _info, _other, shndx, value, _size = struct.unpack_from(
                "<IBBHQQ", blob, base
            )
            start = strtab.offset + st_name
            name = blob[start : blob.index(b"\x00", start)].decode("ascii", "replace")
            if not name.endswith(".kd") or shndx == 0 or shndx >= len(sections):
                continue
            host = sections[shndx]
            off = host.offset + (value - host.addr)
            out[name[: -len(".kd")]] = blob[off : off + 64]
    return out


def _accum_offset(descriptor: bytes) -> int | None:
    """``accum_offset`` (first AGPR-backed slot) from ``compute_pgm_rsrc3``."""
    if len(descriptor) < 48:
        return None
    rsrc3 = struct.unpack_from("<I", descriptor, 44)[0]
    return ((rsrc3 & 0x3F) + 1) * 4


# --------------------------------------------------------------------------- resources
@dataclass(frozen=True)
class KernelResources:
    """Per-kernel resources read from the code object."""

    name: str
    arch: str
    vgpr_total: int  # metadata .vgpr_count (arch VGPR + AGPR on unified-file targets)
    arch_vgpr: int
    agpr: int
    sgpr: int
    lds_bytes: int
    scratch_bytes: int
    vgpr_spill: int
    sgpr_spill: int
    wavefront_size: int
    max_flat_workgroup_size: int
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "arch": self.arch,
            "vgpr_total": self.vgpr_total,
            "arch_vgpr": self.arch_vgpr,
            "agpr": self.agpr,
            "sgpr": self.sgpr,
            "lds_bytes": self.lds_bytes,
            "scratch_bytes": self.scratch_bytes,
            "vgpr_spill": self.vgpr_spill,
            "sgpr_spill": self.sgpr_spill,
            "wavefront_size": self.wavefront_size,
            "max_flat_workgroup_size": self.max_flat_workgroup_size,
        }


def _has_unified_agprs(arch: str) -> bool:
    return arch in ("gfx90a", "gfx940", "gfx941", "gfx942", "gfx950")


def read_code_object_resources(
    hsaco: bytes, *, arch: str, kernel: str | None = None
) -> KernelResources:
    """Read the resources of one kernel from an HSACO blob.

    ``kernel`` selects by symbol name; ``None`` requires exactly one kernel.
    Raises ``ValueError`` when the metadata note is missing or ambiguous.
    """
    blob = bytes(hsaco)
    sections = _elf_sections(blob)
    meta = _metadata_note(blob, sections)
    if not meta or "amdhsa.kernels" not in meta:
        raise ValueError("code object has no AMDGPU metadata note")
    kernels = meta["amdhsa.kernels"]
    if kernel is not None:
        kernels = [k for k in kernels if k.get(".name") == kernel]
    if len(kernels) != 1:
        raise ValueError(f"expected one kernel, found {len(kernels)}")
    k = kernels[0]
    name = k.get(".name", "")
    total = int(k.get(".vgpr_count", 0))
    agpr = int(k.get(".agpr_count", 0))
    arch_vgpr = total
    if _has_unified_agprs(arch) and agpr > 0:
        acc = _accum_offset(_kernel_descriptors(blob, sections).get(name, b""))
        arch_vgpr = acc if acc is not None else total - agpr
    return KernelResources(
        name=name,
        arch=arch,
        vgpr_total=total,
        arch_vgpr=arch_vgpr,
        agpr=agpr,
        sgpr=int(k.get(".sgpr_count", 0)),
        lds_bytes=int(k.get(".group_segment_fixed_size", 0)),
        scratch_bytes=int(k.get(".private_segment_fixed_size", 0)),
        vgpr_spill=int(k.get(".vgpr_spill_count", 0)),
        sgpr_spill=int(k.get(".sgpr_spill_count", 0)),
        wavefront_size=int(k.get(".wavefront_size", 64)),
        max_flat_workgroup_size=int(k.get(".max_flat_workgroup_size", 0)),
        raw=dict(k),
    )


# Register-file facts per arch family (per SIMD, per lane), and wave slots per SIMD.
_VGPR_FILE = {"cdna": (512, 8, 8), "rdna": (1536, 24, 16)}  # (regs, granule, max waves)


def occupancy_waves_per_simd(
    res: KernelResources, *, lds_capacity_bytes: int, simds_per_cu: int = 4
) -> int:
    """Waves per SIMD of the workgroups that can be resident on one CU.

    CDNA (wave64) has a 512-entry unified register file per lane with an
    8-register granule; RDNA wave32 has 1536 entries with a 24-register granule
    (gfx11/gfx12 1.5x register file targets). Residency is counted in whole
    workgroups: a workgroup of W waves needs ``ceil(W / simds)`` wave slots on
    each SIMD when ``W > simds`` (else one slot on W of the SIMDs), and LDS
    bounds the workgroups per CU. Returns 0 when the kernel cannot be resident
    at all, else at least 1.
    """
    family = "cdna" if res.wavefront_size == 64 else "rdna"
    regs, granule, max_waves = _VGPR_FILE[family]
    used = max(res.vgpr_total, 1)
    used = -(-used // granule) * granule
    by_vgpr = min(max_waves, regs // used)
    waves_per_wg = max(1, -(-max(res.max_flat_workgroup_size, 1) // res.wavefront_size))
    if waves_per_wg > simds_per_cu:
        wgs = by_vgpr // -(-waves_per_wg // simds_per_cu)
    else:
        wgs = (by_vgpr * simds_per_cu) // waves_per_wg
    if res.lds_bytes > 0:
        wgs = min(wgs, lds_capacity_bytes // res.lds_bytes)
    if wgs == 0:
        return 0
    return max(1, (wgs * waves_per_wg) // simds_per_cu)


# --------------------------------------------------------------------------- disassembly
_LINE_RE = re.compile(
    r"^\s+(?P<mn>[a-z_][a-z0-9_]*)(?P<ops>[^/]*)//\s*(?P<addr>[0-9A-Fa-f]+):"
)
_TARGET_RE = re.compile(r"<[^>+]*\+0x(?P<off>[0-9a-fA-F]+)>")
_FUNC_RE = re.compile(r"^(?P<addr>[0-9a-fA-F]+)\s+<(?P<name>[^>]+)>:")


@dataclass(frozen=True)
class Instr:
    addr: int
    mnemonic: str
    operands: str
    target: int | None = None  # branch target address (SOPP branches)


def parse_disassembly(text: str) -> list[Instr]:
    """Instructions (address order) from ``llvm-objdump -d`` text of one kernel."""
    out: list[Instr] = []
    func_addr = 0
    for line in text.splitlines():
        fm = _FUNC_RE.match(line)
        if fm:
            func_addr = int(fm.group("addr"), 16)
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        mn = m.group("mn")
        addr = int(m.group("addr"), 16)
        ops = m.group("ops").strip()
        target = None
        if mn == "s_branch" or mn.startswith("s_cbranch_"):
            tm = _TARGET_RE.search(line)
            if tm:
                target = func_addr + int(tm.group("off"), 16)
            else:
                imm = int(ops.split(",")[0].strip(), 0)
                if imm > 0x7FFF:
                    imm -= 0x10000
                target = addr + 4 + 4 * imm
        out.append(Instr(addr, mn, ops, target))
    return out


@dataclass(frozen=True)
class LoopBody:
    """Instructions from a back-edge target (loop header) through the back-edge."""

    header_addr: int
    backedge_addr: int
    start: int  # index into the instruction list
    end: int  # inclusive index of the back-edge branch

    def instructions(self, instrs: Sequence[Instr]) -> Sequence[Instr]:
        return instrs[self.start : self.end + 1]


def find_loops(instrs: Sequence[Instr]) -> list[LoopBody]:
    """Every back-edge (branch to an address at or before itself) as a loop body."""
    index = {ins.addr: i for i, ins in enumerate(instrs)}
    loops = []
    for i, ins in enumerate(instrs):
        if ins.target is None or ins.target > ins.addr:
            continue
        start = index.get(ins.target)
        if start is None:
            continue
        loops.append(LoopBody(ins.target, ins.addr, start, i))
    return loops


def _is_mfma(mn: str) -> bool:
    return mn.startswith(("v_mfma", "v_smfmac", "v_wmma", "v_swmmac"))


# --------------------------------------------------------------------------- barrier order
# LDS traffic per barrier interval, in the emitted LLVM IR and in the ISA. A
# barrier that is relied on to order LDS accesses across waves only does so if
# every LDS access stays on its side of it in the code object.
_IR_SCALAR_BYTES = {"i8": 1, "i16": 2, "half": 2, "bfloat": 2, "i32": 4, "float": 4}
_IR_SCALAR_BYTES.update({"i64": 8, "double": 8})
_IR_VEC_RE = re.compile(r"<(?P<n>\d+) x (?P<t>\w+)>")
_IR_LDS_LOAD_RE = re.compile(
    r"=\s*load (?:volatile )?(?P<ty><[^>]+>|\w+), ptr addrspace\(3\)"
)
_IR_LDS_STORE_RE = re.compile(
    r"^\s*store (?:volatile )?(?P<ty><[^>]+>|\w+) [^,]+, ptr addrspace\(3\)"
)
_IR_LDS_READ_CALL_RE = re.compile(
    r"=\s*call (?P<ty><[^>]+>|\w+) @llvm\.amdgcn\.ds\.read\.[^(]*\(ptr addrspace\(3\)"
)
# CDNA spells the two-address forms ``ds_read2`` / ``ds_write2st64``; RDNA (gfx11
# and later) ``ds_load_2addr`` / ``ds_store_2addr_stride64``.
_ISA_LDS_RE = re.compile(
    r"^ds_(?P<op>read|write|load|store)"
    r"(?P<two>2st64|2|_2addr_stride64|_2addr)?_[bui](?P<bits>\d+)"
)


def _ir_type_bytes(ty: str) -> int:
    m = _IR_VEC_RE.fullmatch(ty)
    if m:
        return int(m.group("n")) * _IR_SCALAR_BYTES[m.group("t")]
    return _IR_SCALAR_BYTES[ty]


@dataclass(frozen=True)
class LdsInterval:
    """LDS bytes read / written between two consecutive barriers."""

    read: int
    write: int


def ir_barrier_intervals(ir_text: str) -> list[LdsInterval]:
    """LDS bytes between consecutive ``s.barrier`` calls, in IR text order.

    Interval ``i`` is the code before barrier ``i`` (after barrier ``i - 1``);
    the code after the last barrier is the final interval.
    """
    out: list[LdsInterval] = []
    rd = wr = 0
    for line in ir_text.splitlines():
        if "@llvm.amdgcn.s.barrier()" in line and "call" in line:
            out.append(LdsInterval(rd, wr))
            rd = wr = 0
            continue
        m = _IR_LDS_LOAD_RE.search(line) or _IR_LDS_READ_CALL_RE.search(line)
        if m:
            rd += _ir_type_bytes(m.group("ty"))
            continue
        m = _IR_LDS_STORE_RE.search(line)
        if m:
            wr += _ir_type_bytes(m.group("ty"))
    out.append(LdsInterval(rd, wr))
    return out


def isa_lds_bytes(ins: Instr) -> tuple[int, int]:
    """(read, write) LDS bytes of one instruction (0, 0 for non-LDS-memory)."""
    m = _ISA_LDS_RE.match(ins.mnemonic)
    if not m:
        return 0, 0
    n = int(m.group("bits")) // 8 * (2 if m.group("two") else 1)
    return (n, 0) if m.group("op") in ("read", "load") else (0, n)


def isa_barrier_intervals(instrs: Sequence[Instr]) -> list[LdsInterval]:
    """LDS bytes between consecutive ``s_barrier`` instructions, address order."""
    out: list[LdsInterval] = []
    rd = wr = 0
    for ins in instrs:
        if ins.mnemonic == "s_barrier":
            out.append(LdsInterval(rd, wr))
            rd = wr = 0
            continue
        r, w = isa_lds_bytes(ins)
        rd += r
        wr += w
    out.append(LdsInterval(rd, wr))
    return out


def prologue_barrier_count(instrs: Sequence[Instr]) -> int:
    """``s_barrier`` count before the first loop that itself holds a barrier.

    The straight-line prologue (staging, fragment preloads) ends at that loop
    header; the IR and the ISA lay those barriers out in the same order.
    """
    loops = [
        lp
        for lp in find_loops(instrs)
        if any(x.mnemonic == "s_barrier" for x in lp.instructions(instrs))
    ]
    end = min((lp.start for lp in loops), default=len(instrs))
    return sum(1 for x in instrs[:end] if x.mnemonic == "s_barrier")


def lds_barrier_crossings(ir_text: str, instrs: Sequence[Instr]) -> list[str]:
    """Prologue barrier intervals whose LDS bytes differ between IR and ISA.

    Every barrier before the first loop that holds a barrier is compared: the
    LDS bytes read and written before it (since the previous barrier) must be
    the same in the code object as in the IR. An LDS access that the compiler
    moved across one of these barriers shows up as a byte difference on both
    of its sides. Returns one message per differing interval (empty: none).
    """
    n = prologue_barrier_count(instrs)
    ir_iv = ir_barrier_intervals(ir_text)
    isa_iv = isa_barrier_intervals(instrs)
    if len(ir_iv) <= n:
        return [f"IR holds {len(ir_iv) - 1} barriers, the ISA prologue {n}"]
    return [
        f"before prologue barrier {i + 1}: IR read/write {a.read}/{a.write} B, "
        f"ISA {b.read}/{b.write} B"
        for i, (a, b) in enumerate(zip(ir_iv[:n], isa_iv[:n]))
        if a != b
    ]


def select_q_loop(instrs: Sequence[Instr], loops: Sequence[LoopBody]) -> LoopBody:
    """The q-loop: the loop body with the most matrix instructions.

    Ties go to the larger body (an outer loop that contains an inner one). A
    kernel without any matrix instruction in a loop raises ``ValueError``.
    """
    best, best_key = None, None
    for lp in loops:
        body = lp.instructions(instrs)
        key = (sum(1 for x in body if _is_mfma(x.mnemonic)), len(body))
        if best_key is None or key > best_key:
            best, best_key = lp, key
    if best is None or best_key[0] == 0:
        raise ValueError("no loop with matrix instructions found")
    return best


@dataclass(frozen=True)
class AgprCopies:
    reads: int
    writes: int
    moves: int

    @property
    def total(self) -> int:
        return self.reads + self.writes + self.moves


def count_agpr_copies(body: Iterable[Instr]) -> AgprCopies:
    """``v_accvgpr_read*`` / ``v_accvgpr_write*`` / ``v_accvgpr_mov*`` counts."""
    r = w = m = 0
    for ins in body:
        if ins.mnemonic.startswith("v_accvgpr_read"):
            r += 1
        elif ins.mnemonic.startswith("v_accvgpr_write"):
            w += 1
        elif ins.mnemonic.startswith("v_accvgpr_mov"):
            m += 1
    return AgprCopies(r, w, m)


def agpr_copy_ceiling(
    *,
    block_m: int | None = None,
    block_n: int | None = None,
    waves: int | None = None,
    s_in_agpr: bool = False,
    head_size: int | None = None,
    spec=None,
    arch: str | None = None,
) -> AgprCopies:
    """Model ceiling of AGPR copies per q step (the hot-loop AGPR-copy rule).

    With ``spec`` and ``arch`` (preferred) this is
    :func:`kernels.common.attention_bwd.agpr_copy_budget`, the one model of the
    rule. The keyword form is the attr-route plan of a wave64 tile: 0 without
    ``s_in_agpr``; with it, S and dP (``2*kM0*kN0/(64*W)``) plus, when
    ``head_size`` is given, the dQ partial (``kM0*D/(64*W)``) are read once per
    step; never writes or moves.
    """
    if spec is not None:
        from kernels.common.attention_bwd import agpr_copy_budget

        b = agpr_copy_budget(spec, arch)
        return AgprCopies(b["reads"], b["writes"], b["moves"])
    if not s_in_agpr:
        return AgprCopies(0, 0, 0)
    reads = (2 * block_m * block_n) // (64 * waves)
    if head_size is not None:
        reads += (block_m * head_size) // (64 * waves)
    return AgprCopies(reads, 0, 0)


def exclusive_loop_bodies(
    instrs: Sequence[Instr], loops: Sequence[LoopBody]
) -> list[tuple[LoopBody, list[Instr]]]:
    """Every loop with the instructions it executes itself: its body minus the
    bodies of the loops nested in it (an outer loop's static body contains its
    inner loops once, which must not count against the outer loop's step)."""
    out = []
    for lp in loops:
        inner = [
            o
            for o in loops
            if o is not lp
            and lp.start <= o.start
            and o.end <= lp.end
            and (o.start, o.end) != (lp.start, lp.end)
        ]
        own = [
            instrs[i]
            for i in range(lp.start, lp.end + 1)
            if not any(o.start <= i <= o.end for o in inner)
        ]
        out.append((lp, own))
    return out


def mfma_dst_classes(body: Iterable[Instr]) -> dict:
    """Matrix instructions of ``body`` by destination register class:
    ``{"agpr": n, "vgpr": n}`` (an ``a[`` / ``aN`` vdst is the AGPR form)."""
    out = {"agpr": 0, "vgpr": 0}
    for ins in body:
        if not _is_mfma(ins.mnemonic):
            continue
        dst = ins.operands.strip().split(",", 1)[0].strip()
        out["agpr" if dst.startswith("a") else "vgpr"] += 1
    return out


# --------------------------------------------------------------------------- tools
def find_objdump() -> str | None:
    """Locate ``llvm-objdump`` (env, PATH, ROCm install, pip ROCm SDK) and export it."""
    cands = [os.environ.get("LLVM_OBJDUMP"), shutil.which("llvm-objdump")]
    for root in (os.environ.get("ROCM_PATH"), os.environ.get("ROCM_HOME"), "/opt/rocm"):
        if root:
            cands.append(str(Path(root) / "llvm" / "bin" / "llvm-objdump"))
    spec = importlib.util.find_spec("_rocm_sdk_core")
    if spec is not None and spec.submodule_search_locations:
        for loc in spec.submodule_search_locations:
            for exe in ("llvm-objdump.exe", "llvm-objdump"):
                cands.append(str(Path(loc) / "lib" / "llvm" / "bin" / exe))
    for c in cands:
        if c and Path(c).is_file():
            os.environ["LLVM_OBJDUMP"] = c
            return c
    return None


def _probe_isa_inspect():
    here = Path(__file__).resolve()
    rocke_root = here.parents[2]
    path = (
        rocke_root
        / "platform"
        / "dsl_docs"
        / "optimization"
        / "utilities"
        / "tools"
        / "dsl_probes"
        / "probe_isa_inspect.py"
    )
    name = "_bwd_probe_isa_inspect"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def disassemble(hsaco: bytes, *, arch: str) -> str:
    """Disassembly text via ``probe_isa_inspect.disassemble_hsaco``."""
    if find_objdump() is None:
        raise FileNotFoundError("llvm-objdump not found (set LLVM_OBJDUMP)")
    return _probe_isa_inspect().disassemble_hsaco(bytes(hsaco), mcpu=arch)


def pin_production_comgr() -> str | None:
    """Make the next compile in this process use the production comgr
    (torch's bundled library, pinned by path) without importing torch; the
    library conftest already did it for the session. Returns the pinned path
    (None: nothing to pin, the gate then reads whichever comgr loads)."""
    from kernels.common.attention_bwd_comgr import ensure_production_comgr

    return ensure_production_comgr()


def comgr_version() -> str | None:
    """``major.minor`` of the loaded comgr library plus its ROCm vintage, if known."""
    try:
        import ctypes

        from rocke.runtime import comgr as _comgr

        lib = _comgr._resolve_lib()
        major, minor = ctypes.c_size_t(), ctypes.c_size_t()
        lib.amd_comgr_get_version(ctypes.byref(major), ctypes.byref(minor))
        rocm = _comgr.resolved_lib_rocm_version()
        rocm_s = f"{rocm[0]}.{rocm[1]}" if rocm else "unknown"
        return f"comgr{major.value}.{minor.value}/rocm{rocm_s}"
    except Exception:  # noqa: BLE001 - recorded as unknown, never fatal
        return None


def comgr_is_production(comgr: str | None) -> bool:
    """True for a :func:`comgr_version` string of the production ROCm vintage."""
    return bool(comgr) and comgr.rsplit("/rocm", 1)[-1] == GATE_COMGR_ROCM


def current_flavor() -> str:
    try:
        from rocke.core.lower_llvm import _detect_llvm_flavor

        return _detect_llvm_flavor()
    except Exception:  # noqa: BLE001
        return os.environ.get("ROCKE_LLVM_FLAVOR", "unknown")


def comgr_targets() -> frozenset[str] | None:
    """The processors (``gfx950``, ...) the loaded comgr library has an ISA
    for (None: the library cannot be asked)."""
    try:
        import ctypes

        from rocke.runtime import comgr as _comgr

        lib = _comgr._resolve_lib()
        count = ctypes.c_size_t()
        if lib.amd_comgr_get_isa_count(ctypes.byref(count)) != 0:
            return None
        out = set()
        for i in range(count.value):
            name = ctypes.c_char_p()
            if lib.amd_comgr_get_isa_name(ctypes.c_size_t(i), ctypes.byref(name)):
                return None
            out.add(name.value.decode().rsplit("-", 1)[-1].split(":")[0])
        return frozenset(out)
    except Exception:  # noqa: BLE001 - unknown, never fatal
        return None


def gate_toolchain_gap(arch: str) -> str | None:
    """Why the loaded toolchain is not the gate's for ``arch`` (None: it is:
    the ``GATE_FLAVOR`` flavor and the production comgr, with a target for
    ``arch``)."""
    flavor, comgr = current_flavor(), comgr_version()
    if flavor != GATE_FLAVOR:
        return f"flavor {flavor}"
    if not comgr_is_production(comgr):
        return f"comgr {comgr}"
    targets = comgr_targets()
    if targets is not None and arch not in targets:
        return f"comgr {comgr} has no {arch} target"
    return None


def compile_or_skip(kernel: Any, *, arch: str, **kw: Any) -> Any:
    """``compile_kernel(kernel, arch=arch, **kw)`` for a test that compiles.

    A compile error skips the test only where the loaded toolchain is not the
    gate's (:func:`gate_toolchain_gap`). On the gate's toolchain (the
    production comgr, ``llvm22``) a kernel the validator admits must compile,
    so the error is raised and fails the test.
    """
    import pytest
    from rocke.helpers.compile import compile_kernel

    try:
        return compile_kernel(kernel, arch=arch, **kw)
    except Exception as exc:  # noqa: BLE001 - a defect unless the toolchain differs
        gap = gate_toolchain_gap(arch)
        if gap is None:
            raise
        pytest.skip(f"comgr cannot compile for {arch} ({gap}): {exc}")


# --------------------------------------------------------------------------- gate
@dataclass(frozen=True)
class ResourceGateResult:
    resources: KernelResources
    occupancy: int
    q_loop_copies: AgprCopies | None
    copy_ceiling: AgprCopies | None
    failures: tuple[str, ...]
    flavor: str
    comgr: str | None
    warnings: tuple[str, ...] = ()
    loops: tuple[dict, ...] = ()  # per matrix loop: copies and MFMA dst classes

    @property
    def valid(self) -> bool:
        """True only for results produced on the production flavor by the
        production comgr (``GATE_COMGR_ROCM``)."""
        return self.flavor == GATE_FLAVOR and comgr_is_production(self.comgr)

    @property
    def passed(self) -> bool:
        return self.valid and not self.failures

    def record(self) -> dict:
        return {
            "resources": self.resources.as_dict(),
            "occupancy": self.occupancy,
            "q_loop_agpr_copies": (
                None
                if self.q_loop_copies is None
                else {
                    "reads": self.q_loop_copies.reads,
                    "writes": self.q_loop_copies.writes,
                    "moves": self.q_loop_copies.moves,
                }
            ),
            "failures": list(self.failures),
            "warnings": list(self.warnings),
            "loops": [dict(x) for x in self.loops],
            "flavor": self.flavor,
            "comgr": self.comgr,
            "valid": self.valid,
        }


def evaluate_resource_gate(
    hsaco: bytes,
    *,
    arch: str,
    lds_capacity_bytes: int,
    block_m: int | None = None,
    block_n: int | None = None,
    waves: int | None = None,
    s_in_agpr: bool = False,
    spec=None,
    isa_text: str | None = None,
    disassembler: Callable[[bytes], str] | None = None,
    kernel: str | None = None,
    flavor: str | None = None,
    comgr: str | None = None,
    ir_text: str | None = None,
) -> ResourceGateResult:
    """Apply the resource rules to one compiled kernel.

    The q-loop AGPR-copy rule runs when the tile (``block_m``, ``block_n``,
    ``waves``) is given and the target has AGPRs; the disassembly comes from
    ``isa_text``, else ``disassembler(hsaco)``, else :func:`disassemble`. With
    the kernel's LLVM IR (``ir_text``), the barrier-order rule checks that no
    LDS access crossed a prologue barrier (:func:`lds_barrier_crossings`).
    Failure tokens: ``scratch``, ``vgpr_spill``, ``sgpr_spill``, ``arch_vgpr``,
    ``agpr``, ``occupancy``, ``lds``, ``agpr_copy_traffic``, ``no_q_loop``,
    ``lds_barrier_order``.
    """
    res = read_code_object_resources(hsaco, arch=arch, kernel=kernel)
    fails = []
    if res.scratch_bytes != 0:
        fails.append("scratch")
    if res.vgpr_spill != 0:
        fails.append("vgpr_spill")
    if res.sgpr_spill != 0:
        fails.append("sgpr_spill")
    if res.arch_vgpr > ARCH_VGPR_BUDGET:
        fails.append("arch_vgpr")
    if res.agpr > AGPR_BUDGET:
        fails.append("agpr")
    occ = occupancy_waves_per_simd(res, lds_capacity_bytes=lds_capacity_bytes)
    if occ < 1:
        fails.append("occupancy")
    warns = ["sgpr_margin"] if res.sgpr > SGPR_WATCH else []
    if res.lds_bytes > lds_capacity_bytes:
        fails.append("lds")
    if spec is not None:
        block_m, block_n, waves = spec.block_m, spec.block_n, spec.waves
        s_in_agpr = bool(spec.s_in_agpr)
    copies = ceiling = None
    loop_records: list[dict] = []
    if None not in (block_m, block_n, waves) and _has_unified_agprs(arch):
        if isa_text is None:
            isa_text = (disassembler or (lambda h: disassemble(h, arch=arch)))(hsaco)
        instrs = parse_disassembly(isa_text)
        if spec is not None:
            ceiling = agpr_copy_ceiling(spec=spec, arch=arch)
        else:
            ceiling = agpr_copy_ceiling(
                block_m=block_m, block_n=block_n, waves=waves, s_in_agpr=s_in_agpr
            )
        # every loop that issues matrix instructions itself (edge_tiles emits
        # several q loops; a copy in one must not hide behind a clean other)
        mloops = [
            (lp, own)
            for lp, own in exclusive_loop_bodies(instrs, find_loops(instrs))
            if any(_is_mfma(x.mnemonic) for x in own)
        ]
        if not mloops:
            fails.append("no_q_loop")
        over = False
        for lp, own in mloops:
            c = count_agpr_copies(own)
            loop_records.append(
                {
                    "header": lp.header_addr,
                    "reads": c.reads,
                    "writes": c.writes,
                    "moves": c.moves,
                    "mfma_dst": mfma_dst_classes(own),
                }
            )
            if copies is None or c.total > copies.total:
                copies = c  # the loop with the most copies
            over |= (
                c.reads > ceiling.reads
                or c.writes > ceiling.writes
                or c.moves > ceiling.moves
            )
        if over:
            fails.append("agpr_copy_traffic")
    if ir_text is not None:
        if isa_text is None:
            isa_text = (disassembler or (lambda h: disassemble(h, arch=arch)))(hsaco)
        if lds_barrier_crossings(ir_text, parse_disassembly(isa_text)):
            fails.append("lds_barrier_order")
    return ResourceGateResult(
        resources=res,
        occupancy=occ,
        q_loop_copies=copies,
        copy_ceiling=ceiling,
        failures=tuple(fails),
        flavor=flavor if flavor is not None else current_flavor(),
        comgr=comgr if comgr is not None else comgr_version(),
        warnings=tuple(warns),
        loops=tuple(loop_records),
    )
