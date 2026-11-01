# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Static checks of the true-mono platform (WP1).

* :func:`check_kv_acquire` (P5, D0 section 5 WP1): on the rocKE IR, every load
  of a launch-written KV row carries ``sc1`` or follows an agent-scope
  ``buffer_inv`` issued after the ready-pair poll, every launch-written KV
  store is ``sc1`` or written back (``buffer_wbl2``) before its ready pair, and
  every scalar re-read of a launch-written word invalidates the scalar cache
  (``s_dcache_inv``; ``strict``: plus the L2 acquire). P5's verifier: a missing
  acquire is caught by numerics 0 times in the gate configuration, so this is
  static. Reads the ``rocke.kv`` markers of :mod:`.kv`.
* :func:`spin_without_reload` (P2 finding 1): in a disassembly, a loop with no
  memory instruction whose instructions read only ``vcc`` / ``exec`` /
  constants: a poll whose loads LLVM hoisted out of the loop (it never
  reloads, so it hangs whenever the first read was not ready).
* :func:`exec_join_findings` (P6, FlyTape ``lint_isa.py``): a vector register
  written in a block before the block restores ``exec`` (``s_or_b64 exec,
  exec, s[..]``): a write under a stale mask.
* :func:`code_object_meta` / :func:`resource_gate` (P6 CI gate): SGPR / VGPR
  spill counts, private segment size and the exec-join lint all zero.
* :func:`check_staging_overlap` (P8): a fused function's private staging
  (e.g. the fused MoE's ``ys`` mailbox) must not alias any mailbox region a
  stage of any layer kind polls under launch tags: the scratch buffer is
  shared by every layer and int32 tags wrap, so a stale private pair can match
  a later launch's tag (P8: the wrapped tag of slot 7 equals ``0x80000088``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass

from rocke.core.ir import CPOL_SC1, KernelDef, Op

from .kv import KV_ATTR

__all__ = [
    "Finding",
    "Staging",
    "check_kv_acquire",
    "check_staging_overlap",
    "code_object_meta",
    "exec_join_findings",
    "fused_moe_staging",
    "ops_in_order",
    "parse_disasm",
    "polled_regions",
    "resource_gate",
    "spin_without_reload",
]


@dataclass(frozen=True)
class Finding:
    rule: str
    where: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.where}: {self.detail}"


# ------------------------------------------------------------ IR traversal


def ops_in_order(kernel: KernelDef) -> Iterator[Op]:
    """Every op of ``kernel`` in program (textual) order, nested regions
    inline (an op's regions follow the op)."""

    def walk(ops: Iterable[Op]) -> Iterator[Op]:
        for op in ops:
            yield op
            for r in op.regions:
                yield from walk(r.ops)

    body = getattr(kernel, "body", None)
    if body is None:
        body = getattr(kernel, "ops", [])
    if hasattr(body, "ops"):
        body = body.ops
    yield from walk(body)


def _asm_text(op: Op) -> str:
    for k in ("asm", "text", "asm_string", "template"):
        v = op.attrs.get(k)
        if isinstance(v, str):
            return v
    return ""


def _is_agent_acquire(op: Op) -> bool:
    t = _asm_text(op)
    return bool(re.search(r"buffer_inv\s+(sc0\s+)?sc1\b", t))


def _is_writeback(op: Op) -> bool:
    return "buffer_wbl2" in _asm_text(op)


def check_kv_acquire(kernel: KernelDef, *, strict: bool = True) -> list[Finding]:
    """The P5 rules over the ``rocke.kv`` markers (see the module docstring),
    in program order (nested regions inline: an acquire by thread 0 behind a
    barrier covers the CTA, as in the bulk rule). Returns the findings (empty
    = clean). A kernel with no markers returns a single ``unmarked`` finding:
    the check cannot vouch for it."""
    # constants and structured-control ops (created outside ``IRBuilder._op``,
    # so never marked) do not delimit a marked group
    ops = [op for op in ops_in_order(kernel) if op.name != "arith.constant" and not op.name.startswith("scf.")]
    if not any(KV_ATTR in op.attrs for op in ops):
        return [Finding("unmarked", kernel.name, "no rocke.kv markers: build KV traffic through persist.kv")]
    out: list[Finding] = []
    st = {"poll": -1, "acq": -1, "pending": 0, "in_pub": False, "wb": False}

    def close_pub() -> None:
        if st["pending"] and not st["wb"]:
            out.append(Finding("kv_store_unreleased", kernel.name,
                               f"{st['pending']} launch-written KV store(s) published without sc1 or "
                               "buffer_wbl2 (P5: 100% stale on other XCDs)"))  # fmt: skip
        st["pending"], st["in_pub"], st["wb"] = 0, False, False

    for i, op in enumerate(ops):
        kind = op.attrs.get(KV_ATTR)
        if kind == "publish":
            st["in_pub"] = True
            st["wb"] = st["wb"] or _is_writeback(op)
            continue
        if st["in_pub"]:
            close_pub()
        if kind is None:
            continue
        aux = int(op.attrs.get("aux", 0))
        if kind == "store":
            if not aux & CPOL_SC1:
                st["pending"] += 1
        elif kind == "poll":
            st["poll"] = i
        elif kind == "acquire":
            if _is_agent_acquire(op):
                st["acq"] = i
        elif kind == "launch":
            if aux & CPOL_SC1:
                continue
            if st["poll"] < 0:
                out.append(Finding("kv_load_unordered", kernel.name,
                                   f"op {i}: launch-written KV load with no ready-pair poll before it"))  # fmt: skip
            elif st["acq"] < st["poll"]:
                out.append(Finding("kv_load_unacquired", kernel.name,
                                   f"op {i}: plain load of a launch-written KV row without buffer_inv sc1 after "
                                   "its poll (P5: vL1D-stale)"))  # fmt: skip
        elif kind == "sread":
            if op.name != "tile.inline_asm":
                continue  # carried in a register (the ready pair's payload): nothing re-read
            t = _asm_text(op)
            if "s_dcache_inv" not in t:
                out.append(Finding("sread_stale_scalar", kernel.name,
                                   f"op {i}: scalar re-read of a launch-written word without s_dcache_inv "
                                   "(P5 C-B5: 100% stale)"))  # fmt: skip
            elif strict and not re.search(r"buffer_inv\s+(sc0\s+)?sc1\b", t):
                out.append(Finding("sread_l2_unacquired", kernel.name,
                                   f"op {i}: s_dcache_inv without the L2 acquire (clean on hardware, not "
                                   "memory-model safe; strict)"))  # fmt: skip
    if st["in_pub"]:
        close_pub()
    if st["pending"]:
        out.append(Finding("kv_store_unpublished", kernel.name, f"{st['pending']} KV store(s) never published"))
    return out


# ------------------------------------------------------------- disassembly


@dataclass(frozen=True)
class Inst:
    addr: int
    op: str
    operands: str
    text: str


_HDR = re.compile(r"^([0-9a-fA-F]+) <([^>]+)>:\s*$")
_INST = re.compile(r"^\s+([a-z_0-9]+)(?:\s+(.*?))?\s*//\s*([0-9A-Fa-f]+):")
_BRANCH = re.compile(r"^s_(c?branch)(?:_\w+)?$")
_MEM = re.compile(
    r"^(global_load|global_atomic|buffer_load|buffer_atomic|flat_load|flat_atomic|scratch_load|"
    r"s_load|s_buffer_load|ds_read|ds_add_rtn|ds_cmpst_rtn|ds_swizzle|s_memrealtime|s_memtime|s_getreg)"
)  # fmt: skip
_INERT_OPS = re.compile(r"^(s_nop|s_sleep|s_waitcnt|s_setprio|s_branch|s_cbranch\w*)$")
_SREG_OK = re.compile(r"^(vcc|vcc_lo|vcc_hi|exec|exec_lo|exec_hi|-?\d+|0x[0-9a-fA-F]+|scc)$")


def parse_disasm(text: str) -> dict[str, list[Inst]]:
    """``llvm-objdump -d`` text -> {function: [instructions]}."""
    funcs: dict[str, list[Inst]] = {}
    cur = None
    for line in text.splitlines():
        m = _HDR.match(line)
        if m:
            cur = funcs.setdefault(m.group(2), [])
            continue
        if cur is None:
            continue
        m = _INST.match(line)
        if m:
            cur.append(Inst(int(m.group(3), 16), m.group(1), (m.group(2) or "").strip(), line.strip()))
    return funcs


def _branch_target(inst: Inst) -> int | None:
    if not _BRANCH.match(inst.op):
        return None
    try:
        imm = int(inst.operands.split()[0], 0)
    except (ValueError, IndexError):
        return None
    if imm >= 0x8000:
        imm -= 0x10000
    return inst.addr + 4 + 4 * imm


def _inert(inst: Inst) -> bool:
    if _INERT_OPS.match(inst.op):
        return True
    if not inst.op.startswith("s_"):
        return False
    srcs = [x.strip() for x in inst.operands.split(",")[1:]]
    return all(_SREG_OK.match(s) for s in srcs if s)


def spin_without_reload(text: str) -> list[Finding]:
    """Backward-branch loops with no memory instruction whose every
    instruction is inert (reads only vcc / exec / constants): their exit
    condition cannot change, i.e. a poll that never reloads."""
    out = []
    for fn, insts in parse_disasm(text).items():
        by_addr = {x.addr: k for k, x in enumerate(insts)}
        for k, x in enumerate(insts):
            t = _branch_target(x)
            if t is None or t > x.addr or t not in by_addr:
                continue
            body = insts[by_addr[t] : k + 1]
            if any(_MEM.match(y.op) for y in body):
                continue
            if all(_inert(y) for y in body):
                out.append(Finding("spin_without_reload", fn,
                                   f"loop 0x{t:x}..0x{x.addr:x} ({len(body)} insts): "
                                   + " | ".join(y.op + (" " + y.operands if y.operands else "") for y in body)))  # fmt: skip
    return out


_RESTORE = re.compile(r"^s_or_b64$")
_VDST = re.compile(r"^(v\d+|v\[\d+:\d+\]|a\d+|a\[\d+:\d+\])\s*(,|$)")
_NOVDST = re.compile(r"^(s_|v_writelane|v_readlane|v_readfirstlane|v_cmp|ds_write|ds_add|global_store|buffer_store)")
_BEND = re.compile(r"^s_(c?branch\w*|endpgm|setpc\w*)$")


def exec_join_findings(text: str) -> list[Finding]:
    """FlyTape's exec-join lint (P6 port) on an objdump listing: blocks start
    at branch targets; a vector write between a block start and the block's
    ``s_or_b64 exec, exec, s[..]`` is flagged."""
    out = []
    for fn, insts in parse_disasm(text).items():
        starts = {t for t in (_branch_target(x) for x in insts) if t is not None}
        label, early = None, []
        for x in insts:
            if x.addr in starts:
                label, early = x.addr, []
            if label is None:
                continue
            ops0 = x.operands.split(",")[0].strip() if x.operands else ""
            if _RESTORE.match(x.op) and x.operands.replace(" ", "").startswith("exec,exec,s["):
                if early:
                    out.append(Finding("exec_join", fn, f"block 0x{label:x}: {len(early)} vector write(s) "
                                       f"before the exec restore: {early[0].text[:80]}"))  # fmt: skip
                label = None
            elif _BEND.match(x.op) or "exec" in ops0 or "saveexec" in x.op:
                label = None
            elif not _NOVDST.match(x.op) and _VDST.match(x.operands):
                early.append(x)
    return out


# ------------------------------------------------------- code-object gate

META_KEYS = (".vgpr_count", ".agpr_count", ".sgpr_count", ".vgpr_spill_count", ".sgpr_spill_count",
             ".group_segment_fixed_size", ".private_segment_fixed_size", ".max_flat_workgroup_size",
             ".uses_dynamic_stack")  # fmt: skip


def code_object_meta(blob: bytes) -> dict:
    """The kernel metadata scalars of a code object (msgpack scan; P6)."""
    out = {}
    for k in META_KEYS:
        kb = k.encode()
        i = blob.find(bytes([0xA0 | len(kb)]) + kb)
        if i < 0:
            continue
        j = i + 1 + len(kb)
        t = blob[j]
        if t < 0x80:
            v = t
        elif t in (0xC2, 0xC3):
            v = t == 0xC3
        elif t == 0xCC:
            v = blob[j + 1]
        elif t == 0xCD:
            v = int.from_bytes(blob[j + 1 : j + 3], "big")
        elif t == 0xCE:
            v = int.from_bytes(blob[j + 1 : j + 5], "big")
        else:
            v = None
        out[k[1:]] = v
    return out


def resource_gate(blob: bytes, disasm: str | None = None, *, strict_exec_join: bool = False) -> tuple[bool, dict]:
    """P6's CI gate: spills, private segment and (with ``disasm``) the spin
    lint all zero; the exec-join lint is reported, and gates only with
    ``strict_exec_join`` (FlyTape's heuristic also flags legitimate join
    blocks of rocKE builds: 6 in the P2 ``pairs`` builds, all temporaries
    consumed under the same mask). Returns ``(ok, report)``."""
    m = code_object_meta(blob)
    rep = {"meta": m}
    ok = all((m.get(k) or 0) == 0 for k in ("vgpr_spill_count", "sgpr_spill_count", "private_segment_fixed_size"))
    if disasm is not None:
        ej = exec_join_findings(disasm)
        sp = spin_without_reload(disasm)
        rep["exec_join"] = [str(f) for f in ej]
        rep["spin"] = [str(f) for f in sp]
        ok = ok and not sp and (not ej or not strict_exec_join)
    rep["ok"] = ok
    return ok, rep


# --------------------------------------------------------- staging (P8)


@dataclass(frozen=True)
class Staging:
    """A private staging range of a fused function: ``owner``, byte
    ``offset`` in the scratch buffer, ``nbytes``."""

    owner: str
    offset: int
    nbytes: int


def polled_regions(flow: Mapping[str, tuple], extra: Mapping[str, tuple] = ()) -> set[str]:
    """Regions some stage reads (polls) in any layer kind: the union of the
    ``reads`` of an ``atom_layout._FLOW``-style map (``stage -> (deps, reads,
    writes)``) and of ``extra`` per-kind overrides."""
    out: set[str] = set()
    for src in (flow, dict(extra)):
        for _st, (_deps, reads, _writes) in src.items():
            out |= set(reads)
    return out


def check_staging_overlap(stagings: Sequence[Staging], regions: Mapping[str, int], region_bytes: Mapping[str, int],
                          row_bytes: int, rows: int, polled: Iterable[str]) -> list[Finding]:  # fmt: skip
    """Every staging range against every row of every polled mailbox region
    (``regions`` = row-0 byte offsets, ``region_bytes`` = bytes per row,
    rows ``row_bytes`` apart)."""
    pol = set(polled)
    out = []
    for s in stagings:
        lo, hi = s.offset, s.offset + s.nbytes
        for name, off in regions.items():
            if name not in pol:
                continue
            for r in range(rows):
                a = off + r * row_bytes
                b = a + region_bytes[name]
                if lo < b and a < hi:
                    out.append(Finding("staging_aliases_mailbox", s.owner,
                                       f"[{lo:#x}, {hi:#x}) overlaps polled region {name!r} row {r} "
                                       f"[{a:#x}, {b:#x}) (P8: a stale private pair can match a later launch's "
                                       "wrapped tag)"))  # fmt: skip
    return out


def fused_moe_staging(rows: int) -> list[Staging]:
    """Where ``moe_fused`` places its private ``ys`` mailbox today: row ``r``'s
    ys at ``SC["ss"] + r * SCRATCH_BYTES``, ``YS_BYTES`` long (P8's finding
    is the overlap with ``ss``, which Q6 polls)."""
    from kernels.gfx950.qwen38_mono import atom_layout as AL
    from kernels.gfx950.qwen38_mono import moe_fused as MF

    return [Staging(f"moe_fused.ys[r{r}]", AL.SC["ss"] + r * AL.SCRATCH_BYTES, MF.YS_BYTES) for r in range(rows)]
