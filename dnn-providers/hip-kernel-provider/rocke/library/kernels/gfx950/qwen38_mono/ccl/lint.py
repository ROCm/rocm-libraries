# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Peer-address lint of a CCL ``KernelDef`` (DESIGN-CCL 4.8).

Taint: every value derived (through any op) from a kernel parameter named
``peer<k>`` or ``ctrl`` (other GPUs' memory, host-mapped memory). A build
fails when a tainted value reaches

* an atomic (IR ``*atomic*`` ops, or ``global_atomic`` / ``buffer_atomic`` /
  ``flat_atomic`` in inline asm): ``P/core/lower_llvm.py`` marks f32 / bf16x2
  ``atomicrmw fadd`` as ``!amdgpu.no.remote.memory``, which is false on peer
  memory, and the protocol needs no peer atomics;
* a buffer resource or buffer op (``H7``: peer pointers in a V# are unreliable);
* a scalar memory read (``s_load`` / ``s_buffer_load``: not coherent);
* an IR load / store (no cache-policy control) or an inline-asm
  ``global_load`` / ``global_store`` without ``sc0 sc1``.

Loaded DATA is not an address: the results of a load-only inline-asm block
(``global_load*`` + ``s_waitcnt`` lines, e.g. the CCL-LM pull form's peer
reads) do not propagate the peer taint.

Local mailbox accesses: an ADDRESS derived from the kernel parameter
``arena`` (this rank's own receive arena, which peers write) is tainted too,
through address arithmetic only (not through loaded values). Its loads and
stores must be inline asm with ``sc0 sc1`` (a poll without system-scope bits
reads stale lines: the coherence negative control); IR loads / stores of it
are rejected; atomics on it are allowed (local exit counter, error claim).
``allow_local_scope=True`` waives only the missing-bits rule for the
declared ``poll_cpol`` levers (``sc1``, ``none``); every default build must
pass without it.

The rocKE lowering itself is unchanged (existing builds stay byte-identical);
the lint is the guard.
"""

from __future__ import annotations

import re
from typing import List, Set

from rocke.core.ir import KernelDef, Region

__all__ = ["lint_peer_access", "PeerLintError"]

_TAINT_PARAM = re.compile(r"^(peer\d+|ctrl)$")
_LOCAL_PARAM = "arena"
# ops through which an arena ADDRESS does not propagate (their results are data)
_NO_ADDR_PROP = ("load", "atomic", "inline_asm", "smem", "readlane")
_MEM_LINE = re.compile(r"^\s*(global_(load|store)\w*|flat_\w+|buffer_\w+|s_load\w*|s_buffer_load\w*|global_atomic\w*|scratch_\w+)\b")


class PeerLintError(AssertionError):
    pass


def _walk(region: Region):
    for op in region.ops:
        yield op
        for r in op.regions:
            yield from _walk(r)


def _is_load_block(op) -> bool:
    """An inline-asm block that only loads (``global_load*`` lines plus
    ``s_waitcnt`` / ``s_nop``): its results are loaded data, which carry no
    peer taint (the pull form reads peer arenas; its data is then stored
    with plain IR). Its own memory lines are still checked for system-scope
    bits like every asm block with a peer operand."""
    if op.name != "tile.inline_asm":
        return False
    lines = [x.strip() for x in op.attrs.get("template", "").split("\n") if x.strip()]
    loads = [x for x in lines if x.startswith("global_load")]
    rest = [x for x in lines if not x.startswith(("global_load", "s_waitcnt", "s_nop"))]
    return bool(loads) and not rest


def _asm_problems(template: str) -> List[str]:
    out = []
    for line in template.split("\n"):
        m = _MEM_LINE.match(line)
        if not m:
            continue
        mnem = m.group(1)
        if "atomic" in mnem:
            out.append(f"atomic on a peer-derived address: {line.strip()}")
        elif mnem.startswith("buffer_"):
            out.append(f"buffer op on a peer-derived address: {line.strip()}")
        elif mnem.startswith(("s_load", "s_buffer_load")):
            out.append(f"scalar load of a peer-derived address: {line.strip()}")
        elif mnem.startswith(("flat_", "scratch_")):
            out.append(f"flat / scratch access on a peer-derived address: {line.strip()}")
        elif "sc0" not in line or "sc1" not in line:
            out.append(f"peer access without system-scope bits (sc0 sc1): {line.strip()}")
    return out


def _local_asm_problems(template: str) -> List[str]:
    out = []
    for line in template.split("\n"):
        m = _MEM_LINE.match(line)
        if not m:
            continue
        mnem = m.group(1)
        if "atomic" in mnem:
            continue  # local atomics (exit counter, error claim) are fine
        if mnem.startswith("buffer_") or mnem.startswith(("s_load", "s_buffer_load", "flat_", "scratch_")):
            out.append(f"arena: non-global access of the receive arena: {line.strip()}")
        elif "sc0" not in line or "sc1" not in line:
            out.append(f"arena: receive-arena access without system-scope bits (sc0 sc1): {line.strip()}")
    return out


def _local_problems(kernel: KernelDef, ops) -> List[str]:
    tainted: Set[str] = {f"%{p.name}" for p in kernel.params if p.name == _LOCAL_PARAM}
    if not tainted:
        return []
    changed = True
    while changed:
        changed = False
        for op in ops:
            if any(k in op.name for k in _NO_ADDR_PROP):
                continue
            if any(v.name in tainted for v in op.operands):
                for res in op.results:
                    if res.name not in tainted:
                        tainted.add(res.name)
                        changed = True
    problems: List[str] = []
    for op in ops:
        hit = [v.name for v in op.operands if v.name in tainted]
        if not hit:
            continue
        name = op.name
        if name == "tile.inline_asm":
            problems += [f"{name}: {p}" for p in _local_asm_problems(op.attrs.get("template", ""))]
        elif "atomic" in name:
            continue
        elif "buffer" in name or name == "tile.buffer_rsrc":
            problems.append(f"{name}: arena: buffer resource / op of the receive arena ({hit[0]})")
        elif name.startswith(("memref.global_load", "memref.global_store", "memref.cooperative", "memref.global_flag", "tile.global_load", "tile.global_store", "tile.buffer_load", "tile.buffer_store")):
            problems.append(f"{name}: arena: plain IR access of the receive arena ({hit[0]}), no system-scope bits")
        elif "smem" in name and "load" in name:
            problems.append(f"{name}: arena: scalar load of the receive arena ({hit[0]})")
    return problems


def lint_peer_access(kernel: KernelDef, *, allow_local_scope: bool = False) -> List[str]:
    """Problems found (empty = clean). Inline asm is checked line by line
    when ANY of its operands is tainted (asm operands are not tied to
    instructions), so keep peer and local accesses in separate asm blocks.
    ``allow_local_scope`` waives the missing-system-scope rule of receive-arena
    loads (declared ``poll_cpol`` lever builds only)."""
    tainted: Set[str] = {f"%{p.name}" for p in kernel.params if _TAINT_PARAM.match(p.name)}
    problems: List[str] = []
    # fixed point (loops carry values through regions)
    changed = True
    ops = list(_walk(kernel.body))
    while changed:
        changed = False
        for op in ops:
            if _is_load_block(op):
                continue  # the DATA a load-only asm block returns (pull reads of a peer) is not an address
            if any(v.name in tainted for v in op.operands):
                for res in op.results:
                    if res.name not in tainted:
                        tainted.add(res.name)
                        changed = True
    for op in ops:
        hit = [v.name for v in op.operands if v.name in tainted]
        if not hit:
            continue
        name = op.name
        if name == "tile.inline_asm":
            problems += [f"{name}: {p}" for p in _asm_problems(op.attrs.get("template", ""))]
        elif "atomic" in name:
            problems.append(f"{name} on a peer-derived address ({hit[0]})")
        elif "buffer" in name or name == "tile.buffer_rsrc":
            problems.append(f"{name}: buffer resource / op from a peer-derived address ({hit[0]})")
        elif name.startswith(("memref.global_load", "memref.global_store", "memref.cooperative", "memref.global_flag", "tile.global_load", "tile.global_store", "tile.buffer_load", "tile.buffer_store")):
            problems.append(f"{name}: plain IR access of a peer-derived address ({hit[0]}), no system-scope bits")
        elif "smem" in name and "load" in name:
            problems.append(f"{name}: scalar load of a peer-derived address ({hit[0]})")
    for lp in _local_problems(kernel, ops):
        if allow_local_scope and "without system-scope bits" in lp:
            continue
        problems.append(lp)
    return problems


def check(kernel: KernelDef, *, allow_local_scope: bool = False) -> None:
    p = lint_peer_access(kernel, allow_local_scope=allow_local_scope)
    if p:
        raise PeerLintError(f"{kernel.name}: " + "; ".join(p[:8]))
