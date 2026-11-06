# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""W15 climb r0 fix: Rust-direct DEVICE FUNCTIONS for the hybrid (prototype of the in-place
``ROCKE_CODEOBJ_DIRECT_LIB`` arm).

The hybrid links every rocKE K2b stage into ATOM's FlyDSL K2 as an LLVM device function; the Rust direct compiler
(``platform/rust/rocke-codeobj``) emits whole code objects only. This module bridges the two without LLVM codegen of
the stage body:

1. the stage's ``KernelDef`` (the SAME body the COMGR device function is lowered from) is compiled by the direct
   compiler as a kernel whose parameters are all kernarg-PRELOADED (``inreg``): every parameter then lives in a fixed
   user SGPR (``s[2 + dword]``) at the kernel's real entry (the code object's first 256 bytes are the compatibility
   entry for firmware without preload; it is skipped), the workitem id in ``v0``;
2. the code object's ``.text`` from that entry is disassembled and turned into ONE inline-asm block: branch targets
   become local labels (``${:uid}``: unique per inlined copy), every ``s_endpgm`` a branch to the block's end, where
   ``s_waitcnt vmcnt(0) lgkmcnt(0)`` drains the memory ops LLVM cannot see and EXEC is restored from an SGPR pair
   saved at the start (the body may end lanes early);
3. a ``define void @<symbol>(<the COMGR device function's ABI>)`` wraps it: ``readfirstlane`` of each argument bound
   to its preload SGPR (tied in / out constraints: the asm may overwrite them), ``workitem.id.x`` to ``v0``, every
   other register the body touches (SGPRs, VGPRs, AGPRs, VCC, M0, SCC) clobbered, plus memory. Leading / trailing
   ``s_nop 4`` cover the VALU-SGPR -> VMEM and VALU -> VALU wait states LLVM cannot see across the asm boundary
   (alola rules 176 / 185 / 212).

Refused (``DirectFnRefused``): a body whose parameters do not all fit the preload (the compiler's lever ``devfn``,
added for this: 64 dwords; without it the hardware's 14), a lane-varying parameter, scratch (spills), static LDS beyond the COMGR function's
pool, the workgroup id, or PC-relative code (``s_getpc`` / ``s_setpc`` / ``s_swappc``).

The wrapper has the COMGR device function's symbol, ABI and attributes, so ATOM's extern link (``rk_extern``) takes
it unchanged (``ATOM_DSV41_RK_EXTERN=<stage>=<dir>``). Bitwise vs the COMGR function is NOT by construction (another
compiler): it is gated in place by the rig like every candidate.

CLI (used by ``stage_moe --direct``): see :func:`write_direct_functions`.
"""

from __future__ import annotations

import json
import os
import re
import subprocess

__all__ = ["DirectFnRefused", "direct_device_function"]

PRELOAD_SGPRS = 64  # the compiler's lever ``devfn`` (rir/opt.rs DEVFN_PRELOAD_SGPRS): every inreg argument preloaded
PRELOAD_ENTRY = 256  # bytes of the compatibility entry (target/gfx9 preload_entry)
_BRANCH = re.compile(r"^s_(branch|cbranch_\w+)$")
_BAD = re.compile(r"^s_(getpc_b64|setpc_b64|swappc_b64|call_b64|rfe_b64|trap|sethalt)$")
_INST = re.compile(r"^\s+([a-z_0-9]+)(?:\s+(.*?))?\s*//\s*([0-9A-Fa-f]+):")
_SYM = re.compile(r"^([0-9a-fA-F]+) <([^>]+)>:")
_SREG = re.compile(r"\bs\[(\d+):(\d+)\]|\bs(\d+)\b")
_VREG = re.compile(r"\bv\[(\d+):(\d+)\]|\bv(\d+)\b")
_AREG = re.compile(r"\ba\[(\d+):(\d+)\]|\ba(\d+)\b")


class DirectFnRefused(Exception):
    """The body cannot become a direct device function (reason in the message)."""


def _regs(pat, text):
    out = set()
    for m in pat.finditer(text):
        if m.group(3) is not None:
            out.add(int(m.group(3)))
        else:
            out.update(range(int(m.group(1)), int(m.group(2)) + 1))
    return out


def _kernarg_layout(kernel):
    """(param index -> (first dword, dwords)), total dwords: the direct compiler's ``opt::kernarg_layout`` (natural
    alignment of each parameter)."""
    off, out = 0, {}
    for i, p in enumerate(kernel.params):
        size = 8 if str(p.type) in ("i64", "f64") or "ptr" in str(p.type) else 4
        off = (off + size - 1) // size * size
        out[i] = (off // 4, size // 4)
        off += size
    return out, (off + 3) // 4


def _uses_block_id(kernel):
    def walk(region):
        for op in region.ops:
            if op.name in ("gpu.block_id", "gpu.block_id_x", "gpu.workgroup_id") or "block_id" in op.name:
                return True
            for r in op.regions:
                if walk(r):
                    return True
        return False

    return walk(kernel.body)


def _disasm(hsaco: bytes, objdump: str, symbol: str, workdir: str) -> list:
    """[(byte offset from the symbol, mnemonic, operands)] of ``symbol``'s instructions."""
    os.makedirs(workdir, exist_ok=True)
    path = os.path.join(workdir, symbol + ".direct.hsaco")
    with open(path, "wb") as f:
        f.write(hsaco)
    txt = subprocess.run([objdump, "-d", "--mcpu=gfx950", path], capture_output=True, text=True, check=True).stdout
    with open(os.path.join(workdir, symbol + ".direct.s"), "w", newline="\n") as f:
        f.write(txt)
    out, base, cur = [], None, False
    for line in txt.splitlines():
        m = _SYM.match(line)
        if m:
            cur = m.group(2) == symbol
            base = int(m.group(1), 16) if cur else base
            continue
        if not cur:
            continue
        m = _INST.match(line)
        if m:
            out.append((int(m.group(3), 16) - base, m.group(1), (m.group(2) or "").strip()))
    if not out:
        raise DirectFnRefused(f"no instructions of {symbol} in the direct code object")
    return out


def _meta(hsaco: bytes) -> dict:
    from ..qwen38_mono.persist.lint import code_object_meta

    return code_object_meta(hsaco)


def asm_body(insts: list, preload: int) -> tuple:
    """The inline-asm template text and the register sets it touches. ``insts`` from :func:`_disasm`."""
    start = PRELOAD_ENTRY if preload else 0
    body = [x for x in insts if x[0] >= start]
    last = max((i for i, x in enumerate(body) if x[1] == "s_endpgm"), default=None)
    if last is None:
        raise DirectFnRefused("no s_endpgm")
    body = body[: last + 1]  # the alignment padding past the last s_endpgm never runs
    for _, mn, ops in body:
        if _BAD.match(mn):
            raise DirectFnRefused(f"PC-relative / call instruction {mn}")
        base = ops.split(",")[1].strip() if mn.startswith("s_load") and "," in ops else ""
        if base == "s[0:1]":
            raise DirectFnRefused(f"a kernarg load in the body ({mn} {ops}): a compiler without the devfn lever")
    targets = set()
    lines = []
    for off, mn, ops in body:
        if _BRANCH.match(mn):
            imm = int(ops.split()[0], 0)
            if imm >= 0x8000:
                imm -= 0x10000
            targets.add(off + 4 + 4 * imm)
    addrs = {x[0] for x in body}
    if not targets <= addrs | {body[-1][0] + 4}:
        raise DirectFnRefused(f"branch target outside the body: {sorted(targets - addrs)[:4]}")
    text_all = []
    for off, mn, ops in body:
        if off in targets:
            lines.append(f".Ldfn${{:uid}}_{off:x}:")
        if _BRANCH.match(mn):
            imm = int(ops.split()[0], 0)
            if imm >= 0x8000:
                imm -= 0x10000
            lines.append(f"{mn} .Ldfn${{:uid}}_{off + 4 + 4 * imm:x}")
        elif mn == "s_endpgm":
            lines.append("s_branch .Ldfn${:uid}_end")
        else:
            lines.append(f"{mn} {ops}".rstrip())
            text_all.append(ops)
    if body[-1][0] + 4 in targets:
        lines.append(f".Ldfn${{:uid}}_{body[-1][0] + 4:x}:")
    joined = " ".join(text_all)
    return lines, _regs(_SREG, joined), _regs(_VREG, joined), _regs(_AREG, joined), ("m0" in joined)


def direct_device_function(kernel, symbol: str, abi, *, objdump: str, workdir: str, lds_bytes: int = 0,
                           flat_wg: str = "64,512", datalayout: str = "") -> dict:  # fmt: skip
    """The LLVM IR text of a direct-compiled device function ``symbol`` with ``abi`` (``[(name, 'i32'|'i64')]``, the
    COMGR device function's, ``lds_base`` last) from ``kernel`` (its parameters are marked ``inreg`` here). Returns
    {"ir", "meta"}; raises :class:`DirectFnRefused`."""
    from rocke.helpers.compile import compile_kernel

    if _uses_block_id(kernel):
        raise DirectFnRefused("the body reads the workgroup id")
    layout, dwords = _kernarg_layout(kernel)
    if dwords > PRELOAD_SGPRS:
        raise DirectFnRefused(f"{dwords} kernarg dwords > the {PRELOAD_SGPRS}-dword preload")
    names = [p.name for p in kernel.params]
    if [n for n, _ in abi][: len(names)] != names:
        raise DirectFnRefused(f"ABI {abi} does not start with the kernel parameters {names}")
    for p in kernel.params:
        p.attrs["inreg"] = True
    os.environ["ROCKE_DIRECT_STRICT"] = "1"
    saved = os.environ.get("ROCKE_DIRECT_LEVERS")
    os.environ["ROCKE_DIRECT_LEVERS"] = ",".join(x for x in ((saved or "").split(",") + ["devfn"]) if x)
    try:
        art = compile_kernel(kernel, arch="gfx950", backend="python", code_object_backend="direct")
    finally:
        if saved is None:
            os.environ.pop("ROCKE_DIRECT_LEVERS", None)
        else:
            os.environ["ROCKE_DIRECT_LEVERS"] = saved
    if getattr(art, "code_object_backend", "direct") != "direct":
        raise DirectFnRefused(f"direct compile refused: {getattr(art, 'direct_fallback', '')}")
    meta = _meta(art.hsaco)
    if meta.get(".private_segment_fixed_size", 0) or meta.get(".sgpr_spill_count", 0) or meta.get(".vgpr_spill_count", 0):
        raise DirectFnRefused(f"scratch / spills: {meta}")
    if meta.get(".group_segment_fixed_size", 0) > lds_bytes:
        raise DirectFnRefused(f"static LDS {meta.get('.group_segment_fixed_size')} > the function's pool {lds_bytes}")
    insts = _disasm(art.hsaco, objdump, kernel.name, workdir)
    lines, sregs, vregs, aregs, m0 = asm_body(insts, dwords)
    inputs = []  # (constraint register, llvm type, argument name)
    for i, p in enumerate(kernel.params):
        first, n = layout[i]
        reg = f"s{2 + first}" if n == 1 else f"s[{2 + first}:{3 + first}]"
        inputs.append((reg, "i32" if n == 1 else "i64", p.name))
    inputs.append(("v0", "i32", "%tid"))
    pinned = set()
    for i, p in enumerate(kernel.params):
        first, n = layout[i]
        pinned.update(range(2 + first, 2 + first + n))
    save = max(sregs | pinned | {1}) + 1
    save += save % 2
    if save + 1 > 101:
        raise DirectFnRefused(f"no free SGPR pair for EXEC past s{save - 1}")
    tmpl = ["s_nop 4", f"s_mov_b64 s[{save}:{save + 1}], exec"] + lines + [
        ".Ldfn${:uid}_end:", "s_waitcnt vmcnt(0) lgkmcnt(0)", f"s_mov_b64 exec, s[{save}:{save + 1}]", "s_nop 4"]
    clob_s = sorted((sregs | {save, save + 1}) - pinned)
    clob_v = sorted(vregs - {0})
    cons = [f"={{{r}}}" for r, _, _ in inputs] + [str(i) for i in range(len(inputs))]
    cons += [f"~{{s{r}}}" for r in clob_s] + [f"~{{v{r}}}" for r in clob_v] + [f"~{{a{r}}}" for r in sorted(aregs)]
    cons += ["~{vcc}", "~{scc}"] + (["~{m0}"] if m0 else []) + ["~{memory}"]

    def esc(s):
        return s.replace("\\", "\\5C").replace('"', "\\22")

    asm = "\\0A".join(esc(x) for x in tmpl)
    args = []
    body = ["entry:", "  %tid = call i32 @llvm.amdgcn.workitem.id.x()"]
    for k, (_, ty, name) in enumerate(inputs[:-1]):
        body.append(f"  %u{k} = call {ty} @llvm.amdgcn.readfirstlane.{ty}({ty} %{name})")
        args.append(f"{ty} %u{k}")
    args.append("i32 %tid")
    rty = "{ " + ", ".join(ty for _, ty, _ in inputs) + " }"
    body.append(f'  %r = call {rty} asm sideeffect "{asm}", "{",".join(cons)}"({", ".join(args)}) #1')
    body.append("  ret void")
    sig = ", ".join(f"{t} %{n}" for n, t in abi)
    dl = next((x for x in datalayout.splitlines() if x.startswith("target datalayout")), None)
    ir = "\n".join(([dl] if dl else []) + [
        'target triple = "amdgcn-amd-amdhsa"',
        f"; W15 r0 fix: Rust-direct device function of {kernel.name} (dsv41_mono/direct_fn.py); direct resources "
        f"{json.dumps({k: v for k, v in meta.items()})}",
        "declare i32 @llvm.amdgcn.workitem.id.x()",
        "declare i32 @llvm.amdgcn.readfirstlane.i32(i32)",
        "declare i64 @llvm.amdgcn.readfirstlane.i64(i64)",
        f"define void @{symbol}({sig}) #0 {{",
        *body,
        "}",
        f'attributes #0 = {{ "amdgpu-flat-work-group-size"="{flat_wg}" alwaysinline convergent norecurse nounwind }}',
        "attributes #1 = { convergent nounwind }",
        "",
    ])
    return {"ir": ir, "meta": {"direct_meta": meta, "kernarg_dwords": dwords, "asm_lines": len(tmpl),
                               "sgprs_clobbered": len(clob_s), "vgprs_clobbered": len(clob_v),
                               "agprs_clobbered": len(aregs), "exec_save": save}}  # fmt: skip
