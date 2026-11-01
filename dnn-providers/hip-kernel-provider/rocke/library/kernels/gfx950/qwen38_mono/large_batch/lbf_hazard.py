# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WP-L1: the inline-asm VMEM hazard as a compile gate for every LB / CCL build (DESIGN-LB 15.4, DESIGN-CCL 10.6).

gfx9 needs 5 wait states between a VALU write of an SGPR (``v_readlane`` restoring a spilled SGPR base,
``v_readfirstlane``, a VOP3 SGPR destination) and a VMEM instruction that reads that SGPR. LLVM pads its own VMEM
instructions but not VMEM inside inline asm, so an asm ``global_*`` / ``buffer_*`` on an ``s`` operand can read a stale
base: grid-barrier arrivals at a wrong address or a memory fault (P-L4 1.3, WP-C5 F1; alola rules 176 / 182 / 191).
Whether it appears depends on register allocation, so a clean build today does not clear the next one: the ISA of
every build is scanned.

* :func:`pad_asm_vmem` (the ``hz`` knob): prefix ``s_nop 4`` to every ``tile.inline_asm`` whose template issues VMEM
  and whose constraints have an ``s`` input (the same pass as ``insitu.pad_asm_vmem``; own copy so the LB framework
  does not depend on a probe module).
* :func:`scan_text`: the ISA scanner of ``p-l4/fix/asm_hazard.py`` (backward walk over control-flow predecessors from
  every VMEM reading an SGPR; branch targets from the SOPP offsets; one wait state per instruction, ``N + 1`` per
  ``s_nop N``; a hit = a VALU write of one of those SGPRs fewer than 5 wait states earlier).
* :func:`compile_gated`: compile, disassemble, scan; :class:`HazardError` on any hit (the compile gate).
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import tempfile

__all__ = ["HazardError", "WAIT", "compile_gated", "disassemble", "objdump", "pad_asm_vmem", "scan_text"]

WAIT = 5
_VMEM_ASM = ("global_", "buffer_", "flat_", "scratch_", "tbuffer_")
VOP3B_SDST1 = ("v_add_co_u32", "v_sub_co_u32", "v_subrev_co_u32", "v_addc_co_u32", "v_subb_co_u32", "v_subbrev_co_u32",
               "v_div_scale_f32", "v_div_scale_f64", "v_mad_u64_u32", "v_mad_i64_i32")
UNCOND = ("s_branch", "s_endpgm", "s_setpc_b64", "s_trap", "s_endpgm_saved")
RE_INS = re.compile(r"^\s+([a-z_0-9]+)(.*?)\s*//\s*([0-9A-Fa-f]+):")
RE_FN = re.compile(r"^([0-9a-f]+) <([^>]+)>:")
RE_SREG = re.compile(r"\bs\[(\d+):(\d+)\]|\bs(\d+)\b")


class HazardError(RuntimeError):
    def __init__(self, name: str, hits: list):
        self.hits = hits
        super().__init__(f"{name}: {len(hits)} unpadded VALU-SGPR -> VMEM hazard(s), e.g. {hits[:2]}")


def pad_asm_vmem(kd) -> int:
    """Prefix ``s_nop 4`` to every inline-asm op that issues VMEM with an SGPR input. Returns the padded count."""
    n = 0

    def walk(region) -> None:
        nonlocal n
        for op in region.ops:
            if op.name in ("tile.inline_asm", "tile.inline_asm_multi"):
                t = op.attrs.get("template", "")
                cons = [c.strip() for c in op.attrs.get("constraints", "").split(",")]
                ins = [c for c in cons if not c.startswith("=") and not c.startswith("~")]
                if any(v in t for v in _VMEM_ASM) and "s" in ins and not t.startswith("s_nop 4\n"):
                    op.attrs["template"] = "s_nop 4\n" + t
                    n += 1
            for r in op.regions:
                walk(r)

    walk(kd.body)
    return n


def _sregs(text: str) -> set:
    out = set()
    for m in RE_SREG.finditer(text):
        if m.group(3) is not None:
            out.add(int(m.group(3)))
        else:
            out.update(range(int(m.group(1)), int(m.group(2)) + 1))
    return out


def _split_ops(rest: str) -> list:
    return [o.strip() for o in rest.split(",")] if rest.strip() else []


def scan_text(dis: str) -> tuple[list, int]:
    """``(hits, n_vmem_reading_sgpr)`` of an ``llvm-objdump -d`` listing; a hit is ``dict(addr, vmem, writer, ws,
    regs)``."""
    ins = []
    for ln in dis.splitlines():
        if RE_FN.match(ln):
            continue
        m = RE_INS.match(ln)
        if not m:
            continue
        addr = int(m.group(3), 16)
        mn, ops = m.group(1), _split_ops(m.group(2))
        tgt = None
        if (mn.startswith("s_cbranch") or mn == "s_branch") and ops:
            try:
                tgt = addr + 4 + 4 * int(ops[0], 0)
            except ValueError:
                tgt = None
        ins.append(dict(addr=addr, mn=mn, ops=ops, txt=ln.split("//")[0].strip(), tgt=tgt))
    idx = {d["addr"]: i for i, d in enumerate(ins)}
    preds = [[] for _ in ins]
    for i, d in enumerate(ins):
        mn = d["mn"]
        if i + 1 < len(ins) and not mn.startswith(UNCOND):
            preds[i + 1].append(i)
        if (mn.startswith("s_cbranch") or mn == "s_branch") and d["tgt"] is not None and d["tgt"] in idx:
            preds[idx[d["tgt"]]].append(i)

    def ws(d):
        if d["mn"] == "s_nop":
            try:
                return int(d["ops"][0], 0) + 1
            except (ValueError, IndexError):
                return 1
        return 1

    def writes(d):
        mn = d["mn"]
        if not mn.startswith("v_") or not d["ops"]:
            return set()
        w = _sregs(d["ops"][0])
        if mn.startswith(VOP3B_SDST1) and len(d["ops"]) > 1:
            w |= _sregs(d["ops"][1])
        return w

    hits, nv = [], 0
    for i, d in enumerate(ins):
        if not d["mn"].startswith(_VMEM_ASM):
            continue
        rd = _sregs(",".join(d["ops"]))
        if not rd:
            continue
        nv += 1
        stack = [(p, 0) for p in preds[i]]
        seen = set()
        while stack:
            j, w = stack.pop()
            if (j, w) in seen or w >= WAIT:
                continue
            seen.add((j, w))
            dj = ins[j]
            hit = writes(dj) & rd
            if hit:
                hits.append(dict(addr=hex(d["addr"]), vmem=d["txt"], writer=dj["txt"], ws=w, regs=sorted(hit)))
                continue
            for p in preds[j]:
                stack.append((p, w + ws(dj)))
    return hits, nv


def objdump() -> str:
    """An ``llvm-objdump`` with gfx950 (``LLVM_OBJDUMP``, PATH, ``/opt/rocm*``, torch's / the venv's ROCm SDK, the
    laptop's LLVM build)."""
    c = [os.environ.get("LLVM_OBJDUMP", ""), shutil.which("llvm-objdump") or ""]
    c += sorted(glob.glob("/opt/rocm*/llvm/bin/llvm-objdump"), reverse=True)
    try:
        import importlib.util

        for mod in ("_rocm_sdk_core", "_rocm_sdk_devel"):
            spec = importlib.util.find_spec(mod)
            if spec and spec.submodule_search_locations:
                for loc in spec.submodule_search_locations:
                    c += glob.glob(os.path.join(loc, "lib", "llvm", "bin", "llvm-objdump*"))
    except Exception:  # noqa: BLE001
        pass
    c += ["C:/Users/vanantha/Documents/dev/llvm-project/build-flydsl/bin/llvm-objdump.exe"]
    for p in c:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError("no llvm-objdump found (set LLVM_OBJDUMP)")


def disassemble(hsaco: bytes, tool: str | None = None) -> str:
    tool = tool or objdump()
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "k.hsaco")
        with open(f, "wb") as fh:
            fh.write(hsaco)
        return subprocess.run([tool, "-d", "--mcpu=gfx950", f], capture_output=True, text=True, check=True).stdout


def compile_gated(kd, *, arch: str = "gfx950", tool: str | None = None, keep_dis: dict | None = None):
    """Compile ``kd`` (Python engine), disassemble and scan; raise :class:`HazardError` on any hit. Returns the
    artifact. ``keep_dis``: a dict receiving ``{kernel name: listing}``."""
    from rocke.helpers.compile import compile_kernel

    art = compile_kernel(kd, arch=arch, backend="python")
    dis = disassemble(art.hsaco, tool)
    if keep_dis is not None:
        keep_dis[art.kernel_name] = dis
    hits, _nv = scan_text(dis)
    if hits:
        raise HazardError(art.kernel_name, hits)
    return art
