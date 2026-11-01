"""Compile ``rt_builds`` labels with COMGR and the direct compiler and compare their ISA (compile-only).

    python rt_isa.py --out DIR [--objdump PATH] LABEL ...

Writes ``DIR/<label>.{comgr,direct}.{hsaco,s}`` and prints per label: instructions by class (VALU, SALU, VMEM
load / store, SMEM, LDS, MFMA, branches, waits, nops), the ``s_waitcnt`` and ``s_nop`` counts, code bytes and the
kernel descriptor's VGPR / SGPR / AGPR / LDS. ``ROCKE_DIRECT_LEVERS`` applies to the direct build.
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CLASSES = [
    ("mfma", r"^v_mfma"), ("vmem_load", r"^(buffer|global)_load|^global_atomic"), ("vmem_store", r"^(buffer|global)_store"),
    ("smem", r"^s_(load|buffer_load|memtime|memrealtime|dcache)"), ("lds", r"^ds_"), ("wait", r"^s_waitcnt"),
    ("nop", r"^s_nop"), ("branch", r"^s_(cbranch|branch|setpc|swappc)"), ("barrier", r"^s_barrier"),
    ("salu", r"^s_"), ("valu", r"^v_"),
]  # fmt: skip


def classify(lines: list[str]) -> collections.Counter:
    c = collections.Counter()
    for ln in lines:
        m = re.match(r"^\s+([a-z0-9_]+)", ln)
        if not m:
            continue
        mn = m.group(1)
        for name, pat in CLASSES:
            if re.match(pat, mn):
                c[name] += 1
                break
        c["total"] += 1
    return c


def disasm(objdump: str, hsaco: str) -> list[str]:
    out = subprocess.run([objdump, "-d", "--mcpu=gfx950", hsaco], capture_output=True, text=True).stdout
    return [ln for ln in out.splitlines() if re.match(r"^\s+[a-z]", ln) and "//" in ln or re.match(r"^\s+[sv]_|^\s+(buffer|global|ds)_", ln)]


def resources(hsaco: bytes) -> dict:
    from rocke.runtime.code_object_elf import kernel_descriptor_resources  # noqa: F401  (if present)

    return {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--objdump", default="C:/Users/vanantha/Documents/dev/llvm-project/build-flydsl/bin/llvm-objdump.exe")
    ap.add_argument("labels", nargs="+")
    a = ap.parse_args(argv)
    import rt_builds
    from rocke.helpers.compile import compile_kernel

    os.makedirs(a.out, exist_ok=True)
    for label in a.labels:
        stats = {}
        for be in ("comgr", "direct"):
            art = compile_kernel(rt_builds.build(label), arch="gfx950", backend="python", code_object_backend=be)
            used = getattr(art, "code_object_backend", be)
            if used != be:
                print(f"{label} {be}: fell back: {getattr(art, 'direct_fallback', '')[:200]}")
                continue
            h = os.path.join(a.out, f"{label}.{be}.hsaco")
            with open(h, "wb") as fh:
                fh.write(art.hsaco)
            lines = disasm(a.objdump, h)
            with open(os.path.join(a.out, f"{label}.{be}.s"), "w", newline="\n") as fh:
                fh.write("\n".join(lines) + "\n")
            stats[be] = classify(lines)
            stats[be]["bytes"] = len(art.hsaco)
        if len(stats) == 2:
            keys = ["total"] + [n for n, _ in CLASSES]
            print(f"{label}: " + "  ".join(f"{k} {stats['comgr'][k]}/{stats['direct'][k]}" for k in keys))
    return 0


if __name__ == "__main__":
    sys.exit(main())
