# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Diff a CK DSL kernel's ISA against a hand-written C++/HIP reference kernel.

The DSL's own ISA introspection (``rocke.analysis``, ``probe_isa_inspect.py``)
only works on ``KernelDef``s. This adds the missing side: compile an arbitrary
C++/HIP reference kernel with ``hipcc``, pull the gfx950 code object out of the
embedded HIP fatbin (``roc-obj-ls``), disassemble it, and diff the instruction
mix against the DSL kernel. Use it whenever you have a hand-tuned reference to
match (a tile-library kernel, a research kernel, a vendor sample).

The diff targets **semantic equivalence** (same opcode mix / run-length), NOT
byte equality: ``hipcc`` and ``libamd_comgr`` use different register allocators
and post-RA schedulers, so register numbers and ``s_nop`` padding differ even
for identical algorithms. We compare instruction-class histograms + run-length
sequences.

CLI:
    python reference_isa_diff.py --cpp ref.cpp --define FOO=1 --include /path
        (prints the reference histogram + run-length)

Programmatic:
    from reference_isa_diff import reference_isa, rocke_isa, diff_report
    print(diff_report("mainloop", reference_isa("ref.cpp"), rocke_isa(kdef)))
"""

from __future__ import annotations

import collections
import itertools
import os
import re
import subprocess
import sys
import tempfile

# Resolve llvm-objdump from $ROCM_PATH (fall back to the conventional path).
ROCM = os.environ.get("ROCM_PATH", "/opt/rocm")
OBJDUMP = os.path.join(ROCM, "llvm", "bin", "llvm-objdump")

# Instruction classes worth tracking for a tiled matmul / attention kernel.
_KEEP = (
    "v_mfma",
    "ds_read",
    "ds_write",
    "buffer_load",
    "global_load",
    "s_barrier",
    "s_setprio",
    "s_waitcnt",
    "v_pk_",
    "sched",
    "iglp",
    "v_perm",
)


def _classes(disasm: str):
    """The tracked-class opcode list, in program order."""
    out = []
    for line in disasm.splitlines():
        if "\t" not in line:
            continue
        toks = line.split("//")[0].strip().split()
        if toks and toks[0].startswith(_KEEP):
            out.append(toks[0])
    return out


def histogram(disasm: str):
    """Instruction-class -> count."""
    return collections.Counter(_classes(disasm))


def runlength(disasm: str, maxlen: int = 60):
    """Run-length-encoded class sequence — the 'shape' of the mainloop."""
    seq = [(k, len(list(g))) for k, g in itertools.groupby(_classes(disasm))]
    return seq[:maxlen]


def disasm_code_object(co_path: str, arch: str = "gfx950") -> str:
    return subprocess.run(
        [OBJDUMP, "-d", f"--mcpu={arch}", co_path],
        capture_output=True,
        text=True,
    ).stdout


def reference_isa(
    cpp_path: str,
    *,
    arch: str = "gfx950",
    defines=(),
    include_dirs=(),
    pybind: bool = False,
    extra=(),
) -> str:
    """Compile a C++/HIP kernel with hipcc and return its ``arch`` disasm.

    ``defines`` / ``include_dirs`` / ``extra`` are passed through verbatim, so
    this is reference-kernel-agnostic — supply whatever the kernel needs.
    ``pybind=True`` adds the pybind11 + Python includes (for kernels that build
    as a Python extension module).
    """
    so = tempfile.mktemp(suffix=".so")
    cmd = [
        "hipcc",
        cpp_path,
        f"--offload-arch={arch}",
        "-std=c++20",
        "-w",
        "-shared",
        "-fPIC",
        "-o",
        so,
    ]
    cmd += [f"-D{d}" for d in defines]
    cmd += list(extra)
    for inc in include_dirs:
        cmd += ["-I", inc]
    cmd += ["-I", os.path.join(ROCM, "include", "hip")]
    if pybind:
        pb = subprocess.run(
            [sys.executable, "-m", "pybind11", "--includes"],
            capture_output=True,
            text=True,
        ).stdout.split()
        import sysconfig

        cmd += pb + ["-I", sysconfig.get_path("include")]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if not os.path.exists(so):
        raise RuntimeError("hipcc failed:\n" + r.stderr[-2000:])
    try:
        ls = subprocess.run(["roc-obj-ls", so], capture_output=True, text=True).stdout
        m = re.search(rf"{arch}\s+file://[^#]+#offset=(\d+)&size=(\d+)", ls)
        if not m:
            raise RuntimeError(f"no {arch} bundle found in fatbin:\n" + ls)
        off, size = int(m.group(1)), int(m.group(2))
        co = tempfile.mktemp(suffix=".co")
        with open(so, "rb") as f:
            f.seek(off)
            data = f.read(size)
        with open(co, "wb") as f:
            f.write(data)
        try:
            return disasm_code_object(co, arch)
        finally:
            os.unlink(co)
    finally:
        os.unlink(so)


def rocke_isa(kernel_def, arch: str = "gfx950") -> str:
    """Lower a rocke ``KernelDef`` and return its ``arch`` disasm."""
    from rocke.helpers import compile_kernel

    art = compile_kernel(kernel_def, arch=arch)
    co = tempfile.mktemp(suffix=".hsaco")
    with open(co, "wb") as f:
        f.write(art.hsaco)
    try:
        return disasm_code_object(co, arch)
    finally:
        os.unlink(co)


def diff_report(name: str, ref_disasm: str, dsl_disasm: str) -> str:
    """Side-by-side instruction-class histogram + a match verdict."""
    rh, dh = histogram(ref_disasm), histogram(dsl_disasm)
    keys = sorted(set(rh) | set(dh))
    lines = [
        f"=== {name}: REF vs ROCKE instruction histogram ===",
        f"  {'class':24s} {'ref':>6s} {'dsl':>6s}  match",
    ]
    ok = True
    for k in keys:
        r, d = rh.get(k, 0), dh.get(k, 0)
        mark = "ok" if r == d else ("~" if abs(r - d) <= max(1, r // 8) else "DIFF")
        if mark == "DIFF":
            ok = False
        lines.append(f"  {k:24s} {r:6d} {d:6d}  {mark}")
    lines.append(f"  verdict: {'SEMANTIC MATCH' if ok else 'MISMATCH (see DIFF rows)'}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpp", required=True, help="reference C++/HIP kernel source")
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument(
        "--define", action="append", default=[], help="-D macro (repeatable)"
    )
    ap.add_argument(
        "--include", action="append", default=[], help="-I dir (repeatable)"
    )
    ap.add_argument("--pybind", action="store_true")
    args = ap.parse_args()
    d = reference_isa(
        args.cpp,
        arch=args.arch,
        defines=args.define,
        include_dirs=args.include,
        pybind=args.pybind,
    )
    print(f"=== {args.cpp} ({args.arch}) instruction histogram ===")
    for k, c in sorted(histogram(d).items()):
        print(f"  {k:24s} {c:6d}")
    print("\n=== run-length (mainloop shape) ===")
    for k, n in runlength(d):
        print(f"  {k} x{n}")
