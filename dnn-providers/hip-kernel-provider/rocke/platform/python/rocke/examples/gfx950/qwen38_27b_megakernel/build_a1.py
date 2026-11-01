# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build the A1 Qwen3.8-27B decode megakernel code object (CPU only, no GPU needed).

    python build_a1.py OUT_DIR [--spec p2c_full_a1]

Steps:
  1. ``a1_kernel.build_full(spec)`` builds the rocKE IR (one kernel, one pointer-table argument).
  2. rocKE lowers it to LLVM IR with the python backend (``backend="python"``; the kernel adds
     intrinsic-call ops that only that backend knows).
  3. The IR calls OCML (``pow``/``cos``/``sin``/``exp2``, used by RoPE and the attention gate), so it is
     linked with the ROCm device library ``ocml.bc`` (``llvm-link --only-needed --internalize``) plus the
     three ``__oclc_*`` control constants that FlyDSL's serializer would also add.
  4. rocKE's comgr path compiles the linked IR at ``-O3`` to ``kernel.hsaco``.

Writes OUT_DIR/{kernel.ll, kernel.hsaco, kernel.s, notes.txt, object.json}. ``object.json`` holds
``symbol`` and ``rows``, which is what ATOM's ``Megakernel(object_dir)`` loader reads.

Toolchain lookup (override with environment variables):
  LLVM_BIN  directory with llvm-link / llvm-readelf / llvm-objdump
            (default: the first that holds llvm-link of ``$ROCM_PATH/../_rocm_sdk_core/lib/llvm/bin``
            (TheRock wheels), ``$ROCM_PATH/llvm/bin``, ``$ROCM_PATH/lib/llvm/bin``; ROCM_PATH defaults to /opt/rocm)
  OCML_BC   path to ocml.bc (default: ``$LLVM_BIN/../amdgcn/bitcode/ocml.bc``)
  ROCKE_COMGR_LIB  comgr library, if rocKE cannot find it on its own.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rocke.helpers.compile import (  # noqa: E402
    _comgr_options_for_kernel,
    _lower_llvm_via_backend,
)
from rocke.runtime.comgr import build_hsaco_from_llvm_ir  # noqa: E402

import a1_kernel  # noqa: E402

ARCH = "gfx950"
OCLC = (
    "@__oclc_finite_only_opt = hidden local_unnamed_addr addrspace(4) constant i8 0, align 1\n"
    "@__oclc_unsafe_math_opt = hidden local_unnamed_addr addrspace(4) constant i8 0, align 1\n"
    "@__oclc_ISA_version = hidden local_unnamed_addr addrspace(4) constant i32 9500, align 4\n"
)
# Reference build of the A1 object (ROCm 10.0 TheRock wheel from the ATOM blog-benchmark image; first built with
# rocKE develop 1e36b592, reproduced on develop d1de2108). A different compiler or ocml.bc gives different bytes.
REFERENCE = {
    "p2c_full_a1": {
        "kernel_ll_sha256": "612296e3afb932f64f20628636b06034643b97fb8327d0f17843aefc3dd66257",
        "hsaco_sha256": "1f710cd33adee782cb2abd7304f4e6d778f7f36d8e1ac896df00d70dc9dd4a9e",
    }
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def llvm_bin() -> Path:
    if os.environ.get("LLVM_BIN"):
        return Path(os.environ["LLVM_BIN"])
    rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    for cand in (
        rocm.parent / "_rocm_sdk_core" / "lib" / "llvm" / "bin",
        rocm / "llvm" / "bin",
        rocm / "lib" / "llvm" / "bin",
    ):
        if (cand / "llvm-link").exists():
            return cand
    raise SystemExit("llvm-link not found: set LLVM_BIN")


def link_ocml(ll: str, out: Path, llvm: Path) -> str:
    if "@__ocml_" not in ll:
        return ll
    ocml = Path(
        os.environ.get("OCML_BC", llvm.parent / "amdgcn" / "bitcode" / "ocml.bc")
    )
    src, dst = out / "kernel.ocml_in.ll", out / "kernel.linked.ll"
    src.write_text(ll + "\n" + OCLC)
    try:
        subprocess.run(
            [
                str(llvm / "llvm-link"),
                "--only-needed",
                "--internalize",
                str(src),
                str(ocml),
                "-S",
                "-o",
                str(dst),
            ],
            check=True,
        )
        return dst.read_text()
    finally:
        src.unlink(missing_ok=True)
        dst.unlink(missing_ok=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("out", type=Path)
    ap.add_argument(
        "--spec", default="p2c_full_a1", help="kernel spec from a1_kernel.specs()"
    )
    args = ap.parse_args(argv)
    spec = next((s for s in a1_kernel.specs() if s.name == args.spec), None)
    if spec is None:
        raise SystemExit(
            f"unknown spec {args.spec}; choose from {[s.name for s in a1_kernel.specs()]}"
        )
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    llvm = llvm_bin()

    kernel = a1_kernel.build_full(spec)
    ll = _lower_llvm_via_backend(kernel, arch=ARCH, backend="python", spec=None)
    (out / "kernel.ll").write_text(ll)
    blob, _ = build_hsaco_from_llvm_ir(
        link_ocml(ll, out, llvm), options=_comgr_options_for_kernel(kernel)
    )
    (out / "kernel.hsaco").write_bytes(blob)

    notes = subprocess.run(
        [str(llvm / "llvm-readelf"), "--notes", str(out / "kernel.hsaco")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    (out / "notes.txt").write_text(notes)
    dis = subprocess.run(
        [str(llvm / "llvm-objdump"), "-d", f"--mcpu={ARCH}", str(out / "kernel.hsaco")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    (out / "kernel.s").write_text(dis)

    meta = {
        "symbol": kernel.name,
        "name": spec.name,
        "rows": 8,
        "bytes": len(blob),
        "sha256": sha256(blob),
        "kernel_ll_sha256": sha256(ll.encode()),
        "knobs": list(spec.knobs),
    }
    (out / "object.json").write_text(json.dumps(meta, indent=1) + "\n")
    ref = REFERENCE.get(spec.name)
    if ref:
        meta["matches_reference_ir"] = (
            meta["kernel_ll_sha256"] == ref["kernel_ll_sha256"]
        )
        meta["matches_reference_hsaco"] = meta["sha256"] == ref["hsaco_sha256"]
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
