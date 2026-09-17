#!/usr/bin/env python3
"""Repeatably compile the POC `vadd` flyDSL kernel to a bare gfx950 HSACO.

This is the M0 "author the missing build step": today the .hsaco was produced by
hand on the laptop; this makes it a one-command, deterministic build on the
gfx950 box. It compiles COMPILE_ONLY (no GPU launch needed), dumps the
`gpu-module-to-binary` stage, and extracts the embedded ELF via extract_hsaco.

Usage:
    build_vadd.py [out.hsaco]     # default: <scratch>/vadd_gfx950.hsaco
"""
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRATCH = _HERE.parent
sys.path.insert(0, str(_SCRATCH))  # so `import vadd` (the kernel source) resolves

from flydsl_build.extract_hsaco import extract_bin_attr  # noqa: E402


def main() -> int:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (_SCRATCH / "vadd_gfx950.hsaco")

    dump_dir = Path(tempfile.mkdtemp(prefix="flydsl-build-vadd-"))
    os.environ["FLYDSL_DUMP_IR"] = "1"
    os.environ["FLYDSL_DUMP_DIR"] = str(dump_dir)
    os.environ["COMPILE_ONLY"] = "1"

    import torch  # noqa: E402

    import vadd  # noqa: E402  (the flyDSL @kernel/@jit source)

    # COMPILE_ONLY: trigger the JIT compile without a real launch. CPU tensors
    # carry enough dtype/shape metadata to compile.
    N = 256
    a = torch.zeros(N, dtype=torch.float32)
    b = torch.zeros(N, dtype=torch.float32)
    out = torch.empty(N, dtype=torch.float32)
    vadd.run(out, a, b, torch.cuda.current_stream())

    stage19 = next(dump_dir.glob("*/19_gpu_module_to_binary.mlir"))
    blob = extract_bin_attr(stage19.read_text(encoding="utf-8", errors="surrogateescape"))
    if blob[:4] != b"\x7fELF":
        raise SystemExit(f"not ELF: {blob[:4]!r}")
    out_path.write_bytes(blob)
    print(f"[build_vadd] wrote {out_path} ({len(blob)} bytes) from {stage19}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
