#!/usr/bin/env python3
"""Compile the toy RMSNorm flyDSL kernel to a bare gfx950 HSACO for one hidden size N.

Mirrors build_vadd.py: COMPILE_ONLY JIT compile (CPU dummy tensors), dump the
gpu-module-to-binary stage, extract the embedded ELF. One N per process invocation
so the two builds get independent dump dirs / kernel-cache state.

Usage:
    build_rmsnorm_toy.py <N> [out.hsaco]
    # default out: <scratch>/rmsnorm_toy_n<N>_gfx950.hsaco
"""
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRATCH = _HERE.parent
sys.path.insert(0, str(_SCRATCH))  # so `import rmsnorm_toy` resolves

from flydsl_build.extract_hsaco import extract_bin_attr  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    N = int(sys.argv[1])
    out_path = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else (_SCRATCH / f"rmsnorm_toy_n{N}_gfx950.hsaco")
    )

    dump_dir = Path(tempfile.mkdtemp(prefix=f"flydsl-build-rmsnorm-n{N}-"))
    os.environ["FLYDSL_DUMP_IR"] = "1"
    os.environ["FLYDSL_DUMP_DIR"] = str(dump_dir)
    os.environ["COMPILE_ONLY"] = "1"

    import torch  # noqa: E402
    import rmsnorm_toy  # noqa: E402

    _kernel, run = rmsnorm_toy.build(N)

    # COMPILE_ONLY: trigger the JIT compile without a real launch. CPU tensors carry
    # enough dtype/shape metadata to compile. Flat (rank-1) views so linear base+k
    # indexing matches rank-1 tensors (rank-2 injects an index-typed stride).
    rows = 4
    x = torch.zeros(rows * N, dtype=torch.float32)
    w = torch.zeros(N, dtype=torch.float32)
    out = torch.empty(rows * N, dtype=torch.float32)
    run(out, x, w, rows, torch.cuda.current_stream())

    stage19 = next(dump_dir.glob("*/19_gpu_module_to_binary.mlir"))
    blob = extract_bin_attr(stage19.read_text(encoding="utf-8", errors="surrogateescape"))
    if blob[:4] != b"\x7fELF":
        raise SystemExit(f"not ELF: {blob[:4]!r}")
    out_path.write_bytes(blob)
    print(f"[build_rmsnorm_toy] N={N} wrote {out_path} ({len(blob)} bytes) from {stage19}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
