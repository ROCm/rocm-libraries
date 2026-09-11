#!/usr/bin/env python3
"""Compile FlyDSL's REAL build_rmsnorm_module to a bare gfx950 HSACO, one (N,dtype).

Mirrors build_rmsnorm_toy.py, but uses the production kernel from the FlyDSL repo
(kernels/norm/rmsnorm_kernel.py::build_rmsnorm_module). Requires the FlyDSL repo on
PYTHONPATH (the repo's kernels/ import cleanly against the venv's flydsl 0.3.2).

Launch wrapper signature (store_rstd=False):  launch_rmsnorm(Input, Gamma, Output, m_in, stream)
Device kernel:  rmsnorm_kernel(Input, Gamma, Rstd, Output)  -- Rstd slot = Gamma ptr (unused).
Geometry: grid=(M,1,1), block=(BLOCK_THREADS=256,1,1).

Usage:
    build_rmsnorm_real.py <N> [dtype=bf16] [out.hsaco]
    # default out: <scratch>/rmsnorm_real_n<N>_<dtype>_gfx950.hsaco
"""
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRATCH = _HERE.parent
_FLYDSL_REPO = Path("/home/AMD/brpepers/FlyDSL")
sys.path.insert(0, str(_SCRATCH))
sys.path.insert(0, str(_FLYDSL_REPO))  # so `import kernels.norm...` resolves

from flydsl_build.extract_hsaco import extract_bin_attr  # noqa: E402


def _torch_dtype(dtype_str):
    import torch

    return {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}[dtype_str]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    N = int(sys.argv[1])
    dtype_str = sys.argv[2] if len(sys.argv) > 2 else "bf16"
    out_path = (
        Path(sys.argv[3])
        if len(sys.argv) > 3
        else (_SCRATCH / f"rmsnorm_real_n{N}_{dtype_str}_gfx950.hsaco")
    )

    dump_dir = Path(tempfile.mkdtemp(prefix=f"flydsl-build-rmsnorm-real-n{N}-"))
    os.environ["FLYDSL_DUMP_IR"] = "1"
    os.environ["FLYDSL_DUMP_DIR"] = str(dump_dir)
    os.environ["COMPILE_ONLY"] = "1"
    os.environ.setdefault("ARCH", "gfx950")

    import torch  # noqa: E402
    from kernels.norm.rmsnorm_kernel import build_rmsnorm_module  # noqa: E402

    launch = build_rmsnorm_module(N, dtype_str)

    td = _torch_dtype(dtype_str)
    M = 4
    Input = torch.zeros((M, N), dtype=td)
    Gamma = torch.zeros((N,), dtype=td)
    Output = torch.empty((M, N), dtype=td)
    launch(Input, Gamma, Output, M, torch.cuda.current_stream())

    stage19 = next(dump_dir.glob("*/19_gpu_module_to_binary.mlir"))
    blob = extract_bin_attr(stage19.read_text(encoding="utf-8", errors="surrogateescape"))
    if blob[:4] != b"\x7fELF":
        raise SystemExit(f"not ELF: {blob[:4]!r}")
    out_path.write_bytes(blob)
    print(f"[build_rmsnorm_real] N={N} dtype={dtype_str} wrote {out_path} ({len(blob)} bytes) from {stage19}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
