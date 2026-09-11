#!/usr/bin/env python3
"""Compile FlyDSL's REAL dense flash-attention (dualwave-swp) to a bare gfx950 HSACO.

M3 ABI-discovery driver: mirrors build_rmsnorm_real.py. Builds ONE attention HSACO for a
baked (num_heads, head_dim, causal, dtype) tuple so we can read its AMDGPU .args offsets/
sizes from the ELF note and decode the multi-tensor kernarg ABI (the plan's flagged
"hardest bit"). seq_len/batch stay runtime, so one HSACO spans all prefill lengths.

Device kernel: flash_attn_dualwave_swp_gfx950_kernel(Q, K, V, O, LSE, ...).
Launch wrapper (dense, no bias/alibi/sink): launch(q, k, v, o, B, S).
Dense tensor layout: Q/K/V/O = [B, S, H, D] bf16 (batch, seqlen, heads, head_dim).

Usage:
    build_flash_attn_real.py <num_heads> <head_dim> [causal=1] [dtype=bf16] [out.hsaco] [--kv N]
    # --kv/--num-kv-heads N: GQA key/value head count (default = num_heads => MHA). num_heads
    #   must be divisible by N. When N != num_heads the default filename encodes both:
    #   flash_attn_real_h<H>kv<N>_d<D>_...  (e.g. Mistral-7B: --kv 8 with num_heads=32).
    # default out: <scratch>/flash_attn_real_h<H>[kv<N>]_d<D>_<causal|noncausal>_<dtype>_gfx950.hsaco
"""
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRATCH = _HERE.parent
_FLYDSL_REPO = Path("/home/AMD/brpepers/FlyDSL")
sys.path.insert(0, str(_SCRATCH))
sys.path.insert(0, str(_FLYDSL_REPO))

from flydsl_build.extract_hsaco import extract_bin_attr  # noqa: E402


def _torch_dtype(dtype_str):
    import torch

    return {"bf16": torch.bfloat16, "f16": torch.float16}[dtype_str]


def main() -> int:
    argv = sys.argv[1:]
    # Pull the optional GQA flag out of argv first so the remaining positionals stay
    # back-compatible with existing callers: <H> <D> [causal] [dtype] [out].
    num_kv_heads = None
    for flag in ("--kv", "--num-kv-heads"):
        if flag in argv:
            i = argv.index(flag)
            num_kv_heads = int(argv[i + 1])
            del argv[i : i + 2]
    if len(argv) < 2:
        print(__doc__)
        return 2
    num_heads = int(argv[0])
    head_dim = int(argv[1])
    causal = bool(int(argv[2])) if len(argv) > 2 else True
    dtype_str = argv[3] if len(argv) > 3 else "bf16"
    if num_kv_heads is None:
        num_kv_heads = num_heads  # MHA back-compat
    if num_heads % num_kv_heads != 0:
        raise SystemExit(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
    tag = "causal" if causal else "noncausal"
    hcode = f"h{num_heads}" if num_kv_heads == num_heads else f"h{num_heads}kv{num_kv_heads}"
    out_path = (
        Path(argv[4])
        if len(argv) > 4
        else (_SCRATCH / f"flash_attn_real_{hcode}_d{head_dim}_{tag}_{dtype_str}_gfx950.hsaco")
    )

    dump_dir = Path(tempfile.mkdtemp(prefix=f"flydsl-build-fa-h{num_heads}-d{head_dim}-"))
    os.environ["FLYDSL_DUMP_IR"] = "1"
    os.environ["FLYDSL_DUMP_DIR"] = str(dump_dir)
    os.environ["COMPILE_ONLY"] = "1"
    os.environ.setdefault("ARCH", "gfx950")

    import torch  # noqa: E402
    from kernels.attention.flash_attn_gfx950 import build_flash_attn_dualwave_swp_module  # noqa: E402

    launch = build_flash_attn_dualwave_swp_module(
        num_heads=num_heads, head_dim=head_dim, causal=causal, dtype_str=dtype_str,
        num_kv_heads=num_kv_heads,
    )

    td = _torch_dtype(dtype_str)
    B, S = 1, 256  # runtime scalars; any legal prefill shape triggers the compile
    q = torch.zeros((B, S, num_heads, head_dim), dtype=td)
    k = torch.zeros((B, S, num_kv_heads, head_dim), dtype=td)  # GQA: K/V carry kv-head count
    v = torch.zeros((B, S, num_kv_heads, head_dim), dtype=td)
    o = torch.empty((B, S, num_heads, head_dim), dtype=td)
    launch(q, k, v, o, B, S)

    stage19 = next(dump_dir.glob("*/19_gpu_module_to_binary.mlir"))
    blob = extract_bin_attr(stage19.read_text(encoding="utf-8", errors="surrogateescape"))
    if blob[:4] != b"\x7fELF":
        raise SystemExit(f"not ELF: {blob[:4]!r}")
    out_path.write_bytes(blob)
    print(
        f"[build_flash_attn_real] H={num_heads} KV={num_kv_heads} D={head_dim} {tag} dtype={dtype_str} "
        f"wrote {out_path} ({len(blob)} bytes) from {stage19}"
    )
    print(f"[build_flash_attn_real] dump dir (for stage-01/20 ABI decode): {dump_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
