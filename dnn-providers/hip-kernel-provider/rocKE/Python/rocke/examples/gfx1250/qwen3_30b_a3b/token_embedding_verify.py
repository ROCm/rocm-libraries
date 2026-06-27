# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU verify for the gfx1250 token-embedding gather kernel.

  PYTHONPATH=Python python3 -m \
      rocke.examples.gfx1250.qwen3_30b_a3b.token_embedding_verify \
      --tokens 8 --hidden 2048 --vocab 4096
"""

from __future__ import annotations

import argparse
import ctypes
import struct


def main() -> int:
    from rocke.runtime.comgr import prefer_bundled_lib

    prefer_bundled_lib()  # pin newest comgr/LLVM flavor before lowering (gfx1250 needs ROCm>=7.2)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="gfx1250")
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--vocab", type=int, default=4096)
    p.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"))
    p.add_argument("--vec", type=int, default=8)
    args = p.parse_args()

    import ml_dtypes
    import numpy as np

    from rocke.helpers import compile_kernel
    from rocke.instances.gfx1250.qwen3_token_embedding import (
        Qwen3TokenEmbeddingSpec,
        build_qwen3_token_embedding,
        qwen3_token_embedding_grid,
    )
    from rocke.runtime.hip_module import Runtime

    T, H, V = args.tokens, args.hidden, args.vocab
    st = ml_dtypes.bfloat16 if args.dtype == "bf16" else np.float16
    rng = np.random.default_rng(0xE3B)

    table = (rng.standard_normal((V, H)) * 0.5).astype(st)
    ids = rng.integers(0, V, size=T).astype(np.int32)
    ref = table[ids].astype(np.float32)

    spec = Qwen3TokenEmbeddingSpec(hidden=H, dtype=args.dtype, vec=args.vec)
    art = compile_kernel(
        build_qwen3_token_embedding(spec, arch=args.arch), arch=args.arch
    )
    print(f"[{args.arch}] built {art.kernel_name} ({art.hsaco_bytes} B)")

    rt = Runtime()
    module = rt.load_module(art.hsaco)
    fn = module.get_function(art.kernel_name)

    def u8(a):
        a = np.ascontiguousarray(a)
        return (ctypes.c_uint8 * int(a.nbytes)).from_buffer_copy(a)

    out = np.zeros((T, H), st)
    idd = rt.alloc(ids.nbytes)
    td = rt.alloc(table.nbytes)
    od = rt.alloc(out.nbytes)
    rt.memcpy_h2d(idd, u8(ids), ids.nbytes)
    rt.memcpy_h2d(td, u8(table.view(np.uint8)), table.nbytes)
    rt.memset(od, 0, out.nbytes)

    grid = qwen3_token_embedding_grid(T, spec)
    block = (spec.block_size, 1, 1)
    rt.launch(fn, grid, block, struct.pack("<3Qi", idd, td, od, T))
    rt.sync()
    obuf = (ctypes.c_uint8 * int(out.nbytes))()
    rt.memcpy_d2h(obuf, od, out.nbytes)
    got = np.frombuffer(bytes(obuf), dtype=st).reshape(T, H).astype(np.float32)
    for ptr in (idd, td, od):
        rt.free(ptr)
    module.unload()

    max_abs = float(np.abs(got - ref).max())
    ok = max_abs == 0.0
    print(
        f"[{args.arch}] token_embedding {args.dtype} T{T} H{H} V{V} vec{args.vec}: "
        f"max_abs={max_abs:.3e} -> {'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
