# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU verify for the gfx1250 greedy (argmax) token sampler.

  PYTHONPATH=Python python3 -m \
      rocke.examples.gfx1250.qwen3_30b_a3b.greedy_sampler_verify --tokens 8 --vocab 151936
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
    p.add_argument("--vocab", type=int, default=151936)
    p.add_argument("--dtype", default="f32", choices=("f32", "bf16", "fp16"))
    p.add_argument("--block-size", type=int, default=256)
    args = p.parse_args()

    import numpy as np

    from rocke.helpers import compile_kernel
    from rocke.instances.gfx1250.qwen3_sampler import (
        Qwen3GreedySamplerSpec,
        build_qwen3_greedy_sampler,
        qwen3_greedy_sampler_grid,
    )
    from rocke.runtime.hip_module import Runtime

    T, V = args.tokens, args.vocab
    rng = np.random.default_rng(0x5A3)
    if args.dtype == "f32":
        npdt = np.float32
        logits = rng.standard_normal((T, V)).astype(np.float32)
    else:
        import ml_dtypes

        npdt = ml_dtypes.bfloat16 if args.dtype == "bf16" else np.float16
        # round to storage dtype so the reference argmax matches the device read
        logits = rng.standard_normal((T, V)).astype(npdt)

    ref = np.argmax(logits.astype(np.float32), axis=1).astype(np.int32)

    spec = Qwen3GreedySamplerSpec(logits_dtype=args.dtype, block_size=args.block_size)
    art = compile_kernel(
        build_qwen3_greedy_sampler(spec, arch=args.arch), arch=args.arch
    )
    print(f"[{args.arch}] built {art.kernel_name} ({art.hsaco_bytes} B)")

    rt = Runtime()
    module = rt.load_module(art.hsaco)
    fn = module.get_function(art.kernel_name)

    def u8(a):
        a = np.ascontiguousarray(a)
        return (ctypes.c_uint8 * int(a.nbytes)).from_buffer_copy(a)

    out = np.zeros(T, np.int32)
    ld = rt.alloc(logits.nbytes)
    od = rt.alloc(out.nbytes)
    rt.memcpy_h2d(ld, u8(logits.view(np.uint8)), logits.nbytes)
    rt.memset(od, 0, out.nbytes)

    grid = qwen3_greedy_sampler_grid(T, spec)
    block = (spec.block_size, 1, 1)
    rt.launch(fn, grid, block, struct.pack("<2Qi", ld, od, V))
    rt.sync()
    obuf = (ctypes.c_uint8 * int(out.nbytes))()
    rt.memcpy_d2h(obuf, od, out.nbytes)
    got = np.frombuffer(bytes(obuf), dtype=np.int32).copy()
    for ptr in (ld, od):
        rt.free(ptr)
    module.unload()

    mism = int((got != ref).sum())
    ok = mism == 0
    print(
        f"[{args.arch}] greedy_sampler {args.dtype} T{T} V{V} bs{args.block_size}: "
        f"mismatches={mism}/{T} -> {'PASS' if ok else 'FAIL'}  "
        f"(got[:4]={got[:4].tolist()} ref[:4]={ref[:4].tolist()})"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
