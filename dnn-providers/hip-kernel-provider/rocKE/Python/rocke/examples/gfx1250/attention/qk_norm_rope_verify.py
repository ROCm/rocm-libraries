# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU verify for the gfx1250 Qwen3 fused QK-norm + RoPE kernel.

Runs the fused per-head RMSNorm + rotary on a gfx1250 device for a Q-or-K tensor
``[tokens, num_heads, head_dim]`` and compares against a numpy reference that
models the kernel's bf16 storage (bf16 in -> f32 compute -> bf16 out).

  PYTHONPATH=Python python3 -m \
      rocke.examples.gfx1250.attention.qk_norm_rope_verify \
      --tokens 4 --num-heads 32 --head-dim 64
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
    p.add_argument("--tokens", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=32)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"))
    p.add_argument("--layout", default="half", choices=("half", "interleaved"))
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--tol", type=float, default=2e-2)
    args = p.parse_args()

    import ml_dtypes
    import numpy as np

    from rocke.helpers import compile_kernel
    from rocke.instances.gfx1250.qwen3_qk_norm_rope import (
        Qwen3QkNormRopeSpec,
        build_qwen3_qk_norm_rope,
        qwen3_qk_norm_rope_grid,
    )
    from rocke.runtime.hip_module import Runtime

    T, nh, H = args.tokens, args.num_heads, args.head_dim
    half = H // 2
    st = ml_dtypes.bfloat16 if args.dtype == "bf16" else np.float16
    rng = np.random.default_rng(0x9E3)

    x = (rng.standard_normal((T, nh, H)) * 0.5).astype(np.float32)
    weight = (rng.uniform(0.5, 1.5, size=H)).astype(np.float32)
    positions = np.arange(T, dtype=np.int32)
    max_pos = int(positions.max()) + 1

    # rotary tables [max_pos, half] (theta arbitrary; ref + device share them).
    inv_freq = (1.0 / (10000.0 ** (np.arange(half) / half))).astype(np.float32)
    ang = positions[:, None].astype(np.float32) * inv_freq[None, :]  # (T, half)
    cos_tab = np.zeros((max_pos, half), np.float32)
    sin_tab = np.zeros((max_pos, half), np.float32)
    cos_tab[positions] = np.cos(ang)
    sin_tab[positions] = np.sin(ang)

    def pair(i):
        return (2 * i, 2 * i + 1) if args.layout == "interleaved" else (i, i + half)

    # ---- numpy reference (bf16 in -> f32 -> bf16 out) ----
    xb = x.astype(st).astype(np.float32)
    ref = np.zeros((T, nh, H), np.float32)
    for t in range(T):
        for h in range(nh):
            v = xb[t, h]
            inv = 1.0 / np.sqrt(np.mean(v * v) + args.eps)
            xn = v * inv * weight
            o = np.empty(H, np.float32)
            for i in range(half):
                lo_i, hi_i = pair(i)
                c, s = cos_tab[positions[t], i], sin_tab[positions[t], i]
                o[lo_i] = xn[lo_i] * c - xn[hi_i] * s
                o[hi_i] = xn[lo_i] * s + xn[hi_i] * c
            ref[t, h] = o
    ref = ref.astype(st).astype(np.float32)

    spec = Qwen3QkNormRopeSpec(
        num_heads=nh,
        head_dim=H,
        dtype=args.dtype,
        eps=args.eps,
        rope_layout=args.layout,
    )
    art = compile_kernel(build_qwen3_qk_norm_rope(spec, arch=args.arch), arch=args.arch)
    print(f"[{args.arch}] built {art.kernel_name} ({art.hsaco_bytes} B)")

    rt = Runtime()
    module = rt.load_module(art.hsaco)
    fn = module.get_function(art.kernel_name)

    def u8(a):
        a = np.ascontiguousarray(a)
        return (ctypes.c_uint8 * int(a.nbytes)).from_buffer_copy(a)

    x_st = x.astype(st)
    out = np.zeros((T, nh, H), st)
    xd = rt.alloc(x_st.nbytes)
    wd = rt.alloc(weight.nbytes)
    cd = rt.alloc(cos_tab.nbytes)
    sd = rt.alloc(sin_tab.nbytes)
    pd = rt.alloc(positions.nbytes)
    od = rt.alloc(out.nbytes)
    rt.memcpy_h2d(xd, u8(x_st.view(np.uint8)), x_st.nbytes)
    rt.memcpy_h2d(wd, u8(weight), weight.nbytes)
    rt.memcpy_h2d(cd, u8(cos_tab), cos_tab.nbytes)
    rt.memcpy_h2d(sd, u8(sin_tab), sin_tab.nbytes)
    rt.memcpy_h2d(pd, u8(positions), positions.nbytes)
    rt.memset(od, 0, out.nbytes)

    grid = qwen3_qk_norm_rope_grid(T, spec)
    block = (spec.block_size, 1, 1)
    packed = struct.pack("<6Qi", xd, wd, cd, sd, pd, od, T)
    rt.launch(fn, grid, block, packed)
    rt.sync()
    obuf = (ctypes.c_uint8 * int(out.nbytes))()
    rt.memcpy_d2h(obuf, od, out.nbytes)
    got = np.frombuffer(bytes(obuf), dtype=st).reshape(T, nh, H).astype(np.float32)
    for ptr in (xd, wd, cd, sd, pd, od):
        rt.free(ptr)
    module.unload()

    diff = np.abs(got - ref)
    denom = np.maximum(np.abs(ref), 1.0)
    max_rel = float((diff / denom).max())
    max_abs = float(diff.max())
    ok = max_rel <= args.tol
    print(
        f"[{args.arch}] qk_norm_rope {args.dtype} T{T} nh{nh} d{H} {args.layout}: "
        f"max_abs={max_abs:.3e} max_rel={max_rel:.3e} tol={args.tol:.0e} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
