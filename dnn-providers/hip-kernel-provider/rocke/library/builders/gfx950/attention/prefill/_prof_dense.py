#!/usr/bin/env python3
"""Minimal single-kernel launcher for rocprofv3 profiling of the rocke dense
prefill kernel. Warmup builds/JITs; then launches the kernel `iters` times so
the profiler captures steady-state dispatches.

Args: S Hq Hkv D causal(0/1) iters
"""
import math
import os
import sys

_HERE = os.path.dirname(__file__)
_RK = os.path.abspath(os.path.join(_HERE, "../../../../.."))
sys.path.insert(0, _RK + "/platform/python")
sys.path.insert(0, _RK + "/library")

import torch  # noqa: E402
from kernels.gfx950.attention_dense import (
    AttentionDenseSpec,
    build_attention_dense,
)  # noqa: E402
from rocke.helpers.compile import compile_kernel  # noqa: E402
from rocke.helpers.spec import SignatureBuilder  # noqa: E402
from rocke.runtime import KernelLauncher, LaunchConfig  # noqa: E402


def main():
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    Hq = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    Hkv = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    D = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    causal = bool(int(sys.argv[5])) if len(sys.argv) > 5 else True
    iters = int(sys.argv[6]) if len(sys.argv) > 6 else 20
    dev = "cuda"
    DT = torch.bfloat16
    torch.manual_seed(0)
    q = (torch.randn(1, S, Hq, D, dtype=DT, device=dev) * 0.2).contiguous()
    k = (torch.randn(1, S, Hkv, D, dtype=DT, device=dev) * 0.2).contiguous()
    v = (torch.randn(1, S, Hkv, D, dtype=DT, device=dev) * 0.2).contiguous()
    out = torch.zeros(1, S, Hq, D, dtype=DT, device=dev)
    spec = AttentionDenseSpec(
        batch=1,
        seqlen_q=S,
        seqlen_kv=S,
        num_query_heads=Hq,
        num_kv_heads=Hkv,
        head_size=D,
        causal=causal,
        dtype="bf16",
        persistent=True,
        num_persistent=256,
    )
    art = compile_kernel(
        build_attention_dense(spec),
        arch="gfx950",
        backend="python",
        capture_ir_text=False,
    )
    sig = (
        SignatureBuilder()
        .ptr("q_ptr", "bf16")
        .ptr("k_ptr", "bf16")
        .ptr("v_ptr", "bf16")
        .ptr("o_ptr", "bf16")
        .scalar("scale", "f32")
        .build()
    )
    lch = KernelLauncher(hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig)
    stream = torch.cuda.current_stream().cuda_stream
    cfg = LaunchConfig(
        grid=(256, 1, 1), block=(spec.num_waves * 64, 1, 1), stream=stream
    )
    vals = {
        "q_ptr": q,
        "k_ptr": k,
        "v_ptr": v,
        "o_ptr": out,
        "scale": 1.0 / math.sqrt(D),
    }
    # warmup / JIT
    lch(vals, config=cfg)
    torch.cuda.synchronize()
    sys.stderr.write(f"[prof_dense] kernel={art.kernel_name}\n")
    for _ in range(iters):
        lch(vals, config=cfg)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
