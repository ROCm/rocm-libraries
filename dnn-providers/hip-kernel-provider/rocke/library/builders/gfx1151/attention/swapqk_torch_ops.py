# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Torch custom-op wrapper for the gfx1151 swapqk FMHA kernel.

Registers ``rocke_gfx1151::fmha_swapqk_fwd`` as a graph-capture-safe
torch custom op.  The op takes Q/K/V in standard layout and writes the
attention output into a caller-supplied ``out`` tensor:

    rocke_gfx1151::fmha_swapqk_fwd(q, k, v, out, scale, causal)

Q  : [B, Sq, Hq, D] float16
K  : [B, Sk, Hk, D] float16
V  : [B, Sk, Hk, D] float16  (row-major; op transposes on device internally)
out: [B, Sq, Hq, D] float16  (caller-allocated, written in-place)
scale  : float   (attention scale, e.g. 1/sqrt(D))
causal : bool

The out-parameter ABI is required for HIP graph replay stability.
The kernel is compiled once per (Hq, Hk, D, causal, block_n, k_lds) tuple and
cached; the strategy pair is chosen from the key length by
:func:`_pick_strategy`.

This module lives under ``library/`` rather than ``platform/python/rocke/
instances/`` because it imports ``kernels.gfx1151.wmma_fmha_swapqk``. The
dependency runs ``library -> platform`` only (see rocke/AGENTS.md "Layout"):
platform is standalone-installable and never reaches into ``kernels``.
"""

from __future__ import annotations

import math
import os
import struct
from typing import Any

_NAMESPACE = "rocke_gfx1151"
_OP_NAME = "fmha_swapqk_fwd"
_FULL_NAME = f"{_NAMESPACE}::{_OP_NAME}"
_SCHEMA = (
    f"{_OP_NAME}(Tensor q, Tensor k, Tensor v, Tensor(a!) out, "
    "float scale, bool causal) -> ()"
)
_ARCH = "gfx1151"

# Kernel cache keyed by
# (num_query_heads, num_kv_heads, head_size, causal, block_n, k_lds).
_KERNEL_CACHE: dict[tuple, Any] = {}

# Module-level torch.library handles kept alive for the process lifetime.
_TORCH_LIBS: list[Any] = []
_REGISTERED = False

# (min_seqlen_k, block_n, k_lds), ASCENDING in min_seqlen_k; the last row that
# both matches and exactly divides seqlen_k wins.
#
# One row, because bn64+k_lds measured the fastest arm at EVERY sequence length
# tried -- there is nothing to switch between. Interleaved, 3 reps, min per rep,
# Hq32/Hk8/D128/causal/gqa_fuse=4, dispatch us (rocprofv3):
#
#   S     bn64    bn64+k_lds   bn128   Triton
#   1024   279.7     269.4     396.0    423.3
#   2048  1231.2    1112.8    1652.4   1795.4
#   4096  7484.4    4730.7    6579.7   6341.1
#   8192 37708.0   20936.7   28320.5  23945.4
#
# This retires the old two-row table. bn128 used to win past S=4096 by buying L0
# hits with a bigger tile; staging K in LDS buys strictly more of them (L2
# requests 59.0M -> 17.5M at S=8192) without bn128's 112 B of scratch, so bn128 is
# now dominated at every S -- by 1.39x at S=4096 and 1.35x at S=8192. Dropping it
# also widens eligibility, since bn64 divides every seqlen bn128 does and more.
# k_lds is a bn64-only lever by LDS budget; see SwapQKCfg.k_lds and
# rocke/docs/k_lds_staging_08_18_2026.md.
_BLOCK_N_TABLE = ((0, 64, True),)

# "auto" (default) walks the table; "64"/"128" pin it for A/B control arms.
_BLOCK_N = os.environ.get("ROCKE_BLOCK_N", "auto")
# "auto" (default) takes k_lds from the table; "0"/"1" pin it for A/B control.
_K_LDS = os.environ.get("ROCKE_K_LDS", "auto")


def _pick_strategy(seqlen_k: int) -> tuple[int, bool]:
    """Measured-winning ``(block_n, k_lds)`` for this key length.

    ``block_n`` is 0 when no tile exactly divides ``seqlen_k``. The kernel's kv
    loop bound is ``seqlen_k // block_n``, which truncates the tail instead of
    masking it, so a non-divisible launch is a wrong answer rather than a slow
    one -- callers must treat 0 as "not eligible for swapqk".

    """
    best = (0, False)
    for lo, bn, klds in _BLOCK_N_TABLE:
        if seqlen_k >= lo and seqlen_k % bn == 0:
            best = (bn, klds)
    if _BLOCK_N != "auto":
        bn = int(_BLOCK_N)
        best = (bn if seqlen_k % bn == 0 else 0, best[1])
    if _K_LDS != "auto":
        best = (best[0], _K_LDS == "1")
    return best


def _get_launcher(
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    causal: bool,
    block_n: int,
    k_lds: bool,
):
    """Return a compiled+cached (hsaco_bytes, kernel_name, block_size) triple."""
    key = (num_query_heads, num_kv_heads, head_size, causal, block_n, k_lds)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    from kernels.gfx1151.wmma_fmha_swapqk import (
        SwapQKCfg,
        build_wmma_fmha_swapqk,
        is_valid_spec,
    )
    from rocke.helpers import compile_kernel

    cfg = SwapQKCfg(
        head_size=head_size,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        mask_mode="causal" if causal else "none",
        v_transposed=True,
        block_n=block_n,
        k_lds=k_lds,
    )
    ok, why = is_valid_spec(cfg, _ARCH)
    if not ok:
        raise RuntimeError(f"invalid swapqk config: {why}")

    art = compile_kernel(build_wmma_fmha_swapqk(cfg, arch=_ARCH), arch=_ARCH)

    from rocke.runtime.launcher import KernelLauncher
    from rocke.runtime.hip_module import Runtime

    rt = Runtime()
    module = rt.load_module(art.hsaco)
    fn = module.get_function(art.kernel_name)

    entry = (art.hsaco, art.kernel_name, cfg.block_size, fn, module, rt, cfg)
    _KERNEL_CACHE[key] = entry
    return entry


def _launch_swapqk(q, k, v, out, scale: float, causal: bool) -> None:
    import math
    import struct

    B, Sq, Hq, D = q.shape
    _B, Sk, Hk, _D = k.shape

    block_n, k_lds = _pick_strategy(Sk)
    if block_n == 0:
        raise RuntimeError(
            f"swapqk cannot serve seqlen_k={Sk}: the kv loop truncates rather "
            f"than masks its tail, so seqlen_k must be a multiple of one of the "
            f"supported kv tiles {sorted({bn for _, bn, _k in _BLOCK_N_TABLE})}"
        )

    hsaco, kernel_name, block_size, fn, module, rt, cfg = _get_launcher(
        Hq, Hk, D, causal, block_n, k_lds
    )

    from kernels.gfx1151.wmma_fmha_swapqk import swapqk_grid

    # V reaches the kernel transposed, [B,Sk,Hk,D] -> [B,Hk,D,Sk]: the device
    # twin of swapqk_transpose_v's np.transpose(v, (0, 2, 3, 1)). permute() is a
    # view, contiguous() is the copy, and it is issued on torch's current stream
    # -- the same stream resolve_stream(0) hands the launch below, so the copy is
    # ordered ahead of the kernel. The numpy round trip this replaces had to
    # synchronize on the device every call, and produced a buffer the torch
    # caching allocator had never seen, which is the one case where that stream
    # ordering does not hold.
    v_t = v.permute(0, 2, 3, 1).contiguous()

    scale_log2 = float(scale * math.log2(math.e))
    grid = swapqk_grid(cfg, seqlen_q=Sq, batch=B)
    block = (block_size, 1, 1)

    qd = q.data_ptr()
    kd = k.data_ptr()
    vd = v_t.data_ptr()
    od = out.data_ptr()

    # ABI: 4×uint64 pointers, 1×float, 10×int32
    # (matches struct.pack "<QQQQfiiiiiiiiii" in wmma_fmha_swapqk_verify.py)
    packed = struct.pack(
        "<QQQQfiiiiiiiiii",
        qd,
        kd,
        vd,
        od,
        scale_log2,
        Sq,
        Sk,
        Hq * D,
        D,  # Q strides: row_stride, head_stride
        Hk * D,
        D,  # K strides
        Hk * D,
        D,  # V strides (over transposed layout, kernel re-derives internally)
        Hq * D,
        D,  # O strides
    )

    import ctypes

    packed_buf = (ctypes.c_uint8 * len(packed)).from_buffer(bytearray(packed))

    from rocke.runtime.torch_interop import resolve_stream

    stream = resolve_stream(0, device=q.device)

    rt.launch(fn, grid, block, packed_buf, stream=stream)

    # v_t falls out of scope here on purpose. It is a torch-allocated tensor
    # produced on the launch stream, so the caching allocator queues its release
    # behind this kernel instead of handing the memory to the next torch.empty.
    # That is exactly the guarantee the old host-staged buffer did not have.


def fmha_swapqk_fwd_impl(q, k, v, out, scale: float, causal: bool) -> None:
    """GPU implementation of rocke_gfx1151::fmha_swapqk_fwd."""
    _launch_swapqk(q, k, v, out, scale, causal)


def fmha_swapqk_fwd_meta(q, k, v, out, scale: float, causal: bool) -> None:
    """Shape-only meta implementation (no GPU, used by torch.compile tracing)."""
    return None


def register_torch_custom_ops(namespace: str = _NAMESPACE) -> bool:
    """Register fmha_swapqk_fwd as a torch custom op. Idempotent."""
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        import torch
    except Exception:
        return False

    try:
        def_lib = torch.library.Library(namespace, "DEF")
        def_lib.define(_SCHEMA)

        impl_lib = torch.library.Library(namespace, "IMPL")
        impl_lib.impl(_OP_NAME, fmha_swapqk_fwd_impl, "CompositeExplicitAutograd")

        meta_lib = torch.library.Library(namespace, "IMPL")
        meta_lib.impl(_OP_NAME, fmha_swapqk_fwd_meta, "Meta")
    except RuntimeError as exc:
        if "Only a single TORCH_LIBRARY" not in str(exc) and "already" not in str(exc):
            raise

    _TORCH_LIBS.extend([def_lib, impl_lib, meta_lib])
    _REGISTERED = True
    return True
