# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""On-GPU numeric verify for the layout-general SDPA forward (ticket 2).

Builds ``kernels.common.fmha_mfma`` with ``free_stride=True`` for a chosen
contiguous layout (BSHD or BHSD), lays Q/K/V/O out in that layout, launches
through the HIP runtime, and compares the output against the torch-free numpy
dense-attention reference. The
point of the test is layout invariance: the same logical problem must give the
same result whether the operands are stored BSHD or BHSD, because the kernel
reads them through runtime ``(batch, head, token)`` strides plus a per-batch
base-pointer rebase.

It compiles through the comgr / LLVM path (``compile_kernel`` with
``backend="python"``), not the hipcc C++ path: the per-batch rebased O store is
a derived write pointer, which the hipcc backend emits as ``const`` (it rejects
the store), while the LLVM path has no const-qualified pointers. It must run on
a device matching ``--arch``.

    PYTHONPATH=platform/python python3 -m builders.common.sdpa_fwd_free_stride_verify \\
        --arch gfx950 --layout bshd --seqlen-q 128 --seqlen-k 128 --head-size 64 \\
        --heads 8 --kv-heads 2

Run both layouts and confirm they agree with the reference (and each other).
The accumulation order of the f32 matmul chain differs from numpy, so parity is
judged within a tolerance, not bit-for-bit.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import struct

import numpy as np

from rocke.core.arch import ArchTarget
from rocke.helpers.compile import compile_kernel
from rocke.numeric.references import dense_attention_reference
from rocke.runtime.hip_module import Runtime

from kernels.common._fmha_common import FmhaCommonSpec, FmhaShape
from kernels.common._sdpa_fwd_strides import contiguous_strides
from kernels.common.fmha_mfma import (
    FmhaMfmaSpec,
    build_fmha_fwd_mfma,
    fmha_fwd_mfma_grid,
    is_valid_spec,
)

_NP_DTYPE = {"f16": np.float16, "bf16": np.float16}  # bf16 modeled as f16 bits here


def _to_physical(logical: np.ndarray, layout: str) -> np.ndarray:
    """C-contiguous physical buffer for ``logical`` (B, H, S, D) in ``layout``."""
    if layout == "bhsd":
        return np.ascontiguousarray(logical)
    if layout == "bshd":  # [B, S, H, D]
        return np.ascontiguousarray(logical.transpose(0, 2, 1, 3))
    raise ValueError(f"unsupported layout {layout!r}")


def _from_physical(phys_flat: np.ndarray, layout: str, shape) -> np.ndarray:
    """Logical (B, H, S, D) view of a flat physical buffer in ``layout``."""
    B, H, S, D = shape
    if layout == "bhsd":
        return phys_flat.reshape(B, H, S, D)
    return phys_flat.reshape(B, S, H, D).transpose(0, 2, 1, 3)  # bshd


def _reference(ql, kl, vl, *, causal):
    """Reference attention over logical (B, H, S, D) operands; the reference
    consumes one batch of shape (S, H, D) with KV heads expanded to Q heads."""
    B, Hq = ql.shape[0], ql.shape[1]
    Hk = kl.shape[1]
    out = []
    for b in range(B):
        if Hk != Hq:
            kb = np.repeat(kl[b], Hq // Hk, axis=0)
            vb = np.repeat(vl[b], Hq // Hk, axis=0)
        else:
            kb, vb = kl[b], vl[b]
        out.append(
            dense_attention_reference(
                ql[b].transpose(1, 0, 2),
                kb.transpose(1, 0, 2),
                vb.transpose(1, 0, 2),
                causal=causal,
                out_dtype=np.float16,
            )
        )
    return np.stack(out)  # (B, S, Hq, D)


def run_verify(
    *,
    arch,
    layout,
    heads,
    head_size,
    seqlen_q,
    seqlen_k,
    batch,
    kv_heads=0,
    causal=False,
    dtype="f16",
    tol=2e-2,
    verbose=True,
):
    """Build fmha_mfma(free_stride=True) for ``layout``, launch it, and compare
    against the numpy reference. Returns ``(passed, max_abs, bad, total)``;
    raises ``ValueError`` on an invalid spec. Shared by the CLI and the suite's
    GPU-gated numeric test."""
    Hq, D, Hk = heads, head_size, (kv_heads or heads)
    Sq, Sk, B = seqlen_q, seqlen_k, batch

    common = FmhaCommonSpec(
        FmhaShape(head_size=D, num_query_heads=Hq, num_kv_heads=Hk),
        dtype=dtype,
        mask_mode="causal" if causal else "none",
    )
    spec = FmhaMfmaSpec(common=common, seqlen_q=Sq, seqlen_k=Sk, free_stride=True)
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid fmha_mfma free-stride spec: {why}")

    # comgr / LLVM path (backend="python"): the free-stride forward is
    # Python-only for now, and the LLVM path has no const-qualified pointers,
    # so the per-batch rebased O store lowers cleanly (unlike the hipcc path).
    art = compile_kernel(
        build_fmha_fwd_mfma(spec, arch=arch), arch=arch, backend="python"
    )
    if verbose:
        print(f"[{arch}] built {art.kernel_name} ({art.hsaco_bytes} B)")

    np_dt = _NP_DTYPE[dtype]
    rng = np.random.default_rng(0xA11E)
    ql = (rng.standard_normal((B, Hq, Sq, D)) * 0.3).astype(np_dt)
    kl = (rng.standard_normal((B, Hk, Sk, D)) * 0.3).astype(np_dt)
    vl = (rng.standard_normal((B, Hk, Sk, D)) * 0.3).astype(np_dt)

    q_phys = _to_physical(ql, layout)
    k_phys = _to_physical(kl, layout)
    v_phys = _to_physical(vl, layout)
    o_phys = np.zeros_like(q_phys)

    q_b, q_h, q_t = contiguous_strides(layout, heads=Hq, seqlen=Sq, head_size=D)
    k_b, k_h, k_t = contiguous_strides(layout, heads=Hk, seqlen=Sk, head_size=D)
    v_b, v_h, v_t = k_b, k_h, k_t  # V has K's shape/layout
    o_b, o_h, o_t = q_b, q_h, q_t  # O has Q's shape/layout
    scale_log2 = float(1.0 / math.sqrt(D) * math.log2(math.e))

    rt = Runtime()
    module = rt.load_module(art.hsaco)
    fn = module.get_function(art.kernel_name)

    def u8(a):
        return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(np.ascontiguousarray(a))

    qd, kd, vd, od = (rt.alloc(x.nbytes) for x in (q_phys, k_phys, v_phys, o_phys))
    rt.memcpy_h2d(qd, u8(q_phys), q_phys.nbytes)
    rt.memcpy_h2d(kd, u8(k_phys), k_phys.nbytes)
    rt.memcpy_h2d(vd, u8(v_phys), v_phys.nbytes)
    rt.memset(od, 0, o_phys.nbytes)

    # fmha_mfma kernarg order: Q,K,V,O, scale_log2, seqlen_q, seqlen_k, the
    # eight token/head strides, then the four free-stride batch strides.
    packed = struct.pack(
        "<QQQQf" + "i" * 14,
        qd,
        kd,
        vd,
        od,
        scale_log2,
        Sq,
        Sk,
        q_t,
        q_h,
        k_t,
        k_h,
        v_t,
        v_h,
        o_t,
        o_h,
        q_b,
        k_b,
        v_b,
        o_b,
    )
    rt.launch(
        fn,
        fmha_fwd_mfma_grid(spec, batch=B),
        (ArchTarget.from_gfx(arch).wave_size, 1, 1),
        packed,
    )
    rt.sync()
    rt.memcpy_d2h(u8(o_phys), od, o_phys.nbytes)
    for ptr in (qd, kd, vd, od):
        rt.free(ptr)
    module.unload()

    got = _from_physical(o_phys.reshape(-1), layout, (B, Hq, Sq, D)).transpose(
        0, 2, 1, 3
    )
    ref = _reference(ql, kl, vl, causal=causal)
    diff = np.abs(got.astype(np.float32) - ref.astype(np.float32))
    max_abs = float(diff.max())
    bad = int(np.count_nonzero(diff > tol))
    if verbose:
        tag = "PASS" if max_abs <= tol else "FAIL"
        print(
            f"[{arch}] layout={layout} Sq={Sq} Sk={Sk} D={D} Hq={Hq} Hk={Hk} "
            f"causal={causal}: max_abs_diff={max_abs:.3e} bad={bad}/{got.size} "
            f"tol={tol:.0e} -> {tag}"
        )
    return max_abs <= tol, max_abs, bad, got.size


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="gfx950")
    p.add_argument("--layout", choices=("bhsd", "bshd"), default="bshd")
    p.add_argument("--seqlen-q", type=int, default=128)
    p.add_argument("--seqlen-k", type=int, default=128)
    p.add_argument("--head-size", type=int, default=64)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--kv-heads", type=int, default=0, help="0 -> MHA (== heads)")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--dtype", choices=("f16", "bf16"), default="f16")
    p.add_argument("--tol", type=float, default=2e-2)
    args = p.parse_args()

    try:
        passed, _max_abs, _bad, _total = run_verify(
            arch=args.arch,
            layout=args.layout,
            heads=args.heads,
            kv_heads=args.kv_heads,
            head_size=args.head_size,
            seqlen_q=args.seqlen_q,
            seqlen_k=args.seqlen_k,
            batch=args.batch,
            causal=args.causal,
            dtype=args.dtype,
            tol=args.tol,
            verbose=True,
        )
    except ValueError as e:
        print(f"[{args.arch}] {e}")
        return 2
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
