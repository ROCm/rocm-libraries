# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""On-GPU numeric verify for the layout-general SDPA forward.

Builds ``kernels.common.fmha_mfma`` with ``free_stride=True``, lays Q/K/V/O out
in a chosen physical placement, launches through the HIP runtime, and compares
the output against the torch-free numpy dense-attention reference. The point of
the test is layout invariance: the same logical problem must give the same
result however the operands are stored, because the kernel reads them through
runtime ``(batch, head, token)`` strides plus a per-batch base-pointer rebase.

Four placements, covering the axes the free-stride ABI claims:

* ``bhsd`` / ``bshd`` -- contiguous, the two 4D orders.
* ``head_pad`` -- a padded head dim ([B, H, S, D+pad] sliced to D), so every
  stride carries the padding while the head dim keeps unit stride.
* ``packed`` -- Q/K/V share one [B, S, 3, H, D] buffer and are distinguished
  only by their base offsets (0, H*D, 2*H*D), so the three operands alias one
  allocation at non-zero offsets.

Strides are read off the numpy view rather than assumed, so a placement
describes itself and the kernel has to reproduce it. It must run on a device
matching ``--arch``.

    PYTHONPATH=platform/python python3 -m builders.common.sdpa_fwd_free_stride_verify \\
        --arch gfx950 --layout bshd --seqlen-q 128 --seqlen-k 128 --head-size 64 \\
        --heads 8 --kv-heads 2

Run the placements and confirm they agree with the reference (and each other).
The accumulation order of the f32 matmul chain differs from numpy, so parity is
judged within a tolerance, not bit-for-bit.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import struct
from typing import NamedTuple

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

# bf16 is rejected by is_valid_spec (the f16-only tiled FMHA v1), so there is no
# bf16 entry to model: adding one would feed f16 bit patterns to a bf16 kernel.
_NP_DTYPE = {"f16": np.float16}


class VerifyResult(NamedTuple):
    passed: bool
    max_abs: float
    bad: int
    total: int
    out: np.ndarray  # logical (B, S, Hq, D), matching the reference


def _place(logical: np.ndarray, layout: str, *, head_pad: int = 0):
    """``(backing, view)`` for one tensor: a C-contiguous buffer and the logical
    ``(B, H, S, D)`` view into it. ``head_pad`` widens the head dim in the
    buffer only, so the batch/head/token strides all carry the padding while the
    head dim keeps unit stride."""
    B, H, S, D = logical.shape
    Dp = D + head_pad
    if layout == "bhsd":  # [B, H, S, D]
        backing = np.zeros((B, H, S, Dp), logical.dtype)
        backing[..., :D] = logical
        return backing, backing[..., :D]
    if layout == "bshd":  # [B, S, H, D]
        backing = np.zeros((B, S, H, Dp), logical.dtype)
        backing[..., :D] = logical.transpose(0, 2, 1, 3)
        return backing, backing[..., :D].transpose(0, 2, 1, 3)
    raise ValueError(f"unsupported layout {layout!r}")


def _place_packed(ql: np.ndarray, kl: np.ndarray, vl: np.ndarray):
    """``(backing, [q_view, k_view, v_view])`` for Q/K/V interleaved in one
    ``[B, S, 3, H, D]`` buffer. The three operands alias a single allocation and
    differ only by base offset (0, H*D, 2*H*D), which is the case that exercises
    the body's aligned vector loads at a non-zero base."""
    B, H, S, D = ql.shape
    backing = np.zeros((B, S, 3, H, D), ql.dtype)
    for i, t in enumerate((ql, kl, vl)):
        backing[:, :, i] = t.transpose(0, 2, 1, 3)
    return backing, [backing[:, :, i].transpose(0, 2, 1, 3) for i in range(3)]


def _base_and_strides(view: np.ndarray, backing: np.ndarray):
    """Element ``(base_offset, (s_b, s_h, s_t))`` of ``view`` into C-contiguous
    ``backing``. The head dim must come out unit-stride: that is the
    ``StridedTensorAddr`` contract the kernel assumes."""
    it = backing.itemsize
    base = (view.ctypes.data - backing.ctypes.data) // it
    s_b, s_h, s_t, s_d = (s // it for s in view.strides)
    if s_d != 1:
        raise ValueError(f"head-dim stride must be 1, got {s_d}")
    return base, (s_b, s_h, s_t)


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
    head_pad=0,
    packed=False,
    verbose=True,
):
    """Build fmha_mfma(free_stride=True), launch it, and compare against the
    numpy reference. Returns a :class:`VerifyResult`; raises ``ValueError`` on
    an invalid spec or placement. Shared by the CLI and the suite's GPU-gated
    numeric test.

    ``layout`` places each tensor (and O in every case). ``head_pad`` widens the
    head dim of the backing buffers. ``packed`` instead puts Q/K/V in one
    [B, S, 3, H, D] buffer at offsets 0, H*D and 2*H*D; O stays separate and
    follows ``layout``."""
    Hq, D, Hk = heads, head_size, (kv_heads or heads)
    Sq, Sk, B = seqlen_q, seqlen_k, batch
    if packed:
        # One H in the packed buffer, so the operands must share a head count.
        if Hk != Hq:
            raise ValueError(f"packed QKV needs Hq == Hk (got {Hq} and {Hk})")
        if Sk != Sq:
            raise ValueError(f"packed QKV needs Sq == Sk (got {Sq} and {Sk})")
        if head_pad:
            raise ValueError("packed QKV and head_pad are separate placements")

    common = FmhaCommonSpec(
        FmhaShape(head_size=D, num_query_heads=Hq, num_kv_heads=Hk),
        dtype=dtype,
        mask_mode="causal" if causal else "none",
    )
    spec = FmhaMfmaSpec(common=common, seqlen_q=Sq, seqlen_k=Sk, free_stride=True)
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid fmha_mfma free-stride spec: {why}")

    art = compile_kernel(build_fmha_fwd_mfma(spec, arch=arch), arch=arch)
    if verbose:
        print(f"[{arch}] built {art.kernel_name} ({art.hsaco_bytes} B)")

    np_dt = _NP_DTYPE[dtype]
    rng = np.random.default_rng(0xA11E)
    ql = (rng.standard_normal((B, Hq, Sq, D)) * 0.3).astype(np_dt)
    kl = (rng.standard_normal((B, Hk, Sk, D)) * 0.3).astype(np_dt)
    vl = (rng.standard_normal((B, Hk, Sk, D)) * 0.3).astype(np_dt)

    # Physical placement. ``backings`` are the distinct allocations; each tensor
    # names the one it lives in plus its own base offset, so packed QKV is just
    # three tensors sharing a backing at three offsets.
    if packed:
        qkv, (q_view, k_view, v_view) = _place_packed(ql, kl, vl)
        backings = {"qkv": qkv}
        owner = {"q": "qkv", "k": "qkv", "v": "qkv"}
    else:
        q_back, q_view = _place(ql, layout, head_pad=head_pad)
        k_back, k_view = _place(kl, layout, head_pad=head_pad)
        v_back, v_view = _place(vl, layout, head_pad=head_pad)
        backings = {"q": q_back, "k": k_back, "v": v_back}
        owner = {"q": "q", "k": "k", "v": "v"}
    o_back, o_view = _place(np.zeros_like(ql), layout, head_pad=head_pad)
    backings["o"] = o_back
    owner["o"] = "o"

    views = {"q": q_view, "k": k_view, "v": v_view, "o": o_view}
    base, stride = {}, {}
    for name, view in views.items():
        base[name], stride[name] = _base_and_strides(view, backings[owner[name]])

    # On the plain contiguous path the strides numpy reports must equal what the
    # shipped launcher helper derives analytically. This is the one place the
    # helper is checked against a stride set a real kernel then runs on.
    if not packed and not head_pad:
        for name, h, s in (("q", Hq, Sq), ("k", Hk, Sk), ("v", Hk, Sk), ("o", Hq, Sq)):
            want = contiguous_strides(layout, heads=h, seqlen=s, head_size=D)
            if stride[name] != want:
                raise ValueError(
                    f"{name}: contiguous_strides gave {want}, numpy {stride[name]}"
                )
    scale_log2 = float(1.0 / math.sqrt(D) * math.log2(math.e))

    rt = Runtime()
    module = rt.load_module(art.hsaco)
    fn = module.get_function(art.kernel_name)

    def u8(a):
        return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(np.ascontiguousarray(a))

    dev = {name: rt.alloc(buf.nbytes) for name, buf in backings.items()}
    for name, buf in backings.items():
        if name == "o":
            rt.memset(dev[name], 0, buf.nbytes)
        else:
            rt.memcpy_h2d(dev[name], u8(buf), buf.nbytes)

    itemsize = o_back.itemsize
    ptr = {n: dev[owner[n]] + base[n] * itemsize for n in ("q", "k", "v", "o")}

    # fmha_mfma kernarg order: Q,K,V,O, scale_log2, seqlen_q, seqlen_k, the
    # eight token/head strides, then the four free-stride batch strides.
    kernargs = struct.pack(
        "<QQQQf" + "i" * 14,
        ptr["q"],
        ptr["k"],
        ptr["v"],
        ptr["o"],
        scale_log2,
        Sq,
        Sk,
        *(x for n in ("q", "k", "v", "o") for x in (stride[n][2], stride[n][1])),
        *(stride[n][0] for n in ("q", "k", "v", "o")),
    )
    rt.launch(
        fn,
        fmha_fwd_mfma_grid(spec, batch=B),
        (ArchTarget.from_gfx(arch).wave_size, 1, 1),
        kernargs,
    )
    rt.sync()
    rt.memcpy_d2h(u8(o_back), dev["o"], o_back.nbytes)
    for p in dev.values():
        rt.free(p)
    module.unload()

    got = o_view.transpose(0, 2, 1, 3)  # (B, S, Hq, D), the reference's order
    ref = _reference(ql, kl, vl, causal=causal)
    diff = np.abs(got.astype(np.float32) - ref.astype(np.float32))
    max_abs = float(diff.max())
    bad = int(np.count_nonzero(diff > tol))
    if verbose:
        tag = "PASS" if max_abs <= tol else "FAIL"
        where = "packed" if packed else f"layout={layout} head_pad={head_pad}"
        print(
            f"[{arch}] {where} Sq={Sq} Sk={Sk} D={D} Hq={Hq} Hk={Hk} "
            f"causal={causal}: max_abs_diff={max_abs:.3e} bad={bad}/{got.size} "
            f"tol={tol:.0e} -> {tag}"
        )
    return VerifyResult(max_abs <= tol, max_abs, bad, got.size, np.array(got))


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
    # f16 only: is_valid_spec rejects bf16 until the bf16 atom factory lands.
    p.add_argument("--dtype", choices=("f16",), default="f16")
    p.add_argument("--tol", type=float, default=2e-2)
    p.add_argument(
        "--head-pad", type=int, default=0, help="pad the head dim in the buffer"
    )
    p.add_argument(
        "--packed", action="store_true", help="Q/K/V share one [B,S,3,H,D] buffer"
    )
    args = p.parse_args()

    try:
        result = run_verify(
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
            head_pad=args.head_pad,
            packed=args.packed,
            verbose=True,
        )
    except ValueError as e:
        print(f"[{args.arch}] {e}")
        return 2
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
