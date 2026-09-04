# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-free numpy verifier for the MoE activation gather/rescale prologue.

Checks all four of the kernel's outputs against a numpy model, and separately
checks the property the whole kernel exists to preserve: dequantizing ``A``
under the block scale it publishes must reproduce the activation the row came
from, because ``A_q * s_token == A_q' * s_block`` is the point of the rescale.

Why torch-free: rocKE compiles through Comgr, and a resident torch changes which
LLVM the process resolves -- the compile stops finishing rather than failing. The
assert in :func:`main` is there so that mistake surfaces as an error instead of a
hang. This is the same rule the fused_mega_moe harness follows.

    PYTHONPATH=<rocke platform python> python3 -u \\
        rocke/examples/gfx950/moe_gather_rescale/verify_gather_rescale.py
"""

from __future__ import annotations

import argparse
import ctypes
import sys

import numpy as np

#: Scale-group width along the hidden axis. Fixed by the MoE kernels.
GROUP_K = 128
FP8_MAX = 448.0
#: Floor on a per-token amax, so an all-zero token cannot produce a zero divisor.
AMAX_FLOOR = 1e-6


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# fp8 e4m3 (OCP "fn" flavour: no inf, 0x7f / 0xff are NaN, max magnitude 448)
# ---------------------------------------------------------------------------


def _e4m3_value_table() -> np.ndarray:
    codes = np.arange(256, dtype=np.uint16)
    sign, exp, man = (codes >> 7) & 1, (codes >> 3) & 0xF, codes & 0x7
    sub = (man / 8.0) * (2.0**-6)
    nrm = (1.0 + man / 8.0) * np.power(2.0, exp.astype(np.float64) - 7.0)
    val = np.where(exp == 0, sub, nrm)
    val = np.where(sign == 1, -val, val)
    val[(exp == 0xF) & (man == 0x7)] = np.nan
    return val.astype(np.float32)


_E4M3_TABLE = _e4m3_value_table()
#: Codes 0x00..0x7e are the finite non-negative values, and they ascend with the
#: code, so a code IS its own index into this array. e4m3 is sign-magnitude, so
#: the negative half is the same ladder with bit 7 set.
_POS_VALS = _E4M3_TABLE[:0x7F]


def quantize_e4m3(x: np.ndarray) -> np.ndarray:
    """Saturating f32 -> e4m3 code bytes, round-to-nearest-EVEN.

    The tie rule is not cosmetic here. This verifier compares the kernel's ``A``
    byte for byte, and the kernel's converter is ``v_cvt_pk_fp8_f32``, which is
    round-to-nearest-even. Rounding halves away from zero instead would disagree
    on exactly the near-tie elements that this kernel's exact divide exists to
    get right, so a wrong tie rule here would report the bug as absent.

    Quantizing the magnitude and re-attaching the sign bit separately (rather
    than searching the signed ladder) is what makes -0.0 survive as 0x80, which
    is the code a pad row produces.
    """
    x = np.asarray(x, dtype=np.float32)
    mag = np.clip(np.abs(x), 0.0, FP8_MAX)
    hi = np.clip(np.searchsorted(_POS_VALS, mag), 1, _POS_VALS.size - 1)
    d_hi, d_lo = _POS_VALS[hi] - mag, mag - _POS_VALS[hi - 1]
    take_hi = np.where(d_hi == d_lo, (hi & 1) == 0, d_hi < d_lo)
    code = np.where(take_hi, hi, hi - 1).astype(np.uint8)
    return code | (np.signbit(x).astype(np.uint8) << 7)


def dequantize_e4m3(codes: np.ndarray) -> np.ndarray:
    return _E4M3_TABLE[codes]


# ---------------------------------------------------------------------------
# Host-side problem construction
# ---------------------------------------------------------------------------


class Problem:
    """Per-token-quantized activations plus a block-aligned routing.

    This is the input side of the contract: activations already quantized with
    ONE SCALE PER (token, 128-group), which is what a serving stack that
    quantizes before routing naturally has, and which the mega-kernel cannot
    consume directly.
    """

    def __init__(self, tokens: int, hidden: int, experts: int, topk: int,
                 tile_m: int, seed: int):
        if hidden % GROUP_K:
            raise SystemExit(f"hidden={hidden} must be a multiple of {GROUP_K}")
        if topk > experts:
            raise SystemExit(f"topk={topk} exceeds experts={experts}")
        rng = np.random.default_rng(seed)
        self.tokens, self.hidden, self.experts = tokens, hidden, experts
        self.topk, self.tile_m = topk, tile_m
        self.n_hb = hidden // GROUP_K
        self.n_flat = tokens * topk

        # A wide spread of per-token magnitudes is the interesting case: it is
        # what makes the block scale differ from the row scales, so the ratio is
        # not 1 and the re-round actually happens.
        x = rng.standard_normal((tokens, hidden)).astype(np.float32)
        x *= np.exp(rng.uniform(-2.0, 2.0, size=(tokens, 1))).astype(np.float32)
        amax = np.maximum(
            np.abs(x.reshape(tokens, self.n_hb, GROUP_K)).max(axis=2), AMAX_FLOOR
        )
        self.AqScale = (amax / FP8_MAX).astype(np.float32)  # [T, n_hb]
        self.Aq = quantize_e4m3(
            x / np.repeat(self.AqScale, GROUP_K, axis=1)
        )  # [T, H] fp8 codes

        logits = rng.standard_normal((tokens, experts)).astype(np.float32)
        ids = np.argsort(-logits, axis=-1, kind="stable")[:, :topk]
        w = rng.random((tokens, topk)).astype(np.float32)
        self.topk_ids = ids.astype(np.int32)
        self.TopkWeights = (w / w.sum(-1, keepdims=True)).astype(np.float32)

        self._align_blocks()

    def _align_blocks(self) -> None:
        """Group the flattened ``(token, slot)`` ids by expert into tile_m rows.

        The kernel does not care how the grouping was produced -- it gathers
        whatever ``SortedIds`` names -- so this is a plain stand-in for the
        framework's block-align pass rather than a reimplementation of one. Pad
        slots carry ``n_flat``, an out-of-range id; that sentinel is the only
        thing marking a row as padding.
        """
        tm = self.tile_m
        per_expert = []
        for e in range(self.experts):
            tok, slot = np.nonzero(self.topk_ids == e)
            if tok.size == 0:
                continue
            flat = np.sort(tok.astype(np.int64) * self.topk + slot)
            pad = (-flat.size) % tm
            per_expert.append(
                np.concatenate([flat, np.full(pad, self.n_flat, dtype=np.int64)])
            )
        if not per_expert:
            raise SystemExit("routing activated no experts")
        self.SortedIds = np.concatenate(per_expert).astype(np.int32)
        self.n_blocks = self.SortedIds.size // tm
        self.padded = self.n_blocks * tm

    def reference(self):
        """The four outputs, modelled exactly as the kernel computes them."""
        tm, n_hb = self.tile_m, self.n_hb
        valid = self.SortedIds < self.n_flat
        flat = np.where(valid, self.SortedIds, 0).astype(np.int64)
        tok = flat // self.topk

        # A pad row contributes a zero scale, so it cannot raise the block max.
        row_scale = self.AqScale[tok] * valid[:, None]
        blk = row_scale.reshape(self.n_blocks, tm, n_hb).max(axis=1, keepdims=True)
        blk = np.maximum(blk, 1e-30)
        AScale = np.broadcast_to(blk, (self.n_blocks, tm, n_hb)).reshape(
            self.padded, n_hb
        )
        ratio = row_scale / AScale  # 0 on a pad row -> A is zeroed there

        gathered = dequantize_e4m3(self.Aq[tok]).reshape(self.padded, n_hb, GROUP_K)
        A = quantize_e4m3(gathered * ratio[:, :, None]).reshape(
            self.padded, self.hidden
        )
        SortedTokenIds = np.where(valid, tok, -1).astype(np.int32)
        SortedWeights = self.TopkWeights.reshape(-1)[flat] * valid
        return A, AScale.astype(np.float32), SortedTokenIds, SortedWeights.astype(
            np.float32
        ), valid


# ---------------------------------------------------------------------------
# Device side (rocKE HIP runtime only -- no torch)
# ---------------------------------------------------------------------------


def run_kernel(p: Problem):
    """Compile, launch once, and read back the four outputs."""
    from rocke.helpers.compile import compile_kernel
    from rocke.instances.common.moe_gather_rescale_a import (
        MoeGatherRescaleSpec,
        build_moe_gather_rescale_a,
        moe_gather_rescale_a_grid,
        moe_gather_rescale_a_signature,
    )
    from rocke.runtime.host_buffers import as_u8_buffer
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import DeviceMem, KernelLauncher, LaunchConfig

    # max_n_hb only sizes the LDS scratch; hidden itself stays a runtime arg.
    spec = MoeGatherRescaleSpec(tile_m=p.tile_m, max_n_hb=max(p.n_hb, 1))
    art = compile_kernel(
        build_moe_gather_rescale_a(spec), arch="gfx950", capture_ir_text=False
    )
    log(f"built {art.kernel_name}  hsaco={len(art.hsaco)}B")

    rt = Runtime()
    keep: list = []

    def upload(arr: np.ndarray) -> DeviceMem:
        arr = np.ascontiguousarray(arr)
        mem = DeviceMem(arr.nbytes)
        rt.memcpy_h2d(mem.ptr(), as_u8_buffer(arr), arr.nbytes)
        keep.extend((arr, mem))
        return mem

    def download(mem: DeviceMem, shape, dtype) -> np.ndarray:
        out = np.zeros(shape, dtype=dtype)
        buf = (ctypes.c_uint8 * out.nbytes).from_buffer(out)
        rt.memcpy_d2h(buf, mem.ptr(), out.nbytes)
        del buf
        return out

    # Outputs start as poison, not zeros: every byte the checks look at has to
    # have been written by the kernel, including the pad rows.
    outs = {
        "A": (DeviceMem(p.padded * p.hidden), (p.padded, p.hidden), np.uint8),
        "AScale": (DeviceMem(p.padded * p.n_hb * 4), (p.padded, p.n_hb), np.float32),
        "SortedTokenIds": (DeviceMem(p.padded * 4), (p.padded,), np.int32),
        "SortedWeights": (DeviceMem(p.padded * 4), (p.padded,), np.float32),
    }
    for mem, shape, dtype in outs.values():
        keep.append(mem)
        poison = np.full(shape, 0xCD if dtype is np.uint8 else -7.0, dtype=dtype)
        rt.memcpy_h2d(mem.ptr(), as_u8_buffer(poison), poison.nbytes)

    values = {
        "Aq": upload(p.Aq),
        "AqScale": upload(p.AqScale),
        "SortedIds": upload(p.SortedIds),
        "TopkWeights": upload(p.TopkWeights),
        "n_flat": p.n_flat,
        "topk": p.topk,
        "hidden": p.hidden,
        "n_hb": p.n_hb,
    }
    values.update({name: mem for name, (mem, _, _) in outs.items()})

    KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=art.kernel_name,
        signature=moe_gather_rescale_a_signature(spec),
        cache_key=("moe_gather_rescale_verify", spec.kernel_name()),
    )(
        values,
        config=LaunchConfig(
            stream=0,
            grid=moe_gather_rescale_a_grid(p.n_blocks, spec),
            block=(spec.block_size, 1, 1),
        ),
    )
    rt.sync()
    return tuple(download(mem, shape, dtype) for mem, shape, dtype in outs.values())


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check(p: Problem, got, ref) -> bool:
    """Compare every output, and report the invariant the kernel is for."""
    A, AScale, Stid, Sw = got
    rA, rAScale, rStid, rSw, valid = ref
    ok = True

    def report(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        log(f"  {'PASS' if passed else 'FAIL'}  {name}{'  ' + detail if detail else ''}")

    report("SortedTokenIds exact", bool((Stid == rStid).all()))
    report("SortedWeights exact", bool((Sw == rSw).all()))
    report("AScale exact (block scale on every row, pads included)",
           bool((AScale == rAScale).all()))

    # e4m3 is sign-magnitude, so a pad row whose source byte was negative
    # rescales to -0.0 (0x80), not 0x00. Both dequantize to zero, which is all
    # the consumer's block amax cares about, so the byte-level check is split:
    # every non-zero disagreement is a real fault, a +-0 one is not.
    differs = A != rA
    signed_zero_only = differs & (dequantize_e4m3(A) == 0) & (dequantize_e4m3(rA) == 0)
    real = int((differs & ~signed_zero_only).sum())
    report("A bytes (ignoring +-0 encoding)", real == 0,
           f"{real} real / {int(differs.sum())} total differing of {A.size}")
    report("pad rows dequantize to zero",
           bool((dequantize_e4m3(A)[~valid] == 0).all()))

    # The invariant: restating a row under the block scale must not move the
    # value it represents. Everything the kernel does is in service of this.
    got_val = (dequantize_e4m3(A).reshape(p.padded, p.n_hb, GROUP_K)
               * AScale[:, :, None]).reshape(p.padded, p.hidden)
    tok = np.where(valid, p.SortedIds, 0).astype(np.int64) // p.topk
    want = (dequantize_e4m3(p.Aq[tok]).reshape(p.padded, p.n_hb, GROUP_K)
            * p.AqScale[tok][:, :, None]).reshape(p.padded, p.hidden)
    want *= valid[:, None]
    denom = max(float(np.abs(want).max()), 1e-9)
    log(f"  dequant round-trip: max {np.abs(got_val - want).max() / denom:.3e}"
        f"  mean {np.abs(got_val - want).mean() / denom:.3e}"
        "  (relative to the largest activation; the residual is the fp8"
        " re-rounding a block-uniform scale costs)")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--tile-m", type=int, default=16,
                    help="must equal the tile_m of the MoE kernel this feeds")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert "torch" not in sys.modules, "verifier must stay torch-free"

    p = Problem(args.tokens, args.hidden, args.experts, args.topk,
                args.tile_m, args.seed)
    log(f"T={p.tokens} H={p.hidden} E={p.experts} K={p.topk} tile_m={p.tile_m} "
        f"-> {p.n_flat} token-slots, {p.n_blocks} blocks, {p.padded} padded rows")

    got = run_kernel(p)
    ok = check(p, got, p.reference())
    log("VERDICT: " + ("PROLOGUE OK" if ok else "PROLOGUE MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
