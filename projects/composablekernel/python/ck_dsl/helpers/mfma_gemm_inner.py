# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Universal MFMA-tiled K-loop helper for GEMM-shaped kernels.

This module factors the inner MFMA loop that every production GEMM
needs into one helper. Concrete kernels (StreamK GEMM, block-scale
GEMM, MX GEMM, plain MFMA GEMM, fused-MoE per-expert GEMM,
implicit-GEMM convolution) supply just the **per-lane load callbacks**
plus an optional **per-tile post-MFMA callback** (for per-group scale
apply / mask / bias); the helper emits the ``scf.for`` K-loop, threads
the f32 accumulator through ``iter_args``, and returns the final
per-lane accumulator vector.

What it abstracts:

* The K-loop trip count math (``K / atom.k`` atoms).
* The per-lane (m_row, n_col, k_base) decomposition from the atom's
  canonical lane layout (lane t = ``k_blk * atom.m + m_in_atom``).
* The f32 accumulator initialization and threading via
  ``scf.for_iter`` -- no LDS reload between K-tiles.
* The atom dispatch -- callers pick the right :class:`MfmaAtom`
  (``f16_16x16x16``, ``bf16_16x16x16``, ``fp8_16x16x32``, etc.) and
  the helper invokes its ``.emit()`` per iter.

What it leaves to callers:

* The actual A / B loads (callbacks). This lets a kernel choose
  between global → register direct loads (fine for small K),
  LDS-staged double-buffered loads (production), preshuffle-B
  layout, paged-KV indirection, codebook-dequant, MX-scaled, etc.
* The output store (cshuffle epilogue / atomic add into f32
  workspace / direct store). Callers receive the final per-lane
  accumulator and write it however they want.

Together this gives every GEMM-shaped kernel a one-line lift from
scalar inner (1 MAC / cycle / lane) to MFMA (256 MACs / cycle / lane).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ..core.ir import F16, F32, BF16, IRBuilder, Value
from .atoms import MfmaAtom


__all__ = [
    "LaneDecode",
    "decode_mfma_lanes",
    "load_a_row_major_contiguous",
    "load_b_col_strided_scalars",
    "mfma_k_loop",
    "mfma_atom_for_dtype",
]


@dataclass(frozen=True)
class LaneDecode:
    """Per-lane MFMA operand coordinates.

    For wave64 atoms with ``m / n`` dim in ``{16, 32}``:

    * ``m_in_atom`` -- the M-row this lane carries (``lane % atom.m``).
    * ``n_in_atom`` -- the N-col this lane carries (``lane % atom.n``).
      Equal to ``m_in_atom`` for the square atoms we expose.
    * ``k_blk`` -- which ``a_per_lane``-sized chunk of K this lane
      carries (``lane / atom.m``). Total K covered = ``atom.k`` per
      MFMA invocation = ``a_per_lane * 64 / atom.m`` (== atom.k).

    Used by the standard row-major-A / col-strided-B load callbacks.
    Custom callers (LDS-staged, preshuffle-B, paged-KV) get the same
    decode and apply their own address math.
    """

    lane: Value
    m_in_atom: Value
    n_in_atom: Value
    k_blk: Value


def decode_mfma_lanes(b: IRBuilder, atom: MfmaAtom, lane: Value) -> LaneDecode:
    """Decompose a wave64 lane id into (m_in_atom, n_in_atom, k_blk).

    Assumes the canonical "square" MFMA atoms (16x16x* / 32x32x*).
    Other atoms (4x4x4 small-batch) need a different decode -- they
    use a ``batch_idx = lane / 4`` instead of a ``k_blk``.
    """
    c_m = b.const_i32(atom.m)
    c_n = b.const_i32(atom.n)
    m_in_atom = b.mod(lane, c_m)
    n_in_atom = b.mod(lane, c_n)
    k_blk = b.div(lane, c_m)
    return LaneDecode(
        lane=lane,
        m_in_atom=m_in_atom,
        n_in_atom=n_in_atom,
        k_blk=k_blk,
    )


def mfma_atom_for_dtype(
    dtype_in: str,
    m: int = 16,
    n: int = 16,
    *,
    prefer_packed_k: bool = True,
) -> MfmaAtom:
    """Pick the right :class:`MfmaAtom` for an in-dtype and tile shape.

    ``prefer_packed_k`` (default) picks the K-packed atom (atom.k=32
    for f16/bf16/fp8 on gfx950+) which is denser. Pass False to fall
    back to the legacy atoms (16x16x16 f16, 32x32x8 f16) when targeting
    pre-CDNA3 cards.
    """
    if dtype_in in ("f16", "fp16"):
        if (m, n) == (16, 16):
            return (
                MfmaAtom.f16_16x16x32() if prefer_packed_k else MfmaAtom.f16_16x16x16()
            )
        if (m, n) == (32, 32):
            return (
                MfmaAtom.f16_32x32x16() if prefer_packed_k else MfmaAtom.f16_32x32x8()
            )
    if dtype_in == "bf16":
        # bf16 atoms aren't in the MfmaAtom factory yet -- v2 follow-on
        # adds them once the bf16 MFMA intrinsics land in IRBuilder.
        raise NotImplementedError(
            "bf16 MFMA atom factories not exposed yet; use the f16 atoms "
            "by converting bf16 operands first"
        )
    if dtype_in == "fp8e4m3":
        if (m, n) == (16, 16):
            return MfmaAtom.fp8_16x16x32()
        if (m, n) == (32, 32):
            return MfmaAtom.fp8_32x32x16()
    if dtype_in == "bf8e5m2":
        if (m, n) == (16, 16):
            return MfmaAtom.bf8_16x16x32()
        if (m, n) == (32, 32):
            return MfmaAtom.bf8_32x32x16()
    raise ValueError(f"no MFMA atom for dtype_in={dtype_in!r} shape {m}x{n}")


def _ir_type_for_dtype(dtype_in: str):
    """Map a dtype string to its IR :class:`Type`."""
    from ..core.ir import BF8E5M2, FP8E4M3

    if dtype_in in ("f16", "fp16"):
        return F16
    if dtype_in == "bf16":
        return BF16
    if dtype_in == "fp8e4m3":
        return FP8E4M3
    if dtype_in == "bf8e5m2":
        return BF8E5M2
    raise ValueError(f"unsupported dtype {dtype_in!r}")


def load_a_row_major_contiguous(
    b: IRBuilder,
    *,
    A: Value,
    atom: MfmaAtom,
    lane_decode: LaneDecode,
    m_tile_base: Value,
    k_tile_base: Value,
    K: int,
) -> Value:
    """Per-lane A load for **row-major (M, K)** layout.

    The K axis is **contiguous** so the per-lane ``a_per_lane`` values
    are at consecutive memory addresses -- one vec_<a_per_lane x dtype>
    load fills the lane's A operand.

    * ``m_tile_base`` -- the M-tile base (e.g. ``bid_m * atom.m``).
    * ``k_tile_base`` -- the K-tile base (e.g. ``k_tile_idx *
      atom.k``).
    * ``K`` -- the matrix's K extent in elements (for stride math).

    Returns a per-lane ``<a_per_lane x dtype>`` vector ready for
    :meth:`MfmaAtom.emit`.
    """
    dtype_ir = _ir_type_for_dtype(atom.dtype_in)
    m_row = b.add(m_tile_base, lane_decode.m_in_atom)
    k_lane_start = b.mul(lane_decode.k_blk, b.const_i32(atom.a_per_lane))
    k_base = b.add(k_tile_base, k_lane_start)
    a_addr = b.add(b.mul(m_row, b.const_i32(K)), k_base)
    if atom.dtype_in in ("f16", "fp16", "bf16"):
        return b.global_load_vN(
            A,
            a_addr,
            dtype_ir,
            atom.a_per_lane,
            align=atom.a_per_lane * 2,
        )
    # fp8 / bf8: no vec load helper -- fall back to scalar loads.
    out = b.zero_vec(dtype_ir, atom.a_per_lane)
    for j in range(atom.a_per_lane):
        addr = b.add(a_addr, b.const_i32(j))
        s = b.global_load(A, addr, dtype_ir, align=1)
        out = b.vec_insert(out, s, j)
    return out


def load_b_col_strided_scalars(
    b: IRBuilder,
    *,
    B: Value,
    atom: MfmaAtom,
    lane_decode: LaneDecode,
    n_tile_base: Value,
    k_tile_base: Value,
    N: int,
) -> Value:
    """Per-lane B load for **row-major (K, N)** layout.

    Each K-element of B is offset by N from the next, so the
    ``b_per_lane`` values are NOT contiguous in memory; we issue
    ``b_per_lane`` scalar loads and pack them into a vector. CK Tile's
    production GEMM avoids this by staging B in LDS first
    (``ds_read_b64_tr_b16`` via :class:`TransposeLdsReader`); the
    v1 path uses scalar loads and relies on L2 caching for adjacent
    lanes' loads of the same column. The vector-vs-scalar choice is
    a performance knob, not a correctness one.
    """
    dtype_ir = _ir_type_for_dtype(atom.dtype_in)
    n_col = b.add(n_tile_base, lane_decode.n_in_atom)
    k_lane_start = b.mul(lane_decode.k_blk, b.const_i32(atom.b_per_lane))
    k_base = b.add(k_tile_base, k_lane_start)
    out = b.zero_vec(dtype_ir, atom.b_per_lane)
    for j in range(atom.b_per_lane):
        addr = b.add(
            b.mul(b.add(k_base, b.const_i32(j)), b.const_i32(N)),
            n_col,
        )
        s = b.global_load(
            B, addr, dtype_ir, align=2 if atom.dtype_in in ("f16", "bf16") else 1
        )
        out = b.vec_insert(out, s, j)
    return out


def mfma_k_loop(
    b: IRBuilder,
    *,
    K: int,
    atom: MfmaAtom,
    load_a: Callable[[IRBuilder, Value], Value],
    load_b: Callable[[IRBuilder, Value], Value],
    per_tile_post_mfma: Optional[Callable[[IRBuilder, Value, Value], Value]] = None,
    initial_acc: Optional[Value] = None,
    iv_name: str = "kt",
    acc_name: str = "acc",
) -> Value:
    """Emit a ``scf.for`` K-loop of MFMA atoms; return the final f32 acc.

    Per iteration ``kt`` ∈ ``[0, K / atom.k)``:

    1. ``a = load_a(b, kt)`` -- per-lane A operand vector.
    2. ``b_op = load_b(b, kt)`` -- per-lane B operand vector.
    3. ``acc = atom.emit(a, b_op, acc)``.
    4. (Optional) ``acc = per_tile_post_mfma(b, acc, kt)`` -- used for
       per-group scale application (block-scale GEMM, MX GEMM) or
       per-tile bias.

    ``initial_acc`` defaults to ``atom.zero_acc(b)``. Pass a non-zero
    initial value to chain multiple K-loops (e.g. across paged blocks)
    without leaving the f32 register accumulator.

    Returns a per-lane ``<c_per_lane x f32>`` vector.
    """
    if K % atom.k != 0:
        raise ValueError(f"mfma_k_loop: K={K} must be divisible by atom.k={atom.k}")
    n_tiles = K // atom.k
    acc0 = initial_acc if initial_acc is not None else atom.zero_acc(b)
    kloop = b.scf_for_iter(
        b.const_i32(0),
        b.const_i32(n_tiles),
        b.const_i32(1),
        [(acc_name, acc0)],
        iv_name=iv_name,
    )
    with kloop as (kt, (acc_v,)):
        a_vec = load_a(b, kt)
        b_vec = load_b(b, kt)
        new_acc = atom.emit(b, a_vec, b_vec, acc_v)
        if per_tile_post_mfma is not None:
            new_acc = per_tile_post_mfma(b, new_acc, kt)
        b.scf_yield(new_acc)
    return kloop.results[0]


def store_acc_to_global(
    b: IRBuilder,
    *,
    C: Value,
    atom: MfmaAtom,
    lane_decode: LaneDecode,
    m_tile_base: Value,
    n_tile_base: Value,
    acc: Value,
    N: int,
    out_dtype: str = "f16",
    atomic_add: bool = False,
) -> None:
    """Write a per-lane MFMA accumulator to global ``C`` in row-major.

    Default ``out_dtype="f16"`` does the canonical f32 → f16 cshuffle
    via :meth:`cast_f32_to`. Pass ``out_dtype="f32"`` for kernels that
    keep their accumulator in f32 (StreamK workspace, block-scale).
    Pass ``atomic_add=True`` to perform ``atomic_add(C[i], val)``
    instead of a plain store (StreamK split-K).

    The atom's :meth:`lane_to_output` returns the per-lane
    (row_in_atom, col_in_atom) for accumulator slot ``i``; combined
    with the tile base offsets, each lane writes ``c_per_lane`` output
    cells.
    """
    out_dtype_ir = F32 if out_dtype == "f32" else _ir_type_for_dtype(out_dtype)
    for i in range(atom.c_per_lane):
        row_in, col_in = atom.lane_to_output(b, lane_decode.lane, i)
        row = b.add(m_tile_base, row_in)
        col = b.add(n_tile_base, col_in)
        addr = b.add(b.mul(row, b.const_i32(N)), col)
        c_f32 = b.vec_extract(acc, i)
        if out_dtype == "f32":
            val = c_f32
        else:
            val = b.cast_f32_to(c_f32, out_dtype_ir)
        if atomic_add:
            if out_dtype != "f32":
                raise ValueError("atomic_add output requires out_dtype='f32'")
            b.global_atomic_add(C, addr, val)
        else:
            b.global_store(
                C,
                addr,
                val,
                align=4 if out_dtype == "f32" else 2,
            )
