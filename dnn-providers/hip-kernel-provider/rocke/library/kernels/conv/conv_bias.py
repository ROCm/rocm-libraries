# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Per-output-channel bias for the convolution forward kernels.

A conv + bias forward computes, for every output element,

    y[n, (d,) h, w, g*kpg + k] = round(acc_fp32 + float(bias[g*kpg + k]))

with one rounding: the bias is added to the fp32 accumulator before any
scalar accumulator epilogue and before the conversion to the output dtype.

The bias vector has K (= groups * kpg) entries in the output dtype. A kernel
built with a bias takes it as a trailing ``Bias`` / ``Bias_bytes`` kernarg
pair (:func:`kernels.common.conv_abi.conv_bias_arg_names`) and reads it
through a buffer resource sized by ``Bias_bytes``, so a channel past the end
reads 0. Nothing here is emitted for a kernel without bias.

The pieces are generic, so every conv family uses the same ones:

* :func:`declare_conv_bias_params` -- the two kernargs, after every other one;
* :func:`conv_bias_rsrc` -- the buffer resource over them;
* :func:`load_conv_bias_f32` -- ``bias[k]`` as an fp32 value;
* :func:`load_conv_bias_vec_f32` -- ``bias[k .. k+n)`` as an ``<n x f32>``
  vector (the direct kernels, which store runs of channels);
* :func:`load_conv_bias_run_f32` -- the same run read one element at a time,
  for a run whose start is not aligned to the vector load;
* :func:`add_conv_bias_to_accs` -- add one value per fragment column to MMA
  accumulator fragments;
* :func:`apply_implicit_gemm_conv_bias` -- the implicit-GEMM forward's use of
  the above on its accumulator fragments.

The C++ engine mirrors them one for one (``instance_conv_bias.h``,
``rocke_conv_bias_*``), so the two emit the same IR.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from rocke.core.ir import BF16, F16, F32, I32, IRBuilder, PtrType, Type, Value

__all__ = [
    "add_conv_bias_to_accs",
    "apply_implicit_gemm_conv_bias",
    "conv_bias_rsrc",
    "declare_conv_bias_params",
    "load_conv_bias_f32",
    "load_conv_bias_run_f32",
    "load_conv_bias_vec_f32",
]


def declare_conv_bias_params(b: IRBuilder, elem: Type) -> Tuple[Value, Value]:
    """Declare the ``Bias`` pointer (``elem`` = the output dtype) and its
    ``Bias_bytes`` size. Call after every other kernarg is declared."""
    bias = b.param(
        "Bias", PtrType(elem, "global"), noalias=True, readonly=True, align=16
    )
    bias_bytes = b.param("Bias_bytes", I32)
    return bias, bias_bytes


def conv_bias_rsrc(b: IRBuilder, bias: Value, bias_bytes: Value) -> Value:
    """The buffer resource the bias is read through (out of range reads 0)."""
    return b.buffer_rsrc(bias, bias_bytes)


def _bias_elem_bytes(elem: Type) -> int:
    """Bytes of one bias element: the output dtypes a conv forward stores."""
    if elem == F32:
        return 4
    if elem in (F16, BF16):
        return 2
    raise ValueError(f"conv bias: no {elem} bias (f16, bf16 or f32 only)")


def load_conv_bias_f32(
    b: IRBuilder, rsrc: Value, k: Value, elem: Type, c0: Value
) -> Value:
    """``bias[k]`` (element index ``k``) widened to fp32. ``c0`` is the
    soffset (an i32 zero)."""
    off = b.mul(k, b.const_i32(_bias_elem_bytes(elem)))
    return b.cast_to_f32(b.buffer_load(rsrc, off, c0, elem))


def add_conv_bias_to_accs(
    b: IRBuilder,
    accs: Sequence[Value],
    bias_per_ni: Sequence[Value],
    *,
    mfmas_m: int,
    mfmas_n: int,
) -> List[Value]:
    """Add ``bias_per_ni[ni]`` (an fp32 scalar) to every slot of the
    accumulator fragments ``accs[mi * mfmas_n + ni]``.

    Valid when a lane's fragment slots all lie in one output column, which
    holds for every MFMA / WMMA fp32 accumulator layout the conv kernels use
    (the column of a slot is a function of the lane only).
    """
    if len(accs) != mfmas_m * mfmas_n or len(bias_per_ni) != mfmas_n:
        raise ValueError(
            f"add_conv_bias_to_accs: {len(accs)} accs / {len(bias_per_ni)} bias "
            f"values for a {mfmas_m}x{mfmas_n} fragment grid"
        )
    splat = [
        b.vec_pack([v] * accs[ni].type.count, F32) for ni, v in enumerate(bias_per_ni)
    ]
    return [b.fadd(acc, splat[i % mfmas_n]) for i, acc in enumerate(accs)]


def apply_implicit_gemm_conv_bias(
    b: IRBuilder,
    accs: Sequence[Value],
    *,
    op,
    elem: Type,
    bias: Value,
    bias_bytes: Value,
    lane: Value,
    warp_n_idx: Value,
    block_n_off: Value,
    k_out_group_base: Optional[Value],
    mfmas_m: int,
    mfmas_n: int,
    warp_tile_n: int,
) -> List[Value]:
    """Add the bias to the implicit-GEMM forward's accumulator fragments.

    The output channel of fragment column ``ni`` of a lane is
    ``k_out_group_base + block_n_off + warp_n_idx * mfmas_n * warp_tile_n +
    ni * warp_tile_n + col(lane)``, where ``col`` is the column the MMA op's
    destination layout gives slot 0 of the lane (the same map the direct and
    cshuffle stores scatter by). ``k_out_group_base`` is ``g * kpg`` (or
    ``g * Gm`` under group merging, whose merged run is contiguous in K), None
    when ungrouped. Columns past the group's kpg read a bias the store then
    drops; columns past K read 0.
    """
    rsrc = conv_bias_rsrc(b, bias, bias_bytes)
    c0 = b.const_i32(0)
    _, col = op.dst_layout().coord(b, lane, 0)
    warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * warp_tile_n))
    k = b.add(b.add(block_n_off, warp_n_off), col)
    if k_out_group_base is not None:
        k = b.add(k_out_group_base, k)
    vals = []
    for ni in range(mfmas_n):
        k_ni = k if ni == 0 else b.add(k, b.const_i32(ni * warp_tile_n))
        vals.append(load_conv_bias_f32(b, rsrc, k_ni, elem, c0))
    return add_conv_bias_to_accs(b, accs, vals, mfmas_m=mfmas_m, mfmas_n=mfmas_n)


def load_conv_bias_vec_f32(
    b: IRBuilder, rsrc: Value, k: Value, n: int, elem: Type, c0: Value
) -> Value:
    """``bias[k .. k+n)`` widened to an ``<n x f32>`` vector.

    The run is read in loads of at most 16 bytes (one for 16-bit ``elem`` up
    to ``n = 8``), each element widened, then packed. ``k`` is the element
    index of the first channel; ``c0`` is the soffset (an i32 zero). A run
    that ends past ``Bias_bytes`` reads 0 there.
    """
    eb = _bias_elem_bytes(elem)
    per = min(n, 16 // eb)
    if n % per != 0:
        raise ValueError(f"load_conv_bias_vec_f32: {n} elements in loads of {per}")
    off = b.mul(k, b.const_i32(eb))
    comps: List[Value] = []
    for j in range(0, n, per):
        voff = off if j == 0 else b.add(off, b.const_i32(j * eb))
        v = b.buffer_load_vN(rsrc, voff, c0, elem, per)
        comps += [b.cast_to_f32(b.vec_extract(v, i)) for i in range(per)]
    return b.vec_pack(comps, F32)


def load_conv_bias_run_f32(
    b: IRBuilder, rsrc: Value, k: Value, n: int, elem: Type, c0: Value
) -> Value:
    """``bias[k .. k+n)`` as an ``<n x f32>`` vector, one scalar load per
    element (:func:`load_conv_bias_f32`): for a run that may start off the
    alignment of a vector load (a per-group channel count that is not a
    multiple of the run)."""
    comps = [
        load_conv_bias_f32(b, rsrc, k if j == 0 else b.add(k, b.const_i32(j)), elem, c0)
        for j in range(n)
    ]
    return b.vec_pack(comps, F32)
