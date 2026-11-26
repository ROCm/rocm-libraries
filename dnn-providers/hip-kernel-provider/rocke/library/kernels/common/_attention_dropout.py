# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Philox dropout on P for the attention forward kernels (hipDNN SEM-015).

The mask is SEM-015's (``rocke.numeric.philox_dropout`` is its oracle), so a
backward of another engine regenerates it from the same seed and offset.
Dropout multiplies the post-softmax P by ``keep / (1 - p)`` before PV: the row
sum and the LSE stay pre-dropout, the dropped probabilities are zeroed before
the PV MFMA, and ``1 / (1 - p)`` is folded into the epilogue's ``1 / l``.

A body that computes S^T = K @ Q^T holds, in each accumulator slot, the score of
the key its MMA C layout maps the slot's row to. The tile helpers take that map
(``slot_key``) and need four consecutive slots ``4g .. 4g + 3`` to hold four
consecutive keys from a multiple of 4: the words of one Philox call, used in
order, so no lane exchange is needed. Slot 0 must hold a lane's lowest key of
its sub-tile (the run counters add their distance from it).
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass

import numpy as np

from rocke.core.ir import F32, I32, I64, PtrType
from rocke.helpers.attention_dropout import (
    dropout_keep,
    dropout_row_counter,
    dropout_words,
)
from rocke.helpers.rng import philox_counter_add
from rocke.numeric.philox_dropout import dropout_scale, dropout_threshold

#: Keys per Philox call: SEM-015 uses the four words of a call for four
#: consecutive keys, so counters advance once per four keys.
KEYS_PER_CALL = 4

# Kernel params, in ABI order. Seed and offset come by value or from device
# memory (a training step can advance the offset on the device, as PyTorch's
# generator does); a null pointer selects the value, so one kernel serves both
# and the forms may mix.
DROPOUT_PARAMS = (
    ("dropout_seed_ptr", "ptr"),
    ("dropout_offset_ptr", "ptr"),
    ("dropout_seed", "i64"),
    ("dropout_offset", "i64"),
    ("dropout_threshold", "i32"),
    ("dropout_scale", "f32"),
)


@dataclass(frozen=True)
class DropoutState:
    """The dropout state a body reads once per CTA."""

    seed_lo: object
    seed_hi: object
    offset_lo: object
    offset_hi: object
    threshold: object
    scale: object
    # Philox round forms (rocke.helpers.rng.PhiloxCall): xor3 needs v_bitop3_b32
    # (gfx950); mad64 pays only where v_mad_u64_u32 is full rate (gfx950).
    xor3: bool = False
    mad64: bool = False


def _value_or_load(b, ptr, value):
    """The load sits behind the null check: a by-value launch passes no address."""
    from_memory = b.cmp_ne(b.global_ptr_to_i64(ptr), b.const_i64(0))
    branch = b.scf_if_else(from_memory, result_types=[I64])
    with branch as (then_ctx, else_ctx):
        with then_ctx:
            b.scf_yield(b.global_load(ptr, b.const_i32(0), I64, align=8))
        with else_ctx:
            b.scf_yield(value)
    (result,) = branch.results
    return result


def declare_dropout_params(
    b, *, xor3: bool = False, mad64: bool = False
) -> DropoutState:
    """Declare the params of :data:`DROPOUT_PARAMS` and resolve seed and offset."""
    seed_ptr = b.param(
        "dropout_seed_ptr", PtrType(I64, "global"), noalias=True, readonly=True, align=8
    )
    offset_ptr = b.param(
        "dropout_offset_ptr",
        PtrType(I64, "global"),
        noalias=True,
        readonly=True,
        align=8,
    )
    seed_val = b.param("dropout_seed", I64)
    offset_val = b.param("dropout_offset", I64)
    threshold = b.param("dropout_threshold", I32)
    scale = b.param("dropout_scale", F32)
    seed = _value_or_load(b, seed_ptr, seed_val)
    offset = _value_or_load(b, offset_ptr, offset_val)
    shift = b.const_i64(32)
    return DropoutState(
        seed_lo=b.to_sgpr_u32(b.trunc(seed, I32)),
        seed_hi=b.to_sgpr_u32(b.trunc(b.lshr(seed, shift), I32)),
        offset_lo=b.to_sgpr_u32(b.trunc(offset, I32)),
        offset_hi=b.to_sgpr_u32(b.trunc(b.lshr(offset, shift), I32)),
        threshold=threshold,
        scale=scale,
        xor3=xor3,
        mad64=mad64,
    )


def dropout_query_counter(
    b, st: DropoutState, *, batch, head, query, num_heads, seqlen_q, seqlen_kv
):
    """Counter words of key 0 of row ``(batch, head, query)``; hoist per work item.

    ``seqlen_q`` / ``seqlen_kv`` are the declared extents (``max_sq`` /
    ``max_sk`` of SEM-015), also for varlen and paged; ``batch`` is the
    sequence and ``query`` the position in it. The row index fits i32 as the
    LSE index does.
    """
    row = b.add(b.mul(b.add(b.mul(batch, num_heads), head), seqlen_q), query)
    return dropout_row_counter(
        b, offset_lo=st.offset_lo, offset_hi=st.offset_hi, row=row, max_sk=seqlen_kv
    )


def dropout_tile_counter(b, row_ctr, tile_key0, slot_key):
    """Counter of this lane's slot 0 in the KV tile starting at key ``tile_key0``:
    the base every run of the tile adds to, so it is computed once per tile.

    Runs add their key's distance from slot 0, which a layout whose lanes differ
    by a fixed key offset reduces to a constant.
    """
    k4 = b.lshr(b.add(tile_key0, slot_key(0)), b.const_i32(2))
    return philox_counter_add(b, row_ctr[0], row_ctr[1], k4)


def _run_counter(b, tile_ctr, *, nsub, g, slot_key, sub_keys):
    """Counter of the call of run ``(nsub, g)``: slots ``4g .. 4g + 3`` of the
    ``sub_keys``-key sub-tile ``nsub``."""
    if nsub == 0 and g == 0:
        return tile_ctr
    dist = b.add(
        b.const_i32(nsub * sub_keys),
        b.sub(slot_key(KEYS_PER_CALL * g), slot_key(0)),
    )
    k4 = b.lshr(dist, b.const_i32(2))
    return philox_counter_add(b, tile_ctr[0], tile_ctr[1], k4)


def dropout_keep_run(b, st: DropoutState, tile_ctr, *, nsub, g, slot_key, sub_keys):
    """Keep predicates of slots ``4g .. 4g + 3`` of sub-tile ``nsub``."""
    c_lo, c_hi = _run_counter(
        b, tile_ctr, nsub=nsub, g=g, slot_key=slot_key, sub_keys=sub_keys
    )
    words = dropout_words(
        b,
        seed_lo=st.seed_lo,
        seed_hi=st.seed_hi,
        ctr_lo=c_lo,
        ctr_hi=c_hi,
        xor3=st.xor3,
        mad64=st.mad64,
    )
    return [dropout_keep(b, w, st.threshold) for w in words]


def apply_dropout_tile(
    b, st: DropoutState, row_ctr, p, *, tile_key0, slot_key, sub_keys
):
    """Zero the dropped entries of one KV tile's P (``p[nsub][slot]``).

    ``slot_key(slot)`` is the key a slot holds within its ``sub_keys``-key
    sub-tile (the C layout's row of S^T). Masked and padded keys already hold P = 0, so they need no case. Returns a new ``p``;
    the scale stays out (folded into ``1 / l``).
    """
    zero = b.const_f32(0.0)
    tile_ctr = dropout_tile_counter(b, row_ctr, tile_key0, slot_key)
    out = []
    for nsub, sub in enumerate(p):
        sub_out = list(sub)
        for g in range(len(sub) // KEYS_PER_CALL):
            keep = dropout_keep_run(
                b, st, tile_ctr, nsub=nsub, g=g, slot_key=slot_key, sub_keys=sub_keys
            )
            for w, k in enumerate(keep):
                slot = g * KEYS_PER_CALL + w
                sub_out[slot] = b.select(k, sub[slot], zero)
        out.append(sub_out)
    return out


def dropout_signature(sig):
    """Append the dropout params to a ``SignatureBuilder`` (ABI order)."""
    for name, ty in DROPOUT_PARAMS:
        sig = sig.ptr(name, "i64") if ty == "ptr" else sig.scalar(name, ty)
    return sig


def _seed_or_offset(name: str, value) -> tuple[object, int]:
    """``(pointer, value)`` kernargs of one of seed / offset.

    An integer (as u64 or i64) goes by value; an array (``DeviceArray`` or
    torch tensor) of one int64 element on the device is read by the kernel.
    """
    from rocke.runtime.device_array import array_info

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an int or a device int64 array, got bool")
    if isinstance(value, numbers.Integral):
        value = int(value)
        if not -(1 << 63) <= value < (1 << 64):
            raise ValueError(f"{name}={value} does not fit 64 bits")
        return None, value - (1 << 64) if value >= (1 << 63) else value
    info = array_info(value)
    if info.dtype != "i64" or info.numel != 1:
        raise ValueError(
            f"{name} must be an int64 array of one element, got {info.dtype} "
            f"of shape {info.shape}"
        )
    if info.device is None:
        raise ValueError(f"{name} must be in device memory")
    return value, 0


def dropout_launch_args(
    enabled: bool, *, dropout_seed, dropout_offset, dropout_p
) -> dict:
    """Kernargs of the dropout params, or ``{}`` when the spec has dropout off.

    ``dropout_seed`` / ``dropout_offset``: an ``int`` each, or a device int64
    array of one element (see :func:`_seed_or_offset`); ``dropout_p`` in
    ``(0, 1)``. Inputs that disagree with the spec raise rather than launch a
    kernel whose ABI has (or lacks) the dropout params.
    """
    named = (
        ("dropout_seed", dropout_seed),
        ("dropout_offset", dropout_offset),
        ("dropout_p", dropout_p),
    )
    given = [n for n, v in named if v is not None]
    if not enabled:
        if given:
            raise ValueError(f"{', '.join(given)} provided but dropout is off")
        return {}
    if len(given) != 3:
        raise ValueError("dropout requires dropout_seed, dropout_offset and dropout_p")
    p = float(dropout_p)
    # p = 0 is no dropout: the dispatch picks a kernel without it.
    if not 0.0 < p < 1.0:
        raise ValueError(f"dropout_p must be in (0, 1), got {p}")
    seed_ptr, seed = _seed_or_offset("dropout_seed", dropout_seed)
    offset_ptr, offset = _seed_or_offset("dropout_offset", dropout_offset)
    return {
        "dropout_seed_ptr": seed_ptr,
        "dropout_offset_ptr": offset_ptr,
        "dropout_seed": seed,
        "dropout_offset": offset,
        "dropout_threshold": dropout_threshold(p),
        "dropout_scale": dropout_scale(p),
    }
