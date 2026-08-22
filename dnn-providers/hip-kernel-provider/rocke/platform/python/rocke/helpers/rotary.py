# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Rotary position embedding (RoPE) helpers.

RoPE rotates pairs of head-dim values by a position-dependent angle::

 pair_idx in [0, head_size / 2)
 theta = inv_freq[pair_idx] # precomputed: 1 / 10000^(2k/H)
 cos_t = cos(token_position * theta)
 sin_t = sin(token_position * theta)
 out[2*i] = in[2*i] * cos_t - in[2*i+1] * sin_t # interleaved
 out[2*i+1] = in[2*i] * sin_t + in[2*i+1] * cos_t

CK Tile's reference exposes two layouts:

* **Interleaved** (``rotate_half=False``) -- the two paired elements
 are at ``2*i`` and ``2*i+1`` in the head-dim. Hugging Face's GPT-J
 / LLaMA-1 cache uses this layout.
* **Non-interleaved** (``rotate_half=True``) -- the head-dim is
 bisected into ``[0, H/2)`` and ``[H/2, H)``; element ``i`` pairs
 with element ``i + H/2``. Hugging Face's LLaMA-2 / 3, Qwen, and
 most modern checkpoints use this layout.

Both layouts use the same ``(cos_t, sin_t)`` table, just a different
pair-index function. The helpers here let a kernel author pick the
layout via a single flag on the spec.

What v1 ships:

* :func:`pair_indices` -- compile-time pair-index pair for ``i`` in
 ``[0, head_size / 2)`` under either layout.
* :func:`load_cos_sin` -- one (cos, sin) f32 pair from the precomputed
 rotary table at offset ``(token_position, pair_idx)``.
* :func:`apply_rotary_pair_f32` -- the 2x2 rotation, f32 in / f32 out.
* :func:`RotarySpec` -- compile-time description (layout + table
 strides) consumed by both fmha-fwd (Q+K pre-rotate) and
 fmha-appendkv (K-only rotate before cache write).

Limitations of v1:

* ``cos`` / ``sin`` tables are f32. Half-precision rotary tables
 exist (the ``rope_fp16`` variant in some CK Tile reference
 kernels); they save half the table bytes at the cost of a
 ``cvt.f32.f16`` per element.
* No NeoX / partial-rotary variants -- ``rotary_dim < head_size``
 is on the v2 follow-on list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ..core.ir import F32, IRBuilder, PtrType, Value


__all__ = [
    "RotaryLayout",
    "RotarySpec",
    "apply_rotary_pair_f32",
    "load_cos_sin",
    "pair_indices",
]


RotaryLayout = Literal["interleaved", "half"]


@dataclass(frozen=True)
class RotarySpec:
    """Compile-time description of one rotary embedding configuration.

    Attributes
    ----------
    head_size
    Total head dimension ``H``. Must be even; the rotary halves
    / pairs each cover ``H/2`` elements.
    layout
    ``"interleaved"`` for the GPT-J / LLaMA-1 layout (pair =
    ``(2*i, 2*i+1)``), ``"half"`` for the LLaMA-2 / 3 / Qwen
    layout (pair = ``(i, i + H/2)``).
    table_stride_pos
    Stride in elements between adjacent positions in the
    precomputed cos / sin tables. Each table has shape
    ``(max_pos, H/2)``; typical layout is row-major so the
    stride equals ``H/2`` (in elements).
    """

    head_size: int
    layout: RotaryLayout = "half"
    table_stride_pos: int = 0  # 0 = compute as head_size // 2

    def __post_init__(self):
        if self.head_size <= 0 or self.head_size % 2 != 0:
            raise ValueError(
                f"rotary head_size must be positive and even, got {self.head_size}"
            )
        if self.layout not in ("interleaved", "half"):
            raise ValueError(
                f"rotary layout must be 'interleaved' or 'half', got {self.layout!r}"
            )

    @property
    def pair_count(self) -> int:
        return self.head_size // 2

    @property
    def stride_pos(self) -> int:
        return self.table_stride_pos or self.pair_count


def pair_indices(spec: RotarySpec, pair_idx: int) -> Tuple[int, int]:
    """Return the ``(lo, hi)`` head-dim indices for one rotary pair.

    For ``"interleaved"``: ``(2*i, 2*i+1)``.
    For ``"half"``: ``(i, i + H/2)``.
    """
    if pair_idx < 0 or pair_idx >= spec.pair_count:
        raise ValueError(f"pair_idx out of range: {pair_idx}")
    if spec.layout == "interleaved":
        return (2 * pair_idx, 2 * pair_idx + 1)
    return (pair_idx, pair_idx + spec.pair_count)


def load_cos_sin(
    b: IRBuilder,
    cos_table: Value,
    sin_table: Value,
    *,
    token_pos: Value,
    pair_idx: Value,
    spec: RotarySpec,
) -> Tuple[Value, Value]:
    """Load one ``(cos, sin)`` f32 pair from the rotary tables.

    The tables are laid out as ``(max_position, H/2)`` row-major in
    f32; ``offset = token_pos * stride_pos + pair_idx``. This helper
    issues two scalar global loads.

    The caller is responsible for promoting ``token_pos`` to a
    wave-uniform i32 if the loads should land in scalar registers
    (the FMHA-fwd path does this via
    :meth:`IRBuilder.to_sgpr_u32`).
    """
    if cos_table.type != sin_table.type:
        raise ValueError("cos / sin tables must have matching pointer type")
    if not isinstance(cos_table.type, PtrType):
        raise ValueError("cos_table must be a pointer")
    if cos_table.type.pointee != F32:
        raise ValueError("rotary tables must be ptr<f32> in v1")
    offset = b.add(b.mul(token_pos, b.const_i32(spec.stride_pos)), pair_idx)
    cos_v = b.global_load_f32(cos_table, offset)
    sin_v = b.global_load_f32(sin_table, offset)
    return cos_v, sin_v


def apply_rotary_pair_f32(
    b: IRBuilder,
    lo: Value,
    hi: Value,
    cos_t: Value,
    sin_t: Value,
) -> Tuple[Value, Value]:
    """Apply the 2x2 rotation to one ``(lo, hi)`` element pair.

    Math::

    lo' = lo * cos_t - hi * sin_t
    hi' = lo * sin_t + hi * cos_t

    Compute is in f32; the AMDGPU backend folds the four
    multiplies and two adds into two ``v_fma_f32`` per pair.
    """
    if lo.type.name != "f32" or hi.type.name != "f32":
        raise ValueError("apply_rotary_pair_f32 expects f32 inputs")
    if cos_t.type.name != "f32" or sin_t.type.name != "f32":
        raise ValueError("apply_rotary_pair_f32 expects f32 cos / sin")
    new_lo = b.fsub(b.fmul(lo, cos_t), b.fmul(hi, sin_t))
    new_hi = b.fadd(b.fmul(lo, sin_t), b.fmul(hi, cos_t))
    return new_lo, new_hi
