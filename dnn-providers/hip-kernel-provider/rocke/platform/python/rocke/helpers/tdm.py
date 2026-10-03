# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 Tensor-DMA (TDM) descriptor construction.

The gfx1250 TDM engine moves a tiled sub-region of a global tensor into LDS
(``tensor_load_to_lds``) or back out (``tensor_store_from_lds``) from a single
wave-uniform descriptor, bypassing the per-lane VGPR staging an ordinary
global->LDS copy needs.

``IRBuilder.tensor_load_to_lds`` / ``tensor_store_from_lds`` take the descriptor
as five opaque vectors (``vec<i32,4>``, ``vec<i32,8>``, ``vec<i32,4>``,
``vec<i32,4>``, ``vec<i32,8>``) and validate only their widths. This module
builds those vectors: it is pure composition over existing IR ops, so it adds no
op to either engine.

The bit layout is documented in
``dsl_docs/optimization/arch/gfx1250.md`` §21.10 and mirrors two in-repo
reference implementations -- ``ck_tile/core/arch/amd_tdm_descriptor.hpp`` and
MIOpen's ``hipconv/.../cdna5/tdm_desc.h``.

Scope of this version: **rank 1-2, non-gather, non-iterating**. Groups 2/3 are
therefore all-zero (the rank>2 / gather / iterate modes union into them) and
group 4 is reserved. Higher ranks raise rather than emit a half-correct
descriptor -- see :func:`tdm_descriptor_groups`.

Dimension order is **fastest-varying first** -- ``dim0`` is the contiguous
dimension, and stride slot ``i`` is the pitch that advances ``dim(i+1)``, so
there is no stride for ``dim0`` (it is implicitly unit-stride). This is the
opposite of NumPy's ``shape`` convention and is verified on device by
``examples/gfx1250/isa_features/tdm_descriptor_verify.py``. Most callers want
:func:`tdm_row_major_2d`, which takes natural ``(rows, cols)`` and cannot be
mis-ordered::

    d0, d1, d2, d3, d4 = tdm_row_major_2d(
        b,
        global_addr=a_addr,           # i64 byte address
        lds_addr=b.smem_addr_of(smem),
        rows=m, cols=k, row_pitch=k,  # full tensor, elements
        tile_rows=tile_m, tile_cols=tile_k,
        elem_bytes=2,                 # fp16
    )
    b.tensor_load_to_lds(d0, d1, d2, d3, d4)
    b.s_wait_tensorcnt(0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence, Union

from rocke.core.ir import I32, I64

if TYPE_CHECKING:
    from rocke.core.ir import IRBuilder, Value

__all__ = [
    "TDM_MAX_RANK",
    "TdmPadding",
    "tdm_data_size_code",
    "tdm_descriptor_groups",
    "tdm_load_to_lds",
    "tdm_row_major_2d",
]

# Groups 2/3 carry dims 2-4, the gather row-index array, or the iterate config.
# Only the rank<=2 non-gather non-iterate case leaves them zero, which is all
# this module emits.
TDM_MAX_RANK = 2

# group1.data_size is a 2-bit log2(element bytes) code.
_DATA_SIZE_CODE = {1: 0, 2: 1, 4: 2, 8: 3}

_GROUP0_COUNT_VALID = 1  # 0 = NULL tensor, 1 = valid tensor
_GROUP0_TYPE = 2  # "must be 2"

ScalarOrValue = Union[int, "Value"]


@dataclass(frozen=True)
class TdmPadding:
    """LDS-side padding applied during a load (not honoured for stores).

    The engine inserts ``amount_dwords`` every ``interval_dwords`` of the
    inner dimension. In practice this is an LDS row stride: set
    ``interval_dwords`` to the tile's inner extent in dwords and
    ``amount_dwords`` to ``lds_row_stride - tile_inner`` in dwords.

    Both descriptor fields are biased by one; :meth:`encode` applies the bias.
    """

    interval_dwords: int
    amount_dwords: int

    def __post_init__(self) -> None:
        iv, amt = self.interval_dwords, self.amount_dwords
        if not 2 <= iv <= 256 or (iv & (iv - 1)) != 0:
            raise ValueError(
                "TdmPadding.interval_dwords must be a power of two in [2, 256], "
                f"got {iv}"
            )
        if not 1 <= amt <= 128:
            raise ValueError(
                f"TdmPadding.amount_dwords must be in [1, 128], got {amt}"
            )

    def encode(self) -> tuple[int, int]:
        """Return the biased ``(pad_interval, pad_amount)`` field values."""
        return self.interval_dwords.bit_length() - 2, self.amount_dwords - 1


def tdm_data_size_code(elem_bytes: int) -> int:
    """``log2(elem_bytes)`` as the 2-bit ``group1.data_size`` field."""
    try:
        return _DATA_SIZE_CODE[elem_bytes]
    except KeyError:
        raise ValueError(
            f"TDM elem_bytes must be 1, 2, 4, or 8, got {elem_bytes}"
        ) from None


def _as_signed_i32(value: int) -> int:
    """Reinterpret an unsigned 32-bit pattern as the signed value LLVM wants."""
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value >= (1 << 31) else value


def _pack_word(
    b: IRBuilder,
    fields: Sequence[tuple[str, ScalarOrValue, int, int]],
) -> Value:
    """Pack ``(name, value, bit_offset, width)`` fields into one i32.

    Compile-time ints fold into a single constant; SSA values are masked,
    shifted, and or-ed in. Out-of-range constants raise rather than silently
    truncating -- a descriptor that is quietly wrong drives a DMA engine.
    """
    const_bits = 0
    dynamic: list[tuple[ScalarOrValue, int, int]] = []

    for name, value, offset, width in fields:
        mask = (1 << width) - 1
        if isinstance(value, int):
            if not 0 <= value <= mask:
                raise ValueError(
                    f"TDM field {name!r} does not fit {width} bits "
                    f"(0..{mask}), got {value}"
                )
            const_bits |= value << offset
        else:
            dynamic.append((value, offset, width))

    if not dynamic:
        return b.const_i32(_as_signed_i32(const_bits))

    acc: Value | None = None
    if const_bits:
        acc = b.const_i32(_as_signed_i32(const_bits))

    for value, offset, width in dynamic:
        part = value
        if width < 32:
            part = b.land(part, b.const_i32(_as_signed_i32((1 << width) - 1)))
        if offset:
            part = b.shl(part, b.const_i32(offset))
        acc = part if acc is None else b.lor(acc, part)

    assert acc is not None
    return acc


def _lo_hi(
    b: IRBuilder, value: ScalarOrValue, shift: int
) -> tuple[ScalarOrValue, ScalarOrValue]:
    """Split ``value`` into ``(low, high)`` about ``shift``.

    Constants split exactly. SSA operands are i32, so a split at or above
    bit 32 has a statically-zero high half -- and it must be returned as the
    literal 0, **not** as ``lshr i32 x, 32``: a shift by the full width is
    poison in LLVM, and poison in a descriptor's stride field points the DMA
    engine at a wild address. That is a memory fault at runtime, not a
    compile error, so it cannot be caught by any static check.
    """
    if isinstance(value, int):
        return value & ((1 << shift) - 1), value >> shift
    if shift >= 32:
        return value, 0
    return value, b.lshr(value, b.const_i32(shift))


def _make_vector(b: IRBuilder, words: Sequence[Value], *, lanes: int) -> Value:
    vec = b.zero_vec(I32, lanes)
    for index, word in enumerate(words):
        vec = b.vec_insert(vec, word, index)
    return vec


def _require_i64(name: str, value: Value) -> None:
    if value.type is not I64:
        raise TypeError(f"{name} must be i64, got {value.type}")


def tdm_descriptor_groups(
    b: IRBuilder,
    *,
    global_addr: Value,
    lds_addr: Value,
    tensor_dims: Sequence[ScalarOrValue],
    tensor_strides: Sequence[ScalarOrValue],
    tile_dims: Sequence[int],
    elem_bytes: int,
    padding: TdmPadding | None = None,
    workgroup_mask: int = 0,
    early_timeout: bool = False,
    is_restore: bool = False,
    scalarize: bool = True,
) -> tuple[Value, Value, Value, Value, Value]:
    """Build the five TDM descriptor groups for a rank-1/2 plain transfer.

    Parameters
    ----------
    global_addr
        Byte address of the tensor origin as an ``i64``. rocke exposes no
        ``ptrtoint``, so pass this as an ``i64`` kernel parameter rather than a
        ``ptr<...,global>``.
    lds_addr
        LDS byte address, ``i32`` or the ``i64`` from
        :meth:`IRBuilder.smem_addr_of` (truncated here).
    tensor_dims
        Full-tensor element counts, **fastest-varying first**: ``dim0`` is the
        contiguous dimension. Reads past these on the positive side return
        zero, so tile tails need no predication.
    tensor_strides
        Element pitches, 48-bit descriptor fields. ``tensor_strides[i]``
        advances ``tensor_dims[i + 1]`` -- ``dim0`` has no stride entry because
        it is implicitly unit-stride, so the last entry is unused. For a
        row-major 2D tensor ``tensor_strides[0]`` is the row pitch.
    tile_dims
        Tile element counts in the same fastest-first order; compile-time,
        16 bits each.
    elem_bytes
        Element size feeding ``data_size`` (1, 2, 4, or 8).
    padding
        Optional LDS padding; load-only (see :class:`TdmPadding`).
    scalarize
        Lift each word into an SGPR. The descriptor must be wave-uniform, so
        leave this on unless the caller already guarantees uniformity.

    Returns
    -------
    ``(d0, d1, d2, d3, d4)`` ready for :meth:`IRBuilder.tensor_load_to_lds` or
    :meth:`IRBuilder.tensor_store_from_lds`.
    """
    rank = len(tensor_dims)
    if rank != len(tensor_strides) or rank != len(tile_dims):
        raise ValueError(
            "tensor_dims, tensor_strides, and tile_dims must agree in length; "
            f"got {rank}, {len(tensor_strides)}, {len(tile_dims)}"
        )
    if not 1 <= rank <= TDM_MAX_RANK:
        raise NotImplementedError(
            f"tdm_descriptor_groups supports rank 1..{TDM_MAX_RANK}, got {rank}. "
            "Higher ranks need groups 2/3, whose mode union (dims 2-4 vs gather "
            "indices vs iterate config) is not modelled here."
        )
    _require_i64("tdm_descriptor_groups global_addr", global_addr)
    data_size = tdm_data_size_code(elem_bytes)

    if lds_addr.type is I64:
        lds_addr = b.trunc(lds_addr, I32)
    elif lds_addr.type is not I32:
        raise TypeError(
            f"tdm_descriptor_groups lds_addr must be i32 or i64, got {lds_addr.type}"
        )

    for index, extent in enumerate(tile_dims):
        if not 0 <= extent <= 0xFFFF:
            raise ValueError(
                f"tile_dims[{index}] must fit 16 bits, got {extent}"
            )
    for index, stride in enumerate(tensor_strides):
        if isinstance(stride, int) and not 0 <= stride < (1 << 48):
            raise ValueError(
                f"tensor_strides[{index}] must fit the 48-bit field, got {stride}"
            )

    # Pad rank-1 out to the two-dimension descriptor slots with zeros.
    dims = list(tensor_dims) + [0] * (TDM_MAX_RANK - rank)
    strides = list(tensor_strides) + [0] * (TDM_MAX_RANK - rank)
    tiles = list(tile_dims) + [0] * (TDM_MAX_RANK - rank)

    pad_interval, pad_amount = padding.encode() if padding else (0, 0)

    # ---- group 0: addresses and op mode -------------------------------------
    # is_store stays 0: direction is chosen by which intrinsic is called.
    # scope/th stay 0: cache policy rides the intrinsic's own immediate.
    global_lo = b.trunc(global_addr, I32)
    global_hi = b.trunc(b.lshr(global_addr, b.const_i64(32)), I32)

    group0_words = [
        _pack_word(
            b,
            [
                ("count", _GROUP0_COUNT_VALID, 0, 2),
                ("is_restore", int(is_restore), 2, 1),
                ("is_store", 0, 3, 1),
                ("nv", 0, 4, 1),
                ("scope", 0, 5, 2),
                ("th", 0, 7, 3),
                ("gather_index_size", 0, 30, 1),
                ("gather_mode", 0, 31, 1),
            ],
        ),
        lds_addr,
        global_lo,
        _pack_word(
            b,
            [
                ("global_addr_hi", global_hi, 0, 25),
                ("type", _GROUP0_TYPE, 30, 2),
            ],
        ),
    ]

    # ---- group 1: shape, stride, modifiers ----------------------------------
    dim0_lo, dim0_hi = _lo_hi(b, dims[0], 16)
    dim1_lo, dim1_hi = _lo_hi(b, dims[1], 16)
    # The two strides are NOT encoded alike: stride[0] is lo32+hi16 (split at
    # bit 32), stride[1] is lo16+hi32 (split at bit 16). See gfx1250.md §21.10 --
    # ck_tile shifts the second mode by 32 and drops bits [16:32]; that is
    # masked only while stride[1] < 2**16. Follow the MIOpen encoding.
    stride0_lo, stride0_hi = _lo_hi(b, strides[0], 32)
    stride1_lo, stride1_hi = _lo_hi(b, strides[1], 16)

    group1_words = [
        _pack_word(
            b,
            [
                ("workgroup_mask", workgroup_mask, 0, 16),
                ("data_size", data_size, 16, 2),
                ("atomic_barrier_enable", 0, 18, 1),
                ("iterate_enable", 0, 19, 1),
                ("pad_enable", int(padding is not None), 20, 1),
                ("early_timeout", int(early_timeout), 21, 1),
                ("pad_interval", pad_interval, 22, 3),
                ("pad_amount", pad_amount, 25, 7),
            ],
        ),
        _pack_word(
            b,
            [
                ("atomic_barrier_address", 0, 0, 16),
                ("tensor_dim0_lo", dim0_lo, 16, 16),
            ],
        ),
        _pack_word(
            b,
            [
                ("tensor_dim0_hi", dim0_hi, 0, 16),
                ("tensor_dim1_lo", dim1_lo, 16, 16),
            ],
        ),
        _pack_word(
            b,
            [
                ("tensor_dim1_hi", dim1_hi, 0, 16),
                ("tile_dim0", tiles[0], 16, 16),
            ],
        ),
        _pack_word(
            b,
            [
                ("tile_dim1", tiles[1], 0, 16),
                ("tile_dim2", 0, 16, 16),
            ],
        ),
        _pack_word(b, [("tensor_dim0_stride_lo", stride0_lo, 0, 32)]),
        _pack_word(
            b,
            [
                ("tensor_dim0_stride_hi", stride0_hi, 0, 16),
                ("tensor_dim1_stride_lo", stride1_lo, 16, 16),
            ],
        ),
        _pack_word(b, [("tensor_dim1_stride_hi", stride1_hi, 0, 32)]),
    ]

    if scalarize:
        group0_words = [b.to_sgpr_u32(w) for w in group0_words]
        group1_words = [b.to_sgpr_u32(w) for w in group1_words]

    d0 = _make_vector(b, group0_words, lanes=4)
    d1 = _make_vector(b, group1_words, lanes=8)
    d2 = b.zero_vec(I32, 4)
    d3 = b.zero_vec(I32, 4)
    d4 = b.zero_vec(I32, 8)
    return d0, d1, d2, d3, d4


def tdm_row_major_2d(
    b: IRBuilder,
    *,
    global_addr: Value,
    lds_addr: Value,
    rows: ScalarOrValue,
    cols: ScalarOrValue,
    row_pitch: ScalarOrValue,
    tile_rows: int,
    tile_cols: int,
    elem_bytes: int,
    **kwargs: object,
) -> tuple[Value, Value, Value, Value, Value]:
    """:func:`tdm_descriptor_groups` for a row-major 2D tensor.

    Takes natural ``(rows, cols)`` order and performs the fastest-first
    transposition the descriptor wants, so the dimension order cannot be got
    backwards. ``row_pitch`` is the element distance between consecutive rows
    (``cols`` for a packed tensor, larger for a padded or sliced one).

    The tile lands in LDS row-major and contiguous: ``tile_rows * tile_cols``
    elements, ``tile_cols`` per row.
    """
    return tdm_descriptor_groups(
        b,
        global_addr=global_addr,
        lds_addr=lds_addr,
        tensor_dims=(cols, rows),
        tensor_strides=(row_pitch, 0),
        tile_dims=(tile_cols, tile_rows),
        elem_bytes=elem_bytes,
        **kwargs,  # type: ignore[arg-type]
    )


def tdm_load_to_lds(
    b: IRBuilder,
    *,
    cachepolicy: int = 0,
    wait: bool = False,
    **descriptor_kwargs: object,
) -> None:
    """Build a descriptor and issue ``tensor_load_to_lds`` in one step.

    ``descriptor_kwargs`` are forwarded verbatim to
    :func:`tdm_descriptor_groups`. Set ``wait=True`` to follow the load with
    ``s_wait_tensorcnt(0)``; leave it off to overlap the transfer with compute
    and place the wait yourself.
    """
    groups = tdm_descriptor_groups(b, **descriptor_kwargs)  # type: ignore[arg-type]
    d0, d1, d2, d3, d4 = groups
    b.tensor_load_to_lds(d0, d1, d2, d3, d4, cachepolicy=cachepolicy)
    if wait:
        b.s_wait_tensorcnt(0)
