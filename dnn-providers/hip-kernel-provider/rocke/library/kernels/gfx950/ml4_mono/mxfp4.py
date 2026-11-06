# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""ML4's native MXFP4 address and operand helpers.

The address equations are the aiter
``common/plan/tiles.py``/``unit_native_mxfp4`` equations.  They are kept in a
small rocKE module so UG and down GEMV use the same byte-exact addressing
instead of growing independent copies.
"""

from __future__ import annotations

from rocke.core.ir import I32


def _v(b, value):
    return value if hasattr(value, "type") else b.const_i32(int(value))


def _add(b, *values):
    out = _v(b, values[0])
    for value in values[1:]:
        out = b.add(out, _v(b, value))
    return out


def _mul(b, left, right):
    return b.mul(_v(b, left), _v(b, right))


def atom_mxfp4_scale_index(row: int, col: int, cols: int) -> tuple[int, int]:
    """Host form of aiter's shuffled E8M0 scale address."""
    r32, a, b = row // 32, (row // 16) % 2, row % 16
    c8, e = col // 8, col % 4
    return (((r32 * ((cols + 7) // 8) + c8) * 4 + e) * 16 + b, a)


def _scale_index_ir(b, row, col, cols):
    """Dynamic version of :func:`atom_mxfp4_scale_index` in i32 words."""
    r32 = b.lshr(row, b.const_i32(5))
    a = b.land(b.lshr(row, b.const_i32(4)), b.const_i32(1))
    bb = b.land(row, b.const_i32(15))
    c8 = b.lshr(col, b.const_i32(3))
    e = b.land(col, b.const_i32(3))
    words_per_row = (cols + 7) // 8
    word = _mul(
        b,
        _add(b, _mul(b, _add(b, _mul(b, r32, words_per_row), c8), 4), e),
        16,
    )
    return _add(b, word, _mul(b, bb, 1)), a


def _load_scale_byte(b, rsrc, row, col, cols):
    word, byte = _scale_index_ir(b, row, col, cols)
    packed = b.buffer_load(rsrc, _mul(b, word, 4), b.const_i32(0), I32)
    return b.land(b.lshr(packed, _mul(b, byte, 8)), b.const_i32(0xFF))


def unit_native_mxfp4(
    b,
    w_rsrc,
    s_rsrc,
    *,
    row_group,
    k_chunk,
    k_size,
    lane,
    ln=None,
    lds,
    activation_word,
    activation_scale_word,
    coefficient=None,
):
    """Issue one aiter-native 128-K MXFP4 tile.

    The returned tuple is ``(raw_w, activation, weight_scale, activation_scale,
    coefficient)`` and is consumed by :func:`mx_mma`.  ``activation_word`` is
    an LDS word containing the first packed FP8 byte pair for the lane's
    128-K chunk.
    """
    # ``ln`` is the logical weight lane.  UG uses a shuffled lane for the
    # paired gate/up rows; using the physical lane here silently produces a
    # plausible but different expert tile.
    ln = lane if ln is None else ln
    row = _add(b, _mul(b, row_group, 16), b.land(ln, b.const_i32(15)))
    raw_off = _add(
        b,
        _mul(
            b,
            _add(
                b,
                _mul(b, row_group, k_size // 64),
                _mul(b, k_chunk, 2),
            ),
            128,
        ),
        _mul(b, ln, 4),
    )
    raw = b.buffer_load_vN(
        w_rsrc, raw_off, b.const_i32(0), I32, 4
    )
    raw = b.vec_pack([b.vec_extract(raw, i) for i in range(4)], I32)
    weight_scale = _load_scale_byte(
        b,
        s_rsrc,
        row,
        _add(b, _mul(b, k_chunk, 4), b.lshr(ln, b.const_i32(4))),
        k_size // 32,
    )
    activation = []
    for half in range(2):
        words = b.smem_load_vN(
            lds,
            _add(b, activation_word, half * 16, _mul(b, b.lshr(lane, b.const_i32(4)), 4)),
            dtype=I32,
            n=4,
        )
        activation.extend(b.vec_extract(words, i) for i in range(4))
    activation = b.vec_pack(activation, I32)
    scale = b.smem_load_vN(
        lds, activation_scale_word, dtype=I32, n=1
    )
    activation_scale = b.vec_extract(scale, 0)
    return raw, activation, weight_scale, activation_scale, coefficient


__all__ = ["atom_mxfp4_scale_index", "unit_native_mxfp4"]
