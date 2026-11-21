# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Emission helpers shared by the gfx942 and gfx950 dense attention kernels.

Each helper here was a byte-for-byte copy in both arch modules
(``kernels/gfx942/attention_dense.py`` and ``kernels/gfx950/attention_dense.py``).
"""

from __future__ import annotations

from kernels.common._lse_store import zero_keyless_row

__all__ = ["emit_o_store"]


def emit_o_store(
    b, o, q_row_byte, d_half, o_acc, rcp_l, dtype, d_tiles, width, row_has_key=None
):
    """O = acc / l, ``width`` contiguous head-dim elements per global store.

    ``row_has_key`` zeroes rows that see no key; each arch's builder decides
    which specs pass it. ``None`` emits the unchanged store sequence.
    """
    for dt in range(d_tiles):
        for g in range(4):
            for c in range(0, 4, width):
                d0 = b.add(b.const_i32(dt * 32 + g * 8 + c), d_half)
                addr = b.add(q_row_byte, d0)
                vals = [
                    b.cast_f32_to(
                        b.fmul(b.vec_extract(o_acc[dt], g * 4 + kk), rcp_l), dtype
                    )
                    for kk in range(c, c + width)
                ]
                packed = b.vec_pack(vals, dtype)
                if row_has_key is not None:
                    packed = zero_keyless_row(b, packed, row_has_key, dtype, width)
                b.global_store_vN(o, addr, packed, width, align=2 * width)
