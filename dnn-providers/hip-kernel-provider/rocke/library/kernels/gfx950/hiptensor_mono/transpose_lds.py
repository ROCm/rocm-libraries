# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""LDS-tiled 2D transpose body (512 threads, 64 x 64 f16 tile per block).

For the permutes that coalesce to a plain 2D transpose (the canonical
hipTensor operands: ``X[R, C] -> Y[C, R]``, row-major numpy view). The gather
body reads one cache line per element when the output's fast axis is the
input's slow axis; this body reads 16-byte vectors along the input rows into
an LDS tile, then writes 16-byte vectors along the output rows (8 strided LDS
reads per thread). Requires ``R % 64 == 0`` and ``C % 64 == 0``.
Signature ``(X, Y)``; ``block_id_x`` = tile index (row-tile major).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from rocke.core.ir import F16, IRBuilder, KernelDef, PtrType

__all__ = ["TransposeLdsSpec", "build_transpose_lds", "transpose_lds_ok", "TILE"]

TILE = 64
_PAD = 8  # halves of row padding (keeps 16-byte aligned LDS rows)


def transpose_lds_ok(
    x_shape: Tuple[int, ...], perm: Tuple[int, ...], block_size: int
) -> bool:
    return (
        len(x_shape) == 2
        and tuple(perm) == (1, 0)
        and x_shape[0] % TILE == 0
        and x_shape[1] % TILE == 0
        and block_size == TILE * TILE // 8
    )


@dataclass(frozen=True)
class TransposeLdsSpec:
    """``X[rows, cols] -> Y[cols, rows]`` (f16, row-major numpy views)."""

    rows: int
    cols: int
    block_size: int = TILE * TILE // 8
    name: str = "rocke_htmono_tlds"


def build_transpose_lds(spec: TransposeLdsSpec, *, arch: str = "gfx950") -> KernelDef:
    R, C, block_size, name = spec.rows, spec.cols, spec.block_size, spec.name
    if not transpose_lds_ok((R, C), (1, 0), block_size):
        raise ValueError(
            f"transpose_lds needs R, C multiples of {TILE} and {TILE * TILE // 8} threads"
        )
    b = IRBuilder(f"{name}_{R}x{C}")
    b.kernel.attrs["max_workgroup_size"] = block_size
    X = b.param("X", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    Y = b.param("Y", PtrType(F16, "global"), noalias=True, writeonly=True, align=16)
    tid = b.thread_id_x()
    bid = b.block_id_x()
    tiles_c = C // TILE
    r0 = b.mul(b.div(bid, b.const_i32(tiles_c)), b.const_i32(TILE))
    c0 = b.mul(b.mod(bid, b.const_i32(tiles_c)), b.const_i32(TILE))
    lds = b.smem_alloc(F16, [TILE, TILE + _PAD], name_hint="tlds")
    c8 = b.const_i32(8)
    r = b.div(tid, c8)  # 0..63
    cc = b.mul(b.mod(tid, c8), c8)  # 0, 8, .., 56
    b.sync()  # a previous tile of the same ticket may still be read from LDS
    v = b.global_load_vN(
        X, b.add(b.mul(b.add(r0, r), b.const_i32(C)), b.add(c0, cc)), F16, 8, align=16
    )
    b.smem_store_vN(lds, [r, cc], v, n=8)
    b.sync()
    # output row (input column) r, output columns (input rows) cc .. cc + 7
    vals = [
        b.vec_extract(
            b.smem_load_vN(lds, b.add(cc, b.const_i32(j)), r, dtype=F16, n=1), 0
        )
        for j in range(8)
    ]
    out = b.vec_pack(vals, F16)
    b.global_store_vN(
        Y, b.add(b.mul(b.add(c0, r), b.const_i32(R)), b.add(r0, cc)), out, 8, align=16
    )
    return b.kernel
