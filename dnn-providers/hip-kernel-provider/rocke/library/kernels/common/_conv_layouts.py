# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Conv-owned LDS fragment readers for the implicit-GEMM conv instances.

:class:`ConvKOuterFragmentReader` reads MMA operand fragments out of a
**K-outer** conv LDS tile through the transpose-read instructions. It is
library code: only the conv wgrad and dgrad instances use it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rocke.core.ir import F16, IRBuilder, Type, Value

__all__ = ["ConvKOuterFragmentReader"]


@dataclass(frozen=True)
class ConvKOuterFragmentReader:
    """Operand-fragment reads from a **K-outer** conv LDS tile.

    Distinct from :class:`TransposeLdsReader`, which is not reusable here:
    that one is pinned to the ``M = 16`` attention shape with ``K_L = K / 4``
    and a fixed ``col = (lane % 4) * 4``, whereas a conv K-outer tile has a
    variable atom edge (16 or 32), an arbitrary per-lane fragment length, a
    running base offset on both axes, and two wave regimes.

    Shared by the wgrad and dgrad instances
    (:mod:`kernels.common.conv_implicit_gemm_wgrad`,
    :mod:`kernels.common.conv_implicit_gemm_dgrad`). Mirrors ``rocke_conv_tr_frag`` in the C++ engine
    (``library/cpp/instances/common/conv_implicit_gemm_conv_compute_phase.cpp``);
    the two must stay byte-identical, so keep the emission order below in
    lockstep with it.

    The dataclass is unbound; call :meth:`bind` once per kernel, at the point
    the lane-derived constants should enter the instruction stream, then call
    :meth:`fragment` per operand fragment.
    """

    wave_size: int

    def bind(self, b: IRBuilder, lane: Value) -> "_BoundConvKOuterFragmentReader":
        """Materialize the lane-derived constants once per kernel.

        Emits IR **only** on the wave64 path, which is the one whose column
        term reuses them. Callers must therefore invoke this exactly where the
        old inline ``if lds_k_outer and wave_size == 64`` block sat: binding
        unconditionally would add ops to every non-K-outer config and move
        every existing golden.
        """
        if self.wave_size == 64:
            return _BoundConvKOuterFragmentReader(
                wave_size=self.wave_size,
                lane=lane,
                lane_mod4=b.mul(b.mod(lane, b.const_i32(4)), b.const_i32(4)),
                grp16=b.div(b.mod(lane, b.const_i32(16)), b.const_i32(4)),
            )
        return _BoundConvKOuterFragmentReader(
            wave_size=self.wave_size, lane=lane, lane_mod4=None, grp16=None
        )


@dataclass(frozen=True)
class _BoundConvKOuterFragmentReader:
    """SSA values produced by :meth:`ConvKOuterFragmentReader.bind`."""

    wave_size: int
    lane: Value
    lane_mod4: Optional[Value]
    grp16: Optional[Value]

    def fragment(
        self,
        b: IRBuilder,
        smem: Value,
        mn_base: Value,
        k_base: Value,
        *,
        mn_atom: int,
        n: int,
        dtype: Type = F16,
    ) -> Value:
        """One MMA operand fragment from a K-outer tile via transpose reads.

        Two regimes, selected on wave size, because the lane mapping is a
        property of how the wave covers the atom edge:

        wave32 (gfx1250 WMMA 16x16x32) -- the *result* layout is lane ``l``
        owning column ``l % 16`` and K-half ``l // 16``, but that is what the
        lane must end up holding, not the address it supplies.
        ``ds_load_tr16_b128`` transposes an 8x8 element block *within each group
        of 8 lanes*: the 8 lanes of a group each read 8 contiguous elements, and
        lane ``j`` of the group receives element ``j`` from all 8 of those runs.
        So to land column ``l % 16`` in lane ``l``, the group addresses the
        8-column block containing it and lane ``l`` supplies the ``l % 8``-th K
        row of the run, not its own column:

            col  = mn_base + ((l % 16) // 8) * 8
            row0 = k_base  + (l // 16) * n + (l % 8)

        Verified on silicon -- addressing this as if the instruction returned a
        straight run of K at the lane's own column reads a transposed operand
        and produces numerically wrong output.

        wave64 (gfx950 MFMA) -- 64 lanes over a 16- or 32-wide edge means four
        (or two) groups *within* the free axis, which is where the
        ``((l % 16) // 4)`` and ``(l % 4) * 4`` terms come from.
        ``ds_read_b64_tr_b16`` returns 4 per lane, so ``n / 4`` reads.
        """
        lane = self.lane
        if self.wave_size == 32:
            c16 = b.const_i32(16)
            c8 = b.const_i32(8)
            lane_mod16 = b.mod(lane, c16)
            col_grp = b.div(lane_mod16, c8)
            col_off = b.mul(col_grp, c8)
            col = b.add(mn_base, col_off)
            lane_div16 = b.div(lane, c16)
            c_n = b.const_i32(n)
            row_mul = b.mul(lane_div16, c_n)
            lane_mod8 = b.mod(lane, c8)
            row_sum = b.add(row_mul, lane_mod8)
            row0 = b.add(k_base, row_sum)
            parts = [
                b.ds_read_tr16_b128(
                    smem, b.add(row0, b.const_i32(8 * r)), col, dtype=dtype
                )
                for r in range(n // 8)
            ]
            out = parts[0]
            for pt in parts[1:]:
                out = b.vec_concat(out, pt)
            return out
        c_mn = b.const_i32(mn_atom)
        col = b.add(
            mn_base,
            b.add(
                b.mul(b.div(b.mod(lane, c_mn), b.const_i32(16)), b.const_i32(16)),
                self.lane_mod4,
            ),
        )
        row0 = b.add(
            k_base, b.add(b.mul(b.div(lane, c_mn), b.const_i32(n)), self.grp16)
        )
        parts = [
            b.ds_read_tr16_b64(smem, b.add(row0, b.const_i32(4 * r)), col, dtype=dtype)
            for r in range(n // 4)
        ]
        out = parts[0]
        for pt in parts[1:]:
            out = b.vec_concat(out, pt)
        return out
