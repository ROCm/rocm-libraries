# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# CPU contract for the masks-1/2 -> DPP split in the warp XOR reduce helpers.
#
# Masks 1 and 2 are intra-quad and must lower through warp_shuffle_xor_dpp
# (``tile.dpp_xor``, quad_perm on gfx9 CDNA: no ds_swizzle / no lgkmcnt(0)
# wait). Masks 4 and 8 cross the quad and stay on warp_shuffle_xor
# (``tile.ds_swizzle_xor``). A non-f32/i32 value must fall back to
# warp_shuffle_xor rather than raise (the warp_shuffle_xor_dpp type guard).
import unittest

from rocke.core.ir import IRBuilder
from rocke.helpers.attention import warp_xor_reduce_sum
from rocke.helpers.reduction import _warp_xor_reduce


def _op_names(kernel):
    out = []

    def walk(region):
        for op in region.ops:
            out.append(op.name)
            for r in getattr(op, "regions", None) or []:
                walk(r)

    walk(kernel.body)
    return out


class WarpXorReduceDppSplit(unittest.TestCase):
    def test_sum_reduce_masks_1_2_use_dpp_wider_use_ds_swizzle(self):
        # stages=4 walks masks 1, 2, 4, 8.
        for const_name, arg in (("const_f32", 1.0), ("const_i32", 1)):
            b = IRBuilder("sum_reduce_split")
            warp_xor_reduce_sum(b, getattr(b, const_name)(arg), stages=4)
            names = _op_names(b.kernel)
            self.assertEqual(names.count("tile.dpp_xor"), 2, const_name)
            self.assertEqual(names.count("tile.ds_swizzle_xor"), 2, const_name)

    def test_generic_reduce_masks_1_2_use_dpp(self):
        # wave_size=16 walks masks 1, 2, 4, 8.
        b = IRBuilder("generic_reduce_split")
        _warp_xor_reduce(b, b.const_f32(1.0), combine="sum", wave_size=16)
        names = _op_names(b.kernel)
        self.assertEqual(names.count("tile.dpp_xor"), 2)
        self.assertEqual(names.count("tile.ds_swizzle_xor"), 2)

    def test_non_f32_i32_falls_back_to_ds_swizzle_without_raising(self):
        # f64 lacks a register-only DPP path; it must split onto ds_swizzle, not
        # raise inside warp_shuffle_xor_dpp.
        b = IRBuilder("sum_reduce_f64")
        warp_xor_reduce_sum(b, b.const_f64(1.0), stages=4)
        names = _op_names(b.kernel)
        self.assertEqual(names.count("tile.dpp_xor"), 0)
        self.assertIn("tile.ds_swizzle_xor", names)


if __name__ == "__main__":
    unittest.main()
