# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# CPU contract for the DPP split in the warp XOR reduce helpers.
#
# For f32/i32 values, the stages inside one 16-lane DPP row lower through
# warp_shuffle_xor_dpp (``tile.dpp_xor``): masks 1 and 2 as themselves
# (quad_perm on gfx9 CDNA), masks 4 and 8 as dpp masks 7 and 15
# (row_half_mirror / row_mirror). Masks 16 and up leave the row and stay on
# warp_shuffle_xor. A non-f32/i32 value falls back to warp_shuffle_xor rather
# than raising.
import unittest

from rocke.core.ir import IRBuilder
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.helpers.attention import warp_xor_reduce_sum
from rocke.helpers.reduction import _warp_xor_reduce


def _ops(kernel):
    out = []

    def walk(region):
        for op in region.ops:
            out.append(op)
            for r in getattr(op, "regions", None) or []:
                walk(r)

    walk(kernel.body)
    return out


def _dpp_masks(kernel):
    return [op.attrs["xor_mask"] for op in _ops(kernel) if op.name == "tile.dpp_xor"]


def _count(kernel, name):
    return sum(op.name == name for op in _ops(kernel))


class WarpXorReduceDppSplit(unittest.TestCase):
    def test_sum_reduce_in_row_stages_use_dpp(self):
        # stages=6 walks masks 1, 2, 4, 8, 16, 32.
        for const_name, arg in (("const_f32", 1.0), ("const_i32", 1)):
            b = IRBuilder("sum_reduce_split")
            warp_xor_reduce_sum(b, getattr(b, const_name)(arg), stages=6)
            self.assertEqual(_dpp_masks(b.kernel), [1, 2, 7, 15], const_name)
            # mask 16 stays on ds_swizzle (mask 32 takes ds_bpermute).
            self.assertEqual(_count(b.kernel, "tile.ds_swizzle_xor"), 1, const_name)

    def test_generic_reduce_in_row_stages_use_dpp(self):
        # wave_size=64 walks masks 1, 2, 4, 8, 16, 32.
        for combine in ("sum", "max"):
            b = IRBuilder("generic_reduce_split")
            _warp_xor_reduce(b, b.const_f32(1.0), combine=combine, wave_size=64)
            self.assertEqual(_dpp_masks(b.kernel), [1, 2, 7, 15], combine)
            self.assertEqual(_count(b.kernel, "tile.ds_swizzle_xor"), 1, combine)

    def test_dpp_masks_lower_on_gfx9_cdna(self):
        # gfx9 has no row_xmask: masks 7 and 15 must hit the row_half_mirror /
        # row_mirror encodings, not raise at lowering.
        b = IRBuilder("sum_reduce_gfx942")
        warp_xor_reduce_sum(b, b.const_f32(1.0), stages=4)
        ll = _lower_kernel_to_llvm_python(b.kernel, arch="gfx942")
        self.assertEqual(ll.count("@llvm.amdgcn.update.dpp.i32(i32 %"), 4)
        self.assertNotIn("@llvm.amdgcn.ds.swizzle(i32 %", ll)

    def test_non_f32_i32_falls_back_to_ds_swizzle_without_raising(self):
        # f64 has no register-only DPP path; it must split onto ds_swizzle, not
        # raise inside warp_shuffle_xor_dpp.
        b = IRBuilder("sum_reduce_f64")
        warp_xor_reduce_sum(b, b.const_f64(1.0), stages=4)
        self.assertEqual(_dpp_masks(b.kernel), [])
        self.assertGreater(_count(b.kernel, "tile.ds_swizzle_xor"), 0)


if __name__ == "__main__":
    unittest.main()
