# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""HIP lowering of the dense gfx1250 16x16x128 f8f6f4 WMMA atom.

Python twin of ``test_dense_f8f6f4_hip.cpp``: the same fp8 GEMM (a 64x64x128
tile on the dense atom, no TDM), lowered to HIP by the native Python lowerer,
must emit the builtin call the C++ test pins -- the 6-operand form with an FP8
format immediate (0) per matrix and an i16 modC of 0.
"""

from __future__ import annotations

import unittest


def _spec():
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
    )

    return UniversalGemmSpec(
        name="dense_f8f6f4_hip",
        tile=TileSpec(
            tile_m=64,
            tile_n=64,
            tile_k=128,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=128,
        ),
        trait=TraitSpec(pipeline="mem", epilogue="default"),
        data=DataSpec(
            dtype_a="fp8e4m3",
            dtype_b="fp8e4m3",
            dtype_c="bf16",
            dtype_acc="fp32",
            layout="RCR",
        ),
        wave_size=32,
        block_size=128,
        batched=False,
    )


class TestDenseF8f6f4Hip(unittest.TestCase):
    def test_dense_atom_lowers_to_hip(self):
        from rocke.core.lower_hip import lower_kernel_to_hip
        from rocke.instances.common.gemm_universal import build_universal_gemm

        kernel = build_universal_gemm(_spec(), arch="gfx1250")
        hip = lower_kernel_to_hip(kernel, arch="gfx1250")
        calls = [l for l in hip.splitlines() if "16x16x128_f8f6f4(" in l]
        self.assertTrue(calls, hip)
        first = calls[0].strip()
        self.assertIn("= __builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(0, ", first)
        self.assertTrue(first.endswith(", (int16_t)0, acc_m0_n0);"), first)


if __name__ == "__main__":
    unittest.main()
