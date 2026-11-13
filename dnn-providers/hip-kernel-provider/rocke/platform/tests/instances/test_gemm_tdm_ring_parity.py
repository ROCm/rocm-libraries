# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU regression tests for the gfx1250 TDM ring's LDS addressing.

Both assertions here guard defects that shipped: the emitted kernel built,
lowered, passed byte-identity between the two engines, and computed garbage on
silicon. Byte-identity cannot catch either one -- it compares our two engines
against each other, and both were ported from the same source, so both were
wrong and in agreement. These are IR-shape assertions precisely because that is
the level the defects lived at, and they are mutation-checked: reverting either
fix fails the corresponding test.

1. ``lds_parity`` reaching the WMMA read path. The load side offsets each
   K-tile into its own ring slot; without the same shift on the read the WMMAs
   always read slot 0 while the mover fills another -- a read/write race, which
   is why the symptom was non-deterministic and worsened with ring depth.
   gfx1250 is a WMMA target, so this path is the one that runs there; the MFMA
   path kept its parity logic, which is what made the omission read as
   plausible code.

2. ``tensor_dim1`` carrying the tensor's remaining extent rather than the tile
   height. The mover bounds address generation on it, so a constant describes a
   tensor ending one tile past every origin. Square shapes cannot expose this;
   a ragged M can.
"""

from __future__ import annotations

import re
import unittest

ARCH = "gfx1250"


def _tdm_spec(*, depth: int, dtype: str = "bf16", lds_k_pad: int = 8):
    """A minimal valid gfx1250 TDM GEMM spec at the given ring depth."""
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
    )

    warp_tile_k = 64 if dtype.startswith(("fp8", "bf8")) else 32
    c_dtype = "bf16" if dtype.startswith(("fp8", "bf8")) else dtype
    return UniversalGemmSpec(
        name="tdm_ring_parity_probe",
        tile=TileSpec(
            tile_m=64,
            tile_n=64,
            tile_k=32 if warp_tile_k == 32 else 64,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=warp_tile_k,
        ),
        trait=TraitSpec(
            pipeline="mem",
            scheduler="intrawave",
            epilogue="default",
            pad_m=True,
            pad_n=True,
            pad_k=True,
            tdm=True,
            tdm_depth=depth,
            lds_k_pad=lds_k_pad,
        ),
        data=DataSpec(
            dtype_a=dtype,
            dtype_b=dtype,
            dtype_c=c_dtype,
            dtype_acc="fp32",
            layout="RCR",
        ),
        wave_size=32,
    )


def _lower(spec) -> str:
    from rocke.core.lower_llvm import lower_kernel_to_llvm
    from rocke.instances.common.gemm_universal import (
        build_universal_gemm,
        is_valid_spec,
    )

    ok, why = is_valid_spec(spec, arch=ARCH)
    if not ok:
        raise AssertionError(f"spec rejected: {why}")
    ir = build_universal_gemm(spec, arch=ARCH)
    # tensor_load_to_lds needs the llvm23 flavor; pin it rather than take the
    # environment's, as core/test_gfx1250_tdm.py does.
    return lower_kernel_to_llvm(ir, arch=ARCH, llvm_flavor="llvm23")


class TestGemmTdmRingParity(unittest.TestCase):
    def test_wmma_read_path_offsets_by_ring_parity(self):
        """Depth >= 2 must multiply the ring parity into the LDS read address.

        Asserted *differentially* against depth 1 rather than as an absolute
        count. ``mul nsw i32 %x, 64`` also appears in unrelated address
        arithmetic -- an absolute threshold passes with the defect present,
        which is how the first version of this test failed to catch it. Depth 1
        is the honest baseline: one buffer, so no parity term belongs there,
        and every extra multiply at depth >= 2 is the parity reaching A and B.
        """
        # block_m == block_n == 64 for this spec, so the parity stride is 64.
        parity_mul = re.compile(r"mul nsw i32 %\w+, 64\b")

        single = _lower(_tdm_spec(depth=1))
        self.assertIn("tensor.load.to.lds", single, "depth 1 did not emit TDM")
        baseline = len(parity_mul.findall(single))

        for depth in (2, 3, 4):
            with self.subTest(tdm_depth=depth):
                ll = _lower(_tdm_spec(depth=depth))
                self.assertIn("tensor.load.to.lds", ll, "not a TDM kernel")
                self.assertGreater(
                    len(parity_mul.findall(ll)),
                    baseline,
                    "WMMA read path is missing the ring-parity row offset; "
                    "every WMMA would read ring slot 0 while the mover fills "
                    "another, which is a read/write race",
                )

    def test_descriptor_extent_is_dynamic_not_the_tile_height(self):
        """``tensor_dim1`` must be the tensor's residual extent, not a constant.

        Upstream emits ``smax(extent - row_off, 0)``. A constant tile height
        makes every descriptor claim the tensor ends one tile past its own
        origin, which is wrong for every tile but the first.
        """
        ll = _lower(_tdm_spec(depth=2))
        self.assertIn("tensor.load.to.lds", ll, "not a TDM kernel")
        # The residual is computed against the M/N kernel params and clamped.
        self.assertRegex(
            ll,
            r"sub nsw i32 %[MN], %\w+",
            "descriptor extent is not derived from the M/N tensor extent",
        )
        self.assertIn(
            "llvm.smax.i32",
            ll,
            "descriptor extent is not clamped at zero",
        )

    def test_fp8_takes_the_same_addressing_path(self):
        """The defects were dtype-independent; fp8 must carry the same terms."""
        parity_mul = re.compile(r"mul nsw i32 %\w+, 64\b")
        kw = dict(dtype="fp8e4m3", lds_k_pad=48)
        baseline = len(parity_mul.findall(_lower(_tdm_spec(depth=1, **kw))))
        ll = _lower(_tdm_spec(depth=2, **kw))
        self.assertIn("tensor.load.to.lds", ll, "not a TDM kernel")
        self.assertGreater(len(parity_mul.findall(ll)), baseline)
        self.assertRegex(ll, r"sub nsw i32 %[MN], %\w+")


if __name__ == "__main__":
    unittest.main()
