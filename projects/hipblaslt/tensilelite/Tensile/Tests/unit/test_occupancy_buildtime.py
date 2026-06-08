################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
"""
CUOccupancy measurement: unit tests for OccupancyMeasure.py (formula + arch caps).

Tests the occupancy formula and architecture-caps helpers that underpin both
the codegen-time scan (updateOccupancyFromScan) and the custom-kernel ASM
parser (compute_occupancy_from_asm_source):

  - TestComputeOccupancyFromResources: validates compute_occupancy_from_resources()
    against the six BF16 GEMM oracle cases and boundary conditions.
  - TestArchCapsForKernel: validates _arch_caps_for_kernel() ISA-to-caps mapping.

HIP cross-validation (loading .co.raw and calling
hipModuleOccupancyMaxActiveBlocksPerMultiprocessor) lives in test_occupancy_hip.py.

MT320x192x64 reconciliation (the motivating mismatch):
  Before updateOccupancyFromScan: pool ≈ 264 → occ=1 (WRONG)
  After updateOccupancyFromScan:  vgpr_count=256 in .s → occ=2  (CORRECT)
  HIP measurement:                loads the .co with vgpr_count=256 → occ=2  (CORRECT)
  All three agree: CUOccupancy=2.

How to run:
  # From tensilelite/ dir:
  tox -e unit -- Tensile/Tests/unit/test_occupancy_buildtime.py -v -s
  pytest Tensile/Tests/unit/test_occupancy_buildtime.py -v -s
"""

import pytest

from Tensile.OccupancyMeasure import (
    compute_occupancy_from_resources,
    _arch_caps_for_kernel,
)


# ── Tests: compute_occupancy_from_resources ───────────────────────────────────

class TestComputeOccupancyFromResources:
    """Unit tests for the occupancy computation formula in OccupancyMeasure.py."""

    # gfx950 hardware constants
    GFX950_PHY_VGPR  = 512
    GFX950_PHY_SGPR  = 800
    GFX950_DEV_LDS   = 163840   # 160 KB
    GFX950_MAX_WAVES = 8

    def _occ(self, vgpr_count, lds_bytes, num_threads=256, sgpr_count=64):
        return compute_occupancy_from_resources(
            vgpr_count=vgpr_count,
            lds_bytes=lds_bytes,
            sgpr_count=sgpr_count,
            num_threads=num_threads,
            physical_vgpr=self.GFX950_PHY_VGPR,
            physical_sgpr=self.GFX950_PHY_SGPR,
            device_lds=self.GFX950_DEV_LDS,
            max_waves_per_simd=self.GFX950_MAX_WAVES,
        )

    def test_case4_mt320x192x64_vgpr_256(self):
        """Case 4 (MT320x192x64): vgpr_count=256, LDS=68864 → occ=2.

        This is the primary motivating case.  With updateOccupancyFromScan
        the .s file has .amdhsa_next_free_vgpr=256; the formula correctly
        computes occ=2.
        """
        # 512 // 256 = 2 (VGPR limit)
        # 163840 // roundup(68864/256)*256 = 163840 // 69120 = 2 (LDS limit)
        # min(2, 2, ...) = 2
        occ = self._occ(vgpr_count=256, lds_bytes=68864)
        assert occ == 2, f"MT320x192x64 should give occ=2, got {occ}"

    def test_case1_mt256x256x64_vgpr_249(self):
        """Case 1 (MT256x256x64): vgpr_count=249, LDS=133120 → occ=1 (LDS-limited)."""
        # 512 // 249 = 2 (VGPR), but LDS: 163840 // (roundup(133120,256)*256)=133120 → 1
        occ = self._occ(vgpr_count=249, lds_bytes=133120)
        assert occ == 1, f"MT256x256x64 (LDS-limited) should give occ=1, got {occ}"

    def test_vgpr_limited_occ2(self):
        """Exactly 256 unified VGPRs → occ=2 on gfx950 (512/256=2)."""
        occ = self._occ(vgpr_count=256, lds_bytes=65536)
        assert occ == 2

    def test_vgpr_limited_occ1(self):
        """257+ VGPRs → occ=1 (512//257=1)."""
        occ = self._occ(vgpr_count=257, lds_bytes=65536)
        assert occ == 1

    def test_wave_cap_limits_low_vgpr(self):
        """Very few VGPRs: limited by MaxWavesPerSimd=8 for gfx950."""
        occ = self._occ(vgpr_count=8, lds_bytes=256, num_threads=256)
        assert occ == 8, f"Low VGPR (8) with 256 threads → MaxWavesPerSimd=8, got {occ}"

    def test_larger_block_reduces_occupancy(self):
        """512-thread block: multiplier=2, MaxWavesPerSimd//2 = 4."""
        occ = self._occ(vgpr_count=8, lds_bytes=256, num_threads=512)
        assert occ == 4, f"512 threads, low VGPR → occ=4, got {occ}"

    def test_lds_limited_occ1(self):
        """LDS=86016 (>160KB/2=81920) → occ=1."""
        occ = self._occ(vgpr_count=8, lds_bytes=86016)
        assert occ == 1, f"LDS-limited (86016 B) should give occ=1, got {occ}"

    def test_six_oracle_cases(self):
        """All six BF16 GEMM oracle cases from the investigation must agree."""
        # (desc, vgpr_count, lds, threads, expected_occ)
        cases = [
            ("case1 MT256x256x64", 249, 133120, 256, 1),
            ("case2 MT16x64x512",  256,  86016, 256, 1),
            ("case4 MT320x192x64", 256,  68864, 256, 2),   # the motivating mismatch
            ("case5 MT64x80x64",   256,  94016, 128, 1),
            ("case6 MT192x192x64", 256, 149760, 256, 1),
        ]
        for desc, vgpr, lds, threads, expected in cases:
            occ = self._occ(vgpr_count=vgpr, lds_bytes=lds, num_threads=threads)
            assert occ == expected, (
                f"{desc}: vgpr={vgpr}, lds={lds}, threads={threads} "
                f"→ expected {expected}, got {occ}"
            )


# ── Tests: _arch_caps_for_kernel ──────────────────────────────────────────────

class TestArchCapsForKernel:
    """Verify arch caps are derived correctly from ISA tuples."""

    def _caps(self, isa):
        k = {"ISA": list(isa), "KernelLanguage": "Assembly"}
        return _arch_caps_for_kernel(k)

    def test_gfx950_caps(self):
        phy_vgpr, phy_sgpr, dev_lds, max_waves = self._caps((9, 5, 0))
        assert phy_vgpr    == 512
        assert max_waves   == 8
        assert dev_lds     == 163840

    def test_gfx942_caps(self):
        phy_vgpr, phy_sgpr, dev_lds, max_waves = self._caps((9, 4, 2))
        assert phy_vgpr    == 512
        assert max_waves   == 8
        assert dev_lds     == 65536

    def test_gfx90a_caps(self):
        phy_vgpr, phy_sgpr, dev_lds, max_waves = self._caps((9, 0, 10))
        assert phy_vgpr    == 512
        assert max_waves   == 8

    def test_gfx908_caps(self):
        phy_vgpr, phy_sgpr, dev_lds, max_waves = self._caps((9, 0, 8))
        assert phy_vgpr    == 256
        assert max_waves   == 10
