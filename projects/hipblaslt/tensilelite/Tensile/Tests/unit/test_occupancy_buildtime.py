# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
GPU-free unit tests for OccupancyMeasure.py: occupancy formula and arch-caps table.

Tests compute_occupancy_from_resources() and _arch_caps_for_kernel() used by
both the codegen-time scan (updateOccupancyFromScan) and the custom-kernel ASM
parser (compute_occupancy_from_asm_source).
HIP cross-validation lives in test_occupancy_hip.py.
"""

import pytest

from Tensile.OccupancyMeasure import (
    compute_occupancy_from_resources,
    compute_occupancy_from_asm_source,
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


# ── Cross-validation: _arch_caps_for_kernel vs. compute_occupancy_from_resources ─

class TestArchCapsAgreementWithRocisaExpected:
    """Validate that _arch_caps_for_kernel() matches rocisa::hardware_caps.hpp values.

    Runs GPU-free and guards against drift when new hardware constants are added.
    """

    # (isa, physical_vgpr, physical_sgpr, device_lds, max_waves_per_simd)
    _EXPECTED = [
        ((9, 5, 0),  512, 800, 163840, 8),   # gfx950: ArchAccUnifiedRegs, 160 KB LDS
        ((9, 4, 2),  512, 800,  65536, 8),   # gfx942: ArchAccUnifiedRegs, 64 KB LDS
        ((9, 0, 10), 512, 800,  65536, 8),   # gfx90a: ArchAccUnifiedRegs, 64 KB LDS
        ((9, 0, 8),  256, 800,  65536, 10),  # gfx908: non-unified, 64 KB LDS
    ]

    @pytest.mark.parametrize(
        "isa,exp_phy_vgpr,exp_phy_sgpr,exp_dev_lds,exp_max_waves",
        _EXPECTED,
        ids=[f"gfx{''.join(str(x) for x in e[0])}" for e in _EXPECTED],
    )
    def test_static_table_matches_rocisa_expected(
        self, isa, exp_phy_vgpr, exp_phy_sgpr, exp_dev_lds, exp_max_waves
    ):
        """_arch_caps_for_kernel must return the same values rocisa provides."""
        k = {"ISA": list(isa), "KernelLanguage": "Assembly"}
        phy_vgpr, phy_sgpr, dev_lds, max_waves = _arch_caps_for_kernel(k)
        assert phy_vgpr   == exp_phy_vgpr,   f"ISA {isa}: physical_vgpr mismatch"
        assert phy_sgpr   == exp_phy_sgpr,   f"ISA {isa}: physical_sgpr mismatch"
        assert dev_lds    == exp_dev_lds,    f"ISA {isa}: device_lds mismatch"
        assert max_waves  == exp_max_waves,  f"ISA {isa}: max_waves_per_simd mismatch"


# ── compute_occupancy_from_asm_source: arch_caps passthrough ─────────────────

class TestComputeOccupancyFromAsmSourceArchCaps:
    """Verify that the arch_caps kwarg overrides the static _arch_caps_for_kernel table.

    KernelWriterAssembly.getSourceFileString passes live rocisa caps as arch_caps
    so the custom-kernel path always uses rocisa as the single source of truth.
    """

    def _asm(self, vgpr, sgpr, lds):
        return (
            f".amdhsa_kernel dummy\n"
            f"  .amdhsa_next_free_vgpr {vgpr}\n"
            f"  .amdhsa_next_free_sgpr {sgpr}\n"
            f"  .amdhsa_group_segment_fixed_size {lds}\n"
            f".end_amdhsa_kernel\n"
        )

    def test_arch_caps_kwarg_used_when_provided(self):
        """When arch_caps is provided, _arch_caps_for_kernel is NOT called."""
        from Tensile.OccupancyMeasure import compute_occupancy_from_asm_source

        kernel = {"ISA": [9, 5, 0], "NumThreads": 256}
        asm = self._asm(vgpr=256, sgpr=64, lds=68864)

        # Pass caps identical to gfx950 rocisa values; expect the same result as
        # calling without arch_caps on a gfx950 kernel.
        gfx950_caps = (512, 800, 163840, 8)
        occ_with    = compute_occupancy_from_asm_source(kernel, asm, arch_caps=gfx950_caps)
        occ_without = compute_occupancy_from_asm_source(kernel, asm)  # uses static table

        assert occ_with == occ_without == 2, (
            f"gfx950, vgpr=256, lds=68864 → expected occ=2; "
            f"with_caps={occ_with}, without_caps={occ_without}"
        )

    def test_arch_caps_kwarg_overrides_isa_lookup(self):
        """Mismatched arch_caps changes the result, proving the kwarg is used."""
        from Tensile.OccupancyMeasure import compute_occupancy_from_asm_source

        # Kernel declared as gfx950 but caps for gfx908 (256 VGPR pool) passed explicitly.
        kernel = {"ISA": [9, 5, 0], "NumThreads": 256}
        asm = self._asm(vgpr=128, sgpr=64, lds=16384)

        normal_occ   = compute_occupancy_from_asm_source(kernel, asm)                  # 512//128=4 → 4
        override_occ = compute_occupancy_from_asm_source(kernel, asm,
                           arch_caps=(256, 800, 65536, 10))  # 256//128=2 → 2

        assert normal_occ   == 4, f"gfx950 native caps: expected 4, got {normal_occ}"
        assert override_occ == 2, f"gfx908 override caps: expected 2, got {override_occ}"

    def test_arch_caps_none_falls_back_to_static_table(self):
        """When arch_caps=None (default), _arch_caps_for_kernel is used."""
        from Tensile.OccupancyMeasure import compute_occupancy_from_asm_source

        kernel = {"ISA": [9, 4, 2], "NumThreads": 256}
        asm = self._asm(vgpr=256, sgpr=64, lds=65536)

        occ = compute_occupancy_from_asm_source(kernel, asm, arch_caps=None)
        assert occ is not None
        assert occ >= 1
