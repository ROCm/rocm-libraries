# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
GPU-free unit tests for OccupancyMeasure.py: occupancy formula and arch-caps table.

Tests compute_occupancy_from_resources() and _arch_caps_for_kernel() used by
both the codegen-time max-VGPR path (updateOccupancyFromMaxVgpr) and the custom-kernel ASM
parser (compute_occupancy_from_asm_source).
HIP cross-validation lives in test_occupancy_hip.py.
"""

from math import ceil
from types import SimpleNamespace

import pytest

# The coverage tox env runs `pytest -m unit Tensile/Tests/unit`; without this
# explicit declaration the file is silently deselected and reports 0% coverage.
pytestmark = pytest.mark.unit

from Tensile.OccupancyMeasure import (
    compute_occupancy_from_resources,
    compute_occupancy_from_asm_source,
    _arch_caps_for_kernel,
)

try:
    from Tensile.KernelWriterAssembly import KernelWriterAssembly as _KWA
    _KWA_AVAILABLE = True
except Exception:
    _KWA_AVAILABLE = False


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

        This is the primary motivating case.  With updateOccupancyFromMaxVgpr
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


# ── Fix #1: >256-thread VGPR multiplier parity ───────────────────────────────

class TestComputeOccupancyMultiplierParity:
    """Fix #1: compute_occupancy_from_resources must scale aligned VGPRs by the
    workgroup multiplier, matching getVgprOccupancy for >256-thread blocks."""

    GFX950_PHY_VGPR  = 512
    GFX950_PHY_SGPR  = 800
    GFX950_DEV_LDS   = 163840
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

    def _get_vgpr_occ_reference(self, num_threads, vgprs, total_vgpr, max_waves):
        """Mirror getVgprOccupancy logic for cross-check."""
        multiplier    = int(ceil(max(num_threads, 256) / 256.0))
        max_occ       = max_waves // multiplier
        align         = 4
        vgprs_aligned = int(ceil(vgprs / align)) * align * multiplier
        if vgprs_aligned < 1:
            return max_occ
        return min(total_vgpr // vgprs_aligned, max_occ)

    def test_512thread_200vgpr_gives_occ1(self):
        """512-thread, 200 VGPRs: multiplier=2, aligned*2=400 → 512//400=1.

        Before Fix #1 the formula ignored the multiplier on the VGPR side and
        returned 512//200=2 (wrong); with the fix it returns 1 (matching
        getVgprOccupancy).
        """
        occ = self._occ(vgpr_count=200, lds_bytes=0, num_threads=512)
        assert occ == 1, f"512-thread, 200 VGPR on gfx950 → expected 1, got {occ}"

    def test_512thread_parity_with_get_vgpr_occupancy(self):
        """compute_occupancy_from_resources must match getVgprOccupancy for 512-thread."""
        cases = [
            (512, 200, 1),   # multiplier=2, aligned=200*2=400 → 512//400=1
            (512, 64,  4),   # multiplier=2, aligned=64*2=128   → 512//128=4 (wave-capped)
            (512, 8,   4),   # multiplier=2, aligned=8*2=16     → 512//16=32 → wave cap=4
            (1024, 128, 1),  # multiplier=4, aligned=128*4=512  → 512//512=1
        ]
        for num_threads, vgprs, expected in cases:
            ref = self._get_vgpr_occ_reference(
                num_threads, vgprs, self.GFX950_PHY_VGPR, self.GFX950_MAX_WAVES)
            occ = self._occ(vgpr_count=vgprs, lds_bytes=0, num_threads=num_threads)
            assert occ == ref == expected, (
                f"threads={num_threads}, vgpr={vgprs}: "
                f"expected occ={expected}, formula={occ}, ref={ref}"
            )

    def test_256thread_unchanged(self):
        """256-thread (multiplier=1): identical to old behaviour — must not regress."""
        oracle_cases = [
            (249, 133120, 256, 1),   # LDS-limited
            (256,  86016, 256, 1),   # LDS-limited
            (256,  68864, 256, 2),   # VGPR/LDS both give 2
            (256,  94016, 128, 1),   # LDS-limited
            (256, 149760, 256, 1),   # LDS-limited
        ]
        for vgpr, lds, threads, expected in oracle_cases:
            occ = self._occ(vgpr_count=vgpr, lds_bytes=lds, num_threads=threads)
            assert occ == expected, (
                f"vgpr={vgpr}, lds={lds}, threads={threads}: "
                f"expected {expected}, got {occ}"
            )


# ── Fix #4: _arch_caps_for_kernel safe fallback ───────────────────────────────

class TestArchCapsForKernelSafeFallback:
    """Fix #4: _arch_caps_for_kernel must not silently emit garbage caps for
    unsupported ISAs; it must return None instead."""

    def _caps(self, isa):
        k = {"ISA": list(isa)}
        return _arch_caps_for_kernel(k)

    def test_gfx1100_returns_none(self):
        """gfx1100 (wave32, gfx11) is not explicitly supported → None."""
        assert self._caps((11, 0, 0)) is None, "gfx1100 must return None"

    def test_gfx1201_returns_none(self):
        """gfx12 non-gfx1250 is not explicitly supported → None."""
        assert self._caps((12, 0, 1)) is None, "gfx1201 must return None"

    def test_gfx1030_returns_none(self):
        """gfx1030 (wave32, gfx10) is not explicitly supported → None."""
        assert self._caps((10, 3, 0)) is None, "gfx1030 must return None"

    def test_gfx1250_caps(self):
        """gfx1250 is explicitly listed; caps must match hardware_caps.hpp."""
        result = self._caps((12, 5, 0))
        assert result is not None, "gfx1250 must not return None"
        phy_vgpr, phy_sgpr, dev_lds, max_waves = result
        assert phy_vgpr  == 1024,   f"gfx1250 physical_vgpr: expected 1024, got {phy_vgpr}"
        assert phy_sgpr  == 800,    f"gfx1250 physical_sgpr: expected 800, got {phy_sgpr}"
        assert dev_lds   == 327680, f"gfx1250 device_lds: expected 327680 (320KB), got {dev_lds}"
        assert max_waves == 10,     f"gfx1250 max_waves: expected 10, got {max_waves}"

    def test_gfx906_caps(self):
        """gfx906 (gfx9, non-unified) is handled by the generic gfx9 branch."""
        result = self._caps((9, 0, 6))
        assert result is not None, "gfx906 must not return None (gfx9 generic branch)"
        phy_vgpr, _, _, max_waves = result
        assert phy_vgpr  == 256, f"gfx906 physical_vgpr: expected 256, got {phy_vgpr}"
        assert max_waves == 10,  f"gfx906 max_waves: expected 10, got {max_waves}"

    def test_compute_asm_source_returns_none_for_unsupported_isa(self):
        """compute_occupancy_from_asm_source returns None for unsupported ISAs."""
        asm = (
            ".amdhsa_kernel test\n"
            "  .amdhsa_next_free_vgpr 64\n"
            "  .amdhsa_next_free_sgpr 32\n"
            "  .amdhsa_group_segment_fixed_size 16384\n"
            ".end_amdhsa_kernel\n"
        )
        result = compute_occupancy_from_asm_source({"ISA": [11, 0, 0]}, asm)
        assert result is None, f"gfx1100 compute_occupancy_from_asm_source must be None, got {result}"


# ── Drift guard: extended _arch_caps_for_kernel table vs rocisa values ────────

class TestArchCapsAgreementExtended:
    """Drift guard for the extended _arch_caps_for_kernel table (includes gfx1250).

    Must be kept in sync with rocisa/hardware_caps.hpp initArchCaps /
    initRegisterCaps when new ISAs are added.
    """

    _EXPECTED = [
        # (isa, physical_vgpr, physical_sgpr, device_lds, max_waves)
        ((9, 5, 0),  512, 800, 163840,  8),   # gfx950
        ((9, 4, 2),  512, 800,  65536,  8),   # gfx942
        ((9, 0, 10), 512, 800,  65536,  8),   # gfx90a
        ((9, 0, 8),  256, 800,  65536, 10),   # gfx908
        ((12, 5, 0), 1024, 800, 327680, 10),  # gfx1250 (Fix #4 addition)
    ]

    @pytest.mark.parametrize(
        "isa,exp_phy_vgpr,exp_phy_sgpr,exp_dev_lds,exp_max_waves",
        _EXPECTED,
        ids=[f"gfx{''.join(str(x) for x in e[0])}" for e in _EXPECTED],
    )
    def test_extended_table_matches_rocisa_expected(
        self, isa, exp_phy_vgpr, exp_phy_sgpr, exp_dev_lds, exp_max_waves
    ):
        k = {"ISA": list(isa)}
        result = _arch_caps_for_kernel(k)
        assert result is not None, f"ISA {isa}: expected caps, got None"
        phy_vgpr, phy_sgpr, dev_lds, max_waves = result
        assert phy_vgpr   == exp_phy_vgpr,   f"ISA {isa}: physical_vgpr mismatch"
        assert phy_sgpr   == exp_phy_sgpr,   f"ISA {isa}: physical_sgpr mismatch"
        assert dev_lds    == exp_dev_lds,    f"ISA {isa}: device_lds mismatch"
        assert max_waves  == exp_max_waves,  f"ISA {isa}: max_waves_per_simd mismatch"


# ── updateOccupancyFromMaxVgpr O(1) path ─────────────────────────────────────

class _MockPool:
    def __init__(self, n):
        self._n = n
    def size(self):
        return self._n


class _MockMkb:
    def __init__(self, body_text=""):
        _text = body_text
        class _Body:
            def __str__(self):
                return _text
        self.body = _Body()
        self.set_gprs_called = False
        self.set_gprs_kwargs = {}

    def setGprs(self, **kwargs):
        self.set_gprs_called = True
        self.set_gprs_kwargs = kwargs


def _make_max_vgpr_writer(arch_acc_unified=True, vgpr_size=256, agpr_size=256, sgpr_size=64):
    """Minimal KernelWriterAssembly stub for updateOccupancyFromMaxVgpr tests."""
    kw = object.__new__(_KWA)
    kw.states = SimpleNamespace(
        archCaps={
            "ArchAccUnifiedRegs": int(arch_acc_unified),
            "MaxWavesPerSimd": 8 if arch_acc_unified else 10,
            "DeviceLDS": 163840,
        },
        regCaps={"MaxVgpr": 256, "PhysicalMaxSgpr": 800},
        doubleVgpr=arch_acc_unified,
    )
    kw.vgprPool = _MockPool(vgpr_size)
    kw.agprPool = _MockPool(agpr_size)
    kw.sgprPool = _MockPool(sgpr_size)
    return kw


@pytest.mark.skipif(not _KWA_AVAILABLE, reason="KernelWriterAssembly import requires rocisa")
class TestUpdateOccupancyFromMaxVgprO1Path:
    """updateOccupancyFromMaxVgpr uses rocIsaPassResult.maxVgpr (O(1)); no regex fallback."""

    def test_missing_max_vgpr_skips_update(self):
        """maxVgpr <= 0 from rocIsaPass → skip update; CUOccupancy unchanged."""
        kw = _make_max_vgpr_writer()
        mkb = _MockMkb()
        kernel = {"CUOccupancy": 2, "NumThreads": 256, "LdsNumBytes": 0}
        kw.updateOccupancyFromMaxVgpr(kernel, mkb, -1)
        assert not mkb.set_gprs_called
        assert kernel["CUOccupancy"] == 2

    def test_pool_estimate_skips_update(self):
        """maxVgpr == pool size → no reduction possible; CUOccupancy unchanged."""
        kw = _make_max_vgpr_writer(vgpr_size=256, agpr_size=256, sgpr_size=64)
        mkb = _MockMkb()
        kernel = {"CUOccupancy": 1, "NumThreads": 256, "LdsNumBytes": 0}
        kw.updateOccupancyFromMaxVgpr(kernel, mkb, 256)
        assert not mkb.set_gprs_called
        assert kernel["CUOccupancy"] == 1

    def test_reduced_vgpr_count_updates(self):
        """maxVgpr < pool → setGprs called and CUOccupancy corrected."""
        kw = _make_max_vgpr_writer(vgpr_size=256, agpr_size=256, sgpr_size=64)
        mkb = _MockMkb()
        kernel = {"CUOccupancy": 1, "NumThreads": 256, "LdsNumBytes": 0}
        kw.updateOccupancyFromMaxVgpr(kernel, mkb, 10)
        assert mkb.set_gprs_called
        assert mkb.set_gprs_kwargs["totalVgprs"] == 10
        assert mkb.set_gprs_kwargs["totalAgprs"] == 256
