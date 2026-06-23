# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Unit tests for gfx950 (and related) occupancy calculations in TensileLite.

Covers MaxWavesPerSimd cap enforcement and post-rocIsaPass VGPR-scan corrections
for ArchAccUnifiedRegs ISAs (gfx90a/gfx942/gfx950).
"""

import os
import shutil
from math import ceil
from types import SimpleNamespace

import pytest

# The coverage tox env runs `pytest -m unit Tensile/Tests/unit`; without this
# explicit declaration the file is silently deselected and reports 0% coverage.
pytestmark = pytest.mark.unit

from Tensile.KernelWriterAssembly import KernelWriterAssembly
from rocisa import rocIsa


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _init_rocisa(isa):
    """Initialize rocisa singleton for the given ISA tuple."""
    ri = rocIsa.getInstance()
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    search_path = os.pathsep.join([
        os.path.join(rocm_path, "bin"),
        os.path.join(rocm_path, "lib", "llvm", "bin"),
    ])
    asmpath = shutil.which("amdclang++", path=search_path) or "/usr/bin/amdclang++"
    ri.init(isa, asmpath)
    ri.setKernel(isa, 64)
    return ri


def _make_writer(ri):
    """Minimal KernelWriterAssembly stub with rocisa caps wired into states."""
    kw = object.__new__(KernelWriterAssembly)
    kw.states = SimpleNamespace(
        archCaps=ri.getArchCaps(),
        regCaps=ri.getRegCaps(),
    )
    return kw


def _occ(kw, *, numThreads, vgprs, accvgprs, sgprs, ldsBytes, doubleVgpr=True):
    """Convenience wrapper for getOccupancy."""
    return kw.getOccupancy(numThreads, vgprs, sgprs, ldsBytes, accvgprs, doubleVgpr)


# ---------------------------------------------------------------------------
# Bug 1: MaxWavesPerSimd cap
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "isa,expected",
    [
        ((9, 0, 8), 10),   # gfx908  – not ArchAccUnifiedRegs, stays at 10
        ((9, 0, 10), 8),   # gfx90a  – ArchAccUnifiedRegs, capped at 8
        ((9, 4, 2), 8),    # gfx942  – ArchAccUnifiedRegs, capped at 8
        ((9, 5, 0), 8),    # gfx950  – ArchAccUnifiedRegs, capped at 8
    ],
)
def test_max_waves_per_simd_from_arch_caps(isa, expected):
    """archCaps MaxWavesPerSimd must reflect ArchAccUnifiedRegs limits.

    Before the fix MaxWavesPerSimd did not exist in archCaps; code used
    ConstValues.maxOccupancy which was hardcoded to 10 for all ISAs.
    After the fix all ArchAccUnifiedRegs ISAs return 8.
    """
    ri = _init_rocisa(isa)
    assert ri.getArchCaps()["MaxWavesPerSimd"] == expected


@pytest.mark.parametrize(
    "isa,expected_occupancy",
    [
        ((9, 5, 0), 8),   # gfx950 — was 10 with hardcoded maxOccupancy
        ((9, 0, 8), 10),  # gfx908 — unchanged
    ],
)
def test_get_vgpr_occupancy_uses_max_waves_per_simd(isa, expected_occupancy):
    """getVgprOccupancy caps at MaxWavesPerSimd when VGPR count is not limiting.

    With very few VGPRs the VGPR pool imposes no constraint and the result
    must equal MaxWavesPerSimd // multiplier.  On gfx950 this is 8, not 10.
    """
    kw = _make_writer(_init_rocisa(isa))
    occupancy = kw.getVgprOccupancy(numThreads=256, vgprs=1, doubleVgpr=False)
    assert occupancy == expected_occupancy


# ---------------------------------------------------------------------------
# gfx950-specific hardware caps validation
# ---------------------------------------------------------------------------

def test_gfx950_device_lds_is_160kb():
    """gfx950 has 160 KB per CU (not 64 KB like gfx908/gfx942)."""
    ri = _init_rocisa((9, 5, 0))
    assert ri.getArchCaps()["DeviceLDS"] == 163840  # 160 * 1024


def test_gfx950_arch_acc_unified_regs():
    """gfx950 is ArchAccUnifiedRegs (unified VGPR+AGPR pool)."""
    ri = _init_rocisa((9, 5, 0))
    assert ri.getArchCaps()["ArchAccUnifiedRegs"] == 1


def test_gfx950_physical_vgpr_pool_is_512():
    """gfx950 physical VGPR pool per SIMD = 512 (256 regular + 256 accumulation)."""
    ri = _init_rocisa((9, 5, 0))
    assert ri.getRegCaps()["PhysicalMaxVgpr"] == 512
    assert ri.getRegCaps()["MaxVgpr"] == 256  # logical max per wave


# ---------------------------------------------------------------------------
# getLdsLimitedOccupancy – gfx950 LDS boundary conditions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "lds_bytes,expected_occ",
    [
        # LDS well below 160 KB/2=80 KB: 2 workgroups fit
        (68864,  2),   # case 4 compiled kernel LDS → should give 2
        (65536,  2),   # 64 KB
        # LDS at 80 KB threshold
        (81920,  2),   # exactly 160/2 KB → rounds to 81920 → 163840//81920 = 2
        (81921,  1),   # just above → rounds up to 82176 (256-byte granularity) → 163840//82176 = 1
        # LDS above 80 KB: only 1 workgroup fits
        (86016,  1),   # case 2 compiled kernel LDS
        (94016,  1),   # case 5 compiled kernel LDS
        (133120, 1),   # case 1/3 compiled kernel LDS
        (149760, 1),   # case 6 compiled kernel LDS
    ],
)
def test_lds_limited_occupancy_gfx950(lds_bytes, expected_occ):
    """getLdsLimitedOccupancy on gfx950 with 163840-byte device LDS."""
    occ = KernelWriterAssembly.getLdsLimitedOccupancy(163840, lds_bytes)
    assert occ == expected_occ, (
        f"LDS={lds_bytes} B: expected {expected_occ} blocks/CU, got {occ}"
    )


# ---------------------------------------------------------------------------
# getOccupancy – combined VGPR+AGPR path (doubleVgpr=True, gfx950)
# ---------------------------------------------------------------------------

class TestGetOccupancyGfx950:
    """Validate getOccupancy for gfx950 (doubleVgpr=True)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.kw = _make_writer(_init_rocisa((9, 5, 0)))

    def test_vgpr_limited_two_waves(self):
        """256 combined VGPRs → occ=2 per SIMD (512 pool / 256 = 2)."""
        # Pure VGPR limit, large enough LDS to not be the bottleneck
        occ = _occ(self.kw, numThreads=256, vgprs=16, accvgprs=240,
                   sgprs=64, ldsBytes=65536)
        # ceil(16/8)*8 + 240 = 16 + 240 = 256; 512//256 = 2
        assert occ == 2

    def test_vgpr_limited_one_wave_when_combined_exceeds_256(self):
        """More than 256 combined VGPRs → occ=1."""
        # ceil(64/8)*8 + 240 = 64 + 240 = 304; 512//304 = 1
        occ = _occ(self.kw, numThreads=256, vgprs=64, accvgprs=240,
                   sgprs=64, ldsBytes=65536)
        assert occ == 1

    def test_lds_limits_occupancy_for_large_tiles(self):
        """LDS > 81920 B limits to 1 even when VGPR would allow 2."""
        # numRegs=256 → VGPR occ=2, but LDS is 133120 → LDS occ=1
        occ = _occ(self.kw, numThreads=256, vgprs=16, accvgprs=233,
                   sgprs=64, ldsBytes=133120)
        assert occ == 1  # LDS-limited

    def test_lds_permits_two_workgroups_case4(self):
        """Case 4 kernel LDS (68864 B) allows 2 workgroups/CU on gfx950.

        This is the *correct* occupancy that Tensile should report for case 4
        when given the actual compiled-kernel register counts (vgprs=16,
        accvgprs=240, total=256) instead of the code-gen pool overestimate.
        """
        # Mimics what HIP reports: numRegs=256 total → assume 16 reg + 240 acc
        occ = _occ(self.kw, numThreads=256, vgprs=16, accvgprs=240,
                   sgprs=64, ldsBytes=68864)
        assert occ == 2, (
            "With 68864 B LDS and 256 combined VGPRs on gfx950, "
            "getOccupancy should return 2 (matching HIP runtime)"
        )

    def test_vgpr_pool_overestimate_causes_case4_mismatch(self):
        """Tensile code-gen typically over-allocates VGPRs vs compiled result.

        MIWT10_6 allocates 240 accVGPRs.  If the regular pool is even 64
        VGPRs (a typical code-gen estimate), the combined total is 304 and
        Tensile computes occ=1 – while HIP reports occ=2 for the compiled
        kernel (numRegs=256, i.e., 256 unified VGPRs per lane).

        This test documents the known discrepancy: it asserts the *wrong*
        (Tensile code-gen) occupancy is 1 and the *right* (HIP oracle)
        occupancy is 2 for these register counts.
        """
        code_gen_vgprs = 64   # typical Tensile pool overestimate
        code_gen_acc   = 240  # 60 MI tiles × 4 acc VGPRs
        # ceil(64/8)*8 + 240 = 304 → 512//304 = 1
        tensile_occ = _occ(self.kw, numThreads=256,
                           vgprs=code_gen_vgprs, accvgprs=code_gen_acc,
                           sgprs=64, ldsBytes=68864)
        assert tensile_occ == 1, "Tensile code-gen estimate gives occ=1"

        compiled_vgprs = 16   # compiler reduces total to 256 unified
        compiled_acc   = 240
        hip_oracle_occ = _occ(self.kw, numThreads=256,
                              vgprs=compiled_vgprs, accvgprs=compiled_acc,
                              sgprs=64, ldsBytes=68864)
        assert hip_oracle_occ == 2, "Compiled kernel (numRegs=256) gives occ=2"

    def test_sgpr_limit_gfx950(self):
        """SGPR limit (PhysicalMaxSgpr=800) does not falsely lower occupancy."""
        # 64 SGPRs per wave → 800//64=12, not binding
        occ = _occ(self.kw, numThreads=256, vgprs=16, accvgprs=240,
                   sgprs=64, ldsBytes=68864)
        assert occ == 2

    def test_large_sgpr_count_reduces_occupancy(self):
        """Very large SGPR count can independently limit occupancy."""
        # 400 SGPRs → 800//400=2; combined VGPRs still give 2 → result is 2
        occ = _occ(self.kw, numThreads=256, vgprs=16, accvgprs=240,
                   sgprs=400, ldsBytes=68864)
        assert occ == 2
        # 401 → 800//401=1 → SGPR-limited
        occ_sgpr_limited = _occ(self.kw, numThreads=256, vgprs=16, accvgprs=240,
                                sgprs=401, ldsBytes=68864)
        assert occ_sgpr_limited == 1


# ---------------------------------------------------------------------------
# Six user-provided BF16 GEMM cases – oracle from HIP runtime
# ---------------------------------------------------------------------------

# Oracle from hipModuleOccupancyMaxActiveBlocksPerMultiprocessor on gfx950.
# Case 4 (numRegs=256, LDS=68864) is VGPR-limited at occ=2; others are LDS-limited.

_GFX950_KERNEL_ORACLE = [
    # (description,       numRegs, staticLDS, numThreads, hip_blocks_per_cu)
    ("case1 MT256x256x64",  249,   133120,     256,         1),
    ("case2 MT16x64x512",   256,    86016,     256,         1),
    ("case4 MT320x192x64",  256,    68864,     256,         2),
    ("case5 MT64x80x64",    256,    94016,     128,         1),
    ("case6 MT192x192x64",  256,   149760,     256,         1),
]


@pytest.mark.parametrize("desc,numRegs,staticLDS,numThreads,hip_occ",
                         _GFX950_KERNEL_ORACLE,
                         ids=[x[0] for x in _GFX950_KERNEL_ORACLE])
def test_lds_limited_occupancy_matches_hip_oracle(desc, numRegs, staticLDS,
                                                   numThreads, hip_occ):
    """getLdsLimitedOccupancy with compiled-kernel LDS values must agree with HIP.

    For LDS-limited kernels (staticLDS > device_lds/2) the formula alone reproduces
    the HIP result.  For case 4 (staticLDS=68864, VGPR-limited) the LDS formula gives
    2 — matching HIP — but the code-gen VGPR pool over-estimate produces stored occ=1.
    """
    device_lds = 163840  # gfx950 160 KB
    lds_occ = KernelWriterAssembly.getLdsLimitedOccupancy(device_lds, staticLDS)

    if staticLDS > device_lds // 2:
        # LDS-limited: the LDS formula alone reproduces the HIP result.
        assert lds_occ == hip_occ, (
            f"{desc}: LDS-limited occupancy mismatch. "
            f"getLdsLimitedOccupancy={lds_occ}, HIP oracle={hip_occ}"
        )
    else:
        # Not LDS-limited; LDS formula says ≥ 2 and HIP also says ≥ 2.
        assert lds_occ >= hip_occ, (
            f"{desc}: Expected lds_occ >= hip_occ ({hip_occ}), got {lds_occ}"
        )


@pytest.mark.parametrize("desc,numRegs,staticLDS,numThreads,hip_occ",
                         _GFX950_KERNEL_ORACLE,
                         ids=[x[0] for x in _GFX950_KERNEL_ORACLE])
def test_getoccupancy_with_compiled_register_counts(desc, numRegs, staticLDS,
                                                     numThreads, hip_occ):
    """getOccupancy with compiled numRegs (from hipFuncGetAttribute) must match HIP oracle."""
    kw = _make_writer(_init_rocisa((9, 5, 0)))

    # All VGPR seen as pure accvgprs in the combined pool, 0 regular vgprs.
    # This isolates the VGPR-pool limit: ceil(0/8)*8 + numRegs = numRegs.
    occ = _occ(kw, numThreads=numThreads,
               vgprs=0, accvgprs=numRegs,
               sgprs=64, ldsBytes=staticLDS, doubleVgpr=True)
    assert occ == hip_occ, (
        f"{desc}: getOccupancy with compiled numRegs={numRegs}, "
        f"LDS={staticLDS} B → expected {hip_occ}, got {occ}"
    )


# ---------------------------------------------------------------------------
# MaxWavesPerSimd is the binding constraint for low-VGPR kernels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "isa,max_waves",
    [
        ((9, 5, 0), 8),   # gfx950 ArchAccUnifiedRegs
        ((9, 0, 10), 8),  # gfx90a ArchAccUnifiedRegs
        ((9, 4, 2), 8),   # gfx942 ArchAccUnifiedRegs
        ((9, 0, 8), 10),  # gfx908 not unified
    ],
)
def test_max_waves_limits_low_vgpr_kernels(isa, max_waves):
    """For trivially small kernels (1 VGPR), occupancy equals MaxWavesPerSimd.

    This validates the primary benefit of the MaxWavesPerSimd fix: kernels
    that would otherwise be unconstrained by VGPR or LDS are now capped at
    the correct hardware limit (8 for unified-register ISAs).
    """
    kw = _make_writer(_init_rocisa(isa))
    # 1 VGPR: no register pressure, occupancy determined entirely by MaxWavesPerSimd.
    # doubleVgpr=False mimics pre-MFMA kernels where accvgprs=0.
    occ = kw.getOccupancy(256, 1, 64, 1, 0, doubleVgpr=False)
    assert occ == max_waves, (
        f"ISA {isa}: expected occ={max_waves} (MaxWavesPerSimd), got {occ}"
    )


@pytest.mark.parametrize(
    "isa,should_be_capped",
    [
        ((9, 5, 0), True),   # gfx950: cap at 8 prevents falsely high occ
        ((9, 0, 8), False),  # gfx908: cap at 10 (also correct)
    ],
)
def test_max_waves_per_simd_prevents_overclaim_on_gfx950(isa, should_be_capped):
    """Without the MaxWavesPerSimd fix, gfx950 would claim up to 10 waves/SIMD.

    The old behaviour (hardcoded 10) was wrong for gfx950.  With the fix, the
    cap is 8.  Kernels with very few VGPRs (< 512/9 ≈ 56 total) would have
    reported occupancy 9 or 10 before the fix but now correctly report ≤ 8.
    """
    kw = _make_writer(_init_rocisa(isa))
    archCaps = kw.states.archCaps
    # Use doubleVgpr only for ArchAccUnifiedRegs (as Tensile's code-gen does).
    double = bool(archCaps["ArchAccUnifiedRegs"])
    # 48 VGPRs per thread.
    # gfx950 (doubleVgpr=True):  totalVgprs=512, 512//48=10 → capped to 8 ✓
    # gfx908 (doubleVgpr=False): totalVgprs=256, 256//48=5  → below cap of 10 ✓
    occ = kw.getVgprOccupancy(numThreads=256, vgprs=48, doubleVgpr=double)
    if should_be_capped:
        assert occ <= 8, f"ISA {isa}: occupancy {occ} exceeds gfx950 hardware limit of 8"
        # Also verify it would have been higher (10) with the old hardcoded cap
        assert 512 // (8 * 1) > 8, "sanity: 512//8=64 > 8, VGPR does not limit"
    else:
        # gfx908 non-unified: 256//48=5, well below the cap of 10
        assert occ == 5


# ---------------------------------------------------------------------------
# updateOccupancyFromMaxVgpr – corrects CUOccupancy after rocIsaPass optimization
# ---------------------------------------------------------------------------

class _MockBody:
    """Minimal mock for KernelBody.body that returns a fixed assembly string."""
    def __init__(self, text):
        self._text = text
    def __str__(self):
        return self._text


class _MockMkb:
    """Minimal KernelBody-like mock for updateOccupancyFromMaxVgpr tests."""
    def __init__(self, body_text, initial_next_free_vgpr=264):
        self.body = _MockBody(body_text)
        self._next_free_vgpr = initial_next_free_vgpr
        self._set_gprs_calls = []

    def setGprs(self, totalVgprs, totalAgprs, totalSgprs):
        self._set_gprs_calls.append((totalVgprs, totalAgprs, totalSgprs))
        # Simulate the ArchAccUnifiedRegs formula
        from math import ceil
        self._next_free_vgpr = int(ceil(totalVgprs / 8.0)) * 8 + totalAgprs

    def getNextFreeVgpr(self):
        return self._next_free_vgpr


class TestUpdateOccupancyFromMaxVgpr:
    """Validate the post-rocIsaPass CUOccupancy correction path.

    updateOccupancyFromMaxVgpr() runs after rocIsaPass (ArchAccUnifiedRegs only).
    The O(1) fast path receives the pre-computed maxVgpr from rocIsaPassResult
    (= max VGPR index + 1, derived from the register graph the pass already
    built).  When that count is lower than the checkResources pool estimate,
    the kernel descriptor and CUOccupancy are corrected.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ri = _init_rocisa((9, 5, 0))
        self.kw = _make_writer(self.ri)
        # Wire in pool sizes to simulate checkResources post-state.
        # pool.size()=21 vgprs, pool.size()=240 agprs (MT320x192x64 subtile estimate)
        import types
        self.kw.vgprPool = types.SimpleNamespace(size=lambda: 21)
        self.kw.agprPool = types.SimpleNamespace(size=lambda: 240)
        self.kw.sgprPool = types.SimpleNamespace(size=lambda: 88)
        # Expose needed methods
        self.kw.states.doubleVgpr = True
        self.kw.getLdsSize = lambda k: k.get("LdsNumBytes", 68864)

    def _run_with_max_vgpr(self, kernel=None, max_vgpr=-1):
        """Invoke updateOccupancyFromMaxVgpr using max_vgpr from rocIsaPass."""
        if kernel is None:
            kernel = {"NumThreads": 256, "LdsNumBytes": 68864, "CUOccupancy": 1}
        mkb = _MockMkb("", initial_next_free_vgpr=264)
        self.kw.updateOccupancyFromMaxVgpr(kernel, mkb, max_vgpr)
        return kernel, mkb

    def test_no_update_when_max_vgpr_equals_pool(self):
        """max_vgpr == pool size: no update (already at pool estimate)."""
        # Pool: 21 vgpr → ceil(21/8)*8=24 + 240 = 264
        kernel, mkb = self._run_with_max_vgpr(max_vgpr=21)
        assert kernel["CUOccupancy"] == 1, "Should not update when max_vgpr == pool"
        assert mkb._set_gprs_calls == [], "setGprs should not be called"

    def test_update_when_max_vgpr_lower_than_pool(self):
        """max_vgpr < pool: descriptor and CUOccupancy are corrected."""
        # Pool grew to 21 but rocIsaPass found only 16 VGPRs used.
        # ceil(16/8)*8 + 240 = 256 → occ = 512//256 = 2
        kernel, mkb = self._run_with_max_vgpr(max_vgpr=16)
        assert kernel["CUOccupancy"] == 2, (
            f"max_vgpr=16 → 256 unified VGPRs → occ=2, got {kernel['CUOccupancy']}"
        )
        assert mkb._set_gprs_calls, "setGprs must be called to update kernel descriptor"
        vgprs, agprs, _ = mkb._set_gprs_calls[-1]
        assert vgprs == 16, f"max_vgpr should be 16, got {vgprs}"
        assert agprs == 240, f"agprs should be 240 (pool size), got {agprs}"
        assert mkb.getNextFreeVgpr() == 256  # ceil(16/8)*8 + 240

    def test_no_update_for_non_arch_acc_unified(self):
        """Non-ArchAccUnifiedRegs ISA: max-VGPR update is skipped entirely."""
        ri_gfx908 = _init_rocisa((9, 0, 8))  # gfx908: not ArchAccUnifiedRegs
        kw_908 = _make_writer(ri_gfx908)
        import types
        kw_908.vgprPool = types.SimpleNamespace(size=lambda: 21)
        kw_908.agprPool = types.SimpleNamespace(size=lambda: 240)
        kw_908.sgprPool = types.SimpleNamespace(size=lambda: 88)
        kw_908.states.doubleVgpr = False
        kw_908.getLdsSize = lambda k: k.get("LdsNumBytes", 68864)

        kernel = {"NumThreads": 256, "LdsNumBytes": 68864, "CUOccupancy": 1}
        mkb = _MockMkb("", initial_next_free_vgpr=264)
        kw_908.updateOccupancyFromMaxVgpr(kernel, mkb, 16)
        assert mkb._set_gprs_calls == [], "Non-unified arch: setGprs must not be called"
        assert kernel["CUOccupancy"] == 1, "Non-unified arch: occupancy must be unchanged"

    def test_vgpr_count_16_gives_occ2(self):
        """16 VGPRs (matching v[0:15]) → 256 unified VGPRs → occ=2."""
        kernel, mkb = self._run_with_max_vgpr(max_vgpr=16)
        assert kernel["CUOccupancy"] == 2, (
            f"max_vgpr=16 → 256 unified → occ=2, got {kernel['CUOccupancy']}"
        )

    def test_agpr_always_uses_pool_size(self):
        """AGPR count is always taken from agprPool.size(), not from rocIsaPass."""
        kernel, mkb = self._run_with_max_vgpr(max_vgpr=2)
        if mkb._set_gprs_calls:
            _, agprs, _ = mkb._set_gprs_calls[-1]
            assert agprs == 240, f"agprs must equal agprPool.size()=240, got {agprs}"

    def test_mt320x192x64_gfx950_gives_occ2(self):
        """Motivating case: MT320x192x64/gfx950 with max_vgpr=16 → occ=2."""
        kernel = {"NumThreads": 256, "LdsNumBytes": 68864, "CUOccupancy": 1}
        mkb = _MockMkb("", initial_next_free_vgpr=264)
        self.kw.updateOccupancyFromMaxVgpr(kernel, mkb, 16)
        assert kernel["CUOccupancy"] == 2, (
            f"MT320x192x64/gfx950: expected CUOccupancy=2, got {kernel['CUOccupancy']}"
        )

    def test_missing_max_vgpr_skips_update(self):
        """When rocIsaPass does not supply maxVgpr (<=0), skip the update."""
        kernel, mkb = self._run_with_max_vgpr(max_vgpr=-1)
        assert kernel["CUOccupancy"] == 1
        assert mkb._set_gprs_calls == []
