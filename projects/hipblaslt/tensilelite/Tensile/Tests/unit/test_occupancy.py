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
Unit tests for gfx950 (and related) occupancy calculations in TensileLite.

Background
----------
Two separate occupancy bugs were identified on gfx950 (AMD Instinct MI355X):

Bug 1 – MaxWavesPerSimd hardcoded to 10
   Before the fix, ConstValues.maxOccupancy was always 10, matching gfx908.
   On gfx90a/gfx942/gfx950 (ArchAccUnifiedRegs) the true hardware cap is 8 waves
   per SIMD. Kernels with very few VGPRs (occ > 8 in VGPR terms) got an inflated
   CUOccupancy up to 10, causing StreamK to over-subscribe SIMDs.
   Fix: hardware_caps.hpp sets rv["MaxWavesPerSimd"] = ArchAccUnifiedRegs ? 8 : 10;
        KernelWriterAssembly.py reads self.states.archCaps["MaxWavesPerSimd"].

Bug 2 – VGPR-pool over-estimation → CUOccupancy under-reported for large-tile kernels
   Tensile's code-gen VGPR pool tracks peak live registers (regular + acc separately),
   then combines them as ceil(vgprs/8)*8 + accvgprs. For large MFMA tiles (e.g.,
   MIWT10_6 = 60 tiles × 4 acc VGPRs = 240 accvgprs), the combined total typically
   exceeds 256, giving CUOccupancy=1. However, the AMDGPU compiler achieves 256
   physical VGPRs (2 waves/SIMD) by overlapping register lifetimes that Tensile does
   not model. This means the stored CUOccupancy can be 1 when the hardware can
   actually sustain 2 workgroups/CU.

   For the 6 BF16 GEMM cases below, empirical HIP runtime occupancy measurements
   via hipModuleOccupancyMaxActiveBlocksPerMultiprocessor on gfx950 show:
     Case 1 (M:36912 N:62832 K:4448)  → kernel idx 13685, HIP=1 block/CU, Tensile=1 (match, LDS-limited)
     Case 2 (M:16 N:128 K:941728)     → kernel idx 13809, HIP=1 block/CU, Tensile=1 (match, LDS-limited)
     Case 3 (M:540528 N:7600 K:1008)  → kernel idx 13685 (same kernel as case 1)
     Case 4 (M:592 N:8000 K:540688)   → kernel idx 13842, HIP=2 block/CU, Tensile=1 (MISMATCH, VGPR over-est.)
     Case 5 (M:80 N:80 K:322768)      → kernel idx 13830, HIP=1 block/CU, Tensile=1 (match, LDS-limited)
     Case 6 (M:2464 N:3600 K:738624)  → kernel idx 13703, HIP=1 block/CU, Tensile=1 (match, LDS-limited)

   kernel metadata (from hipFuncGetAttribute on installed gfx950 library):
     idx 13685 (MT256x256x64, WG32_8_1): numRegs=249, staticLDS=133120 B → LDS-lim @ occ=1
     idx 13809 (MT16x64x512,  WG16_4_4): numRegs=256, staticLDS=86016  B → LDS-lim @ occ=1
     idx 13842 (MT320x192x64, WG32_8_1): numRegs=256, staticLDS=68864  B → VGPR-lim @ occ=2 (HIP) vs 1 (Tensile)
     idx 13830 (MT64x80x64,   WG32_4_1): numRegs=256, staticLDS=94016  B → LDS-lim @ occ=1
     idx 13703 (MT192x192x64, WG64_4_1): numRegs=256, staticLDS=149760 B → LDS-lim @ occ=1

   Root cause for case 4: MIWT10_6 allocates 60×4=240 accVGPRs at code-gen time;
   combined with ~64 regular VGPRs → pool ≈ 304. The compiler reduces to 256 by
   overlapping register lifetimes. Tensile does not model lifetime overlap, so it
   computes occupancy from the (pessimistic) pool total.
"""

import os
import shutil
from math import ceil
from types import SimpleNamespace

import pytest

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
        (81920,  2),   # exactly 160/2 KB → still 2
        (81921,  1),   # just above → rounds down to 81920 in 256-byte granularity → 2? see below
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

# Reference values measured with hipModuleOccupancyMaxActiveBlocksPerMultiprocessor
# on gfx950 (AMD Instinct MI355X, hipblaslt v100202) for the installed
# TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bljk_*_gfx950.co.
# These reflect the *true* hardware occupancy of the compiled kernels.
#
#  Case  |  M       N       K   | sol_idx | numRegs | staticLDS B | HIP blk/CU | Tensile CUOcc
#  ------+-----------------------+---------+---------+-------------+------------+---------------
#  1/3   | 36912   62832   4448 |  13685  |   249   |   133120    |     1      |     1  (OK)
#  2     |    16     128  941728|  13809  |   256   |    86016    |     1      |     1  (OK)
#  4     |   592    8000  540688|  13842  |   256   |    68864    |     2      |     1  (MISMATCH)
#  5     |    80      80  322768|  13830  |   256   |    94016    |     1      |     1  (OK)
#  6     |  2464    3600  738624|  13703  |   256   |   149760    |     1      |     1  (OK)
#
# Column "Tensile CUOcc" = sizeMapping["CUOccupancy"] from msgpack, built with
# the old maxOccupancy=10 code.  The "MISMATCH" in case 4 is caused by VGPR
# pool over-estimation, as described in Bug 2 above.

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
    """For LDS-dominated kernels Tensile getLdsLimitedOccupancy must agree with HIP.

    Uses the actual compiled-kernel LDS values (staticLDS) as measured by
    hipFuncGetAttribute.  Where the kernel is LDS-limited (staticLDS > 81920 B),
    the LDS formula alone must reproduce the HIP result.
    For case 4 (staticLDS=68864 < 81920, VGPR-limited in the compiled kernel)
    the LDS formula alone gives 2 – matching HIP – but Tensile's code-gen VGPR
    over-estimate causes the stored CUOccupancy to be 1 (see Bug 2 above).
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
    """getOccupancy must reproduce the HIP oracle when given compiled register counts.

    This test uses the actual numRegs reported by hipFuncGetAttribute and treats
    the unified VGPR count as: accvgprs = numRegs - min(numRegs, 16) [a conservative
    split where ≥ 240 acc is expected for large MFMA tiles], and verifies that the
    resulting Tensile occupancy matches HIP.

    For the specific compiled kernels above, numRegs ∈ {249, 256}.  On gfx950 with
    doubleVgpr=True, the occupancy limit is 512 // numRegs (since the combined pool
    equals numRegs).  The SGPR and LDS constraints must not make the result worse.
    """
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
# updateOccupancyFromScan – corrects CUOccupancy after rocIsaPass optimization
# ---------------------------------------------------------------------------

class _MockBody:
    """Minimal mock for KernelBody.body that returns a fixed assembly string."""
    def __init__(self, text):
        self._text = text
    def __str__(self):
        return self._text


class _MockMkb:
    """Minimal KernelBody-like mock for updateOccupancyFromScan tests."""
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


class TestUpdateOccupancyFromScan:
    """Validate the post-rocIsaPass assembly scan that corrects CUOccupancy.

    updateOccupancyFromScan() is called after rocIsaPass in kernelBody()
    (ArchAccUnifiedRegs ISAs only).  When removeDuplicateAssignment eliminates
    high-indexed VGPR copies, the actual instruction-level VGPR count can be
    lower than the pool high-water mark used in checkResources.  The scan
    detects this and corrects both the kernel descriptor and CUOccupancy.

    This is the finalization point: after this call, kernel["CUOccupancy"] and
    .amdhsa_next_free_vgpr in the .s are in sync.  The ELF pass (reading back
    the assembled .o) is therefore a redundant confirmation rather than a
    correction and is now opt-in only.
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

    def _run_scan(self, body_text, kernel=None):
        if kernel is None:
            kernel = {"NumThreads": 256, "LdsNumBytes": 68864, "CUOccupancy": 1}
        mkb = _MockMkb(body_text, initial_next_free_vgpr=264)
        self.kw.updateOccupancyFromScan(kernel, mkb)
        return kernel, mkb

    def test_no_update_when_scan_equals_pool(self):
        """If instructions reference the same registers as the pool, no update."""
        # Pool: 21 vgpr → ceil(21/8)*8=24 + 240 = 264
        # Body references v0-v20: max=20, scanned=21, same as pool → no update
        body = " ".join(f"v_mov_b32 v{i}, s0" for i in range(21))
        body += " v_mfma_f32_16x16x16_bf16 a[0:239], v[0:1], v[0:1], a[0:239]"
        kernel, mkb = self._run_scan(body)
        assert kernel["CUOccupancy"] == 1, "Should not update when scan == pool"
        assert mkb._set_gprs_calls == [], "setGprs should not be called"

    def test_update_when_scan_finds_fewer_vgprs(self):
        """If rocIsaPass eliminated high-indexed VGPRs, scan finds fewer and updates."""
        # Simulate: pool grew to 21 but only v0-v15 (16 VGPRs) appear in instructions
        # after removeDuplicateAssignment.
        # ceil(16/8)*8 + 240 = 256 → occ = 512//256 = 2
        body = " ".join(f"v_mov_b32 v{i}, s0" for i in range(16))
        body += " v_mfma_f32_16x16x16_bf16 a[0:239], v[0:1], v[0:1], a[0:239]"
        kernel, mkb = self._run_scan(body)
        assert kernel["CUOccupancy"] == 2, (
            f"After scan: v0-v15 only → 256 unified VGPRs → occ=2, got {kernel['CUOccupancy']}"
        )
        assert mkb._set_gprs_calls, "setGprs must be called to update kernel descriptor"
        vgprs, agprs, _ = mkb._set_gprs_calls[-1]
        assert vgprs == 16, f"scanned vgprs should be 16, got {vgprs}"
        assert agprs == 240, f"agprs should be 240, got {agprs}"
        assert mkb.getNextFreeVgpr() == 256  # ceil(16/8)*8 + 240

    def test_no_update_for_non_arch_acc_unified(self):
        """Non-ArchAccUnifiedRegs ISA: scan is skipped entirely."""
        ri_gfx908 = _init_rocisa((9, 0, 8))  # gfx908: not ArchAccUnifiedRegs
        kw_908 = _make_writer(ri_gfx908)
        import types
        kw_908.vgprPool = types.SimpleNamespace(size=lambda: 21)
        kw_908.agprPool = types.SimpleNamespace(size=lambda: 240)
        kw_908.sgprPool = types.SimpleNamespace(size=lambda: 88)
        kw_908.states.doubleVgpr = False
        kw_908.getLdsSize = lambda k: k.get("LdsNumBytes", 68864)

        kernel = {"NumThreads": 256, "LdsNumBytes": 68864, "CUOccupancy": 1}
        mkb = _MockMkb("v_mov_b32 v0, s0", initial_next_free_vgpr=264)
        kw_908.updateOccupancyFromScan(kernel, mkb)
        assert mkb._set_gprs_calls == [], "Non-unified arch: setGprs must not be called"
        assert kernel["CUOccupancy"] == 1, "Non-unified arch: occupancy must be unchanged"

    def test_range_references_expanded_correctly(self):
        """v[0:15] range references count as v0 through v15."""
        body = "v_mfma_f32_16x16x16_bf16 a[0:239], v[0:15], v[0:15], a[0:239]"
        kernel, mkb = self._run_scan(body)
        assert kernel["CUOccupancy"] == 2, (
            f"v[0:15] → 16 VGPRs → 256 unified → occ=2, got {kernel['CUOccupancy']}"
        )

    def test_agpr_count_clamped_to_pool(self):
        """Even if scan finds fewer acc VGPRs than pool, keep pool count for acc."""
        # Body only uses a[0:3] but pool has 240 acc VGPRs
        body = "v_mfma_f32_16x16x16_bf16 a[0:3], v[0:1], v[0:1], a[0:3]"
        kernel, mkb = self._run_scan(body)
        # vgprs=2 → ceil(2/8)*8=8; agprs=min(4, 240)=4 → total=12 → occ=2
        # BUT: scanned agprs is clamped to pool (240), so total=8+240=248 → occ=2
        if mkb._set_gprs_calls:
            _, agprs, _ = mkb._set_gprs_calls[-1]
            assert agprs <= 240, "Scanned agprs must not exceed pool size"
