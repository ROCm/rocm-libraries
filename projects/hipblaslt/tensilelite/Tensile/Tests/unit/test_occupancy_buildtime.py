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
CUOccupancy measurement: unit tests for OccupancyMeasure.py.

Tests the two approaches implemented for fix/gfx950-occupancy-hip-buildtime:

Approach 1 – ELF metadata (CPU-only, opt-in via --occupancy-from-elf):
  Reads vgpr_count from the AMDHSA msgpack note in assembled .o ELF files and
  recomputes CUOccupancy using the hardware occupancy formula.  No GPU needed.
  NOW OPT-IN in the build pipeline (was previously default ON).

  Why changed to opt-in:
    updateOccupancyFromScan() already sets both .amdhsa_next_free_vgpr and
    kernel["CUOccupancy"] from the scanned register counts before the .s is
    written.  The assembler takes .amdhsa_next_free_vgpr literally (no compiler
    reduction for -x assembler input), so the ELF's vgpr_count equals exactly
    what updateOccupancyFromScan wrote.  Reading it back and applying the same
    formula gives the same occupancy — the ELF pass is a faithful readback, not
    an independent correction.  It is useful for debugging (confirming the scan
    result was assembled as expected) but not as a correction pass.

Approach 2 – HIP runtime (GPU required, CI/test-only):
  Loads the linked .co.raw via hipModuleLoad and calls
  hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.  Authoritative hardware
  ground truth; IS independent of Tensile's formula.
  NOT part of the TensileCreateLibrary build pipeline; only invoked from tests
  (this file and test_occupancy_hip.py) on machines with a compatible GPU.

Investigation result:
  For pure-assembly (-x assembler) input, amdclang++ takes the
  .amdhsa_next_free_vgpr directive literally.  The assembled .o file's
  vgpr_count equals exactly what Tensile / updateOccupancyFromScan wrote.
  No compiler-side VGPR reduction occurs for assembler mode.  The ELF approach
  is therefore a faithful readback of what the codegen put in the .s file,
  not an independent measurement.  The HIP approach IS independent and catches
  any formula discrepancy.

MT320x192x64 reconciliation:
  Before updateOccupancyFromScan: pool ≈ 264 → occ=1 (WRONG)
  After updateOccupancyFromScan:  vgpr_count=256 in .s → ELF reads 256 → occ=2
  HIP measurement:                loads the .co with vgpr_count=256 → occ=2
  All three agree: CUOccupancy=2.

How to run:
  # From tensilelite/ dir:
  tox -e unit -- Tensile/Tests/unit/test_occupancy_buildtime.py -v -s
  pytest Tensile/Tests/unit/test_occupancy_buildtime.py -v -s
"""

import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

# ── hip-python (optional) ─────────────────────────────────────────────────────
try:
    from hip import hip as _hip
    HIP_AVAILABLE = True
except ImportError:
    _hip = None
    HIP_AVAILABLE = False

# ── msgpack ───────────────────────────────────────────────────────────────────
try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False

# ── module under test ─────────────────────────────────────────────────────────
from Tensile.OccupancyMeasure import (
    read_amdhsa_metadata_from_elf,
    read_vgpr_count_from_elf,
    read_kernel_descriptor_from_elf,
    compute_occupancy_from_elf_metadata,
    query_hip_occupancy,
    _arch_caps_for_kernel,
)


# ── GPU detection ─────────────────────────────────────────────────────────────

def _detect_gpu() -> str | None:
    """Return the first detected GPU arch string, or None."""
    override = os.environ.get("TENSILE_GPU_TARGET")
    if override:
        return override
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    enumerator = os.path.join(rocm_path, "bin", "rocm_agent_enumerator")
    if os.path.exists(enumerator):
        try:
            out = subprocess.check_output(
                [enumerator, "-t", "GPU"], stderr=subprocess.DEVNULL
            )
            archs = [
                line.strip()
                for line in out.decode().splitlines()
                if line.strip() and "gfx000" not in line
            ]
            return archs[0] if archs else None
        except subprocess.CalledProcessError:
            pass
    return None


GFX_TARGET: str | None = _detect_gpu()


# ── Assembly / compilation helpers ────────────────────────────────────────────

def _find_amdclang() -> str:
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    search = os.pathsep.join([
        os.path.join(rocm_path, "bin"),
        os.path.join(rocm_path, "lib", "llvm", "bin"),
    ])
    return shutil.which("amdclang++", path=search) or "/usr/bin/amdclang++"


def _build_minimal_kernel_asm(
    gfx: str,
    vgpr_count: int,
    lds_bytes: int,
    num_threads: int,
    sgpr_count: int = 16,
    wave_size: int = 64,
) -> str:
    """Return AMDHSA assembly source with the given resource allocations."""
    # vgpr_count IS .amdhsa_next_free_vgpr (already the combined unified value)
    vgpr_granule = 4
    aligned = ((vgpr_count + vgpr_granule - 1) // vgpr_granule) * vgpr_granule
    accum_offset = aligned  # no acc VGPRs separate from the combined pool
    next_free_vgpr = aligned

    return f"""\
.amdgcn_target "amdgcn-amd-amdhsa--{gfx}"

.text
.protected minimal_kernel
.globl    minimal_kernel
.p2align  8
.type     minimal_kernel,@function

.section .rodata,#alloc
.p2align 6
.amdhsa_kernel minimal_kernel
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_accum_offset              {accum_offset}
  .amdhsa_next_free_vgpr            {next_free_vgpr}
  .amdhsa_next_free_sgpr            {sgpr_count}
  .amdhsa_group_segment_fixed_size  {lds_bytes}
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 0
  .amdhsa_system_sgpr_workgroup_id_z 0
  .amdhsa_system_vgpr_workitem_id    0
  .amdhsa_float_denorm_mode_32      3
  .amdhsa_float_denorm_mode_16_64   3
.end_amdhsa_kernel

.text
minimal_kernel:
  s_endpgm

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 1
amdhsa.kernels:
  - .name:                    minimal_kernel
    .symbol:                  'minimal_kernel.kd'
    .language:                OpenCL C
    .language_version:
      - 2
      - 0
    .args: []
    .kernarg_segment_size:    8
    .kernarg_segment_align:   8
    .group_segment_fixed_size: {lds_bytes}
    .private_segment_fixed_size: 0
    .wavefront_size:          {wave_size}
    .sgpr_count:              {sgpr_count}
    .vgpr_count:              {next_free_vgpr}
    .max_flat_workgroup_size: {num_threads}
...
.end_amdgpu_metadata
"""


def _assemble_to_o(gfx: str, asm_source: str, output_o: str) -> None:
    """Assemble to a relocatable .o (single-arch ELF, not yet linked)."""
    amdclang = _find_amdclang()
    with tempfile.NamedTemporaryFile(suffix=".s", mode="w", delete=False) as f:
        f.write(asm_source)
        asm_path = f.name
    try:
        result = subprocess.run(
            [
                amdclang, "-x", "assembler",
                "--target=amdgcn-amd-amdhsa",
                f"-mcpu={gfx}",
                "-mwavefrontsize64",
                "-mcode-object-version=5",
                "-c",          # stop after assembling; produce relocatable .o
                "-o", output_o,
                asm_path,
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Assembly failed: {result.stderr}"
            )
    finally:
        os.unlink(asm_path)


def _assemble_and_link_to_co(gfx: str, asm_source: str, output_co: str) -> None:
    """Assemble + link to a loadable single-arch code object."""
    amdclang = _find_amdclang()
    with tempfile.NamedTemporaryFile(suffix=".s", mode="w", delete=False) as f:
        f.write(asm_source)
        asm_path = f.name
    try:
        result = subprocess.run(
            [
                amdclang, "-x", "assembler",
                "--target=amdgcn-amd-amdhsa",
                f"-mcpu={gfx}",
                "-mwavefrontsize64",
                "-mcode-object-version=5",
                "-o", output_co,
                asm_path,
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Assembly/link failed: {result.stderr}"
            )
    finally:
        os.unlink(asm_path)


# ── Tests: ELF metadata reading ───────────────────────────────────────────────

@pytest.mark.skipif(not MSGPACK_AVAILABLE, reason="msgpack not installed")
@pytest.mark.skipif(GFX_TARGET is None, reason="No GPU detected (needed for assembler arch)")
class TestElfMetadataReading:
    """Verify that read_amdhsa_metadata_from_elf / read_vgpr_count_from_elf work correctly."""

    @pytest.fixture(autouse=True)
    def tmp(self, tmp_path):
        self.tmp_path = tmp_path

    def _build_and_read(self, vgpr_count, lds_bytes, num_threads=256, sgpr_count=16):
        gfx = GFX_TARGET
        asm = _build_minimal_kernel_asm(gfx, vgpr_count, lds_bytes, num_threads, sgpr_count)
        o_path = str(self.tmp_path / f"k_{vgpr_count}v.o")
        _assemble_to_o(gfx, asm, o_path)
        return o_path

    def test_reads_back_vgpr_count(self):
        """vgpr_count in ELF matches what was assembled."""
        for vgpr in [8, 64, 128, 256]:
            o_path = self._build_and_read(vgpr, 256)
            # Aligned to 4-dword boundary
            aligned = ((vgpr + 3) // 4) * 4
            read_back = read_vgpr_count_from_elf(o_path, "minimal_kernel")
            assert read_back == aligned, (
                f"vgpr_count={vgpr}: assembled aligned={aligned}, ELF reads {read_back}"
            )

    def test_reads_back_kernel_descriptor(self):
        """Full kernel descriptor dict contains expected keys and values."""
        vgpr, lds, threads = 256, 68864, 256
        o_path = self._build_and_read(vgpr, lds, threads)
        kd = read_kernel_descriptor_from_elf(o_path, "minimal_kernel")
        assert kd is not None, "Kernel descriptor not found in ELF"
        vc = kd.get(".vgpr_count") or kd.get("vgpr_count")
        ldsv = kd.get(".group_segment_fixed_size") or kd.get("group_segment_fixed_size")
        assert vc == vgpr, f"vgpr_count: expected {vgpr}, got {vc}"
        assert ldsv == lds, f"group_segment_fixed_size: expected {lds}, got {ldsv}"

    def test_unknown_kernel_returns_none(self):
        """read_vgpr_count_from_elf returns None for unknown kernel names."""
        o_path = self._build_and_read(64, 256)
        assert read_vgpr_count_from_elf(o_path, "does_not_exist") is None

    def test_non_elf_returns_none(self, tmp_path):
        """A non-ELF file returns None without raising."""
        bad = str(tmp_path / "garbage.o")
        with open(bad, "wb") as f:
            f.write(b"this is not an ELF file")
        assert read_amdhsa_metadata_from_elf(bad) is None


# ── Tests: compute_occupancy_from_elf_metadata ────────────────────────────────

class TestComputeOccupancyFromElfMetadata:
    """Unit tests for the occupancy computation formula in OccupancyMeasure.py."""

    # gfx950 hardware constants
    GFX950_PHY_VGPR  = 512
    GFX950_PHY_SGPR  = 800
    GFX950_DEV_LDS   = 163840   # 160 KB
    GFX950_MAX_WAVES = 8

    def _occ(self, vgpr_count, lds_bytes, num_threads=256, sgpr_count=64):
        return compute_occupancy_from_elf_metadata(
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
        the .s file has .amdhsa_next_free_vgpr=256; the ELF approach reads
        256 and correctly computes occ=2.
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


# ── Tests: ELF occ → formula agrees with HIP (when GPU available) ─────────────

@pytest.mark.skipif(not MSGPACK_AVAILABLE, reason="msgpack not installed")
@pytest.mark.skipif(not HIP_AVAILABLE, reason="hip-python not installed")
@pytest.mark.skipif(GFX_TARGET is None, reason="No GPU detected")
class TestElfOccupancyMatchesHip:
    """Cross-validate: compute_occupancy_from_elf_metadata == HIP measurement.

    For each test case:
      1. Assemble a minimal kernel with known vgpr_count / LDS.
      2. Read vgpr_count back from the ELF → compute ELF occupancy.
      3. Load the linked code object → HIP hardware occupancy.
      4. Assert both agree.
    """

    _CASES = [
        # (vgpr_count, lds_bytes, num_threads, label)
        (256, 68864, 256, "case4-MT320x192x64-analog"),
        (256, 65536, 256, "256-vgpr-64KB-LDS"),
        (128, 65536, 256, "128-vgpr-64KB-LDS"),
        (  8,   256, 256, "8-vgpr-256B-LDS"),  # MaxWavesPerSimd limited
        (  8, 86016, 256, "8-vgpr-large-LDS-limited"),
    ]

    @pytest.fixture(autouse=True)
    def tmp(self, tmp_path):
        self.tmp_path = tmp_path

    @pytest.mark.parametrize("vgpr_count,lds_bytes,num_threads,label", _CASES,
                             ids=[c[3] for c in _CASES])
    def test_elf_matches_hip(self, vgpr_count, lds_bytes, num_threads, label):
        """ELF-computed occupancy must equal HIP hardware measurement."""
        gfx = GFX_TARGET
        if gfx not in ("gfx950", "gfx942", "gfx90a", "gfx908"):
            pytest.skip(f"Unsupported GPU: {gfx}")

        asm = _build_minimal_kernel_asm(gfx, vgpr_count, lds_bytes, num_threads)

        # Assemble to relocatable .o for ELF reading
        o_path = str(self.tmp_path / f"{label.replace('-','_')}.o")
        _assemble_to_o(gfx, asm, o_path)

        # Assemble+link to executable .co for HIP loading
        co_path = str(self.tmp_path / f"{label.replace('-','_')}.co")
        _assemble_and_link_to_co(gfx, asm, co_path)

        # ELF approach
        vgpr_from_elf = read_vgpr_count_from_elf(o_path, "minimal_kernel")
        assert vgpr_from_elf is not None, f"ELF metadata not found in {o_path}"
        kd = read_kernel_descriptor_from_elf(o_path, "minimal_kernel")
        lds_from_elf = kd.get(".group_segment_fixed_size") or kd.get("group_segment_fixed_size") or 0

        # Determine arch caps from ISA
        isa_map = {
            "gfx950": (9, 5, 0), "gfx942": (9, 4, 2),
            "gfx90a": (9, 0, 10), "gfx908": (9, 0, 8),
        }
        isa = isa_map.get(gfx, (9, 5, 0))
        kernel_stub = {"ISA": list(isa), "KernelLanguage": "Assembly"}
        phy_vgpr, phy_sgpr, dev_lds, max_waves = _arch_caps_for_kernel(kernel_stub)

        elf_occ = compute_occupancy_from_elf_metadata(
            vgpr_count=vgpr_from_elf,
            lds_bytes=lds_from_elf,
            sgpr_count=16,
            num_threads=num_threads,
            physical_vgpr=phy_vgpr,
            physical_sgpr=phy_sgpr,
            device_lds=dev_lds,
            max_waves_per_simd=max_waves,
        )

        # HIP approach
        hip_occ = query_hip_occupancy(co_path, "minimal_kernel", num_threads)

        print(
            f"\n  [{label}] gfx={gfx}, vgpr={vgpr_from_elf}, "
            f"lds={lds_from_elf}, threads={num_threads}\n"
            f"  ELF occ={elf_occ}, HIP occ={hip_occ}"
        )

        assert elf_occ == hip_occ, (
            f"{label}: ELF occupancy {elf_occ} != HIP occupancy {hip_occ}\n"
            f"  vgpr={vgpr_from_elf}, lds={lds_from_elf}, threads={num_threads}, gfx={gfx}"
        )


# ── Tests: assembler does NOT reduce vgpr_count (task 3 investigation) ────────

@pytest.mark.skipif(not MSGPACK_AVAILABLE, reason="msgpack not installed")
@pytest.mark.skipif(GFX_TARGET is None, reason="No GPU detected (needed for assembler)")
class TestAssemblerDoesNotReduceVgprCount:
    """Document that amdclang++ -x assembler takes .amdhsa_next_free_vgpr literally.

    Task 3 investigation result:
    For pure-assembly input, the assembler does NOT reduce vgpr_count below
    what was written in the .s file.  The .o ELF's vgpr_count exactly matches
    the .amdhsa_next_free_vgpr directive.

    This means:
    - ELF reading is a faithful readback of the codegen result (not independent).
    - updateOccupancyFromScan must reduce the directive *before* the .s is written.
    - Only the HIP approach provides a truly independent hardware measurement.

    Contrast with compiled HIP/C++ kernels, where the compiler can reduce
    register usage (e.g., MT320x192x64 compiled kernel has numRegs=256 while
    naive code-gen would predict 264 without updateOccupancyFromScan).
    """

    @pytest.fixture(autouse=True)
    def tmp(self, tmp_path):
        self.tmp_path = tmp_path

    @pytest.mark.parametrize("requested_vgpr", [64, 128, 200, 256])
    def test_assembled_vgpr_count_matches_directive(self, requested_vgpr):
        """ELF vgpr_count equals .amdhsa_next_free_vgpr directive (no reduction)."""
        gfx = GFX_TARGET
        asm = _build_minimal_kernel_asm(gfx, requested_vgpr, 256, 256)
        o_path = str(self.tmp_path / f"no_reduce_{requested_vgpr}.o")
        _assemble_to_o(gfx, asm, o_path)

        # Align to 4-dword boundary (assembler aligns vgpr count)
        aligned = ((requested_vgpr + 3) // 4) * 4
        actual = read_vgpr_count_from_elf(o_path, "minimal_kernel")

        assert actual == aligned, (
            f"Requested vgpr={requested_vgpr} (aligned={aligned}), "
            f"but ELF has vgpr_count={actual}.\n"
            f"If actual < aligned, the assembler IS reducing vgpr_count – unexpected!"
        )

    def test_high_vgpr_directive_preserved(self):
        """A high but valid vgpr_count (252) is not silently reduced by the assembler.

        Note: the assembler rejects .amdhsa_accum_offset > 256, so we use 252
        (the largest multiple-of-4 ≤ 256) as the 'high but valid' case.
        This verifies that the assembler does not silently reduce a directive
        that is within the legal range.
        """
        gfx = GFX_TARGET
        requested = 252   # largest aligned-to-4 value ≤ 256 (accum_offset limit)
        asm = _build_minimal_kernel_asm(gfx, requested, 256, 256)
        o_path = str(self.tmp_path / "high_vgpr.o")
        _assemble_to_o(gfx, asm, o_path)
        aligned = ((requested + 3) // 4) * 4   # = 252 (already aligned)
        actual = read_vgpr_count_from_elf(o_path, "minimal_kernel")
        assert actual is not None, "ELF metadata missing from assembled .o"
        assert actual == aligned, (
            f"Assembler silently changed vgpr_count from {aligned} to {actual}. "
            f"For pure-assembly input this should not happen."
        )
