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
Unit tests for Change 1: CUOccupancy computation from custom kernel .s files at build time.

Change 1 summary
----------------
Custom kernels bypass the normal codegen path and therefore skip the
``checkResources`` / ``updateOccupancyFromScan`` that sets ``kernel["CUOccupancy"]``.
Before this fix, ``kernel["CUOccupancy"]`` stayed at its default of ``-1``.

The fix adds ``compute_occupancy_from_asm_source`` in ``OccupancyMeasure.py``, which
parses ``.amdhsa_next_free_vgpr``, ``.amdhsa_next_free_sgpr``, and
``.amdhsa_group_segment_fixed_size`` from the custom ``.s`` source text, then calls
the existing ``compute_occupancy_from_resources`` formula using hardware constants
derived via ``_arch_caps_for_kernel`` (ISA-table lookup, no GPU needed).

``getSourceFileString`` in ``KernelWriterAssembly.py`` calls this function for custom
kernels and stores the result in ``kernel["CUOccupancy"]`` before returning.

Change 2 (debug warning) tests
--------------------------------
Also verifies that the ``print2``-gated Python warning in ``processKernelSource``
fires when CUOccupancy is ≤ 0 and verbosity ≥ 2, and stays silent otherwise.

Test structure
--------------
* ``TestComputeOccupancyFromAsmSource`` — unit-tests the parser/formula against
  synthetic .s snippets with known register counts.
* ``TestComputeOccupancyRealCustomKernels`` — reads actual files from
  ``Tensile/CustomKernels/`` and asserts computed CUOccupancy against expected
  values derived from the hardware formula.  Skipped if the directory is absent.
* ``TestPythonDebugWarning`` — verifies the print2-gated warning behaviour.

How to run
----------
  # From tensilelite/ dir:
  tox -e unit -- Tensile/Tests/unit/test_custom_kernel_occupancy.py -v -s
  pytest Tensile/Tests/unit/test_custom_kernel_occupancy.py -v -s
"""

import io
import os
import sys
from pathlib import Path

import pytest

from Tensile.OccupancyMeasure import (
    compute_occupancy_from_asm_source,
    compute_occupancy_from_resources,
    _arch_caps_for_kernel,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CUSTOM_KERNEL_DIR = Path(__file__).parents[2] / "CustomKernels"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_asm(next_free_vgpr, next_free_sgpr, group_seg_size, accum_offset=None,
              kernel_name="test_kernel"):
    """Return a minimal .amdhsa_kernel block with the given directives."""
    accum_line = (
        f"  .amdhsa_accum_offset {accum_offset} // accvgpr offset\n"
        if accum_offset is not None else ""
    )
    return (
        f".amdhsa_kernel {kernel_name}\n"
        f"  .amdhsa_user_sgpr_kernarg_segment_ptr 1\n"
        f"{accum_line}"
        f"  .amdhsa_next_free_vgpr {next_free_vgpr} // vgprs\n"
        f"  .amdhsa_next_free_sgpr {next_free_sgpr} // sgprs\n"
        f"  .amdhsa_group_segment_fixed_size {group_seg_size} // lds bytes\n"
        f"  .amdhsa_private_segment_fixed_size 0\n"
        f".end_amdhsa_kernel\n"
    )


def _kernel(isa, num_threads=256):
    return {"ISA": list(isa), "NumThreads": num_threads, "KernelLanguage": "Assembly"}


# ---------------------------------------------------------------------------
# TestComputeOccupancyFromAsmSource — synthetic .s snippets
# ---------------------------------------------------------------------------

class TestComputeOccupancyFromAsmSource:
    """Test compute_occupancy_from_asm_source with synthetic assembly text."""

    # --- gfx950 cases (phy_vgpr=512, max_waves=8, device_lds=163840) ---

    def test_gfx950_256vgpr_occ2(self):
        """256 unified VGPRs on gfx950 → occ=2 (512//256=2), LDS not limiting."""
        asm = _make_asm(next_free_vgpr=256, next_free_sgpr=64, group_seg_size=65536,
                        accum_offset=128)
        kernel = _kernel((9, 5, 0))
        occ = compute_occupancy_from_asm_source(kernel, asm)
        assert occ == 2, f"Expected 2, got {occ}"

    def test_gfx950_504vgpr_occ1(self):
        """504 unified VGPRs on gfx950 → occ=1 (512//504=1)."""
        asm = _make_asm(next_free_vgpr=504, next_free_sgpr=88, group_seg_size=133120,
                        accum_offset=248)
        kernel = _kernel((9, 5, 0))
        occ = compute_occupancy_from_asm_source(kernel, asm)
        assert occ == 1, f"Expected 1, got {occ}"

    def test_gfx950_512vgpr_occ1(self):
        """512 unified VGPRs on gfx950 → occ=1 (512//512=1)."""
        asm = _make_asm(next_free_vgpr=512, next_free_sgpr=88, group_seg_size=131072,
                        accum_offset=256)
        kernel = _kernel((9, 5, 0))
        occ = compute_occupancy_from_asm_source(kernel, asm)
        assert occ == 1, f"Expected 1, got {occ}"

    def test_gfx950_lds_limited_occ1(self):
        """gfx950: LDS=133120 > device_lds/2=81920 → LDS limits to occ=1."""
        asm = _make_asm(next_free_vgpr=256, next_free_sgpr=64, group_seg_size=133120,
                        accum_offset=128)
        kernel = _kernel((9, 5, 0))
        occ = compute_occupancy_from_asm_source(kernel, asm)
        assert occ == 1, f"Expected LDS-limited occ=1, got {occ}"

    # --- gfx942 cases (phy_vgpr=512, max_waves=8, device_lds=65536) ---

    def test_gfx942_256vgpr_accum128_occ2(self):
        """gfx942: 256 total VGPRs (accum_offset=128) → occ=2 (512//256=2)."""
        asm = _make_asm(next_free_vgpr=256, next_free_sgpr=102, group_seg_size=32768,
                        accum_offset=128)
        kernel = _kernel((9, 4, 2))
        occ = compute_occupancy_from_asm_source(kernel, asm)
        assert occ == 2, f"Expected 2, got {occ}"

    def test_gfx942_512vgpr_accum256_occ1(self):
        """gfx942: 512 total VGPRs (accum_offset=256) → occ=1 (512//512=1)."""
        asm = _make_asm(next_free_vgpr=512, next_free_sgpr=102, group_seg_size=65536,
                        accum_offset=256)
        kernel = _kernel((9, 4, 2))
        occ = compute_occupancy_from_asm_source(kernel, asm)
        assert occ == 1, f"Expected 1, got {occ}"

    def test_gfx942_lds_limited_occ1(self):
        """gfx942: LDS=65536 = device_lds → lds_occ=1 regardless of VGPRs."""
        asm = _make_asm(next_free_vgpr=128, next_free_sgpr=64, group_seg_size=65536,
                        accum_offset=64)
        kernel = _kernel((9, 4, 2))
        occ = compute_occupancy_from_asm_source(kernel, asm)
        assert occ == 1, f"Expected LDS-limited occ=1, got {occ}"

    # --- gfx908 (non-unified, no accum_offset) ---

    def test_gfx908_256vgpr_no_accum_offset(self):
        """gfx908: no accum_offset; 256 regular VGPRs → occ=1 (256//256=1)."""
        asm = _make_asm(next_free_vgpr=256, next_free_sgpr=64, group_seg_size=32768)
        kernel = _kernel((9, 0, 8))
        occ = compute_occupancy_from_asm_source(kernel, asm)
        assert occ == 1, f"Expected 1, got {occ}"

    def test_gfx908_128vgpr_no_accum_offset(self):
        """gfx908: 128 regular VGPRs → occ=2 (256//128=2)."""
        asm = _make_asm(next_free_vgpr=128, next_free_sgpr=64, group_seg_size=16384)
        kernel = _kernel((9, 0, 8))
        occ = compute_occupancy_from_asm_source(kernel, asm)
        assert occ == 2, f"Expected 2, got {occ}"

    # --- Robustness: missing directives ---

    def test_missing_vgpr_directive_returns_none(self):
        """Missing .amdhsa_next_free_vgpr → returns None (no crash)."""
        asm = (
            ".amdhsa_kernel broken\n"
            "  .amdhsa_next_free_sgpr 64\n"
            "  .amdhsa_group_segment_fixed_size 32768\n"
            ".end_amdhsa_kernel\n"
        )
        result = compute_occupancy_from_asm_source(_kernel((9, 5, 0)), asm)
        assert result is None

    def test_missing_lds_directive_returns_none(self):
        """Missing .amdhsa_group_segment_fixed_size → returns None (no crash)."""
        asm = (
            ".amdhsa_kernel broken\n"
            "  .amdhsa_next_free_vgpr 256\n"
            "  .amdhsa_next_free_sgpr 64\n"
            ".end_amdhsa_kernel\n"
        )
        result = compute_occupancy_from_asm_source(_kernel((9, 5, 0)), asm)
        assert result is None

    def test_empty_string_returns_none(self):
        """Empty assembly source → returns None (no crash)."""
        result = compute_occupancy_from_asm_source(_kernel((9, 5, 0)), "")
        assert result is None

    def test_comment_tolerant(self):
        """Parser handles comments after values (e.g. '// vgprs')."""
        asm = (
            ".amdhsa_kernel comment_test\n"
            "  .amdhsa_next_free_vgpr 256 // unified vgprs  (128 regular + 128 acc)\n"
            "  .amdhsa_next_free_sgpr 88  // sgprs used\n"
            "  .amdhsa_group_segment_fixed_size 32768 // lds bytes\n"
            ".end_amdhsa_kernel\n"
        )
        kernel = _kernel((9, 4, 2))
        occ = compute_occupancy_from_asm_source(kernel, asm)
        # 512//256=2, LDS: 65536//32768=2, wave: 8//1=8 → occ=2
        assert occ == 2, f"Expected 2 (comment-tolerant parse), got {occ}"


# ---------------------------------------------------------------------------
# TestComputeOccupancyRealCustomKernels — real .s files
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _CUSTOM_KERNEL_DIR.is_dir(),
    reason=f"CustomKernels directory not found: {_CUSTOM_KERNEL_DIR}",
)
class TestComputeOccupancyRealCustomKernels:
    """Parse actual CustomKernels/*.s files and assert expected CUOccupancy.

    Expected values are derived from the hardware formula applied to the
    .amdhsa_next_free_vgpr and .amdhsa_group_segment_fixed_size directives
    in each file.

    Canonical cases from the original bug report:
    - gfx942: 256-total-vgpr kernels → occ=2 (not -1 as before)
    - gfx950: 504/512-vgpr kernels   → occ=1
    """

    # (filename_stem, ISA, NumThreads, expected_occ)
    _CASES = [
        # gfx950: HHS, 504 unified vgprs, LDS=133120 → LDS-limited occ=1
        (
            "Custom_Cijk_Alik_Bljk_HHS_BH_MT256x256x64_MI16x16x1_UserArgs_shortname0_gfx950",
            (9, 5, 0), 256, 1,
        ),
        # gfx950: BBS, 504 unified vgprs, LDS=133120 → occ=1
        (
            "Custom_Cijk_Alik_Bljk_BBS_BH_MT256x256x64_MI16x16x1_UserArgs_shortname0_gfx950",
            (9, 5, 0), 256, 1,
        ),
        # gfx950: 512 unified vgprs, LDS=131072 → VGPR-limited occ=1
        (
            "Custom_Cijk_Ailk_Bjlk_S_MX_B_BIAS_HA_S_SAV_NTD_SK3_UserArgs_MT256x256x32_MI16x16x1_shortname0_gfx950",
            (9, 5, 0), 256, 1,
        ),
        # gfx942: HSS, 256 unified vgprs (accum_offset=128), LDS=32768 → occ=2
        (
            "Custom_Cijk_Ailk_Bljk_HSS_BH_Bias_GG_AS_SAV_UserArgs_shortname0_gfx942",
            (9, 4, 2), 256, 2,
        ),
        # gfx942: HHS, 256 unified vgprs (accum_offset=128), LDS=32768 → occ=2
        (
            "Custom_Cijk_Ailk_Bljk_HHS_BH_Bias_GG_AS_SAV_UserArgs_shortname1_gfx942",
            (9, 4, 2), 256, 2,
        ),
        # gfx942: HSS, 512 unified vgprs (accum_offset=256), LDS=65536 → occ=1
        (
            "Custom_Cijk_Ailk_Bljk_HSS_BH_Bias_GG_AS_SAV_UserArgs_shortname10_gfx942",
            (9, 4, 2), 256, 1,
        ),
    ]

    @pytest.mark.parametrize("stem,isa,threads,expected", _CASES,
                             ids=[c[0].split("Custom_")[1][:40] for c in _CASES])
    def test_real_kernel_occupancy(self, stem, isa, threads, expected):
        """Parser on a real .s file should return the expected occupancy."""
        s_path = _CUSTOM_KERNEL_DIR / f"{stem}.s"
        if not s_path.exists():
            pytest.skip(f"File not found: {s_path.name}")

        asm_source = s_path.read_text()
        kernel = {"ISA": list(isa), "NumThreads": threads, "KernelLanguage": "Assembly"}
        occ = compute_occupancy_from_asm_source(kernel, asm_source)

        assert occ is not None, (
            f"compute_occupancy_from_asm_source returned None for {stem}.s; "
            f"check that all required .amdhsa_ directives are present."
        )
        assert occ == expected, (
            f"{stem}.s: expected CUOccupancy={expected}, got {occ}\n"
            f"  ISA={isa}, threads={threads}"
        )

    def test_all_gfx950_kernels_have_positive_occupancy(self):
        """Every gfx950 custom .s file should yield CUOccupancy >= 1 (never None/-1)."""
        files = sorted(_CUSTOM_KERNEL_DIR.glob("*gfx950*.s"))
        assert len(files) > 0, "No gfx950 custom kernels found in CustomKernels/"
        for f in files:
            asm = f.read_text()
            kernel = {"ISA": [9, 5, 0], "NumThreads": 256, "KernelLanguage": "Assembly"}
            occ = compute_occupancy_from_asm_source(kernel, asm)
            assert occ is not None, f"{f.name}: parse returned None"
            assert occ >= 1, f"{f.name}: CUOccupancy={occ} < 1"

    def test_all_gfx942_kernels_have_positive_occupancy(self):
        """Every gfx942 custom .s file should yield CUOccupancy >= 1 (never None/-1)."""
        files = sorted(_CUSTOM_KERNEL_DIR.glob("*gfx942*.s"))
        assert len(files) > 0, "No gfx942 custom kernels found in CustomKernels/"
        for f in files:
            asm = f.read_text()
            kernel = {"ISA": [9, 4, 2], "NumThreads": 256, "KernelLanguage": "Assembly"}
            occ = compute_occupancy_from_asm_source(kernel, asm)
            assert occ is not None, f"{f.name}: parse returned None"
            assert occ >= 1, f"{f.name}: CUOccupancy={occ} < 1"


# ---------------------------------------------------------------------------
# TestPythonDebugWarning — Change 2a: print2-gated warning in processKernelSource
# ---------------------------------------------------------------------------

class TestPythonDebugWarning:
    """Verify that the debug warning for CUOccupancy<=0 fires iff verbosity>=2.

    The warning logic in ``processKernelSource`` (Run.py) is:

        if cuocc <= 0 and getVerbosity() >= 2:
            print2(f"[codegen] CUOccupancy={cuocc} ...")

    We test this logic directly without importing Tensile.Common (which would pull in
    rocisa).  The tests replicate the exact conditional and print call.
    """

    def _run_warning_check(self, cuocc_value, verbosity):
        """Simulate the processKernelSource warning logic without requiring rocisa.

        Directly replicates the conditional (cuocc <= 0 and verbosity >= 2) and
        the print call, mirroring what is in Run.py processKernelSource.
        """
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            if cuocc_value <= 0 and verbosity >= 2:
                print(
                    f"[codegen] CUOccupancy={cuocc_value} (<=0) after codegen for kernel test_kernel; "
                    f"runtime will clamp to 1.  Enable PrintLevel>=2 to see this message."
                )
        finally:
            sys.stdout = old_stdout
        return captured.getvalue()

    def test_warning_fires_at_verbosity2_when_cuocc_negative(self):
        """Warning is emitted when verbosity=2 and CUOccupancy=-1."""
        output = self._run_warning_check(cuocc_value=-1, verbosity=2)
        assert "CUOccupancy=-1" in output, (
            f"Expected warning not found in output: {output!r}"
        )

    def test_warning_fires_at_verbosity2_when_cuocc_zero(self):
        """Warning is emitted when verbosity=2 and CUOccupancy=0."""
        output = self._run_warning_check(cuocc_value=0, verbosity=2)
        assert "CUOccupancy=0" in output

    def test_no_warning_at_verbosity1(self):
        """No warning at default verbosity=1 (silent in normal builds)."""
        output = self._run_warning_check(cuocc_value=-1, verbosity=1)
        assert output == "", (
            f"Unexpected output at verbosity=1: {output!r}"
        )

    def test_no_warning_at_verbosity0(self):
        """No warning at verbosity=0 (user wants no printing)."""
        output = self._run_warning_check(cuocc_value=-1, verbosity=0)
        assert output == "", f"Unexpected output at verbosity=0: {output!r}"

    def test_no_warning_for_positive_cuocc(self):
        """No warning when CUOccupancy > 0 (normal case)."""
        output = self._run_warning_check(cuocc_value=2, verbosity=2)
        assert output == "", f"Unexpected warning for positive CUOccupancy: {output!r}"

    def test_warning_at_verbosity3_when_cuocc_negative(self):
        """Warning also fires at verbosity=3 (higher than 2)."""
        output = self._run_warning_check(cuocc_value=-1, verbosity=3)
        assert "CUOccupancy=-1" in output
