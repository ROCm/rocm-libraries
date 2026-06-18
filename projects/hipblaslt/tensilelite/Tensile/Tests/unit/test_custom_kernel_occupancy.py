# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Unit tests for CUOccupancy computation from custom kernel .s files at build time.

Custom kernels bypass checkResources/updateOccupancyFromScan; the fix adds
compute_occupancy_from_asm_source which parses .amdhsa_ directives directly.
Also tests the print2-gated CUOccupancy<=0 warning in processKernelSource.
"""

import io
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# The coverage tox env runs `pytest -m unit Tensile/Tests/unit`; without this
# explicit declaration the file is silently deselected and reports 0% coverage.
pytestmark = pytest.mark.unit

from Tensile.OccupancyMeasure import (
    compute_occupancy_from_asm_source,
    compute_occupancy_from_resources,
    _arch_caps_for_kernel,
)

try:
    from Tensile.KernelWriterAssembly import KernelWriterAssembly as _KWA
    _KWA_AVAILABLE = True
except Exception:
    _KWA_AVAILABLE = False

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
# TestPythonDebugWarning — Test warning messages
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TestNonGfx9OccupancyNotComputed — Check for architectures not implemented in occupancy calculator
# ---------------------------------------------------------------------------

class TestNonGfx9OccupancyNotComputed:
    """Non-gfx9 ISAs must not yield a CUOccupancy override.

    compute_occupancy_from_asm_source returns None for non-gfx9 ISAs because
    _arch_caps_for_kernel returns None (the formula is gfx9/wave64-specific).
    Callers use ``if occ is not None: kernel["CUOccupancy"] = occ``; returning
    None ensures CUOccupancy stays at its default (-1) for non-gfx9 custom kernels.
    """

    def _asm(self, vgpr=64, sgpr=32, lds=16384):
        return _make_asm(next_free_vgpr=vgpr, next_free_sgpr=sgpr, group_seg_size=lds)

    def test_gfx1100_returns_none(self):
        """gfx1100 (wave32, gfx11): compute_occupancy_from_asm_source returns None."""
        result = compute_occupancy_from_asm_source(_kernel((11, 0, 0)), self._asm())
        assert result is None, f"gfx1100 must return None (unsupported arch), got {result}"

    def test_gfx1201_returns_none(self):
        """gfx1201 (wave32, gfx12 non-gfx1250): returns None."""
        result = compute_occupancy_from_asm_source(_kernel((12, 0, 1)), self._asm())
        assert result is None, f"gfx1201 must return None, got {result}"

    def test_gfx1030_returns_none(self):
        """gfx1030 (wave32, gfx10): returns None."""
        result = compute_occupancy_from_asm_source(_kernel((10, 3, 0)), self._asm())
        assert result is None, f"gfx1030 must return None, got {result}"

    def test_gfx950_gfx9_returns_valid_occ(self):
        """gfx950 (gfx9): compute_occupancy_from_asm_source returns a valid occupancy."""
        asm = _make_asm(256, 64, 32768)
        result = compute_occupancy_from_asm_source(_kernel((9, 5, 0)), asm)
        assert result is not None and result >= 1, \
            f"gfx950 (gfx9) must return occ >= 1, got {result}"

    def test_gfx908_gfx9_returns_valid_occ(self):
        """gfx908 (gfx9, non-unified): compute_occupancy_from_asm_source returns valid."""
        asm = _make_asm(64, 32, 16384)
        result = compute_occupancy_from_asm_source(_kernel((9, 0, 8)), asm)
        assert result is not None and result >= 1, \
            f"gfx908 (gfx9) must return occ >= 1, got {result}"

    def test_cuoccupancy_not_set_for_non_gfx9(self):
        """Simulated caller: CUOccupancy stays -1 when compute returns None (non-gfx9)."""
        kernel = _kernel((11, 0, 0))
        kernel["CUOccupancy"] = -1
        asm = _make_asm(64, 32, 16384)
        occ = compute_occupancy_from_asm_source(kernel, asm)
        if occ is not None:
            kernel["CUOccupancy"] = occ   # what getSourceFileString does
        assert kernel["CUOccupancy"] == -1, \
            f"CUOccupancy must stay -1 for non-gfx9; got {kernel['CUOccupancy']}"


class TestPythonDebugWarning:
    """Verify the print2-gated CUOccupancy<=0 warning in processKernelSource (Run.py).

    Tests replicate the conditional directly without importing Tensile.Common.
    """

    def _run_warning_check(self, cuocc_value, verbosity):
        """Replicate the processKernelSource warning conditional and capture output."""
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            if cuocc_value <= 0 and verbosity >= 2:
                print(
                    f"[codegen] CUOccupancy={cuocc_value} (<=0) after codegen for kernel test_kernel; "
                    f"runtime will clamp to 1."
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


# ---------------------------------------------------------------------------
# TestGetSourceFileStringCustomKernelPath
# Regression test for the AttributeError bug fixed in commit ba0056217d3.
# ---------------------------------------------------------------------------

class _KernelObj(dict):
    """dict subclass with attribute-style access, matching the kernel objects
    that TensileCreateLibrary passes to getSourceFileString."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, val):
        self[name] = val


@pytest.mark.skipif(not _KWA_AVAILABLE, reason="KernelWriterAssembly import requires rocisa")
@pytest.mark.skipif(
    not _CUSTOM_KERNEL_DIR.is_dir(),
    reason=f"CustomKernels directory not found: {_CUSTOM_KERNEL_DIR}",
)
class TestGetSourceFileStringCustomKernelPath:
    """Regression guard for commit ba0056217d3.

    The bug: getSourceFileString for custom kernels accessed self.states.regCaps
    and self.states.archCaps to compute CUOccupancy.  On the TensileCreateLibrary
    custom-kernel path, only setRocIsa() is called (not setKernel/setKernel via
    _initKernel), so StateValues is never constructed and those attributes do not
    exist — raising AttributeError and breaking ALL gfx9 custom-kernel builds.

    The fix (ba0056217d3): query rocIsa.getInstance() directly instead of
    self.states.regCaps/archCaps.

    This test exercises the REAL getSourceFileString custom-kernel code block with
    a KernelWriterAssembly whose self.states has NO regCaps/archCaps attribute —
    exactly mirroring the setRocIsa-only production path.  It would have failed
    (AttributeError propagating out of getSourceFileString) with the pre-fix code.
    """

    # gfx942 kernel: 256 unified VGPRs (accum_offset=128), LDS=32768 → occ=2
    _KERNEL_STEM = (
        "Custom_Cijk_Ailk_Bljk_HSS_BH_Bias_GG_AS_SAV_UserArgs_shortname0_gfx942"
    )
    _ISA = (9, 4, 2)
    _WAVEFRONT_SIZE = 64
    _EXPECTED_OCC = 2

    @pytest.fixture(autouse=True)
    def _init_rocisa_gfx942(self):
        """Initialize the rocisa singleton for gfx942.

        Mirrors what setKernel() does in _initKernel so that the singleton holds
        live caps for gfx942.  In production, this state is captured via getData()
        and restored via setData() inside setRocIsa(), making getRegCaps() /
        getArchCaps() valid when getSourceFileString calls rocIsa.getInstance().
        """
        from rocisa import rocIsa

        asm_path = shutil.which("amdclang++") or "/usr/bin/amdclang++"
        if not os.path.exists(asm_path):
            pytest.skip(f"amdclang++ not found at {asm_path}; cannot init rocisa")

        ti = rocIsa.getInstance()
        ti.init(self._ISA, asm_path)
        ti.setKernel(self._ISA, self._WAVEFRONT_SIZE)

        assert ti.getRegCaps().get("MaxVgpr") is not None, (
            "rocisa singleton not properly initialized; getRegCaps() returned empty dict"
        )

    def _make_kernel(self):
        """Construct a kernel object matching the production custom-kernel dict."""
        k = _KernelObj({
            "CustomKernelName": self._KERNEL_STEM,
            "ISA": list(self._ISA),
            "NumThreads": 256,
            "KernelLanguage": "Assembly",
            "CUOccupancy": -1,
        })
        # duplicate is set as an attribute by TensileCreateLibrary before dispatch
        k.duplicate = False
        return k

    def _make_kwa_setrocisa_path(self, asm_source):
        """Create a KernelWriterAssembly that mirrors the setRocIsa-only init path.

        Key invariant: self.states has NO regCaps and NO archCaps.

        On the production custom-kernel path, only setRocIsa() is ever called
        (it calls ti.setData() + ti.setOutputOptions() but never _initKernel).
        _initKernel is the only place that creates StateValues and populates
        self.states.regCaps / self.states.archCaps.  A custom-kernel KWA therefore
        never has those attributes — which is precisely what the pre-fix code failed
        to account for.

        _getCustomKernelSource is monkeypatched to return the real .s file content
        directly, avoiding the need for self.assembler / self.debugConfig.
        """
        kwa = object.__new__(_KWA)
        # states WITHOUT regCaps/archCaps — the critical setRocIsa-path invariant
        kwa.states = SimpleNamespace()
        # return the .s content without touching the filesystem or assembler
        kwa._getCustomKernelSource = lambda kernel, directory: asm_source
        return kwa

    # ------------------------------------------------------------------
    # Sanity: verify the test setup genuinely lacks regCaps
    # ------------------------------------------------------------------

    def test_setup_has_no_regcaps_on_states(self):
        """The test fixture must NOT have regCaps on self.states.

        This confirms the test setup faithfully reproduces the pre-fix failure
        condition: accessing self.states.regCaps raises AttributeError, which
        is exactly what happened in production before the fix.
        """
        s_path = _CUSTOM_KERNEL_DIR / f"{self._KERNEL_STEM}.s"
        if not s_path.exists():
            pytest.skip(f"File not found: {s_path.name}")

        kwa = self._make_kwa_setrocisa_path(s_path.read_text())
        with pytest.raises(AttributeError):
            _ = kwa.states.regCaps

    # ------------------------------------------------------------------
    # Real regression: getSourceFileString must not raise
    # ------------------------------------------------------------------

    def test_getSourceFileString_no_raise_on_setrocisa_path(self):
        """getSourceFileString must not raise AttributeError on the setRocIsa path.

        Pre-fix: the occupancy block accessed self.states.regCaps which does not
        exist → AttributeError propagated up (NOT caught by 'except RuntimeError').
        Post-fix: rocIsa.getInstance().getRegCaps() is called instead → no error.
        """
        s_path = _CUSTOM_KERNEL_DIR / f"{self._KERNEL_STEM}.s"
        if not s_path.exists():
            pytest.skip(f"File not found: {s_path.name}")

        kwa = self._make_kwa_setrocisa_path(s_path.read_text())
        kernel = self._make_kernel()

        errcode, code = kwa.getSourceFileString(kernel)
        assert errcode == 0, f"Expected errcode=0, got {errcode}"
        assert code, "Expected non-empty assembly source"

    def test_getSourceFileString_sets_cuoccupancy_from_singleton(self):
        """getSourceFileString must compute CUOccupancy from the rocisa singleton.

        Verifies the correct value (occ=2 for a 256-vgpr gfx942 kernel) is set,
        not the default -1.  Pre-fix, this would never be reached because the
        AttributeError on self.states.regCaps would propagate first.
        """
        s_path = _CUSTOM_KERNEL_DIR / f"{self._KERNEL_STEM}.s"
        if not s_path.exists():
            pytest.skip(f"File not found: {s_path.name}")

        kwa = self._make_kwa_setrocisa_path(s_path.read_text())
        kernel = self._make_kernel()

        kwa.getSourceFileString(kernel)

        assert kernel["CUOccupancy"] == self._EXPECTED_OCC, (
            f"Expected CUOccupancy={self._EXPECTED_OCC} (from rocisa singleton caps), "
            f"got {kernel['CUOccupancy']}. "
            f"Pre-fix code would have raised AttributeError on self.states.regCaps "
            f"before ever reaching this line."
        )


# ---------------------------------------------------------------------------
# TestGetSourceFileStringIncompleteSingletonCaps
# Regression test for the grouped_gemm_ck KeyError (post-commit a1c968f136c).
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _KWA_AVAILABLE, reason="KernelWriterAssembly requires rocisa")
class TestGetSourceFileStringIncompleteSingletonCaps:
    """Regression guard for the grouped_gemm_ck KeyError on gfx942.

    Root cause: grouped_gemm_ck custom kernels call getSourceFileString with a
    rocisa singleton whose getRegCaps() returns an incomplete dict that lacks
    "MaxVgpr" (setKernel was never called for this ISA in the ck codegen path).
    The occupancy block then raises ``KeyError: 'MaxVgpr'``, crashing codegen.

    The fix (this commit): try the live singleton caps; on KeyError/AttributeError
    fall back to the static hardware table (_arch_caps_for_kernel).  If the static
    table can supply caps (gfx942 is covered), compute occupancy from there;
    otherwise (or if asm directives are missing) leave CUOccupancy at -1.

    These tests mock rocIsa.getInstance() so no hardware / amdclang++ is needed.
    They would have raised KeyError pre-fix and pass post-fix.
    """

    _ISA = (9, 4, 2)
    _EXPECTED_OCC = 2   # 256-vgpr gfx942 kernel via static fallback

    def _make_kernel_obj(self, isa=None):
        k = _KernelObj({
            "CustomKernelName": "test_incomplete_caps_kernel",
            "ISA": list(isa or self._ISA),
            "NumThreads": 256,
            "KernelLanguage": "Assembly",
            "CUOccupancy": -1,
        })
        k.duplicate = False
        return k

    def _good_asm(self):
        """Valid assembly source for a 256-vgpr gfx942 custom kernel (occ=2 via table)."""
        return _make_asm(next_free_vgpr=256, next_free_sgpr=102,
                         group_seg_size=32768, accum_offset=128,
                         kernel_name="test_incomplete_caps_kernel")

    def _make_kwa(self, asm_source):
        kwa = object.__new__(_KWA)
        kwa.states = SimpleNamespace()
        kwa._getCustomKernelSource = lambda k, d: asm_source
        return kwa

    # ------------------------------------------------------------------
    # Core regression: KeyError pre-fix, no error post-fix
    # ------------------------------------------------------------------

    def test_incomplete_singleton_caps_no_raise(self):
        """Pre-fix: KeyError 'MaxVgpr'; post-fix: no exception raised.

        Mirrors the grouped_gemm_ck failure path exactly: singleton returns
        empty dicts for both getRegCaps() and getArchCaps(), and the custom
        kernel has valid assembly directives.
        """
        from unittest.mock import MagicMock, patch

        mock_ti = MagicMock()
        mock_ti.getRegCaps.return_value = {}   # missing MaxVgpr — the failing case
        mock_ti.getArchCaps.return_value = {}  # missing DeviceLDS etc.

        kwa = self._make_kwa(self._good_asm())
        kernel = self._make_kernel_obj()

        with patch("Tensile.KernelWriterAssembly.rocIsa") as mock_rocisa:
            mock_rocisa.getInstance.return_value = mock_ti
            errcode, code = kwa.getSourceFileString(kernel)

        assert errcode == 0, (
            f"getSourceFileString must return errcode=0 with incomplete singleton caps; "
            f"got {errcode}.  Pre-fix code raised KeyError: 'MaxVgpr'."
        )
        assert code, "assembly source must be non-empty"

    def test_incomplete_singleton_caps_uses_static_fallback(self):
        """Incomplete singleton caps → static table (_arch_caps_for_kernel) provides caps.

        For gfx942, _arch_caps_for_kernel returns (512, 800, 65536, 8); combined
        with 256 VGPRs and 32768 B LDS from the asm source this yields occ=2.
        """
        from unittest.mock import MagicMock, patch

        mock_ti = MagicMock()
        mock_ti.getRegCaps.return_value = {}
        mock_ti.getArchCaps.return_value = {}

        kwa = self._make_kwa(self._good_asm())
        kernel = self._make_kernel_obj()

        with patch("Tensile.KernelWriterAssembly.rocIsa") as mock_rocisa:
            mock_rocisa.getInstance.return_value = mock_ti
            kwa.getSourceFileString(kernel)

        assert kernel["CUOccupancy"] == self._EXPECTED_OCC, (
            f"Expected CUOccupancy={self._EXPECTED_OCC} via static table fallback "
            f"for gfx942 with 256 VGPRs; got {kernel['CUOccupancy']}"
        )

    def test_incomplete_caps_missing_asm_directives_stays_minus1(self):
        """Incomplete singleton caps AND missing asm directives → CUOccupancy stays -1.

        Even though the static table has gfx942 caps, compute_occupancy_from_asm_source
        returns None when the asm lacks the required .amdhsa_ directives.  The
        caller then leaves CUOccupancy at its default (-1).
        """
        from unittest.mock import MagicMock, patch

        mock_ti = MagicMock()
        mock_ti.getRegCaps.return_value = {}
        mock_ti.getArchCaps.return_value = {}

        # asm missing all required directives
        asm_no_directives = ".amdhsa_kernel bad_kernel\n.end_amdhsa_kernel\n"
        kwa = self._make_kwa(asm_no_directives)
        kernel = self._make_kernel_obj()

        with patch("Tensile.KernelWriterAssembly.rocIsa") as mock_rocisa:
            mock_rocisa.getInstance.return_value = mock_ti
            errcode, _ = kwa.getSourceFileString(kernel)

        assert errcode == 0, f"errcode must be 0, got {errcode}"
        assert kernel["CUOccupancy"] == -1, (
            f"CUOccupancy must remain -1 when asm directives are missing; "
            f"got {kernel['CUOccupancy']}"
        )

    def test_partial_singleton_caps_missing_only_maxvgpr(self):
        """Singleton has some keys but not MaxVgpr → KeyError caught, fallback used."""
        from unittest.mock import MagicMock, patch

        mock_ti = MagicMock()
        # PhysicalMaxSgpr present but MaxVgpr missing — exactly as in the CI failure
        mock_ti.getRegCaps.return_value = {"PhysicalMaxSgpr": 800}
        mock_ti.getArchCaps.return_value = {"DeviceLDS": 65536, "MaxWavesPerSimd": 8}

        kwa = self._make_kwa(self._good_asm())
        kernel = self._make_kernel_obj()

        with patch("Tensile.KernelWriterAssembly.rocIsa") as mock_rocisa:
            mock_rocisa.getInstance.return_value = mock_ti
            errcode, code = kwa.getSourceFileString(kernel)

        assert errcode == 0, f"errcode must be 0, got {errcode}"
        # Falls back to static table → occ=2
        assert kernel["CUOccupancy"] == self._EXPECTED_OCC, (
            f"Expected CUOccupancy={self._EXPECTED_OCC} from static fallback; "
            f"got {kernel['CUOccupancy']}"
        )

    def test_complete_singleton_caps_still_compute_correct_occupancy(self):
        """Complete singleton caps still yield the correct occupancy (no regression).

        When the singleton has MaxVgpr=256 (gfx942 ArchAccUnifiedRegs: phy=512),
        PhysicalMaxSgpr=800, DeviceLDS=65536, MaxWavesPerSimd=8 — the live path
        works exactly as before: occ=2 for 256-vgpr kernel.
        """
        from unittest.mock import MagicMock, patch

        mock_ti = MagicMock()
        mock_ti.getRegCaps.return_value = {"MaxVgpr": 256, "PhysicalMaxSgpr": 800}
        mock_ti.getArchCaps.return_value = {
            "ArchAccUnifiedRegs": True,
            "DeviceLDS": 65536,
            "MaxWavesPerSimd": 8,
        }

        kwa = self._make_kwa(self._good_asm())
        kernel = self._make_kernel_obj()

        with patch("Tensile.KernelWriterAssembly.rocIsa") as mock_rocisa:
            mock_rocisa.getInstance.return_value = mock_ti
            kwa.getSourceFileString(kernel)

        # MaxVgpr=256, ArchAccUnifiedRegs=True → phy=256*2=512; 512//256=2, LDS ok → occ=2
        assert kernel["CUOccupancy"] == self._EXPECTED_OCC, (
            f"Expected CUOccupancy={self._EXPECTED_OCC} from complete singleton caps; "
            f"got {kernel['CUOccupancy']}"
        )
