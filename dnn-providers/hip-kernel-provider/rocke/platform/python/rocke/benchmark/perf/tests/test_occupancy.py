# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the occupancy primitive's pure logic (no GPU)."""
import re
import shutil
import subprocess
import unittest
from types import SimpleNamespace

from rocke.benchmark.perf import occupancy


class TestResourcesSourceTag(unittest.TestCase):
    def setUp(self):
        self._orig = occupancy.parse_notes

    def tearDown(self):
        occupancy.parse_notes = self._orig

    def test_source_tagged_elf_notes(self):
        occupancy.parse_notes = lambda b: {
            "vgpr": 24,
            "sgpr": 16,
            "lds_bytes": 2048,
            "wave_size": 64,
        }
        res = occupancy.resources(b"fake", "gfx950")
        self.assertEqual(res["source"], "elf_notes")  # distinguishes from rocprofv3
        self.assertEqual(res["vgpr"], 24)
        self.assertIsNotNone(res["occupancy"])

    def test_empty_notes_returns_empty(self):
        occupancy.parse_notes = lambda b: {}
        self.assertEqual(occupancy.resources(b"x", "gfx950"), {})

    def test_elf_target_overrides_a_stale_arch_flag(self):
        # Real gfx1201 binary, caller passed --arch gfx950: the RDNA occupancy model
        # must win, otherwise the wrong wave count is reported with no signal.
        occupancy.parse_notes = lambda b: {
            "vgpr": 111,
            "target": "gfx1201",
            "wave_size": 32,
        }
        res = occupancy.resources(b"fake", "gfx950")
        self.assertEqual(res["target_arch"], "gfx1201")
        self.assertEqual(
            res["occupancy"], occupancy._occupancy_estimate(111, "gfx1201", 32)
        )
        self.assertNotEqual(
            res["occupancy"], occupancy._occupancy_estimate(111, "gfx950", 64)
        )

    def test_arch_argument_is_the_fallback_when_notes_lack_a_target(self):
        occupancy.parse_notes = lambda b: {"vgpr": 111, "wave_size": 64}
        res = occupancy.resources(b"fake", "gfx950")
        self.assertEqual(res["target_arch"], "gfx950")
        self.assertEqual(
            res["occupancy"], occupancy._occupancy_estimate(111, "gfx950", 64)
        )


class TestParseNotes(unittest.TestCase):
    """Parse a real `llvm-readelf --notes` excerpt (captured from a gfx1201 HSACO)."""

    _NOTES = (
        "Displaying notes found in: .note\n"
        "  amdhsa.kernels:\n"
        "    - .agpr_count:    0\n"
        "      .group_segment_fixed_size: 512\n"
        "      .sgpr_count:    23\n"
        "      .sgpr_spill_count: 0\n"
        "      .vgpr_count:    111\n"
        "      .vgpr_spill_count: 0\n"
        "      .wavefront_size: 32\n"
        "  amdhsa.target:   amdgcn-amd-amdhsa--gfx1201\n"
        "  amdhsa.version:\n"
    )

    def setUp(self):
        self._orig_readelf = occupancy._readelf
        self._orig_run = occupancy.subprocess.run
        occupancy._readelf = lambda: "llvm-readelf"
        occupancy.subprocess.run = lambda *a, **k: SimpleNamespace(
            returncode=0, stdout=self._NOTES, stderr=""
        )

    def tearDown(self):
        occupancy._readelf = self._orig_readelf
        occupancy.subprocess.run = self._orig_run

    def test_fields_and_target_parsed(self):
        fields = occupancy.parse_notes(b"fake")
        self.assertEqual(fields["vgpr"], 111)
        self.assertEqual(fields["lds_bytes"], 512)
        self.assertEqual(fields["target"], "gfx1201")

    def test_resources_uses_the_parsed_target(self):
        res = occupancy.resources(b"fake", "gfx950")  # deliberately wrong flag
        self.assertEqual(res["target_arch"], "gfx1201")
        self.assertEqual(res["occupancy"], 12)  # RDNA model, not the CDNA 4


class TestOccupancyEstimate(unittest.TestCase):
    def test_estimate_is_capped(self):
        # tiny VGPR -> capped at max_waves_per_simd, not unbounded
        est = occupancy._occupancy_estimate(4, "gfx950", 64)
        self.assertEqual(est, 8)  # gfx950 max_waves_per_simd

    def test_zero_vgpr_none(self):
        self.assertIsNone(occupancy._occupancy_estimate(0, "gfx950", 64))

    def test_gfx90a_uses_eight_vgpr_allocation_granularity(self):
        self.assertEqual(occupancy._occupancy_estimate(65, "gfx90a", 64), 7)


class TestEstimateOccupancyDetail(unittest.TestCase):
    """Multi-limiter model: min over VGPR / AGPR / LDS / workgroup / wave cap."""

    def setUp(self):
        self._orig = occupancy.parse_notes

    def tearDown(self):
        occupancy.parse_notes = self._orig

    def _patch(self, **notes):
        occupancy.parse_notes = lambda b: notes

    def test_vgpr_limited(self):
        # 256 VGPR on gfx950 -> 512//256 = 2 waves/SIMD; LDS/AGPR slack.
        self._patch(vgpr=256, agpr=0, lds_bytes=2048, max_flat_workgroup_size=256)
        det = occupancy.estimate_occupancy_detail(b"x", "gfx950")
        self.assertEqual(det["waves_per_simd"], 2)
        self.assertEqual(det["limited_by"], "VGPR")

    def test_lds_limited(self):
        # low VGPR (8 waves) but a fat 80 KB workgroup -> LDS binds below regs.
        self._patch(vgpr=64, agpr=0, lds_bytes=80000, max_flat_workgroup_size=512)
        det = occupancy.estimate_occupancy_detail(b"x", "gfx950")
        self.assertEqual(det["limited_by"], "LDS")
        self.assertEqual(det["waves_per_simd"], 4)  # (163840//80000=2 wg) * 8 / 4

    def test_agpr_limited_separate_pool(self):
        # gfx942 (CDNA3) keeps separate VGPR/AGPR pools -> heavy accumulator use
        # starves waves before VGPR does.
        self._patch(vgpr=64, agpr=200, lds_bytes=2048, max_flat_workgroup_size=256)
        det = occupancy.estimate_occupancy_detail(b"x", "gfx942")
        self.assertEqual(det["limited_by"], "AGPR")

    def test_gfx950_combined_reg_pool(self):
        # gfx950 (CDNA4): VGPR+AGPR share one file, so the SUM sets the wave count.
        # Calibrated vs gfx950 hardware MeanOccupancyPerCU: an AGPR-heavy kernel (44 VGPR +
        # 132 AGPR) measured ~7 waves/CU. 512 // align_up(176,16)=176 -> 2/SIMD
        # (8/CU); a separate 256-AGPR pool would wrongly give 1/SIMD (4/CU).
        self._patch(vgpr=44, agpr=132, lds_bytes=24576, max_flat_workgroup_size=256)
        det = occupancy.estimate_occupancy_detail(b"x", "gfx950")
        self.assertEqual(det["waves_per_simd"], 2)
        self.assertEqual(det["limited_by"], "VGPR+AGPR")

    def test_waves_per_wg_override_beats_notes(self):
        # Both paths label "LDS", so assert the VALUE: the note's 64//64=1 wave/WG
        # gives 2*1 = 2 waves/CU (waves_per_simd 0); the override (8) gives 2*8 = 16
        # (waves_per_simd 4). Dropping the parameter would fail these asserts.
        self._patch(vgpr=64, agpr=0, lds_bytes=80000, max_flat_workgroup_size=64)
        det = occupancy.estimate_occupancy_detail(b"x", "gfx950", waves_per_wg=8)
        self.assertEqual(det["waves_per_cu"], 16)
        self.assertEqual(det["waves_per_simd"], 4)

    def test_unknown_arch_returns_empty(self):
        self._patch(vgpr=64, lds_bytes=2048)
        self.assertEqual(occupancy.estimate_occupancy_detail(b"x", "gfx1201"), {})

    def test_empty_notes_returns_empty(self):
        self._patch()  # {}
        self.assertEqual(occupancy.estimate_occupancy_detail(b"x", "gfx950"), {})

    def test_missing_required_field_returns_empty(self):
        # A partial parse (e.g. the vgpr regex stopped matching) must return {},
        # not clamp the missing vgpr to 1 and report the arch maximum.
        self._patch(lds_bytes=2048, agpr=0)  # no vgpr
        self.assertEqual(occupancy.estimate_occupancy_detail(b"x", "gfx950"), {})
        self._patch(vgpr=64)  # no lds_bytes
        self.assertEqual(occupancy.estimate_occupancy_detail(b"x", "gfx950"), {})


def _find_llvm_tool(name):
    """The ROCm LLVM tool ``name`` (llc, llvm-readelf), or None when absent."""
    for candidate in (f"/opt/rocm/llvm/bin/{name}", name):
        path = shutil.which(candidate)
        if path:
            return path
    return None


def _vgpr_kernel_ir(vgprs, wave_size=32):
    """An AMDGPU kernel that uses exactly ``vgprs`` VGPRs (it clobbers the top one).

    The workgroup is capped at 64 lanes so the workgroup size never limits the
    occupancy the compiler reports; only the VGPR count does.
    """
    features = ' "target-features"="+wavefrontsize64"' if wave_size == 64 else ""
    return (
        "define amdgpu_kernel void @k() #0 {\n"
        f'  call void asm sideeffect "", "~{{v{vgprs - 1}}}"()\n'
        "  ret void\n"
        "}\n"
        f'attributes #0 = {{ "amdgpu-flat-work-group-size"="1,64"{features} }}\n'
    )


def _run_llc(llc, processor, ir, *extra_args):
    """Run llc on ``ir``; None when this llc does not know ``processor``."""
    out = subprocess.run(
        [llc, "-mtriple=amdgcn-amd-amdhsa", f"-mcpu={processor}", *extra_args],
        input=ir.encode(),
        capture_output=True,
        timeout=120,
        check=True,
    )
    if b"not a recognized processor" in out.stderr:
        return None
    return out.stdout


def _llc_occupancy(llc, processor, vgprs):
    """The waves per SIMD llc reports for a kernel using ``vgprs`` VGPRs."""
    asm = _run_llc(llc, processor, _vgpr_kernel_ir(vgprs), "-o", "-")
    if asm is None:
        return None
    return int(re.search(rb"; Occupancy: (\d+)", asm).group(1))


def _llc_code_object(llc, processor, vgprs, wave_size=32):
    """A real relocatable code object (ELF with AMDGPU notes) built by llc."""
    obj = _run_llc(
        llc,
        processor,
        _vgpr_kernel_ir(vgprs, wave_size),
        "-filetype=obj",
        "-o",
        "-",
    )
    if obj is None:
        raise unittest.SkipTest(f"this llc does not know {processor}")
    return obj


# VGPR counts at which the two register-file sizes give distinct wave counts.
_VGPR_SWEEP = (24, 64, 72, 84, 96, 104, 120, 128, 168, 176, 200, 256)
# Waves per SIMD at each _VGPR_SWEEP count, as the compiler reports them.
_WAVES_WITH_1536_VGPRS = (16, 16, 16, 16, 16, 12, 12, 10, 9, 8, 7, 5)
_WAVES_WITH_1024_VGPRS = (16, 16, 12, 10, 10, 9, 8, 8, 5, 5, 4, 4)
_MEASURED_RDNA_WAVES = {
    "gfx1100": _WAVES_WITH_1536_VGPRS,
    "gfx1101": _WAVES_WITH_1536_VGPRS,
    "gfx1102": _WAVES_WITH_1024_VGPRS,
    "gfx1103": _WAVES_WITH_1024_VGPRS,
    "gfx1150": _WAVES_WITH_1024_VGPRS,
    "gfx1151": _WAVES_WITH_1536_VGPRS,
    "gfx1152": _WAVES_WITH_1024_VGPRS,
    "gfx1153": _WAVES_WITH_1024_VGPRS,
    "gfx1154": _WAVES_WITH_1024_VGPRS,
    "gfx11-generic": _WAVES_WITH_1024_VGPRS,
    "gfx1200": _WAVES_WITH_1536_VGPRS,
    "gfx1201": _WAVES_WITH_1536_VGPRS,
    "gfx12-generic": _WAVES_WITH_1536_VGPRS,
    "gfx1250": _WAVES_WITH_1024_VGPRS,
}


class TestRdnaVgprFilePerProcessor(unittest.TestCase):
    """The VGPR-limited estimate follows each RDNA processor's own register file.

    gfx1100, gfx1101, gfx1151 and the gfx12 parts have 1536 VGPRs per SIMD
    (allocated in steps of 24); the other gfx11 parts, gfx11-generic and gfx1250
    have 1024 (steps of 16). The pinned wave counts were measured with the
    compiler's own occupancy report, identical at clang 20, 23 and 24 (gfx1154
    needs clang 24, gfx1250 clang 23 or later):

        llc -mtriple=amdgcn-amd-amdhsa -mcpu=<processor> -o - kernel.ll | grep Occupancy

    where kernel.ll is ``_vgpr_kernel_ir(n)`` for each n in ``_VGPR_SWEEP``.
    """

    def test_model_matches_the_measured_wave_counts(self):
        for processor, measured in _MEASURED_RDNA_WAVES.items():
            for vgprs, waves in zip(_VGPR_SWEEP, measured):
                with self.subTest(processor=processor, vgprs=vgprs):
                    self.assertEqual(
                        occupancy._occupancy_estimate(vgprs, processor, 32), waves
                    )

    def test_every_modelled_rdna_processor_is_measured(self):
        rdna = {
            processor
            for processor, vgpr_file in occupancy._VGPR_FILES.items()
            if vgpr_file.wave_size == 32
        }
        self.assertEqual(rdna, set(_MEASURED_RDNA_WAVES))

    def test_model_matches_the_local_compiler(self):
        llc = _find_llvm_tool("llc")
        if llc is None:
            self.skipTest("llc not found")
        checked = 0
        for processor in _MEASURED_RDNA_WAVES:
            for vgprs in _VGPR_SWEEP:
                reported = _llc_occupancy(llc, processor, vgprs)
                if reported is None:
                    break  # this llc predates the processor; the pinned test covers it
                checked += 1
                with self.subTest(processor=processor, vgprs=vgprs):
                    self.assertEqual(
                        occupancy._occupancy_estimate(vgprs, processor, 32), reported
                    )
        self.assertGreater(checked, 0)

    def test_code_object_target_selects_the_processor(self):
        # Real code objects: the notes name the processor (a generic target too),
        # and that processor's register file sets the estimate.
        llc = _find_llvm_tool("llc")
        if llc is None or _find_llvm_tool("llvm-readelf") is None:
            self.skipTest("llc or llvm-readelf not found")
        for processor in ("gfx1100", "gfx1102", "gfx11-generic", "gfx12-generic"):
            with self.subTest(processor=processor):
                code_object = _llc_code_object(llc, processor, 128)
                res = occupancy.resources(code_object, "gfx950")
                self.assertEqual(res["target_arch"], processor)
                self.assertEqual(res["vgpr"], 128)
                self.assertEqual(res["occupancy"], _llc_occupancy(llc, processor, 128))


class TestUnmodelledCodeIsRefused(unittest.TestCase):
    """A processor or wave size without recorded data is an error, not a guess."""

    def test_unknown_processor(self):
        with self.assertRaisesRegex(ValueError, "processor 'gfx1030'"):
            occupancy._occupancy_estimate(64, "gfx1030", 32)

    def test_wave64_code_on_a_wave32_register_file(self):
        with self.assertRaisesRegex(ValueError, "wave64 code on gfx1100"):
            occupancy._occupancy_estimate(64, "gfx1100", 64)

    def test_real_code_objects(self):
        llc = _find_llvm_tool("llc")
        if llc is None or _find_llvm_tool("llvm-readelf") is None:
            self.skipTest("llc or llvm-readelf not found")
        unknown = _llc_code_object(llc, "gfx1030", 64)
        with self.assertRaisesRegex(ValueError, "processor 'gfx1030'"):
            occupancy.resources(unknown, "gfx1100")
        wave64 = _llc_code_object(llc, "gfx1100", 64, wave_size=64)
        with self.assertRaisesRegex(ValueError, "wave64 code on gfx1100"):
            occupancy.resources(wave64, "gfx1100")


if __name__ == "__main__":
    unittest.main()
