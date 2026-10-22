# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the occupancy primitive's pure logic (no GPU)."""
import unittest
from types import SimpleNamespace

from rocke.benchmark.perf import occupancy
from rocke.tests.arch_marks import arch_specific
from rocke.tests.llc import find_llvm_tool, llc_code_object, llc_occupancy

_CDNA_ARCHS = ("gfx942", "gfx950")


class TestResourcesSourceTag(unittest.TestCase):
    ARCH = "gfx950"

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
        res = occupancy.resources(b"fake", self.ARCH)
        self.assertEqual(res["source"], "elf_notes")  # distinguishes from rocprofv3
        self.assertEqual(res["vgpr"], 24)
        self.assertIsNotNone(res["occupancy"])

    def test_empty_notes_returns_empty(self):
        occupancy.parse_notes = lambda b: {}
        self.assertEqual(occupancy.resources(b"x", self.ARCH), {})

    def test_elf_target_overrides_a_stale_arch_flag(self):
        # Real gfx1201 binary, caller passed a CDNA --arch: the RDNA occupancy model
        # must win, otherwise the wrong wave count is reported with no signal.
        occupancy.parse_notes = lambda b: {
            "vgpr": 111,
            "target": "gfx1201",
            "wave_size": 32,
        }
        res = occupancy.resources(b"fake", self.ARCH)
        self.assertEqual(res["target_arch"], "gfx1201")
        self.assertEqual(
            res["occupancy"], occupancy._occupancy_estimate(111, "gfx1201", 32)
        )
        self.assertNotEqual(
            res["occupancy"], occupancy._occupancy_estimate(111, self.ARCH, 64)
        )

    def test_arch_argument_is_the_fallback_when_notes_lack_a_target(self):
        occupancy.parse_notes = lambda b: {"vgpr": 111, "wave_size": 64}
        res = occupancy.resources(b"fake", self.ARCH)
        self.assertEqual(res["target_arch"], self.ARCH)
        self.assertEqual(
            res["occupancy"], occupancy._occupancy_estimate(111, self.ARCH, 64)
        )


class TestResourcesSourceTagGfx942(TestResourcesSourceTag):
    ARCH = "gfx942"


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
        for arch in _CDNA_ARCHS:
            with self.subTest(arch=arch):
                res = occupancy.resources(b"fake", arch)  # deliberately wrong flag
                self.assertEqual(res["target_arch"], "gfx1201")
                self.assertEqual(res["occupancy"], 12)  # RDNA model, not the CDNA 4

    def _patch_target(self, target):
        notes = self._NOTES.replace("amdgcn-amd-amdhsa--gfx1201", target)
        occupancy.subprocess.run = lambda *a, **k: SimpleNamespace(
            returncode=0, stdout=notes, stderr=""
        )

    def test_the_recorded_target_names_the_processor(self):
        # The note's value is read with rocke.core.arch.parse_target_id, whose
        # tests cover every spelling; here, the processor reaches the fields.
        self._patch_target("amdgcn-amd-amdhsa-unknown-gfx950:xnack+")
        self.assertEqual(occupancy.parse_notes(b"fake")["target"], "gfx950")

    def test_a_recorded_target_that_is_not_a_target_id_is_refused(self):
        # Never fall back to the caller's arch for a target that cannot be read,
        # nor take a processor name from the wrong place in the value.
        refusal = r"amdhsa\.target note: .* is not a target ID"
        for target in ("''", "notgfx950", "amdgcn-amd-amdhsa-unknown-:gfx950"):
            with self.subTest(target=target):
                self._patch_target(target)
                with self.assertRaisesRegex(ValueError, refusal):
                    occupancy.parse_notes(b"fake")
                with self.assertRaisesRegex(ValueError, refusal):
                    occupancy.resources(b"fake", "gfx950")
                with self.assertRaisesRegex(ValueError, refusal):
                    occupancy.estimate_occupancy_detail(b"fake", "gfx950")


class TestOccupancyEstimate(unittest.TestCase):
    def test_estimate_is_capped(self):
        # tiny VGPR -> capped at max_waves_per_simd, not unbounded
        for arch in _CDNA_ARCHS:
            with self.subTest(arch=arch):
                est = occupancy._occupancy_estimate(4, arch, 64)
                self.assertEqual(est, 8)  # max_waves_per_simd

    def test_zero_vgpr_none(self):
        for arch in _CDNA_ARCHS:
            with self.subTest(arch=arch):
                self.assertIsNone(occupancy._occupancy_estimate(0, arch, 64))

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
        # 256 VGPR on a CDNA arch -> 512//256 = 2 waves/SIMD; LDS/AGPR slack.
        self._patch(vgpr=256, agpr=0, lds_bytes=2048, max_flat_workgroup_size=256)
        for arch in _CDNA_ARCHS:
            with self.subTest(arch=arch):
                det = occupancy.estimate_occupancy_detail(b"x", arch)
                self.assertEqual(det["waves_per_simd"], 2)
                self.assertEqual(det["limited_by"], "VGPR")

    @arch_specific(
        ("gfx950",), reason="163840 B LDS per CU gives 2 workgroups; gfx942 has 65536 B"
    )
    def test_lds_limited(self):
        # low VGPR (8 waves) but a fat 80 KB workgroup -> LDS binds below regs.
        self._patch(vgpr=64, agpr=0, lds_bytes=80000, max_flat_workgroup_size=512)
        det = occupancy.estimate_occupancy_detail(b"x", "gfx950")
        self.assertEqual(det["limited_by"], "LDS")
        self.assertEqual(det["waves_per_simd"], 4)  # (163840//80000=2 wg) * 8 / 4

    @arch_specific(
        ("gfx942",),
        reason="gfx942 keeps separate VGPR and AGPR pools; gfx950 combines them",
    )
    def test_agpr_limited_separate_pool(self):
        # gfx942 (CDNA3) keeps separate VGPR/AGPR pools -> heavy accumulator use
        # starves waves before VGPR does.
        self._patch(vgpr=64, agpr=200, lds_bytes=2048, max_flat_workgroup_size=256)
        det = occupancy.estimate_occupancy_detail(b"x", "gfx942")
        self.assertEqual(det["limited_by"], "AGPR")

    @arch_specific(
        ("gfx950",), reason="gfx950 shares one VGPR+AGPR register file; gfx942 does not"
    )
    def test_gfx950_combined_reg_pool(self):
        # gfx950 (CDNA4): VGPR+AGPR share one file, so the SUM sets the wave count.
        # Calibrated vs gfx950 hardware MeanOccupancyPerCU: an AGPR-heavy kernel (44 VGPR +
        # 132 AGPR) measured ~7 waves/CU. 512 // align_up(176,16)=176 -> 2/SIMD
        # (8/CU); a separate 256-AGPR pool would wrongly give 1/SIMD (4/CU).
        self._patch(vgpr=44, agpr=132, lds_bytes=24576, max_flat_workgroup_size=256)
        det = occupancy.estimate_occupancy_detail(b"x", "gfx950")
        self.assertEqual(det["waves_per_simd"], 2)
        self.assertEqual(det["limited_by"], "VGPR+AGPR")

    @arch_specific(
        ("gfx950",), reason="expected waves come from the 163840 B LDS per CU of gfx950"
    )
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

    def test_code_object_target_selects_the_caps(self):
        # A gfx942 code object (64 VGPRs, 32 KiB LDS, 256-lane workgroup) with a
        # stale gfx950 argument: gfx942's 64 KiB LDS fits 2 workgroups of 4 waves,
        # 2 waves per SIMD. gfx950's 160 KiB would give 5.
        self._patch(
            vgpr=64,
            agpr=0,
            lds_bytes=32768,
            max_flat_workgroup_size=256,
            target="gfx942",
        )
        det = occupancy.estimate_occupancy_detail(b"x", "gfx950")
        self.assertEqual(det["waves_per_simd"], 2)
        self.assertEqual(det["limited_by"], "LDS")

    def test_code_object_target_without_caps_returns_empty(self):
        # An RDNA code object has no multi-limiter caps, whatever the argument says.
        self._patch(vgpr=64, lds_bytes=2048, target="gfx1201")
        self.assertEqual(occupancy.estimate_occupancy_detail(b"x", "gfx950"), {})

    def test_empty_notes_returns_empty(self):
        self._patch()  # {}
        for arch in _CDNA_ARCHS:
            with self.subTest(arch=arch):
                self.assertEqual(occupancy.estimate_occupancy_detail(b"x", arch), {})

    def test_missing_required_field_returns_empty(self):
        # A partial parse (e.g. the vgpr regex stopped matching) must return {},
        # not clamp the missing vgpr to 1 and report the arch maximum.
        for arch in _CDNA_ARCHS:
            with self.subTest(arch=arch):
                self._patch(lds_bytes=2048, agpr=0)  # no vgpr
                self.assertEqual(occupancy.estimate_occupancy_detail(b"x", arch), {})
                self._patch(vgpr=64)  # no lds_bytes
                self.assertEqual(occupancy.estimate_occupancy_detail(b"x", arch), {})


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
        llc = find_llvm_tool("llc")
        if llc is None:
            self.skipTest("llc not found")
        checked = 0
        for processor in _MEASURED_RDNA_WAVES:
            for vgprs in _VGPR_SWEEP:
                reported = llc_occupancy(llc, processor, vgprs)
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
        llc = find_llvm_tool("llc")
        if llc is None or find_llvm_tool("llvm-readelf") is None:
            self.skipTest("llc or llvm-readelf not found")
        for processor in ("gfx1100", "gfx1102", "gfx11-generic", "gfx12-generic"):
            for arch in _CDNA_ARCHS:
                with self.subTest(processor=processor, arch=arch):
                    code_object = llc_code_object(llc, processor, 128)
                    res = occupancy.resources(code_object, arch)
                    self.assertEqual(res["target_arch"], processor)
                    self.assertEqual(res["vgpr"], 128)
                    self.assertEqual(
                        res["occupancy"], llc_occupancy(llc, processor, 128)
                    )


class TestUnmodelledCodeIsRefused(unittest.TestCase):
    """A processor or wave size without recorded data is an error, not a guess."""

    def test_unknown_processor(self):
        with self.assertRaisesRegex(ValueError, "processor 'gfx1030'"):
            occupancy._occupancy_estimate(64, "gfx1030", 32)

    def test_wave64_code_on_a_wave32_register_file(self):
        with self.assertRaisesRegex(ValueError, "wave64 code on gfx1100"):
            occupancy._occupancy_estimate(64, "gfx1100", 64)

    def test_real_code_objects(self):
        llc = find_llvm_tool("llc")
        if llc is None or find_llvm_tool("llvm-readelf") is None:
            self.skipTest("llc or llvm-readelf not found")
        unknown = llc_code_object(llc, "gfx1030", 64)
        with self.assertRaisesRegex(ValueError, "processor 'gfx1030'"):
            occupancy.resources(unknown, "gfx1100")
        wave64 = llc_code_object(llc, "gfx1100", 64, wave_size=64)
        with self.assertRaisesRegex(ValueError, "wave64 code on gfx1100"):
            occupancy.resources(wave64, "gfx1100")


if __name__ == "__main__":
    unittest.main()
