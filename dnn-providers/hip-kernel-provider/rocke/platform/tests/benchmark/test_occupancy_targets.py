# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The occupancy model covers every target rocKE lowers for, gfx908 included.

The gfx908 cases check the model against the compiler's own occupancy report,
read from kernels built with the ``rocke.tests.llc`` helpers.
"""

import unittest

from rocke.benchmark.perf import occupancy
from rocke.core.arch import ArchTarget, known_arches
from rocke.tests.llc import llc_code_object, llc_occupancy, require_amdgpu_llc

# gfx908 register counts at the edges of every wave count, and the waves per SIMD
# the compiler reports at each (separate 256-entry VGPR and AGPR files, steps of 4,
# at most 10 waves).
_GFX908_REGISTER_SWEEP = (1, 24, 25, 28, 29, 32, 33, 36, 37, 40, 41, 48, 49, 64, 65)
_GFX908_REGISTER_SWEEP += (84, 85, 128, 129, 256)
_GFX908_WAVES = (10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1)
# (VGPRs, AGPRs) used by a kernel: VGPRs only, AGPRs only, and mixed. The code
# object's ``.vgpr_count`` is the larger of the two.
_GFX908_REGISTER_USES = (
    (64, 0),
    (65, 0),
    (1, 64),
    (1, 200),
    (10, 64),
    (100, 20),
    (129, 128),
    (256, 256),
)


class TestGfx908RegisterFiles(unittest.TestCase):
    """The occupancy estimates on gfx908 match the compiler.

    gfx908 has separate VGPR and AGPR files of 256 per SIMD each, both allocated in
    steps of 4, at most 10 waves. Measured with the compiler's own occupancy report
    for every VGPR count 1..256, every AGPR count 1..256 and a grid of mixed uses at
    clang 20:

        llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -o - kernel.ll | grep Occupancy

    where kernel.ll is ``_vgpr_kernel_ir(v, agprs=a)``.
    """

    def test_model_matches_the_measured_wave_counts(self):
        for count, waves in zip(_GFX908_REGISTER_SWEEP, _GFX908_WAVES):
            with self.subTest(count=count):
                self.assertEqual(
                    occupancy._occupancy_estimate(count, "gfx908", 64), waves
                )

    def test_code_objects_match_the_local_compiler(self):
        # Real code objects, with and without AGPRs: both the VGPR-file estimate
        # and the multi-limiter model read from the notes equal the wave count llc
        # reports for the same kernel.
        llc = require_amdgpu_llc()
        for vgprs, agprs in _GFX908_REGISTER_USES:
            with self.subTest(vgprs=vgprs, agprs=agprs):
                code_object = llc_code_object(
                    llc, "gfx908", vgprs, wave_size=64, agprs=agprs
                )
                reported = llc_occupancy(llc, "gfx908", vgprs, agprs)
                res = occupancy.resources(code_object, "gfx1100")
                self.assertEqual(res["target_arch"], "gfx908")
                self.assertEqual(res["occupancy"], reported)
                detail = occupancy.estimate_occupancy_detail(code_object, "gfx908")
                self.assertEqual(detail["waves_per_simd"], reported)


class TestEveryTargetIsModelled(unittest.TestCase):
    """Every target rocKE lowers for has a register file in the occupancy model at
    its wave size, so its VGPR-limited estimate is a number, never a gap. Every
    wave64 target also has the caps of the multi-limiter model, which models
    wave64 code only (for any other target it returns an empty result)."""

    def test_every_known_target_has_a_register_file(self):
        for arch in known_arches():
            target = ArchTarget.from_gfx(arch)
            with self.subTest(arch=arch):
                self.assertIn(arch, occupancy._VGPR_FILES)
                self.assertEqual(
                    occupancy._VGPR_FILES[arch].wave_size, target.wave_size
                )
                if target.wave_size == 64:
                    self.assertIn(arch, occupancy._ARCH_CAPS)


if __name__ == "__main__":
    unittest.main()
