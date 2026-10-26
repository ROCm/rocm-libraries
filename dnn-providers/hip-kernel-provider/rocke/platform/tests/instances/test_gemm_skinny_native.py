# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gemm_skinny with MN-contiguous operands (layouts RRR / CRR / CCR) and the
both-engine emission of the family (CPU only).

GPU numerics of the native forms (random and integer-exact data, odd leading
dimensions, odd K, empty split slices, guard-page buffers, counter self-reset,
run-to-run determinism) are covered by the GPU harness on gfx950
and gfx942; these tests pin the static contract.
"""

from __future__ import annotations

import dataclasses
import os
import unittest
from unittest import mock

from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.instances.common.gemm_skinny import (
    SkinnyGemmSpec,
    build_gemm_skinny,
    is_valid_spec,
    skinny_launch_error,
    skinny_signature,
)

_NATIVE = ("RRR", "CRR", "CCR")


def _llvm(spec, arch="gfx950"):
    return lower_kernel_to_llvm(build_gemm_skinny(spec, arch), arch=arch)


def _engine_available() -> bool:
    try:
        import rocke_engine  # type: ignore # noqa: F401
    except ImportError:
        return False
    return True


class TestNativeValidity(unittest.TestCase):
    def test_layouts_and_operands(self):
        self.assertEqual(SkinnyGemmSpec().mn_operands, (False, False))
        self.assertEqual(SkinnyGemmSpec(layout="RRR").mn_operands, (False, True))
        self.assertEqual(SkinnyGemmSpec(layout="CRR").mn_operands, (True, True))
        self.assertEqual(SkinnyGemmSpec(layout="CCR").mn_operands, (True, False))
        for lay in _NATIVE:
            with self.subTest(lay=lay):
                self.assertTrue(is_valid_spec(SkinnyGemmSpec(layout=lay), "gfx950")[0])
                self.assertTrue(
                    is_valid_spec(SkinnyGemmSpec(layout=lay, mn_lds="u16"), "gfx942")[0]
                )

    def test_rejects(self):
        bad = (
            ("gfx950", dict(layout="RRC")),
            ("gfx950", dict(layout="RCR", mnalign=2)),  # unused MN levers
            ("gfx950", dict(layout="RCR", mn_pad=16)),
            ("gfx950", dict(layout="RCR", mn_lds="u16")),
            ("gfx950", dict(layout="RRR", kmap="lane")),
            ("gfx950", dict(layout="RRR", b_load="global")),
            ("gfx950", dict(layout="CRR", kalign=2)),  # no K-contiguous operand
            ("gfx950", dict(layout="CCR", mnalign=4)),
            ("gfx950", dict(layout="CCR", mn_pad=8)),
            ("gfx950", dict(layout="CCR", mn_lds="b8")),
            ("gfx942", dict(layout="CCR")),  # no transpose read on gfx942
        )
        for arch, kw in bad:
            with self.subTest(arch=arch, kw=kw):
                self.assertFalse(is_valid_spec(SkinnyGemmSpec(**kw), arch)[0])

    def test_k_contiguous_b_keeps_its_load_levers(self):
        # CCR: B is K-contiguous, so b_load / kalign stay available
        self.assertTrue(
            is_valid_spec(SkinnyGemmSpec(layout="CCR", b_load="global"), "gfx950")[0]
        )
        self.assertTrue(
            is_valid_spec(SkinnyGemmSpec(layout="CCR", kalign=1), "gfx950")[0]
        )

    def test_lds_staging_is_budgeted(self):
        # 4 waves, both operands MN-contiguous, 64-wide wave extents, 2 pieces
        # per step: the staging does not fit gfx942's 64 KiB with the pad
        spec = SkinnyGemmSpec(
            layout="CRR",
            wave_m_atoms=4,
            wave_n_atoms=4,
            warp_m=2,
            warp_n=2,
            kc=16,
            mn_pad=16,
            mn_lds="u16",
        )
        ok, why = is_valid_spec(spec, "gfx942")
        self.assertFalse(ok)
        self.assertIn("LDS", why)
        self.assertTrue(
            is_valid_spec(dataclasses.replace(spec, kc=8, mn_pad=0), "gfx942")[0]
        )


class TestNativeLaunchContract(unittest.TestCase):
    def test_mn_leading_dimension_alignment(self):
        s8 = SkinnyGemmSpec(layout="RRR")
        self.assertIsNone(skinny_launch_error(s8, 7, 4096, 4096, 1))
        self.assertIn("mnalign", skinny_launch_error(s8, 7, 4098, 4096, 1))
        s2 = dataclasses.replace(s8, mnalign=2)
        self.assertIsNone(skinny_launch_error(s2, 7, 4098, 4096, 1))
        self.assertIn("mnalign", skinny_launch_error(s2, 7, 4097, 4096, 1))
        s1 = dataclasses.replace(s8, mnalign=1)
        self.assertIsNone(skinny_launch_error(s1, 7, 4097, 4096, 1))
        # A is MN-contiguous in CRR / CCR: M is its leading dimension
        a8 = SkinnyGemmSpec(layout="CCR")
        self.assertIn("M=", skinny_launch_error(a8, 4097, 64, 4096, 1))
        self.assertIsNone(skinny_launch_error(a8, 4096, 65, 4096, 1))

    def test_k_alignment_only_for_k_contiguous_operands(self):
        self.assertIn(
            "kalign", skinny_launch_error(SkinnyGemmSpec(layout="RRR"), 8, 8, 9, 1)
        )
        self.assertIsNone(skinny_launch_error(SkinnyGemmSpec(layout="CRR"), 8, 8, 9, 1))

    def test_slice_offsets_fit_i32(self):
        # one slice of 65536 K rows x 32768 columns overflows the i32 offsets;
        # a split that shortens the slice brings it back
        s = SkinnyGemmSpec(layout="RRR", wave_n_atoms=4)
        self.assertIn("i32", skinny_launch_error(s, 16, 32768, 65536, 1))
        self.assertIsNone(skinny_launch_error(s, 16, 32768, 65536, 8))

    def test_same_abi_as_rcr(self):
        for lay in _NATIVE:
            self.assertEqual(
                skinny_signature(SkinnyGemmSpec(layout=lay)),
                skinny_signature(SkinnyGemmSpec()),
            )


class TestNativeEmission(unittest.TestCase):
    def test_names_distinguish_the_native_levers(self):
        base = SkinnyGemmSpec(layout="CRR")
        names = {SkinnyGemmSpec().kernel_name(), base.kernel_name()}
        for kw in (
            dict(layout="RRR"),
            dict(layout="CCR"),
            dict(mnalign=2),
            dict(mnalign=1),
            dict(mn_pad=16),
            dict(mn_pad=32),
            dict(mn_lds="u16"),
        ):
            n = dataclasses.replace(base, **kw).kernel_name()
            self.assertNotIn(n, names, kw)
            names.add(n)

    def test_rcr_names_unchanged(self):
        # the RCR kernel names (and so their caches / tables) are unchanged
        self.assertEqual(
            SkinnyGemmSpec().kernel_name(),
            "rocke_gemm_skinny_bf16_a1x4_w1x1x1_kc16_pf2",
        )

    def test_transpose_read_per_mode(self):
        tr = _llvm(SkinnyGemmSpec(layout="CRR"))
        self.assertIn("ds.read.tr16.b64", tr)
        u16 = _llvm(SkinnyGemmSpec(layout="CRR", mn_lds="u16"))
        self.assertNotIn("ds.read.tr16.b64", u16)
        u16_942 = _llvm(SkinnyGemmSpec(layout="RRR", mn_lds="u16"), "gfx942")
        self.assertNotIn("ds.read.tr16.b64", u16_942)
        self.assertNotIn("ds.read.tr16.b64", _llvm(SkinnyGemmSpec()))

    def test_odd_leading_dimension_realigns(self):
        self.assertIn("alignbyte", _llvm(SkinnyGemmSpec(layout="RRR", mnalign=1)))
        self.assertNotIn("alignbyte", _llvm(SkinnyGemmSpec(layout="RRR")))

    def test_split_is_launch_only(self):
        base = SkinnyGemmSpec(layout="CCR", wave_n_atoms=2, warp_k=2, red_sync="sc1")
        self.assertEqual(_llvm(base), _llvm(dataclasses.replace(base, split=9)))


@unittest.skipUnless(_engine_available(), "rocke_engine (C++ engine) is not built")
class TestBothEngines(unittest.TestCase):
    """The C++ engine lowers every gemm_skinny form byte-identically, incl. the
    scalar ``tile.buffer_load`` of the kalign-1 / mnalign-1 loads."""

    def test_python_and_cpp_agree(self):
        specs = []
        for kalign in (1, 2, 8):
            specs.append(SkinnyGemmSpec(kalign=kalign, red_sync="sc1"))
        for lay in _NATIVE:
            for mna in (1, 2, 8):
                specs.append(
                    SkinnyGemmSpec(layout=lay, mnalign=mna, mn_pad=16, warp_k=2)
                )
        specs.append(SkinnyGemmSpec(layout="CCR", kalign=1, red_sync="agent"))
        with mock.patch.dict(os.environ, {"ROCKE_BACKEND": "both"}):
            for spec in specs:
                for arch in ("gfx950", "gfx942"):
                    s = (
                        spec
                        if arch == "gfx950" or spec.layout == "RCR"
                        else (dataclasses.replace(spec, mn_lds="u16"))
                    )
                    with self.subTest(arch=arch, name=s.kernel_name()):
                        self.assertTrue(_llvm(s, arch))


if __name__ == "__main__":
    unittest.main()
