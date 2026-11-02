# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-only tests for the universal-GEMM verify harness manifest build.

``examples.common.universal_gemm_verify.build_manifest`` is the call site that
carries ``--dtype`` into the manifest. It matters because the failure mode is
silent: ``helpers.manifest.make_gemm_manifest`` falls back to
``gemm_args_signature()`` -- fp16 -- when ``args_signature`` is omitted, and the
manifest ``kind`` is ``gemm_fp16`` for every GEMM regardless of operand dtype.
The element type is carried only by the ``A`` pointer type, which
``manifest_runner.gemm._gemm_is_bf16`` reads back to pick the reference dtype.
So a dropped kwarg produces a well-formed manifest that verifies a bf16 kernel
against an fp16 reference.

These tests assert the round trip -- build the manifest the way the harness
does, then read it back through the real consumer -- with stub tile/spec/
artifact objects, so no compile, GPU, or torch is required. The atom and wave
size come from the arch catalog via the harness's own ``_pick_atom``, since
atom geometry is arch-specific and no single literal stands in for all targets.

The 8-bit GEMMs break three assumptions the fp16/bf16 path was written under, so
they are covered separately below: ``kind`` is ``gemm_fp8`` rather than
``gemm_fp16``, A/B and C no longer share a dtype (fp8 in, bf16 out), and the
atom name carries a ``<a>_<b>`` operand pair rather than one dtype. Each is a
place where falling back to the fp16 default produces a manifest that is
well-formed, routes to a runner that will happily execute, and verifies against
the wrong reference.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from rocke.core.arch import ArchTarget
from rocke.examples.common.universal_gemm_verify import (
    _c_dtype_for,
    _pick_atom,
    build_manifest,
)
from rocke.instances.common.manifest_runner.gemm import (
    _F8_ENCODERS,
    _bf8e5m2_decode_table,
    _bf8e5m2_encode,
    _fp8e4m3_encode,
    _gemm_f8_dtype,
    _gemm_is_bf16,
)

# make_gemm_manifest reads exactly these three attributes off the artifact.
_ARTIFACT = SimpleNamespace(
    kernel_name="ugemm_stub", hsaco_bytes=4096, timings={"total": 1.0}
)
_TILE = SimpleNamespace(tile_m=64, tile_n=64, tile_k=32)
_SPEC = SimpleNamespace(block_size=256)

# The bf16 atom the harness resolves per arch, plus the MMA family its wave
# size implies (wave64 -> mfma, wave32 -> wmma). K width differs by target, so
# a single hardcoded atom would be wrong for most of them. Each entry is
# re-checked against the catalog in test_bf16_atom_matches_arch_catalog.
_ARCH_BF16_ATOM = {
    "gfx942": ("mfma", (16, 16, 16)),
    "gfx950": ("mfma", (16, 16, 32)),
    "gfx1151": ("wmma", (16, 16, 16)),
    "gfx1250": ("wmma", (16, 16, 32)),
}

# The 8-bit atoms, same shape of claim. gfx1250's K=64 atom is the one Phase 0
# verified on silicon; the others are listed because the manifest path is not
# gfx1250-only and a catalog edit that moved them should be visible here.
_ARCH_F8_ATOM = {
    "gfx942": ("mfma", (16, 16, 32)),
    "gfx950": ("mfma", (16, 16, 32)),
    "gfx1250": ("wmma", (16, 16, 64)),
}
# Where the largest-K atom differs by 8-bit dtype: gfx1250 fp8 has the dense
# 16x16x128 f8f6f4 atom, bf8 only the K=64 one.
_ARCH_F8_ATOM_OVERRIDE = {("gfx1250", "fp8e4m3"): (16, 16, 128)}


def _manifest(dtype, arch="gfx950", c_dtype=None):
    """Build a manifest the way ``main`` does, with that arch's own atom."""
    target = ArchTarget.from_gfx(arch)
    return build_manifest(
        _ARTIFACT,
        tile=_TILE,
        spec=_SPEC,
        dtype=dtype,
        shape=(512, 512, 512),
        wave_size=target.wave_size,
        atom=_pick_atom(target, dtype, None),
        c_dtype=c_dtype,
    )


def _ptr_types(manifest):
    sig = manifest["args_signature"]
    return {a["name"]: a["type"] for a in sig if a["name"] in ("A", "B", "C")}


class TestUniversalGemmVerifyManifest(unittest.TestCase):
    def test_bf16_manifest_advertises_bf16_ptrs(self):
        self.assertEqual(
            _ptr_types(_manifest("bf16")),
            {
                "A": "ptr<bf16, global>",
                "B": "ptr<bf16, global>",
                "C": "ptr<bf16, global>",
            },
        )

    def test_fp16_manifest_advertises_f16_ptrs(self):
        self.assertEqual(
            _ptr_types(_manifest("fp16")),
            {
                "A": "ptr<f16, global>",
                "B": "ptr<f16, global>",
                "C": "ptr<f16, global>",
            },
        )

    def test_runner_recovers_dtype_from_manifest(self):
        # The assertion that matters: the manifest the harness writes is read
        # back by the runner as the dtype the user asked for. This is what the
        # pre-fix manifest got wrong -- bf16 in, fp16 reference out.
        for arch in _ARCH_BF16_ATOM:
            with self.subTest(arch=arch):
                self.assertTrue(_gemm_is_bf16(_manifest("bf16", arch=arch)))
                self.assertFalse(_gemm_is_bf16(_manifest("fp16", arch=arch)))

    def test_kind_is_gemm_fp16_for_both_dtypes(self):
        # Pins why args_signature has to carry the dtype: ``kind`` cannot.
        for dtype in ("fp16", "bf16"):
            with self.subTest(dtype=dtype):
                self.assertEqual(_manifest(dtype)["kind"], "gemm_fp16")

    def test_bf16_atom_matches_arch_catalog(self):
        # Two claims: the catalog still resolves the atom this table records,
        # and build_manifest formats that atom -- not a stale or reordered one
        # -- into the atoms entry, with the family its wave size implies.
        for arch, (family, atom) in _ARCH_BF16_ATOM.items():
            with self.subTest(arch=arch):
                target = ArchTarget.from_gfx(arch)
                self.assertEqual(_pick_atom(target, "bf16", None), atom)
                m, n, k = atom
                self.assertEqual(
                    _manifest("bf16", arch=arch)["atoms"][0],
                    f"{family}_f32_{m}x{n}x{k}_bf16",
                )


class TestEightBitManifest(unittest.TestCase):
    def test_fp8_manifest_pairs_fp8_operands_with_a_bf16_c(self):
        # The mixed ABI in one assertion. fp8 has no accumulator range to spare,
        # so C is bf16; if C ever came back as ptr<fp8e4m3, global> the kernel
        # would write bytes the runner then decodes as e4m3 and compares against
        # an fp32-accumulated reference.
        self.assertEqual(
            _ptr_types(_manifest("fp8e4m3", arch="gfx1250")),
            {
                "A": "ptr<fp8e4m3, global>",
                "B": "ptr<fp8e4m3, global>",
                "C": "ptr<bf16, global>",
            },
        )

    def test_bf8_manifest_pairs_bf8_operands_with_a_bf16_c(self):
        self.assertEqual(
            _ptr_types(_manifest("bf8e5m2", arch="gfx1250")),
            {
                "A": "ptr<bf8e5m2, global>",
                "B": "ptr<bf8e5m2, global>",
                "C": "ptr<bf16, global>",
            },
        )

    def test_eight_bit_c_is_widened_and_others_are_left_alone(self):
        # _c_dtype_for is the single place the widening decision is made, and it
        # must not widen anything else: a bf16 GEMM silently acquiring a
        # different C dtype would be the same bug in the other direction.
        for dtype in ("fp8e4m3", "bf8e5m2"):
            self.assertEqual(_c_dtype_for(dtype), "bf16")
        for dtype in ("fp16", "bf16"):
            self.assertEqual(_c_dtype_for(dtype), dtype)

    def test_eight_bit_c_can_be_fp16(self):
        # --c-dtype fp16 must reach the manifest's C pointer, or the runner
        # would decode an f16 kernel's output as bf16.
        for dtype in ("fp8e4m3", "bf8e5m2"):
            self.assertEqual(_c_dtype_for(dtype, "fp16"), "fp16")
            types = _ptr_types(_manifest(dtype, arch="gfx1250", c_dtype="fp16"))
            self.assertEqual(types["C"], "ptr<f16, global>")
            self.assertEqual(types["A"], f"ptr<{dtype}, global>")

    def test_c_dtype_override_is_refused_where_it_does_not_apply(self):
        with self.assertRaises(ValueError):
            _c_dtype_for("fp8e4m3", "fp8e4m3")  # no 8-bit C: no fp8 accumulator
        with self.assertRaises(ValueError):
            _c_dtype_for("fp16", "bf16")  # 16-bit GEMMs are homogeneous
        self.assertEqual(_c_dtype_for("bf16", "bf16"), "bf16")

    def test_kind_is_gemm_fp8_not_the_fp16_default(self):
        # make_gemm_manifest defaults kind to gemm_fp16. That default is the
        # failure this test exists for: an fp8 manifest carrying it would route
        # to the fp16 runner, which reads A as two-byte elements and would
        # mis-shape the operand before any comparison happened.
        for dtype in ("fp8e4m3", "bf8e5m2"):
            with self.subTest(dtype=dtype):
                self.assertEqual(_manifest(dtype, arch="gfx1250")["kind"], "gemm_fp8")

    def test_the_bf16_sniff_does_not_claim_eight_bit_manifests(self):
        # _gemm_is_bf16 keys off the A pointer type, and the fp8 manifest's C is
        # bf16. If the sniff ever moved to C, every fp8 GEMM would be handed the
        # bf16 reference path.
        for dtype in ("fp8e4m3", "bf8e5m2"):
            with self.subTest(dtype=dtype):
                self.assertFalse(_gemm_is_bf16(_manifest(dtype, arch="gfx1250")))

    def test_atom_name_carries_both_operand_dtypes(self):
        # The 8-bit atoms are named <a>_<b> because the hardware op takes a
        # mixed pair; the single-dtype spelling used for fp16/bf16 resolves to
        # no atom at all. Also re-checks the catalog still returns these shapes.
        for arch, (family, atom) in _ARCH_F8_ATOM.items():
            for dtype in ("fp8e4m3", "bf8e5m2"):
                with self.subTest(arch=arch, dtype=dtype):
                    target = ArchTarget.from_gfx(arch)
                    want = _ARCH_F8_ATOM_OVERRIDE.get((arch, dtype), atom)
                    self.assertEqual(_pick_atom(target, dtype, None), want)
                    m, n, k = want
                    self.assertEqual(
                        _manifest(dtype, arch=arch)["atoms"][0],
                        f"{family}_f32_{m}x{n}x{k}_{dtype}_{dtype}",
                    )

    def test_gfx1250_eight_bit_atom_is_twice_the_k_of_its_bf16_atom(self):
        # Half-width operands, double the K per instruction. This is the whole
        # reason the fp8 path needed its own lane map rather than reusing the
        # bf16 one, so it is asserted rather than assumed. bf8 has only this
        # K=64 atom; fp8 additionally has the dense f8f6f4 one at four times.
        target = ArchTarget.from_gfx("gfx1250")
        bf16_k = _pick_atom(target, "bf16", None)[2]
        self.assertEqual(_pick_atom(target, "bf8e5m2", None)[2], 2 * bf16_k)
        self.assertEqual(_pick_atom(target, "fp8e4m3", None)[2], 4 * bf16_k)


class TestEightBitEncoders(unittest.TestCase):
    """The gemm_fp8 runner must hand each kernel bytes in its own format.

    A bf8 kernel fed e4m3 bytes decodes nonsense, so a correct kernel fails
    nearly every element and the verify lane cannot pass for bf8 at all.
    """

    def test_e5m2_table_matches_the_format(self):
        import numpy as np

        t = _bf8e5m2_decode_table(np)
        for code, want in (
            (0x3C, 1.0),
            (0xBC, -1.0),
            (0x45, 5.0),
            (0x7B, 57344.0),  # largest finite
            (0x04, 2.0**-14),  # smallest normal
            (0x01, 2.0**-16),  # smallest subnormal
        ):
            with self.subTest(code=hex(code)):
                self.assertEqual(float(t[code]), want)
        # e == 0x1F is inf / NaN; none of those codes may be encodable.
        self.assertTrue(np.isnan(t[0x7C:0x80]).all())
        self.assertTrue(np.isnan(t[0xFC:0x100]).all())

    def test_e5m2_encode_round_trips_every_finite_code(self):
        import numpy as np

        t = _bf8e5m2_decode_table(np)
        finite = np.flatnonzero(np.isfinite(t) & (np.arange(256) != 0x80))
        codes = _bf8e5m2_encode(np, t[finite])
        self.assertTrue(np.array_equal(codes, finite.astype(np.uint8)))

    def test_encoders_reject_what_their_format_cannot_hold(self):
        import numpy as np

        # 1.125 needs three significand bits: exact in e4m3, not in e5m2.
        x = np.array([1.125], dtype=np.float32)
        self.assertEqual(int(_fp8e4m3_encode(np, x)[0]), 0x39)
        with self.assertRaisesRegex(ValueError, "bf8e5m2"):
            _bf8e5m2_encode(np, x)

    def test_test_inputs_are_exact_in_both_formats(self):
        import numpy as np

        x = np.arange(-5, 6, dtype=np.float32)
        for dtype, encode in _F8_ENCODERS.items():
            with self.subTest(dtype=dtype):
                encode(np, x)  # raises if any input is inexact

    def test_runner_picks_the_encoder_from_the_manifest(self):
        for dtype in ("fp8e4m3", "bf8e5m2"):
            with self.subTest(dtype=dtype):
                self.assertEqual(
                    _gemm_f8_dtype(_manifest(dtype, arch="gfx1250")), dtype
                )
        with self.assertRaisesRegex(ValueError, "A pointer type"):
            _gemm_f8_dtype(_manifest("bf16", arch="gfx1250"))

    def test_each_dtype_maps_to_its_own_encoder(self):
        import numpy as np

        # 1.0 is 0x38 in e4m3 and 0x3C in e5m2; the mapping is the bug.
        one = np.array([1.0], dtype=np.float32)
        for dtype, code in (("fp8e4m3", 0x38), ("bf8e5m2", 0x3C)):
            with self.subTest(dtype=dtype):
                self.assertEqual(int(_F8_ENCODERS[dtype](np, one)[0]), code)


if __name__ == "__main__":
    unittest.main()
