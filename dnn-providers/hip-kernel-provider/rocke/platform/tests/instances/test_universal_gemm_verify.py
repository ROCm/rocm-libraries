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
from rocke.instances.common.manifest_runner.gemm import _gemm_is_bf16

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


def _manifest(dtype, arch="gfx950"):
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
                    self.assertEqual(_pick_atom(target, dtype, None), atom)
                    m, n, k = atom
                    self.assertEqual(
                        _manifest(dtype, arch=arch)["atoms"][0],
                        f"{family}_f32_{m}x{n}x{k}_{dtype}_{dtype}",
                    )

    def test_gfx1250_eight_bit_atom_is_twice_the_k_of_its_bf16_atom(self):
        # Half-width operands, double the K per instruction. This is the whole
        # reason the fp8 path needed its own lane map rather than reusing the
        # bf16 one, so it is asserted rather than assumed.
        target = ArchTarget.from_gfx("gfx1250")
        self.assertEqual(
            _pick_atom(target, "fp8e4m3", None)[2],
            2 * _pick_atom(target, "bf16", None)[2],
        )


if __name__ == "__main__":
    unittest.main()
