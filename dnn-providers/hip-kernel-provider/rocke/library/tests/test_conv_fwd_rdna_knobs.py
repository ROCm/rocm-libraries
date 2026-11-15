# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests for the RDNA forward implicit-GEMM knobs: the uniform per-tile
reduction decode (``uniform_k_decode``). Validator gates, kernel names, the
cache identity / label round trip, and that the defaults leave the kernel
name and the identity hash unchanged."""

import unittest

from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
from kernels.common.conv_implicit_gemm import (
    ImplicitGemmConvSpec,
    is_valid_spec,
)
from kernels.conv.kernel_identity import KernelIdentity

_WMMA = "gfx1201"
_MFMA = "gfx950"


def _spec(**kw):
    base = dict(
        problem=ConvProblem(N=2, Hi=16, Wi=16, C=64, K=128, Y=1, X=1, sH=2, sW=2),
        data=ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d="fp16"),
        tile_m=128,
        tile_n=256,
        tile_k=32,
        warp_m=2,
        warp_n=4,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
        wave_size=32,
        pipeline="mem",
        epilogue="cshuffle",
        vector_size_a=8,
        vector_size_b=8,
        vector_size_c=8,
    )
    base.update(kw)
    return ImplicitGemmConvSpec(**base)


def _identity(**kw):
    base = dict(
        arch=_WMMA,
        direction="fwd",
        algorithm="implicit_gemm",
        llvm_flavor="llvm24",
        dtype_a="fp16",
        dtype_b="fp16",
        dtype_d="fp16",
        grouped=True,
        tile_m=128,
        tile_n=256,
        tile_k=32,
        warp_m=2,
        warp_n=4,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
        pipeline="mem",
        epilogue="cshuffle",
        wave_size=32,
        vector_size_a=8,
        vector_size_b=8,
        vector_size_c=8,
    )
    base.update(kw)
    return KernelIdentity(**base)


class TestDefaultsUnchanged(unittest.TestCase):
    def test_kernel_name_has_no_new_tag(self):
        name = _spec().kernel_name()
        for tag in ("ukd",):
            self.assertNotIn(tag, name)

    def test_identity_hash_ignores_defaults(self):
        """The knobs this file covers stay out of the hashed payload at their
        defaults (so existing cache entries keep their hash) and enter it when
        set. The payload is captured from stable_hash itself rather than
        rebuilt from a field list, so identity fields added elsewhere do not
        break the check."""
        import json
        from unittest import mock

        from kernels.conv import kernel_identity as ki

        def payload(ident):
            seen = []
            real = json.dumps

            def capture(obj, *a, **kw):
                seen.append(dict(obj))
                return real(obj, *a, **kw)

            with mock.patch.object(ki.json, "dumps", side_effect=capture):
                ident.stable_hash()
            (fields,) = seen
            return fields

        knobs = {
            "regt_prefetch": 1,
            "uniform_k_decode": True,
            "lds_swizzle": True,
            "group_fast_raster": True,
        }
        at_default = payload(_identity())
        for name, value in knobs.items():
            with self.subTest(field=name):
                self.assertNotIn(name, at_default)
                self.assertEqual(payload(_identity(**{name: value}))[name], value)
                self.assertNotEqual(
                    _identity(**{name: value}).stable_hash(),
                    _identity().stable_hash(),
                )


class TestValidator(unittest.TestCase):
    def test_wmma_accepts(self):
        for kw in (dict(uniform_k_decode=True),):
            ok, why = is_valid_spec(_spec(**kw), _WMMA)
            self.assertTrue(ok, f"{kw}: {why}")

    def test_rejects(self):
        for kw, why in (
            (
                dict(uniform_k_decode=True, pipeline="wavelet", tile_n=64, warp_n=2),
                "mem pipeline",
            ),
        ):
            ok, reason = is_valid_spec(_spec(**kw), _WMMA)
            self.assertFalse(ok, kw)
            self.assertIn(why, reason, kw)


class TestNamesAndLabels(unittest.TestCase):
    def test_kernel_name_tags(self):
        self.assertIn("ukd", _spec(uniform_k_decode=True).kernel_name())

    def test_label_round_trip(self):
        for kw in (dict(uniform_k_decode=True),):
            ident = _identity(**kw)
            self.assertEqual(KernelIdentity.from_label(ident.label()), ident, kw)
            self.assertNotEqual(ident.stable_hash(), _identity().stable_hash())


if __name__ == "__main__":
    unittest.main(verbosity=2)
