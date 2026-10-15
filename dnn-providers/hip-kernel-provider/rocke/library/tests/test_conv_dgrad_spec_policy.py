# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-only policy tests for the implicit-GEMM dgrad spec (no GPU needed).

Covers the validator rules that replaced silently-ignored or unsafe knobs:

  - async_dma / unroll_k / chiplet_swizzle are rejected (the dgrad builder has
    no such path);
  - an accumulator tile above 256 fp32 registers per lane is rejected (it
    fills a wave64's whole register file);
  - compv4 is a schedule policy here and is not charged a second LDS buffer,
    and the LDS charge equals what the builder allocates;
  - the K-outer B fragment rule the builder enforces is in the validator too;
  - a non-default ``lds_k_pad`` reaches the kernel name.
"""

from __future__ import annotations

import unittest
from unittest import mock

import kernels.common.conv_implicit_gemm_dgrad as dgrad_mod
from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_implicit_gemm import ConvDataSpec
from kernels.common.conv_implicit_gemm_dgrad import (
    DgradConvSpec,
    build_implicit_gemm_conv_dgrad,
    dgrad_lds_bytes,
    is_valid_dgrad_spec,
)

_ARCH = "gfx950"


def _problem(**kw):
    base = {
        "N": 2,
        "Hi": 14,
        "Wi": 14,
        "C": 64,
        "K": 64,
        "Y": 3,
        "X": 3,
        "pH": 1,
        "pW": 1,
    }
    base.update(kw)
    return ConvProblem(**base)


def _spec(problem=None, dtype="bf16", **kw):
    base = {
        "problem": problem if problem is not None else _problem(),
        "data": ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype),
        "tile_m": 64,
        "tile_n": 64,
        "tile_k": 64,
        "warp_m": 2,
        "warp_n": 2,
        "warp_tile_m": 32,
        "warp_tile_n": 32,
        "warp_tile_k": 16,
        "pipeline": "mem",
        "epilogue": "cshuffle",
        "lds_k_outer": True,
    }
    base.update(kw)
    return DgradConvSpec(**base)


def _smem_bytes(kernel, spec):
    """(A+B bytes, C bytes) of the kernel's LDS allocations (16-bit dtypes)."""
    ab = c = 0
    for op in kernel.body.ops:
        if op.name != "tile.smem_alloc":
            continue
        shape = op.results[0].type.shape
        n = shape[0] * shape[1] * 2
        if spec.epilogue == "cshuffle" and tuple(shape) == (spec.tile_m, spec.tile_n):
            c += n
        else:
            ab += n
    return ab, c


class TestValidatorRules(unittest.TestCase):
    def test_ignored_knobs_are_rejected(self):
        for kw in ({"async_dma": True}, {"unroll_k": True}, {"chiplet_swizzle": True}):
            with self.subTest(**kw):
                ok, why = is_valid_dgrad_spec(_spec(lds_k_outer=False, **kw), _ARCH)
                self.assertFalse(ok)
                self.assertIn("does not implement", why)

    def test_spilling_accumulator_tile_rejected(self):
        # A 256x256 tile with 1x2 warps of the 16x16x32 atom: 512 fp32
        # accumulators per lane, the whole wave64 register file.
        big = _problem(N=4, Hi=12, Wi=12, C=384, K=384)
        bad = _spec(
            problem=big,
            tile_m=256,
            tile_n=256,
            tile_k=64,
            warp_m=1,
            warp_n=2,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=32,
        )
        ok, why = is_valid_dgrad_spec(bad, _ARCH)
        self.assertFalse(ok)
        self.assertIn("accumulator", why)
        # 256 accumulators per lane (2x2 warps) is the largest accepted.
        ok, why = is_valid_dgrad_spec(
            _spec(
                problem=big,
                tile_m=256,
                tile_n=256,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=32,
            ),
            _ARCH,
        )
        self.assertTrue(ok, why)

    def test_compv4_is_single_buffered(self):
        # A/B single-buffered fits the gfx950 LDS; a doubled charge would not.
        spec = _spec(
            problem=_problem(C=256, K=256),
            tile_m=256,
            tile_n=256,
            tile_k=128,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=32,
            pipeline="compv4",
        )
        ok, why = is_valid_dgrad_spec(spec, _ARCH)
        self.assertTrue(ok, why)
        self.assertEqual(dgrad_lds_bytes(spec), dgrad_lds_bytes(_with(spec, "mem")))

    def test_lds_charge_equals_the_allocation(self):
        p16 = _problem(Hi=16, Wi=16, C=128, K=128)
        for kw in (
            {},
            {"pipeline": "compv4"},
            {"epilogue": "default"},
            {"lds_k_outer": False},
            {"lds_k_outer": False, "lds_k_pad": 16},
            {"tile_m": 128, "warp_tile_m": 16, "warp_tile_n": 16, "warp_tile_k": 32},
        ):
            with self.subTest(**kw):
                spec = _spec(problem=p16, **kw)
                k = build_implicit_gemm_conv_dgrad(spec, arch=_ARCH)
                ab, c = _smem_bytes(k, spec)
                self.assertEqual(dgrad_lds_bytes(spec), max(ab, c))

    def test_kouter_fragment_rule_is_in_the_validator(self):
        # No atom the targets carry violates it today; a fragment of 2
        # elements per lane (half a ds_read_tr16_b64) must be refused by the
        # validator rather than by the builder.
        spec = _spec()
        real = dgrad_mod._resolve_dgrad_op(spec, _ARCH)
        fake = mock.Mock(wraps=real)
        fake.b_frag_len = 2
        with mock.patch.object(dgrad_mod, "_resolve_dgrad_op", return_value=fake):
            ok, why = is_valid_dgrad_spec(spec, _ARCH)
        self.assertFalse(ok)
        self.assertIn("B fragment length", why)
        # M-outer specs do not read through the transpose.
        with mock.patch.object(dgrad_mod, "_resolve_dgrad_op", return_value=fake):
            ok, why = is_valid_dgrad_spec(_spec(lds_k_outer=False), _ARCH)
        self.assertTrue(ok, why)

    def test_every_accepted_kouter_atom_builds(self):
        for wt, wk in ((16, 16), (16, 32), (32, 8), (32, 16)):
            spec = _spec(warp_tile_m=wt, warp_tile_n=wt, warp_tile_k=wk)
            ok, why = is_valid_dgrad_spec(spec, _ARCH)
            with self.subTest(atom=(wt, wk)):
                self.assertTrue(ok, why)
                build_implicit_gemm_conv_dgrad(spec, arch=_ARCH)


def _with(spec, pipeline):
    from dataclasses import replace

    return replace(spec, pipeline=pipeline)


class TestKernelName(unittest.TestCase):
    def test_lds_k_pad_reaches_the_name(self):
        self.assertNotIn("_kp", _spec(lds_k_outer=False).kernel_name())
        # The default pad, spelled out, is the default build.
        self.assertNotIn("_kp", _spec(lds_k_outer=False, lds_k_pad=8).kernel_name())
        self.assertIn("_kp16", _spec(lds_k_outer=False, lds_k_pad=16).kernel_name())
        self.assertIn("_kp0", _spec(lds_k_outer=False, lds_k_pad=0).kernel_name())
        # tile_k < 16 defaults to no pad.
        self.assertNotIn(
            "_kp",
            _spec(
                lds_k_outer=False, tile_k=8, warp_tile_k=8, lds_k_pad=0, dtype="fp16"
            ).kernel_name(),
        )


if __name__ == "__main__":
    unittest.main()
