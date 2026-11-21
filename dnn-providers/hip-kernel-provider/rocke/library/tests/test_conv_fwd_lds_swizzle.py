# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests for the RDNA4 forward implicit-GEMM knobs: the XOR-swizzled
unpadded A/B LDS (``lds_swizzle``): the swizzle's bank mapping, validator gates, LDS
bytes, kernel names, the identity / label round trip, and that the defaults
leave names, hashes and IR untouched."""

import unittest

from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
from kernels.common.conv_implicit_gemm import (
    ImplicitGemmConvSpec,
    build_implicit_gemm_conv,
    implicit_gemm_lds_bytes,
    is_valid_spec,
)
from kernels.conv.kernel_identity import KernelIdentity
from rocke.core.ir_print import print_ir
from rocke.helpers.loads import lds_col_xor_params

_WMMA = "gfx1201"


def _spec(**kw):
    base = dict(
        problem=ConvProblem(N=2, Hi=16, Wi=16, C=64, K=128, Y=3, X=3, pH=1, pW=1),
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


def _chunk(row, col, params):
    shift, mask, log2, flip = params
    return (col ^ ((((row >> shift) & mask) ^ ((row >> flip) & 1)) << log2)) >> log2


class TestSwizzleBanks(unittest.TestCase):
    """Whether an LDS phase serves 8 lanes x 16 bytes (32 banks) or 16 lanes
    x 16 bytes (64 banks): the consecutive rows a fragment read touches at one
    chunk, and the chunks one row-major store phase writes, must land on
    distinct 16-byte bank groups."""

    def test_reads_and_stores_conflict_free(self):
        for lanes in (8, 16):
            for tile_k in (16, 32, 64, 128):
                params = lds_col_xor_params(tile_k)
                chunks = tile_k // 8
                for h in range(chunks):
                    for r0 in range(0, 64, lanes):
                        groups = {
                            (r * chunks + _chunk(r, h * 8, params)) % lanes
                            for r in range(r0, r0 + lanes)
                        }
                        self.assertEqual(len(groups), lanes, (lanes, tile_k, h, r0))
                per_phase = max(1, lanes // chunks)
                for r0 in range(0, 64, per_phase):
                    for c0 in range(0, chunks, lanes // per_phase):
                        groups = {
                            (r * chunks + _chunk(r, c * 8, params)) % lanes
                            for r in range(r0, r0 + per_phase)
                            for c in range(c0, c0 + lanes // per_phase)
                        }
                        self.assertEqual(len(groups), lanes, (lanes, tile_k, r0, c0))

    def test_permutes_within_the_row(self):
        for tile_k in (16, 32, 64, 128):
            params = lds_col_xor_params(tile_k)
            for r in range(32):
                cols = sorted(_chunk(r, c, params) for c in range(0, tile_k, 8))
                self.assertEqual(cols, list(range(tile_k // 8)))


class TestDefaultsUnchanged(unittest.TestCase):
    def test_kernel_name_has_no_new_tag(self):
        name = _spec().kernel_name()
        self.assertNotIn("lxs", name)

    def test_default_ir_has_no_xor(self):
        ir = print_ir(build_implicit_gemm_conv(_spec(), arch=_WMMA))
        self.assertNotIn("arith.xor", ir)
        ir = print_ir(build_implicit_gemm_conv(_spec(lds_swizzle=True), arch=_WMMA))
        self.assertIn("arith.xor", ir)

    def test_identity_hash_neutral(self):
        self.assertEqual(
            _identity(lds_swizzle=False).stable_hash(),
            _identity().stable_hash(),
        )


class TestValidator(unittest.TestCase):
    def test_accepts(self):
        for kw in (
            dict(lds_swizzle=True),
            dict(lds_swizzle=True, tile_k=64),
            dict(lds_swizzle=True, uniform_k_decode=True),
        ):
            ok, why = is_valid_spec(_spec(**kw), _WMMA)
            self.assertTrue(ok, f"{kw}: {why}")

    def test_rejects(self):
        for kw, why in (
            (dict(lds_swizzle=True, lds_k_pad=8), "lds_layout"),
            (
                dict(lds_swizzle=True, pipeline="wavelet", tile_n=64, warp_n=2),
                "mem pipeline",
            ),
        ):
            ok, reason = is_valid_spec(_spec(**kw), _WMMA)
            self.assertFalse(ok, kw)
            self.assertIn(why, reason, kw)
        mfma = dict(wave_size=64, warp_tile_m=32, warp_tile_n=32, warp_tile_k=8)
        ok, _ = is_valid_spec(_spec(lds_swizzle=True, **mfma), "gfx950")
        self.assertFalse(ok)


class TestLds(unittest.TestCase):
    def test_swizzle_drops_the_k_pad(self):
        _, ab_pad, _, _ = implicit_gemm_lds_bytes(_spec())
        _, ab_xor, _, _ = implicit_gemm_lds_bytes(_spec(lds_swizzle=True))
        self.assertEqual(ab_pad, (128 + 256) * (32 + 8) * 2)
        self.assertEqual(ab_xor, (128 + 256) * 32 * 2)


class TestNamesAndLabels(unittest.TestCase):
    def test_kernel_name_tags(self):
        self.assertIn("lxs", _spec(lds_swizzle=True).kernel_name())

    def test_label_round_trip(self):
        for kw in (
            dict(lds_swizzle=True),
            dict(lds_swizzle=True, regt_prefetch=1),
        ):
            ident = _identity(**kw)
            self.assertEqual(KernelIdentity.from_label(ident.label()), ident, kw)
            self.assertNotEqual(ident.stable_hash(), _identity().stable_hash())
        self.assertIn("lxs", _identity(lds_swizzle=True).label())


class TestAotTwins(unittest.TestCase):
    def test_twins_on_rdna_only(self):
        from kernels.conv.aot_jobs import (
            CACHE_FWD_RDNA4_KNOB_ROWS,
            _fwd_jobs,
            _geometries,
        )
        from rocke.core.arch import ArchTarget

        for arch, rdna in (
            ("gfx1201", True),
            ("gfx12-generic", True),
            ("gfx11-generic", False),
            ("gfx1250", False),
        ):
            with self.subTest(arch=arch):
                t = ArchTarget.from_gfx(arch)
                tiles = {tuple(r[:5]) for r in CACHE_FWD_RDNA4_KNOB_ROWS if r[6]}
                geos = [
                    g
                    for g in _geometries(t, "wmma", "fp16", "fp16", "fwd")
                    if tuple(g[:5]) in tiles and g[6] == "mem" and g[7] == "cshuffle"
                ]
                jobs = list(_fwd_jobs(arch, "fp16", 32, "wmma", t, geometries=geos))
                lxs = [j for j in jobs if j.identity.lds_swizzle]
                self.assertEqual(bool(lxs), rdna)
                for j in lxs:
                    self.assertTrue(j.spec_kwargs["lds_swizzle"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
