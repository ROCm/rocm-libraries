# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Support predicate tests for GEMM dispatcher candidates."""

from __future__ import annotations

import unittest
from dataclasses import replace

from rocke.core.arch import ArchTarget
from rocke.dispatch import GemmRequest, dispatch_gemm_fp16
from rocke.dispatch.gemm.support import (
    GemmSupportQuery,
    gemm_config_supported,
    request_shape_supported,
    support_query_from_universal_spec,
)
from rocke.tests.arch_marks import COVERED_ARCHES


def _warp_tile_32x32(arch: str) -> tuple[int, int, int]:
    """The 32x32 fp16 MFMA warp tile that the arch catalog accepts (K differs by arch)."""
    op = ArchTarget.from_gfx(arch).mma.select_largest_k(
        a_dtype="fp16", b_dtype="fp16", c_dtype="fp32", m=32, n=32
    )
    return (op.m, op.n, op.k)


def _wmma_arches() -> list[str]:
    from rocke.core.arch import known_arches

    return [g for g in known_arches() if ArchTarget.from_gfx(g).wave_size == 32]


def _wmma_spec(dtype, tile, atom, epilogue):
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
    )

    return UniversalGemmSpec(
        name="wmma_support_probe",
        tile=TileSpec(
            tile_m=tile[0],
            tile_n=tile[1],
            tile_k=tile[2],
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=atom[0],
            warp_tile_n=atom[1],
            warp_tile_k=atom[2],
        ),
        trait=TraitSpec(pipeline="mem", scheduler="intrawave", epilogue=epilogue),
        data=DataSpec(dtype_a=dtype, dtype_b=dtype, dtype_c=dtype, layout="RCR"),
        wave_size=32,
    )


def _wmma_query(arch, *, atom, epilogue="default", dtype="fp16") -> GemmSupportQuery:
    return support_query_from_universal_spec(
        _wmma_spec(dtype, (128, 128, 128), atom, epilogue), arch=arch
    )


class TestGemmSupportPredicates(unittest.TestCase):
    def _cdna_query(self, arch: str) -> GemmSupportQuery:
        result = dispatch_gemm_fp16(GemmRequest(M=128, N=128, K=32, arch=arch))
        return support_query_from_universal_spec(result.spec, arch=arch)

    def test_valid_query_is_supported(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                ok, why = gemm_config_supported(self._cdna_query(arch))
                self.assertTrue(ok, why)

    def test_rejects_wave_size_mismatch(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                ok, why = gemm_config_supported(
                    replace(self._cdna_query(arch), wave_size=32)
                )
                self.assertFalse(ok)
                self.assertIn("wave_size", why)

    def test_rejects_block_size_mismatch(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                ok, why = gemm_config_supported(
                    replace(self._cdna_query(arch), block_size=128)
                )
                self.assertFalse(ok)
                self.assertIn("block_size", why)

    def test_rejects_unsupported_mma_intrinsic_shape(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                ok, why = gemm_config_supported(
                    replace(self._cdna_query(arch), warp_tile=(64, 64, 16))
                )
                self.assertFalse(ok)
                self.assertIn("unsupported", why)

    def test_rejects_lds_overflow(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                q = replace(
                    self._cdna_query(arch),
                    cta_tile=(512, 512, 256),
                    warp_shape=(4, 4, 1),
                    warp_tile=_warp_tile_32x32(arch),
                    block_size=1024,
                )
                ok, why = gemm_config_supported(q)
                self.assertFalse(ok)
                self.assertIn("LDS budget", why)

    def test_rejects_wmma_pipeline_and_epilogue_restrictions(self):
        rdna = dispatch_gemm_fp16(GemmRequest(M=64, N=32, K=16, arch="gfx1151"))
        q = support_query_from_universal_spec(rdna.spec, arch="gfx1151")
        ok, why = gemm_config_supported(replace(q, pipeline="compv4"))
        self.assertFalse(ok)
        self.assertIn("WMMA path supports only the 'mem' pipeline", why)
        ok, why = gemm_config_supported(replace(q, epilogue="cshuffle"))
        self.assertFalse(ok)
        self.assertIn("WMMA path supports only the 'default' epilogue", why)

    def test_wmma_atoms_and_cshuffle_follow_the_arch_table(self):
        # The WMMA atom set and the cshuffle epilogue come from ArchTarget.gemm,
        # the table the universal GEMM's validator reads, not a copy of the rule.
        # An atom the MMA catalog lacks is rejected earlier, so only catalog
        # atoms test the rule; gfx1250's fp32 16x16x4 is the one the table drops.
        atoms = ((16, 16, 4), (16, 16, 16), (16, 16, 32), (16, 16, 64), (16, 16, 128))
        outcomes = set()
        for arch in _wmma_arches():
            target = ArchTarget.from_gfx(arch)
            for dtype in ("fp16", "bf16", "fp32"):
                for atom in atoms:
                    if not target.mma.has_shape(
                        family="wmma",
                        a_dtype=dtype,
                        b_dtype=dtype,
                        c_dtype="fp32",
                        m=atom[0],
                        n=atom[1],
                        k=atom[2],
                    ):
                        continue
                    in_table = atom in target.gemm.wmma_atoms
                    outcomes.add(in_table)
                    with self.subTest(arch=arch, dtype=dtype, atom=atom):
                        _, why = gemm_config_supported(
                            _wmma_query(arch, atom=atom, dtype=dtype)
                        )
                        rejected_for_atom = why.startswith(
                            "WMMA path supports only"
                        ) and ("pipeline" not in why and "epilogue" not in why)
                        self.assertEqual(rejected_for_atom, not in_table, why)
            with self.subTest(arch=arch, epilogue="cshuffle"):
                atom = min(target.gemm.wmma_atoms)
                _, why = gemm_config_supported(
                    _wmma_query(arch, atom=atom, epilogue="cshuffle")
                )
                self.assertEqual(
                    "'default' epilogue" in why, not target.gemm.wmma_cshuffle, why
                )
        # Both outcomes occur, so the loop above is not vacuous.
        self.assertEqual(outcomes, {True, False})

    def test_wmma_arches_agree_with_the_gemm_validator(self):
        # Dispatch must neither admit a WMMA spec the kernel rejects nor reject,
        # for a WMMA reason, one it builds.
        from rocke.instances.common.gemm_universal import is_valid_spec

        tiles = ((64, 64, 64), (128, 128, 64), (128, 128, 128), (64, 128, 128))
        atoms = ((16, 16, 4), (16, 16, 16), (16, 16, 32), (16, 16, 64), (16, 16, 128))
        for arch in _wmma_arches():
            for dtype in ("fp16", "bf16"):
                for tile in tiles:
                    for atom in atoms:
                        for epilogue in ("default", "cshuffle"):
                            spec = _wmma_spec(dtype, tile, atom, epilogue)
                            with self.subTest(
                                arch=arch,
                                dtype=dtype,
                                tile=tile,
                                atom=atom,
                                epilogue=epilogue,
                            ):
                                d_ok, d_why = gemm_config_supported(
                                    support_query_from_universal_spec(spec, arch=arch)
                                )
                                v_ok, v_why = is_valid_spec(spec, arch)
                                self.assertEqual(d_ok, v_ok, f"{d_why} | {v_why}")

    def test_request_shape_support_respects_padding_flags(self):
        for arch in COVERED_ARCHES:
            with self.subTest(arch=arch):
                result = dispatch_gemm_fp16(GemmRequest(M=128, N=128, K=32, arch=arch))
                ok, why = request_shape_supported(
                    GemmRequest(M=130, N=128, K=32, arch=arch), result.spec
                )
                self.assertFalse(ok)
                self.assertIn("M=130", why)

                padded = replace(
                    result.spec, trait=replace(result.spec.trait, pad_m=True)
                )
                ok, why = request_shape_supported(
                    GemmRequest(M=130, N=128, K=32, arch=arch), padded
                )
                self.assertTrue(ok, why)


if __name__ == "__main__":
    unittest.main()
