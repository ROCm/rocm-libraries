# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU tests for gfx1250 Qwen3-30B-A3B GEMM contracts (bf16 + low-bit)."""

from __future__ import annotations

import unittest
from dataclasses import replace


class TestGfx1250Gemm(unittest.TestCase):
    @staticmethod
    def _dtl_spec(
        dtype: str = "bf16", pipeline: str = "mem", *, prefetch: bool = False
    ):
        from rocke.instances.common.gemm_universal import (
            DataSpec,
            TileSpec,
            TraitSpec,
            UniversalGemmSpec,
        )

        return UniversalGemmSpec(
            name="gfx1250_dtl_test",
            tile=TileSpec(
                tile_m=128,
                tile_n=128,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=32,
            ),
            trait=TraitSpec(
                pipeline=pipeline,
                scheduler="intrawave",
                epilogue="default",
                direct_to_lds=True,
                dtl_prefetch=prefetch,
                pad_m=True,
                pad_n=True,
                pad_k=True,
            ),
            data=DataSpec(
                dtype_a=dtype,
                dtype_b=dtype,
                dtype_c=dtype,
                dtype_acc="fp32",
                layout="RCR",
            ),
            wave_size=32,
        )

    def test_wmma_dtl_validation_contract(self):
        from rocke.instances.common.gemm_universal import is_valid_spec

        for dtype in ("fp16", "bf16"):
            for pipeline in ("mem", "wmma_v1"):
                for prefetch in (False, True):
                    with self.subTest(
                        dtype=dtype, pipeline=pipeline, prefetch=prefetch
                    ):
                        ok, why = is_valid_spec(
                            self._dtl_spec(dtype, pipeline, prefetch=prefetch),
                            arch="gfx1250",
                        )
                        self.assertTrue(ok, why)

        base = self._dtl_spec()
        # lds_k_pad is supported on the gfx1250 direct-to-LDS path: the async
        # instruction is per-lane addressed, so the padded row stride is free.
        for pad in (16,):
            for prefetch in (False, True):
                with self.subTest(lds_k_pad=pad, prefetch=prefetch):
                    spec = replace(
                        base,
                        trait=replace(
                            base.trait, lds_k_pad=pad, dtl_prefetch=prefetch
                        ),
                    )
                    ok, why = is_valid_spec(spec, arch="gfx1250")
                    self.assertTrue(ok, why)
        prefetch_without_dtl = replace(
            base,
            trait=replace(
                base.trait, direct_to_lds=False, dtl_prefetch=True
            ),
        )
        ok, why = is_valid_spec(prefetch_without_dtl, arch="gfx1250")
        self.assertFalse(ok)
        self.assertIn("requires direct_to_lds", why)
        # Each lane copies 8 elements per pass and the pass loop has no bound
        # check, so a 16x32 A tile (64 chunks for 128 lanes) would address past
        # the tile. The same tile is valid without direct_to_lds.
        under_sized_a = replace(
            base,
            tile=replace(base.tile, tile_m=16, warp_m=1, warp_n=4),
        )
        ok, why = is_valid_spec(under_sized_a, arch="gfx1250")
        self.assertFalse(ok)
        self.assertIn("A tile to fill whole 128-lane passes", why)
        ok, why = is_valid_spec(
            replace(
                under_sized_a,
                trait=replace(under_sized_a.trait, direct_to_lds=False),
            ),
            arch="gfx1250",
        )
        self.assertTrue(ok, why)

    def test_wmma_dtl_lowers_to_gfx1250_async_instruction(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.common.gemm_universal import build_universal_gemm

        for prefetch in (False, True):
            with self.subTest(prefetch=prefetch):
                ll = lower_kernel_to_llvm(
                    build_universal_gemm(
                        self._dtl_spec(prefetch=prefetch), arch="gfx1250"
                    ),
                    arch="gfx1250",
                )
                self.assertIn("llvm.amdgcn.global.load.async.to.lds.b128", ll)
                self.assertIn("llvm.amdgcn.s.wait.asynccnt", ll)
                self.assertNotIn("llvm.amdgcn.raw.ptr.buffer.load.lds", ll)
                expected_lds = 32768 if prefetch else 16384
                self.assertIn(f"[{expected_lds} x i8]", ll)

    def test_wmma_dtl_lds_k_pad_strides_every_lds_access(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.common.gemm_universal import build_universal_gemm

        base = self._dtl_spec()
        for prefetch in (False, True):
            with self.subTest(prefetch=prefetch):
                spec = replace(
                    base,
                    trait=replace(base.trait, lds_k_pad=16, dtl_prefetch=prefetch),
                )
                ll = lower_kernel_to_llvm(
                    build_universal_gemm(spec, arch="gfx1250"), arch="gfx1250"
                )
                self.assertIn("llvm.amdgcn.global.load.async.to.lds.b128", ll)
                # tile_k 32 + pad 16: async writes and WMMA reads both use the
                # padded row, so no access may keep the unpadded stride. The
                # row count doubles with the prefetch ring.
                self.assertIn(" x [48 x bfloat]]", ll)
                self.assertNotIn(" x [32 x bfloat]]", ll)
                expected_lds = 49152 if prefetch else 24576
                self.assertIn(f"[{expected_lds} x i8]", ll)

    def test_wmma_dtl_compiles_to_hsaco(self):
        from rocke.helpers.compile import compile_kernel
        from rocke.instances.common.gemm_universal import build_universal_gemm

        artifact = compile_kernel(
            build_universal_gemm(
                self._dtl_spec(prefetch=True), arch="gfx1250"
            ),
            arch="gfx1250",
        )
        self.assertGreater(artifact.hsaco_bytes, 0)

    @classmethod
    def _tdm_spec(cls, *, depth: int = 1, lds_k_pad: int = 8, **trait):
        base = cls._dtl_spec()
        overrides = {
            "direct_to_lds": False,
            "dtl_prefetch": False,
            "tdm": True,
            "tdm_depth": depth,
            "lds_k_pad": lds_k_pad,
            **trait,
        }
        return replace(
            base,
            name="gfx1250_tdm_test",
            trait=replace(base.trait, **overrides),
        )

    def test_wmma_tdm_validation_contract(self):
        from rocke.instances.common.gemm_universal import is_valid_spec

        ok, why = is_valid_spec(self._tdm_spec(), arch="gfx1250")
        self.assertTrue(ok, why)
        # lds_k_pad is honoured by the mover, so every pad the sweep lists must
        # encode; pad 0 simply disables hardware padding.
        for pad in (0, 8, 16, 24, 40, 56):
            with self.subTest(lds_k_pad=pad):
                ok, why = is_valid_spec(self._tdm_spec(lds_k_pad=pad), arch="gfx1250")
                self.assertTrue(ok, why)
        for depth in (1, 2, 3, 4):
            with self.subTest(tdm_depth=depth):
                ok, why = is_valid_spec(self._tdm_spec(depth=depth), arch="gfx1250")
                self.assertTrue(ok, why)

        for label, spec, needle in (
            (
                "with direct_to_lds",
                self._tdm_spec(direct_to_lds=True),
                "alternative load paths",
            ),
            ("depth 5", self._tdm_spec(depth=5), "tdm_depth must be in 1..4"),
            ("depth 0", self._tdm_spec(depth=0), "tdm_depth must be in 1..4"),
        ):
            with self.subTest(label):
                ok, why = is_valid_spec(spec, arch="gfx1250")
                self.assertFalse(ok)
                self.assertIn(needle, why)

        # An odd pad cannot be expressed as a whole number of dwords.
        ok, why = is_valid_spec(self._tdm_spec(lds_k_pad=1), arch="gfx1250")
        self.assertFalse(ok)
        self.assertIn("cannot encode lds_k_pad", why)

        # tdm_depth without tdm would silently allocate a second LDS buffer.
        base = self._dtl_spec()
        ok, why = is_valid_spec(
            replace(base, trait=replace(base.trait, tdm_depth=2)), arch="gfx1250"
        )
        self.assertFalse(ok)
        self.assertIn("only meaningful with tdm", why)

    def test_tdm_rejected_off_gfx1250(self):
        """No target without the mover may admit ``tdm``, whatever its MMA family.

        The gate used to sit inside the WMMA branch only, so an MFMA spec with
        ``tdm=True`` validated and the builder emitted ``tensor_load_to_lds``
        for gfx942/gfx950. The build must now fail before any IR is emitted.
        """
        from rocke.instances.common.gemm_universal import (
            DataSpec,
            TileSpec,
            TraitSpec,
            UniversalGemmSpec,
            build_universal_gemm,
            is_valid_spec,
        )

        def spec(warp_tile, wave_size):
            return UniversalGemmSpec(
                name="non_gfx1250_tdm",
                tile=TileSpec(
                    tile_m=128,
                    tile_n=128,
                    tile_k=32,
                    warp_m=2,
                    warp_n=2,
                    warp_k=1,
                    warp_tile_m=warp_tile[0],
                    warp_tile_n=warp_tile[1],
                    warp_tile_k=warp_tile[2],
                ),
                trait=TraitSpec(
                    pipeline="mem",
                    scheduler="intrawave",
                    epilogue="default",
                    tdm=True,
                    lds_k_pad=8,
                ),
                data=DataSpec(dtype_a="bf16", dtype_b="bf16", dtype_c="bf16"),
                wave_size=wave_size,
            )

        for arch, candidate in (
            ("gfx950", spec((16, 16, 16), 64)),
            ("gfx942", spec((32, 32, 8), 64)),
            ("gfx1151", spec((16, 16, 16), 32)),
        ):
            with self.subTest(arch):
                # Control: the same spec without tdm is admitted, so the
                # rejection below is attributable to tdm alone.
                ok, why = is_valid_spec(
                    replace(candidate, trait=replace(candidate.trait, tdm=False)),
                    arch=arch,
                )
                self.assertTrue(ok, why)

                ok, why = is_valid_spec(candidate, arch=arch)
                self.assertFalse(ok)
                self.assertIn("Tensor Data Mover", why)
                with self.assertRaisesRegex(ValueError, "Tensor Data Mover"):
                    build_universal_gemm(candidate, arch=arch)

    def test_wmma_tdm_lowers_to_tensor_load_to_lds(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.common.gemm_universal import build_universal_gemm

        for depth in (1, 2, 3, 4):
            with self.subTest(tdm_depth=depth):
                ll = lower_kernel_to_llvm(
                    build_universal_gemm(
                        self._tdm_spec(depth=depth), arch="gfx1250"
                    ),
                    arch="gfx1250",
                )
                self.assertIn("llvm.amdgcn.tensor.load.to.lds", ll)
                self.assertIn("llvm.amdgcn.s.wait.tensorcnt", ll)
                # The mover writes LDS itself, so neither staged path appears.
                self.assertNotIn("llvm.amdgcn.global.load.async.to.lds", ll)
                self.assertNotIn("llvm.amdgcn.s.wait.asynccnt", ll)
                # 128x(32+8) halves per operand, one region per ring buffer.
                self.assertIn(f"[{20480 * depth} x i8]", ll)

    def test_wmma_tdm_deep_ring_waits_partially(self):
        """Depth >= 3 must leave ``depth - 2`` fills in flight across the wait.

        This is the whole lever: the extra LDS buffers only pay if the per-tile
        wait stops draining TENSORcnt to zero. A deep ring that still emitted
        ``s_wait_tensorcnt(0)`` would verify, run, and buy nothing, so the count
        is asserted rather than inferred from a timer.

        The count is per *issuing* wave, and this spec's 2x2 warp grid puts A on
        wave 0 and B on wave 1, so one descriptor per tile per wave -- hence
        ``depth - 2`` and not ``2 * (depth - 2)``. Waiting on the latter would be
        a whole tile too shallow and would read LDS before the fill landed.
        """
        import re

        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.common.gemm_universal import build_universal_gemm

        for depth in (1, 2, 3, 4):
            with self.subTest(tdm_depth=depth):
                ll = lower_kernel_to_llvm(
                    build_universal_gemm(
                        self._tdm_spec(depth=depth), arch="gfx1250"
                    ),
                    arch="gfx1250",
                )
                waits = [
                    int(n)
                    for n in re.findall(r"wait\.tensorcnt\(i16 (\d+)\)", ll)
                ]
                if depth >= 3:
                    # Two distinct waits on the ring path, and the distinction
                    # is the point: the in-loop wait is partial (the lever),
                    # while the loop is followed by exactly one full drain so no
                    # mover write outlives the K-loop. Without that drain the
                    # trailing look-ahead fills stay in flight into the
                    # epilogue, where the smem packer has aliased the cshuffle C
                    # tile onto the ring's bytes.
                    self.assertEqual(sorted(waits), [0, depth - 2])
                else:
                    # Depths 1 and 2 keep the original emission: every wait on
                    # the ping-pong path is a full drain.
                    self.assertEqual(set(waits), {0})
                if depth >= 3:
                    # One prologue fill per ring slot bar the one computed
                    # first, plus the single look-ahead fill in the loop body,
                    # for each of the two operands. The declare line is not a
                    # call site, so match on the call prefix.
                    issues = ll.count("call void @llvm.amdgcn.tensor.load.to.lds")
                    self.assertEqual(issues, 2 * (depth - 1) + 2)

    def test_wmma_rejects_lds_swizzle_on_every_load_path(self):
        """``lds_swizzle`` miscompiles on WMMA, so the family gate must catch it.

        It XORs the *global* column so the LDS destination stays wave-contiguous
        -- a gfx9-shaped assumption that does not carry over to WMMA's ds_read
        geometry. The direct-to-LDS and TDM paths cannot even express it and
        rejected it already; the gap was the plain VGPR-staged path, where it
        emits and runs *fast* while returning wrong results. Since the sweep
        verifies only its top candidates, such a config ranks as a winner while
        being incorrect, so the gate has to refuse it rather than the config file.
        """
        from rocke.instances.common.gemm_universal import is_valid_spec

        base = self._dtl_spec()
        for label, trait in (
            ("vgpr-staged", {"direct_to_lds": False, "dtl_prefetch": False}),
            ("direct-to-lds", {"direct_to_lds": True}),
            ("tdm", {"direct_to_lds": False, "dtl_prefetch": False, "tdm": True}),
        ):
            with self.subTest(label):
                spec = replace(
                    base,
                    trait=replace(base.trait, lds_swizzle=True, **trait),
                )
                ok, why = is_valid_spec(spec, arch="gfx1250")
                self.assertFalse(ok)
                self.assertIn("lds_swizzle", why)
        # The swizzle is bit-exact and worth ~+3% on CDNA MFMA, so the gate is
        # scoped to the WMMA family and must not reach the MFMA path.
        from rocke.instances.common.gemm_universal import (
            DataSpec,
            TileSpec,
            TraitSpec,
            UniversalGemmSpec,
        )

        mfma = UniversalGemmSpec(
            name="gfx950_swizzle_control",
            tile=TileSpec(
                tile_m=128,
                tile_n=128,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=16,
            ),
            trait=TraitSpec(
                pipeline="compv3",
                scheduler="intrawave",
                epilogue="default",
                lds_swizzle=True,
            ),
            data=DataSpec(dtype_a="bf16", dtype_b="bf16", dtype_c="bf16"),
            wave_size=64,
        )
        ok, why = is_valid_spec(mfma, arch="gfx950")
        self.assertTrue(ok, why)

    def test_wmma_tdm_compiles_to_hsaco_per_depth(self):
        from rocke.helpers.compile import compile_kernel
        from rocke.instances.common.gemm_universal import build_universal_gemm

        blobs = {}
        for depth in (1, 2, 3, 4):
            artifact = compile_kernel(
                build_universal_gemm(self._tdm_spec(depth=depth), arch="gfx1250"),
                arch="gfx1250",
            )
            self.assertGreater(artifact.hsaco_bytes, 0)
            blobs[depth] = artifact.hsaco
        # Every depth is a distinct kernel; a ring that collapsed onto the
        # ping-pong would otherwise pass every other assertion here.
        self.assertEqual(len(set(blobs.values())), len(blobs))

    @staticmethod
    def _cshuffle_spec(
        *, epilogue: str = "cshuffle", depth: int = 2, pad: bool = False
    ):
        """The m114688 case-study winner tile, epilogue/pad parameterised.

        256x256x64 on a 8x4 warp grid (block 1024), WMMA 16x16x32 bf16, TDM
        ping-pong, ``lds_k_pad=8``.
        """
        from rocke.instances.common.gemm_universal import (
            DataSpec,
            TileSpec,
            TraitSpec,
            UniversalGemmSpec,
        )

        return UniversalGemmSpec(
            name="gfx1250_cshuffle_test",
            tile=TileSpec(
                tile_m=256,
                tile_n=256,
                tile_k=64,
                warp_m=8,
                warp_n=4,
                warp_k=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=32,
            ),
            trait=TraitSpec(
                pipeline="mem",
                scheduler="intrawave",
                epilogue=epilogue,
                tdm=True,
                tdm_depth=depth,
                lds_k_pad=8,
                pad_m=pad,
                pad_n=pad,
                pad_k=pad,
            ),
            data=DataSpec(
                dtype_a="bf16",
                dtype_b="bf16",
                dtype_c="bf16",
                dtype_acc="fp32",
                layout="RCR",
            ),
            wave_size=32,
        )

    def test_wmma_cshuffle_validation_contract(self):
        """The WMMA path accepts both epilogues, and the LDS gate models the
        A/B <-> C aliasing the emitter actually performs.

        Against gfx1250's 320 KiB per-WG LDS, the aliasing is what decides the
        outcome one tile_k up from the winner: at 256x256x128 the C staging
        tile is 128 KiB and the double-buffered TDM A/B is 272 KiB, so the real
        peak max(A/B, C) = 272 KiB fits while an additive gate would compute
        400 KiB and reject. ``cshuffle_no_alias`` opts out of the aliasing and
        must therefore be the one that fails.
        """
        from rocke.instances.common.gemm_universal import is_valid_spec

        for epilogue in ("default", "cshuffle"):
            for depth in (1, 2):
                for pad in (False, True):
                    with self.subTest(epilogue=epilogue, depth=depth, pad=pad):
                        ok, why = is_valid_spec(
                            self._cshuffle_spec(
                                epilogue=epilogue, depth=depth, pad=pad
                            ),
                            arch="gfx1250",
                        )
                        self.assertTrue(ok, why)

        # One tile_k up, the aliased peak still fits but the additive sum does
        # not, so the two gates disagree and the aliasing is observable.
        spec = replace(
            self._cshuffle_spec(),
            tile=replace(self._cshuffle_spec().tile, tile_k=128),
        )
        ok, why = is_valid_spec(spec, arch="gfx1250")
        self.assertTrue(ok, why)

        # cshuffle_no_alias opts out of the aliasing, so the budget really is
        # additive there and this tile no longer fits.
        no_alias = replace(
            spec, trait=replace(spec.trait, cshuffle_no_alias=True)
        )
        ok, why = is_valid_spec(no_alias, arch="gfx1250")
        self.assertFalse(ok)
        self.assertIn("LDS budget", why)

        # A persistent tile loop keeps A/B and C live together, so the packer
        # cannot alias them and the budget is additive there as well.
        persistent = replace(
            spec, trait=replace(spec.trait, persistent=True, persistent_ctas=256)
        )
        ok, why = is_valid_spec(persistent, arch="gfx1250")
        self.assertFalse(ok)
        self.assertIn("LDS budget", why)

    def test_wmma_cshuffle_stages_c_through_lds_without_extra_lds(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.common.gemm_universal import build_universal_gemm

        def _lower(**kw):
            return lower_kernel_to_llvm(
                build_universal_gemm(self._cshuffle_spec(**kw), arch="gfx1250"),
                arch="gfx1250",
            )

        cshuffle = _lower()
        default = _lower(epilogue="default")

        # Two 256x(64+8) bf16 operand buffers, ping-ponged = 144 KiB. The C
        # staging tile (128 KiB) aliases onto them, so the pool is unchanged
        # from the direct epilogue -- cshuffle is LDS-free at this tile.
        self.assertIn("[147456 x i8]", cshuffle)
        self.assertIn("[147456 x i8]", default)

        # The accumulator reaches C through LDS, and the global stores are
        # 8-wide (16 B) instead of the direct epilogue's per-slot scalars.
        self.assertIn("store <8 x bfloat>", cshuffle)
        self.assertNotIn("store <8 x bfloat>", default)
        self.assertIn("addrspace(3)", cshuffle)

    def test_bf16_qwen_gemm_shapes_validate_and_lower(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.examples.gfx1250.qwen3_30b_a3b.qwen3_30b_a3b_shapes import (
            ALL_BF16_GEMM_SHAPES,
            bf16_universal_gemm_spec,
            shape_by_name,
        )
        from rocke.instances.common.gemm_universal import (
            build_universal_gemm,
            is_valid_spec,
        )

        for shape in ALL_BF16_GEMM_SHAPES:
            with self.subTest(shape=shape.name):
                spec = bf16_universal_gemm_spec(shape)
                ok, why = is_valid_spec(spec, arch="gfx1250")
                self.assertTrue(ok, why)

        for name in ("qkv_decode_bf16", "qkv_prefill_bf16"):
            spec = bf16_universal_gemm_spec(shape_by_name(name))
            ll = lower_kernel_to_llvm(
                build_universal_gemm(spec, arch="gfx1250"), arch="gfx1250"
            )
            self.assertIn("llvm.amdgcn.wmma.f32.16x16x32.bf16", ll)
            self.assertIn("<16 x bfloat>", ll)

    def test_block_scaled_lowbit_expert_gemms_lower(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.examples.gfx1250.qwen3_30b_a3b.qwen3_30b_a3b_shapes import (
            EXPERT_BLOCK_SCALED_GEMM_SHAPES,
        )
        from rocke.instances.gfx1250.block_scaled_gemm import (
            BlockScaledGemmSpec,
            block_scaled_gemm_grid,
            block_scaled_gemm_signature,
            build_block_scaled_gemm,
            is_valid_spec,
        )

        for shape in EXPERT_BLOCK_SCALED_GEMM_SHAPES:
            with self.subTest(shape=shape.name):
                spec = BlockScaledGemmSpec(
                    name=f"gfx1250_{shape.name}",
                    M=shape.M,
                    N=shape.N,
                    K=shape.K,
                    dtype_a=shape.dtype,
                    dtype_b=shape.dtype,
                )
                ok, why = is_valid_spec(spec, arch="gfx1250")
                self.assertTrue(ok, why)
                self.assertIn("K=64 FP8/BF8 WMMA", why)
                self.assertIn("wmma", spec.kernel_name())
                self.assertEqual(block_scaled_gemm_grid(spec)[2], 1)
                sig = block_scaled_gemm_signature(spec)
                self.assertEqual(
                    [p["name"] for p in sig[:5]],
                    ["A", "B", "A_scale", "B_scale", "C"],
                )

                mfma_spec = replace(spec, matrix_path="mfma")
                ok, why = is_valid_spec(mfma_spec, arch="gfx1250")
                self.assertFalse(ok)
                self.assertIn("no MFMA block_scale path", why)

                ll = lower_kernel_to_llvm(
                    build_block_scaled_gemm(spec, arch="gfx1250"), arch="gfx1250"
                )
                self.assertIn("define amdgpu_kernel", ll)
                # The real K=64 FP8/BF8 WMMA intrinsic + per-block f32 scale.
                lowbit = "fp8" if shape.dtype == "fp8e4m3" else "bf8"
                self.assertIn(
                    f"llvm.amdgcn.wmma.f32.16x16x64.{lowbit}.{lowbit}.v8f32.v8i32",
                    ll,
                )
                self.assertIn("fmul float", ll)


if __name__ == "__main__":
    unittest.main(verbosity=2)
