# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU tests for gfx1250 Qwen3-30B-A3B GEMM contracts (bf16 + low-bit)."""

from __future__ import annotations

import unittest
import unittest.mock
from dataclasses import asdict, replace


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
        # lds_swizzle is still rejected: it XORs the global column so the LDS
        # destination stays wave-contiguous, which does not carry over.
        for changes, expected in (({"lds_swizzle": True}, "lds_swizzle"),):
            with self.subTest(changes=changes):
                spec = replace(base, trait=replace(base.trait, **changes))
                ok, why = is_valid_spec(spec, arch="gfx1250")
                self.assertFalse(ok)
                self.assertIn(expected, why)
        prefetch_without_dtl = replace(
            base,
            trait=replace(
                base.trait, direct_to_lds=False, dtl_prefetch=True
            ),
        )
        ok, why = is_valid_spec(prefetch_without_dtl, arch="gfx1250")
        self.assertFalse(ok)
        self.assertIn("requires direct_to_lds", why)

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

    def test_wmma_dtl_cpp_builder_matches_python(self):
        try:
            import rocke_engine
        except Exception as exc:
            self.skipTest(f"rocke_engine C++ binding not importable: {exc}")

        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.common.gemm_universal import build_universal_gemm

        spec = self._dtl_spec(prefetch=True)
        with unittest.mock.patch.dict("os.environ", {"ROCKE_BACKEND": "python"}):
            py_ll = lower_kernel_to_llvm(
                build_universal_gemm(spec, arch="gfx1250"),
                arch="gfx1250",
            )
        cpp_ll = rocke_engine.gemm_lower_llvm(asdict(spec), arch="gfx1250")
        self.assertEqual(py_ll, cpp_ll)

    def test_dtl_is_enumerated_in_trait_sweep(self):
        from rocke.examples.gfx1250.bf16_gemm_sweep import (
            enumerate_tile_configs,
            enumerate_trait_configs,
            load_config,
        )

        config = load_config()
        knobs = config["trait_config"]
        finalists = enumerate_tile_configs(config)[:12]
        traits = enumerate_trait_configs(config, finalists)
        dtl_traits = [spec.trait for spec in traits if spec.trait.direct_to_lds]
        self.assertTrue(dtl_traits, "config enumerates no direct-to-LDS candidates")

        # Asserted as invariants rather than a hardcoded total: the sweep config
        # is routinely re-pinned per experiment (e.g. direct_to_lds fixed to
        # [true] for an isolated A/B), which would make a fixed count brittle.

        # dtl_prefetch is only ever paired with direct_to_lds.
        self.assertTrue(
            all(spec.trait.direct_to_lds for spec in traits if spec.trait.dtl_prefetch)
        )
        # lds_swizzle stays pruned on the direct-to-LDS path.
        self.assertTrue(all(not trait.lds_swizzle for trait in dtl_traits))
        # lds_k_pad is NOT pruned: every pad the config lists is enumerated for
        # direct-to-LDS, and orthogonally to dtl_prefetch.
        self.assertEqual(
            {trait.lds_k_pad for trait in dtl_traits}, set(knobs["lds_k_pad"])
        )
        for pad in knobs["lds_k_pad"]:
            for prefetch in knobs.get("dtl_prefetch", [False]):
                with self.subTest(lds_k_pad=pad, dtl_prefetch=prefetch):
                    self.assertTrue(
                        any(
                            trait.lds_k_pad == pad and trait.dtl_prefetch == prefetch
                            for trait in dtl_traits
                        )
                    )
        # The pad arms are balanced, so padding doubled the direct-to-LDS space
        # rather than replacing part of it.
        by_pad = {
            pad: sum(1 for trait in dtl_traits if trait.lds_k_pad == pad)
            for pad in knobs["lds_k_pad"]
        }
        self.assertEqual(len(set(by_pad.values())), 1, by_pad)

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
        for depth in (1, 2):
            with self.subTest(tdm_depth=depth):
                ok, why = is_valid_spec(self._tdm_spec(depth=depth), arch="gfx1250")
                self.assertTrue(ok, why)

        for label, spec, needle in (
            (
                "with direct_to_lds",
                self._tdm_spec(direct_to_lds=True),
                "alternative load paths",
            ),
            ("with lds_swizzle", self._tdm_spec(lds_swizzle=True), "lds_swizzle"),
            ("depth 3", self._tdm_spec(depth=3), "tdm_depth must be 1 or 2"),
        ):
            with self.subTest(label):
                ok, why = is_valid_spec(spec, arch="gfx1250")
                self.assertFalse(ok)
                self.assertIn(needle, why)

        # The mover is a gfx1250 opcode, so the knob is gated on the capability
        # rather than on the architecture name.
        from rocke.core.arch.target import ArchTarget

        self.assertTrue(ArchTarget.from_gfx("gfx1250").memory.has_tdm)
        for other in ("gfx950", "gfx942", "gfx1151"):
            with self.subTest(other):
                self.assertFalse(ArchTarget.from_gfx(other).memory.has_tdm)

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

    def test_wmma_tdm_lowers_to_tensor_load_to_lds(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.common.gemm_universal import build_universal_gemm

        for depth in (1, 2):
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
                # 128x(32+8) halves per operand, doubled when ping-ponging.
                self.assertIn(f"[{20480 * depth} x i8]", ll)

    def test_wmma_tdm_compiles_to_hsaco_per_depth(self):
        from rocke.helpers.compile import compile_kernel
        from rocke.instances.common.gemm_universal import build_universal_gemm

        blobs = {}
        for depth in (1, 2):
            artifact = compile_kernel(
                build_universal_gemm(self._tdm_spec(depth=depth), arch="gfx1250"),
                arch="gfx1250",
            )
            self.assertGreater(artifact.hsaco_bytes, 0)
            blobs[depth] = artifact.hsaco
        self.assertNotEqual(blobs[1], blobs[2])

    def test_tdm_is_enumerated_in_trait_sweep(self):
        import copy

        from rocke.examples.gfx1250.bf16_gemm_sweep import (
            enumerate_tile_configs,
            enumerate_trait_configs,
            load_config,
        )

        base = load_config()
        finalists = enumerate_tile_configs(base)[:8]

        # The shipped config enables the knob, so take the baseline from an
        # explicitly tdm-off copy rather than assuming its value.
        off = copy.deepcopy(base)
        off["trait_config"]["tdm"] = [False]
        without = enumerate_trait_configs(off, finalists)
        self.assertFalse([s for s in without if s.trait.tdm])

        config = copy.deepcopy(base)
        config["trait_config"]["tdm"] = [False, True]
        config["trait_config"]["tdm_depth"] = [1, 2]
        traits = enumerate_trait_configs(config, finalists)
        tdm_traits = [spec.trait for spec in traits if spec.trait.tdm]
        self.assertTrue(tdm_traits, "tdm knob did not enumerate any candidate")
        # The knob adds configurations rather than displacing the existing ones.
        self.assertGreater(len(traits), len(without))
        self.assertEqual({t.tdm_depth for t in tdm_traits}, {1, 2})
        # TDM is its own load path, never combined with the other two.
        self.assertTrue(
            all(not t.direct_to_lds and not t.dtl_prefetch for t in tdm_traits)
        )
        # Padding is swept on the TDM path too -- the mover applies it.
        self.assertEqual(
            {t.lds_k_pad for t in tdm_traits},
            set(config["trait_config"]["lds_k_pad"]),
        )

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
