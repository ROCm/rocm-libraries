# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Coverage for the configurable gfx1250 TDM/LDS MX FP4 GEMM."""

from __future__ import annotations

import re
import unittest

import numpy as np
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.examples.gfx1250.gemm.mxfp4_tdm_gemm import (
    SATURATION_ALPHA,
    YAML_TILE,
    kernel_inputs,
    make_inputs,
    reference,
    run_case,
    swizzle_mx_scales_gfx1250,
)
from rocke.instances.gfx1250.mxfp4_tdm_gemm import (
    IntOps,
    MxFp4TdmGemmSpec,
    build_mxfp4_tdm_gemm,
    is_valid_spec,
    mxfp4_lds_layout,
    tdm_pad_fields,
)
from rocke.instances.gfx1250.mxfp4_tdm_gemm_variants import (
    BASE_VARIANT,
    MXFP4_GEMM_VARIANTS,
    assert_variants_cumulative,
    variant_flags,
    variant_names,
    variant_spec,
    variant_step,
)
from rocke.runtime.hip_module import get_device_arch

_SMOKE = (1024, 1024, 2048)


def _chain_llvm() -> dict[str, str]:
    if not hasattr(_chain_llvm, "cache"):
        _chain_llvm.cache = {
            name: _lower_kernel_to_llvm_python(
                build_mxfp4_tdm_gemm(
                    variant_spec(name, *_SMOKE, **YAML_TILE), arch="gfx1250"
                ),
                arch="gfx1250",
                llvm_flavor="llvm23",
            )
            for name in variant_names(trials=True)
        }
    return _chain_llvm.cache


def _count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text))


def _tdm_pad(fields: dict) -> tuple[int, int]:
    """(period, pad) bytes decoded the way the TDM engine reads the fields."""
    return 4 << (fields["pad_interval"] + 1), 4 * (fields["pad_amount"] + 1)


def _tdm_write(base: int, row_bytes: int, rows: int, pad: tuple[int, int] | None):
    """Simulated LDS placement of one 2D TDM transfer, 16 B at a time.

    Returns the LDS address of each row's first byte and every 16 B chunk
    written. Padding counts from the start of the transfer.
    """
    period, pad_bytes = pad if pad else (0, 0)
    address, since_pad = base, 0
    row_start, chunks = [], []
    for _ in range(rows):
        row_start.append(address)
        for _ in range(row_bytes // 16):
            chunks.append(address)
            address += 16
            since_pad += 16
            if period and since_pad == period:
                address += pad_bytes
                since_pad = 0
    return row_start, chunks


class TestGfx1250MxFp4TdmGemm(unittest.TestCase):
    def test_geometry_and_resource_validation(self):
        from rocke.core.arch import ArchTarget

        vgpr_cap = ArchTarget.from_gfx("gfx1250").limits.vgprs
        self.assertEqual(vgpr_cap, 1024)

        spec = MxFp4TdmGemmSpec(name="valid", M=128, N=64, K=256)
        self.assertEqual((spec.wave_repeat_m, spec.wave_repeat_n), (2, 2))
        self.assertEqual(spec.block_size, 128)
        self.assertEqual(spec.accumulator_values_per_lane, 64)
        self.assertEqual(spec.lds_bytes, 24576)
        self.assertEqual(is_valid_spec(spec), (True, "ok"))

        # YAML MacroTile with SourceSwap: 512 FP32 acc / lane fits MaxVgpr=1024.
        yaml_ss = MxFp4TdmGemmSpec(
            name="yaml_ss",
            M=256,
            N=256,
            K=512,
            tile_m=256,
            tile_n=256,
            tile_k=512,
        )
        self.assertEqual(yaml_ss.accumulator_values_per_lane, 512)
        self.assertLessEqual(yaml_ss.lds_bytes, 327680)
        self.assertEqual(is_valid_spec(yaml_ss), (True, "ok"))

        # Accumulators alone past MaxVgpr (LDS kept under the 320 KiB cap).
        over_vgpr = MxFp4TdmGemmSpec(
            name="over_vgpr",
            M=512,
            N=512,
            K=128,
            tile_m=512,
            tile_n=512,
            tile_k=128,
        )
        ok, reason = is_valid_spec(over_vgpr)
        self.assertFalse(ok)
        self.assertEqual(over_vgpr.accumulator_values_per_lane, 2048)
        self.assertIn("2048 FP32 values per lane", reason)
        self.assertIn("1024 MaxVgpr", reason)

    def test_step5_accepted_geometries_pass_resource_checks(self):
        from rocke.core.arch import ArchTarget

        vgpr_cap = ArchTarget.from_gfx("gfx1250").limits.vgprs
        cases = (
            MxFp4TdmGemmSpec(
                name="grow_m",
                M=256,
                N=64,
                K=512,
                tile_m=256,
                tile_n=64,
                tile_k=512,
            ),
            MxFp4TdmGemmSpec(
                name="grow_n",
                M=128,
                N=128,
                K=512,
                tile_m=128,
                tile_n=128,
                tile_k=512,
            ),
            MxFp4TdmGemmSpec(
                name="waves_4x1",
                M=128,
                N=64,
                K=256,
                warp_m=4,
                warp_n=1,
            ),
            MxFp4TdmGemmSpec(
                name="waves_4x2",
                M=256,
                N=64,
                K=256,
                tile_m=256,
                tile_n=64,
                warp_m=4,
                warp_n=2,
            ),
            MxFp4TdmGemmSpec(
                name="yaml_mt256",
                M=256,
                N=256,
                K=512,
                tile_m=256,
                tile_n=256,
                tile_k=512,
            ),
        )
        for spec in cases:
            with self.subTest(name=spec.name):
                self.assertLessEqual(spec.accumulator_values_per_lane, vgpr_cap)
                self.assertEqual(is_valid_spec(spec), (True, "ok"))

    def test_invalid_geometry_is_rejected(self):
        cases = (
            (
                MxFp4TdmGemmSpec(name="bad", M=128, N=64, K=256, warp_k=2),
                "warp_k must be 1",
            ),
            (
                MxFp4TdmGemmSpec(name="bad", M=128, N=64, K=384, tile_k=256),
                "M/N/K must divide",
            ),
            (
                MxFp4TdmGemmSpec(
                    name="bad",
                    M=128,
                    N=64,
                    K=256,
                    tdm_prefetch_depth=3,
                ),
                "tdm_prefetch_depth must be 2",
            ),
        )
        for spec, message in cases:
            with self.subTest(message=message):
                ok, reason = is_valid_spec(spec)
                self.assertFalse(ok)
                self.assertIn(message, reason)

    def test_step6_dependencies_are_rejected(self):
        problem = {"M": 1024, "N": 1024, "K": 2048, **YAML_TILE}
        chain = variant_flags("step6_8_mxfp4_gemm")
        cases = (
            (
                {"M": 128, "N": 96, "K": 256, "tile_n": 96, "vector_epilogue": True},
                "vector_epilogue needs wave_repeat_n in (4, 8, 16)",
            ),
            (
                {
                    "M": 128,
                    "N": 96,
                    "K": 256,
                    "tile_n": 96,
                    "local_read_prefetch": True,
                },
                "needs an even wave_repeat_n",
            ),
            (
                {**problem, "warp_m": 4, "warp_n": 1, "wave_parity_tdm": True},
                "wave_parity_tdm needs 2x2 waves",
            ),
            (
                {**problem, "vector_epilogue": True, "scale_tdm": True},
                "scale_tdm requires wave_parity_tdm",
            ),
            (
                {**problem, "wave_parity_tdm": True, "scale_tdm": True},
                "scale_tdm requires vector_epilogue",
            ),
            (
                {
                    "M": 512,
                    "N": 256,
                    "K": 1024,
                    **YAML_TILE,
                    "tile_m": 512,
                    "tile_n": 128,
                    "vector_epilogue": True,
                    "wave_parity_tdm": True,
                    "scale_tdm": True,
                },
                "scale_tdm needs wave_repeat_m == 4",
            ),
            (
                {"M": 128, "N": 64, "K": 768, "tile_k": 384, "lds_pad": True},
                "lds_pad needs tile_k / 2 to divide",
            ),
            (
                {**problem, "lds_pad": True, "lds_pad_ck": True},
                "lds_pad_ck and lds_pad are alternative LDS paddings",
            ),
            (
                {"M": 128, "N": 64, "K": 768, "tile_k": 384, "lds_pad_ck": True},
                "lds_pad_ck needs tile_k / 2 to be a power of two",
            ),
            (
                {**problem, "lds_pad": True, "lds_segment_interleave": True},
                "lds_segment_interleave requires lds_pad and wave_parity_tdm",
            ),
            (
                {**problem, "K": 512, "init_c_wmma": True},
                "init_c_wmma peels the first and last K tiles",
            ),
            (
                {**problem, "cluster_dim": (4, 4, 2)},
                "cluster_dim must be 2D with at most 16 workgroups",
            ),
            (
                {**problem, "M": 768, "cluster_dim": (4, 4, 1)},
                "must be divisible by cluster_dim",
            ),
            (
                {**problem, "wave_parity_tdm": True, "tdm_multicast": True},
                "tdm_multicast requires cluster_dim and wave_parity_tdm",
            ),
            (
                {**problem, "prefetch_gl2": True},
                "prefetch_gl2 requires cluster_dim",
            ),
            (
                {
                    "M": 1024,
                    "N": 1024,
                    "K": 1024,
                    **YAML_TILE,
                    "tile_k": 128,
                    "cluster_dim": (4, 4, 1),
                    "prefetch_gl2": True,
                },
                "prefetch_gl2 needs tile_k / 2 to be a multiple of 256 B",
            ),
            (
                {**problem, **chain, "tile_refill_mid": True},
                "tile_refill_mid requires local_read_prefetch, init_c_wmma and scale_tdm",
            ),
        )
        for kwargs, message in cases:
            with self.subTest(message=message):
                ok, reason = is_valid_spec(MxFp4TdmGemmSpec(name="bad", **kwargs))
                self.assertFalse(ok)
                self.assertIn(message, reason)
        full = MxFp4TdmGemmSpec(name="chain", **problem, **chain)
        self.assertEqual(is_valid_spec(full), (True, "ok"))

    def test_variant_registry_chain_and_base_identity(self):
        assert_variants_cumulative()
        names = variant_names()
        self.assertEqual(names[0], BASE_VARIANT)
        self.assertEqual(
            [variant_step(name) for name in names],
            [f"6.{i}" for i in range(1, 9)],
        )
        self.assertEqual(
            [name for name in names[1:]],
            [f"step6_{i}_mxfp4_gemm" for i in range(2, 9)],
        )
        self.assertEqual(MXFP4_GEMM_VARIANTS[0][1], {})
        for name in names:
            with self.subTest(variant=name):
                spec = variant_spec(name, 1024, 1024, 2048, **YAML_TILE)
                self.assertEqual(is_valid_spec(spec), (True, "ok"))
                self.assertEqual(spec.uses_step6, name != BASE_VARIANT)
        base = variant_spec(BASE_VARIANT, 128, 64, 256)
        self.assertEqual(base.name, BASE_VARIANT)
        self.assertFalse(base.local_read_prefetch)
        self.assertEqual(is_valid_spec(base), (True, "ok"))
        default = MxFp4TdmGemmSpec(name="mxfp4_tdm_gemm", M=128, N=64, K=256)
        base_llvm = _lower_kernel_to_llvm_python(
            build_mxfp4_tdm_gemm(base, arch="gfx1250"),
            arch="gfx1250",
            llvm_flavor="llvm23",
        )
        default_llvm = _lower_kernel_to_llvm_python(
            build_mxfp4_tdm_gemm(default, arch="gfx1250"),
            arch="gfx1250",
            llvm_flavor="llvm23",
        )
        self.assertEqual(
            base_llvm.replace(BASE_VARIANT, "mxfp4_tdm_gemm"),
            default_llvm,
        )

    def test_llvm_contains_double_buffered_tdm_lds_wmma_and_fp8_epilogue(self):
        spec = MxFp4TdmGemmSpec(name="llvm", M=128, N=64, K=256)
        llvm = _lower_kernel_to_llvm_python(
            build_mxfp4_tdm_gemm(spec, arch="gfx1250"),
            arch="gfx1250",
            llvm_flavor="llvm23",
        )
        self.assertGreaterEqual(llvm.count("llvm.amdgcn.tensor.load.to.lds"), 4)
        self.assertGreaterEqual(llvm.count("llvm.amdgcn.s.wait.tensorcnt"), 2)
        self.assertIn("llvm.amdgcn.wmma.scale.f32.32x16x128.f4", llvm)
        self.assertIn("load <4 x i32>, ptr addrspace(3)", llvm)
        self.assertIn("llvm.amdgcn.cvt.f32.fp8", llvm)
        self.assertIn("llvm.amdgcn.cvt.pk.fp8.f32", llvm)

    def test_step_lds_bytes(self):
        expected = {
            "6.1": 262144,
            "6.2": 262144,
            "6.3": 262144,
            "6.4": 262144,
            "6.5": 278528,
        }
        for name in variant_names():
            step = variant_step(name)
            spec = variant_spec(name, *_SMOKE, **YAML_TILE)
            with self.subTest(step=step):
                self.assertEqual(spec.lds_bytes, expected.get(step, 282624))
                self.assertLessEqual(spec.lds_bytes, 327680)
        ck_pad = variant_spec("trial_ck_lds_pad_mxfp4_gemm", *_SMOKE, **YAML_TILE)
        self.assertEqual(ck_pad.lds_bytes, 294912)

    def test_each_step_adds_its_ir_feature(self):
        llvm = _chain_llvm()
        by_step = {variant_step(name): text for name, text in llvm.items()}
        store_i8 = r"store i8 [^\n]*ptr addrspace\(1\)"
        store_v8 = r"store <8 x i8> [^\n]*ptr addrspace\(1\)"
        lds_b128 = r"load <4 x i32>, ptr addrspace\(3\)"
        tdm = r"call void @llvm\.amdgcn\.tensor\.load\.to\.lds\("
        scale_u8 = r"load i8, ptr addrspace\(1\)"
        zero_c = r"wmma\.scale\.f32\.32x16x128\.f4\.v16f32[^\n]*<16 x float> %cz\d+"
        k_tiles = _SMOKE[2] // YAML_TILE["tile_k"]
        k_steps = YAML_TILE["tile_k"] // 128

        # 6.2: 8-wide FP8 stores and a NaN-preserving saturation.
        self.assertGreater(_count(store_i8, by_step["6.1"]), 0)
        self.assertEqual(_count(store_i8, by_step["6.2"]), 0)
        self.assertEqual(_count(store_v8, by_step["6.2"]), 128)
        self.assertIn("fcmp uno", by_step["6.2"])
        self.assertNotIn("fcmp uno", by_step["6.1"])
        # 6.3: each A/B fragment is read once per K step (4 A + 8 B atoms,
        # 64 B and 32 B per lane), in the loop body and the peeled last tile.
        self.assertEqual(
            _count(lds_b128, by_step["6.3"]), (4 * 4 + 8 * 2) * k_steps * 2
        )
        # 6.4: one branch-free TDM issue per operand, descriptors chosen by
        # wave parity. 6.5 uses the same select and adds the scale load.
        self.assertEqual(_count(tdm, by_step["6.3"]), 4)
        self.assertEqual(_count(tdm, by_step["6.4"]), 2)
        self.assertIn("tensor.load.to.lds(<4 x i32> %sel", by_step["6.4"])
        # 6.5: no per-lane global scale loads; 12 scale reads per K tile copy.
        self.assertGreater(_count(scale_u8, by_step["6.4"]), 0)
        self.assertEqual(_count(scale_u8, by_step["6.5"]), 0)
        self.assertEqual(
            _count(lds_b128, by_step["6.5"]) - _count(lds_b128, by_step["6.4"]),
            (4 + 8) * 2,
        )
        # 6.6: A pad, B iterate (bit 19, 16 iterations), B-scale pad.
        a_pad = 1 << 20 | 7 << 22 | 3 << 25
        b_scale_pad = 1 << 20 | 5 << 22 | 3 << 25
        for constant in (a_pad, 1 << 19, (16 - 1) << 16, b_scale_pad):
            self.assertNotRegex(by_step["6.5"], rf"\b{constant}\b")
            self.assertRegex(by_step["6.6"], rf"\b{constant}\b")
        # Trial CK LDS pad: 6.5 plus a 16 B / 256 B pad on A and B (the
        # word 6.6 uses for its B-scale pad), and no iterate mode.
        ck_pad = by_step["T4"]
        self.assertRegex(ck_pad, rf"\b{b_scale_pad}\b")
        for constant in (a_pad, 1 << 19, (16 - 1) << 16):
            self.assertNotRegex(ck_pad, rf"\b{constant}\b")
        self.assertEqual(_count(tdm, ck_pad), _count(tdm, by_step["6.5"]))
        # 6.7: B component 1 moves past A.
        layout = mxfp4_lds_layout(
            variant_spec("step6_7_mxfp4_gemm", *_SMOKE, **YAML_TILE)
        )
        b1 = layout.b_component_base(IntOps, 1)
        self.assertEqual(b1, 99584)
        self.assertNotRegex(by_step["6.6"], rf"\b{b1}\b")
        self.assertRegex(by_step["6.7"], rf"\b{b1}\b")
        # 6.8: cluster barriers without fences replace the workgroup barrier.
        # They are inline asm, so LLVM sees no branch in the main loop: the
        # prologue, the loop body and the peeled last tile.
        self.assertIn("@llvm.amdgcn.s.barrier()", by_step["6.7"])
        self.assertNotIn("@llvm.amdgcn.s.barrier()", by_step["6.8"])
        self.assertNotIn("@llvm.amdgcn.s.cluster.barrier", by_step["6.8"])
        cluster_asm = re.escape(
            'asm sideeffect "s_barrier_signal_isfirst -1\\0As_barrier_wait -1'
            '\\0As_cbranch_scc0 1\\0As_barrier_signal -3\\0As_barrier_wait -3"'
        )
        self.assertEqual(_count(cluster_asm, by_step["6.7"]), 0)
        self.assertEqual(_count(cluster_asm, by_step["6.8"]), 3)
        self.assertNotRegex(by_step["6.8"], r"\bfence\b")
        # Trial prefetch_gl2: two NT, SE-scope GL2 prefetches in the loop
        # body, with no guard branch.
        prefetch_gl2, init_c, refill_mid = (by_step[t] for t in ("T1", "T2", "T3"))
        prefetch = r"call void @llvm\.amdgcn\.global\.prefetch\(ptr addrspace\(1\) %\w+, i32 9\)"
        self.assertEqual(
            _count(r"@llvm\.amdgcn\.global\.prefetch\(", by_step["6.8"]), 0
        )
        self.assertEqual(_count(prefetch, prefetch_gl2), 2)
        branches = r"\bbr i1\b"
        self.assertEqual(
            _count(branches, prefetch_gl2), _count(branches, by_step["6.8"])
        )
        # Trial init_c_wmma: the first K step of the peeled first tile
        # multiplies into C = 0 (peeled first tile, loop body, peeled last
        # tile).
        wmma = r"call <16 x float> @llvm\.amdgcn\.wmma\.scale\.f32\.32x16x128\.f4"
        self.assertEqual(_count(zero_c, by_step["6.8"]), 0)
        self.assertEqual(_count(zero_c, init_c), 32)
        self.assertEqual(_count(wmma, by_step["6.8"]), 32 * k_steps * 2)
        self.assertEqual(_count(wmma, init_c), 32 * k_steps * 3)
        self.assertGreaterEqual(k_tiles, 3)
        # Trial tile_refill_mid: the TDM runs two tiles ahead and refills
        # mid-tile, so only the peeled first tile and the loop body remain
        # (the clamped tail needs no peel). Tile k+1 is waited for at the
        # tile boundary with tile k+2's two loads in flight; tensorcnt 0 is
        # only the drain after the loop. Scheduling barriers that let only
        # ALU work cross pin the refill and the first K-step's reads (three
        # per tile copy), and no guard is added.
        self.assertEqual(_count(wmma, refill_mid), 32 * k_steps * 2)
        self.assertEqual(_count(tdm, refill_mid), 8)
        wait_tensor = r"call void @llvm\.amdgcn\.s\.wait\.tensorcnt\(i16 {}\)"
        self.assertEqual(_count(wait_tensor.format(0), init_c), 3)
        self.assertEqual(_count(wait_tensor.format(0), refill_mid), 1)
        self.assertEqual(_count(wait_tensor.format(2), refill_mid), 2)
        sched = r"call void @llvm\.amdgcn\.sched\.barrier\(i32 1030\)"
        self.assertEqual(_count(sched, init_c), 0)
        self.assertEqual(_count(sched, refill_mid), 6)
        self.assertEqual(_count(cluster_asm, refill_mid), 4)
        self.assertEqual(_count(branches, refill_mid), _count(branches, init_c))

    def test_tdm_pad_fields_encoding(self):
        self.assertEqual(tdm_pad_fields(1024, 16), {"pad_interval": 7, "pad_amount": 3})
        self.assertEqual(tdm_pad_fields(256, 16), {"pad_interval": 5, "pad_amount": 3})
        for period in (8, 16, 64, 256, 1024):
            for pad in (4, 16, 32):
                self.assertEqual(_tdm_pad(tdm_pad_fields(period, pad)), (period, pad))

    def test_lds_layout_matches_simulated_tdm_writes(self):
        for name in variant_names(trials=True):
            spec = variant_spec(name, *_SMOKE, **YAML_TILE)
            layout = mxfp4_lds_layout(spec)
            with self.subTest(step=variant_step(name)):
                self.assertEqual(2 * layout.buffer_bytes, spec.lds_bytes)
                chunks = []
                a_pad = (
                    _tdm_pad(tdm_pad_fields(layout.a_pad_period, 16))
                    if layout.a_pad_period
                    else None
                )
                for c in range(layout.a_components):
                    rows, written = _tdm_write(
                        layout.a_component_base(IntOps, c),
                        layout.row_bytes,
                        layout.a_component_rows,
                        a_pad,
                    )
                    chunks += written
                    for r, address in enumerate(rows):
                        row = c * layout.a_component_rows + r
                        self.assertEqual(layout.a_row_offset(IntOps, row), address)
                for c in range(layout.b_components):
                    base = layout.b_component_base(IntOps, c)
                    per_iter = layout.b_iterate_rows or layout.b_component_rows
                    iterations = layout.b_component_rows // per_iter
                    stride = layout.b_iterate_stride
                    b_pad = (
                        _tdm_pad(tdm_pad_fields(layout.b_pad_period, 16))
                        if layout.b_pad_period
                        else None
                    )
                    for it in range(iterations):
                        rows, written = _tdm_write(
                            base + it * stride, layout.row_bytes, per_iter, b_pad
                        )
                        chunks += written
                        for r, address in enumerate(rows):
                            row = c * layout.b_component_rows + it * per_iter + r
                            self.assertEqual(layout.b_row_offset(IntOps, row), address)
                if spec.scale_tdm:
                    b_scale_pad = (
                        _tdm_pad(tdm_pad_fields(layout.b_scale_pad_period, 16))
                        if layout.b_scale_pad_period
                        else None
                    )
                    for c in range(layout.a_components):
                        rows, written = _tdm_write(
                            layout.a_scale_component_base(IntOps, c),
                            layout.a_scale_row_bytes,
                            layout.k_steps,
                            None,
                        )
                        chunks += written
                        for step, address in enumerate(rows):
                            row = c * layout.a_component_rows + 5
                            self.assertEqual(
                                layout.a_scale_offset(IntOps, step, row),
                                address + 5 * 4,
                            )
                    for c in range(layout.b_components):
                        rows, written = _tdm_write(
                            layout.b_scale_component_base(IntOps, c),
                            layout.b_scale_row_bytes,
                            layout.k_steps,
                            b_scale_pad,
                        )
                        chunks += written
                        period = layout.b_scale_pad_period
                        for step in range(layout.k_steps):
                            for col in range(0, layout.b_component_rows, 4):
                                linear = step * layout.b_scale_row_bytes + col * 4
                                pads = linear // period * 16 if period else 0
                                expected = (
                                    layout.b_scale_component_base(IntOps, c)
                                    + linear
                                    + pads
                                )
                                self.assertEqual(
                                    layout.b_scale_offset(
                                        IntOps, step, c * layout.b_component_rows + col
                                    ),
                                    expected,
                                )
                        # Every 16 B scale read starts on a written chunk.
                        written_set = set(written)
                        for step in range(layout.k_steps):
                            for col in range(0, layout.b_component_rows, 4):
                                self.assertIn(
                                    layout.b_scale_offset(
                                        IntOps, step, c * layout.b_component_rows + col
                                    ),
                                    written_set,
                                )
                self.assertEqual(len(chunks), len(set(chunks)), "TDM writes overlap")
                self.assertGreaterEqual(min(chunks), 0)
                self.assertLessEqual(max(chunks) + 16, layout.buffer_bytes)

    def test_scale_swizzle_matches_tensilelite_layout(self):
        spec = variant_spec("step6_5_mxfp4_gemm", 256, 256, 1024, **YAML_TILE)
        inputs = make_inputs(spec)
        _, _, a_sw, b_sw, _ = kernel_inputs(spec, inputs)
        a_scale, b_scale = inputs[2], inputs[3]
        steps = spec.K // 128
        self.assertEqual(a_sw.shape, (steps, spec.M, 4))
        self.assertEqual(b_sw.shape, (steps, spec.N, 4))
        for s, mn, j in ((0, 0, 0), (1, 7, 3), (steps - 1, 255, 2)):
            self.assertEqual(a_sw[s, mn, j], a_scale[mn, 4 * s + j])
            self.assertEqual(b_sw[s, mn, j], b_scale[4 * s + j, mn])
        base = variant_spec(BASE_VARIANT, 256, 256, 1024, **YAML_TILE)
        self.assertIs(kernel_inputs(base, inputs)[2], a_scale)
        np.testing.assert_array_equal(
            swizzle_mx_scales_gfx1250(np.arange(8, dtype=np.uint8).reshape(1, 8)),
            np.array([[[0, 1, 2, 3]], [[4, 5, 6, 7]]], dtype=np.uint8),
        )

    def test_saturating_reference_clamps_to_fp8_max(self):
        spec = MxFp4TdmGemmSpec(name="sat", M=128, N=64, K=256)
        inputs = make_inputs(spec, signed=True)
        alpha, beta = np.float32(SATURATION_ALPHA), np.float32(0.0)
        plain = reference(spec, inputs, alpha=alpha, beta=beta).astype(np.float32)
        saturated = reference(
            spec, inputs, alpha=alpha, beta=beta, saturate=True
        ).astype(np.float32)
        self.assertTrue(np.isnan(plain).any(), "inputs must overflow FP8")
        self.assertFalse(np.isnan(saturated).any())
        self.assertEqual(saturated.max(), 448.0)
        self.assertEqual(saturated.min(), -448.0)
        finite = ~np.isnan(plain)
        np.testing.assert_array_equal(saturated[finite], plain[finite])

    def test_variant_chain_matches_reference_on_gfx1250(self):
        if get_device_arch(0) != "gfx1250":
            self.skipTest("visible HIP device 0 is not gfx1250")
        for name in variant_names():
            spec = variant_spec(name, *_SMOKE, **YAML_TILE)
            with self.subTest(variant=name):
                for beta in (0.0, 0.5):
                    run_case(spec, beta=beta)
                if spec.vector_epilogue:
                    run_case(spec, alpha=SATURATION_ALPHA, beta=0.0, signed=True)

    def test_fp8_result_matches_reference_on_gfx1250(self):
        if get_device_arch(0) != "gfx1250":
            self.skipTest("visible HIP device 0 is not gfx1250")
        run_case(
            MxFp4TdmGemmSpec(
                name="numeric",
                M=64,
                N=32,
                K=256,
                tile_m=64,
                tile_n=32,
                warp_m=1,
                warp_n=1,
            )
        )


if __name__ == "__main__":
    unittest.main()
