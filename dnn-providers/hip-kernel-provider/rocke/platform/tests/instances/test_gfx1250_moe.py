# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU tests for the gfx1250 Qwen3-30B-A3B MoE stack.

Covers router top-k softmax, MoE sorting / static-offset scatter, the fused-MoE
forward contract (bf16 batched GEMM + smoothquant + graph-capture API), the
explicit low-bit block-scale gap, and the FP8/BF8 fused-MoE driver wiring.
"""

from __future__ import annotations

import math
import unittest
from typing import List, Sequence, Tuple

from rocke import lower_kernel_to_llvm


# --------------------------------------------------------------------------- #
# CPU reference helpers (router top-k + sorting)
# --------------------------------------------------------------------------- #
def _topk_softmax_reference(
    rows: Sequence[Sequence[float]], k: int
) -> Tuple[List[List[float]], List[List[int]]]:
    weights: List[List[float]] = []
    indices: List[List[int]] = []
    for row in rows:
        winners = sorted(enumerate(row), key=lambda item: (-item[1], item[0]))[:k]
        picked_vals = [v for _, v in winners]
        vmax = picked_vals[0]
        exps = [math.exp(v - vmax) for v in picked_vals]
        denom = sum(exps)
        weights.append([v / denom for v in exps])
        indices.append([idx for idx, _ in winners])
    return weights, indices


def _moe_sort_reference(
    topk_ids: Sequence[Sequence[int]],
    topk_weights: Sequence[Sequence[float]],
    experts: int,
) -> Tuple[List[int], List[int], List[int], List[int], List[float]]:
    counts = [0 for _ in range(experts)]
    buckets: List[List[Tuple[int, int, float]]] = [[] for _ in range(experts)]
    for token, row in enumerate(topk_ids):
        for k_idx, expert in enumerate(row):
            if 0 <= expert < experts:
                counts[expert] += 1
                buckets[expert].append((token, k_idx, topk_weights[token][k_idx]))

    offsets: List[int] = []
    running = 0
    for count in counts:
        offsets.append(running)
        running += count

    sorted_token_ids: List[int] = []
    sorted_topk_ids: List[int] = []
    sorted_weights: List[float] = []
    for bucket in buckets:
        for token, k_idx, weight in bucket:
            sorted_token_ids.append(token)
            sorted_topk_ids.append(k_idx)
            sorted_weights.append(weight)
    return offsets, counts, sorted_token_ids, sorted_topk_ids, sorted_weights


def _static_offset_scatter_reference(
    topk_ids: Sequence[Sequence[int]],
    experts: int,
    slot_size: int,
) -> Tuple[List[int], List[int]]:
    static_offsets = [expert * slot_size for expert in range(experts)]
    sorted_token_ids = [-1 for _ in range(experts * slot_size)]
    counters = [0 for _ in range(experts)]
    for token, row in enumerate(topk_ids):
        for expert in row:
            if 0 <= expert < experts:
                local = counters[expert]
                if local >= slot_size:
                    raise AssertionError(
                        f"static slot overflow for expert {expert}: "
                        f"slot_size={slot_size}"
                    )
                sorted_token_ids[static_offsets[expert] + local] = token
                counters[expert] += 1
    return static_offsets, sorted_token_ids


class TestGfx1250RouterTopkSort(unittest.TestCase):
    def test_topk_softmax_e128_k8_wave32_contract(self):
        from rocke.core.lower_hip import lower_kernel_to_hip
        from rocke.examples.gfx1250.qwen3_30b_a3b.qwen3_30b_a3b_shapes import (
            ARCH,
            NUM_EXPERTS,
            QWEN3_30B_A3B_DECODE,
            TOPK,
            topk_softmax_spec,
        )
        from rocke.instances import build_topk_softmax, is_valid_topk_softmax_spec

        spec = topk_softmax_spec()
        ok, why = is_valid_topk_softmax_spec(spec, ARCH)
        self.assertTrue(ok, why)
        self.assertEqual(spec.n_per_row, NUM_EXPERTS)
        self.assertEqual(spec.k, TOPK)
        self.assertEqual(spec.block_size, 128)

        hip = lower_kernel_to_hip(build_topk_softmax(spec, ARCH))
        self.assertIn("__shared__", hip)
        self.assertIn("__syncthreads()", hip)

        rows = [
            [float((i * 37) % NUM_EXPERTS) / 17.0 for i in range(NUM_EXPERTS)],
            [
                float(((NUM_EXPERTS - i) * 19) % NUM_EXPERTS) / 23.0
                for i in range(NUM_EXPERTS)
            ],
        ]
        weights, indices = _topk_softmax_reference(rows, TOPK)
        self.assertEqual(len(indices), QWEN3_30B_A3B_DECODE.tokens)
        for row_weights, row_indices, row in zip(weights, indices, rows):
            self.assertEqual(len(row_indices), TOPK)
            self.assertAlmostEqual(sum(row_weights), 1.0, places=6)
            expected = sorted(range(NUM_EXPERTS), key=lambda i: (-row[i], i))[:TOPK]
            self.assertEqual(row_indices, expected)

    def test_moe_sorting_and_static_offset_decode_contract(self):
        from rocke.core.lower_hip import lower_kernel_to_hip
        from rocke.examples.gfx1250.qwen3_30b_a3b.qwen3_30b_a3b_shapes import (
            ARCH,
            NUM_EXPERTS,
            QWEN3_30B_A3B_DECODE,
            SORT_BLOCK_SIZE,
            TOPK,
            fused_moe_forward_spec,
            moe_sorting_spec,
        )
        from rocke.instances import (
            FusedMoeForward,
            MoeSortingSpec,
            build_moe_sort_scan,
            is_valid_moe_sorting_spec,
        )

        sort_spec = moe_sorting_spec()
        self.assertEqual(sort_spec.block_size, SORT_BLOCK_SIZE)
        self.assertEqual(sort_spec.experts, NUM_EXPERTS)
        self.assertGreaterEqual(sort_spec.block_size, sort_spec.experts)
        ok, why = is_valid_moe_sorting_spec(sort_spec, ARCH)
        self.assertTrue(ok, why)

        too_small = MoeSortingSpec(
            tokens=2, topk=TOPK, experts=NUM_EXPERTS, block_size=64
        )
        ok, why = is_valid_moe_sorting_spec(too_small, ARCH)
        self.assertFalse(ok)
        self.assertIn("experts", why)

        scan_hip = lower_kernel_to_hip(build_moe_sort_scan(sort_spec, ARCH))
        self.assertIn("__shared__", scan_hip)
        self.assertIn("__syncthreads()", scan_hip)

        topk_ids = [
            [0, 127, 5, 9, 13, 17, 21, 25],
            [0, 126, 5, 10, 14, 18, 22, 26],
        ]
        topk_weights = [
            [0.40, 0.20, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04],
            [0.35, 0.18, 0.12, 0.10, 0.08, 0.07, 0.06, 0.04],
        ]
        offsets, counts, sorted_tokens, sorted_topk, sorted_weights = (
            _moe_sort_reference(topk_ids, topk_weights, NUM_EXPERTS)
        )
        self.assertEqual(sum(counts), QWEN3_30B_A3B_DECODE.total_pairs)
        self.assertEqual(counts[0], 2)
        self.assertEqual(counts[5], 2)
        self.assertEqual(counts[127], 1)
        self.assertEqual(offsets[0], 0)
        self.assertEqual(offsets[1], counts[0])
        self.assertEqual(len(sorted_tokens), QWEN3_30B_A3B_DECODE.total_pairs)
        self.assertEqual(len(sorted_topk), QWEN3_30B_A3B_DECODE.total_pairs)
        self.assertEqual(len(sorted_weights), QWEN3_30B_A3B_DECODE.total_pairs)

        fwd = FusedMoeForward(fused_moe_forward_spec())
        self.assertTrue(fwd._use_static_offsets)
        self.assertEqual(fwd._static_slot_size, QWEN3_30B_A3B_DECODE.static_slot_size)
        static_offsets, static_tokens = _static_offset_scatter_reference(
            topk_ids,
            NUM_EXPERTS,
            QWEN3_30B_A3B_DECODE.static_slot_size,
        )
        self.assertEqual(static_offsets[0], 0)
        self.assertEqual(static_offsets[1], QWEN3_30B_A3B_DECODE.static_slot_size)
        self.assertEqual(static_tokens[0], 0)
        self.assertEqual(static_tokens[1], 1)

    def test_static_slot_size_rejects_non_positive_override(self):
        from rocke.examples.gfx1250.qwen3_30b_a3b.qwen3_30b_a3b_shapes import (
            fused_moe_forward_spec,
        )
        from rocke.instances import FusedMoeForward

        spec = fused_moe_forward_spec()
        spec.static_slot_size = 0
        with self.assertRaisesRegex(ValueError, "static_slot_size"):
            FusedMoeForward(spec)


class TestGfx1250FusedMoeForward(unittest.TestCase):
    def test_static_forward_contract_and_batched_gemm(self):
        from rocke.examples.gfx1250.qwen3_30b_a3b.qwen3_30b_a3b_shapes import (
            QWEN3_30B_A3B_DECODE_MOE,
        )
        from rocke.instances.common.batched_gemm import (
            build_batched_gemm,
            is_valid_spec,
        )
        from rocke.instances.common.fused_moe_e2e import FusedMoeForward

        shape = QWEN3_30B_A3B_DECODE_MOE
        self.assertEqual(
            (shape.tokens, shape.experts, shape.topk, shape.hidden, shape.intermediate),
            (2, 128, 8, 2048, 768),
        )
        self.assertEqual(shape.active_pairs, 16)

        spec = shape.to_fused_moe_forward_spec(use_static_offsets=True)
        self.assertEqual(spec.sort_block_size, 128)
        fwd = FusedMoeForward(spec)
        self.assertEqual(fwd.arch, "gfx1250")
        self.assertTrue(fwd._use_static_offsets)
        self.assertEqual(fwd._static_slot_size, 16)

        gemm = fwd.spec.to_batched_gemm_spec()
        self.assertEqual(gemm.wave_size, 32)
        self.assertEqual(gemm.block_size, 32)
        self.assertEqual(
            (
                gemm.tile.tile_m,
                gemm.tile.tile_n,
                gemm.tile.tile_k,
                gemm.tile.warp_tile_m,
                gemm.tile.warp_tile_n,
                gemm.tile.warp_tile_k,
            ),
            (16, 16, 32, 16, 16, 32),
        )
        self.assertEqual(gemm.trait.pipeline, "mem")
        self.assertEqual(gemm.trait.epilogue, "default")
        ok, why = is_valid_spec(gemm, arch="gfx1250")
        self.assertTrue(ok, why)
        ll = lower_kernel_to_llvm(
            build_batched_gemm(gemm, arch="gfx1250"), arch="gfx1250"
        )
        self.assertIn("llvm.amdgcn.wmma.f32.16x16x32.bf16", ll)

    def test_dynamic_prefill_sort_contract(self):
        from rocke.examples.gfx1250.qwen3_30b_a3b.qwen3_30b_a3b_shapes import MoeShape
        from rocke.instances.common.moe_sorting import (
            build_moe_sort_histogram,
            build_moe_sort_scan,
            build_moe_sort_scatter,
            is_valid_spec,
        )

        shape = MoeShape(tokens=128, experts=128, topk=8, hidden=2048, intermediate=768)
        spec = shape.to_fused_moe_forward_spec(use_static_offsets=False)
        sort_spec = spec.to_sort_spec()
        self.assertEqual(sort_spec.block_size, 128)
        ok, why = is_valid_spec(sort_spec, arch="gfx1250")
        self.assertTrue(ok, why)
        for build in (
            build_moe_sort_histogram,
            build_moe_sort_scan,
            build_moe_sort_scatter,
        ):
            ll = lower_kernel_to_llvm(build(sort_spec, arch="gfx1250"), arch="gfx1250")
            self.assertIn("define amdgpu_kernel", ll)

    def test_streaming_and_smoothquant_scaffolding_builds(self):
        from rocke.examples.gfx1250.qwen3_30b_a3b.qwen3_30b_a3b_shapes import (
            QWEN3_30B_A3B_DECODE_MOE,
        )
        from rocke.instances.common.fused_moe import (
            build_moe_silu_mul,
            build_moe_topk_weighted_reduce,
        )
        from rocke.instances.common.moe_smoothquant import (
            MoeSmoothQuantSpec,
            build_moe_smoothquant,
            is_valid_spec as is_valid_moe_smoothquant_spec,
            moe_smoothquant_grid,
        )

        shape = QWEN3_30B_A3B_DECODE_MOE
        fmoe = shape.to_fused_moe_forward_spec().to_fused_moe_spec()
        for kernel in (build_moe_silu_mul(fmoe), build_moe_topk_weighted_reduce(fmoe)):
            ll = lower_kernel_to_llvm(kernel, arch="gfx1250")
            self.assertIn("define amdgpu_kernel", ll)

        sq = MoeSmoothQuantSpec(
            n_per_block=shape.hidden,
            topk=shape.topk,
            experts=shape.experts,
            dtype="bf16",
            out_dtype="i8",
            block_size=256,
            vec=8,
            tokens=shape.tokens,
        )
        ok, why = is_valid_moe_smoothquant_spec(sq, arch="gfx1250")
        self.assertTrue(ok, why)
        self.assertEqual(
            moe_smoothquant_grid(shape.tokens, sq), (shape.active_pairs, 1, 1)
        )
        ll = lower_kernel_to_llvm(
            build_moe_smoothquant(sq, arch="gfx1250"), arch="gfx1250"
        )
        self.assertIn("define amdgpu_kernel", ll)

    def test_graph_capture_api_dynamic_smoke(self):
        from rocke.examples.gfx1250.qwen3_30b_a3b.qwen3_30b_a3b_shapes import (
            QWEN3_30B_A3B_DECODE_MOE,
        )
        from rocke.instances.common.fused_moe_e2e import FusedMoeForward

        spec = QWEN3_30B_A3B_DECODE_MOE.to_fused_moe_forward_spec(
            use_static_offsets=False
        )
        fwd = FusedMoeForward(spec)
        with self.assertRaisesRegex(RuntimeError, "requires static-offset mode"):
            fwd.capture_graph(
                routing_logits=None,
                X=None,
                W_gate=None,
                W_up=None,
                W_down=None,
                Y=None,
            )
        with self.assertRaisesRegex(RuntimeError, "before capture_graph"):
            fwd.replay_graph()

    def test_low_bit_block_scale_gap_is_explicit(self):
        from rocke.instances.common.block_scale_gemm import (
            BlockScaleGemmSpec,
            is_valid_spec,
        )
        from rocke.instances.common.gemm_universal import TileSpec, TraitSpec
        from rocke.instances.common.moe_gemm_fused import (
            FusedInterleavedGateUpSiluGemmSpec,
            build_moe_interleaved_gate_up_silu_gemm,
        )

        low_bit = BlockScaleGemmSpec(
            M=16,
            N=16,
            K=64,
            quant_mode="abquant",
            mantissa_dtype="fp8e4m3",
            group_size_mnk=(1, 1, 64),
        )
        ok, why = is_valid_spec(low_bit, arch="gfx1250")
        self.assertFalse(ok)
        self.assertIn("WMMA block-scale expert GEMMs are not implemented", why)

        mfma_only = FusedInterleavedGateUpSiluGemmSpec(
            name="gfx1250_gap",
            tile=TileSpec(
                tile_m=16,
                tile_n=16,
                tile_k=32,
                warp_m=1,
                warp_n=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=32,
            ),
            trait=TraitSpec(pipeline="mem", epilogue="default", pad_m=True, pad_n=True),
            wave_size=32,
            dtype="bf16",
        )
        with self.assertRaisesRegex(NotImplementedError, "MFMA-only.*WMMA"):
            build_moe_interleaved_gate_up_silu_gemm(mfma_only, arch="gfx1250")


class TestGfx1250Fp8MoeDriver(unittest.TestCase):
    def test_spec_static_offset_layout(self):
        from rocke.instances.gfx1250.fused_moe_fp8 import Gfx1250Fp8MoeSpec

        spec = Gfx1250Fp8MoeSpec(
            tokens=2, experts=128, topk=8, hidden=2048, intermediate=768
        )
        # slot_size rounds T*K=16 up to a multiple of the 16x16 WMMA tile.
        self.assertEqual(spec.slot_size, 16)
        self.assertEqual(spec.rows, 128 * 16)

    def test_expert_gemms_lower_to_k64_fp8_wmma(self):
        from rocke.instances.gfx1250.block_scaled_gemm import (
            BlockScaledGemmSpec,
            build_block_scaled_gemm,
        )
        from rocke.instances.gfx1250.fused_moe_fp8 import (
            Gfx1250Fp8MoeSpec,
            _block_k_for,
        )

        spec = Gfx1250Fp8MoeSpec(
            tokens=2, experts=4, topk=2, hidden=256, intermediate=128
        )
        gu = BlockScaledGemmSpec(
            name="t_gu",
            M=spec.slot_size,
            N=spec.intermediate,
            K=spec.hidden,
            dtype_a="fp8e4m3",
            dtype_b="fp8e4m3",
            dtype_c="bf16",
            block_k=_block_k_for(spec.hidden),
        )
        down = BlockScaledGemmSpec(
            name="t_down",
            M=spec.slot_size,
            N=spec.hidden,
            K=spec.intermediate,
            dtype_a="fp8e4m3",
            dtype_b="fp8e4m3",
            dtype_c="bf16",
            block_k=_block_k_for(spec.intermediate),
        )
        for g in (gu, down):
            ll = lower_kernel_to_llvm(
                build_block_scaled_gemm(g, arch="gfx1250"), arch="gfx1250"
            )
            self.assertIn("llvm.amdgcn.wmma.f32.16x16x64.fp8.fp8.v8f32.v8i32", ll)

    def test_full_driver_compiles_for_gfx1250(self):
        from rocke.instances.gfx1250.fused_moe_fp8 import (
            Gfx1250Fp8Moe,
            Gfx1250Fp8MoeSpec,
        )
        from rocke.runtime.comgr import resolved_lib_rocm_version

        # gfx1250 codegen requires comgr >= 7.2; when the resolved comgr is older
        # (e.g. a system ROCm 7.0 with no torch-bundled 7.2 loaded in the process)
        # set_isa rejects the gfx1250 ISA. Skip cleanly rather than fail -- the
        # compile path is covered wherever a gfx1250-capable comgr is active
        # (e.g. the ROCm 7.2 venv with torch imported).
        ver = resolved_lib_rocm_version()
        if ver is None or ver < (7, 2):
            self.skipTest(f"comgr {ver} cannot target gfx1250 (needs >= 7.2)")

        # Compiling the whole driver exercises every component-kernel build for
        # gfx1250 and the per-expert GEMM specs (no GPU needed to compile).
        spec = Gfx1250Fp8MoeSpec(
            tokens=2, experts=4, topk=2, hidden=256, intermediate=128
        )
        moe = Gfx1250Fp8Moe(spec)
        self.assertEqual(moe.arch, "gfx1250")
        self.assertEqual(moe._gu_spec.K, 256)
        self.assertEqual(moe._down_spec.K, 128)
        self.assertEqual(moe._gu_spec.N, 128)
        self.assertEqual(moe._down_spec.N, 256)


if __name__ == "__main__":
    unittest.main(verbosity=2)
