# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU IR-level drift gate for the gfx950 hand-scheduled grouped bf16 GEMM
family: dense ``grouped_gemm``, pure ragged ``ragged_gemm``, and fused MoE
``ragged_moe``.

These builders are lowered by the Python engine only (no C++ mirror yet), so
this file guards the *emitted IR contract* on CPU: each spec builds, lowers to
AMDGPU LLVM, emits the expected 16x16x32 bf16 MFMA cluster, and is a
deterministic function of the spec (the invariant the representative-IR golden
relies on). The on-GPU numeric coverage lives in the example harnesses under
``examples/gfx950/grouped_gemm/``.

Run:  PYTHONPATH=Python python3 tests/instances/test_gfx950_grouped_gemm.py
"""

from __future__ import annotations

import unittest


def _lower(kernel) -> str:
    # Pin the native Python lowerer + the host's autodetected llvm flavor so the
    # emitted IR is a deterministic function of the built kernel alone.
    from rocke.core.lower_llvm import (
        _lower_kernel_to_llvm_python,
        _resolve_llvm_flavor,
    )

    return _lower_kernel_to_llvm_python(
        kernel, arch="gfx950", llvm_flavor=_resolve_llvm_flavor()
    )


_MFMA_BF16 = "llvm.amdgcn.mfma.f32.16x16x32.bf16"


class TestGroupedGemmGfx950(unittest.TestCase):
    def test_builds_with_expected_geometry(self):
        from rocke.instances.gfx950.grouped_gemm import (
            GroupedGemmSpec,
            build_grouped_gemm,
            grouped_gemm_signature,
        )

        kernel, bs, tm, tn = build_grouped_gemm(
            GroupedGemmSpec(M=8192, N=1024, K=512, E=64)
        )
        self.assertEqual(kernel.name, "grouped_gemm")
        self.assertEqual((bs, tm, tn), (512, 256, 256))
        # ABI: (A, B, C, M, N, K, stride_a, stride_b, stride_c)
        self.assertEqual(len(grouped_gemm_signature()), 9)

    def test_lowers_to_bf16_mfma_kernel(self):
        from rocke.instances.gfx950.grouped_gemm import (
            GroupedGemmSpec,
            build_grouped_gemm,
        )

        ir = _lower(build_grouped_gemm(GroupedGemmSpec(M=8192, N=1024, K=512, E=64))[0])
        self.assertIn("define amdgpu_kernel", ir)
        self.assertIn(_MFMA_BF16, ir)

    def test_lowering_is_deterministic(self):
        from rocke.instances.gfx950.grouped_gemm import (
            GroupedGemmSpec,
            build_grouped_gemm,
        )

        spec = GroupedGemmSpec(M=8192, N=1024, K=512, E=64)
        self.assertEqual(
            _lower(build_grouped_gemm(spec)[0]),
            _lower(build_grouped_gemm(spec)[0]),
        )

    def test_nn_layout_lowers(self):
        # b_rrr=True (NN weights) takes the transpose-read path; it must still
        # build and emit the bf16 MFMA cluster.
        from rocke.instances.gfx950.grouped_gemm import (
            GroupedGemmSpec,
            build_grouped_gemm,
        )

        ir = _lower(
            build_grouped_gemm(
                GroupedGemmSpec(M=8192, N=1024, K=512, E=64, b_rrr=True)
            )[0]
        )
        self.assertIn(_MFMA_BF16, ir)


class TestRaggedGemmGfx950(unittest.TestCase):
    def test_builds_and_lowers(self):
        from rocke.instances.gfx950.ragged_gemm import (
            RaggedGemmSpec,
            build_ragged_gemm,
            ragged_gemm_signature,
        )

        kernel, bs, tm, tn = build_ragged_gemm(RaggedGemmSpec(N=1024, K=512, E=64))
        self.assertEqual(kernel.name, "ragged_gemm")
        self.assertGreater(bs, 0)
        self.assertTrue(ragged_gemm_signature())
        ir = _lower(kernel)
        self.assertIn("define amdgpu_kernel", ir)
        self.assertIn(_MFMA_BF16, ir)


class TestRaggedMoeGfx950(unittest.TestCase):
    def test_builds_and_lowers(self):
        from rocke.instances.gfx950.ragged_moe import (
            RaggedMoeSpec,
            build_ragged_moe,
            ragged_moe_signature,
        )

        kernel, bs, tm, tn = build_ragged_moe(
            RaggedMoeSpec(N=1024, K=512, E=64, TOPK=2)
        )
        self.assertEqual(kernel.name, "ragged_moe")
        self.assertGreater(bs, 0)
        self.assertTrue(ragged_moe_signature())
        ir = _lower(kernel)
        self.assertIn("define amdgpu_kernel", ir)
        self.assertIn(_MFMA_BF16, ir)


if __name__ == "__main__":
    unittest.main(verbosity=2)
