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

Run:  PYTHONPATH=python python3 tests/instances/test_gfx950_grouped_gemm.py
"""

from __future__ import annotations

import os
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


class TestRaggedGemmHarnessGfx950(unittest.TestCase):
    """Host-side scheduling helpers of the ragged_gemm workflow harness
    (``examples/gfx950/grouped_gemm/ragged_gemm_hip.py``).

    The harness imports torch/numpy at module scope and the numeric path needs a
    GPU, so this loads the module by file path and skips cleanly when torch (or
    the rocke runtime) is unavailable. It exercises the pure CPU scheduling glue:
    ``make_group_sizes`` (per-expert row counts) and ``build_sched`` (per-tile
    expert/offset/valid-row schedule).
    """

    @classmethod
    def setUpClass(cls):
        import importlib.util
        import pathlib

        # tests/instances/<this> -> parents[2] == rocke/platform
        harness = (
            pathlib.Path(__file__).resolve().parents[2]
            / "python/rocke/examples/gfx950/grouped_gemm/ragged_gemm_hip.py"
        )
        if not harness.is_file():
            raise unittest.SkipTest(f"harness not found: {harness}")
        try:
            import torch  # noqa: F401

            spec = importlib.util.spec_from_file_location("_rgemm_harness", harness)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:  # torch / rocke runtime / numpy missing
            raise unittest.SkipTest(f"ragged_gemm_hip harness not importable: {e}")
        cls.h = mod

    def test_make_group_sizes_partitions_total(self):
        # Reference shape (M_total=524288, E=64); the bimodal distribution's
        # hardcoded 4096/12288 per-expert counts are sized for this.
        m_total, e = 524288, 64
        for dist in ("equal", "ragged", "bimodal"):
            sizes = self.h.make_group_sizes(m_total=m_total, e=e, dist=dist, seed=0)
            self.assertEqual(tuple(sizes.shape), (e,))
            self.assertEqual(int(sizes.sum().item()), m_total, dist)
            self.assertGreaterEqual(int(sizes.min().item()), 0, dist)

    def test_build_sched_is_consistent(self):
        sizes = self.h.make_group_sizes(m_total=16384, e=8, dist="ragged", seed=1)
        block_m = 256
        expert_ids, m_offsets, m_valid, num_tiles, m_starts = self.h.build_sched(
            sizes, block_m, device="cpu"
        )
        # One tile per (ceil(size/block_m)) per expert.
        expected_tiles = int(
            sum((int(s) + block_m - 1) // block_m for s in sizes.tolist())
        )
        self.assertEqual(num_tiles, expected_tiles)
        self.assertEqual(int(expert_ids.shape[0]), num_tiles)
        self.assertEqual(int(m_offsets.shape[0]), num_tiles)
        # Valid rows across all tiles must cover exactly the total rows.
        self.assertEqual(int(m_valid.sum().item()), int(sizes.sum().item()))
        # No tile claims more than a full block.
        self.assertLessEqual(int(m_valid.max().item()), block_m)
        # Per-expert start offsets are the exclusive prefix sum of sizes.
        self.assertEqual(int(m_starts[0].item()), 0)


class TestGfx950GroupedGemmCppByteIdentity(unittest.TestCase):
    """Dual-engine byte-identity for the grouped/ragged/MoE kernels.

    The C++ engine lowers any Python-built kernel through the family-agnostic
    serialized-IR seam (``rocke_engine.lower_serialized_ir``), so no hand-written
    C++ builder is needed: this asserts the C++ engine emits **byte-identical**
    LLVM IR to the Python engine via ``ROCKE_BACKEND=both`` (the differential
    oracle, which raises on any mismatch). Skipped when the ``rocke_engine``
    pybind module is not built/importable (build ``cpp/bindings`` to enable).
    """

    @classmethod
    def setUpClass(cls):
        try:
            import rocke_engine  # noqa: F401
        except Exception as e:  # not built / not on sys.path
            raise unittest.SkipTest(f"rocke_engine C++ binding not importable: {e}")

    def _assert_cpp_matches_python(self, kernel) -> None:
        import unittest.mock as mock

        from rocke.core.lower_llvm import lower_kernel_to_llvm, _resolve_llvm_flavor

        # backend='both' lowers with both engines and raises BackendMismatch on
        # any divergence; strict mode turns an engine rejection into an error
        # rather than a silent Python fallback. Scope the env to this call.
        with mock.patch.dict(
            os.environ, {"ROCKE_BACKEND": "both", "ROCKE_CPP_STRICT": "1"}
        ):
            lower_kernel_to_llvm(
                kernel, arch="gfx950", llvm_flavor=_resolve_llvm_flavor()
            )

    def test_grouped_gemm_cpp_byte_identical(self):
        from rocke.instances.gfx950.grouped_gemm import (
            GroupedGemmSpec,
            build_grouped_gemm,
        )

        self._assert_cpp_matches_python(
            build_grouped_gemm(GroupedGemmSpec(M=8192, N=1024, K=512, E=64))[0]
        )

    def test_ragged_gemm_cpp_byte_identical(self):
        from rocke.instances.gfx950.ragged_gemm import (
            RaggedGemmSpec,
            build_ragged_gemm,
        )

        self._assert_cpp_matches_python(
            build_ragged_gemm(RaggedGemmSpec(N=1024, K=512, E=64))[0]
        )

    def test_ragged_moe_cpp_byte_identical(self):
        from rocke.instances.gfx950.ragged_moe import (
            RaggedMoeSpec,
            build_ragged_moe,
        )

        self._assert_cpp_matches_python(
            build_ragged_moe(RaggedMoeSpec(N=1024, K=512, E=64, TOPK=2))[0]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
