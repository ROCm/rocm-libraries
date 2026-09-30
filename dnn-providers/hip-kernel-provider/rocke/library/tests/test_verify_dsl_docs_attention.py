# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Verify the attention (kernels.*) doc claims extracted from verify_dsl_docs.

This file carries the portion of ``platform/dsl_docs/development/verify_dsl_docs.py``
that exercised ``kernels.*`` attention builders and was removed from the platform
file to keep ``platform/dsl_docs/`` free of ``kernels`` imports.

The single test class mirrors the ``check("UnifiedAttention 2D + 3D + Reduce:
build + LLVM", t_attention)`` case from the original verifier.
"""

from __future__ import annotations

import unittest

from rocke import lower_kernel_to_llvm

from kernels import (
    UnifiedAttentionProblem,
    UnifiedAttention2DSpec,
    UnifiedAttention3DSpec,
    UnifiedAttentionReduceSpec,
    build_unified_attention_2d,
    build_unified_attention_3d,
    build_unified_attention_reduce,
)


class TestVerifyDslDocsAttention(unittest.TestCase):
    """Attention doc-example verification (originally in verify_dsl_docs.py)."""

    def _make_problem(self) -> UnifiedAttentionProblem:
        return UnifiedAttentionProblem(
            total_q=1,
            num_seqs=1,
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            max_seqlen_q=1,
            max_seqlen_k=1024,
            dtype="fp16",
        )

    def test_unified_attention_2d_build_and_lower(self) -> None:
        p = self._make_problem()
        ir = lower_kernel_to_llvm(
            build_unified_attention_2d(UnifiedAttention2DSpec(problem=p))
        )
        self.assertIsInstance(ir, str)
        self.assertGreater(len(ir), 0)

    def test_unified_attention_3d_build_and_lower(self) -> None:
        p = self._make_problem()
        ir = lower_kernel_to_llvm(
            build_unified_attention_3d(UnifiedAttention3DSpec(problem=p))
        )
        self.assertIsInstance(ir, str)
        self.assertGreater(len(ir), 0)

    def test_unified_attention_reduce_build_and_lower(self) -> None:
        p = self._make_problem()
        ir = lower_kernel_to_llvm(
            build_unified_attention_reduce(
                UnifiedAttentionReduceSpec(problem=p, num_segments=8)
            )
        )
        self.assertIsInstance(ir, str)
        self.assertGreater(len(ir), 0)


if __name__ == "__main__":
    unittest.main()
