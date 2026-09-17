# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU lowering tests for the gfx1250 Qwen3 model-glue kernels.

Covers the full-offload gap fillers: fused QK-norm+RoPE, token-embedding gather,
and the greedy (argmax) sampler. On-device numeric verification lives under
``examples/gfx1250/{attention,model}/*_verify.py`` (validated exact on gfx950 as
the kernels are arch-neutral; they also assemble for gfx1250).
"""

from __future__ import annotations

import unittest

from rocke.core.lower_llvm import lower_kernel_to_llvm


class TestQwen3Glue(unittest.TestCase):
    def test_qk_norm_rope_lowers(self):
        from rocke.instances.gfx1250.qwen3_qk_norm_rope import (
            Qwen3QkNormRopeSpec,
            build_qwen3_qk_norm_rope,
            qwen3_qk_norm_rope_grid,
        )

        for nh, lay in ((32, "half"), (4, "half"), (32, "interleaved")):
            spec = Qwen3QkNormRopeSpec(num_heads=nh, rope_layout=lay)
            ll = lower_kernel_to_llvm(
                build_qwen3_qk_norm_rope(spec, arch="gfx1250"), arch="gfx1250"
            )
            self.assertIn("define amdgpu_kernel", ll)
            self.assertIn("rsq", ll)  # RMS normalization -> llvm.amdgcn.rsq.f32
        # one thread per (token, head): T=2, nh=32 -> 64 -> 1 block of 64
        self.assertEqual(
            qwen3_qk_norm_rope_grid(2, Qwen3QkNormRopeSpec(num_heads=32)), (1, 1, 1)
        )

    def test_token_embedding_lowers(self):
        from rocke.instances.gfx1250.qwen3_token_embedding import (
            Qwen3TokenEmbeddingSpec,
            build_qwen3_token_embedding,
            qwen3_token_embedding_grid,
        )

        spec = Qwen3TokenEmbeddingSpec(hidden=2048, vec=8)
        ll = lower_kernel_to_llvm(
            build_qwen3_token_embedding(spec, arch="gfx1250"), arch="gfx1250"
        )
        self.assertIn("define amdgpu_kernel", ll)
        # T=4, hidden=2048, vec=8 -> 4*256=1024 vecs / 256 = 4 blocks
        self.assertEqual(qwen3_token_embedding_grid(4, spec), (4, 1, 1))

    def test_greedy_sampler_lowers(self):
        from rocke.instances.gfx1250.qwen3_sampler import (
            Qwen3GreedySamplerSpec,
            build_qwen3_greedy_sampler,
            qwen3_greedy_sampler_grid,
        )

        for dt in ("f32", "bf16"):
            ll = lower_kernel_to_llvm(
                build_qwen3_greedy_sampler(
                    Qwen3GreedySamplerSpec(logits_dtype=dt), arch="gfx1250"
                ),
                arch="gfx1250",
            )
            self.assertIn("define amdgpu_kernel", ll)
            self.assertIn("barrier", ll)  # block LDS index-reduction
        self.assertEqual(
            qwen3_greedy_sampler_grid(8, Qwen3GreedySamplerSpec()), (8, 1, 1)
        )

    def test_exports(self):
        import rocke.instances.gfx1250 as g

        for name in (
            "build_qwen3_qk_norm_rope",
            "build_qwen3_token_embedding",
            "build_qwen3_greedy_sampler",
        ):
            self.assertTrue(hasattr(g, name), name)


if __name__ == "__main__":
    unittest.main()
