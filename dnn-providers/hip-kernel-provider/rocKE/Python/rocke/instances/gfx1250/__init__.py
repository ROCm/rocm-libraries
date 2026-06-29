# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250-class (gfx1250) instance builders.

gfx1250 is a CDNA multi-chip device on the GFX12 programming model (wave32,
WMMA, no MFMA). The primary fp16/bf16 WMMA atom is 16x16x32 (K=32), distinct
from the RDNA4 gfx1201 16x16x16. See
``examples/gfx1250/attention/gfx1250_universal_attention_plan.md``.
"""

from .wmma_attention_fwd import (
    WmmaAttentionFwdSpec,
    build_wmma_attention_fwd,
    wmma_attention_fwd_grid,
)

# Qwen3-30B-A3B attention now goes through the unified attention instance
# (instances/common/attention_unified.py + attention_tiled_2d / attention_tiled_3d).
# Only the KV-cache-side kernels remain as dedicated gfx1250 builders.
from .qwen3_kv_cache import (
    Qwen3KvAppendRopeSpec,
    Qwen3KvDequantSpec,
    build_qwen3_kv_append_rope,
    build_qwen3_kv_dequant_smoke,
)
from .block_scaled_gemm import (
    BlockScaledGemmSpec,
    block_scaled_gemm_grid,
    block_scaled_gemm_signature,
    build_block_scaled_gemm,
    is_valid_spec as is_valid_block_scaled_gemm_spec,
)
from .fused_moe_fp8 import Gfx1250Fp8Moe, Gfx1250Fp8MoeSpec
from .fused_moe_mega_wmma import (
    FusedMegaWmmaSpec,
    build_moe_fused_mega_wmma,
    moe_fused_mega_wmma_grid,
    moe_fused_mega_wmma_signature,
)

# Qwen3-30B-A3B model-glue ops (fill the full-offload gaps): fused QK-norm+RoPE,
# token-embedding gather, greedy (argmax) sampler. Pure elementwise/reduction
# math (no WMMA), so arch-neutral; namespaced here as part of the gfx1250 day-0
# operator set.
from .qwen3_qk_norm_rope import (
    Qwen3QkNormRopeSpec,
    build_qwen3_qk_norm_rope,
    qwen3_qk_norm_rope_grid,
    qwen3_qk_norm_rope_signature,
)
from .qwen3_token_embedding import (
    Qwen3TokenEmbeddingSpec,
    build_qwen3_token_embedding,
    qwen3_token_embedding_grid,
    qwen3_token_embedding_signature,
)
from .qwen3_sampler import (
    Qwen3GreedySamplerSpec,
    build_qwen3_greedy_sampler,
    qwen3_greedy_sampler_grid,
    qwen3_greedy_sampler_signature,
)
from .wmma_gemm import WmmaGemmSpec, build_wmma_gemm, wmma_gemm_grid

__all__ = [
    "BlockScaledGemmSpec",
    "block_scaled_gemm_grid",
    "block_scaled_gemm_signature",
    "build_block_scaled_gemm",
    "is_valid_block_scaled_gemm_spec",
    "Gfx1250Fp8Moe",
    "Gfx1250Fp8MoeSpec",
    "FusedMegaWmmaSpec",
    "build_moe_fused_mega_wmma",
    "moe_fused_mega_wmma_grid",
    "moe_fused_mega_wmma_signature",
    "WmmaGemmSpec",
    "build_wmma_gemm",
    "wmma_gemm_grid",
    "WmmaAttentionFwdSpec",
    "build_wmma_attention_fwd",
    "wmma_attention_fwd_grid",
    "Qwen3KvAppendRopeSpec",
    "Qwen3KvDequantSpec",
    "build_qwen3_kv_append_rope",
    "build_qwen3_kv_dequant_smoke",
    "Qwen3QkNormRopeSpec",
    "build_qwen3_qk_norm_rope",
    "qwen3_qk_norm_rope_grid",
    "qwen3_qk_norm_rope_signature",
    "Qwen3TokenEmbeddingSpec",
    "build_qwen3_token_embedding",
    "qwen3_token_embedding_grid",
    "qwen3_token_embedding_signature",
    "Qwen3GreedySamplerSpec",
    "build_qwen3_greedy_sampler",
    "qwen3_greedy_sampler_grid",
    "qwen3_greedy_sampler_signature",
]
