# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# rocke Heuristics: ML-based kernel selection (GEMM + conv + SDPA + MoE + norm).
#
# Training data is generated natively from the rocke sweep ecosystem. The
# op-parameterized generator gen_sweep_data.py (--op gemm|conv|sdpa|moe|norm)
# enumerates each op's spec variants x shape corpus, builds them, and writes an
# op-appropriate feature parquet; gen_gemm_sweep_data.py remains the GEMM shim it
# delegates to. Conv reuses the GEMM 72-feature engine via the implicit-GEMM
# projection; SDPA uses FmhaFeatureEngine (68-feature parity with the C++
# ml_extract_fmha_features); MoE/norm use minimal latency/bandwidth engines. The
# downstream train / predict / evaluate / search / feature_engine modules are
# data-source-agnostic and consume the canonical per-op parquet schema.
