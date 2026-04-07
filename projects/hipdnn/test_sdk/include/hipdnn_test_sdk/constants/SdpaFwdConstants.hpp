// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Standard SDPA fprop constants for testing.
// Represents: batch=2, num_heads=4, seq_len=128, head_dim=64

// Required input tensors
constexpr int64_t K_SDPA_TENSOR_Q_UID = 40;
constexpr std::array<int64_t, 4> K_SDPA_TENSOR_Q_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_TENSOR_Q_STRIDES = {32768, 8192, 64, 1};

constexpr int64_t K_SDPA_TENSOR_K_UID = 41;
constexpr std::array<int64_t, 4> K_SDPA_TENSOR_K_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_TENSOR_K_STRIDES = {32768, 8192, 64, 1};

constexpr int64_t K_SDPA_TENSOR_V_UID = 42;
constexpr std::array<int64_t, 4> K_SDPA_TENSOR_V_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_TENSOR_V_STRIDES = {32768, 8192, 64, 1};

// Required output tensor
constexpr int64_t K_SDPA_TENSOR_O_UID = 43;
constexpr std::array<int64_t, 4> K_SDPA_TENSOR_O_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_TENSOR_O_STRIDES = {32768, 8192, 64, 1};

// Optional tensors (1D scalars or masks)
constexpr int64_t K_SDPA_TENSOR_ATTN_MASK_UID = 5;
constexpr int64_t K_SDPA_TENSOR_SCALE_UID = 6;
constexpr int64_t K_SDPA_TENSOR_SEQ_LEN_Q_UID = 7;
constexpr int64_t K_SDPA_TENSOR_SEQ_LEN_KV_UID = 8;
constexpr int64_t K_SDPA_TENSOR_SEED_UID = 9;
constexpr int64_t K_SDPA_TENSOR_OFFSET_UID = 10;
constexpr int64_t K_SDPA_TENSOR_DROPOUT_MASK_UID = 11;
constexpr int64_t K_SDPA_TENSOR_DROPOUT_SCALE_UID = 12;
constexpr int64_t K_SDPA_TENSOR_PAGE_TABLE_K_UID = 13;
constexpr int64_t K_SDPA_TENSOR_PAGE_TABLE_V_UID = 14;
constexpr int64_t K_SDPA_TENSOR_BLOCK_MASK_UID = 15;
constexpr int64_t K_SDPA_TENSOR_SINK_TOKEN_UID = 16;
constexpr int64_t K_SDPA_TENSOR_DESCALE_Q_UID = 17;
constexpr int64_t K_SDPA_TENSOR_DESCALE_K_UID = 18;
constexpr int64_t K_SDPA_TENSOR_DESCALE_V_UID = 19;
constexpr int64_t K_SDPA_TENSOR_DESCALE_S_UID = 20;
constexpr int64_t K_SDPA_TENSOR_SCALE_S_UID = 21;
constexpr int64_t K_SDPA_TENSOR_SCALE_O_UID = 22;
constexpr int64_t K_SDPA_TENSOR_STATS_UID = 23;
constexpr int64_t K_SDPA_TENSOR_MAX_UID = 24;
constexpr int64_t K_SDPA_TENSOR_SUM_EXP_UID = 25;
constexpr int64_t K_SDPA_TENSOR_RNG_DUMP_UID = 26;
constexpr int64_t K_SDPA_TENSOR_AMAX_S_UID = 27;
constexpr int64_t K_SDPA_TENSOR_AMAX_O_UID = 28;

constexpr std::array<int64_t, 4> K_SDPA_TENSOR_STATS_DIMS = {2, 4, 128, 1};
constexpr std::array<int64_t, 4> K_SDPA_TENSOR_STATS_STRIDES = {512, 128, 1, 1};

// Attention mask: [batch=2, num_heads=4, seq_q=128, seq_kv=128]
constexpr std::array<int64_t, 4> K_SDPA_TENSOR_ATTN_MASK_DIMS = {2, 4, 128, 128};
constexpr std::array<int64_t, 4> K_SDPA_TENSOR_ATTN_MASK_STRIDES = {65536, 16384, 128, 1};

// Scalar tensor (volume == 1) for descale/scale/amax/seed/offset
constexpr std::array<int64_t, 4> K_SDPA_TENSOR_SCALAR_DIMS = {1, 1, 1, 1};
constexpr std::array<int64_t, 4> K_SDPA_TENSOR_SCALAR_STRIDES = {1, 1, 1, 1};

} // namespace hipdnn_tests::constants
