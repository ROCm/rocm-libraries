// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Standard SDPA bprop constants for testing.
// Represents: batch=2, num_heads=4, seq_len=128, head_dim=64

// Required input tensors
constexpr int64_t K_SDPA_BWD_TENSOR_Q_UID = 90;
constexpr std::array<int64_t, 4> K_SDPA_BWD_TENSOR_Q_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_BWD_TENSOR_Q_STRIDES = {32768, 8192, 64, 1};

constexpr int64_t K_SDPA_BWD_TENSOR_K_UID = 91;
constexpr std::array<int64_t, 4> K_SDPA_BWD_TENSOR_K_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_BWD_TENSOR_K_STRIDES = {32768, 8192, 64, 1};

constexpr int64_t K_SDPA_BWD_TENSOR_V_UID = 92;
constexpr std::array<int64_t, 4> K_SDPA_BWD_TENSOR_V_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_BWD_TENSOR_V_STRIDES = {32768, 8192, 64, 1};

constexpr int64_t K_SDPA_BWD_TENSOR_O_UID = 93;
constexpr std::array<int64_t, 4> K_SDPA_BWD_TENSOR_O_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_BWD_TENSOR_O_STRIDES = {32768, 8192, 64, 1};

constexpr int64_t K_SDPA_BWD_TENSOR_DO_UID = 94;
constexpr std::array<int64_t, 4> K_SDPA_BWD_TENSOR_DO_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_BWD_TENSOR_DO_STRIDES = {32768, 8192, 64, 1};

constexpr int64_t K_SDPA_BWD_TENSOR_STATS_UID = 95;
constexpr std::array<int64_t, 4> K_SDPA_BWD_TENSOR_STATS_DIMS = {2, 4, 128, 1};
constexpr std::array<int64_t, 4> K_SDPA_BWD_TENSOR_STATS_STRIDES = {512, 128, 1, 1};

// Required output tensors
constexpr int64_t K_SDPA_BWD_TENSOR_DQ_UID = 96;
constexpr std::array<int64_t, 4> K_SDPA_BWD_TENSOR_DQ_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_BWD_TENSOR_DQ_STRIDES = {32768, 8192, 64, 1};

constexpr int64_t K_SDPA_BWD_TENSOR_DK_UID = 97;
constexpr std::array<int64_t, 4> K_SDPA_BWD_TENSOR_DK_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_BWD_TENSOR_DK_STRIDES = {32768, 8192, 64, 1};

constexpr int64_t K_SDPA_BWD_TENSOR_DV_UID = 98;
constexpr std::array<int64_t, 4> K_SDPA_BWD_TENSOR_DV_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_BWD_TENSOR_DV_STRIDES = {32768, 8192, 64, 1};

// Optional tensors
constexpr int64_t K_SDPA_BWD_TENSOR_SCALE_UID = 100;
constexpr int64_t K_SDPA_BWD_TENSOR_ATTN_MASK_UID = 101;
constexpr int64_t K_SDPA_BWD_TENSOR_SEQ_LEN_Q_UID = 102;
constexpr int64_t K_SDPA_BWD_TENSOR_SEQ_LEN_KV_UID = 103;
constexpr int64_t K_SDPA_BWD_TENSOR_SEED_UID = 104;
constexpr int64_t K_SDPA_BWD_TENSOR_OFFSET_UID = 105;
constexpr int64_t K_SDPA_BWD_TENSOR_DROPOUT_MASK_UID = 106;
constexpr int64_t K_SDPA_BWD_TENSOR_DROPOUT_SCALE_UID = 107;
constexpr int64_t K_SDPA_BWD_TENSOR_DROPOUT_SCALE_INV_UID = 108;
constexpr int64_t K_SDPA_BWD_TENSOR_DBIAS_UID = 109;

// Scalar tensor (volume == 1) for scale/seed/offset
constexpr std::array<int64_t, 1> K_SDPA_BWD_TENSOR_SCALAR_DIMS = {1};
constexpr std::array<int64_t, 1> K_SDPA_BWD_TENSOR_SCALAR_STRIDES = {1};

// Attention mask: [batch=2, num_heads=4, seq_q=128, seq_kv=128]
constexpr std::array<int64_t, 4> K_SDPA_BWD_TENSOR_ATTN_MASK_DIMS = {2, 4, 128, 128};
constexpr std::array<int64_t, 4> K_SDPA_BWD_TENSOR_ATTN_MASK_STRIDES = {65536, 16384, 128, 1};

// Sequence length tensors: [batch=2, 1, 1, 1]
constexpr std::array<int64_t, 4> K_SDPA_BWD_TENSOR_SEQ_LEN_DIMS = {2, 1, 1, 1};
constexpr std::array<int64_t, 4> K_SDPA_BWD_TENSOR_SEQ_LEN_STRIDES = {1, 1, 1, 1};

// Dropout mask: same shape as attention weights [batch=2, num_heads=4, seq_q=128, seq_kv=128]
constexpr std::array<int64_t, 4> K_SDPA_BWD_TENSOR_DROPOUT_MASK_DIMS = {2, 4, 128, 128};
constexpr std::array<int64_t, 4> K_SDPA_BWD_TENSOR_DROPOUT_MASK_STRIDES = {65536, 16384, 128, 1};

// dBias output: [batch=2, num_heads=4, seq_q=128, seq_kv=128]
constexpr std::array<int64_t, 4> K_SDPA_BWD_TENSOR_DBIAS_DIMS = {2, 4, 128, 128};
constexpr std::array<int64_t, 4> K_SDPA_BWD_TENSOR_DBIAS_STRIDES = {65536, 16384, 128, 1};

} // namespace hipdnn_tests::constants
