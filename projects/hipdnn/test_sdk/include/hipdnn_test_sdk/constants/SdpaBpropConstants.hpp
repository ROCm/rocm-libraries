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
constexpr int64_t K_SDPA_BPROP_TENSOR_Q_UID = 1900;
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_Q_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_Q_STRIDES = {32768, 8192, 64, 1};

constexpr int64_t K_SDPA_BPROP_TENSOR_K_UID = 1901;
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_K_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_K_STRIDES = {32768, 8192, 64, 1};

constexpr int64_t K_SDPA_BPROP_TENSOR_V_UID = 1902;
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_V_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_V_STRIDES = {32768, 8192, 64, 1};

constexpr int64_t K_SDPA_BPROP_TENSOR_O_UID = 1903;
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_O_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_O_STRIDES = {32768, 8192, 64, 1};

constexpr int64_t K_SDPA_BPROP_TENSOR_DO_UID = 1904;
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_DO_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_DO_STRIDES = {32768, 8192, 64, 1};

constexpr int64_t K_SDPA_BPROP_TENSOR_STATS_UID = 1905;
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_STATS_DIMS = {2, 4, 128, 1};
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_STATS_STRIDES = {512, 128, 1, 1};

// Required output tensors
constexpr int64_t K_SDPA_BPROP_TENSOR_DQ_UID = 1906;
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_DQ_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_DQ_STRIDES = {32768, 8192, 64, 1};

constexpr int64_t K_SDPA_BPROP_TENSOR_DK_UID = 1907;
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_DK_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_DK_STRIDES = {32768, 8192, 64, 1};

constexpr int64_t K_SDPA_BPROP_TENSOR_DV_UID = 1908;
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_DV_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_DV_STRIDES = {32768, 8192, 64, 1};

// Optional tensors
constexpr int64_t K_SDPA_BPROP_TENSOR_SCALE_UID = 1909;
constexpr int64_t K_SDPA_BPROP_TENSOR_ATTN_MASK_UID = 1910;
constexpr int64_t K_SDPA_BPROP_TENSOR_SEQ_LEN_Q_UID = 1911;
constexpr int64_t K_SDPA_BPROP_TENSOR_SEQ_LEN_KV_UID = 1912;
constexpr int64_t K_SDPA_BPROP_TENSOR_SEED_UID = 1913;
constexpr int64_t K_SDPA_BPROP_TENSOR_OFFSET_UID = 1914;
constexpr int64_t K_SDPA_BPROP_TENSOR_DROPOUT_MASK_UID = 1915;
constexpr int64_t K_SDPA_BPROP_TENSOR_DROPOUT_SCALE_UID = 1916;
constexpr int64_t K_SDPA_BPROP_TENSOR_DROPOUT_SCALE_INV_UID = 1917;
constexpr int64_t K_SDPA_BPROP_TENSOR_DBIAS_UID = 1918;

// Scalar tensor (volume == 1) for scale/seed/offset
constexpr std::array<int64_t, 1> K_SDPA_BPROP_TENSOR_SCALAR_DIMS = {1};
constexpr std::array<int64_t, 1> K_SDPA_BPROP_TENSOR_SCALAR_STRIDES = {1};

// Attention mask: [batch=2, num_heads=4, seq_q=128, seq_kv=128]
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_ATTN_MASK_DIMS = {2, 4, 128, 128};
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_ATTN_MASK_STRIDES = {65536, 16384, 128, 1};

// Sequence length tensors: [batch=2, 1, 1, 1]
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_SEQ_LEN_DIMS = {2, 1, 1, 1};
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_SEQ_LEN_STRIDES = {1, 1, 1, 1};

// Dropout mask: same shape as attention weights [batch=2, num_heads=4, seq_q=128, seq_kv=128]
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_DROPOUT_MASK_DIMS = {2, 4, 128, 128};
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_DROPOUT_MASK_STRIDES = {65536, 16384, 128, 1};

// dBias output: [batch=2, num_heads=4, seq_q=128, seq_kv=128]
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_DBIAS_DIMS = {2, 4, 128, 128};
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_DBIAS_STRIDES = {65536, 16384, 128, 1};

} // namespace hipdnn_tests::constants
