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
constexpr int64_t K_SDPA_BPROP_TENSOR_Q_UID = 50;
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_Q_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_Q_STRIDES = {32768, 8192, 64, 1};

constexpr int64_t K_SDPA_BPROP_TENSOR_K_UID = 51;
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_K_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_K_STRIDES = {32768, 8192, 64, 1};

constexpr int64_t K_SDPA_BPROP_TENSOR_V_UID = 52;
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_V_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_V_STRIDES = {32768, 8192, 64, 1};

constexpr int64_t K_SDPA_BPROP_TENSOR_O_UID = 53;
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_O_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_O_STRIDES = {32768, 8192, 64, 1};

constexpr int64_t K_SDPA_BPROP_TENSOR_DO_UID = 54;
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_DO_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_DO_STRIDES = {32768, 8192, 64, 1};

constexpr int64_t K_SDPA_BPROP_TENSOR_STATS_UID = 55;
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_STATS_DIMS = {2, 4, 128, 1};
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_STATS_STRIDES = {512, 128, 1, 1};

// Required output tensors
constexpr int64_t K_SDPA_BPROP_TENSOR_DQ_UID = 56;
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_DQ_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_DQ_STRIDES = {32768, 8192, 64, 1};

constexpr int64_t K_SDPA_BPROP_TENSOR_DK_UID = 57;
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_DK_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_DK_STRIDES = {32768, 8192, 64, 1};

constexpr int64_t K_SDPA_BPROP_TENSOR_DV_UID = 58;
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_DV_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_DV_STRIDES = {32768, 8192, 64, 1};

// Optional tensors
constexpr int64_t K_SDPA_BPROP_TENSOR_SCALE_UID = 60;
constexpr int64_t K_SDPA_BPROP_TENSOR_ATTN_MASK_UID = 61;
constexpr int64_t K_SDPA_BPROP_TENSOR_SEQ_LEN_Q_UID = 62;
constexpr int64_t K_SDPA_BPROP_TENSOR_SEQ_LEN_KV_UID = 63;
constexpr int64_t K_SDPA_BPROP_TENSOR_SEED_UID = 64;
constexpr int64_t K_SDPA_BPROP_TENSOR_OFFSET_UID = 65;
constexpr int64_t K_SDPA_BPROP_TENSOR_DROPOUT_MASK_UID = 66;
constexpr int64_t K_SDPA_BPROP_TENSOR_DROPOUT_SCALE_UID = 67;
constexpr int64_t K_SDPA_BPROP_TENSOR_DROPOUT_SCALE_INV_UID = 68;
constexpr int64_t K_SDPA_BPROP_TENSOR_DBIAS_UID = 69;

// Scalar tensor (volume == 1) for scale/seed/offset
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_SCALAR_DIMS = {1, 1, 1, 1};
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_SCALAR_STRIDES = {1, 1, 1, 1};

// Attention mask: [batch=2, num_heads=4, seq_q=128, seq_kv=128]
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_ATTN_MASK_DIMS = {2, 4, 128, 128};
constexpr std::array<int64_t, 4> K_SDPA_BPROP_TENSOR_ATTN_MASK_STRIDES = {65536, 16384, 128, 1};

} // namespace hipdnn_tests::constants
