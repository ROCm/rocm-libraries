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
constexpr int64_t K_SDPA_TENSOR_Q_UID = 1800;
constexpr std::array<int64_t, 4> K_SDPA_TENSOR_Q_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_TENSOR_Q_STRIDES = {32768, 8192, 64, 1};

constexpr int64_t K_SDPA_TENSOR_K_UID = 1801;
constexpr std::array<int64_t, 4> K_SDPA_TENSOR_K_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_TENSOR_K_STRIDES = {32768, 8192, 64, 1};

constexpr int64_t K_SDPA_TENSOR_V_UID = 1802;
constexpr std::array<int64_t, 4> K_SDPA_TENSOR_V_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_TENSOR_V_STRIDES = {32768, 8192, 64, 1};

// Required output tensor
constexpr int64_t K_SDPA_TENSOR_O_UID = 1803;
constexpr std::array<int64_t, 4> K_SDPA_TENSOR_O_DIMS = {2, 4, 128, 64};
constexpr std::array<int64_t, 4> K_SDPA_TENSOR_O_STRIDES = {32768, 8192, 64, 1};

// Optional tensors (1D scalars or masks)
constexpr int64_t K_SDPA_TENSOR_ATTN_MASK_UID = 1804;
constexpr int64_t K_SDPA_TENSOR_SCALE_UID = 1805;
constexpr int64_t K_SDPA_TENSOR_SEQ_LEN_Q_UID = 1806;
constexpr int64_t K_SDPA_TENSOR_SEQ_LEN_KV_UID = 1807;
constexpr int64_t K_SDPA_TENSOR_SEED_UID = 1808;
constexpr int64_t K_SDPA_TENSOR_OFFSET_UID = 1809;
constexpr int64_t K_SDPA_TENSOR_DROPOUT_MASK_UID = 1810;
constexpr int64_t K_SDPA_TENSOR_DROPOUT_SCALE_UID = 1811;
constexpr int64_t K_SDPA_TENSOR_PAGE_TABLE_K_UID = 1812;
constexpr int64_t K_SDPA_TENSOR_PAGE_TABLE_V_UID = 1813;
constexpr int64_t K_SDPA_TENSOR_BLOCK_MASK_UID = 1814;
constexpr int64_t K_SDPA_TENSOR_SINK_TOKEN_UID = 1815;
constexpr int64_t K_SDPA_TENSOR_DESCALE_Q_UID = 1816;
constexpr int64_t K_SDPA_TENSOR_DESCALE_K_UID = 1817;
constexpr int64_t K_SDPA_TENSOR_DESCALE_V_UID = 1818;
constexpr int64_t K_SDPA_TENSOR_DESCALE_S_UID = 1819;
constexpr int64_t K_SDPA_TENSOR_SCALE_S_UID = 1820;
constexpr int64_t K_SDPA_TENSOR_SCALE_O_UID = 1821;
constexpr int64_t K_SDPA_TENSOR_STATS_UID = 1822;
constexpr int64_t K_SDPA_TENSOR_MAX_UID = 1823;
constexpr int64_t K_SDPA_TENSOR_SUM_EXP_UID = 1824;
constexpr int64_t K_SDPA_TENSOR_RNG_DUMP_UID = 1825;
constexpr int64_t K_SDPA_TENSOR_AMAX_S_UID = 1826;
constexpr int64_t K_SDPA_TENSOR_AMAX_O_UID = 1827;

constexpr std::array<int64_t, 4> K_SDPA_TENSOR_STATS_DIMS = {2, 4, 128, 1};
constexpr std::array<int64_t, 4> K_SDPA_TENSOR_STATS_STRIDES = {512, 128, 1, 1};

// Attention mask: [batch=2, num_heads=4, seq_q=128, seq_kv=128]
constexpr std::array<int64_t, 4> K_SDPA_TENSOR_ATTN_MASK_DIMS = {2, 4, 128, 128};
constexpr std::array<int64_t, 4> K_SDPA_TENSOR_ATTN_MASK_STRIDES = {65536, 16384, 128, 1};

// Scalar tensor (volume == 1) for descale/scale/amax/seed/offset
constexpr std::array<int64_t, 4> K_SDPA_TENSOR_SCALAR_DIMS = {1, 1, 1, 1};
constexpr std::array<int64_t, 4> K_SDPA_TENSOR_SCALAR_STRIDES = {1, 1, 1, 1};

} // namespace hipdnn_tests::constants
