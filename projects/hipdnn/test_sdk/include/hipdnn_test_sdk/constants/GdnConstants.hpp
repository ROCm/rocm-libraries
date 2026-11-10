// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Standard Gdn constants for testing get/set of valid operations.
// These represent "any valid gdn" — specific values are not significant.

constexpr int64_t K_GDN_TENSOR_Q_UID = 9800;
constexpr std::array<int64_t, 3> K_GDN_TENSOR_Q_DIMS = {4, 2, 4};
constexpr std::array<int64_t, 3> K_GDN_TENSOR_Q_STRIDES = {8, 4, 1};

constexpr int64_t K_GDN_TENSOR_K_UID = 9801;
constexpr std::array<int64_t, 3> K_GDN_TENSOR_K_DIMS = {4, 2, 4};
constexpr std::array<int64_t, 3> K_GDN_TENSOR_K_STRIDES = {8, 4, 1};

constexpr int64_t K_GDN_TENSOR_V_UID = 9802;
constexpr std::array<int64_t, 3> K_GDN_TENSOR_V_DIMS = {4, 2, 4};
constexpr std::array<int64_t, 3> K_GDN_TENSOR_V_STRIDES = {8, 4, 1};

constexpr int64_t K_GDN_TENSOR_G_UID = 9803;
constexpr std::array<int64_t, 2> K_GDN_TENSOR_G_DIMS = {4, 2};
constexpr std::array<int64_t, 2> K_GDN_TENSOR_G_STRIDES = {2, 1};

constexpr int64_t K_GDN_TENSOR_BETA_UID = 9804;
constexpr std::array<int64_t, 2> K_GDN_TENSOR_BETA_DIMS = {4, 2};
constexpr std::array<int64_t, 2> K_GDN_TENSOR_BETA_STRIDES = {2, 1};

constexpr int64_t K_GDN_TENSOR_CU_SEQLENS_UID = 9805;
constexpr std::array<int64_t, 1> K_GDN_TENSOR_CU_SEQLENS_DIMS = {3};
constexpr std::array<int64_t, 1> K_GDN_TENSOR_CU_SEQLENS_STRIDES = {1};

constexpr int64_t K_GDN_TENSOR_O_UID = 9806;
constexpr std::array<int64_t, 3> K_GDN_TENSOR_O_DIMS = {4, 2, 4};
constexpr std::array<int64_t, 3> K_GDN_TENSOR_O_STRIDES = {8, 4, 1};

constexpr int64_t K_GDN_TENSOR_INITIAL_STATE_UID = 9807;
constexpr std::array<int64_t, 4> K_GDN_TENSOR_INITIAL_STATE_DIMS = {2, 2, 4, 4};
constexpr std::array<int64_t, 4> K_GDN_TENSOR_INITIAL_STATE_STRIDES = {32, 16, 4, 1};

constexpr int64_t K_GDN_TENSOR_A_LOG_UID = 9808;
constexpr std::array<int64_t, 1> K_GDN_TENSOR_A_LOG_DIMS = {2};
constexpr std::array<int64_t, 1> K_GDN_TENSOR_A_LOG_STRIDES = {1};

constexpr int64_t K_GDN_TENSOR_DT_BIAS_UID = 9809;
constexpr std::array<int64_t, 1> K_GDN_TENSOR_DT_BIAS_DIMS = {2};
constexpr std::array<int64_t, 1> K_GDN_TENSOR_DT_BIAS_STRIDES = {1};

constexpr int64_t K_GDN_TENSOR_STATE_INDICES_UID = 9810;
constexpr std::array<int64_t, 1> K_GDN_TENSOR_STATE_INDICES_DIMS = {2};
constexpr std::array<int64_t, 1> K_GDN_TENSOR_STATE_INDICES_STRIDES = {1};

constexpr int64_t K_GDN_TENSOR_FINAL_STATE_UID = 9811;
constexpr std::array<int64_t, 4> K_GDN_TENSOR_FINAL_STATE_DIMS = {2, 2, 4, 4};
constexpr std::array<int64_t, 4> K_GDN_TENSOR_FINAL_STATE_STRIDES = {32, 16, 4, 1};

constexpr int64_t K_GDN_TENSOR_STATE_CHECKPOINTS_UID = 9812;
constexpr std::array<int64_t, 4> K_GDN_TENSOR_STATE_CHECKPOINTS_DIMS = {4, 2, 4, 4};
constexpr std::array<int64_t, 4> K_GDN_TENSOR_STATE_CHECKPOINTS_STRIDES = {32, 16, 4, 1};

} // namespace hipdnn_tests::constants
