// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace hipdnn_tests::constants
{

// Standard Gdn constants for testing get/set of valid operations.
// These represent "any valid gdn" — specific values are not significant.

constexpr int64_t K_GDN_TENSOR_Q_UID = 2000;
constexpr std::array<int64_t, 3> K_GDN_TENSOR_Q_DIMS = {8, 2, 8};
constexpr std::array<int64_t, 3> K_GDN_TENSOR_Q_STRIDES = {16, 8, 1};

constexpr int64_t K_GDN_TENSOR_K_UID = 2001;
constexpr std::array<int64_t, 3> K_GDN_TENSOR_K_DIMS = {8, 2, 8};
constexpr std::array<int64_t, 3> K_GDN_TENSOR_K_STRIDES = {16, 8, 1};

constexpr int64_t K_GDN_TENSOR_V_UID = 2002;
constexpr std::array<int64_t, 3> K_GDN_TENSOR_V_DIMS = {8, 2, 8};
constexpr std::array<int64_t, 3> K_GDN_TENSOR_V_STRIDES = {16, 8, 1};

constexpr int64_t K_GDN_TENSOR_G_UID = 2003;
constexpr std::array<int64_t, 2> K_GDN_TENSOR_G_DIMS = {8, 2};
constexpr std::array<int64_t, 2> K_GDN_TENSOR_G_STRIDES = {2, 1};

constexpr int64_t K_GDN_TENSOR_BETA_UID = 2004;
constexpr std::array<int64_t, 2> K_GDN_TENSOR_BETA_DIMS = {8, 2};
constexpr std::array<int64_t, 2> K_GDN_TENSOR_BETA_STRIDES = {2, 1};

constexpr int64_t K_GDN_TENSOR_CU_SEQLENS_UID = 2005;
constexpr std::array<int64_t, 1> K_GDN_TENSOR_CU_SEQLENS_DIMS = {3};
constexpr std::array<int64_t, 1> K_GDN_TENSOR_CU_SEQLENS_STRIDES = {1};

constexpr int64_t K_GDN_TENSOR_INITIAL_STATE_UID = 2006;
constexpr std::array<int64_t, 4> K_GDN_TENSOR_INITIAL_STATE_DIMS = {2, 2, 8, 8};
constexpr std::array<int64_t, 4> K_GDN_TENSOR_INITIAL_STATE_STRIDES = {128, 64, 8, 1};

constexpr int64_t K_GDN_TENSOR_A_LOG_UID = 2007;
constexpr std::array<int64_t, 1> K_GDN_TENSOR_A_LOG_DIMS = {2};
constexpr std::array<int64_t, 1> K_GDN_TENSOR_A_LOG_STRIDES = {1};

constexpr int64_t K_GDN_TENSOR_DT_BIAS_UID = 2008;
constexpr std::array<int64_t, 1> K_GDN_TENSOR_DT_BIAS_DIMS = {2};
constexpr std::array<int64_t, 1> K_GDN_TENSOR_DT_BIAS_STRIDES = {1};

constexpr int64_t K_GDN_TENSOR_STATE_INDICES_UID = 2009;
constexpr std::array<int64_t, 1> K_GDN_TENSOR_STATE_INDICES_DIMS = {2};
constexpr std::array<int64_t, 1> K_GDN_TENSOR_STATE_INDICES_STRIDES = {1};

constexpr int64_t K_GDN_TENSOR_O_UID = 2010;
constexpr std::array<int64_t, 3> K_GDN_TENSOR_O_DIMS = {8, 2, 8};
constexpr std::array<int64_t, 3> K_GDN_TENSOR_O_STRIDES = {16, 8, 1};

constexpr int64_t K_GDN_TENSOR_FINAL_STATE_UID = 2011;
constexpr std::array<int64_t, 4> K_GDN_TENSOR_FINAL_STATE_DIMS = {2, 2, 8, 8};
constexpr std::array<int64_t, 4> K_GDN_TENSOR_FINAL_STATE_STRIDES = {128, 64, 8, 1};

constexpr int64_t K_GDN_TENSOR_STATE_CHECKPOINTS_UID = 2012;
constexpr std::array<int64_t, 4> K_GDN_TENSOR_STATE_CHECKPOINTS_DIMS = {4, 2, 8, 8};
constexpr std::array<int64_t, 4> K_GDN_TENSOR_STATE_CHECKPOINTS_STRIDES = {128, 64, 8, 1};

} // namespace hipdnn_tests::constants
