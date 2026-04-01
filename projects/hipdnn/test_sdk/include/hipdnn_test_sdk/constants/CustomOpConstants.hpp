// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace hipdnn_tests::constants
{

// Standard custom op constants for testing get/set of valid custom op operations.
// Represents: custom_op("test.op") with up to 2 inputs and 2 outputs, all 2D tensors (2,3).

constexpr int64_t K_CUSTOM_OP_INPUT_UID_0 = 800;
constexpr int64_t K_CUSTOM_OP_INPUT_UID_1 = 801;
constexpr int64_t K_CUSTOM_OP_OUTPUT_UID_0 = 802;
constexpr int64_t K_CUSTOM_OP_OUTPUT_UID_1 = 803;

inline const std::string K_CUSTOM_OP_ID = "test.op";
inline const std::vector<uint8_t> K_CUSTOM_OP_OPAQUE_DATA = {0xDE, 0xAD};

inline const std::array<int64_t, 2> K_CUSTOM_OP_TENSOR_DIMS = {2, 3};
inline const std::array<int64_t, 2> K_CUSTOM_OP_TENSOR_STRIDES = {3, 1};

} // namespace hipdnn_tests::constants
