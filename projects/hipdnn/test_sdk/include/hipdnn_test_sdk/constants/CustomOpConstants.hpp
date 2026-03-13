// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace hipdnn_tests::constants
{

// Standard custom op constants for testing get/set of valid custom op operations.
// Represents: custom_op("test.op") with 2 inputs and 1 output, all 2D tensors (2,3).

constexpr int64_t K_CUSTOM_OP_INPUT_UID_0 = 100;
constexpr int64_t K_CUSTOM_OP_INPUT_UID_1 = 101;
constexpr int64_t K_CUSTOM_OP_OUTPUT_UID_0 = 200;

inline const std::string K_CUSTOM_OP_ID = "test.op";
inline const std::vector<uint8_t> K_CUSTOM_OP_OPAQUE_DATA = {0xDE, 0xAD};

} // namespace hipdnn_tests::constants
