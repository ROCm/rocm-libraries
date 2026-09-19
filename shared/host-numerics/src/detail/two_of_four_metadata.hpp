// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include "structured_sparsity_plan.hpp"

namespace roc::host_numerics::detail {
Shape validateTwoOfFourMetadataProblem(const TwoOfFourMetadataArguments& problem);
Shape validateTwoOfFourMetadataRequest(const TwoOfFourMetadataInvocation& request);
uint8_t twoOfFourMetadataNibble(uint8_t first, uint8_t second);
}  // namespace roc::host_numerics::detail
