// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace roc::host_numerics::detail {
std::vector<std::byte> packRawValues(std::span<const uint8_t> rawValues, uint16_t bitsPerValue,
                                     int threadCount);
}  // namespace roc::host_numerics::detail
