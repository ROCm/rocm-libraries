// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <span>

namespace roc::host_numerics::detail {
bool byteRangesOverlap(std::span<const std::byte> left, std::span<const std::byte> right);
}  // namespace roc::host_numerics::detail
