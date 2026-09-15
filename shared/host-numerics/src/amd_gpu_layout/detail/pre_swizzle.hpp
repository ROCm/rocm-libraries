// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <vector>

namespace roc::host_numerics::amd_gpu_layout::detail {
size_t preSwizzleBytes(const std::byte* input, size_t inputElementCount, size_t elementSize,
                       const std::vector<size_t>& sizes, const std::vector<size_t>& preSwizzleSize,
                       const std::vector<size_t>& preTileSize, std::byte* output);
}  // namespace roc::host_numerics::amd_gpu_layout::detail
