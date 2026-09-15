// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <vector>

namespace roc::host_numerics::amd_gpu_layout::detail {
size_t copyGfx950ScaleStorageBytes(const std::byte* input, size_t inputElementCount,
                                   size_t elementSize, const std::vector<size_t>& sizes,
                                   std::byte* output);
}  // namespace roc::host_numerics::amd_gpu_layout::detail
