// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>

namespace roc::host_numerics::amd_gpu_layout::detail {
size_t copyGfx1250ScaleStorageBytes(const std::byte* input, size_t inputElementCount,
                                    size_t elementSize, size_t slowDimension, size_t fastDimension,
                                    size_t mxBlock, std::byte* output);
}  // namespace roc::host_numerics::amd_gpu_layout::detail
