// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>

namespace roc::hostnumerics::amd_gpu_layout::detail {
size_t copyGfx950ScaleStorageBytes(const std::byte* input, size_t inputElementCount,
                                   size_t elementSize, size_t rowCount, size_t columnCount,
                                   std::byte* output);
}  // namespace roc::hostnumerics::amd_gpu_layout::detail
