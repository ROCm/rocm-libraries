// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstring>
#include <roc/hostnumerics/reshape_permutation.hpp>

namespace roc::hostnumerics::amd_gpu_layout::detail {

// The destination already contains padding; copy only the natural elements.
inline void copyPermutationBytes(const std::byte* input, std::byte* output, size_t elementSize,
                                 const ReshapePermutation& transform) {
    for (size_t index = 0; index < transform.naturalElementCount(); ++index)
        std::memcpy(output + transform.physicalIndex(index) * elementSize,
                    input + index * elementSize, elementSize);
}

}  // namespace roc::hostnumerics::amd_gpu_layout::detail
