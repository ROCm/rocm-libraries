// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstring>
#include <roc/hostnumerics/reshape_permutation.hpp>

#include "parallel_chunks.hpp"

namespace roc::hostnumerics::amd_gpu_layout::detail {

// The destination already contains padding. Each natural element has a unique
// destination, so copying chunks concurrently cannot overwrite another chunk.
inline void copyPermutationBytes(const std::byte* input, std::byte* output, size_t elementSize,
                                 const ReshapePermutation& transform) {
    const size_t count = transform.naturalElementCount();
    parallelForChunks(count, count, [&](size_t begin, size_t end) noexcept {
        for (size_t index = begin; index < end; ++index)
            std::memcpy(output + transform.physicalIndex(index) * elementSize,
                        input + index * elementSize, elementSize);
    });
}

}  // namespace roc::hostnumerics::amd_gpu_layout::detail
