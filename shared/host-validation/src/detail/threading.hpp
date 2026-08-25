// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <roc/host_validation/tensor.hpp>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace roc::host_validation::detail {
inline uint64_t strideMagnitude(ptrdiff_t stride) {
    if (stride >= 0) return static_cast<uint64_t>(stride);
    return static_cast<uint64_t>(-(stride + 1)) + 1;
}

inline bool hasProvablyIndependentElements(const Tensor& destination) {
    if (scalarTypeInfo(destination.type()).storageBits % 8 != 0) return false;

    std::vector<std::pair<uint64_t, size_t>> dimensions;
    dimensions.reserve(destination.shape().rank());
    for (size_t dimension = 0; dimension < destination.shape().rank(); ++dimension) {
        const size_t extent = destination.shape()[dimension];
        if (extent <= 1) continue;

        const uint64_t stride = strideMagnitude(destination.layout().strides()[dimension]);
        if (stride == 0) return false;
        dimensions.emplace_back(stride, extent);
    }
    std::ranges::sort(dimensions);

    uint64_t addressedSpan = 1;
    for (const auto& [stride, extent] : dimensions) {
        if (stride < addressedSpan) return false;
        const uint64_t additionalExtent = static_cast<uint64_t>(extent - 1);
        if (additionalExtent > (std::numeric_limits<uint64_t>::max() - addressedSpan) / stride)
            return false;
        addressedSpan += additionalExtent * stride;
    }
    return true;
}

inline int operationThreadCount(size_t workItemCount, size_t minimumWorkItemsPerThread = 4096,
                                int defaultMaximumThreadCount = 8) {
#ifdef _OPENMP
    if (workItemCount == 0 || omp_in_parallel()) return 1;

    const int runtimeMaximum = std::max(1, omp_get_max_threads());
    const char* configuredThreadCount = std::getenv("OMP_NUM_THREADS");
    const int maximum = configuredThreadCount != nullptr && configuredThreadCount[0] != '\0'
                            ? runtimeMaximum
                            : std::min(runtimeMaximum, defaultMaximumThreadCount);
    const size_t usefulThreadCount = std::max(
        size_t{1}, workItemCount / minimumWorkItemsPerThread +
                       static_cast<size_t>(workItemCount % minimumWorkItemsPerThread != 0));
    return static_cast<int>(std::min(usefulThreadCount, static_cast<size_t>(maximum)));
#else
    (void)workItemCount;
    (void)minimumWorkItemsPerThread;
    (void)defaultMaximumThreadCount;
    return 1;
#endif
}
}  // namespace roc::host_validation::detail
