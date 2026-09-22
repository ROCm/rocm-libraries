// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <roc/host_numerics/tensor.hpp>
#include <span>

namespace roc::host_numerics::detail {
// Canonical implementation plan for elementwise Tensor copies and conversions.
// Tensor owns the public NumPy-like allocation API; internal consumers may execute
// independent plan blocks in parallel without reimplementing traversal or codecs.
class TensorConversion {
   public:
    TensorConversion(const Tensor& source, Tensor destination,
                     const ScalarConversionOptions& options);

    size_t elementCount() const;
    size_t blockCount() const;
    bool canParallelize() const;

    // Blocks may execute concurrently only when canParallelize() is true.
    void copyBlock(size_t block) const;
    void copyAll() const;

   private:
    using CopyRangeFunction = void (*)(std::span<const std::byte> sourceStorage,
                                       ptrdiff_t sourceOffset, ptrdiff_t sourceStride,
                                       std::span<std::byte> destinationStorage,
                                       ptrdiff_t destinationOffset, ptrdiff_t destinationStride,
                                       size_t count, const ScalarConversionOptions& options);

    enum class Traversal {
        Linear,
        TiledMatrix,
        General,
    };

    Tensor m_source;
    Tensor m_destination;
    ScalarConversionOptions m_options;
    CopyRangeFunction m_copyRange = nullptr;
    Traversal m_traversal = Traversal::General;
    IndexOrder m_indexOrder = IndexOrder::LastDimensionFastest;
    bool m_sourceFirstDimensionFastest = false;
    size_t m_blockCount = 0;
    size_t m_tileExtent = 0;
};
}  // namespace roc::host_numerics::detail
