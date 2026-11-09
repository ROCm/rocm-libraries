// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <roc/hostnumerics/amd_gpu_layout/mx.hpp>
#include <stdexcept>

#include "copy_permutation.hpp"
#include "detail/checked_arithmetic.hpp"

namespace roc::hostnumerics::amd_gpu_layout {
namespace {
size_t roundUp(size_t value, size_t multiple) {
    const size_t remainder = value % multiple;
    return remainder == 0
               ? value
               : ::roc::hostnumerics::detail::checkedAdd(value, multiple - remainder,
                                                         "MX scale storage size overflows size_t.");
}

ReshapePermutation scaleTransform(std::array<size_t, 2> shape, size_t blockSize,
                                  MxScaleStorageLayout layout) {
    if (blockSize == 0) throw std::runtime_error("MX scale block size must be nonzero.");
    const Shape natural{shape[0], shape[1]};
    switch (layout) {
        case MxScaleStorageLayout::Natural:
            return ReshapePermutation::identity(natural);
        case MxScaleStorageLayout::Tile32x8: {
            const size_t rows = roundUp(shape[0], 32), columns = roundUp(shape[1], 8);
            return {natural, Shape{rows, columns}, Shape{rows / 32, 2, 16, columns / 8, 2, 4},
                    Permutation{0, 3, 5, 2, 4, 1}};
        }
        case MxScaleStorageLayout::FastDimensionTiles: {
            if (blockSize != 16 && blockSize != 32)
                throw std::runtime_error(
                    "Fast dimension scale tiles require a block size of 16 or 32.");
            const size_t q = 128 / blockSize;
            const size_t columns = roundUp(shape[1], q);
            return {natural, Shape{shape[0], columns}, Shape{shape[0], columns / q, q},
                    Permutation{1, 0, 2}};
        }
    }
    throw std::runtime_error("Invalid MX scale storage layout.");
}
}  // namespace

MxScaleStoragePlan::MxScaleStoragePlan(std::array<size_t, 2> naturalShape, size_t blockSize,
                                       MxScaleStorageLayout layout)
    : m_layout(layout),
      m_blockSize(blockSize),
      m_transform(scaleTransform(naturalShape, blockSize, layout)) {}

std::vector<std::byte> copyMxScaleStorageToPhysicalLayout(const std::byte* naturalScaleStorage,
                                                          size_t naturalScaleByteCount,
                                                          const MxScaleStoragePlan& plan) {
    if (naturalScaleStorage == nullptr && naturalScaleByteCount != 0)
        throw std::runtime_error("MX scale layout input storage is null.");
    if (naturalScaleByteCount != plan.naturalByteCount())
        throw std::runtime_error("MX scale storage does not match its plan.");
    std::vector<std::byte> physical(plan.physicalByteCount());
    if (plan.layout() == MxScaleStorageLayout::Natural) {
        if (naturalScaleByteCount != 0)
            std::memcpy(physical.data(), naturalScaleStorage, naturalScaleByteCount);
    } else {
        detail::copyPermutationBytes(naturalScaleStorage, physical.data(), 1, plan.transform());
    }
    return physical;
}

}  // namespace roc::hostnumerics::amd_gpu_layout
