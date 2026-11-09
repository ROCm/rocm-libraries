// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstring>
#include <roc/hostnumerics/amd_gpu_layout/mx.hpp>
#include <stdexcept>
#include <utility>

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

}  // namespace

MxScaleStoragePlan MxScaleStoragePlan::natural(std::array<size_t, 2> shape) {
    return MxScaleStoragePlan{ReshapePermutation::identity(Shape{shape[0], shape[1]})};
}

MxScaleStoragePlan MxScaleStoragePlan::tile32x8(std::array<size_t, 2> shape) {
    const Shape natural{shape[0], shape[1]};
    const size_t rows = roundUp(shape[0], 32), columns = roundUp(shape[1], 8);
    return MxScaleStoragePlan{{natural, Shape{rows, columns},
                               Shape{rows / 32, 2, 16, columns / 8, 2, 4},
                               Permutation{0, 3, 5, 2, 4, 1}}};
}

MxScaleStoragePlan MxScaleStoragePlan::fastDimensionTiles(std::array<size_t, 2> shape,
                                                          size_t scalesPerTile) {
    if (scalesPerTile == 0) throw std::runtime_error("MX scale tile width must be nonzero.");
    const size_t columns = roundUp(shape[1], scalesPerTile);
    return MxScaleStoragePlan{{Shape{shape[0], shape[1]}, Shape{shape[0], columns},
                               Shape{shape[0], columns / scalesPerTile, scalesPerTile},
                               Permutation{1, 0, 2}}};
}

MxScaleStoragePlan::MxScaleStoragePlan(ReshapePermutation transform)
    : m_transform(std::move(transform)) {}

std::vector<std::byte> copyMxScaleStorageToPhysicalLayout(const std::byte* naturalScaleStorage,
                                                          size_t naturalScaleByteCount,
                                                          const MxScaleStoragePlan& plan) {
    if (naturalScaleStorage == nullptr && naturalScaleByteCount != 0)
        throw std::runtime_error("MX scale layout input storage is null.");
    if (naturalScaleByteCount != plan.naturalByteCount())
        throw std::runtime_error("MX scale storage does not match its plan.");
    std::vector<std::byte> physical(plan.physicalByteCount());
    const auto& transform = plan.transform();
    // Reshaping alone preserves flat order when there is no padding or axis reordering.
    if (transform.naturalShape() == transform.paddedShape() &&
        std::is_sorted(transform.order().begin(), transform.order().end())) {
        if (naturalScaleByteCount != 0)
            std::memcpy(physical.data(), naturalScaleStorage, naturalScaleByteCount);
    } else {
        detail::copyPermutationBytes(naturalScaleStorage, physical.data(), 1, transform);
    }
    return physical;
}

}  // namespace roc::hostnumerics::amd_gpu_layout
