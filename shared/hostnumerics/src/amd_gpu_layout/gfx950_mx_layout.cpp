// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "detail/gfx950_mx_layout.hpp"

#include <sstream>
#include <stdexcept>

#include "detail/checked_arithmetic.hpp"
#include "detail/mx_layout_common.hpp"

namespace roc::hostnumerics::amd_gpu_layout::detail {
using ::roc::hostnumerics::detail::checkedMultiply;

namespace {
struct GFX950ScalePlan {
    size_t rowCount = 0;
    size_t columnCount = 0;
    size_t paddedRowCount = 0;
    size_t paddedColumnCount = 0;
    size_t outputElementCount = 0;
    DimensionShufflePlan shuffle;
};

GFX950ScalePlan makeGFX950ScalePlan(size_t inputElementCount, size_t rowCount, size_t columnCount) {
    GFX950ScalePlan plan;
    plan.rowCount = rowCount;
    plan.columnCount = columnCount;
    const size_t inputElements = checkedMultiply(
        plan.rowCount, plan.columnCount, "gfx950 MX scale layout: size multiplication overflow");
    if (inputElements != inputElementCount) {
        std::ostringstream message;
        message << "gfx950 MX scale layout input size " << inputElementCount
                << " doesn't match rows * columns = " << inputElements;
        throw std::runtime_error(message.str());
    }

    plan.paddedRowCount = roundUp(plan.rowCount, 32);
    plan.paddedColumnCount = roundUp(plan.columnCount, 8);
    plan.outputElementCount =
        checkedMultiply(plan.paddedRowCount, plan.paddedColumnCount,
                        "gfx950 MX scale layout: size multiplication overflow");
    plan.shuffle.sizes = {plan.paddedRowCount / 32, 2, 16, plan.paddedColumnCount / 8, 2, 4};
    plan.shuffle.sourceStrides = {
        checkedMultiply(32, plan.paddedColumnCount,
                        "gfx950 MX scale layout strides: size multiplication overflow"),
        checkedMultiply(16, plan.paddedColumnCount,
                        "gfx950 MX scale layout strides: size multiplication overflow"),
        plan.paddedColumnCount,
        8,
        4,
        1};
    plan.shuffle.destinationStrides =
        computeShuffledStrides(plan.shuffle.sizes, {1, 4, 2, 5, 3, 0});
    validateShuffle(plan.outputElementCount, plan.shuffle);
    return plan;
}
}  // namespace

size_t copyGfx950ScaleStorageBytes(const std::byte* input, size_t inputElementCount,
                                   size_t elementSize, size_t rowCount, size_t columnCount,
                                   std::byte* output) {
    const GFX950ScalePlan plan = makeGFX950ScalePlan(inputElementCount, rowCount, columnCount);
    validateByteStorage(inputElementCount, plan.outputElementCount, elementSize,
                        "gfx950 MX scale layout");
    if (output == nullptr) return plan.outputElementCount;

    forEachShuffledElement(
        plan.shuffle, plan.outputElementCount,
        [&](size_t paddedSourceIndex, size_t destinationIndex) {
            const size_t sourceRow = paddedSourceIndex / plan.paddedColumnCount;
            const size_t sourceColumn = paddedSourceIndex % plan.paddedColumnCount;
            if (sourceRow < plan.rowCount && sourceColumn < plan.columnCount) {
                const size_t sourceIndex = sourceRow * plan.columnCount + sourceColumn;
                copyElement(input, sourceIndex, output, destinationIndex, elementSize);
            }
        });
    return plan.outputElementCount;
}
}  // namespace roc::hostnumerics::amd_gpu_layout::detail
