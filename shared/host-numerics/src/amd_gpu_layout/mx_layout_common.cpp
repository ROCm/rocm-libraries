// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "detail/mx_layout_common.hpp"

#include <cstring>
#include <sstream>
#include <stdexcept>

#include "detail/checked_arithmetic.hpp"

namespace roc::host_numerics::amd_gpu_layout::detail {
using ::roc::host_numerics::detail::checkedAdd;
using ::roc::host_numerics::detail::checkedMultiply;

size_t roundUp(size_t value, size_t multiple) {
    if (multiple == 0) throw std::runtime_error("roundUp: multiple must be non-zero");
    const size_t remainder = value % multiple;
    return remainder == 0
               ? value
               : checkedAdd(value, multiple - remainder, "roundUp: size addition overflow");
}

size_t checkedProduct(const std::vector<size_t>& values) {
    size_t result = 1;
    for (const size_t value : values)
        result = checkedMultiply(result, value, "product: size multiplication overflow");
    return result;
}

std::vector<size_t> computeStrides(const std::vector<size_t>& sizes) {
    std::vector<size_t> strides(sizes.size());
    if (sizes.empty()) return strides;

    strides[0] = 1;
    for (size_t index = 1; index < sizes.size(); ++index)
        strides[index] = checkedMultiply(strides[index - 1], sizes[index - 1],
                                         "computeStrides: size multiplication overflow");
    return strides;
}

std::vector<size_t> computeShuffledStrides(const std::vector<size_t>& sizes,
                                           const std::vector<size_t>& dimensionOrder) {
    if (dimensionOrder.size() != sizes.size())
        throw std::runtime_error(
            "computeShuffledStrides: dimension order must contain every dimension");

    std::vector<size_t> strides(sizes.size(), 0);
    std::vector<bool> seen(sizes.size(), false);
    size_t stride = 1;
    for (const size_t index : dimensionOrder) {
        if (index >= sizes.size() || seen[index])
            throw std::runtime_error(
                "computeShuffledStrides: dimension order must be a permutation");
        seen[index] = true;
        strides[index] = stride;
        stride = checkedMultiply(stride, sizes[index],
                                 "computeShuffledStrides: size multiplication overflow");
    }
    return strides;
}

namespace {
size_t maximumOffset(const std::vector<size_t>& sizes, const std::vector<size_t>& strides,
                     const std::string& context) {
    size_t offset = 0;
    for (size_t dimension = 0; dimension < sizes.size(); ++dimension) {
        if (sizes[dimension] == 0) return 0;
        const size_t contribution = checkedMultiply(sizes[dimension] - 1, strides[dimension],
                                                    context + ": size multiplication overflow");
        offset = checkedAdd(offset, contribution, context + ": size addition overflow");
    }
    return offset;
}
}  // namespace

size_t validateShuffle(size_t inputElementCount, const DimensionShufflePlan& plan) {
    if (plan.sizes.size() != plan.destinationStrides.size() ||
        plan.sizes.size() != plan.sourceStrides.size())
        throw std::runtime_error("shuffleDims: size/stride dimension mismatch");
    if (plan.sizes.size() < 2) throw std::runtime_error("shuffleDims: need at least 2 dimensions");

    const size_t totalElements = checkedProduct(plan.sizes);
    if (inputElementCount != totalElements) {
        std::ostringstream message;
        message << "shuffleDims: input size " << inputElementCount << " doesn't match expected "
                << totalElements;
        throw std::runtime_error(message.str());
    }

    if (totalElements != 0 &&
        (maximumOffset(plan.sizes, plan.sourceStrides, "shuffleDims source") >= inputElementCount ||
         maximumOffset(plan.sizes, plan.destinationStrides, "shuffleDims destination") >=
             inputElementCount))
        throw std::runtime_error("shuffleDims: strides address outside the storage");
    return totalElements;
}

void validateByteStorage(size_t inputElementCount, size_t outputElementCount, size_t elementSize,
                         const std::string& context) {
    checkedMultiply(inputElementCount, elementSize,
                    context + " input bytes: size multiplication overflow");
    checkedMultiply(outputElementCount, elementSize,
                    context + " output bytes: size multiplication overflow");
}

void copyElement(const std::byte* input, size_t sourceIndex, std::byte* output,
                 size_t destinationIndex, size_t elementSize) {
    std::memcpy(output + destinationIndex * elementSize, input + sourceIndex * elementSize,
                elementSize);
}
}  // namespace roc::host_numerics::amd_gpu_layout::detail
