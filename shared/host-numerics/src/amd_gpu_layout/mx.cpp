// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <roc/host_numerics/amd_gpu_layout/mx.hpp>

#include "detail/checked_arithmetic.hpp"
#include "detail/gfx1250_mx_layout.hpp"
#include "detail/gfx950_mx_layout.hpp"
#include "detail/pre_swizzle.hpp"

namespace roc::host_numerics::amd_gpu_layout {
MxScaleStorageLayout mxScaleStorageLayoutForArchitectureName(std::string_view architectureName) {
    const auto matchesArchitectureToken = [architectureName](std::string_view token) {
        return architectureName == token ||
               (architectureName.size() > token.size() &&
                architectureName.compare(0, token.size(), token) == 0 &&
                architectureName[token.size()] == ':');
    };

    if (matchesArchitectureToken("gfx950")) return MxScaleStorageLayout::Gfx950;
    if (matchesArchitectureToken("gfx1250")) return MxScaleStorageLayout::Gfx1250;
    return MxScaleStorageLayout::Natural;
}

MxScaleStoragePlan planMxScaleStorage(std::array<size_t, 2> naturalShape, size_t blockSize,
                                      MxScaleStorageLayout layout) {
    const size_t naturalByteCount = ::roc::host_numerics::detail::checkedMultiply(
        naturalShape[0], naturalShape[1], "MX natural scale storage");
    size_t physicalByteCount = naturalByteCount;

    switch (layout) {
        case MxScaleStorageLayout::Natural:
            break;
        case MxScaleStorageLayout::Gfx950:
            physicalByteCount = detail::copyGfx950ScaleStorageBytes(
                nullptr, naturalByteCount, 1, {naturalShape[0], naturalShape[1]}, nullptr);
            break;
        case MxScaleStorageLayout::Gfx1250:
            physicalByteCount = detail::copyGfx1250ScaleStorageBytes(
                nullptr, naturalByteCount, 1, naturalShape[0], naturalShape[1], blockSize, nullptr);
            break;
        default:
            throw std::invalid_argument("Invalid MX scale storage layout.");
    }

    return {
        .layout = layout,
        .naturalShape = naturalShape,
        .blockSize = blockSize,
        .naturalByteCount = naturalByteCount,
        .physicalByteCount = physicalByteCount,
    };
}

std::vector<std::byte> copyMxScaleStorageToPhysicalLayout(const std::byte* naturalScaleStorage,
                                                          size_t naturalScaleByteCount,
                                                          const MxScaleStoragePlan& plan) {
    const MxScaleStoragePlan expected =
        planMxScaleStorage(plan.naturalShape, plan.blockSize, plan.layout);
    if (plan.naturalByteCount != expected.naturalByteCount ||
        plan.physicalByteCount != expected.physicalByteCount)
        throw std::invalid_argument("MX scale storage plan is inconsistent.");
    if (naturalScaleStorage == nullptr && naturalScaleByteCount != 0)
        throw std::invalid_argument("MX scale layout input storage is null.");
    if (naturalScaleByteCount != expected.naturalByteCount)
        throw std::invalid_argument("MX scale storage does not match its plan.");

    if (plan.layout == MxScaleStorageLayout::Natural) {
        if (naturalScaleByteCount == 0) return {};
        return {naturalScaleStorage, naturalScaleStorage + naturalScaleByteCount};
    }

    std::vector<std::byte> output(expected.physicalByteCount, std::byte{0});
    switch (plan.layout) {
        case MxScaleStorageLayout::Gfx950: {
            const std::vector<size_t> dimensions{plan.naturalShape[0], plan.naturalShape[1]};
            detail::copyGfx950ScaleStorageBytes(naturalScaleStorage, naturalScaleByteCount, 1,
                                                dimensions, output.data());
            return output;
        }
        case MxScaleStorageLayout::Gfx1250: {
            detail::copyGfx1250ScaleStorageBytes(naturalScaleStorage, naturalScaleByteCount, 1,
                                                 plan.naturalShape[0], plan.naturalShape[1],
                                                 plan.blockSize, output.data());
            return output;
        }
        case MxScaleStorageLayout::Natural:
            break;
    }
    throw std::invalid_argument("Invalid MX scale storage layout.");
}
}  // namespace roc::host_numerics::amd_gpu_layout
