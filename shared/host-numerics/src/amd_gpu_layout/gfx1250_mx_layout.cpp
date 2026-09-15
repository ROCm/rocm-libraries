// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "detail/gfx1250_mx_layout.hpp"

#include <sstream>
#include <stdexcept>

#include "detail/checked_arithmetic.hpp"
#include "detail/mx_layout_common.hpp"
#include "mx_threading.hpp"

namespace roc::host_numerics::amd_gpu_layout::detail {
using ::roc::host_numerics::detail::checkedMultiply;

namespace {
struct GFX1250ScalePlan {
    size_t slowDimension = 0;
    size_t fastDimension = 0;
    size_t dimK = 0;
    size_t outputElementCount = 0;
};

GFX1250ScalePlan makeGFX1250ScalePlan(size_t inputElementCount, size_t slowDimension,
                                      size_t fastDimension, size_t mxBlock) {
    if (mxBlock != 16 && mxBlock != 32)
        throw std::runtime_error("gfx1250 MX scale layout requires a block size of 16 or 32");

    const size_t expectedInput =
        checkedMultiply(slowDimension, fastDimension,
                        "gfx1250 MX scale layout input: size multiplication overflow");
    if (expectedInput != inputElementCount) {
        std::ostringstream message;
        message << "gfx1250 MX scale layout input size " << inputElementCount
                << " doesn't match slowDim*fastDim = " << expectedInput;
        throw std::runtime_error(message.str());
    }

    GFX1250ScalePlan plan;
    plan.slowDimension = slowDimension;
    plan.fastDimension = fastDimension;
    plan.dimK = 128 / mxBlock;
    const size_t paddedFastDimension = roundUp(fastDimension, plan.dimK);
    plan.outputElementCount =
        checkedMultiply(slowDimension, paddedFastDimension,
                        "gfx1250 MX scale layout output: size multiplication overflow");
    return plan;
}
}  // namespace

size_t copyGfx1250ScaleStorageBytes(const std::byte* input, size_t inputElementCount,
                                    size_t elementSize, size_t slowDimension, size_t fastDimension,
                                    size_t mxBlock, std::byte* output) {
    const GFX1250ScalePlan plan =
        makeGFX1250ScalePlan(inputElementCount, slowDimension, fastDimension, mxBlock);
    validateByteStorage(inputElementCount, plan.outputElementCount, elementSize,
                        "gfx1250 MX scale layout");
    if (output == nullptr) return plan.outputElementCount;

    const size_t copyGroupCount = plan.outputElementCount / plan.dimK;
    parallelForChunks(copyGroupCount, plan.outputElementCount, [&](size_t begin, size_t end) {
        for (size_t copyGroup = begin; copyGroup < end; ++copyGroup) {
            const size_t slowIndex = copyGroup % plan.slowDimension;
            const size_t tile = copyGroup / plan.slowDimension;
            const size_t outputBase = copyGroup * plan.dimK;
            const size_t sourceFastBase = tile * plan.dimK;
            for (size_t elementInTile = 0; elementInTile < plan.dimK; ++elementInTile) {
                const size_t sourceFast = sourceFastBase + elementInTile;
                if (sourceFast < plan.fastDimension) {
                    const size_t sourceIndex = slowIndex * plan.fastDimension + sourceFast;
                    copyElement(input, sourceIndex, output, outputBase + elementInTile,
                                elementSize);
                }
            }
        }
    });
    return plan.outputElementCount;
}
}  // namespace roc::host_numerics::amd_gpu_layout::detail
