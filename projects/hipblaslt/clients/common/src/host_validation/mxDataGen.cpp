// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Product-private hipBLASLt translation and architecture-selected upload
// transforms around component-owned MX tensor generation.

#include <hipblaslt/host_validation/Types.hpp>
#include <hipblaslt/host_validation/mxDataGen.hpp>
#include <roc/host_validation/mx.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <roc/host_validation/amd_gpu_layout/mx.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{
    using roc::host_validation::MxDataRecipe;
    using roc::host_validation::MxGenerationProblem;
    using roc::host_validation::MxScaleGenerationMode;
    using roc::host_validation::ScalarType;

    std::pair<int, int> randIntRangeFor(ScalarType dataType)
    {
        switch(dataType)
        {
        case ScalarType::Float4E2M1:
            return {-4, 4};
        case ScalarType::Float6E2M3:
            return {-7, 7};
        case ScalarType::Float6E3M2:
            return {-28, 28};
        default:
            return {1, 10};
        }
    }

    double normDistStdDevFor(ScalarType dataType)
    {
        return dataType == ScalarType::Float4E2M1 ? 5.0 : 1.0;
    }

    ScalarType scaleScalarType(hipDataType scaleType)
    {
        if(scaleType == HIP_R_8F_E4M3)
            return ScalarType::E4M3;
        return hipblaslt::host_validation::scalarType(scaleType);
    }

    std::optional<MxScaleGenerationMode> scaleGenerationMode(std::string_view initMethod)
    {
        if(initMethod == "Zeros" || initMethod == "zero")
            return MxScaleGenerationMode::Minimum;
        if(initMethod == "Identity" || initMethod == "Ones" || initMethod == "NegOnes"
           || initMethod == "DenormMins" || initMethod == "DenormMaxs" || initMethod == "Infs"
           || initMethod == "Sequential" || initMethod == "RowIndex" || initMethod == "ColIndex"
           || initMethod == "Checkerboard" || initMethod == "ScaledDiagonal"
           || initMethod == "rand_int")
            return MxScaleGenerationMode::One;
        if(initMethod == "Twos")
            return MxScaleGenerationMode::Two;
        if(initMethod == "MaxVals")
            return MxScaleGenerationMode::Maximum;
        if(initMethod == "NaNs")
            return MxScaleGenerationMode::NaN;
        return std::nullopt;
    }

    MxDataRecipe generationRecipe(std::string_view initMethod,
                                  ScalarType       dataType,
                                  float            minimum,
                                  float            maximum)
    {
        if(initMethod == "Sequential")
            return MxDataRecipe::sequential();
        if(initMethod == "RowIndex")
            return MxDataRecipe::rowIndex();
        if(initMethod == "ColIndex")
            return MxDataRecipe::columnIndex();
        if(initMethod == "Checkerboard")
            return MxDataRecipe::checkerboard();
        if(initMethod == "ScaledDiagonal")
            return MxDataRecipe::scaledDiagonal();
        if(initMethod == "Identity")
            return MxDataRecipe::identity();
        if(initMethod == "Ones")
            return MxDataRecipe::constant(1.0);
        if(initMethod == "Zeros" || initMethod == "zero")
            return MxDataRecipe::constant(0.0);
        if(initMethod == "Twos")
            return MxDataRecipe::constant(2.0);
        if(initMethod == "NegOnes")
            return MxDataRecipe::constant(-1.0);
        if(initMethod == "MaxVals")
            return MxDataRecipe::typeMaximum();
        if(initMethod == "DenormMins")
            return MxDataRecipe::typeDenormalMinimum();
        if(initMethod == "DenormMaxs")
            return MxDataRecipe::typeDenormalMaximum();
        if(initMethod == "NaNs")
            return MxDataRecipe::typeNaN();
        if(initMethod == "Infs")
            return MxDataRecipe::typeInfinity();
        if(initMethod == "Bounded")
            return MxDataRecipe::bounded({.lower = minimum, .upper = maximum});
        if(initMethod == "uniform_01")
            return MxDataRecipe::bounded({.lower = 0.0, .upper = 1.0});
        if(initMethod == "hpl")
            return MxDataRecipe::bounded({.lower = -0.5, .upper = 0.5});
        if(initMethod == "uniform_low_precision")
            return MxDataRecipe::bounded({.lower = -6.0, .upper = 6.0});
        if(initMethod == "TrigonometricFromFloat" || initMethod == "trig_float")
            return MxDataRecipe::trigonometric();
        if(initMethod == "norm_dist")
            return MxDataRecipe::normal(
                {.mean = 0.0, .standardDeviation = normDistStdDevFor(dataType)});
        if(initMethod == "rand_int")
        {
            const auto [lower, upper] = randIntRangeFor(dataType);
            return MxDataRecipe::uniformInteger({.lower = lower, .upper = upper});
        }
        throw std::invalid_argument("Unsupported hipBLASLt MX initialization mode.");
    }

    std::vector<uint8_t> swizzleScaleBytes(std::vector<uint8_t> scaleBytes,
                                           MXScaleLayout        scaleLayout,
                                           size_t               slowDimension,
                                           size_t               fastDimension,
                                           size_t               blockSize)
    {
        switch(scaleLayout)
        {
        case MXScaleLayout::GFX950:
            return roc::host_validation::amd_gpu_layout::preSwizzleScalesGFX950(
                scaleBytes, {slowDimension, fastDimension});
        case MXScaleLayout::GFX1250:
            if(blockSize > 0)
                return roc::host_validation::amd_gpu_layout::preSwizzleScalesGFX1250(
                    scaleBytes, slowDimension, fastDimension, blockSize);
            break;
        case MXScaleLayout::None:
            break;
        }
        return scaleBytes;
    }
} // namespace

std::vector<float> generateMXInput(hipDataType            dataType,
                                   hipDataType            scaleType,
                                   std::span<uint8_t>     data,
                                   std::span<uint8_t>     scale,
                                   uint64_t               row,
                                   uint64_t               col,
                                   uint64_t               stride,
                                   int const              scaleBlockRowSize,
                                   int const              scaleBlockColSize,
                                   MXScaleLayout          scaleLayout,
                                   std::string_view const initMethod,
                                   float                  min_val,
                                   float                  max_val,
                                   std::string_view const scaleInitMethod,
                                   uint32_t               seed)
{
    if(data.data() == nullptr || scale.data() == nullptr)
        throw std::invalid_argument("generateMXInput requires non-null data and scale outputs.");
    if constexpr(sizeof(size_t) < sizeof(uint64_t))
    {
        if(row > std::numeric_limits<size_t>::max() || col > std::numeric_limits<size_t>::max())
            throw std::overflow_error("generateMXInput dimensions exceed size_t.");
    }
    if(stride > static_cast<uint64_t>(std::numeric_limits<ptrdiff_t>::max()))
        throw std::overflow_error("generateMXInput leading dimension exceeds ptrdiff_t.");
    if(scaleBlockRowSize <= 0 || scaleBlockColSize <= 0)
        throw std::invalid_argument("generateMXInput scale block dimensions must be positive.");
    const size_t blockRows    = static_cast<size_t>(scaleBlockRowSize);
    const size_t blockColumns = static_cast<size_t>(scaleBlockColSize);
    if(blockRows > std::numeric_limits<size_t>::max() / blockColumns)
        throw std::overflow_error("generateMXInput scale block size overflow.");
    if(blockRows > 1 && blockColumns > 1)
        throw std::invalid_argument("generateMXInput supports blocking along one tensor axis.");

    const ScalarType hostDataType = hipblaslt::host_validation::scalarType(dataType);
    MxGenerationProblem problem;
    problem.dataType  = hostDataType;
    problem.scaleType = scaleScalarType(scaleType);
    problem.shape = roc::host_validation::Shape{static_cast<size_t>(row), static_cast<size_t>(col)};
    problem.leadingDimension = static_cast<ptrdiff_t>(stride);
    problem.blockSize        = blockRows * blockColumns;
    problem.blockAxis        = blockColumns > 1 ? 1 : 0;
    problem.data             = generationRecipe(initMethod, hostDataType, min_val, max_val);
    const std::string_view scaleMethod
        = scaleInitMethod.empty() ? initMethod : scaleInitMethod;
    if(!scaleInitMethod.empty())
        (void)generationRecipe(scaleInitMethod, hostDataType, -1.0f, 1.0f);
    if(const auto scaleMode = scaleGenerationMode(scaleMethod))
        problem.scale = *scaleMode;
    problem.seed = seed;

    roc::host_validation::MxGenerationResult result = roc::host_validation::generateMx(problem);

    std::vector<uint8_t> scaleBytes(result.scales.storage().size());
    std::memcpy(scaleBytes.data(), result.scales.storage().data(), scaleBytes.size());
    const size_t blockedScaleExtent
        = (problem.shape[problem.blockAxis] + problem.blockSize - 1) / problem.blockSize;
    const size_t fastScaleExtent = problem.blockAxis == 0 ? blockedScaleExtent : problem.shape[0];
    const size_t slowScaleExtent = problem.blockAxis == 0 ? problem.shape[1] : blockedScaleExtent;
    scaleBytes                   = swizzleScaleBytes(
        std::move(scaleBytes), scaleLayout, slowScaleExtent, fastScaleExtent, problem.blockSize);

    if(data.size() < result.data.storage().size())
        throw std::invalid_argument("generateMXInput data output is too small.");
    if(scale.size() < scaleBytes.size())
        throw std::invalid_argument("generateMXInput scale output is too small.");

    std::memcpy(data.data(), result.data.storage().data(), result.data.storage().size());
    std::memcpy(scale.data(), scaleBytes.data(), scaleBytes.size());

    std::vector<float> reference(problem.shape.elementCount());
    std::memcpy(
        reference.data(), result.reference.storage().data(), reference.size() * sizeof(float));
    return reference;
}

void restrideMXScaleBufferKFast(std::span<uint8_t> buffer,
                                size_t             compactFreeDim,
                                size_t             compactKBlocks,
                                size_t             paddedKBlocks,
                                size_t             elemBytes)
{
    if(compactKBlocks == paddedKBlocks || compactFreeDim == 0)
        return;
    if(buffer.data() == nullptr)
        throw std::invalid_argument("restrideMXScaleBufferKFast: buffer must not be null");
    if(elemBytes == 0)
        throw std::invalid_argument("restrideMXScaleBufferKFast: element size must be non-zero");
    if(paddedKBlocks < compactKBlocks)
        throw std::invalid_argument(
            "restrideMXScaleBufferKFast: padded extent is smaller than compact extent");
    if(compactKBlocks > std::numeric_limits<size_t>::max() / elemBytes
       || paddedKBlocks > std::numeric_limits<size_t>::max() / elemBytes)
        throw std::overflow_error("restrideMXScaleBufferKFast: row size overflow");
    size_t const compactRow = compactKBlocks * elemBytes;
    size_t const paddedRow  = paddedKBlocks * elemBytes;
    if(compactFreeDim > std::numeric_limits<size_t>::max() / paddedRow)
        throw std::overflow_error("restrideMXScaleBufferKFast: buffer size overflow");
    if(buffer.size() < compactFreeDim * paddedRow)
        throw std::invalid_argument("restrideMXScaleBufferKFast: destination buffer is too small");
    size_t const padTail = paddedRow - compactRow;
    for(size_t f = compactFreeDim; f-- > 1;)
    {
        std::memmove(buffer.data() + f * paddedRow, buffer.data() + f * compactRow, compactRow);
        std::memset(buffer.data() + f * paddedRow + compactRow, 0x00, padTail);
    }
    std::memset(buffer.data() + compactRow, 0x00, padTail);
}

MXScaleLayout mxScaleLayoutForArchName(std::string_view archName)
{
    if(archName.find("gfx950") != std::string_view::npos)
        return MXScaleLayout::GFX950;
    if(archName.find("gfx1250") != std::string_view::npos)
        return MXScaleLayout::GFX1250;
    return MXScaleLayout::None;
}

MXScaleLayout mxScaleLayoutForFormat(hipblaslt_scaling_format scalingFormat,
                                     std::string_view         archName)
{
    if(scalingFormat == hipblaslt_scaling_format::Block_32_UE8M0_32_8_EXT)
        return MXScaleLayout::GFX950;
    if(mxScaleLayoutForArchName(archName) == MXScaleLayout::GFX1250)
        return MXScaleLayout::GFX1250;
    return MXScaleLayout::None;
}
