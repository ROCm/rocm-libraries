// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Product-private hipBLASLt translation and architecture-selected upload
// transforms around component-owned MX tensor generation.

#include <roc/host_validation/adapters/hipblaslt/Types.hpp>
#include <roc/host_validation/adapters/hipblaslt/mxDataGen.hpp>
#include <roc/host_validation/mx.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <roc/host_validation/amd_gpu_layout/mx.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{
    using roc::host_validation::MxGenerationMode;
    using roc::host_validation::MxGenerationProblem;
    using roc::host_validation::MxGenerationRecipe;
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
        return roc::host_validation::hipblaslt_adapter::scalarType(scaleType);
    }

    bool recipesEqual(const MxGenerationRecipe& first, const MxGenerationRecipe& second)
    {
        return first.mode == second.mode && first.parameter0 == second.parameter0
               && first.parameter1 == second.parameter1;
    }

    bool isRandomLike(MxGenerationMode mode)
    {
        return mode == MxGenerationMode::Bounded || mode == MxGenerationMode::BoundedAlternatingSign
               || mode == MxGenerationMode::Unbounded || mode == MxGenerationMode::Normal;
    }

    bool isConstantScaleRecipe(MxGenerationMode mode)
    {
        switch(mode)
        {
        case MxGenerationMode::Zeros:
        case MxGenerationMode::Ones:
        case MxGenerationMode::NegativeOnes:
        case MxGenerationMode::Twos:
        case MxGenerationMode::Maximum:
        case MxGenerationMode::DenormalMinimum:
        case MxGenerationMode::DenormalMaximum:
        case MxGenerationMode::NaN:
        case MxGenerationMode::Infinity:
            return true;
        default:
            return false;
        }
    }

    MxGenerationRecipe generationRecipe(std::string_view initMethod,
                                        ScalarType       dataType,
                                        float            minimum,
                                        float            maximum)
    {
        MxGenerationRecipe recipe;
        recipe.parameter0 = minimum;
        recipe.parameter1 = maximum;
        if(initMethod == "Sequential")
            recipe.mode = MxGenerationMode::Sequential;
        else if(initMethod == "RowIndex")
            recipe.mode = MxGenerationMode::RowIndex;
        else if(initMethod == "ColIndex")
            recipe.mode = MxGenerationMode::ColumnIndex;
        else if(initMethod == "Checkerboard")
            recipe.mode = MxGenerationMode::Checkerboard;
        else if(initMethod == "ScaledDiagonal")
            recipe.mode = MxGenerationMode::ScaledDiagonal;
        else if(initMethod == "Identity")
            recipe.mode = MxGenerationMode::Identity;
        else if(initMethod == "Ones")
            recipe.mode = MxGenerationMode::Ones;
        else if(initMethod == "Zeros" || initMethod == "zero")
            recipe.mode = MxGenerationMode::Zeros;
        else if(initMethod == "Twos")
            recipe.mode = MxGenerationMode::Twos;
        else if(initMethod == "NegOnes")
            recipe.mode = MxGenerationMode::NegativeOnes;
        else if(initMethod == "MaxVals")
            recipe.mode = MxGenerationMode::Maximum;
        else if(initMethod == "DenormMins")
            recipe.mode = MxGenerationMode::DenormalMinimum;
        else if(initMethod == "DenormMaxs")
            recipe.mode = MxGenerationMode::DenormalMaximum;
        else if(initMethod == "NaNs")
            recipe.mode = MxGenerationMode::NaN;
        else if(initMethod == "Infs")
            recipe.mode = MxGenerationMode::Infinity;
        else if(initMethod == "Bounded")
            recipe.mode = MxGenerationMode::Bounded;
        else if(initMethod == "uniform_01")
        {
            recipe.mode       = MxGenerationMode::Bounded;
            recipe.parameter0 = 0;
            recipe.parameter1 = 1;
        }
        else if(initMethod == "hpl")
        {
            recipe.mode       = MxGenerationMode::Bounded;
            recipe.parameter0 = -0.5;
            recipe.parameter1 = 0.5;
        }
        else if(initMethod == "uniform_low_precision")
        {
            recipe.mode       = MxGenerationMode::Bounded;
            recipe.parameter0 = -6;
            recipe.parameter1 = 6;
        }
        else if(initMethod == "TrigonometricFromFloat" || initMethod == "trig_float")
            recipe.mode = MxGenerationMode::Trigonometric;
        else if(initMethod == "norm_dist")
        {
            recipe.mode       = MxGenerationMode::Normal;
            recipe.parameter0 = 0;
            recipe.parameter1 = normDistStdDevFor(dataType);
        }
        else if(initMethod == "rand_int")
        {
            const auto [lower, upper] = randIntRangeFor(dataType);
            recipe.mode               = MxGenerationMode::UniformInteger;
            recipe.parameter0         = lower;
            recipe.parameter1         = upper;
        }
        else
            throw std::invalid_argument("Unsupported hipBLASLt MX initialization mode.");
        return recipe;
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

    const ScalarType hostDataType = roc::host_validation::hipblaslt_adapter::scalarType(dataType);
    MxGenerationProblem problem;
    problem.dataType  = hostDataType;
    problem.scaleType = scaleScalarType(scaleType);
    problem.shape = roc::host_validation::Shape{static_cast<size_t>(row), static_cast<size_t>(col)};
    problem.leadingDimension = static_cast<ptrdiff_t>(stride);
    problem.blockSize        = blockRows * blockColumns;
    problem.blockAxis        = blockColumns > 1 ? 1 : 0;
    problem.data             = generationRecipe(initMethod, hostDataType, min_val, max_val);
    if(!scaleInitMethod.empty())
    {
        MxGenerationRecipe const scaleRecipe
            = generationRecipe(scaleInitMethod, hostDataType, -1.0f, 1.0f);
        if(recipesEqual(problem.data, scaleRecipe)
           || (isRandomLike(problem.data.mode) && isConstantScaleRecipe(scaleRecipe.mode)))
            problem.scale = scaleRecipe;
    }
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
