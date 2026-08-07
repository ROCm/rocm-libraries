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
#include <mxDataGenerator/PreSwizzle.hpp>
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
            return DGen::preSwizzleScalesGFX950(scaleBytes, {slowDimension, fastDimension});
        case MXScaleLayout::GFX1250:
            if(blockSize > 0)
                return DGen::preSwizzleScalesGFX1250(
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
                                   void*                  data,
                                   void*                  scale,
                                   uint64_t               row,
                                   uint64_t               col,
                                   uint64_t               stride,
                                   bool                   isTranspose,
                                   int const              scaleBlockRowSize,
                                   int const              scaleBlockColSize,
                                   bool                   isMatrixA,
                                   MXScaleLayout          scaleLayout,
                                   std::string_view const initMethod,
                                   float                  min_val,
                                   float                  max_val,
                                   std::string_view const scaleInitMethod)
{
    const ScalarType hostDataType = roc::host_validation::hipblaslt_adapter::scalarType(dataType);
    MxGenerationProblem problem;
    problem.dataType  = hostDataType;
    problem.scaleType = roc::host_validation::hipblaslt_adapter::scalarType(scaleType);
    problem.shape = roc::host_validation::Shape{static_cast<size_t>(row), static_cast<size_t>(col)};
    problem.leadingDimension = static_cast<ptrdiff_t>(stride);
    problem.blockSize        = static_cast<size_t>(scaleBlockRowSize * scaleBlockColSize);
    problem.blockAxis        = ((isMatrixA && isTranspose) || (!isMatrixA && !isTranspose)) ? 0 : 1;
    problem.data             = generationRecipe(initMethod, hostDataType, min_val, max_val);
    if(!scaleInitMethod.empty())
        problem.scale = generationRecipe(scaleInitMethod, hostDataType, -1.0f, 1.0f);

    roc::host_validation::MxGenerationResult result = roc::host_validation::generateMx(problem);
    std::memcpy(data, result.data.storage().data(), result.data.storage().size());

    std::vector<uint8_t> scaleBytes(result.scales.storage().size());
    std::memcpy(scaleBytes.data(), result.scales.storage().data(), scaleBytes.size());
    const size_t elementsPerBlock = problem.blockSize;
    const size_t scaleRows
        = elementsPerBlock == 0
              ? 0
              : (static_cast<size_t>(row) + elementsPerBlock - 1) / elementsPerBlock;
    scaleBytes = swizzleScaleBytes(
        std::move(scaleBytes), scaleLayout, static_cast<size_t>(col), scaleRows, elementsPerBlock);
    std::memcpy(scale, scaleBytes.data(), scaleBytes.size());

    std::vector<float> reference(row * col);
    std::memcpy(
        reference.data(), result.reference.storage().data(), reference.size() * sizeof(float));
    return reference;
}

void restrideMXScaleBufferKFast(uint8_t* buffer,
                                size_t   compactFreeDim,
                                size_t   compactKBlocks,
                                size_t   paddedKBlocks,
                                size_t   elemBytes)
{
    if(compactKBlocks == paddedKBlocks || compactFreeDim == 0)
        return;
    size_t const compactRow = compactKBlocks * elemBytes;
    size_t const paddedRow  = paddedKBlocks * elemBytes;
    size_t const padTail    = paddedRow - compactRow;
    for(size_t f = compactFreeDim; f-- > 1;)
    {
        std::memmove(buffer + f * paddedRow, buffer + f * compactRow, compactRow);
        std::memset(buffer + f * paddedRow + compactRow, 0x00, padTail);
    }
    std::memset(buffer + compactRow, 0x00, padTail);
}

void applyMXScaleLayoutInPlace(uint8_t*      scale,
                               size_t        scaleElemCount,
                               MXScaleLayout scaleLayout,
                               size_t        slowDim,
                               size_t        fastDim,
                               size_t        mxBlock)
{
    if(scaleLayout == MXScaleLayout::None || scaleElemCount == 0)
        return;
    std::vector<uint8_t> scaleBytes(scale, scale + scaleElemCount);
    scaleBytes = swizzleScaleBytes(std::move(scaleBytes), scaleLayout, slowDim, fastDim, mxBlock);
    std::memcpy(scale, scaleBytes.data(), scaleBytes.size());
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
