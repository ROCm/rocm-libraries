// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <roc/hostnumerics/generation.hpp>
#include <roc/hostnumerics/mx.hpp>
#include <stdexcept>
#include <vector>

namespace hipblaslt::hostnumerics::maf
{
    enum class Mode
    {
        Maf2,
        Maf2UnitScale,
        Maf1
    };

    inline uint64_t seed(size_t operand, size_t batch, bool scales = false)
    {
        constexpr uint64_t base        = 1'713'573'849;
        constexpr uint64_t scaleDomain = 0xa24baed4963ee407ULL;
        return (base + 32 * (operand + 3 * static_cast<uint64_t>(batch)))
               ^ (scales ? scaleDomain : 0);
    }

    inline roc::hostnumerics::ScalarConversionOptions conversion(Mode                          mode,
                                                                 roc::hostnumerics::ScalarType type)
    {
        using namespace roc::hostnumerics;
        return {IntegerRounding::NearestEven,
                IntegerOverflow::Saturate,
                BFloat16Rounding::NearestEven,
                mode != Mode::Maf1 && type == ScalarType::BFloat16
                    ? FloatingSourcePrecision::Float32
                    : FloatingSourcePrecision::Direct,
                FloatingOverflow::SaturateFinite,
                E8M0Underflow::FlushBelowFloat32Normal};
    }

    inline void validateType(roc::hostnumerics::ScalarType type)
    {
        using namespace roc::hostnumerics;
        if(scalarTypeInfo(type).category != ScalarCategory::FloatingPoint
           || !scalarTypeInfo(type).isSigned)
            throw std::runtime_error("MAF initialization requires a signed floating data type.");
    }

    inline roc::hostnumerics::GenerationRecipe
        dataRecipe(Mode mode, size_t operand, size_t batch, size_t rank, size_t indexOffset = 0)
    {
        using namespace roc::hostnumerics;
        return GenerationRecipe::realOnly(
            mode == Mode::Maf1
                ? (operand == 0 ? GenerationRecipe::sine() : GenerationRecipe::cosine())
                : GenerationRecipe::normal({}),
            {.seed        = seed(operand, batch),
             .indexOrder  = Permutation::reverse(rank),
             .indexOffset = indexOffset});
    }

    // Both clients use dimensions zero and one as the stored matrix dimensions.
    inline void fill(roc::hostnumerics::Tensor destination,
                     Mode                      mode,
                     size_t                    operand,
                     size_t                    firstBatch = 0)
    {
        using namespace roc::hostnumerics;
        validateType(destination.type());
        if(destination.elementCount() == 0)
            return;
        const auto& shape = destination.shape();
        if(shape.rank() == 0)
        {
            generate(destination,
                     dataRecipe(mode, operand, firstBatch, 0),
                     conversion(mode, destination.type()));
            return;
        }
        const size_t        rows           = shape[0];
        const size_t        columns        = shape.rank() > 1 ? shape[1] : 1;
        const size_t        matrixElements = Shape{rows, columns}.elementCount();
        std::vector<size_t> batchExtents;
        for(size_t axis = 2; axis < shape.rank(); ++axis)
            batchExtents.push_back(destination.layout().stride(axis) == 0 ? 1 : shape[axis]);
        Shape       batches(batchExtents);
        const Shape originalBatches(
            std::vector<size_t>(shape.dimensions().begin() + std::min(size_t{2}, shape.rank()),
                                shape.dimensions().end()));
        const auto          batchOrder = Permutation::reverse(batches.rank());
        std::vector<size_t> lastBatch(batches.rank());
        for(size_t axis = 0; axis < batches.rank(); ++axis)
            lastBatch[axis] = batches[axis] - 1;
        const size_t lastOrdinal = originalBatches.linearIndex(lastBatch, batchOrder);
        if(firstBatch > std::numeric_limits<size_t>::max() - lastOrdinal)
            throw std::runtime_error("MAF batch index overflow.");
        if(mode == Mode::Maf1
           && firstBatch + lastOrdinal
                  > (std::numeric_limits<size_t>::max() - (matrixElements - 1)) / matrixElements)
            throw std::runtime_error("MAF logical index overflow.");
        batches.forEachCoordinate(
            IndexRange{0, batches.elementCount(), 1},
            batchOrder,
            [&](size_t ordinal, std::span<const size_t> coordinates) {
                ordinal                   = originalBatches.linearIndex(coordinates, batchOrder);
                const size_t        batch = firstBatch + ordinal;
                std::vector<size_t> origin(shape.rank(), 0);
                std::copy(coordinates.begin(),
                          coordinates.end(),
                          origin.begin() + std::min(size_t{2}, shape.rank()));
                const auto offsetResult = destination.layout().elementOffset(origin);
                if(!offsetResult)
                    throw std::runtime_error(offsetResult.error().message());
                const ptrdiff_t offset       = *offsetResult;
                const bool      broadcastRow = destination.layout().stride(0) == 0;
                const bool      broadcastColumn
                    = shape.rank() > 1 && destination.layout().stride(1) == 0;
                Shape      matrixShape{broadcastRow ? 1 : rows, broadcastColumn ? 1 : columns};
                const auto view = destination.shareStorageWithLayout(
                    Layout(matrixShape,
                           {destination.layout().stride(0),
                            shape.rank() > 1 ? destination.layout().stride(1)
                                             : static_cast<ptrdiff_t>(rows)},
                           offset));
                const auto options = conversion(mode, destination.type());
                if(mode == Mode::Maf1 && (broadcastRow || broadcastColumn))
                    generate(
                        view,
                        [&](std::span<const size_t> index) {
                            const double phase = static_cast<double>(index[0] + index[1] * rows
                                                                     + batch * matrixElements);
                            return operand == 0 ? std::sin(phase) : std::cos(phase);
                        },
                        options);
                else
                    generate(view,
                             dataRecipe(mode,
                                        operand,
                                        batch,
                                        2,
                                        mode == Mode::Maf1 ? batch * matrixElements : 0),
                             options);
            });
    }

    inline roc::hostnumerics::MxTensor generateMx(roc::hostnumerics::Shape               shape,
                                                  roc::hostnumerics::MxGenerationOptions options,
                                                  Mode                                   mode,
                                                  size_t                                 operand,
                                                  size_t                                 batch)
    {
        using namespace roc::hostnumerics;
        if(options.dataType == ScalarType::Float4E2M1)
        {
            const size_t expectedBlock = options.scaleType == ScalarType::E8M0 ? 32 : 16;
            if(options.blockSize != expectedBlock
               || (options.scaleType != ScalarType::E8M0 && options.scaleType != ScalarType::E4M3
                   && options.scaleType != ScalarType::E5M3))
                throw std::runtime_error("MAF FP4 requires E8M0/B32 or E4M3/E5M3/B16 scales.");
        }
        else if(options.blockSize != 32 || options.scaleType != ScalarType::E8M0)
            throw std::runtime_error("MAF FP8/FP6 requires E8M0/B32 scales.");
        options.scale = MxScaleGenerationMode::One;
        const bool numericalData
            = mode == Mode::Maf1
              || (mode == Mode::Maf2UnitScale && options.dataType == ScalarType::Float4E2M1);
        auto generated = roc::hostnumerics::generateMx(
            shape,
            numericalData ? MxDataGeneration::quantize(
                GenerationRecipe::realOnlyLastDimensionFastest(GenerationRecipe::zero(), 0))
                          : MxDataGeneration::preserveGeneratedEncoding(GenerationRecipe::realOnly(
                              GenerationRecipe::uniformFiniteEncodedValue(),
                              {.seed       = seed(operand, batch),
                               .indexOrder = roc::hostnumerics::Permutation{1, 0}})),
            options);
        if(numericalData)
            fill(generated.data, mode, operand, batch);
        auto scaleComponent = GenerationRecipe::constant({.value = 1.0});
        if(mode == Mode::Maf1)
            scaleComponent = GenerationRecipe::constant({.value = 0.5});
        else if(!numericalData)
            scaleComponent
                = options.dataType == ScalarType::Float4E2M1
                      ? GenerationRecipe::normal({.mean = 0.34375, .standardDeviation = 0.08})
                            .withAbsoluteTransform()
                      : GenerationRecipe::binomialHalf({.trials = 21})
                            .withExp2Transform()
                            .withAffineValueMapping({.scale = 0x1p-11});
        generate(generated.scales,
                 GenerationRecipe::realOnlyLastDimensionFastest(scaleComponent,
                                                                seed(operand, batch, true)),
                 conversion(mode, options.scaleType));
        generated.refreshReference();
        return generated;
    }
}
