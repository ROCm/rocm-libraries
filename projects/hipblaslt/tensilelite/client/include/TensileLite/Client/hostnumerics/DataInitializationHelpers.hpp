// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// TensileLite client-private adapter.

// TensileLite adapter over hostnumerics-owned MX generation.

#include "DataInitialization.hpp"
#include <TensileLite/Client/hostnumerics/HostNumericsBridge.hpp>
#include <TensileLite/Client/hostnumerics/MafInitialization.hpp>
#include <roc/hostnumerics/amd_gpu_layout/mx.hpp>
#include <roc/hostnumerics/mx.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace TensileLite
{
    namespace Client
    {
        namespace hostnumerics::detail
        {
            inline bool isMaf(InitMode mode)
            {
                return mode == InitMode::Maf2_0 || mode == InitMode::Maf2_0UnitScale
                       || mode == InitMode::Maf1_0;
            }
            inline hipblaslt::hostnumerics::maf::Mode mafMode(InitMode mode)
            {
                using hipblaslt::hostnumerics::maf::Mode;
                return mode == InitMode::Maf1_0            ? Mode::Maf1
                       : mode == InitMode::Maf2_0UnitScale ? Mode::Maf2UnitScale
                                                           : Mode::Maf2;
            }
            inline roc::hostnumerics::amd_gpu_layout::MxScaleStorageLayout
                mxScaleStorageLayoutForArchName(std::string_view archName)
            {
                return roc::hostnumerics::amd_gpu_layout::mxScaleStorageLayoutForArchitectureName(
                    archName);
            }

            inline roc::hostnumerics::MxScaleGenerationMode mxScaleGenerationMode(InitMode mode)
            {
                using roc::hostnumerics::MxScaleGenerationMode;
                switch(mode)
                {
                case InitMode::Zero:
                    return MxScaleGenerationMode::Minimum;
                case InitMode::Two:
                    return MxScaleGenerationMode::Two;
                case InitMode::Max:
                    return MxScaleGenerationMode::Maximum;
                case InitMode::NaN:
                case InitMode::BadInput:
                    return MxScaleGenerationMode::NaN;
                case InitMode::One:
                case InitMode::NegOne:
                case InitMode::DenormMin:
                case InitMode::DenormMax:
                case InitMode::Inf:
                case InitMode::BadOutput:
                case InitMode::Identity:
                case InitMode::SerialIdx:
                case InitMode::SerialDim0:
                case InitMode::SerialDim1:
                case InitMode::Random:
                    return MxScaleGenerationMode::One;
                case InitMode::TrigSin:
                case InitMode::TrigCos:
                case InitMode::TrigAbsSin:
                case InitMode::TrigAbsCos:
                case InitMode::RandomNarrow:
                case InitMode::RandomNegPosLimited:
                case InitMode::TrigIndSin:
                case InitMode::TrigIndCos:
                case InitMode::TrigIndAbsSin:
                case InitMode::TrigIndAbsCos:
                case InitMode::UniformLowPrecision:
                    return MxScaleGenerationMode::Derived;
                case InitMode::Maf2_0:
                case InitMode::Maf2_0UnitScale:
                case InitMode::Maf1_0:
                case InitMode::Free:
                case InitMode::Count:
                    break;
                }
                throw std::runtime_error("Unsupported TensileLite MX scale initialization.");
            }

            inline roc::hostnumerics::MxDataGeneration
                mxDataGeneration(InitMode                        mode,
                                 roc::hostnumerics::ScalarType   dataType,
                                 roc::hostnumerics::Shape const& shape,
                                 uint64_t                        seed)
            {
                using roc::hostnumerics::GenerationRecipe;
                using roc::hostnumerics::MxDataGeneration;
                using roc::hostnumerics::ScalarType;
                auto recipe = [&](GenerationRecipe::Component component) {
                    return GenerationRecipe::realOnly(std::move(component),
                                                      {
                                                          .seed       = seed,
                                                          .indexOrder = std::vector<size_t>{1, 0},
                                                      });
                };

                switch(mode)
                {
                case InitMode::Zero:
                    return MxDataGeneration::quantize(recipe(GenerationRecipe::zero()));
                case InitMode::One:
                    return MxDataGeneration::quantize(
                        recipe(GenerationRecipe::constant({.value = 1.0})));
                case InitMode::Two:
                    return MxDataGeneration::quantize(
                        recipe(GenerationRecipe::constant({.value = 2.0})));
                case InitMode::NegOne:
                    return MxDataGeneration::quantize(
                        recipe(GenerationRecipe::constant({.value = -1.0})));
                case InitMode::Max:
                    return MxDataGeneration::quantize(recipe(GenerationRecipe::typeMaximum()));
                case InitMode::DenormMin:
                    return MxDataGeneration::quantize(
                        recipe(GenerationRecipe::typeDenormalMinimum()));
                case InitMode::DenormMax:
                    return MxDataGeneration::quantize(
                        recipe(GenerationRecipe::typeDenormalMaximum()));
                case InitMode::NaN:
                case InitMode::BadInput:
                    return MxDataGeneration::quantize(recipe(GenerationRecipe::constant(
                        {.value = std::numeric_limits<double>::quiet_NaN()})));
                case InitMode::Inf:
                case InitMode::BadOutput:
                    return MxDataGeneration::quantize(recipe(GenerationRecipe::typeInfinity()));
                case InitMode::Identity:
                    return MxDataGeneration::quantize(recipe(GenerationRecipe::identity()));
                case InitMode::SerialIdx:
                case InitMode::SerialDim0:
                case InitMode::SerialDim1:
                    return MxDataGeneration::quantize(recipe(GenerationRecipe::affineIndexRemainder(
                        {.dimensionCoefficients = {static_cast<int64_t>(shape[1] % 256U), 1},
                         .positiveDivisor       = 256})));
                case InitMode::TrigSin:
                case InitMode::TrigCos:
                case InitMode::TrigAbsSin:
                case InitMode::TrigAbsCos:
                case InitMode::TrigIndSin:
                case InitMode::TrigIndCos:
                case InitMode::TrigIndAbsSin:
                case InitMode::TrigIndAbsCos:
                    return MxDataGeneration::quantize(
                        recipe(GenerationRecipe::uniformReal(
                                   {.lower = 0.0, .upper = 6.28318530717958647692528676655900576})
                                   .withCosineTransform()));
                case InitMode::Random:
                {
                    std::pair<int, int> range{1, 10};
                    if(dataType == ScalarType::Float4E2M1)
                        range = {-4, 4};
                    else if(dataType == ScalarType::Float6E2M3)
                        range = {-7, 7};
                    else if(dataType == ScalarType::Float6E3M2)
                        range = {-28, 28};
                    return MxDataGeneration::quantize(recipe(GenerationRecipe::uniformInteger(
                        {.lower = range.first, .upper = range.second})));
                }
                case InitMode::RandomNarrow:
                case InitMode::RandomNegPosLimited:
                    return MxDataGeneration::preserveRange(
                        recipe(GenerationRecipe::uniformReal({.lower = -1.0, .upper = 1.0})),
                        {.lower = -1.0, .upper = 1.0});
                case InitMode::UniformLowPrecision:
                    return MxDataGeneration::preserveRange(
                        recipe(GenerationRecipe::uniformReal({.lower = -6.0, .upper = 6.0})),
                        {.lower = -6.0, .upper = 6.0});
                case InitMode::Maf2_0:
                case InitMode::Maf2_0UnitScale:
                case InitMode::Maf1_0:
                case InitMode::Free:
                case InitMode::Count:
                    break;
                }
                throw std::runtime_error("Unsupported TensileLite MX data initialization.");
            }

            void initializeMxDataIndependently(rocisa::DataType dataType,
                                               rocisa::DataType scaleType,
                                               InitMode         dataInitialization,
                                               InitMode         scaleInitialization,
                                               uint64_t         seed,
                                               const roc::hostnumerics::MxTensor& generated);

            inline roc::hostnumerics::MxTensor generateMxData(rocisa::DataType         dataType,
                                                              rocisa::DataType         scaleType,
                                                              roc::hostnumerics::Shape shape,
                                                              size_t   leadingDimension,
                                                              size_t   blockAxis,
                                                              size_t   blockSize,
                                                              InitMode dataInitialization,
                                                              InitMode scaleInitialization,
                                                              uint64_t seed,
                                                              bool     independentData,
                                                              size_t   operand    = 0,
                                                              size_t   batchIndex = 0)
            {
                using roc::hostnumerics::generateMx;
                using roc::hostnumerics::MxDataGeneration;
                using roc::hostnumerics::MxGenerationOptions;
                using roc::hostnumerics::ScalarType;
                const ScalarType hostDataType = toHostNumericsScalarType(dataType);
                if(leadingDimension > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()))
                    throw std::runtime_error("TensileLite MX leading dimension exceeds ptrdiff_t.");
                if(isMaf(dataInitialization))
                {
                    MxGenerationOptions options;
                    options.dataType = hostDataType;
                    options.scaleType
                        = scaleType == rocisa::DataType::E8 || scaleType == rocisa::DataType::None
                              ? ScalarType::E8M0
                              : toHostNumericsMxScaleType(scaleType);
                    options.leadingDimension = static_cast<ptrdiff_t>(leadingDimension);
                    options.blockAxis        = blockAxis;
                    options.blockSize        = blockSize;
                    return hipblaslt::hostnumerics::maf::generateMx(std::move(shape),
                                                                    options,
                                                                    mafMode(dataInitialization),
                                                                    operand,
                                                                    batchIndex);
                }
                MxDataGeneration dataGeneration
                    = mxDataGeneration(independentData ? InitMode::Zero : dataInitialization,
                                       hostDataType,
                                       shape,
                                       seed);

                MxGenerationOptions options;
                options.dataType         = hostDataType;
                options.scaleType        = toHostNumericsMxScaleType(scaleType);
                options.leadingDimension = static_cast<ptrdiff_t>(leadingDimension);
                options.blockAxis        = blockAxis;
                options.blockSize        = blockSize;
                options.scale  = independentData ? roc::hostnumerics::MxScaleGenerationMode::One
                                                 : mxScaleGenerationMode(scaleInitialization);
                auto generated = generateMx(std::move(shape), std::move(dataGeneration), options);
                if(independentData)
                    initializeMxDataIndependently(dataType,
                                                  scaleType,
                                                  dataInitialization,
                                                  scaleInitialization,
                                                  seed,
                                                  generated);
                return generated;
            }
        } // namespace hostnumerics::detail
    } // namespace Client
} // namespace TensileLite
