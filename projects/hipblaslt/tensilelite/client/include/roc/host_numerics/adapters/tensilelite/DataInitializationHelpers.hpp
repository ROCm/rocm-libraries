// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private TensileLite adapter.

// TensileLite adapter over host-numerics-owned MX generation.

#if HIPBLASLT_ENABLE_MXDATAGENERATOR

#include "DataInitialization.hpp"
#include <roc/host_numerics/adapters/tensilelite/HostNumericsBridge.hpp>
#include <roc/host_numerics/amd_gpu_layout/mx.hpp>
#include <roc/host_numerics/mx.hpp>

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
        namespace detail
        {
            inline roc::host_numerics::amd_gpu_layout::MxScaleStorageLayout
                mxScaleStorageLayoutForArchName(std::string_view archName)
            {
                return roc::host_numerics::amd_gpu_layout::mxScaleStorageLayoutForArchitectureName(
                    archName);
            }

            inline roc::host_numerics::MxScaleGenerationMode mxScaleGenerationMode(InitMode mode)
            {
                using roc::host_numerics::MxScaleGenerationMode;
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
                case InitMode::Free:
                case InitMode::Count:
                    break;
                }
                throw std::invalid_argument("Unsupported TensileLite MX scale initialization.");
            }

            inline roc::host_numerics::MxDataGeneration
                mxDataGeneration(InitMode                         mode,
                                 roc::host_numerics::ScalarType   dataType,
                                 roc::host_numerics::Shape const& shape,
                                 uint32_t                         seed)
            {
                using namespace roc::host_numerics;
                auto recipe
                    = [&](GenerationRecipe::Component component,
                          uint64_t randomDomain = mx_generation_random_domain_version_1::data) {
                          return GenerationRecipe::realOnly(
                              std::move(component),
                              {
                                  .seed         = seed,
                                  .indexOrder   = IndexOrder::FirstDimensionFastest,
                                  .randomDomain = randomDomain,
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
                case InitMode::Free:
                case InitMode::Count:
                    break;
                }
                throw std::invalid_argument("Unsupported TensileLite MX data initialization.");
            }

            inline roc::host_numerics::MxGenerationProblem
                makeMxGenerationProblem(rocisa::DataType          dataType,
                                        rocisa::DataType          scaleType,
                                        roc::host_numerics::Shape shape,
                                        size_t                    leadingDimension,
                                        size_t                    blockAxis,
                                        size_t                    blockSize,
                                        InitMode                  dataInitialization,
                                        InitMode                  scaleInitialization,
                                        uint32_t                  seed)
            {
                using namespace roc::host_numerics;
                const ScalarType hostDataType = toHostNumericsScalarType(dataType);
                if(leadingDimension > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()))
                    throw std::overflow_error(
                        "TensileLite MX leading dimension exceeds ptrdiff_t.");
                MxDataGeneration dataGeneration
                    = mxDataGeneration(dataInitialization, hostDataType, shape, seed);

                MxGenerationProblem problem(std::move(shape), std::move(dataGeneration));
                problem.dataType         = hostDataType;
                problem.scaleType        = toHostNumericsMxScaleType(scaleType);
                problem.leadingDimension = static_cast<ptrdiff_t>(leadingDimension);
                problem.blockAxis        = blockAxis;
                problem.blockSize        = blockSize;
                problem.scale            = mxScaleGenerationMode(scaleInitialization);
                return problem;
            }
        } // namespace detail
    } // namespace Client
} // namespace TensileLite
#endif // HIPBLASLT_ENABLE_MXDATAGENERATOR
