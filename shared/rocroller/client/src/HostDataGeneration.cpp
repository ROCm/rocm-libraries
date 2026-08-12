// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "client/HostDataGeneration.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

#include <roc/host_validation/generation.hpp>
#include <roc/host_validation/mx.hpp>
#include <rocRoller/DataTypes/DataTypes_BF8.hpp>
#include <rocRoller/DataTypes/DataTypes_FP8.hpp>
#include <rocRoller/Utilities/Settings.hpp>

namespace rocRoller::Client::GEMMClient
{
    namespace
    {
        using roc::host_validation::GenerationOptions;
        using roc::host_validation::GenerationPattern;
        using roc::host_validation::GenerationTransform;
        using roc::host_validation::Layout;
        using roc::host_validation::LogicalIndexOrder;
        using roc::host_validation::MxGenerationMode;
        using roc::host_validation::MxGenerationProblem;
        using roc::host_validation::MxGenerationRecipe;
        using roc::host_validation::ScalarType;
        using roc::host_validation::Shape;
        using roc::host_validation::Tensor;

        struct GeneratedOperand
        {
            Tensor                data;
            std::optional<Tensor> scales;
        };

        ptrdiff_t checkedPtrdiff(size_t value, char const* description)
        {
            if(value > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()))
                throw std::overflow_error(description);
            return static_cast<ptrdiff_t>(value);
        }

        Layout tensorLayout(TensorDescriptor const& descriptor)
        {
            std::vector<ptrdiff_t> strides;
            strides.reserve(descriptor.strides().size());
            for(auto const stride : descriptor.strides())
                strides.push_back(
                    checkedPtrdiff(stride, "rocRoller tensor stride exceeds ptrdiff_t."));

            return Layout(
                Shape(descriptor.sizes()),
                std::move(strides),
                checkedPtrdiff(descriptor.offset(), "rocRoller tensor offset exceeds ptrdiff_t."));
        }

        ScalarType unscaledF8ScalarType(DataType type)
        {
            auto const mode = Settings::Get(Settings::F8ModeOption);
            switch(type)
            {
            case DataType::FP8:
                return mode == F8Mode::NaNoo ? ScalarType::Float8E4M3Fnuz : ScalarType::Float8E4M3;
            case DataType::BF8:
                return mode == F8Mode::NaNoo ? ScalarType::Float8E5M2Fnuz : ScalarType::Float8E5M2;
            default:
                throw std::invalid_argument("Requested rocRoller type is not an F8 type.");
            }
        }

        ScalarType dataScalarType(DataType type, bool scaled)
        {
            switch(type)
            {
            case DataType::Float:
                return ScalarType::Float32;
            case DataType::Half:
                return ScalarType::Float16;
            case DataType::BFloat16:
                return ScalarType::BFloat16;
            case DataType::FP8:
                return scaled ? ScalarType::Float8E4M3 : unscaledF8ScalarType(type);
            case DataType::BF8:
                return scaled ? ScalarType::Float8E5M2 : unscaledF8ScalarType(type);
            case DataType::FP6:
                return ScalarType::Float6E2M3;
            case DataType::BF6:
                return ScalarType::Float6E3M2;
            case DataType::FP4:
                return ScalarType::Float4E2M1;
            default:
                throw std::invalid_argument("Unsupported rocRoller host-generation data type.");
            }
        }

        ScalarType scaleScalarType(DataType type)
        {
            switch(type)
            {
            case DataType::E8M0:
                return ScalarType::E8M0;
            case DataType::E5M3:
                return ScalarType::E5M3;
            case DataType::E4M3:
                return ScalarType::E4M3;
            default:
                throw std::invalid_argument("Unsupported rocRoller host-generation scale type.");
            }
        }

        std::pair<int, int> unboundedExponentRange(ScalarType type)
        {
            switch(type)
            {
            case ScalarType::Float16:
                return {-15, 15};
            case ScalarType::BFloat16:
            case ScalarType::Float32:
                return {-127, 127};
            case ScalarType::Float8E4M3:
                return {-7, 7};
            case ScalarType::Float8E5M2:
                return {-15, 15};
            case ScalarType::Float8E4M3Fnuz:
                return {-7, 7};
            case ScalarType::Float8E5M2Fnuz:
                return {-15, 15};
            case ScalarType::Float6E2M3:
            case ScalarType::Float4E2M1:
                return {-1, 2};
            case ScalarType::Float6E3M2:
                return {-3, 4};
            default:
                throw std::invalid_argument(
                    "Unsupported scalar type for unbounded rocRoller generation.");
            }
        }

        LogicalIndexOrder indexOrder(TensorDescriptor const& descriptor)
        {
            if(descriptor.dimensions() != 2)
                throw std::invalid_argument("rocRoller GEMM generation requires rank-two tensors.");
            if(descriptor.stride(0) == descriptor.stride(1))
            {
                if(descriptor.size(0) == 1)
                    return LogicalIndexOrder::LastDimensionFastest;
                if(descriptor.size(1) == 1)
                    return LogicalIndexOrder::FirstDimensionFastest;
                throw std::invalid_argument(
                    "rocRoller GEMM generation requires non-overlapping matrix strides.");
            }
            return descriptor.stride(0) < descriptor.stride(1)
                       ? LogicalIndexOrder::FirstDimensionFastest
                       : LogicalIndexOrder::LastDimensionFastest;
        }

        GenerationOptions generationOptions(TensorDescriptor const&   descriptor,
                                            DataInitialization const& initialization,
                                            ScalarType                type,
                                            float                     minimum,
                                            float                     maximum,
                                            uint32_t                  seed)
        {
            GenerationOptions options;
            options.seed       = seed;
            options.indexOrder = indexOrder(descriptor);

            switch(initialization.mode)
            {
            case DataInitializationMode::Bounded:
                options.real.pattern    = GenerationPattern::UniformReal;
                options.real.parameter0 = minimum;
                options.real.parameter1 = maximum;
                break;
            case DataInitializationMode::BoundedAlternatingSign:
                options.real.pattern    = GenerationPattern::UniformReal;
                options.real.parameter0 = 0;
                options.real.parameter1 = std::max(std::abs(minimum), std::abs(maximum));
                for(size_t dimension = 0; dimension < descriptor.dimensions(); ++dimension)
                {
                    if((descriptor.stride(dimension) & 1U) != 0)
                        options.real.alternatingDimensions.push_back(dimension);
                }
                options.real.negativeParity = 1U ^ (descriptor.offset() & 1U);
                break;
            case DataInitializationMode::Unbounded:
            {
                auto const [lowerExponent, upperExponent] = unboundedExponentRange(type);
                options.real.pattern    = GenerationPattern::RandomEncodedExponent;
                options.real.parameter0 = lowerExponent;
                options.real.parameter1 = upperExponent;
                options.real.sourceType = type;
                break;
            }
            case DataInitializationMode::Identity:
                options.real.pattern = GenerationPattern::Identity;
                break;
            case DataInitializationMode::Ones:
                options.real.pattern    = GenerationPattern::Constant;
                options.real.parameter0 = 1;
                break;
            case DataInitializationMode::Zeros:
                options.real.pattern = GenerationPattern::Zero;
                break;
            case DataInitializationMode::TrigonometricFromFloat:
                options.real.pattern    = GenerationPattern::UniformReal;
                options.real.parameter0 = 0;
                options.real.parameter1 = 6.28318530717958647692528676655900576;
                options.real.transform  = GenerationTransform::Cosine;
                break;
            case DataInitializationMode::NormalFromFloat:
                options.real.pattern    = GenerationPattern::Normal;
                options.real.parameter0 = initialization.normalMean;
                options.real.parameter1 = initialization.normalStandardDeviation;
                break;
            }
            return options;
        }

        MxGenerationRecipe mxGenerationRecipe(DataInitialization const& initialization,
                                              float                     minimum,
                                              float                     maximum)
        {
            MxGenerationRecipe recipe;
            switch(initialization.mode)
            {
            case DataInitializationMode::Bounded:
                recipe.mode       = MxGenerationMode::Bounded;
                recipe.parameter0 = minimum;
                recipe.parameter1 = maximum;
                break;
            case DataInitializationMode::BoundedAlternatingSign:
                recipe.mode       = MxGenerationMode::BoundedAlternatingSign;
                recipe.parameter0 = minimum;
                recipe.parameter1 = maximum;
                break;
            case DataInitializationMode::Unbounded:
                recipe.mode = MxGenerationMode::Unbounded;
                break;
            case DataInitializationMode::Identity:
                recipe.mode = MxGenerationMode::Identity;
                break;
            case DataInitializationMode::Ones:
                recipe.mode = MxGenerationMode::Ones;
                break;
            case DataInitializationMode::Zeros:
                recipe.mode = MxGenerationMode::Zeros;
                break;
            case DataInitializationMode::TrigonometricFromFloat:
                recipe.mode = MxGenerationMode::Trigonometric;
                break;
            case DataInitializationMode::NormalFromFloat:
                recipe.mode       = MxGenerationMode::Normal;
                recipe.parameter0 = initialization.normalMean;
                recipe.parameter1 = initialization.normalStandardDeviation;
                break;
            }
            return recipe;
        }

        Tensor generateUnscaledF8(TensorDescriptor const&   descriptor,
                                  DataInitialization const& initialization,
                                  ScalarType                type,
                                  float                     minimum,
                                  float                     maximum,
                                  uint32_t                  seed)
        {
            auto const layout = tensorLayout(descriptor);
            Tensor     source(ScalarType::Float32, layout);
            roc::host_validation::generate(
                source.mutableView(),
                generationOptions(descriptor, initialization, type, minimum, maximum, seed));

            auto const sourceStorage = source.storage();
            if(sourceStorage.size() % sizeof(float) != 0)
                throw std::logic_error("Float32 generation storage is not float-aligned.");

            std::vector<std::byte> storage(sourceStorage.size() / sizeof(float));
            for(size_t index = 0; index < storage.size(); ++index)
            {
                float value;
                std::memcpy(&value, sourceStorage.data() + index * sizeof(float), sizeof(float));

                uint8_t encoded;
                if(type == ScalarType::Float8E4M3 || type == ScalarType::Float8E4M3Fnuz)
                    encoded = FP8(value).data;
                else
                    encoded = BF8(value).data;
                storage[index] = static_cast<std::byte>(encoded);
            }

            return Tensor::fromStorage(type, layout, std::move(storage));
        }

        GeneratedOperand generateUnscaled(TensorDescriptor const&   descriptor,
                                          DataInitialization const& initialization,
                                          float                     minimum,
                                          float                     maximum,
                                          uint32_t                  seed)
        {
            auto const type = dataScalarType(descriptor.dataType(), false);
            if(descriptor.dataType() == DataType::FP8 || descriptor.dataType() == DataType::BF8)
            {
                return {
                    generateUnscaledF8(descriptor, initialization, type, minimum, maximum, seed),
                    std::nullopt};
            }

            Tensor data(type, tensorLayout(descriptor));
            roc::host_validation::generate(
                data.mutableView(),
                generationOptions(descriptor, initialization, type, minimum, maximum, seed));
            return {std::move(data), std::nullopt};
        }

        GeneratedOperand generateScaled(TensorDescriptor const&   descriptor,
                                        DataInitialization const& initialization,
                                        DataType                  scaleType,
                                        size_t                    blockedDimension,
                                        size_t                    scaleBlockSize,
                                        float                     minimum,
                                        float                     maximum,
                                        uint32_t                  seed)
        {
            if(descriptor.dimensions() != 2)
                throw std::invalid_argument("rocRoller MX generation requires rank-two tensors.");
            if(descriptor.offset() != 0)
                throw std::invalid_argument("rocRoller MX generation does not support offsets.");
            if(blockedDimension >= descriptor.dimensions())
                throw std::invalid_argument(
                    "rocRoller MX blocked dimension exceeds the tensor rank.");

            std::array<size_t, 2> dimensions{0, 1};
            std::ranges::sort(dimensions, [&](size_t first, size_t second) {
                return descriptor.stride(first) < descriptor.stride(second);
            });
            auto const contiguousDimension = dimensions[0];
            auto const freeDimension       = dimensions[1];
            if(descriptor.stride(contiguousDimension) != 1)
                throw std::invalid_argument(
                    "rocRoller MX generation requires a stride-one matrix dimension.");

            MxGenerationProblem problem;
            problem.dataType  = dataScalarType(descriptor.dataType(), true);
            problem.scaleType = scaleScalarType(scaleType);
            problem.shape
                = Shape{descriptor.size(contiguousDimension), descriptor.size(freeDimension)};
            problem.leadingDimension
                = checkedPtrdiff(descriptor.stride(freeDimension),
                                 "rocRoller MX leading dimension exceeds ptrdiff_t.");
            problem.blockAxis = blockedDimension == contiguousDimension ? 0 : 1;
            problem.blockSize = scaleBlockSize;
            problem.data      = mxGenerationRecipe(initialization, minimum, maximum);
            problem.seed      = seed;

            auto   result      = roc::host_validation::generateMx(problem);
            auto   dataStorage = std::vector<std::byte>(result.data.storage().begin(),
                                                        result.data.storage().end());
            Tensor data        = Tensor::fromStorage(
                result.data.type(), tensorLayout(descriptor), std::move(dataStorage));

            auto const logicalFreeDimension = size_t{1} - blockedDimension;
            auto const logicalFreeExtent    = descriptor.size(logicalFreeDimension);
            auto const blockCount
                = descriptor.size(blockedDimension) / scaleBlockSize
                  + static_cast<size_t>(descriptor.size(blockedDimension) % scaleBlockSize != 0);
            auto const scaleLayout
                = blockedDimension == contiguousDimension
                      ? Layout(Shape{logicalFreeExtent, blockCount},
                               {checkedPtrdiff(blockCount,
                                               "rocRoller MX scale stride exceeds ptrdiff_t."),
                                1})
                      : Layout(Shape{logicalFreeExtent, blockCount},
                               {1,
                                checkedPtrdiff(logicalFreeExtent,
                                               "rocRoller MX scale stride exceeds ptrdiff_t.")});
            auto   scaleStorage = std::vector<std::byte>(result.scales.storage().begin(),
                                                         result.scales.storage().end());
            Tensor scales
                = Tensor::fromStorage(result.scales.type(), scaleLayout, std::move(scaleStorage));
            return {std::move(data), std::move(scales)};
        }

        GeneratedOperand generateOperand(TensorDescriptor const&   descriptor,
                                         DataInitialization const& initialization,
                                         DataType                  scaleType,
                                         size_t                    blockedDimension,
                                         size_t                    scaleBlockSize,
                                         float                     minimum,
                                         float                     maximum,
                                         uint32_t                  seed)
        {
            if(scaleType == DataType::None)
                return generateUnscaled(descriptor, initialization, minimum, maximum, seed);
            return generateScaled(descriptor,
                                  initialization,
                                  scaleType,
                                  blockedDimension,
                                  scaleBlockSize,
                                  minimum,
                                  maximum,
                                  seed);
        }
    }

    GeneratedGEMMInputs generateGEMMInputs(TensorDescriptor const&   descriptorA,
                                           TensorDescriptor const&   descriptorB,
                                           TensorDescriptor const&   descriptorC,
                                           DataInitialization const& initializationA,
                                           DataInitialization const& initializationB,
                                           DataInitialization const& initializationC,
                                           DataType                  scaleTypeA,
                                           DataType                  scaleTypeB,
                                           size_t                    scaleBlockSize,
                                           float                     minimum,
                                           float                     maximum,
                                           uint32_t                  seed)
    {
        auto generatedA = generateOperand(descriptorA,
                                          initializationA,
                                          scaleTypeA,
                                          1,
                                          scaleBlockSize,
                                          minimum,
                                          maximum,
                                          seed + 1);
        auto generatedB = generateOperand(descriptorB,
                                          initializationB,
                                          scaleTypeB,
                                          0,
                                          scaleBlockSize,
                                          minimum,
                                          maximum,
                                          seed + 2);
        auto generatedC = generateOperand(
            descriptorC, initializationC, DataType::None, 0, 1, minimum, maximum, seed);

        return {std::move(generatedA.data),
                std::move(generatedB.data),
                std::move(generatedC.data),
                std::move(generatedA.scales),
                std::move(generatedB.scales)};
    }
}
