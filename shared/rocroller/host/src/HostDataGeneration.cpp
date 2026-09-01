// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/HostNumerics/HostDataGeneration.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string_view>
#include <utility>

#include <roc/host_numerics/generation.hpp>
#include <roc/host_numerics/mx.hpp>
#include <rocRoller/DataTypes/DataTypes_BF8.hpp>
#include <rocRoller/DataTypes/DataTypes_FP8.hpp>
#include <rocRoller/Utilities/Settings.hpp>

namespace rocRoller::HostNumerics
{
    namespace
    {
        using roc::host_numerics::GenerationRecipe;
        using roc::host_numerics::GenerationRecipeSettings;
        using roc::host_numerics::IndexOrder;
        using roc::host_numerics::Layout;
        using roc::host_numerics::MxDataGeneration;
        using roc::host_numerics::MxGenerationProblem;
        using roc::host_numerics::MxScaleGenerationMode;
        using roc::host_numerics::ScalarType;
        using roc::host_numerics::Shape;
        using roc::host_numerics::Tensor;

        ptrdiff_t checkedPtrdiff(size_t value, char const* description)
        {
            if(value > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()))
                throw std::overflow_error(description);
            return static_cast<ptrdiff_t>(value);
        }

        Layout makeTensorLayout(TensorDescriptor const& descriptor)
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

        ScalarType makeHostScalarType(DataType type, DataTypeInterpretation interpretation)
        {
            switch(type)
            {
            case DataType::UInt8:
                return ScalarType::UInt8;
            case DataType::Int8:
                return ScalarType::Int8;
            case DataType::UInt16:
                return ScalarType::UInt16;
            case DataType::Int16:
                return ScalarType::Int16;
            case DataType::UInt32:
                return ScalarType::UInt32;
            case DataType::Int32:
                return ScalarType::Int32;
            case DataType::UInt64:
                return ScalarType::UInt64;
            case DataType::Int64:
                return ScalarType::Int64;
            case DataType::Float:
                return ScalarType::Float32;
            case DataType::Double:
                return ScalarType::Float64;
            case DataType::ComplexFloat:
                return ScalarType::ComplexFloat32;
            case DataType::ComplexDouble:
                return ScalarType::ComplexFloat64;
            case DataType::Half:
                return ScalarType::Float16;
            case DataType::BFloat16:
                return ScalarType::BFloat16;
            case DataType::FP8:
                return interpretation == DataTypeInterpretation::BlockScaled
                           ? ScalarType::Float8E4M3
                           : unscaledF8ScalarType(type);
            case DataType::BF8:
                return interpretation == DataTypeInterpretation::BlockScaled
                           ? ScalarType::Float8E5M2
                           : unscaledF8ScalarType(type);
            case DataType::FP6:
                return ScalarType::Float6E2M3;
            case DataType::BF6:
                return ScalarType::Float6E3M2;
            case DataType::FP4:
                return ScalarType::Float4E2M1;
            case DataType::E8M0:
                return ScalarType::E8M0;
            case DataType::E5M3:
                return ScalarType::E5M3;
            case DataType::E4M3:
                return ScalarType::E4M3;
            default:
                throw std::invalid_argument("Unsupported rocRoller host-numerics data type.");
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

        IndexOrder indexOrder(TensorDescriptor const& descriptor)
        {
            if(descriptor.dimensions() != 2)
                throw std::invalid_argument("rocRoller GEMM generation requires rank-two tensors.");
            if(descriptor.stride(0) == descriptor.stride(1))
            {
                if(descriptor.size(0) == 1)
                    return IndexOrder::LastDimensionFastest;
                if(descriptor.size(1) == 1)
                    return IndexOrder::FirstDimensionFastest;
                throw std::invalid_argument(
                    "rocRoller GEMM generation requires non-overlapping matrix strides.");
            }
            return descriptor.stride(0) < descriptor.stride(1) ? IndexOrder::FirstDimensionFastest
                                                               : IndexOrder::LastDimensionFastest;
        }

        GenerationRecipe generationRecipe(TensorDescriptor const&   descriptor,
                                          DataInitialization const& initialization,
                                          ScalarType                type,
                                          float                     minimum,
                                          float                     maximum,
                                          uint32_t                  seed)
        {
            GenerationRecipeSettings const settings{
                .seed       = seed,
                .indexOrder = indexOrder(descriptor),
            };
            auto realOnly = [&settings](GenerationRecipe::Component component) {
                return GenerationRecipe::realOnly(std::move(component), settings);
            };

            switch(initialization.mode)
            {
            case DataInitializationMode::Bounded:
                return realOnly(
                    GenerationRecipe::uniformReal({.lower = minimum, .upper = maximum}));
            case DataInitializationMode::BoundedAlternatingSign:
            {
                std::vector<size_t> alternatingDimensions;
                for(size_t dimension = 0; dimension < descriptor.dimensions(); ++dimension)
                {
                    if((descriptor.stride(dimension) & 1U) != 0)
                        alternatingDimensions.push_back(dimension);
                }
                auto component = GenerationRecipe::uniformReal(
                    {.lower = 0.0, .upper = std::max(std::abs(minimum), std::abs(maximum))});
                if(!alternatingDimensions.empty())
                {
                    component = component.withAlternatingSign(
                        {.dimensions      = std::move(alternatingDimensions),
                         .negativeWhenOdd = (1U ^ (descriptor.offset() & 1U)) != 0});
                }
                return realOnly(std::move(component));
            }
            case DataInitializationMode::Unbounded:
            {
                auto const [lowerExponent, upperExponent] = unboundedExponentRange(type);
                return realOnly(
                    GenerationRecipe::randomEncodedExponent({.lowerUnbiasedExponent = lowerExponent,
                                                             .upperUnbiasedExponent = upperExponent,
                                                             .sourceType            = type}));
            }
            case DataInitializationMode::Identity:
                return realOnly(GenerationRecipe::identity());
            case DataInitializationMode::Ones:
                return realOnly(GenerationRecipe::constant({.value = 1.0}));
            case DataInitializationMode::Zeros:
                return realOnly(GenerationRecipe::zero());
            case DataInitializationMode::TrigonometricFromFloat:
                return realOnly(GenerationRecipe::uniformReal(
                                    {.lower = 0.0, .upper = 6.28318530717958647692528676655900576})
                                    .withCosineTransform());
            case DataInitializationMode::NormalFromFloat:
                return realOnly(GenerationRecipe::normal(
                    {.mean              = initialization.normalMean,
                     .standardDeviation = initialization.normalStandardDeviation}));
            }
            throw std::invalid_argument("Unknown rocRoller data initialization mode.");
        }

        MxDataGeneration mxDataGeneration(DataInitialization const& initialization,
                                          Shape const&              shape,
                                          float                     minimum,
                                          float                     maximum,
                                          uint32_t                  seed)
        {
            auto recipe = [&](GenerationRecipe::Component component,
                              uint64_t                    randomDomain
                              = roc::host_numerics::mx_generation_random_domain_version_1::data) {
                return GenerationRecipe::realOnly(
                    std::move(component),
                    {
                        .seed         = seed,
                        .indexOrder   = IndexOrder::FirstDimensionFastest,
                        .randomDomain = randomDomain,
                    });
            };

            switch(initialization.mode)
            {
            case DataInitializationMode::Bounded:
                return MxDataGeneration::preserveRange(
                    recipe(GenerationRecipe::uniformReal({.lower = minimum, .upper = maximum})),
                    {.lower = minimum, .upper = maximum});
            case DataInitializationMode::BoundedAlternatingSign:
            {
                std::vector<size_t> alternatingDimensions{0};
                if((shape[0] & 1U) != 0)
                    alternatingDimensions.push_back(1);
                const double maximumMagnitude = std::max(std::abs(minimum), std::abs(maximum));
                return MxDataGeneration::preserveRange(
                    recipe(GenerationRecipe::uniformReal({.lower = 0.0, .upper = maximumMagnitude})
                               .withAlternatingSign({.dimensions = std::move(alternatingDimensions),
                                                     .negativeWhenOdd = true})),
                    {.lower = -maximumMagnitude, .upper = maximumMagnitude});
            }
            case DataInitializationMode::Unbounded:
                return MxDataGeneration::preserveGeneratedEncoding(recipe(
                    GenerationRecipe::uniformFiniteEncodedValue(),
                    roc::host_numerics::mx_generation_random_domain_version_1::unboundedData));
            case DataInitializationMode::Identity:
                return MxDataGeneration::quantize(recipe(GenerationRecipe::identity()));
            case DataInitializationMode::Ones:
                return MxDataGeneration::quantize(
                    recipe(GenerationRecipe::constant({.value = 1.0})));
            case DataInitializationMode::Zeros:
                return MxDataGeneration::quantize(recipe(GenerationRecipe::zero()));
            case DataInitializationMode::TrigonometricFromFloat:
                return MxDataGeneration::quantize(
                    recipe(GenerationRecipe::uniformReal(
                               {.lower = 0.0, .upper = 6.28318530717958647692528676655900576})
                               .withCosineTransform()));
            case DataInitializationMode::NormalFromFloat:
                return MxDataGeneration::quantize(
                    recipe(GenerationRecipe::normal(
                               {.mean              = initialization.normalMean,
                                .standardDeviation = initialization.normalStandardDeviation}),
                           roc::host_numerics::mx_generation_random_domain_version_1::normal));
            }
            throw std::invalid_argument("Unknown rocRoller MX data initialization mode.");
        }

        MxScaleGenerationMode mxScaleGenerationMode(DataInitializationMode mode)
        {
            switch(mode)
            {
            case DataInitializationMode::Identity:
            case DataInitializationMode::Ones:
                return MxScaleGenerationMode::One;
            case DataInitializationMode::Zeros:
                return MxScaleGenerationMode::Minimum;
            case DataInitializationMode::Bounded:
            case DataInitializationMode::BoundedAlternatingSign:
            case DataInitializationMode::TrigonometricFromFloat:
            case DataInitializationMode::NormalFromFloat:
                return MxScaleGenerationMode::Derived;
            case DataInitializationMode::Unbounded:
                return MxScaleGenerationMode::RandomFinite;
            }
            throw std::invalid_argument("Unknown rocRoller MX scale initialization mode.");
        }

        Tensor generateUnscaledF8(TensorDescriptor const&   descriptor,
                                  DataInitialization const& initialization,
                                  ScalarType                type,
                                  float                     minimum,
                                  float                     maximum,
                                  uint32_t                  seed)
        {
            auto const layout = hostTensorLayout(descriptor);
            Tensor     source(ScalarType::Float32, layout);
            roc::host_numerics::generate(
                source, generationRecipe(descriptor, initialization, type, minimum, maximum, seed));

            return source.copyConvertedTo(type);
        }

        GeneratedTensor generateUnscaled(TensorDescriptor const&   descriptor,
                                         DataInitialization const& initialization,
                                         float                     minimum,
                                         float                     maximum,
                                         uint32_t                  seed,
                                         bool                      includeReference)
        {
            auto const type = hostScalarType(descriptor.dataType());
            if(descriptor.dataType() == DataType::FP8 || descriptor.dataType() == DataType::BF8)
            {
                auto data
                    = generateUnscaledF8(descriptor, initialization, type, minimum, maximum, seed);
                std::optional<Tensor> reference;
                if(includeReference)
                    reference = data.copyConvertedTo(ScalarType::Float32);
                return {std::move(data), std::nullopt, std::move(reference)};
            }

            Tensor data(type, hostTensorLayout(descriptor));
            roc::host_numerics::generate(
                data, generationRecipe(descriptor, initialization, type, minimum, maximum, seed));
            std::optional<Tensor> reference;
            if(includeReference)
                reference = data.copyConvertedTo(ScalarType::Float32);
            return {std::move(data), std::nullopt, std::move(reference)};
        }

        GeneratedTensor generateScaled(TensorDescriptor const&   descriptor,
                                       DataInitialization const& initialization,
                                       DataType                  scaleType,
                                       size_t                    blockedDimension,
                                       size_t                    scaleBlockSize,
                                       float                     minimum,
                                       float                     maximum,
                                       uint32_t                  seed,
                                       bool                      includeReference)
        {
            if(descriptor.dimensions() != 2)
                throw std::invalid_argument("rocRoller MX generation requires rank-two tensors.");
            if(descriptor.offset() != 0)
                throw std::invalid_argument("rocRoller MX generation does not support offsets.");
            if(blockedDimension >= descriptor.dimensions())
                throw std::invalid_argument(
                    "rocRoller MX blocked dimension exceeds the tensor rank.");
            if(scaleBlockSize == 0)
                throw std::invalid_argument("rocRoller MX scale block size must be nonzero.");

            std::array<size_t, 2> dimensions{0, 1};
            std::ranges::sort(dimensions, [&](size_t first, size_t second) {
                return descriptor.stride(first) < descriptor.stride(second);
            });
            auto const contiguousDimension = dimensions[0];
            auto const freeDimension       = dimensions[1];
            if(descriptor.stride(contiguousDimension) != 1)
                throw std::invalid_argument(
                    "rocRoller MX generation requires a stride-one matrix dimension.");

            Shape mxShape{descriptor.size(contiguousDimension), descriptor.size(freeDimension)};
            MxDataGeneration dataGeneration
                = mxDataGeneration(initialization, mxShape, minimum, maximum, seed);
            MxGenerationProblem problem(std::move(mxShape), std::move(dataGeneration));
            problem.dataType
                = hostScalarType(descriptor.dataType(), DataTypeInterpretation::BlockScaled);
            problem.scaleType = hostScalarType(scaleType);
            problem.leadingDimension
                = checkedPtrdiff(descriptor.stride(freeDimension),
                                 "rocRoller MX leading dimension exceeds ptrdiff_t.");
            problem.blockAxis = blockedDimension == contiguousDimension ? 0 : 1;
            problem.blockSize = scaleBlockSize;
            problem.scale     = mxScaleGenerationMode(initialization.mode);

            auto   result = roc::host_numerics::generateMx(problem);
            Tensor data   = result.data.shareStorageWithLayout(hostTensorLayout(descriptor));

            auto const scaleLayout = hostScaleLayout(descriptor, blockedDimension, scaleBlockSize);
            Tensor     scales      = result.scales.shareStorageWithLayout(scaleLayout);
            std::optional<Tensor> reference;
            if(includeReference)
                reference = result.reference.shareStorageWithLayout(hostTensorLayout(descriptor));
            return {std::move(data), std::move(scales), std::move(reference)};
        }
    }

    roc::host_numerics::ScalarType hostScalarType(DataType               type,
                                                  DataTypeInterpretation interpretation)
    {
        return makeHostScalarType(type, interpretation);
    }

    std::string toString(DataInitialization const& initialization)
    {
        auto modeName = [](DataInitializationMode mode) -> std::string_view {
            switch(mode)
            {
            case DataInitializationMode::Bounded:
                return "Bounded";
            case DataInitializationMode::BoundedAlternatingSign:
                return "BoundedAlternatingSign";
            case DataInitializationMode::Unbounded:
                return "Unbounded";
            case DataInitializationMode::Identity:
                return "Identity";
            case DataInitializationMode::Ones:
                return "Ones";
            case DataInitializationMode::Zeros:
                return "Zeros";
            case DataInitializationMode::TrigonometricFromFloat:
                return "TrigonometricFromFloat";
            case DataInitializationMode::NormalFromFloat:
                return "NormalFromFloat";
            }
            throw std::invalid_argument("Unknown rocRoller data initialization mode.");
        };

        auto description = std::string(modeName(initialization.mode));
        if(initialization.mode == DataInitializationMode::NormalFromFloat)
        {
            description += "(" + std::to_string(initialization.normalMean) + ", "
                           + std::to_string(initialization.normalStandardDeviation) + ")";
        }
        return "DataInitMode(" + description + ")";
    }

    roc::host_numerics::Layout hostTensorLayout(TensorDescriptor const& descriptor)
    {
        return makeTensorLayout(descriptor);
    }

    roc::host_numerics::Layout hostScaleLayout(TensorDescriptor const& descriptor,
                                               size_t                  blockedDimension,
                                               size_t                  blockSize)
    {
        if(descriptor.dimensions() != 2)
            throw std::invalid_argument("rocRoller block scales require a rank-two tensor.");
        if(blockedDimension >= descriptor.dimensions())
            throw std::invalid_argument("rocRoller block-scale dimension exceeds the tensor rank.");
        if(blockSize == 0)
            throw std::invalid_argument("rocRoller block-scale size must be nonzero.");

        std::array<size_t, 2> dimensions{0, 1};
        std::ranges::sort(dimensions, [&](size_t first, size_t second) {
            return descriptor.stride(first) < descriptor.stride(second);
        });
        auto const contiguousDimension = dimensions[0];
        if(descriptor.stride(contiguousDimension) != 1)
            throw std::invalid_argument(
                "rocRoller block scales require a stride-one matrix dimension.");

        auto const freeDimension = size_t{1} - blockedDimension;
        auto const freeExtent    = descriptor.size(freeDimension);
        auto const blockCount
            = descriptor.size(blockedDimension) / blockSize
              + static_cast<size_t>(descriptor.size(blockedDimension) % blockSize != 0);

        if(blockedDimension == contiguousDimension)
        {
            return Layout(
                Shape{freeExtent, blockCount},
                {checkedPtrdiff(blockCount, "rocRoller block-scale stride exceeds ptrdiff_t."), 1});
        }
        return Layout(
            Shape{freeExtent, blockCount},
            {1, checkedPtrdiff(freeExtent, "rocRoller block-scale stride exceeds ptrdiff_t.")});
    }

    GeneratedTensor generateHostTensor(TensorDescriptor const&             descriptor,
                                       DataInitialization const&           initialization,
                                       std::optional<BlockScaleGeneration> blockScale,
                                       float                               minimum,
                                       float                               maximum,
                                       uint32_t                            seed,
                                       bool                                includeReference)
    {
        if(!blockScale)
            return generateUnscaled(
                descriptor, initialization, minimum, maximum, seed, includeReference);
        if(blockScale->type == DataType::None)
            throw std::invalid_argument(
                "rocRoller block-scale generation requires a scale data type.");
        return generateScaled(descriptor,
                              initialization,
                              blockScale->type,
                              blockScale->blockedDimension,
                              blockScale->blockSize,
                              minimum,
                              maximum,
                              seed,
                              includeReference);
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
        auto generatedA = generateHostTensor(
            descriptorA,
            initializationA,
            scaleTypeA == DataType::None ? std::nullopt
                                         : std::optional<BlockScaleGeneration>{BlockScaleGeneration{
                                             scaleTypeA, 1, scaleBlockSize}},
            minimum,
            maximum,
            seed + 1);
        auto generatedB = generateHostTensor(
            descriptorB,
            initializationB,
            scaleTypeB == DataType::None ? std::nullopt
                                         : std::optional<BlockScaleGeneration>{BlockScaleGeneration{
                                             scaleTypeB, 0, scaleBlockSize}},
            minimum,
            maximum,
            seed + 2);
        auto generatedC = generateHostTensor(
            descriptorC, initializationC, std::nullopt, minimum, maximum, seed);

        return {std::move(generatedA.data),
                std::move(generatedB.data),
                std::move(generatedC.data),
                std::move(generatedA.scales),
                std::move(generatedB.scales)};
    }
}
