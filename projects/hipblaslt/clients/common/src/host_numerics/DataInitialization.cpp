// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Product-private translation from hipBLASLt initialization modes to
// product-independent host-numerics tensor generation recipes.

#include <hipblaslt/host_numerics/HipblasltDataInitialization.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace hipblaslt::host_numerics
{
    using ::roc::host_numerics::GenerationRecipe;
    using ::roc::host_numerics::IndexOrder;
    using ::roc::host_numerics::MxDataGeneration;
    using ::roc::host_numerics::MxGenerationOptions;
    using ::roc::host_numerics::MxScaleGenerationMode;
    using ::roc::host_numerics::MxTensor;
    using ::roc::host_numerics::ScalarCategory;
    using ::roc::host_numerics::ScalarType;
    using ::roc::host_numerics::Shape;
    using ::roc::host_numerics::Tensor;
    using ::roc::host_numerics::generate;
    using ::roc::host_numerics::generateAt;
    using ::roc::host_numerics::scalarTypeInfo;

    namespace amd_gpu_layout = ::roc::host_numerics::amd_gpu_layout;

    namespace
    {
        // Compatibility values inherited from hipblaslt_init_alt_impl_big/small.
        constexpr double specialInitializationAValue = 65'280.0;
        constexpr double specialInitializationBValue = 0.0000607967376708984375;

        // The historical FP16 accumulator probe alternates 2 * (max_finite - 4)
        // so the running sum exposes premature Float16 rounding without overflow.
        constexpr double maximumFiniteFloat16Value = 65'504.0;
        constexpr double fp16AccumulatorProbeStep  = 4.0;

        enum class ComplexGenerationPolicy
        {
            RealOnly,
            Replicated,
            Cartesian,
        };

        struct RandomIntegerRecipeConfiguration
        {
            bool                    small         = false;
            bool                    alternating   = false;
            ComplexGenerationPolicy complexPolicy = ComplexGenerationPolicy::Cartesian;
            uint64_t                seed          = defaultInitializationSeed;
        };

        struct HplRecipeConfiguration
        {
            bool                    positiveOnly  = false;
            bool                    alternating   = false;
            ComplexGenerationPolicy complexPolicy = ComplexGenerationPolicy::Cartesian;
            uint64_t                seed          = defaultInitializationSeed;
        };

        GenerationRecipe bindComponentRecipe(ScalarType                  destinationType,
                                               GenerationRecipe::Component component,
                                               ComplexGenerationPolicy     policy,
                                               uint64_t                    seed)
        {
            if(scalarTypeInfo(destinationType).category != ScalarCategory::Complex
               || policy == ComplexGenerationPolicy::RealOnly)
                return GenerationRecipe::realOnly(std::move(component), {.seed = seed});
            if(policy == ComplexGenerationPolicy::Replicated)
                return GenerationRecipe::replicated(std::move(component), {.seed = seed});
            GenerationRecipe::Component imaginary = component;
            return GenerationRecipe::cartesian(
                std::move(component), std::move(imaginary), {.seed = seed});
        }

        GenerationRecipe bindComponentPairRecipe(ScalarType                  destinationType,
                                                   GenerationRecipe::Component real,
                                                   GenerationRecipe::Component imaginary,
                                                   ComplexGenerationPolicy     policy,
                                                   uint64_t                    seed)
        {
            if(scalarTypeInfo(destinationType).category != ScalarCategory::Complex
               || policy == ComplexGenerationPolicy::RealOnly)
                return GenerationRecipe::realOnly(std::move(real), {.seed = seed});
            if(policy == ComplexGenerationPolicy::Replicated)
                return GenerationRecipe::replicated(std::move(real), {.seed = seed});
            return GenerationRecipe::cartesian(
                std::move(real), std::move(imaginary), {.seed = seed});
        }

        GenerationRecipe::Component
            trigonometricComponent(TrigonometricComponent component, bool absolute = false)
        {
            GenerationRecipe::Component result = component == TrigonometricComponent::Sine
                                                     ? GenerationRecipe::sine()
                                                     : GenerationRecipe::cosine();
            return absolute ? result.withAbsoluteTransform() : result;
        }

        GenerationRecipe
            randomIntegerRecipe(ScalarType                       type,
                                RandomIntegerRecipeConfiguration configuration = {})
        {
            const ScalarCategory category = scalarTypeInfo(type).category;
            if(category == ScalarCategory::Boolean)
                throw std::invalid_argument(
                    "Random-integer initialization does not support Boolean tensors.");

            GenerationRecipe::Component component = [&] {
                if(configuration.small)
                    return GenerationRecipe::uniformInteger({.lower = 1, .upper = 10})
                        .withAffineValueMapping({.scale = 0.1});

                switch(type)
                {
                case ScalarType::Float16:
                case ScalarType::BFloat16:
                    return GenerationRecipe::uniformInteger({.lower = -2, .upper = 2});
                case ScalarType::Int8:
                    return GenerationRecipe::uniformInteger({.lower = 1, .upper = 3});
                case ScalarType::Float4E2M1:
                    return GenerationRecipe::uniformInteger({.lower = -4, .upper = 4});
                case ScalarType::Float6E2M3:
                    return GenerationRecipe::uniformInteger({.lower = -7, .upper = 7});
                case ScalarType::Float6E3M2:
                    return GenerationRecipe::uniformInteger({.lower = -28, .upper = 28});
                case ScalarType::E8M0:
                    return GenerationRecipe::randomEncodedExponent(
                        {.lowerUnbiasedExponent = -3, .upperUnbiasedExponent = 3});
                default:
                    break;
                }

                if(category == ScalarCategory::SignedInteger
                   || category == ScalarCategory::UnsignedInteger
                   || category == ScalarCategory::FloatingPoint
                   || category == ScalarCategory::Complex || category == ScalarCategory::Scale)
                    return GenerationRecipe::uniformInteger({.lower = 1, .upper = 10});
                throw std::invalid_argument(
                    "Random-integer initialization requires an arithmetic tensor type.");
            }();

            if(configuration.alternating)
                component = component.withAlternatingSign(
                    {.dimensions = {0, 1}, .negativeWhenOdd = false});
            return bindComponentRecipe(
                type, std::move(component), configuration.complexPolicy, configuration.seed);
        }

        GenerationRecipe trigonometricRecipe(
            ScalarType              type,
            TrigonometricComponent  realComponent,
            bool                    positiveOnly = false,
            ComplexGenerationPolicy complexPolicy = ComplexGenerationPolicy::Cartesian,
            uint64_t                seed = defaultInitializationSeed)
        {
            const TrigonometricComponent imaginaryComponent
                = realComponent == TrigonometricComponent::Sine ? TrigonometricComponent::Cosine
                                                                : TrigonometricComponent::Sine;
            return bindComponentPairRecipe(type,
                                           trigonometricComponent(realComponent, positiveOnly),
                                           trigonometricComponent(imaginaryComponent, positiveOnly),
                                           complexPolicy,
                                           seed);
        }

        GenerationRecipe hplRecipe(ScalarType type, HplRecipeConfiguration configuration = {})
        {
            GenerationRecipe::Component component = [&] {
                if(type == ScalarType::E8M0)
                    return GenerationRecipe::randomEncodedExponent(
                        {.lowerUnbiasedExponent = -3, .upperUnbiasedExponent = 3});
                if(type == ScalarType::Int8)
                    return GenerationRecipe::uniformInteger(
                        {.lower = configuration.positiveOnly ? 0 : -1, .upper = 1});

                GenerationRecipe::Component uniform
                    = GenerationRecipe::uniformReal({.lower = -0.5, .upper = 0.5});
                return configuration.positiveOnly || type == ScalarType::E5M3
                           ? uniform.withAbsoluteTransform()
                           : uniform;
            }();

            if(configuration.alternating)
                component = component.withAlternatingSign(
                    {.dimensions = {0, 1}, .negativeWhenOdd = false});
            return bindComponentRecipe(
                type, std::move(component), configuration.complexPolicy, configuration.seed);
        }

        GenerationRecipe lowPrecisionRecipe(
            ScalarType              type,
            ComplexGenerationPolicy complexPolicy = ComplexGenerationPolicy::RealOnly,
            uint64_t                seed = defaultInitializationSeed)
        {
            GenerationRecipe::Component component = [&] {
                if(type == ScalarType::E8M0)
                    return GenerationRecipe::randomEncodedExponent(
                        {.lowerUnbiasedExponent = -3, .upperUnbiasedExponent = 3});
                if(type == ScalarType::Int8)
                    return GenerationRecipe::uniformInteger({.lower = -6, .upper = 6});

                GenerationRecipe::Component uniform
                    = GenerationRecipe::uniformReal({.lower = -6.0, .upper = 6.0});
                return type == ScalarType::E5M3 ? uniform.withAbsoluteTransform() : uniform;
            }();
            return bindComponentRecipe(type, std::move(component), complexPolicy, seed);
        }

        GenerationRecipe nanRecipe(
            ScalarType              type,
            ComplexGenerationPolicy complexPolicy = ComplexGenerationPolicy::RealOnly,
            uint64_t                seed = defaultInitializationSeed)
        {
            if(!scalarTypeInfo(type).supportsNaN)
                return GenerationRecipe::realOnly(
                    GenerationRecipe::randomRawBits(), {.seed = seed});
            return bindComponentRecipe(type, GenerationRecipe::typeNaN(), complexPolicy, seed);
        }

        GenerationRecipe normalRecipe(
            ScalarType              type,
            ComplexGenerationPolicy complexPolicy = ComplexGenerationPolicy::RealOnly,
            uint64_t                seed = defaultInitializationSeed)
        {
            return bindComponentRecipe(
                type, GenerationRecipe::normal({}), complexPolicy, seed);
        }

        GenerationRecipe uniformZeroOneRecipe(
            ScalarType              type,
            ComplexGenerationPolicy complexPolicy = ComplexGenerationPolicy::RealOnly,
            uint64_t                seed = defaultInitializationSeed)
        {
            const ScalarCategory category = scalarTypeInfo(type).category;
            const bool integerDestination = category == ScalarCategory::Boolean
                                            || category == ScalarCategory::SignedInteger
                                            || category == ScalarCategory::UnsignedInteger;
            GenerationRecipe::Component component
                = integerDestination
                      ? GenerationRecipe::uniformInteger({.lower = 0, .upper = 1})
                      : GenerationRecipe::uniformReal({.lower = 0.0, .upper = 1.0});
            return bindComponentRecipe(type, std::move(component), complexPolicy, seed);
        }

        // First output of the historical 32-bit linear-congruential selector.
        // The second output always selected positive infinity by default.
        inline constexpr uint32_t legacyOneSpecialIndexSelection = 3'554'416'254u;

        void validateInitialization(MatrixRole                     role,
                                    hipblaslt_initialization       initialization,
                                    std::optional<OneSpecialValue> oneSpecialValue,
                                    bool                           positiveOnly)
        {
            switch(role)
            {
            case MatrixRole::A:
            case MatrixRole::B:
            case MatrixRole::C:
                break;
            default:
                throw std::invalid_argument("Unsupported hipBLASLt matrix role.");
            }

            if(oneSpecialValue && initialization != hipblaslt_initialization::norm_dist_one_special)
                throw std::invalid_argument(
                    "A one-special value requires norm_dist_one_special initialization.");

            if(positiveOnly && initialization != hipblaslt_initialization::hpl
               && initialization != hipblaslt_initialization::trig_float)
                throw std::invalid_argument(
                    "Positive-only initialization is only supported for hpl and trig_float.");
        }

        GenerationRecipe sentinelRecipe(GenerationRecipe::Component component, bool complexOutput)
        {
            return complexOutput ? GenerationRecipe::replicated(std::move(component))
                                 : GenerationRecipe::realOnly(std::move(component));
        }

        bool supportsExplicitFloatingSentinel(ScalarType type)
        {
            return type == ScalarType::Float16 || type == ScalarType::BFloat16
                   || type == ScalarType::Float32 || type == ScalarType::Float64;
        }

        MxDataGeneration mxDataGeneration(hipblaslt_initialization initialization,
                                          ScalarType               dataType,
                                          uint64_t                 seed)
        {
            auto recipe = [&](GenerationRecipe::Component component) {
                return GenerationRecipe::realOnly(
                    std::move(component),
                    {
                        .seed       = seed,
                        .indexOrder = IndexOrder::FirstDimensionFastest,
                    });
            };

            switch(initialization)
            {
            case hipblaslt_initialization::hpl:
                return MxDataGeneration::preserveRange(
                    recipe(GenerationRecipe::uniformReal({.lower = -0.5, .upper = 0.5})),
                    {.lower = -0.5, .upper = 0.5});
            case hipblaslt_initialization::trig_float:
                return MxDataGeneration::quantize(
                    recipe(GenerationRecipe::uniformReal(
                               {.lower = 0.0, .upper = 6.28318530717958647692528676655900576})
                               .withCosineTransform()));
            case hipblaslt_initialization::uniform_01:
                return MxDataGeneration::preserveRange(
                    recipe(GenerationRecipe::uniformReal({.lower = 0.0, .upper = 1.0})),
                    {.lower = 0.0, .upper = 1.0});
            case hipblaslt_initialization::zero:
                return MxDataGeneration::quantize(recipe(GenerationRecipe::zero()));
            case hipblaslt_initialization::norm_dist:
                return MxDataGeneration::quantize(recipe(GenerationRecipe::normal(
                    {.mean              = 0.0,
                     .standardDeviation = dataType == ScalarType::Float4E2M1 ? 5.0 : 1.0})));
            case hipblaslt_initialization::rand_int:
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
            case hipblaslt_initialization::uniform_low_precision:
                return MxDataGeneration::preserveRange(
                    recipe(GenerationRecipe::uniformReal({.lower = -6.0, .upper = 6.0})),
                    {.lower = -6.0, .upper = 6.0});
            default:
                throw std::invalid_argument("Unsupported hipBLASLt MX data initialization mode.");
            }
        }

        MxScaleGenerationMode mxScaleGenerationMode(hipblaslt_initialization initialization)
        {
            if(initialization == hipblaslt_initialization::zero)
                return MxScaleGenerationMode::Minimum;
            if(initialization == hipblaslt_initialization::rand_int)
                return MxScaleGenerationMode::One;
            return MxScaleGenerationMode::Derived;
        }

        GenerationRecipe matrixGenerationRecipe(const Tensor&            destination,
                                                MatrixRole               role,
                                                hipblaslt_initialization initialization,
                                                uint64_t                 seed,
                                                bool                     forceNaN,
                                                bool                     positiveOnly)
        {
            const ScalarType type        = destination.type();
            const bool     complexOutput = scalarTypeInfo(type).category == ScalarCategory::Complex;
            if(forceNaN)
            {
                if(!scalarTypeInfo(type).supportsNaN)
                    throw std::invalid_argument(
                        "hipBLASLt input type has no supported NaN initialization.");
                return nanRecipe(type, ComplexGenerationPolicy::Replicated, seed);
            }

            switch(initialization)
            {
            case hipblaslt_initialization::rand_int:
            {
                const ScalarCategory category    = scalarTypeInfo(type).category;
                const bool           alternating = role == MatrixRole::B
                                         && category != ScalarCategory::Boolean
                                         && category != ScalarCategory::UnsignedInteger
                                         && category != ScalarCategory::Scale;
                return randomIntegerRecipe(type, {.alternating = alternating, .seed = seed});
            }
            case hipblaslt_initialization::trig_float:
            {
                const TrigonometricComponent realComponent = role == MatrixRole::B
                                                                 ? TrigonometricComponent::Cosine
                                                                 : TrigonometricComponent::Sine;
                return trigonometricRecipe(
                    type, realComponent, positiveOnly, ComplexGenerationPolicy::Cartesian, seed);
            }
            case hipblaslt_initialization::hpl:
                return hplRecipe(type, {.positiveOnly = positiveOnly, .seed = seed});
            case hipblaslt_initialization::uniform_low_precision:
                return lowPrecisionRecipe(type, ComplexGenerationPolicy::RealOnly, seed);
            case hipblaslt_initialization::special:
            {
                if(role == MatrixRole::A)
                    return GenerationRecipe::realOnly(
                        GenerationRecipe::constant({.value = specialInitializationAValue}));
                if(role == MatrixRole::B)
                    return GenerationRecipe::realOnly(
                        GenerationRecipe::constant({.value = specialInitializationBValue}));
                return GenerationRecipe::realOnly(
                    GenerationRecipe::uniformInteger({.lower = 1, .upper = 10}), {.seed = seed});
            }
            case hipblaslt_initialization::zero:
                return GenerationRecipe::realOnly(GenerationRecipe::zero());
            case hipblaslt_initialization::norm_dist:
                return normalRecipe(type, ComplexGenerationPolicy::RealOnly, seed);
            case hipblaslt_initialization::norm_dist_one_special:
            {
                if(!supportsExplicitFloatingSentinel(type))
                    throw std::invalid_argument("hipBLASLt one-special normal initialization "
                                                "requires an ordinary floating type.");
                return normalRecipe(type, ComplexGenerationPolicy::RealOnly, seed);
            }
            case hipblaslt_initialization::uniform_01:
                return uniformZeroOneRecipe(type, ComplexGenerationPolicy::RealOnly, seed);
            case hipblaslt_initialization::integer_exact:
            {
                GenerationRecipe::Component component
                    = GenerationRecipe::uniformInteger({.lower = 0, .upper = 2});
                if(role == MatrixRole::B)
                {
                    component = component.withAlternatingSign(
                        {.dimensions = {0, 1}, .negativeWhenOdd = false});
                }
                return GenerationRecipe::realOnly(std::move(component), {.seed = seed});
            }
            case hipblaslt_initialization::fp16_accumulator_probe:
            {
                if(type != ScalarType::Float16)
                    throw std::invalid_argument(
                        "hipBLASLt FP16 accumulator probe requires Float16 storage.");
                if(role == MatrixRole::A)
                    return GenerationRecipe::realOnly(GenerationRecipe::constant(
                        {.value = maximumFiniteFloat16Value - fp16AccumulatorProbeStep}));
                if(role == MatrixRole::B)
                    return GenerationRecipe::realOnly(
                        GenerationRecipe::constant({.value = 2.0})
                            .withAlternatingSign({.dimensions = {0}, .negativeWhenOdd = true}));
                return GenerationRecipe::realOnly(GenerationRecipe::zero());
            }
            case hipblaslt_initialization::inf:
                if(!supportsExplicitFloatingSentinel(type))
                    throw std::invalid_argument(
                        "hipBLASLt infinity initialization requires an ordinary floating type.");
                return sentinelRecipe(GenerationRecipe::typeInfinity(), complexOutput);
            case hipblaslt_initialization::neg_zero:
                if(!supportsExplicitFloatingSentinel(type))
                    throw std::invalid_argument("hipBLASLt negative-zero initialization requires "
                                                "an ordinary floating type.");
                return sentinelRecipe(GenerationRecipe::typeNegativeZero(), complexOutput);
            case hipblaslt_initialization::neg_inf:
                if(!supportsExplicitFloatingSentinel(type))
                    throw std::invalid_argument("hipBLASLt negative-infinity initialization "
                                                "requires an ordinary floating type.");
                return sentinelRecipe(GenerationRecipe::typeNegativeInfinity(), complexOutput);
            case hipblaslt_initialization::nan:
                if(!supportsExplicitFloatingSentinel(type))
                    throw std::invalid_argument(
                        "hipBLASLt NaN initialization requires an ordinary floating type.");
                return nanRecipe(type, ComplexGenerationPolicy::RealOnly, seed);
            }
            throw std::invalid_argument("Unsupported hipBLASLt host matrix initialization mode.");
        }

        void injectOneSpecial(Tensor view, std::optional<OneSpecialValue> requestedValue)
        {
            const size_t logicalElements = view.shape().elementCount();
            if(logicalElements == 0)
                return;

            const size_t specialLinearIndex
                = size_t(legacyOneSpecialIndexSelection) % logicalElements;
            const OneSpecialValue specialValue
                = requestedValue.value_or(OneSpecialValue::PositiveInfinity);

            const GenerationRecipe::Component component = [&] {
                switch(specialValue)
                {
                case OneSpecialValue::PositiveInfinity:
                    return GenerationRecipe::typeInfinity();
                case OneSpecialValue::NegativeInfinity:
                    return GenerationRecipe::typeNegativeInfinity();
                case OneSpecialValue::NaN:
                    return GenerationRecipe::typeNaN();
                }
                throw std::invalid_argument("Unsupported hipBLASLt one-special value.");
            }();
            generateAt(
                view,
                specialLinearIndex,
                sentinelRecipe(component,
                               scalarTypeInfo(view.type()).category == ScalarCategory::Complex));
        }
    } // namespace

    MxTensor generateMxData(hipDataType              dataType,
                            hipDataType              scaleType,
                            Shape                    shape,
                            uint64_t                 leadingDimension,
                            size_t                   blockAxis,
                            size_t                   blockSize,
                            hipblaslt_initialization initialization,
                            uint64_t                 seed)
    {
        if(shape.rank() != 2)
            throw std::invalid_argument("hipBLASLt MX generation requires a rank-two shape.");
        if(leadingDimension > static_cast<uint64_t>(std::numeric_limits<ptrdiff_t>::max()))
            throw std::overflow_error("hipBLASLt MX leading dimension exceeds ptrdiff_t.");
        const ScalarType hostDataType = scalarType(dataType);
        const ScalarType hostScaleType
            = scaleType == HIP_R_8F_E4M3 ? ScalarType::E4M3 : scalarType(scaleType);
        MxDataGeneration dataGeneration = mxDataGeneration(initialization, hostDataType, seed);

        MxGenerationOptions options;
        options.dataType         = hostDataType;
        options.scaleType        = hostScaleType;
        options.leadingDimension = static_cast<ptrdiff_t>(leadingDimension);
        options.blockAxis        = blockAxis;
        options.blockSize        = blockSize;
        options.scale            = mxScaleGenerationMode(initialization);
        return generateMx(std::move(shape), std::move(dataGeneration), options);
    }

    amd_gpu_layout::MxScaleStorageLayout mxScaleStorageLayoutForArchName(std::string_view archName)
    {
        return amd_gpu_layout::mxScaleStorageLayoutForArchitectureName(archName);
    }

    amd_gpu_layout::MxScaleStorageLayout
        mxScaleStorageLayoutForFormat(hipblaslt_scaling_format scalingFormat,
                                      std::string_view         archName)
    {
        if(scalingFormat == hipblaslt_scaling_format::Block_32_UE8M0_32_8_EXT)
            return amd_gpu_layout::MxScaleStorageLayout::Gfx950;
        if(mxScaleStorageLayoutForArchName(archName)
           == amd_gpu_layout::MxScaleStorageLayout::Gfx1250)
            return amd_gpu_layout::MxScaleStorageLayout::Gfx1250;
        return amd_gpu_layout::MxScaleStorageLayout::Natural;
    }

    GenerationRecipe initializationRecipe(ScalarType               type,
                                          hipblaslt_initialization initialization,
                                          uint64_t                 seed,
                                          TrigonometricComponent   trigonometric)
    {
        switch(initialization)
        {
        case hipblaslt_initialization::rand_int:
            return randomIntegerRecipe(type, {.seed = seed});
        case hipblaslt_initialization::trig_float:
            return trigonometricRecipe(
                type, trigonometric, false, ComplexGenerationPolicy::Cartesian, seed);
        case hipblaslt_initialization::hpl:
            return hplRecipe(type, {.seed = seed});
        case hipblaslt_initialization::uniform_low_precision:
            return lowPrecisionRecipe(type, ComplexGenerationPolicy::Cartesian, seed);
        case hipblaslt_initialization::special:
            return GenerationRecipe::realOnly(
                GenerationRecipe::constant({.value = specialInitializationAValue}), {.seed = seed});
        case hipblaslt_initialization::zero:
            return GenerationRecipe::realOnly(GenerationRecipe::zero(), {.seed = seed});
        case hipblaslt_initialization::norm_dist:
            return normalRecipe(type, ComplexGenerationPolicy::Cartesian, seed);
        case hipblaslt_initialization::uniform_01:
            return uniformZeroOneRecipe(type, ComplexGenerationPolicy::Cartesian, seed);
        case hipblaslt_initialization::integer_exact:
            return bindComponentRecipe(type,
                                       GenerationRecipe::uniformInteger({.lower = 0, .upper = 2}),
                                       ComplexGenerationPolicy::Cartesian,
                                       seed);
        case hipblaslt_initialization::inf:
            return GenerationRecipe::realOnly(GenerationRecipe::typeInfinity(), {.seed = seed});
        case hipblaslt_initialization::neg_zero:
            return GenerationRecipe::realOnly(GenerationRecipe::typeNegativeZero(), {.seed = seed});
        case hipblaslt_initialization::neg_inf:
            return GenerationRecipe::realOnly(GenerationRecipe::typeNegativeInfinity(),
                                              {.seed = seed});
        case hipblaslt_initialization::nan:
            return nanRecipe(type, ComplexGenerationPolicy::RealOnly, seed);
        case hipblaslt_initialization::fp16_accumulator_probe:
        case hipblaslt_initialization::norm_dist_one_special:
            throw std::invalid_argument(
                "Requested hipBLASLt initialization requires matrix role and layout information.");
        }
        throw std::invalid_argument("Unsupported hipBLASLt tensor initialization mode.");
    }

    GenerationRecipe groupedGemmInitializationRecipe(ScalarType                      type,
                                                     hipblaslt_initialization        mode,
                                                     initialization::OperandSequence operand,
                                                     uint64_t                        seed)
    {
        switch(mode)
        {
        case hipblaslt_initialization::rand_int:
        case hipblaslt_initialization::trig_float:
        case hipblaslt_initialization::hpl:
        case hipblaslt_initialization::uniform_low_precision:
        case hipblaslt_initialization::special:
        case hipblaslt_initialization::zero:
            break;
        default:
            throw std::invalid_argument(
                "Grouped GEMM does not support the requested initialization mode.");
        }

        switch(operand)
        {
        case initialization::OperandSequence::MatrixA:
        case initialization::OperandSequence::MatrixB:
        case initialization::OperandSequence::MatrixC:
        case initialization::OperandSequence::Bias:
            break;
        default:
            throw std::invalid_argument("Unsupported grouped GEMM initialization operand.");
        }

        if(mode == hipblaslt_initialization::rand_int)
        {
            GenerationRecipe::Component component
                = GenerationRecipe::uniformInteger({.lower = 1, .upper = 10});
            if(operand == initialization::OperandSequence::MatrixB)
                component
                    = component.withAlternatingSign({.dimensions = {0}, .negativeWhenOdd = false});
            return bindComponentRecipe(
                type, std::move(component), ComplexGenerationPolicy::RealOnly, seed);
        }

        if(mode == hipblaslt_initialization::special)
        {
            if(operand == initialization::OperandSequence::MatrixA)
                return GenerationRecipe::realOnly(
                    GenerationRecipe::constant({.value = specialInitializationAValue}),
                    {.seed = seed});
            if(operand == initialization::OperandSequence::MatrixB)
                return GenerationRecipe::realOnly(
                    GenerationRecipe::constant({.value = specialInitializationBValue}),
                    {.seed = seed});
            return hplRecipe(type, {.seed = seed});
        }

        return initializationRecipe(type,
                                    mode,
                                    seed,
                                    operand == initialization::OperandSequence::MatrixB
                                        ? TrigonometricComponent::Cosine
                                        : TrigonometricComponent::Sine);
    }

    void initializeMatrix(Tensor                         destination,
                          MatrixRole                     role,
                          hipblaslt_initialization       initialization,
                          uint64_t                       seed,
                          bool                           forceNaN,
                          std::optional<OneSpecialValue> oneSpecialValue,
                          bool                           positiveOnly)
    {
        validateInitialization(role, initialization, oneSpecialValue, positiveOnly);
        generate(destination,
                 matrixGenerationRecipe(
                     destination, role, initialization, seed, forceNaN, positiveOnly));

        if(initialization == hipblaslt_initialization::norm_dist_one_special)
            injectOneSpecial(destination, oneSpecialValue);
    }
} // namespace hipblaslt::host_numerics
