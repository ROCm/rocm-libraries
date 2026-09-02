// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Product-private translation from hipBLASLt initialization modes to
// product-independent host-numerics tensor generation recipes.

#include <hipblaslt/host_numerics/HipblasltDataInitialization.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace hipblaslt::host_numerics
{
    using namespace ::roc::host_numerics;

    namespace
    {
        // First output of the historical 32-bit linear-congruential selector.
        // The second output always selected positive infinity by default.
        inline constexpr uint32_t legacyOneSpecialIndexSelection = 3'554'416'254u;

        size_t checkedMultiply(size_t left, size_t right)
        {
            if(left != 0 && right > std::numeric_limits<size_t>::max() / left)
                throw std::overflow_error("hipBLASLt matrix initialization size overflow.");
            return left * right;
        }

        size_t checkedAdd(size_t left, size_t right)
        {
            if(right > std::numeric_limits<size_t>::max() - left)
                throw std::overflow_error("hipBLASLt matrix initialization size overflow.");
            return left + right;
        }

        size_t matrixElements(const MatrixInitialization& initialization)
        {
            return checkedMultiply(initialization.leadingDimension, initialization.columns);
        }

        ptrdiff_t layoutStride(size_t stride)
        {
            if(stride > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()))
                throw std::overflow_error(
                    "hipBLASLt matrix initialization stride exceeds ptrdiff_t.");
            return static_cast<ptrdiff_t>(stride);
        }

        bool usesPackedBatchStride(const MatrixInitialization& initialization)
        {
            return initialization.initialization == hipblaslt_initialization::norm_dist_one_special
                   || (initialization.role == MatrixRole::B
                       && (initialization.initialization == hipblaslt_initialization::integer_exact
                           || initialization.initialization
                                  == hipblaslt_initialization::fp16_accumulator_probe));
        }

        size_t effectiveBatchStride(const MatrixInitialization& initialization)
        {
            const size_t oneMatrixElements = matrixElements(initialization);
            if(usesPackedBatchStride(initialization))
                return initialization.batchStride
                           ? std::max(initialization.batchStride, oneMatrixElements)
                           : oneMatrixElements;
            return initialization.batchStride;
        }

        size_t storageElements(const MatrixInitialization& initialization, size_t batchStride)
        {
            if(initialization.batchCount == 0)
                return 0;
            const size_t oneMatrixElements = matrixElements(initialization);
            if(batchStride >= initialization.leadingDimension)
                return checkedAdd(oneMatrixElements,
                                  checkedMultiply(initialization.batchCount - 1, batchStride));
            return oneMatrixElements;
        }

        void validateInitialization(const MatrixInitialization& initialization)
        {
            switch(initialization.role)
            {
            case MatrixRole::A:
            case MatrixRole::B:
            case MatrixRole::C:
                break;
            default:
                throw std::invalid_argument("Unsupported hipBLASLt matrix role.");
            }

            if(initialization.oneSpecialValue
               && initialization.initialization != hipblaslt_initialization::norm_dist_one_special)
                throw std::invalid_argument(
                    "A one-special value requires norm_dist_one_special initialization.");

            if(initialization.positiveOnly
               && initialization.initialization != hipblaslt_initialization::hpl
               && initialization.initialization != hipblaslt_initialization::trig_float)
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

        initialization::OperandSequence operandSequence(MatrixRole role)
        {
            switch(role)
            {
            case MatrixRole::A:
                return initialization::OperandSequence::MatrixA;
            case MatrixRole::B:
                return initialization::OperandSequence::MatrixB;
            case MatrixRole::C:
                return initialization::OperandSequence::MatrixC;
            }
            throw std::invalid_argument("Unsupported hipBLASLt matrix role.");
        }

        uint64_t matrixSeed(uint64_t seed, MatrixRole role)
        {
            return initialization::seedForSequence(seed, operandSequence(role));
        }

#if HIPBLASLT_ENABLE_MXDATAGENERATOR
        MxDataGeneration mxDataGeneration(hipblaslt_initialization initialization,
                                          ScalarType               dataType,
                                          float                    minimum,
                                          float                    maximum,
                                          uint32_t                 seed)
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
#endif

        GenerationRecipe matrixGenerationRecipe(const MatrixInitialization& initialization,
                                                const Tensor&               destination)
        {
            const ScalarType type        = destination.type();
            const bool     complexOutput = scalarTypeInfo(type).category == ScalarCategory::Complex;
            const uint64_t seed = matrixSeed(defaultInitializationSeed, initialization.role);
            if(initialization.forceNaN)
            {
                if(!scalarTypeInfo(type).supportsNaN)
                    throw std::invalid_argument(
                        "hipBLASLt input type has no supported NaN initialization.");
                return nanRecipe(type, ComplexGenerationPolicy::Replicated, seed);
            }

            switch(initialization.initialization)
            {
            case hipblaslt_initialization::rand_int:
            {
                const ScalarCategory category    = scalarTypeInfo(type).category;
                const bool           alternating = initialization.role == MatrixRole::B
                                         && category != ScalarCategory::Boolean
                                         && category != ScalarCategory::UnsignedInteger
                                         && category != ScalarCategory::Scale;
                return randomIntegerRecipe(type, {.alternating = alternating, .seed = seed});
            }
            case hipblaslt_initialization::trig_float:
            {
                const TrigonometricComponent realComponent = initialization.role == MatrixRole::B
                                                                 ? TrigonometricComponent::Cosine
                                                                 : TrigonometricComponent::Sine;
                return trigonometricRecipe(type, realComponent, initialization.positiveOnly);
            }
            case hipblaslt_initialization::hpl:
                return hplRecipe(type, {.positiveOnly = initialization.positiveOnly, .seed = seed});
            case hipblaslt_initialization::uniform_low_precision:
                return lowPrecisionRecipe(type, ComplexGenerationPolicy::RealOnly, seed);
            case hipblaslt_initialization::special:
            {
                if(initialization.role == MatrixRole::A)
                    return GenerationRecipe::realOnly(
                        GenerationRecipe::constant({.value = specialInitializationAValue}));
                if(initialization.role == MatrixRole::B)
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
                return normalRecipe(type,
                                    ComplexGenerationPolicy::RealOnly,
                                    matrixSeed(oneSpecialInitializationSeed, initialization.role));
            }
            case hipblaslt_initialization::uniform_01:
                return uniformZeroOneRecipe(type, ComplexGenerationPolicy::RealOnly, seed);
            case hipblaslt_initialization::integer_exact:
            {
                GenerationRecipe::Component component
                    = GenerationRecipe::uniformInteger({.lower = 0, .upper = 2});
                if(initialization.role == MatrixRole::B)
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
                if(initialization.role == MatrixRole::A)
                    return GenerationRecipe::realOnly(GenerationRecipe::constant(
                        {.value = maximumFiniteFloat16Value - fp16AccumulatorProbeStep}));
                if(initialization.role == MatrixRole::B)
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

#if HIPBLASLT_ENABLE_MXDATAGENERATOR
    MxTensor generateMxData(hipDataType              dataType,
                            hipDataType              scaleType,
                            Shape                    shape,
                            uint64_t                 leadingDimension,
                            size_t                   blockAxis,
                            size_t                   blockSize,
                            hipblaslt_initialization initialization,
                            uint32_t                 seed)
    {
        if(shape.rank() != 2)
            throw std::invalid_argument("hipBLASLt MX generation requires a rank-two shape.");
        if(leadingDimension > static_cast<uint64_t>(std::numeric_limits<ptrdiff_t>::max()))
            throw std::overflow_error("hipBLASLt MX leading dimension exceeds ptrdiff_t.");
        const ScalarType hostDataType = scalarType(dataType);
        const ScalarType hostScaleType
            = scaleType == HIP_R_8F_E4M3 ? ScalarType::E4M3 : scalarType(scaleType);
        MxDataGeneration dataGeneration
            = mxDataGeneration(initialization, hostDataType, -1.0f, 1.0f, seed);

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
#endif

    Tensor generateMatrix(const MatrixInitialization& initialization)
    {
        validateInitialization(initialization);
        if(initialization.leadingDimension < initialization.rows)
            throw std::invalid_argument(
                "hipBLASLt initialization leading dimension is smaller than rows.");

        const ScalarType type        = scalarType(initialization.type);
        const size_t     batchStride = effectiveBatchStride(initialization);
        const size_t     elements    = storageElements(initialization, batchStride);
        const size_t     generatedBatchCount
            = initialization.batchCount == 0
                  ? 0
                  : (batchStride >= initialization.leadingDimension ? initialization.batchCount
                                                                    : 1);
        Layout layout(
            Shape{initialization.rows, initialization.columns, generatedBatchCount},
            {1, layoutStride(initialization.leadingDimension), layoutStride(batchStride)});
        Tensor matrix = Tensor(type, Layout::contiguousLastDimensionFastest(Shape{elements}))
                            .shareStorageWithLayout(std::move(layout));
        if(initialization.rows == 0 || initialization.columns == 0
           || initialization.batchCount == 0)
            return matrix;

        generate(matrix, matrixGenerationRecipe(initialization, matrix));

        if(initialization.initialization == hipblaslt_initialization::norm_dist_one_special)
            injectOneSpecial(matrix, initialization.oneSpecialValue);
        return matrix;
    }
} // namespace hipblaslt::host_numerics
