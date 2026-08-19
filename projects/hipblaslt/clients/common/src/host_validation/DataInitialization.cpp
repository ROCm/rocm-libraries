// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Product-private translation from hipBLASLt initialization modes to
// product-independent host-validation tensor generation recipes.

#include <hipblaslt/host_validation/HipblasltDataInitialization.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hipblaslt::host_validation
{
    using namespace ::roc::host_validation;

    namespace
    {
        bool usesPackedBatchStride(const MatrixStorageInitialization& initialization)
        {
            return initialization.initialization == hipblaslt_initialization::norm_dist_one_special
                   || (initialization.role == MatrixRole::B
                       && (initialization.initialization == hipblaslt_initialization::integer_exact
                           || initialization.initialization
                                  == hipblaslt_initialization::fp16_accumulator_probe));
        }

        size_t effectiveBatchStride(const MatrixStorageInitialization& initialization)
        {
            const size_t matrixElements = initialization.leadingDimension * initialization.columns;
            if(usesPackedBatchStride(initialization))
                return initialization.batchStride
                           ? std::max(initialization.batchStride, matrixElements)
                           : matrixElements;
            return initialization.batchStride;
        }

        size_t storageElements(const MatrixStorageInitialization& initialization,
                               size_t                             batchStride)
        {
            if(initialization.batchCount == 0)
                return 0;
            const size_t matrixElements = initialization.leadingDimension * initialization.columns;
            if(batchStride >= initialization.leadingDimension)
                return matrixElements + (initialization.batchCount - 1) * batchStride;
            return matrixElements;
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

        GenerationRecipe matrixGenerationRecipe(const MatrixStorageInitialization& initialization,
                                                const Tensor&                      destination)
        {
            const ScalarType type    = destination.type();
            const bool complexOutput = scalarTypeInfo(type).category == ScalarCategory::Complex;
            if(initialization.forceNaN)
            {
                const ScalarTypeInfo& info = scalarTypeInfo(type);
                if(info.category == ScalarCategory::Scale
                   || (info.category == ScalarCategory::FloatingPoint && !info.supportsNaN))
                    throw std::invalid_argument(
                        "hipBLASLt input type has no supported NaN initialization.");
                return nanRecipe(type, ComplexGenerationPolicy::Replicated);
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
                return randomIntegerRecipe(type, {.alternating = alternating});
            }
            case hipblaslt_initialization::trig_float:
            {
                const TrigonometricComponent realComponent = initialization.role == MatrixRole::B
                                                                 ? TrigonometricComponent::Cosine
                                                                 : TrigonometricComponent::Sine;
                return trigonometricRecipe(type, realComponent, initialization.positiveOnly);
            }
            case hipblaslt_initialization::hpl:
                return hplRecipe(type, {.positiveOnly = initialization.positiveOnly});
            case hipblaslt_initialization::uniform_low_precision:
                return lowPrecisionRecipe(type);
            case hipblaslt_initialization::special:
            {
                if(initialization.role == MatrixRole::A)
                    return GenerationRecipe::realOnly(
                        GenerationRecipe::constant({.value = specialInitializationAValue}));
                if(initialization.role == MatrixRole::B)
                    return GenerationRecipe::realOnly(
                        GenerationRecipe::constant({.value = specialInitializationBValue}));
                return GenerationRecipe::realOnly(
                    GenerationRecipe::uniformInteger({.lower = 1, .upper = 10}));
            }
            case hipblaslt_initialization::zero:
                return GenerationRecipe::realOnly(GenerationRecipe::zero());
            case hipblaslt_initialization::norm_dist:
                return normalRecipe(type);
            case hipblaslt_initialization::norm_dist_one_special:
            {
                if(!supportsExplicitFloatingSentinel(type))
                    throw std::invalid_argument("hipBLASLt one-special normal initialization "
                                                "requires an ordinary floating type.");
                return normalRecipe(
                    type, ComplexGenerationPolicy::RealOnly, oneSpecialInitializationSeed);
            }
            case hipblaslt_initialization::uniform_01:
                return uniformZeroOneRecipe(type);
            case hipblaslt_initialization::integer_exact:
            {
                GenerationRecipe::Component component
                    = GenerationRecipe::uniformInteger({.lower = 0, .upper = 2});
                uint64_t recipeSeed = defaultInitializationSeed;
                if(initialization.role == MatrixRole::B)
                {
                    component = component.withAlternatingSign(
                        {.dimensions = {0, 1}, .negativeWhenOdd = false});
                    recipeSeed = initialization::seedForSequence(
                        defaultInitializationSeed, initialization::integerExactMatrixBSequence);
                }
                return GenerationRecipe::realOnly(std::move(component), {.seed = recipeSeed});
            }
            case hipblaslt_initialization::fp16_accumulator_probe:
            {
                if(type != ScalarType::Float16)
                    return GenerationRecipe::realOnly(GenerationRecipe::zero());
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
                return nanRecipe(type);
            }
            throw std::invalid_argument("Unsupported hipBLASLt host matrix initialization mode.");
        }

        enum class OneSpecialValue
        {
            PositiveInfinity = 0,
            NegativeInfinity = 1,
            NaN              = 2,
        };

        void injectOneSpecial(Tensor view, int requestedSpecialType)
        {
            const size_t logicalElements = view.shape().elementCount();
            if(logicalElements == 0)
                return;

            uint32_t     state              = static_cast<uint32_t>(oneSpecialInitializationSeed)
                                                  * initialization::oneSpecialLcgMultiplier
                                              + initialization::oneSpecialLcgIncrement;
            const size_t specialLinearIndex = size_t(state) % logicalElements;
            state                           = state * initialization::oneSpecialLcgMultiplier
                                              + initialization::oneSpecialLcgIncrement;
            const int specialType
                = requestedSpecialType >= 0
                          && requestedSpecialType < initialization::oneSpecialValueCount
                      ? requestedSpecialType
                      : int(state >> initialization::oneSpecialLcgValueShift)
                            % initialization::oneSpecialValueCount;

            const GenerationRecipe::Component component
                = specialType == static_cast<int>(OneSpecialValue::PositiveInfinity)
                      ? GenerationRecipe::typeInfinity()
                  : specialType == static_cast<int>(OneSpecialValue::NegativeInfinity)
                      ? GenerationRecipe::typeNegativeInfinity()
                      : GenerationRecipe::typeNaN();
            generateAt(
                view,
                specialLinearIndex,
                sentinelRecipe(component,
                               scalarTypeInfo(view.type()).category == ScalarCategory::Complex));
        }
    } // namespace

    std::vector<std::byte> generateMatrixStorage(const MatrixStorageInitialization& initialization)
    {
        if(initialization.leadingDimension < initialization.rows)
            throw std::invalid_argument(
                "hipBLASLt initialization leading dimension is smaller than rows.");

        const ScalarType       type        = scalarType(initialization.type);
        const size_t           batchStride = effectiveBatchStride(initialization);
        const size_t           elements    = storageElements(initialization, batchStride);
        std::vector<std::byte> storage(
            storageBytesForLayout(type, Layout::contiguous(Shape{elements})));
        if(initialization.rows == 0 || initialization.columns == 0
           || initialization.batchCount == 0)
            return storage;

        const size_t generatedBatchCount
            = batchStride >= initialization.leadingDimension ? initialization.batchCount : 1;
        Layout layout(Shape{initialization.rows, initialization.columns, generatedBatchCount},
                      {1,
                       static_cast<ptrdiff_t>(initialization.leadingDimension),
                       static_cast<ptrdiff_t>(batchStride)});
        Tensor view(type, layout, storage);
        generate(view, matrixGenerationRecipe(initialization, view));

        if(initialization.initialization == hipblaslt_initialization::norm_dist_one_special)
            injectOneSpecial(view, initialization.specialValueType);
        return std::vector<std::byte>(view.storage().begin(), view.storage().end());
    }
} // namespace hipblaslt::host_validation
