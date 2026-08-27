// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private hipBLASLt adapter.

#include <cstddef>
#include <cstdint>
#include <hipblaslt/host_validation/GenerationRecipes.hpp>
#include <hipblaslt/host_validation/Types.hpp>
#include <hipblaslt_datatype2string.hpp>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>

namespace hipblaslt::host_validation
{
    using namespace ::roc::host_validation;

    enum class MatrixRole
    {
        A,
        B,
        C,
    };

    enum class OneSpecialValue : uint8_t
    {
        PositiveInfinity,
        NegativeInfinity,
        NaN,
    };

    struct MatrixInitialization
    {
        MatrixRole                     role             = MatrixRole::A;
        hipblaslt_initialization       initialization   = hipblaslt_initialization::zero;
        bool                           forceNaN         = false;
        hipDataType                    type             = HIP_R_32F;
        size_t                         rows             = 0;
        size_t                         columns          = 0;
        size_t                         leadingDimension = 0;
        size_t                         batchStride      = 0;
        size_t                         batchCount       = 1;
        std::optional<OneSpecialValue> oneSpecialValue;
        bool                           positiveOnly = false;
    };

    ::roc::host_validation::Tensor generateMatrix(const MatrixInitialization& initialization);

    namespace detail
    {
        inline void generateIntoCallerStorage(void*                   data,
                                              ScalarType              type,
                                              Layout                  layout,
                                              const GenerationRecipe& recipe)
        {
            const size_t storageBytes = storageBytesForLayout(type, layout);
            if(storageBytes != 0 && data == nullptr)
                throw std::invalid_argument(
                    "hipBLASLt initialization destination storage is null.");

            ::roc::host_validation::Tensor generated
                = generate(type, std::move(layout), recipe);
            std::span<std::byte> destinationStorage(
                static_cast<std::byte*>(data), storageBytes);
            const uint16_t bits = scalarTypeInfo(type).storageBits;
            ::roc::host_validation::detail::forEachIndex(
                generated.shape(), [&](std::span<const size_t> indices, size_t) {
                    const ptrdiff_t offset = generated.layout().elementOffset(indices);
                    ::roc::host_validation::detail::copyBitRange(
                        generated.rawEncodedBackingStorage(),
                        ::roc::host_validation::detail::bitOffset(type, offset),
                        destinationStorage,
                        ::roc::host_validation::detail::bitOffset(type, offset),
                        bits);
                });
        }
    } // namespace detail

    template <typename T>
    void initializeTensor(T* data, Layout layout, const GenerationRecipe& recipe)
    {
        detail::generateIntoCallerStorage(
            static_cast<void*>(data), scalarType<T>(), std::move(layout), recipe);
    }

    inline void
        initializeTensor(void* data, ScalarType type, Layout layout, const GenerationRecipe& recipe)
    {
        detail::generateIntoCallerStorage(data, type, std::move(layout), recipe);
    }

    inline void initializeTensor(void*                   data,
                                 hipDataType             type,
                                 Layout                  layout,
                                 const GenerationRecipe& recipe)
    {
        initializeTensor(data, scalarType(type), std::move(layout), recipe);
    }

    template <typename T>
    void initializeMatrixBatches(T*                      data,
                                 size_t                  rows,
                                 size_t                  columns,
                                 ptrdiff_t               leadingDimension,
                                 ptrdiff_t               batchStride,
                                 size_t                  batchCount,
                                 const GenerationRecipe& recipe)
    {
        initializeTensor(
            data,
            Layout(Shape{rows, columns, batchCount}, {1, leadingDimension, batchStride}),
            recipe);
    }

    inline void initializeMatrixBatches(void*                   data,
                                        ScalarType              type,
                                        size_t                  rows,
                                        size_t                  columns,
                                        ptrdiff_t               leadingDimension,
                                        ptrdiff_t               batchStride,
                                        size_t                  batchCount,
                                        const GenerationRecipe& recipe)
    {
        initializeTensor(
            data,
            type,
            Layout(Shape{rows, columns, batchCount}, {1, leadingDimension, batchStride}),
            recipe);
    }

    inline void initializeMatrixBatches(void*                   data,
                                        hipDataType             type,
                                        size_t                  rows,
                                        size_t                  columns,
                                        ptrdiff_t               leadingDimension,
                                        ptrdiff_t               batchStride,
                                        size_t                  batchCount,
                                        const GenerationRecipe& recipe)
    {
        initializeMatrixBatches(data,
                                scalarType(type),
                                rows,
                                columns,
                                leadingDimension,
                                batchStride,
                                batchCount,
                                recipe);
    }

    inline GenerationRecipe vectorInitializationRecipe(ScalarType               type,
                                                       hipblaslt_initialization initialization,
                                                       TrigonometricComponent   trigonometric)
    {
        switch(initialization)
        {
        case hipblaslt_initialization::rand_int:
            return randomIntegerRecipe(type);
        case hipblaslt_initialization::trig_float:
            return trigonometricRecipe(type, trigonometric);
        case hipblaslt_initialization::hpl:
            return hplRecipe(type);
        case hipblaslt_initialization::uniform_low_precision:
            return lowPrecisionRecipe(type, ComplexGenerationPolicy::Cartesian);
        case hipblaslt_initialization::special:
            return GenerationRecipe::realOnly(
                GenerationRecipe::constant({.value = specialInitializationAValue}));
        case hipblaslt_initialization::zero:
            return GenerationRecipe::realOnly(GenerationRecipe::zero());
        case hipblaslt_initialization::norm_dist:
            return normalRecipe(type, ComplexGenerationPolicy::Cartesian);
        case hipblaslt_initialization::uniform_01:
            return uniformZeroOneRecipe(type, ComplexGenerationPolicy::Cartesian);
        case hipblaslt_initialization::integer_exact:
            return bindComponentRecipe(type,
                                       GenerationRecipe::uniformInteger({.lower = 0, .upper = 2}),
                                       ComplexGenerationPolicy::Cartesian,
                                       defaultInitializationSeed);
        case hipblaslt_initialization::inf:
            return GenerationRecipe::realOnly(
                GenerationRecipe::constant({.value = std::numeric_limits<double>::infinity()}));
        case hipblaslt_initialization::neg_zero:
            return GenerationRecipe::realOnly(GenerationRecipe::constant({.value = -0.0}));
        case hipblaslt_initialization::neg_inf:
            return GenerationRecipe::realOnly(
                GenerationRecipe::constant({.value = -std::numeric_limits<double>::infinity()}));
        case hipblaslt_initialization::nan:
            return GenerationRecipe::realOnly(
                GenerationRecipe::constant({.value = std::numeric_limits<double>::quiet_NaN()}));
        case hipblaslt_initialization::fp16_accumulator_probe:
        case hipblaslt_initialization::norm_dist_one_special:
            throw std::invalid_argument(
                "Requested hipBLASLt initialization requires matrix role and layout information.");
        }
        throw std::invalid_argument("Unsupported hipBLASLt vector initialization mode.");
    }

    template <typename T>
    void initialize(std::span<T>             values,
                    hipblaslt_initialization initialization,
                    TrigonometricComponent   trigonometric = TrigonometricComponent::Cosine)
    {
        initializeTensor(
            values.data(),
            Layout::contiguousLastDimensionFastest(Shape{values.size()}),
            vectorInitializationRecipe(scalarType<T>(), initialization, trigonometric));
    }

    template <typename T>
    void initialize(T*                       data,
                    size_t                   size,
                    hipblaslt_initialization initialization,
                    TrigonometricComponent   trigonometric = TrigonometricComponent::Cosine)
    {
        initialize(std::span<T>(data, size), initialization, trigonometric);
    }

    template <typename T>
    void initializeCosineMatrix(T*        data,
                                size_t    rows,
                                size_t    columns,
                                ptrdiff_t leadingDimension,
                                ptrdiff_t batchStride,
                                size_t    batchCount)
    {
        initializeMatrixBatches(data,
                                rows,
                                columns,
                                leadingDimension,
                                batchStride,
                                batchCount,
                                GenerationRecipe::realOnly(GenerationRecipe::cosine()));
    }
} // namespace hipblaslt::host_validation
