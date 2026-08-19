// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private hipBLASLt adapter.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <hipblaslt/host_validation/GenerationRecipes.hpp>
#include <hipblaslt/host_validation/Types.hpp>
#include <hipblaslt_datatype2string.hpp>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace hipblaslt::host_validation
{
    using namespace ::roc::host_validation;

    enum class MatrixRole
    {
        A,
        B,
        C,
    };

    struct MatrixStorageInitialization
    {
        MatrixRole               role             = MatrixRole::A;
        hipblaslt_initialization initialization   = hipblaslt_initialization::zero;
        bool                     forceNaN         = false;
        hipDataType              type             = HIP_R_32F;
        size_t                   rows             = 0;
        size_t                   columns          = 0;
        size_t                   leadingDimension = 0;
        size_t                   batchStride      = 0;
        size_t                   batchCount       = 1;
        int                      specialValueType = -1;
        bool                     positiveOnly     = false;
    };

    std::vector<std::byte> generateMatrixStorage(const MatrixStorageInitialization& initialization);

    template <typename T>
    void initializeTensor(T* data, Layout layout, const GenerationRecipe& recipe)
    {
        const size_t elements = storageBytesForLayout(scalarType<T>(), layout) / sizeof(T);
        auto         tensor   = tensorFromMutableStorage(data, elements, std::move(layout));
        generate(tensor, recipe);
        if(!tensor.storage().empty())
            std::memcpy(data, tensor.storage().data(), tensor.storage().size());
    }

    inline void
        initializeTensor(void* data, ScalarType type, Layout layout, const GenerationRecipe& recipe)
    {
        auto tensor = tensorFromMutableStorage(data, type, std::move(layout));
        generate(tensor, recipe);
        if(!tensor.storage().empty())
            std::memcpy(data, tensor.storage().data(), tensor.storage().size());
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
            return GenerationRecipe::realOnly(GenerationRecipe::zero());
        }
        return GenerationRecipe::realOnly(GenerationRecipe::zero());
    }

    template <typename T>
    void initialize(std::span<T>             values,
                    hipblaslt_initialization initialization,
                    TrigonometricComponent   trigonometric = TrigonometricComponent::Cosine)
    {
        initializeTensor(
            values.data(),
            Layout::contiguous(Shape{values.size()}),
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
