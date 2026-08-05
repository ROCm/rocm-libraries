// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private hipBLASLt adapter.

#include <cmath>
#include <cstddef>
#include <hipblaslt_datatype2string.hpp>
#include <limits>
#include <roc/host_validation/adapters/hipblaslt/Types.hpp>
#include <roc/host_validation/validation.hpp>
#include <span>
#include <utility>

namespace roc::host_validation::hipblaslt_adapter
{
    template <typename T>
    void initializeTensor(T* data, Layout layout, const GenerationOptions& options)
    {
        const size_t elements = storageBytesForLayout(scalarType<T>(), layout) / sizeof(T);
        generate(mutableTensorView(data, elements, std::move(layout)), options);
    }

    template <typename T>
    void initializeMatrixBatches(T*                       data,
                                 size_t                   rows,
                                 size_t                   columns,
                                 ptrdiff_t                leadingDimension,
                                 ptrdiff_t                batchStride,
                                 size_t                   batchCount,
                                 const GenerationOptions& options)
    {
        initializeTensor(
            data,
            Layout(Shape{rows, columns, batchCount}, {1, leadingDimension, batchStride}),
            options);
    }

    template <typename T>
    void initialize(std::span<T>             values,
                    hipblaslt_initialization initialization,
                    DataPattern              trigonometricPattern = DataPattern::Cosine)
    {
        GenerationOptions options;
        options.seed = 69069;

        switch(initialization)
        {
        case hipblaslt_initialization::rand_int:
            options.real = {
                .pattern    = GenerationPattern::UniformInteger,
                .parameter0 = 1,
                .parameter1 = 10,
            };
            options.imaginary        = options.real;
            options.imaginary.stream = 1;
            break;
        case hipblaslt_initialization::trig_float:
            options.real.pattern      = trigonometricPattern == DataPattern::Sine
                                            ? GenerationPattern::Sine
                                            : GenerationPattern::Cosine;
            options.imaginary.pattern = options.real.pattern == GenerationPattern::Sine
                                            ? GenerationPattern::Cosine
                                            : GenerationPattern::Sine;
            break;
        case hipblaslt_initialization::hpl:
            options.real = {
                .pattern    = GenerationPattern::UniformReal,
                .parameter0 = -0.5,
                .parameter1 = 0.5,
            };
            options.imaginary        = options.real;
            options.imaginary.stream = 1;
            break;
        case hipblaslt_initialization::uniform_low_precision:
            options.real = {
                .pattern    = GenerationPattern::UniformReal,
                .parameter0 = -6.0,
                .parameter1 = 6.0,
            };
            options.imaginary        = options.real;
            options.imaginary.stream = 1;
            break;
        case hipblaslt_initialization::special:
            options.real = {
                .pattern    = GenerationPattern::Constant,
                .parameter0 = 65280.0,
            };
            break;
        case hipblaslt_initialization::zero:
            break;
        case hipblaslt_initialization::norm_dist:
            options.real.pattern      = GenerationPattern::Normal;
            options.imaginary.pattern = GenerationPattern::Normal;
            options.imaginary.stream  = 1;
            break;
        case hipblaslt_initialization::uniform_01:
            options.real = {
                .pattern    = GenerationPattern::UniformReal,
                .parameter0 = 0.0,
                .parameter1 = 1.0,
            };
            options.imaginary        = options.real;
            options.imaginary.stream = 1;
            break;
        case hipblaslt_initialization::integer_exact:
            options.real = {
                .pattern    = GenerationPattern::UniformInteger,
                .parameter0 = 0,
                .parameter1 = 2,
            };
            options.imaginary        = options.real;
            options.imaginary.stream = 1;
            break;
        case hipblaslt_initialization::inf:
            options.real = {
                .pattern    = GenerationPattern::Constant,
                .parameter0 = std::numeric_limits<double>::infinity(),
            };
            break;
        case hipblaslt_initialization::neg_zero:
            options.real = {
                .pattern    = GenerationPattern::Constant,
                .parameter0 = -0.0,
            };
            break;
        case hipblaslt_initialization::neg_inf:
            options.real = {
                .pattern    = GenerationPattern::Constant,
                .parameter0 = -std::numeric_limits<double>::infinity(),
            };
            break;
        case hipblaslt_initialization::nan:
            options.real = {
                .pattern    = GenerationPattern::Constant,
                .parameter0 = std::numeric_limits<double>::quiet_NaN(),
            };
            break;
        case hipblaslt_initialization::fp16_accumulator_probe:
        case hipblaslt_initialization::norm_dist_one_special:
            break;
        }

        initializeTensor(values.data(), Layout::contiguous(Shape{values.size()}), options);
    }

    template <typename T>
    void initialize(T*                       data,
                    size_t                   size,
                    hipblaslt_initialization initialization,
                    DataPattern              trigonometricPattern = DataPattern::Cosine)
    {
        initialize(std::span<T>(data, size), initialization, trigonometricPattern);
    }

    template <typename T>
    void initializeCosineMatrix(T*        data,
                                size_t    rows,
                                size_t    columns,
                                ptrdiff_t leadingDimension,
                                ptrdiff_t batchStride,
                                size_t    batchCount)
    {
        GenerationOptions options;
        options.real.pattern = GenerationPattern::Cosine;
        initializeMatrixBatches(
            data, rows, columns, leadingDimension, batchStride, batchCount, options);
    }
} // namespace roc::host_validation::hipblaslt_adapter
