// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private hipBLASLt adapter.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <hipblaslt_datatype2string.hpp>
#include <limits>
#include <roc/host_validation/adapters/hipblaslt/Types.hpp>
#include <roc/host_validation/validation.hpp>
#include <span>
#include <utility>
#include <vector>

namespace roc::host_validation::hipblaslt_adapter
{
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

    inline GenerationOptions randomIntegerOptions(ScalarType type,
                                                  bool       small                = false,
                                                  bool       alternating          = false,
                                                  bool       independentImaginary = true)
    {
        GenerationOptions options;
        options.seed         = 69069;
        options.real.pattern = GenerationPattern::UniformInteger;
        if(small)
        {
            options.real.parameter0 = 1;
            options.real.parameter1 = 10;
            options.real.valueScale = 0.1;
            return options;
        }
        switch(type)
        {
        case ScalarType::Float16:
        case ScalarType::BFloat16:
            options.real.parameter0 = -2;
            options.real.parameter1 = 2;
            break;
        case ScalarType::Int8:
            options.real.parameter0 = 1;
            options.real.parameter1 = 3;
            break;
        case ScalarType::Float4E2M1:
            options.real.parameter0 = -4;
            options.real.parameter1 = 4;
            break;
        case ScalarType::Float6E2M3:
            options.real.parameter0 = -7;
            options.real.parameter1 = 7;
            break;
        case ScalarType::Float6E3M2:
            options.real.parameter0 = -28;
            options.real.parameter1 = 28;
            break;
        case ScalarType::E8M0:
            options.real.pattern    = GenerationPattern::RandomEncodedExponent;
            options.real.parameter0 = -3;
            options.real.parameter1 = 3;
            break;
        default:
            options.real.parameter0 = 1;
            options.real.parameter1 = 10;
            break;
        }
        if(alternating)
            options.real.alternatingDimensions = {0, 1};
        if(independentImaginary)
        {
            options.imaginary        = options.real;
            options.imaginary.stream = 1;
        }
        return options;
    }

    inline GenerationOptions hplOptions(ScalarType type,
                                        bool       positiveOnly         = false,
                                        bool       alternating          = false,
                                        bool       independentImaginary = true)
    {
        GenerationOptions options;
        options.seed = 69069;
        if(type == ScalarType::E8M0)
        {
            options.real.pattern    = GenerationPattern::RandomEncodedExponent;
            options.real.parameter0 = -3;
            options.real.parameter1 = 3;
        }
        else if(type == ScalarType::Int8)
        {
            options.real.pattern    = GenerationPattern::UniformInteger;
            options.real.parameter0 = positiveOnly ? 0 : -1;
            options.real.parameter1 = 1;
        }
        else
        {
            options.real.pattern    = GenerationPattern::UniformReal;
            options.real.parameter0 = -0.5;
            options.real.parameter1 = 0.5;
            if(positiveOnly || type == ScalarType::E5M3)
                options.real.transform = GenerationTransform::Absolute;
        }
        if(alternating)
            options.real.alternatingDimensions = {0, 1};
        if(independentImaginary)
        {
            options.imaginary        = options.real;
            options.imaginary.stream = 1;
        }
        return options;
    }

    inline GenerationOptions lowPrecisionOptions(ScalarType type)
    {
        GenerationOptions options;
        options.seed = 69069;
        if(type == ScalarType::E8M0)
        {
            options.real.pattern    = GenerationPattern::RandomEncodedExponent;
            options.real.parameter0 = -3;
            options.real.parameter1 = 3;
        }
        else
        {
            options.real.pattern    = type == ScalarType::Int8 ? GenerationPattern::UniformInteger
                                                               : GenerationPattern::UniformReal;
            options.real.parameter0 = -6;
            options.real.parameter1 = 6;
            if(type == ScalarType::E5M3)
                options.real.transform = GenerationTransform::Absolute;
        }
        return options;
    }

    inline GenerationOptions nanOptions(ScalarType type, bool independentImaginary = false)
    {
        GenerationOptions options;
        options.seed         = 69069;
        options.real.pattern = scalarTypeInfo(type).supportsNaN ? GenerationPattern::TypeNaN
                                                                : GenerationPattern::RandomRawBits;
        if(independentImaginary && scalarTypeInfo(type).category == ScalarCategory::Complex)
            options.imaginary = options.real;
        return options;
    }

    template <typename T>
    void initializeTensor(T* data, Layout layout, const GenerationOptions& options)
    {
        const size_t elements = storageBytesForLayout(scalarType<T>(), layout) / sizeof(T);
        generate(mutableTensorView(data, elements, std::move(layout)), options);
    }

    inline void initializeTensor(void*                    data,
                                 hipDataType              type,
                                 Layout                   layout,
                                 const GenerationOptions& options)
    {
        const size_t bytes = storageBytesForLayout(scalarType(type), layout);
        generate(mutableTensorView(data, bytes, type, std::move(layout)), options);
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

    inline void initializeMatrixBatches(void*                    data,
                                        hipDataType              type,
                                        size_t                   rows,
                                        size_t                   columns,
                                        ptrdiff_t                leadingDimension,
                                        ptrdiff_t                batchStride,
                                        size_t                   batchCount,
                                        const GenerationOptions& options)
    {
        initializeTensor(
            data,
            type,
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
