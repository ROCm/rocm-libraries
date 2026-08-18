// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private hipBLASLt adapter.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <hipblaslt/host_validation/Types.hpp>
#include <hipblaslt_datatype2string.hpp>
#include <limits>
#include <roc/host_validation/generation.hpp>
#include <span>
#include <utility>
#include <vector>

namespace hipblaslt::host_validation
{
    using namespace ::roc::host_validation;

    inline constexpr uint64_t defaultInitializationSeed    = 69069;
    inline constexpr uint64_t oneSpecialInitializationSeed = 12345;
    inline constexpr uint64_t independentImaginaryStream   = 1;
    inline constexpr uint64_t integerExactBStream          = 1000003;
    inline constexpr double   specialInitializationAValue  = 65280.0;
    inline constexpr double   specialInitializationBValue  = 0.0000607967376708984375;
    inline constexpr double   maximumFiniteFloat16Value    = 65504.0;

    inline GenerationOptions makeGenerationOptions(GenerationPattern pattern,
                                                   double            parameter0 = 0.0,
                                                   double            parameter1 = 1.0,
                                                   uint64_t seed = defaultInitializationSeed)
    {
        GenerationOptions options;
        options.seed            = seed;
        options.real.pattern    = pattern;
        options.real.parameter0 = parameter0;
        options.real.parameter1 = parameter1;
        return options;
    }

    inline GenerationOptions withIndependentImaginary(GenerationOptions options)
    {
        options.imaginary        = options.real;
        options.imaginary.stream = independentImaginaryStream;
        return options;
    }

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
        options.seed         = defaultInitializationSeed;
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
            options = withIndependentImaginary(std::move(options));
        return options;
    }

    inline GenerationOptions legacyRandomOptions(ScalarType type)
    {
        if(type == ScalarType::E8M0)
        {
            GenerationOptions options;
            options.real.pattern    = GenerationPattern::UniformRawInteger;
            options.real.parameter0 = 1;
            options.real.parameter1 = 10;
            return options;
        }
        return randomIntegerOptions(type, false, false, false);
    }

    inline GenerationOptions sineOptions(ScalarType type)
    {
        GenerationOptions options;
        options.real.pattern = GenerationPattern::Sine;
        if(scalarTypeInfo(type).category == ScalarCategory::Complex)
            options.imaginary.pattern = GenerationPattern::Cosine;
        return options;
    }

    inline GenerationOptions hplOptions(ScalarType type,
                                        bool       positiveOnly         = false,
                                        bool       alternating          = false,
                                        bool       independentImaginary = true)
    {
        GenerationOptions options;
        options.seed = defaultInitializationSeed;
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
            options = withIndependentImaginary(std::move(options));
        return options;
    }

    inline GenerationOptions lowPrecisionOptions(ScalarType type)
    {
        GenerationOptions options;
        options.seed = defaultInitializationSeed;
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
        options.seed         = defaultInitializationSeed;
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
        auto         tensor   = tensorFromMutableStorage(data, elements, std::move(layout));
        generate(tensor, options);
        if(!tensor.storage().empty())
            std::memcpy(data, tensor.storage().data(), tensor.storage().size());
    }

    inline void initializeTensor(void*                    data,
                                 ScalarType               type,
                                 Layout                   layout,
                                 const GenerationOptions& options)
    {
        auto tensor = tensorFromMutableStorage(data, type, std::move(layout));
        generate(tensor, options);
        if(!tensor.storage().empty())
            std::memcpy(data, tensor.storage().data(), tensor.storage().size());
    }

    inline void initializeTensor(void*                    data,
                                 hipDataType              type,
                                 Layout                   layout,
                                 const GenerationOptions& options)
    {
        initializeTensor(data, scalarType(type), std::move(layout), options);
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
                                        ScalarType               type,
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

    inline void initializeMatrixBatches(void*                    data,
                                        hipDataType              type,
                                        size_t                   rows,
                                        size_t                   columns,
                                        ptrdiff_t                leadingDimension,
                                        ptrdiff_t                batchStride,
                                        size_t                   batchCount,
                                        const GenerationOptions& options)
    {
        initializeMatrixBatches(data,
                                scalarType(type),
                                rows,
                                columns,
                                leadingDimension,
                                batchStride,
                                batchCount,
                                options);
    }

    inline GenerationOptions vectorInitializationOptions(hipblaslt_initialization initialization,
                                                         GenerationPattern trigonometricPattern)
    {
        switch(initialization)
        {
        case hipblaslt_initialization::rand_int:
            return withIndependentImaginary(
                makeGenerationOptions(GenerationPattern::UniformInteger, 1, 10));
        case hipblaslt_initialization::trig_float:
        {
            GenerationOptions options;
            options.seed              = defaultInitializationSeed;
            options.real.pattern      = trigonometricPattern == GenerationPattern::Sine
                                            ? GenerationPattern::Sine
                                            : GenerationPattern::Cosine;
            options.imaginary.pattern = options.real.pattern == GenerationPattern::Sine
                                            ? GenerationPattern::Cosine
                                            : GenerationPattern::Sine;
            return options;
        }
        case hipblaslt_initialization::hpl:
            return withIndependentImaginary(
                makeGenerationOptions(GenerationPattern::UniformReal, -0.5, 0.5));
        case hipblaslt_initialization::uniform_low_precision:
            return withIndependentImaginary(
                makeGenerationOptions(GenerationPattern::UniformReal, -6.0, 6.0));
        case hipblaslt_initialization::special:
            return makeGenerationOptions(GenerationPattern::Constant, specialInitializationAValue);
        case hipblaslt_initialization::zero:
            return makeGenerationOptions(GenerationPattern::Zero);
        case hipblaslt_initialization::norm_dist:
            return withIndependentImaginary(makeGenerationOptions(GenerationPattern::Normal));
        case hipblaslt_initialization::uniform_01:
            return withIndependentImaginary(
                makeGenerationOptions(GenerationPattern::UniformReal, 0.0, 1.0));
        case hipblaslt_initialization::integer_exact:
            return withIndependentImaginary(
                makeGenerationOptions(GenerationPattern::UniformInteger, 0, 2));
        case hipblaslt_initialization::inf:
            return makeGenerationOptions(GenerationPattern::Constant,
                                         std::numeric_limits<double>::infinity());
        case hipblaslt_initialization::neg_zero:
            return makeGenerationOptions(GenerationPattern::Constant, -0.0);
        case hipblaslt_initialization::neg_inf:
            return makeGenerationOptions(GenerationPattern::Constant,
                                         -std::numeric_limits<double>::infinity());
        case hipblaslt_initialization::nan:
            return makeGenerationOptions(GenerationPattern::Constant,
                                         std::numeric_limits<double>::quiet_NaN());
        case hipblaslt_initialization::fp16_accumulator_probe:
        case hipblaslt_initialization::norm_dist_one_special:
            return makeGenerationOptions(GenerationPattern::Zero);
        }
        return makeGenerationOptions(GenerationPattern::Zero);
    }

    template <typename T>
    void initialize(std::span<T>             values,
                    hipblaslt_initialization initialization,
                    GenerationPattern        trigonometricPattern = GenerationPattern::Cosine)
    {
        initializeTensor(values.data(),
                         Layout::contiguous(Shape{values.size()}),
                         vectorInitializationOptions(initialization, trigonometricPattern));
    }

    template <typename T>
    void initialize(T*                       data,
                    size_t                   size,
                    hipblaslt_initialization initialization,
                    GenerationPattern        trigonometricPattern = GenerationPattern::Cosine)
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
} // namespace hipblaslt::host_validation
