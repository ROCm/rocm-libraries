// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Product-private translation from hipBLASLt initialization modes to
// product-independent host-validation tensor generation recipes.

#include <roc/host_validation/adapters/hipblaslt/HipblasltDataInitialization.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace roc::host_validation::hipblaslt_adapter
{
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

        GenerationOptions sentinelOptions(GenerationPattern pattern, bool complexOutput)
        {
            GenerationOptions options;
            options.real.pattern = pattern;
            if(complexOutput)
                options.imaginary = options.real;
            return options;
        }

        bool supportsExplicitFloatingSentinel(ScalarType type)
        {
            return type == ScalarType::Float16 || type == ScalarType::BFloat16
                   || type == ScalarType::Float32 || type == ScalarType::Float64;
        }

        GenerationOptions matrixGenerationOptions(const MatrixStorageInitialization& initialization,
                                                  ScalarType                         type)
        {
            const bool complexOutput = scalarTypeInfo(type).category == ScalarCategory::Complex;
            if(initialization.forceNaN)
            {
                const ScalarTypeInfo& info = scalarTypeInfo(type);
                if(info.category == ScalarCategory::Scale
                   || (info.category == ScalarCategory::FloatingPoint && !info.supportsNaN))
                    throw std::invalid_argument(
                        "hipBLASLt input type has no supported NaN initialization.");
                return nanOptions(type, true);
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
                return randomIntegerOptions(type, false, alternating);
            }
            case hipblaslt_initialization::trig_float:
            {
                GenerationOptions options;
                options.real.pattern = initialization.role == MatrixRole::B
                                           ? GenerationPattern::Cosine
                                           : GenerationPattern::Sine;
                if(initialization.positiveOnly)
                    options.real.transform = GenerationTransform::Absolute;
                if(complexOutput)
                {
                    options.imaginary.pattern = initialization.role == MatrixRole::B
                                                    ? GenerationPattern::Sine
                                                    : GenerationPattern::Cosine;
                    if(initialization.positiveOnly)
                        options.imaginary.transform = GenerationTransform::Absolute;
                }
                return options;
            }
            case hipblaslt_initialization::hpl:
                return hplOptions(type, initialization.positiveOnly);
            case hipblaslt_initialization::uniform_low_precision:
                return lowPrecisionOptions(type);
            case hipblaslt_initialization::special:
            {
                GenerationOptions options;
                if(initialization.role == MatrixRole::A)
                {
                    options.real.pattern    = GenerationPattern::Constant;
                    options.real.parameter0 = 65280.0;
                }
                else if(initialization.role == MatrixRole::B)
                {
                    options.real.pattern    = GenerationPattern::Constant;
                    options.real.parameter0 = 0.0000607967376708984375;
                }
                else
                {
                    options.real.pattern    = GenerationPattern::UniformInteger;
                    options.real.parameter0 = 1;
                    options.real.parameter1 = 10;
                }
                return options;
            }
            case hipblaslt_initialization::zero:
                return {};
            case hipblaslt_initialization::norm_dist:
            {
                GenerationOptions options;
                options.seed         = 69069;
                options.real.pattern = GenerationPattern::Normal;
                return options;
            }
            case hipblaslt_initialization::norm_dist_one_special:
            {
                if(!supportsExplicitFloatingSentinel(type))
                    throw std::invalid_argument("hipBLASLt one-special normal initialization "
                                                "requires an ordinary floating type.");
                GenerationOptions options;
                options.seed         = 12345;
                options.real.pattern = GenerationPattern::Normal;
                return options;
            }
            case hipblaslt_initialization::uniform_01:
            {
                GenerationOptions options;
                options.seed         = 69069;
                options.real.pattern = type == ScalarType::Int8 ? GenerationPattern::UniformInteger
                                                                : GenerationPattern::UniformReal;
                options.real.parameter0 = 0;
                options.real.parameter1 = 1;
                return options;
            }
            case hipblaslt_initialization::integer_exact:
            {
                GenerationOptions options;
                options.seed            = 69069;
                options.real.pattern    = GenerationPattern::UniformInteger;
                options.real.parameter0 = 0;
                options.real.parameter1 = 2;
                options.real.stream     = initialization.role == MatrixRole::B ? 1000003 : 0;
                if(initialization.role == MatrixRole::B)
                    options.real.alternatingDimensions = {0, 1};
                return options;
            }
            case hipblaslt_initialization::fp16_accumulator_probe:
            {
                GenerationOptions options;
                if(type != ScalarType::Float16)
                    return options;
                if(initialization.role == MatrixRole::A)
                {
                    options.real.pattern    = GenerationPattern::Constant;
                    options.real.parameter0 = 65504.0 - 4.0;
                }
                else if(initialization.role == MatrixRole::B)
                {
                    options.real.pattern               = GenerationPattern::Constant;
                    options.real.parameter0            = 2.0;
                    options.real.alternatingDimensions = {0};
                    options.real.negativeParity        = 1;
                }
                return options;
            }
            case hipblaslt_initialization::inf:
                if(!supportsExplicitFloatingSentinel(type))
                    throw std::invalid_argument(
                        "hipBLASLt infinity initialization requires an ordinary floating type.");
                return sentinelOptions(GenerationPattern::TypeInfinity, complexOutput);
            case hipblaslt_initialization::neg_zero:
                if(!supportsExplicitFloatingSentinel(type))
                    throw std::invalid_argument("hipBLASLt negative-zero initialization requires "
                                                "an ordinary floating type.");
                return sentinelOptions(GenerationPattern::TypeNegativeZero, complexOutput);
            case hipblaslt_initialization::neg_inf:
                if(!supportsExplicitFloatingSentinel(type))
                    throw std::invalid_argument("hipBLASLt negative-infinity initialization "
                                                "requires an ordinary floating type.");
                return sentinelOptions(GenerationPattern::TypeNegativeInfinity, complexOutput);
            case hipblaslt_initialization::nan:
                if(!supportsExplicitFloatingSentinel(type))
                    throw std::invalid_argument(
                        "hipBLASLt NaN initialization requires an ordinary floating type.");
                return nanOptions(type);
            }
            throw std::invalid_argument("Unsupported hipBLASLt host matrix initialization mode.");
        }

        void injectOneSpecial(MutableTensorView view, int requestedSpecialType)
        {
            const size_t logicalElements = view.shape().elementCount();
            if(logicalElements == 0)
                return;

            uint32_t     state              = 12345u * 1103515245u + 12345u;
            const size_t specialLinearIndex = size_t(state) % logicalElements;
            state                           = state * 1103515245u + 12345u;
            const int specialType           = requestedSpecialType >= 0 && requestedSpecialType <= 2
                                                  ? requestedSpecialType
                                                  : int(state >> 16) % 3;

            std::array<size_t, 3> indices{};
            size_t                remainder = specialLinearIndex;
            for(size_t dimension = 0; dimension < indices.size(); ++dimension)
            {
                indices[dimension] = remainder % view.shape()[dimension];
                remainder /= view.shape()[dimension];
            }

            GenerationOptions special;
            special.real.pattern = specialType == 0   ? GenerationPattern::TypeInfinity
                                   : specialType == 1 ? GenerationPattern::TypeNegativeInfinity
                                                      : GenerationPattern::TypeNaN;
            if(scalarTypeInfo(view.type()).category == ScalarCategory::Complex)
                special.imaginary = special.real;

            const ptrdiff_t offset = view.layout().elementOffset(indices);
            generate(MutableTensorView(view.type(), Layout(Shape{1}, {1}, offset), view.storage()),
                     special);
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
        const uint16_t         storageBits = scalarTypeInfo(type).storageBits;
        std::vector<std::byte> storage((elements * static_cast<size_t>(storageBits) + 7) / 8);
        if(initialization.rows == 0 || initialization.columns == 0
           || initialization.batchCount == 0)
            return storage;

        const size_t generatedBatchCount
            = batchStride >= initialization.leadingDimension ? initialization.batchCount : 1;
        Layout layout(Shape{initialization.rows, initialization.columns, generatedBatchCount},
                      {1,
                       static_cast<ptrdiff_t>(initialization.leadingDimension),
                       static_cast<ptrdiff_t>(batchStride)});
        MutableTensorView view(type, layout, storage);
        generate(view, matrixGenerationOptions(initialization, type));

        if(initialization.initialization == hipblaslt_initialization::norm_dist_one_special)
            injectOneSpecial(view, initialization.specialValueType);
        return storage;
    }
} // namespace roc::host_validation::hipblaslt_adapter
