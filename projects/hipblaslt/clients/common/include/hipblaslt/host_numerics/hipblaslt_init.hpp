// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private hipBLASLt adapter over host-numerics-owned initialization.

#include "hipblaslt_datatype2string.hpp"

#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/host_numerics/HipblasltDataInitialization.hpp>

#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

enum class ABC_dims
{
    A,
    B,
    C
};

void hipblaslt_init_device(ABC_dims                 ABC_dims,
                           hipblaslt_initialization init,
                           bool                     is_nan,
                           void*                    A,
                           size_t                   M,
                           size_t                   N,
                           size_t                   lda,
                           hipDataType              type,
                           size_t                   stride,
                           size_t                   batch_count,
                           bool                     positiveOnly = false,
                           std::optional<hipblaslt::host_numerics::OneSpecialValue> oneSpecialValue
                           = std::nullopt);

namespace hipblaslt::host_numerics::detail
{
    enum class RuntimeInitialization : uint8_t
    {
        General = 1U << 0,
        Random  = 1U << 1,
        Small   = 1U << 2,
    };

    inline constexpr uint8_t runtimeInitializationCapabilities(ScalarType type)
    {
        constexpr uint8_t general = static_cast<uint8_t>(RuntimeInitialization::General);
        constexpr uint8_t random  = static_cast<uint8_t>(RuntimeInitialization::Random);
        constexpr uint8_t small   = static_cast<uint8_t>(RuntimeInitialization::Small);

        switch(type)
        {
        case ScalarType::Float32:
        case ScalarType::Float64:
        case ScalarType::Float16:
        case ScalarType::Int32:
        case ScalarType::ComplexFloat32:
        case ScalarType::ComplexFloat64:
            return general | random | small;
        case ScalarType::BFloat16:
        case ScalarType::Float8E4M3:
        case ScalarType::Float8E5M2:
        case ScalarType::Float8E4M3Fnuz:
        case ScalarType::Float8E5M2Fnuz:
        case ScalarType::Int8:
            return general | random;
        case ScalarType::E8M0:
            return random;
        default:
            return 0;
        }
    }

    inline constexpr bool supportsRuntimeInitialization(ScalarType            type,
                                                        RuntimeInitialization required)
    {
        return (runtimeInitializationCapabilities(type) & static_cast<uint8_t>(required)) != 0;
    }

    [[noreturn]] inline void
        throwUnsupportedRuntimeInitialization(std::string_view                 functionName,
                                              const std::optional<ScalarType>& type,
                                              bool                             identifyPackedType)
    {
        std::string message(functionName);
        if(identifyPackedType && type)
        {
            switch(*type)
            {
            case ScalarType::Float6E2M3:
                throw std::invalid_argument(message + " does not support FP6.");
            case ScalarType::Float6E3M2:
                throw std::invalid_argument(message + " does not support BF6.");
            case ScalarType::Float4E2M1:
                throw std::invalid_argument(message + " does not support FP4.");
            default:
                break;
            }
        }
        throw std::invalid_argument(message + " does not support the requested data type.");
    }

    template <typename RecipeFactory>
    inline void initializeRuntimeTensor(void*                 data,
                                        hipDataType           runtimeType,
                                        Layout                layout,
                                        RuntimeInitialization required,
                                        std::string_view      functionName,
                                        bool                  identifyPackedType,
                                        RecipeFactory&&       recipeFactory)
    {
        const std::optional<ScalarType> type = tryScalarType(runtimeType);
        if(!type || !supportsRuntimeInitialization(*type, required))
            throwUnsupportedRuntimeInitialization(functionName, type, identifyPackedType);

        GenerationRecipe recipe = std::forward<RecipeFactory>(recipeFactory)(*type);
        initializeTensor(data, *type, std::move(layout), recipe);
    }

    inline Layout matrixBatchLayout(
        size_t rows, size_t columns, size_t leadingDimension, size_t batchStride, size_t batchCount)
    {
        return Layout(
            Shape{rows, columns, batchCount},
            {1, static_cast<ptrdiff_t>(leadingDimension), static_cast<ptrdiff_t>(batchStride)});
    }
} // namespace hipblaslt::host_numerics::detail

template <typename T>
inline void
    hipblaslt_init(T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    const auto recipe
        = hipblaslt::host_numerics::realOnlyRandomRecipe(hipblaslt::host_numerics::scalarType<T>());
    hipblaslt::host_numerics::initializeMatrixBatches(A, M, N, lda, stride, batch_count, recipe);
}

inline void hipblaslt_init(void*       A,
                           size_t      M,
                           size_t      N,
                           size_t      lda,
                           hipDataType type,
                           size_t      stride      = 0,
                           size_t      batch_count = 1)
{
    hipblaslt::host_numerics::detail::initializeRuntimeTensor(
        A,
        type,
        hipblaslt::host_numerics::detail::matrixBatchLayout(M, N, lda, stride, batch_count),
        hipblaslt::host_numerics::detail::RuntimeInitialization::Random,
        "hipblaslt_init",
        true,
        [](roc::host_numerics::ScalarType scalar) {
            return hipblaslt::host_numerics::realOnlyRandomRecipe(scalar);
        });
}

inline void hipblaslt_init_small(void*       A,
                                 size_t      M,
                                 size_t      N,
                                 size_t      lda,
                                 hipDataType type,
                                 size_t      stride      = 0,
                                 size_t      batch_count = 1)
{
    hipblaslt::host_numerics::detail::initializeRuntimeTensor(
        A,
        type,
        hipblaslt::host_numerics::detail::matrixBatchLayout(M, N, lda, stride, batch_count),
        hipblaslt::host_numerics::detail::RuntimeInitialization::Small,
        "hipblaslt_init_small",
        false,
        [](roc::host_numerics::ScalarType scalar) {
            return hipblaslt::host_numerics::randomIntegerRecipe(
                scalar,
                {.small         = true,
                 .complexPolicy = hipblaslt::host_numerics::ComplexGenerationPolicy::RealOnly});
        });
}

template <typename T>
inline void hipblaslt_init_nan(T* values, size_t elements)
{
    const auto recipe
        = hipblaslt::host_numerics::nanRecipe(hipblaslt::host_numerics::scalarType<T>());
    hipblaslt::host_numerics::initializeTensor(
        values,
        roc::host_numerics::Layout::contiguousLastDimensionFastest(
            roc::host_numerics::Shape{elements}),
        recipe);
}

inline void hipblaslt_init_zero(void*       data,
                                size_t      rows,
                                size_t      columns,
                                size_t      leadingDimension,
                                hipDataType type,
                                size_t      batchStride = 0,
                                size_t      batchCount  = 1)
{
    hipblaslt::host_numerics::detail::initializeRuntimeTensor(
        data,
        type,
        hipblaslt::host_numerics::detail::matrixBatchLayout(
            rows, columns, leadingDimension, batchStride, batchCount),
        hipblaslt::host_numerics::detail::RuntimeInitialization::General,
        "hipblaslt_init_zero",
        false,
        [](roc::host_numerics::ScalarType) {
            return roc::host_numerics::GenerationRecipe::realOnly(
                roc::host_numerics::GenerationRecipe::zero());
        });
}
