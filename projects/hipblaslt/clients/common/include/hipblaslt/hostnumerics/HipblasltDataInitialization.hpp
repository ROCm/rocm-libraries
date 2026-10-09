// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private translation from hipBLASLt initialization modes, data types,
// and architecture selection to HostNumerics recipes and tensors. Storage,
// random generation, and physical-layout loops remain in hostnumerics.

#include <cstddef>
#include <cstdint>
#include <hipblaslt/hostnumerics/InitializationPolicy.hpp>
#include <hipblaslt/hostnumerics/Types.hpp>
#include <hipblaslt_arguments.hpp>
#include <hipblaslt_scaling_format.hpp>
#include <optional>
#include <roc/hostnumerics/generation.hpp>
#include <string_view>

#include <roc/hostnumerics/amd_gpu_layout/mx.hpp>
#include <roc/hostnumerics/mx.hpp>

namespace hipblaslt::hostnumerics
{
    enum class TrigonometricComponent
    {
        Sine,
        Cosine,
        Count,
    };

    enum class MatrixRole
    {
        A,
        B,
        C,
        Count,
    };

    enum class IntegerExactPattern
    {
        Standard,
        Ternary,
        SparseK,
        Count,
    };

    bool parseIntegerExactPattern(std::string_view name, IntegerExactPattern& pattern);

    struct IntegerExactOptions
    {
        IntegerExactPattern pattern = IntegerExactPattern::Standard;
        // Stored matrix dimension containing K, before applying op(A).
        // SparseK restricts A to at most 17 nonzero terms per logical row.
        size_t reductionAxis = 1;
    };

    enum class OneSpecialValue : uint8_t
    {
        PositiveInfinity,
        NegativeInfinity,
        NaN,
        Count,
    };

    // Preserve the seed used by the former mxDataGenerator implementation.
    // Callers derive explicit operand and batch seeds from this compatibility
    // base; generateMxData does not advance or hide seed state.
    inline constexpr uint64_t mxDefaultSeed = 1'713'573'849U;

    ::roc::hostnumerics::MxTensor generateMxData(hipDataType                dataType,
                                                 hipDataType                scaleType,
                                                 ::roc::hostnumerics::Shape shape,
                                                 uint64_t                   leadingDimension,
                                                 size_t                     blockAxis,
                                                 size_t                     blockSize,
                                                 hipblaslt_initialization   initialization,
                                                 uint64_t                   seed,
                                                 MatrixRole                 role);

    ::roc::hostnumerics::amd_gpu_layout::MxScaleStorageLayout
        mxScaleStorageLayoutForArchName(std::string_view archName);

    ::roc::hostnumerics::amd_gpu_layout::MxScaleStorageLayout
        mxScaleStorageLayoutForFormat(hipblaslt_scaling_format scalingFormat,
                                      std::string_view         archName);

    // Translates a product-level initialization mode into a tensor recipe.
    // The caller supplies the complete seed; no implicit stream is added.
    ::roc::hostnumerics::GenerationRecipe
        initializationRecipe(::roc::hostnumerics::ScalarType type,
                             hipblaslt_initialization         initialization,
                             uint64_t                         seed,
                             TrigonometricComponent trigonometric = TrigonometricComponent::Cosine);

    // Preserves the grouped-GEMM client's historical operand patterns while
    // leaving tensor allocation and seed sequencing with the caller.
    ::roc::hostnumerics::GenerationRecipe
        groupedGemmInitializationRecipe(::roc::hostnumerics::ScalarType type,
                                        hipblaslt_initialization         mode,
                                        initialization::OperandSequence  operand,
                                        uint64_t                         seed);

    // Applies the hipBLASLt initialization policy directly to an existing
    // Tensor. The caller owns storage and supplies the exact seed used by the
    // selected recipe; MatrixRole only selects operand-specific value rules.
    void initializeMatrix(::roc::hostnumerics::Tensor    destination,
                          MatrixRole                     role,
                          hipblaslt_initialization       initialization,
                          uint64_t                       seed,
                          bool                           forceNaN        = false,
                          std::optional<OneSpecialValue> oneSpecialValue = std::nullopt,
                          bool                           positiveOnly    = false,
                          IntegerExactOptions            integerExact    = {});

} // namespace hipblaslt::hostnumerics
