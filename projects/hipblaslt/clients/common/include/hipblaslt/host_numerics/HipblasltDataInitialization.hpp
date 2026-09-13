// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private translation from hipBLASLt initialization modes, data types,
// and architecture selection to host-numerics recipes and tensors. Storage,
// random generation, and physical-layout loops remain in host-numerics.

#include <cstddef>
#include <cstdint>
#include <hipblaslt/host_numerics/InitializationPolicy.hpp>
#include <hipblaslt/host_numerics/Types.hpp>
#include <hipblaslt_arguments.hpp>
#include <hipblaslt_scaling_format.hpp>
#include <optional>
#include <roc/host_numerics/generation.hpp>
#include <string_view>

#include <roc/host_numerics/amd_gpu_layout/mx.hpp>
#include <roc/host_numerics/mx.hpp>

namespace hipblaslt::host_numerics
{
    enum class TrigonometricComponent
    {
        Sine,
        Cosine,
    };

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

    // Preserve the seed used by the former mxDataGenerator implementation.
    // Callers derive explicit operand and batch seeds from this compatibility
    // base; generateMxData does not advance or hide seed state.
    inline constexpr uint64_t mxDefaultSeed = 1'713'573'849U;

    ::roc::host_numerics::MxTensor generateMxData(hipDataType                 dataType,
                                                  hipDataType                 scaleType,
                                                  ::roc::host_numerics::Shape shape,
                                                  uint64_t                    leadingDimension,
                                                  size_t                      blockAxis,
                                                  size_t                      blockSize,
                                                  hipblaslt_initialization    initialization,
                                                  uint64_t                    seed);

    ::roc::host_numerics::amd_gpu_layout::MxScaleStorageLayout
        mxScaleStorageLayoutForArchName(std::string_view archName);

    ::roc::host_numerics::amd_gpu_layout::MxScaleStorageLayout
        mxScaleStorageLayoutForFormat(hipblaslt_scaling_format scalingFormat,
                                      std::string_view         archName);

    // Translates a product-level initialization mode into a tensor recipe.
    // The caller supplies the complete seed; no implicit stream is added.
    ::roc::host_numerics::GenerationRecipe
        initializationRecipe(::roc::host_numerics::ScalarType type,
                             hipblaslt_initialization         initialization,
                             uint64_t                         seed,
                             TrigonometricComponent trigonometric = TrigonometricComponent::Cosine);

    // Preserves the grouped-GEMM client's historical operand patterns while
    // leaving tensor allocation and seed sequencing with the caller.
    ::roc::host_numerics::GenerationRecipe
        groupedGemmInitializationRecipe(::roc::host_numerics::ScalarType type,
                                        hipblaslt_initialization         mode,
                                        initialization::OperandSequence  operand,
                                        uint64_t                         seed);

    // Applies the hipBLASLt initialization policy directly to an existing
    // Tensor. The caller owns storage and supplies the exact seed used by the
    // selected recipe; MatrixRole only selects operand-specific value rules.
    void initializeMatrix(::roc::host_numerics::Tensor   destination,
                          MatrixRole                     role,
                          hipblaslt_initialization       initialization,
                          uint64_t                       seed,
                          bool                           forceNaN        = false,
                          std::optional<OneSpecialValue> oneSpecialValue = std::nullopt,
                          bool                           positiveOnly    = false);

} // namespace hipblaslt::host_numerics
