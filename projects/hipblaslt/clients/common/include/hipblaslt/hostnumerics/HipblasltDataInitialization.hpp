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

#include <TensileLite/Client/hostnumerics/MxScaleStorageLayout.hpp>
#include <roc/hostnumerics/amd_gpu_layout/mx.hpp>
#include <roc/hostnumerics/mx.hpp>
#include <vector>

namespace hipblaslt::hostnumerics
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

    enum class IntegerExactPattern
    {
        Standard,
        Ternary,
        SparseK,
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
    };

    // Preserve the seed used by the former mxDataGenerator implementation.
    // Callers derive explicit operand and batch seeds from this compatibility
    // base; generateMxData does not advance or hide seed state.
    inline constexpr uint64_t mxDefaultSeed = 1'713'573'849U;

    // integer_exact MX elements are integers of magnitude at most
    // integerExactMxMaxElement, and their E8M0 scales at most integerExactMxMaxScale,
    // so each product of a dequantized A and B element is an integer of magnitude at
    // most integerExactMxMaxProduct. K of them bound every partial sum of D, in any
    // summation order. FP32 holds every integer below 2^24 exactly, so the device and
    // the host reference agree exactly while K * integerExactMxMaxProduct < 2^24, that
    // is K < 262144.
    inline constexpr int     integerExactMxMaxElement = 2;
    inline constexpr int     integerExactMxMaxScale   = 4;
    inline constexpr int64_t integerExactMxMaxProduct
        = int64_t(integerExactMxMaxElement) * integerExactMxMaxScale * integerExactMxMaxElement
          * integerExactMxMaxScale;
    inline constexpr int64_t integerExactMxExactLimit = int64_t(1) << 24;

    constexpr int64_t integerExactMxSumBound(int64_t k)
    {
        return k * integerExactMxMaxProduct;
    }

    // integer_exact accepts FP8 (E4M3, E5M2) and FP6 (E2M3, E3M2) A and B with E8M0
    // scales: A elements in {0, 1, 2}, B elements in {-2, ..., 2}, scales in {1, 2, 4}.
    ::roc::hostnumerics::MxTensor generateMxData(hipDataType                dataType,
                                                 hipDataType                scaleType,
                                                 ::roc::hostnumerics::Shape shape,
                                                 uint64_t                   leadingDimension,
                                                 size_t                     blockAxis,
                                                 size_t                     blockSize,
                                                 hipblaslt_initialization   initialization,
                                                 uint64_t                   seed,
                                                 MatrixRole                 role,
                                                 size_t                     batchIndex = 0);

    MxScaleStorageLayout mxScaleStorageLayoutForArchName(std::string_view archName);

    MxScaleStorageLayout mxScaleStorageLayoutForFormat(hipblaslt_scaling_format scalingFormat,
                                                       std::string_view         archName);

    // Returns the scale bytes a kernel reads for generated MX data. blockAxis is the
    // data axis that generateMxData blocked. Tile32x8 and Tile128x4 take the natural
    // scales as [MN, K blocks], so scales blocked along axis 1 are transposed first.
    // The plan's natural shape must match the oriented scales. Tile128x4 applies the
    // plan, which must be natural, and then shuffles the bytes into 128-row by
    // 4-K-block tiles; it requires MN % 128 == 0 and K blocks % 4 == 0.
    std::vector<std::byte>
        mxScaleBytesForDevice(const ::roc::hostnumerics::MxTensor& generated,
                              size_t                               blockAxis,
                              const ::roc::hostnumerics::amd_gpu_layout::MxScaleStoragePlan& plan,
                              MxScaleStorageLayout layout);

    // Translates a product-level initialization mode into a tensor recipe.
    // The caller supplies the complete seed; no implicit stream is added.
    ::roc::hostnumerics::GenerationRecipe
        initializationRecipe(::roc::hostnumerics::ScalarType type,
                             hipblaslt_initialization        initialization,
                             uint64_t                        seed,
                             TrigonometricComponent          trigonometric);

    // Preserves the grouped-GEMM client's historical operand patterns while
    // leaving tensor allocation and seed sequencing with the caller.
    ::roc::hostnumerics::GenerationRecipe
        groupedGemmInitializationRecipe(::roc::hostnumerics::ScalarType type,
                                        hipblaslt_initialization        mode,
                                        initialization::OperandSequence operand,
                                        uint64_t                        seed);

    // Applies the hipBLASLt initialization policy directly to an existing
    // Tensor. The caller owns storage and supplies the exact seed used by the
    // selected recipe; MatrixRole only selects operand-specific value rules.
    struct MatrixInitializationOptions
    {
        // Fill every element with NaN instead of the selected values.
        bool                           forceNaN = false;
        std::optional<OneSpecialValue> oneSpecialValue;
        bool                           positiveOnly = false;
        IntegerExactOptions            integerExact;
        // The batch index of the tensor's first batch, which MAF seeds use.
        size_t batchIndex = 0;
    };

    void initializeMatrix(::roc::hostnumerics::Tensor        destination,
                          MatrixRole                         role,
                          hipblaslt_initialization           initialization,
                          uint64_t                           seed,
                          const MatrixInitializationOptions& options = {});

} // namespace hipblaslt::hostnumerics
