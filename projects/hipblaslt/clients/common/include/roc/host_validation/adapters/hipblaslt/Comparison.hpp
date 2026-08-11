// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private hipBLASLt descriptor adapter. Numerical comparison is owned
// by roc::host-validation.

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <roc/host_validation/adapters/hipblaslt/Types.hpp>
#include <roc/host_validation/comparison.hpp>
#include <roc/host_validation/typed_comparison.hpp>
#include <span>
#include <stdexcept>
#include <vector>

namespace roc::host_validation::hipblaslt_adapter
{
    inline Layout comparisonLayout(int64_t rows,
                                   int64_t columns,
                                   int64_t leadingDimension,
                                   int64_t batchStride,
                                   int64_t batchCount)
    {
        if(rows < 0 || columns < 0 || leadingDimension < 0 || batchStride < 0 || batchCount < 0)
            throw std::invalid_argument("hipBLASLt comparison dimensions must be non-negative.");
        return Layout(
            Shape{static_cast<size_t>(rows),
                  static_cast<size_t>(columns),
                  static_cast<size_t>(batchCount)},
            {1, static_cast<ptrdiff_t>(leadingDimension), static_cast<ptrdiff_t>(batchStride)});
    }

    inline TensorView comparisonView(const void* data, hipDataType type, const Layout& layout)
    {
        const ScalarType scalar       = scalarType(type);
        const size_t     storageBytes = storageBytesForLayout(scalar, layout);
        if(data == nullptr && storageBytes != 0)
            throw std::invalid_argument("hipBLASLt comparison buffer is null.");
        return TensorView(
            scalar,
            layout,
            std::span<const std::byte>(static_cast<const std::byte*>(data), storageBytes));
    }

    template <typename T>
    ComparisonResult compareTypedBuffers(const void*       expected,
                                         const void*       observed,
                                         const Layout&     layout,
                                         ComparisonOptions options)
    {
        constexpr ScalarType scalar = scalarType<T>();
        static_assert(scalarTypeInfo(scalar).storageBits == sizeof(T) * 8,
                      "Typed hipBLASLt comparison requires native scalar storage.");
        const size_t storageElements = storageBytesForLayout(scalar, layout) / sizeof(T);
        options.selection.indexOrder = ComparisonIndexOrder::FirstDimensionFastest;
        return compare(
            TypedTensorView<T>(
                layout, std::span<const T>(static_cast<const T*>(observed), storageElements)),
            TypedTensorView<T>(
                layout, std::span<const T>(static_cast<const T*>(expected), storageElements)),
            options);
    }

    inline ComparisonResult compareBuffers(int64_t           rows,
                                           int64_t           columns,
                                           int64_t           leadingDimension,
                                           int64_t           batchStride,
                                           const void*       expected,
                                           const void*       observed,
                                           int64_t           batchCount,
                                           hipDataType       type,
                                           ComparisonOptions options)
    {
        const Layout layout
            = comparisonLayout(rows, columns, leadingDimension, batchStride, batchCount);
        switch(type)
        {
        case HIP_R_32F:
            return compareTypedBuffers<float>(expected, observed, layout, options);
        case HIP_R_64F:
            return compareTypedBuffers<double>(expected, observed, layout, options);
        case HIP_C_32F:
            return compareTypedBuffers<std::complex<float>>(expected, observed, layout, options);
        case HIP_C_64F:
            return compareTypedBuffers<std::complex<double>>(expected, observed, layout, options);
        case HIP_R_16F:
            return compareTypedBuffers<hipblasLtHalf>(expected, observed, layout, options);
        case HIP_R_16BF:
            return compareTypedBuffers<hip_bfloat16>(expected, observed, layout, options);
        case HIP_R_8F_E4M3_FNUZ:
            return compareTypedBuffers<hipblaslt_f8_fnuz>(expected, observed, layout, options);
        case HIP_R_8F_E5M2_FNUZ:
            return compareTypedBuffers<hipblaslt_bf8_fnuz>(expected, observed, layout, options);
        case HIP_R_8F_E4M3:
            return compareTypedBuffers<hipblaslt_f8>(expected, observed, layout, options);
        case HIP_R_8F_E5M2:
            return compareTypedBuffers<hipblaslt_bf8>(expected, observed, layout, options);
        case HIP_R_32I:
            return compareTypedBuffers<int32_t>(expected, observed, layout, options);
        case HIP_R_8I:
            return compareTypedBuffers<hipblasLtInt8>(expected, observed, layout, options);
        default:
            options.selection.indexOrder = ComparisonIndexOrder::FirstDimensionFastest;
            return compare(comparisonView(observed, type, layout),
                           comparisonView(expected, type, layout),
                           options);
        }
    }
} // namespace roc::host_validation::hipblaslt_adapter
