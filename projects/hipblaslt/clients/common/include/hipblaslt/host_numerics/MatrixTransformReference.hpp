// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private hipBLASLt adapter.

#include <complex>
#include <optional>
#include <ostream>
#include <roc/host_numerics/comparison.hpp>
#include <roc/host_numerics/tensor_operations.hpp>
#include <stdexcept>
#include <utility>

namespace hipblaslt::host_numerics
{
    using ::roc::host_numerics::compare;
    using ::roc::host_numerics::ComparisonOptions;
    using ::roc::host_numerics::ComparisonReport;
    using ::roc::host_numerics::Layout;
    using ::roc::host_numerics::addInto;
    using ::roc::host_numerics::multiply;
    using ::roc::host_numerics::multiplyInto;
    using ::roc::host_numerics::ScalarType;
    using ::roc::host_numerics::scalarTypeInfo;
    using ::roc::host_numerics::Shape;
    using ::roc::host_numerics::Tensor;

    inline void reportMatrixTransformMismatches(std::ostream&           output,
                                                const ComparisonReport& comparison)
    {
        output << "MatrixTransform " << formatComparisonReport(comparison);
    }

    inline Layout matrixTransformLayout(size_t    rows,
                                        size_t    columns,
                                        size_t    batchCount,
                                        ptrdiff_t leadingDimension,
                                        ptrdiff_t batchStride,
                                        bool      rowMajor,
                                        bool      transpose)
    {
        const ptrdiff_t physicalRowStride    = rowMajor ? leadingDimension : 1;
        const ptrdiff_t physicalColumnStride = rowMajor ? 1 : leadingDimension;
        return Layout(Shape{rows, columns, batchCount},
                      {transpose ? physicalColumnStride : physicalRowStride,
                       transpose ? physicalRowStride : physicalColumnStride,
                       batchStride});
    }

    inline ComparisonReport referenceMatrixTransform(const Tensor&                observed,
                                                     const std::optional<Tensor>& a,
                                                     const std::optional<Tensor>& b,
                                                     std::complex<double>     alpha = {1.0, 0.0},
                                                     std::complex<double>     beta  = {1.0, 0.0},
                                                     const ComparisonOptions& comparisonOptions
                                                     = {})
    {
        if(scalarTypeInfo(observed.type()).isPacked())
            throw std::invalid_argument(
                "MatrixTransform reference does not support packed scalar storage.");

        Tensor expected(ScalarType::Float32, observed.shape());
        if(a && b)
        {
            Tensor scaledA = multiply(*a,
                                      Tensor::scalar(ScalarType::Float32, alpha),
                                      ScalarType::Float32,
                                      ScalarType::Float32);
            Tensor scaledB = multiply(*b,
                                      Tensor::scalar(ScalarType::Float32, beta),
                                      ScalarType::Float32,
                                      ScalarType::Float32);
            addInto(scaledA, scaledB, expected, ScalarType::Float32);
        }
        else if(a)
            multiplyInto(
                *a, Tensor::scalar(ScalarType::Float32, alpha), expected, ScalarType::Float32);
        else if(b)
            multiplyInto(
                *b, Tensor::scalar(ScalarType::Float32, beta), expected, ScalarType::Float32);
        else
            throw std::invalid_argument("MatrixTransform reference requires A or B.");

        return compare(observed, expected, comparisonOptions);
    }
} // namespace hipblaslt::host_numerics
