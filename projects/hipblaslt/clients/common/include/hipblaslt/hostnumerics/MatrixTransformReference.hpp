// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private hipBLASLt adapter.

#include <complex>
#include <optional>
#include <ostream>
#include <roc/hostnumerics/comparison.hpp>
#include <roc/hostnumerics/tensor_operations.hpp>
#include <stdexcept>
#include <utility>

namespace hipblaslt::hostnumerics
{
    using ::roc::hostnumerics::addInto;
    using ::roc::hostnumerics::compare;
    using ::roc::hostnumerics::ComparisonOptions;
    using ::roc::hostnumerics::ComparisonReport;
    using ::roc::hostnumerics::Layout;
    using ::roc::hostnumerics::multiply;
    using ::roc::hostnumerics::multiplyInto;
    using ::roc::hostnumerics::ScalarType;
    using ::roc::hostnumerics::scalarTypeInfo;
    using ::roc::hostnumerics::Shape;
    using ::roc::hostnumerics::Tensor;

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
                       batchStride},
                      0);
    }

    inline ComparisonReport referenceMatrixTransform(const Tensor&                observed,
                                                     const std::optional<Tensor>& a,
                                                     const std::optional<Tensor>& b,
                                                     std::complex<double>         alpha,
                                                     std::complex<double>         beta,
                                                     const ComparisonOptions&     comparisonOptions)
    {
        if(scalarTypeInfo(observed.type()).isPacked())
            throw std::runtime_error(
                "MatrixTransform reference does not support packed scalar storage.");

        Tensor expected(ScalarType::Float32, observed.shape());
        if(a && b)
        {
            Tensor scaledA = multiply(*a,
                                      Tensor::scalar(ScalarType::Float32, alpha),
                                      ScalarType::Float32,
                                      ScalarType::Float32,
                                      std::nullopt);
            Tensor scaledB = multiply(*b,
                                      Tensor::scalar(ScalarType::Float32, beta),
                                      ScalarType::Float32,
                                      ScalarType::Float32,
                                      std::nullopt);
            addInto(scaledA,
                    scaledB,
                    expected,
                    ScalarType::Float32,
                    roc::hostnumerics::OutputSelection::all({}));
        }
        else if(a)
            multiplyInto(*a,
                         Tensor::scalar(ScalarType::Float32, alpha),
                         expected,
                         ScalarType::Float32,
                         roc::hostnumerics::OutputSelection::all({}));
        else if(b)
            multiplyInto(*b,
                         Tensor::scalar(ScalarType::Float32, beta),
                         expected,
                         ScalarType::Float32,
                         roc::hostnumerics::OutputSelection::all({}));
        else
            throw std::runtime_error("MatrixTransform reference requires A or B.");

        return compare(observed, expected, comparisonOptions);
    }
} // namespace hipblaslt::hostnumerics
