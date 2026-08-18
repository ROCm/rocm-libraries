// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private hipBLASLt adapter.

#include <complex>
#include <cstddef>
#include <hip/library_types.h>
#include <optional>
#include <ostream>
#include <hipblaslt/host_validation/Types.hpp>
#include <roc/host_validation/axpby.hpp>
#include <roc/host_validation/comparison.hpp>
#include <span>
#include <stdexcept>
#include <utility>

namespace hipblaslt::host_validation
{
    using namespace ::roc::host_validation;

    struct MatrixTransformReferenceArguments
    {
        const void*          observed               = nullptr;
        size_t               observedStorageBytes   = 0;
        const void*          a                      = nullptr;
        size_t               aStorageBytes          = 0;
        const void*          b                      = nullptr;
        size_t               bStorageBytes          = 0;
        hipDataType          type                   = HIP_R_32F;
        size_t               rows                   = 0;
        size_t               columns                = 0;
        size_t               batchCount             = 1;
        ptrdiff_t            leadingDimensionA      = 0;
        ptrdiff_t            leadingDimensionB      = 0;
        ptrdiff_t            leadingDimensionOutput = 0;
        ptrdiff_t            batchStride            = 0;
        bool                 rowMajorA              = false;
        bool                 rowMajorB              = false;
        bool                 rowMajorOutput         = false;
        bool                 transposeA             = false;
        bool                 transposeB             = false;
        std::complex<double> alpha{1.0, 0.0};
        std::complex<double> beta{1.0, 0.0};
        ComparisonOptions    comparison;
    };

    struct MatrixTransformReferenceResult
    {
        AxpbyRunInfo     runInfo;
        ComparisonResult comparison;
    };

    inline void reportMatrixTransformMismatches(std::ostream&           output,
                                                const ComparisonResult& comparison)
    {
        output << "MatrixTransform validation found " << comparison.mismatches
               << " mismatches among " << comparison.compared << " compared elements";

        for(const auto& mismatch : comparison.reportedMismatches)
        {
            output << "\n  index " << mismatch.index;
            if(!mismatch.coordinates.empty())
            {
                output << " coordinates [";
                for(size_t i = 0; i < mismatch.coordinates.size(); ++i)
                {
                    if(i != 0)
                        output << ", ";
                    output << mismatch.coordinates[i];
                }
                output << "]";
            }
            output << ": expected " << mismatch.expected << ", observed " << mismatch.observed
                   << ", absolute difference " << mismatch.absoluteDifference << ", tolerance "
                   << mismatch.tolerance;
        }

        if(comparison.mismatches > comparison.reportedMismatches.size())
            output << "\n  " << comparison.mismatches - comparison.reportedMismatches.size()
                   << " additional mismatches not shown";
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

    inline Tensor matrixTransformTensor(const void* pointer,
                                        size_t      storageBytes,
                                        ScalarType  type,
                                        Layout      layout)
    {
        return Tensor(
            type,
            std::move(layout),
            std::span<const std::byte>(static_cast<const std::byte*>(pointer), storageBytes));
    }

    inline MatrixTransformReferenceResult
        referenceMatrixTransform(const MatrixTransformReferenceArguments& arguments)
    {
        const ScalarType type = scalarType(arguments.type);
        if(!arguments.observed)
            throw std::invalid_argument("MatrixTransform reference requires observed output.");
        if(scalarTypeInfo(type).isPacked())
            throw std::invalid_argument(
                "MatrixTransform reference does not support packed scalar storage.");

        const Layout aLayout      = matrixTransformLayout(arguments.rows,
                                                          arguments.columns,
                                                          arguments.batchCount,
                                                          arguments.leadingDimensionA,
                                                          arguments.batchStride,
                                                          arguments.rowMajorA,
                                                          arguments.transposeA);
        const Layout bLayout      = matrixTransformLayout(arguments.rows,
                                                          arguments.columns,
                                                          arguments.batchCount,
                                                          arguments.leadingDimensionB,
                                                          arguments.batchStride,
                                                          arguments.rowMajorB,
                                                          arguments.transposeB);
        const Layout outputLayout = matrixTransformLayout(arguments.rows,
                                                          arguments.columns,
                                                          arguments.batchCount,
                                                          arguments.leadingDimensionOutput,
                                                          arguments.batchStride,
                                                          arguments.rowMajorOutput,
                                                          false);

        std::optional<Tensor> a;
        std::optional<Tensor> b;
        if(arguments.a)
            a = matrixTransformTensor(arguments.a, arguments.aStorageBytes, type, aLayout);
        if(arguments.b)
            b = matrixTransformTensor(arguments.b, arguments.bStorageBytes, type, bLayout);

        Tensor       expected(ScalarType::Float32, outputLayout);
        AxpbyProblem problem(std::move(a), std::move(b), expected, ScalarType::Float32);
        problem.alpha              = arguments.alpha;
        problem.beta               = arguments.beta;
        const AxpbyRunInfo runInfo = referenceAxpby(problem);

        Tensor observed
            = matrixTransformTensor(
                  arguments.observed, arguments.observedStorageBytes, type, outputLayout)
                  .to(ScalarType::Float32);
        const ComparisonResult comparison = compare(observed, expected, arguments.comparison);
        return {
            .runInfo    = runInfo,
            .comparison = comparison,
        };
    }
} // namespace hipblaslt::host_validation
