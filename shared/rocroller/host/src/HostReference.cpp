// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/hostnumerics/HostReference.hpp>

#include <algorithm>
#include <sstream>
#include <utility>

#include <roc/hostnumerics/backends/blas.hpp>
#include <roc/hostnumerics/gemm.hpp>
#include <roc/hostnumerics/tensor_operations.hpp>

namespace rocRoller::hostnumerics
{
    namespace
    {
        using roc::hostnumerics::Shape;
        using roc::hostnumerics::Tensor;

        bool isSupportedScaleType(DataType type)
        {
            return type == DataType::E4M3 || type == DataType::E5M3 || type == DataType::E8M0;
        }

        void validateBlockScaleShape(roc::hostnumerics::MatmulBlockScale const& scale,
                                     size_t                                     freeExtent,
                                     size_t                                     reductionExtent,
                                     char const*                                name)
        {
            const size_t blockSize = scale.blockSize;
            if(blockSize == 0)
                throw std::runtime_error("rocRoller scale block size must be nonzero.");

            const size_t blockCount = reductionExtent / blockSize
                                      + static_cast<size_t>(reductionExtent % blockSize != 0);
            if(scale.scales.shape() != Shape{freeExtent, blockCount})
                throw std::runtime_error(std::string("rocRoller ") + name
                                         + " scales must have shape [free extent, K block].");
        }
    }

    std::string HostComparisonResult::message() const
    {
        std::ostringstream output;
        output << (ok() ? "Comparison PASSED." : "Comparison FAILED.");
        output << "  Relative norms:";
        output << " L2 " << std::scientific << statistics.relativeFrobeniusError;
        output << " Inf " << std::scientific << statistics.relativeMaximumError;
        output << "  Norms (x/ref):";
        output << " L2 " << std::scientific << statistics.frobeniusObserved << "/"
               << statistics.frobeniusExpected;
        output << " Inf " << std::scientific << statistics.maximumObservedMagnitude << "/"
               << statistics.maximumExpectedMagnitude;
        output << "  Tolerance: " << std::scientific << acceptableError.relativeL2Tolerance;
        output << " (" << acceptableError.reasoning << ")";
        return output.str();
    }

    bool HostComparisonResult::ok() const
    {
        return statistics.passed();
    }

    roc::hostnumerics::MatmulBlockScale hostBlockScale(DataType                 type,
                                                       std::span<const uint8_t> values,
                                                       TensorDescriptor const&  dataDescriptor,
                                                       size_t                   blockedDimension,
                                                       size_t                   blockSize)
    {
        using roc::hostnumerics::storageBytesForLayout;
        using roc::hostnumerics::Tensor;

        if(!isSupportedScaleType(type))
            throw std::runtime_error("rocRoller block scales require a scale data type.");
        auto const layout     = hostScaleLayout(dataDescriptor, blockedDimension, blockSize);
        auto const scalarType = hostScalarType(type, DataTypeInterpretation::Unscaled);
        if(values.size_bytes() != storageBytesForLayout(scalarType, layout))
            throw std::runtime_error(
                "rocRoller block-scale storage does not match its data descriptor.");
        return {Tensor::copyEncodedBackingStorage(scalarType, layout, std::as_bytes(values)),
                blockSize};
    }

    roc::hostnumerics::MatmulBlockScale
        hostUniformScale(DataType type, uint8_t value, size_t freeExtent, size_t reductionExtent)
    {
        using roc::hostnumerics::Layout;
        using roc::hostnumerics::Shape;
        using roc::hostnumerics::Tensor;

        if(!isSupportedScaleType(type))
            throw std::runtime_error("rocRoller uniform scale requires a scale data type.");
        // An empty reduction still needs a nonzero block size; it then has no blocks.
        const size_t blockSize  = std::max<size_t>(reductionExtent, 1);
        const size_t blockCount = reductionExtent == 0 ? 0 : 1;
        const auto   bytes      = std::as_bytes(std::span(&value, 1));
        return {Tensor::copyEncodedBackingStorage(
                    hostScalarType(type, DataTypeInterpretation::Unscaled),
                    Layout(Shape{freeExtent, blockCount}, {0, 0}, 0),
                    bytes),
                blockSize};
    }

    roc::hostnumerics::Tensor
        computeHostReference(roc::hostnumerics::Tensor                          a,
                             roc::hostnumerics::Tensor                          b,
                             roc::hostnumerics::Tensor                          c,
                             std::optional<roc::hostnumerics::MatmulBlockScale> scaleA,
                             std::optional<roc::hostnumerics::MatmulBlockScale> scaleB,
                             float                                              alpha,
                             float                                              beta)
    {
        using roc::hostnumerics::add;
        using roc::hostnumerics::Layout;
        using roc::hostnumerics::MatmulOptions;
        using roc::hostnumerics::matmulWithBlasBackend;
        using roc::hostnumerics::ScalarType;
        using roc::hostnumerics::Shape;
        using roc::hostnumerics::Tensor;

        if(a.shape().rank() != 2 || b.shape().rank() != 2 || c.shape().rank() != 2)
            throw std::runtime_error("rocRoller host GEMM requires rank-two A, B, and C tensors.");
        if(b.shape()[0] != a.shape()[1])
            throw std::runtime_error("rocRoller host GEMM A and B reduction extents do not match.");
        if(c.shape() != Shape{a.shape()[0], b.shape()[1]})
            throw std::runtime_error(
                "rocRoller host GEMM C shape does not match the output shape.");

        MatmulOptions options(ScalarType::Float32);
        if(scaleA)
            validateBlockScaleShape(*scaleA, a.shape()[0], a.shape()[1], "A");
        if(scaleB)
            validateBlockScaleShape(*scaleB, b.shape()[1], b.shape()[0], "B");
        options.blockScaleA = std::move(scaleA);
        options.blockScaleB = std::move(scaleB);

        const size_t rows = a.shape()[0];
        if(rows > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()))
            throw std::runtime_error("rocRoller host GEMM output stride exceeds ptrdiff_t.");
        const Layout outputLayout(Shape{rows, b.shape()[1]}, {1, static_cast<ptrdiff_t>(rows)}, 0);
        std::optional<Tensor> result;
        if(alpha != 0.0f && a.shape()[1] != 0)
        {
            const Tensor product
                = matmulWithBlasBackend(a, b, ScalarType::Float32, options, std::nullopt);
            result = product * alpha;
        }
        if(beta != 0.0f)
        {
            Tensor addend = c * beta;
            if(result)
                return add(*result, addend, ScalarType::Float32, ScalarType::Float32, outputLayout);
            return addend.copyConvertedTo(ScalarType::Float32, outputLayout);
        }
        return result ? result->copyConvertedTo(ScalarType::Float32, outputLayout)
                      : Tensor(ScalarType::Float32, outputLayout);
    }

    roc::hostnumerics::Tensor
        computeHostReference(GeneratedGEMMInputs const& inputs, float alpha, float beta)
    {
        return computeHostReference(
            inputs.a, inputs.b, inputs.c, inputs.scaleA, inputs.scaleB, alpha, beta);
    }

    HostComparisonResult compareHostReference(roc::hostnumerics::Tensor observed,
                                              roc::hostnumerics::Tensor expected,
                                              AcceptableGEMMError       acceptableError)
    {
        using roc::hostnumerics::compare;
        using roc::hostnumerics::ComparisonOptions;
        using roc::hostnumerics::ComparisonReport;

        ComparisonOptions options;
        options.allClose                               = false;
        options.computeElementwiseStatistics           = true;
        options.computeFrobenius                       = true;
        options.computeUlp                             = false;
        options.maxReportedMismatches                  = 0;
        options.relativeFrobeniusTolerance             = acceptableError.relativeL2Tolerance;
        options.zeroExpectedNormIsNaN                  = true;
        options.nonFiniteValuesInvalidateRelativeNorms = true;

        ComparisonReport statistics = compare(observed, expected, options);

        return {
            .acceptableError = std::move(acceptableError),
            .statistics      = std::move(statistics),
        };
    }
}
