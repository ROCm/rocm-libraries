// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/HostNumerics/HostReference.hpp>

#include <sstream>
#include <utility>

#include <roc/host_numerics/backends/blas.hpp>
#include <roc/host_numerics/gemm.hpp>

namespace rocRoller::HostNumerics
{
    namespace
    {
        using namespace roc::host_numerics;

        Tensor normalizeBlockScale(Tensor      values,
                                   size_t      freeExtent,
                                   size_t      reductionExtent,
                                   size_t      blockSize,
                                   char const* name)
        {
            if(blockSize == 0)
                throw std::invalid_argument("rocRoller scale block size must be nonzero.");

            const size_t blockCount = reductionExtent / blockSize
                                      + static_cast<size_t>(reductionExtent % blockSize != 0);
            if(values.shape() != Shape{freeExtent, blockCount})
                throw std::invalid_argument(std::string("rocRoller ") + name
                                            + " scales must have shape [free extent, K block].");

            return values;
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

    roc::host_numerics::Tensor hostScaleTensor(DataType                 type,
                                               std::span<const uint8_t> values,
                                               size_t                   freeExtent,
                                               size_t                   reductionExtent,
                                               size_t                   blockSize)
    {
        using namespace roc::host_numerics;
        auto const scalarType = hostScalarType(type);
        if(scalarTypeInfo(scalarType).category != ScalarCategory::Scale)
            throw std::invalid_argument("rocRoller runtime scale requires a scale data type.");
        if(values.size() != 1)
            throw std::invalid_argument("rocRoller runtime scale storage must contain one scalar.");
        if(blockSize == 0)
            throw std::invalid_argument("rocRoller runtime scale block size must be nonzero.");
        const size_t blockCount
            = reductionExtent / blockSize + static_cast<size_t>(reductionExtent % blockSize != 0);
        return Tensor::copyEncodedBackingStorage(
            scalarType, Layout(Shape{freeExtent, blockCount}, {0, 0}), std::as_bytes(values));
    }

    roc::host_numerics::Tensor hostScaleTensor(DataType                 type,
                                               std::span<const uint8_t> values,
                                               TensorDescriptor const&  dataDescriptor,
                                               size_t                   blockedDimension,
                                               size_t                   blockSize)
    {
        using namespace roc::host_numerics;

        auto const layout     = hostScaleLayout(dataDescriptor, blockedDimension, blockSize);
        auto const scalarType = hostScalarType(type);
        if(scalarTypeInfo(scalarType).category != ScalarCategory::Scale)
            throw std::invalid_argument("rocRoller block scales require a scale data type.");
        if(values.size() == 1)
            return Tensor::copyEncodedBackingStorage(
                scalarType, Layout(layout.shape(), {0, 0}), std::as_bytes(values));

        auto const requiredBytes = storageBytesForLayout(scalarType, layout);
        if(values.size_bytes() != requiredBytes)
            throw std::invalid_argument(
                "rocRoller block-scale storage does not match its data descriptor.");
        return Tensor::copyEncodedBackingStorage(scalarType, layout, std::as_bytes(values));
    }

    roc::host_numerics::Tensor
        computeHostReference(roc::host_numerics::Tensor                a,
                             roc::host_numerics::Tensor                b,
                             roc::host_numerics::Tensor                c,
                             std::optional<roc::host_numerics::Tensor> scaleA,
                             std::optional<roc::host_numerics::Tensor> scaleB,
                             size_t                                    scaleBlockSize,
                             float                                     alpha,
                             float                                     beta)
    {
        using namespace roc::host_numerics;

        if(a.shape().rank() != 2 || b.shape().rank() != 2 || c.shape().rank() != 2)
            throw std::invalid_argument(
                "rocRoller host GEMM requires rank-two A, B, and C tensors.");
        if(b.shape()[0] != a.shape()[1])
            throw std::invalid_argument(
                "rocRoller host GEMM A and B reduction extents do not match.");
        if(c.shape() != Shape{a.shape()[0], b.shape()[1]})
            throw std::invalid_argument(
                "rocRoller host GEMM C shape does not match the output shape.");

        GemmOptions options(ScalarType::Float32);
        if(scaleA)
            options.blockScaleA = normalizeBlockScale(
                std::move(*scaleA), a.shape()[0], a.shape()[1], scaleBlockSize, "A");
        if(scaleB)
            options.blockScaleB = normalizeBlockScale(
                std::move(*scaleB), b.shape()[1], b.shape()[0], scaleBlockSize, "B");
        if(options.blockScaleA)
            options.blockSizeA = scaleBlockSize;
        if(options.blockScaleB)
            options.blockSizeB = scaleBlockSize;

        options.alpha = alpha;
        options.beta  = beta;

        const size_t rows = a.shape()[0];
        if(rows > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()))
            throw std::overflow_error("rocRoller host GEMM output stride exceeds ptrdiff_t.");
        const Layout outputLayout(Shape{rows, b.shape()[1]}, {1, static_cast<ptrdiff_t>(rows)});
        return referenceGemmWithBlasBackend(
            std::move(a), std::move(b), std::move(c), ScalarType::Float32, options, outputLayout);
    }

    roc::host_numerics::Tensor
        computeHostReference(GeneratedGEMMInputs const&                inputs,
                             std::optional<roc::host_numerics::Tensor> runtimeScaleA,
                             std::optional<roc::host_numerics::Tensor> runtimeScaleB,
                             size_t                                    scaleBlockSize,
                             float                                     alpha,
                             float                                     beta)
    {
        return computeHostReference(
            inputs.a,
            inputs.b,
            inputs.c,
            runtimeScaleA
                ? std::move(runtimeScaleA)
                : (inputs.scaleA ? std::optional<roc::host_numerics::Tensor>(inputs.scaleA)
                                 : std::nullopt),
            runtimeScaleB
                ? std::move(runtimeScaleB)
                : (inputs.scaleB ? std::optional<roc::host_numerics::Tensor>(inputs.scaleB)
                                 : std::nullopt),
            scaleBlockSize,
            alpha,
            beta);
    }

    HostComparisonResult compareHostReference(roc::host_numerics::Tensor observed,
                                              roc::host_numerics::Tensor expected,
                                              AcceptableGEMMError        acceptableError)
    {
        using namespace roc::host_numerics;

        ComparisonOptions options;
        options.allClose                               = false;
        options.computeElementwiseStatistics           = true;
        options.computeFrobenius                       = true;
        options.computeUlp                             = false;
        options.maxReportedMismatches                  = 0;
        options.relativeFrobeniusTolerance             = acceptableError.relativeL2Tolerance;
        options.strictTolerance                        = true;
        options.zeroExpectedNormIsNaN                  = true;
        options.nonFiniteValuesInvalidateRelativeNorms = true;

        ComparisonReport statistics = compare(observed, expected, options);

        return {
            .acceptableError = std::move(acceptableError),
            .statistics      = std::move(statistics),
        };
    }
}
