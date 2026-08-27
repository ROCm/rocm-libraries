// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/HostNumerics/HostReference.hpp>

#include <sstream>
#include <utility>

#include <roc/host_validation/backends/blas.hpp>
#include <roc/host_validation/gemm.hpp>

namespace rocRoller::HostNumerics
{
    namespace
    {
        using namespace roc::host_validation;

        size_t checkedElementCount(size_t first, size_t second, char const* description)
        {
            if(second != 0 && first > std::numeric_limits<size_t>::max() / second)
                throw std::overflow_error(description);
            return first * second;
        }

        BlockScaleBinding normalizeBlockScale(Tensor      values,
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

            return {std::move(values), blockSize};
        }
    }

    std::string HostComparisonResult::message() const
    {
        std::ostringstream output;
        output << (ok ? "Comparison PASSED." : "Comparison FAILED.");
        output << "  Relative norms:";
        output << " L2 " << std::scientific << relativeNormL2;
        output << " Inf " << std::scientific << relativeNormInf;
        output << "  Norms (x/ref):";
        output << " L2 " << std::scientific << observedNormL2 << "/" << referenceNormL2;
        output << " Inf " << std::scientific << observedNormInf << "/" << referenceNormInf;
        output << "  Tolerance: " << std::scientific << acceptableError.relativeL2Tolerance;
        output << " (" << acceptableError.reasoning << ")";
        return output.str();
    }

    roc::host_validation::Tensor hostScaleTensor(DataType                 type,
                                                 std::span<const uint8_t> values,
                                                 size_t                   freeExtent,
                                                 size_t                   reductionExtent,
                                                 size_t                   blockSize)
    {
        using namespace roc::host_validation;
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

    roc::host_validation::Tensor hostScaleTensor(DataType                 type,
                                                 std::span<const uint8_t> values,
                                                 TensorDescriptor const&  dataDescriptor,
                                                 size_t                   blockedDimension,
                                                 size_t                   blockSize)
    {
        using namespace roc::host_validation;

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

    HostReferenceProblem
        makeHostReferenceProblem(GeneratedGEMMInputs const&                  inputs,
                                 std::optional<roc::host_validation::Tensor> runtimeScaleA,
                                 std::optional<roc::host_validation::Tensor> runtimeScaleB,
                                 size_t                                      scaleBlockSize,
                                 float                                       alpha,
                                 float                                       beta)
    {
        HostReferenceProblem result(inputs.a, inputs.b, inputs.c);
        result.scaleA
            = runtimeScaleA
                  ? std::move(runtimeScaleA)
                  : (inputs.scaleA ? std::optional<roc::host_validation::Tensor>(inputs.scaleA)
                                   : std::nullopt);
        result.scaleB
            = runtimeScaleB
                  ? std::move(runtimeScaleB)
                  : (inputs.scaleB ? std::optional<roc::host_validation::Tensor>(inputs.scaleB)
                                   : std::nullopt);
        result.scaleBlockSize = scaleBlockSize;
        result.alpha          = alpha;
        result.beta           = beta;
        return result;
    }

    roc::host_validation::Tensor computeHostReference(HostReferenceProblem const& problem)
    {
        using namespace roc::host_validation;

        if(problem.a.shape().rank() != 2 || problem.b.shape().rank() != 2
           || problem.c.shape().rank() != 2)
            throw std::invalid_argument(
                "rocRoller host GEMM requires rank-two A, B, and C tensors.");

        const size_t rows            = problem.a.shape()[0];
        const size_t reductionExtent = problem.a.shape()[1];
        const size_t columns         = problem.b.shape()[1];
        if(problem.b.shape()[0] != reductionExtent)
            throw std::invalid_argument(
                "rocRoller host GEMM A and B reduction extents do not match.");
        if(problem.c.shape() != Shape{rows, columns})
            throw std::invalid_argument(
                "rocRoller host GEMM C shape does not match the output shape.");
        if(rows > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()))
            throw std::overflow_error("rocRoller host GEMM output stride exceeds ptrdiff_t.");

        Tensor output(ScalarType::Float32,
                      Layout(Shape{rows, columns}, {1, static_cast<ptrdiff_t>(rows)}));

        GemmOperand           operandA(problem.a);
        GemmOperand           operandB(problem.b);
        std::optional<Tensor> unitScaleA;
        std::optional<Tensor> unitScaleB;
        if(problem.scaleA || problem.scaleB)
        {
            if(problem.scaleBlockSize == 0)
                throw std::invalid_argument("rocRoller scale block size must be nonzero.");
            const size_t blockCount
                = reductionExtent / problem.scaleBlockSize
                  + static_cast<size_t>(reductionExtent % problem.scaleBlockSize != 0);
            if(!problem.scaleA)
            {
                const std::vector<float> values(
                    checkedElementCount(
                        rows, blockCount, "rocRoller A scale element count overflow."),
                    1.0f);
                unitScaleA = Tensor::copyValuesWithConversion(
                    ScalarType::Float32, Shape{rows, blockCount}, std::span<const float>(values));
            }
            if(!problem.scaleB)
            {
                const std::vector<float> values(
                    checkedElementCount(
                        columns, blockCount, "rocRoller B scale element count overflow."),
                    1.0f);
                unitScaleB = Tensor::copyValuesWithConversion(ScalarType::Float32,
                                                Shape{columns, blockCount},
                                                std::span<const float>(values));
            }
            operandA.blockScale
                = normalizeBlockScale(problem.scaleA ? *problem.scaleA : *unitScaleA,
                                      rows,
                                      reductionExtent,
                                      problem.scaleBlockSize,
                                      "A");
            operandB.blockScale
                = normalizeBlockScale(problem.scaleB ? *problem.scaleB : *unitScaleB,
                                      columns,
                                      reductionExtent,
                                      problem.scaleBlockSize,
                                      "B");
        }

        GemmRequest request(
            std::move(operandA), std::move(operandB), problem.c, output, ScalarType::Float32);
        request.epilogue.alpha = problem.alpha;
        request.epilogue.beta  = problem.beta;

        static const TransformingBlasGemmBackend backend;
        referenceGemm(request,
                      {
                          .backend                 = GemmBackend::Blas,
                          .requireRequestedBackend = false,
                      },
                      &backend);
        return output;
    }

    HostComparisonResult compareHostReference(roc::host_validation::Tensor observed,
                                              roc::host_validation::Tensor expected,
                                              AcceptableGEMMError          acceptableError)
    {
        using namespace roc::host_validation;

        ComparisonOptions options;
        options.pointwise                  = false;
        options.computePointwiseStatistics = true;
        options.computeFrobenius           = true;
        options.computeUlp                 = false;
        options.maxReportedMismatches      = 0;

        ComparisonResult statistics = compare(observed, expected, options);
        const bool hasNonFinite = statistics.matchedNaNs != 0 || statistics.matchedInfinities != 0
                                  || statistics.nonFiniteMismatches != 0;
        const double relativeNormL2
            = hasNonFinite ? std::numeric_limits<double>::quiet_NaN()
                           : statistics.frobeniusDifference / statistics.frobeniusExpected;
        const double relativeNormInf
            = hasNonFinite ? std::numeric_limits<double>::quiet_NaN()
                           : statistics.maxAbsoluteDifference / statistics.maximumExpectedMagnitude;
        const bool passed = relativeNormL2 < acceptableError.relativeL2Tolerance;

        statistics.relativeFrobeniusError = relativeNormL2;
        statistics.frobeniusPassed        = passed;

        return {
            .ok               = passed,
            .relativeNormL2   = relativeNormL2,
            .relativeNormInf  = relativeNormInf,
            .referenceNormL2  = statistics.frobeniusExpected,
            .referenceNormInf = statistics.maximumExpectedMagnitude,
            .observedNormL2   = statistics.frobeniusObserved,
            .observedNormInf  = statistics.maximumObservedMagnitude,
            .acceptableError  = std::move(acceptableError),
            .statistics       = std::move(statistics),
        };
    }
}
