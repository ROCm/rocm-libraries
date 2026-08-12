// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "client/HostReference.hpp"

#include <sstream>
#include <utility>

#include <roc/host_validation/backends/blas.hpp>
#include <roc/host_validation/gemm.hpp>

namespace rocRoller::Client::GEMMClient
{
    namespace
    {
        using namespace roc::host_validation;

        ScalarType scaleScalarType(DataType type)
        {
            switch(type)
            {
            case DataType::E8M0:
                return ScalarType::E8M0;
            case DataType::E5M3:
                return ScalarType::E5M3;
            case DataType::E4M3:
                return ScalarType::E4M3;
            default:
                throw std::invalid_argument("Unsupported rocRoller host-reference scale type.");
            }
        }

        BlockScaleBinding normalizeBlockScale(TensorView  values,
                                              size_t      freeExtent,
                                              size_t      reductionExtent,
                                              size_t      blockSize,
                                              char const* name)
        {
            if(blockSize == 0)
                throw std::invalid_argument("rocroller-gemm scale block size must be nonzero.");

            const size_t blockCount = reductionExtent / blockSize
                                      + static_cast<size_t>(reductionExtent % blockSize != 0);
            if(values.shape() != Shape{freeExtent, blockCount})
                throw std::invalid_argument(std::string("rocroller-gemm ") + name
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

    roc::host_validation::TensorView hostScaleTensorView(DataType                 type,
                                                         std::span<const uint8_t> values,
                                                         size_t                   freeExtent,
                                                         size_t                   reductionExtent,
                                                         size_t                   blockSize)
    {
        using namespace roc::host_validation;
        if(values.size() != 1)
            throw std::invalid_argument(
                "rocroller-gemm runtime scale storage must contain one scalar.");
        if(blockSize == 0)
            throw std::invalid_argument("rocroller-gemm runtime scale block size must be nonzero.");
        const size_t blockCount
            = reductionExtent / blockSize + static_cast<size_t>(reductionExtent % blockSize != 0);
        return TensorView(scaleScalarType(type),
                          Layout(Shape{freeExtent, blockCount}, {0, 0}),
                          std::as_bytes(values));
    }

    HostReferenceProblem
        makeHostReferenceProblem(GeneratedGEMMInputs const&                      inputs,
                                 std::optional<roc::host_validation::TensorView> runtimeScaleA,
                                 std::optional<roc::host_validation::TensorView> runtimeScaleB,
                                 size_t                                          scaleBlockSize,
                                 float                                           alpha,
                                 float                                           beta)
    {
        HostReferenceProblem result(inputs.a.view(), inputs.b.view(), inputs.c.view());
        result.scaleA
            = runtimeScaleA
                  ? std::move(runtimeScaleA)
                  : (inputs.scaleA
                         ? std::optional<roc::host_validation::TensorView>(inputs.scaleA->view())
                         : std::nullopt);
        result.scaleB
            = runtimeScaleB
                  ? std::move(runtimeScaleB)
                  : (inputs.scaleB
                         ? std::optional<roc::host_validation::TensorView>(inputs.scaleB->view())
                         : std::nullopt);
        result.scaleBlockSize = scaleBlockSize;
        result.alpha          = alpha;
        result.beta           = beta;
        return result;
    }

    roc::host_validation::Tensor computeHostReference(HostReferenceProblem const& problem)
    {
        using namespace roc::host_validation;

        const size_t rows            = problem.a.shape()[0];
        const size_t reductionExtent = problem.a.shape()[1];
        const size_t columns         = problem.b.shape()[1];
        Tensor       output(ScalarType::Float32,
                            Layout(Shape{rows, columns}, {1, static_cast<ptrdiff_t>(rows)}));

        GemmOperand           operandA(problem.a);
        GemmOperand           operandB(problem.b);
        std::optional<Tensor> unitScaleA;
        std::optional<Tensor> unitScaleB;
        if(problem.scaleA || problem.scaleB)
        {
            if(problem.scaleBlockSize == 0)
                throw std::invalid_argument("rocroller-gemm scale block size must be nonzero.");
            const size_t blockCount
                = reductionExtent / problem.scaleBlockSize
                  + static_cast<size_t>(reductionExtent % problem.scaleBlockSize != 0);
            if(!problem.scaleA)
            {
                const std::vector<float> values(rows * blockCount, 1.0f);
                unitScaleA = Tensor::fromValues(
                    ScalarType::Float32, Shape{rows, blockCount}, std::span<const float>(values));
            }
            if(!problem.scaleB)
            {
                const std::vector<float> values(columns * blockCount, 1.0f);
                unitScaleB = Tensor::fromValues(ScalarType::Float32,
                                                Shape{columns, blockCount},
                                                std::span<const float>(values));
            }
            operandA.blockScale
                = normalizeBlockScale(problem.scaleA ? *problem.scaleA : unitScaleA->view(),
                                      rows,
                                      reductionExtent,
                                      problem.scaleBlockSize,
                                      "A");
            operandB.blockScale
                = normalizeBlockScale(problem.scaleB ? *problem.scaleB : unitScaleB->view(),
                                      columns,
                                      reductionExtent,
                                      problem.scaleBlockSize,
                                      "B");
        }

        GemmRequest request(std::move(operandA),
                            std::move(operandB),
                            problem.c,
                            output.mutableView(),
                            ScalarType::Float32);
        request.epilogue.alpha = problem.alpha;
        request.epilogue.beta  = problem.beta;

        static const TransformingBlasGemmBackend backend;
        referenceGemm(request,
                      {
                          .backend                 = GemmBackend::Blas,
                          .requireRequestedBackend = true,
                      },
                      &backend);
        return output;
    }

    HostComparisonResult compareHostReference(roc::host_validation::TensorView observed,
                                              roc::host_validation::TensorView expected,
                                              AcceptableGEMMError              acceptableError)
    {
        using namespace roc::host_validation;

        ComparisonOptions options;
        options.pointwise                  = false;
        options.computePointwiseStatistics = true;
        options.computeFrobenius           = true;
        options.computeUlp                 = false;
        options.maxReportedMismatches      = 0;

        ComparisonResult statistics = compare(observed, expected, options);
        const bool   hasNonFinite = statistics.matchedNaNs != 0 || statistics.matchedInfinities != 0
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
