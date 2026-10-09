// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// TensileLite client-private translation from GEMM descriptors to
// product-independent HostNumerics operations.

#include <Tensile/ContractionProblem.hpp>

#include <roc/hostnumerics/epilogue.hpp>
#include <roc/hostnumerics/gemm.hpp>
#include <roc/hostnumerics/reduction.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace TensileLite::Client::hostnumerics
{
    enum class TranslationFailureCode
    {
        UnsupportedContraction,
        UnsupportedBiasSource,
        MissingInput,
        InvalidBatchPointer,
        UnsupportedDataType,
        UnsupportedActivation,
        InvalidScaleConfiguration,
        InvalidDescriptor,
        InvalidBatchIndex,
        Count,
    };

    struct TranslationFailure
    {
        TranslationFailureCode code;
        std::string            reason;
    };

    struct TranslatedGemmBatch
    {
        TranslatedGemmBatch(TranslatedGemmBatch&&) noexcept            = default;
        TranslatedGemmBatch& operator=(TranslatedGemmBatch&&) noexcept = default;

        TranslatedGemmBatch(const TranslatedGemmBatch&)            = delete;
        TranslatedGemmBatch& operator=(const TranslatedGemmBatch&) = delete;

        roc::hostnumerics::MatmulOptions& matmulOptions()
        {
            return options;
        }

        const roc::hostnumerics::MatmulOptions& matmulOptions() const
        {
            return options;
        }

        void runGemm(roc::hostnumerics::GemmBackend backend
                     = roc::hostnumerics::GemmBackend::Automatic) const;
        void runPostGemmOperationsAndCopyOutputs() const;

    private:
        explicit TranslatedGemmBatch(roc::hostnumerics::Tensor     aTensor,
                                     roc::hostnumerics::Tensor     bTensor,
                                     roc::hostnumerics::Tensor     cTensor,
                                     roc::hostnumerics::Tensor     dTensor,
                                     roc::hostnumerics::ScalarType accumulatorType)
            : a(std::move(aTensor))
            , b(std::move(bTensor))
            , c(std::move(cTensor))
            , d(std::move(dTensor))
            , options(accumulatorType)
            , alpha(roc::hostnumerics::Tensor::scalar(accumulatorType, 1))
            , beta(roc::hostnumerics::Tensor::scalar(accumulatorType, 0))
            , scaleC(roc::hostnumerics::Tensor::scalar(accumulatorType, 1))
        {
        }

        struct CopyBack
        {
            std::span<std::byte>                destination;
            roc::hostnumerics::Tensor          source;
            roc::hostnumerics::OutputSelection selection;
        };

        struct BiasReduction
        {
            BiasReduction(roc::hostnumerics::Tensor     inputTensor,
                          roc::hostnumerics::Tensor     outputTensor,
                          roc::hostnumerics::ScalarType accumulator,
                          std::vector<size_t>            reductionAxes)
                : input(std::move(inputTensor))
                , output(std::move(outputTensor))
                , accumulatorType(accumulator)
                , axes(std::move(reductionAxes))
            {
            }

            roc::hostnumerics::Tensor     input;
            roc::hostnumerics::Tensor     output;
            roc::hostnumerics::ScalarType accumulatorType;
            std::vector<size_t>            axes;
        };

        struct BoundEpilogue
        {
            BoundEpilogue(roc::hostnumerics::Tensor     inputTensor,
                          roc::hostnumerics::Tensor     outputTensor,
                          roc::hostnumerics::ScalarType computeType)
                : input(std::move(inputTensor))
                , outputs{.output = std::move(outputTensor)}
                , options(computeType)
            {
            }

            roc::hostnumerics::Tensor          input;
            roc::hostnumerics::EpilogueOutputs outputs;
            roc::hostnumerics::EpilogueOptions options;
        };

        roc::hostnumerics::Tensor      a;
        roc::hostnumerics::Tensor      b;
        roc::hostnumerics::Tensor      c;
        roc::hostnumerics::Tensor      d;
        roc::hostnumerics::MatmulOptions options;
        roc::hostnumerics::Tensor        alpha;
        roc::hostnumerics::Tensor        beta;
        roc::hostnumerics::Tensor        scaleC;
        std::optional<roc::hostnumerics::Tensor> scaleAlpha;
        std::optional<roc::hostnumerics::Tensor> scaleA;
        std::optional<roc::hostnumerics::Tensor> scaleB;
        roc::hostnumerics::OutputSelection       outputSelection;
        std::optional<BoundEpilogue>    epilogue;
        std::optional<BiasReduction>    biasReduction;
        std::vector<CopyBack>           copyBacks;

        friend class GemmInvocationAdapter;
    };

    // Move-only translation plan for one TensileLite invocation. Preflight
    // validates every batch before execution, while translateBatch snapshots
    // only the requested batch. Pointer-array addresses are captured during
    // preflight; the adapter then borrows the input backing buffers, whose bytes
    // must remain valid and unchanged until their batch is translated. A returned
    // TranslatedGemmBatch owns its input snapshot;
    // output buffers must remain valid through
    // runPostGemmOperationsAndCopyOutputs(). Batches must be translated, executed,
    // and copied back in ascending order when invocation-wide AMax is enabled.
    class GemmInvocationAdapter
    {
    public:
        ~GemmInvocationAdapter();
        GemmInvocationAdapter(GemmInvocationAdapter&&) noexcept;
        GemmInvocationAdapter& operator=(GemmInvocationAdapter&&) noexcept;

        GemmInvocationAdapter(const GemmInvocationAdapter&)            = delete;
        GemmInvocationAdapter& operator=(const GemmInvocationAdapter&) = delete;

        size_t batchCount() const;

        // Executes every preflighted batch in the required order, including
        // post-GEMM operations and copies back to caller-owned storage.
        void execute(roc::hostnumerics::GemmBackend backend) const;

        std::variant<TranslatedGemmBatch, TranslationFailure>
            translateBatch(size_t batch) const;

    private:
        struct State;

        explicit GemmInvocationAdapter(std::unique_ptr<const State> state);

        friend std::variant<GemmInvocationAdapter, TranslationFailure>
            translateGemmInvocation(ContractionProblemGemm const& problem,
                                    ContractionInputs const&      inputs,
                                    roc::hostnumerics::OutputSelection outputSelection);

        std::unique_ptr<const State> m_state;
    };

    std::variant<GemmInvocationAdapter, TranslationFailure>
        translateGemmInvocation(ContractionProblemGemm const& problem,
                                ContractionInputs const&      inputs,
                                roc::hostnumerics::OutputSelection outputSelection);

    std::variant<GemmInvocationAdapter, TranslationFailure>
        translateGemmInvocation(ContractionProblemGemm const& problem,
                                ContractionInputs const&      inputs,
                                size_t                        elementsToValidate);
} // namespace TensileLite::Client::hostnumerics
