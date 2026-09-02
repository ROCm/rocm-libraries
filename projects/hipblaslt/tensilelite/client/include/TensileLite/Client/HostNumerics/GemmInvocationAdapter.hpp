// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// TensileLite client-private translation from GEMM descriptors to
// product-independent host-numerics operations.

#include <Tensile/ContractionProblem.hpp>

#include <roc/host_numerics/epilogue.hpp>
#include <roc/host_numerics/gemm.hpp>
#include <roc/host_numerics/reduction.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace TensileLite::Client::HostNumerics
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

        roc::host_numerics::GemmOptions& gemmOptions()
        {
            return options;
        }

        const roc::host_numerics::GemmOptions& gemmOptions() const
        {
            return options;
        }

        roc::host_numerics::GemmBackend runGemm(roc::host_numerics::GemmBackend backend
                                                = roc::host_numerics::GemmBackend::Automatic) const;
        void runPostGemmOperationsAndCopyOutputs() const;

    private:
        explicit TranslatedGemmBatch(roc::host_numerics::Tensor     aTensor,
                                     roc::host_numerics::Tensor     bTensor,
                                     roc::host_numerics::Tensor     cTensor,
                                     roc::host_numerics::Tensor     dTensor,
                                     roc::host_numerics::ScalarType accumulatorType)
            : a(std::move(aTensor))
            , b(std::move(bTensor))
            , c(std::move(cTensor))
            , d(std::move(dTensor))
            , options(accumulatorType)
        {
        }

        struct CopyBack
        {
            std::span<std::byte>                destination;
            roc::host_numerics::Tensor          source;
            roc::host_numerics::OutputSelection selection;
        };

        struct BiasReduction
        {
            BiasReduction(roc::host_numerics::Tensor     inputTensor,
                          roc::host_numerics::Tensor     outputTensor,
                          roc::host_numerics::ScalarType accumulator,
                          std::vector<size_t>            reductionAxes)
                : input(std::move(inputTensor))
                , output(std::move(outputTensor))
                , accumulatorType(accumulator)
                , axes(std::move(reductionAxes))
            {
            }

            roc::host_numerics::Tensor     input;
            roc::host_numerics::Tensor     output;
            roc::host_numerics::ScalarType accumulatorType;
            std::vector<size_t>            axes;
        };

        struct BoundEpilogue
        {
            BoundEpilogue(roc::host_numerics::Tensor     inputTensor,
                          roc::host_numerics::Tensor     outputTensor,
                          roc::host_numerics::ScalarType computeType)
                : input(std::move(inputTensor))
                , outputs{.output = std::move(outputTensor)}
                , options(computeType)
            {
            }

            roc::host_numerics::Tensor          input;
            roc::host_numerics::EpilogueOutputs outputs;
            roc::host_numerics::EpilogueOptions options;
        };

        roc::host_numerics::Tensor      a;
        roc::host_numerics::Tensor      b;
        roc::host_numerics::Tensor      c;
        roc::host_numerics::Tensor      d;
        roc::host_numerics::GemmOptions options;
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
        roc::host_numerics::GemmBackend execute(roc::host_numerics::GemmBackend backend) const;

        std::variant<TranslatedGemmBatch, TranslationFailure>
            translateBatch(size_t batch) const;

    private:
        struct State;

        explicit GemmInvocationAdapter(std::unique_ptr<const State> state);

        friend std::variant<GemmInvocationAdapter, TranslationFailure>
            translateGemmInvocation(ContractionProblemGemm const& problem,
                                    ContractionInputs const&      inputs,
                                    roc::host_numerics::OutputSelection outputSelection);

        std::unique_ptr<const State> m_state;
    };

    std::variant<GemmInvocationAdapter, TranslationFailure>
        translateGemmInvocation(ContractionProblemGemm const& problem,
                                ContractionInputs const&      inputs,
                                roc::host_numerics::OutputSelection outputSelection);

    std::variant<GemmInvocationAdapter, TranslationFailure>
        translateGemmInvocation(ContractionProblemGemm const& problem,
                                ContractionInputs const&      inputs,
                                size_t                        elementsToValidate);
} // namespace TensileLite::Client::HostNumerics
