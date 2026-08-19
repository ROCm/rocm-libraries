// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private translation from TensileLite GEMM descriptors to
// product-independent host-validation operations.

#include <Tensile/ContractionProblem.hpp>

#include <roc/host_validation/epilogue.hpp>
#include <roc/host_validation/gemm.hpp>
#include <roc/host_validation/reduction.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace TensileLite::Client::reference_adapter
{
    enum class TranslationFailureCode
    {
        UnsupportedContraction,
        UnsupportedBiasSource,
        MissingInput,
        InvalidBatchPointer,
        UnsupportedDataType,
        UnsupportedAccumulator,
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
        struct CopyBack
        {
            std::span<std::byte>                                 destination;
            roc::host_validation::Tensor                         source;
            std::optional<roc::host_validation::OutputSelection> selection;
        };

        roc::host_validation::GemmRequest& gemm()
        {
            return *gemmRequest;
        }

        const roc::host_validation::GemmRequest& gemm() const
        {
            return *gemmRequest;
        }

        void copyOutputs() const;

        std::optional<roc::host_validation::Tensor> runtimeScaleA;
        std::optional<roc::host_validation::Tensor> runtimeScaleB;
        std::optional<roc::host_validation::Tensor> intermediate;
        std::optional<roc::host_validation::Tensor> gradientAuxiliary;
        std::optional<roc::host_validation::Tensor> biasWorkspace;

        std::optional<roc::host_validation::GemmRequest>      gemmRequest;
        std::optional<roc::host_validation::EpilogueRequest>  epilogue;
        std::optional<roc::host_validation::ReductionRequest> biasReduction;
        std::vector<CopyBack>                                 copyBacks;
    };

    // Move-only translation plan for one TensileLite invocation. The adapter
    // borrows the problem and input descriptors supplied to
    // translateGemmInvocation and must not outlive either object.
    class GemmInvocationAdapter
    {
    public:
        ~GemmInvocationAdapter();
        GemmInvocationAdapter(GemmInvocationAdapter&&) noexcept;
        GemmInvocationAdapter& operator=(GemmInvocationAdapter&&) noexcept;

        GemmInvocationAdapter(const GemmInvocationAdapter&)            = delete;
        GemmInvocationAdapter& operator=(const GemmInvocationAdapter&) = delete;

        size_t batchCount() const;

        roc::host_validation::ScalarType operationAccumulatorType() const;

        bool usesStandaloneEpilogue() const;

        std::variant<TranslatedGemmBatch, TranslationFailure>
            translateBatch(size_t batch, roc::host_validation::ScalarType accumulatorType) const;

    private:
        struct State;

        explicit GemmInvocationAdapter(std::unique_ptr<const State> state);

        friend std::variant<GemmInvocationAdapter, TranslationFailure>
            translateGemmInvocation(ContractionProblemGemm const& problem,
                                    ContractionInputs const&      inputs,
                                    size_t                        elementsToValidate);

        std::unique_ptr<const State> m_state;
    };

    std::variant<GemmInvocationAdapter, TranslationFailure>
        translateGemmInvocation(ContractionProblemGemm const& problem,
                                ContractionInputs const&      inputs,
                                size_t                        elementsToValidate);
} // namespace TensileLite::Client::reference_adapter
