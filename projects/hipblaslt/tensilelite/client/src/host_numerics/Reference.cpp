// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Product-private TensileLite execution policy around the descriptor adapter.

#include <roc/host_numerics/adapters/tensilelite/GemmInvocationAdapter.hpp>
#include <roc/host_numerics/adapters/tensilelite/HostNumericsBridge.hpp>
#include <roc/host_numerics/adapters/tensilelite/Reference.hpp>
#include <roc/host_numerics/validation.hpp>

#include "TimingInstrumentation.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <variant>

namespace TensileLite
{
    namespace
    {
        using namespace roc::host_numerics;
        using Client::reference_adapter::GemmInvocationAdapter;
        using Client::reference_adapter::TranslationFailure;

        [[noreturn]] void throwTranslationFailure(const TranslationFailure& failure)
        {
            throw std::invalid_argument("TensileLite host-numerics translation failed: "
                                        + failure.reason);
        }

        GemmRunInfo executeTranslatedGemm(ContractionProblemGemm const& problem,
                                          ContractionInputs const&      inputs,
                                          OutputSelection               outputSelection,
                                          GemmBackend                    backend)
        {
            using namespace Client::reference_adapter;

            auto translation
                = translateGemmInvocation(problem, inputs, std::move(outputSelection));
            if(std::holds_alternative<TranslationFailure>(translation))
                throwTranslationFailure(std::get<TranslationFailure>(translation));
            GemmInvocationAdapter adapter = std::move(std::get<GemmInvocationAdapter>(translation));
            return adapter.execute(backend);
        }
    } // namespace

    namespace Client
    {
        roc::host_numerics::GemmRunInfo
            executeReferenceGemm(ContractionProblemGemm const& problem,
                                 ContractionInputs const&      inputs,
                                 roc::host_numerics::OutputSelection outputSelection,
                                 roc::host_numerics::GemmBackend     backend)
        {
            return executeTranslatedGemm(
                problem, inputs, std::move(outputSelection), backend);
        }

        roc::host_numerics::GemmRunInfo
            executeReferenceGemm(ContractionProblemGemm const& problem,
                                 ContractionInputs const&      inputs,
                                 size_t                        elementsToValidate,
                                 roc::host_numerics::GemmBackend backend)
        {
            return executeReferenceGemm(problem,
                                        inputs,
                                        referenceOutputSelection(problem.d(), elementsToValidate),
                                        backend);
        }

        roc::host_numerics::GemmRunInfo SolveGemmCPU(ContractionProblemGemm const& problem,
                                                     ContractionInputs const&      inputs,
                                                     roc::host_numerics::OutputSelection
                                                         outputSelection)
        {
            using roc::host_numerics::GemmRunInfo;

            ScopedTimer timer("solve_cpu_reference");
            return executeReferenceGemm(problem,
                                        inputs,
                                        std::move(outputSelection),
                                        roc::host_numerics::GemmBackend::Automatic);
        }

        roc::host_numerics::GemmRunInfo SolveGemmCPU(ContractionProblemGemm const& problem,
                                                     ContractionInputs const&      inputs,
                                                     size_t elementsToValidate)
        {
            return SolveGemmCPU(
                problem, inputs, referenceOutputSelection(problem.d(), elementsToValidate));
        }

        void SolveCPU(ContractionProblem const* problem,
                      ProblemInputs const*      inputs,
                      std::span<const roc::host_numerics::OutputSelection> outputSelections)
        {
            if(auto groupedProblem = dynamic_cast<ContractionProblemGroupedGemm const*>(problem))
            {
                auto refInput = dynamic_cast<ContractionGroupedInputs const*>(inputs);
                if(!refInput)
                    throw std::runtime_error("Unable to cast input to ContractionGroupedInputs.");
                if(groupedProblem->gemms.size() != refInput->grouped.size())
                    throw std::runtime_error("Mismatched number of grouped problems and inputs.");
                if(groupedProblem->gemms.size() != outputSelections.size())
                    throw std::runtime_error(
                        "Mismatched number of grouped problems and output selections.");

                for(uint64_t i = 0; i < groupedProblem->gemms.size(); ++i)
                {
                    ContractionProblemGemm groupedGemm  = groupedProblem->gemms[i];
                    ContractionInputs      groupedInput = refInput->grouped[i];
                    SolveGemmCPU(groupedGemm, groupedInput, outputSelections[i]);
                }
                return;
            }

            if(auto gemmProblem = dynamic_cast<ContractionProblemGemm const*>(problem))
            {
                auto refInput = dynamic_cast<ContractionInputs const*>(inputs);
                if(!refInput)
                    throw std::runtime_error("Unable to cast input to ContractionInputs.");
                if(outputSelections.size() != 1)
                    throw std::runtime_error(
                        "An ungrouped GEMM requires exactly one output selection.");
                SolveGemmCPU(*gemmProblem, *refInput, outputSelections.front());
                return;
            }

            throw std::runtime_error("[Reference] Failed to cast to any ContractionProblem");
        }
    } // namespace Client
} // namespace TensileLite
