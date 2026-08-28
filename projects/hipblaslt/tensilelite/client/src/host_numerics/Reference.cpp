// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Product-private TensileLite execution policy around the descriptor adapter.

#include <roc/host_numerics/adapters/tensilelite/GemmInvocationAdapter.hpp>
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
        using Client::reference_adapter::TranslatedGemmBatch;
        using Client::reference_adapter::TranslationFailure;

        class RunInfoRecorder
        {
        public:
            void record(const GemmRunInfo& runInfo)
            {
                if(!m_hasRunInfo)
                {
                    m_combined.backendUsed    = runInfo.backendUsed;
                    m_combined.fallbackReason = runInfo.fallbackReason;
                    m_hasRunInfo              = true;
                }
                else if(m_combined.backendUsed != runInfo.backendUsed)
                {
                    m_combined.backendUsed = GemmBackend::Mixed;
                }
                m_combined.outputElementsWritten += runInfo.outputElementsWritten;
                m_combined.outputElementsCovered += runInfo.outputElementsCovered;
                if(!m_combined.fallbackReason && runInfo.fallbackReason)
                    m_combined.fallbackReason = runInfo.fallbackReason;
            }

            GemmRunInfo result() const
            {
                return m_combined;
            }

        private:
            GemmRunInfo m_combined;
            bool        m_hasRunInfo = false;
        };

        void executeGemmBatch(TranslatedGemmBatch& translated,
                              GemmBackend          backend,
                              RunInfoRecorder&     recorder)
        {
            recorder.record(referenceGemm(translated.gemm(), backend));
        }

        [[noreturn]] void throwTranslationFailure(const TranslationFailure& failure)
        {
            throw std::invalid_argument("TensileLite host-numerics translation failed: "
                                        + failure.reason);
        }

        GemmRunInfo executeTranslatedGemm(ContractionProblemGemm const& problem,
                                          ContractionInputs const&      inputs,
                                          size_t                        elementsToValidate,
                                          GemmBackend                    backend)
        {
            using namespace Client::reference_adapter;

            auto translation = translateGemmInvocation(problem, inputs, elementsToValidate);
            if(std::holds_alternative<TranslationFailure>(translation))
                throwTranslationFailure(std::get<TranslationFailure>(translation));
            GemmInvocationAdapter adapter = std::move(std::get<GemmInvocationAdapter>(translation));

            RunInfoRecorder recorder;

            for(size_t batch = 0; batch < adapter.batchCount(); ++batch)
            {
                auto batchTranslation = adapter.translateBatch(batch);
                if(std::holds_alternative<TranslationFailure>(batchTranslation))
                    throwTranslationFailure(std::get<TranslationFailure>(batchTranslation));
                TranslatedGemmBatch translated
                    = std::move(std::get<TranslatedGemmBatch>(batchTranslation));

                executeGemmBatch(translated, backend, recorder);
                translated.runPostGemmOperationsAndCopyOutputs();
            }
            return recorder.result();
        }
    } // namespace

    namespace Client
    {
        roc::host_numerics::GemmRunInfo
            executeReferenceGemm(ContractionProblemGemm const& problem,
                                 ContractionInputs const&      inputs,
                                 size_t                        elementsToValidate,
                                 roc::host_numerics::GemmBackend backend)
        {
            return executeTranslatedGemm(problem, inputs, elementsToValidate, backend);
        }

        roc::host_numerics::GemmRunInfo SolveGemmCPU(ContractionProblemGemm const& problem,
                                                       ContractionInputs const&      inputs,
                                                       size_t elementsToValidate)
        {
            using roc::host_numerics::GemmRunInfo;

            ScopedTimer timer("solve_cpu_reference");
            return executeReferenceGemm(problem,
                                        inputs,
                                        elementsToValidate,
                                        roc::host_numerics::GemmBackend::Automatic);
        }

        void SolveCPU(ContractionProblem const* problem,
                      ProblemInputs const*      inputs,
                      size_t                    elementsToValidate)
        {
            if(auto groupedProblem = dynamic_cast<ContractionProblemGroupedGemm const*>(problem))
            {
                auto refInput = dynamic_cast<ContractionGroupedInputs const*>(inputs);
                if(!refInput)
                    throw std::runtime_error("Unable to cast input to ContractionGroupedInputs.");
                if(groupedProblem->gemms.size() != refInput->grouped.size())
                    throw std::runtime_error("Mismatched number of grouped problems and inputs.");

                for(uint64_t i = 0; i < groupedProblem->gemms.size(); ++i)
                {
                    ContractionProblemGemm groupedGemm  = groupedProblem->gemms[i];
                    ContractionInputs      groupedInput = refInput->grouped[i];
                    SolveGemmCPU(groupedGemm, groupedInput, elementsToValidate);
                }
                return;
            }

            if(auto gemmProblem = dynamic_cast<ContractionProblemGemm const*>(problem))
            {
                auto refInput = dynamic_cast<ContractionInputs const*>(inputs);
                if(!refInput)
                    throw std::runtime_error("Unable to cast input to ContractionInputs.");
                SolveGemmCPU(*gemmProblem, *refInput, elementsToValidate);
                return;
            }

            throw std::runtime_error("[Reference] Failed to cast to any ContractionProblem");
        }
    } // namespace Client
} // namespace TensileLite
