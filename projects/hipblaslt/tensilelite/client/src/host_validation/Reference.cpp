// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Product-private TensileLite execution policy around the descriptor adapter.

#include <roc/host_validation/adapters/tensilelite/GemmInvocationAdapter.hpp>
#include <roc/host_validation/adapters/tensilelite/Reference.hpp>
#include <roc/host_validation/backends/blocked.hpp>
#include <roc/host_validation/validation.hpp>

#include <Tensile/Utils.hpp>

#include "TimingInstrumentation.hpp"

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
#include <variant>

namespace TensileLite
{
    namespace
    {
        using namespace roc::host_validation;
        using Client::reference_adapter::GemmInvocationAdapter;
        using Client::reference_adapter::TranslatedGemmBatch;
        using Client::reference_adapter::TranslationFailure;
        using Client::ReferenceGemmExecution;

        class ReferenceGemmPolicy
        {
        public:
            explicit ReferenceGemmPolicy(ReferenceGemmExecution execution)
                : m_execution(execution)
            {
            }

            ScalarType accumulatorType(ScalarType operationType) const
            {
                if(m_execution != ReferenceGemmExecution::Pointwise
                   && (operationType == ScalarType::Float16
                       || operationType == ScalarType::BFloat16))
                    return ScalarType::Float32;
                return operationType;
            }

            bool requestsBlockedBackend() const
            {
                return m_execution != ReferenceGemmExecution::Pointwise;
            }

            bool requiresBlockedBackend() const
            {
                return m_execution == ReferenceGemmExecution::BlockedRequired;
            }

        private:
            ReferenceGemmExecution m_execution;
        };

        BlockedGemmBackend const& blockedGemmBackend()
        {
            static const BlockedGemmBackend backend;
            return backend;
        }

        class RunInfoRecorder
        {
        public:
            void record(const GemmRunInfo& runInfo)
            {
                if(!m_hasRunInfo)
                {
                    m_combined.backendUsed    = runInfo.backendUsed;
                    m_combined.fallbackReason = runInfo.fallbackReason;
                    m_hasRunInfo               = true;
                }
                else if(m_combined.backendUsed != runInfo.backendUsed)
                {
                    throw std::runtime_error(
                        "Reference GEMM batches selected different host backends.");
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

        bool executeGemmBatch(ContractionProblemGemm const& problem,
                              GemmInvocationAdapter const&  adapter,
                              TranslatedGemmBatch&          translated,
                              ReferenceGemmPolicy const&    policy,
                              RunInfoRecorder&              recorder)
        {
            GemmRequest& request = translated.gemm();
            if(policy.requestsBlockedBackend())
            {
                const GemmExecution execution = {
                    .backend                 = GemmBackend::Blocked,
                    .requireRequestedBackend = true,
                };
                const GemmSupportInfo support
                    = queryGemmSupport(request, execution, &blockedGemmBackend());
                if(support)
                {
                    recorder.record(referenceGemm(request, execution, &blockedGemmBackend()));
                    return true;
                }
                if(policy.requiresBlockedBackend())
                    return false;
                if(adapter.usesStandaloneEpilogue()
                   && adapter.operationAccumulatorType() != request.accumulatorType)
                    return false;

                request.accumulatorType = adapter.operationAccumulatorType();
                request.mathMode  = request.accumulatorType == ScalarType::Float32
                                            && problem.f32XdlMathOp() == rocisa::DataType::XFloat32
                                        ? MathMode::XFloat32
                                        : MathMode::Default;
                GemmRunInfo runInfo = referenceGemm(request,
                                                    {
                                                        .backend = GemmBackend::Pointwise,
                                                        .requireRequestedBackend = true,
                                                    });
                runInfo.fallbackReason = support.reason;
                recorder.record(runInfo);
                return true;
            }

            recorder.record(referenceGemm(request,
                                          {
                                              .backend                 = GemmBackend::Pointwise,
                                              .requireRequestedBackend = true,
                                          }));
            return true;
        }

        std::optional<GemmRunInfo> tryTranslatedGemm(ContractionProblemGemm const& problem,
                                                     ContractionInputs const&      inputs,
                                                     size_t elementsToValidate,
                                                     ReferenceGemmExecution execution)
        {
            using namespace Client::reference_adapter;

            auto translation = translateGemmInvocation(problem, inputs, elementsToValidate);
            if(std::holds_alternative<TranslationFailure>(translation))
                return std::nullopt;
            GemmInvocationAdapter adapter = std::move(std::get<GemmInvocationAdapter>(translation));

            const ReferenceGemmPolicy policy(execution);
            const ScalarType          accumulatorType
                = policy.accumulatorType(adapter.operationAccumulatorType());
            RunInfoRecorder recorder;

            for(size_t batch = 0; batch < adapter.batchCount(); ++batch)
            {
                auto batchTranslation = adapter.translateBatch(batch, accumulatorType);
                if(std::holds_alternative<TranslationFailure>(batchTranslation))
                    return std::nullopt;
                TranslatedGemmBatch translated
                    = std::move(std::get<TranslatedGemmBatch>(batchTranslation));

                if(!executeGemmBatch(problem, adapter, translated, policy, recorder))
                    return std::nullopt;
                if(translated.epilogue)
                    referenceEpilogue(*translated.epilogue);
                if(translated.biasReduction)
                    referenceSum(*translated.biasReduction);
                translated.copyOutputs();
            }
            return recorder.result();
        }
    } // namespace

    namespace Client
    {
        std::optional<roc::host_validation::GemmRunInfo>
            tryReferenceGemm(ContractionProblemGemm const& problem,
                             ContractionInputs const&      inputs,
                             size_t                        elementsToValidate,
                             ReferenceGemmExecution        execution)
        {
            return tryTranslatedGemm(problem, inputs, elementsToValidate, execution);
        }

        roc::host_validation::GemmRunInfo SolveGemmCPU(ContractionProblemGemm const& problem,
                                                       ContractionInputs const&      inputs,
                                                       size_t elementsToValidate)
        {
            using roc::host_validation::GemmRunInfo;

            const bool partialSelection
                = elementsToValidate != 0
                  && elementsToValidate < problem.d().totalLogicalElements();
            const bool preserveStepwiseAccumulator
                = partialSelection
                  && (problem.computeType() == rocisa::DataType::Half
                      || problem.computeType() == rocisa::DataType::BFloat16);

            if(!preserveStepwiseAccumulator)
            {
                ScopedTimer timer("solve_cpu_fast");
                if(const auto runInfo = tryReferenceGemm(problem,
                                                        inputs,
                                                        elementsToValidate,
                                                        ReferenceGemmExecution::BlockedPreferred))
                    return *runInfo;
            }
            else if(const auto runInfo
                    = tryReferenceGemm(problem,
                                       inputs,
                                       elementsToValidate,
                                       ReferenceGemmExecution::Pointwise))
                return *runInfo;

            throw std::runtime_error(
                concatenate("Unsupported host-validation GEMM descriptor: ",
                            problem.operationIdentifier(),
                            ". The product-local typed reference fallback has been disabled."));
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
