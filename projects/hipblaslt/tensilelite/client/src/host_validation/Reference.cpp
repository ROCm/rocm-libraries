// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Product-private TensileLite execution policy around the descriptor adapter.

#include <roc/host_validation/adapters/tensilelite/GemmProblemAdapter.hpp>
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
        using Client::reference_adapter::GemmProblemAdapter;
        using Client::reference_adapter::TranslatedGemmBatch;
        using Client::reference_adapter::TranslationFailure;

        enum class RuntimeGemmExecution
        {
            Pointwise,
            BlockedPreferred,
            BlockedRequired,
        };

        class RuntimeGemmPolicy
        {
        public:
            explicit RuntimeGemmPolicy(RuntimeGemmExecution execution)
                : m_execution(execution)
            {
            }

            ScalarType accumulatorType(ScalarType operationType) const
            {
                if(m_execution != RuntimeGemmExecution::Pointwise
                   && (operationType == ScalarType::Float16
                       || operationType == ScalarType::BFloat16))
                    return ScalarType::Float32;
                return operationType;
            }

            bool requestsBlockedBackend() const
            {
                return m_execution != RuntimeGemmExecution::Pointwise;
            }

            bool requiresBlockedBackend() const
            {
                return m_execution == RuntimeGemmExecution::BlockedRequired;
            }

        private:
            RuntimeGemmExecution m_execution;
        };

        BlockedGemmBackend const& blockedGemmBackend()
        {
            static const BlockedGemmBackend backend;
            return backend;
        }

        class RunInfoRecorder
        {
        public:
            explicit RunInfoRecorder(GemmRunInfo* combined)
                : m_combined(combined)
            {
                if(m_combined != nullptr)
                    *m_combined = {};
            }

            void record(const GemmRunInfo& runInfo)
            {
                if(m_combined == nullptr)
                    return;
                if(!m_hasRunInfo)
                {
                    m_combined->backendUsed    = runInfo.backendUsed;
                    m_combined->fallbackReason = runInfo.fallbackReason;
                    m_hasRunInfo               = true;
                }
                else if(m_combined->backendUsed != runInfo.backendUsed)
                {
                    throw std::runtime_error(
                        "Reference GEMM batches selected different host backends.");
                }
                m_combined->outputElementsWritten += runInfo.outputElementsWritten;
                m_combined->outputElementsCovered += runInfo.outputElementsCovered;
                if(!m_combined->fallbackReason && runInfo.fallbackReason)
                    m_combined->fallbackReason = runInfo.fallbackReason;
            }

        private:
            GemmRunInfo* m_combined   = nullptr;
            bool         m_hasRunInfo = false;
        };

        bool executeGemmBatch(ContractionProblemGemm const& problem,
                              GemmProblemAdapter const&     adapter,
                              TranslatedGemmBatch&          translated,
                              RuntimeGemmPolicy const&      policy,
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
                    recorder.record(
                        referenceGemm(request, execution, &blockedGemmBackend()).runInfo);
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
                GemmResult result = referenceGemm(request,
                                                  {
                                                      .backend = GemmBackend::Pointwise,
                                                      .requireRequestedBackend = true,
                                                  });
                result.runInfo.fallbackReason = support.reason;
                recorder.record(result.runInfo);
                return true;
            }

            recorder.record(referenceGemm(request,
                                          {
                                              .backend                 = GemmBackend::Pointwise,
                                              .requireRequestedBackend = true,
                                          })
                                .runInfo);
            return true;
        }

        bool tryRuntimeGemm(ContractionProblemGemm const&      problem,
                            ContractionInputs const&           inputs,
                            size_t                             elementsToValidate,
                            RuntimeGemmExecution               execution,
                            roc::host_validation::GemmRunInfo* combinedRunInfo = nullptr)
        {
            using namespace Client::reference_adapter;

            auto translation = translateGemmProblem(problem, inputs, elementsToValidate);
            if(std::holds_alternative<TranslationFailure>(translation))
                return false;
            GemmProblemAdapter adapter = std::move(std::get<GemmProblemAdapter>(translation));

            const RuntimeGemmPolicy policy(execution);
            const ScalarType        accumulatorType
                = policy.accumulatorType(adapter.operationAccumulatorType());
            RunInfoRecorder recorder(combinedRunInfo);

            for(size_t batch = 0; batch < adapter.batchCount(); ++batch)
            {
                auto batchTranslation = adapter.translateBatch(batch, accumulatorType);
                if(std::holds_alternative<TranslationFailure>(batchTranslation))
                    return false;
                TranslatedGemmBatch translated
                    = std::move(std::get<TranslatedGemmBatch>(batchTranslation));

                if(!executeGemmBatch(problem, adapter, translated, policy, recorder))
                    return false;
                if(translated.epilogue)
                    referenceEpilogue(*translated.epilogue);
                if(translated.biasReduction)
                    referenceSum(*translated.biasReduction);
                translated.copyOutputs();
            }
            return true;
        }
    } // namespace

    namespace Client
    {
        bool tryRuntimePointwiseGemm(ContractionProblemGemm const& problem,
                                     ContractionInputs const&      inputs,
                                     size_t                        elementsToValidate)
        {
            return tryRuntimeGemm(
                problem, inputs, elementsToValidate, RuntimeGemmExecution::Pointwise);
        }

        bool tryRuntimeBlockedGemm(ContractionProblemGemm const& problem,
                                   ContractionInputs const&      inputs,
                                   size_t                        elementsToValidate)
        {
            return tryRuntimeGemm(
                problem, inputs, elementsToValidate, RuntimeGemmExecution::BlockedRequired);
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

            GemmRunInfo runInfo;
            if(!preserveStepwiseAccumulator)
            {
                ScopedTimer timer("solve_cpu_fast");
                if(tryRuntimeGemm(problem,
                                  inputs,
                                  elementsToValidate,
                                  RuntimeGemmExecution::BlockedPreferred,
                                  &runInfo))
                    return runInfo;
            }
            else if(tryRuntimeGemm(problem,
                                   inputs,
                                   elementsToValidate,
                                   RuntimeGemmExecution::Pointwise,
                                   &runInfo))
                return runInfo;

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
