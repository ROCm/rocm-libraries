// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ClientRunScheduler.hpp"
#include "DataInitialization.hpp"

#include <memory>

namespace TensileLite
{
    namespace Client
    {
        // Production seam shared by the benchmark client and scheduler tests:
        // it adapts real DataInitialization state to RunDataCoordinator.
        class SchedulerDataCoordinatorAdapter final : public RunDataCoordinator
        {
        public:
            explicit SchedulerDataCoordinatorAdapter(std::shared_ptr<DataInitialization> dataInit);

            void resetPreparedSlotsForProblem() override;

            std::shared_ptr<ProblemInputs> prepareGPUInputs(ContractionProblem const* problem) override;

            std::vector<std::shared_ptr<ProblemInputs>> prepareRotatingGPUOutput(
                int32_t                        maxRotatingBufferNum,
                ContractionProblem const*      problem,
                std::shared_ptr<ProblemInputs> inputs,
                hipStream_t                   stream) override;

            void waitForPreparedSlot(hipStream_t stream) override;

            void primeNextInputSlot(ContractionProblem const* problem) override;

        private:
            std::shared_ptr<DataInitialization> m_dataInit;
        };
    } // namespace Client
} // namespace TensileLite
