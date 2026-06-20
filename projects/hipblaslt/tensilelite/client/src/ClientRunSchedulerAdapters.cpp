// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ClientRunSchedulerAdapters.hpp"

namespace TensileLite
{
    namespace Client
    {
        SchedulerDataCoordinatorAdapter::SchedulerDataCoordinatorAdapter(
            std::shared_ptr<DataInitialization> dataInit)
            : m_dataInit(std::move(dataInit))
        {
        }

        void SchedulerDataCoordinatorAdapter::resetPreparedSlotsForProblem()
        {
            m_dataInit->resetPreparedSlotsForProblem();
        }

        std::shared_ptr<ProblemInputs>
            SchedulerDataCoordinatorAdapter::prepareGPUInputs(ContractionProblem const* problem)
        {
            return m_dataInit->prepareGPUInputs(problem);
        }

        std::vector<std::shared_ptr<ProblemInputs>>
            SchedulerDataCoordinatorAdapter::prepareRotatingGPUOutput(
                int32_t                        maxRotatingBufferNum,
                ContractionProblem const*      problem,
                std::shared_ptr<ProblemInputs> inputs,
                hipStream_t                   stream)
        {
            return m_dataInit->prepareRotatingGPUOutput(
                maxRotatingBufferNum, problem, std::move(inputs), stream);
        }

        void SchedulerDataCoordinatorAdapter::waitForPreparedSlot(hipStream_t stream)
        {
            m_dataInit->waitForPreparedSlot(stream);
        }

        void SchedulerDataCoordinatorAdapter::primeNextInputSlot(ContractionProblem const* problem)
        {
            m_dataInit->primeNextInputSlot(problem);
        }
    } // namespace Client
} // namespace TensileLite
