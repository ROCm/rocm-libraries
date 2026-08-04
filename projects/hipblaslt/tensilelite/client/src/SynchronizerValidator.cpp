// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "SynchronizerValidator.hpp"

#include "ResultReporter.hpp"

#include <Tensile/hip/HipUtils.hpp>

#include <sstream>

namespace TensileLite
{
    namespace Client
    {
        SynchronizerValidator::SynchronizerValidator(po::variables_map const& args)
            : m_enabled(args["check-streamk-sync"].as<bool>())
        {
        }

        void SynchronizerValidator::preProblem(ContractionProblem* const problem)
        {
            m_problem = problem;
        }

        void SynchronizerValidator::preSolution(ContractionSolution* const solution)
        {
            m_dirtyInSolution = false;
        }

        void SynchronizerValidator::postSolution()
        {
            if(m_dirtyInSolution)
            {
                m_errorsReported++;
                // Overrides the reference verdict so the result row, and the CSV
                // and library logic fed from it, cannot crown a solution that
                // corrupts the shared buffer.
                m_reporter->report(ResultKey::Validation, "FAILED");
            }

            m_dirtyInSolution = false;
        }

        void SynchronizerValidator::validateWarmups(std::shared_ptr<ProblemInputs> inputs,
                                                    TimingEvents const&            startEvents,
                                                    TimingEvents const&            stopEvents)
        {
            checkInputs(inputs, "warmup");
        }

        void SynchronizerValidator::checkInputs(std::shared_ptr<ProblemInputs> inputs,
                                                char const*                    stage)
        {
            if(!m_enabled || m_problem == nullptr || inputs == nullptr)
                return;

            if(auto problems = dynamic_cast<ContractionProblemGroupedGemm*>(m_problem))
            {
                auto const& result = dynamic_cast<ContractionGroupedInputs const&>(*inputs);
                for(size_t j = 0; j < problems->gemms.size(); j++)
                {
                    if(!checkBuffer(problems->gemms[j], result.grouped[j].Synchronizer, stage, j))
                        m_dirtyInSolution = true;
                }
            }
            else if(auto problem = dynamic_cast<ContractionProblemGemm*>(m_problem))
            {
                auto const& result = dynamic_cast<ContractionInputs const&>(*inputs);
                if(!checkBuffer(*problem, result.Synchronizer, stage, 0))
                    m_dirtyInSolution = true;
            }
            else
            {
                throw std::runtime_error("Failed to cast to any ContractionProblem.");
            }
        }

        SynchronizerValidator::~SynchronizerValidator()
        {
            if(m_staging != nullptr)
                static_cast<void>(hipHostFree(m_staging));
        }

        uint8_t* SynchronizerValidator::stagingBuffer(size_t bytes)
        {
            if(m_stagingBytes >= bytes)
                return m_staging;

            if(m_staging != nullptr)
                HIP_CHECK_EXC(hipHostFree(m_staging));

            m_staging      = nullptr;
            m_stagingBytes = 0;
            HIP_CHECK_EXC(hipHostMalloc(&m_staging, bytes));
            m_stagingBytes = bytes;

            return m_staging;
        }

        bool SynchronizerValidator::checkBuffer(ContractionProblemGemm const& problem,
                                                void*                         deviceSynchronizer,
                                                char const*                   stage,
                                                size_t                        gemmIdx)
        {
            if(deviceSynchronizer == nullptr)
                return true;

            auto const&  tensor = problem.tensors()[ContractionProblemGemm::TENSOR::Synchronizer];
            size_t const bytes  = tensor.totalAllocatedBytes();
            if(bytes == 0)
                return true;

            // Runs once per solution, so the copy is hot: pinned staging avoids
            // the pageable bounce buffer.
            uint8_t* host = stagingBuffer(bytes);
            HIP_CHECK_EXC(hipMemcpy(host, deviceSynchronizer, bytes, hipMemcpyDeviceToHost));

            // Raw bytes, not the tensor type: the buffer is declared with the alpha
            // type but holds integer counters. Word-at-a-time on the clean path;
            // the per-byte tally below only runs once a nonzero word is found.
            size_t const    words = bytes / sizeof(uint64_t);
            bool            dirty = false;
            uint64_t const* w     = reinterpret_cast<uint64_t const*>(host);
            for(size_t i = 0; i < words && !dirty; i++)
                dirty = (w[i] != 0);
            for(size_t i = words * sizeof(uint64_t); i < bytes && !dirty; i++)
                dirty = (host[i] != 0);

            if(!dirty)
                return true;

            size_t nonzero = 0;
            size_t first   = bytes;
            for(size_t i = 0; i < bytes; i++)
            {
                if(host[i] != 0)
                {
                    if(nonzero == 0)
                        first = i;
                    nonzero++;
                }
            }

            std::ostringstream msg;
            msg << "StreamK Synchronizer left dirty after " << stage << " (gemm " << gemmIdx
                << "): " << nonzero << "/" << bytes << " bytes nonzero, first at byte offset "
                << first << " -- the kernel did not self-clean its work-queue state.\n";
            m_reporter->log(LogLevel::Error, msg.str());

            // Only reached when residue was found, so the clean path pays nothing.
            // The client zeroes the Synchronizer once per problem (it is not an
            // output tensor, so the per-solution resetOutput skips it), so without
            // this every later solution would inherit and re-report this residue.
            HIP_CHECK_EXC(hipMemset(deviceSynchronizer, 0, bytes));

            return false;
        }
    } // namespace Client
} // namespace TensileLite
