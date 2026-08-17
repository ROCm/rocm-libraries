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
        bool scanSynchronizerResidue(uint8_t const* host, size_t bytes, SynchronizerResidue& out)
        {
            // Ints are both the scan step and the reported unit: the buffer is
            // declared with the alpha type but holds 32-bit counters, one per XCD
            // at the head, so an offset in ints names the counter left set. The
            // element count is a multiple of 8 and alpha is at least int-sized,
            // so bytes always divides evenly.
            size_t const    ints    = bytes / sizeof(uint32_t);
            uint32_t const* v       = reinterpret_cast<uint32_t const*>(host);
            size_t          nonzero = 0;
            size_t          first   = ints;
            for(size_t i = 0; i < ints; i++)
            {
                if(v[i] != 0)
                {
                    if(nonzero == 0)
                        first = i;
                    nonzero++;
                }
            }

            if(nonzero == 0)
                return false;

            out.totalInts   = ints;
            out.nonzeroInts = nonzero;
            out.firstInt    = first;

            return true;
        }

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
            m_checkedSolution = false;
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
            m_checkedSolution = true;
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

            // The buffer is declared with the alpha type but written as int
            // counters. A narrower alpha leaves the tail of the kernel's range
            // outside the allocation, where residue is unreadable rather than
            // merely unreported.
            if(bytes < tensor.totalAllocatedElements() * sizeof(int))
            {
                // Thrown, not reported: this is a client configuration limit, not
                // a kernel that failed to self-clean.
                std::ostringstream msg;
                msg << "Synchronizer is declared with a type narrower than int (" << bytes
                    << " bytes for " << tensor.totalAllocatedElements()
                    << " elements), so --check-streamk-sync cannot cover the range the kernel "
                       "uses.";
                throw std::runtime_error(msg.str());
            }

            // Runs once per solution, so the copy is hot: pinned staging avoids
            // the pageable bounce buffer.
            uint8_t* host = stagingBuffer(bytes);
            HIP_CHECK_EXC(hipMemcpy(host, deviceSynchronizer, bytes, hipMemcpyDeviceToHost));

            SynchronizerResidue residue;
            if(!scanSynchronizerResidue(host, bytes, residue))
                return true;

            std::ostringstream msg;
            msg << "StreamK Synchronizer left dirty after " << stage << " (gemm " << gemmIdx
                << "): " << residue.nonzeroInts << "/" << residue.totalInts
                << " ints nonzero, first at int offset " << residue.firstInt
                << " -- the kernel did not self-clean its work-queue state.\n";
            m_reporter->log(LogLevel::Error, msg.str());

            // The client zeroes the Synchronizer only on the first
            // prepareGPUInputs -- it is not an output tensor, so the per-solution
            // resetOutput skips it. Clear the residue so it is reported once
            // instead of again by every solution that follows.
            HIP_CHECK_EXC(hipMemset(deviceSynchronizer, 0, bytes));

            return false;
        }
    } // namespace Client
} // namespace TensileLite
