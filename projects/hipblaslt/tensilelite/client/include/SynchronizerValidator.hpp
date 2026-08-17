// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*! \file
 * \brief Post-launch dirty-buffer check for the StreamK Synchronizer buffer.
 *
 * StreamK kernels share one Synchronizer allocation across launches and must
 * leave it at zero on exit so the next launch starts clean. Residue is silent
 * -- it corrupts a later launch, not the one that left it -- so this listener
 * reads the buffer back and fails the run on any nonzero byte, clearing the
 * residue so it is reported once rather than by every solution that follows.
 * It also fails when the buffer is declared too narrow to scan in full.
 *
 * Enabled by --check-streamk-sync (GlobalParameters CheckStreamKSync).
 */

#pragma once

#include "RunListener.hpp"

#include "ProgramOptions.hpp"

#include <Tensile/ContractionProblem.hpp>

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>

namespace TensileLite
{
    namespace Client
    {
        struct SynchronizerResidue
        {
            size_t totalInts   = 0;
            size_t nonzeroInts = 0;
            size_t firstInt    = 0;
        };

        /// Scans *bytes* of *host* for nonzero data. Returns false when clean,
        /// leaving *out* untouched; otherwise fills *out* and returns true.
        bool scanSynchronizerResidue(uint8_t const* host, size_t bytes, SynchronizerResidue& out);

        class SynchronizerValidator : public RunListener
        {
        public:
            SynchronizerValidator(po::variables_map const& args);
            virtual ~SynchronizerValidator();

            virtual bool needMoreBenchmarkRuns() const override
            {
                return false;
            }
            virtual void preBenchmarkRun() override { }
            virtual void postBenchmarkRun() override { }

            virtual void preProblem(ContractionProblem* const problem) override;
            virtual void postProblem() override { }

            virtual void preSolution(ContractionSolution* const solution) override;
            virtual void postSolution() override;

            virtual bool needMoreRunsInSolution() const override
            {
                return false;
            }

            // Request one warmup so validateWarmups always has a launch to
            // inspect, even when no other listener asks for warmups.
            virtual size_t numWarmupRuns() override
            {
                return m_enabled ? 1 : 0;
            }
            virtual void setNumWarmupRuns(size_t count) override { }
            virtual void preWarmup() override { }
            virtual void postWarmup(TimingEvents const& startEvents,
                                    TimingEvents const& stopEvents,
                                    hipStream_t const&  stream) override
            {
            }
            // Runs after the first warmup launch. Any further warmups other
            // listeners ask for run past this point and are not observed, so a
            // check covers the launches since the previous one.
            virtual void validateWarmups(std::shared_ptr<ProblemInputs> inputs,
                                         TimingEvents const&            startEvents,
                                         TimingEvents const&            stopEvents) override;

            virtual size_t numSyncs() override
            {
                return 0;
            }
            virtual void setNumSyncs(size_t count) override { }
            virtual void preSyncs() override { }
            virtual void postSyncs() override { }

            virtual size_t numEnqueuesPerSync() override
            {
                return 0;
            }
            virtual void setNumEnqueuesPerSync(size_t count) override { }
            virtual void preEnqueues(hipStream_t const& stream) override { }
            virtual void postEnqueues(TimingEvents const& startEvents,
                                      TimingEvents const& stopEvents,
                                      hipStream_t const&  stream) override
            {
            }
            // Deliberately unchecked. Back-to-back launches are where work-queue
            // races surface, but the readback synchronizes the stream, so checking
            // here would perturb the very cadence the enqueues exercise.
            virtual void validateEnqueues(std::shared_ptr<ProblemInputs> inputs,
                                          TimingEvents const&            startEvents,
                                          TimingEvents const&            stopEvents) override
            {
            }

            virtual void finalizeReport() override { }

            virtual int error() const override
            {
                return m_errorsReported;
            }

        private:
            /// Checks every Synchronizer buffer reachable from *inputs*.
            /// `stage` names the launch phase, for the diagnostic.
            void checkInputs(std::shared_ptr<ProblemInputs> inputs, char const* stage);

            /// Reads back one buffer, re-zeroing it only if it was left dirty.
            /// Returns false when it was left dirty.
            bool checkBuffer(ContractionProblemGemm const& problem,
                             void*                         deviceSynchronizer,
                             char const*                   stage,
                             size_t                        gemmIdx);

            /// Pinned host staging buffer of at least *bytes*, grown on demand
            /// and reused across launches.
            uint8_t* stagingBuffer(size_t bytes);

            bool                m_enabled         = false;
            ContractionProblem* m_problem         = nullptr;
            uint8_t*            m_staging         = nullptr;
            size_t              m_stagingBytes    = 0;

        protected:
            // Protected so a test-only subclass can drive these without a GPU.
            bool m_dirtyInSolution = false;
            int  m_errorsReported  = 0;
        };
    } // namespace Client
} // namespace TensileLite
