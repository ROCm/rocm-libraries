// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*! \file
 * \brief Post-launch dirty-buffer check for the shared Synchronizer buffer.
 *
 * The standalone benchmark client reuses each problem's Synchronizer
 * buffer across solutions. Stream-K uses it for queue counters and fixup
 * flags; GSU MultipleBufferSingleKernel and output-amax use it for
 * synchronization. Each consumer must leave the buffer at zero on exit.
 * Residue is silent -- it corrupts a later launch, not the one that left it --
 * so this listener reads the buffer back, fails the run on any nonzero int,
 * and clears it so it is reported once. Buffers declared too narrow to scan
 * in full produce a warning and are skipped.
 *
 * The check covers every launch of a solution: main hands each run
 * iteration's inputs to beginRunIteration() before its launches and calls
 * checkRunIteration() at its end (GuardPageAll solutions run two iterations,
 * and the second one places the buffer at a different address), and
 * postSolution() checks launches an exception left unchecked, then reports
 * the solution's verdict.
 *
 * On by default (GlobalParameters CheckSynchronizer); --check-synchronizer=0
 * turns it off. A default must not change what a config does, so the listener
 * is passive and skips solutions known not to use the Synchronizer.
 */

#pragma once

#include "RunListener.hpp"

#include "ProgramOptions.hpp"

#include <Tensile/ContractionProblem.hpp>

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <memory>
#include <vector>

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
            virtual void preBenchmarkRun() override {}
            virtual void postBenchmarkRun() override {}

            virtual void preProblem(ContractionProblem* const problem) override;
            virtual void postProblem() override {}

            virtual void preSolution(ContractionSolution* const solution) override;
            // Checks launches the end-of-iteration check missed (an exception
            // ended the iteration), then reports. Never throws.
            virtual void postSolution() override;

            /// Inputs the run iteration's kernels are solved with (one per
            /// rotating buffer), before its launches.
            void beginRunIteration(std::vector<std::shared_ptr<ProblemInputs>> const& inputs);

            /// Checks the buffers after the launches of one run iteration,
            /// before the next iteration's input reset. Does nothing if nothing
            /// launched since the last check. Throws on a HIP error (e.g. a
            /// kernel fault surfacing in the readback), after marking the
            /// solution failed.
            void checkRunIteration();

            // Passive: inspect launches other listeners drive, never request
            // one. Configs with NumElementsToValidate 0 and SyncsPerBenchmark 0
            // launch no kernels by design; requesting a run there would cost
            // NumWarmups launches per solution and observe nothing. Where
            // nothing launches there is no residue, so this costs no coverage.
            virtual bool needMoreRunsInSolution() const override
            {
                return false;
            }
            virtual size_t numWarmupRuns() override
            {
                return 0;
            }
            virtual void setNumWarmupRuns(size_t count) override {}
            virtual void preWarmup() override
            {
                m_launchedSinceCheck = true;
            }
            virtual void postWarmup(TimingEvents const& startEvents,
                                    TimingEvents const& stopEvents,
                                    hipStream_t const&  stream) override
            {
            }
            // Not a check point: later launches of the iteration would go
            // unobserved (see checkRunIteration()).
            virtual void validateWarmups(std::shared_ptr<ProblemInputs> inputs,
                                         TimingEvents const&            startEvents,
                                         TimingEvents const&            stopEvents) override
            {
            }

#if TENSILELITE_CLIENT_ENABLE_ROCPROFSDK
            virtual void preProfiler() override
            {
                m_launchedSinceCheck = true;
            }
#endif

            virtual size_t numSyncs() override
            {
                return 0;
            }
            virtual void setNumSyncs(size_t count) override {}
            virtual void preSyncs() override {}
            virtual void postSyncs() override {}

            virtual size_t numEnqueuesPerSync() override
            {
                return 0;
            }
            virtual void setNumEnqueuesPerSync(size_t count) override {}
            virtual void preEnqueues(hipStream_t const& stream) override
            {
                m_launchedSinceCheck = true;
            }
            virtual void postEnqueues(TimingEvents const& startEvents,
                                      TimingEvents const& stopEvents,
                                      hipStream_t const&  stream) override
            {
            }
            // Deliberately unchecked: a readback here would synchronize the
            // stream, perturbing the back-to-back cadence these enqueues exercise.
            virtual void validateEnqueues(std::shared_ptr<ProblemInputs> inputs,
                                          TimingEvents const&            startEvents,
                                          TimingEvents const&            stopEvents) override
            {
            }

            virtual void finalizeReport() override {}

            virtual int error() const override
            {
                return m_errorsReported;
            }

        private:
            /// Checks the launches since the last check, if any (with the
            /// iteration's inputs); marks the solution failed if dirty or if
            /// the check throws, and rethrows.
            void checkLaunches(char const* stage);

            /// Checks every distinct Synchronizer buffer reachable from the
            /// iteration's inputs (rotating buffers may share it or not).
            /// `stage` names the launch phase, for the diagnostic.
            void checkInputs(char const* stage);

            /// Reads back one buffer, re-zeroing it only when dirty.
            /// Returns false when dirty.
            bool checkBuffer(ContractionProblemGemm const& problem,
                             void*                         deviceSynchronizer,
                             char const*                   stage,
                             size_t                        gemmIdx);

            /// Pinned host staging buffer of at least *bytes*, grown on demand
            /// and reused across launches.
            uint8_t* stagingBuffer(size_t bytes);

            bool                m_enabled      = false;
            // One narrow-alpha warning per run, not one per solution.
            bool                m_narrowAlphaWarned = false;
            ContractionProblem* m_problem      = nullptr;
            uint8_t*            m_staging      = nullptr;
            size_t              m_stagingBytes = 0;
            // The current run iteration's inputs: its buffer addresses depend on
            // the iteration (GuardPageFront / GuardPageBack).
            std::vector<std::shared_ptr<ProblemInputs>> m_iterationInputs;
            // Set by the pre-launch hooks, cleared by a check.
            bool m_launchedSinceCheck = false;

        protected:
            // Keep device transfers overridable for host-only lifecycle tests.
            virtual uint8_t* readBuffer(void* device, size_t bytes);
            virtual void clearBuffer(void* device, size_t bytes);

            /// Whether the check should do anything right now: switched on, and
            /// the solution in hand may use the Synchronizer.
            bool active() const
            {
                return m_enabled && m_mayUseSynchronizer;
            }

            // Protected so a test-only subclass can drive these without a GPU.
            bool m_failedInSolution = false;
            int  m_errorsReported  = 0;
            // Whether the solution in hand may use the Synchronizer. True for
            // an unknown solution, so an unrecognised launch is scanned rather
            // than silently skipped.
            bool m_mayUseSynchronizer = true;
        };
    } // namespace Client
} // namespace TensileLite
