// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*! \file
 * \brief Gate check for TriggeredGateEpilogue kernels (--triggered-gate-epilogue).
 *
 * Builds the gate table the kernels read, zeroes the gates before the first
 * warmup launch and, after it, checks that every band's gate equals the number
 * of work-groups that store into the band. Requests that launch itself.
 */

#pragma once

#include "RunListener.hpp"

#include "ProgramOptions.hpp"

#include <Tensile/ContractionProblem.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace TensileLite
{
    namespace Client
    {
        struct GateTableLayout
        {
            size_t   bands       = 0;
            uint32_t publishers  = 0; // expected gate value after one launch
            size_t   gatesOffset = 0;
            size_t   bytes       = 0;
        };

        GateTableLayout gateTableLayout(size_t bands, uint32_t publishers);

        /// Host image of the table for a buffer at *deviceBase*; gates start at zero.
        void fillGateTable(std::vector<uint8_t>&  host,
                           GateTableLayout const& layout,
                           uint64_t               deviceBase);

        class TriggeredGateValidator : public RunListener
        {
        public:
            TriggeredGateValidator(po::variables_map const& args);

            virtual bool needMoreBenchmarkRuns() const override
            {
                return false;
            }
            virtual void preBenchmarkRun() override {}
            virtual void postBenchmarkRun() override {}

            virtual void preProblem(ContractionProblem* const problem) override
            {
                m_problem = problem;
            }
            virtual void postProblem() override {}

            virtual void preSolution(ContractionSolution* const solution) override;
            virtual void postSolution() override;

            virtual bool needMoreRunsInSolution() const override
            {
                return m_active && !m_checked;
            }
            virtual size_t numWarmupRuns() override
            {
                return m_active && !m_checked ? 1 : 0;
            }
            virtual void setNumWarmupRuns(size_t count) override {}
            virtual void preWarmup() override;
            virtual void postWarmup(TimingEvents const& startEvents,
                                    TimingEvents const& stopEvents) override
            {
            }
            virtual void validateWarmups(std::vector<size_t> const& slots,
                                         TimingEvents const&        startEvents,
                                         TimingEvents const&        stopEvents) override;

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
            virtual void preEnqueues() override {}
            virtual void postEnqueues(TimingEvents const& startEvents,
                                      TimingEvents const& stopEvents) override
            {
            }
            virtual void validateEnqueues(std::vector<size_t> const& slots,
                                          TimingEvents const&        startEvents,
                                          TimingEvents const&        stopEvents) override
            {
            }

            virtual void finalizeReport() override {}

            virtual int error() const override
            {
                return m_errorsReported;
            }

        private:
            void setUpTables(ContractionSolution const& solution);
            void checkGates();

            bool                 m_enabled          = false;
            bool                 m_active           = false;
            bool                 m_checked          = false;
            bool                 m_failedInSolution = false;
            int                  m_errorsReported   = 0;
            ContractionProblem*  m_problem          = nullptr;
            GateTableLayout      m_layout;
            std::vector<uint8_t> m_host;
        };
    } // namespace Client
} // namespace TensileLite
