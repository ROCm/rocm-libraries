/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include "RunListener.hpp"

#include <chrono>
#include <cstddef>

#include <boost/program_options.hpp>

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>

namespace Tensile
{
    namespace Client
    {
        namespace po = boost::program_options;

        class BenchmarkTimer : public RunListener
        {
        public:
            using clock = std::chrono::steady_clock;

            BenchmarkTimer(po::variables_map const& args, Hardware const& hardware);

            virtual bool needMoreBenchmarkRuns() const override;
            virtual void preBenchmarkRun() override;
            virtual void postBenchmarkRun() override;

            virtual void preProblem(ContractionProblem const& problem) override;
            virtual void postProblem() override;

            virtual void preSolution(ContractionSolution const& solution) override;
            virtual void postSolution() override;

            virtual bool needMoreRunsInSolution() const override;

            virtual size_t numWarmupRuns() override;
            virtual void   setNumWarmupRuns(size_t count) override;
            virtual void   preWarmup() override;
            virtual void   postWarmup() override;
            virtual void   validateWarmups(std::shared_ptr<ContractionInputs> inputs,
                                           TimingEvents const&                startEvents,
                                           TimingEvents const&                stopEvents) override;

            virtual size_t numSyncs() override;
            virtual void   setNumSyncs(size_t count) override;
            virtual void   preSyncs() override;
            virtual void   postSyncs() override;

            virtual size_t numEnqueuesPerSync() override;
            virtual void   setNumEnqueuesPerSync(size_t count) override;
            virtual void   preEnqueues() override;
            virtual void   postEnqueues(TimingEvents const& startEvents,
                                        TimingEvents const& stopEvents) override;
            virtual void   validateEnqueues(std::shared_ptr<ContractionInputs> inputs,
                                            TimingEvents const&                startEvents,
                                            TimingEvents const&                stopEvents) override;

            virtual void finalizeReport() override;
            virtual int  error() const override;

        private:
            const int    m_numWarmups;
            const int    m_numBenchmarks;
            const int    m_numEnqueuesPerSync;
            const int    m_numSyncsPerBenchmark;
            const int    m_numEnqueuesPerSolution;
            const size_t m_minFlopsPerSync = 0;

            const bool m_useGPUTimer;
            const bool m_syncAfterWarmups = true;
            const int  m_sleepPercent;

            int m_numBenchmarksRun = 0;

            Hardware const&     m_hardware;
            ContractionProblem  m_problem;
            ContractionSolution m_solution;

            int m_numEnqueuesInSolution = 0;
            int m_numSyncsInBenchmark   = 0;
            int m_curNumEnqueuesPerSync = 0;

            clock::time_point m_startTime;
            clock::time_point m_endTime;

            using double_millis = std::chrono::duration<double, std::milli>;
            using double_micros = std::chrono::duration<double, std::micro>;
            using double_nanos  = std::chrono::duration<double, std::nano>;

            double_millis m_timeInSolution;
            double_millis m_totalGPUTime;
        };
    } // namespace Client
} // namespace Tensile
