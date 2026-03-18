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

#include "BenchmarkTimer.hpp"
#include "PerformanceReporter.hpp"
#include "ResultReporter.hpp"

#include "Reference.hpp"

#include <Tensile/hip/HipUtils.hpp>

#include <csignal>
#include <cstddef>
#include <thread>

namespace Tensile
{
    namespace Client
    {
        static_assert(BenchmarkTimer::clock::is_steady, "Clock must be steady.");

        BenchmarkTimer::BenchmarkTimer(po::variables_map const& args, Hardware const& hardware)
            : m_numWarmups(args["num-warmups"].as<int>())
            , m_syncAfterWarmups(args["sync-after-warmups"].as<bool>())
            , m_numBenchmarks(args["num-benchmarks"].as<int>())
            , m_numEnqueuesPerSync(args["num-enqueues-per-sync"].as<int>())
            , m_minFlopsPerSync(args["min-flops-per-sync"].as<size_t>())
            , m_numSyncsPerBenchmark(args["num-syncs-per-benchmark"].as<int>())
            , m_hardware(hardware)
            , m_numEnqueuesPerSolution(m_numEnqueuesPerSync * m_numSyncsPerBenchmark)
            , m_useGPUTimer(args["use-gpu-timer"].as<bool>())
            , m_sleepPercent(args["sleep-percent"].as<int>())
            , m_timeInSolution(0)
            , m_totalGPUTime(0)
        {
        }

        bool BenchmarkTimer::needMoreBenchmarkRuns() const
        {
            return m_numBenchmarksRun < m_numBenchmarks;
        }

        void BenchmarkTimer::preBenchmarkRun() {}

        void BenchmarkTimer::postBenchmarkRun()
        {
            m_numBenchmarksRun++;
        }

        void BenchmarkTimer::preProblem(ContractionProblem const& problem)
        {
            m_problem = problem;
        }

        void BenchmarkTimer::postProblem() {}

        void BenchmarkTimer::preSolution(ContractionSolution const& solution)
        {
            m_numEnqueuesInSolution = 0;
            m_timeInSolution        = double_millis::zero();

            ContractionSolution::ProjectedPerformance pp
                = solution.projectedPerformance(m_problem, m_hardware);
            m_solution = solution;

            m_reporter->report(ResultKey::Tile0Granularity, pp.granularities.tile0Granularity);
            m_reporter->report(ResultKey::Tile1Granularity, pp.granularities.tile1Granularity);
            m_reporter->report(ResultKey::CuGranularity, pp.granularities.cuGranularity);
            m_reporter->report(ResultKey::WaveGranularity, pp.granularities.waveGranularity);
            m_reporter->report(ResultKey::TotalGranularity, pp.granularities.totalGranularity);

            m_reporter->report(ResultKey::NumCus, perf.CUs);
            m_reporter->report(ResultKey::TilesPerCu, pp.granularities.tilesPerCu);
            m_reporter->report(ResultKey::MemReadBytes, pp.staticModel.memReadBytes);
            m_reporter->report(ResultKey::MemWriteBytes, pp.staticModel.memWriteBytesD);
        }

        void BenchmarkTimer::postSolution()
        {
            double timePerEnqueue_us
                = double_micros(m_timeInSolution).count() / m_numEnqueuesInSolution;

            ContractionSolution::ProjectedPerformance pp
                = m_solution.projectedPerformance(m_problem, m_hardware);

            double gflops      = m_problem.flopCount() / (timePerEnqueue_us) / 1000.0;
            int    tiles       = pp.granularities.tilesPerCu * perf.CUs;
            int    usedCus     = std::min(tiles, perf.CUs);
            double gflopsPerCu = gflops / usedCus;

            m_reporter->report(ResultKey::TimeUS, timePerEnqueue_us);
            m_reporter->report(ResultKey::SpeedGFlopsPerCu, gflopsPerCu);
            m_reporter->report(ResultKey::SpeedGFlops, gflops);

            m_timeInSolution        = double_millis::zero();
            m_numEnqueuesInSolution = 0;
        }

        bool BenchmarkTimer::needMoreRunsInSolution() const
        {
            return m_numEnqueuesInSolution < m_numEnqueuesPerSolution;
        }

        size_t BenchmarkTimer::numWarmupRuns()
        {
            return m_numWarmups;
        }

        void BenchmarkTimer::setNumWarmupRuns(size_t count)
        {
            if(count < m_numWarmups)
                throw std::runtime_error(concatenate(
                    "Expected at least", m_numWarmups, " warmup runs, got ", count, "."));
        }

        void BenchmarkTimer::preWarmup() {}

        void BenchmarkTimer::postWarmup() {}

        void BenchmarkTimer::validateWarmups(std::shared_ptr<ContractionInputs> inputs,
                                             TimingEvents const&                startEvents,
                                             TimingEvents const&                stopEvents)
        {
            if(m_syncAfterWarmups && (stopEvents->size() > 0) && (stopEvents->back().size() > 0))
                HIP_CHECK_EXC(hipEventSynchronize(stopEvents->back().back()));
        }

        size_t BenchmarkTimer::numSyncs()
        {
            return m_numSyncsPerBenchmark;
        }

        void BenchmarkTimer::setNumSyncs(size_t count)
        {
            m_numSyncsInBenchmark = count;
        }

        void BenchmarkTimer::preSyncs() {}

        void BenchmarkTimer::postSyncs() {}

        size_t BenchmarkTimer::numEnqueuesPerSync()
        {
            size_t enqueuesByFlops = 0;
            if(m_minFlopsPerSync > 0)
            {
                size_t flopsInProblem = m_problem.flopCount();
                enqueuesByFlops       = CeilDivide(m_minFlopsPerSync, flopsInProblem);
            }

            return std::max<size_t>(m_numEnqueuesPerSync, enqueuesByFlops);
        }

        void BenchmarkTimer::setNumEnqueuesPerSync(size_t count)
        {
            m_curNumEnqueuesPerSync = count;
        }

        void BenchmarkTimer::preEnqueues()
        {
            if(!m_useGPUTimer)
            {
                // Synchronize before timer so warmup runs are not included in benchmark time
                HIP_CHECK_EXC(hipDeviceSynchronize());
                m_startTime = clock::now();
            }
        }

        void BenchmarkTimer::postEnqueues(TimingEvents const& startEvents,
                                          TimingEvents const& stopEvents)
        {
            if(!m_useGPUTimer)
            {
                HIP_CHECK_EXC(hipDeviceSynchronize());
                m_endTime = clock::now();
            }
        }

        void BenchmarkTimer::validateEnqueues(std::shared_ptr<ContractionInputs> inputs,
                                              TimingEvents const&                startEvents,
                                              TimingEvents const&                stopEvents)
        {
            double_millis totalTime(0.0);

            if(m_useGPUTimer)
            {
                HIP_CHECK_EXC(hipEventSynchronize(stopEvents->back().back()));
                for(size_t i = 0; i < startEvents->size(); i++)
                {
                    float enqTime = 0.0f;

                    HIP_CHECK_EXC(hipEventElapsedTime(
                        &enqTime, startEvents->at(i).front(), stopEvents->at(i).back()));

                    totalTime += double_millis(enqTime);
                }
            }
            else
            {
                totalTime = double_millis(m_endTime - m_startTime);
            }

            m_timeInSolution += totalTime;
            m_totalGPUTime += totalTime;
            m_numEnqueuesInSolution += startEvents->size();

            if(m_sleepPercent > 0)
            {
                auto sleepTime = totalTime * (m_sleepPercent / 100.0);

                std::this_thread::sleep_for(sleepTime);
            }
        }

        void BenchmarkTimer::finalizeReport() {}

        int BenchmarkTimer::error() const
        {
            return 0;
        }
    } // namespace Client
} // namespace Tensile
