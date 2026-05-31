/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "TimingInstrumentation.hpp"

#include "Reference.hpp"

#include <Tensile/hip/HipUtils.hpp>
#include <Tensile/ModifiedZ.hpp>

#include <algorithm>
#include <csignal>
#include <cstddef>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace TensileLite
{
    namespace Client
    {
        static_assert(BenchmarkTimer::clock::is_steady, "Clock must be steady.");

        namespace
        {
            size_t warmupTimingStartIndex(size_t warmupCount)
            {
                if(warmupCount <= 1)
                {
                    throw std::runtime_error(
                        "SkipSlowSolutionRatio warmup timing requires at least 2 warmups "
                        "so the first warmup can be discarded.");
                }

                size_t skippedWarmups = std::max<size_t>(1, (warmupCount + 9) / 10);
                return std::min(warmupCount - 1, skippedWarmups);
            }
        }

        BenchmarkTimer::BenchmarkTimer(po::variables_map const& args,
                                       Hardware const&          hardware,
                                       float                    flushTimeUs)
            : m_numWarmups(args["num-warmups"].as<int>())
            , m_syncAfterWarmups(args["sync-after-warmups"].as<bool>())
            , m_numBenchmarks(args["num-benchmarks"].as<int>())
            , m_numEnqueuesPerSync(args["num-enqueues-per-sync"].as<int>())
            , m_maxEnqueuesPerSync(args["max-enqueues-per-sync"].as<int>())
            , m_minFlopsPerSync(args["min-flops-per-sync"].as<size_t>())
            , m_numSyncsPerBenchmark(args["num-syncs-per-benchmark"].as<int>())
            , m_hardware(hardware)
            , m_numEnqueuesPerSolution(m_numEnqueuesPerSync * m_numSyncsPerBenchmark)
            , m_useGPUTimer(args["use-gpu-timer"].as<bool>())
            , m_sleepPercent(args["sleep-percent"].as<int>())
            , m_timeInSolution(0)
            , m_currentBestWarmUpTime(std::numeric_limits<double>::max())
            , m_flushTimeUs(flushTimeUs)
            , m_skip_slow_solution_ratio(args["skip-slow-solution-ratio"].as<float>())
            , m_skip_slow_solution(0)
            , m_skiprun_from_map(0)
            , m_numSolutionSkip(0)
            , m_prob_sol_map(args["prob-sol-map"].as<prob_sol_map>())
        {
            if(m_skip_slow_solution_ratio && m_numWarmups <= 1)
            {
                throw std::runtime_error(
                    "SkipSlowSolutionRatio warmup timing requires at least 2 warmups "
                    "so the first warmup can be discarded.");
            }
        }

        bool BenchmarkTimer::needMoreBenchmarkRuns() const
        {
            return m_numBenchmarksRun < m_numBenchmarks;
        }

        void BenchmarkTimer::preBenchmarkRun()
        {
            m_currProblemIdx = -1; // init
        }

        void BenchmarkTimer::postBenchmarkRun()
        {
            m_numBenchmarksRun++;
            m_currProblemIdx = -1; // reset
        }

        void BenchmarkTimer::preProblem(ContractionProblem* const problem)
        {
            ++m_currProblemIdx; // update current prob-idx

            // test if we only have a specific solution to run (When restore-from-log)
            m_probOnlyRunSolIdx = -1;
            if(m_prob_sol_map.count(m_currProblemIdx) != 0)
                m_probOnlyRunSolIdx = m_prob_sol_map.at(m_currProblemIdx);

            m_problem               = problem;
            m_currentBestWarmUpTime = double_millis(std::numeric_limits<double>::max());
            m_numSolutionSkip       = 0;
            m_currSolutionIdx       = -1; // init
        }

        void BenchmarkTimer::postProblem()
        {
            m_currSolutionIdx = -1; // reset
            if(m_numSolutionSkip)
            {
                std::cout << "########################## " << m_numSolutionSkip
                          << " solutions were skipped in total. (Skip Ratio: "
                          << m_skip_slow_solution_ratio << ")##########################"
                          << std::endl;
            }
            else
            {
                // print this as an indication of end-of-problem
                std::cout << "########################## " << std::endl;
            }
        }

        void BenchmarkTimer::preSolution(ContractionSolution* const solution)
        {
            m_numEnqueuesInSolution = 0;
            m_timeInSolution        = double_millis::zero();
            m_hotWindowTimeSamplesUS.clear();
            m_skip_slow_solution    = false;

            ++m_currSolutionIdx; // update current sol-idx
            // When restore-from-log: check if this solution is skipped if it's not the specified one, init this flag as false
            m_skiprun_from_map = false;
            if((m_probOnlyRunSolIdx != -1) && (m_probOnlyRunSolIdx != m_currSolutionIdx))
                m_skiprun_from_map = true;

            ContractionSolution::ProjectedPerformance pp;

            if(auto problem = dynamic_cast<ContractionProblemGroupedGemm*>(m_problem))
            {
                pp = solution->projectedPerformance(problem->gemms[0], m_hardware);
            }
            else if(auto problem = dynamic_cast<ContractionProblemGemm*>(m_problem))
            {
                pp = solution->projectedPerformance(*problem, m_hardware);
            }
            else
            {
                throw std::runtime_error(
                    "[BenchmarkTimer] Failed to cast problem to any ContractionProblem.");
            }

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
            bool sol_is_skipped = (m_skiprun_from_map || m_skip_slow_solution);
            if(sol_is_skipped)
            {
                m_reporter->report(ResultKey::TimeUS, std::numeric_limits<double>::quiet_NaN());
                m_reporter->report(ResultKey::SpeedGFlopsPerCu, 0);
                m_reporter->report(ResultKey::SpeedGFlops, 0);
                m_timeInSolution        = double_millis::zero();
                m_hotWindowTimeSamplesUS.clear();
                m_numEnqueuesInSolution = 0;
                return;
            }

            double timePerEnqueue_us;
            double gflops;
            double gflopsPerCu;

            {
                ScopedTimer timer("post_solution_perf_calc");
                if(m_numEnqueuesInSolution > 0)
                {
                    if(!m_hotWindowTimeSamplesUS.empty())
                    {
                        std::ostringstream rawTimingSamples;
                        rawTimingSamples << "tensilelite-client raw timing samples problem="
                                         << m_currProblemIdx << " solution=" << m_currSolutionIdx
                                         << " raw_us=[";
                        for(size_t sampleIdx = 0; sampleIdx < m_hotWindowTimeSamplesUS.size();
                            ++sampleIdx)
                        {
                            if(sampleIdx != 0)
                                rawTimingSamples << ',';
                            rawTimingSamples << m_hotWindowTimeSamplesUS[sampleIdx];
                        }
                        rawTimingSamples << ']';
                        std::cerr << rawTimingSamples.str() << '\n';

                        timePerEnqueue_us = ModifiedZ::removeOutliersAndGetMean(
                                                m_hotWindowTimeSamplesUS, 2.0)
                                            - m_flushTimeUs;
                    }
                    else
                    {
                        timePerEnqueue_us = double_micros(m_timeInSolution).count()
                                            / m_numEnqueuesInSolution - m_flushTimeUs;
                    }
                }
                else
                {
                    m_timeInSolution        = double_millis::zero();
                    m_hotWindowTimeSamplesUS.clear();
                    m_numEnqueuesInSolution = 0;
                    return;
                }

                ContractionSolution::ProjectedPerformance pp;
                double                                    flopCount = 0;
                if(auto problem = dynamic_cast<ContractionProblemGroupedGemm*>(m_problem))
                {
                    pp        = m_solution->projectedPerformance(problem->gemms[0], m_hardware);
                    flopCount = problem->gemms[0].flopCount();
                }
                else if(auto problem = dynamic_cast<ContractionProblemGemm*>(m_problem))
                {
                    pp        = m_solution->projectedPerformance(*problem, m_hardware);
                    flopCount = problem->flopCount();
                }
                else
                {
                    throw std::runtime_error(
                        "[BenchmarkTimer] Failed to cast problem to any ContractionProblem.");
                }

                gflops      = flopCount / (timePerEnqueue_us) / 1000.0;
                int    tiles       = pp.granularities.tilesPerCu * perf.CUs;
                int    usedCus     = std::min(tiles, perf.CUs);
                gflopsPerCu = gflops / usedCus;
            }

            {
                ScopedTimer timer("post_solution_reporting");
                m_reporter->report(ResultKey::TimeUS, timePerEnqueue_us);
                m_reporter->report(ResultKey::SpeedGFlopsPerCu, gflopsPerCu);
                m_reporter->report(ResultKey::SpeedGFlops, gflops);
            }

            m_timeInSolution        = double_millis::zero();
            m_hotWindowTimeSamplesUS.clear();
            m_numEnqueuesInSolution = 0;
        }

        bool BenchmarkTimer::needMoreRunsInSolution() const
        {
            bool sol_is_skipped = (m_skiprun_from_map || m_skip_slow_solution);
            return m_numEnqueuesInSolution < m_numEnqueuesPerSolution && !sol_is_skipped;
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

        void BenchmarkTimer::preWarmup()
        {
            // When restore-from-log, we only run on the specified solution
            if(m_skiprun_from_map)
                return;
        }

        void BenchmarkTimer::postWarmup(TimingEvents const& startEvents,
                                        TimingEvents const& stopEvents,
                                        hipStream_t const&  stream)
        {
            // no need to do the warmup test when:
            // 1. skip_from_map (When restore-from-log) or
            // 2. skip_slow_solution_ratio is not set
            if(m_skiprun_from_map || !m_skip_slow_solution_ratio)
                return;

            double_millis totalTime(0.0);
            float         eventMs = 0.0f;
            size_t        timingStartIdx = warmupTimingStartIndex(stopEvents->size());
            if((*startEvents)[timingStartIdx].empty() || stopEvents->back().empty())
            {
                throw std::runtime_error("Warmup timing events are empty.");
            }

            HIP_CHECK_EXC(hipEventSynchronize(stopEvents->back().back()));
            HIP_CHECK_EXC(hipEventElapsedTime(&eventMs,
                                              (*startEvents)[timingStartIdx].front(),
                                              stopEvents->back().back()));
            totalTime = double_millis(eventMs);
            if(totalTime < m_currentBestWarmUpTime)
                m_currentBestWarmUpTime = totalTime;
            else if(totalTime * m_skip_slow_solution_ratio > m_currentBestWarmUpTime)
            {
                //std::cout << "current fast time " << double_micros(m_currentBestWarmUpTime).count()/m_numWarmups \
                //  << " us, warm up time " << double_micros(totalTime).count()/m_numWarmups << " us"<< std::endl;
                m_skip_slow_solution = true;
                m_numSolutionSkip++;
            }
        }

        void BenchmarkTimer::validateWarmups(std::shared_ptr<ProblemInputs> inputs,
                                             TimingEvents const&            startEvents,
                                             TimingEvents const&            stopEvents)
        {
            if(m_syncAfterWarmups && (stopEvents->size() > 0) && (stopEvents->back().size() > 0))
            {
                ScopedTimer timer("validate_gpu_sync");
                HIP_CHECK_EXC(hipEventSynchronize(stopEvents->back().back()));
            }
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
            // No need to run when
            // 1. this solution is skip_from_map (restore-from-log) or
            // 2. m_skip_slow_solution
            if(m_skiprun_from_map || m_skip_slow_solution)
                return 0;
            size_t enqueuesByFlops = 0;
            if(m_minFlopsPerSync > 0)
            {
                double flopCount = 0;
                if(auto problem = dynamic_cast<ContractionProblemGroupedGemm*>(m_problem))
                {
                    for(int i = 0; i < problem->gemms.size(); i++)
                        flopCount += problem->gemms[i].flopCount();
                }
                else if(auto problem = dynamic_cast<ContractionProblemGemm*>(m_problem))
                {
                    flopCount = problem->flopCount();
                }
                else
                {
                    throw std::runtime_error(
                        "[BenchmarkTimer] Failed to cast problem to any ContractionProblem.");
                }
                // avoid zero division
                size_t flopsInProblem = flopCount != 0 ? flopCount : 1;
                enqueuesByFlops       = CeilDivide(m_minFlopsPerSync, flopsInProblem);
            }

            return std::min<size_t>(std::max<size_t>(m_numEnqueuesPerSync, enqueuesByFlops),
                                    m_maxEnqueuesPerSync);
        }

        void BenchmarkTimer::setNumEnqueuesPerSync(size_t count)
        {
            m_curNumEnqueuesPerSync = count;
        }

        void BenchmarkTimer::preEnqueues(hipStream_t const& stream)
        {
            if(!m_useGPUTimer)
            {
                HIP_CHECK_EXC(hipDeviceSynchronize());
                m_startTime = clock::now();
            }
        }

        void BenchmarkTimer::postEnqueues(TimingEvents const& startEvents,
                                          TimingEvents const& stopEvents,
                                          hipStream_t const&  stream)
        {
            if(!m_useGPUTimer)
            {
                HIP_CHECK_EXC(hipDeviceSynchronize());
                m_endTime = clock::now();
            }
        }

        void BenchmarkTimer::validateEnqueues(std::shared_ptr<ProblemInputs> inputs,
                                              TimingEvents const&            startEvents,
                                              TimingEvents const&            stopEvents)
        {
            double_millis totalTime(0.0);

            if(m_curNumEnqueuesPerSync == 0)
                throw std::runtime_error(
                    "[BenchmarkTimer] Effective enqueue count was not initialized.");

            if(m_useGPUTimer)
            {
                if(startEvents->empty() || stopEvents->empty() || stopEvents->back().empty())
                    throw std::runtime_error(
                        "[BenchmarkTimer] GPU timing requires at least one timing event pair.");
                if(startEvents->size() != stopEvents->size())
                    throw std::runtime_error(
                        "[BenchmarkTimer] Timing event count mismatch for benchmark enqueues.");

                HIP_CHECK_EXC(hipEventSynchronize(stopEvents->back().back()));

                size_t kernelEventCount = 0;

                for(size_t enqueueIdx = 0; enqueueIdx < startEvents->size(); ++enqueueIdx)
                {
                    auto const& iterationStarts = startEvents[enqueueIdx];
                    auto const& iterationStops  = stopEvents[enqueueIdx];

                    if(iterationStarts.empty() || iterationStops.empty())
                        throw std::runtime_error(
                            "[BenchmarkTimer] Missing precise timing events for a sync window.");
                    if(iterationStarts.size() != iterationStops.size())
                        throw std::runtime_error(
                            "[BenchmarkTimer] Precise timing event count mismatch.");

                    for(size_t eventIdx = 0; eventIdx < iterationStarts.size(); ++eventIdx)
                    {
                        float eventMs = 0.0f;
                        HIP_CHECK_EXC(hipEventElapsedTime(
                            &eventMs, iterationStarts[eventIdx], iterationStops[eventIdx]));
                        totalTime += double_millis(eventMs);
                        ++kernelEventCount;
                    }
                }

                if(kernelEventCount == 0)
                    throw std::runtime_error(
                        "[BenchmarkTimer] Precise GPU timing requires at least one kernel event pair.");

                m_hotWindowTimeSamplesUS.push_back(double_micros(totalTime).count()
                                                   / m_curNumEnqueuesPerSync);
            }
            else
            {
                totalTime = double_millis(m_endTime - m_startTime);
                const double rawSampleUs = double_micros(totalTime).count() / m_curNumEnqueuesPerSync;
                m_hotWindowTimeSamplesUS.push_back(rawSampleUs);
            }

            m_timeInSolution += totalTime;
            m_numEnqueuesInSolution += static_cast<int>(m_curNumEnqueuesPerSync);

            // reportTiming("gpu_kernel_execution", totalTime.count());

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
} // namespace TensileLite
