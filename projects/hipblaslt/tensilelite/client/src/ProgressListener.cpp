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

#include <ProgressListener.hpp>
#include <rocisa/include/enum.hpp>

#include <cstddef>
#include <iomanip>

#include <sys/time.h>

namespace TensileLite
{
    namespace Client
    {
        ProgressListener::ProgressListener(po::variables_map const& args)
            : m_runOnce(args["selection-only"].as<bool>())
        {
        }

        bool ProgressListener::needMoreBenchmarkRuns() const
        {
            return m_runOnce;
        }

        void ProgressListener::preBenchmarkRun()
        {
            m_runOnce = false;
            m_reporter->report(ResultKey::BenchmarkRunNumber, m_benchmarkRun);
        }

        void ProgressListener::postBenchmarkRun()
        {
            m_benchmarkRun++;
        }

        void ProgressListener::writeReport(ContractionProblemGemm const& problem)
        {
            m_reporter->report(ResultKey::OperationIdentifier, problem.operationIdentifier());

            m_reporter->report(ResultKey::TotalFlops, problem.flopCount());

            m_reporter->report(ResultKey::ASizes, problem.a().sizes());
            m_reporter->report(ResultKey::BSizes, problem.b().sizes());
            m_reporter->report(ResultKey::CSizes, problem.c().sizes());
            m_reporter->report(ResultKey::DSizes, problem.d().sizes());

            m_reporter->report(ResultKey::AStrides, problem.a().strides());
            m_reporter->report(ResultKey::BStrides, problem.b().strides());
            m_reporter->report(ResultKey::CStrides, problem.c().strides());
            m_reporter->report(ResultKey::DStrides, problem.d().strides());

            m_reporter->report(ResultKey::LDA, problem.a().strides()[1]);
            m_reporter->report(ResultKey::LDB, problem.b().strides()[1]);
            m_reporter->report(ResultKey::LDC, problem.c().strides()[1]);
            m_reporter->report(ResultKey::LDD, problem.d().strides()[1]);

            if(problem.useBias())
            {
                m_reporter->report(ResultKey::BiasType,
                                   rocisa::toString(problem.getParams().biasEnum()));
            }
            else
            {
                m_reporter->report(ResultKey::BiasType, "None");
            }
            if(problem.useScaleAlphaVec() || problem.useBias())
            {
                m_reporter->report(ResultKey::FactorDim, problem.getParams().factorDim());
            }
            m_reporter->report(ResultKey::ActivationType,
                               ToString(problem.getParams().activationEnum()));
        }

        void ProgressListener::preProblem(ContractionProblem* const problem)
        {
            if(auto groupedProblem = dynamic_cast<const ContractionProblemGroupedGemm*>(problem))
            {
                writeReport(groupedProblem->gemms[0]);
                std::vector<std::vector<size_t>> sizes;
                for(auto& it : groupedProblem->gemms)
                    sizes.push_back(it.problemSizes());
                m_reporter->report(ResultKey::ProblemSizes, sizes);
            }
            else if(auto gemmProblem = dynamic_cast<const ContractionProblemGemm*>(problem))
            {
                writeReport(*gemmProblem);
                m_reporter->report(ResultKey::ProblemSizes, gemmProblem->problemSizes());
            }
            else
            {
                throw std::runtime_error("Failed to cast to any ContractionProblem.");
            }
        }

        void ProgressListener::postProblem() {}

        void ProgressListener::preSolution(ContractionSolution const& solution)
        {
            m_reporter->report(ResultKey::SolutionName, solution.name());
            m_reporter->report(ResultKey::SolutionIndex, solution.index);
        }

        void ProgressListener::postSolution() {}

        bool ProgressListener::needMoreRunsInSolution() const
        {
            return false;
        }

        size_t ProgressListener::numWarmupRuns()
        {
            return 0;
        }

        void ProgressListener::setNumWarmupRuns(size_t count) {}

        void ProgressListener::preWarmup() {}

        void ProgressListener::postWarmup(TimingEvents const& startEvents,
                                          TimingEvents const& stopEvents,
                                          hipStream_t const&  stream)
        {
        }

        void ProgressListener::validateWarmups(std::shared_ptr<ProblemInputs> inputs,
                                               TimingEvents const&            startEvents,
                                               TimingEvents const&            stopEvents)
        {
        }

        size_t ProgressListener::numSyncs()
        {
            return 0;
        }

        void ProgressListener::setNumSyncs(size_t count) {}

        void ProgressListener::preSyncs() {}

        void ProgressListener::postSyncs() {}

        size_t ProgressListener::numEnqueuesPerSync()
        {
            return 0;
        }

        void ProgressListener::setNumEnqueuesPerSync(size_t count) {}

        void ProgressListener::preEnqueues(hipStream_t const& stream) {}

        void ProgressListener::postEnqueues(TimingEvents const& startEvents,
                                            TimingEvents const& stopEvents,
                                            hipStream_t const&  stream)
        {
        }

        void ProgressListener::validateEnqueues(std::shared_ptr<ProblemInputs> inputs,
                                                TimingEvents const&            startEvents,
                                                TimingEvents const&            stopEvents)
        {
            struct timeval tmnow;
            struct tm*     tm;
            gettimeofday(&tmnow, NULL); // microsecond resolution
            tm = localtime(&tmnow.tv_sec);
            std::cout.fill('0');

            std::ostringstream msg;
            msg.fill('0');
            msg << (tm->tm_year + 1900) << "-" << std::setw(2) << (tm->tm_mon + 1) << "-"
                << std::setw(2) << tm->tm_mday << " " << std::setw(2) << tm->tm_hour << ":"
                << std::setw(2) << tm->tm_min << ":" << std::setw(2) << tm->tm_sec << "."
                << std::setw(6) << static_cast<int>(tmnow.tv_usec);

            m_reporter->report(ResultKey::EnqueueTime, msg.str());
        }

        void ProgressListener::finalizeReport() {}

        int ProgressListener::error() const
        {
            return 0;
        }

    } // namespace Client
} // namespace TensileLite
