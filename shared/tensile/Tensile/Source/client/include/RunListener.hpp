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

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>

#include "ResultReporter_fwd.hpp"
#include "TimingEvents.hpp"

#include <cstddef>

namespace Tensile
{
    namespace Client
    {
        class RunListener
        {
        public:
            virtual ~RunListener() = default;

            virtual void setReporter(std::shared_ptr<ResultReporter> reporter)
            {
                m_reporter = reporter;
            }

            /**********************************
   * Benchmark run - Outermost loop.
   **********************************/

            /// A benchmark run is a loop over all the problems.
            /// Return true if we need to loop once more.
            /// Note that it's not guaranteed that each listener gets this call.
            virtual bool needMoreBenchmarkRuns() const = 0;

            /// Called at the beginning of each benchmark run.
            virtual void preBenchmarkRun() = 0;

            /// Called at the end of each benchmark run.
            virtual void postBenchmarkRun() = 0;

            /**********
   * Problem
   **********/

            /// Called at the beginning of each problem.
            virtual void preProblem(ContractionProblem const& problem) = 0;

            /// Called at end of each problem.
            virtual void postProblem() = 0;

            /***********
   * Solution
   ***********/

            /// Called at the beginning of each solution.
            virtual void preSolution(ContractionSolution const& solution) = 0;

            /// Called at end of each solution.
            virtual void postSolution() = 0;

            /// Loop condition.  Return true if we need another run for this solution.
            /// Note that it's not guaranteed that each listener gets this call.
            virtual bool needMoreRunsInSolution() const = 0;

            /***********************************************************************
   * Kernel invocation run - Innermost loop.
   *
   * Within a solution, we will have zero or more warmup runs followed by
   * zero or more benchmark runs.
   ***********************************************************************/

            virtual size_t numWarmupRuns()                = 0;
            virtual void   setNumWarmupRuns(size_t count) = 0;
            virtual void   preWarmup()                    = 0;
            virtual void   postWarmup()                   = 0;
            virtual void   validateWarmups(std::shared_ptr<ContractionInputs> inputs,
                                           TimingEvents const&                startEvents,
                                           TimingEvents const&                stopEvents)
                = 0;

            virtual size_t numSyncs()                = 0;
            virtual void   setNumSyncs(size_t count) = 0;
            virtual void   preSyncs()                = 0;
            virtual void   postSyncs()               = 0;

            virtual size_t numEnqueuesPerSync()                = 0;
            virtual void   setNumEnqueuesPerSync(size_t count) = 0;
            virtual void   preEnqueues()                       = 0;
            virtual void   postEnqueues(TimingEvents const& startEvents,
                                        TimingEvents const& stopEvents)
                = 0;
            virtual void validateEnqueues(std::shared_ptr<ContractionInputs> inputs,
                                          TimingEvents const&                startEvents,
                                          TimingEvents const&                stopEvents)
                = 0;

            /// Called at end of program execution.  Print out a summary of the runs.
            virtual void finalizeReport() = 0;

            /// Called at end of program execution.  Return a non-zero value if a
            /// non-fatal error was previously recorded.
            virtual int error() const = 0;

        protected:
            std::shared_ptr<ResultReporter> m_reporter;
        };

    } // namespace Client
} // namespace Tensile
