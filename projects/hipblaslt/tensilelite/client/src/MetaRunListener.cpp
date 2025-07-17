/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "MetaRunListener.hpp"

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>

#include "ResultReporter.hpp"
#include "RunListener.hpp"

#include <cstddef>

namespace TensileLite
{
    namespace Client
    {
        void MetaRunListener::addListener(std::shared_ptr<RunListener> listener)
        {
            if(!m_firstProblem)
                throw std::runtime_error("Can't add listeners after benchmarking has begun.");

            listener->setReporter(m_reporter);

            m_listeners.push_back(listener);
        }

        void MetaRunListener::setReporter(std::shared_ptr<ResultReporter> reporter)
        {
            if(!m_firstProblem)
                throw std::runtime_error("Can't set reporter after benchmarking has begun.");
            if(m_reporter)
                throw std::runtime_error("Can't set reporter more than once.");

            m_reporter = reporter;
            for(auto const& listener : m_listeners)
                listener->setReporter(reporter);

            m_listeners.insert(m_listeners.begin(), reporter);
        }

        bool MetaRunListener::needMoreBenchmarkRuns() const
        {
            for(auto iter = m_listeners.begin(); iter != m_listeners.end(); iter++)
                if((*iter)->needMoreBenchmarkRuns())
                    return true;

            return false;
        }

        void MetaRunListener::preBenchmarkRun()
        {
            for(auto iter = m_listeners.begin(); iter != m_listeners.end(); iter++)
                (*iter)->preBenchmarkRun();
        }

        void MetaRunListener::postBenchmarkRun()
        {
            for(auto iter = m_listeners.rbegin(); iter != m_listeners.rend(); iter++)
                (*iter)->postBenchmarkRun();
        }

        void MetaRunListener::preProblem(ContractionProblem* const problem)
        {
            for(auto iter = m_listeners.begin(); iter != m_listeners.end(); iter++)
                (*iter)->preProblem(problem);
        }

        void MetaRunListener::postProblem()
        {
            for(auto iter = m_listeners.rbegin(); iter != m_listeners.rend(); iter++)
                (*iter)->postProblem();
        }

        void MetaRunListener::preSolution(ContractionSolution const& solution)
        {
            for(auto iter = m_listeners.begin(); iter != m_listeners.end(); iter++)
                (*iter)->preSolution(solution);
        }

        void MetaRunListener::postSolution()
        {
            for(auto iter = m_listeners.rbegin(); iter != m_listeners.rend(); iter++)
                (*iter)->postSolution();
        }

        bool MetaRunListener::needMoreRunsInSolution() const
        {
            for(auto iter = m_listeners.begin(); iter != m_listeners.end(); iter++)
                if((*iter)->needMoreRunsInSolution())
                    return true;

            return false;
        }

        size_t MetaRunListener::numWarmupRuns()
        {
            size_t count = 0;
            for(auto iter = m_listeners.begin(); iter != m_listeners.end(); iter++)
                count = std::max(count, (*iter)->numWarmupRuns());

            setNumWarmupRuns(count);

            return count;
        }

        void MetaRunListener::setNumWarmupRuns(size_t count)
        {
            for(auto iter = m_listeners.begin(); iter != m_listeners.end(); iter++)
                (*iter)->setNumWarmupRuns(count);
        }

        void MetaRunListener::preWarmup()
        {
            for(auto iter = m_listeners.begin(); iter != m_listeners.end(); iter++)
                (*iter)->preWarmup();
        }

        void MetaRunListener::postWarmup(TimingEvents const& startEvents,
                                         TimingEvents const& stopEvents,
                                         hipStream_t const&  stream)
        {
            for(auto iter = m_listeners.rbegin(); iter != m_listeners.rend(); iter++)
                (*iter)->postWarmup(startEvents, stopEvents, stream);
        }

        void MetaRunListener::validateWarmups(std::shared_ptr<ProblemInputs> inputs,
                                              TimingEvents const&            startEvents,
                                              TimingEvents const&            stopEvents)
        {
            for(auto iter = m_listeners.begin(); iter != m_listeners.end(); iter++)
                (*iter)->validateWarmups(inputs, startEvents, stopEvents);
        }

        size_t MetaRunListener::numSyncs()
        {
            size_t count = 0;
            for(auto iter = m_listeners.begin(); iter != m_listeners.end(); iter++)
                count = std::max(count, (*iter)->numSyncs());

            setNumSyncs(count);

            return count;
        }

        void MetaRunListener::setNumSyncs(size_t count)
        {
            for(auto iter = m_listeners.begin(); iter != m_listeners.end(); iter++)
                (*iter)->setNumSyncs(count);
        }

        void MetaRunListener::preSyncs()
        {
            for(auto iter = m_listeners.begin(); iter != m_listeners.end(); iter++)
                (*iter)->preSyncs();
        }

        void MetaRunListener::postSyncs()
        {
            for(auto iter = m_listeners.rbegin(); iter != m_listeners.rend(); iter++)
                (*iter)->postSyncs();
        }

        size_t MetaRunListener::numEnqueuesPerSync()
        {
            size_t count = 0;
            for(auto iter = m_listeners.begin(); iter != m_listeners.end(); iter++)
                count = std::max(count, (*iter)->numEnqueuesPerSync());

            setNumEnqueuesPerSync(count);

            return count;
        }

        void MetaRunListener::setNumEnqueuesPerSync(size_t count)
        {
            for(auto iter = m_listeners.begin(); iter != m_listeners.end(); iter++)
                (*iter)->setNumEnqueuesPerSync(count);
        }

        void MetaRunListener::preEnqueues(hipStream_t const& stream)
        {
            for(auto iter = m_listeners.begin(); iter != m_listeners.end(); iter++)
                (*iter)->preEnqueues(stream);
        }

        void MetaRunListener::postEnqueues(TimingEvents const& startEvents,
                                           TimingEvents const& stopEvents,
                                           hipStream_t const&  stream)
        {
            for(auto iter = m_listeners.rbegin(); iter != m_listeners.rend(); iter++)
                (*iter)->postEnqueues(startEvents, stopEvents, stream);
        }

        void MetaRunListener::validateEnqueues(std::shared_ptr<ProblemInputs> inputs,
                                               TimingEvents const&            startEvents,
                                               TimingEvents const&            stopEvents)
        {
            for(auto iter = m_listeners.begin(); iter != m_listeners.end(); iter++)
                (*iter)->validateEnqueues(inputs, startEvents, stopEvents);
        }

        void MetaRunListener::finalizeReport()
        {
            for(auto iter = m_listeners.begin(); iter != m_listeners.end(); iter++)
                (*iter)->finalizeReport();
        }

        int MetaRunListener::error() const
        {
            for(auto iter = m_listeners.begin(); iter != m_listeners.end(); iter++)
            {
                int rv = (*iter)->error();
                if(rv)
                    return rv;
            }

            return 0;
        }
    } // namespace Client
} // namespace TensileLite
