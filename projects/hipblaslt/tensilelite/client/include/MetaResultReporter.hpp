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

#pragma once

#include "RunListener.hpp"

#include <cstddef>

namespace TensileLite
{
    namespace Client
    {
        class MetaResultReporter : public ResultReporter
        {
        public:
            virtual void addReporter(std::shared_ptr<ResultReporter> reporter)
            {
                if(reporter != nullptr)
                    m_reporters.push_back(reporter);
            }

            virtual void reportValue_string(std::string const& key,
                                            std::string const& value) override
            {
                for(auto r : m_reporters)
                    r->reportValue_string(key, value);
            }

            virtual void reportValue_uint(std::string const& key, uint64_t value) override
            {
                for(auto r : m_reporters)
                    r->reportValue_uint(key, value);
            }

            virtual void reportValue_int(std::string const& key, int64_t value) override
            {
                for(auto r : m_reporters)
                    r->reportValue_int(key, value);
            }

            virtual void reportValue_double(std::string const& key, double value) override
            {
                for(auto r : m_reporters)
                    r->reportValue_double(key, value);
            }

            virtual void reportValue_sizes(std::string const&         key,
                                           std::vector<size_t> const& value) override
            {
                for(auto r : m_reporters)
                    r->reportValue_sizes(key, value);
            }

            virtual void
                reportValue_vecOfSizes(std::string const&                      key,
                                       std::vector<std::vector<size_t>> const& value) override
            {
                for(auto r : m_reporters)
                    r->reportValue_vecOfSizes(key, value);
            }

            virtual bool logAtLevel(LogLevel level) override
            {
                for(auto r : m_reporters)
                    if(r->logAtLevel(level))
                        return true;
                return false;
            }

            virtual void logMessage(LogLevel level, std::string const& message) override
            {
                for(auto r : m_reporters)
                    r->logMessage(level, message);
            }

            virtual void logTensor(LogLevel                level,
                                   std::string const&      name,
                                   void const*             data,
                                   TensorDescriptor const& tensor,
                                   void const*             ptrValue) override
            {
                for(auto r : m_reporters)
                    r->logTensor(level, name, data, tensor, ptrValue);
            }

            /// RunListener interface functions

            virtual bool needMoreBenchmarkRuns() const override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    if((*iter)->needMoreBenchmarkRuns())
                        return true;

                return false;
            }

            virtual void preBenchmarkRun() override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->preBenchmarkRun();
            }

            virtual void postBenchmarkRun() override
            {
                for(auto iter = m_reporters.rbegin(); iter != m_reporters.rend(); iter++)
                    (*iter)->postBenchmarkRun();
            }

            virtual void preProblem(ContractionProblem* const problem) override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->preProblem(problem);
            }

            virtual void postProblem() override
            {
                for(auto iter = m_reporters.rbegin(); iter != m_reporters.rend(); iter++)
                    (*iter)->postProblem();
            }

            virtual void preSolution(ContractionSolution const& solution) override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->preSolution(solution);
            }

            virtual void postSolution() override
            {
                for(auto iter = m_reporters.rbegin(); iter != m_reporters.rend(); iter++)
                    (*iter)->postSolution();
            }

            virtual bool needMoreRunsInSolution() const override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    if((*iter)->needMoreRunsInSolution())
                        return true;

                return false;
            }

            virtual size_t numWarmupRuns() override
            {
                return 0;
            }
            virtual void setNumWarmupRuns(size_t count) override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->setNumWarmupRuns(count);
            }

            virtual void preWarmup() override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->preWarmup();
            }

            virtual void postWarmup(TimingEvents const& startEvents,
                                    TimingEvents const& stopEvents,
                                    hipStream_t const&  stream) override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->postWarmup(startEvents, stopEvents, stream);
            }

            virtual void validateWarmups(std::shared_ptr<ProblemInputs> inputs,
                                         TimingEvents const&            startEvents,
                                         TimingEvents const&            stopEvents) override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->validateWarmups(inputs, startEvents, stopEvents);
            }

            virtual size_t numSyncs() override
            {
                return 0;
            }

            virtual void setNumSyncs(size_t count) override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->setNumSyncs(count);
            }

            virtual void preSyncs() override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->preSyncs();
            }

            virtual void postSyncs() override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->postSyncs();
            }

            virtual size_t numEnqueuesPerSync() override
            {
                return 0;
            }

            virtual void setNumEnqueuesPerSync(size_t count) override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->setNumEnqueuesPerSync(count);
            }

            virtual void preEnqueues(hipStream_t const& stream) override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->preEnqueues(stream);
            }
            virtual void postEnqueues(TimingEvents const& startEvents,
                                      TimingEvents const& stopEvents,
                                      hipStream_t const&  stream) override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->postEnqueues(startEvents, stopEvents, stream);
            }

            virtual void validateEnqueues(std::shared_ptr<ProblemInputs> inputs,
                                          TimingEvents const&            startEvents,
                                          TimingEvents const&            stopEvents) override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->validateEnqueues(inputs, startEvents, stopEvents);
            }

            virtual void finalizeReport() override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                    (*iter)->finalizeReport();
            }

            virtual int error() const override
            {
                for(auto iter = m_reporters.begin(); iter != m_reporters.end(); iter++)
                {
                    int rv = (*iter)->error();
                    if(rv)
                        return rv;
                }

                return 0;
            }

        private:
            std::vector<std::shared_ptr<ResultReporter>> m_reporters;
        };
    } // namespace Client
} // namespace TensileLite
