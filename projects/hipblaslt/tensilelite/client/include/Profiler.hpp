/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ENABLE_ROCPROFSDK
#define ENABLE_ROCPROFSDK 0
#endif
#if ENABLE_ROCPROFSDK

#include <boost/program_options.hpp>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/registration.h>

#include <cstddef>
#include <string>
#include <vector>
#include <set>
#include <unordered_map>

#include "RunListener.hpp"

namespace TensileLite
{
    namespace Client
    {
        namespace po = boost::program_options;
        class Profiler : public RunListener
        {
        public:
            struct ProfileInfo {
                uint64_t dispatch_id;
                uint64_t kernel_id;
                double execution_time_us;
                std::unordered_map<std::string, uint64_t> counters;
                int solutionIdx;
            };

            static std::shared_ptr<Profiler>
                Default(po::variables_map const& args);

            Profiler(int deviceIdx, std::vector<std::string> counters);
            ~Profiler();

            virtual void preSolution(ContractionSolution* const solution) override;
            virtual void postSolution() override;
            virtual void preProfiler() override;
            virtual void postProfiler() override;
            virtual void postProblem() override;

            friend class RocProfiler;

            virtual void preProblem(ContractionProblem* const problem) override {}

            virtual bool needMoreBenchmarkRuns() const override
            {
                return false;
            }
            virtual void preBenchmarkRun() override {}
            virtual void postBenchmarkRun() override {}

            virtual bool needMoreRunsInSolution() const override
            {
                return false;
            }

            virtual size_t numWarmupRuns() override
            {
                return 0;
            }
            virtual void setNumWarmupRuns(size_t count) override {}
            virtual void preWarmup() override {}
            virtual void postWarmup(TimingEvents const& startEvents,
                                    TimingEvents const& stopEvents,
                                    hipStream_t const&  stream) override
            {
            }
            virtual void validateWarmups(std::shared_ptr<ProblemInputs> inputs,
                                         TimingEvents const&            startEvents,
                                         TimingEvents const&            stopEvents) override
            {
            }

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
            virtual void preEnqueues(hipStream_t const& stream) override {}
            virtual void postEnqueues(TimingEvents const& startEvents,
                                      TimingEvents const& stopEvents,
                                      hipStream_t const&  stream) override
            {
            }
            virtual void validateEnqueues(std::shared_ptr<ProblemInputs> inputs,
                                          TimingEvents const&            startEvents,
                                          TimingEvents const&            stopEvents) override
            {
            }

            virtual void finalizeReport() override {}

            virtual int error() const override
            {
                return 0;
            }
        private:
            int m_currentSolutionIdx = 0;
            bool m_currentDone = false;
            std::set<std::string> m_counterNames;
            std::unordered_map<int, uint64_t> m_solutionIdx2DispatchId;
            std::unordered_map<uint64_t, ProfileInfo> m_dispatchId2ProfileInfo;
        };

        namespace rocprof
        {
            int tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data);
        }

        class RocProfiler {
        public:
            static RocProfiler& getInstance() {
                static RocProfiler instance;
                return instance;
            }

            RocProfiler() : m_do(false), m_initialized(false), m_context_started(false) {
                m_context.handle = 0;
            }

            // Prevent copying
            RocProfiler(const RocProfiler&) = delete;
            RocProfiler& operator=(const RocProfiler&) = delete;

            // Initialize profiler with desired counters
            bool initialize(int deviceIdx, const std::vector<std::string>& counter_names, Profiler* profiler);

            // Query GPU agents
            void queryAgents(int deviceIdx);

            // Create counter profiles for all agents
            void createProfiles();

            // Start context and enable profiling
            bool start();
            // Stop context
            void stop();

            // Enable rocprof
            void enable() { m_do = true; }

            // Disable rocprof
            void disable() { m_do = false; }

            // Get coounter in string
            std::string fetch(int index);

            void shutdown();

            ~RocProfiler() {
                shutdown();
            }

            friend int rocprof::tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data);
        private:
            bool m_do = false;
            bool m_initialized = false;
            bool m_context_started = false;
            uint32_t m_locationId;
            std::mutex m_mutex;
            rocprofiler_context_id_t m_context;
            rocprofiler_buffer_id_t m_buffer;
            rocprofiler_agent_v0_t m_agent;
            rocprofiler_counter_config_id_t m_agentProfile;
            Profiler* m_profiler;
            std::unordered_map<std::string, rocprofiler_counter_id_t> m_counterName2Id;

            // Tool initialization callback
            static int tool_init_impl(rocprofiler_client_finalize_t fini_func, void* tool_data);

            // Dispatch callback - called for each kernel dispatch
            static void dispatchCallback(
                                  rocprofiler_dispatch_counting_service_data_t dispatch_data,
                                  rocprofiler_counter_config_id_t* config,
                                  rocprofiler_user_data_t* user_data,
                                  void* callback_data);

            // Buffered callback - receives counter data
            static void bufferedCallback(
                                  rocprofiler_context_id_t,
                                  rocprofiler_buffer_id_t,
                                  rocprofiler_record_header_t** headers,
                                  size_t num_headers,
                                  void* user_data,
                                  uint64_t);
        };
    }
}

#endif
