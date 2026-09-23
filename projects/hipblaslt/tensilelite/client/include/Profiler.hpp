// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifndef TENSILELITE_CLIENT_ENABLE_ROCPROFSDK
#define TENSILELITE_CLIENT_ENABLE_ROCPROFSDK 0
#endif
#if TENSILELITE_CLIENT_ENABLE_ROCPROFSDK

#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/registration.h>

#include <cstddef>
#include <string>
#include <vector>
#include <variant>
#include <set>
#include <mutex>
#include <future>
#include <unordered_map>

#include "ProgramOptions.hpp"
#include "RunListener.hpp"

namespace TensileLite
{
    namespace Client
    {
        class Profiler : public RunListener
        {
        public:
            struct ProfileInfo {
                uint64_t kernel_id;
                std::unordered_map<uint64_t, std::variant<double, std::vector<double>>> counters;
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
            uint64_t m_currentKernelId = 0;
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
            struct CounterInfo {
                uint64_t id;
                size_t num;
                std::vector<rocprofiler_counter_record_dimension_info_t> dimInfos;
                std::vector<std::string> dimNames;
                std::vector<size_t> strides;
            };

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
            bool initialize(int deviceIdx, Profiler* profiler);

            // Query GPU agents
            void queryAgents(int deviceIdx);

            // Create counter profiles for all agents
            void createProfiles();

            // Start context and enable profiling
            bool start();
            // Stop context
            void stop();

            // Enable rocprof
            void enable() {
                m_do = true;
                m_promise = std::promise<void>();
                m_future = m_promise.get_future();
            }

            // get kernel id of given kernel name
            uint64_t getKernelId(std::string kernelName);

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
            std::promise<void> m_promise;
            std::future<void> m_future;
            rocprofiler_context_id_t m_context;
            rocprofiler_agent_v0_t m_agent;
            rocprofiler_counter_config_id_t m_agentProfile;
            std::unordered_map<uint64_t, CounterInfo> m_counterInfos;
            Profiler* m_profiler;
            std::unordered_map<std::string, rocprofiler_counter_id_t> m_counterName2Id;
            std::unordered_map<std::string, uint64_t> m_kernelName2Id;

            // Tool initialization callback
            static int tool_init_impl(rocprofiler_client_finalize_t fini_func, void* tool_data);

            // Tracing callback - configured to record kernel id and name during loading
            static void kernelLoadingCallback(
                                              rocprofiler_callback_tracing_record_t record,
                                              rocprofiler_user_data_t *user_data,
                                              void *callback_data);

            // Dispatch callback - called for each kernel dispatch
            static void dispatchCallback(
                                         rocprofiler_dispatch_counting_service_data_t dispatch_data,
                                         rocprofiler_counter_config_id_t* config,
                                         rocprofiler_user_data_t* user_data,
                                         void* callback_data);

            // Record callback - called after kernel finished
            static void recordCallback(
                                       rocprofiler_dispatch_counting_service_data_t dispatch_data,
                                       rocprofiler_counter_record_t* record_data,
                                       unsigned long record_count,
                                       rocprofiler_user_data_t user_data,
                                       void* callback_data);

        };
    }
}

#endif
