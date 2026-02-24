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

#include <Tensile/hip/HipUtils.hpp>

#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/registration.h>
#include <hip/hip_runtime.h>

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <iostream>

#include "Profiler.hpp"

#define ROCPROFILER_CALL(result, msg)                                                              \
    {                                                                                              \
        rocprofiler_status_t CHECKSTATUS = result;                                                 \
        if (CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS)                                             \
        {                                                                                          \
            std::string status_msg = rocprofiler_get_status_string(CHECKSTATUS);                   \
            std::cerr << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg            \
                      << " failed with error code " << CHECKSTATUS << ": " << status_msg           \
                      << std::endl;                                                                \
            std::stringstream errmsg{};                                                            \
            errmsg << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg " failure ("  \
                   << status_msg << ")";                                                           \
            throw std::runtime_error(errmsg.str());                                                \
        }                                                                                          \
    }

#define ROCPROFILER_CHECK(result, msg) \
    (rocprof::rocprofiler_call_ok((result), #result, __FILE__, __LINE__, (msg)))

namespace TensileLite
{
    namespace Client
    {
        namespace rocprof
        {
            inline bool rocprofiler_call_ok(rocprofiler_status_t status, const char* expr, const char* file, int line, const char* msg)
            {
                if (status != ROCPROFILER_STATUS_SUCCESS)
                {
                    std::string status_msg = rocprofiler_get_status_string(status);
                    std::cerr << "[" << expr << "][" << file << ":" << line << "] " << msg
                              << " failed with error code " << status << ": " << status_msg
                              << std::endl;
                    return false;
                }
                return true;
            }

            uint32_t locationIdFromDeviceId(int deviceIdx)
            {
                char pciStr[32];
                HIP_CHECK_EXC(hipDeviceGetPCIBusId(pciStr, sizeof(pciStr), deviceIdx));
                bool parseSuccess = false;
                int dom, bus, dev, fnc;
                if (std::sscanf(pciStr, "%x:%x:%x.%u", &dom, &bus, &dev, &fnc) == 4) { parseSuccess = true; }
                if (parseSuccess || std::sscanf(pciStr, "%x:%x.%u", &bus, &dev, &fnc) == 3) { parseSuccess = true; }
                if (!parseSuccess) { throw std::runtime_error("failed to parse pci bus id"); }
                uint32_t locationId = ((bus & 0xFF) << 8) | ((dev & 0x1F) << 3) | (fnc & 0x07);
                return locationId;
            }
        }

        bool RocProfiler::initialize(int deviceIdx, const std::vector<std::string>& counter_names, Profiler* profiler) {
            if (m_initialized) return true;
            std::lock_guard<std::mutex> lock(m_mutex);
            m_profiler = profiler;

            queryAgents(deviceIdx);
            createProfiles();

            m_initialized = true;
            return true;
        }

        void RocProfiler::queryAgents(int deviceIdx) {
            m_locationId = rocprof::locationIdFromDeviceId(deviceIdx);
            auto agent_cb = [](rocprofiler_agent_version_t version,
                               const void** agents,
                               size_t num_agents,
                               void* user_data) -> rocprofiler_status_t {
                auto* rocprofiler = static_cast<RocProfiler*>(user_data);
                bool found = false;
                for (size_t i = 0; i < num_agents; ++i) {
                    const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents[i]);
                    if (agent->type == ROCPROFILER_AGENT_TYPE_GPU && agent->location_id == rocprofiler->m_locationId) {
                        rocprofiler->m_agent = *agent;
                        found = true;
                    }
                }
                if (!found)
                    return ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND;
                return ROCPROFILER_STATUS_SUCCESS;
            };
            ROCPROFILER_CALL(rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0, agent_cb, sizeof(rocprofiler_agent_v0_t), this),
                             "Failed to find GPU agent of with device-idx");
        }

        void RocProfiler::createProfiles() {
            auto counter_cb = [](rocprofiler_agent_id_t,
                                 rocprofiler_counter_id_t* counters,
                                 size_t num_counters,
                                 void* user_data) -> rocprofiler_status_t {
                auto* rocprofiler = static_cast<RocProfiler*>(user_data);
                std::vector<rocprofiler_counter_id_t> counter_ids;
                rocprofiler_counter_info_v0_t info;
                rocprofiler_status_t status;
                for (size_t i = 0; i < num_counters; i++) {
                    auto counter_id = counters[i];
                    status = rocprofiler_query_counter_info(counter_id, ROCPROFILER_COUNTER_INFO_VERSION_0, static_cast<void*>(&info));
                    if (status == ROCPROFILER_STATUS_SUCCESS)
                    {
                        if (rocprofiler->m_profiler->m_counterNames.count(info.name)) {
                            counter_ids.push_back(counter_id);
                            rocprofiler->m_counterName2Id[info.name] = counter_id;
                        }
                    }
                }
                bool failed = false;
                for (auto& counter_name : rocprofiler->m_profiler->m_counterNames)
                {
                    if (rocprofiler->m_counterName2Id.find(counter_name) == rocprofiler->m_counterName2Id.end())
                    {
                        std::cerr << "Counter " << counter_name << " not available for this agent" << std::endl;
                        failed = true;
                    }
                }
                if (failed)
                    return ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND;
                rocprofiler_counter_config_id_t profile;
                status = rocprofiler_create_counter_config(rocprofiler->m_agent.id, counter_ids.data(), counter_ids.size(), &profile);
                if (status != ROCPROFILER_STATUS_SUCCESS)
                {
                    std::cerr << "Failed to create counter profile" << std::endl;
                    return status;
                }
                rocprofiler->m_agentProfile = profile;
                return ROCPROFILER_STATUS_SUCCESS;
            };
            ROCPROFILER_CALL(rocprofiler_iterate_agent_supported_counters(m_agent.id, counter_cb, this),
                             "Failed to query counters for agent");
        }

        bool RocProfiler::start() {
            if (!m_context_started) {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (m_context.handle != 0) {
                    rocprofiler_start_context(m_context);
                    m_context_started = true;
                }
            }
            return m_context_started;
        }

        void RocProfiler::stop() {
            if (m_context_started && m_context.handle != 0) {
                std::lock_guard<std::mutex> lock(m_mutex);
                rocprofiler_stop_context(m_context);
                m_context_started = false;
            }
        }

        std::string RocProfiler::fetch(int index) {
            std::lock_guard<std::mutex> lock(m_mutex);
            auto it = m_profiler->m_solutionIdx2DispatchId.find(index);
            if (it == m_profiler->m_solutionIdx2DispatchId.end())
                throw std::runtime_error("no counter data for solution " + std::to_string(index));
            auto dispatchId = it->second;
            auto counters = m_profiler->m_dispatchId2ProfileInfo[dispatchId].counters;
            std::ostringstream ss;
            int i = 0;
            for (const auto& [name, counter_id] : m_counterName2Id) {
                auto cit = counters.find(counter_id.handle);
                if (cit == counters.end())
                    throw std::runtime_error("counter " + name + " value not found.");
                auto value = cit->second;
                if (i != 0) {
                    ss << ",";
                }
                ss << name << ": " << value;
                i++;
            }
            return ss.str();
        }

        void RocProfiler::shutdown() {
            if (m_initialized) {
                stop();
                m_initialized = false;
            }
        }

        int RocProfiler::tool_init_impl(rocprofiler_client_finalize_t fini_func, void* tool_data)
        {
            auto* rocprofiler = static_cast<RocProfiler*>(tool_data);

            // Create context
            if (!ROCPROFILER_CHECK(rocprofiler_create_context(&rocprofiler->m_context), "Failed to create context in tool_init"))
                return -1;

            // configure dispatch counting service
            if (!ROCPROFILER_CHECK(rocprofiler_configure_callback_dispatch_counting_service(rocprofiler->m_context,
                                                                                            RocProfiler::dispatchCallback, tool_data,
                                                                                            RocProfiler::recordCallback, tool_data),
                                   "Failed to configure callback dispatch counting service"))
                return -1;

            return 0;
        }

        void RocProfiler::dispatchCallback(
                              rocprofiler_dispatch_counting_service_data_t dispatch_data,
                              rocprofiler_counter_config_id_t* config,
                              rocprofiler_user_data_t* user_data,
                              void* callback_data)
        {
            auto* rocprofiler = static_cast<RocProfiler*>(callback_data);
            std::lock_guard<std::mutex> lock(rocprofiler->m_mutex);
            if (rocprofiler->m_do) {
                *config = rocprofiler->m_agentProfile;
                user_data->value = rocprofiler->m_profiler->m_currentSolutionIdx;
            } else {
                *config = rocprofiler_counter_config_id_t{0}; // no profiling
            }
        }

        void RocProfiler::recordCallback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                                         rocprofiler_counter_record_t* record_data,
                                         unsigned long record_count,
                                         rocprofiler_user_data_t user_data,
                                         void* callback_data)
        {
            auto* rocprofiler = static_cast<RocProfiler*>(callback_data);
            std::lock_guard<std::mutex> lock(rocprofiler->m_mutex);
            int solutionIdx = user_data.value;
            auto dispatch_id = dispatch_data.dispatch_info.dispatch_id;
            rocprofiler->m_profiler->m_solutionIdx2DispatchId[solutionIdx] = dispatch_id;
            Profiler::ProfileInfo pinfo;
            pinfo.kernel_id = dispatch_data.dispatch_info.kernel_id;
            pinfo.execution_time_us = (dispatch_data.end_timestamp - dispatch_data.start_timestamp) / 1e3;
            for (unsigned long i = 0; i < record_count; ++i) {
                const auto& record = record_data[i];
                rocprofiler_counter_id_t counter_id = {.handle = 0};
                rocprofiler_query_record_counter_id(record.id, &counter_id);
                pinfo.counters[counter_id.handle] = record.counter_value;
            }
            rocprofiler->m_profiler->m_dispatchId2ProfileInfo.emplace(dispatch_id, pinfo);
        }

        namespace rocprof
        {
            inline int tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
            {
                return TensileLite::Client::RocProfiler::tool_init_impl(fini_func, tool_data);
            }
        }
    }
}

extern "C" {
    rocprofiler_tool_configure_result_t*
    rocprofiler_configure(uint32_t version, const char* runtime_version, uint32_t priority, rocprofiler_client_id_t* client_id)
    {
        // Initialize result structure
        static rocprofiler_tool_configure_result_t result;
        result.size = sizeof(rocprofiler_tool_configure_result_t);
        result.initialize = TensileLite::Client::rocprof::tool_init;
        result.finalize = nullptr;
        result.tool_data = &TensileLite::Client::RocProfiler::getInstance();
        return &result;
    }
}
