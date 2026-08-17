// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

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
#include <regex>

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

            std::string dimensionNameAbbrev(std::string dimName)
            {
                if (dimName == "DIMENSION_SHADER_ENGINE") {
                    return "SE";
                } else if (dimName == "DIMENSION_SHADER_ARRAY") {
                    return "SA";
                } else if (dimName == "DIMENSION_INSTANCE") {
                    return "INST";
                } else if (dimName.rfind("DIMENSION_", 0) == 0) {
                    return dimName.substr(10);
                } else {
                    return dimName;
                }
            }
        }

        bool RocProfiler::initialize(int deviceIdx, Profiler* profiler) {
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
                rocprofiler_counter_info_v1_t info;
                rocprofiler_status_t status;
                for (size_t i = 0; i < num_counters; i++) {
                    auto counter_id = counters[i];
                    status = rocprofiler_query_counter_info(counter_id, ROCPROFILER_COUNTER_INFO_VERSION_1, static_cast<void*>(&info));
                    if (status == ROCPROFILER_STATUS_SUCCESS)
                    {
                        if (rocprofiler->m_profiler->m_counterNames.count(info.name)) {
                            counter_ids.push_back(counter_id);
                            rocprofiler->m_counterName2Id[info.name] = counter_id;
                            auto* dims_p = *info.dimensions;
                            RocProfiler::CounterInfo counterInfo{
                                .id = counter_id.handle,
                                .dimInfos = std::vector<rocprofiler_counter_record_dimension_info_t>(dims_p, dims_p + info.dimensions_count) };
                            auto& strides = counterInfo.strides;
                            strides.push_back(1);
                            for (size_t j = 0; j < counterInfo.dimInfos.size() - 1; j++) {
                                auto& dim = counterInfo.dimInfos[j];
                                strides.push_back(dim.instance_size * strides.back());
                            }
                            counterInfo.num = counterInfo.dimInfos.back().instance_size * strides.back();
                            auto& dimNames = counterInfo.dimNames;
                            for (auto& dim : counterInfo.dimInfos) {
                                std::string name = dim.name;
                                dimNames.push_back(rocprof::dimensionNameAbbrev(name));
                            }
                            rocprofiler->m_counterInfos.emplace(counter_id.handle, std::move(counterInfo));
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
            m_future.get(); // wait callback finished
            std::lock_guard<std::mutex> lock(m_mutex);
            auto it = m_profiler->m_solutionIdx2DispatchId.find(index);
            if (it == m_profiler->m_solutionIdx2DispatchId.end())
                throw std::runtime_error("no counter data for solution " + std::to_string(index));
            auto dispatchId = it->second;
            auto& counters = m_profiler->m_dispatchId2ProfileInfo[dispatchId].counters;
            std::ostringstream ss;
            size_t i = 0;
            for (const auto& [name, counter_id] : m_counterName2Id) {
                auto cit = counters.find(counter_id.handle);
                if (cit == counters.end())
                    throw std::runtime_error("counter " + name + " value not found.");
                if (i != 0) ss << ",";
                auto value = cit->second;
                if (std::holds_alternative<double>(value)) {
                    auto counterValue = std::get<double>(value);
                    ss << name << ": ";
                    if (std::trunc(counterValue) == counterValue) {
                        ss << std::fixed << std::setprecision(0);
                    } else {
                        ss << std::defaultfloat;
                    }
                    ss << counterValue;
                } else {
                    auto& counterValues = std::get<std::vector<double>>(value);
                    auto& counterInfo = m_counterInfos[counter_id.handle];
                    auto& strides = counterInfo.strides;
                    for (size_t lindex = 0; lindex < counterInfo.num; lindex++) {
                        if (lindex != 0) ss << ",";
                        ss << name << "(";
                        size_t pos = lindex, ndim = strides.size();
                        for (size_t j = 0; j < ndim; j++) {
                            size_t k = ndim - 1 - j;
                            size_t stride = strides[k];
                            size_t idx = pos / stride;
                            ss << counterInfo.dimNames[k] << "=" << idx;
                            if (k != 0) ss << ",";
                            pos %= stride;
                        }
                        auto counterValue = counterValues[lindex];
                        ss << "): ";
                        if (std::trunc(counterValue) == counterValue) {
                            ss << std::fixed << std::setprecision(0);
                        } else {
                            ss << std::defaultfloat;
                        }
                        ss << counterValue;
                    }
                }
                i++;
            }
            return ss.str();
        }

        uint64_t RocProfiler::getKernelId(std::string kernelName) {
            auto it = m_kernelName2Id.find(kernelName);
            if (it == m_kernelName2Id.end())
                throw std::runtime_error("kernel id not found: " + kernelName);
            return it->second;
        }

        void RocProfiler::shutdown() {
            if (m_initialized) {
                stop();
                rocprofiler_destroy_counter_config(m_agentProfile);
                m_initialized = false;
            }
        }

        int RocProfiler::tool_init_impl(rocprofiler_client_finalize_t fini_func, void* tool_data)
        {
            auto* rocprofiler = static_cast<RocProfiler*>(tool_data);

            // Create context
            if (!ROCPROFILER_CHECK(rocprofiler_create_context(&rocprofiler->m_context), "Failed to create context in tool_init"))
                return -1;

            // configure tracing service for kernel loading
            rocprofiler_tracing_operation_t operation = ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER;
            if (!ROCPROFILER_CHECK(rocprofiler_configure_callback_tracing_service(rocprofiler->m_context,
                                                                                  ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                                                  &operation, 1,
                                                                                  RocProfiler::kernelLoadingCallback, tool_data),
                                   "Failed to configure callback tracing service"))
                return -1;

            // configure dispatch counting service
            if (!ROCPROFILER_CHECK(rocprofiler_configure_callback_dispatch_counting_service(rocprofiler->m_context,
                                                                                            RocProfiler::dispatchCallback, tool_data,
                                                                                            RocProfiler::recordCallback, tool_data),
                                   "Failed to configure callback dispatch counting service"))
                return -1;

            return 0;
        }

        void RocProfiler::kernelLoadingCallback(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t *user_data, void *callback_data)
        {
            auto* rocprofiler = static_cast<RocProfiler*>(callback_data);
            std::lock_guard<std::mutex> lock(rocprofiler->m_mutex);
            if (record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD &&
                record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
                record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER) {
                auto* kernelinfo = static_cast<rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t*>(record.payload);
                std::string kernel_name = std::regex_replace(kernelinfo->kernel_name, std::regex{"(\\.kd)$"}, "");
                rocprofiler->m_kernelName2Id.emplace(std::move(kernel_name), kernelinfo->kernel_id);
            }
        }

        void RocProfiler::dispatchCallback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                                           rocprofiler_counter_config_id_t* config,
                                           rocprofiler_user_data_t* user_data,
                                           void* callback_data)
        {
            auto* rocprofiler = static_cast<RocProfiler*>(callback_data);
            std::lock_guard<std::mutex> lock(rocprofiler->m_mutex);
            if (rocprofiler->m_do && dispatch_data.dispatch_info.kernel_id == rocprofiler->m_profiler->m_currentKernelId) {
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
            auto& counters = pinfo.counters;
            for (size_t i = 0; i < record_count; ++i) {
                const auto& record = record_data[i];
                rocprofiler_counter_id_t counter_id = {.handle = 0};
                rocprofiler_query_record_counter_id(record.id, &counter_id);
                std::ostringstream ss;
                auto& counterInfo = rocprofiler->m_counterInfos[counter_id.handle];
                if (counterInfo.num <= 1) {
                    counters[counterInfo.id] = record.counter_value;
                } else {
                    size_t index = 0;
                    for (size_t j = 0; j < counterInfo.dimInfos.size(); j++) {
                        auto& dim = counterInfo.dimInfos[j];
                        size_t pos = 0;
                        rocprofiler_query_record_dimension_position(record.id, dim.id, &pos);
                        index += pos * counterInfo.strides[j];
                    }
                    auto it = counters.find(counterInfo.id);
                    if (it == counters.end()) {
                        std::vector<double> counterValues(counterInfo.num);
                        counterValues[index] = record.counter_value;
                        counters.emplace(counterInfo.id, std::move(counterValues));
                    } else {
                        auto& counterValues = std::get<std::vector<double>>(it->second);
                        counterValues[index] = record.counter_value;
                    }
                }
            }
            rocprofiler->m_profiler->m_dispatchId2ProfileInfo.emplace(dispatch_id, pinfo);
            rocprofiler->m_promise.set_value(); // notify finished
        }

        namespace rocprof
        {
            inline int tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
            {
                return RocProfiler::tool_init_impl(fini_func, tool_data);
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
