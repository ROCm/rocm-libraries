// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "Agent.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/cxx/codeobj/code_printing.hpp>
#include <rocprofiler-sdk/experimental/thread_trace.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/HipUtils.hpp>
#include <rocRoller/Utilities/Logging.hpp>
#include <rocRoller/Utilities/Settings.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <thread>

// Based on https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-sdk/samples/thread_trace

#define ROCPROFILER_CALL(result, msg)                                                   \
    if(auto ec = (result); ec != ROCPROFILER_STATUS_SUCCESS)                            \
    {                                                                                   \
        rocRoller::Throw<rocRoller::FatalError>(                                        \
            "rocprofiler-sdk error: ", rocprofiler_get_status_string(ec), " :: ", msg); \
    }

// Set to false when another profiler (e.g. rocprofv3) is registered first
bool enable_agent = true;

namespace rocRoller
{
    struct TraceDecodeCallbackUserData
    {
        bool                            ok;
        profiler::InstructionLatencyMap instruction_map;
    };

    rocprofiler_thread_trace_decoder_id_t decoder{};
    rocprofiler_context_id_t              client_ctx{};

    rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate address_table;

    // To enable waiting for dispatch data
    std::vector<uint8_t>    profile_data;
    std::condition_variable profile_data_cv;
    std::mutex              profile_data_mutex;

    bool       enable_profiler = false;
    std::mutex enable_profiler_mutex;

    void codeobj_callback(rocprofiler_callback_tracing_record_t record,
                          rocprofiler_user_data_t*,
                          void*)
    {
        if(record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT
           || record.operation != ROCPROFILER_CODE_OBJECT_LOAD
           || record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD)
            return;

        auto* data
            = static_cast<rocprofiler_callback_tracing_code_object_load_data_t*>(record.payload);

        Log::debug("codeobj_callback: code_object_id {}, storage_type {}",
                   data->code_object_id,
                   static_cast<std::underlying_type_t<rocprofiler_code_object_storage_type_t>>(
                       data->storage_type));

        if(data->storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE)
        {
            address_table.addDecoder(
                data->uri, data->code_object_id, data->load_delta, data->load_size);
        }
        else
        {
            auto* memorybase = reinterpret_cast<const void*>(data->memory_base);
            if(memorybase)
            {
                rocprofiler_thread_trace_decoder_codeobj_load(decoder,
                                                              data->code_object_id,
                                                              data->load_delta,
                                                              data->load_size,
                                                              memorybase,
                                                              data->memory_size);

                address_table.addDecoder(memorybase,
                                         data->memory_size,
                                         data->code_object_id,
                                         data->load_delta,
                                         data->load_size);
            }
        }
    }

    void trace_decode_callback(rocprofiler_thread_trace_decoder_record_type_t record_type_id,
                               void*                                          events,
                               uint64_t                                       num_events,
                               void*                                          userdata_raw)
    {
        /*
        Don't use AssertFatal macro, as that is caught as a rocprofiler error and masks the real issue.
        Handle other record types https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/api-reference/thread_trace.html#trace-decoder-info-events
        */
        Log::debug(
            "trace_decode_callback: record_type_id {}, num_events {}",
            static_cast<std::underlying_type_t<rocprofiler_thread_trace_decoder_record_type_t>>(
                record_type_id),
            num_events);

        TraceDecodeCallbackUserData* userdata
            = static_cast<TraceDecodeCallbackUserData*>(userdata_raw);

        if(!userdata->ok)
            return;

        switch(record_type_id)
        {
        case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE:
        {
            assert(num_events == 1 && "expected one event for WAVE record type");
            auto* wave = static_cast<rocprofiler_thread_trace_decoder_wave_t*>(events);
            Log::debug("  wave : cu {}, simd {}, wave_id {}, contexts {}, instructions_size {}",
                       wave->cu,
                       wave->simd,
                       wave->wave_id,
                       wave->contexts,
                       wave->instructions_size);

            assert(wave->instructions_size >= 1 && "expected at least one instruction in wave");
            uint32_t prev_time    = wave->instructions_array[0].time;
            uint32_t prev_latency = 0;
            for(size_t i = 0; i < wave->instructions_size; i++)
            {
                auto& inst = wave->instructions_array[i];

                Log::debug("    inst {}: code_object_id {}, address 0x{:x}, duration {}",
                           i,
                           inst.pc.code_object_id,
                           inst.pc.address,
                           inst.duration);

                if(inst.pc.code_object_id == 0)
                {
                    Log::warn("trace_decode_callback: code_object_id is 0");
                    assert(userdata->instruction_map.empty()
                           && "expected all instructions to have code_object_id of 0 if any do");
                    userdata->ok = false;
                    return;
                }

                uint32_t latencyWithPrecedingNone = inst.time - prev_time;
                Log::trace("trace_decode_callback: duration {}, stall {}, time {}, "
                           "prev_time {}, prev_latency {}, latencyWithPrecedingNone {}",
                           inst.duration,
                           static_cast<uint32_t>(inst.stall),
                           inst.time,
                           prev_time,
                           prev_latency,
                           latencyWithPrecedingNone);
                prev_time    = inst.time;
                prev_latency = inst.duration;

                auto& data = userdata->instruction_map[inst.pc];
                data.totalLatency += inst.duration;
                data.totalLatencyWithPrecedingNone += latencyWithPrecedingNone;
                data.hitcount += 1;
            }
            return;
        }
        case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY:
        case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_GFXIP:
        case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_REALTIME:
        case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_RT_FREQUENCY:
            // these data not used currently
            return;

        case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_INFO:
        {
            assert(num_events == 1 && "expected one event for INFO record type");
            auto* info = static_cast<rocprofiler_thread_trace_decoder_info_t*>(events);
            Log::warn("parse: record info {}",
                      static_cast<std::underlying_type_t<rocprofiler_thread_trace_decoder_info_t>>(
                          *info));
            // Only seen these enumerations; investigate if others are encountered
            assert(((*info == ROCPROFILER_THREAD_TRACE_DECODER_INFO_STITCH_INCOMPLETE)
                    || (*info == ROCPROFILER_THREAD_TRACE_DECODER_INFO_DATA_LOST))
                   && "received unexpected INFO record type");
            userdata->ok = false;
            userdata->instruction_map.clear();
            return;
        }
        default:
            Log::error(
                "parse: unhandled record type {}",
                static_cast<std::underlying_type_t<rocprofiler_thread_trace_decoder_record_type_t>>(
                    record_type_id));
            assert(!"unhandled record type");
            return;
        }
    }

    void shader_data_callback(rocprofiler_agent_id_t  agent,
                              int64_t                 shader_engine_id,
                              void*                   data,
                              size_t                  data_size,
                              rocprofiler_user_data_t userdata)
    {
        // Based on https://github.com/ROCm/rocm-systems/blob/e9dac39102606ac6f7ab2778f74745e27841fb6c/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk-tool/tool.cpp#L1387-L1408
        // Avoid decoding directly in this function body per https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/api-reference/thread_trace.html#processing-thread-trace-data

        rocprofiler_dispatch_id_t dispatch_id
            = static_cast<rocprofiler_dispatch_id_t>(userdata.value);

        Log::debug("shader_data_callback: dispatch_id {}, data_size {}", dispatch_id, data_size);

        // Note this exists to also prevent a deadlock, as profile_data.size() is checked under profile_data_cv.wait
        assert(data != nullptr && data_size > 0 && "invalid shader data callback");

        {
            std::lock_guard lock(profile_data_mutex);
            profile_data = std::vector<uint8_t>(static_cast<uint8_t*>(data),
                                                static_cast<uint8_t*>(data) + data_size);
        }
        profile_data_cv.notify_all();
    }

    rocprofiler_thread_trace_control_flags_t
        dispatch_callback(rocprofiler_agent_id_t             agent_id,
                          rocprofiler_queue_id_t             queue_id,
                          rocprofiler_async_correlation_id_t correlation_id,
                          rocprofiler_kernel_id_t            kernel_id,
                          rocprofiler_dispatch_id_t          dispatch_id,
                          void*                              userdata_config,
                          rocprofiler_user_data_t*           userdata_shader)
    {
        // Protect against multiple dispatches, current behavior is to only profile the first dispatch after enabling
        std::lock_guard lock(enable_profiler_mutex);

        Log::debug(
            "dispatch_callback: dispatch_id {}, enable_profiler {}", dispatch_id, enable_profiler);

        if(enable_profiler)
        {
            userdata_shader->value = dispatch_id;
            enable_profiler        = false;

            return ROCPROFILER_THREAD_TRACE_CONTROL_START_AND_STOP;
        }
        return ROCPROFILER_THREAD_TRACE_CONTROL_NONE;
    }

    rocprofiler_status_t query_agents(rocprofiler_agent_version_t,
                                      const void** agents,
                                      size_t       num_agents,
                                      void*        user_data)
    {
        for(size_t idx = 0; idx < num_agents; idx++)
        {
            const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents[idx]);
            if(agent->type != ROCPROFILER_AGENT_TYPE_GPU)
                continue;

            auto parameters = std::vector<rocprofiler_thread_trace_parameter_t>{};
            parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_TARGET_CU, 1});
            parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK, 0x1});
            parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SERIALIZE_ALL, 1});
            parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_BUFFER_SIZE, 0x100000000});

            ROCPROFILER_CALL(
                rocprofiler_configure_dispatch_thread_trace_service(client_ctx,
                                                                    agent->id,
                                                                    parameters.data(),
                                                                    parameters.size(),
                                                                    dispatch_callback,
                                                                    shader_data_callback,
                                                                    user_data),
                "configure dispatch thread trace");
        }
        return ROCPROFILER_STATUS_SUCCESS;
    }

    int tool_init(rocprofiler_client_finalize_t, void*)
    {
        ROCPROFILER_CALL(
            rocprofiler_thread_trace_decoder_create(
                &decoder, (Settings::getInstance()->get(Settings::ROCMPath) + "/lib").c_str()),
            "create decoder");

        ROCPROFILER_CALL(rocprofiler_create_context(&client_ctx), "create context");

        ROCPROFILER_CALL(
            rocprofiler_configure_callback_tracing_service(client_ctx,
                                                           ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                           nullptr,
                                                           0,
                                                           codeobj_callback,
                                                           nullptr),
            "configure code object tracing");

        ROCPROFILER_CALL(rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                                            &query_agents,
                                                            sizeof(rocprofiler_agent_t),
                                                            nullptr),
                         "query agents");

        ROCPROFILER_CALL(rocprofiler_start_context(client_ctx), "start context");

        return 0;
    }

    void tool_fini(void*)
    {
        rocprofiler_thread_trace_decoder_destroy(decoder);
    }

    namespace profiler
    {
        std::optional<std::vector<InstructionProfile>> waitForData()
        {
            if(!enable_agent)
                return std::nullopt;

            Log::debug("waitForData");

            std::unique_lock<std::mutex> lock(profile_data_mutex);

            profile_data_cv.wait(lock, [] { return !profile_data.empty(); });

            Log::debug("waitForData: acquired profile data, decoding {} bytes",
                       profile_data.size());

            TraceDecodeCallbackUserData callback_user_data{.ok = true, .instruction_map = {}};

            const auto status = rocprofiler_trace_decode(decoder,
                                                         trace_decode_callback,
                                                         profile_data.data(),
                                                         profile_data.size(),
                                                         &callback_user_data);

            Log::debug("waitForData: decoding complete, clearing {} bytes of profile data",
                       profile_data.size());

            profile_data.clear();
            lock.unlock();

            if(status != ROCPROFILER_STATUS_SUCCESS)
            {
                Log::warn("waitForData: decoding error, rocprofiler_trace_decode returned {}",
                          rocprofiler_get_status_string(status));
                return std::nullopt;
            }

            if(not callback_user_data.ok)
            {
                Log::warn("waitForData: decoding error in callback, check previous logs");
                return std::nullopt;
            }

            if(callback_user_data.instruction_map.empty())
            {
                Log::warn("waitForData: no instructions decoded");
                return std::nullopt;
            }

            for(auto& [pc, data] : callback_user_data.instruction_map)
            {
                auto inst = address_table.get(pc.code_object_id, pc.address);
                if(inst != nullptr)
                    data.instruction = inst->inst;
            }

            std::vector<InstructionProfile> result;
            result.reserve(callback_user_data.instruction_map.size());
            for(const auto& [pc, data] : callback_user_data.instruction_map)
            {
                result.push_back(data);
            }
            Log::debug("waitForData: retrieved {} instructions", result.size());

            return result;
        }

        std::optional<std::vector<InstructionProfile>>
            getDispatchData(std::function<void()> dispatch)
        {
            Log::debug("getDispatchData");

            HIP_CHECK(hipDeviceSynchronize()); // Ensure all prior dispatches finished

            {
                std::lock_guard lock(enable_profiler_mutex);
                enable_profiler = true;
            }
            dispatch();

            const auto data = waitForData();
            return data;
        }

        std::vector<InstructionProfile> loopUntilDispatchData(std::function<void()> dispatch)
        {
            Log::debug("loopUntilDispatchData: starting loop to get dispatch data");

            std::optional<std::vector<InstructionProfile>> data;

            while(true)
            {
                data = getDispatchData(dispatch);
                if(data.has_value())
                {
                    return *data;
                }
                if(!enable_agent)
                {
                    Log::debug(
                        "loopUntilDispatchData: profiler disabled, returning empty data set");
                    return {};
                }
                Log::debug("loopUntilDispatchData: got no data, invoking another dispatch");
            }
        }

        void reset()
        {
            std::scoped_lock lock{profile_data_mutex, enable_profiler_mutex};
            profile_data.clear();
            enable_profiler = false;
        }

        uint64_t InstructionProfile::meanLatency() const
        {
            if(hitcount == 0)
                return 0;
            return totalLatency / hitcount;
        }

        uint64_t InstructionProfile::meanLatencyWithPrecedingNone() const
        {
            if(hitcount == 0)
                return 0;
            return totalLatencyWithPrecedingNone / hitcount;
        }

        std::string InstructionProfile::toString() const
        {
            return fmt::format("'{}', totalLatency: {}, totalLatencyWithPrecedingNone: {}, "
                               "hitcount: {}, meanLatency: {}, meanLatencyWithPrecedingNone: {}",
                               instruction,
                               totalLatency,
                               totalLatencyWithPrecedingNone,
                               hitcount,
                               meanLatency(),
                               meanLatencyWithPrecedingNone());
        }

        std::string toString(std::vector<InstructionProfile> const& profiles)
        {
            std::ostringstream result;
            for(const auto& profile : profiles)
            {
                result << profile.toString() + "\n";
            }
            result << "\n";
            return result.str();
        }

    } // namespace profiler
} // namespace rocRoller

extern "C" rocprofiler_tool_configure_result_t*
    rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if(priority > 0)
    {
        enable_agent = false;
        return nullptr;
    }

    id->name = "rocRoller rocprofiler: I'm a string literal, just do a text search to find me :)";

    static auto cfg
        = rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                              &rocRoller::tool_init,
                                              &rocRoller::tool_fini,
                                              nullptr};

    return &cfg;
}
