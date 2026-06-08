// Probe heuristic plugin: logs that it was called and returns engine IDs unchanged.

#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>
#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>
#include <hipdnn_plugin_sdk/heuristic_api_version.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace
{

constexpr const char* kPluginName = "ProbeHeuristicPlugin";
constexpr const char* kPluginVersion = "0.1.0";
constexpr const char* kPolicyName = "ProbeHeuristic::PassThrough";

thread_local std::string g_last_error;

int64_t policy_id()
{
    return hipdnn_data_sdk::utilities::policyNameToId(kPolicyName);
}

void log(const char* msg)
{
    std::fprintf(stderr, "[PROBE_HEURISTIC] %s\n", msg);
    std::fflush(stderr);
}

struct ProbeHandle
{
};

struct ProbePolicy
{
    int64_t policy_id = 0;
    std::vector<int64_t> engine_ids;
    bool has_graph = false;
    bool finalized = false;
};

} // namespace

extern "C"
{

hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)
{
    if(name == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *name = kPluginName;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)
{
    if(version == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *version = kPluginVersion;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnPluginGetApiVersion(const char** version)
{
    if(version == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *version = HIPDNN_HEURISTIC_API_VERSION;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)
{
    if(type == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *type = HIPDNN_PLUGIN_TYPE_HEURISTIC;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

void hipdnnPluginGetLastErrorString(const char** error_str)
{
    if(error_str != nullptr)
    {
        *error_str = g_last_error.c_str();
    }
}

hipdnnPluginStatus_t hipdnnPluginSetLoggingCallback(hipdnnCallback_t)
{
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnPluginSetLogLevel(hipdnnSeverity_t)
{
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t
    hipdnnHeuristicPluginGetAllPolicyIds(int64_t* policy_ids, uint32_t max_policies, uint32_t* num_policies)
{
    if(num_policies == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    if(max_policies == 0)
    {
        *num_policies = 1;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    if(policy_ids == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    policy_ids[0] = policy_id();
    *num_policies = 1;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnHeuristicPluginGetPolicyName(int64_t id, const char** name)
{
    if(name == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    if(id != policy_id())
    {
        g_last_error = "unknown policy id";
        return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
    }
    *name = kPolicyName;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnHeuristicHandleCreate(hipdnnHeuristicHandle_t* out_handle)
{
    if(out_handle == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *out_handle = reinterpret_cast<hipdnnHeuristicHandle_t>(new ProbeHandle{});
    log("HandleCreate");
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnHeuristicHandleDestroy(hipdnnHeuristicHandle_t handle)
{
    delete reinterpret_cast<ProbeHandle*>(handle);
    log("HandleDestroy");
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t
    hipdnnHeuristicHandleSetDeviceProperties(hipdnnHeuristicHandle_t, const hipdnnPluginConstData_t* device_props)
{
    std::string msg = "HandleSetDeviceProperties size="
                      + std::to_string(device_props != nullptr ? device_props->size : 0);
    log(msg.c_str());
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnHeuristicPolicyDescriptorCreate(
    hipdnnHeuristicHandle_t, int64_t id, hipdnnHeuristicPolicyDescriptor_t* out_desc)
{
    if(out_desc == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    if(id != policy_id())
    {
        g_last_error = "unknown policy id";
        return HIPDNN_PLUGIN_STATUS_INVALID_VALUE;
    }
    *out_desc = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(new ProbePolicy{id});
    log("PolicyDescriptorCreate");
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnHeuristicPolicyDescriptorDestroy(hipdnnHeuristicPolicyDescriptor_t desc)
{
    delete reinterpret_cast<ProbePolicy*>(desc);
    log("PolicyDescriptorDestroy");
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t
    hipdnnHeuristicPolicySetEngineIds(hipdnnHeuristicPolicyDescriptor_t desc,
                                      const int64_t* ids,
                                      size_t count)
{
    if(desc == nullptr || (count > 0 && ids == nullptr))
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    auto* policy = reinterpret_cast<ProbePolicy*>(desc);
    policy->engine_ids.assign(ids, ids + count);
    std::string msg = "PolicySetEngineIds count=" + std::to_string(count);
    for(auto id : policy->engine_ids)
    {
        msg += " " + std::to_string(id);
    }
    log(msg.c_str());
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t
    hipdnnHeuristicPolicySetSerializedGraph(hipdnnHeuristicPolicyDescriptor_t desc,
                                            const hipdnnPluginConstData_t* graph)
{
    if(desc == nullptr || graph == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    auto* policy = reinterpret_cast<ProbePolicy*>(desc);
    policy->has_graph = graph->ptr != nullptr && graph->size > 0;
    std::string msg = "PolicySetSerializedGraph size=" + std::to_string(graph->size);
    log(msg.c_str());
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnHeuristicPolicyFinalize(hipdnnHeuristicPolicyDescriptor_t desc,
                                                   int32_t* out_applied)
{
    if(desc == nullptr || out_applied == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    auto* policy = reinterpret_cast<ProbePolicy*>(desc);
    policy->finalized = true;
    *out_applied = 1;
    log("PolicyFinalize called - HEURISTIC IS ACTIVE");
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t
    hipdnnHeuristicPolicyGetSortedEngineIds(hipdnnHeuristicPolicyDescriptor_t desc,
                                            int64_t* out_ids,
                                            size_t* inout_count)
{
    if(desc == nullptr || inout_count == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    auto* policy = reinterpret_cast<ProbePolicy*>(desc);
    if(!policy->finalized)
    {
        return HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED;
    }
    if(out_ids == nullptr)
    {
        *inout_count = policy->engine_ids.size();
        log("PolicyGetSortedEngineIds count query");
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    const auto n = std::min(*inout_count, policy->engine_ids.size());
    std::memcpy(out_ids, policy->engine_ids.data(), n * sizeof(int64_t));
    *inout_count = n;
    log("PolicyGetSortedEngineIds retrieve");
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

} // extern "C"
