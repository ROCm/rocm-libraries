// Convolution regime heuristic plugin: detects conv regimes and falls through without routing.

#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>
#include <hipdnn_plugin_sdk/heuristic_api_version.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <exception>
#include <optional>
#include <string>
#include <vector>

namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace flatbuffer_utilities = hipdnn_flatbuffers_sdk::flatbuffer_utilities;

constexpr const char* kPluginName = "ConvRegimeHeuristicPlugin";
constexpr const char* kPluginVersion = "0.1.0";
constexpr const char* kPolicyName = "ConvHeuristic::RegimeClassifier";

thread_local std::string g_last_error;

int64_t policy_id()
{
    return hipdnn_data_sdk::utilities::policyNameToId(kPolicyName);
}

void log(const char* msg)
{
    std::fprintf(stderr, "[CONV_HEURISTIC] %s\n", msg);
    std::fflush(stderr);
}

void set_error(const std::string& error)
{
    g_last_error = error;
    log(error.c_str());
}

struct ConvHandle
{
};

struct ConvRegimeInfo
{
    std::array<int64_t, 4> x_dims{};
    std::array<int64_t, 4> w_dims{};
    int64_t stride_h = 1;
    int64_t stride_w = 1;
    int64_t pad_h = 0;
    int64_t pad_w = 0;
    data_objects::DataType data_type = data_objects::DataType::UNSET;
};

struct ConvPolicy
{
    int64_t policy_id = 0;
    std::vector<int64_t> engine_ids;
    std::optional<ConvRegimeInfo> regime_info;
    bool finalized = false;
};

std::optional<std::array<int64_t, 4>> get_4d_dims(const data_objects::TensorAttributes* tensor)
{
    if(tensor == nullptr || tensor->dims() == nullptr || tensor->dims()->size() < 4)
    {
        return std::nullopt;
    }

    return std::array<int64_t, 4>{tensor->dims()->Get(0),
                                  tensor->dims()->Get(1),
                                  tensor->dims()->Get(2),
                                  tensor->dims()->Get(3)};
}

int64_t vector_value_or(const flatbuffers::Vector<int64_t>* values, size_t index, int64_t fallback)
{
    if(values == nullptr || values->size() <= index)
    {
        return fallback;
    }
    return values->Get(index);
}

std::optional<ConvRegimeInfo> detect_conv_regime(const hipdnnPluginConstData_t* graph)
{
    if(graph == nullptr || graph->ptr == nullptr || graph->size == 0)
    {
        set_error("serialized graph is empty");
        return std::nullopt;
    }

    flatbuffer_utilities::GraphWrapper wrapper(graph->ptr, graph->size);
    if(!wrapper.isValid())
    {
        set_error("serialized graph is not a valid hipDNN graph flatbuffer");
        return std::nullopt;
    }

    const auto& tensor_map = wrapper.getTensorMap();
    for(const auto& node_wrapper : wrapper.nodeWrappers())
    {
        if(node_wrapper == nullptr
           || node_wrapper->attributesType() != data_objects::NodeAttributes::ConvolutionFwdAttributes)
        {
            continue;
        }

        const auto& attrs = node_wrapper->attributesAs<data_objects::ConvolutionFwdAttributes>();
        const auto x_it = tensor_map.find(attrs.x_tensor_uid());
        const auto w_it = tensor_map.find(attrs.w_tensor_uid());
        if(x_it == tensor_map.end() || w_it == tensor_map.end())
        {
            set_error("conv node references missing input or filter tensor");
            return std::nullopt;
        }

        const auto x_dims = get_4d_dims(x_it->second);
        const auto w_dims = get_4d_dims(w_it->second);
        if(!x_dims || !w_dims)
        {
            set_error("conv input or filter tensor does not have at least 4 dims");
            return std::nullopt;
        }

        const auto* pre_padding = attrs.pre_padding();
        return ConvRegimeInfo{*x_dims,
                              *w_dims,
                              vector_value_or(attrs.stride(), 0, 1),
                              vector_value_or(attrs.stride(), 1, 1),
                              vector_value_or(pre_padding, 0, 0),
                              vector_value_or(pre_padding, 1, 0),
                              x_it->second->data_type()};
    }

    log("no convolution forward node found in serialized graph");
    return std::nullopt;
}

void log_regime(const ConvRegimeInfo& info)
{
    const int64_t n = info.x_dims[0];
    const int64_t c = info.x_dims[1];
    const int64_t h = info.x_dims[2];
    const int64_t w = info.x_dims[3];
    const int64_t k = info.w_dims[0];
    const int64_t filter_c = info.w_dims[1];
    const int64_t r = info.w_dims[2];
    const int64_t s = info.w_dims[3];

    const char* regime = "GENERAL_CONV";
    if(filter_c == 1 && c > 1 && k == c)
    {
        regime = "DEPTHWISE";
    }
    else if(r == 1 && s == 1)
    {
        regime = "GEMM_CONV";
    }
    else if(r == 3 && s == 3 && info.stride_h == 1 && info.stride_w == 1)
    {
        regime = "WINOGRAD";
    }
    else if(r > 3 || s > 3)
    {
        regime = "DIRECT";
    }

    std::fprintf(stderr,
                 "[CONV_HEURISTIC] regime=%s N=%ld C=%ld H=%ld W=%ld K=%ld filterC=%ld R=%ld "
                 "S=%ld stride=%ldx%ld pad=%ldx%ld dtype=%s\n",
                 regime,
                 static_cast<long>(n),
                 static_cast<long>(c),
                 static_cast<long>(h),
                 static_cast<long>(w),
                 static_cast<long>(k),
                 static_cast<long>(filter_c),
                 static_cast<long>(r),
                 static_cast<long>(s),
                 static_cast<long>(info.stride_h),
                 static_cast<long>(info.stride_w),
                 static_cast<long>(info.pad_h),
                 static_cast<long>(info.pad_w),
                 data_objects::EnumNameDataType(info.data_type));
    std::fflush(stderr);
}

} // namespace

extern "C"
{

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)
{
    if(name == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *name = kPluginName;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)
{
    if(version == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *version = kPluginVersion;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginGetApiVersion(const char** version)
{
    if(version == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *version = HIPDNN_HEURISTIC_API_VERSION;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)
{
    if(type == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *type = HIPDNN_PLUGIN_TYPE_HEURISTIC;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT void hipdnnPluginGetLastErrorString(const char** error_str)
{
    if(error_str != nullptr)
    {
        *error_str = g_last_error.c_str();
    }
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginSetLoggingCallback(hipdnnCallback_t)
{
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginSetLogLevel(hipdnnSeverity_t)
{
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPluginGetAllPolicyIds(int64_t* policy_ids,
                                         uint32_t max_policies,
                                         uint32_t* num_policies)
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

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPluginGetPolicyName(int64_t id, const char** name)
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

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicHandleCreate(hipdnnHeuristicHandle_t* out_handle)
{
    if(out_handle == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *out_handle = reinterpret_cast<hipdnnHeuristicHandle_t>(new ConvHandle{});
    log("HandleCreate");
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicHandleDestroy(hipdnnHeuristicHandle_t handle)
{
    delete reinterpret_cast<ConvHandle*>(handle);
    log("HandleDestroy");
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicHandleSetDeviceProperties(hipdnnHeuristicHandle_t,
                                             const hipdnnPluginConstData_t* device_props)
{
    std::string msg = "HandleSetDeviceProperties size="
                      + std::to_string(device_props != nullptr ? device_props->size : 0);
    log(msg.c_str());
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnHeuristicPolicyDescriptorCreate(
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
    *out_desc = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(new ConvPolicy{id});
    log("PolicyDescriptorCreate");
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPolicyDescriptorDestroy(hipdnnHeuristicPolicyDescriptor_t desc)
{
    delete reinterpret_cast<ConvPolicy*>(desc);
    log("PolicyDescriptorDestroy");
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPolicySetEngineIds(hipdnnHeuristicPolicyDescriptor_t desc,
                                      const int64_t* ids,
                                      size_t count)
{
    if(desc == nullptr || (count > 0 && ids == nullptr))
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    auto* policy = reinterpret_cast<ConvPolicy*>(desc);
    policy->engine_ids.assign(ids, ids + count);
    log(("PolicySetEngineIds count=" + std::to_string(count)).c_str());
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPolicySetSerializedGraph(hipdnnHeuristicPolicyDescriptor_t desc,
                                            const hipdnnPluginConstData_t* graph)
{
    if(desc == nullptr || graph == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

    auto* policy = reinterpret_cast<ConvPolicy*>(desc);
    log(("PolicySetSerializedGraph size=" + std::to_string(graph->size)).c_str());
    try
    {
        policy->regime_info = detect_conv_regime(graph);
    }
    catch(const std::exception& e)
    {
        set_error(std::string("failed to parse serialized graph: ") + e.what());
        policy->regime_info.reset();
    }
    catch(...)
    {
        set_error("failed to parse serialized graph: unknown exception");
        policy->regime_info.reset();
    }
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPolicyFinalize(hipdnnHeuristicPolicyDescriptor_t desc, int32_t* out_applied)
{
    if(desc == nullptr || out_applied == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

    auto* policy = reinterpret_cast<ConvPolicy*>(desc);
    policy->finalized = true;
    if(policy->regime_info)
    {
        log_regime(*policy->regime_info);
    }
    else
    {
        log("PolicyFinalize no convolution regime detected");
    }

    *out_applied = 0;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

HIPDNN_HEURISTIC_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnHeuristicPolicyGetSortedEngineIds(hipdnnHeuristicPolicyDescriptor_t desc,
                                            int64_t* out_ids,
                                            size_t* inout_count)
{
    if(desc == nullptr || inout_count == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

    auto* policy = reinterpret_cast<ConvPolicy*>(desc);
    if(!policy->finalized)
    {
        return HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED;
    }
    if(out_ids == nullptr)
    {
        *inout_count = policy->engine_ids.size();
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }

    const auto n = std::min(*inout_count, policy->engine_ids.size());
    std::memcpy(out_ids, policy->engine_ids.data(), n * sizeof(int64_t));
    *inout_count = n;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

} // extern "C"
