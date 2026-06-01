// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ExecutionPlanDescriptor.hpp"
#include "BackendEnumStringUtils.hpp"
#include "EngineConfigDescriptor.hpp"
#include "EngineDescriptor.hpp"
#include "GraphDescriptor.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "handle/Handle.hpp"
#include <cstring>
#include <flatbuffers/flatbuffer_builder.h>
#include <flatbuffers/verifier.h>
#include <hipdnn_flatbuffers_sdk/data_objects/execution_plan_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <vector>

namespace hipdnn_backend
{
namespace
{
constexpr uint32_t PLAN_SERIALIZATION_VERSION = 1;

std::vector<int64_t> collectTensorUids(const GraphDescriptor& graph)
{
    auto serializedGraph = graph.getSerializedGraph();
    if(serializedGraph.ptr == nullptr || serializedGraph.size == 0)
    {
        return {};
    }

    auto* graphData = static_cast<const uint8_t*>(serializedGraph.ptr);
    flatbuffers::Verifier verifier(graphData, serializedGraph.size);
    THROW_IF_FALSE(hipdnn_flatbuffers_sdk::data_objects::VerifyGraphBuffer(verifier),
                   HIPDNN_STATUS_BAD_PARAM,
                   "ExecutionPlanDescriptor::finalize() failed: serialized graph is invalid.");

    auto graphObject = hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphData);
    auto tensors = graphObject->tensors();
    if(tensors == nullptr)
    {
        return {};
    }

    std::vector<int64_t> tensorUids;
    tensorUids.reserve(tensors->size());
    for(const auto* tensor : *tensors)
    {
        tensorUids.push_back(tensor->uid());
    }
    return tensorUids;
}
} // namespace

void ExecutionPlanDescriptor::finalize()
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "ExecutionPlanDescriptor::finalize() failed: Already finalized.");

    THROW_IF_NULL(_engineConfig,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ExecutionPlanDescriptor::finalize() failed: Engine was not set.");

    auto engine = _engineConfig->getEngine();
    auto engineId = engine->getEngineId();
    auto graph = engine->getGraph();
    auto handle = graph->getHandle();
    auto pluginResourceManager = handle->getPluginResourceManager();
    auto engineConfigPluginData = _engineConfig->getSerializedEngineConfig();
    _pluginResourceManager = pluginResourceManager;
    _engineId = engineId;
    _tensorUids = collectTensorUids(*graph);
    _isOverrideShapeEnabled = graph->isOverrideShapeEnabled();

    _executionContext = plugin::EnginePluginResourceManager::createExecutionContext(
        pluginResourceManager, engineId, &engineConfigPluginData, graph.get());

    _workspaceSize = static_cast<int64_t>(
        pluginResourceManager->getWorkspaceSize(engineId, _executionContext->get()));

    THROW_IF_LT(_workspaceSize,
                0,
                HIPDNN_STATUS_INTERNAL_ERROR,
                "ExecutionPlanDescriptor::finalize() failed: Invalid workspace size.");

    HipdnnBackendDescriptorImpl<ExecutionPlanDescriptor>::finalize();
}

void ExecutionPlanDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                           hipdnnBackendAttributeType_t attributeType,
                                           int64_t requestedElementCount,
                                           int64_t* elementCount,
                                           void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "ExecutionPlanDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE:
        getWorkspaceSize(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG:
        getEngineConfig(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_EXECUTION_PLAN_TENSOR_UIDS_EXT:
        getTensorUids(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_EXECUTION_PLAN_HANDLE:
    case HIPDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS:
    case HIPDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS:
    case HIPDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION:
    case HIPDNN_ATTR_EXECUTION_PLAN_KERNEL_CACHE:
    case HIPDNN_ATTR_EXECUTION_PLAN_DEVICEPROP:
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("ExecutionPlanDescriptor::getAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }
}

void ExecutionPlanDescriptor::getWorkspaceSize(hipdnnBackendAttributeType_t attributeType,
                                               int64_t requestedElementCount,
                                               int64_t* elementCount,
                                               void* arrayOfElements) const
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_INT64,
                HIPDNN_STATUS_BAD_PARAM,
                "ExecutionPlanDescriptor failed to get workspace size: Invalid attribute type.");

    THROW_IF_NE(requestedElementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "ExecutionPlanDescriptor failed to get workspace size: Invalid element count.");

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "ExecutionPlanDescriptor failed to get workspace size: Null pointer.");

    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }

    *static_cast<int64_t*>(arrayOfElements) = _workspaceSize;
}

void ExecutionPlanDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                           hipdnnBackendAttributeType_t attributeType,
                                           int64_t elementCount,
                                           const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "ExecutionPlanDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_EXECUTION_PLAN_HANDLE:
        setHandle(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG:
        setEngineConfig(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE:
    case HIPDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS:
    case HIPDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS:
    case HIPDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION:
    case HIPDNN_ATTR_EXECUTION_PLAN_KERNEL_CACHE:
    case HIPDNN_ATTR_EXECUTION_PLAN_DEVICEPROP:
    case HIPDNN_ATTR_EXECUTION_PLAN_TENSOR_UIDS_EXT:
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("ExecutionPlanDescriptor::setAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }
}

void ExecutionPlanDescriptor::setHandle(hipdnnBackendAttributeType_t attributeType,
                                        int64_t elementCount,
                                        const void* arrayOfElements)
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_HANDLE,
                HIPDNN_STATUS_BAD_PARAM,
                "ExecutionPlanDescriptor failed to set handle: Invalid attribute type.");
    THROW_IF_NE(elementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "ExecutionPlanDescriptor failed to set handle: Invalid element count.");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "ExecutionPlanDescriptor failed to set handle: Null pointer.");

    hipdnnHandle_t handle = *static_cast<const hipdnnHandle_t*>(arrayOfElements);

    THROW_IF_NULL(handle,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "ExecutionPlanDescriptor failed to set handle: Handle is null.");

    // Just ignore handle since it's deprecated for execution plan, but still do the checks and keep the API.
}

void ExecutionPlanDescriptor::setEngineConfig(hipdnnBackendAttributeType_t attributeType,
                                              int64_t elementCount,
                                              const void* arrayOfElements)
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "ExecutionPlanDescriptor failed to set engine config: Invalid attribute type.");

    THROW_IF_NE(elementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "ExecutionPlanDescriptor failed to set engine config: Invalid element count.");

    auto engineConfig = HipdnnBackendDescriptor::unpackDescriptor<const EngineConfigDescriptor>(
        arrayOfElements,
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
        "ExecutionPlanDescriptor failed to set engine config: Null pointer.");

    THROW_IF_FALSE(engineConfig->isFinalized(),
                   HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                   "ExecutionPlanDescriptor failed to set engine config: Engine config "
                   "descriptor is not finalized.");

    _engineConfig = engineConfig;
}

void ExecutionPlanDescriptor::getEngineConfig(hipdnnBackendAttributeType_t attributeType,
                                              int64_t requestedElementCount,
                                              int64_t* elementCount,
                                              void* arrayOfElements) const
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "ExecutionPlanDescriptor failed to get engine config: Invalid attribute type.");

    THROW_IF_NE(requestedElementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "ExecutionPlanDescriptor failed to get engine config: "
                "Invalid element count.");

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "ExecutionPlanDescriptor failed to get engine config: Null pointer for "
                  "arrayOfElements.");

    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }

    THROW_IF_NULL(_engineConfig,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "ExecutionPlanDescriptor failed to get engine config: Engine config is null. "
                  "Engine config was not set.");

    HipdnnBackendDescriptor::packDescriptor(_engineConfig, arrayOfElements);
}

void ExecutionPlanDescriptor::getTensorUids(hipdnnBackendAttributeType_t attributeType,
                                            int64_t requestedElementCount,
                                            int64_t* elementCount,
                                            void* arrayOfElements) const
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_INT64,
                HIPDNN_STATUS_BAD_PARAM,
                "ExecutionPlanDescriptor failed to get tensor UIDs: Invalid attribute type.");

    auto count = static_cast<int64_t>(_tensorUids.size());
    if(arrayOfElements == nullptr || requestedElementCount == 0)
    {
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "ExecutionPlanDescriptor failed to get tensor UIDs: elementCount is null.");
        *elementCount = count;
        return;
    }

    THROW_IF_LT(requestedElementCount,
                count,
                HIPDNN_STATUS_BAD_PARAM,
                "ExecutionPlanDescriptor failed to get tensor UIDs: requested element count is "
                "too small.");

    if(elementCount != nullptr)
    {
        *elementCount = count;
    }

    std::memcpy(arrayOfElements, _tensorUids.data(), _tensorUids.size() * sizeof(int64_t));
}

std::shared_ptr<const EngineConfigDescriptor> ExecutionPlanDescriptor::getEngineConfig() const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_INTERNAL_ERROR,
                   "ExecutionPlanDescriptor::getEngineConfig() failed: Not finalized.");

    return _engineConfig;
}

int64_t ExecutionPlanDescriptor::getEngineId() const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_INTERNAL_ERROR,
                   "ExecutionPlanDescriptor::getEngineId() failed: Not finalized.");
    THROW_IF_EQ(_engineId,
                INVALID_ENGINE_ID,
                HIPDNN_STATUS_INTERNAL_ERROR,
                "ExecutionPlanDescriptor::getEngineId() failed: Engine id is invalid.");

    return _engineId;
}

const std::vector<int64_t>& ExecutionPlanDescriptor::getTensorUids() const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_INTERNAL_ERROR,
                   "ExecutionPlanDescriptor::getTensorUids() failed: Not finalized.");

    return _tensorUids;
}

bool ExecutionPlanDescriptor::isOverrideShapeEnabled() const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_INTERNAL_ERROR,
                   "ExecutionPlanDescriptor::isOverrideShapeEnabled() failed: Not finalized.");

    return _isOverrideShapeEnabled;
}

hipdnnEnginePluginExecutionContext_t ExecutionPlanDescriptor::getExecutionContext() const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_INTERNAL_ERROR,
                   "ExecutionPlanDescriptor::getExecutionContext() failed: Not finalized.");

    return _executionContext->get();
}

void ExecutionPlanDescriptor::serializeBackendPlan(size_t requestedByteSize,
                                                   size_t* planByteSize,
                                                   uint8_t* serializedPlan) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "ExecutionPlanDescriptor::serializeBackendPlan() failed: Not finalized.");
    THROW_IF_NULL(planByteSize,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "ExecutionPlanDescriptor::serializeBackendPlan() failed: planByteSize is null.");
    THROW_IF_NULL(_pluginResourceManager,
                  HIPDNN_STATUS_INTERNAL_ERROR,
                  "ExecutionPlanDescriptor::serializeBackendPlan() failed: resource manager is "
                  "null.");

    std::vector<uint8_t> pluginPayload;
    _pluginResourceManager->serializeExecutionContext(
        _engineId, _executionContext->get(), pluginPayload);

    flatbuffers::FlatBufferBuilder builder;
    auto serializedPluginPayload = builder.CreateVector(pluginPayload);
    auto serializedTensorUids = builder.CreateVector(_tensorUids);
    auto executionPlan = hipdnn_flatbuffers_sdk::data_objects::CreateSerializedExecutionPlan(
        builder,
        PLAN_SERIALIZATION_VERSION,
        _engineId,
        _workspaceSize,
        serializedTensorUids,
        serializedPluginPayload,
        _isOverrideShapeEnabled);
    builder.Finish(executionPlan);

    *planByteSize = builder.GetSize();
    if(serializedPlan != nullptr)
    {
        THROW_IF_LT(requestedByteSize,
                    builder.GetSize(),
                    HIPDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT,
                    "Requested buffer size is smaller than the serialized execution plan size.");
        std::memcpy(serializedPlan, builder.GetBufferPointer(), builder.GetSize());
    }
}

void ExecutionPlanDescriptor::deserializeBackendPlan(
    const std::shared_ptr<plugin::EnginePluginResourceManager>& pluginResourceManager,
    const uint8_t* serializedPlan,
    size_t planByteSize)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "ExecutionPlanDescriptor::deserializeBackendPlan() failed: Already finalized.");
    THROW_IF_NULL(pluginResourceManager,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "ExecutionPlanDescriptor::deserializeBackendPlan() failed: resource manager is "
                  "null.");
    THROW_IF_NULL(serializedPlan,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "ExecutionPlanDescriptor::deserializeBackendPlan() failed: serialized plan is "
                  "null.");
    THROW_IF_TRUE(planByteSize == 0,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ExecutionPlanDescriptor::deserializeBackendPlan() failed: plan size is zero.");

    flatbuffers::Verifier verifier(serializedPlan, planByteSize);
    THROW_IF_FALSE(
        hipdnn_flatbuffers_sdk::data_objects::VerifySerializedExecutionPlanBuffer(verifier),
        HIPDNN_STATUS_BAD_PARAM,
        "Serialized execution plan FlatBuffer is invalid.");

    auto executionPlan
        = hipdnn_flatbuffers_sdk::data_objects::GetSerializedExecutionPlan(serializedPlan);
    THROW_IF_NE(executionPlan->version(),
                PLAN_SERIALIZATION_VERSION,
                HIPDNN_STATUS_NOT_SUPPORTED,
                "Serialized execution plan version is not supported.");

    auto serializedPluginPayload = executionPlan->plugin_payload();
    auto serializedTensorUids = executionPlan->tensor_uids();

    THROW_IF_TRUE(serializedPluginPayload == nullptr || serializedPluginPayload->empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "Serialized execution plan contains an empty plugin payload.");
    THROW_IF_TRUE(serializedTensorUids == nullptr || serializedTensorUids->empty(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "Serialized execution plan contains no tensor UIDs.");
    _engineId = executionPlan->engine_id();
    _workspaceSize = executionPlan->workspace_size();
    _isOverrideShapeEnabled = executionPlan->is_override_shape_enabled();
    THROW_IF_LT(_workspaceSize,
                0,
                HIPDNN_STATUS_BAD_PARAM,
                "Serialized execution plan contains an invalid workspace size.");

    _tensorUids.assign(serializedTensorUids->begin(), serializedTensorUids->end());
    std::vector<uint8_t> pluginPayload(serializedPluginPayload->begin(),
                                       serializedPluginPayload->end());

    const hipdnnPluginConstData_t pluginPayloadData{pluginPayload.data(), pluginPayload.size()};

    _pluginResourceManager = pluginResourceManager;
    _executionContext = plugin::EnginePluginResourceManager::createExecutionContextFromSerialized(
        _pluginResourceManager, _engineId, &pluginPayloadData);

    HipdnnBackendDescriptorImpl<ExecutionPlanDescriptor>::finalize();
}

hipdnnBackendDescriptorType_t ExecutionPlanDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR;
}

std::string ExecutionPlanDescriptor::toString() const
{
    std::string str = "ExecutionPlanDescriptor: {workspaceSize=" + std::to_string(_workspaceSize);
    str += ", engineId=" + std::to_string(_engineId);
    str += _engineConfig ? ", engineConfig="
                               + fmt::format("{:p}", static_cast<const void*>(_engineConfig.get()))
                         : ", engineConfig=null";
    str += "}";
    return str;
}

} // namespace hipdnn_backend
