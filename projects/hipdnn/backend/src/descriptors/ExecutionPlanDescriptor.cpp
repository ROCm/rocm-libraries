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

namespace hipdnn_backend
{

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

std::shared_ptr<const EngineConfigDescriptor> ExecutionPlanDescriptor::getEngineConfig() const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_INTERNAL_ERROR,
                   "ExecutionPlanDescriptor::getEngineConfig() failed: Not finalized.");

    return _engineConfig;
}

hipdnnEnginePluginExecutionContext_t ExecutionPlanDescriptor::getExecutionContext() const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_INTERNAL_ERROR,
                   "ExecutionPlanDescriptor::getExecutionContext() failed: Not finalized.");

    return _executionContext->get();
}

hipdnnBackendDescriptorType_t ExecutionPlanDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR;
}

} // namespace hipdnn_backend
