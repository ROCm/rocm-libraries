// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "EngineConfigDescriptor.hpp"
#include "BackendEnumStringUtils.hpp"
#include "EngineDescriptor.hpp"
#include "GraphDescriptor.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "handle/Handle.hpp"

#include <hipdnn_sdk/data_objects/engine_config_generated.h>

namespace hipdnn_backend
{

EngineConfigDescriptor::EngineConfigDescriptor()
{
    _engineConfigData = std::make_unique<hipdnn_sdk::data_objects::EngineConfigT>();
}

void EngineConfigDescriptor::finalize()
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "EngineConfigDescriptor::finalize() failed: Already finalized.");

    THROW_IF_NULL(_engine,
                  HIPDNN_STATUS_BAD_PARAM,
                  "EngineConfigDescriptor::finalize() failed: Engine is not set.");

    auto graph = _engine->getGraph();
    auto handle = graph->getHandle();
    auto pluginResourceManager = handle->getPluginResourceManager();

    auto engineId = _engine->getEngineId();

    auto engineConfigPluginData = getSerializedEngineConfig();
    auto workspaceSize = static_cast<int64_t>(
        pluginResourceManager->getWorkspaceSize(engineId, &engineConfigPluginData, graph.get()));

    THROW_IF_LT(workspaceSize,
                0,
                HIPDNN_STATUS_INTERNAL_ERROR,
                "EngineConfigDescriptor::setMaxWorkspaceSize() failed: "
                "Max workspace size cannot be negative.");

    _maxWorkspaceSize = workspaceSize;
    HipdnnBackendDescriptorImpl<EngineConfigDescriptor>::finalize();
}

void EngineConfigDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                          hipdnnBackendAttributeType_t attributeType,
                                          int64_t requestedElementCount,
                                          int64_t* elementCount,
                                          void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "EngineConfigDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_ENGINECFG_ENGINE:
        getEngine(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE:
        getMaxWorkspaceSize(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO:
    case HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES:
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("EngineConfigDescriptor::getAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }
}

void EngineConfigDescriptor::getEngine(hipdnnBackendAttributeType_t attributeType,
                                       int64_t requestedElementCount,
                                       int64_t* elementCount,
                                       void* arrayOfElements) const
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineConfigDescriptor failed to get engine: "
                "Invalid attribute type.");

    THROW_IF_NE(requestedElementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineConfigDescriptor failed to get engine: "
                "Invalid element count.");

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "EngineConfigDescriptor failed to get engine: "
                  "Null pointer.");

    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }

    HipdnnBackendDescriptor::packDescriptor(_engine, arrayOfElements);
}

void EngineConfigDescriptor::getMaxWorkspaceSize(hipdnnBackendAttributeType_t attributeType,
                                                 int64_t requestedElementCount,
                                                 int64_t* elementCount,
                                                 void* arrayOfElements) const
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_INT64,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineConfigDescriptor failed to get max workspace size: Invalid attribute type.");

    THROW_IF_NE(requestedElementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineConfigDescriptor failed to get max workspace size: Invalid element count.");

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "EngineConfigDescriptor failed to get max workspace size: Null pointer.");

    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }

    *static_cast<int64_t*>(arrayOfElements) = _maxWorkspaceSize;
}

void EngineConfigDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                          hipdnnBackendAttributeType_t attributeType,
                                          int64_t elementCount,
                                          const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "EngineConfigDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_ENGINECFG_ENGINE:
        setEngine(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO:
    case HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES:
    case HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE:
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("EngineConfigDescriptor::setAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }

    // reset the serialized buffer when an attribute is set to ensure it's not cached out of date.
    _engineConfigSerializedBuffer = flatbuffers::DetachedBuffer();
}

void EngineConfigDescriptor::setEngine(hipdnnBackendAttributeType_t attributeType,
                                       int64_t elementCount,
                                       const void* arrayOfElements)
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineConfigDescriptor failed to set engine: "
                "Invalid attribute type.");

    THROW_IF_NE(elementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineConfigDescriptor failed to set engine: "
                "Invalid element count.");

    auto engine = HipdnnBackendDescriptor::unpackDescriptor<const EngineDescriptor>(
        arrayOfElements,
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
        "EngineConfigDescriptor failed to set engine: Engine is null.");

    THROW_IF_FALSE(engine->isFinalized(),
                   HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                   "EngineConfigDescriptor failed to set engine: "
                   "Engine is not finalized.");

    _engine = engine;
    _engineConfigData->engine_id = _engine->getEngineId();
}

std::shared_ptr<const EngineDescriptor> EngineConfigDescriptor::getEngine() const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_INTERNAL_ERROR,
                   "EngineConfigDescriptor::getEngine() failed: Not finalized.");
    return _engine;
}

hipdnnBackendDescriptorType_t EngineConfigDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR;
}

hipdnnPluginConstData_t EngineConfigDescriptor::getSerializedEngineConfig() const
{
    if(_engineConfigSerializedBuffer.size() == 0)
    {
        THROW_IF_NULL(_engine,
                      HIPDNN_STATUS_INTERNAL_ERROR,
                      "EngineConfigDescriptor::getSerializedEngineConfig: engine is null");

        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(
            hipdnn_sdk::data_objects::EngineConfig::Pack(builder, _engineConfigData.get()));
        _engineConfigSerializedBuffer = builder.Release();
    }

    return {_engineConfigSerializedBuffer.data(), _engineConfigSerializedBuffer.size()};
}

} // namespace hipdnn_backend
