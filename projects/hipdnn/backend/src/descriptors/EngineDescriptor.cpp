// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "EngineDescriptor.hpp"
#include "BackendEnumStringUtils.hpp"
#include "GraphDescriptor.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "handle/Handle.hpp"

namespace hipdnn_backend
{

void EngineDescriptor::finalize()
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "EngineDescriptor::finalize() failed: Already finalized.");

    THROW_IF_NULL(
        _graph, HIPDNN_STATUS_BAD_PARAM, "EngineDescriptor::finalize() failed: Graph is not set.");

    THROW_IF_FALSE(_engineIdSet,
                   HIPDNN_STATUS_BAD_PARAM,
                   "EngineDescriptor::finalize() failed: Engine id is not set.");

    auto handle = _graph->getHandle();
    auto pluginResourceManager = handle->getPluginResourceManager();

    auto engineIds = pluginResourceManager->getApplicableEngineIds(_graph.get());
    if(std::find(engineIds.begin(), engineIds.end(), _engineId) == engineIds.end())
    {
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM,
                              "EngineDescriptor::finalize() failed: Engine id is not in a valid "
                              "range of engine IDs");
    }

    _engineDetails = plugin::EnginePluginResourceManager::getEngineDetails(
        pluginResourceManager, _engineId, _graph.get());

    HipdnnBackendDescriptorImpl<EngineDescriptor>::finalize();
}

void EngineDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                    hipdnnBackendAttributeType_t attributeType,
                                    int64_t requestedElementCount,
                                    int64_t* elementCount,
                                    void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "EngineDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_ENGINE_OPERATION_GRAPH:
        getGraph(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINE_GLOBAL_INDEX:
        getGlobalId(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINE_KNOB_INFO:
    case HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE:
    case HIPDNN_ATTR_ENGINE_LAYOUT_INFO:
    case HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE:
    case HIPDNN_ATTR_ENGINE_SM_COUNT_TARGET:
    case HIPDNN_ATTR_ENGINE_DEVICEPROP:
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("EngineDescriptor::getAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }
}

void EngineDescriptor::getGraph(hipdnnBackendAttributeType_t attributeType,
                                int64_t requestedElementCount,
                                int64_t* elementCount,
                                void* arrayOfElements) const
{

    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineDescriptor failed to get graph: Invalid attribute type.");

    THROW_IF_NE(requestedElementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineDescriptor failed to get graph: Invalid element count.");

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "EngineDescriptor failed to get graph: Null pointer.");

    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }

    HipdnnBackendDescriptor::packDescriptor(_graph, arrayOfElements);
}

void EngineDescriptor::getGlobalId(hipdnnBackendAttributeType_t attributeType,
                                   int64_t requestedElementCount,
                                   int64_t* elementCount,
                                   void* arrayOfElements) const
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_INT64,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineDescriptor failed to get global engine ID: Invalid attribute type.");

    THROW_IF_NE(requestedElementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineDescriptor failed to get global engine ID: Invalid element count.");

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "EngineDescriptor failed to get global engine ID: Null pointer.");

    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }

    *static_cast<int64_t*>(arrayOfElements) = _engineId;
}

void EngineDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                    hipdnnBackendAttributeType_t attributeType,
                                    int64_t elementCount,
                                    const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "EngineDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_ENGINE_OPERATION_GRAPH:
        setGraph(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINE_GLOBAL_INDEX:
        setGlobalId(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINE_KNOB_INFO:
    case HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE:
    case HIPDNN_ATTR_ENGINE_LAYOUT_INFO:
    case HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE:
    case HIPDNN_ATTR_ENGINE_SM_COUNT_TARGET:
    case HIPDNN_ATTR_ENGINE_DEVICEPROP:
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("EngineDescriptor::setAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }
}

void EngineDescriptor::setGraph(hipdnnBackendAttributeType_t attributeType,
                                int64_t elementCount,
                                const void* arrayOfElements)
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineDescriptor failed to set graph: Invalid attribute type.");

    THROW_IF_NE(elementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineDescriptor failed to set graph: Invalid element count.");

    auto graph = HipdnnBackendDescriptor::unpackDescriptor<const GraphDescriptor>(
        arrayOfElements,
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
        "EngineDescriptor failed to set graph: Graph is null.");

    THROW_IF_FALSE(graph->isFinalized(),
                   HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                   "EngineDescriptor failed to set graph: Graph is not finalized.");

    _graph = graph;
}

void EngineDescriptor::setGlobalId(hipdnnBackendAttributeType_t attributeType,
                                   int64_t elementCount,
                                   const void* arrayOfElements)
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_INT64,
                HIPDNN_STATUS_BAD_PARAM,
                "Engine failed to set engine id: Invalid attribute type.");

    THROW_IF_NE(elementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "Engine failed to set engine id: Invalid element count.");

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "Engine failed to set engine id: Null pointer.");

    _engineId = *static_cast<const int64_t*>(arrayOfElements);
    _engineIdSet = true;
}

std::shared_ptr<const GraphDescriptor> EngineDescriptor::getGraph() const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_INTERNAL_ERROR,
                   "EngineDescriptor::getGraph() failed: Not finalized.");

    return _graph;
}

int64_t EngineDescriptor::getEngineId() const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_INTERNAL_ERROR,
                   "EngineDescriptor::getEngineId() failed: Not finalized.");

    return _engineId;
}

hipdnnBackendDescriptorType_t EngineDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_ENGINE_DESCRIPTOR;
}

} // namespace hipdnn_backend
