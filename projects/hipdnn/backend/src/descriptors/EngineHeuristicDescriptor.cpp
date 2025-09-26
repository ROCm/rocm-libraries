// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "EngineHeuristicDescriptor.hpp"
#include "BackendEnumStringUtils.hpp"
#include "EngineConfigDescriptor.hpp"
#include "EngineDescriptor.hpp"
#include "GraphDescriptor.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "ScopedDescriptor.hpp"
#include "handle/Handle.hpp"

namespace hipdnn_backend
{

void EngineHeuristicDescriptor::finalize()
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "EngineHeuristicDescriptor::finalize() failed: Already finalized.");

    THROW_IF_NULL(_graph,
                  HIPDNN_STATUS_BAD_PARAM,
                  "EngineHeuristicDescriptor::finalize() failed: Graph is not set.");

    THROW_IF_FALSE(_heuristicModeSet,
                   HIPDNN_STATUS_BAD_PARAM,
                   "EngineHeuristicDescriptor::finalize() failed: Heuristic mode is not set.");

    auto handle = _graph->getHandle();
    auto pluginResourceManager = handle->getPluginResourceManager();

    // TODO - For now we are going to return the engine IDs we get from the plugin resource manager.
    // In the future, we will need to implement a plugin system for engine heuristics that allows plugins to determine sort order of the returned engines.
    _engineIds = pluginResourceManager->getApplicableEngineIds(_graph.get());

    HipdnnBackendDescriptorImpl<EngineHeuristicDescriptor>::finalize();
}

void EngineHeuristicDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                             hipdnnBackendAttributeType_t attributeType,
                                             int64_t requestedElementCount,
                                             int64_t* elementCount,
                                             void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                   "EngineHeuristicDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH:
        getGraph(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINEHEUR_MODE:
        getHeuristicMode(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINEHEUR_RESULTS:
        getEngineConfigs(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("EngineHeuristicDescriptor::getAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }
}

void EngineHeuristicDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                             hipdnnBackendAttributeType_t attributeType,
                                             int64_t elementCount,
                                             const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "EngineHeuristicDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_ENGINEHEUR_MODE:
        setHeuristicMode(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH:
        setGraph(attributeType, elementCount, arrayOfElements);
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("EngineHeuristicDescriptor::setAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }
}

void EngineHeuristicDescriptor::setHeuristicMode(hipdnnBackendAttributeType_t attributeType,
                                                 int64_t elementCount,
                                                 const void* arrayOfElements)
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_HEUR_MODE,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineHeuristicDescriptor failed to set heuristic mode: Invalid attribute type.");

    THROW_IF_NE(elementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineHeuristicDescriptor failed to set heuristic mode: Invalid element count.");

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "EngineHeuristicDescriptor failed to set heuristic mode: Null pointer.");

    auto heurMode = static_cast<const hipdnnBackendHeurMode_t*>(arrayOfElements);
    THROW_IF_NULL(
        heurMode,
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
        "EngineHeuristicDescriptor failed to set heuristic mode: Heuristic mode is null.");

    auto heurModeValue = *heurMode;
    if(heurModeValue != HIPDNN_HEUR_MODE_FALLBACK)
    {
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "EngineHeuristicDescriptor::setHeuristicMode() is not "
                              "supported for the given heuristic mode.");
    }

    _heuristicMode = heurModeValue;
    _heuristicModeSet = true;
}

void EngineHeuristicDescriptor::setGraph(hipdnnBackendAttributeType_t attributeType,
                                         int64_t elementCount,
                                         const void* arrayOfElements)
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineHeuristicDescriptor failed to set graph: Invalid attribute type.");

    THROW_IF_NE(elementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineHeuristicDescriptor failed to set graph: Invalid element count.");

    auto graph = HipdnnBackendDescriptor::unpackDescriptor<const GraphDescriptor>(
        arrayOfElements,
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
        "EngineHeuristicDescriptor failed to set graph: Null pointer.");

    THROW_IF_FALSE(graph->isFinalized(),
                   HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                   "EngineHeuristicDescriptor failed to set graph: Graph is not finalized.");

    _graph = graph;
}

void EngineHeuristicDescriptor::getGraph(hipdnnBackendAttributeType_t attributeType,
                                         int64_t requestedElementCount,
                                         int64_t* elementCount,
                                         void* arrayOfElements) const
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineHeuristicDescriptor failed to get graph: Invalid attribute type.");

    THROW_IF_NE(requestedElementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineHeuristicDescriptor failed to get graph: Invalid element count.");

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "EngineHeuristicDescriptor failed to get graph: Null pointer.");

    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }

    HipdnnBackendDescriptor::packDescriptor(_graph, arrayOfElements);
}

void EngineHeuristicDescriptor::getEngineConfigs(hipdnnBackendAttributeType_t attributeType,
                                                 int64_t requestedElementCount,
                                                 int64_t* elementCount,
                                                 void* arrayOfElements) const
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineHeuristicDescriptor failed to get engine configs: Invalid attribute type.");

    // Return the number of engine configs if they aren't requesting any.
    if(requestedElementCount == 0)
    {
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "EngineHeuristicDescriptor failed to get engine config count: Null pointer "
                      "for element count.");
        *elementCount = static_cast<int64_t>(_engineIds.size());
    }
    else
    {
        THROW_IF_NULL(arrayOfElements,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "EngineHeuristicDescriptor failed to get engine configs: Null pointer.");

        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "EngineHeuristicDescriptor failed to get engine config: Null pointer for "
                      "element count.");

        // Create engine config descriptors for each engine ID
        auto outputArray = static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements);
        for(size_t i = 0; i < _engineIds.size() && i < static_cast<size_t>(requestedElementCount);
            ++i)
        {
            auto config = HipdnnBackendDescriptor::unpackDescriptor<EngineConfigDescriptor>(
                outputArray[i],
                HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                "EngineHeuristicDescriptor failed to get engine config: Config "
                "descriptor is null.");

            auto engine = std::make_shared<EngineDescriptor>();

            engine->setAttribute(
                HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, &_engineIds[i]);

            ScopedDescriptor graphDesc(HipdnnBackendDescriptor::packDescriptor(_graph));
            engine->setAttribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                 HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                 1,
                                 graphDesc.getPtr());
            engine->finalize();

            ScopedDescriptor engineDesc(HipdnnBackendDescriptor::packDescriptor(engine));
            config->setAttribute(HIPDNN_ATTR_ENGINECFG_ENGINE,
                                 HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                 1,
                                 engineDesc.getPtr());
        }

        *elementCount = std::min(requestedElementCount, static_cast<int64_t>(_engineIds.size()));
    }
}

void EngineHeuristicDescriptor::getHeuristicMode(hipdnnBackendAttributeType_t attributeType,
                                                 int64_t requestedElementCount,
                                                 int64_t* elementCount,
                                                 void* arrayOfElements) const
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_HEUR_MODE,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineHeuristicDescriptor failed to get heuristic mode: Invalid attribute type.");

    THROW_IF_NE(requestedElementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineHeuristicDescriptor failed to get heuristic mode: Invalid element count.");

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "EngineHeuristicDescriptor failed to get heuristic mode: Null pointer.");

    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }

    auto heurModeOut = static_cast<hipdnnBackendHeurMode_t*>(arrayOfElements);
    *heurModeOut = _heuristicMode;
}

std::shared_ptr<const GraphDescriptor> EngineHeuristicDescriptor::getGraph() const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_INTERNAL_ERROR,
                   "EngineHeuristicDescriptor::getGraph() failed: Not finalized.");

    return _graph;
}

hipdnnBackendDescriptorType_t EngineHeuristicDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR;
}

} // namespace hipdnn_backend
