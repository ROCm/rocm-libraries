// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorFactory.hpp"
#include "BackendEnumStringUtils.hpp"
#include "EngineConfigDescriptor.hpp"
#include "EngineDescriptor.hpp"
#include "EngineHeuristicDescriptor.hpp"
#include "ExecutionPlanDescriptor.hpp"
#include "GraphDescriptor.hpp"
#include "HipdnnException.hpp"
#include "VariantDescriptor.hpp"
#include "logging/Logging.hpp"

namespace hipdnn_backend
{

void DescriptorFactory::create(hipdnnBackendDescriptorType_t descriptorType,
                               hipdnnBackendDescriptor_t* descriptor)
{
    THROW_IF_NULL(
        descriptor, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER, "hipdnnBackendDescriptor_t* is null.");

    HIPDNN_LOG_INFO("Creating descriptor of type: {}",
                    hipdnnGetBackendDescriptorTypeName(descriptorType));

    std::shared_ptr<IBackendDescriptor> privateDesc;
    switch(descriptorType)
    {
    case HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR:
        privateDesc = std::make_shared<EngineConfigDescriptor>();
        break;
    case HIPDNN_BACKEND_ENGINE_DESCRIPTOR:
        privateDesc = std::make_shared<EngineDescriptor>();
        break;
    case HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR:
        privateDesc = std::make_shared<ExecutionPlanDescriptor>();
        break;
    case HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR:
        privateDesc = std::make_shared<GraphDescriptor>();
        break;
    case HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR:
        privateDesc = std::make_shared<VariantDescriptor>();
        break;
    case HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR:
        privateDesc = std::make_shared<EngineHeuristicDescriptor>();
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              std::string("Descriptor type ")
                                  + hipdnnGetBackendDescriptorTypeName(descriptorType)
                                  + " is not supported.");
    }

    *descriptor = HipdnnBackendDescriptor::packDescriptor(privateDesc);

    HIPDNN_LOG_INFO("Created descriptor: {:p}", static_cast<void*>(*descriptor));
}

void DescriptorFactory::createGraphExt(hipdnnBackendDescriptor_t* descriptor,
                                       const uint8_t* serializedGraph,
                                       size_t graphByteSize)
{
    THROW_IF_NULL(
        descriptor, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER, "hipdnnBackendDescriptor_t* is null.");
    THROW_IF_NULL(
        serializedGraph, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER, "serializedGraph is null.");
    THROW_IF_TRUE(graphByteSize == 0, HIPDNN_STATUS_BAD_PARAM, "graphByteSize is 0.");

    auto graphDescriptor = std::make_shared<GraphDescriptor>();
    graphDescriptor->deserializeGraph(serializedGraph, graphByteSize);
    *descriptor = HipdnnBackendDescriptor::packDescriptor(graphDescriptor);

    HIPDNN_LOG_INFO("Created graph descriptor: {:p}", static_cast<void*>(*descriptor));
}

void DescriptorFactory::destroy(hipdnnBackendDescriptor_t descriptor)
{
    THROW_IF_NULL(
        descriptor, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER, "hipdnnBackendDescriptor_t is null.");

    delete descriptor;

    HIPDNN_LOG_INFO("Destroyed descriptor: {:p}", static_cast<void*>(descriptor));
}

} // namespace hipdnn_backend
