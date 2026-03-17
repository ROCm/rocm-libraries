// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "NodeFactory.hpp"
#include "HipdnnException.hpp"

namespace hipdnn_backend
{

std::shared_ptr<IGraphOperation> NodeFactory::createOperationFromNode(
    const hipdnn_data_sdk::data_objects::NodeT& nodeT,
    const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap)
{
    using NodeAttributes = hipdnn_data_sdk::data_objects::NodeAttributes;

    switch(nodeT.attributes.type)
    {
    case NodeAttributes::ConvolutionFwdAttributes:
        return ConvolutionFwdOperationDescriptor::fromNode(nodeT, tensorMap);
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            "NodeFactory::createOperationFromNode: unsupported node type "
                + std::string(
                    hipdnn_data_sdk::data_objects::EnumNameNodeAttributes(nodeT.attributes.type)));
    }
}

std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> NodeFactory::buildTensorMap(
    const std::vector<std::unique_ptr<hipdnn_data_sdk::data_objects::TensorAttributesT>>& tensors)
{
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> tensorMap;
    for(const auto& tensorT : tensors)
    {
        THROW_IF_NULL(
            tensorT, HIPDNN_STATUS_INTERNAL_ERROR, "buildTensorMap: null tensor in graph");

        THROW_IF_TRUE(tensorMap.count(tensorT->uid) > 0,
                      HIPDNN_STATUS_INTERNAL_ERROR,
                      "buildTensorMap: duplicate tensor UID " + std::to_string(tensorT->uid)
                          + " in graph");

        tensorMap[tensorT->uid] = TensorDescriptor::fromFlatBuffer(*tensorT);
    }
    return tensorMap;
}

} // namespace hipdnn_backend
