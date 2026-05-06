// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "NodeFactory.hpp"
#include "BatchnormOperationDescriptor.hpp"
#include "HipdnnException.hpp"

namespace hipdnn_backend
{

std::shared_ptr<IBackendDescriptor> NodeFactory::createOperationFromNode(
    const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
    const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap)
{
    using NodeAttributes = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;

    switch(nodeT.attributes.type)
    {
    case NodeAttributes::BatchnormAttributes:
        return BatchnormOperationDescriptor::fromNode(nodeT, tensorMap);
    case NodeAttributes::BatchnormBackwardAttributes:
        return BatchnormBackwardOperationDescriptor::fromNode(nodeT, tensorMap);
    case NodeAttributes::BatchnormInferenceAttributes:
        return BatchnormInferenceOperationDescriptor::fromNode(nodeT, tensorMap);
    case NodeAttributes::BatchnormInferenceAttributesVarianceExt:
        return BatchnormInferenceVarianceExtOperationDescriptor::fromNode(nodeT, tensorMap);
    case NodeAttributes::BlockScaleDequantizeAttributes:
        return BlockScaleDequantizeOperationDescriptor::fromNode(nodeT, tensorMap);
    case NodeAttributes::BlockScaleQuantizeAttributes:
        return BlockScaleQuantizeOperationDescriptor::fromNode(nodeT, tensorMap);
    case NodeAttributes::ConvolutionBwdAttributes:
        return ConvolutionBwdOperationDescriptor::fromNode(nodeT, tensorMap);
    case NodeAttributes::ConvolutionFwdAttributes:
        return ConvolutionFwdOperationDescriptor::fromNode(nodeT, tensorMap);
    case NodeAttributes::ConvolutionWrwAttributes:
        return ConvolutionWrwOperationDescriptor::fromNode(nodeT, tensorMap);
    case NodeAttributes::CustomOpAttributes:
        return CustomOpOperationDescriptor::fromNode(nodeT, tensorMap);
    case NodeAttributes::LayernormAttributes:
        return LayernormOperationDescriptor::fromNode(nodeT, tensorMap);
    case NodeAttributes::MatmulAttributes:
        return MatmulOperationDescriptor::fromNode(nodeT, tensorMap);
    case NodeAttributes::PointwiseAttributes:
        return PointwiseOperationDescriptor::fromNode(nodeT, tensorMap);
    case NodeAttributes::ReductionAttributes:
        return ReductionOperationDescriptor::fromNode(nodeT, tensorMap);
    case NodeAttributes::RMSNormAttributes:
        return RMSNormOperationDescriptor::fromNode(nodeT, tensorMap);
    case NodeAttributes::SdpaAttributes:
        return SdpaFwdOperationDescriptor::fromNode(nodeT, tensorMap);
    case NodeAttributes::SdpaBackwardAttributes:
        return SdpaBwdOperationDescriptor::fromNode(nodeT, tensorMap);
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            "NodeFactory::createOperationFromNode: unsupported node type "
                + std::string(hipdnn_flatbuffers_sdk::data_objects::EnumNameNodeAttributes(
                    nodeT.attributes.type)));
    }
}

std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> NodeFactory::buildTensorMap(
    const std::vector<std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT>>&
        tensors)
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
