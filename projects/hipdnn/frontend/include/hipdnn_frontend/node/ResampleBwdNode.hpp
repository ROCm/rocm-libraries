// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/ResampleBwdAttributes.hpp>
#include <hipdnn_frontend/detail/ResampleBwdPacker.hpp>
#include <hipdnn_frontend/detail/ResampleBwdUnpacker.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>

namespace hipdnn_frontend::graph
{
class ResampleBwdNode : public BaseNode<ResampleBwdNode, NodeType::RESAMPLE_BWD>
{
public:
    ResampleBwdAttributes attributes;

    ResampleBwdNode(ResampleBwdAttributes&& attrs, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(attrs))
    {
    }

    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        ResampleBwdAttributes attrs;
        HIPDNN_CHECK_ERROR(detail::unpackResampleBwdOperation(opDesc, tensorMap, attrs));
        attributes = std::move(attrs);
        return {};
    }

    Error pre_validate_node() const override
    {
        // Validate required input tensors
        HIPDNN_RETURN_IF_FALSE(attributes.get_dy(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "ResampleBwdNode missing dy (input) for pre-validation");

        // Validate required output tensors
        HIPDNN_RETURN_IF_FALSE(attributes.get_dx(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "ResampleBwdNode missing dx (output) for pre-validation");

        // Validate required tensor dimensions
        HIPDNN_RETURN_IF_TRUE(attributes.get_dy()->get_dim().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "ResampleBwdNode missing dy dimensions for pre-validation");

        // Validate required resample parameters
        HIPDNN_RETURN_IF_TRUE(attributes.get_pre_padding().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "ResampleBwdNode missing pre_padding for pre-validation");
        HIPDNN_RETURN_IF_TRUE(attributes.get_post_padding().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "ResampleBwdNode missing post_padding for pre-validation");
        HIPDNN_RETURN_IF_TRUE(attributes.get_stride().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "ResampleBwdNode missing stride for pre-validation");
        HIPDNN_RETURN_IF_TRUE(attributes.get_window().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "ResampleBwdNode missing window for pre-validation");

        return {};
    }

    Error infer_properties_node() override
    {
        // Validate required tensor pointers
        HIPDNN_RETURN_IF_FALSE(attributes.get_dy(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "ResampleBwdNode missing dy for setting properties");

        HIPDNN_RETURN_IF_FALSE(attributes.get_dx(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "ResampleBwdNode missing dx for setting properties");

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        auto dxTensor = attributes.get_dx();
        auto dyTensor = attributes.get_dy();

        // Infer output dimensions if not set
        auto dxDims = dxTensor->get_dim();
        if(dxDims.empty())
        {
            auto& dyDims = dyTensor->get_dim();
            dxDims.resize(dyDims.size());

            auto& prePadding = attributes.get_pre_padding();
            auto& postPadding = attributes.get_post_padding();
            auto& strideVec = attributes.get_stride();
            auto& windowVec = attributes.get_window();

            dxDims[0] = dyDims[0]; // N (batch)
            dxDims[1] = dyDims[1]; // C (channels preserved)

            // Calculate spatial dimensions (inverse of forward)
            // forward: y = (x + pre + post - window) / stride + 1
            // inverse: x = (y - 1) * stride - pre - post + window
            for(size_t i = 2; i < dyDims.size(); ++i)
            {
                const size_t spatialIdx = i - 2;
                dxDims[i] = (dyDims[i] - 1) * strideVec[spatialIdx] - prePadding[spatialIdx]
                            - postPadding[spatialIdx] + windowVec[spatialIdx];
            }

            dxTensor->set_dim(dxDims);
        }

        // Infer output strides if not set
        if(dxTensor->get_stride().empty())
        {
            auto& dyStrides = dyTensor->get_stride();
            auto& currentDxDims = dxTensor->get_dim();

            HIPDNN_RETURN_IF_TRUE(
                dyStrides.empty(),
                ErrorCode::ATTRIBUTE_NOT_SET,
                "ResampleBwdNode: Cannot infer output strides - missing input strides");

            HIPDNN_RETURN_IF_TRUE(
                currentDxDims.empty(),
                ErrorCode::ATTRIBUTE_NOT_SET,
                "ResampleBwdNode: Cannot infer output strides - missing output dimensions");

            HIPDNN_RETURN_IF_NE(
                dyStrides.size(),
                currentDxDims.size(),
                ErrorCode::ATTRIBUTE_NOT_SET,
                "ResampleBwdNode: Stride dimension mismatch between input and output tensors");

            auto strideOrder = hipdnn_data_sdk::utilities::extractStrideOrder(dyStrides);
            auto dxStrides
                = hipdnn_data_sdk::utilities::generateStrides(currentDxDims, strideOrder);
            dxTensor->set_stride(dxStrides);
        }

        return {};
    }

    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createResampleBwdOperation(attributes, tensorDescs, operations);
    }
};
}
