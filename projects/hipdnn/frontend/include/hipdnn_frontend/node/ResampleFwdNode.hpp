// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/ResampleFwdAttributes.hpp>
#include <hipdnn_frontend/detail/ResampleFwdPacker.hpp>
#include <hipdnn_frontend/detail/ResampleFwdUnpacker.hpp>

namespace hipdnn_frontend::graph
{
class ResampleFwdNode : public BaseNode<ResampleFwdNode, NodeType::RESAMPLE_FWD>
{
public:
    ResampleFwdAttributes attributes;

    ResampleFwdNode(ResampleFwdAttributes&& resampleFwdAttributes,
                    const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(resampleFwdAttributes))
    {
    }

    Error pre_validate_node() const override
    {
        if(!attributes.get_x())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "ResampleFwdNode missing input X for pre-validation"};
        }
        if(!attributes.get_y())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "ResampleFwdNode missing output Y for pre-validation"};
        }

        // Validate tensor ranks if dimensions are already set
        auto x = attributes.get_x();
        if(!x->get_dim().empty() && x->get_dim().size() < 3)
        {
            return {ErrorCode::INVALID_VALUE,
                    "ResampleFwdNode input X must have at least rank 3 (N, C, spatial...)"};
        }
        auto y = attributes.get_y();
        if(!y->get_dim().empty() && y->get_dim().size() < 3)
        {
            return {ErrorCode::INVALID_VALUE,
                    "ResampleFwdNode output Y must have at least rank 3 (N, C, spatial...)"};
        }

        return {};
    }

    Error infer_properties_node() override
    {
        auto x = attributes.get_x();
        auto y = attributes.get_y();

        HIPDNN_RETURN_IF_FALSE(
            x, ErrorCode::ATTRIBUTE_NOT_SET, "ResampleFwdNode missing x for setting properties");
        HIPDNN_RETURN_IF_FALSE(
            y, ErrorCode::ATTRIBUTE_NOT_SET, "ResampleFwdNode missing y for setting properties");

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        auto yDims = y->get_dim();

        // Infer output dimensions if not set
        if(yDims.empty())
        {
            auto& xDims = x->get_dim();
            yDims.resize(xDims.size());

            auto& prePadding = attributes.get_pre_padding();
            auto& postPadding = attributes.get_post_padding();
            auto& strideVec = attributes.get_stride();
            auto& windowVec = attributes.get_window();

            yDims[0] = xDims[0]; // N (batch)
            yDims[1] = xDims[1]; // C (channels preserved)

            // Calculate spatial dimensions
            for(size_t i = 2; i < xDims.size(); ++i)
            {
                auto spatialIdx = i - 2;
                auto inputSize = xDims[i];
                auto prePad = prePadding[spatialIdx];
                auto postPad = postPadding[spatialIdx];
                auto strideVal = strideVec[spatialIdx];
                auto windowVal = windowVec[spatialIdx];

                yDims[i] = (inputSize + prePad + postPad - windowVal) / strideVal + 1;
            }

            y->set_dim(yDims);
        }

        // Infer output strides if not set
        if(y->get_stride().empty())
        {
            std::vector<int64_t> yStrides(yDims.size());
            yStrides.back() = 1;
            for(int64_t i = static_cast<int64_t>(yDims.size()) - 2; i >= 0; --i)
            {
                yStrides[static_cast<size_t>(i)]
                    = yStrides[static_cast<size_t>(i + 1)] * yDims[static_cast<size_t>(i + 1)];
            }
            y->set_stride(yStrides);
        }

        return {};
    }

    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createResampleFwdOperation(attributes, tensorDescs, operations);
    }

    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        ResampleFwdAttributes attrs;
        HIPDNN_CHECK_ERROR(detail::unpackResampleFwdOperation(opDesc, tensorMap, attrs));
        attributes = std::move(attrs);
        return {};
    }
};
} // namespace hipdnn_frontend::graph
