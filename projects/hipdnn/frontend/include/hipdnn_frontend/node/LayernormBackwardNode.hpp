// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/LayernormBackwardAttributes.hpp>
#include <hipdnn_frontend/detail/LayernormBackwardPacker.hpp>
#include <hipdnn_frontend/detail/LayernormBackwardUnpacker.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>

namespace hipdnn_frontend::graph
{
class LayernormBackwardNode : public BaseNode<LayernormBackwardNode, NodeType::LAYERNORM_BACKWARD>
{
public:
    LayernormBackwardAttributes attributes;

    LayernormBackwardNode(LayernormBackwardAttributes&& attrs, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(attrs))
    {
    }

    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        LayernormBackwardAttributes attrs;
        HIPDNN_CHECK_ERROR(detail::unpackLayernormBackwardOperation(opDesc, tensorMap, attrs));
        attributes = std::move(attrs);
        return {};
    }

    Error pre_validate_node() const override
    {
        // Validate required tensor pointers
        HIPDNN_RETURN_IF_FALSE(attributes.get_dy(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "LayernormBackwardNode missing dy (input) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_x(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "LayernormBackwardNode missing x (input) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_scale(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "LayernormBackwardNode missing scale (input) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_dx(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "LayernormBackwardNode missing dx (output) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_dscale(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "LayernormBackwardNode missing dscale (output) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_dbias(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "LayernormBackwardNode missing dbias (output) for pre-validation");

        return {};
    }

    Error infer_properties_node() override
    {
        // Validate required tensor pointers
        HIPDNN_RETURN_IF_FALSE(attributes.get_dy(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "LayernormBackwardNode missing dy for setting properties");

        HIPDNN_RETURN_IF_FALSE(attributes.get_x(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "LayernormBackwardNode missing x for setting properties");

        HIPDNN_RETURN_IF_FALSE(attributes.get_scale(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "LayernormBackwardNode missing scale for setting properties");

        HIPDNN_RETURN_IF_FALSE(attributes.get_dx(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "LayernormBackwardNode missing dx for setting properties");

        HIPDNN_RETURN_IF_FALSE(attributes.get_dscale(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "LayernormBackwardNode missing dscale for setting properties");

        HIPDNN_RETURN_IF_FALSE(attributes.get_dbias(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "LayernormBackwardNode missing dbias for setting properties");

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        if(attributes.get_dx()->get_dim().empty())
        {
            attributes.get_dx()->set_dim(attributes.get_x()->get_dim());
        }
        if(attributes.get_dx()->get_stride().empty())
        {
            attributes.get_dx()->set_stride(attributes.get_x()->get_stride());
        }

        if(attributes.get_dscale()->get_dim().empty())
        {
            attributes.get_dscale()->set_dim(attributes.get_scale()->get_dim());
        }
        if(attributes.get_dscale()->get_stride().empty())
        {
            attributes.get_dscale()->set_stride(attributes.get_scale()->get_stride());
        }

        if(attributes.get_dbias()->get_dim().empty())
        {
            attributes.get_dbias()->set_dim(attributes.get_scale()->get_dim());
        }
        if(attributes.get_dbias()->get_stride().empty())
        {
            attributes.get_dbias()->set_stride(attributes.get_scale()->get_stride());
        }

        auto dxTensor = attributes.get_dx();
        auto dscaleTensor = attributes.get_dscale();
        auto dbiasTensor = attributes.get_dbias();
        auto dyTensor = attributes.get_dy();
        auto scaleTensor = attributes.get_scale();

        // Infer output strides if not set
        if(dxTensor->get_stride().empty())
        {
            auto& dyStrides = dyTensor->get_stride();
            auto& dxDims = dxTensor->get_dim();

            HIPDNN_RETURN_IF_TRUE(
                dyStrides.empty(),
                ErrorCode::ATTRIBUTE_NOT_SET,
                "LayernormBackwardNode: Cannot infer output strides - missing input strides");

            HIPDNN_RETURN_IF_TRUE(
                dxDims.empty(),
                ErrorCode::ATTRIBUTE_NOT_SET,
                "LayernormBackwardNode: Cannot infer output strides - missing output dimensions");

            HIPDNN_RETURN_IF_NE(dyStrides.size(),
                                dxDims.size(),
                                ErrorCode::ATTRIBUTE_NOT_SET,
                                "LayernormBackwardNode: Stride dimension mismatch between input "
                                "and output tensors");

            auto strideOrder = hipdnn_data_sdk::utilities::extractStrideOrder(dyStrides);
            auto dxStrides = hipdnn_data_sdk::utilities::generateStrides(dxDims, strideOrder);
            dxTensor->set_stride(dxStrides);
        }

        if(dscaleTensor->get_stride().empty())
        {
            auto& scaleStrides = scaleTensor->get_stride();
            auto& dscaleDims = dscaleTensor->get_dim();

            HIPDNN_RETURN_IF_TRUE(scaleStrides.empty(),
                                  ErrorCode::ATTRIBUTE_NOT_SET,
                                  "LayernormBackwardNode: Cannot infer gradient scale strides - "
                                  "missing scale strides");

            HIPDNN_RETURN_IF_TRUE(dscaleDims.empty(),
                                  ErrorCode::ATTRIBUTE_NOT_SET,
                                  "LayernormBackwardNode: Cannot infer gradient scale strides - "
                                  "missing gradient scale dimensions");

            HIPDNN_RETURN_IF_NE(scaleStrides.size(),
                                dscaleDims.size(),
                                ErrorCode::ATTRIBUTE_NOT_SET,
                                "LayernormBackwardNode: Stride dimension mismatch between scale "
                                "and gradient scale tensors");

            auto strideOrder = hipdnn_data_sdk::utilities::extractStrideOrder(scaleStrides);
            auto dscaleStrides
                = hipdnn_data_sdk::utilities::generateStrides(dscaleDims, strideOrder);
            dscaleTensor->set_stride(dscaleStrides);
        }

        if(dbiasTensor->get_stride().empty())
        {
            auto& scaleStrides = scaleTensor->get_stride();
            auto& dbiasDims = dbiasTensor->get_dim();

            HIPDNN_RETURN_IF_TRUE(scaleStrides.empty(),
                                  ErrorCode::ATTRIBUTE_NOT_SET,
                                  "LayernormBackwardNode: Cannot infer gradient bias strides - "
                                  "missing scale strides");

            HIPDNN_RETURN_IF_TRUE(dbiasDims.empty(),
                                  ErrorCode::ATTRIBUTE_NOT_SET,
                                  "LayernormBackwardNode: Cannot infer gradient bias strides - "
                                  "missing gradient bias dimensions");

            HIPDNN_RETURN_IF_NE(scaleStrides.size(),
                                dbiasDims.size(),
                                ErrorCode::ATTRIBUTE_NOT_SET,
                                "LayernormBackwardNode: Stride dimension mismatch between scale "
                                "and gradient bias tensors");

            auto strideOrder = hipdnn_data_sdk::utilities::extractStrideOrder(scaleStrides);
            auto dbiasStrides = hipdnn_data_sdk::utilities::generateStrides(dbiasDims, strideOrder);
            dbiasTensor->set_stride(dbiasStrides);
        }

        // Infer normalized dimension count if not available
        if(attributes.get_normalized_dim_count() <= 0)
        {
            if(attributes.get_x()->get_dim().size()
               == attributes.get_scale()
                      ->get_dim()
                      .size()) // Dimensions not used by scale have been set to 1
            {
                int64_t normalizedDimCount = 1;
                for(int64_t i = static_cast<int64_t>(attributes.get_scale()->get_dim().size()) - 1;
                    i >= 0;
                    --i)
                {
                    if(attributes.get_scale()->get_dim()[static_cast<size_t>(i)] == 1)
                    {
                        break;
                    }
                    normalizedDimCount
                        = static_cast<int64_t>(attributes.get_scale()->get_dim().size()) - i;
                }
                attributes.set_normalized_dim_count(normalizedDimCount);
            }
            else // Dimensions not used by scale have been omitted
            {
                attributes.set_normalized_dim_count(
                    static_cast<int64_t>(attributes.get_scale()->get_dim().size()));
            }
        }

        return {};
    }

    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createLayernormBackwardOperation(attributes, tensorDescs, operations);
    }
};
}
