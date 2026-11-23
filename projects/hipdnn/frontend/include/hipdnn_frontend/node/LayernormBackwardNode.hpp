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
#include <hipdnn_frontend/node/detail/Utilities.hpp>

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

        HIPDNN_RETURN_IF_TRUE(attributes.get_mean() == nullptr
                                  ^ attributes.get_inv_variance() == nullptr,
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "LayernormBackwardNode requires mean (input) and inv_variance "
                              "(input) to both be set or both not be set");

        HIPDNN_RETURN_IF_FALSE(attributes.get_dx(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "LayernormBackwardNode missing dx (output) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_dscale(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "LayernormBackwardNode missing dscale (output) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_dbias(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "LayernormBackwardNode missing dbias (output) for pre-validation");

        // Validate tensor properties
        HIPDNN_RETURN_IF_FALSE(
            attributes.get_dy()->get_dim().empty()
                || attributes.get_dy()->get_dim() == attributes.get_x()->get_dim(),
            ErrorCode::INVALID_VALUE,
            "LayernormBackwardNode requires equal shapes for dy (input) and x (input)");
        HIPDNN_RETURN_IF_FALSE(
            attributes.get_dx()->get_dim().empty()
                || attributes.get_dx()->get_dim() == attributes.get_x()->get_dim(),
            ErrorCode::INVALID_VALUE,
            "LayernormBackwardNode requires equal shapes for dx (output) and x (input)");
        HIPDNN_RETURN_IF_FALSE(
            attributes.get_mean() == nullptr || attributes.get_inv_variance() == nullptr
                || attributes.get_mean()->get_dim() == attributes.get_inv_variance()->get_dim(),
            ErrorCode::INVALID_VALUE,
            "LayernormBackwardNode requires equal shapes for mean (input) and inv_variance "
            "(input)");
        HIPDNN_RETURN_IF_FALSE(
            attributes.get_dscale()->get_dim().empty()
                || attributes.get_dscale()->get_dim() == attributes.get_scale()->get_dim(),
            ErrorCode::INVALID_VALUE,
            "LayernormBackwardNode requires equal shapes for dscale (output) and scale (input)");
        HIPDNN_RETURN_IF_FALSE(
            attributes.get_dbias()->get_dim().empty()
                || attributes.get_dbias()->get_dim() == attributes.get_scale()->get_dim(),
            ErrorCode::INVALID_VALUE,
            "LayernormBackwardNode requires equal shapes for dbias (output) and scale (input)");
        HIPDNN_RETURN_IF_FALSE(
            attributes.get_normalized_dim_count() >= 0,
            ErrorCode::INVALID_VALUE,
            "LayernormBackwardNode requires a non-negative normalized_dim_count");

        const auto& xDims = attributes.get_x()->get_dim();
        const auto& scaleDims = attributes.get_scale()->get_dim();
        if(xDims.empty() || scaleDims.empty())
        {
            return {};
        }

        // Scale holds the last k dims of x, optionally preceded by 1s; mean and inv_variance
        // hold the first n - k (batch) dims of x, optionally followed by k 1s.
        int64_t normalizedDimCount = 0;
        HIPDNN_CHECK_ERROR(
            detail::deriveLayernormNormalizedDimCount(xDims,
                                                      scaleDims,
                                                      attributes.get_normalized_dim_count(),
                                                      "LayernormBackwardNode",
                                                      normalizedDimCount));
        HIPDNN_CHECK_ERROR(detail::validateLayernormAffineIfSet(
            attributes.get_scale(), xDims, normalizedDimCount, "Scale tensor"));
        HIPDNN_CHECK_ERROR(detail::validateLayernormStatsIfSet(
            attributes.get_mean(), xDims, normalizedDimCount, "Mean tensor"));
        HIPDNN_CHECK_ERROR(detail::validateLayernormStatsIfSet(
            attributes.get_inv_variance(), xDims, normalizedDimCount, "Inverse variance tensor"));

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

        // Infer dy shape and strides if not set
        if(attributes.get_dy()->get_dim().empty())
        {
            attributes.get_dy()->set_dim(attributes.get_x()->get_dim());
        }
        if(attributes.get_dy()->get_stride().empty())
        {
            attributes.get_dy()->set_stride(attributes.get_x()->get_stride());
        }

        // Infer dx shape and strides if not set
        if(attributes.get_dx()->get_dim().empty())
        {
            attributes.get_dx()->set_dim(attributes.get_x()->get_dim());
        }
        if(attributes.get_dx()->get_stride().empty())
        {
            attributes.get_dx()->set_stride(attributes.get_x()->get_stride());
        }

        // Infer dscale shape and strides if not set
        if(attributes.get_dscale()->get_dim().empty())
        {
            attributes.get_dscale()->set_dim(attributes.get_scale()->get_dim());
        }
        if(attributes.get_dscale()->get_stride().empty())
        {
            attributes.get_dscale()->set_stride(attributes.get_scale()->get_stride());
        }

        // Infer dbias shape and strides if not set
        if(attributes.get_dbias()->get_dim().empty())
        {
            attributes.get_dbias()->set_dim(attributes.get_scale()->get_dim());
        }
        if(attributes.get_dbias()->get_stride().empty())
        {
            attributes.get_dbias()->set_stride(attributes.get_scale()->get_stride());
        }

        // Normalized dim count: the explicit attribute if set, else derived from x and scale
        const auto& xDims = attributes.get_x()->get_dim();
        const auto& scaleDims = attributes.get_scale()->get_dim();
        if(!xDims.empty() && !scaleDims.empty())
        {
            int64_t normalizedDimCount = 0;
            HIPDNN_CHECK_ERROR(
                detail::deriveLayernormNormalizedDimCount(xDims,
                                                          scaleDims,
                                                          attributes.get_normalized_dim_count(),
                                                          "LayernormBackwardNode",
                                                          normalizedDimCount));
            attributes.set_normalized_dim_count(normalizedDimCount);
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
