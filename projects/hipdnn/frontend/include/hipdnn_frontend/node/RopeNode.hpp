// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/RopeAttributes.hpp>
#include <hipdnn_frontend/detail/RopePacker.hpp>
#include <hipdnn_frontend/detail/RopeUnpacker.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/detail/Utilities.hpp>

namespace hipdnn_frontend::graph
{
class RopeNode : public BaseNode<RopeNode, NodeType::ROPE>
{
public:
    RopeAttributes attributes;

    RopeNode(RopeAttributes&& attrs, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(attrs))
    {
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        RopeAttributes attrs;
        HIPDNN_CHECK_ERROR(detail::unpackRopeOperation(opDesc, tensorMap, attrs));
        attributes = std::move(attrs);
        return {};
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    Error pre_validate_node() const override
    {
        const auto x = attributes.get_x();
        const auto freqs = attributes.get_freqs();
        const auto y = attributes.get_y();

        HIPDNN_RETURN_IF_FALSE(x, ErrorCode::ATTRIBUTE_NOT_SET, "RopeNode missing x input");
        HIPDNN_RETURN_IF_FALSE(freqs, ErrorCode::ATTRIBUTE_NOT_SET, "RopeNode missing freqs input");
        HIPDNN_RETURN_IF_FALSE(y, ErrorCode::ATTRIBUTE_NOT_SET, "RopeNode missing y output");

        constexpr size_t K_MIN_RANK = 2;
        HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(x, K_MIN_RANK, "RoPE x tensor"));
        HIPDNN_CHECK_ERROR(
            detail::validateMinimumTensorDimensions(freqs, K_MIN_RANK, "RoPE freqs tensor"));

        const auto& xDims = x->get_dim();
        const auto& freqsDims = freqs->get_dim();
        HIPDNN_RETURN_IF_NE(freqsDims.size(),
                            xDims.size(),
                            ErrorCode::INVALID_VALUE,
                            "RoPE freqs tensor must have the same rank as x");
        HIPDNN_RETURN_IF_TRUE(freqs->get_data_type() != DataType::FLOAT,
                              ErrorCode::INVALID_VALUE,
                              "RoPE freqs tensor must have FLOAT data type");

        const auto headDim = xDims.back();
        const auto ropeDim = attributes.get_rope_dim() == 0 ? headDim : attributes.get_rope_dim();
        HIPDNN_RETURN_IF_TRUE(attributes.get_rope_dim() < 0,
                              ErrorCode::INVALID_VALUE,
                              "RoPE rope_dim must not be negative");
        HIPDNN_RETURN_IF_TRUE(ropeDim <= 0 || ropeDim > headDim,
                              ErrorCode::INVALID_VALUE,
                              "RoPE rope_dim must be in (0, D] where D is the last dim of x");
        HIPDNN_RETURN_IF_TRUE(
            ropeDim % 2 != 0, ErrorCode::INVALID_VALUE, "RoPE rope_dim must be even");

        const auto ropeOffset = attributes.get_rope_offset();
        if(ropeOffset.has_value())
        {
            HIPDNN_RETURN_IF_TRUE(
                *ropeOffset < 0, ErrorCode::INVALID_VALUE, "RoPE rope_offset must not be negative");
            HIPDNN_RETURN_IF_TRUE(*ropeOffset + ropeDim > headDim,
                                  ErrorCode::INVALID_VALUE,
                                  "RoPE rope_offset + rope_dim must not exceed the last dim of x");
        }

        HIPDNN_RETURN_IF_NE(freqsDims.back(),
                            ropeDim,
                            ErrorCode::INVALID_VALUE,
                            "RoPE freqs last dim must equal rope_dim");
        for(size_t i = 0; i + 1 < xDims.size(); ++i)
        {
            HIPDNN_RETURN_IF_TRUE(freqsDims[i] != xDims[i] && freqsDims[i] != 1,
                                  ErrorCode::INVALID_VALUE,
                                  "RoPE freqs dims must equal x dims or be 1 (except the last)");
        }

        if(!y->get_dim().empty())
        {
            HIPDNN_RETURN_IF_NE(y->get_dim(),
                                xDims,
                                ErrorCode::INVALID_VALUE,
                                "RoPE y tensor dimensions must equal x dimensions");
        }

        return {};
    }

    // NOLINTNEXTLINE(readability-identifier-naming,readability-make-member-function-const)
    Error infer_properties_node() override
    {
        HIPDNN_RETURN_IF_FALSE(attributes.get_x(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "RopeNode missing x for setting properties");
        HIPDNN_RETURN_IF_FALSE(attributes.get_y(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "RopeNode missing y for setting properties");

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        const auto x = attributes.get_x();
        const auto y = attributes.get_y();

        if(y->get_dim().empty())
        {
            y->set_dim(x->get_dim());
        }
        else
        {
            HIPDNN_RETURN_IF_NE(y->get_dim(),
                                x->get_dim(),
                                ErrorCode::INVALID_VALUE,
                                "RoPE y tensor dimensions must equal x dimensions");
        }

        if(y->get_stride().empty())
        {
            const auto& xStrides = x->get_stride();
            if(xStrides.size() == y->get_dim().size())
            {
                y->set_stride(xStrides);
            }
            else
            {
                y->set_stride(hipdnn_data_sdk::utilities::generateStrides(y->get_dim()));
            }
        }

        return {};
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createRopeOperation(attributes, tensorDescs, operations);
    }
};
} // namespace hipdnn_frontend::graph
