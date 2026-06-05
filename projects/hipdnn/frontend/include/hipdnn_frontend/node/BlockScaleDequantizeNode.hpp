// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/BlockScaleDequantizeAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/detail/BlockScaleDequantizePacker.hpp>
#include <hipdnn_frontend/detail/BlockScaleDequantizeUnpacker.hpp>
#include <hipdnn_frontend/node/detail/Utilities.hpp>

namespace hipdnn_frontend::graph
{
class BlockScaleDequantizeNode
    : public BaseNode<BlockScaleDequantizeNode, NodeType::BLOCK_SCALE_DEQUANTIZE>
{
public:
    BlockScaleDequantizeAttributes attributes;

    BlockScaleDequantizeNode(BlockScaleDequantizeAttributes&& blockScaleDequantizeAttrs,
                             const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(blockScaleDequantizeAttrs))
    {
    }

    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        BlockScaleDequantizeAttributes attrs;
        HIPDNN_CHECK_ERROR(detail::unpackBlockScaleDequantizeOperation(opDesc, tensorMap, attrs));
        attributes = std::move(attrs);
        return {};
    }

    Error pre_validate_node() const override
    {
        // SECTION 1: Validate required tensor pointers
        HIPDNN_RETURN_IF_FALSE(attributes.get_x(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BlockScaleDequantizeNode missing x for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_scale(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BlockScaleDequantizeNode missing scale for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_y(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BlockScaleDequantizeNode missing y for pre-validation");

        // Dequantize output must be a virtual tensor — it is consumed by downstream
        // operations in a fused graph rather than written to user memory.
        HIPDNN_RETURN_IF_FALSE(attributes.get_y()->get_is_virtual(),
                               ErrorCode::INVALID_VALUE,
                               "BlockScaleDequantizeNode output tensor y must be virtual");

        HIPDNN_RETURN_IF_FALSE(!attributes.get_block_size().empty(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BlockScaleDequantizeNode block_size must not be empty");

        // SECTION 2: Validate tensor dimensions (when set)
        auto x = attributes.get_x();
        auto y = attributes.get_y();

        if(detail::areTensorDimensionsSet(x))
        {
            HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(x, 1, "Input tensor (x)"));
        }

        // Dequantize preserves shape — output dims must match input if set
        if(detail::areTensorDimensionsSet(x) && detail::areTensorDimensionsSet(y))
        {
            HIPDNN_CHECK_ERROR(
                detail::validateTensorShapesMatch(x, y, "Input tensor (x)", "Output tensor (y)"));
        }

        // SECTION 3: Validate block_size values
        const auto& blockSize = attributes.get_block_size();

        for(size_t i = 0; i < blockSize.size(); ++i)
        {
            HIPDNN_RETURN_IF_FALSE(blockSize[i] > 0,
                                   ErrorCode::INVALID_VALUE,
                                   "BlockScaleDequantizeNode block_size[" + std::to_string(i)
                                       + "] must be positive, got " + std::to_string(blockSize[i]));
        }

        // block_size cannot have more entries than tensor dimensions
        if(detail::areTensorDimensionsSet(x))
        {
            HIPDNN_RETURN_IF_FALSE(blockSize.size() <= x->get_dim().size(),
                                   ErrorCode::INVALID_VALUE,
                                   "BlockScaleDequantizeNode block_size has "
                                       + std::to_string(blockSize.size())
                                       + " entries but input tensor has only "
                                       + std::to_string(x->get_dim().size()) + " dimensions");
        }

        return {ErrorCode::OK, ""};
    }

    Error infer_properties_node() override
    {
        auto x = attributes.get_x();
        auto y = attributes.get_y();

        if(!x)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "BlockScaleDequantizeNode missing x for setting properties"};
        }

        if(!y)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "BlockScaleDequantizeNode missing y for setting properties"};
        }

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        if(y->get_dim().empty())
        {
            y->set_dim(x->get_dim());
        }

        if(y->get_stride().empty())
        {
            if(!x->get_stride().empty())
            {
                y->set_stride(x->get_stride());
            }
            else if(!y->get_dim().empty())
            {
                y->set_stride(hipdnn_data_sdk::utilities::generateStrides(y->get_dim()));
            }
        }

        return {};
    }

    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createBlockScaleDequantizeOperation(attributes, tensorDescs, operations);
    }
};
} // namespace hipdnn_frontend::graph
