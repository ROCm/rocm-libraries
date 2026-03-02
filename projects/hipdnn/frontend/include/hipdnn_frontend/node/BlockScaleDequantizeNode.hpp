// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/BlockScaleDequantizeAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>

namespace hipdnn_frontend::graph
{
class BlockScaleDequantizeNode : public BaseNode<BlockScaleDequantizeNode>
{
public:
    BlockScaleDequantizeAttributes attributes;

    BlockScaleDequantizeNode(BlockScaleDequantizeAttributes&& blockScaleDequantizeAttrs,
                             const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(blockScaleDequantizeAttrs))
    {
    }

    Error pre_validate_node() const override
    {
        // ====================================================================
        // BLOCK SCALE DEQUANTIZE VALIDATION
        // Dequantizes blocked low-precision data using per-block scales.
        // ====================================================================

        // SECTION 1: Validate Required Tensor Pointers
        HIPDNN_RETURN_IF_FALSE(attributes.get_x(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BlockScaleDequantizeNode missing x for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_scale(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BlockScaleDequantizeNode missing scale for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_y(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BlockScaleDequantizeNode missing y for pre-validation");

        // SECTION 2: Validate block_size is not empty
        HIPDNN_RETURN_IF_FALSE(!attributes.get_block_size().empty(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BlockScaleDequantizeNode block_size must not be empty");

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

        if(y->get_stride().empty() && !x->get_stride().empty())
        {
            y->set_stride(x->get_stride());
        }

        return {};
    }

    flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>
        pack_node(flatbuffers::FlatBufferBuilder& builder) const override
    {
        return hipdnn_data_sdk::data_objects::CreateNodeDirect(
            builder,
            attributes.get_name().c_str(),
            toSdkType(attributes.compute_data_type),
            hipdnn_data_sdk::data_objects::NodeAttributes::BlockScaleDequantizeAttributes,
            attributes.pack_attributes(builder).Union());
    }
};
} // namespace hipdnn_frontend::graph
