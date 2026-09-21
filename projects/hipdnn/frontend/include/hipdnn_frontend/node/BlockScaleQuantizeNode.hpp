// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/BlockScaleQuantizeAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/detail/BlockScaleQuantizePacker.hpp>
#include <hipdnn_frontend/detail/BlockScaleQuantizeUnpacker.hpp>
#include <hipdnn_frontend/node/detail/Utilities.hpp>

namespace hipdnn_frontend::graph
{
class BlockScaleQuantizeNode
    : public BaseNode<BlockScaleQuantizeNode, NodeType::BLOCK_SCALE_QUANTIZE>
{
public:
    BlockScaleQuantizeAttributes attributes;

    BlockScaleQuantizeNode(BlockScaleQuantizeAttributes&& blockScaleQuantizeAttrs,
                           const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(blockScaleQuantizeAttrs))
    {
    }

    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        BlockScaleQuantizeAttributes bsqAttr;
        HIPDNN_CHECK_ERROR(detail::unpackBlockScaleQuantizeOperation(opDesc, tensorMap, bsqAttr));
        attributes = std::move(bsqAttr);
        return {};
    }

    Error pre_validate_node() const override
    {
        // SECTION 1: Validate required tensor pointers
        HIPDNN_RETURN_IF_FALSE(attributes.get_x(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BlockScaleQuantizeNode missing x for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_y(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BlockScaleQuantizeNode missing y for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_scale(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BlockScaleQuantizeNode missing scale for pre-validation");

        // SECTION 2: Validate block_size is set and positive
        auto blockSize = attributes.get_block_size();
        HIPDNN_RETURN_IF_FALSE(blockSize.has_value(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BlockScaleQuantizeNode block_size not set");

        HIPDNN_RETURN_IF_FALSE(blockSize.value() > 0,
                               ErrorCode::INVALID_VALUE,
                               "BlockScaleQuantizeNode block_size must be positive, got "
                                   + std::to_string(blockSize.value()));

        // SECTION 3: Validate tensor dimensions (when set)
        auto x = attributes.get_x();
        auto y = attributes.get_y();

        if(detail::areTensorDimensionsSet(x))
        {
            HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(x, 1, "Input tensor (x)"));

            const auto& xDims = x->get_dim();

            // Validate axis is within tensor rank
            auto axis = attributes.get_axis();
            if(axis.has_value())
            {
                HIPDNN_RETURN_IF_FALSE(axis.value() >= 0,
                                       ErrorCode::INVALID_VALUE,
                                       "BlockScaleQuantizeNode axis must be non-negative, got "
                                           + std::to_string(axis.value()));

                HIPDNN_RETURN_IF_FALSE(static_cast<size_t>(axis.value()) < xDims.size(),
                                       ErrorCode::INVALID_VALUE,
                                       "BlockScaleQuantizeNode axis " + std::to_string(axis.value())
                                           + " exceeds input tensor rank "
                                           + std::to_string(xDims.size()));
            }

            // Validate divisibility of the target dimension by block_size
            const size_t targetAxis
                = axis.has_value() ? static_cast<size_t>(axis.value()) : xDims.size() - 1;
            if(targetAxis < xDims.size() && xDims[targetAxis] > 0)
            {
                HIPDNN_RETURN_IF_FALSE(
                    xDims[targetAxis] % blockSize.value() == 0,
                    ErrorCode::INVALID_VALUE,
                    "BlockScaleQuantizeNode dimension at axis " + std::to_string(targetAxis) + " ("
                        + std::to_string(xDims[targetAxis]) + ") must be divisible by block_size ("
                        + std::to_string(blockSize.value()) + ")");
            }
        }

        // Quantize preserves shape -- output dims must match input if set
        if(detail::areTensorDimensionsSet(x) && detail::areTensorDimensionsSet(y))
        {
            HIPDNN_CHECK_ERROR(
                detail::validateTensorShapesMatch(x, y, "Input tensor (x)", "Output tensor (y)"));
        }

        return {ErrorCode::OK, ""};
    }

    Error infer_properties_node() override
    {
        auto x = attributes.get_x();
        auto y = attributes.get_y();
        auto scale = attributes.get_scale();

        if(!x)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "BlockScaleQuantizeNode missing x for setting properties"};
        }

        if(!y)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "BlockScaleQuantizeNode missing y for setting properties"};
        }

        if(!scale)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "BlockScaleQuantizeNode missing scale for setting properties"};
        }

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        auto axis = attributes.get_axis();

        // Infer Y dims and strides from X
        if(y->get_dim().empty())
        {
            y->set_dim(x->get_dim());
        }

        if(y->get_stride().empty())
        {
            if(attributes.get_transpose() && !x->get_stride().empty())
            {
                y->set_stride(hipdnn_data_sdk::utilities::generateStridesWithPackedAxis(
                    x->get_stride(), x->get_dim(), y->get_dim(), axis));
            }
            else if(!x->get_stride().empty())
            {
                y->set_stride(x->get_stride());
            }
            else if(!y->get_dim().empty())
            {
                y->set_stride(hipdnn_data_sdk::utilities::generateStrides(y->get_dim()));
            }
        }

        // Infer scale dims from X dims and block_size
        if(scale->get_dim().empty() && !x->get_dim().empty())
        {
            auto blockSize = attributes.get_block_size();
            if(blockSize.has_value() && blockSize.value() > 0)
            {
                auto scaleDims = x->get_dim();
                const size_t scaleAxis
                    = axis.has_value() ? static_cast<size_t>(axis.value()) : scaleDims.size() - 1;

                if(scaleAxis < scaleDims.size())
                {
                    scaleDims[scaleAxis] /= blockSize.value();
                }
                scale->set_dim(scaleDims);
            }
        }

        if(scale->get_stride().empty())
        {
            if(attributes.get_transpose() && !x->get_stride().empty() && !scale->get_dim().empty())
            {
                scale->set_stride(hipdnn_data_sdk::utilities::generateStridesWithPackedAxis(
                    x->get_stride(), x->get_dim(), scale->get_dim(), axis));
            }
            else if(!x->get_stride().empty() && !scale->get_dim().empty())
            {
                auto strideOrder = hipdnn_data_sdk::utilities::extractStrideOrder(x->get_stride());
                scale->set_stride(
                    hipdnn_data_sdk::utilities::generateStrides(scale->get_dim(), strideOrder));
            }
            else if(!scale->get_dim().empty())
            {
                scale->set_stride(hipdnn_data_sdk::utilities::generateStrides(scale->get_dim()));
            }
        }

        return {};
    }

    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createBlockScaleQuantizeOperation(attributes, tensorDescs, operations);
    }
};
} // namespace hipdnn_frontend::graph
