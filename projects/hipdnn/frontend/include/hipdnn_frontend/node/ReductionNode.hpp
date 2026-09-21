// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/ReductionAttributes.hpp>
#include <hipdnn_frontend/detail/ReductionPacker.hpp>
#include <hipdnn_frontend/detail/ReductionUnpacker.hpp>

namespace hipdnn_frontend::graph
{
class ReductionNode : public BaseNode<ReductionNode, NodeType::REDUCTION>
{
public:
    ReductionAttributes attributes;

    ReductionNode(ReductionAttributes&& reductionAttributes, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(reductionAttributes))
    {
    }

    Error pre_validate_node() const override
    {
        if(!attributes.get_x())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "ReductionNode missing input X for pre-validation"};
        }
        if(!attributes.get_y())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "ReductionNode missing output Y for pre-validation"};
        }
        if(!attributes.get_mode().has_value() || attributes.get_mode() == ReductionMode::NOT_SET)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET, "ReductionNode missing mode for pre-validation"};
        }

        const auto& xDims = attributes.get_x()->get_dim();
        const auto& yDims = attributes.get_y()->get_dim();

        if(!xDims.empty() && !yDims.empty())
        {
            HIPDNN_RETURN_IF_NE(xDims.size(),
                                yDims.size(),
                                ErrorCode::INVALID_VALUE,
                                "ReductionNode: X and Y must have the same rank. X rank="
                                    + std::to_string(xDims.size())
                                    + ", Y rank=" + std::to_string(yDims.size()));

            bool hasReduction = false;
            for(size_t i = 0; i < xDims.size(); ++i)
            {
                if(yDims[i] < xDims[i])
                {
                    HIPDNN_RETURN_IF_NE(yDims[i],
                                        static_cast<int64_t>(1),
                                        ErrorCode::INVALID_VALUE,
                                        "ReductionNode: Y dim[" + std::to_string(i)
                                            + "]=" + std::to_string(yDims[i])
                                            + " is less than X dim[" + std::to_string(i)
                                            + "]=" + std::to_string(xDims[i]) + " but is not 1");
                    hasReduction = true;
                }
                else
                {
                    HIPDNN_RETURN_IF_TRUE(yDims[i] > xDims[i],
                                          ErrorCode::INVALID_VALUE,
                                          "ReductionNode: Y dim[" + std::to_string(i)
                                              + "]=" + std::to_string(yDims[i]) + " exceeds X dim["
                                              + std::to_string(i)
                                              + "]=" + std::to_string(xDims[i]));
                }
            }

            HIPDNN_RETURN_IF_FALSE(hasReduction,
                                   ErrorCode::INVALID_VALUE,
                                   "ReductionNode: Y dims must be strictly smaller than X dims in "
                                   "at least one dimension");
        }

        return {};
    }

    Error infer_properties_node() override
    {
        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));
        return {};
    }

    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createReductionOperation(attributes, tensorDescs, operations);
    }

    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        ReductionAttributes attrs;
        HIPDNN_CHECK_ERROR(detail::unpackReductionOperation(opDesc, tensorMap, attrs));
        attributes = std::move(attrs);
        return {};
    }
};
} // namespace hipdnn_frontend::graph
