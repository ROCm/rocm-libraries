// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/PoolingFwdAttributes.hpp>
#include <hipdnn_frontend/detail/PoolingFwdPacker.hpp>
#include <hipdnn_frontend/detail/PoolingFwdUnpacker.hpp>

namespace hipdnn_frontend::graph
{
class PoolingFwdNode : public BaseNode<PoolingFwdNode, NodeType::POOLING_FWD>
{
public:
    PoolingFwdAttributes attributes;

    PoolingFwdNode(PoolingFwdAttributes&& poolingFwdAttributes, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(poolingFwdAttributes))
    {
    }

    Error pre_validate_node() const override
    {
        if(!attributes.get_x())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "PoolingFwdNode missing input X for pre-validation"};
        }
        if(!attributes.get_y())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "PoolingFwdNode missing output Y for pre-validation"};
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
        return detail::createPoolingFwdOperation(attributes, tensorDescs, operations);
    }

    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        PoolingFwdAttributes attrs;
        HIPDNN_CHECK_ERROR(detail::unpackPoolingFwdOperation(opDesc, tensorMap, attrs));
        attributes = std::move(attrs);
        return {};
    }
};
} // namespace hipdnn_frontend::graph
