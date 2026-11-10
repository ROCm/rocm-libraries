// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <cstdint>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/TopkAttributes.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/detail/TopkPacker.hpp>
#include <hipdnn_frontend/detail/TopkUnpacker.hpp>
#include <vector>

namespace hipdnn_frontend::graph
{
/**
 * @brief Top-k selection along the last dimension.
 *
 * indices.dims = values.dims = x.dims with the last dim replaced by k. Indices are INT32 or
 * INT64 (INT64 when left unset, matching torch.topk); values take the data type of x when
 * left unset. The values output is optional.
 */
class TopkNode : public BaseNode<TopkNode, NodeType::TOPK>
{
public:
    TopkAttributes attributes;

    TopkNode(TopkAttributes&& attrs, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(attrs))
    {
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        TopkAttributes attrs;
        HIPDNN_CHECK_ERROR(detail::unpackTopkOperation(opDesc, tensorMap, attrs));
        attributes = std::move(attrs);
        return {};
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    Error pre_validate_node() const override
    {
        const auto x = attributes.get_x();
        const auto indices = attributes.get_indices();
        const auto values = attributes.get_values();

        HIPDNN_RETURN_IF_FALSE(x, ErrorCode::ATTRIBUTE_NOT_SET, "TopkNode missing x input");
        HIPDNN_RETURN_IF_FALSE(
            indices, ErrorCode::ATTRIBUTE_NOT_SET, "TopkNode missing indices output");

        HIPDNN_RETURN_IF_TRUE(x->get_dim().empty(),
                              ErrorCode::INVALID_VALUE,
                              "TopkNode x tensor must have at least one dimension");

        const auto k = attributes.get_k();
        const auto lastDim = x->get_dim().back();
        HIPDNN_RETURN_IF_TRUE(k < 1 || k > lastDim,
                              ErrorCode::INVALID_VALUE,
                              "TopkNode k must satisfy 1 <= k <= x last dim");

        const auto indicesType = indices->get_data_type();
        HIPDNN_RETURN_IF_TRUE(indicesType != DataType::NOT_SET && indicesType != DataType::INT32
                                  && indicesType != DataType::INT64,
                              ErrorCode::INVALID_VALUE,
                              "TopkNode indices tensor must have INT32 or INT64 data type");

        const auto expectedDims = expectedOutputDims();
        if(!indices->get_dim().empty())
        {
            HIPDNN_RETURN_IF_NE(indices->get_dim(),
                                expectedDims,
                                ErrorCode::INVALID_VALUE,
                                "TopkNode indices dims must equal x dims with last dim k");
        }

        if(values)
        {
            if(!values->get_dim().empty())
            {
                HIPDNN_RETURN_IF_NE(values->get_dim(),
                                    expectedDims,
                                    ErrorCode::INVALID_VALUE,
                                    "TopkNode values dims must equal x dims with last dim k");
            }

            HIPDNN_RETURN_IF_TRUE(x->get_data_type() != DataType::NOT_SET
                                      && values->get_data_type() != DataType::NOT_SET
                                      && x->get_data_type() != values->get_data_type(),
                                  ErrorCode::INVALID_VALUE,
                                  "TopkNode values data type must match x data type");
        }

        return {};
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    Error infer_properties_node() override
    {
        HIPDNN_RETURN_IF_FALSE(attributes.get_x(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "TopkNode missing x for setting properties");
        HIPDNN_RETURN_IF_FALSE(attributes.get_indices(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "TopkNode missing indices for setting properties");

        const auto x = attributes.get_x();
        const auto indices = attributes.get_indices();
        const auto values = attributes.get_values();

        HIPDNN_RETURN_IF_TRUE(x->get_dim().empty(),
                              ErrorCode::INVALID_VALUE,
                              "TopkNode x tensor must have at least one dimension");

        // Resolve x from the graph context first so values inherit its type; indices default
        // to INT64 (torch.topk convention) instead of the graph's io/intermediate type.
        x->fill_from_context(graph_attributes);
        if(indices->get_data_type() == DataType::NOT_SET)
        {
            indices->set_data_type(DataType::INT64);
        }
        if(values && values->get_data_type() == DataType::NOT_SET)
        {
            values->set_data_type(x->get_data_type());
        }

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        const auto expectedDims = expectedOutputDims();
        HIPDNN_CHECK_ERROR(inferOutput(*indices, expectedDims, "indices"));
        if(values)
        {
            HIPDNN_CHECK_ERROR(inferOutput(*values, expectedDims, "values"));
        }

        return {};
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createTopkOperation(attributes, tensorDescs, operations);
    }

private:
    // Requires x set with rank >= 1 (checked by pre_validate_node).
    std::vector<int64_t> expectedOutputDims() const
    {
        auto dims = attributes.get_x()->get_dim();
        dims.back() = attributes.get_k();
        return dims;
    }

    static Error inferOutput(TensorAttributes& tensor,
                             const std::vector<int64_t>& expectedDims,
                             const std::string& label)
    {
        if(tensor.get_dim().empty())
        {
            tensor.set_dim(expectedDims);
        }
        else
        {
            HIPDNN_RETURN_IF_NE(tensor.get_dim(),
                                expectedDims,
                                ErrorCode::INVALID_VALUE,
                                "TopkNode " + label + " dims do not match the inferred dimensions");
        }

        if(tensor.get_stride().empty())
        {
            tensor.set_stride(hipdnn_data_sdk::utilities::generateStrides(tensor.get_dim()));
        }
        return {};
    }
};
} // namespace hipdnn_frontend::graph
