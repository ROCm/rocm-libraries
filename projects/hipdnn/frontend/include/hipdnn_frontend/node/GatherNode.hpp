// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <cstddef>
#include <cstdint>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GatherAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/detail/GatherPacker.hpp>
#include <hipdnn_frontend/detail/GatherUnpacker.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <vector>

namespace hipdnn_frontend::graph
{
/**
 * @brief Gather (index_select / embedding lookup) along one axis.
 *
 * y.dims = x.dims[:axis] ++ indices.dims ++ x.dims[axis + 1:]; y has the data type of x.
 * Indices are INT32 or INT64.
 */
class GatherNode : public BaseNode<GatherNode, NodeType::GATHER>
{
public:
    GatherAttributes attributes;

    GatherNode(GatherAttributes&& attrs, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(attrs))
    {
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        GatherAttributes attrs;
        HIPDNN_CHECK_ERROR(detail::unpackGatherOperation(opDesc, tensorMap, attrs));
        attributes = std::move(attrs);
        return {};
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    Error pre_validate_node() const override
    {
        const auto x = attributes.get_x();
        const auto indices = attributes.get_indices();
        const auto y = attributes.get_y();

        HIPDNN_RETURN_IF_FALSE(x, ErrorCode::ATTRIBUTE_NOT_SET, "GatherNode missing x input");
        HIPDNN_RETURN_IF_FALSE(
            indices, ErrorCode::ATTRIBUTE_NOT_SET, "GatherNode missing indices input");
        HIPDNN_RETURN_IF_FALSE(y, ErrorCode::ATTRIBUTE_NOT_SET, "GatherNode missing y output");

        HIPDNN_RETURN_IF_TRUE(x->get_dim().empty(),
                              ErrorCode::INVALID_VALUE,
                              "GatherNode x tensor must have at least one dimension");
        HIPDNN_RETURN_IF_TRUE(indices->get_dim().empty(),
                              ErrorCode::INVALID_VALUE,
                              "GatherNode indices tensor must have at least one dimension");

        const auto axis = attributes.get_axis();
        const auto xRank = static_cast<int64_t>(x->get_dim().size());
        HIPDNN_RETURN_IF_TRUE(axis < 0 || axis >= xRank,
                              ErrorCode::INVALID_VALUE,
                              "GatherNode axis must satisfy 0 <= axis < rank(x)");

        HIPDNN_RETURN_IF_TRUE(indices->get_data_type() != DataType::INT32
                                  && indices->get_data_type() != DataType::INT64,
                              ErrorCode::INVALID_VALUE,
                              "GatherNode indices tensor must have INT32 or INT64 data type");

        if(!y->get_dim().empty())
        {
            HIPDNN_RETURN_IF_NE(y->get_dim().size(),
                                x->get_dim().size() - 1 + indices->get_dim().size(),
                                ErrorCode::INVALID_VALUE,
                                "GatherNode y rank must equal rank(x) - 1 + rank(indices)");
            HIPDNN_RETURN_IF_NE(y->get_dim(),
                                expectedOutputDims(),
                                ErrorCode::INVALID_VALUE,
                                "GatherNode y dims must equal x.dims[:axis] ++ indices.dims ++ "
                                "x.dims[axis + 1:]");
        }

        HIPDNN_RETURN_IF_TRUE(x->get_data_type() != DataType::NOT_SET
                                  && y->get_data_type() != DataType::NOT_SET
                                  && x->get_data_type() != y->get_data_type(),
                              ErrorCode::INVALID_VALUE,
                              "GatherNode y data type must match x data type");

        return {};
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    Error infer_properties_node() override
    {
        HIPDNN_RETURN_IF_FALSE(attributes.get_x(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "GatherNode missing x for setting properties");
        HIPDNN_RETURN_IF_FALSE(attributes.get_indices(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "GatherNode missing indices for setting properties");
        HIPDNN_RETURN_IF_FALSE(attributes.get_y(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "GatherNode missing y for setting properties");

        const auto x = attributes.get_x();
        const auto y = attributes.get_y();

        // y takes the data type of x; resolve x from the graph context first so an x that
        // relies on io_data_type propagates it to y rather than y falling back separately.
        x->fill_from_context(graph_attributes);
        if(y->get_data_type() == DataType::NOT_SET)
        {
            y->set_data_type(x->get_data_type());
        }

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        const auto expectedDims = expectedOutputDims();
        if(y->get_dim().empty())
        {
            y->set_dim(expectedDims);
        }
        else
        {
            HIPDNN_RETURN_IF_NE(y->get_dim(),
                                expectedDims,
                                ErrorCode::INVALID_VALUE,
                                "GatherNode y dims do not match the inferred dimensions");
        }

        if(y->get_stride().empty())
        {
            y->set_stride(hipdnn_data_sdk::utilities::generateStrides(y->get_dim()));
        }

        return {};
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createGatherOperation(attributes, tensorDescs, operations);
    }

private:
    // Requires x, indices set and 0 <= axis < rank(x) (checked by pre_validate_node).
    std::vector<int64_t> expectedOutputDims() const
    {
        const auto& xDims = attributes.get_x()->get_dim();
        const auto& indexDims = attributes.get_indices()->get_dim();
        const auto axis = static_cast<size_t>(attributes.get_axis());

        std::vector<int64_t> dims;
        dims.reserve(xDims.size() - 1 + indexDims.size());
        dims.insert(dims.end(), xDims.begin(), xDims.begin() + static_cast<std::ptrdiff_t>(axis));
        dims.insert(dims.end(), indexDims.begin(), indexDims.end());
        dims.insert(dims.end(), xDims.begin() + static_cast<std::ptrdiff_t>(axis + 1), xDims.end());
        return dims;
    }
};
} // namespace hipdnn_frontend::graph
