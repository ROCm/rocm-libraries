// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/MoeRoutingAttributes.hpp>
#include <hipdnn_frontend/detail/MoeRoutingPacker.hpp>
#include <hipdnn_frontend/detail/MoeRoutingUnpacker.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/detail/Utilities.hpp>

#include <string>
#include <vector>

namespace hipdnn_frontend::graph
{

/// MoE routing: turns topk_ids [S, top_k] into the expert group boundaries
/// first_token_offset [num_experts + 1, 1, 1] and the routed-row indices
/// token_index / token_ks [1, S * top_k, 1] consumed by MoE grouped matmul.
class MoeRoutingNode : public BaseNode<MoeRoutingNode, NodeType::MOE_ROUTING>
{
public:
    MoeRoutingAttributes attributes;

    MoeRoutingNode(MoeRoutingAttributes&& attrs, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(attrs))
    {
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        MoeRoutingAttributes attrs;
        HIPDNN_CHECK_ERROR(detail::unpackMoeRoutingOperation(opDesc, tensorMap, attrs));
        attributes = std::move(attrs);
        return {};
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    Error pre_validate_node() const override
    {
        const auto topkIds = attributes.get_topk_ids();
        const auto firstTokenOffset = attributes.get_first_token_offset();
        const auto tokenIndex = attributes.get_token_index();
        const auto tokenKs = attributes.get_token_ks();

        HIPDNN_RETURN_IF_FALSE(
            topkIds, ErrorCode::ATTRIBUTE_NOT_SET, "MoeRoutingNode missing topk_ids input");
        HIPDNN_RETURN_IF_FALSE(firstTokenOffset,
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "MoeRoutingNode missing first_token_offset output");
        HIPDNN_RETURN_IF_FALSE(
            tokenIndex, ErrorCode::ATTRIBUTE_NOT_SET, "MoeRoutingNode missing token_index output");
        HIPDNN_RETURN_IF_FALSE(
            tokenKs, ErrorCode::ATTRIBUTE_NOT_SET, "MoeRoutingNode missing token_ks output");
        // A single expert needs no routing: every token goes to expert 0 in source order.
        HIPDNN_RETURN_IF_TRUE(attributes.get_num_experts() < 2,
                              ErrorCode::INVALID_VALUE,
                              "MoeRoutingNode num_experts must be at least 2 (a single expert "
                              "needs no routing)");

        constexpr size_t K_TOPK_IDS_RANK = 2;
        HIPDNN_RETURN_IF_NE(topkIds->get_dim().size(),
                            K_TOPK_IDS_RANK,
                            ErrorCode::INVALID_VALUE,
                            "MoeRoutingNode topk_ids tensor must have shape [S, top_k]");
        HIPDNN_RETURN_IF_TRUE(topkIds->get_data_type() != DataType::INT32,
                              ErrorCode::INVALID_VALUE,
                              "MoeRoutingNode topk_ids tensor must have INT32 data type");

        // Outputs may still be unset here (infer_properties_node() runs afterwards and fills
        // them in); whatever the caller did set must already satisfy the contract.
        const auto offsetType = firstTokenOffset->get_data_type();
        HIPDNN_RETURN_IF_TRUE(
            offsetType != DataType::NOT_SET && offsetType != DataType::INT32
                && offsetType != DataType::INT64,
            ErrorCode::INVALID_VALUE,
            "MoeRoutingNode first_token_offset tensor must have INT32 or INT64 data type");
        HIPDNN_RETURN_IF_TRUE(tokenIndex->get_data_type() != DataType::NOT_SET
                                  && tokenIndex->get_data_type() != DataType::INT32,
                              ErrorCode::INVALID_VALUE,
                              "MoeRoutingNode token_index tensor must have INT32 data type");
        HIPDNN_RETURN_IF_TRUE(tokenKs->get_data_type() != DataType::NOT_SET
                                  && tokenKs->get_data_type() != DataType::INT32,
                              ErrorCode::INVALID_VALUE,
                              "MoeRoutingNode token_ks tensor must have INT32 data type");

        HIPDNN_RETURN_IF_TRUE(!firstTokenOffset->get_dim().empty()
                                  && firstTokenOffset->get_dim() != expectedOffsetDims(),
                              ErrorCode::INVALID_VALUE,
                              "MoeRoutingNode first_token_offset tensor must have shape "
                              "[num_experts + 1, 1, 1]");
        const auto rowDims = expectedRowDims();
        HIPDNN_RETURN_IF_TRUE(
            !tokenIndex->get_dim().empty() && tokenIndex->get_dim() != rowDims,
            ErrorCode::INVALID_VALUE,
            "MoeRoutingNode token_index tensor must have shape [1, S * top_k, 1]");
        HIPDNN_RETURN_IF_TRUE(!tokenKs->get_dim().empty() && tokenKs->get_dim() != rowDims,
                              ErrorCode::INVALID_VALUE,
                              "MoeRoutingNode token_ks tensor must have shape [1, S * top_k, 1]");

        return {};
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    Error infer_properties_node() override
    {
        const auto topkIds = attributes.get_topk_ids();
        const auto firstTokenOffset = attributes.get_first_token_offset();
        const auto tokenIndex = attributes.get_token_index();
        const auto tokenKs = attributes.get_token_ks();

        HIPDNN_RETURN_IF_FALSE(topkIds,
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "MoeRoutingNode missing topk_ids for setting properties");
        HIPDNN_RETURN_IF_FALSE(firstTokenOffset,
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "MoeRoutingNode missing first_token_offset for setting properties");
        HIPDNN_RETURN_IF_FALSE(tokenIndex,
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "MoeRoutingNode missing token_index for setting properties");
        HIPDNN_RETURN_IF_FALSE(tokenKs,
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "MoeRoutingNode missing token_ks for setting properties");

        constexpr size_t K_TOPK_IDS_RANK = 2;
        HIPDNN_RETURN_IF_NE(topkIds->get_dim().size(),
                            K_TOPK_IDS_RANK,
                            ErrorCode::INVALID_VALUE,
                            "MoeRoutingNode topk_ids tensor must have shape [S, top_k]");

        // Routing outputs are index tensors: give them their integer types before
        // fill_from_context() would hand them the graph's floating-point io type.
        if(firstTokenOffset->get_data_type() == DataType::NOT_SET)
        {
            firstTokenOffset->set_data_type(DataType::INT32);
        }
        if(tokenIndex->get_data_type() == DataType::NOT_SET)
        {
            tokenIndex->set_data_type(DataType::INT32);
        }
        if(tokenKs->get_data_type() == DataType::NOT_SET)
        {
            tokenKs->set_data_type(DataType::INT32);
        }

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        HIPDNN_CHECK_ERROR(
            inferOutput(firstTokenOffset, expectedOffsetDims(), "first_token_offset"));
        const auto rowDims = expectedRowDims();
        HIPDNN_CHECK_ERROR(inferOutput(tokenIndex, rowDims, "token_index"));
        HIPDNN_CHECK_ERROR(inferOutput(tokenKs, rowDims, "token_ks"));

        return {};
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createMoeRoutingOperation(attributes, tensorDescs, operations);
    }

private:
    std::vector<int64_t> expectedOffsetDims() const
    {
        return {attributes.get_num_experts() + 1, 1, 1};
    }

    std::vector<int64_t> expectedRowDims() const
    {
        const auto& idsDims = attributes.get_topk_ids()->get_dim();
        return {1, idsDims[0] * idsDims[1], 1};
    }

    static Error inferOutput(const std::shared_ptr<TensorAttributes>& tensor,
                             const std::vector<int64_t>& expectedDims,
                             const char* name)
    {
        if(tensor->get_dim().empty())
        {
            tensor->set_dim(expectedDims);
        }
        else
        {
            HIPDNN_RETURN_IF_NE(tensor->get_dim(),
                                expectedDims,
                                ErrorCode::INVALID_VALUE,
                                std::string("MoeRoutingNode ") + name
                                    + " tensor dimensions do not match the inferred dimensions");
        }
        if(tensor->get_stride().empty())
        {
            tensor->set_stride(hipdnn_data_sdk::utilities::generateStrides(tensor->get_dim()));
        }
        return {};
    }
};
} // namespace hipdnn_frontend::graph
