// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/MoeGroupedMatmulAttributes.hpp>
#include <hipdnn_frontend/detail/MoeGroupedMatmulPacker.hpp>
#include <hipdnn_frontend/detail/MoeGroupedMatmulUnpacker.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/detail/Utilities.hpp>

namespace hipdnn_frontend::graph
{
class MoeGroupedMatmulNode : public BaseNode<MoeGroupedMatmulNode, NodeType::MOE_GROUPED_MATMUL>
{
public:
    MoeGroupedMatmulAttributes attributes;

    MoeGroupedMatmulNode(MoeGroupedMatmulAttributes&& attrs, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(attrs))
    {
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        MoeGroupedMatmulAttributes attrs;
        HIPDNN_CHECK_ERROR(detail::unpackMoeGroupedMatmulOperation(opDesc, tensorMap, attrs));
        attributes = std::move(attrs);
        return {};
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    Error pre_validate_node() const override
    {
        const auto token = attributes.get_token();
        const auto weight = attributes.get_weight();
        const auto firstTokenOffset = attributes.get_first_token_offset();
        const auto output = attributes.get_output();

        HIPDNN_RETURN_IF_FALSE(
            token, ErrorCode::ATTRIBUTE_NOT_SET, "MoeGroupedMatmulNode missing token input");
        HIPDNN_RETURN_IF_FALSE(
            weight, ErrorCode::ATTRIBUTE_NOT_SET, "MoeGroupedMatmulNode missing weight input");
        HIPDNN_RETURN_IF_FALSE(firstTokenOffset,
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "MoeGroupedMatmulNode missing first_token_offset input");
        HIPDNN_RETURN_IF_FALSE(
            output, ErrorCode::ATTRIBUTE_NOT_SET, "MoeGroupedMatmulNode missing output");

        constexpr size_t K_TENSOR_RANK = 3;
        HIPDNN_CHECK_ERROR(
            detail::validateMinimumTensorDimensions(token, K_TENSOR_RANK, "MoE token tensor"));
        HIPDNN_CHECK_ERROR(
            detail::validateMinimumTensorDimensions(weight, K_TENSOR_RANK, "MoE weight tensor"));
        HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(
            firstTokenOffset, K_TENSOR_RANK, "MoE first-token-offset tensor"));

        HIPDNN_RETURN_IF_NE(token->get_dim().size(),
                            K_TENSOR_RANK,
                            ErrorCode::INVALID_VALUE,
                            "MoE token tensor must have shape [1, tokens, K]");
        HIPDNN_RETURN_IF_NE(weight->get_dim().size(),
                            K_TENSOR_RANK,
                            ErrorCode::INVALID_VALUE,
                            "MoE weight tensor must have shape [experts, K, N]");
        HIPDNN_RETURN_IF_NE(
            firstTokenOffset->get_dim().size(),
            K_TENSOR_RANK,
            ErrorCode::INVALID_VALUE,
            "MoE first-token-offset tensor must have shape [batch * experts, 1, 1]");
        HIPDNN_RETURN_IF_NE(token->get_dim()[0],
                            1,
                            ErrorCode::INVALID_VALUE,
                            "MoE token tensor must have a singleton leading dimension");
        HIPDNN_RETURN_IF_NE(firstTokenOffset->get_dim()[1],
                            1,
                            ErrorCode::INVALID_VALUE,
                            "MoE first-token-offset tensor must have trailing dimensions [1, 1]");
        HIPDNN_RETURN_IF_NE(firstTokenOffset->get_dim()[2],
                            1,
                            ErrorCode::INVALID_VALUE,
                            "MoE first-token-offset tensor must have trailing dimensions [1, 1]");
        HIPDNN_RETURN_IF_NE(token->get_dim()[2],
                            weight->get_dim()[1],
                            ErrorCode::INVALID_VALUE,
                            "MoE token K dimension must match the weight K dimension");
        HIPDNN_RETURN_IF_TRUE(weight->get_dim()[0] <= 0,
                              ErrorCode::INVALID_VALUE,
                              "MoE weight tensor must describe at least one expert");
        HIPDNN_RETURN_IF_TRUE(
            firstTokenOffset->get_dim()[0] % weight->get_dim()[0] != 0,
            ErrorCode::INVALID_VALUE,
            "MoE first-token-offset tensor length must be divisible by expert count");
        HIPDNN_RETURN_IF_TRUE(firstTokenOffset->get_data_type() != DataType::INT32,
                              ErrorCode::INVALID_VALUE,
                              "MoE first-token-offset tensor must have INT32 data type");

        const auto validateRoutingTensor
            = [](const std::shared_ptr<TensorAttributes>& tensor, const char* name) -> Error {
            constexpr size_t K_ROUTING_TENSOR_RANK = 3;
            HIPDNN_RETURN_IF_FALSE(tensor,
                                   ErrorCode::ATTRIBUTE_NOT_SET,
                                   std::string("MoeGroupedMatmulNode missing ") + name + " input");
            HIPDNN_RETURN_IF_TRUE(tensor->get_data_type() != DataType::INT32,
                                  ErrorCode::INVALID_VALUE,
                                  std::string(name) + " must have INT32 data type");
            HIPDNN_CHECK_ERROR(
                detail::validateMinimumTensorDimensions(tensor, K_ROUTING_TENSOR_RANK, name));
            HIPDNN_RETURN_IF_NE(tensor->get_dim().size(),
                                K_ROUTING_TENSOR_RANK,
                                ErrorCode::INVALID_VALUE,
                                std::string(name) + " must have shape [1, routed_tokens, 1]");
            HIPDNN_RETURN_IF_NE(tensor->get_dim()[0],
                                1,
                                ErrorCode::INVALID_VALUE,
                                std::string(name) + " must have a singleton leading dimension");
            HIPDNN_RETURN_IF_NE(tensor->get_dim()[2],
                                1,
                                ErrorCode::INVALID_VALUE,
                                std::string(name) + " must have a singleton trailing dimension");
            return {};
        };

        switch(attributes.get_mode())
        {
        case MoeGroupedMatmulMode::NONE:
            return {};
        case MoeGroupedMatmulMode::GATHER:
            return validateRoutingTensor(attributes.get_token_index(), "MoE token-index tensor");
        case MoeGroupedMatmulMode::SCATTER:
            HIPDNN_CHECK_ERROR(
                validateRoutingTensor(attributes.get_token_index(), "MoE token-index tensor"));
            HIPDNN_CHECK_ERROR(
                validateRoutingTensor(attributes.get_token_ks(), "MoE token-ks tensor"));
            HIPDNN_RETURN_IF_NE(
                attributes.get_token_index()->get_dim()[1],
                attributes.get_token_ks()->get_dim()[1],
                ErrorCode::INVALID_VALUE,
                "MoE token-index and token-ks tensors must have the same routed-token count");
            HIPDNN_RETURN_IF_TRUE(attributes.get_top_k() <= 0,
                                  ErrorCode::INVALID_VALUE,
                                  "MoE SCATTER mode requires top_k to be positive");
            HIPDNN_RETURN_IF_TRUE(attributes.get_top_k() > weight->get_dim()[0],
                                  ErrorCode::INVALID_VALUE,
                                  "MoE top_k must not exceed the number of experts");
            return {};
        default:
            return {ErrorCode::INVALID_VALUE, "MoeGroupedMatmulNode has an unknown routing mode"};
        }
    }

    // NOLINTNEXTLINE(readability-identifier-naming,readability-make-member-function-const)
    Error infer_properties_node() override
    {
        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        const std::vector<int64_t> expectedOutputDims
            = {1,
               attributes.get_mode() == MoeGroupedMatmulMode::GATHER
                   ? attributes.get_token_index()->get_dim()[1]
                   : attributes.get_token()->get_dim()[1],
               attributes.get_weight()->get_dim()[2]};

        const auto output = attributes.get_output();
        if(output->get_dim().empty())
        {
            output->set_dim(expectedOutputDims);
        }
        else
        {
            HIPDNN_RETURN_IF_NE(
                output->get_dim(),
                expectedOutputDims,
                ErrorCode::INVALID_VALUE,
                "MoeGroupedMatmul output tensor dimensions do not match the inferred dimensions");
        }

        if(output->get_stride().empty())
        {
            output->set_stride(hipdnn_data_sdk::utilities::generateStrides(output->get_dim()));
        }

        return {};
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createMoeGroupedMatmulOperation(attributes, tensorDescs, operations);
    }
};
} // namespace hipdnn_frontend::graph
