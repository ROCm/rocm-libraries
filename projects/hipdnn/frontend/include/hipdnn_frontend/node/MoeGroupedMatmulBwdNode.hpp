// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/MoeGroupedMatmulBwdAttributes.hpp>
#include <hipdnn_frontend/detail/MoeGroupedMatmulBwdPacker.hpp>
#include <hipdnn_frontend/detail/MoeGroupedMatmulBwdUnpacker.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/detail/Utilities.hpp>

namespace hipdnn_frontend::graph
{
// Hand-maintained; do not regenerate with --mode frontend (node.hpp.j2's stub/
// custom_checks paths would overwrite this with TODO placeholders).
class MoeGroupedMatmulBwdNode
    : public BaseNode<MoeGroupedMatmulBwdNode, NodeType::MOE_GROUPED_MATMUL_BWD>
{
public:
    MoeGroupedMatmulBwdAttributes attributes;

    MoeGroupedMatmulBwdNode(MoeGroupedMatmulBwdAttributes&& attrs,
                            const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(attrs))
    {
    }

    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        MoeGroupedMatmulBwdAttributes attrs;
        HIPDNN_CHECK_ERROR(detail::unpackMoeGroupedMatmulBwdOperation(opDesc, tensorMap, attrs));
        attributes = std::move(attrs);
        return {};
    }

    Error pre_validate_node() const override
    {
        // Validate required input tensors
        HIPDNN_RETURN_IF_FALSE(
            attributes.get_doutput(),
            ErrorCode::ATTRIBUTE_NOT_SET,
            "MoeGroupedMatmulBwdNode missing doutput (input) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_token(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "MoeGroupedMatmulBwdNode missing token (input) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(
            attributes.get_first_token_offset(),
            ErrorCode::ATTRIBUTE_NOT_SET,
            "MoeGroupedMatmulBwdNode missing first_token_offset (input) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(
            attributes.get_dweight(),
            ErrorCode::ATTRIBUTE_NOT_SET,
            "MoeGroupedMatmulBwdNode missing dweight (output) for pre-validation");

        // Validate required tensor dimensions
        HIPDNN_RETURN_IF_TRUE(
            attributes.get_doutput()->get_dim().empty(),
            ErrorCode::ATTRIBUTE_NOT_SET,
            "MoeGroupedMatmulBwdNode missing doutput dimensions for pre-validation");

        HIPDNN_RETURN_IF_TRUE(
            attributes.get_token()->get_dim().empty(),
            ErrorCode::ATTRIBUTE_NOT_SET,
            "MoeGroupedMatmulBwdNode missing token dimensions for pre-validation");

        HIPDNN_RETURN_IF_TRUE(
            attributes.get_first_token_offset()->get_dim().empty(),
            ErrorCode::ATTRIBUTE_NOT_SET,
            "MoeGroupedMatmulBwdNode missing first_token_offset dimensions for pre-validation");

        HIPDNN_RETURN_IF_TRUE(
            attributes.get_dweight()->get_dim().empty(),
            ErrorCode::ATTRIBUTE_NOT_SET,
            "MoeGroupedMatmulBwdNode missing dweight dimensions for pre-validation");

        // Custom validation checks
        const auto doutputTensor = attributes.get_doutput();
        const auto tokenTensor = attributes.get_token();
        const auto firstTokenOffsetTensor = attributes.get_first_token_offset();
        const auto dweightTensor = attributes.get_dweight();

        constexpr size_t K_TENSOR_RANK = 3;

        HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(
            doutputTensor, K_TENSOR_RANK, "MoE doutput tensor"));
        HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(
            tokenTensor, K_TENSOR_RANK, "MoE token tensor"));
        HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(
            firstTokenOffsetTensor, K_TENSOR_RANK, "MoE first_token_offset tensor"));

        // Validate doutput and token have exact rank 3 (minimum-rank checks above only
        // reject rank < 3; without these, a rank 4+ tensor would pass here and only be
        // caught later, at backend finalize()).
        HIPDNN_RETURN_IF_NE(doutputTensor->get_dim().size(),
                            K_TENSOR_RANK,
                            ErrorCode::INVALID_VALUE,
                            "MoE doutput tensor must have rank 3");
        HIPDNN_RETURN_IF_NE(tokenTensor->get_dim().size(),
                            K_TENSOR_RANK,
                            ErrorCode::INVALID_VALUE,
                            "MoE token tensor must have rank 3");

        // Real tokens are flattened into dim[1]; dim[0] is a singleton placeholder axis.
        HIPDNN_RETURN_IF_NE(doutputTensor->get_dim()[0],
                            1,
                            ErrorCode::INVALID_VALUE,
                            "MoE doutput tensor must have a singleton leading dimension");
        HIPDNN_RETURN_IF_NE(tokenTensor->get_dim()[0],
                            1,
                            ErrorCode::INVALID_VALUE,
                            "MoE token tensor must have a singleton leading dimension");

        // Validate dweight has rank 3 [experts, K, N] - reject rather than infer, the expert
        // count is not derivable from any input.
        HIPDNN_RETURN_IF_NE(dweightTensor->get_dim().size(),
                            K_TENSOR_RANK,
                            ErrorCode::INVALID_VALUE,
                            "MoE dweight tensor must have shape [experts, K, N]");

        HIPDNN_RETURN_IF_TRUE(dweightTensor->get_dim()[0] <= 0,
                              ErrorCode::INVALID_VALUE,
                              "MoE dweight tensor must describe at least one expert");

        // Validate dweight dim[1] (K) matches token dim[2]
        HIPDNN_RETURN_IF_NE(dweightTensor->get_dim()[1],
                            tokenTensor->get_dim()[2],
                            ErrorCode::INVALID_VALUE,
                            "MoE dweight K dimension must match the token K dimension");

        // Validate dweight dim[2] (N) matches doutput dim[2]
        HIPDNN_RETURN_IF_NE(dweightTensor->get_dim()[2],
                            doutputTensor->get_dim()[2],
                            ErrorCode::INVALID_VALUE,
                            "MoE dweight N dimension must match the doutput N dimension");

        // Validate doutput dim[1] (token count) matches token dim[1]
        HIPDNN_RETURN_IF_NE(doutputTensor->get_dim()[1],
                            tokenTensor->get_dim()[1],
                            ErrorCode::INVALID_VALUE,
                            "MoE doutput token-count dimension must match the token tensor");

        // Validate first_token_offset has rank 3 with shape [experts, 1, 1] and dim[0] equal
        // to dweight dim[0] (expert count)
        HIPDNN_RETURN_IF_NE(firstTokenOffsetTensor->get_dim().size(),
                            K_TENSOR_RANK,
                            ErrorCode::INVALID_VALUE,
                            "MoE first-token-offset tensor must have shape [experts, 1, 1]");
        HIPDNN_RETURN_IF_NE(firstTokenOffsetTensor->get_dim()[1],
                            1,
                            ErrorCode::INVALID_VALUE,
                            "MoE first-token-offset tensor must have trailing dimensions [1, 1]");
        HIPDNN_RETURN_IF_NE(firstTokenOffsetTensor->get_dim()[2],
                            1,
                            ErrorCode::INVALID_VALUE,
                            "MoE first-token-offset tensor must have trailing dimensions [1, 1]");
        HIPDNN_RETURN_IF_NE(
            firstTokenOffsetTensor->get_dim()[0],
            dweightTensor->get_dim()[0],
            ErrorCode::INVALID_VALUE,
            "MoE first-token-offset expert count must match the dweight expert count");
        HIPDNN_RETURN_IF_TRUE(firstTokenOffsetTensor->get_data_type() != DataType::INT32,
                              ErrorCode::INVALID_VALUE,
                              "MoE first-token-offset tensor must have INT32 data type");

        return {};
    }

    Error infer_properties_node() override
    {
        // Validate required tensor pointers
        HIPDNN_RETURN_IF_FALSE(attributes.get_doutput(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "MoeGroupedMatmulBwdNode missing doutput for setting properties");

        HIPDNN_RETURN_IF_FALSE(attributes.get_token(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "MoeGroupedMatmulBwdNode missing token for setting properties");

        HIPDNN_RETURN_IF_FALSE(
            attributes.get_first_token_offset(),
            ErrorCode::ATTRIBUTE_NOT_SET,
            "MoeGroupedMatmulBwdNode missing first_token_offset for setting properties");

        HIPDNN_RETURN_IF_FALSE(attributes.get_dweight(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "MoeGroupedMatmulBwdNode missing dweight for setting properties");

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        // dweight = [experts, token.dim[2], doutput.dim[2]], but the expert count cannot be
        // derived from any input tensor, so dweight dimensions are caller-supplied and
        // validated (see pre_validate_node) rather than inferred here.
        auto dweightTensor = attributes.get_dweight();

        HIPDNN_RETURN_IF_TRUE(dweightTensor->get_dim().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "MoeGroupedMatmulBwdNode: dweight dimensions must be set by the "
                              "caller (expert count cannot be inferred)");

        constexpr size_t K_TENSOR_RANK = 3;
        HIPDNN_RETURN_IF_NE(dweightTensor->get_dim().size(),
                            K_TENSOR_RANK,
                            ErrorCode::INVALID_VALUE,
                            "MoeGroupedMatmulBwdNode: dweight must have rank 3 [experts, K, N]");

        // Infer output strides if not set
        if(dweightTensor->get_stride().empty())
        {
            dweightTensor->set_stride(
                hipdnn_data_sdk::utilities::generateStrides(dweightTensor->get_dim()));
        }

        return {};
    }

    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createMoeGroupedMatmulBwdOperation(attributes, tensorDescs, operations);
    }
};
}
