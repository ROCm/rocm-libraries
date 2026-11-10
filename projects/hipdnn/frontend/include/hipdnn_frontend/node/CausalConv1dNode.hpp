// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/CausalConv1dAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/detail/CausalConv1dPacker.hpp>
#include <hipdnn_frontend/detail/CausalConv1dUnpacker.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/detail/Utilities.hpp>

#include <string>
#include <vector>

namespace hipdnn_frontend::graph
{
class CausalConv1dNode : public BaseNode<CausalConv1dNode, NodeType::CAUSAL_CONV1D>
{
public:
    CausalConv1dAttributes attributes;

    CausalConv1dNode(CausalConv1dAttributes&& attrs, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(attrs))
    {
    }

    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        CausalConv1dAttributes attrs;
        HIPDNN_CHECK_ERROR(detail::unpackCausalConv1dOperation(opDesc, tensorMap, attrs));
        attributes = std::move(attrs);
        return {};
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    Error pre_validate_node() const override
    {
        const auto x = attributes.get_x();
        const auto weight = attributes.get_weight();
        const auto cuSeqlens = attributes.get_cu_seqlens();
        const auto y = attributes.get_y();
        const auto bias = attributes.get_bias();
        const auto initialState = attributes.get_initial_state();
        const auto stateIndices = attributes.get_state_indices();
        const auto finalState = attributes.get_final_state();

        HIPDNN_RETURN_IF_FALSE(
            x, ErrorCode::ATTRIBUTE_NOT_SET, "CausalConv1dNode missing x (input)");
        HIPDNN_RETURN_IF_FALSE(
            weight, ErrorCode::ATTRIBUTE_NOT_SET, "CausalConv1dNode missing weight (input)");
        HIPDNN_RETURN_IF_FALSE(
            cuSeqlens, ErrorCode::ATTRIBUTE_NOT_SET, "CausalConv1dNode missing cu_seqlens (input)");
        HIPDNN_RETURN_IF_FALSE(
            y, ErrorCode::ATTRIBUTE_NOT_SET, "CausalConv1dNode missing y (output)");

        HIPDNN_RETURN_IF_NE(x->get_dim().size(),
                            2u,
                            ErrorCode::INVALID_VALUE,
                            "CausalConv1d x tensor must have shape [T, C]");
        HIPDNN_RETURN_IF_NE(weight->get_dim().size(),
                            2u,
                            ErrorCode::INVALID_VALUE,
                            "CausalConv1d weight tensor must have shape [C, W]");
        HIPDNN_RETURN_IF_NE(cuSeqlens->get_dim().size(),
                            1u,
                            ErrorCode::INVALID_VALUE,
                            "CausalConv1d cu_seqlens tensor must have shape [N + 1]");

        const auto channels = x->get_dim()[1];
        const auto kernelWidth = weight->get_dim()[1];
        const auto numSequences = cuSeqlens->get_dim()[0] - 1;

        HIPDNN_RETURN_IF_TRUE(channels <= 0,
                              ErrorCode::INVALID_VALUE,
                              "CausalConv1d x tensor must have at least one channel");
        HIPDNN_RETURN_IF_NE(weight->get_dim()[0],
                            channels,
                            ErrorCode::INVALID_VALUE,
                            "CausalConv1d weight channel dimension must match x channels");
        HIPDNN_RETURN_IF_TRUE(kernelWidth < 2,
                              ErrorCode::INVALID_VALUE,
                              "CausalConv1d kernel width (weight dim 1) must be at least 2");
        HIPDNN_RETURN_IF_TRUE(numSequences < 1,
                              ErrorCode::INVALID_VALUE,
                              "CausalConv1d cu_seqlens must hold at least two offsets");
        HIPDNN_RETURN_IF_FALSE(isIndexType(cuSeqlens->get_data_type()),
                               ErrorCode::INVALID_VALUE,
                               "CausalConv1d cu_seqlens tensor must have INT32 or INT64 data type");

        if(bias)
        {
            HIPDNN_RETURN_IF_NE(bias->get_dim(),
                                std::vector<int64_t>{channels},
                                ErrorCode::INVALID_VALUE,
                                "CausalConv1d bias tensor must have shape [C]");
        }

        if(stateIndices)
        {
            HIPDNN_RETURN_IF_FALSE(initialState,
                                   ErrorCode::INVALID_VALUE,
                                   "CausalConv1d state_indices requires initial_state");
            HIPDNN_RETURN_IF_NE(stateIndices->get_dim(),
                                std::vector<int64_t>{numSequences},
                                ErrorCode::INVALID_VALUE,
                                "CausalConv1d state_indices tensor must have shape [N]");
            HIPDNN_RETURN_IF_FALSE(
                isIndexType(stateIndices->get_data_type()),
                ErrorCode::INVALID_VALUE,
                "CausalConv1d state_indices tensor must have INT32 or INT64 data type");
        }

        if(initialState)
        {
            HIPDNN_CHECK_ERROR(
                validateStateDims(initialState->get_dim(), "initial_state", numSequences));
        }

        HIPDNN_RETURN_IF_NE(attributes.get_output_final_state(),
                            finalState != nullptr,
                            ErrorCode::INVALID_VALUE,
                            "CausalConv1d final_state output must be set exactly when "
                            "output_final_state is requested");

        if(finalState && !finalState->get_dim().empty())
        {
            HIPDNN_CHECK_ERROR(
                validateStateDims(finalState->get_dim(), "final_state", numSequences));
            if(stateIndices)
            {
                HIPDNN_RETURN_IF_NE(
                    finalState->get_dim()[0],
                    initialState->get_dim()[0],
                    ErrorCode::INVALID_VALUE,
                    "CausalConv1d final_state pool must have as many rows as initial_state");
            }
        }

        return {};
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    Error infer_properties_node() override
    {
        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        const auto x = attributes.get_x();
        const auto y = attributes.get_y();

        if(y->get_dim().empty())
        {
            y->set_dim(x->get_dim());
        }
        else
        {
            HIPDNN_RETURN_IF_NE(y->get_dim(),
                                x->get_dim(),
                                ErrorCode::INVALID_VALUE,
                                "CausalConv1d y tensor dimensions must match x [T, C]");
        }

        if(y->get_stride().empty())
        {
            if(x->get_stride().size() == x->get_dim().size())
            {
                const auto strideOrder
                    = hipdnn_data_sdk::utilities::extractStrideOrder(x->get_stride());
                y->set_stride(
                    hipdnn_data_sdk::utilities::generateStrides(y->get_dim(), strideOrder));
            }
            else
            {
                y->set_stride(hipdnn_data_sdk::utilities::generateStrides(y->get_dim()));
            }
        }

        const auto finalState = attributes.get_final_state();
        if(finalState)
        {
            if(finalState->get_dim().empty())
            {
                const auto initialState = attributes.get_initial_state();
                const auto stateRows = attributes.get_state_indices()
                                           ? initialState->get_dim()[0]
                                           : attributes.get_cu_seqlens()->get_dim()[0] - 1;
                finalState->set_dim(
                    {stateRows, x->get_dim()[1], attributes.get_weight()->get_dim()[1] - 1});
            }
            if(finalState->get_stride().empty())
            {
                finalState->set_stride(
                    hipdnn_data_sdk::utilities::generateStrides(finalState->get_dim()));
            }
        }

        return {};
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createCausalConv1dOperation(attributes, tensorDescs, operations);
    }

private:
    static bool isIndexType(DataType dataType)
    {
        return dataType == DataType::INT32 || dataType == DataType::INT64;
    }

    // A state is [N, C, W - 1], or a pool [P, C, W - 1] with P >= 1 when state_indices is set.
    Error validateStateDims(const std::vector<int64_t>& dims,
                            const std::string& name,
                            int64_t numSequences) const
    {
        HIPDNN_RETURN_IF_NE(dims.size(),
                            3u,
                            ErrorCode::INVALID_VALUE,
                            "CausalConv1d " + name + " tensor must have shape [N, C, W - 1]");
        if(attributes.get_state_indices())
        {
            HIPDNN_RETURN_IF_TRUE(dims[0] < 1,
                                  ErrorCode::INVALID_VALUE,
                                  "CausalConv1d " + name + " pool must have at least one row");
        }
        else
        {
            HIPDNN_RETURN_IF_NE(dims[0],
                                numSequences,
                                ErrorCode::INVALID_VALUE,
                                "CausalConv1d " + name
                                    + " leading dimension must match the sequence count N");
        }
        HIPDNN_RETURN_IF_NE(dims[1],
                            attributes.get_x()->get_dim()[1],
                            ErrorCode::INVALID_VALUE,
                            "CausalConv1d " + name + " channel dimension must match x channels");
        HIPDNN_RETURN_IF_NE(dims[2],
                            attributes.get_weight()->get_dim()[1] - 1,
                            ErrorCode::INVALID_VALUE,
                            "CausalConv1d " + name + " last dimension must be W - 1");
        return {};
    }
};
} // namespace hipdnn_frontend::graph
