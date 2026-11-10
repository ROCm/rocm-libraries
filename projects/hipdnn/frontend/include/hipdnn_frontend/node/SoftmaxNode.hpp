// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/SoftmaxAttributes.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/detail/SoftmaxPacker.hpp>
#include <hipdnn_frontend/detail/SoftmaxUnpacker.hpp>

namespace hipdnn_frontend::graph
{
class SoftmaxNode : public BaseNode<SoftmaxNode, NodeType::SOFTMAX>
{
public:
    SoftmaxAttributes attributes;

    SoftmaxNode(SoftmaxAttributes&& attrs, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(attrs))
    {
    }

    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        SoftmaxAttributes attrs;
        HIPDNN_CHECK_ERROR(detail::unpackSoftmaxOperation(opDesc, tensorMap, attrs));
        attributes = std::move(attrs);
        return {};
    }

    Error pre_validate_node() const override
    {
        auto pTensor = attributes.get_p();
        auto sTensor = attributes.get_s();
        auto statsTensor = attributes.get_stats();

        HIPDNN_RETURN_IF_FALSE(pTensor,
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "SoftmaxNode missing p (input) for pre-validation");
        HIPDNN_RETURN_IF_FALSE(sTensor,
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "SoftmaxNode missing s (output) for pre-validation");

        const auto& pDims = pTensor->get_dim();
        HIPDNN_RETURN_IF_TRUE(
            pDims.empty(), ErrorCode::INVALID_VALUE, "SoftmaxNode: p must have rank >= 1");

        // S has the dims of P (checked only when set; inferred otherwise).
        if(!sTensor->get_dim().empty())
        {
            HIPDNN_RETURN_IF_TRUE(sTensor->get_dim() != pDims,
                                  ErrorCode::INVALID_VALUE,
                                  "SoftmaxNode: s dims must equal p dims");
        }

        // Stats has the dims of P with the last (softmax) dim 1.
        if(statsTensor && !statsTensor->get_dim().empty())
        {
            HIPDNN_RETURN_IF_TRUE(statsTensor->get_dim() != statsDims(pDims),
                                  ErrorCode::INVALID_VALUE,
                                  "SoftmaxNode: stats dims must equal p dims with last dim 1");
        }

        return {};
    }

    Error infer_properties_node() override
    {
        auto pTensor = attributes.get_p();
        auto sTensor = attributes.get_s();
        auto statsTensor = attributes.get_stats();

        HIPDNN_RETURN_IF_FALSE(
            pTensor, ErrorCode::ATTRIBUTE_NOT_SET, "SoftmaxNode missing p for setting properties");
        HIPDNN_RETURN_IF_FALSE(
            sTensor, ErrorCode::ATTRIBUTE_NOT_SET, "SoftmaxNode missing s for setting properties");

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        const auto& pDims = pTensor->get_dim();
        const auto& pStrides = pTensor->get_stride();

        if(sTensor->get_dim().empty())
        {
            sTensor->set_dim(pDims);
        }
        if(statsTensor && statsTensor->get_dim().empty())
        {
            statsTensor->set_dim(statsDims(pDims));
        }

        const bool needsStrides
            = sTensor->get_stride().empty() || (statsTensor && statsTensor->get_stride().empty());
        if(!needsStrides)
        {
            return {};
        }

        HIPDNN_RETURN_IF_TRUE(pStrides.empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "SoftmaxNode: Cannot infer output strides - missing input strides");
        HIPDNN_RETURN_IF_NE(pStrides.size(),
                            pDims.size(),
                            ErrorCode::INVALID_VALUE,
                            "SoftmaxNode: p stride rank must equal p dim rank");

        auto strideOrder = hipdnn_data_sdk::utilities::extractStrideOrder(pStrides);
        if(sTensor->get_stride().empty())
        {
            sTensor->set_stride(
                hipdnn_data_sdk::utilities::generateStrides(sTensor->get_dim(), strideOrder));
        }
        if(statsTensor && statsTensor->get_stride().empty())
        {
            statsTensor->set_stride(
                hipdnn_data_sdk::utilities::generateStrides(statsTensor->get_dim(), strideOrder));
        }

        return {};
    }

    /// Dims of the Stats output: dims of P with the last (softmax) dim set to 1.
    static std::vector<int64_t> statsDims(const std::vector<int64_t>& pDims)
    {
        auto dims = pDims;
        if(!dims.empty())
        {
            dims.back() = 1;
        }
        return dims;
    }

    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createSoftmaxOperation(attributes, tensorDescs, operations);
    }
};
}
