// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/MatmulAttributes.hpp>
#include <hipdnn_frontend/detail/MatmulPacker.hpp>
#include <hipdnn_frontend/detail/MatmulUnpacker.hpp>
#include <hipdnn_frontend/node/detail/Utilities.hpp>

namespace hipdnn_frontend::graph
{
class MatmulNode : public BaseNode<MatmulNode, NodeType::MATMUL>
{

public:
    MatmulAttributes attributes;

    MatmulNode(MatmulAttributes&& matmulAttributes, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(matmulAttributes))
    {
    }

    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        MatmulAttributes attrs;
        HIPDNN_CHECK_ERROR(detail::unpackMatmulOperation(opDesc, tensorMap, attrs));
        attributes = std::move(attrs);
        return {};
    }

    Error pre_validate_node() const override
    {
        // Validate tensor pointers
        const auto a = attributes.get_a();
        const auto b = attributes.get_b();
        const auto c = attributes.get_c();
        const auto minRank = 2;

        // Validate minimum dimensionality for matmul operands
        HIPDNN_CHECK_ERROR(validateTensors(a, b, c));
        HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(a, minRank, "Input tensor A"));
        HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(b, minRank, "Input tensor B"));

        const auto& aDims = a->get_dim(); // [..., M, K]
        const auto& bDims = b->get_dim(); // [..., K, N]

        // Require A and B to have the same number of dims
        const auto aRank = aDims.size();
        const auto bRank = bDims.size();
        HIPDNN_RETURN_IF_NE(aRank,
                            bRank,
                            ErrorCode::INVALID_VALUE,
                            "Matmul input tensors A and B must have the same rank: A has rank="
                                + std::to_string(aRank) + ", B has rank=" + std::to_string(bRank));

        // Validate inner reduction dimension K matches
        const auto aK = aDims[aDims.size() - 1];
        const auto bK = bDims[bDims.size() - 2];
        HIPDNN_RETURN_IF_NE(
            aK,
            bK,
            ErrorCode::INVALID_VALUE,
            "MatmulNode: Inner dimensions must match (a[...,K] vs b[...,K,...]). Got "
                + std::to_string(aK) + " vs " + std::to_string(bK));

        // Validate broadcast-compatibility of batch dimensions (leading dims)
        const auto batchDims = aRank - minRank;
        HIPDNN_CHECK_ERROR(validateBroadcastableBatchDims(aDims, bDims, batchDims));

        return {};
    }

    Error infer_properties_node() override
    {
        // Validate tensor pointers
        const auto a = attributes.get_a();
        const auto b = attributes.get_b();
        const auto c = attributes.get_c();

        HIPDNN_CHECK_ERROR(validateTensors(a, b, c));
        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        // Infer output dimensions if not set
        if(c->get_dim().empty())
        {
            const auto& aDims = a->get_dim(); // [..., M, K]
            const auto& bDims = b->get_dim(); // [..., K, N]

            // Compute common broadcasted batch shape from leading dims using divisibility rule:
            // for each batch dim i: require a[i] % b[i] == 0 || b[i] % a[i] == 0, and take max(a[i], b[i])
            const auto minRank = 2;
            const auto batchDims = aDims.size() - minRank;
            HIPDNN_CHECK_ERROR(validateBroadcastableBatchDims(aDims, bDims, batchDims));

            std::vector<int64_t> cDims;
            cDims.reserve(aDims.size());
            for(size_t i = 0; i < batchDims; ++i)
            {
                cDims.push_back(std::max(aDims[i], bDims[i]));
            }
            cDims.push_back(aDims[aDims.size() - 2]); // M
            cDims.push_back(bDims[bDims.size() - 1]); // N

            c->set_dim(cDims);
        }

        // Infer output strides if not set
        if(c->get_stride().empty())
        {
            auto cStrides = hipdnn_data_sdk::utilities::generateStrides(c->get_dim());
            c->set_stride(cStrides);
        }

        return {};
    }

    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createMatmulOperation(attributes, tensorDescs, operations);
    }

private:
    static Error validateTensors(const std::shared_ptr<TensorAttributes>& a,
                                 const std::shared_ptr<TensorAttributes>& b,
                                 const std::shared_ptr<TensorAttributes>& c)
    {
        HIPDNN_RETURN_IF_FALSE(
            a, ErrorCode::ATTRIBUTE_NOT_SET, std::string("MatmulNode missing A input"));
        HIPDNN_RETURN_IF_FALSE(
            b, ErrorCode::ATTRIBUTE_NOT_SET, std::string("MatmulNode missing B input"));
        HIPDNN_RETURN_IF_FALSE(
            c, ErrorCode::ATTRIBUTE_NOT_SET, std::string("MatmulNode missing C output"));

        return {};
    }

    static Error validateBroadcastableBatchDims(const std::vector<int64_t>& aDims,
                                                const std::vector<int64_t>& bDims,
                                                size_t batchDims)
    {
        for(size_t i = 0; i < batchDims; ++i)
        {
            const auto aDimVal = aDims[i];
            const auto bDimVal = bDims[i];
            HIPDNN_RETURN_IF_TRUE(aDimVal % bDimVal != 0 && bDimVal % aDimVal != 0,
                                  ErrorCode::INVALID_VALUE,
                                  std::string("Matmul input tensors A and B have incompatible ")
                                      + "batch dimensions for broadcasting at index "
                                      + std::to_string(i) + ": A has dim=" + std::to_string(aDimVal)
                                      + ", B has dim=" + std::to_string(bDimVal));
        }

        return {};
    }
};
}
