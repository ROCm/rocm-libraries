// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/detail/SdpaFwdPacker.hpp>
#include <hipdnn_frontend/detail/SdpaFwdUnpacker.hpp>
#include <hipdnn_frontend/node/detail/Utilities.hpp>

namespace hipdnn_frontend::graph
{
class SdpaFwdNode : public BaseNode<SdpaFwdNode, NodeType::SDPA_FWD>
{

public:
    SdpaAttributes attributes;

    SdpaFwdNode(SdpaAttributes&& sdpaAttributes, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(sdpaAttributes))
    {
    }

    Error pre_validate_node() const override
    {
        const auto q = attributes.get_q();
        const auto k = attributes.get_k();
        const auto v = attributes.get_v();
        const auto o = attributes.get_o();

        // Validate required tensors are present
        HIPDNN_RETURN_IF_FALSE(
            q, ErrorCode::ATTRIBUTE_NOT_SET, std::string("SdpaFwdNode missing Q input"));
        HIPDNN_RETURN_IF_FALSE(
            k, ErrorCode::ATTRIBUTE_NOT_SET, std::string("SdpaFwdNode missing K input"));
        HIPDNN_RETURN_IF_FALSE(
            v, ErrorCode::ATTRIBUTE_NOT_SET, std::string("SdpaFwdNode missing V input"));
        HIPDNN_RETURN_IF_FALSE(
            o, ErrorCode::ATTRIBUTE_NOT_SET, std::string("SdpaFwdNode missing O output"));

        const auto& qDims = q->get_dim();
        const auto& kDims = k->get_dim();
        const auto& vDims = v->get_dim();

        // Rule 1: Q, K, V must be exactly rank-4
        const size_t reqRank = 4;
        HIPDNN_RETURN_IF_NE(qDims.size(),
                            reqRank,
                            ErrorCode::INVALID_VALUE,
                            "SdpaFwdNode: Query tensor Q must be rank-4, got rank="
                                + std::to_string(qDims.size()));
        HIPDNN_RETURN_IF_NE(kDims.size(),
                            reqRank,
                            ErrorCode::INVALID_VALUE,
                            "SdpaFwdNode: Key tensor K must be rank-4, got rank="
                                + std::to_string(kDims.size()));
        HIPDNN_RETURN_IF_NE(vDims.size(),
                            reqRank,
                            ErrorCode::INVALID_VALUE,
                            "SdpaFwdNode: Value tensor V must be rank-4, got rank="
                                + std::to_string(vDims.size()));
        HIPDNN_RETURN_IF_NE(qDims[0],
                            kDims[0],
                            ErrorCode::INVALID_VALUE,
                            "SdpaFwdNode: batch size mismatch between Q and K: "
                                + std::to_string(qDims[0]) + " vs " + std::to_string(kDims[0]));
        HIPDNN_RETURN_IF_NE(qDims[0],
                            vDims[0],
                            ErrorCode::INVALID_VALUE,
                            "SdpaFwdNode: batch size mismatch between Q and V: "
                                + std::to_string(qDims[0]) + " vs " + std::to_string(vDims[0]));
        // Rule 2: head_dim: Q[-1] == K[-1]; V[-1] is independent (may differ)
        const auto headDimQ = qDims[3];
        const auto headDimK = kDims[3];
        HIPDNN_RETURN_IF_NE(headDimQ,
                            headDimK,
                            ErrorCode::INVALID_VALUE,
                            "SdpaFwdNode: head_dim mismatch between Q and K: "
                                + std::to_string(headDimQ) + " vs " + std::to_string(headDimK));

        // Rule 3: seq_kv: K[-2] == V[-2]
        const auto seqKvK = kDims[2];
        const auto seqKvV = vDims[2];
        HIPDNN_RETURN_IF_NE(seqKvK,
                            seqKvV,
                            ErrorCode::INVALID_VALUE,
                            "SdpaFwdNode: seq_kv mismatch between K and V: "
                                + std::to_string(seqKvK) + " vs " + std::to_string(seqKvV));

        // Rule 4: num_heads % K_heads == 0; num_heads % V_heads == 0 (GQA/MQA)
        const auto numHeads = qDims[1];
        const auto numHeadsK = kDims[1];
        const auto numHeadsV = vDims[1];
        HIPDNN_RETURN_IF_TRUE(numHeadsK <= 0,
                              ErrorCode::INVALID_VALUE,
                              "SdpaFwdNode: num_heads_k must be positive, got "
                                  + std::to_string(numHeadsK));
        HIPDNN_RETURN_IF_TRUE(numHeads % numHeadsK != 0,
                              ErrorCode::INVALID_VALUE,
                              "SdpaFwdNode: num_heads must be divisible by num_heads_k for "
                              "GQA/MQA. num_heads="
                                  + std::to_string(numHeads)
                                  + ", num_heads_k=" + std::to_string(numHeadsK));
        HIPDNN_RETURN_IF_TRUE(numHeadsV <= 0,
                              ErrorCode::INVALID_VALUE,
                              "SdpaFwdNode: num_heads_v must be positive, got "
                                  + std::to_string(numHeadsV));
        HIPDNN_RETURN_IF_TRUE(numHeads % numHeadsV != 0,
                              ErrorCode::INVALID_VALUE,
                              "SdpaFwdNode: num_heads must be divisible by num_heads_v for "
                              "GQA/MQA. num_heads="
                                  + std::to_string(numHeads)
                                  + ", num_heads_v=" + std::to_string(numHeadsV));

        // Rule 5: Optional attention mask validation
        const auto attnMask = attributes.get_bias();
        if(attnMask)
        {
            const auto& maskDims = attnMask->get_dim();
            const auto maskRank = maskDims.size();
            HIPDNN_RETURN_IF_TRUE(maskRank > static_cast<size_t>(4),
                                  ErrorCode::INVALID_VALUE,
                                  "SdpaFwdNode: ATTN_MASK rank must be <= 4, got rank="
                                      + std::to_string(maskRank));

            const auto seqQ = qDims[2];
            const auto seqKv = kDims[2];
            const auto maskLast = maskDims[maskRank - 1];
            HIPDNN_RETURN_IF_TRUE(maskLast != seqKv && maskLast != 1,
                                  ErrorCode::INVALID_VALUE,
                                  "SdpaFwdNode: ATTN_MASK last dim must equal seq_kv ("
                                      + std::to_string(seqKv) + ") or 1, got "
                                      + std::to_string(maskLast));

            if(maskRank >= 2)
            {
                const auto maskSecondLast = maskDims[maskRank - 2];
                HIPDNN_RETURN_IF_TRUE(maskSecondLast != seqQ && maskSecondLast != 1,
                                      ErrorCode::INVALID_VALUE,
                                      "SdpaFwdNode: ATTN_MASK second-to-last dim must equal "
                                      "seq_q ("
                                          + std::to_string(seqQ) + ") or 1, got "
                                          + std::to_string(maskSecondLast));
            }
        }

        // Rule 6: Optional scale must be a scalar tensor (volume == 1)
        const auto scale = attributes.get_attn_scale();
        if(scale)
        {
            HIPDNN_CHECK_ERROR(detail::validateScalarParameter(scale, "SCALE tensor"));
        }

        // Rule 7: If O has dims already set, validate them
        const auto& oDims = o->get_dim();
        if(!oDims.empty())
        {
            const auto headDimV = vDims[3];
            HIPDNN_RETURN_IF_NE(oDims.size(),
                                reqRank,
                                ErrorCode::INVALID_VALUE,
                                "SdpaFwdNode: Output O must be rank-4, got rank="
                                    + std::to_string(oDims.size()));
            HIPDNN_RETURN_IF_NE(oDims[0],
                                qDims[0],
                                ErrorCode::INVALID_VALUE,
                                "SdpaFwdNode: Output O batch mismatch: expected "
                                    + std::to_string(qDims[0]) + " got "
                                    + std::to_string(oDims[0]));
            HIPDNN_RETURN_IF_NE(oDims[1],
                                qDims[1],
                                ErrorCode::INVALID_VALUE,
                                "SdpaFwdNode: Output O num_heads mismatch: expected "
                                    + std::to_string(qDims[1]) + " got "
                                    + std::to_string(oDims[1]));
            HIPDNN_RETURN_IF_NE(oDims[2],
                                qDims[2],
                                ErrorCode::INVALID_VALUE,
                                "SdpaFwdNode: Output O seq_q mismatch: expected "
                                    + std::to_string(qDims[2]) + " got "
                                    + std::to_string(oDims[2]));
            HIPDNN_RETURN_IF_NE(oDims[3],
                                headDimV,
                                ErrorCode::INVALID_VALUE,
                                "SdpaFwdNode: Output O head_dim must match V head_dim: expected "
                                    + std::to_string(headDimV) + " got "
                                    + std::to_string(oDims[3]));
        }

        return {};
    }

    Error infer_properties_node() override
    {
        const auto q = attributes.get_q();
        const auto v = attributes.get_v();
        const auto o = attributes.get_o();

        HIPDNN_RETURN_IF_FALSE(
            q, ErrorCode::ATTRIBUTE_NOT_SET, std::string("SdpaFwdNode missing Q input"));
        HIPDNN_RETURN_IF_FALSE(
            v, ErrorCode::ATTRIBUTE_NOT_SET, std::string("SdpaFwdNode missing V input"));
        HIPDNN_RETURN_IF_FALSE(
            o, ErrorCode::ATTRIBUTE_NOT_SET, std::string("SdpaFwdNode missing O output"));

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        // Output O shape: [batch, num_heads, seq_q, head_dim_v]
        // head_dim_v (V's last dim) may differ from head_dim_qk
        if(o->get_dim().empty())
        {
            const auto& qDims = q->get_dim();
            const auto& vDims = v->get_dim();
            o->set_dim({qDims[0], qDims[1], qDims[2], vDims[3]});
        }

        if(o->get_stride().empty())
        {
            auto oStrides = hipdnn_data_sdk::utilities::generateStrides(o->get_dim());
            o->set_stride(oStrides);
        }

        // Stats output: softmax logsumexp values, shape [batch, num_heads, seq_q, 1]
        const auto stats = attributes.get_stats();
        if(stats)
        {
            if(stats->get_dim().empty())
            {
                const auto& qDims = q->get_dim();
                stats->set_dim({qDims[0], qDims[1], qDims[2], 1});
            }
            if(stats->get_stride().empty())
            {
                auto statsStrides = hipdnn_data_sdk::utilities::generateStrides(stats->get_dim());
                stats->set_stride(statsStrides);
            }
        }

        return {};
    }

    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        SdpaAttributes attrs;
        HIPDNN_CHECK_ERROR(detail::unpackSdpaFwdOperation(opDesc, tensorMap, attrs));
        attributes = std::move(attrs);
        return {};
    }

    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createSdpaFwdOperation(attributes, tensorDescs, operations);
    }
};
} // namespace hipdnn_frontend::graph
