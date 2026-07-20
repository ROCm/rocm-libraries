// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/SdpaBackwardAttributes.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/detail/SdpaBwdPacker.hpp>
#include <hipdnn_frontend/detail/SdpaBwdUnpacker.hpp>
#include <hipdnn_frontend/node/detail/Utilities.hpp>

namespace hipdnn_frontend::graph
{

/**
 * @class SdpaBwdNode
 * @brief Node implementing scaled dot-product attention backward pass
 *
 * Computes gradients dQ, dK, dV given:
 * - Forward inputs Q, K, V, O
 * - Upstream gradient dO
 * - Softmax statistics (logsumexp) from the forward pass
 *
 * The backward pass uses the flash attention algorithm to compute:
 *   dV = softmax(Q * K^T / sqrt(d_k))^T * dO
 *   dP = dO * V^T         (where P = softmax(Q * K^T / sqrt(d_k)))
 *   dS = P * (dP - rowsum(dO * O))
 *   dQ = dS * K / sqrt(d_k)
 *   dK = dS^T * Q / sqrt(d_k)
 */
class SdpaBwdNode : public BaseNode<SdpaBwdNode, NodeType::SDPA_BWD>
{
public:
    SdpaBackwardAttributes attributes;

    SdpaBwdNode(SdpaBackwardAttributes&& sdpaAttrs, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(sdpaAttrs))
    {
    }

    Error pre_validate_node() const override
    {
        const auto q = attributes.get_q();
        const auto k = attributes.get_k();
        const auto v = attributes.get_v();
        const auto o = attributes.get_o();
        const auto dOut = attributes.get_do();
        const auto stats = attributes.get_stats();
        const auto dq = attributes.get_dq();
        const auto dk = attributes.get_dk();
        const auto dv = attributes.get_dv();

        // Validate required input tensors
        HIPDNN_RETURN_IF_FALSE(
            q, ErrorCode::ATTRIBUTE_NOT_SET, std::string("SdpaBwdNode missing Q input"));
        HIPDNN_RETURN_IF_FALSE(
            k, ErrorCode::ATTRIBUTE_NOT_SET, std::string("SdpaBwdNode missing K input"));
        HIPDNN_RETURN_IF_FALSE(
            v, ErrorCode::ATTRIBUTE_NOT_SET, std::string("SdpaBwdNode missing V input"));
        HIPDNN_RETURN_IF_FALSE(
            o, ErrorCode::ATTRIBUTE_NOT_SET, std::string("SdpaBwdNode missing O input"));
        HIPDNN_RETURN_IF_FALSE(
            dOut, ErrorCode::ATTRIBUTE_NOT_SET, std::string("SdpaBwdNode missing dO input"));
        HIPDNN_RETURN_IF_FALSE(
            stats, ErrorCode::ATTRIBUTE_NOT_SET, std::string("SdpaBwdNode missing Stats input"));

        // Validate required output tensors
        HIPDNN_RETURN_IF_FALSE(
            dq, ErrorCode::ATTRIBUTE_NOT_SET, std::string("SdpaBwdNode missing dQ output"));
        HIPDNN_RETURN_IF_FALSE(
            dk, ErrorCode::ATTRIBUTE_NOT_SET, std::string("SdpaBwdNode missing dK output"));
        HIPDNN_RETURN_IF_FALSE(
            dv, ErrorCode::ATTRIBUTE_NOT_SET, std::string("SdpaBwdNode missing dV output"));

        const auto& qDims = q->get_dim();
        const auto& kDims = k->get_dim();
        const auto& vDims = v->get_dim();

        // Rule 1: Q, K, V must be exactly rank-4
        const size_t reqRank = 4;
        HIPDNN_RETURN_IF_NE(qDims.size(),
                            reqRank,
                            ErrorCode::INVALID_VALUE,
                            "SdpaBwdNode: Query tensor Q must be rank-4, got rank="
                                + std::to_string(qDims.size()));
        HIPDNN_RETURN_IF_NE(kDims.size(),
                            reqRank,
                            ErrorCode::INVALID_VALUE,
                            "SdpaBwdNode: Key tensor K must be rank-4, got rank="
                                + std::to_string(kDims.size()));
        HIPDNN_RETURN_IF_NE(vDims.size(),
                            reqRank,
                            ErrorCode::INVALID_VALUE,
                            "SdpaBwdNode: Value tensor V must be rank-4, got rank="
                                + std::to_string(vDims.size()));

        // Rule 2: batch size consistency
        HIPDNN_RETURN_IF_NE(qDims[0],
                            kDims[0],
                            ErrorCode::INVALID_VALUE,
                            "SdpaBwdNode: batch size mismatch between Q and K: "
                                + std::to_string(qDims[0]) + " vs " + std::to_string(kDims[0]));
        HIPDNN_RETURN_IF_NE(qDims[0],
                            vDims[0],
                            ErrorCode::INVALID_VALUE,
                            "SdpaBwdNode: batch size mismatch between Q and V: "
                                + std::to_string(qDims[0]) + " vs " + std::to_string(vDims[0]));

        // Rule 3: head_dim: Q[-1] == K[-1]
        const auto headDimQ = qDims[3];
        const auto headDimK = kDims[3];
        HIPDNN_RETURN_IF_NE(headDimQ,
                            headDimK,
                            ErrorCode::INVALID_VALUE,
                            "SdpaBwdNode: head_dim mismatch between Q and K: "
                                + std::to_string(headDimQ) + " vs " + std::to_string(headDimK));

        // Rule 4: seq_kv: K[-2] == V[-2]
        const auto seqKvK = kDims[2];
        const auto seqKvV = vDims[2];
        HIPDNN_RETURN_IF_NE(seqKvK,
                            seqKvV,
                            ErrorCode::INVALID_VALUE,
                            "SdpaBwdNode: seq_kv mismatch between K and V: "
                                + std::to_string(seqKvK) + " vs " + std::to_string(seqKvV));

        // Rule 5: num_heads % K_heads == 0; num_heads % V_heads == 0 (GQA/MQA)
        const auto numHeads = qDims[1];
        const auto numHeadsK = kDims[1];
        const auto numHeadsV = vDims[1];
        HIPDNN_RETURN_IF_TRUE(numHeadsK <= 0,
                              ErrorCode::INVALID_VALUE,
                              "SdpaBwdNode: num_heads_k must be positive, got "
                                  + std::to_string(numHeadsK));
        HIPDNN_RETURN_IF_TRUE(numHeads % numHeadsK != 0,
                              ErrorCode::INVALID_VALUE,
                              "SdpaBwdNode: num_heads must be divisible by num_heads_k for "
                              "GQA/MQA. num_heads="
                                  + std::to_string(numHeads)
                                  + ", num_heads_k=" + std::to_string(numHeadsK));
        HIPDNN_RETURN_IF_TRUE(numHeadsV <= 0,
                              ErrorCode::INVALID_VALUE,
                              "SdpaBwdNode: num_heads_v must be positive, got "
                                  + std::to_string(numHeadsV));
        HIPDNN_RETURN_IF_TRUE(numHeads % numHeadsV != 0,
                              ErrorCode::INVALID_VALUE,
                              "SdpaBwdNode: num_heads must be divisible by num_heads_v for "
                              "GQA/MQA. num_heads="
                                  + std::to_string(numHeads)
                                  + ", num_heads_v=" + std::to_string(numHeadsV));

        // Rule 6: dO must match O shape [batch, num_heads, seq_q, head_dim_v]
        const auto& oDims = o->get_dim();
        if(!oDims.empty())
        {
            const auto& dOutDims = dOut->get_dim();
            if(!dOutDims.empty())
            {
                HIPDNN_RETURN_IF_NE(dOutDims.size(),
                                    reqRank,
                                    ErrorCode::INVALID_VALUE,
                                    "SdpaBwdNode: dO must be rank-4, got rank="
                                        + std::to_string(dOutDims.size()));
                HIPDNN_RETURN_IF_NE(dOutDims[0],
                                    oDims[0],
                                    ErrorCode::INVALID_VALUE,
                                    "SdpaBwdNode: dO batch mismatch with O");
                HIPDNN_RETURN_IF_NE(dOutDims[1],
                                    oDims[1],
                                    ErrorCode::INVALID_VALUE,
                                    "SdpaBwdNode: dO num_heads mismatch with O");
                HIPDNN_RETURN_IF_NE(dOutDims[2],
                                    oDims[2],
                                    ErrorCode::INVALID_VALUE,
                                    "SdpaBwdNode: dO seq_q mismatch with O");
                HIPDNN_RETURN_IF_NE(dOutDims[3],
                                    oDims[3],
                                    ErrorCode::INVALID_VALUE,
                                    "SdpaBwdNode: dO head_dim mismatch with O");
            }
        }

        // Rule 7: Optional attention mask validation
        const auto attnMask = attributes.get_bias();
        if(attnMask)
        {
            const auto& maskDims = attnMask->get_dim();
            const auto maskRank = maskDims.size();
            HIPDNN_RETURN_IF_TRUE(maskRank > static_cast<size_t>(4),
                                  ErrorCode::INVALID_VALUE,
                                  "SdpaBwdNode: ATTN_MASK rank must be <= 4, got rank="
                                      + std::to_string(maskRank));

            const auto seqQ = qDims[2];
            const auto seqKv = kDims[2];
            const auto maskLast = maskDims[maskRank - 1];
            HIPDNN_RETURN_IF_TRUE(maskLast != seqKv && maskLast != 1,
                                  ErrorCode::INVALID_VALUE,
                                  "SdpaBwdNode: ATTN_MASK last dim must equal seq_kv ("
                                      + std::to_string(seqKv) + ") or 1, got "
                                      + std::to_string(maskLast));

            if(maskRank >= 2)
            {
                const auto maskSecondLast = maskDims[maskRank - 2];
                HIPDNN_RETURN_IF_TRUE(maskSecondLast != seqQ && maskSecondLast != 1,
                                      ErrorCode::INVALID_VALUE,
                                      "SdpaBwdNode: ATTN_MASK second-to-last dim must equal "
                                      "seq_q ("
                                          + std::to_string(seqQ) + ") or 1, got "
                                          + std::to_string(maskSecondLast));
            }
        }

        // Rule 8: Optional dropout mask validation
        // Shape must be rank-4 (B or 1, H_q or 1, S_q, S_kv) — last two dims are exact,
        // first two dims may broadcast (1 allowed).
        const auto dropoutMask = attributes.get_dropout_mask();
        if(dropoutMask)
        {
            const auto& dmDims = dropoutMask->get_dim();
            HIPDNN_RETURN_IF_NE(dmDims.size(),
                                static_cast<size_t>(4),
                                ErrorCode::INVALID_VALUE,
                                "SdpaBwdNode: DROPOUT_MASK must be rank-4, got rank="
                                    + std::to_string(dmDims.size()));

            const auto seqQ = qDims[2];
            const auto seqKv = kDims[2];
            const auto batch = qDims[0];

            HIPDNN_RETURN_IF_TRUE(dmDims[0] != batch && dmDims[0] != 1,
                                  ErrorCode::INVALID_VALUE,
                                  "SdpaBwdNode: DROPOUT_MASK dim[0] must equal batch ("
                                      + std::to_string(batch) + ") or 1, got "
                                      + std::to_string(dmDims[0]));
            HIPDNN_RETURN_IF_TRUE(dmDims[1] != numHeads && dmDims[1] != 1,
                                  ErrorCode::INVALID_VALUE,
                                  "SdpaBwdNode: DROPOUT_MASK dim[1] must equal num_heads ("
                                      + std::to_string(numHeads) + ") or 1, got "
                                      + std::to_string(dmDims[1]));
            HIPDNN_RETURN_IF_NE(dmDims[2],
                                seqQ,
                                ErrorCode::INVALID_VALUE,
                                "SdpaBwdNode: DROPOUT_MASK dim[2] must equal seq_q ("
                                    + std::to_string(seqQ) + "), got " + std::to_string(dmDims[2]));
            HIPDNN_RETURN_IF_NE(dmDims[3],
                                seqKv,
                                ErrorCode::INVALID_VALUE,
                                "SdpaBwdNode: DROPOUT_MASK dim[3] must equal seq_kv ("
                                    + std::to_string(seqKv) + "), got "
                                    + std::to_string(dmDims[3]));
        }

        // Rule 9: Padding mask — when padding_mask=true, seq_len_q and seq_len_kv must
        // be provided as rank-4 tensors with shape (B, 1, 1, 1) and INT32 data type.
        if(attributes.padding_mask)
        {
            const auto seqLenQ = attributes.get_seq_len_q();
            const auto seqLenKv = attributes.get_seq_len_kv();

            HIPDNN_RETURN_IF_FALSE(
                seqLenQ,
                ErrorCode::ATTRIBUTE_NOT_SET,
                std::string("SdpaBwdNode: padding_mask=true requires seq_len_q tensor to be set"));
            HIPDNN_RETURN_IF_FALSE(
                seqLenKv,
                ErrorCode::ATTRIBUTE_NOT_SET,
                std::string("SdpaBwdNode: padding_mask=true requires seq_len_kv tensor to be set"));

            const auto& sqDims = seqLenQ->get_dim();
            const auto& skvDims = seqLenKv->get_dim();
            const auto batch = qDims[0];

            HIPDNN_RETURN_IF_NE(sqDims.size(),
                                static_cast<size_t>(4),
                                ErrorCode::INVALID_VALUE,
                                "SdpaBwdNode: seq_len_q must be rank-4, got rank="
                                    + std::to_string(sqDims.size()));
            HIPDNN_RETURN_IF_NE(sqDims[0],
                                batch,
                                ErrorCode::INVALID_VALUE,
                                "SdpaBwdNode: seq_len_q dim[0] must equal batch ("
                                    + std::to_string(batch) + "), got "
                                    + std::to_string(sqDims[0]));
            HIPDNN_RETURN_IF_NE(sqDims[1],
                                static_cast<int64_t>(1),
                                ErrorCode::INVALID_VALUE,
                                "SdpaBwdNode: seq_len_q dim[1] must be 1, got "
                                    + std::to_string(sqDims[1]));
            HIPDNN_RETURN_IF_NE(sqDims[2],
                                static_cast<int64_t>(1),
                                ErrorCode::INVALID_VALUE,
                                "SdpaBwdNode: seq_len_q dim[2] must be 1, got "
                                    + std::to_string(sqDims[2]));
            HIPDNN_RETURN_IF_NE(sqDims[3],
                                static_cast<int64_t>(1),
                                ErrorCode::INVALID_VALUE,
                                "SdpaBwdNode: seq_len_q dim[3] must be 1, got "
                                    + std::to_string(sqDims[3]));
            HIPDNN_RETURN_IF_TRUE(seqLenQ->get_data_type() != DataType::INT32,
                                  ErrorCode::INVALID_VALUE,
                                  std::string("SdpaBwdNode: seq_len_q must have INT32 data "
                                              "type"));

            HIPDNN_RETURN_IF_NE(skvDims.size(),
                                static_cast<size_t>(4),
                                ErrorCode::INVALID_VALUE,
                                "SdpaBwdNode: seq_len_kv must be rank-4, got rank="
                                    + std::to_string(skvDims.size()));
            HIPDNN_RETURN_IF_NE(skvDims[0],
                                batch,
                                ErrorCode::INVALID_VALUE,
                                "SdpaBwdNode: seq_len_kv dim[0] must equal batch ("
                                    + std::to_string(batch) + "), got "
                                    + std::to_string(skvDims[0]));
            HIPDNN_RETURN_IF_NE(skvDims[1],
                                static_cast<int64_t>(1),
                                ErrorCode::INVALID_VALUE,
                                "SdpaBwdNode: seq_len_kv dim[1] must be 1, got "
                                    + std::to_string(skvDims[1]));
            HIPDNN_RETURN_IF_NE(skvDims[2],
                                static_cast<int64_t>(1),
                                ErrorCode::INVALID_VALUE,
                                "SdpaBwdNode: seq_len_kv dim[2] must be 1, got "
                                    + std::to_string(skvDims[2]));
            HIPDNN_RETURN_IF_NE(skvDims[3],
                                static_cast<int64_t>(1),
                                ErrorCode::INVALID_VALUE,
                                "SdpaBwdNode: seq_len_kv dim[3] must be 1, got "
                                    + std::to_string(skvDims[3]));
            HIPDNN_RETURN_IF_TRUE(seqLenKv->get_data_type() != DataType::INT32,
                                  ErrorCode::INVALID_VALUE,
                                  std::string("SdpaBwdNode: seq_len_kv must have INT32 data "
                                              "type"));
        }

        // Rule 10: Optional attention scale must be a scalar tensor
        const auto scale = attributes.get_attn_scale();
        if(scale)
        {
            HIPDNN_CHECK_ERROR(detail::validateScalarParameter(scale, "SCALE tensor"));
        }

        return {};
    }

    Error infer_properties_node() override
    {
        const auto q = attributes.get_q();
        const auto k = attributes.get_k();
        const auto v = attributes.get_v();
        const auto dq = attributes.get_dq();
        const auto dk = attributes.get_dk();
        const auto dv = attributes.get_dv();

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        // dQ has same shape as Q: [batch, num_heads, seq_q, head_dim]
        if(dq->get_dim().empty())
        {
            dq->set_dim(q->get_dim());
        }
        if(dq->get_stride().empty())
        {
            dq->set_stride(hipdnn_data_sdk::utilities::generateStrides(dq->get_dim()));
        }

        // dK has same shape as K: [batch, num_kv_heads, seq_kv, head_dim]
        if(dk->get_dim().empty())
        {
            dk->set_dim(k->get_dim());
        }
        if(dk->get_stride().empty())
        {
            dk->set_stride(hipdnn_data_sdk::utilities::generateStrides(dk->get_dim()));
        }

        // dV has same shape as V: [batch, num_kv_heads, seq_kv, head_dim_v]
        if(dv->get_dim().empty())
        {
            dv->set_dim(v->get_dim());
        }
        if(dv->get_stride().empty())
        {
            dv->set_stride(hipdnn_data_sdk::utilities::generateStrides(dv->get_dim()));
        }

        return {};
    }

    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createSdpaBwdOperation(attributes, tensorDescs, operations);
    }

    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        SdpaBackwardAttributes sdpaAttr;
        HIPDNN_CHECK_ERROR(detail::unpackSdpaBwdOperation(opDesc, tensorMap, sdpaAttr));
        attributes = std::move(sdpaAttr);
        return {};
    }
};
} // namespace hipdnn_frontend::graph
