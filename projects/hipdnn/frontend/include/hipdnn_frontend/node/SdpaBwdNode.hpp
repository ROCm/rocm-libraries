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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

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
 *   dV = softmax(scale * Q * K^T)^T * dO
 *   dP = dO * V^T         (where P = softmax(scale * Q * K^T))
 *   dS = P * (dP - rowsum(dO * O))
 *   dQ = scale * dS * K
 *   dK = scale * dS^T * Q
 * where scale is the attention scale, 1.0 (no scaling) when none is set.
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
        HIPDNN_RETURN_IF_TRUE(attributes.hasUnsupportedUsage(),
                              ErrorCode::INVALID_VALUE,
                              attributes.getUnsupportedReason());

        // Band domain (SdpaSemanticsRegister SEM-005): left_bound -1 or >= 1, right_bound >= -1.
        HIPDNN_CHECK_ERROR(detail::validateSdpaDiagonalBand(
            attributes.left_bound, attributes.right_bound, "SdpaBwdNode"));

        // The deprecated causal flags name different alignments; both set is invalid (SEM-002).
        HIPDNN_CHECK_ERROR(detail::validateSdpaDeprecatedCausalFlags(
            attributes.causal_mask, attributes.causal_mask_bottom_right, "SdpaBwdNode"));

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

            const auto batch = qDims[0];
            HIPDNN_CHECK_ERROR(
                detail::validateIndexTensor(seqLenQ, "SdpaBwdNode", "seq_len_q", batch, "batch"));
            HIPDNN_CHECK_ERROR(
                detail::validateIndexTensor(seqLenKv, "SdpaBwdNode", "seq_len_kv", batch, "batch"));
        }

        // Rule 10: attn_scale may be given as a tensor or a baked value, not both
        const auto scale = attributes.get_attn_scale();
        HIPDNN_RETURN_IF_TRUE(
            scale && attributes.attn_scale_value.has_value(),
            ErrorCode::INVALID_VALUE,
            "SdpaBwdNode: attn_scale tensor and attn_scale_value cannot both be set");

        // Rule 10b: Optional attention scale must be a scalar tensor
        if(scale)
        {
            HIPDNN_CHECK_ERROR(detail::validateScalarParameter(scale, "SCALE tensor"));
        }

        // Rule 11: a gradient has the shape of the tensor it differentiates. Dims and strides
        // left empty are inferred, so only caller-supplied ones are checked. As in cuDNN, the
        // head dimension of dQ, dK and dV must be contiguous.
        HIPDNN_CHECK_ERROR(validateGradientOutput(dq, qDims, "dQ", "Q", true));
        HIPDNN_CHECK_ERROR(validateGradientOutput(dk, kDims, "dK", "K", true));
        HIPDNN_CHECK_ERROR(validateGradientOutput(dv, vDims, "dV", "V", true));

        // Rule 12: dBias has the bias tensor's shape (SdpaSemanticsRegister SEM-014), so it
        // needs a bias to take that shape from. cuDNN's last-stride-1 check does not list
        // dBias, so it is not held to that rule.
        const auto dBias = attributes.get_dbias();
        if(dBias)
        {
            HIPDNN_RETURN_IF_FALSE(
                attnMask,
                ErrorCode::ATTRIBUTE_NOT_SET,
                std::string("SdpaBwdNode: dBias requires a Bias tensor, whose shape it takes"));
            HIPDNN_CHECK_ERROR(
                validateGradientOutput(dBias, attnMask->get_dim(), "dBias", "Bias", false));
            // An unset dtype on either side is inferred, so only two explicit ones can differ.
            const auto dBiasType = dBias->get_data_type();
            const auto biasType = attnMask->get_data_type();
            HIPDNN_RETURN_IF_TRUE(dBiasType != DataType::NOT_SET && biasType != DataType::NOT_SET
                                      && dBiasType != biasType,
                                  ErrorCode::INVALID_VALUE,
                                  std::string("SdpaBwdNode: dBias data type must equal the Bias "
                                              "data type"));
        }

        // Rule 13: as in cuDNN, the last dimension of each input is contiguous. Stats is held
        // to it only at rank 4 ([B, H, Sq, 1]): the rank-3 and ragged forms that hipDNN also
        // accepts are left unchecked.
        HIPDNN_CHECK_ERROR(validateLastStrideIsOne(q, "Q"));
        HIPDNN_CHECK_ERROR(validateLastStrideIsOne(k, "K"));
        HIPDNN_CHECK_ERROR(validateLastStrideIsOne(v, "V"));
        HIPDNN_CHECK_ERROR(validateLastStrideIsOne(o, "O"));
        HIPDNN_CHECK_ERROR(validateLastStrideIsOne(dOut, "dO"));
        if(attnMask)
        {
            HIPDNN_CHECK_ERROR(validateLastStrideIsOne(attnMask, "Bias"));
        }
        if(stats->get_dim().size() == reqRank)
        {
            HIPDNN_CHECK_ERROR(validateLastStrideIsOne(stats, "Stats"));
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
        const auto dBias = attributes.get_dbias();
        const auto bias = attributes.get_bias();

        // dBias takes the Bias dtype. This runs before fill_from_context(), which would give
        // an unset dBias dtype the graph IO type instead.
        if(dBias && bias && dBias->get_data_type() == DataType::NOT_SET
           && bias->get_data_type() != DataType::NOT_SET)
        {
            dBias->set_data_type(bias->get_data_type());
        }

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

        // dBias has same shape as Bias. It is never inferred from Q, K and V: a bias may
        // broadcast over batch, heads or queries, and only the bias tensor says which.
        if(dBias && bias)
        {
            if(dBias->get_dim().empty())
            {
                dBias->set_dim(bias->get_dim());
            }
            if(dBias->get_stride().empty())
            {
                dBias->set_stride(hipdnn_data_sdk::utilities::generateStrides(dBias->get_dim()));
            }
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

private:
    // Checks the caller-supplied dims and strides of a gradient output against the tensor it
    // differentiates. Empty dims or strides are skipped: infer_properties_node() fills them.
    static Error validateGradientOutput(const std::shared_ptr<TensorAttributes>& grad,
                                        const std::vector<int64_t>& sourceDims,
                                        const std::string& gradName,
                                        const std::string& sourceName,
                                        bool lastDimMustBeContiguous)
    {
        const auto& dims = grad->get_dim();
        if(!dims.empty())
        {
            HIPDNN_RETURN_IF_NE(dims.size(),
                                sourceDims.size(),
                                ErrorCode::INVALID_VALUE,
                                "SdpaBwdNode: " + gradName + " rank must equal " + sourceName
                                    + " rank (" + std::to_string(sourceDims.size()) + "), got "
                                    + std::to_string(dims.size()));
            for(size_t i = 0; i < dims.size(); ++i)
            {
                if(dims[i] != sourceDims[i])
                {
                    const auto index = std::to_string(i);
                    std::string message = "SdpaBwdNode: ";
                    message.append(gradName).append(" dim[").append(index);
                    message.append("] must equal ").append(sourceName);
                    message.append(" dim[").append(index).append("] (");
                    message.append(std::to_string(sourceDims[i])).append("), got ");
                    message.append(std::to_string(dims[i]));
                    return {ErrorCode::INVALID_VALUE, message};
                }
            }
        }

        const auto& strides = grad->get_stride();
        if(strides.empty())
        {
            return {};
        }
        HIPDNN_RETURN_IF_NE(strides.size(),
                            sourceDims.size(),
                            ErrorCode::INVALID_VALUE,
                            "SdpaBwdNode: " + gradName + " needs one stride per dimension ("
                                + std::to_string(sourceDims.size()) + "), got "
                                + std::to_string(strides.size()));
        for(size_t i = 0; i < strides.size(); ++i)
        {
            if(strides[i] <= 0)
            {
                std::string message = "SdpaBwdNode: ";
                message.append(gradName).append(" stride[").append(std::to_string(i));
                message.append("] must be positive, got ").append(std::to_string(strides[i]));
                return {ErrorCode::INVALID_VALUE, message};
            }
        }
        if(lastDimMustBeContiguous)
        {
            HIPDNN_CHECK_ERROR(validateLastStrideIsOne(grad, gradName));
        }
        return {};
    }

    // cuDNN's rule for the SDPA tensors: the last dimension is contiguous. A layout that
    // breaks it is legal but no kernel addresses it, hence cuDNN's code for the same check.
    // Empty strides are skipped, and no other stride is checked: a zero stride on another
    // dimension is a valid broadcast for an input.
    static Error validateLastStrideIsOne(const std::shared_ptr<TensorAttributes>& tensor,
                                         const std::string& tensorName)
    {
        const auto& strides = tensor->get_stride();
        HIPDNN_RETURN_IF_TRUE(!strides.empty() && strides.back() != 1,
                              ErrorCode::GRAPH_NOT_SUPPORTED,
                              "SdpaBwdNode: " + tensorName
                                  + " stride for the last dimension must be 1, got "
                                  + std::to_string(strides.back()));
        return {};
    }
};
} // namespace hipdnn_frontend::graph
