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
        HIPDNN_RETURN_IF_TRUE(attributes.hasUnsupportedUsage(),
                              ErrorCode::INVALID_VALUE,
                              attributes.getUnsupportedReason());

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
        // Paged K/V (cuDNN semantics): K and V are block containers
        // [num_blocks, H_kv, block_size, D], so their dim 0 is not the batch.
        const bool isPaged
            = attributes.get_page_table_k() != nullptr || attributes.get_page_table_v() != nullptr;
        if(!isPaged)
        {
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
        }
        // Rule 2: head_dim: Q[-1] == K[-1]; V[-1] is independent (may differ)
        const auto headDimQ = qDims[3];
        const auto headDimK = kDims[3];
        HIPDNN_RETURN_IF_NE(headDimQ,
                            headDimK,
                            ErrorCode::INVALID_VALUE,
                            "SdpaFwdNode: head_dim mismatch between Q and K: "
                                + std::to_string(headDimQ) + " vs " + std::to_string(headDimK));

        // Rule 3: seq_kv: K[-2] == V[-2]. For paged K/V this is the block size, so both
        // containers share one block size.
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

        // Rule 4b: variable sequence lengths. seq_len_q / seq_len_kv are INT32 [B, 1, 1, 1].
        HIPDNN_CHECK_ERROR(validateSeqLenTensor(attributes.get_seq_len_q(), "seq_len_q", qDims[0]));
        HIPDNN_CHECK_ERROR(
            validateSeqLenTensor(attributes.get_seq_len_kv(), "seq_len_kv", qDims[0]));

        // Rule 4c: paged K/V cache.
        HIPDNN_CHECK_ERROR(validatePagedKv(qDims, kDims, q->has_ragged_offset()));

        // Rule 4d: attention sink, one logit per query head.
        HIPDNN_CHECK_ERROR(validateSinkToken(qDims[1], q));

        // The KV extent an attention mask must cover: the paged bound, or K's sequence dim.
        const int64_t seqKvExtent
            = isPaged ? static_cast<int64_t>(attributes.max_seq_len_kv.value_or(0)) : kDims[2];

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
            const auto seqKv = seqKvExtent;
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

        // Rule 6: attn_scale may be given as a tensor or a baked value, not both
        const auto scale = attributes.get_attn_scale();
        HIPDNN_RETURN_IF_TRUE(
            scale && attributes.attn_scale_value.has_value(),
            ErrorCode::INVALID_VALUE,
            "SdpaFwdNode: attn_scale tensor and attn_scale_value cannot both be set");

        // Rule 6b: Optional scale must be a scalar tensor (volume == 1)
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

private:
    // The data type a tensor will have once fill_from_context ran: its own, or the graph's
    // io_data_type when unset.
    DataType effectiveDataType(const std::shared_ptr<TensorAttributes>& tensor) const
    {
        const auto dataType = tensor->get_data_type();
        return dataType == DataType::NOT_SET ? graph_attributes.get_io_data_type() : dataType;
    }

    // sink_token, when present: [1, H_q, 1, 1] with data type FLOAT (cuDNN) or the Q data type
    // when that is HALF or BFLOAT16 (hipDNN extension: vLLM keeps sinks in the model dtype).
    Error validateSinkToken(int64_t numHeadsQ, const std::shared_ptr<TensorAttributes>& q) const
    {
        const auto sink = attributes.get_sink_token();
        if(!sink)
        {
            return {};
        }
        const auto& dims = sink->get_dim();
        HIPDNN_RETURN_IF_TRUE(dims.size() != 4 || dims[0] != 1 || dims[1] != numHeadsQ
                                  || dims[2] != 1 || dims[3] != 1,
                              ErrorCode::INVALID_VALUE,
                              "SdpaFwdNode: sink_token must have dims [1, H_q, 1, 1] with H_q="
                                  + std::to_string(numHeadsQ));
        const auto sinkType = effectiveDataType(sink);
        const auto qDataType = effectiveDataType(q);
        const bool sinkMatchesHalfQ
            = sinkType == qDataType
              && (qDataType == DataType::HALF || qDataType == DataType::BFLOAT16);
        HIPDNN_RETURN_IF_TRUE(sinkType != DataType::FLOAT && !sinkMatchesHalfQ,
                              ErrorCode::INVALID_VALUE,
                              "SdpaFwdNode: sink_token data type must be FLOAT, or HALF/BFLOAT16 "
                              "matching Q");
        return {};
    }

    // seq_len_q / seq_len_kv, when present: INT32 [B, 1, 1, 1] (cuDNN).
    static Error validateSeqLenTensor(const std::shared_ptr<TensorAttributes>& seqLen,
                                      const std::string& name,
                                      int64_t batch)
    {
        if(!seqLen)
        {
            return {};
        }
        const auto& dims = seqLen->get_dim();
        HIPDNN_RETURN_IF_TRUE(dims.size() != 4 || dims[0] != batch || dims[1] != 1 || dims[2] != 1
                                  || dims[3] != 1,
                              ErrorCode::INVALID_VALUE,
                              "SdpaFwdNode: " + name
                                  + " must have dims [B, 1, 1, 1] with B=" + std::to_string(batch));
        HIPDNN_RETURN_IF_TRUE(seqLen->get_data_type() != DataType::INT32,
                              ErrorCode::INVALID_VALUE,
                              "SdpaFwdNode: " + name + " must have INT32 data type");
        return {};
    }

    // One page table: INT32 [B, 1, max_blocks_per_seq, 1], not packed (ragged), and exactly as
    // wide as cuDNN requires: max_blocks_per_seq == ceil(max_seq_len_kv / block_size).
    static Error validatePageTable(const std::shared_ptr<TensorAttributes>& pageTable,
                                   const std::string& name,
                                   int64_t batch,
                                   int64_t blockSize,
                                   int64_t maxSeqLenKv)
    {
        const auto& dims = pageTable->get_dim();
        HIPDNN_RETURN_IF_NE(dims.size(),
                            static_cast<size_t>(4),
                            ErrorCode::INVALID_VALUE,
                            "SdpaFwdNode: " + name
                                + " must be rank-4 [B, 1, max_blocks_per_seq, 1], got rank="
                                + std::to_string(dims.size()));
        HIPDNN_RETURN_IF_TRUE(dims[0] != batch || dims[1] != 1 || dims[2] <= 0 || dims[3] != 1,
                              ErrorCode::INVALID_VALUE,
                              "SdpaFwdNode: " + name
                                  + " must have dims [B, 1, max_blocks_per_seq, 1] with B="
                                  + std::to_string(batch) + " and max_blocks_per_seq >= 1");
        HIPDNN_RETURN_IF_TRUE(pageTable->get_data_type() != DataType::INT32,
                              ErrorCode::INVALID_VALUE,
                              "SdpaFwdNode: " + name + " must have INT32 data type");
        HIPDNN_RETURN_IF_TRUE(pageTable->has_ragged_offset(),
                              ErrorCode::INVALID_VALUE,
                              "SdpaFwdNode: packed (ragged) " + name + " is not supported");
        const auto expectedBlocks = (maxSeqLenKv + blockSize - 1) / blockSize;
        HIPDNN_RETURN_IF_NE(dims[2],
                            expectedBlocks,
                            ErrorCode::INVALID_VALUE,
                            "SdpaFwdNode: " + name
                                + " max_blocks_per_seq must equal ceil(max_seq_len_kv / "
                                  "block_size) = "
                                + std::to_string(expectedBlocks) + ", got "
                                + std::to_string(dims[2]));
        return {};
    }

    // Paged K/V rules. Unlike cuDNN, hipDNN needs both tables (or neither) and an explicit
    // max_seq_len_kv, and allows dense Q without seq_len_q.
    Error validatePagedKv(const std::vector<int64_t>& qDims,
                          const std::vector<int64_t>& kDims,
                          bool raggedQ) const
    {
        const auto pageTableK = attributes.get_page_table_k();
        const auto pageTableV = attributes.get_page_table_v();
        if(!pageTableK && !pageTableV)
        {
            HIPDNN_RETURN_IF_TRUE(attributes.max_seq_len_kv.has_value(),
                                  ErrorCode::INVALID_VALUE,
                                  "SdpaFwdNode: max_seq_len_kv is only valid with paged K/V "
                                  "(page_table_k / page_table_v)");
            return {};
        }

        HIPDNN_RETURN_IF_TRUE(!pageTableK || !pageTableV,
                              ErrorCode::INVALID_VALUE,
                              "SdpaFwdNode: paged K/V needs both page_table_k and page_table_v");
        HIPDNN_RETURN_IF_FALSE(attributes.get_seq_len_kv(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               std::string("SdpaFwdNode: paged K/V requires seq_len_kv"));
        HIPDNN_RETURN_IF_TRUE(raggedQ && !attributes.get_seq_len_q(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "SdpaFwdNode: paged K/V with ragged Q requires seq_len_q");
        HIPDNN_RETURN_IF_FALSE(attributes.max_seq_len_kv.has_value(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               std::string("SdpaFwdNode: paged K/V requires max_seq_len_kv"));
        const auto maxSeqLenKv = static_cast<int64_t>(attributes.max_seq_len_kv.value());
        HIPDNN_RETURN_IF_TRUE(maxSeqLenKv <= 0,
                              ErrorCode::INVALID_VALUE,
                              "SdpaFwdNode: max_seq_len_kv must be positive, got "
                                  + std::to_string(maxSeqLenKv));

        const auto blockSize = kDims[2];
        HIPDNN_RETURN_IF_TRUE(kDims[0] <= 0 || blockSize <= 0,
                              ErrorCode::INVALID_VALUE,
                              "SdpaFwdNode: paged K/V containers [num_blocks, H_kv, block_size, D] "
                              "need positive num_blocks and block_size");
        HIPDNN_CHECK_ERROR(
            validatePageTable(pageTableK, "page_table_k", qDims[0], blockSize, maxSeqLenKv));
        HIPDNN_CHECK_ERROR(
            validatePageTable(pageTableV, "page_table_v", qDims[0], blockSize, maxSeqLenKv));
        return {};
    }

public:
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
