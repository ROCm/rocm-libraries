// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <flatbuffers/flatbuffer_builder.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

// A configurable single-node SDPA-forward op-graph builder for unit tests. It
// constructs a real flatbuffer graph (tensors [B, H, S, D] + an SdpaAttributes
// node) that SdpaGraphAdapter::translate() can decode. Defaults mirror the
// PR #8866 fp16 / BSHD / mask-none / no-dropout / default-scale smoke instance.
namespace rocke_client::dispatcher::test
{

struct SdpaGraphConfig
{
    hipdnn_flatbuffers_sdk::data_objects::DataType dtype
        = hipdnn_flatbuffers_sdk::data_objects::DataType::HALF;
    std::int64_t batch = 2;
    std::int64_t numQueryHeads = 4;
    std::int64_t numKvHeads = 4;
    std::int64_t seqlenQ = 64;
    std::int64_t seqlenK = 64;
    std::int64_t headSizeQK = 64;
    std::int64_t headSizeV = 64;
    bool bshd = true; // true => BSHD physical strides, false => BHSD
    bool causalMask = false;
    bool causalMaskBottomRight = false;
    bool paddingMask = false;
    bool alibiMask = false;
    std::optional<float> dropoutProbability;
    std::optional<float> attnScaleValue;
    bool includeScaleTensor = false;
    std::optional<std::int64_t> leftBound;
    std::optional<std::int64_t> rightBound;
    hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment diagonalAlignment
        = hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment::TOP_LEFT;
    bool sdpaNode = true; // false => emit a non-SDPA (pointwise) node instead
    int nodeCount = 1; // >1 exercises the single-node gate
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT; // node accumulation dtype (fp32)
    hipdnn_flatbuffers_sdk::data_objects::DataType mmaCoreMode
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET; // optional SDPA MMA-core override
    hipdnn_flatbuffers_sdk::data_objects::AttentionImplementation implementation
        = hipdnn_flatbuffers_sdk::data_objects::AttentionImplementation::AUTO; // exec strategy
    // Adapter structural-gate knobs (exercise translate()'s decline branches).
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> keyDtype; // mixed-dtype reject
    bool queryRank3 = false; // emit Q as rank-3 (non-rank-4 reject)
    bool omitOutputTensor = false; // drop O from the tensor map (missing-uid reject)
    bool nonContiguousStrides = false; // pad Q strides so layout is OTHER
    bool overrideShapeEnabled = false; // graph opts into execute-time override shapes
    // Cross-tensor consistency knobs (exercise the per-tensor layout/dim gates).
    bool mismatchKeyLayout = false; // K packed with the opposite layout of Q
    std::optional<std::int64_t> mismatchBatch; // K batch (dim0) differs from Q
    std::optional<std::int64_t> mismatchVSeqlen; // V seqlen (dim2) differs from K
    std::optional<std::int64_t> mismatchOHeads; // O heads (dim1) differ from Q
    // Unsupported-feature knobs: set a UID/flag to exercise translate()'s
    // allowlist decline branches. A UID need not reference a real tensor; the
    // adapter declines on the attribute's mere presence.
    std::optional<std::int64_t> attnMaskTensorUid;
    std::optional<std::int64_t> pageTableKTensorUid;
    std::optional<std::int64_t> seqLenQTensorUid;
    std::optional<std::int64_t> blockMaskTensorUid;
    std::optional<std::int64_t> sinkTokenTensorUid;
    std::optional<std::int64_t> statsTensorUid;
    std::optional<std::int64_t> descaleQTensorUid;
    std::optional<bool> generateStats;
    std::optional<std::int32_t> maxSeqLenKv;
};

struct SdpaGraphFixture
{
    std::shared_ptr<flatbuffers::DetachedBuffer> buffer;

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper() const
    {
        return hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(buffer->data(),
                                                                          buffer->size());
    }
};

namespace detail
{

// Contiguous strides for dims [B, H, S, D] under the requested physical layout.
inline std::vector<std::int64_t>
    contiguousStrides(std::int64_t h, std::int64_t s, std::int64_t d, bool bshd)
{
    if(bshd)
    {
        // physical B, S, H, D
        return {s * h * d, d, h * d, 1};
    }
    // physical B, H, S, D (row-major over the dims order)
    return {h * s * d, s * d, d, 1};
}

template <typename T>
flatbuffers::Optional<T> opt(const std::optional<T>& value)
{
    // flatbuffers::Optional is a distinct type from std::optional; the round-trip
    // warning is a false positive here.
    // NOLINTNEXTLINE(bugprone-optional-value-conversion)
    return value.has_value() ? flatbuffers::Optional<T>(*value) : flatbuffers::nullopt;
}

} // namespace detail

inline SdpaGraphFixture buildSdpaGraph(const SdpaGraphConfig& config)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;

    std::vector<std::int64_t> qDims
        = {config.batch, config.numQueryHeads, config.seqlenQ, config.headSizeQK};
    std::vector<std::int64_t> qStrides = detail::contiguousStrides(
        config.numQueryHeads, config.seqlenQ, config.headSizeQK, config.bshd);
    if(config.queryRank3)
    {
        // Drop the head-size dim so Q is not rank-4.
        qDims = {config.batch, config.numQueryHeads, config.seqlenQ};
        qStrides = {config.numQueryHeads * config.seqlenQ, config.seqlenQ, 1};
    }
    else if(config.nonContiguousStrides)
    {
        // Pad the batch stride so the packing matches neither BSHD nor BHSD.
        qStrides[0] *= 2;
    }

    const std::int64_t kBatch = config.mismatchBatch.value_or(config.batch);
    const std::int64_t vSeqlen = config.mismatchVSeqlen.value_or(config.seqlenK);
    const std::int64_t oHeads = config.mismatchOHeads.value_or(config.numQueryHeads);
    const bool keyBshd = config.mismatchKeyLayout ? !config.bshd : config.bshd;

    const std::vector<std::int64_t> kDims
        = {kBatch, config.numKvHeads, config.seqlenK, config.headSizeQK};
    const std::vector<std::int64_t> vDims
        = {config.batch, config.numKvHeads, vSeqlen, config.headSizeV};
    const std::vector<std::int64_t> oDims
        = {config.batch, oHeads, config.seqlenQ, config.headSizeV};

    const std::vector<std::int64_t> kStrides
        = detail::contiguousStrides(config.numKvHeads, config.seqlenK, config.headSizeQK, keyBshd);
    const std::vector<std::int64_t> vStrides
        = detail::contiguousStrides(config.numKvHeads, vSeqlen, config.headSizeV, config.bshd);
    const std::vector<std::int64_t> oStrides
        = detail::contiguousStrides(oHeads, config.seqlenQ, config.headSizeV, config.bshd);

    // K may use a distinct element type to exercise the mixed-dtype gate.
    const DataType keyDtype = config.keyDtype.value_or(config.dtype);

    std::int64_t uid = 1;
    const std::int64_t qUid = uid++;
    tensors.push_back(
        CreateTensorAttributesDirect(builder, qUid, "q", config.dtype, &qStrides, &qDims));
    const std::int64_t kUid = uid++;
    tensors.push_back(
        CreateTensorAttributesDirect(builder, kUid, "k", keyDtype, &kStrides, &kDims));
    const std::int64_t vUid = uid++;
    tensors.push_back(
        CreateTensorAttributesDirect(builder, vUid, "v", config.dtype, &vStrides, &vDims));
    const std::int64_t oUid = uid++;
    if(!config.omitOutputTensor)
    {
        tensors.push_back(
            CreateTensorAttributesDirect(builder, oUid, "o", config.dtype, &oStrides, &oDims));
    }

    flatbuffers::Optional<std::int64_t> scaleUid = flatbuffers::nullopt;
    if(config.includeScaleTensor)
    {
        const std::vector<std::int64_t> scaleDims = {1};
        const std::int64_t scaleTensorUid = uid++;
        tensors.push_back(CreateTensorAttributesDirect(
            builder, scaleTensorUid, "scale", DataType::FLOAT, &scaleDims, &scaleDims));
        scaleUid = flatbuffers::Optional<std::int64_t>(scaleTensorUid);
    }

    const auto sdpaAttributes
        = CreateSdpaAttributes(builder,
                               qUid,
                               kUid,
                               vUid,
                               oUid,
                               detail::opt(config.attnMaskTensorUid), // attn_mask_tensor_uid
                               scaleUid,
                               detail::opt(config.seqLenQTensorUid), // seq_len_q_tensor_uid
                               flatbuffers::nullopt, // seq_len_kv_tensor_uid
                               flatbuffers::nullopt, // seed_tensor_uid
                               flatbuffers::nullopt, // offset_tensor_uid
                               flatbuffers::nullopt, // dropout_mask_tensor_uid
                               flatbuffers::nullopt, // dropout_scale_tensor_uid
                               detail::opt(config.pageTableKTensorUid), // page_table_k_tensor_uid
                               flatbuffers::nullopt, // page_table_v_tensor_uid
                               detail::opt(config.blockMaskTensorUid), // block_mask_tensor_uid
                               detail::opt(config.sinkTokenTensorUid), // sink_token_tensor_uid
                               detail::opt(config.descaleQTensorUid), // descale_q_tensor_uid
                               flatbuffers::nullopt, // descale_k_tensor_uid
                               flatbuffers::nullopt, // descale_v_tensor_uid
                               flatbuffers::nullopt, // descale_s_tensor_uid
                               flatbuffers::nullopt, // scale_s_tensor_uid
                               flatbuffers::nullopt, // scale_o_tensor_uid
                               detail::opt(config.statsTensorUid), // stats_tensor_uid
                               flatbuffers::nullopt, // max_tensor_uid
                               flatbuffers::nullopt, // sum_exp_tensor_uid
                               flatbuffers::nullopt, // rng_dump_tensor_uid
                               flatbuffers::nullopt, // amax_s_tensor_uid
                               flatbuffers::nullopt, // amax_o_tensor_uid
                               detail::opt(config.generateStats), // generate_stats
                               config.alibiMask,
                               config.paddingMask,
                               config.causalMask,
                               config.causalMaskBottomRight,
                               detail::opt(config.dropoutProbability),
                               detail::opt(config.attnScaleValue),
                               detail::opt(config.leftBound),
                               detail::opt(config.rightBound),
                               detail::opt(config.maxSeqLenKv), // max_seq_len_kv
                               config.diagonalAlignment,
                               config.mmaCoreMode, // mma_core_mode
                               config.implementation);

    std::vector<flatbuffers::Offset<Node>> nodes;
    for(int i = 0; i < config.nodeCount; ++i)
    {
        if(config.sdpaNode)
        {
            nodes.push_back(CreateNodeDirect(builder,
                                             "sdpa_fwd",
                                             config.computeDataType,
                                             NodeAttributes::SdpaAttributes,
                                             sdpaAttributes.Union()));
        }
        else
        {
            nodes.push_back(
                CreateNodeDirect(builder, "not_sdpa", config.dtype, NodeAttributes::NONE, 0));
        }
    }

    const auto graphOffset = CreateGraphDirect(builder,
                                               "test",
                                               DataType::FLOAT,
                                               DataType::HALF,
                                               DataType::BFLOAT16,
                                               &tensors,
                                               &nodes,
                                               flatbuffers::nullopt,
                                               config.overrideShapeEnabled);
    builder.Finish(graphOffset);

    SdpaGraphFixture fixture;
    fixture.buffer = std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
    return fixture;
}

} // namespace rocke_client::dispatcher::test
