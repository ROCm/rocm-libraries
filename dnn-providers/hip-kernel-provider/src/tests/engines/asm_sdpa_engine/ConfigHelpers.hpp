// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "GraphTest.hpp"
#include "asm_fmha_v3_fwd_configs.hpp"

#include <flatbuffers/flatbuffer_builder.h>
#include <hip_kernel_provider_common/SdpaConfigEnumerations.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <string>
#include <vector>

namespace asm_sdpa_engine
{

inline std::vector<fmha_v3_fwdConfig> getConfigsForArch(const std::string& archId)
{
    std::vector<fmha_v3_fwdConfig> configs;
    for(const auto& [key, config] : cfg_fmha_fwd)
    {
        if(config.arch == archId)
        {
            configs.push_back(config);
        }
    }

    return configs;
}

/**
 * @brief Converts an SDPA forward config to a valid graph builder.
 *
 * Creates a graph with dimensions matching the config's hdim_q and hdim_v,
 * using arbitrary values for batch, num_heads, seq_q, and seq_kv.
 * Supports all MaskType values and BATCH/GROUP modes.
 *
 * @param config The fmha_v3_fwdConfig containing kernel configuration
 * @return flatbuffers::FlatBufferBuilder A builder containing the graph
 */
inline flatbuffers::FlatBufferBuilder configToCompatibleGraph(const fmha_v3_fwdConfig& config)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<TensorAttributes>> tensorAttributes;

    // Map dtype string to DataType enum
    DataType dataType = DataType::BFLOAT16;
    if(config.dtype == "fp16")
    {
        dataType = DataType::HALF;
    }
    else if(config.dtype == "fp32")
    {
        dataType = DataType::FLOAT;
    }
    // Default to BFLOAT16 for "bf16" or any other value

    // Use arbitrary values for batch, num_heads, and sequence lengths
    const int64_t batch = 4;
    const int64_t numHeads = 8;
    const int64_t seqQ = 256;
    const int64_t seqKv = 256;

    // Define tensor dimensions
    const std::vector<int64_t> qDims = {batch, numHeads, seqQ, config.hdim_q};
    const std::vector<int64_t> qStrides = hipdnn_data_sdk::utilities::generateStrides(qDims);

    const std::vector<int64_t> kDims = {batch, numHeads, seqKv, config.hdim_q};
    const std::vector<int64_t> kStrides = hipdnn_data_sdk::utilities::generateStrides(kDims);

    const std::vector<int64_t> vDims = {batch, numHeads, seqKv, config.hdim_v};
    const std::vector<int64_t> vStrides = hipdnn_data_sdk::utilities::generateStrides(vDims);

    const std::vector<int64_t> oDims = {batch, numHeads, seqQ, config.hdim_v};
    const std::vector<int64_t> oStrides = hipdnn_data_sdk::utilities::generateStrides(oDims);

    int64_t uid = 1;

    // Q tensor
    const auto qUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, qUid, "q", dataType, &qStrides, &qDims));

    // K tensor
    const auto kUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, kUid, "k", dataType, &kStrides, &kDims));

    // V tensor
    const auto vUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, vUid, "v", dataType, &vStrides, &vDims));

    // O tensor
    const auto oUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, oUid, "o", dataType, &oStrides, &oDims));

    // Scale tensor (always include for SDPA)
    const std::vector<int64_t> scaleDims = {1};
    const Float32Value scaleVal(1.0f);
    const auto scaleUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder,
                                     scaleUid,
                                     "scale",
                                     DataType::FLOAT,
                                     &scaleDims,
                                     &scaleDims,
                                     false,
                                     TensorValue::Float32Value,
                                     builder.CreateStruct(scaleVal).Union()));

    // Handle GROUP mode - add sequence length tensors
    flatbuffers::Optional<int64_t> seqLenQUid = flatbuffers::nullopt;
    flatbuffers::Optional<int64_t> seqLenKvUid = flatbuffers::nullopt;

    if(config.mode == BatchMode::GROUP)
    {
        // Sequence length tensors: shape [batch]
        const std::vector<int64_t> seqLenDims = {batch};
        const std::vector<int64_t> seqLenStrides = {1};

        const auto seqLenQTensorUid = uid++;
        tensorAttributes.push_back(CreateTensorAttributesDirect(
            builder, seqLenQTensorUid, "seq_len_q", DataType::INT32, &seqLenStrides, &seqLenDims));
        seqLenQUid = flatbuffers::Optional<int64_t>(seqLenQTensorUid);

        const auto seqLenKvTensorUid = uid++;
        tensorAttributes.push_back(CreateTensorAttributesDirect(builder,
                                                                seqLenKvTensorUid,
                                                                "seq_len_kv",
                                                                DataType::INT32,
                                                                &seqLenStrides,
                                                                &seqLenDims));
        seqLenKvUid = flatbuffers::Optional<int64_t>(seqLenKvTensorUid);
    }

    // Determine mask-related attributes based on MaskType
    flatbuffers::Optional<int64_t> leftBound = flatbuffers::nullopt;
    flatbuffers::Optional<int64_t> rightBound = flatbuffers::nullopt;
    DiagonalAlignment diagAlignment = DiagonalAlignment::TOP_LEFT;
    bool causalMask = false;
    bool causalMaskBottomRight = false;

    switch(config.mask)
    {
    case MaskType::NO_MASK:
        // No mask - leave all defaults
        break;

    case MaskType::TOP_LEFT_CAUSAL:
        // Causal mask with top-left alignment
        leftBound = flatbuffers::Optional<int64_t>(-1);
        rightBound = flatbuffers::Optional<int64_t>(0);
        diagAlignment = DiagonalAlignment::TOP_LEFT;
        causalMask = true;
        break;

    case MaskType::BOTTOM_RIGHT_CAUSAL:
        // Causal mask with bottom-right alignment
        leftBound = flatbuffers::Optional<int64_t>(-1);
        rightBound = flatbuffers::Optional<int64_t>(0);
        diagAlignment = DiagonalAlignment::BOTTOM_RIGHT;
        causalMaskBottomRight = true;
        break;

    case MaskType::WINDOW_GENERIC:
        // Sliding window mask - use arbitrary window bounds
        leftBound = flatbuffers::Optional<int64_t>(64); // Arbitrary left window bound
        rightBound = flatbuffers::Optional<int64_t>(64); // Arbitrary right window bound
        break;

    default:
        // Unknown mask type - treat as no mask
        break;
    }

    // Create SDPA attributes with all the configured options
    auto sdpaAttributes = CreateSdpaAttributes(builder,
                                               qUid,
                                               kUid,
                                               vUid,
                                               oUid,
                                               flatbuffers::nullopt, // attn_mask_tensor_uid
                                               scaleUid,
                                               seqLenQUid,
                                               seqLenKvUid,
                                               flatbuffers::nullopt, // seed_tensor_uid
                                               flatbuffers::nullopt, // offset_tensor_uid
                                               flatbuffers::nullopt, // dropout_mask_tensor_uid
                                               flatbuffers::nullopt, // dropout_scale_tensor_uid
                                               flatbuffers::nullopt, // page_table_k_tensor_uid
                                               flatbuffers::nullopt, // page_table_v_tensor_uid
                                               flatbuffers::nullopt, // block_mask_tensor_uid
                                               flatbuffers::nullopt, // sink_token_tensor_uid
                                               flatbuffers::nullopt, // descale_q_tensor_uid
                                               flatbuffers::nullopt, // descale_k_tensor_uid
                                               flatbuffers::nullopt, // descale_v_tensor_uid
                                               flatbuffers::nullopt, // descale_s_tensor_uid
                                               flatbuffers::nullopt, // scale_s_tensor_uid
                                               flatbuffers::nullopt, // scale_o_tensor_uid
                                               flatbuffers::nullopt, // stats_tensor_uid
                                               flatbuffers::nullopt, // max_tensor_uid
                                               flatbuffers::nullopt, // sum_exp_tensor_uid
                                               flatbuffers::nullopt, // rng_dump_tensor_uid
                                               flatbuffers::nullopt, // amax_s_tensor_uid
                                               flatbuffers::nullopt, // amax_o_tensor_uid
                                               flatbuffers::nullopt, // generate_stats
                                               false, // alibi_mask
                                               false, // padding_mask
                                               causalMask,
                                               causalMaskBottomRight,
                                               flatbuffers::nullopt, // dropout_probability
                                               flatbuffers::nullopt, // attn_scale_value
                                               leftBound,
                                               rightBound,
                                               flatbuffers::nullopt, // max_seq_len_kv
                                               diagAlignment,
                                               DataType::FLOAT, // mma_core_mode
                                               AttentionImplementation::AUTO);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(
        builder, "sdpa_fwd", dataType, NodeAttributes::SdpaAttributes, sdpaAttributes.Union()));

    auto graphOffset = CreateGraphDirect(builder,
                                         "test",
                                         DataType::FLOAT,
                                         DataType::HALF,
                                         DataType::BFLOAT16,
                                         &tensorAttributes,
                                         &nodes);
    builder.Finish(graphOffset);
    return builder;
}

inline auto getCompatibleGraphsForArch(const std::string& archId)
{
    std::vector<GraphTest> graphs;
    for(const auto& config : getConfigsForArch(archId))
    {
        graphs.emplace_back(configToCompatibleGraph(config), config.co_name);
    }

    return graphs;
}

} // namespace asm_sdpa_engine
