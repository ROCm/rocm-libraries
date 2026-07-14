// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ConfigHelpers.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaPlanUtils.hpp"
#include "hip_kernel_provider_common/SdpaConfigConstants.hpp"
#include "hip_kernel_provider_common/SdpaConfigEnumerations.hpp"

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>

namespace asm_sdpa_engine
{

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hip_kernel_provider_common;
using plan_utils::MaskType;

static DataType toDataType(const std::string& configDataType)
{
    const std::unordered_map<std::string, DataType> typeMap
        = {{config::BFLOAT16, DataType::BFLOAT16},
           {config::HALF, DataType::HALF},
           {config::FLOAT, DataType::FLOAT}};

    const auto it = typeMap.find(configDataType);
    if(it == typeMap.end())
    {
        throw std::runtime_error("Unsupported datatype name in config: " + configDataType);
    }
    return it->second;
}

flatbuffers::FlatBufferBuilder configToCompatibleGraph(const fmha_v3_fwdConfig& config)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<TensorAttributes>> tensorAttributes;

    // Map dtype string to DataType enums. FP8 configs use FP8_E4M3 inputs with a
    // BFLOAT16 output and require q/k/v descale tensors (mirrors the engine's fp8
    // applicability contract); every other config is single-dtype in and out.
    const bool isFp8 = (config.dtype == config::FP8BF16);
    const DataType inDataType = isFp8 ? DataType::FP8_E4M3_FNUZ : toDataType(config.dtype);
    const DataType outDataType = isFp8 ? DataType::BFLOAT16 : inDataType;

    // Use arbitrary values for batch, num_heads, and sequence lengths
    const int64_t batch = 2;
    const int64_t numHeads = 4;
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
        CreateTensorAttributesDirect(builder, qUid, "q", inDataType, &qStrides, &qDims));

    // K tensor
    const auto kUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, kUid, "k", inDataType, &kStrides, &kDims));

    // V tensor
    const auto vUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, vUid, "v", inDataType, &vStrides, &vDims));

    // O tensor
    const auto oUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, oUid, "o", outDataType, &oStrides, &oDims));

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

    // FP8 inputs require per-tensor (scalar) q/k/v descale tensors, else the engine
    // declines the graph (mirrors AITER's TORCH_CHECK).
    flatbuffers::Optional<int64_t> descaleQUid = flatbuffers::nullopt;
    flatbuffers::Optional<int64_t> descaleKUid = flatbuffers::nullopt;
    flatbuffers::Optional<int64_t> descaleVUid = flatbuffers::nullopt;
    if(isFp8)
    {
        const std::vector<int64_t> descaleDims = {1, 1, 1, 1};
        const std::vector<int64_t> descaleStrides
            = hipdnn_data_sdk::utilities::generateStrides(descaleDims);

        const auto descaleQTensorUid = uid++;
        tensorAttributes.push_back(CreateTensorAttributesDirect(builder,
                                                                descaleQTensorUid,
                                                                "descale_q",
                                                                DataType::FLOAT,
                                                                &descaleStrides,
                                                                &descaleDims));
        descaleQUid = flatbuffers::Optional<int64_t>(descaleQTensorUid);

        const auto descaleKTensorUid = uid++;
        tensorAttributes.push_back(CreateTensorAttributesDirect(builder,
                                                                descaleKTensorUid,
                                                                "descale_k",
                                                                DataType::FLOAT,
                                                                &descaleStrides,
                                                                &descaleDims));
        descaleKUid = flatbuffers::Optional<int64_t>(descaleKTensorUid);

        const auto descaleVTensorUid = uid++;
        tensorAttributes.push_back(CreateTensorAttributesDirect(builder,
                                                                descaleVTensorUid,
                                                                "descale_v",
                                                                DataType::FLOAT,
                                                                &descaleStrides,
                                                                &descaleDims));
        descaleVUid = flatbuffers::Optional<int64_t>(descaleVTensorUid);
    }

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

    switch(static_cast<MaskType>(config.mask))
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

    case MaskType::SLIDING_WINDOW:
        // Sliding window mask - use arbitrary window bounds
        leftBound = flatbuffers::Optional<int64_t>(64); // Arbitrary left window bound
        rightBound = flatbuffers::Optional<int64_t>(64); // Arbitrary right window bound
        break;

    default:
        // Unknown mask type - treat as no mask
        break;
    }

    // Create SDPA attributes with all the configured options
    const auto sdpaAttributes
        = CreateSdpaAttributes(builder,
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
                               descaleQUid, // descale_q_tensor_uid
                               descaleKUid, // descale_k_tensor_uid
                               descaleVUid, // descale_v_tensor_uid
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
                               DataType::UNSET, // mma_core_mode
                               AttentionImplementation::AUTO);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "sdpa_fwd",
                                     DataType::FLOAT,
                                     NodeAttributes::SdpaAttributes,
                                     sdpaAttributes.Union()));

    const auto graphOffset = CreateGraphDirect(builder,
                                               "test",
                                               DataType::FLOAT,
                                               DataType::HALF,
                                               DataType::BFLOAT16,
                                               &tensorAttributes,
                                               &nodes);
    builder.Finish(graphOffset);
    return builder;
}

} // namespace asm_sdpa_engine
