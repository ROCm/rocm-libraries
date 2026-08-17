// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "AsmSdpaConfigHelpers.hpp"
#include "../../../engines/asm_sdpa_engine/plans/SdpaPlanUtils.hpp"
#include "hip_kernel_provider_common/SdpaConfigConstants.hpp"
#include "hip_kernel_provider_common/SdpaConfigEnumerations.hpp"

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Graph.hpp>

namespace asm_sdpa_engine
{
using namespace hipdnn_frontend;
using namespace hip_kernel_provider_common;
using plan_utils::MaskType;

static DataType toDataType(const std::string& configDataType)
{
    std::unordered_map<std::string, DataType> typeMap = {{config::BFLOAT16, DataType::BFLOAT16},
                                                         {config::HALF, DataType::HALF},
                                                         {config::FLOAT, DataType::FLOAT}};

    auto it = typeMap.find(configDataType);
    if(it == typeMap.end())
    {
        throw std::runtime_error("Unsupported datatype name in config: " + configDataType);
    }
    return it->second;
}

std::string getConfigDescription(const fmha_v3_fwdConfig& config)
{
    std::string maskStr;
    switch(static_cast<MaskType>(config.mask))
    {
    case MaskType::NO_MASK:
        maskStr = "NoMask";
        break;
    case MaskType::TOP_LEFT_CAUSAL:
        maskStr = "TopLeftCausal";
        break;
    case MaskType::BOTTOM_RIGHT_CAUSAL:
        maskStr = "BottomRightCausal";
        break;
    case MaskType::SLIDING_WINDOW:
        maskStr = "SlidingWindow";
        break;
    default:
        maskStr = "UnknownMask";
        break;
    }

    const std::string modeStr
        = (static_cast<BatchMode>(config.mode) == BatchMode::GROUP) ? "Group" : "Batch";

    return config.arch + config.dtype + "HdimQ" + std::to_string(config.hdim_q) + "HdimV"
           + std::to_string(config.hdim_v) + maskStr + modeStr;
}

SdpaFwdTestCase::SdpaFwdTestCase(std::vector<int64_t> qDimsIn,
                                 std::vector<int64_t> vDimsIn,
                                 std::string archIn,
                                 int64_t leftBoundIn,
                                 int64_t rightBoundIn,
                                 bool topLeftAlignmentIn)
    : qDims(std::move(qDimsIn))
    , vDims(std::move(vDimsIn))
    , arch(std::move(archIn))
    , leftBound(leftBoundIn)
    , rightBound(rightBoundIn)
    , topLeftAlignment(topLeftAlignmentIn)
{
    // K tensor is [B, H_kv, S_kv, D_qk]: B and D_qk from Q, H_kv and S_kv from V
    kDims = {qDims[0], vDims[1], vDims[2], qDims[3]};
}

std::string SdpaFwdTestCase::getName(const testing::TestParamInfo<SdpaFwdTestCase>& info)
{
    const auto& tc = info.param;

    std::string maskStr;
    if(tc.leftBound < 0 && tc.rightBound < 0)
    {
        maskStr = "NoMask";
    }
    else if(tc.topLeftAlignment)
    {
        maskStr = "TopLeftCausal";
    }
    else
    {
        maskStr = "BottomRightCausal";
    }

    return tc.arch + "_B" + std::to_string(tc.qDims[0]) + "_Hq" + std::to_string(tc.qDims[1])
           + "_Hkv" + std::to_string(tc.vDims[1]) + "_Sq" + std::to_string(tc.qDims[2]) + "_Skv"
           + std::to_string(tc.vDims[2]) + "_Dqk" + std::to_string(tc.qDims[3]) + "_Dv"
           + std::to_string(tc.vDims[3]) + "_" + maskStr;
}

GraphTestCase configToTestCase(const fmha_v3_fwdConfig& config)
{
    return {config, getConfigDescription(config), config.arch};
}

std::shared_ptr<hipdnn_frontend::graph::Graph> buildSdpaFwdGraph(const GraphTestCase& testCase)

{
    using namespace hipdnn_frontend;
    using namespace hipdnn_frontend::graph;
    using namespace hipdnn_data_sdk::utilities;

    const fmha_v3_fwdConfig& config = testCase.config;

    const int64_t batch = testCase.batch;
    const int64_t numHeads = testCase.numHeads;
    const int64_t seqQ = testCase.seqQ;
    const int64_t seqKv = testCase.seqKv;

    // Determine data type
    const DataType dataType = toDataType(config.dtype);

    // Create tensor dimensions
    const std::vector<int64_t> qDims = {batch, numHeads, seqQ, config.hdim_q};
    const std::vector<int64_t> kDims = {batch, numHeads, seqKv, config.hdim_q};
    const std::vector<int64_t> vDims = {batch, numHeads, seqKv, config.hdim_v};

    auto graph = std::make_shared<Graph>();
    graph->set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto q = std::make_shared<TensorAttributes>();
    q->set_dim(qDims).set_stride(generateStrides(qDims)).set_data_type(dataType);

    auto k = std::make_shared<TensorAttributes>();
    k->set_dim(kDims).set_stride(generateStrides(kDims)).set_data_type(dataType);

    auto v = std::make_shared<TensorAttributes>();
    v->set_dim(vDims).set_stride(generateStrides(vDims)).set_data_type(dataType);

    // Configure SDPA attributes based on config
    SdpaAttributes attributes;
    attributes.set_name("SdpaFwdKernelConfigTest");

    if(testCase.attnScale.has_value())
    {
        attributes.set_attn_scale(testCase.attnScale.value());
    }

    // Configure mask type
    auto maskType = static_cast<MaskType>(config.mask);
    switch(maskType)
    {
    case MaskType::NO_MASK:
        // No mask - default behavior
        break;

    case MaskType::TOP_LEFT_CAUSAL:
        attributes.set_diagonal_band_left_bound(-1);
        attributes.set_diagonal_band_right_bound(0);
        attributes.set_diagonal_alignment(DiagonalAlignment::TOP_LEFT);
        attributes.set_causal_mask(true);
        break;

    case MaskType::BOTTOM_RIGHT_CAUSAL:
        attributes.set_diagonal_band_left_bound(-1);
        attributes.set_diagonal_band_right_bound(0);
        attributes.set_diagonal_alignment(DiagonalAlignment::BOTTOM_RIGHT);
        attributes.set_causal_mask_bottom_right(true);
        break;

    case MaskType::SLIDING_WINDOW:
        // Sliding window mask with arbitrary bounds
        attributes.set_diagonal_band_left_bound(64);
        attributes.set_diagonal_band_right_bound(64);
        break;

    default:
        break;
    }

    // Configure batch mode (GROUP requires sequence length tensors)
    auto batchMode = static_cast<BatchMode>(config.mode);
    if(batchMode == BatchMode::GROUP)
    {
        const std::vector<int64_t> seqLenDims = {batch};
        auto seqLenStrides = generateStrides(seqLenDims);

        auto seqLenQ = std::make_shared<TensorAttributes>();
        seqLenQ->set_dim(seqLenDims).set_stride(seqLenStrides).set_data_type(DataType::INT32);

        auto seqLenKv = std::make_shared<TensorAttributes>();
        seqLenKv->set_dim(seqLenDims).set_stride(seqLenStrides).set_data_type(DataType::INT32);

        attributes.set_seq_len_q(seqLenQ);
        attributes.set_seq_len_kv(seqLenKv);
    }

    auto [o, stats] = graph->sdpa(q, k, v, attributes);

    o->set_output(true);
    o->set_data_type(dataType);

    return graph;
}

} // namespace asm_sdpa_engine
