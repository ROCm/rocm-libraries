// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "AsmSdpaConfigHelpers.hpp"
#include "../../../engines/asm_sdpa_engine/asm/AsmKernelPath.hpp"
#include "../../../engines/asm_sdpa_engine/plans/SdpaPlanUtils.hpp"
#include "AsmSdpaTestSupport.hpp"
#include "hip_kernel_provider_common/SdpaConfigConstants.hpp"
#include "hip_kernel_provider_common/SdpaConfigEnumerations.hpp"

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Graph.hpp>

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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

GraphTestCase configToTestCase(const fmha_v3_fwdConfig& config, bool withStats)
{
    auto tc = GraphTestCase{config, getConfigDescription(config), config.arch};
    // Registry cases use the standard attention scale 1/sqrt(hdim_q), as AITER's own tests
    // do. The gfx950 hd256 kernels multiply Q by scale * log2(e) and round the product to
    // bf16 before QK^T, so their error grows with the scale: at scale 1.0 a few outputs
    // exceed the bf16 tolerance, while at 1/sqrt(256) the error stays well inside it.
    tc.attnScale = 1.0f / std::sqrt(static_cast<float>(config.hdim_q));
    // Group mode gives Q and K/V a token capacity of 512, twice the largest ts_qo of any
    // group row, so the per-batch lengths a test writes into the offset tables can span
    // two Q-tiles of every row on either side.
    if(config.mode == BatchMode::GROUP)
    {
        constexpr int64_t K_GROUP_SEQ_CAPACITY = 512;
        tc.seqQ = K_GROUP_SEQ_CAPACITY;
        tc.seqKv = K_GROUP_SEQ_CAPACITY;
    }
    if(withStats)
    {
        tc.withStats = true;
        tc.name += "Stats";
    }
    return tc;
}

// Fixed tensor uids, so buildSdpaFwdReferenceGraph's graph addresses the same tensors as
// buildSdpaFwdGraph's.
namespace
{
constexpr int64_t K_Q_UID = 1;
constexpr int64_t K_K_UID = 2;
constexpr int64_t K_V_UID = 3;
constexpr int64_t K_O_UID = 4;
constexpr int64_t K_STATS_UID = 5;
constexpr int64_t K_QO_OFFSET_UID = 6;
constexpr int64_t K_KV_OFFSET_UID = 7;
constexpr int64_t K_DESCALE_Q_UID = 8;
constexpr int64_t K_DESCALE_K_UID = 9;
constexpr int64_t K_DESCALE_V_UID = 10;
constexpr int64_t K_CU_SEQ_LEN_Q_UID = 11;
constexpr int64_t K_CU_SEQ_LEN_KV_UID = 12;
constexpr int64_t K_SEQ_LEN_Q_UID = 13;
constexpr int64_t K_SEQ_LEN_KV_UID = 14;

// Which length inputs a cuSeqLens graph carries: the cumulative tables the engine reads, or
// the per-batch lengths the CPU reference reads.
enum class LengthInputs
{
    CUMULATIVE,
    PER_BATCH,
};

// An INT32 [count, 1, 1, 1] token table (ragged offsets, cumulative or per-batch lengths).
std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>
    makeTokenTable(int64_t uid, const char* name, int64_t count)
{
    const std::vector<int64_t> dims = {count, 1, 1, 1};
    auto table = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    table->set_uid(uid)
        .set_name(name)
        .set_dim(dims)
        .set_stride(hipdnn_data_sdk::utilities::generateStrides(dims))
        .set_data_type(DataType::INT32);
    return table;
}

SdpaFwdGraph buildGraph(const GraphTestCase& testCase, LengthInputs lengthInputs)
{
    using namespace hipdnn_frontend::graph;
    using namespace hipdnn_data_sdk::utilities;

    const fmha_v3_fwdConfig& config = testCase.config;
    const bool isGroup = (config.mode == BatchMode::GROUP);
    if(!isGroup && (testCase.sharedOffsetTable || testCase.cuSeqLens))
    {
        throw std::invalid_argument("sharedOffsetTable and cuSeqLens need a group-mode config ("
                                    + testCase.name + ")");
    }
    if(testCase.sharedCuSeqLen && !testCase.cuSeqLens)
    {
        throw std::invalid_argument("sharedCuSeqLen needs cuSeqLens (" + testCase.name + ")");
    }

    const int64_t batch = testCase.batch;
    const int64_t numHeads = testCase.numHeads;
    const int64_t seqQ = testCase.seqQ;
    const int64_t seqKv = testCase.seqKv;

    // Determine data types. For fp8 the inputs are FP8_E4M3_FNUZ (the gfx942/MI300
    // hardware fp8 encoding the AITER kernels consume) and the output is BFLOAT16.
    const bool isFp8 = (config.dtype == config::FP8BF16);
    const DataType inDataType = isFp8 ? DataType::FP8_E4M3_FNUZ : toDataType(config.dtype);
    const DataType outDataType = isFp8 ? DataType::BFLOAT16 : inDataType;

    // Create tensor dimensions
    const std::vector<int64_t> qDims = {batch, numHeads, seqQ, config.hdim_q};
    const std::vector<int64_t> kDims = {batch, numHeads, seqKv, config.hdim_q};
    const std::vector<int64_t> vDims = {batch, numHeads, seqKv, config.hdim_v};
    // Use physical memory layout: B, Seq, H, D for compatibility with group mode
    const std::vector<int64_t> layout = {3, 1, 2, 0};

    auto graph = std::make_shared<Graph>();
    graph->set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto q = std::make_shared<TensorAttributes>();
    q->set_uid(K_Q_UID)
        .set_name("q")
        .set_dim(qDims)
        .set_stride(generateStrides(qDims, layout))
        .set_data_type(inDataType);

    auto k = std::make_shared<TensorAttributes>();
    k->set_uid(K_K_UID)
        .set_name("k")
        .set_dim(kDims)
        .set_stride(generateStrides(kDims, layout))
        .set_data_type(inDataType);

    auto v = std::make_shared<TensorAttributes>();
    v->set_uid(K_V_UID)
        .set_name("v")
        .set_dim(vDims)
        .set_stride(generateStrides(vDims, layout))
        .set_data_type(inDataType);

    std::shared_ptr<TensorAttributes> qoOffset;
    if(isGroup)
    {
        qoOffset = makeTokenTable(K_QO_OFFSET_UID, K_QO_RAGGED_OFFSET_NAME, batch + 1);
        const auto kvOffset
            = testCase.sharedOffsetTable
                  ? qoOffset
                  : makeTokenTable(K_KV_OFFSET_UID, K_KV_RAGGED_OFFSET_NAME, batch + 1);

        q->set_ragged_offset(qoOffset).set_ragged_offset_multiplier(q->get_stride()[2]);
        k->set_ragged_offset(kvOffset).set_ragged_offset_multiplier(k->get_stride()[2]);
        v->set_ragged_offset(kvOffset).set_ragged_offset_multiplier(v->get_stride()[2]);
    }

    // Configure SDPA attributes based on config
    SdpaAttributes attributes;
    attributes.set_name(testCase.withStats ? "SdpaFwdKernelConfigStatsTest"
                                           : "SdpaFwdKernelConfigTest");

    if(testCase.withStats)
    {
        attributes.set_generate_stats(true);
    }

    if(testCase.attnScale.has_value())
    {
        attributes.set_attn_scale(testCase.attnScale.value());
    }

    if(testCase.mmaCoreMode.has_value())
    {
        attributes.set_mma_core_mode(testCase.mmaCoreMode.value());
    }

    if(testCase.cuSeqLens)
    {
        attributes.set_padding_mask(true);
        if(lengthInputs == LengthInputs::CUMULATIVE)
        {
            const auto cuSeqLenQ
                = makeTokenTable(K_CU_SEQ_LEN_Q_UID, K_CU_SEQ_LEN_Q_NAME, batch + 1);
            attributes.set_cu_seq_len_q(cuSeqLenQ);
            attributes.set_cu_seq_len_kv(
                testCase.sharedCuSeqLen
                    ? cuSeqLenQ
                    : makeTokenTable(K_CU_SEQ_LEN_KV_UID, K_CU_SEQ_LEN_KV_NAME, batch + 1));
        }
        else
        {
            const auto seqLenQ = makeTokenTable(K_SEQ_LEN_Q_UID, K_SEQ_LEN_Q_NAME, batch);
            attributes.set_seq_len_q(seqLenQ);
            attributes.set_seq_len_kv(
                testCase.sharedCuSeqLen
                    ? seqLenQ
                    : makeTokenTable(K_SEQ_LEN_KV_UID, K_SEQ_LEN_KV_NAME, batch));
        }
    }

    // FP8 inputs require per-tensor (scalar) q/k/v descale tensors.
    if(isFp8)
    {
        const std::vector<int64_t> descaleDims = {1, 1, 1, 1};
        const auto descaleStrides = generateStrides(descaleDims);
        // Name each descale so the integration fixture can identify it by name (not by
        // scalar element count) when choosing per-tensor fill values.
        const auto makeDescale = [&](int64_t uid, const std::string& name) {
            auto descale = std::make_shared<TensorAttributes>();
            descale->set_uid(uid)
                .set_dim(descaleDims)
                .set_stride(descaleStrides)
                .set_data_type(DataType::FLOAT)
                .set_name(name);
            return descale;
        };
        attributes.set_descale_q(makeDescale(K_DESCALE_Q_UID, "descale_q"));
        attributes.set_descale_k(makeDescale(K_DESCALE_K_UID, "descale_k"));
        attributes.set_descale_v(makeDescale(K_DESCALE_V_UID, "descale_v"));
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
        break;

    case MaskType::BOTTOM_RIGHT_CAUSAL:
        attributes.set_diagonal_band_left_bound(-1);
        attributes.set_diagonal_band_right_bound(0);
        attributes.set_diagonal_alignment(DiagonalAlignment::BOTTOM_RIGHT);
        break;

    case MaskType::SLIDING_WINDOW:
        // Sliding window mask with arbitrary bounds
        attributes.set_diagonal_band_left_bound(64);
        attributes.set_diagonal_band_right_bound(64);
        break;

    default:
        break;
    }

    auto [o, stats] = graph->sdpa(q, k, v, attributes);

    o->set_uid(K_O_UID).set_name("o").set_output(true).set_data_type(outDataType);

    if(testCase.withStats)
    {
        stats->set_uid(K_STATS_UID)
            .set_name("stats")
            .set_output(true)
            .set_data_type(DataType::FLOAT);
        if(isGroup)
        {
            // AITER's varlen LSE [H, B * S_q] in token units on the Q/O offset table: the
            // dims stay [B, H, S_q, 1], and B * S_q, the Q token capacity, is the head stride.
            const std::vector<int64_t> statsDims = {batch, numHeads, seqQ, 1};
            stats->set_dim(statsDims)
                .set_stride({seqQ, batch * seqQ, 1, 1})
                .set_ragged_offset(qoOffset)
                .set_ragged_offset_multiplier(1);
        }
    }

    if(isGroup)
    {
        const std::vector<int64_t> oDims = {batch, numHeads, seqQ, config.hdim_v};
        o->set_dim(oDims).set_stride(generateStrides(oDims, layout));
        o->set_ragged_offset(qoOffset).set_ragged_offset_multiplier(o->get_stride()[2]);
    }

    return {graph, testCase.withStats ? stats : nullptr};
}

} // namespace

SdpaFwdGraph buildSdpaFwdGraph(const GraphTestCase& testCase)
{
    return buildGraph(testCase, LengthInputs::CUMULATIVE);
}

SdpaFwdGraph buildSdpaFwdReferenceGraph(const GraphTestCase& testCase)
{
    if(!testCase.cuSeqLens)
    {
        throw std::invalid_argument("buildSdpaFwdReferenceGraph needs a cuSeqLens case ("
                                    + testCase.name + ")");
    }
    return buildGraph(testCase, LengthInputs::PER_BATCH);
}

std::vector<std::string> getFwdKernelTocKeys(const fmha_v3_fwdConfig& config,
                                             const std::string& arch)
{
    std::vector<std::string> tocKeys{asm_kernels::getFwdKernelTocKey(config.co_name, arch, false)};
    if(arch == "gfx942")
    {
        tocKeys.push_back(asm_kernels::getFwdKernelTocKey(config.co_name, arch, true));
    }
    return tocKeys;
}

std::vector<AsmKernelTocEntry> getFwdKernelTocEntries(const std::string& arch)
{
    std::vector<AsmKernelTocEntry> entries;
    for(const auto& [key, config] : cfg_fmha_fwd)
    {
        if(config.arch != arch)
        {
            continue;
        }
        for(auto& tocKey : getFwdKernelTocKeys(config, arch))
        {
            entries.push_back({std::move(tocKey), config.knl_name});
        }
    }
    return entries;
}

} // namespace asm_sdpa_engine
