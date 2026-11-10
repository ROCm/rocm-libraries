// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/gdn_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/linear_attention_common_generated.h>

#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/// hipkernel:flydsl_gdn. One pack, so `operationMatcher` is empty: `GdnAttributes` is its
/// own union arm, and the graph matcher admits the node type itself.
inline constexpr PackSymbols FLYDSL_GDN{"hipkernel:flydsl_gdn",
                                        "hipkernel.flydsl_gdn.graph_match",
                                        "",
                                        "hipkernel.flydsl_gdn.kernel_match",
                                        "hipkernel.flydsl_gdn.score",
                                        "hipkernel.flydsl_gdn.dispatch",
                                        "flydsl_gdn.q.uid",
                                        "flydsl_gdn.k.uid",
                                        "flydsl_gdn.o.uid"};

/// The tokens the graph match binds beside q, k and o.
constexpr const char* FLYDSL_GDN_V_TOKEN = "flydsl_gdn.v.uid";
constexpr const char* FLYDSL_GDN_INITIAL_STATE_TOKEN = "flydsl_gdn.initial_state.uid";
constexpr const char* FLYDSL_GDN_FINAL_STATE_TOKEN = "flydsl_gdn.final_state.uid";
constexpr const char* FLYDSL_GDN_STATE_INDICES_TOKEN = "flydsl_gdn.state_indices.uid";
constexpr const char* FLYDSL_GDN_TOKENS_TOKEN = "flydsl_gdn.tokens";
constexpr const char* FLYDSL_GDN_SEQUENCES_TOKEN = "flydsl_gdn.sequences";
constexpr const char* FLYDSL_GDN_HEADS_OUT_TOKEN = "flydsl_gdn.heads_out";
constexpr const char* FLYDSL_GDN_KEY_DIM_TOKEN = "flydsl_gdn.key_dim";
constexpr const char* FLYDSL_GDN_VALUE_DIM_TOKEN = "flydsl_gdn.value_dim";

/// The fields the GDN KMD declares.
constexpr const char* FLYDSL_GDN_DTYPE_FIELD = "dtype";
constexpr const char* FLYDSL_GDN_HEAD_DIM_FIELD = "head_dim";
constexpr const char* FLYDSL_GDN_HEAD_DIM_V_FIELD = "head_dim_v";
constexpr const char* FLYDSL_GDN_STATE_STRIDED_FIELD = "state_strided";
constexpr const char* FLYDSL_GDN_DV_SPLIT_FIELD = "dv_split";
constexpr const char* FLYDSL_GDN_BLOCK_THREADS_FIELD = "block_threads";

/// Tensor uids buildFlydslGdnGraph() uses.
constexpr int64_t FLYDSL_GDN_Q_UID = 1;
constexpr int64_t FLYDSL_GDN_K_UID = 2;
constexpr int64_t FLYDSL_GDN_V_UID = 3;
constexpr int64_t FLYDSL_GDN_G_UID = 4;
constexpr int64_t FLYDSL_GDN_BETA_UID = 5;
constexpr int64_t FLYDSL_GDN_CU_SEQLENS_UID = 6;
constexpr int64_t FLYDSL_GDN_O_UID = 7;
constexpr int64_t FLYDSL_GDN_INITIAL_STATE_UID = 8;
constexpr int64_t FLYDSL_GDN_FINAL_STATE_UID = 9;
constexpr int64_t FLYDSL_GDN_A_LOG_UID = 10;
constexpr int64_t FLYDSL_GDN_DT_BIAS_UID = 11;
constexpr int64_t FLYDSL_GDN_STATE_INDICES_UID = 12;
constexpr int64_t FLYDSL_GDN_CHECKPOINTS_UID = 13;

/// Everything buildFlydslGdnGraph() varies. The defaults are Qwen3.8's Gated DeltaNet
/// layer at two decode tokens: bf16 q/k/v/o, 16 q/k heads and 48 v heads of 128, the safe
/// gate with a_log and dt_bias, sigmoid beta, q/k L2 norm, an f32 [N, HO, V, K] state in
/// and out.
struct GdnGraphSpec
{
    using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;

    DataType ioType = DataType::BFLOAT16;
    DataType gType = DataType::FLOAT;
    DataType betaType = DataType::FLOAT;
    DataType paramType = DataType::FLOAT; ///< a_log and dt_bias
    DataType stateType = DataType::FLOAT;
    DataType cuSeqlensType = DataType::INT32;
    DataType indexType = DataType::INT32;
    DataType computeType = DataType::FLOAT;
    std::optional<DataType> oType; ///< unset: ioType

    int64_t tokens = 2;
    int64_t sequences = 2;
    int64_t headsQ = 16;
    int64_t headsK = 16;
    int64_t headsV = 48;
    int64_t keyDim = 128;
    int64_t valueDim = 128;
    /// State rows with state_indices; unset: the sequence count.
    std::optional<int64_t> poolRows;

    bool initialState = true;
    bool finalState = true;
    bool aLog = true;
    bool dtBias = true;
    bool stateIndices = false;
    bool checkpoints = false;

    std::optional<float> scale;
    std::optional<bool> outputFinalState; ///< unset: finalState
    bool useQkL2norm = true;
    bool useBetaSigmoid = true;
    bool allowNegEigval = false;
    bool safeGate = true;
    hipdnn_flatbuffers_sdk::data_objects::GateDomain gateDomain
        = hipdnn_flatbuffers_sdk::data_objects::GateDomain::LOG;
    int64_t checkpointEveryNTokens = 0;
    bool batchInvariant = false;
    bool overwriteInitialState = false;

    /// Dims and strides by uid; unset means the op's shape and packed strides.
    std::map<int64_t, std::vector<int64_t>> dims;
    std::map<int64_t, std::vector<int64_t>> strides;
    /// Declared base alignment by uid; unset means the schema's 16.
    std::map<int64_t, int64_t> alignment;
    std::set<int64_t> virtualTensors;
    bool secondNode = false;
    bool overrideShapeEnabled = false;

    int64_t headsOut() const
    {
        return headsQ > headsV ? headsQ : headsV;
    }

    int64_t stateRows() const
    {
        return poolRows.value_or(sequences);
    }
};

/// The K-major strides of a [rows, HO, V, K] state: V innermost.
inline std::vector<int64_t> flydslGdnKMajorStateStrides(int64_t headsOut, int64_t v, int64_t k)
{
    return {headsOut * v * k, v * k, 1, v};
}

/// A single-node GDN graph built from @p spec.
inline flatbuffers::FlatBufferBuilder buildFlydslGdnGraph(const GdnGraphSpec& spec = {})
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

    const auto ho = spec.headsOut();
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    const auto tensor = [&](int64_t uid,
                            const char* name,
                            data_objects::DataType type,
                            const std::vector<int64_t>& defaultDims) {
        const auto dimsIt = spec.dims.find(uid);
        const auto dims = dimsIt == spec.dims.end() ? defaultDims : dimsIt->second;
        const auto stridesIt = spec.strides.find(uid);
        const auto strides
            = stridesIt == spec.strides.end() ? flydslPackedStrides(dims) : stridesIt->second;
        const auto declared = spec.alignment.find(uid);
        tensors.push_back(data_objects::CreateTensorAttributesDirect(
            builder,
            uid,
            name,
            type,
            &strides,
            &dims,
            spec.virtualTensors.count(uid) != 0,
            data_objects::TensorValue::NONE,
            0,
            false,
            flatbuffers::nullopt,
            declared == spec.alignment.end() ? 16 : declared->second));
    };

    const auto t = spec.tokens;
    tensor(FLYDSL_GDN_Q_UID, "q", spec.ioType, {t, spec.headsQ, spec.keyDim});
    tensor(FLYDSL_GDN_K_UID, "k", spec.ioType, {t, spec.headsK, spec.keyDim});
    tensor(FLYDSL_GDN_V_UID, "v", spec.ioType, {t, spec.headsV, spec.valueDim});
    tensor(FLYDSL_GDN_G_UID, "g", spec.gType, {t, ho});
    tensor(FLYDSL_GDN_BETA_UID, "beta", spec.betaType, {t, ho});
    tensor(FLYDSL_GDN_CU_SEQLENS_UID, "cu_seqlens", spec.cuSeqlensType, {spec.sequences + 1});
    tensor(FLYDSL_GDN_O_UID, "o", spec.oType.value_or(spec.ioType), {t, ho, spec.valueDim});
    const std::vector<int64_t> stateDims{spec.stateRows(), ho, spec.valueDim, spec.keyDim};
    if(spec.initialState)
    {
        tensor(FLYDSL_GDN_INITIAL_STATE_UID, "initial_state", spec.stateType, stateDims);
    }
    if(spec.finalState)
    {
        tensor(FLYDSL_GDN_FINAL_STATE_UID, "final_state", spec.stateType, stateDims);
    }
    if(spec.aLog)
    {
        tensor(FLYDSL_GDN_A_LOG_UID, "a_log", spec.paramType, {ho});
    }
    if(spec.dtBias)
    {
        tensor(FLYDSL_GDN_DT_BIAS_UID, "dt_bias", spec.paramType, {ho});
    }
    if(spec.stateIndices)
    {
        tensor(FLYDSL_GDN_STATE_INDICES_UID, "state_indices", spec.indexType, {spec.sequences});
    }
    if(spec.checkpoints)
    {
        tensor(FLYDSL_GDN_CHECKPOINTS_UID,
               "state_checkpoints",
               spec.stateType,
               {t + spec.sequences, ho, spec.valueDim, spec.keyDim});
    }

    const auto optionalUid = [](bool present, int64_t uid) {
        return present ? flatbuffers::Optional<int64_t>(uid) : flatbuffers::nullopt;
    };
    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    for(int node = 0; node < (spec.secondNode ? 2 : 1); ++node)
    {
        auto attributes = data_objects::CreateGdnAttributes(
            builder,
            FLYDSL_GDN_Q_UID,
            FLYDSL_GDN_K_UID,
            FLYDSL_GDN_V_UID,
            FLYDSL_GDN_G_UID,
            FLYDSL_GDN_BETA_UID,
            FLYDSL_GDN_CU_SEQLENS_UID,
            optionalUid(spec.initialState, FLYDSL_GDN_INITIAL_STATE_UID),
            optionalUid(spec.aLog, FLYDSL_GDN_A_LOG_UID),
            optionalUid(spec.dtBias, FLYDSL_GDN_DT_BIAS_UID),
            optionalUid(spec.stateIndices, FLYDSL_GDN_STATE_INDICES_UID),
            FLYDSL_GDN_O_UID,
            optionalUid(spec.finalState, FLYDSL_GDN_FINAL_STATE_UID),
            optionalUid(spec.checkpoints, FLYDSL_GDN_CHECKPOINTS_UID),
            spec.scale,
            spec.outputFinalState.value_or(spec.finalState),
            spec.useQkL2norm,
            spec.checkpointEveryNTokens,
            spec.useBetaSigmoid,
            spec.allowNegEigval,
            spec.safeGate,
            spec.gateDomain,
            spec.batchInvariant,
            spec.overwriteInitialState);
        nodes.push_back(data_objects::CreateNodeDirect(builder,
                                                       "gdn",
                                                       spec.computeType,
                                                       data_objects::NodeAttributes::GdnAttributes,
                                                       attributes.Union()));
    }

    auto name = builder.CreateString("flydsl_gdn_test");
    auto tensorsVector = builder.CreateVector(tensors);
    auto nodesVector = builder.CreateVector(nodes);

    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_name(name);
    graphBuilder.add_tensors(tensorsVector);
    graphBuilder.add_nodes(nodesVector);
    graphBuilder.add_is_override_shape_enabled(spec.overrideShapeEnabled);
    builder.Finish(graphBuilder.Finish());
    return builder;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
