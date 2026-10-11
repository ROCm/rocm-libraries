// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Adapted from dnn-providers/hip-kernel-provider/src/tests/engines/kernel_ingestor_engine/packs/FlydslRmsNormTestGraphs.hpp at commit 03911535bd0 (FlyDSL provider)

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/layernorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_attributes_generated.h>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/// The Triton RMS-norm pack's symbols. `operationMatcher` is empty: `RMSNormAttributes` is
/// its own union arm, so the graph matcher's node-type test is the operation test.
///
/// Restated from the descriptors and the native file because a descriptor cannot reference
/// a C++ constant; a drifted string fails the census or the registry lookup.
inline constexpr PackSymbols TRITON_RMSNORM{"hipkernel:TritonRmsNorm",
                                            "hipkernel.triton_rmsnorm.graph_match",
                                            "",
                                            "hipkernel.triton_rmsnorm.kernel_match",
                                            "hipkernel.triton_rmsnorm.score",
                                            "hipkernel.triton_rmsnorm.dispatch",
                                            "triton_rmsnorm.x.uid",
                                            "triton_rmsnorm.scale.uid",
                                            "triton_rmsnorm.y.uid"};

/// The tokens the matcher binds beyond x/scale/y.
constexpr const char* TRITON_ROWS_TOKEN = "triton_rmsnorm.rows";
constexpr const char* TRITON_COLUMNS_TOKEN = "triton_rmsnorm.columns";
constexpr const char* TRITON_EPSILON_TOKEN = "triton_rmsnorm.epsilon.uid";

/// The kernel-metadata fields the Triton KMD declares.
constexpr const char* TRITON_DTYPE_FIELD = "dtype";
constexpr const char* TRITON_BLOCK_N_FIELD = "block_n";
constexpr const char* TRITON_BLOCK_THREADS_FIELD = "block_threads";
constexpr const char* TRITON_SHARED_MEM_BYTES_FIELD = "shared_mem_bytes";

/// The largest `block_n` the shipped instances cover; a normalised width above it matches
/// no kernel and the engine declines the graph.
constexpr int64_t TRITON_MAX_BLOCK_N = 8192;

/// Epsilon the default graph bakes. Not a kernel property: any epsilon is a kernarg.
constexpr float TRITON_DEFAULT_EPSILON = 1e-5F;

/// Tensor uids buildTritonRmsNormGraph() uses.
constexpr int64_t TRITON_X_UID = 1;
constexpr int64_t TRITON_SCALE_UID = 2;
constexpr int64_t TRITON_EPSILON_UID = 3;
constexpr int64_t TRITON_Y_UID = 4;
constexpr int64_t TRITON_BIAS_UID = 5;
constexpr int64_t TRITON_INV_RMS_UID = 6;

/// The INT32 offsets tensor a ragged operand names; emitted only when one is ragged.
constexpr int64_t TRITON_RAGGED_OFFSET_UID = 7;

/// @p values[field] when it is present and holds a T, else nullptr.
///
/// Looked up rather than indexed so a missing field fails the checking case instead of
/// throwing out_of_range.
template <typename T>
const T* tryGetMetadataField(const hipdnn_plugin_sdk::ingestor::MetadataValues& values,
                             const char* field)
{
    const auto it = values.find(std::string(field));
    return it == values.end() ? nullptr : std::get_if<T>(&it->second);
}

/// @brief Row-major packed strides for @p dims, the only layout the kernel addresses
/// (row `r` at `r * N`).
inline std::vector<int64_t> tritonPackedStrides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for(size_t i = dims.size(); i-- > 1;)
    {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

/// Everything this pack's graph matcher gates, so each refusal case names the one field it
/// changes. Defaults are a graph the shipped instances serve: packed row-major rank-2 bf16
/// x/y, a scale vector as wide as the normalised axis, a baked epsilon and alignment 16.
struct TritonRmsNormGraphOptions
{
    hipdnn_flatbuffers_sdk::data_objects::DataType dataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16;
    std::vector<int64_t> xDims = {8, 4096};
    /// Ignored when `runtimeEpsilon` is true: the tensor then carries no value and the
    /// caller supplies one at execute.
    float epsilon = TRITON_DEFAULT_EPSILON;
    bool runtimeEpsilon = false;
    /// Overrides scale's shape away from the normalised axis, for the broadcast refusal.
    std::optional<std::vector<int64_t>> scaleDims;
    /// Overrides scale's dtype away from `dataType`, for the dtype-mismatch refusal.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> scaleDataType;
    /// Overrides x's strides away from packed row-major.
    std::optional<std::vector<int64_t>> xStrides;
    bool withBias = false;
    bool withInvRms = false;
    hipdnn_flatbuffers_sdk::data_objects::NormFwdPhase forwardPhase
        = hipdnn_flatbuffers_sdk::data_objects::NormFwdPhase::NOT_SET;
    /// Declared alignment per operand; the matcher requires a positive multiple of 16.
    int64_t xAlignment = 16;
    int64_t scaleAlignment = 16;
    int64_t yAlignment = 16;
    /// Appends a second identical RMS-norm node.
    bool secondNode = false;
    /// Makes the single node a LayernormAttributes node over the same uids; its required
    /// bias names TRITON_BIAS_UID.
    bool asLayernorm = false;
    /// Emits no y tensor; the node still names TRITON_Y_UID.
    bool omitY = false;
    bool xVirtual = false;
    bool scaleVirtual = false;
    bool yVirtual = false;
    /// Marks x as a runtime pass-by-value operand.
    bool xPassByValue = false;
    /// Overrides y's dtype away from `dataType`.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> yDataType;
    /// Overrides y's dims away from x's; y's strides stay packed for these dims.
    std::optional<std::vector<int64_t>> yDims;
    /// Overrides y's strides away from packed row-major.
    std::optional<std::vector<int64_t>> yStrides;
    /// Overrides scale's strides away from packed row-major.
    std::optional<std::vector<int64_t>> scaleStrides;
    /// Overrides the epsilon tensor's dtype away from FLOAT.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> epsilonDataType;
    /// Points that operand's ragged offset at TRITON_RAGGED_OFFSET_UID.
    bool xRagged = false;
    bool scaleRagged = false;
    bool yRagged = false;
};

/**
 * @brief Builds an RMS-norm-forward graph from @p options: a single node unless
 *        `secondNode` is set.
 *
 * The stock hipdnn_test_sdk::utilities::createValidRMSNormGraph() cannot stand in: it
 * hard-codes epsilon, so it cannot vary the conditions under test here.
 */
inline flatbuffers::FlatBufferBuilder
    buildTritonRmsNormGraph(const TritonRmsNormGraphOptions& options = {})
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

    // Scale defaults to x's innermost extent, the normalised width at any rank.
    const auto scaleDims = options.scaleDims.has_value()
                               ? *options.scaleDims
                               : std::vector<int64_t>{options.xDims.back()};
    const auto scaleDataType = options.scaleDataType.value_or(options.dataType);

    const auto xStrides = options.xStrides.value_or(tritonPackedStrides(options.xDims));
    const auto yDims = options.yDims.value_or(options.xDims);
    const auto yStrides = options.yStrides.value_or(tritonPackedStrides(yDims));
    const auto scaleStrides = options.scaleStrides.value_or(tritonPackedStrides(scaleDims));

    const auto raggedOffset = [](bool ragged) {
        return ragged ? flatbuffers::Optional<int64_t>(TRITON_RAGGED_OFFSET_UID)
                      : flatbuffers::Optional<int64_t>(flatbuffers::nullopt);
    };

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    tensors.push_back(data_objects::CreateTensorAttributesDirect(builder,
                                                                 TRITON_X_UID,
                                                                 "x",
                                                                 options.dataType,
                                                                 &xStrides,
                                                                 &options.xDims,
                                                                 options.xVirtual,
                                                                 data_objects::TensorValue::NONE,
                                                                 0,
                                                                 options.xPassByValue,
                                                                 raggedOffset(options.xRagged),
                                                                 options.xAlignment));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(builder,
                                                                 TRITON_SCALE_UID,
                                                                 "scale",
                                                                 scaleDataType,
                                                                 &scaleStrides,
                                                                 &scaleDims,
                                                                 options.scaleVirtual,
                                                                 data_objects::TensorValue::NONE,
                                                                 0,
                                                                 false,
                                                                 raggedOffset(options.scaleRagged),
                                                                 options.scaleAlignment));
    if(!options.omitY)
    {
        tensors.push_back(
            data_objects::CreateTensorAttributesDirect(builder,
                                                       TRITON_Y_UID,
                                                       "y",
                                                       options.yDataType.value_or(options.dataType),
                                                       &yStrides,
                                                       &yDims,
                                                       options.yVirtual,
                                                       data_objects::TensorValue::NONE,
                                                       0,
                                                       false,
                                                       raggedOffset(options.yRagged),
                                                       options.yAlignment));
    }

    // Epsilon is pass-by-value either way (baked: a Float32Value; runtime: the
    // is_runtime_pass_by_value flag and no value), so it is not a device operand.
    const std::vector<int64_t> scalarDims = {1};
    const data_objects::Float32Value epsilonValue(options.epsilon);
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder,
        TRITON_EPSILON_UID,
        "epsilon",
        options.epsilonDataType.value_or(data_objects::DataType::FLOAT),
        &scalarDims,
        &scalarDims,
        false,
        options.runtimeEpsilon ? data_objects::TensorValue::NONE
                               : data_objects::TensorValue::Float32Value,
        options.runtimeEpsilon ? flatbuffers::Offset<void>(0)
                               : builder.CreateStruct(epsilonValue).Union(),
        options.runtimeEpsilon));

    if(options.withBias)
    {
        tensors.push_back(data_objects::CreateTensorAttributesDirect(
            builder, TRITON_BIAS_UID, "bias", scaleDataType, &scaleStrides, &scaleDims));
    }
    if(options.withInvRms)
    {
        static const std::vector<int64_t> s_invRmsDims{1};
        static const std::vector<int64_t> s_invRmsStrides{1};
        tensors.push_back(data_objects::CreateTensorAttributesDirect(builder,
                                                                     TRITON_INV_RMS_UID,
                                                                     "inv_rms",
                                                                     data_objects::DataType::FLOAT,
                                                                     &s_invRmsStrides,
                                                                     &s_invRmsDims));
    }

    if(options.xRagged || options.scaleRagged || options.yRagged)
    {
        // One offset per row plus the end.
        const std::vector<int64_t> raggedDims = {options.xDims.front() + 1};
        const std::vector<int64_t> raggedStrides = {1};
        tensors.push_back(data_objects::CreateTensorAttributesDirect(builder,
                                                                     TRITON_RAGGED_OFFSET_UID,
                                                                     "ragged_offset",
                                                                     data_objects::DataType::INT32,
                                                                     &raggedStrides,
                                                                     &raggedDims));
    }

    const auto makeNode = [&builder, &options]() {
        if(options.asLayernorm)
        {
            const auto layernorm = data_objects::CreateLayernormAttributes(builder,
                                                                           TRITON_X_UID,
                                                                           TRITON_SCALE_UID,
                                                                           TRITON_BIAS_UID,
                                                                           TRITON_EPSILON_UID,
                                                                           TRITON_Y_UID,
                                                                           1);
            return data_objects::CreateNodeDirect(builder,
                                                  "layernorm",
                                                  options.dataType,
                                                  data_objects::NodeAttributes::LayernormAttributes,
                                                  layernorm.Union());
        }

        const auto attributes = data_objects::CreateRMSNormAttributes(
            builder,
            TRITON_X_UID,
            TRITON_SCALE_UID,
            TRITON_EPSILON_UID,
            TRITON_Y_UID,
            options.withBias ? flatbuffers::Optional<int64_t>(TRITON_BIAS_UID)
                             : flatbuffers::nullopt,
            options.withInvRms ? flatbuffers::Optional<int64_t>(TRITON_INV_RMS_UID)
                               : flatbuffers::nullopt,
            options.forwardPhase);
        return data_objects::CreateNodeDirect(builder,
                                              "rmsnorm",
                                              options.dataType,
                                              data_objects::NodeAttributes::RMSNormAttributes,
                                              attributes.Union());
    };

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(makeNode());
    if(options.secondNode)
    {
        nodes.push_back(makeNode());
    }

    auto name = builder.CreateString("triton_rmsnorm_test");
    auto tensorsVector = builder.CreateVector(tensors);
    auto nodesVector = builder.CreateVector(nodes);

    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_name(name);
    graphBuilder.add_tensors(tensorsVector);
    graphBuilder.add_nodes(nodesVector);
    builder.Finish(graphBuilder.Finish());

    return builder;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
