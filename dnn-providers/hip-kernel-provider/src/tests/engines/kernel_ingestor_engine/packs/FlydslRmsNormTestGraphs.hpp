// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_attributes_generated.h>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/// The third engine, split from Pointwise and ConvFwd by graph node type. One pack, so
/// `operationMatcher` is empty -- `RMSNormAttributes` is its own union arm, so admitting
/// the node type IS the operation test and the graph matcher does it in one pass.
///
/// The three operand tokens are x/scale/y; epsilon, rows and columns are bound too but
/// the shared harness only names three, and those are the three that carry pointers.
inline constexpr PackSymbols FLYDSL_RMSNORM{"hipkernel:flydsl_rmsnorm",
                                            "hipkernel.flydsl_rmsnorm.graph_match",
                                            "",
                                            "hipkernel.flydsl_rmsnorm.kernel_match",
                                            "hipkernel.flydsl_rmsnorm.score",
                                            "hipkernel.flydsl_rmsnorm.dispatch",
                                            "flydsl_rmsnorm.x.uid",
                                            "flydsl_rmsnorm.scale.uid",
                                            "flydsl_rmsnorm.y.uid"};

/// The two tokens the matcher binds that are neither uids nor named by PackSymbols.
constexpr const char* FLYDSL_ROWS_TOKEN = "flydsl_rmsnorm.rows";
constexpr const char* FLYDSL_COLUMNS_TOKEN = "flydsl_rmsnorm.columns";
constexpr const char* FLYDSL_EPSILON_TOKEN = "flydsl_rmsnorm.epsilon.uid";

/// The kernel-metadata fields the flydsl KMD declares. Distinct from the pointwise pack's
/// BLOCK_SIZE/DTYPE/OPERATION triple, so the shared makeKernel() cannot serve this pack.
constexpr const char* FLYDSL_DTYPE_FIELD = "dtype";
constexpr const char* FLYDSL_N_FIELD = "N";
constexpr const char* FLYDSL_BLOCK_THREADS_FIELD = "block_threads";

/// `N: 0` in a UKD means "takes the width as a kernarg". Mirrored from the pack's own
/// GENERIC_N_SENTINEL rather than shared, so a change there has to be made twice and is
/// seen -- the value is a wire contract with the descriptors, not an implementation
/// detail either side may move alone.
constexpr int64_t FLYDSL_GENERIC_N = 0;

/// `EPS` in kernels_src/kernels/norm/rmsnorm_common.py, baked into every instance. A graph
/// asking for anything materially different is not one these kernels compute.
constexpr float FLYDSL_BAKED_EPSILON = 1e-5f;

/// Tensor uids buildFlydslRmsNormGraph() uses, in kernel argument order.
constexpr int64_t FLYDSL_X_UID = 1;
constexpr int64_t FLYDSL_SCALE_UID = 2;
constexpr int64_t FLYDSL_EPSILON_UID = 3;
constexpr int64_t FLYDSL_Y_UID = 4;
constexpr int64_t FLYDSL_BIAS_UID = 5;
constexpr int64_t FLYDSL_INV_RMS_UID = 6;

/// @p values[field] when it is present and holds a T, else nullptr.
///
/// Looked up rather than indexed: a kernel missing the field is a finding for the case
/// that checks the field, not an out_of_range from a helper three frames away.
///
/// Shared rather than file-local because both FlyDSL test files read the same metadata
/// triple -- the census over the staged shard and the suites that build kernels by hand.
template <typename T>
const T* tryGetMetadataField(const hipdnn_plugin_sdk::ingestor::MetadataValues& values,
                             const char* field)
{
    const auto it = values.find(std::string(field));
    return it == values.end() ? nullptr : std::get_if<T>(&it->second);
}

/// @brief Row-major packed strides for @p dims. The kernel takes one row stride and an
/// implied innermost stride of 1, so a graph whose strides are not these is refused
/// outright rather than approximated.
inline std::vector<int64_t> flydslPackedStrides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for(size_t i = dims.size(); i-- > 1;)
    {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

/**
 * @brief Builds a single-node RMS-norm-forward graph, parameterized on everything this
 *        pack's graph matcher gates.
 *
 * Defaults to the shape the shipped instances serve: a packed row-major rank-2 x/y in a
 * supported dtype, a gamma vector exactly as wide as the normalised axis, and the baked
 * epsilon as a compile-time constant. Every other parameter exists for one refusal case.
 *
 * The stock hipdnn_test_sdk::utilities::createValidRMSNormGraph() cannot stand in: it
 * hard-codes epsilon at 1e-5 with no way to vary it, types x as FLOAT, and gives gamma
 * x's full rank-4 shape -- three of the conditions under test here.
 *
 * @param scaleDims Overrides gamma's shape away from the normalised axis, for the
 *        broadcast-gamma refusal.
 * @param scaleDataType Overrides gamma's dtype away from @p dataType, for the
 *        cross-operand dtype-mismatch refusal.
 * @param xStridesOverride Overrides x's strides away from packed row-major.
 * @param epsilon The baked epsilon. Ignored when @p runtimeEpsilon is true, in which case
 *        the tensor carries no value and the caller supplies one at execute.
 */
inline flatbuffers::FlatBufferBuilder buildFlydslRmsNormGraph(
    hipdnn_flatbuffers_sdk::data_objects::DataType dataType
    = hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
    const std::vector<int64_t>& xDims = {8, 4096},
    float epsilon = FLYDSL_BAKED_EPSILON,
    bool runtimeEpsilon = false,
    const std::optional<std::vector<int64_t>>& scaleDims = std::nullopt,
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> scaleDataType = std::nullopt,
    const std::optional<std::vector<int64_t>>& xStridesOverride = std::nullopt,
    bool withBias = false,
    bool withInvRms = false,
    hipdnn_flatbuffers_sdk::data_objects::NormFwdPhase forwardPhase
    = hipdnn_flatbuffers_sdk::data_objects::NormFwdPhase::NOT_SET)
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

    // Gamma defaults to the innermost extent of x, whatever rank x has: the matcher
    // collapses everything outside the innermost axis into the row count, so that extent
    // is the normalised width in every accepted shape.
    const auto resolvedScaleDims
        = scaleDims.has_value() ? *scaleDims : std::vector<int64_t>{xDims.back()};
    const auto resolvedScaleDataType = scaleDataType.value_or(dataType);

    const auto xStrides = xStridesOverride.value_or(flydslPackedStrides(xDims));
    const auto yStrides = flydslPackedStrides(xDims);
    const auto scaleStrides = flydslPackedStrides(resolvedScaleDims);

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, FLYDSL_X_UID, "x", dataType, &xStrides, &xDims));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(builder,
                                                                 FLYDSL_SCALE_UID,
                                                                 "scale",
                                                                 resolvedScaleDataType,
                                                                 &scaleStrides,
                                                                 &resolvedScaleDims));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, FLYDSL_Y_UID, "y", dataType, &yStrides, &xDims));

    // Epsilon is pass-by-value either way: baked carries a Float32Value, runtime carries
    // the is_runtime_pass_by_value flag and no value at all. Both make
    // isPassByValueTensor() true, which is what keeps it out of isDeviceOperand().
    const std::vector<int64_t> scalarDims = {1};
    const data_objects::Float32Value epsilonValue(epsilon);
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder,
        FLYDSL_EPSILON_UID,
        "epsilon",
        data_objects::DataType::FLOAT,
        &scalarDims,
        &scalarDims,
        false,
        runtimeEpsilon ? data_objects::TensorValue::NONE : data_objects::TensorValue::Float32Value,
        runtimeEpsilon ? flatbuffers::Offset<void>(0) : builder.CreateStruct(epsilonValue).Union(),
        runtimeEpsilon));

    if(withBias)
    {
        tensors.push_back(data_objects::CreateTensorAttributesDirect(builder,
                                                                     FLYDSL_BIAS_UID,
                                                                     "bias",
                                                                     resolvedScaleDataType,
                                                                     &scaleStrides,
                                                                     &resolvedScaleDims));
    }
    if(withInvRms)
    {
        // One stat per row, so gamma's shape does not describe it -- build it from the
        // leading axes of x with a trailing 1.
        static const std::vector<int64_t> s_invRmsDims{1};
        static const std::vector<int64_t> s_invRmsStrides{1};
        tensors.push_back(data_objects::CreateTensorAttributesDirect(builder,
                                                                     FLYDSL_INV_RMS_UID,
                                                                     "inv_rms",
                                                                     data_objects::DataType::FLOAT,
                                                                     &s_invRmsStrides,
                                                                     &s_invRmsDims));
    }

    auto attributes = data_objects::CreateRMSNormAttributes(
        builder,
        FLYDSL_X_UID,
        FLYDSL_SCALE_UID,
        FLYDSL_EPSILON_UID,
        FLYDSL_Y_UID,
        withBias ? flatbuffers::Optional<int64_t>(FLYDSL_BIAS_UID) : flatbuffers::nullopt,
        withInvRms ? flatbuffers::Optional<int64_t>(FLYDSL_INV_RMS_UID) : flatbuffers::nullopt,
        forwardPhase);

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(data_objects::CreateNodeDirect(builder,
                                                   "rmsnorm",
                                                   dataType,
                                                   data_objects::NodeAttributes::RMSNormAttributes,
                                                   attributes.Union()));

    auto name = builder.CreateString("flydsl_rmsnorm_test");
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
