// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "ConfigHelpers.hpp"
#include "GraphTest.hpp"
#include "asm_fmha_v3_fwd_configs.hpp"
#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaFwdPlanBuilder.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaPlanUtils.hpp"
#include "hip_kernel_provider_common/HipDeviceUtils.hpp"
#include "tests/engines/asm_sdpa_engine/DeprecatedCausalMaskCases.hpp"
#include "tests/engines/asm_sdpa_engine/SdpaSinkTokenGraph.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace asm_sdpa_engine
{
namespace
{

class TestSdpaFwdPlanBuilder : public ::testing::Test
{
protected:
    SdpaFwdPlanBuilder _planBuilder;
    Handle _handle;
};

TEST_F(TestSdpaFwdPlanBuilder, IsApplicableReturnsFalseForNonSdpaGraph)
{
    // Create a batchnorm inference graph - this does not use SDPA attributes
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
}

auto createSdpaFwdGraph(const std::vector<int64_t>& qDims = {4, 8, 256, 128},
                        const std::vector<int64_t>& vDims = {4, 8, 256, 128},
                        hipdnn_flatbuffers_sdk::data_objects::DataType inputType
                        = hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                        hipdnn_flatbuffers_sdk::data_objects::DataType computeType
                        = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                        bool withAttnMask = false,
                        bool withScale = false,
                        bool withStats = false,
                        bool alibiMask = false,
                        bool paddingMask = false,
                        bool causalMask = false,
                        bool overrideShapeEnabled = false,
                        hipdnn_flatbuffers_sdk::data_objects::DataType mmaCoreMode
                        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET)
{
    if(qDims.size() != 4 || vDims.size() != 4)
    {
        throw std::runtime_error("Q, K and V tensors must have a dimension of 4");
    }
    const std::vector<int64_t> kDims = {qDims[0], vDims[1], vDims[2], qDims[3]};
    const std::vector<int64_t> oDims = {qDims[0], qDims[1], qDims[2], vDims[3]};

    return hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        qDims,
        hipdnn_data_sdk::utilities::generateStrides(qDims),
        kDims,
        hipdnn_data_sdk::utilities::generateStrides(kDims),
        vDims,
        hipdnn_data_sdk::utilities::generateStrides(vDims),
        oDims,
        hipdnn_data_sdk::utilities::generateStrides(oDims),
        inputType,
        computeType,
        withAttnMask,
        withScale,
        withStats,
        alibiMask,
        paddingMask,
        causalMask,
        overrideShapeEnabled,
        mmaCoreMode);
}

TEST_F(TestSdpaFwdPlanBuilder, IsApplicableReturnsFalseForOverrideShapeEnabledGraph)
{
    auto builder = createSdpaFwdGraph({4, 8, 256, 128},
                                      {4, 8, 256, 128},
                                      hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                                      hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                      false,
                                      false,
                                      false,
                                      false,
                                      false,
                                      false,
                                      /*overrideShapeEnabled=*/true);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaFwdPlanBuilder, IsApplicableAvailableKernels)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    SKIP_IF_NO_DEVICES();

    const std::string deviceString
        = hip_kernel_provider_common::getDeviceString(_handle.getStream());

    for(const auto& test : getCompatibleGraphsForArch(deviceString, cfg_fmha_fwd))
    {
        EXPECT_TRUE(_planBuilder.isApplicable(_handle, test.graphWrapper())) << test.message;
    }
}

TEST_F(TestSdpaFwdPlanBuilder, IsApplicableSdpaVariations)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    SKIP_IF_NO_DEVICES();

    const std::string deviceString
        = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }

    const std::vector<std::pair<GraphTest, bool>> applicabilityTests = {
        {GraphTest{createSdpaFwdGraph(), "Valid test with q head dim 128"}, true},
        {GraphTest{createSdpaFwdGraph({4, 8, 256, 192}), "Valid test with q head dim 192"}, true},
        {GraphTest{createSdpaFwdGraph({4, 8, 256, 100}), "Final dimension not 128"}, false},
        {GraphTest{createSdpaFwdGraph({4, 8, 256, 128}, {4, 8, 256, 128}, DataType::HALF),
                   "Half precision tensor data type"},
         false},
        {GraphTest{
             createSdpaFwdGraph(
                 {4, 8, 256, 128}, {4, 8, 256, 128}, DataType::BFLOAT16, DataType::FLOAT, true),
             "attn_mask = true"},
         false},
        {GraphTest{createSdpaFwdGraph({4, 8, 256, 128},
                                      {4, 8, 256, 128},
                                      DataType::BFLOAT16,
                                      DataType::FLOAT,
                                      false,
                                      true),
                   "scale = true"},
         true},
        {GraphTest{createSdpaFwdGraph({4, 8, 256, 128},
                                      {4, 8, 256, 128},
                                      DataType::BFLOAT16,
                                      DataType::FLOAT,
                                      false,
                                      false,
                                      true),
                   "withStats = true"},
         true},
        {GraphTest{createSdpaFwdGraph({4, 8, 256, 128},
                                      {4, 8, 256, 128},
                                      DataType::BFLOAT16,
                                      DataType::FLOAT,
                                      false,
                                      true,
                                      false,
                                      true),
                   "alibi_mask = true"},
         false},
        {GraphTest{createSdpaFwdGraph({4, 8, 256, 128},
                                      {4, 8, 256, 128},
                                      DataType::BFLOAT16,
                                      DataType::FLOAT,
                                      false,
                                      true,
                                      false,
                                      false,
                                      true),
                   "padding_mask = true"},
         false},
        {GraphTest{createSdpaFwdGraph({4, 8, 256, 128},
                                      {4, 8, 256, 128},
                                      DataType::BFLOAT16,
                                      DataType::FLOAT,
                                      false,
                                      true,
                                      false,
                                      false,
                                      false,
                                      true),
                   "causal_mask = true"},
         false},
        {GraphTest{createSdpaFwdGraph({4, 8, 256, 128},
                                      {4, 8, 256, 128},
                                      DataType::BFLOAT16,
                                      DataType::BFLOAT16,
                                      false,
                                      true),
                   "compute_data_type != FLOAT"},
         false},
        {GraphTest{createSdpaFwdGraph({4, 8, 256, 128},
                                      {4, 8, 256, 128},
                                      DataType::BFLOAT16,
                                      DataType::FLOAT,
                                      false,
                                      false,
                                      false,
                                      false,
                                      false,
                                      false,
                                      false,
                                      DataType::HALF),
                   "mma_core_mode = HALF (cuDNN-frontend default) on bf16 inputs"},
         true},
        {GraphTest{createSdpaFwdGraph({4, 8, 256, 128},
                                      {4, 8, 256, 128},
                                      DataType::BFLOAT16,
                                      DataType::FLOAT,
                                      false,
                                      false,
                                      false,
                                      false,
                                      false,
                                      false,
                                      false,
                                      DataType::BFLOAT16),
                   "mma_core_mode = BFLOAT16 on bf16 inputs"},
         true},
        {GraphTest{createSdpaFwdGraph({4, 8, 256, 128},
                                      {4, 8, 256, 128},
                                      DataType::BFLOAT16,
                                      DataType::FLOAT,
                                      false,
                                      false,
                                      false,
                                      false,
                                      false,
                                      false,
                                      false,
                                      DataType::FLOAT),
                   "mma_core_mode = FLOAT"},
         false}};

    for(const auto& [test, applicability] : applicabilityTests)
    {
        EXPECT_EQ(_planBuilder.isApplicable(_handle, test.graphWrapper()), applicability)
            << test.message;
    }
}

// =============================================================================
// Runtime pass-by-value scale tensor (RFC 0016)
// =============================================================================

// Build a forward SDPA graph with a runtime pass-by-value scale tensor
// (is_runtime_pass_by_value=true, no baked value).
flatbuffers::FlatBufferBuilder createSdpaFwdGraphWithRuntimePbvScale()
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<TensorAttributes>> tensorAttributes;

    const std::vector<int64_t> dims = {4, 8, 256, 128};
    const std::vector<int64_t> strides = hipdnn_data_sdk::utilities::generateStrides(dims);

    int64_t uid = 1;
    const auto qUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, qUid, "q", DataType::BFLOAT16, &strides, &dims));
    const auto kUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, kUid, "k", DataType::BFLOAT16, &strides, &dims));
    const auto vUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, vUid, "v", DataType::BFLOAT16, &strides, &dims));
    const auto oUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, oUid, "o", DataType::BFLOAT16, &strides, &dims));

    // Runtime pass-by-value scale tensor: is_runtime_pass_by_value=true, no value
    const std::vector<int64_t> scaleDims = {1};
    const auto scaleUid = uid++;
    tensorAttributes.push_back(CreateTensorAttributesDirect(builder,
                                                            scaleUid,
                                                            "scale",
                                                            DataType::FLOAT,
                                                            &scaleDims,
                                                            &scaleDims,
                                                            false, // virtual
                                                            TensorValue::NONE, // no baked value
                                                            0, // value offset
                                                            true)); // is_runtime_pass_by_value

    const auto sdpaAttributes = CreateSdpaAttributes(builder,
                                                     qUid,
                                                     kUid,
                                                     vUid,
                                                     oUid,
                                                     flatbuffers::nullopt, // attn_mask
                                                     scaleUid); // scale_tensor_uid

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "sdpa_fwd",
                                     DataType::FLOAT,
                                     NodeAttributes::SdpaAttributes,
                                     sdpaAttributes.Union()));

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

// Build a forward SDPA graph with a non-pass-by-value scale tensor (a regular
// device tensor, not a scalar). This should be rejected by isApplicable.
flatbuffers::FlatBufferBuilder createSdpaFwdGraphWithNonPbvScaleTensor()
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<TensorAttributes>> tensorAttributes;

    const std::vector<int64_t> dims = {4, 8, 256, 128};
    const std::vector<int64_t> strides = hipdnn_data_sdk::utilities::generateStrides(dims);

    int64_t uid = 1;
    const auto qUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, qUid, "q", DataType::BFLOAT16, &strides, &dims));
    const auto kUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, kUid, "k", DataType::BFLOAT16, &strides, &dims));
    const auto vUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, vUid, "v", DataType::BFLOAT16, &strides, &dims));
    const auto oUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, oUid, "o", DataType::BFLOAT16, &strides, &dims));

    // Regular device tensor — NOT pass-by-value, NOT a scalar
    const std::vector<int64_t> scaleDims = {1};
    const auto scaleUid = uid++;
    tensorAttributes.push_back(CreateTensorAttributesDirect(
        builder, scaleUid, "scale", DataType::FLOAT, &scaleDims, &scaleDims));

    const auto sdpaAttributes = CreateSdpaAttributes(builder,
                                                     qUid,
                                                     kUid,
                                                     vUid,
                                                     oUid,
                                                     flatbuffers::nullopt, // attn_mask
                                                     scaleUid); // scale_tensor_uid

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "sdpa_fwd",
                                     DataType::FLOAT,
                                     NodeAttributes::SdpaAttributes,
                                     sdpaAttributes.Union()));

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

// Build an fp8-in / bf16-out forward SDPA graph, optionally with q/k/v descale
// tensors, to exercise the fp8 applicability policy. descaleDims defaults to a
// per-tensor scalar ([1,1,1,1]); pass a larger shape (e.g. [B,H_kv,1,1]) to build
// the non-scalar descales the engine must reject. qStrides and vStrides override
// the contiguous Q and V strides, which is what the fp8 Ts and byte-stride
// boundary cases need.
flatbuffers::FlatBufferBuilder
    createSdpaFwdFp8Graph(bool withDescale,
                          const std::vector<int64_t>& descaleDims = {1, 1, 1, 1},
                          const std::vector<int64_t>& qStrides = {},
                          const std::vector<int64_t>& vStrides = {},
                          hipdnn_flatbuffers_sdk::data_objects::DataType mmaCoreMode
                          = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<TensorAttributes>> tensorAttributes;

    const std::vector<int64_t> dims = {4, 8, 256, 128};
    const std::vector<int64_t> strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    const std::vector<int64_t>& qStridesUsed = qStrides.empty() ? strides : qStrides;
    const std::vector<int64_t>& vStridesUsed = vStrides.empty() ? strides : vStrides;
    const std::vector<int64_t>& scalarDims = descaleDims;
    const std::vector<int64_t> scalarStrides
        = hipdnn_data_sdk::utilities::generateStrides(scalarDims);

    int64_t uid = 1;
    const auto qUid = uid++;
    tensorAttributes.push_back(CreateTensorAttributesDirect(
        builder, qUid, "q", DataType::FP8_E4M3_FNUZ, &qStridesUsed, &dims));
    const auto kUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, kUid, "k", DataType::FP8_E4M3_FNUZ, &strides, &dims));
    const auto vUid = uid++;
    tensorAttributes.push_back(CreateTensorAttributesDirect(
        builder, vUid, "v", DataType::FP8_E4M3_FNUZ, &vStridesUsed, &dims));
    const auto oUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, oUid, "o", DataType::BFLOAT16, &strides, &dims));

    flatbuffers::Optional<int64_t> descaleQUid;
    flatbuffers::Optional<int64_t> descaleKUid;
    flatbuffers::Optional<int64_t> descaleVUid;
    if(withDescale)
    {
        descaleQUid = uid++;
        tensorAttributes.push_back(CreateTensorAttributesDirect(
            builder, *descaleQUid, "descale_q", DataType::FLOAT, &scalarStrides, &scalarDims));
        descaleKUid = uid++;
        tensorAttributes.push_back(CreateTensorAttributesDirect(
            builder, *descaleKUid, "descale_k", DataType::FLOAT, &scalarStrides, &scalarDims));
        descaleVUid = uid++;
        tensorAttributes.push_back(CreateTensorAttributesDirect(
            builder, *descaleVUid, "descale_v", DataType::FLOAT, &scalarStrides, &scalarDims));
    }

    const auto sdpaAttributes
        = CreateSdpaAttributes(builder,
                               qUid,
                               kUid,
                               vUid,
                               oUid,
                               flatbuffers::nullopt, // attn_mask_tensor_uid
                               flatbuffers::nullopt, // scale_tensor_uid
                               flatbuffers::nullopt, // seq_len_q_tensor_uid
                               flatbuffers::nullopt, // seq_len_kv_tensor_uid
                               flatbuffers::nullopt, // seed_tensor_uid
                               flatbuffers::nullopt, // offset_tensor_uid
                               flatbuffers::nullopt, // dropout_mask_tensor_uid
                               flatbuffers::nullopt, // dropout_scale_tensor_uid
                               flatbuffers::nullopt, // page_table_k_tensor_uid
                               flatbuffers::nullopt, // page_table_v_tensor_uid
                               flatbuffers::nullopt, // block_mask_tensor_uid
                               flatbuffers::nullopt, // sink_token_tensor_uid
                               descaleQUid,
                               descaleKUid,
                               descaleVUid,
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
                               false, // causal_mask
                               false, // causal_mask_bottom_right
                               flatbuffers::nullopt, // dropout_probability
                               flatbuffers::nullopt, // attn_scale_value
                               flatbuffers::nullopt, // left_bound
                               flatbuffers::nullopt, // right_bound
                               flatbuffers::nullopt, // max_seq_len_kv
                               DiagonalAlignment::TOP_LEFT,
                               mmaCoreMode,
                               AttentionImplementation::AUTO);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "sdpa_fwd",
                                     DataType::FLOAT, // node compute type (engine requires FLOAT)
                                     NodeAttributes::SdpaAttributes,
                                     sdpaAttributes.Union()));

    const auto graphOffset = CreateGraphDirect(builder,
                                               "test",
                                               DataType::FLOAT,
                                               DataType::FLOAT,
                                               DataType::BFLOAT16,
                                               &tensorAttributes,
                                               &nodes);
    builder.Finish(graphOffset);
    return builder;
}

TEST_F(TestSdpaFwdPlanBuilder, IsApplicableAcceptsRuntimePassByValueScale)
{
    SKIP_IF_NO_DEVICES();

    const std::string deviceString
        = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }

    auto builder = createSdpaFwdGraphWithRuntimePbvScale();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaFwdPlanBuilder, IsApplicableRejectsNonPassByValueScaleTensor)
{
    SKIP_IF_NO_DEVICES();

    const std::string deviceString
        = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }

    auto builder = createSdpaFwdGraphWithNonPbvScaleTensor();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaFwdPlanBuilder, IsApplicableAcceptsCompileTimeConstantScaleTensor)
{
    // Existing withScale=true path uses a compile-time constant scale tensor
    // (Float32Value baked in). Verify it still passes after the PBV changes.
    SKIP_IF_NO_DEVICES();

    const std::string deviceString
        = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }

    auto builder = createSdpaFwdGraph({4, 8, 256, 128},
                                      {4, 8, 256, 128},
                                      hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                                      hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                      /*withAttnMask=*/false,
                                      /*withScale=*/true);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_handle, graphWrapper));
}

// =============================================================================
// Stats (LSE) output tensor validation
// =============================================================================

// Build a forward SDPA graph with a custom stats tensor (arbitrary dtype/dims).
flatbuffers::FlatBufferBuilder
    createSdpaFwdGraphWithCustomStats(hipdnn_flatbuffers_sdk::data_objects::DataType statsDataType,
                                      const std::vector<int64_t>& statsDims)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<TensorAttributes>> tensorAttributes;

    const std::vector<int64_t> dims = {4, 8, 256, 128};
    const std::vector<int64_t> strides = hipdnn_data_sdk::utilities::generateStrides(dims);

    int64_t uid = 1;
    const auto qUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, qUid, "q", DataType::BFLOAT16, &strides, &dims));
    const auto kUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, kUid, "k", DataType::BFLOAT16, &strides, &dims));
    const auto vUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, vUid, "v", DataType::BFLOAT16, &strides, &dims));
    const auto oUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, oUid, "o", DataType::BFLOAT16, &strides, &dims));

    const auto statsStrides = hipdnn_data_sdk::utilities::generateStrides(statsDims);
    const auto statsUidVal = uid++;
    tensorAttributes.push_back(CreateTensorAttributesDirect(
        builder, statsUidVal, "stats", statsDataType, &statsStrides, &statsDims));

    const auto sdpaAttributes
        = CreateSdpaAttributes(builder,
                               qUid,
                               kUid,
                               vUid,
                               oUid,
                               flatbuffers::nullopt, // attn_mask_tensor_uid
                               flatbuffers::nullopt, // scale_tensor_uid
                               flatbuffers::nullopt, // seq_len_q_tensor_uid
                               flatbuffers::nullopt, // seq_len_kv_tensor_uid
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
                               flatbuffers::Optional<int64_t>(statsUidVal), // stats_tensor_uid
                               flatbuffers::nullopt, // max_tensor_uid
                               flatbuffers::nullopt, // sum_exp_tensor_uid
                               flatbuffers::nullopt, // rng_dump_tensor_uid
                               flatbuffers::nullopt, // amax_s_tensor_uid
                               flatbuffers::nullopt, // amax_o_tensor_uid
                               flatbuffers::Optional<bool>(true), // generate_stats
                               false, // alibi_mask
                               false, // padding_mask
                               false, // causal_mask
                               false); // causal_mask_bottom_right

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

TEST_F(TestSdpaFwdPlanBuilder, IsApplicableRejectsStatsWithWrongDataType)
{
    SKIP_IF_NO_DEVICES();

    const std::string deviceString
        = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }

    // Stats tensor with HALF instead of FLOAT -> rejected
    auto builder = createSdpaFwdGraphWithCustomStats(
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF, {4, 8, 256, 1});

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaFwdPlanBuilder, IsApplicableRejectsStatsWithWrongShape)
{
    SKIP_IF_NO_DEVICES();

    const std::string deviceString
        = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }

    // Stats tensor with wrong batch dimension (3 != 4) -> rejected
    auto builder = createSdpaFwdGraphWithCustomStats(
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, {3, 8, 256, 1});

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaFwdPlanBuilder, IsApplicableRejectsStatsWithWrongRank)
{
    SKIP_IF_NO_DEVICES();

    const std::string deviceString
        = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }

    // Stats tensor with rank 2 -> rejected
    auto builder = createSdpaFwdGraphWithCustomStats(
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, {4, 8});

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaFwdPlanBuilder, IsApplicableAcceptsStatsRank3)
{
    SKIP_IF_NO_DEVICES();

    const std::string deviceString
        = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }

    // Stats tensor with rank 3 [B, H, Sq] -> accepted
    auto builder = createSdpaFwdGraphWithCustomStats(
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, {4, 8, 256});

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaFwdPlanBuilder, IsApplicableAcceptsStatsRank4)
{
    SKIP_IF_NO_DEVICES();

    const std::string deviceString
        = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }

    // Stats tensor with rank 4 [B, H, Sq, 1] -> accepted
    auto builder = createSdpaFwdGraphWithCustomStats(
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, {4, 8, 256, 1});

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_handle, graphWrapper));
}

// FP8 forward kernels are only shipped for gfx942, so these applicability tests
// are gated on that architecture.
TEST_F(TestSdpaFwdPlanBuilder, IsApplicableRejectsFp8WithoutDescale)
{
    SKIP_IF_NO_DEVICES();
    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
    {
        GTEST_SKIP() << "fp8 forward kernels are gfx942-only";
    }

    auto builder = createSdpaFwdFp8Graph(/*withDescale=*/false);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper))
        << "fp8 inputs without q/k/v descales must be rejected";
}

// Note: the positive "fp8 + scalar descales is accepted" case is intentionally not a
// standalone test — IsApplicableAvailableKernels already covers it. On gfx942 that loop
// iterates every cfg_fmha_fwd config, which includes the fp8bf16 rows; configToCompatibleGraph
// builds them with scalar q/k/v descales and asserts isApplicable. Only the negative cases
// below (missing / non-scalar descale) need explicit coverage.

// The kernel-arg builder only wires per-tensor (scalar) descales (all s_descale_*
// strides are zero), so a non-scalar descale (here per-[B, H_kv]) must be declined
// rather than silently mis-read as descale[0] for every batch/head.
TEST_F(TestSdpaFwdPlanBuilder, IsApplicableRejectsFp8WithNonScalarDescale)
{
    SKIP_IF_NO_DEVICES();
    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
    {
        GTEST_SKIP() << "fp8 forward kernels are gfx942-only";
    }

    // Q dims are {B=4, H_kv=8, ...}; a [B, H_kv, 1, 1] descale is a valid per-head
    // shape for the CPU reference but is not supported by the ASM engine.
    auto builder = createSdpaFwdFp8Graph(/*withDescale=*/true, /*descaleDims=*/{4, 8, 1, 1});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper))
        << "fp8 inputs with non-scalar (per-[B, H_kv]) descales must be rejected";
}

// The fp8 kernels run fp8 MFMA, so a 16-bit mma_core_mode (HALF, the cuDNN-frontend
// default that is accepted for bf16 inputs) does not describe them and must decline.
TEST_F(TestSdpaFwdPlanBuilder, IsApplicableRejectsFp8WithHalfMmaCoreMode)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;

    SKIP_IF_NO_DEVICES();
    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
    {
        GTEST_SKIP() << "fp8 forward kernels are gfx942-only";
    }

    auto unsetBuilder = createSdpaFwdFp8Graph(
        /*withDescale=*/true, {1, 1, 1, 1}, /*qStrides=*/{}, /*vStrides=*/{}, DataType::UNSET);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper unsetGraph(
        unsetBuilder.GetBufferPointer(), unsetBuilder.GetSize());
    ASSERT_TRUE(_planBuilder.isApplicable(_handle, unsetGraph))
        << "control: fp8 inputs with scalar descales and UNSET mma_core_mode are accepted";

    auto halfBuilder = createSdpaFwdFp8Graph(
        /*withDescale=*/true, {1, 1, 1, 1}, /*qStrides=*/{}, /*vStrides=*/{}, DataType::HALF);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper halfGraph(
        halfBuilder.GetBufferPointer(), halfBuilder.GetSize());
    EXPECT_FALSE(_planBuilder.isApplicable(_handle, halfGraph))
        << "fp8 inputs with mma_core_mode = HALF must be rejected";
}

TEST_F(TestSdpaFwdPlanBuilder, GetMaxWorkspaceSizeCalculatesCorrectly)
{
    // Create an SDPA graph with known dimensions (withStats = false by default)
    auto builder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph();

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    // Get the workspace size from the plan builder
    const Settings settings;
    const size_t workspaceSize = _planBuilder.getMaxWorkspaceSize(_handle, graphWrapper, settings);

    // Forward-only kernel uses LDS internally, no external workspace needed
    // LSE (when present) is an optional output tensor (stats_tensor_uid), not workspace
    // The default test graph has withStats = false, so workspace should be 0
    EXPECT_EQ(workspaceSize, 0u);
}

// =============================================================================
// Canonical mask-attribute policy (resolveDiagonalBand + getMaskType)
// =============================================================================
//
// These tests exercise the shared mask policy directly through
// plan_utils::resolveDiagonalBand and plan_utils::getMaskType rather than through isApplicable. A deprecated causal
// boolean is merged with the stored bounds and alignment (see
// plan_utils::resolveDiagonalBand); only setting both booleans throws. The policy is
// hardware-agnostic (it runs before any device dispatch and independent of the
// kernel registry), so testing the helper keeps the assertions meaningful on
// any device — including this gfx950 box. Driving the policy through
// isApplicable would not discriminate the policy result from the unrelated "no
// matching kernel" rejection that gfx950 produces for causal configurations
// (the gfx950 forward registry carries NO_MASK rows only).

// Extract the diagonal band from a graph buffer and classify the mask.
plan_utils::MaskType classifyMask(const flatbuffers::FlatBufferBuilder& builder)
{
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());
    const auto& attrs = graphWrapper.nodeWrappers()
                            .front()
                            ->attributesAs<hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes>();
    const auto band = plan_utils::resolveDiagonalBand(attrs);
    return plan_utils::getMaskType(band.leftBound, band.rightBound, band.alignment);
}

// cuDNN's default: no attn_scale_value and no scale tensor means no scaling.
TEST_F(TestSdpaFwdPlanBuilder, AbsentAttnScaleValueIsOne)
{
    auto builder = createSdpaFwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/false,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment::TOP_LEFT);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());
    const auto& attrs = graphWrapper.nodeWrappers()
                            .front()
                            ->attributesAs<hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes>();
    ASSERT_FALSE(attrs.attn_scale_value().has_value());
    EXPECT_EQ(plan_utils::attnScaleOrDefault(attrs), 1.0f);
}

class TestSdpaFwdDeprecatedCausalMask : public ::testing::TestWithParam<DeprecatedCausalMaskCase>
{
};

TEST_P(TestSdpaFwdDeprecatedCausalMask, Classifies)
{
    const auto& param = GetParam();
    auto builder = createSdpaFwdGraphWithMask(param.causalMask,
                                              param.causalMaskBottomRight,
                                              param.leftBound,
                                              param.rightBound,
                                              param.alignment);

    if(!param.expected.has_value())
    {
        EXPECT_THROW(classifyMask(builder), hipdnn_plugin_sdk::HipdnnPluginException);
        return;
    }
    plan_utils::MaskType maskType = plan_utils::MaskType::NO_MASK;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, param.expected.value());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestSdpaFwdDeprecatedCausalMask,
                         ::testing::ValuesIn(deprecatedCausalMaskCases()),
                         deprecatedCausalMaskCaseName);

// The graph's left_bound follows cuDNN (L keys ending at the diagonal, diagonal
// included); the band the kernels take is FlashAttention's window_size_left, L - 1.
TEST(TestSdpaFwdResolveDiagonalBand, ConvertsCudnnLeftBoundToWindowLeft)
{
    // Literal expectations, so the test does not restate the conversion it checks.
    for(const auto& [leftBound, expectedWindowLeft] : {std::pair<int64_t, int64_t>{-1, -1},
                                                       std::pair<int64_t, int64_t>{1, 0},
                                                       std::pair<int64_t, int64_t>{2, 1},
                                                       std::pair<int64_t, int64_t>{64, 63}})
    {
        auto builder = createSdpaFwdGraphWithMask(
            /*causalMask=*/false,
            /*causalMaskBottomRight=*/false,
            leftBound,
            /*rightBound=*/0,
            hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment::TOP_LEFT);
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
            builder.GetBufferPointer(), builder.GetSize());
        const auto& attrs
            = graphWrapper.nodeWrappers()
                  .front()
                  ->attributesAs<hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes>();

        const auto band = plan_utils::resolveDiagonalBand(attrs);
        EXPECT_EQ(band.leftBound, expectedWindowLeft) << "left_bound=" << leftBound;
        EXPECT_EQ(band.rightBound, 0);
    }
}

// =============================================================================
// plan_utils::makeCoveringBoundsUnbound
// =============================================================================
//
// A window that already covers the whole sequence span masks nothing on that
// side, so it classifies as the plain causal or unmasked kernel. Narrower windows
// stay sliding windows. Spans: top-left left=seqLenQ-1 right=seqLenKv-1;
// bottom-right left=seqLenKv-1 right=seqLenQ-1.

namespace
{
using hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment;

constexpr int64_t COVERING_SEQ_LEN_Q = 128;
constexpr int64_t COVERING_SEQ_LEN_KV = 512;

plan_utils::MaskType classifyAfterUnbounding(int64_t leftBound,
                                             int64_t rightBound,
                                             DiagonalAlignment alignment,
                                             int64_t seqLenQ = COVERING_SEQ_LEN_Q,
                                             int64_t seqLenKv = COVERING_SEQ_LEN_KV)
{
    const auto band = plan_utils::makeCoveringBoundsUnbound(
        {leftBound, rightBound, alignment}, seqLenQ, seqLenKv);
    return plan_utils::getMaskType(band.leftBound, band.rightBound, band.alignment);
}
} // namespace

TEST(TestSdpaFwdMakeCoveringBoundsUnbound, TopLeftCoveringLeftBoundIsCausal)
{
    EXPECT_EQ(classifyAfterUnbounding(127, 0, DiagonalAlignment::TOP_LEFT),
              plan_utils::MaskType::TOP_LEFT_CAUSAL);
    EXPECT_EQ(classifyAfterUnbounding(126, 0, DiagonalAlignment::TOP_LEFT),
              plan_utils::MaskType::SLIDING_WINDOW);
}

TEST(TestSdpaFwdMakeCoveringBoundsUnbound, TopLeftCoveringRightBoundIsNoMask)
{
    EXPECT_EQ(classifyAfterUnbounding(-1, 511, DiagonalAlignment::TOP_LEFT),
              plan_utils::MaskType::NO_MASK);
    EXPECT_EQ(classifyAfterUnbounding(-1, 510, DiagonalAlignment::TOP_LEFT),
              plan_utils::MaskType::SLIDING_WINDOW);
}

TEST(TestSdpaFwdMakeCoveringBoundsUnbound, BottomRightCoveringLeftBoundIsCausal)
{
    EXPECT_EQ(classifyAfterUnbounding(511, 0, DiagonalAlignment::BOTTOM_RIGHT),
              plan_utils::MaskType::BOTTOM_RIGHT_CAUSAL);
    EXPECT_EQ(classifyAfterUnbounding(510, 0, DiagonalAlignment::BOTTOM_RIGHT),
              plan_utils::MaskType::SLIDING_WINDOW);
}

TEST(TestSdpaFwdMakeCoveringBoundsUnbound, BottomRightCoveringRightBoundIsNoMask)
{
    EXPECT_EQ(classifyAfterUnbounding(-1, 127, DiagonalAlignment::BOTTOM_RIGHT),
              plan_utils::MaskType::NO_MASK);
    EXPECT_EQ(classifyAfterUnbounding(-1, 126, DiagonalAlignment::BOTTOM_RIGHT),
              plan_utils::MaskType::SLIDING_WINDOW);
}

TEST(TestSdpaFwdMakeCoveringBoundsUnbound, BothBoundsCoveringIsNoMask)
{
    EXPECT_EQ(classifyAfterUnbounding(127, 511, DiagonalAlignment::TOP_LEFT),
              plan_utils::MaskType::NO_MASK);
    EXPECT_EQ(classifyAfterUnbounding(-1, -1, DiagonalAlignment::TOP_LEFT),
              plan_utils::MaskType::NO_MASK);
}

TEST(TestSdpaFwdMakeCoveringBoundsUnbound, BoundsInsideSequencePassThroughUnchanged)
{
    for(const auto alignment : {DiagonalAlignment::TOP_LEFT, DiagonalAlignment::BOTTOM_RIGHT})
    {
        const auto band = plan_utils::makeCoveringBoundsUnbound(
            {10, 20, alignment}, COVERING_SEQ_LEN_Q, COVERING_SEQ_LEN_KV);
        EXPECT_EQ(band.leftBound, 10);
        EXPECT_EQ(band.rightBound, 20);
        EXPECT_EQ(band.alignment, alignment);
    }
}

// A raw int64 bound of 4294967298 would wrap to 2 in an int32 kernel field and
// shrink the window; once covering it is the unbounded request instead.
TEST(TestSdpaFwdMakeCoveringBoundsUnbound, HugeCoveringBoundBecomesUnboundedRequest)
{
    constexpr int64_t HUGE_BOUND = 4294967298;
    const auto band = plan_utils::makeCoveringBoundsUnbound(
        {HUGE_BOUND, 0, DiagonalAlignment::TOP_LEFT}, COVERING_SEQ_LEN_Q, COVERING_SEQ_LEN_KV);
    EXPECT_EQ(band.leftBound, -1);
    EXPECT_EQ(plan_utils::getMaskType(band.leftBound, band.rightBound, band.alignment),
              plan_utils::MaskType::TOP_LEFT_CAUSAL);

    constexpr auto SEQ_Q = static_cast<int32_t>(COVERING_SEQ_LEN_Q);
    constexpr auto SEQ_KV = static_cast<int32_t>(COVERING_SEQ_LEN_KV);
    EXPECT_EQ(plan_utils::computeMaskCoordinates(
                  static_cast<int32_t>(band.leftBound), 0, SEQ_Q, SEQ_KV, true),
              plan_utils::computeMaskCoordinates(SEQ_Q - 1, 0, SEQ_Q, SEQ_KV, true));
}

TEST(TestSdpaFwdMakeCoveringBoundsUnbound, HugeBoundBelowHugeSequenceStaysSlidingWindow)
{
    constexpr int64_t HUGE_BOUND = 4294967298;
    constexpr int64_t HUGE_SEQ_LEN = int64_t{1} << 40;
    const auto band = plan_utils::makeCoveringBoundsUnbound(
        {HUGE_BOUND, 0, DiagonalAlignment::TOP_LEFT}, HUGE_SEQ_LEN, HUGE_SEQ_LEN);
    EXPECT_EQ(band.leftBound, HUGE_BOUND);
    EXPECT_EQ(plan_utils::getMaskType(band.leftBound, band.rightBound, band.alignment),
              plan_utils::MaskType::SLIDING_WINDOW);
}

// A sequence length of 0 or 1 gives a span <= 0, so a causal right bound of 0
// covers it and the mask is dropped on that side.
TEST(TestSdpaFwdMakeCoveringBoundsUnbound, SingleTokenSequenceDropsCausalMaskOnItsAxis)
{
    EXPECT_EQ(classifyAfterUnbounding(-1, 0, DiagonalAlignment::TOP_LEFT, 128, 1),
              plan_utils::MaskType::NO_MASK);
    EXPECT_EQ(classifyAfterUnbounding(-1, 0, DiagonalAlignment::BOTTOM_RIGHT, 1, 128),
              plan_utils::MaskType::NO_MASK);
    EXPECT_EQ(classifyAfterUnbounding(-1, 0, DiagonalAlignment::TOP_LEFT, 1, 128),
              plan_utils::MaskType::TOP_LEFT_CAUSAL);
    EXPECT_EQ(classifyAfterUnbounding(-1, 0, DiagonalAlignment::BOTTOM_RIGHT, 128, 1),
              plan_utils::MaskType::BOTTOM_RIGHT_CAUSAL);
}

// Modern bounds-trio path (no deprecated boolean set). An unset bound is treated
// as unbounded (-1), so a partially specified trio still derives the mask it
// describes.

TEST_F(TestSdpaFwdPlanBuilder, MaskBoundsTrioRightZeroLeftUnsetDerivesTopLeftCausal)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    // The canonical causal request: left unbounded (unset) with right_bound=0.
    // An unset left bound must be read as unbounded, not as "no mask".
    auto builder = createSdpaFwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/false,
        flatbuffers::nullopt,
        flatbuffers::Optional<int64_t>(0),
        DiagonalAlignment::TOP_LEFT);

    plan_utils::MaskType maskType = plan_utils::MaskType::NO_MASK;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::TOP_LEFT_CAUSAL);
}

TEST_F(TestSdpaFwdPlanBuilder, MaskBoundsTrioRightZeroBottomRightDerivesBottomRightCausal)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    // Same causal request with bottom-right diagonal alignment.
    auto builder = createSdpaFwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/false,
        flatbuffers::nullopt,
        flatbuffers::Optional<int64_t>(0),
        DiagonalAlignment::BOTTOM_RIGHT);

    plan_utils::MaskType maskType = plan_utils::MaskType::NO_MASK;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::BOTTOM_RIGHT_CAUSAL);
}

TEST_F(TestSdpaFwdPlanBuilder, MaskBoundsTrioExplicitCausalBoundsDeriveTopLeftCausal)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    // Explicit unbounded-left (left=-1) with right_bound=0 matches the unset case.
    auto builder = createSdpaFwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/false,
        flatbuffers::Optional<int64_t>(-1),
        flatbuffers::Optional<int64_t>(0),
        DiagonalAlignment::TOP_LEFT);

    plan_utils::MaskType maskType = plan_utils::MaskType::NO_MASK;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::TOP_LEFT_CAUSAL);
}

TEST_F(TestSdpaFwdPlanBuilder, MaskBoundsTrioBothUnsetDerivesNoMask)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    auto builder = createSdpaFwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/false,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        DiagonalAlignment::TOP_LEFT);

    plan_utils::MaskType maskType = plan_utils::MaskType::SLIDING_WINDOW;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::NO_MASK);
}

TEST_F(TestSdpaFwdPlanBuilder, MaskBoundsTrioBothUnboundedDerivesNoMask)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    auto builder = createSdpaFwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/false,
        flatbuffers::Optional<int64_t>(-1),
        flatbuffers::Optional<int64_t>(-1),
        DiagonalAlignment::TOP_LEFT);

    plan_utils::MaskType maskType = plan_utils::MaskType::SLIDING_WINDOW;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::NO_MASK);
}

TEST_F(TestSdpaFwdPlanBuilder, MaskBoundsTrioSymmetricWindowDerivesSlidingWindow)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    auto builder = createSdpaFwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/false,
        flatbuffers::Optional<int64_t>(64),
        flatbuffers::Optional<int64_t>(64),
        DiagonalAlignment::TOP_LEFT);

    plan_utils::MaskType maskType = plan_utils::MaskType::NO_MASK;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::SLIDING_WINDOW);
}

TEST_F(TestSdpaFwdPlanBuilder, MaskBoundsTrioLeftOnlyDerivesSlidingWindow)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    // A bounded left with an unset (unbounded) right is a one-sided window.
    auto builder = createSdpaFwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/false,
        flatbuffers::Optional<int64_t>(64),
        flatbuffers::nullopt,
        DiagonalAlignment::TOP_LEFT);

    plan_utils::MaskType maskType = plan_utils::MaskType::NO_MASK;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::SLIDING_WINDOW);
}

// =============================================================================
// Byte-stride overflow guard
// =============================================================================

// Helper: create a forward SDPA graph with explicit strides for all tensors.
auto createSdpaFwdGraphWithStrides(const std::vector<int64_t>& dims,
                                   const std::vector<int64_t>& qStrides,
                                   const std::vector<int64_t>& kStrides,
                                   const std::vector<int64_t>& vStrides,
                                   const std::vector<int64_t>& oStrides)
{
    const std::vector<int64_t> kDims = {dims[0], dims[1], dims[2], dims[3]};
    const std::vector<int64_t> oDims = {dims[0], dims[1], dims[2], dims[3]};

    return hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        dims,
        qStrides,
        kDims,
        kStrides,
        dims,
        vStrides,
        oDims,
        oStrides,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
}

TEST_F(TestSdpaFwdPlanBuilder, IsApplicableRejectsOversizedByteStrides)
{
    SKIP_IF_NO_DEVICES();

    const std::string deviceString
        = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }

    // A batch stride just above UINT32_MAX / 2 will overflow when scaled to
    // bytes (stride * sizeof(bf16)).  The engine must decline via isApplicable.
    constexpr int64_t K_OVERFLOW_STRIDE = static_cast<int64_t>(UINT32_MAX) / 2 + 1;
    const std::vector<int64_t> dims = {4, 8, 256, 128};
    const std::vector<int64_t> overflowStrides
        = {K_OVERFLOW_STRIDE, static_cast<int64_t>(256 * 128), 128, 1};
    const std::vector<int64_t> normalStrides = hipdnn_data_sdk::utilities::generateStrides(dims);

    // Q has overflow stride
    auto builderQ = createSdpaFwdGraphWithStrides(
        dims, overflowStrides, normalStrides, normalStrides, normalStrides);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper gwQ(
        builderQ.GetBufferPointer(), builderQ.GetSize());
    EXPECT_FALSE(_planBuilder.isApplicable(_handle, gwQ));

    // K has overflow stride
    auto builderK = createSdpaFwdGraphWithStrides(
        dims, normalStrides, overflowStrides, normalStrides, normalStrides);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper gwK(
        builderK.GetBufferPointer(), builderK.GetSize());
    EXPECT_FALSE(_planBuilder.isApplicable(_handle, gwK));

    // Normal strides pass
    auto builderOk = createSdpaFwdGraphWithStrides(
        dims, normalStrides, normalStrides, normalStrides, normalStrides);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper gwOk(
        builderOk.GetBufferPointer(), builderOk.GetSize());
    EXPECT_TRUE(_planBuilder.isApplicable(_handle, gwOk));
}

TEST_F(TestSdpaFwdPlanBuilder, IsApplicableRejectsSinkToken)
{
    SKIP_IF_NO_DEVICES();

    const std::string deviceString
        = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }

    // The same graph is accepted without the sink and declined with it: the vendored kernels
    // cannot add the sink logit to the softmax denominator.
    auto base = createSdpaFwdGraph();
    auto withSinkBuilder = addSinkToken(base);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper withSink(
        withSinkBuilder.GetBufferPointer(), withSinkBuilder.GetSize());
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper withoutSink(
        base.GetBufferPointer(), base.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_handle, withoutSink));
    EXPECT_FALSE(_planBuilder.isApplicable(_handle, withSink));
}

// =============================================================================
// Kernarg range guards
// =============================================================================

// The predicates every per-field guard is built from, at their exact
// boundaries. Pure functions, so these run without a device.

TEST(TestSdpaKernargRanges, DimFitsU32AcceptsItsLimit)
{
    EXPECT_TRUE(plan_utils::dimFitsU32("batch", static_cast<int64_t>(UINT32_MAX)));
}

TEST(TestSdpaKernargRanges, DimFitsU32RejectsOnePastItsLimit)
{
    EXPECT_FALSE(plan_utils::dimFitsU32("batch", static_cast<int64_t>(UINT32_MAX) + 1));
}

TEST(TestSdpaKernargRanges, DimFitsU32RejectsANegativeDimension)
{
    EXPECT_FALSE(plan_utils::dimFitsU32("batch", -1));
}

TEST(TestSdpaKernargRanges, DimFitsI32AcceptsItsLimit)
{
    EXPECT_TRUE(plan_utils::dimFitsI32("seqlen_q", static_cast<int64_t>(INT32_MAX)));
}

TEST(TestSdpaKernargRanges, DimFitsI32RejectsOnePastItsLimit)
{
    EXPECT_FALSE(plan_utils::dimFitsI32("seqlen_q", static_cast<int64_t>(INT32_MAX) + 1));
}

TEST(TestSdpaKernargRanges, DimFitsI32RejectsANegativeDimension)
{
    EXPECT_FALSE(plan_utils::dimFitsI32("seqlen_q", -1));
}

TEST(TestSdpaKernargRanges, ByteStrideFitsU32AcceptsItsLimit)
{
    EXPECT_TRUE(plan_utils::byteStrideFitsU32("stride_q", static_cast<int64_t>(UINT32_MAX) / 2, 2));
}

TEST(TestSdpaKernargRanges, ByteStrideFitsU32RejectsOnePastItsLimit)
{
    EXPECT_FALSE(
        plan_utils::byteStrideFitsU32("stride_q", static_cast<int64_t>(UINT32_MAX) / 2 + 1, 2));
}

// The predicate divides instead of multiplying, so an element count whose byte
// product does not even fit in int64 is declined rather than wrapped into
// range.
TEST(TestSdpaKernargRanges, ByteStrideFitsU32DeclinesAnElementCountThatOverflowsTheProduct)
{
    EXPECT_FALSE(plan_utils::byteStrideFitsU32("stride_q", (int64_t{1} << 62) + 1, 2));
}

// Both Ts kernargs fold a tile size into the element size, which shrinks the
// accepted element range by that factor.
TEST(TestSdpaKernargRanges, ByteStrideFitsU32FoldsInATileFactor)
{
    constexpr int64_t K_TILE_BYTES = int64_t{2} * 192;
    EXPECT_TRUE(plan_utils::byteStrideFitsU32(
        "Ts", static_cast<int64_t>(UINT32_MAX) / K_TILE_BYTES, K_TILE_BYTES));
    EXPECT_FALSE(plan_utils::byteStrideFitsU32(
        "Ts", static_cast<int64_t>(UINT32_MAX) / K_TILE_BYTES + 1, K_TILE_BYTES));
}

TEST(TestSdpaKernargRanges, ByteStrideFitsU32RejectsANonPositiveElementSize)
{
    EXPECT_FALSE(plan_utils::byteStrideFitsU32("Ts", 1, 0));
}

// Build a forward graph with per-tensor dims and one explicit stride set, so a
// dimension can be driven to the limit of its field without the contiguous
// stride for that shape tripping the byte-stride guard first. The strides
// describe a [1, 1, 256, 128] tile; applicability ties nothing else to them.
// Only descriptors are built, so the dims cost no memory.
auto createSdpaFwdGraphForDims(const std::vector<int64_t>& qDims,
                               const std::vector<int64_t>& kDims,
                               const std::vector<int64_t>& vDims)
{
    const std::vector<int64_t> oDims = {qDims[0], qDims[1], qDims[2], vDims[3]};
    const std::vector<int64_t> strides = {32768, 32768, 128, 1};
    return hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        qDims,
        strides,
        kDims,
        strides,
        vDims,
        strides,
        oDims,
        strides,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
}

struct FwdDimRangeCase
{
    const char* label;
    std::vector<int64_t> qDims;
    std::vector<int64_t> kDims;
    std::vector<int64_t> vDims;
};

constexpr int64_t K_U32_DIM_LIMIT = static_cast<int64_t>(UINT32_MAX);
constexpr int64_t K_I32_DIM_LIMIT = static_cast<int64_t>(INT32_MAX);

// Batch and the head counts reach the kernel through a uint32 field or an
// unsigned grid dimension; the sequence lengths take the signed limit because
// the launch grid's ceil-division is evaluated in unsigned int.
TEST_F(TestSdpaFwdPlanBuilder, IsApplicableAcceptsDimensionsAtTheir32BitLimits)
{
    SKIP_IF_NO_DEVICES();

    const std::string deviceString
        = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }

    const std::vector<FwdDimRangeCase> cases = {
        {"batch at UINT32_MAX",
         {K_U32_DIM_LIMIT, 8, 256, 128},
         {K_U32_DIM_LIMIT, 8, 256, 128},
         {K_U32_DIM_LIMIT, 8, 256, 128}},
        {"nhead_q at UINT32_MAX",
         {4, K_U32_DIM_LIMIT, 256, 128},
         {4, 8, 256, 128},
         {4, 8, 256, 128}},
        {"nhead_kv at UINT32_MAX",
         {4, 8, 256, 128},
         {4, K_U32_DIM_LIMIT, 256, 128},
         {4, K_U32_DIM_LIMIT, 256, 128}},
        {"seqlen_q at INT32_MAX", {4, 8, K_I32_DIM_LIMIT, 128}, {4, 8, 256, 128}, {4, 8, 256, 128}},
        {"seqlen_kv at INT32_MAX",
         {4, 8, 256, 128},
         {4, 8, K_I32_DIM_LIMIT, 128},
         {4, 8, K_I32_DIM_LIMIT, 128}},
    };

    for(const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.label);
        auto builder = createSdpaFwdGraphForDims(testCase.qDims, testCase.kDims, testCase.vDims);
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper gw(
            builder.GetBufferPointer(), builder.GetSize());
        EXPECT_TRUE(_planBuilder.isApplicable(_handle, gw));
    }
}

TEST_F(TestSdpaFwdPlanBuilder, IsApplicableDeclinesDimensionsPastTheir32BitLimits)
{
    SKIP_IF_NO_DEVICES();

    const std::string deviceString
        = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }

    const std::vector<FwdDimRangeCase> cases = {
        {"batch one past UINT32_MAX",
         {K_U32_DIM_LIMIT + 1, 8, 256, 128},
         {K_U32_DIM_LIMIT + 1, 8, 256, 128},
         {K_U32_DIM_LIMIT + 1, 8, 256, 128}},
        {"nhead_q one past UINT32_MAX",
         {4, K_U32_DIM_LIMIT + 1, 256, 128},
         {4, 8, 256, 128},
         {4, 8, 256, 128}},
        {"nhead_kv one past UINT32_MAX",
         {4, 8, 256, 128},
         {4, K_U32_DIM_LIMIT + 1, 256, 128},
         {4, K_U32_DIM_LIMIT + 1, 256, 128}},
        {"seqlen_q one past INT32_MAX",
         {4, 8, K_I32_DIM_LIMIT + 1, 128},
         {4, 8, 256, 128},
         {4, 8, 256, 128}},
        {"seqlen_kv one past INT32_MAX",
         {4, 8, 256, 128},
         {4, 8, K_I32_DIM_LIMIT + 1, 128},
         {4, 8, K_I32_DIM_LIMIT + 1, 128}},
    };

    for(const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.label);
        auto builder = createSdpaFwdGraphForDims(testCase.qDims, testCase.kDims, testCase.vDims);
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper gw(
            builder.GetBufferPointer(), builder.GetSize());
        EXPECT_FALSE(_planBuilder.isApplicable(_handle, gw));
    }
}

// The head dims are narrowed to int for the kernel-registry lookup, so a head
// dim whose low 32 bits are a supported value used to resolve a kernel built
// for that supported value. The registry set {64, 128, 192} is what bounds the
// head dims in practice, which is why there is no case at INT32_MAX here.
TEST_F(TestSdpaFwdPlanBuilder, IsApplicableDeclinesHeadDimsThatTruncateToASupportedValue)
{
    SKIP_IF_NO_DEVICES();

    const std::string deviceString
        = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }

    constexpr int64_t K_WRAPS_TO_128 = (int64_t{1} << 32) + 128;

    const std::vector<FwdDimRangeCase> cases = {
        {"supported head dims (control)", {4, 8, 256, 128}, {4, 8, 256, 128}, {4, 8, 256, 128}},
        {"hdim_qk wraps to 128",
         {4, 8, 256, K_WRAPS_TO_128},
         {4, 8, 256, K_WRAPS_TO_128},
         {4, 8, 256, 128}},
        {"hdim_v wraps to 128", {4, 8, 256, 128}, {4, 8, 256, 128}, {4, 8, 256, K_WRAPS_TO_128}},
    };

    for(const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.label);
        auto builder = createSdpaFwdGraphForDims(testCase.qDims, testCase.kDims, testCase.vDims);
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper gw(
            builder.GetBufferPointer(), builder.GetSize());
        const bool isControl = testCase.label == std::string("supported head dims (control)");
        EXPECT_EQ(_planBuilder.isApplicable(_handle, gw), isControl);
    }
}

// ts_qo of the row isApplicable resolves for an hd128 no-mask batch graph of
// `dtype`, selected by the same rule getKernelNameKey applies. gfx942 also
// matches bf16_cvt, which getBf16ConvertMode sets to RTNE (0) for bf16 and
// RTNA (1) for fp8.
std::optional<int> resolveFwdTileSizeQo(const std::string& archId, const std::string& dtype)
{
    const int expectedBf16Cvt = (dtype == "fp8bf16") ? 1 : 0;
    for(const auto& entry : cfg_fmha_fwd)
    {
        const auto& cfg = entry.second;
        if(cfg.arch == archId && cfg.dtype == dtype && cfg.hdim_q == 128 && cfg.hdim_v == 128
           && cfg.mask == 0 && cfg.mode == 0
           && (archId == "gfx950" || cfg.bf16_cvt == expectedBf16Cvt))
        {
            return cfg.ts_qo;
        }
    }
    return std::nullopt;
}

// s_Ts is tileSizeQo * stride_q * bytes-per-element, computed in unsigned int.
// The per-field stride guard only bounds stride_q * bytes-per-element, so the
// tile factor is what this guard adds: the accepted stride shrinks by ts_qo.
// The element size is the resolved input size, two bytes for bf16 and one for
// fp8, so the fp8 boundary sits at twice the bf16 one for the same ts_qo.
TEST_F(TestSdpaFwdPlanBuilder, IsApplicableBoundsTheTsProductByTheResolvedTileSize)
{
    SKIP_IF_NO_DEVICES();

    const std::string deviceString
        = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }

    const auto tileSizeQo = resolveFwdTileSizeQo(deviceString, "bf16");
    ASSERT_TRUE(tileSizeQo.has_value()) << "no bf16 hd128 no-mask row for " << deviceString;

    constexpr int64_t K_BF16_BYTES = 2;
    const int64_t limit = static_cast<int64_t>(UINT32_MAX) / (K_BF16_BYTES * *tileSizeQo);
    // One past the Ts limit must still clear the per-field stride guard, or the
    // test would prove nothing about Ts.
    ASSERT_LE((limit + 1) * K_BF16_BYTES, static_cast<int64_t>(UINT32_MAX));

    const std::vector<int64_t> dims = {4, 8, 256, 128};
    const std::vector<int64_t> normalStrides = hipdnn_data_sdk::utilities::generateStrides(dims);

    const std::vector<int64_t> atLimit = {limit, limit, limit, 1};
    auto builderAtLimit
        = createSdpaFwdGraphWithStrides(dims, atLimit, normalStrides, normalStrides, normalStrides);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper gwAtLimit(
        builderAtLimit.GetBufferPointer(), builderAtLimit.GetSize());
    EXPECT_TRUE(_planBuilder.isApplicable(_handle, gwAtLimit));

    const std::vector<int64_t> pastLimit = {limit + 1, limit + 1, limit + 1, 1};
    auto builderPastLimit = createSdpaFwdGraphWithStrides(
        dims, pastLimit, normalStrides, normalStrides, normalStrides);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper gwPastLimit(
        builderPastLimit.GetBufferPointer(), builderPastLimit.GetSize());
    EXPECT_FALSE(_planBuilder.isApplicable(_handle, gwPastLimit));

    // fp8 forward kernels ship for gfx942 only. Their kernarg builder scales the
    // strides by one byte, so bounding the fp8 Ts at the bf16 size would decline
    // descriptors the kernel runs.
    if(deviceString != "gfx942")
    {
        return;
    }

    const auto fp8TileSizeQo = resolveFwdTileSizeQo(deviceString, "fp8bf16");
    ASSERT_TRUE(fp8TileSizeQo.has_value()) << "no fp8bf16 hd128 no-mask row for " << deviceString;

    constexpr int64_t K_FP8_BYTES = 1;
    const int64_t fp8Limit = static_cast<int64_t>(UINT32_MAX) / (K_FP8_BYTES * *fp8TileSizeQo);
    ASSERT_LE((fp8Limit + 1) * K_FP8_BYTES, static_cast<int64_t>(UINT32_MAX));
    // The bf16 element size would have cut this in half, which is the reach the
    // resolved element size restores.
    ASSERT_GT(fp8Limit, static_cast<int64_t>(UINT32_MAX) / (K_BF16_BYTES * *fp8TileSizeQo));

    const std::vector<int64_t> fp8AtLimit = {fp8Limit, fp8Limit, fp8Limit, 1};
    auto fp8BuilderAtLimit
        = createSdpaFwdFp8Graph(/*withDescale=*/true, /*descaleDims=*/{1, 1, 1, 1}, fp8AtLimit);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper gwFp8AtLimit(
        fp8BuilderAtLimit.GetBufferPointer(), fp8BuilderAtLimit.GetSize());
    EXPECT_TRUE(_planBuilder.isApplicable(_handle, gwFp8AtLimit));

    const std::vector<int64_t> fp8PastLimit = {fp8Limit + 1, fp8Limit + 1, fp8Limit + 1, 1};
    auto fp8BuilderPastLimit
        = createSdpaFwdFp8Graph(/*withDescale=*/true, /*descaleDims=*/{1, 1, 1, 1}, fp8PastLimit);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper gwFp8PastLimit(
        fp8BuilderPastLimit.GetBufferPointer(), fp8BuilderPastLimit.GetSize());
    EXPECT_FALSE(_planBuilder.isApplicable(_handle, gwFp8PastLimit));
}

// The forward byte-stride guard checks the batch, head and sequence stride of V
// one field at a time (batch_stride_v, nhead_stride_v, stride_v). Each case
// moves one V stride to its bound and leaves every other tensor contiguous, so
// the at-limit graph is accepted and the past-limit graph can only be declined
// by that one V field. Q, K and O stay normal, which keeps the Ts guard (a Q
// property) out of the picture.
struct VStrideFieldCase
{
    const char* label;
    size_t strideIndex;
};

const std::vector<VStrideFieldCase> K_V_STRIDE_FIELDS = {
    {"batch_stride_v", 0},
    {"nhead_stride_v", 1},
    {"stride_v", 2},
};

std::vector<int64_t>
    contiguousStridesWithField(const std::vector<int64_t>& dims, size_t strideIndex, int64_t value)
{
    std::vector<int64_t> strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    strides[strideIndex] = value;
    return strides;
}

TEST_F(TestSdpaFwdPlanBuilder, IsApplicableBoundsEachBf16VByteStrideField)
{
    SKIP_IF_NO_DEVICES();

    const std::string deviceString
        = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }

    // bf16 scales the element stride by two bytes, so the largest accepted
    // element stride is UINT32_MAX / 2.
    constexpr int64_t K_BF16_STRIDE_LIMIT = static_cast<int64_t>(UINT32_MAX) / 2;
    const std::vector<int64_t> dims = {4, 8, 256, 128};
    const std::vector<int64_t> normalStrides = hipdnn_data_sdk::utilities::generateStrides(dims);

    for(const auto& field : K_V_STRIDE_FIELDS)
    {
        SCOPED_TRACE(field.label);

        auto atLimitBuilder = createSdpaFwdGraphWithStrides(
            dims,
            normalStrides,
            normalStrides,
            contiguousStridesWithField(dims, field.strideIndex, K_BF16_STRIDE_LIMIT),
            normalStrides);
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper gwAtLimit(
            atLimitBuilder.GetBufferPointer(), atLimitBuilder.GetSize());
        EXPECT_TRUE(_planBuilder.isApplicable(_handle, gwAtLimit));

        auto pastLimitBuilder = createSdpaFwdGraphWithStrides(
            dims,
            normalStrides,
            normalStrides,
            contiguousStridesWithField(dims, field.strideIndex, K_BF16_STRIDE_LIMIT + 1),
            normalStrides);
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper gwPastLimit(
            pastLimitBuilder.GetBufferPointer(), pastLimitBuilder.GetSize());
        EXPECT_FALSE(_planBuilder.isApplicable(_handle, gwPastLimit));
    }
}

// fp8 forward kernels ship for gfx942 only. The kernarg builder scales the Q/K/V
// strides by one byte, so V's bound is UINT32_MAX elements, twice the bf16 one.
// The element stride that bf16 declines (UINT32_MAX / 2 + 1) is therefore a
// case fp8 must still accept, which is what the resolved element size restores.
TEST_F(TestSdpaFwdPlanBuilder, IsApplicableBoundsEachFp8VByteStrideField)
{
    SKIP_IF_NO_DEVICES();

    const std::string deviceString
        = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString != "gfx942")
    {
        GTEST_SKIP();
    }

    constexpr auto K_FP8_STRIDE_LIMIT = static_cast<int64_t>(UINT32_MAX);
    constexpr int64_t K_BF16_PAST_LIMIT = static_cast<int64_t>(UINT32_MAX) / 2 + 1;
    const std::vector<int64_t> dims = {4, 8, 256, 128};

    const auto isApplicableWithV = [&](size_t strideIndex, int64_t value) {
        auto builder = createSdpaFwdFp8Graph(/*withDescale=*/true,
                                             /*descaleDims=*/{1, 1, 1, 1},
                                             /*qStrides=*/{},
                                             contiguousStridesWithField(dims, strideIndex, value));
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper gw(
            builder.GetBufferPointer(), builder.GetSize());
        return _planBuilder.isApplicable(_handle, gw);
    };

    for(const auto& field : K_V_STRIDE_FIELDS)
    {
        SCOPED_TRACE(field.label);
        EXPECT_TRUE(isApplicableWithV(field.strideIndex, K_BF16_PAST_LIMIT));
        EXPECT_TRUE(isApplicableWithV(field.strideIndex, K_FP8_STRIDE_LIMIT));
        EXPECT_FALSE(isApplicableWithV(field.strideIndex, K_FP8_STRIDE_LIMIT + 1));
    }
}

// =============================================================================
// Tensor layouts the forward kernels can and cannot address
// =============================================================================

// Replaces the strides, and the dims when non-empty, of the tensor called name.
// createSdpaFwdGraph names its tensors q, k, v, o and stats.
struct LayoutOverride
{
    std::string name;
    std::vector<int64_t> strides;
    std::vector<int64_t> dims;
};

// Copy of a forward graph with the given tensor layouts applied.
flatbuffers::FlatBufferBuilder withLayouts(const flatbuffers::FlatBufferBuilder& base,
                                           const std::vector<LayoutOverride>& layouts)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    GraphT graph;
    GetGraph(base.GetBufferPointer())->UnPackTo(&graph);
    for(const auto& layout : layouts)
    {
        const auto it
            = std::find_if(graph.tensors.begin(), graph.tensors.end(), [&](const auto& tensor) {
                  return tensor->name == layout.name;
              });
        if(it == graph.tensors.end())
        {
            throw std::logic_error("withLayouts: no tensor named " + layout.name);
        }
        (*it)->strides = layout.strides;
        if(!layout.dims.empty())
        {
            (*it)->dims = layout.dims;
        }
    }

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(Graph::Pack(builder, &graph));
    return builder;
}

constexpr std::array<const char*, 4> K_IO_TENSOR_NAMES = {"q", "k", "v", "o"};

// The same strides on the named tensors.
std::vector<LayoutOverride> onTensors(const std::vector<const char*>& names,
                                      const std::vector<int64_t>& strides)
{
    std::vector<LayoutOverride> layouts;
    layouts.reserve(names.size());
    for(const char* name : names)
    {
        layouts.push_back({name, strides, {}});
    }
    return layouts;
}

// The same strides on all four Q/K/V/O tensors.
std::vector<LayoutOverride> onAllIoTensors(const std::vector<int64_t>& strides)
{
    return onTensors({K_IO_TENSOR_NAMES.begin(), K_IO_TENSOR_NAMES.end()}, strides);
}

// Packed bf16 forward graph with no mask. Q is qDims [B, H_q, S_q, D], V is vDims
// [B, H_kv, S_kv, D_v], K is [B, H_kv, S_kv, D] and O is [B, H_q, S_q, D_v]. With
// withStats, stats are packed [B, H_q, S_q, 1].
flatbuffers::FlatBufferBuilder createLayoutGraph(const std::vector<int64_t>& qDims,
                                                 const std::vector<int64_t>& vDims,
                                                 bool withStats = false)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    return createSdpaFwdGraph(qDims,
                              vDims,
                              DataType::BFLOAT16,
                              DataType::FLOAT,
                              /*withAttnMask=*/false,
                              /*withScale=*/false,
                              withStats);
}

// Layouts the kernels address correctly. These were GPU-verified with bf16 on
// gfx942: BSHD, padded and odd row strides, odd head and batch strides, broadcast
// (stride-0) inputs, and packed stats or stats with a padded head stride. The
// padded and odd strides were also verified on gfx950. The size-1-axis cases and
// rank-3 stats differ from them only in strides or dims no kernel reads.
TEST_F(TestSdpaFwdPlanBuilder, IsApplicableAcceptsAddressableLayouts)
{
    SKIP_IF_NO_DEVICES();

    const std::string deviceString
        = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }

    // B=2 and H=2 so that every batch and head stride is used.
    const std::vector<int64_t> dims = {2, 2, 256, 128};
    // [B, H, S, D] dims stored as [B, S, H, D].
    const std::vector<int64_t> bshd = {int64_t{256} * 2 * 128, 128, int64_t{2} * 128, 1};
    // BHSD with each row padded from D=128 to D+8, D+2 or D+1 elements.
    const std::vector<int64_t> rowsPlus8 = {int64_t{2} * 256 * 136, int64_t{256} * 136, 136, 1};
    const std::vector<int64_t> rowsPlus2 = {int64_t{2} * 256 * 130, int64_t{256} * 130, 130, 1};
    const std::vector<int64_t> rowsPlus1 = {int64_t{2} * 256 * 129, int64_t{256} * 129, 129, 1};
    // BHSD with an odd head or batch stride.
    const std::vector<int64_t> oddHead
        = {int64_t{2} * (int64_t{256} * 128 + 1), int64_t{256} * 128 + 1, 128, 1};
    const std::vector<int64_t> oddBatch = {int64_t{2} * 256 * 128 + 1, int64_t{256} * 128, 128, 1};
    // BHSD with the batch or head axis broadcast.
    const std::vector<int64_t> zeroBatch = {0, int64_t{256} * 128, 128, 1};
    const std::vector<int64_t> zeroHead = {int64_t{256} * 128, 0, 128, 1};

    std::vector<std::pair<GraphTest, bool>> applicabilityTests;
    applicabilityTests.emplace_back(GraphTest{createLayoutGraph(dims, dims), "packed BHSD"}, true);
    applicabilityTests.emplace_back(
        GraphTest{withLayouts(createLayoutGraph(dims, dims), onAllIoTensors(bshd)),
                  "BSHD on Q, K, V and O"},
        true);
    applicabilityTests.emplace_back(
        GraphTest{withLayouts(createLayoutGraph(dims, dims), {{"o", bshd, {}}}),
                  "O BSHD while Q, K and V are BHSD"},
        true);
    applicabilityTests.emplace_back(
        GraphTest{withLayouts(createLayoutGraph(dims, dims), onAllIoTensors(rowsPlus8)),
                  "row stride D+8 on Q, K, V and O"},
        true);
    applicabilityTests.emplace_back(
        GraphTest{withLayouts(createLayoutGraph(dims, dims), onAllIoTensors(rowsPlus2)),
                  "row stride D+2 on Q, K, V and O"},
        true);
    applicabilityTests.emplace_back(
        GraphTest{withLayouts(createLayoutGraph({2, 2, 256, 192}, dims),
                              {{"q", {int64_t{2} * 256 * 194, int64_t{256} * 194, 194, 1}, {}},
                               {"k", {int64_t{2} * 256 * 194, int64_t{256} * 194, 194, 1}, {}},
                               {"v", rowsPlus2, {}},
                               {"o", rowsPlus2, {}}}),
                  "hd192 with row stride D+2 on Q, K, V and O"},
        true);
    // Unlike the backward kernels, the forward kernels addressed odd strides
    // correctly on every tensor and axis, with bf16 on gfx942 and gfx950. The fp8
    // kernels were not run with them.
    for(const char* name : K_IO_TENSOR_NAMES)
    {
        applicabilityTests.emplace_back(
            GraphTest{withLayouts(createLayoutGraph(dims, dims), {{name, rowsPlus1, {}}}),
                      std::string(name) + " row stride D+1"},
            true);
        applicabilityTests.emplace_back(
            GraphTest{withLayouts(createLayoutGraph(dims, dims), {{name, oddHead, {}}}),
                      std::string(name) + " odd head stride"},
            true);
        applicabilityTests.emplace_back(
            GraphTest{withLayouts(createLayoutGraph(dims, dims), {{name, oddBatch, {}}}),
                      std::string(name) + " odd batch stride"},
            true);
    }
    // A single row of D elements fits any row stride of at least D.
    applicabilityTests.emplace_back(GraphTest{createLayoutGraph({2, 2, 1, 128}, dims), "S_q=1"},
                                    true);
    applicabilityTests.emplace_back(GraphTest{withLayouts(createLayoutGraph({2, 2, 1, 128}, dims),
                                                          {{"o", {256, 128, 256, 1}, {}}}),
                                              "S_q=1 with O BSHD"},
                                    true);
    applicabilityTests.emplace_back(GraphTest{createLayoutGraph(dims, {2, 2, 1, 128}), "S_kv=1"},
                                    true);
    // A zero batch or head stride on an input (a broadcast) only re-reads one
    // slice.
    applicabilityTests.emplace_back(
        GraphTest{withLayouts(createLayoutGraph(dims, dims), onTensors({"k", "v"}, zeroBatch)),
                  "K and V batch stride 0"},
        true);
    applicabilityTests.emplace_back(
        GraphTest{withLayouts(createLayoutGraph(dims, dims), onTensors({"k", "v"}, zeroHead)),
                  "K and V head stride 0"},
        true);
    applicabilityTests.emplace_back(
        GraphTest{withLayouts(createLayoutGraph(dims, dims), {{"q", zeroBatch, {}}}),
                  "Q batch stride 0"},
        true);
    // On a length-1 axis a zero stride does not alias, also on O.
    applicabilityTests.emplace_back(
        GraphTest{withLayouts(createLayoutGraph({1, 2, 256, 128}, {1, 2, 256, 128}),
                              {{"o", zeroBatch, {}}}),
                  "O batch stride 0 with B=1"},
        true);
    applicabilityTests.emplace_back(
        GraphTest{withLayouts(createLayoutGraph({2, 1, 256, 128}, {2, 1, 256, 128}),
                              {{"o", zeroHead, {}}}),
                  "O head stride 0 with H=1"},
        true);

    // Stats, which the kernels write at ((b * H_q + h) * head stride + s) * 4 bytes.
    applicabilityTests.emplace_back(GraphTest{createLayoutGraph(dims, dims, true), "packed stats"},
                                    true);
    applicabilityTests.emplace_back(GraphTest{withLayouts(createLayoutGraph(dims, dims, true),
                                                          {{"stats", {512, 256, 1}, {2, 2, 256}}}),
                                              "rank-3 packed stats"},
                                    true);
    applicabilityTests.emplace_back(
        GraphTest{withLayouts(createLayoutGraph(dims, dims, true), onAllIoTensors(bshd)),
                  "BSHD on Q, K, V and O with packed stats"},
        true);
    applicabilityTests.emplace_back(
        GraphTest{createLayoutGraph({2, 8, 256, 128}, {2, 2, 256, 128}, true),
                  "GQA with packed stats"},
        true);
    applicabilityTests.emplace_back(
        GraphTest{createLayoutGraph({2, 2, 256, 192}, dims, true), "hd192 with packed stats"},
        true);
    applicabilityTests.emplace_back(
        GraphTest{
            withLayouts(createLayoutGraph(dims, dims, true), {{"stats", {640, 320, 1, 1}, {}}}),
            "stats head stride S_q + 64, batch stride H_q * (S_q + 64)"},
        true);
    applicabilityTests.emplace_back(
        GraphTest{createLayoutGraph({2, 1, 256, 128}, {2, 1, 256, 128}, true),
                  "packed stats with H_q=1"},
        true);
    // No kernel uses the stride of a size-1 stats batch or sequence axis, nor the
    // head stride when there is a single (batch, head) slice.
    applicabilityTests.emplace_back(
        GraphTest{withLayouts(createLayoutGraph({1, 2, 256, 128}, {1, 2, 256, 128}, true),
                              {{"stats", {12345, 256, 1, 1}, {}}}),
                  "unused stats batch stride, B=1"},
        true);
    applicabilityTests.emplace_back(
        GraphTest{withLayouts(createLayoutGraph({2, 2, 1, 128}, dims, true),
                              {{"stats", {2, 1, 5, 1}, {}}}),
                  "unused stats seq stride, S_q=1"},
        true);
    applicabilityTests.emplace_back(
        GraphTest{withLayouts(createLayoutGraph({1, 1, 256, 128}, {1, 1, 256, 128}, true),
                              {{"stats", {256, 7, 1, 1}, {}}}),
                  "unused stats head stride, B=1 H_q=1"},
        true);
    applicabilityTests.emplace_back(GraphTest{withLayouts(createLayoutGraph(dims, dims, true),
                                                          {{"stats", {512, 256, 1, 7}, {}}}),
                                              "arbitrary stride on the size-1 last stats dim"},
                                    true);

    for(const auto& [test, applicability] : applicabilityTests)
    {
        EXPECT_EQ(_planBuilder.isApplicable(_handle, test.graphWrapper()), applicability)
            << test.message;
    }
}

// No kernel takes a head-dim stride, so a strided head dim on Q, K, V or O must
// decline. On gfx942 each of these gave a wrong O, and a stride-2 O left half of
// its elements unwritten.
TEST_F(TestSdpaFwdPlanBuilder, IsApplicableRejectsNonUnitHeadDimStride)
{
    SKIP_IF_NO_DEVICES();

    const std::string deviceString
        = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }

    const std::vector<int64_t> dims = {2, 2, 256, 128};
    // BHSD with head-dim stride 2 and row stride 2 * D.
    const std::vector<int64_t> headDimStride2
        = {int64_t{2} * 256 * 256, int64_t{256} * 256, 256, 2};

    std::vector<GraphTest> declineTests;
    declineTests.emplace_back(
        withLayouts(createLayoutGraph({2, 2, 256, 192}, dims),
                    {{"q", {int64_t{2} * 256 * 384, int64_t{256} * 384, 384, 2}, {}}}),
        "hd192 q head-dim stride 2");
    declineTests.emplace_back(
        withLayouts(createLayoutGraph(dims, dims),
                    {{"q", {int64_t{2} * 128 * 256, int64_t{128} * 256, 1, 256}, {}}}),
        "q stored as [B, H, D, S]");
    // Strides of another rank are declined before any of them is read.
    declineTests.emplace_back(
        withLayouts(createLayoutGraph(dims, dims),
                    {{"q", {int64_t{2} * 256 * 128, int64_t{256} * 128, 128, 1, 1}, {}}}),
        "5 q strides for rank-4 q dims");
    for(const char* name : K_IO_TENSOR_NAMES)
    {
        declineTests.emplace_back(
            withLayouts(createLayoutGraph(dims, dims), {{name, headDimStride2, {}}}),
            std::string(name) + " head-dim stride 2");
    }

    for(const auto& test : declineTests)
    {
        EXPECT_FALSE(_planBuilder.isApplicable(_handle, test.graphWrapper())) << test.message;
    }
}

// A row stride below the tensor's own head dim, also 0 and also on a length-1
// sequence axis: each buffer ends S * row stride bytes into its (batch, head)
// slice, cutting off the end of the last row. On gfx942 the cases run gave a
// wrong O, an O of 0, or an O left NaN, except a K row stride of 1 with S_kv = 1:
// the one key takes all the softmax weight, so O was right, but by the address
// formula its score, and so the stats, are wrong.
TEST_F(TestSdpaFwdPlanBuilder, IsApplicableRejectsShortRows)
{
    SKIP_IF_NO_DEVICES();

    const std::string deviceString
        = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }

    const std::vector<int64_t> dims = {2, 2, 256, 128};
    // Rows overlap: 64-element row stride with D=128.
    const std::vector<int64_t> overlappingRows
        = {int64_t{2} * (int64_t{255} * 64 + 128), int64_t{255} * 64 + 128, 64, 1};
    // Every row of a (batch, head) slice at the same address.
    const std::vector<int64_t> zeroRows = {256, 128, 0, 1};

    std::vector<GraphTest> declineTests;
    for(const char* name : K_IO_TENSOR_NAMES)
    {
        declineTests.emplace_back(
            withLayouts(createLayoutGraph(dims, dims), {{name, overlappingRows, {}}}),
            std::string(name) + " row stride D/2");
        declineTests.emplace_back(
            withLayouts(createLayoutGraph(dims, dims), {{name, zeroRows, {}}}),
            std::string(name) + " row stride 0");
    }
    // Q and O have S_q rows, K and V have S_kv rows.
    for(const char* name : {"q", "o"})
    {
        declineTests.emplace_back(
            withLayouts(createLayoutGraph({2, 2, 1, 128}, dims), {{name, {256, 128, 1, 1}, {}}}),
            std::string(name) + " row stride 1 with S_q=1");
        declineTests.emplace_back(
            withLayouts(createLayoutGraph({2, 2, 2, 128}, dims), {{name, {384, 192, 64, 1}, {}}}),
            std::string(name) + " row stride D/2 with S_q=2");
    }
    for(const char* name : {"k", "v"})
    {
        declineTests.emplace_back(
            withLayouts(createLayoutGraph(dims, {2, 2, 1, 128}), {{name, {256, 128, 1, 1}, {}}}),
            std::string(name) + " row stride 1 with S_kv=1");
        declineTests.emplace_back(
            withLayouts(createLayoutGraph(dims, {2, 2, 2, 128}), {{name, {384, 192, 64, 1}, {}}}),
            std::string(name) + " row stride D/2 with S_kv=2");
    }
    // hd192x128: Q and K rows are 192 elements and V and O rows 128, so a row
    // stride of 160 is too short for Q and K only.
    const std::vector<int64_t> rows160 = {int64_t{2} * 256 * 160, int64_t{256} * 160, 160, 1};
    for(const char* name : {"q", "k"})
    {
        declineTests.emplace_back(
            withLayouts(createLayoutGraph({2, 2, 256, 192}, dims), {{name, rows160, {}}}),
            std::string("hd192 ") + name + " row stride 160");
    }

    for(const auto& test : declineTests)
    {
        EXPECT_FALSE(_planBuilder.isApplicable(_handle, test.graphWrapper())) << test.message;
    }
}

// Each (batch, head) slice of O is written by its own workgroups, so a zero O
// batch or head stride on an axis longer than 1 makes slices overwrite each
// other. On gfx942 an O batch or head stride of 0 left half or two thirds of O
// wrong, and O changed between some runs with identical inputs. Only a zero
// stride is checked: nonzero strides that make O overlap itself, such as equal
// batch and head strides, are not.
TEST_F(TestSdpaFwdPlanBuilder, IsApplicableRejectsZeroStrideOutput)
{
    SKIP_IF_NO_DEVICES();

    const std::string deviceString
        = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }

    const std::vector<int64_t> dims = {2, 2, 256, 128};

    std::vector<GraphTest> declineTests;
    // Only one axis aliases in each of these.
    declineTests.emplace_back(
        withLayouts(createLayoutGraph(dims, dims), {{"o", {0, int64_t{256} * 128, 128, 1}, {}}}),
        "O batch stride 0");
    declineTests.emplace_back(
        withLayouts(createLayoutGraph(dims, dims), {{"o", {int64_t{256} * 128, 0, 128, 1}, {}}}),
        "O head stride 0");
    declineTests.emplace_back(
        withLayouts(createLayoutGraph(dims, dims), {{"o", {0, 0, 128, 1}, {}}}),
        "O batch and head stride 0");
    // The head stride may be 0 here, as H=1, but the batch stride may not.
    declineTests.emplace_back(withLayouts(createLayoutGraph({2, 1, 256, 128}, {2, 1, 256, 128}),
                                          {{"o", {0, 0, 128, 1}, {}}}),
                              "O batch and head stride 0, B=2 H=1");

    for(const auto& test : declineTests)
    {
        EXPECT_FALSE(_planBuilder.isApplicable(_handle, test.graphWrapper())) << test.message;
    }
}

// Every forward kernel writes the LSE of (b, h, s) at
// ((b * H_q + h) * lse_Hs + s) * 4 bytes, with lse_Hs the stats head stride. On
// gfx942 each stats layout that this formula does not describe gave wrong or NaN
// stats in the cases run; by the formula, [H_q, B, S_q] storage also writes past
// the end of the stats buffer. A head stride below S_q makes (batch, head) slices
// overlap, so it declines too.
TEST_F(TestSdpaFwdPlanBuilder, IsApplicableRejectsUnaddressableStats)
{
    SKIP_IF_NO_DEVICES();

    const std::string deviceString
        = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }

    // Packed stats for these dims are [2, 2, 256, 1] with strides [512, 256, 1, 1].
    const std::vector<int64_t> dims = {2, 2, 256, 128};
    const auto statsGraph = [](const std::vector<int64_t>& qDims,
                               const std::vector<int64_t>& vDims,
                               const std::vector<int64_t>& statsStrides,
                               const std::vector<int64_t>& statsDims = {}) {
        return withLayouts(createLayoutGraph(qDims, vDims, true),
                           {{"stats", statsStrides, statsDims}});
    };

    std::vector<GraphTest> declineTests;
    // Only the seq stride is wrong in the next three.
    declineTests.emplace_back(statsGraph(dims, dims, {512, 256, 2, 1}), "stats seq stride 2");
    declineTests.emplace_back(statsGraph(dims, dims, {512, 256, 0, 1}),
                              "stats broadcast along S_q");
    declineTests.emplace_back(statsGraph({1, 1, 256, 128}, {1, 1, 256, 128}, {512, 512, 2, 1}),
                              "stats seq stride 2, B=1 H_q=1");
    // Only the batch stride is wrong in the next four.
    declineTests.emplace_back(statsGraph(dims, dims, {640, 256, 1, 1}),
                              "stats batch stride H_q * S_q + 128");
    declineTests.emplace_back(statsGraph(dims, dims, {256, 512, 1, 1}),
                              "stats stored as [H_q, B, S_q]");
    declineTests.emplace_back(
        statsGraph({2, 8, 256, 128}, {2, 2, 256, 128}, {int64_t{8} * 256 + 128, 256, 1, 1}),
        "GQA stats batch stride H_q * S_q + 128");
    declineTests.emplace_back(statsGraph({2, 2, 256, 192}, dims, {640, 256, 1, 1}),
                              "hd192 stats batch stride H_q * S_q + 128");
    // With H_q=1 the kernel still steps between batches by the head stride, so a
    // batch stride that differs from it declines too.
    declineTests.emplace_back(statsGraph({2, 1, 256, 128}, {2, 1, 256, 128}, {512, 256, 1, 1}),
                              "stats batch stride 2 * S_q, head stride S_q, H_q=1");
    // Only the head stride is wrong in the next four: (batch, head) slices overlap.
    declineTests.emplace_back(statsGraph(dims, dims, {256, 128, 1, 1}),
                              "stats head stride S_q / 2");
    declineTests.emplace_back(statsGraph(dims, dims, {0, 0, 1, 1}),
                              "stats batch and head stride 0");
    declineTests.emplace_back(statsGraph({1, 2, 256, 128}, {1, 2, 256, 128}, {512, 0, 1, 1}),
                              "stats head stride 0, B=1");
    declineTests.emplace_back(statsGraph({2, 1, 256, 128}, {2, 1, 256, 128}, {128, 128, 1, 1}),
                              "stats batch and head stride S_q / 2, H_q=1");
    // More than one stride is wrong in the next three. With H_q=1, [S_q, 1, 1, 1]
    // has both a batch stride other than the head stride and a head stride
    // below S_q.
    declineTests.emplace_back(statsGraph(dims, dims, {512, 1, 2, 1}),
                              "stats stored as [B, S_q, H_q]");
    declineTests.emplace_back(statsGraph(dims, dims, {512, 1, 2}, {2, 2, 256}),
                              "rank-3 stats stored as [B, S_q, H_q]");
    declineTests.emplace_back(statsGraph({2, 1, 256, 128}, {2, 1, 256, 128}, {256, 1, 1, 1}),
                              "stats strides [S_q, 1, 1, 1] with H_q=1");
    // Strides of another rank are declined before any of them is read.
    declineTests.emplace_back(statsGraph(dims, dims, {512, 256, 1}),
                              "3 stats strides for rank-4 stats dims");

    for(const auto& test : declineTests)
    {
        EXPECT_FALSE(_planBuilder.isApplicable(_handle, test.graphWrapper())) << test.message;
    }
}

// =============================================================================
// buildPlan exception contract (IPlanBuilder::buildPlan)
// =============================================================================

TEST_F(TestSdpaFwdPlanBuilder, BuildPlanThrowsForUnsupportedDtype)
{
    SKIP_IF_NO_DEVICES();

    // HALF is not a supported input dtype for the forward ASM SDPA kernels, so
    // the registry lookup will fail and buildPlan must throw
    // HipdnnPluginException rather than silently returning.
    auto builder = createSdpaFwdGraph(
        {4, 8, 256, 128}, {4, 8, 256, 128}, hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    Context ctx;
    const hipdnn_test_sdk::utilities::MockEngineConfig mockEngineConfig;

    EXPECT_THROW(_planBuilder.buildPlan(_handle, graphWrapper, mockEngineConfig, ctx),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST_F(TestSdpaFwdPlanBuilder, BuildPlanThrowsOnEmptyKernelKey)
{
    SKIP_IF_NO_DEVICES();

    const std::string deviceString
        = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }

    // Use an unsupported head dimension so the kernel registry lookup returns
    // an empty key.  isApplicable would reject this, but buildPlan must also
    // throw rather than crashing via cfg_fmha_fwd.at("").
    auto builder = createSdpaFwdGraph({4, 8, 256, 100});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper gw(builder.GetBufferPointer(),
                                                                        builder.GetSize());

    // Forward ignores engineConfig — a null-buffer wrapper suffices.
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper dummyCfg(nullptr, 0);
    Context context;
    EXPECT_THROW(_planBuilder.buildPlan(_handle, gw, dummyCfg, context),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

// =============================================================================
// initializeExecutionSettings
// =============================================================================

TEST_F(TestSdpaFwdPlanBuilder, InitializeExecutionSettingsIsNoOp)
{
    auto builder = createSdpaFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper gw(builder.GetBufferPointer(),
                                                                        builder.GetSize());

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper dummyCfg(nullptr, 0);
    Settings settings;
    // Must not throw or log an error — forward has no knobs.
    EXPECT_NO_THROW(_planBuilder.initializeExecutionSettings(_handle, gw, dummyCfg, settings));
}

} // namespace
} // namespace asm_sdpa_engine
