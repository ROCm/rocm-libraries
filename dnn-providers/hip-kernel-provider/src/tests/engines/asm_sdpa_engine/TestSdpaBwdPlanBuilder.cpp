// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "HipKernelHandle.hpp"
#include "HipKernelSettings.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaBwdPlanBuilder.hpp"
#include "hip_kernel_provider_common/HipDeviceUtils.hpp"

namespace asm_sdpa_engine
{
namespace
{

class TestSdpaBwdPlanBuilder : public ::testing::Test
{
protected:
    SdpaBwdPlanBuilder _planBuilder;
    HipKernelHandle _handle;
};

struct GraphTest
{
    std::shared_ptr<flatbuffers::DetachedBuffer> buffer;
    std::string message;

    GraphTest(flatbuffers::FlatBufferBuilder&& builder, std::string inMessage)
        : buffer(std::make_shared<flatbuffers::DetachedBuffer>(builder.Release()))
        , message(std::move(inMessage))
    {
    }

    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graphWrapper() const
    {
        return hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper(buffer->data(), buffer->size());
    }
};

// Custom backward graph builder with BF16 tensors and FP32 stats by default
auto createSdpaBwdGraph(const std::vector<int64_t>& dims = {4, 8, 256, 128},
                        hipdnn_data_sdk::data_objects::DataType dataType
                        = hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
                        hipdnn_data_sdk::data_objects::DataType statsDataType
                        = hipdnn_data_sdk::data_objects::DataType::FLOAT,
                        bool alibiMask = false,
                        bool paddingMask = false,
                        bool causalMask = false,
                        bool withScale = false)
{
    using namespace hipdnn_data_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<TensorAttributes>> tensorAttributes;

    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    int64_t uid = 1;

    // Q [B, H_q, S_q, D]
    const auto qUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, qUid, "q", dataType, &strides, &dims));

    // K [B, H_kv, S_kv, D]
    const auto kUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, kUid, "k", dataType, &strides, &dims));

    // V [B, H_kv, S_kv, D]
    const auto vUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, vUid, "v", dataType, &strides, &dims));

    // O [B, H_q, S_q, D] (forward output)
    const auto oUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, oUid, "o", dataType, &strides, &dims));

    // dO [B, H_q, S_q, D] (upstream gradient, same shape as O)
    const auto doUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, doUid, "do", dataType, &strides, &dims));

    // Stats [B, H_q, S_q, 1] (LSE from forward pass, FP32)
    const std::vector<int64_t> statsDims = {dims[0], dims[1], dims[2], 1};
    const std::vector<int64_t> statsStrides = {dims[1] * dims[2], dims[2], 1, 1};
    const auto statsUid = uid++;
    tensorAttributes.push_back(CreateTensorAttributesDirect(
        builder, statsUid, "stats", statsDataType, &statsStrides, &statsDims));

    // dQ [B, H_q, S_q, D]
    const auto dqUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, dqUid, "dq", dataType, &strides, &dims));

    // dK [B, H_kv, S_kv, D]
    const auto dkUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, dkUid, "dk", dataType, &strides, &dims));

    // dV [B, H_kv, S_kv, D]
    const auto dvUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, dvUid, "dv", dataType, &strides, &dims));

    // Optional scale tensor
    flatbuffers::Optional<int64_t> scaleUid = flatbuffers::nullopt;
    if(withScale)
    {
        const std::vector<int64_t> passByValueDims = {1};
        const Float32Value scaleVal(1.0f);
        const auto sUid = uid++;
        tensorAttributes.push_back(
            CreateTensorAttributesDirect(builder,
                                         sUid,
                                         "scale",
                                         DataType::FLOAT,
                                         &passByValueDims,
                                         &passByValueDims,
                                         false,
                                         TensorValue::Float32Value,
                                         builder.CreateStruct(scaleVal).Union()));
        scaleUid = flatbuffers::Optional<int64_t>(sUid);
    }

    auto sdpaBwdAttributes = CreateSdpaBackwardAttributes(builder,
                                                          qUid,
                                                          kUid,
                                                          vUid,
                                                          oUid,
                                                          doUid,
                                                          statsUid,
                                                          dqUid,
                                                          dkUid,
                                                          dvUid,
                                                          scaleUid, // scale_tensor_uid
                                                          flatbuffers::nullopt, // attn_mask
                                                          flatbuffers::nullopt, // seq_len_q
                                                          flatbuffers::nullopt, // seq_len_kv
                                                          flatbuffers::nullopt, // seed
                                                          flatbuffers::nullopt, // offset
                                                          flatbuffers::nullopt, // dropout_mask
                                                          flatbuffers::nullopt, // dropout_scale
                                                          flatbuffers::nullopt, // dropout_scale_inv
                                                          flatbuffers::nullopt, // dbias
                                                          alibiMask,
                                                          paddingMask,
                                                          causalMask);

    std::vector<::flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "sdpa_bwd",
                                     dataType,
                                     NodeAttributes::SdpaBackwardAttributes,
                                     sdpaBwdAttributes.Union()));

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

TEST_F(TestSdpaBwdPlanBuilder, IsApplicableReturnsFalseForNonSdpaBwdGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();

    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaBwdPlanBuilder, IsApplicableSdpaBwdVariations)
{
    using namespace hipdnn_data_sdk::data_objects;

    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
    {
        GTEST_SKIP();
    }

    std::vector<std::pair<GraphTest, bool>> applicabilityTests = {
        // Valid backward graph: BF16, HD=128, FP32 stats, no masking
        {GraphTest{createSdpaBwdGraph(), "Valid BF16 HD128 backward"}, true},

        // Wrong head dimension
        {GraphTest{createSdpaBwdGraph({4, 8, 256, 64}), "Head dimension 64"}, false},

        // Wrong tensor dtype (FP16)
        {GraphTest{createSdpaBwdGraph({4, 8, 256, 128}, DataType::HALF), "FP16 tensors"}, false},

        // Causal mask enabled
        {GraphTest{createSdpaBwdGraph(
                       {4, 8, 256, 128}, DataType::BFLOAT16, DataType::FLOAT, false, false, true),
                   "causal_mask = true"},
         false},

        // Alibi mask enabled
        {GraphTest{createSdpaBwdGraph({4, 8, 256, 128}, DataType::BFLOAT16, DataType::FLOAT, true),
                   "alibi_mask = true"},
         false},

        // Padding mask enabled
        {GraphTest{
             createSdpaBwdGraph({4, 8, 256, 128}, DataType::BFLOAT16, DataType::FLOAT, false, true),
             "padding_mask = true"},
         false},

        // With scale tensor (should still be accepted)
        {GraphTest{
             createSdpaBwdGraph(
                 {4, 8, 256, 128}, DataType::BFLOAT16, DataType::FLOAT, false, false, false, true),
             "with scale tensor"},
         true},

        // Stats tensor wrong dtype (BF16 instead of FP32)
        {GraphTest{createSdpaBwdGraph({4, 8, 256, 128}, DataType::BFLOAT16, DataType::BFLOAT16),
                   "stats tensor BF16 (should be FP32)"},
         false},
    };

    for(const auto& [test, applicability] : applicabilityTests)
    {
        EXPECT_EQ(_planBuilder.isApplicable(_handle, test.graphWrapper()), applicability)
            << test.message;
    }
}

TEST_F(TestSdpaBwdPlanBuilder, GetMaxWorkspaceSizeReturnsZero)
{
    auto builder = createSdpaBwdGraph();

    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    HipKernelSettings settings;
    size_t workspaceSize = _planBuilder.getMaxWorkspaceSize(_handle, graphWrapper, settings);

    // Stub: returns 0 until Task I4 implements actual workspace calculation
    EXPECT_EQ(workspaceSize, 0u);
}

} // namespace
} // namespace asm_sdpa_engine
