// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/SdpaGraphMutators.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "GraphTest.hpp"
#include "asm_fmha_v3_bwd_configs.hpp"
#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaBwdPlan.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaBwdPlanBuilder.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaKernelUtils.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaPlanUtils.hpp"
#include "hip_kernel_provider_common/HipDeviceUtils.hpp"
#include "tests/engines/asm_sdpa_engine/AsmSdpaTestSupport.hpp"
#include "tests/engines/asm_sdpa_engine/DeclaredAlignmentTestUtils.hpp"
#include "tests/engines/asm_sdpa_engine/DeprecatedCausalMaskCases.hpp"
#include "tests/engines/asm_sdpa_engine/SdpaAttnScaleCases.hpp"
#include "tests/utilities/RaggedGraphTestUtils.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

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
namespace sdpa = hipdnn_test_sdk::utilities::sdpa;

class TestSdpaBwdPlanBuilder : public ::testing::Test
{
protected:
    // The test device's architecture; call after SKIP_IF_NO_DEVICES().
    std::string deviceString() const
    {
        return hip_kernel_provider_common::getDeviceString(_handle.getStream());
    }

    SdpaBwdPlanBuilder _planBuilder;
    Handle _handle;
};

TEST_F(TestSdpaBwdPlanBuilder, IsApplicableReturnsFalseForNonSdpaBwdGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
}

auto createSdpaBwdGraph(const std::vector<int64_t>& dims = {4, 8, 256, 128},
                        hipdnn_flatbuffers_sdk::data_objects::DataType dataType
                        = hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                        bool withScale = false,
                        bool alibiMask = false,
                        bool paddingMask = false,
                        bool causalMask = false,
                        bool overrideShapeEnabled = false)
{
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    return hipdnn_test_sdk::utilities::createValidSdpaBwdGraph(dims,
                                                               strides,
                                                               dims,
                                                               strides,
                                                               dims,
                                                               strides,
                                                               dims,
                                                               strides,
                                                               dataType,
                                                               withScale,
                                                               alibiMask,
                                                               paddingMask,
                                                               causalMask,
                                                               overrideShapeEnabled);
}

TEST_F(TestSdpaBwdPlanBuilder, IsApplicableReturnsFalseForOverrideShapeEnabledGraph)
{
    auto builder = createSdpaBwdGraph({4, 8, 256, 128},
                                      hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                                      false,
                                      false,
                                      false,
                                      false,
                                      /*overrideShapeEnabled=*/true);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaBwdPlanBuilder, IsApplicableReturnsFalseForRaggedGraph)
{
    // The backward engine does not support group mode; any ragged tensor must be
    // rejected.
    auto builder = sdpa::withRaggedTensors(createSdpaBwdGraph(), {"q", "k", "v", "o"});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaBwdPlanBuilder, IsApplicableReturnsTrueForMultipleOf16ByteAlignment)
{
    SKIP_IF_NO_DEVICES();

    if(!bwd_dispatch::isSupportedArch(deviceString()))
    {
        GTEST_SKIP();
    }

    auto builder
        = hipdnn_test_sdk::utilities::makeGraphWithTensorAlignment(createSdpaBwdGraph(), 32);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaBwdPlanBuilder, IsApplicableReturnsFalseForNon16ByteAlignment)
{
    SKIP_IF_NO_DEVICES();

    // On other devices the device check declines first, masking the alignment check.
    if(!bwd_dispatch::isSupportedArch(deviceString()))
    {
        GTEST_SKIP();
    }

    auto builder
        = hipdnn_test_sdk::utilities::makeGraphWithTensorAlignment(createSdpaBwdGraph(), 8);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
}

// Bug caught: a ragged SDPA backward graph accepted and its packed buffers read through dense
// strides. The unmodified graph is accepted, and the same graph is declined with any one of
// its tensors ragged.
TEST_F(TestSdpaBwdPlanBuilder, IsApplicableRejectsRaggedTensors)
{
    SKIP_IF_NO_DEVICES();

    if(!bwd_dispatch::isSupportedArch(deviceString()))
    {
        GTEST_SKIP();
    }

    hip_kernel_provider::test_utils::expectRaggedGraphsDeclined(
        createSdpaBwdGraph(),
        [this](const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph) {
            return _planBuilder.isApplicable(_handle, graph);
        });
}

// Bug caught: the backward kernels' wide row loads and stores (odo, dqdkdv, dq_convert) run on a
// graph tensor whose declared alignment is below the need (the backend then rejects the pointer
// at execute, or the kernel reads misaligned), or a declared alignment that is a larger multiple
// is wrongly declined. Each of Q, K, V, O, dO, dQ, dK, dV and stats is under-declared, declared
// exactly at its need, and declared at a larger multiple, one tensor at a time; default-16
// graphs stay accepted.
TEST_F(TestSdpaBwdPlanBuilder, IsApplicableHonorsDeclaredAlignmentOfEveryTensor)
{
    SKIP_IF_NO_DEVICES();

    if(!bwd_dispatch::isSupportedArch(deviceString()))
    {
        GTEST_SKIP();
    }

    using hip_kernel_provider::test_utils::AlignmentRole;

    const auto valid = createSdpaBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper(
        valid.GetBufferPointer(), valid.GetSize());
    const auto& attrs
        = wrapper.getNodeWrapper(0)
              .attributesAs<hipdnn_flatbuffers_sdk::data_objects::SdpaBackwardAttributes>();

    const std::vector<AlignmentRole> roles{
        {"q", attrs.q_tensor_uid(), plan_utils::K_ROW_ALIGNMENT_BYTES},
        {"k", attrs.k_tensor_uid(), plan_utils::K_ROW_ALIGNMENT_BYTES},
        {"v", attrs.v_tensor_uid(), plan_utils::K_ROW_ALIGNMENT_BYTES},
        {"o", attrs.o_tensor_uid(), plan_utils::K_ROW_ALIGNMENT_BYTES},
        {"do", attrs.do_tensor_uid(), plan_utils::K_ROW_ALIGNMENT_BYTES},
        {"dq", attrs.dq_tensor_uid(), plan_utils::K_ROW_ALIGNMENT_BYTES},
        {"dk", attrs.dk_tensor_uid(), plan_utils::K_ROW_ALIGNMENT_BYTES},
        {"dv", attrs.dv_tensor_uid(), plan_utils::K_ROW_ALIGNMENT_BYTES},
        {"stats", attrs.stats_tensor_uid(), plan_utils::K_STATS_ALIGNMENT_BYTES},
    };

    hip_kernel_provider::test_utils::expectDeclaredAlignmentHonored(
        valid, roles, [this](const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph) {
            return _planBuilder.isApplicable(_handle, graph);
        });
}

TEST_F(TestSdpaBwdPlanBuilder, IsApplicableSdpaBwdVariations)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    SKIP_IF_NO_DEVICES();

    if(!bwd_dispatch::isSupportedArch(deviceString()))
    {
        GTEST_SKIP();
    }

    const bool isGfx942 = deviceString() == "gfx942";

    const std::vector<std::pair<GraphTest, bool>> applicabilityTests = {
        // Valid backward graph: BF16, HD=128, FP32 stats, no masking. It is calibrated
        // against the CPU reference on gfx942 (logs INFO there, WARN on gfx950).
        {GraphTest{createSdpaBwdGraph(), "Valid BF16 HD128 backward"}, true},

        // HD=64: gfx942 serves it with the (pssk = 1, pddv = 0) rows; AITER sends
        // gfx950 hd64 to CK, so gfx950 declines it.
        {GraphTest{createSdpaBwdGraph({4, 8, 256, 64}), "Head dimension 64"}, isGfx942},

        // HD=192 BF16: registry-supported and dispatched on gfx942. The CPU
        // reference is not yet calibrated, so buildPlan logs a one-time WARN;
        // isApplicable accepts. AITER sends gfx950 hd192x192 to CK, so gfx950
        // declines it.
        {GraphTest{createSdpaBwdGraph({4, 8, 256, 192}), "Head dimension 192"}, isGfx942},

        // FP16 HD128: registry-supported and dispatched; calibrated on gfx942 (logs
        // INFO there, WARN on gfx950). isApplicable accepts.
        {GraphTest{createSdpaBwdGraph({4, 8, 256, 128}, DataType::HALF), "FP16 tensors"}, true},

        // Causal mask dispatched: CSV registry carries mask=1 DQDKDV rows.
        {GraphTest{
             createSdpaBwdGraph({4, 8, 256, 128}, DataType::BFLOAT16, false, false, false, true),
             "causal_mask = true"},
         true},

        // Alibi mask not supported.
        {GraphTest{createSdpaBwdGraph({4, 8, 256, 128}, DataType::BFLOAT16, false, true),
                   "alibi_mask = true"},
         false},

        // Padding mask not supported.
        {GraphTest{createSdpaBwdGraph({4, 8, 256, 128}, DataType::BFLOAT16, false, false, true),
                   "padding_mask = true"},
         false},

        // With scale tensor (should still be accepted)
        {GraphTest{createSdpaBwdGraph({4, 8, 256, 128}, DataType::BFLOAT16, true),
                   "with scale tensor"},
         true},
    };

    for(const auto& [test, applicability] : applicabilityTests)
    {
        EXPECT_EQ(_planBuilder.isApplicable(_handle, test.graphWrapper()), applicability)
            << test.message;
    }
}

// =============================================================================
// Runtime pass-by-value scale tensor (RFC 0016)
// =============================================================================

// Build a backward SDPA graph with a runtime pass-by-value scale tensor
// (is_runtime_pass_by_value=true, no baked value).
flatbuffers::FlatBufferBuilder createSdpaBwdGraphWithRuntimePbvScale()
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
    const auto doUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, doUid, "do", DataType::BFLOAT16, &strides, &dims));

    const std::vector<int64_t> statsDims = {dims[0], dims[1], dims[2], 1};
    const std::vector<int64_t> statsStrides = {dims[1] * dims[2], dims[2], 1, 1};
    const auto statsUid = uid++;
    tensorAttributes.push_back(CreateTensorAttributesDirect(
        builder, statsUid, "stats", DataType::FLOAT, &statsStrides, &statsDims));

    const auto dqUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, dqUid, "dq", DataType::BFLOAT16, &strides, &dims));
    const auto dkUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, dkUid, "dk", DataType::BFLOAT16, &strides, &dims));
    const auto dvUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, dvUid, "dv", DataType::BFLOAT16, &strides, &dims));

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

    const auto sdpaAttributes = CreateSdpaBackwardAttributes(builder,
                                                             qUid,
                                                             kUid,
                                                             vUid,
                                                             oUid,
                                                             doUid,
                                                             statsUid,
                                                             dqUid,
                                                             dkUid,
                                                             dvUid,
                                                             scaleUid); // scale_tensor_uid

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "sdpa_bwd",
                                     DataType::FLOAT,
                                     NodeAttributes::SdpaBackwardAttributes,
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

// Build a backward SDPA graph with a non-pass-by-value scale tensor.
flatbuffers::FlatBufferBuilder createSdpaBwdGraphWithNonPbvScaleTensor()
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
    const auto doUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, doUid, "do", DataType::BFLOAT16, &strides, &dims));

    const std::vector<int64_t> statsDims = {dims[0], dims[1], dims[2], 1};
    const std::vector<int64_t> statsStrides = {dims[1] * dims[2], dims[2], 1, 1};
    const auto statsUid = uid++;
    tensorAttributes.push_back(CreateTensorAttributesDirect(
        builder, statsUid, "stats", DataType::FLOAT, &statsStrides, &statsDims));

    const auto dqUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, dqUid, "dq", DataType::BFLOAT16, &strides, &dims));
    const auto dkUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, dkUid, "dk", DataType::BFLOAT16, &strides, &dims));
    const auto dvUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, dvUid, "dv", DataType::BFLOAT16, &strides, &dims));

    // Regular device tensor — NOT pass-by-value
    const std::vector<int64_t> scaleDims = {1};
    const auto scaleUid = uid++;
    tensorAttributes.push_back(CreateTensorAttributesDirect(
        builder, scaleUid, "scale", DataType::FLOAT, &scaleDims, &scaleDims));

    const auto sdpaAttributes = CreateSdpaBackwardAttributes(builder,
                                                             qUid,
                                                             kUid,
                                                             vUid,
                                                             oUid,
                                                             doUid,
                                                             statsUid,
                                                             dqUid,
                                                             dkUid,
                                                             dvUid,
                                                             scaleUid); // scale_tensor_uid

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "sdpa_bwd",
                                     DataType::FLOAT,
                                     NodeAttributes::SdpaBackwardAttributes,
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

TEST_F(TestSdpaBwdPlanBuilder, IsApplicableAcceptsRuntimePassByValueScale)
{
    SKIP_IF_NO_DEVICES();

    if(!bwd_dispatch::isSupportedArch(deviceString()))
    {
        GTEST_SKIP();
    }

    auto builder = createSdpaBwdGraphWithRuntimePbvScale();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaBwdPlanBuilder, IsApplicableRejectsNonPassByValueScaleTensor)
{
    SKIP_IF_NO_DEVICES();

    if(!bwd_dispatch::isSupportedArch(deviceString()))
    {
        GTEST_SKIP();
    }

    auto builder = createSdpaBwdGraphWithNonPbvScaleTensor();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaBwdPlanBuilder, IsApplicableAcceptsCompileTimeConstantScaleTensor)
{
    // Existing withScale=true path uses a compile-time constant scale tensor.
    // Verify it still passes after the PBV changes.
    SKIP_IF_NO_DEVICES();

    if(!bwd_dispatch::isSupportedArch(deviceString()))
    {
        GTEST_SKIP();
    }

    auto builder = createSdpaBwdGraph({4, 8, 256, 128},
                                      hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                                      /*withScale=*/true);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_handle, graphWrapper));
}

// =============================================================================
// Attention scale range
// =============================================================================

// Scales the kernels mishandle, from attn_scale_value, a constant scale tensor or
// the default of a runtime scale tensor, and a scale tensor whose value cannot be
// read. On gfx942, 0, -0.0, 1e-40 and -1.2 gave NaN or Inf dQ, dK and dV with a
// causal or sliding-window mask, and NaN dQ with S_kv = 128 and no mask (the
// hd128 K tile is 192).
TEST_F(TestSdpaBwdPlanBuilder, IsApplicableRejectsUnsupportedAttnScale)
{
    SKIP_IF_NO_DEVICES();

    if(!bwd_dispatch::isSupportedArch(deviceString()))
    {
        GTEST_SKIP();
    }

    std::vector<GraphTest> declineTests;
    for(const auto& [scale, scaleName] : unsupportedAttnScales())
    {
        for(const auto& [source, sourceName] : attnScaleSources())
        {
            declineTests.emplace_back(withAttnScale(createSdpaBwdGraph(), source, scale),
                                      attnScaleCaseName(scaleName, sourceName));
        }
    }
    for(const auto& [scale, name] : unsupportedDoubleAttnScales())
    {
        for(const bool runtimeWithDefault : {false, true})
        {
            declineTests.emplace_back(
                withDoubleScaleTensor(createSdpaBwdGraph(), scale, runtimeWithDefault),
                name + (runtimeWithDefault ? " as runtime default DOUBLE" : " as constant DOUBLE"));
        }
    }
    declineTests.emplace_back(withUnreadableScaleTensor(createSdpaBwdGraph()),
                              "FP8_E4M3 constant scale tensor");

    for(const auto& test : declineTests)
    {
        EXPECT_FALSE(_planBuilder.isApplicable(_handle, test.graphWrapper())) << test.message;
    }
}

// Finite, positive, normal scales from every source, the unset default (1.0), and
// a pure runtime scale tensor, which execute checks.
TEST_F(TestSdpaBwdPlanBuilder, IsApplicableAcceptsSupportedAttnScale)
{
    SKIP_IF_NO_DEVICES();

    if(!bwd_dispatch::isSupportedArch(deviceString()))
    {
        GTEST_SKIP();
    }

    std::vector<GraphTest> acceptTests;
    acceptTests.emplace_back(createSdpaBwdGraph(), "unset attn_scale_value");
    for(const auto& [scale, scaleName] : supportedAttnScales())
    {
        for(const auto& [source, sourceName] : attnScaleSources())
        {
            acceptTests.emplace_back(withAttnScale(createSdpaBwdGraph(), source, scale),
                                     attnScaleCaseName(scaleName, sourceName));
        }
    }
    for(const auto& [scale, name] : supportedAttnScales())
    {
        for(const bool runtimeWithDefault : {false, true})
        {
            acceptTests.emplace_back(
                withDoubleScaleTensor(
                    createSdpaBwdGraph(), static_cast<double>(scale), runtimeWithDefault),
                name + (runtimeWithDefault ? " as runtime default DOUBLE" : " as constant DOUBLE"));
        }
    }
    acceptTests.emplace_back(createSdpaBwdGraphWithRuntimePbvScale(), "pure runtime scale tensor");

    for(const auto& test : acceptTests)
    {
        EXPECT_TRUE(_planBuilder.isApplicable(_handle, test.graphWrapper())) << test.message;
    }
}

// A backward A32 plan for a [1, 1, 128, 128] bf16 problem, with no loaded kernels,
// whose scale is a pure runtime pass-by-value tensor. Launching it fails with
// INTERNAL_ERROR, so BAD_PARAM shows that execute threw before the first launch.
constexpr int64_t K_BWD_RUNTIME_SCALE_UID = 10;
const std::vector<int64_t> K_BWD_TENSOR_UIDS = {1, 2, 3, 4, 5, 6, 7, 8, 9};

SdpaBwdPlan createBwdPlanWithRuntimeScale(hipdnn_flatbuffers_sdk::data_objects::DataType scaleType
                                          = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
{
    constexpr unsigned int K_ROW = 128;
    constexpr unsigned int K_HEAD = 128 * 128;

    SdpaBwdParams params{};
    params.qUid = 1;
    params.kUid = 2;
    params.vUid = 3;
    params.oUid = 4;
    params.doUid = 5;
    params.statsUid = 6;
    params.dqUid = 7;
    params.dkUid = 8;
    params.dvUid = 9;
    params.batchSize = 1;
    params.numHeadsQ = 1;
    params.numHeadsKv = 1;
    params.seqLenQ = 128;
    params.seqLenKv = 128;
    params.headDimQk = 128;
    params.headDimV = 128;
    for(auto* stride : {&params.qStrideSeq,
                        &params.kStrideSeq,
                        &params.vStrideSeq,
                        &params.oStrideSeq,
                        &params.doStrideSeq,
                        &params.dqStrideSeq,
                        &params.dkStrideSeq,
                        &params.dvStrideSeq})
    {
        *stride = K_ROW;
    }
    for(auto* stride : {&params.qStrideHead,
                        &params.qStrideBatch,
                        &params.kStrideHead,
                        &params.kStrideBatch,
                        &params.vStrideHead,
                        &params.vStrideBatch,
                        &params.oStrideHead,
                        &params.oStrideBatch,
                        &params.doStrideHead,
                        &params.doStrideBatch,
                        &params.dqStrideHead,
                        &params.dqStrideBatch,
                        &params.dkStrideHead,
                        &params.dkStrideBatch,
                        &params.dvStrideHead,
                        &params.dvStrideBatch})
    {
        *stride = K_HEAD;
    }
    params.statsStrideHead = 128;
    params.statsStrideBatch = 128;
    params.odoTiles.ts = 128;
    params.dqdkdvTiles.ts = 192;
    params.postTiles.ts = 64;
    params.accumulatorType = AccumulatorType::A32;
    params.attnScale
        = hipdnn_plugin_sdk::ScalarOperand{K_BWD_RUNTIME_SCALE_UID, scaleType, true, 0.0};
    return {std::make_shared<HipModuleGuard>(),
            std::make_shared<HipModuleGuard>(),
            std::make_shared<HipModuleGuard>(),
            params};
}

// A pure runtime scale tensor's value arrives only at execute, so execute returns
// BAD_PARAM for an unsupported one, before the first launch.
TEST(TestSdpaBwdPlanExecute, RejectsUnsupportedRuntimeAttnScale)
{
    const SdpaBwdPlan plan = createBwdPlanWithRuntimeScale();
    const size_t workspaceSize = plan.getWorkspaceSize(Handle{});
    for(const auto& [scale, name] : unsupportedAttnScales())
    {
        EXPECT_EQ(executeWithRuntimeScale(
                      plan, K_BWD_TENSOR_UIDS, K_BWD_RUNTIME_SCALE_UID, scale, workspaceSize),
                  HIPDNN_PLUGIN_STATUS_BAD_PARAM)
            << name;
    }
    const SdpaBwdPlan doublePlan
        = createBwdPlanWithRuntimeScale(hipdnn_flatbuffers_sdk::data_objects::DataType::DOUBLE);
    for(const auto& [scale, name] : unsupportedDoubleAttnScales())
    {
        EXPECT_EQ(executeWithRuntimeScale(
                      doublePlan, K_BWD_TENSOR_UIDS, K_BWD_RUNTIME_SCALE_UID, scale, workspaceSize),
                  HIPDNN_PLUGIN_STATUS_BAD_PARAM)
            << name;
    }
}

// A supported runtime scale passes the check and reaches the first launch, which
// fails because the plan has no kernels.
TEST(TestSdpaBwdPlanExecute, PassesSupportedRuntimeAttnScaleToLaunch)
{
    SKIP_IF_NO_DEVICES();

    const SdpaBwdPlan plan = createBwdPlanWithRuntimeScale();
    const size_t workspaceSize = plan.getWorkspaceSize(Handle{});
    for(const auto& [scale, name] : supportedAttnScales())
    {
        EXPECT_EQ(executeWithRuntimeScale(
                      plan, K_BWD_TENSOR_UIDS, K_BWD_RUNTIME_SCALE_UID, scale, workspaceSize),
                  HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR)
            << name;
    }

    // Clear the HIP error state left by the intentional failed launch, or the
    // HipErrorHandler listener fails this test for it.
    static_cast<void>(hipGetLastError());
    static_cast<void>(hipExtGetLastError());
}

TEST_F(TestSdpaBwdPlanBuilder, BackwardWorkspaceSizeSmallUnaligned)
{
    // B=1, H=3, S=255, D=128 — chosen so D buffer raw size (3060) is NOT a multiple of 64,
    // exercising the alignUp() rounding: 3060 → 3072
    auto builder = createSdpaBwdGraph({1, 3, 255, 128});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());
    const Settings settings;
    const size_t workspaceSize = _planBuilder.getMaxWorkspaceSize(_handle, graphWrapper, settings);

    // D buffer: 1*3*255*4 = 3060, aligned to 64 → 3072  (exercises alignment rounding)
    // dq_acc:   1*3*255*128*4 = 391680, aligned to 64 → 391680 (already aligned)
    EXPECT_EQ(workspaceSize, 3072u + 391680u);
}

TEST_F(TestSdpaBwdPlanBuilder, BackwardWorkspaceSizeMedium)
{
    // B=2, H=8, S=512, D=128
    auto builder = createSdpaBwdGraph({2, 8, 512, 128});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());
    const Settings settings;
    const size_t workspaceSize = _planBuilder.getMaxWorkspaceSize(_handle, graphWrapper, settings);

    // D buffer: 2*8*512*4 = 32768, aligned to 64 → 32768
    // dq_acc:   2*8*512*128*4 = 4194304, aligned to 64 → 4194304
    EXPECT_EQ(workspaceSize, 32768u + 4194304u);
}

TEST_F(TestSdpaBwdPlanBuilder, BackwardWorkspaceSizeLarge)
{
    // B=4, H=16, S=1024, D=128
    auto builder = createSdpaBwdGraph({4, 16, 1024, 128});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());
    const Settings settings;
    const size_t workspaceSize = _planBuilder.getMaxWorkspaceSize(_handle, graphWrapper, settings);

    // D buffer: 4*16*1024*4 = 262144, aligned to 64 → 262144
    // dq_acc:   4*16*1024*128*4 = 33554432, aligned to 64 → 33554432
    EXPECT_EQ(workspaceSize, 262144u + 33554432u);
}

// CSV-derived constants used by the registry-lookup tests.
namespace
{
constexpr int MASK_NONE = 0; // MaskType::NO_MASK
constexpr int MASK_TOP_LEFT_CAUSAL = 1; // MaskType::TOP_LEFT_CAUSAL
constexpr int MASK_BOTTOM_RIGHT_CAUSAL = 2; // MaskType::BOTTOM_RIGHT_CAUSAL
constexpr int MASK_SLIDING_WINDOW = 3; // MaskType::SLIDING_WINDOW
constexpr int MODE_BATCH = 0; // BatchMode::BATCH
constexpr int MODE_GROUP = 1; // BatchMode::GROUP
constexpr int ATOMIC_A16 = 0; // AccumulatorMode::A16
constexpr int ATOMIC_A32 = 1; // AccumulatorMode::A32
constexpr int BF16_CVT_RTNE = 0; // RoundingMode::RTNE
using asm_sdpa_engine::bwd_dispatch::BF16_CVT_FP16_SENTINEL;

// A batch-mode A32 request. S_q = S_kv = 256 is not a multiple of any A32
// K/V tile (192 or 256), so it asks for pssk = 1 on every row.
bwd_dispatch::KernelRequest makeRequest(const std::string& dtype,
                                        int hdimQ,
                                        int hdimV,
                                        int mask = MASK_NONE,
                                        int bf16Cvt = BF16_CVT_RTNE)
{
    bwd_dispatch::KernelRequest request{};
    request.dataType = dtype;
    request.hdimQ = hdimQ;
    request.hdimV = hdimV;
    request.seqLenQ = 256;
    request.seqLenKv = 256;
    request.mask = mask;
    request.atomic32 = ATOMIC_A32;
    request.mode = MODE_BATCH;
    request.bf16Cvt = dtype == "fp16" ? BF16_CVT_FP16_SENTINEL : bf16Cvt;
    request.psskOffStridesOk = true;
    request.batch = 1;
    request.nheadQ = 1;
    request.qBatchStrideMatchesDqAcc = true;
    return request;
}

// The same request with the given sequence lengths.
bwd_dispatch::KernelRequest
    withSeqLens(bwd_dispatch::KernelRequest request, int64_t seqLenQ, int64_t seqLenKv)
{
    request.seqLenQ = seqLenQ;
    request.seqLenKv = seqLenKv;
    return request;
}

// The registry row the lookup resolves for stage, or std::nullopt.
std::optional<fmha_v3_bwdConfig> resolvedRow(bwd_dispatch::PipelineStage stage,
                                             const std::string& arch,
                                             const bwd_dispatch::KernelRequest& request)
{
    const auto key = bwd_dispatch::lookupKernelNameKey(stage, arch, request);
    if(key.empty())
    {
        return std::nullopt;
    }
    switch(stage)
    {
    case bwd_dispatch::PipelineStage::ODO:
        return cfg_fmha_bwd_odo.at(key);
    case bwd_dispatch::PipelineStage::DQDKDV:
        return cfg_fmha_bwd_dqdkdv.at(key);
    case bwd_dispatch::PipelineStage::DQ_CONVERT:
        return cfg_fmha_bwd_dq_convert.at(key);
    case bwd_dispatch::PipelineStage::DQ_SHUFFLE:
        return cfg_fmha_bwd_dq_shuffle.at(key);
    default:
        return std::nullopt;
    }
}

std::optional<fmha_v3_bwdConfig> resolvedDqdkdv(const std::string& arch,
                                                const bwd_dispatch::KernelRequest& request)
{
    return resolvedRow(bwd_dispatch::PipelineStage::DQDKDV, arch, request);
}

// (pssk, pddv) of the dqdkdv row the lookup resolves, or std::nullopt.
std::optional<std::pair<int, int>> resolvedPsskPddv(const std::string& arch,
                                                    const bwd_dispatch::KernelRequest& request)
{
    const auto row = resolvedDqdkdv(arch, request);
    if(!row)
    {
        return std::nullopt;
    }
    return std::pair<int, int>{row->pssk, row->pddv};
}

const std::pair<int, int> K_PSSK_PDDV{1, 1};
const std::pair<int, int> K_PSSK_ONLY{1, 0};
const std::pair<int, int> K_UNPADDED{0, 0};
} // namespace

// gfx942 serves hd128 with every mask in batch mode for both dtypes; gfx950
// adds the bottom-right window kernel (mask 4), which gfx942 does not have.
TEST(TestSdpaBwdRegistryLookup, Hd128MasksResolveEveryStage)
{
    using bwd_dispatch::PipelineStage;
    struct Case
    {
        const char* arch;
        int mask;
    };
    for(const auto& [arch, mask] :
        std::vector<Case>{{"gfx942", MASK_NONE},
                          {"gfx942", MASK_TOP_LEFT_CAUSAL},
                          {"gfx942", MASK_BOTTOM_RIGHT_CAUSAL},
                          {"gfx942", MASK_SLIDING_WINDOW},
                          {"gfx950", MASK_NONE},
                          {"gfx950", MASK_SLIDING_WINDOW},
                          {"gfx950", BWD_MASK_BOTTOM_RIGHT_SLIDING_WINDOW}})
    {
        for(const char* dtype : {"bf16", "fp16"})
        {
            SCOPED_TRACE(std::string(arch) + " " + dtype + " mask=" + std::to_string(mask));
            const auto request = makeRequest(dtype, 128, 128, mask);
            for(const auto stage :
                {PipelineStage::ODO, PipelineStage::DQDKDV, PipelineStage::DQ_CONVERT})
            {
                EXPECT_TRUE(resolvedRow(stage, arch, request).has_value());
            }
        }
    }

    EXPECT_FALSE(resolvedDqdkdv("gfx942",
                                makeRequest("bf16", 128, 128, BWD_MASK_BOTTOM_RIGHT_SLIDING_WINDOW))
                     .has_value());
}

// get_padded_hdim(): D_qk == D_v rounds up to 64, 128 or 192; gfx950 (192, 128)
// keeps its dims, and every other pair passes through unpadded.
TEST(TestSdpaBwdPaddedHeadDims, RoundsUpToTheRegistryHeadDims)
{
    using bwd_dispatch::paddedHeadDims;
    struct Case
    {
        const char* arch;
        int hdimQ;
        int hdimV;
        int paddedQ;
        int paddedV;
    };
    for(const auto& c : std::vector<Case>{{"gfx942", 8, 8, 64, 64},
                                          {"gfx942", 64, 64, 64, 64},
                                          {"gfx942", 72, 72, 128, 128},
                                          {"gfx942", 128, 128, 128, 128},
                                          {"gfx942", 136, 136, 192, 192},
                                          {"gfx942", 192, 192, 192, 192},
                                          {"gfx942", 256, 256, 256, 256},
                                          {"gfx942", 192, 128, 192, 128},
                                          {"gfx942", 96, 64, 96, 64},
                                          {"gfx950", 96, 96, 128, 128},
                                          {"gfx950", 192, 128, 192, 128}})
    {
        SCOPED_TRACE(std::string(c.arch) + " " + std::to_string(c.hdimQ) + "x"
                     + std::to_string(c.hdimV));
        const auto padded = paddedHeadDims(c.arch, c.hdimQ, c.hdimV);
        EXPECT_EQ(padded.q, c.paddedQ);
        EXPECT_EQ(padded.v, c.paddedV);
    }
}

// Each stage matches on its own AITER key: odo on padded hdim_v, dq_convert on
// padded hdim_q. gfx950 (192, 128) therefore runs the hd128 odo kernel, the
// 192 x 128 dqdkdv kernel and the hd192 dq_convert kernel.
TEST(TestSdpaBwdRegistryLookup, Gfx950Hd192x128ResolvesPerStageHeadDims)
{
    using bwd_dispatch::PipelineStage;
    for(const char* dtype : {"bf16", "fp16"})
    {
        SCOPED_TRACE(dtype);
        const auto request = makeRequest(dtype, 192, 128);

        const auto odo = resolvedRow(PipelineStage::ODO, "gfx950", request);
        ASSERT_TRUE(odo.has_value());
        EXPECT_EQ(odo->hdim_v, 128);

        const auto dqdkdv = resolvedDqdkdv("gfx950", request);
        ASSERT_TRUE(dqdkdv.has_value());
        EXPECT_EQ(std::make_pair(dqdkdv->hdim_q, dqdkdv->hdim_v), std::make_pair(192, 128));

        const auto dqConvert = resolvedRow(PipelineStage::DQ_CONVERT, "gfx950", request);
        ASSERT_TRUE(dqConvert.has_value());
        EXPECT_EQ(dqConvert->hdim_q, 192);
        EXPECT_EQ(dqConvert->dtype, dtype);
    }
}

// gfx942 hd64 ships only (pssk = 1, pddv = 0) A32 rows; with no padding the
// lookup requests pddv = 0 and resolves them.
TEST(TestSdpaBwdRegistryLookup, Gfx942Hd64ResolvesThePsskRows)
{
    using bwd_dispatch::PipelineStage;
    for(const char* dtype : {"bf16", "fp16"})
    {
        for(const int mask : {MASK_NONE, MASK_TOP_LEFT_CAUSAL, MASK_BOTTOM_RIGHT_CAUSAL})
        {
            SCOPED_TRACE(std::string(dtype) + " mask=" + std::to_string(mask));
            const auto request = makeRequest(dtype, 64, 64, mask);
            EXPECT_TRUE(resolvedRow(PipelineStage::ODO, "gfx942", request).has_value());
            EXPECT_TRUE(resolvedRow(PipelineStage::DQ_CONVERT, "gfx942", request).has_value());
            EXPECT_EQ(resolvedPsskPddv("gfx942", request), K_PSSK_ONLY);
            // S_q == S_kv on a tile multiple asks for (0, 0); hd64 A32 has no
            // such row, so the (1, 0) row above it serves.
            EXPECT_EQ(resolvedPsskPddv("gfx942", withSeqLens(request, 384, 384)), K_PSSK_ONLY);
        }
    }
}

// A padded head dim asks for pddv = 1. D = 96 runs the hd128 psskddv kernel on
// gfx942 even when S_q == S_kv is a multiple of the tile: the unpadded (0, 0)
// row does not handle padding and is not above (pssk, 1).
TEST(TestSdpaBwdRegistryLookup, PaddedHeadDimSelectsAPddvRow)
{
    using bwd_dispatch::PipelineStage;
    for(const auto& [arch, hdim] :
        std::vector<std::pair<const char*, int>>{{"gfx942", 96}, {"gfx942", 72}, {"gfx950", 72}})
    {
        for(const auto& request : {makeRequest("bf16", hdim, hdim),
                                   withSeqLens(makeRequest("bf16", hdim, hdim), 384, 384)})
        {
            SCOPED_TRACE(std::string(arch) + " D=" + std::to_string(hdim)
                         + " S=" + std::to_string(request.seqLenKv));
            const auto dqdkdv = resolvedDqdkdv(arch, request);
            ASSERT_TRUE(dqdkdv.has_value());
            EXPECT_EQ(dqdkdv->hdim_q, 128);
            EXPECT_EQ(std::make_pair(dqdkdv->pssk, dqdkdv->pddv), K_PSSK_PDDV);
            EXPECT_EQ(resolvedRow(PipelineStage::ODO, arch, request).value().hdim_v, 128);
            EXPECT_EQ(resolvedRow(PipelineStage::DQ_CONVERT, arch, request).value().hdim_q, 128);
        }
    }
}

// gfx942 hd128 A32: S_q == S_kv on a multiple of the 192-row tile asks for
// pssk = 0, and the unpadded (0, 0) kernel serves it when the strides allow.
// Any other sequence pair, or strides the pssk = 0 kernels cannot take, falls
// back to the psskddv row.
TEST(TestSdpaBwdRegistryLookup, Gfx942Hd128SelectsTheUnpaddedRowOnlyWhenItApplies)
{
    for(const char* dtype : {"bf16", "fp16"})
    {
        for(const int mask : {MASK_NONE, MASK_TOP_LEFT_CAUSAL})
        {
            SCOPED_TRACE(std::string(dtype) + " mask=" + std::to_string(mask));
            const auto request = makeRequest(dtype, 128, 128, mask);
            EXPECT_EQ(resolvedPsskPddv("gfx942", withSeqLens(request, 384, 384)), K_UNPADDED);
            EXPECT_EQ(resolvedPsskPddv("gfx942", withSeqLens(request, 256, 256)), K_PSSK_PDDV);
            EXPECT_EQ(resolvedPsskPddv("gfx942", withSeqLens(request, 384, 768)), K_PSSK_PDDV);

            auto stridesOff = withSeqLens(request, 384, 384);
            stridesOff.psskOffStridesOk = false;
            EXPECT_EQ(resolvedPsskPddv("gfx942", stridesOff), K_PSSK_PDDV);
        }
    }
    // Bottom-right causal and window kernels ship only as psskddv rows.
    for(const int mask : {MASK_BOTTOM_RIGHT_CAUSAL, MASK_SLIDING_WINDOW})
    {
        SCOPED_TRACE(mask);
        EXPECT_EQ(
            resolvedPsskPddv("gfx942", withSeqLens(makeRequest("bf16", 128, 128, mask), 384, 384)),
            K_PSSK_PDDV);
    }
}

// gfx942 A16 rows count 64-row K/V tiles whatever their ts (192): S = 128
// asks for pssk = 0. They are all pssk = 0, so without the stride layout no
// row is left.
TEST(TestSdpaBwdRegistryLookup, Gfx942A16RowsUse64RowTilesAndNeedTheStrideLayout)
{
    auto request = withSeqLens(makeRequest("bf16", 128, 128), 128, 128);
    request.atomic32 = ATOMIC_A16;
    EXPECT_EQ(resolvedPsskPddv("gfx942", request), K_UNPADDED);

    request.psskOffStridesOk = false;
    EXPECT_FALSE(resolvedPsskPddv("gfx942", request).has_value());
}

// When the requested (pssk, pddv) row is missing and more than one row lies
// above it, the smallest wins; AITER keeps whichever its unordered_map yields
// last. gfx942 hd128 A32 group mode ships (1, 0) and (1, 1); a request for
// (0, 0) gets (1, 0).
TEST(TestSdpaBwdRegistryLookup, FallbackPicksTheSmallestRowAboveTheRequest)
{
    auto request = withSeqLens(makeRequest("bf16", 128, 128), 384, 384);
    request.mode = MODE_GROUP;
    EXPECT_EQ(resolvedPsskPddv("gfx942", request), K_PSSK_ONLY);
}

// gfx950 serves only psskddv rows on its ASM path; the (192, 128) request is
// unpadded (pddv = 0) and resolves the (1, 1) row above it.
TEST(TestSdpaBwdRegistryLookup, Gfx950ResolvesThePsskddvRows)
{
    for(const auto& request : {withSeqLens(makeRequest("bf16", 128, 128), 512, 512),
                               withSeqLens(makeRequest("bf16", 192, 128), 384, 384)})
    {
        SCOPED_TRACE(std::to_string(request.hdimQ) + "x" + std::to_string(request.hdimV));
        EXPECT_EQ(resolvedPsskPddv("gfx950", request), K_PSSK_PDDV);
    }
}

// =============================================================================
// The A16 accumulator
// =============================================================================

namespace
{
// A batch-mode A16 request with the given mask and sequence lengths. The
// mask goes into the request as it is (no requestedMaskOrdinal remap).
bwd_dispatch::KernelRequest makeA16Request(
    const std::string& dtype, int hdimQ, int hdimV, int mask, int64_t seqLenQ, int64_t seqLenKv)
{
    auto request = withSeqLens(makeRequest(dtype, hdimQ, hdimV, mask), seqLenQ, seqLenKv);
    request.atomic32 = ATOMIC_A16;
    return request;
}
} // namespace

// mha_bwd.cu fmha_v3_bwd() (:327-355): A32 runs dq_convert on both arches;
// A16 runs dq_shuffle on gfx950 for every head dim but 64, and has no post
// stage on gfx942 (dqdkdv writes dQ).
TEST(TestSdpaBwdPostStage, FollowsTheAccumulatorAndArch)
{
    using bwd_dispatch::PipelineStage;
    using bwd_dispatch::postStage;
    for(const char* arch : {"gfx942", "gfx950"})
    {
        for(const int hdim : {64, 96, 128, 192})
        {
            EXPECT_EQ(postStage(arch, ATOMIC_A32, hdim), PipelineStage::DQ_CONVERT);
        }
    }
    for(const int hdim : {96, 128, 192})
    {
        EXPECT_EQ(postStage("gfx950", ATOMIC_A16, hdim), PipelineStage::DQ_SHUFFLE);
        EXPECT_FALSE(postStage("gfx942", ATOMIC_A16, hdim).has_value());
    }
    EXPECT_FALSE(postStage("gfx950", ATOMIC_A16, 64).has_value());
}

// gfx950 A16 runs the dq_shuffle row of the padded hdim_q (mha_bwd.cu
// :113-133). A16 ignores the dtype there (:123), so fp16 runs the bf16 rows,
// the only dq_shuffle rows AITER ships.
TEST(TestSdpaBwdRegistryLookup, Gfx950A16ResolvesTheDqShuffleRows)
{
    using bwd_dispatch::PipelineStage;
    struct Case
    {
        int hdimQ;
        int hdimV;
        int shuffleHdim;
    };
    for(const char* dtype : {"bf16", "fp16"})
    {
        for(const auto& [hdimQ, hdimV, shuffleHdim] :
            std::vector<Case>{{128, 128, 128}, {96, 96, 128}, {192, 128, 192}})
        {
            for(const int mask : {MASK_NONE, MASK_TOP_LEFT_CAUSAL, MASK_BOTTOM_RIGHT_CAUSAL})
            {
                SCOPED_TRACE(std::string(dtype) + " " + std::to_string(hdimQ) + "x"
                             + std::to_string(hdimV) + " mask=" + std::to_string(mask));
                const auto request = makeA16Request(dtype, hdimQ, hdimV, mask, 200, 512);

                const auto dqdkdv = resolvedDqdkdv("gfx950", request);
                ASSERT_TRUE(dqdkdv.has_value());
                EXPECT_EQ(dqdkdv->atomic32, ATOMIC_A16);
                EXPECT_EQ(dqdkdv->dtype, dtype);

                const auto shuffle = resolvedRow(PipelineStage::DQ_SHUFFLE, "gfx950", request);
                ASSERT_TRUE(shuffle.has_value());
                EXPECT_EQ(shuffle->hdim_q, shuffleHdim);
                EXPECT_EQ(shuffle->mode, MODE_BATCH);
                EXPECT_EQ(shuffle->dtype, "bf16");
            }
        }
    }
    // The windows have A16 rows on gfx950 hd128.
    for(const int mask : {MASK_SLIDING_WINDOW, BWD_MASK_BOTTOM_RIGHT_SLIDING_WINDOW})
    {
        EXPECT_TRUE(
            resolvedDqdkdv("gfx950", makeA16Request("bf16", 128, 128, mask, 256, 512)).has_value());
    }
}

// Every gfx942 A16 row is pssk = 0, so A16 resolves only when S_q == S_kv,
// S_kv is a multiple of 64 (the A16 tile count, mha_bwd.cu:89-95) and the
// strides allow the pssk = 0 kernels, for D in [64, 128], no mask or top-left
// causal (mha.py can_impl_fmha_v3_bwd() np() and pddv()). There are no A16
// rows for D > 128, for windows, or for bottom-right causal.
TEST(TestSdpaBwdRegistryLookup, Gfx942A16ResolvesOnlyWhereThePssk0RowsApply)
{
    for(const char* dtype : {"bf16", "fp16"})
    {
        for(const int hdim : {64, 96, 128})
        {
            for(const int mask : {MASK_NONE, MASK_TOP_LEFT_CAUSAL})
            {
                SCOPED_TRACE(std::string(dtype) + " D=" + std::to_string(hdim)
                             + " mask=" + std::to_string(mask));
                const auto request = makeA16Request(dtype, hdim, hdim, mask, 256, 256);
                const auto row = resolvedDqdkdv("gfx942", request);
                ASSERT_TRUE(row.has_value());
                EXPECT_EQ(row->atomic32, ATOMIC_A16);
                EXPECT_EQ(row->pssk, 0);
                EXPECT_EQ(row->pddv, hdim == 96 ? 1 : 0);

                EXPECT_FALSE(resolvedDqdkdv("gfx942", withSeqLens(request, 256, 512)).has_value());
                EXPECT_FALSE(resolvedDqdkdv("gfx942", withSeqLens(request, 200, 200)).has_value());
                auto stridesOff = request;
                stridesOff.psskOffStridesOk = false;
                EXPECT_FALSE(resolvedDqdkdv("gfx942", stridesOff).has_value());
            }
        }
        for(const int hdim : {136, 192})
        {
            EXPECT_FALSE(
                resolvedDqdkdv("gfx942", makeA16Request(dtype, hdim, hdim, MASK_NONE, 256, 256))
                    .has_value());
        }
        for(const int mask : {MASK_BOTTOM_RIGHT_CAUSAL, MASK_SLIDING_WINDOW})
        {
            EXPECT_FALSE(resolvedDqdkdv("gfx942", makeA16Request(dtype, 128, 128, mask, 256, 256))
                             .has_value());
        }
    }
}

// The pssk = 0 dqdkdv kernels add (batch, head) offsets to their bases in 32
// bits. A32 rows qualify only while dq_acc, dK and dV span at most UINT32_MAX
// bytes and Q's batch stride is one they derive the dq_acc batch offset from;
// otherwise the lookup falls back to the psskddv row, whose bases are 64-bit.
// bf16/fp16 hd128, H_q = 32, S = 12288 (a multiple of the 192-row tile): B = 21
// is 4,227,858,432 dq_acc bytes and B = 22 is 4,429,185,024.
TEST(TestSdpaBwdRegistryLookup, Gfx942Pssk0RowsNeedEveryOffsetToFit32Bits)
{
    constexpr int64_t K_U32_MAX = UINT32_MAX;
    for(const char* dtype : {"bf16", "fp16"})
    {
        for(const int mask : {MASK_NONE, MASK_TOP_LEFT_CAUSAL})
        {
            SCOPED_TRACE(std::string(dtype) + " mask=" + std::to_string(mask));
            auto request = withSeqLens(makeRequest(dtype, 128, 128, mask), 12288, 12288);
            request.nheadQ = 32;
            request.batch = 21;
            EXPECT_EQ(resolvedPsskPddv("gfx942", request), K_UNPADDED);
            request.batch = 22;
            EXPECT_EQ(resolvedPsskPddv("gfx942", request), K_PSSK_PDDV);

            auto grads = withSeqLens(makeRequest(dtype, 128, 128, mask), 384, 384);
            grads.gradKvSpanBytes = K_U32_MAX;
            EXPECT_EQ(resolvedPsskPddv("gfx942", grads), K_UNPADDED);
            grads.gradKvSpanBytes = K_U32_MAX + 1;
            EXPECT_EQ(resolvedPsskPddv("gfx942", grads), K_PSSK_PDDV);

            // The A32 kernels read Q, K, V and dO through 64-bit bases.
            auto inputs = withSeqLens(makeRequest(dtype, 128, 128, mask), 384, 384);
            inputs.inputSpanBytes = K_U32_MAX + 1;
            EXPECT_EQ(resolvedPsskPddv("gfx942", inputs), K_UNPADDED);

            auto qBatch = withSeqLens(makeRequest(dtype, 128, 128, mask), 384, 384);
            qBatch.qBatchStrideMatchesDqAcc = false;
            EXPECT_EQ(resolvedPsskPddv("gfx942", qBatch), K_PSSK_PDDV);
        }
    }

    // gfx942 A16 has only pssk = 0 rows, which form every tensor's offset in 32
    // bits: a tensor past UINT32_MAX bytes leaves no row. They have no dq_acc,
    // so Q's batch stride does not matter.
    auto a16 = makeA16Request("bf16", 128, 128, MASK_NONE, 256, 256);
    a16.qBatchStrideMatchesDqAcc = false;
    a16.inputSpanBytes = K_U32_MAX;
    a16.gradKvSpanBytes = K_U32_MAX;
    EXPECT_EQ(resolvedPsskPddv("gfx942", a16), K_UNPADDED);
    auto bigInput = a16;
    bigInput.inputSpanBytes = K_U32_MAX + 1;
    EXPECT_FALSE(resolvedPsskPddv("gfx942", bigInput).has_value());
    auto bigGrad = a16;
    bigGrad.gradKvSpanBytes = K_U32_MAX + 1;
    EXPECT_FALSE(resolvedPsskPddv("gfx942", bigGrad).has_value());
}

namespace
{
// Dims and strides of one [B, H, S, D] tensor.
struct BhsdLayout
{
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
};

BhsdLayout packedLayout(int64_t batch, int64_t heads, int64_t seq, int64_t headDim)
{
    std::vector<int64_t> dims{batch, heads, seq, headDim};
    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    return {std::move(dims), std::move(strides)};
}

// The same layout with its batch stride set to batchStride elements.
BhsdLayout withBatchStride(BhsdLayout layout, int64_t batchStride)
{
    layout.strides[0] = batchStride;
    return layout;
}

// The gfx942 request bwd_dispatch::makeKernelRequest() builds for an unmasked
// bf16 graph whose Q, O, dO and dQ have layout qo and K, V, dK and dV layout kv.
bwd_dispatch::KernelRequest gfx942GraphRequest(const BhsdLayout& qo,
                                               const BhsdLayout& kv,
                                               bwd_dispatch::AccumulatorMode accMode)
{
    using hipdnn_flatbuffers_sdk::data_objects::SdpaBackwardAttributes;
    auto builder = hipdnn_test_sdk::utilities::createValidSdpaBwdGraph(
        qo.dims,
        qo.strides,
        kv.dims,
        kv.strides,
        kv.dims,
        kv.strides,
        qo.dims,
        qo.strides,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto& attrs = graph.nodeWrappers().front()->attributesAs<SdpaBackwardAttributes>();
    const auto& tensors = graph.getTensorMap();
    return bwd_dispatch::makeKernelRequest("gfx942",
                                           "bf16",
                                           attrs,
                                           *tensors.at(attrs.q_tensor_uid()),
                                           *tensors.at(attrs.k_tensor_uid()),
                                           *tensors.at(attrs.v_tensor_uid()),
                                           *tensors.at(attrs.do_tensor_uid()),
                                           *tensors.at(attrs.dk_tensor_uid()),
                                           *tensors.at(attrs.dv_tensor_uid()),
                                           accMode)
        .request;
}
} // namespace

// The request built from a graph carries the facts the pssk = 0 offset guard
// reads, so the guard sees the graph's dq_acc and tensor spans. Packed bf16
// hd128, H_q = 32, S = 12288: the FP32 dq_acc is 4,227,858,432 bytes at B = 21
// and 4,429,185,024 at B = 22.
TEST(TestSdpaBwdGraphRequest, Gfx942A32DqAccPast4GiBLeavesThePssk0Row)
{
    using bwd_dispatch::AccumulatorMode;
    constexpr int64_t K_BATCH_BYTES = int64_t{32} * 12288 * 128 * 2;

    const auto fits = packedLayout(21, 32, 12288, 128);
    const auto request = gfx942GraphRequest(fits, fits, AccumulatorMode::A32);
    EXPECT_EQ(request.batch, 21);
    EXPECT_EQ(request.nheadQ, 32);
    EXPECT_EQ(request.atomic32, ATOMIC_A32);
    EXPECT_EQ(request.gradKvSpanBytes, 21 * K_BATCH_BYTES);
    EXPECT_EQ(request.inputSpanBytes, 21 * K_BATCH_BYTES);
    EXPECT_TRUE(request.psskOffStridesOk);
    EXPECT_TRUE(request.qBatchStrideMatchesDqAcc);
    EXPECT_EQ(resolvedPsskPddv("gfx942", request), K_UNPADDED);

    const auto wraps = packedLayout(22, 32, 12288, 128);
    EXPECT_EQ(resolvedPsskPddv("gfx942", gfx942GraphRequest(wraps, wraps, AccumulatorMode::A32)),
              K_PSSK_PDDV);
}

// K, V, dK and dV with a 2^31-element batch stride span 2^32 + 786,432 bytes,
// while dq_acc (packed [2, 8, 384, 128] FP32) stays small: the A32 pssk = 0
// row is left for its dK and dV offsets alone.
TEST(TestSdpaBwdGraphRequest, Gfx942A32GradKvPast4GiBLeavesThePssk0Row)
{
    using bwd_dispatch::AccumulatorMode;
    const auto qo = packedLayout(2, 8, 384, 128);
    const auto kv = withBatchStride(qo, int64_t{1} << 31);
    const auto request = gfx942GraphRequest(qo, kv, AccumulatorMode::A32);
    EXPECT_EQ(request.gradKvSpanBytes, (int64_t{1} << 32) + 786432);
    EXPECT_TRUE(request.psskOffStridesOk);
    EXPECT_TRUE(request.qBatchStrideMatchesDqAcc);
    EXPECT_EQ(resolvedPsskPddv("gfx942", request), K_PSSK_PDDV);
}

// Q, O, dO and dQ with a 2^31-element batch stride span 2^32 + 786,432 bytes;
// K, V, dK and dV stay packed. gfx942 A16 has only pssk = 0 rows, so no row
// serves the graph, and A32 leaves the pssk = 0 row because the padded Q batch
// stride is not the dq_acc one.
TEST(TestSdpaBwdGraphRequest, Gfx942BatchPaddedQLeavesEveryPssk0Row)
{
    using bwd_dispatch::AccumulatorMode;
    const auto kv = packedLayout(2, 8, 384, 128);
    const auto qo = withBatchStride(kv, int64_t{1} << 31);

    const auto packedA16 = gfx942GraphRequest(kv, kv, AccumulatorMode::A16);
    EXPECT_EQ(packedA16.atomic32, ATOMIC_A16);
    EXPECT_EQ(resolvedPsskPddv("gfx942", packedA16), K_UNPADDED);

    const auto a16 = gfx942GraphRequest(qo, kv, AccumulatorMode::A16);
    EXPECT_EQ(a16.inputSpanBytes, (int64_t{1} << 32) + 786432);
    EXPECT_EQ(a16.gradKvSpanBytes, int64_t{2} * 8 * 384 * 128 * 2);
    EXPECT_TRUE(a16.psskOffStridesOk);
    EXPECT_FALSE(resolvedPsskPddv("gfx942", a16).has_value());

    const auto a32 = gfx942GraphRequest(qo, kv, AccumulatorMode::A32);
    EXPECT_FALSE(a32.qBatchStrideMatchesDqAcc);
    EXPECT_EQ(resolvedPsskPddv("gfx942", a32), K_PSSK_PDDV);
}

// Bytes from the first element to one past the last; (n - 1) * stride on each
// axis plus one row of D elements.
TEST(TestSdpaBwdSpanBytes, CountsToOnePastTheLastElement)
{
    using bwd_dispatch::bhsdSpanBytes;
    // Packed [2, 1, 2^23, 128] bf16 is exactly 2^32 bytes.
    EXPECT_EQ(bhsdSpanBytes({int64_t{1} << 30, int64_t{1} << 30, 128}, 2, 1, 8388608, 128, 2),
              int64_t{1} << 32);
    // BSHD [B = 1, H = 4, S = 8, D = 128]: strides {batch 4096, head 128, seq 512}.
    EXPECT_EQ(bhsdSpanBytes({4096, 128, 512}, 1, 4, 8, 128, 2), 4096 * 2);
    // Rows padded to 136 elements: the last row ends after D, not after 136.
    EXPECT_EQ(bhsdSpanBytes({int64_t{2} * 136, 136, 0}, 1, 2, 1, 128, 2), (136 + 128) * 2);
    EXPECT_EQ(bhsdSpanBytes({INT64_MAX / 2, 1, 1}, 4, 1, 1, 1, 2), INT64_MAX);
    EXPECT_EQ(bhsdSpanBytes({4096, 128, 512}, 0, 4, 8, 128, 2), 0);
}

// The A32 pssk = 0 kernels form the dq_acc batch offset from Q's batch stride;
// it is the packed FP32 [B, H_q, S_q, D_qk] offset for packed BHSD and BSHD
// (H_q * S_q * D_qk) and for SBHD (H_q * D_qk, below the sequence stride).
TEST(TestSdpaBwdQBatchStride, MatchesDqAccOnlyForBatchPackedQ)
{
    using bwd_dispatch::qBatchStrideMatchesDqAcc;
    constexpr int64_t K_B = 2;
    constexpr int64_t K_H = 4;
    constexpr int64_t K_S = 384;
    constexpr int64_t K_D = 128;
    EXPECT_TRUE(qBatchStrideMatchesDqAcc({K_H * K_S * K_D, K_S * K_D, K_D}, K_B, K_H, K_S, K_D));
    EXPECT_TRUE(qBatchStrideMatchesDqAcc({K_S * K_H * K_D, K_D, K_H * K_D}, K_B, K_H, K_S, K_D));
    EXPECT_TRUE(qBatchStrideMatchesDqAcc({K_H * K_D, K_D, K_B * K_H * K_D}, K_B, K_H, K_S, K_D));
    // Batch-padded Q, and H_q * D_qk at or above the sequence stride.
    EXPECT_FALSE(
        qBatchStrideMatchesDqAcc({(K_H * K_S + 1) * K_D, K_S * K_D, K_D}, K_B, K_H, K_S, K_D));
    EXPECT_FALSE(qBatchStrideMatchesDqAcc({K_H * K_D, K_D, K_H * K_D}, K_B, K_H, K_S, K_D));
    // B == 1 has no batch offset.
    EXPECT_TRUE(
        qBatchStrideMatchesDqAcc({(K_H * K_S + 1) * K_D, K_S * K_D, K_D}, 1, K_H, K_S, K_D));
}

// mha_bwd.cu fmha_v3_bwd() (:374-377): gfx942 A16 asks for the top-left
// causal kernel for a bottom-right causal mask when S_q == S_kv, where the two
// agree, and so serves it from the A16 causal rows.
TEST(TestSdpaBwdRequestedMaskOrdinal, Gfx942A16RunsBottomRightCausalAsTopLeftOnSquareShapes)
{
    using bwd_dispatch::requestedMaskOrdinal;
    EXPECT_EQ(requestedMaskOrdinal("gfx942", MASK_BOTTOM_RIGHT_CAUSAL, ATOMIC_A16, 256, 256),
              MASK_TOP_LEFT_CAUSAL);
    EXPECT_EQ(requestedMaskOrdinal("gfx942", MASK_BOTTOM_RIGHT_CAUSAL, ATOMIC_A16, 256, 512),
              MASK_BOTTOM_RIGHT_CAUSAL);
    EXPECT_EQ(requestedMaskOrdinal("gfx942", MASK_BOTTOM_RIGHT_CAUSAL, ATOMIC_A32, 256, 256),
              MASK_BOTTOM_RIGHT_CAUSAL);
    EXPECT_EQ(requestedMaskOrdinal("gfx950", MASK_BOTTOM_RIGHT_CAUSAL, ATOMIC_A16, 256, 256),
              MASK_BOTTOM_RIGHT_CAUSAL);
    for(const int mask : {MASK_NONE, MASK_TOP_LEFT_CAUSAL, MASK_SLIDING_WINDOW})
    {
        EXPECT_EQ(requestedMaskOrdinal("gfx942", mask, ATOMIC_A16, 256, 256), mask);
    }

    const auto row = resolvedDqdkdv(
        "gfx942",
        makeA16Request(
            "bf16",
            128,
            128,
            requestedMaskOrdinal("gfx942", MASK_BOTTOM_RIGHT_CAUSAL, ATOMIC_A16, 256, 256),
            256,
            256));
    ASSERT_TRUE(row.has_value());
    EXPECT_EQ(row->mask, MASK_TOP_LEFT_CAUSAL);
}

// dQ written directly takes Q's strides; a stride on an axis of length 1 is
// never used.
TEST(TestSdpaBwdDqSharesQStrides, ComparesTheAxesLongerThanOne)
{
    using bwd_dispatch::dqSharesQStrides;
    const bwd_dispatch::BhsStrides q{4096, 1024, 128};
    EXPECT_TRUE(dqSharesQStrides(q, q, 2, 4, 8));
    EXPECT_FALSE(dqSharesQStrides(q, {4096, 1024, 136}, 2, 4, 8));
    EXPECT_FALSE(dqSharesQStrides(q, {4096, 512, 128}, 2, 4, 8));
    EXPECT_FALSE(dqSharesQStrides(q, {8192, 1024, 128}, 2, 4, 8));
    EXPECT_TRUE(dqSharesQStrides(q, {7, 1024, 128}, 1, 4, 8));
    EXPECT_TRUE(dqSharesQStrides(q, {4096, 7, 128}, 2, 1, 8));
    EXPECT_TRUE(dqSharesQStrides(q, {4096, 1024, 7}, 2, 4, 1));
}

// =============================================================================
// The stride layout the pssk = 0 kernels need
// =============================================================================

namespace
{
// Packed [B, H, S, D] strides for H heads, S rows and head dim D.
bwd_dispatch::BhsStrides packedBhs(int64_t heads, int64_t seq, int64_t headDim)
{
    return {heads * seq * headDim, seq * headDim, headDim};
}

bool stridesMatch(const bwd_dispatch::BhsStrides& q,
                  const bwd_dispatch::BhsStrides& k,
                  const bwd_dispatch::BhsStrides& v,
                  const bwd_dispatch::BhsStrides& dO,
                  const bwd_dispatch::BhsStrides& dk,
                  const bwd_dispatch::BhsStrides& dv,
                  int64_t nheadQ = 8,
                  int64_t nheadK = 8)
{
    return bwd_dispatch::psskOffStridesMatch(q, k, v, dO, dk, dv, nheadQ, nheadK);
}
} // namespace

TEST(TestSdpaBwdPsskOffStrides, PackedTensorsMatch)
{
    const auto packed = packedBhs(8, 384, 128);
    EXPECT_TRUE(stridesMatch(packed, packed, packed, packed, packed, packed));
}

// One mismatch per AITER condition: Q vs dO on each axis, K vs V on each axis,
// the dK and dV head strides, and the dK and dV batch strides.
TEST(TestSdpaBwdPsskOffStrides, EachAiterConditionIsChecked)
{
    const auto p = packedBhs(8, 384, 128);
    const bwd_dispatch::BhsStrides bshd{int64_t{384} * 8 * 128, 128, int64_t{8} * 128};
    const bwd_dispatch::BhsStrides paddedRows{int64_t{8} * 384 * 136, int64_t{384} * 136, 136};
    const bwd_dispatch::BhsStrides biggerBatch{p.batch * 2, p.head, p.seq};

    EXPECT_FALSE(stridesMatch(p, p, p, bshd, p, p)) << "dO layout differs from Q";
    EXPECT_FALSE(stridesMatch(p, p, p, paddedRows, p, p)) << "dO rows differ from Q";
    EXPECT_FALSE(stridesMatch(p, p, p, biggerBatch, p, p)) << "dO batch stride differs from Q";
    EXPECT_FALSE(stridesMatch(p, p, bshd, p, p, p)) << "V layout differs from K";
    EXPECT_FALSE(stridesMatch(p, p, paddedRows, p, p, p)) << "V rows differ from K";
    EXPECT_FALSE(stridesMatch(p, p, biggerBatch, p, p, p)) << "V batch stride differs from K";
    EXPECT_FALSE(stridesMatch(p, p, p, p, bshd, p)) << "dK head stride differs from K";
    EXPECT_FALSE(stridesMatch(p, p, p, p, p, bshd)) << "dV head stride differs from V";
    EXPECT_FALSE(stridesMatch(p, p, p, p, biggerBatch, p)) << "dK batch stride differs from K";
    EXPECT_FALSE(stridesMatch(p, p, p, p, p, biggerBatch)) << "dV batch stride differs from V";
}

// Not checked by AITER: O and dQ, and the dK and dV row strides.
TEST(TestSdpaBwdPsskOffStrides, DkDvRowStridesAreNotChecked)
{
    const auto p = packedBhs(8, 384, 128);
    const bwd_dispatch::BhsStrides widerRows{p.batch, p.head, 136};
    EXPECT_TRUE(stridesMatch(p, p, p, p, widerRows, widerRows));
}

// The batch ratio is nhead_q / nhead_k: GQA dK and dV hold nhead_q heads.
// A zero K batch stride (a broadcast) fails it, as it raises in Python.
TEST(TestSdpaBwdPsskOffStrides, BatchStrideRatioFollowsTheHeadRatio)
{
    const auto kv = packedBhs(2, 384, 128);
    const auto qo = packedBhs(8, 384, 128);
    const bwd_dispatch::BhsStrides expandedGrad{kv.batch * 4, kv.head, kv.seq};
    EXPECT_TRUE(stridesMatch(qo, kv, kv, qo, expandedGrad, expandedGrad, 8, 2));
    EXPECT_FALSE(stridesMatch(qo, kv, kv, qo, kv, kv, 8, 2));

    const bwd_dispatch::BhsStrides broadcastBatch{0, kv.head, kv.seq};
    EXPECT_FALSE(
        stridesMatch(qo, broadcastBatch, broadcastBatch, qo, broadcastBatch, broadcastBatch, 2, 2));
}

// =============================================================================
// bf16_cvt matching: `cfg.bf16_cvt == 3 || cfg.bf16_cvt == requested`
// =============================================================================

// The bf16_cvt of the dqdkdv row an A32 batch lookup resolves, or std::nullopt
// when no row matches.
std::optional<int> resolvedDqdkdvBf16Cvt(const char* arch, const char* dtype, int hdim, int bf16Cvt)
{
    auto request = makeRequest(dtype, hdim, hdim);
    request.bf16Cvt = bf16Cvt;
    const auto row = resolvedDqdkdv(arch, request);
    if(!row)
    {
        return std::nullopt;
    }
    return row->bf16_cvt;
}

// gfx950 hd192 bf16 ships one row per rounding mode (0, 1, 2) for an otherwise
// identical tuple, as gfx942 does for every bf16 row. The lookup must return the
// requested one every time, not whichever the unordered registry yields first.
TEST(TestSdpaBwdRegistryLookup, PerRoundingModeRowsResolveTheRequestedMode)
{
    for(const auto& [arch, hdim] :
        std::vector<std::pair<const char*, int>>{{"gfx950", 192}, {"gfx942", 128}})
    {
        for(const int bf16Cvt : {0, 1, 2})
        {
            SCOPED_TRACE(std::string(arch) + " hd" + std::to_string(hdim)
                         + " bf16_cvt=" + std::to_string(bf16Cvt));
            EXPECT_EQ(resolvedDqdkdvBf16Cvt(arch, "bf16", hdim, bf16Cvt), bf16Cvt);
        }
    }
}

// A bf16 row stored with bf16_cvt=3 (gfx950 hd128) serves every requested mode.
TEST(TestSdpaBwdRegistryLookup, AllModeRowServesEveryRoundingMode)
{
    for(const int bf16Cvt : {0, 1, 2})
    {
        SCOPED_TRACE(bf16Cvt);
        EXPECT_EQ(resolvedDqdkdvBf16Cvt("gfx950", "bf16", 128, bf16Cvt), BF16_CVT_FP16_SENTINEL);
    }
}

// The rule is not "ignore bf16_cvt": gfx942 has no all-mode bf16 dqdkdv rows,
// so requesting the sentinel for bf16 there finds nothing.
TEST(TestSdpaBwdRegistryLookup, PerRoundingModeRowsDoNotServeTheSentinel)
{
    EXPECT_FALSE(resolvedDqdkdvBf16Cvt("gfx942", "bf16", 128, BF16_CVT_FP16_SENTINEL).has_value());
}

// Every gfx950 dq_convert row carries bf16_cvt = 3, so for the A32 accumulator
// only the dtype key keeps an fp16 graph off the bf16 cast kernel and the
// reverse (AITER matches dtype unless atomic32 == 0).
TEST(TestSdpaBwdRegistryLookup, A32DqConvertMatchesTheDtype)
{
    for(const char* dtype : {"bf16", "fp16"})
    {
        SCOPED_TRACE(dtype);
        const auto row = resolvedRow(
            bwd_dispatch::PipelineStage::DQ_CONVERT, "gfx950", makeRequest(dtype, 128, 128));
        ASSERT_TRUE(row.has_value());
        EXPECT_EQ(row->dtype, dtype);
    }
}

// =============================================================================
// Backward window dispatch: dqdkdv mask ordinal and kernarg coordinates
// =============================================================================

// AITER v0.1.29 mha_bwd.cu asm_mask_type(): only a gfx950 window aligned
// bottom-right selects mask 4; masks keep their MaskType ordinal otherwise.
TEST(TestSdpaBwdDqdkdvMaskOrdinal, Gfx950BottomRightWindowSelectsMask4)
{
    using hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment;
    using plan_utils::MaskType;

    EXPECT_EQ(bwd_dispatch::dqdkdvMaskOrdinal(
                  "gfx950", MaskType::SLIDING_WINDOW, DiagonalAlignment::BOTTOM_RIGHT),
              BWD_MASK_BOTTOM_RIGHT_SLIDING_WINDOW);
    EXPECT_EQ(bwd_dispatch::dqdkdvMaskOrdinal(
                  "gfx950", MaskType::SLIDING_WINDOW, DiagonalAlignment::TOP_LEFT),
              MASK_SLIDING_WINDOW);
    EXPECT_EQ(bwd_dispatch::dqdkdvMaskOrdinal(
                  "gfx942", MaskType::SLIDING_WINDOW, DiagonalAlignment::BOTTOM_RIGHT),
              MASK_SLIDING_WINDOW);
    EXPECT_EQ(bwd_dispatch::dqdkdvMaskOrdinal(
                  "gfx950", MaskType::BOTTOM_RIGHT_CAUSAL, DiagonalAlignment::BOTTOM_RIGHT),
              MASK_BOTTOM_RIGHT_CAUSAL);
    EXPECT_EQ(bwd_dispatch::dqdkdvMaskOrdinal(
                  "gfx950", MaskType::NO_MASK, DiagonalAlignment::BOTTOM_RIGHT),
              MASK_NONE);
}

// Params for a 256 x 512 window plan on arch with the given dqdkdv mask.
SdpaBwdParams makeWindowParams(const char* arch, int32_t maskOrdinal, bool topLeftAlignment)
{
    SdpaBwdParams params{};
    params.archString = arch;
    params.seqLenQ = 256;
    params.seqLenKv = 512;
    params.maskOrdinal = maskOrdinal;
    params.windowLeft = 63;
    params.windowRight = -1;
    params.topLeftAlignment = topLeftAlignment;
    return params;
}

// The gfx950 window kernels take the FlashAttention window sizes unchanged,
// -1 (unbounded) included, for both alignments.
TEST(TestSdpaBwdDqdkdvMaskCoordinates, Gfx950WindowsPassRawWindowSizes)
{
    const std::pair<int32_t, int32_t> raw{63, -1};
    EXPECT_EQ(dqdkdvMaskCoordinates(
                  makeWindowParams("gfx950", BWD_MASK_BOTTOM_RIGHT_SLIDING_WINDOW, false)),
              raw);
    EXPECT_EQ(dqdkdvMaskCoordinates(makeWindowParams("gfx950", MASK_SLIDING_WINDOW, true)), raw);
}

// The gfx942 window kernel takes compute_mask_coordinates() output:
// bottom-right with S_q = 256, S_kv = 512, left 63 and unbounded right gives
// mask_y = 1 + 63 + (256 - 512) and mask_x = 1 + (256 - 1) + (512 - 256).
TEST(TestSdpaBwdDqdkdvMaskCoordinates, Gfx942WindowPassesComputedCoordinates)
{
    const std::pair<int32_t, int32_t> expected{-192, 512};
    EXPECT_EQ(dqdkdvMaskCoordinates(makeWindowParams("gfx942", MASK_SLIDING_WINDOW, false)),
              expected);
}

// Causal and unmasked kernels bake the mask in; their coordinates stay 0.
TEST(TestSdpaBwdDqdkdvMaskCoordinates, NonWindowMasksPassZero)
{
    const std::pair<int32_t, int32_t> zero{0, 0};
    for(const int32_t mask : {MASK_NONE, MASK_TOP_LEFT_CAUSAL, MASK_BOTTOM_RIGHT_CAUSAL})
    {
        SCOPED_TRACE(mask);
        EXPECT_EQ(dqdkdvMaskCoordinates(makeWindowParams("gfx950", mask, false)), zero);
        EXPECT_EQ(dqdkdvMaskCoordinates(makeWindowParams("gfx942", mask, false)), zero);
    }
}

TEST_F(TestSdpaBwdPlanBuilder, IsApplicableRejectsFp8)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    SKIP_IF_NO_DEVICES();

    if(!bwd_dispatch::isSupportedArch(deviceString()))
    {
        GTEST_SKIP();
    }

    auto builder = createSdpaBwdGraph({2, 8, 256, 128}, DataType::FP8_E4M3);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
}

// =============================================================================
// Head dims AITER v0.1.29 sends to the ASM kernels
// =============================================================================

namespace
{
struct HeadDimCase
{
    int64_t headDimQk;
    int64_t headDimV;
    bool gfx942;
    bool gfx950;
};

// gfx942: 64 <= D <= 192, D % 8 == 0, D_qk == D_v (mha.py can_impl_fmha_v3_bwd).
// gfx950: 64 < D <= 128 with D_qk == D_v, or D_qk = 192 with D_v = 128
// (can_impl_fmha_v3_bwd_gfx950); its hd64 and hd192 x hd192 rows go to CK.
const std::vector<HeadDimCase> K_HEAD_DIM_CASES = {
    {56, 56, false, false},
    {64, 64, true, false},
    {72, 72, true, true},
    {96, 96, true, true},
    {100, 100, false, false},
    {128, 128, true, true},
    {136, 136, true, false},
    {192, 192, true, false},
    {200, 200, false, false},
    {192, 128, false, true},
    {128, 64, false, false},
    {128, 192, false, false},
};
} // namespace

TEST(TestSdpaBwdBatchShape, HeadDimsPerArch)
{
    for(const auto& c : K_HEAD_DIM_CASES)
    {
        SCOPED_TRACE(std::to_string(c.headDimQk) + "x" + std::to_string(c.headDimV));
        EXPECT_EQ(bwd_dispatch::isBatchShapeSupported("gfx942", c.headDimQk, c.headDimV), c.gfx942);
        EXPECT_EQ(bwd_dispatch::isBatchShapeSupported("gfx950", c.headDimQk, c.headDimV), c.gfx950);
    }
}

// End to end: every head dim the arch serves also resolves all three kernels
// (padded dims included), and the others are declined.
TEST_F(TestSdpaBwdPlanBuilder, IsApplicableServesTheHeadDimsAiterSendsToAsm)
{
    SKIP_IF_NO_DEVICES();

    const auto arch = deviceString();
    if(!bwd_dispatch::isSupportedArch(arch))
    {
        GTEST_SKIP();
    }

    for(const auto& c : K_HEAD_DIM_CASES)
    {
        SCOPED_TRACE(std::to_string(c.headDimQk) + "x" + std::to_string(c.headDimV));
        auto builder = createPackedSdpaBwdGraph({2, 4, 256, c.headDimQk}, {2, 4, 256, c.headDimV});
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
            builder.GetBufferPointer(), builder.GetSize());
        EXPECT_EQ(_planBuilder.isApplicable(_handle, graphWrapper),
                  arch == "gfx942" ? c.gfx942 : c.gfx950);
    }
}

TEST_F(TestSdpaBwdPlanBuilder, IsApplicableRejectsFractionalGqaRatio)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    SKIP_IF_NO_DEVICES();

    if(!bwd_dispatch::isSupportedArch(deviceString()))
    {
        GTEST_SKIP();
    }

    // nhead_q = 6, nhead_k = 4.  GQA/MQA is now rejected outright (dK/dV head
    // expansion is unimplemented); a fractional ratio would additionally truncate
    // to ratio = 6/4 = 1 and silently drop K/V heads in dispatch.
    const std::vector<int64_t> qDims = {2, 6, 256, 128};
    const std::vector<int64_t> kDims = {2, 4, 256, 128};
    const std::vector<int64_t> vDims = {2, 4, 256, 128};
    const std::vector<int64_t> oDims = {2, 6, 256, 128};

    auto builder = hipdnn_test_sdk::utilities::createValidSdpaBwdGraph(
        qDims,
        hipdnn_data_sdk::utilities::generateStrides(qDims),
        kDims,
        hipdnn_data_sdk::utilities::generateStrides(kDims),
        vDims,
        hipdnn_data_sdk::utilities::generateStrides(vDims),
        oDims,
        hipdnn_data_sdk::utilities::generateStrides(oDims),
        DataType::BFLOAT16);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaBwdPlanBuilder, IsApplicableRejectsIntegralGqaRatio)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    SKIP_IF_NO_DEVICES();

    if(!bwd_dispatch::isSupportedArch(deviceString()))
    {
        GTEST_SKIP();
    }

    // llama-3-8b backward shape: nhead_q = 32, nhead_k = 8 (ratio 4).  The dqdkdv
    // kernel writes nhead_q head slots into dK/dV, so passing the user's 8-head
    // tensors overruns them by (32 - 8) * 128 * sizeof(bf16) = 6144 bytes and
    // faults with HSA_STATUS_ERROR_MEMORY_FAULT.  Reject until dK/dV head expansion
    // and the ratio-group reduction are implemented.
    const std::vector<int64_t> qDims = {1, 32, 256, 128};
    const std::vector<int64_t> kDims = {1, 8, 256, 128};
    const std::vector<int64_t> vDims = {1, 8, 256, 128};
    const std::vector<int64_t> oDims = {1, 32, 256, 128};

    auto builder = hipdnn_test_sdk::utilities::createValidSdpaBwdGraph(
        qDims,
        hipdnn_data_sdk::utilities::generateStrides(qDims),
        kDims,
        hipdnn_data_sdk::utilities::generateStrides(kDims),
        vDims,
        hipdnn_data_sdk::utilities::generateStrides(vDims),
        oDims,
        hipdnn_data_sdk::utilities::generateStrides(oDims),
        DataType::BFLOAT16);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
}

// =============================================================================
// Tensor layouts the backward kernels can and cannot address
// =============================================================================

// Replaces the strides, and the dims when non-empty, of the tensor called name.
// createValidSdpaBwdGraph names its tensors q, k, v, o, do, stats, dq, dk, dv.
struct LayoutOverride
{
    std::string name;
    std::vector<int64_t> strides;
    std::vector<int64_t> dims;
};

// Copy of a backward graph with the given tensor layouts applied.
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

constexpr std::array<const char*, 8> K_IO_TENSOR_NAMES
    = {"q", "k", "v", "o", "do", "dq", "dk", "dv"};

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

// The same strides on all eight Q/K/V/O/dO/dQ/dK/dV tensors.
std::vector<LayoutOverride> onAllIoTensors(const std::vector<int64_t>& strides)
{
    return onTensors({K_IO_TENSOR_NAMES.begin(), K_IO_TENSOR_NAMES.end()}, strides);
}

// Layouts the kernels address correctly: BSHD, padded row strides, dO strides that
// differ from O, odd strides on every axis whose offset the kernels do not put in a
// 4-byte record index, odd or zero batch and head strides on a length-1 axis (outputs
// included), an odd row stride of at least D with S=1, zero batch or head strides on
// the inputs (broadcasts), and stats whose size-1 axes have any stride (buildPlan
// derives the LSE/D strides from Q's dims). The B=2, H=2 non-packed cases of
// IntegrationGpuSdpaBackward run BSHD Q, K, V and dO with O and dQ rows padded to
// D+1 on the device.
TEST_F(TestSdpaBwdPlanBuilder, IsApplicableAcceptsAddressableLayouts)
{
    SKIP_IF_NO_DEVICES();

    if(!bwd_dispatch::isSupportedArch(deviceString()))
    {
        GTEST_SKIP();
    }

    // B=2 and H=2 so that every batch and head stride is used.
    const std::vector<int64_t> dims = {2, 2, 256, 128};
    // [B, H, S, D] dims stored as [B, S, H, D].
    const std::vector<int64_t> bshd = {int64_t{256} * 2 * 128, 128, int64_t{2} * 128, 1};
    // BHSD with each row padded from D=128 to D+8 or D+2 elements.
    const std::vector<int64_t> paddedRows = {int64_t{2} * 256 * 136, int64_t{256} * 136, 136, 1};
    const std::vector<int64_t> rowsPlus2 = {int64_t{2} * 256 * 130, int64_t{256} * 130, 130, 1};
    // BHSD with an odd batch, head or row stride (every row 2-byte aligned only).
    const std::vector<int64_t> oddBatch = {int64_t{2} * 256 * 128 + 1, int64_t{256} * 128, 128, 1};
    const std::vector<int64_t> oddHead
        = {int64_t{2} * (int64_t{256} * 128 + 1), int64_t{256} * 128 + 1, 128, 1};
    const std::vector<int64_t> oddRows = {int64_t{2} * 256 * 129, int64_t{256} * 129, 129, 1};

    std::vector<std::pair<GraphTest, bool>> applicabilityTests;
    applicabilityTests.emplace_back(GraphTest{createSdpaBwdGraph(dims), "packed BHSD"}, true);
    applicabilityTests.emplace_back(
        GraphTest{withLayouts(createSdpaBwdGraph(dims), onAllIoTensors(bshd)),
                  "BSHD on all eight tensors"},
        true);
    applicabilityTests.emplace_back(
        GraphTest{withLayouts(createSdpaBwdGraph(dims), onAllIoTensors(paddedRows)),
                  "row stride D+8 on all eight tensors"},
        true);
    applicabilityTests.emplace_back(
        GraphTest{withLayouts(createSdpaBwdGraph(dims), onAllIoTensors(rowsPlus2)),
                  "row stride D+2 on all eight tensors"},
        true);
    applicabilityTests.emplace_back(
        GraphTest{withLayouts(createSdpaBwdGraph(dims), {{"do", bshd, {}}}),
                  "dO BSHD while O is BHSD"},
        true);
    // Batch offsets go to the buffer base in every kernel.
    applicabilityTests.emplace_back(
        GraphTest{withLayouts(createSdpaBwdGraph(dims), onAllIoTensors(oddBatch)),
                  "odd batch stride on all eight tensors"},
        true);
    // DQDKDV adds these head offsets to the buffer base.
    applicabilityTests.emplace_back(
        GraphTest{withLayouts(createSdpaBwdGraph(dims),
                              onTensors({"q", "k", "v", "o", "do", "dq"}, oddHead)),
                  "odd head stride on Q, K, V, O, dO and dQ"},
        true);
    // ODO reads O and DQ_CONVERT writes dQ by byte offset.
    applicabilityTests.emplace_back(
        GraphTest{withLayouts(createSdpaBwdGraph(dims), onTensors({"o", "dq"}, oddRows)),
                  "row stride D+1 on O and dQ"},
        true);
    // On a size-1 axis the stride only multiplies index 0, so an odd one is addressed
    // correctly. With S=1 the row stride must still be at least D, because S * row
    // stride bounds each buffer.
    applicabilityTests.emplace_back(GraphTest{withLayouts(createSdpaBwdGraph({2, 2, 1, 128}),
                                                          onAllIoTensors({260, 130, 129, 1})),
                                              "row stride D+1 with S=1 on all eight tensors"},
                                    true);
    applicabilityTests.emplace_back(
        GraphTest{withLayouts(createSdpaBwdGraph({2, 1, 256, 128}),
                              onTensors({"dk", "dv"},
                                        {int64_t{256} * 128, int64_t{256} * 128 + 1, 128, 1})),
                  "odd dK and dV head stride with H=1"},
        true);
    // A zero batch or head stride on an input (a broadcast) only re-reads one
    // slice.
    const std::vector<int64_t> zeroBatch = {0, int64_t{256} * 128, 128, 1};
    const std::vector<int64_t> zeroHead = {int64_t{256} * 128, 0, 128, 1};
    const std::vector<int64_t> zeroBatchAndHead = {0, 0, 128, 1};
    const std::vector<int64_t> threeHeads = {2, 3, 256, 128};
    applicabilityTests.emplace_back(
        GraphTest{
            withLayouts(createSdpaBwdGraph(dims), onTensors({"q", "k", "v", "o", "do"}, zeroBatch)),
            "batch stride 0 on Q, K, V, O and dO"},
        true);
    applicabilityTests.emplace_back(
        GraphTest{withLayouts(createSdpaBwdGraph(threeHeads), onTensors({"k", "v"}, zeroHead)),
                  "head stride 0 on K and V, H=3"},
        true);
    applicabilityTests.emplace_back(GraphTest{withLayouts(createSdpaBwdGraph(threeHeads),
                                                          onTensors({"k", "v"}, zeroBatchAndHead)),
                                              "batch and head stride 0 on K and V, H=3"},
                                    true);
    applicabilityTests.emplace_back(
        GraphTest{withLayouts(createSdpaBwdGraph(threeHeads), {{"do", zeroBatchAndHead, {}}}),
                  "batch and head stride 0 on dO, H=3"},
        true);
    applicabilityTests.emplace_back(
        GraphTest{withLayouts(createSdpaBwdGraph(threeHeads), onTensors({"q", "o"}, zeroHead)),
                  "head stride 0 on Q and O, H=3"},
        true);
    // On a length-1 axis a zero stride does not alias, also on an output.
    applicabilityTests.emplace_back(GraphTest{withLayouts(createSdpaBwdGraph({1, 2, 256, 128}),
                                                          onTensors({"dq", "dk", "dv"}, zeroBatch)),
                                              "batch stride 0 on dQ, dK and dV with B=1"},
                                    true);
    applicabilityTests.emplace_back(GraphTest{withLayouts(createSdpaBwdGraph({2, 1, 256, 128}),
                                                          onTensors({"dq", "dk", "dv"}, zeroHead)),
                                              "head stride 0 on dQ, dK and dV with H=1"},
                                    true);
    applicabilityTests.emplace_back(
        GraphTest{withLayouts(createSdpaBwdGraph(dims), {{"stats", {512, 256, 1}, {2, 2, 256}}}),
                  "rank-3 packed stats"},
        true);
    // No kernel reads the stats strides, so the stride of a size-1 stats axis
    // must not decline, even when it does not fit a 32-bit byte stride.
    applicabilityTests.emplace_back(GraphTest{withLayouts(createSdpaBwdGraph({1, 1, 256, 128}),
                                                          {{"stats", {1024, 640, 1, 1}, {}}}),
                                              "unused stats strides of size-1 axes"},
                                    true);
    applicabilityTests.emplace_back(
        GraphTest{withLayouts(createSdpaBwdGraph({1, 1, 256, 128}),
                              {{"stats", {int64_t{1} << 40, -7, 1, 1}, {}}}),
                  "huge and negative stats strides of size-1 axes"},
        true);
    applicabilityTests.emplace_back(GraphTest{withLayouts(createSdpaBwdGraph({2, 1, 256, 128}),
                                                          {{"stats", {256, 999, 1, 1}, {}}}),
                                              "unused stats head stride, H=1"},
                                    true);
    applicabilityTests.emplace_back(
        GraphTest{withLayouts(createSdpaBwdGraph(dims), {{"stats", {512, 256, 1, 7}, {}}}),
                  "arbitrary stride on the size-1 last stats dim"},
        true);
    applicabilityTests.emplace_back(
        GraphTest{withLayouts(createSdpaBwdGraph({2, 2, 1, 128}), {{"stats", {2, 1, 5, 1}, {}}}),
                  "unused stats seq stride, S_q=1"},
        true);

    for(const auto& [test, applicability] : applicabilityTests)
    {
        EXPECT_EQ(_planBuilder.isApplicable(_handle, test.graphWrapper()), applicability)
            << test.message;
    }
}

// Row strides the kernels cannot address. On gfx942 each of these gave wrong
// gradients:
// - an odd row stride on a tensor DQDKDV indexes in 4-byte records (Q, K, V, dO,
//   dK, dV), and an odd dK/dV head stride: rows start 2 bytes into a dword;
// - a row stride below D, also on a length-1 sequence axis: each buffer ends
//   S * row stride elements into its (batch, head) slice, cutting off the last row.
TEST_F(TestSdpaBwdPlanBuilder, IsApplicableRejectsMisalignedOrShortRows)
{
    SKIP_IF_NO_DEVICES();

    if(!bwd_dispatch::isSupportedArch(deviceString()))
    {
        GTEST_SKIP();
    }

    const std::vector<int64_t> dims = {2, 2, 256, 128};
    const std::vector<int64_t> oddRows = {int64_t{2} * 256 * 129, int64_t{256} * 129, 129, 1};
    const std::vector<int64_t> oddHead
        = {int64_t{2} * (int64_t{256} * 128 + 1), int64_t{256} * 128 + 1, 128, 1};
    // Rows overlap: 64-element row stride with D=128, all strides even.
    const std::vector<int64_t> overlappingRows
        = {int64_t{2} * (int64_t{255} * 64 + 128), int64_t{255} * 64 + 128, 64, 1};
    // S=1 with a row stride of 1.
    const std::vector<int64_t> unitRowStride = {256, 128, 1, 1};
    // Every row at the same address (a sequence broadcast) with S=256.
    const std::vector<int64_t> zeroRowStride = {256, 128, 0, 1};
    // D=192 with a 128-element row stride, all strides even.
    const std::vector<int64_t> hd192Dims = {2, 2, 256, 192};
    const std::vector<int64_t> hd192ShortRows
        = {int64_t{2} * (int64_t{255} * 128 + 192), int64_t{255} * 128 + 192, 128, 1};

    std::vector<GraphTest> declineTests;
    for(const char* name : {"q", "k", "v", "do", "dk", "dv"})
    {
        declineTests.emplace_back(withLayouts(createSdpaBwdGraph(dims), {{name, oddRows, {}}}),
                                  std::string(name) + " row stride D+1");
    }
    // dO row pitch D+1, fp16.
    declineTests.emplace_back(
        withLayouts(createSdpaBwdGraph({2, 2, 64, 128},
                                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF),
                    {{"do", {16512, 8256, 129, 1}, {}}}),
        "dO strides [16512, 8256, 129, 1]");
    for(const char* name : {"dk", "dv"})
    {
        declineTests.emplace_back(withLayouts(createSdpaBwdGraph(dims), {{name, oddHead, {}}}),
                                  std::string(name) + " odd head stride");
    }
    for(const char* name : K_IO_TENSOR_NAMES)
    {
        declineTests.emplace_back(
            withLayouts(createSdpaBwdGraph(dims), {{name, overlappingRows, {}}}),
            std::string(name) + " row stride D/2");
        declineTests.emplace_back(
            withLayouts(createSdpaBwdGraph({2, 2, 1, 128}), {{name, unitRowStride, {}}}),
            std::string(name) + " row stride 1 with S=1");
        declineTests.emplace_back(
            withLayouts(createSdpaBwdGraph(dims), {{name, zeroRowStride, {}}}),
            std::string(name) + " row stride 0 with S=256");
    }
    // At D=192 a 128-element row stride is short for O, dO, dQ, dK and dV. (For Q, K
    // and V the hd192 kernels read past the record bound through the instruction
    // offset, so the uniform rule is conservative there; see isApplicable.)
    for(const char* name : {"o", "do", "dq", "dk", "dv"})
    {
        declineTests.emplace_back(
            withLayouts(createSdpaBwdGraph(hd192Dims), {{name, hd192ShortRows, {}}}),
            std::string(name) + " row stride 128 with D=192");
    }

    for(const auto& test : declineTests)
    {
        EXPECT_FALSE(_planBuilder.isApplicable(_handle, test.graphWrapper())) << test.message;
    }
}

// No kernel takes a head-dim stride, so a strided head dim on any of the eight
// tensors must decline. On gfx942 a stride-2 Q gave Inf dK/dV and a stride-2 dQ
// left dQ elements unwritten.
TEST_F(TestSdpaBwdPlanBuilder, IsApplicableRejectsNonUnitHeadDimStride)
{
    SKIP_IF_NO_DEVICES();

    if(!bwd_dispatch::isSupportedArch(deviceString()))
    {
        GTEST_SKIP();
    }

    const std::vector<int64_t> dims = {2, 2, 256, 128};
    // BHSD with head-dim stride 2 and row stride 2 * D.
    const std::vector<int64_t> headDimStride2
        = {int64_t{2} * 256 * 256, int64_t{256} * 256, 256, 2};

    // Packed BHSD with head-dim stride 0 (every element of a row at one address).
    const std::vector<int64_t> headDimStride0
        = {int64_t{2} * 256 * 128, int64_t{256} * 128, 128, 0};

    for(const char* name : K_IO_TENSOR_NAMES)
    {
        for(const auto& [strides, label] : {std::pair{headDimStride2, "head-dim stride 2"},
                                            std::pair{headDimStride0, "head-dim stride 0"}})
        {
            const GraphTest test{withLayouts(createSdpaBwdGraph(dims), {{name, strides, {}}}),
                                 std::string(name) + " " + label};
            EXPECT_FALSE(_planBuilder.isApplicable(_handle, test.graphWrapper())) << test.message;
        }
    }
}

// The kernels read LSE and D only as packed [B, H_q, S_q]. On gfx942 a stats head
// stride padded to S_q + 64 gave wrong, run-to-run varying gradients for both
// B=2 H=2 and B=1 H=2.
TEST_F(TestSdpaBwdPlanBuilder, IsApplicableRejectsNonPackedStats)
{
    SKIP_IF_NO_DEVICES();

    if(!bwd_dispatch::isSupportedArch(deviceString()))
    {
        GTEST_SKIP();
    }

    // Packed stats for these dims are [2, 2, 256, 1] with strides [512, 256, 1, 1].
    const std::vector<int64_t> dims = {2, 2, 256, 128};
    std::vector<GraphTest> declineTests;
    declineTests.emplace_back(
        withLayouts(createSdpaBwdGraph(dims), {{"stats", {640, 320, 1, 1}, {}}}),
        "stats head stride S_q + 64, B=2 H=2");
    declineTests.emplace_back(
        withLayouts(createSdpaBwdGraph({1, 2, 256, 128}), {{"stats", {640, 320, 1, 1}, {}}}),
        "stats head stride S_q + 64, B=1 H=2");
    declineTests.emplace_back(
        withLayouts(createSdpaBwdGraph(dims), {{"stats", {576, 256, 1, 1}, {}}}),
        "stats batch stride H * S_q + 64");
    // Only the seq stride is wrong in the next three, so only the seq stride
    // check can decline them.
    declineTests.emplace_back(
        withLayouts(createSdpaBwdGraph(dims), {{"stats", {512, 256, 2, 1}, {}}}),
        "stats seq stride 2");
    declineTests.emplace_back(
        withLayouts(createSdpaBwdGraph(dims), {{"stats", {512, 256, 0, 1}, {}}}),
        "stats broadcast along S_q");
    declineTests.emplace_back(
        withLayouts(createSdpaBwdGraph({1, 1, 256, 128}), {{"stats", {512, 512, 2, 1}, {}}}),
        "stats seq stride 2, B=1 H=1");
    declineTests.emplace_back(
        withLayouts(createSdpaBwdGraph(dims), {{"stats", {512, 1, 2, 1}, {}}}),
        "stats stored as [B, S_q, H]");
    // These strides are packed for Q's [B, H_q, S_q], so only the dims check can
    // decline the rest.
    declineTests.emplace_back(
        withLayouts(createSdpaBwdGraph(dims), {{"stats", {512, 256, 1, 1}, {2, 2, 128, 1}}}),
        "stats S_q differs from Q");
    declineTests.emplace_back(
        withLayouts(createSdpaBwdGraph(dims), {{"stats", {512, 256, 1, 1}, {2, 1, 256, 1}}}),
        "stats H differs from Q");
    declineTests.emplace_back(
        withLayouts(createSdpaBwdGraph(dims), {{"stats", {512, 256, 1, 1}, {1, 2, 256, 1}}}),
        "stats B differs from Q");
    declineTests.emplace_back(
        withLayouts(createSdpaBwdGraph(dims), {{"stats", {512, 256, 1, 1}, {2, 2, 256, 2}}}),
        "stats trailing dim 2");
    declineTests.emplace_back(
        withLayouts(createSdpaBwdGraph(dims), {{"stats", {512, 256, 1, 1, 1}, {2, 2, 256, 1, 1}}}),
        "rank-5 stats");
    declineTests.emplace_back(withLayouts(createSdpaBwdGraph(dims), {{"stats", {512, 256, 1}, {}}}),
                              "3 stats strides for rank-4 stats dims");
    declineTests.emplace_back(
        withLayouts(createSdpaBwdGraph(dims), {{"stats", {512, 1}, {2, 512}}}), "rank-2 stats");

    for(const auto& test : declineTests)
    {
        EXPECT_FALSE(_planBuilder.isApplicable(_handle, test.graphWrapper())) << test.message;
    }
}

// Every kernel takes B, H, S and D from Q, K and V, so a tensor whose dims differ
// is addressed with the wrong shape. On gfx942, Q and K with 4 heads and V with
// 2 or 1 gave wrong dQ, dK and dV, and DQDKDV wrote dV past its last head.
TEST_F(TestSdpaBwdPlanBuilder, IsApplicableRejectsMismatchedTensorDims)
{
    using hipdnn_data_sdk::utilities::generateStrides;

    SKIP_IF_NO_DEVICES();

    if(!bwd_dispatch::isSupportedArch(deviceString()))
    {
        GTEST_SKIP();
    }

    std::vector<GraphTest> declineTests;
    const std::vector<int64_t> qkDims = {1, 4, 128, 128};
    for(const int64_t headsV : {2, 1})
    {
        const std::vector<int64_t> vDims = {1, headsV, 128, 128};
        declineTests.emplace_back(hipdnn_test_sdk::utilities::createValidSdpaBwdGraph(
                                      qkDims,
                                      generateStrides(qkDims),
                                      qkDims,
                                      generateStrides(qkDims),
                                      vDims,
                                      generateStrides(vDims),
                                      qkDims,
                                      generateStrides(qkDims),
                                      hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16),
                                  "V and dV with " + std::to_string(headsV)
                                      + " heads, Q and K with 4");
    }

    // Only one tensor differs in each of the rest, so only its check can decline.
    const std::vector<int64_t> dims = {2, 2, 256, 128};
    const std::vector<int64_t> oneHead = {2, 1, 256, 128};
    for(const char* name : {"o", "do", "dq", "v", "dk", "dv"})
    {
        declineTests.emplace_back(
            withLayouts(createSdpaBwdGraph(dims), {{name, generateStrides(oneHead), oneHead}}),
            std::string(name) + " with 1 head, Q and K with 2");
    }
    // A sequence length or head dim that differs, with the head count matching.
    const std::vector<int64_t> headDim64 = {2, 2, 256, 64};
    for(const char* name : {"o", "do", "dq", "dk", "dv"})
    {
        declineTests.emplace_back(
            withLayouts(createSdpaBwdGraph(dims), {{name, generateStrides(headDim64), headDim64}}),
            std::string(name) + " with head dim 64, Q and K with 128");
    }
    const std::vector<int64_t> seq128 = {2, 2, 128, 128};
    for(const char* name : {"o", "do", "dq", "dk", "dv"})
    {
        declineTests.emplace_back(
            withLayouts(createSdpaBwdGraph(dims), {{name, generateStrides(seq128), seq128}}),
            std::string(name) + " with sequence length 128, Q and K with 256");
    }
    // Rank-4 dims and strides are required before any of them is read.
    declineTests.emplace_back(
        withLayouts(createSdpaBwdGraph(dims),
                    {{"dk", {int64_t{2} * 256 * 128, int64_t{256} * 128, 128, 1, 1}, {}}}),
        "dk with 5 strides for rank-4 dims");
    declineTests.emplace_back(withLayouts(createSdpaBwdGraph(dims),
                                          {{"dk", {int64_t{256} * 128, 128, 1}, {4, 256, 128}}}),
                              "rank-3 dk");

    const std::vector<int64_t> oneBatch = {1, 2, 256, 128};
    const auto oneBatchStrides = generateStrides(oneBatch);
    declineTests.emplace_back(withLayouts(createSdpaBwdGraph(dims),
                                          {{"k", oneBatchStrides, oneBatch},
                                           {"v", oneBatchStrides, oneBatch},
                                           {"dk", oneBatchStrides, oneBatch},
                                           {"dv", oneBatchStrides, oneBatch}}),
                              "K, V, dK and dV with batch 1, Q with 2");

    for(const auto& test : declineTests)
    {
        EXPECT_FALSE(_planBuilder.isApplicable(_handle, test.graphWrapper())) << test.message;
    }
}

// A zero batch or head stride on a dQ, dK or dV axis longer than 1 makes (batch,
// head) slices share memory, and the workgroups that write them overwrite each
// other. On gfx942 such dQ, dK and dV were wrong and changed between runs with
// identical inputs. Nonzero strides that make an output overlap itself are not
// checked.
TEST_F(TestSdpaBwdPlanBuilder, IsApplicableRejectsZeroStrideOutputs)
{
    SKIP_IF_NO_DEVICES();

    if(!bwd_dispatch::isSupportedArch(deviceString()))
    {
        GTEST_SKIP();
    }

    const std::vector<int64_t> dims = {2, 2, 256, 128};
    const std::vector<int64_t> zeroBatch = {0, int64_t{256} * 128, 128, 1};
    const std::vector<int64_t> zeroHead = {int64_t{256} * 128, 0, 128, 1};

    std::vector<GraphTest> declineTests;
    // Only one tensor aliases along one axis in each of these.
    for(const char* name : {"dq", "dk", "dv"})
    {
        declineTests.emplace_back(withLayouts(createSdpaBwdGraph(dims), {{name, zeroBatch, {}}}),
                                  std::string(name) + " batch stride 0");
        declineTests.emplace_back(withLayouts(createSdpaBwdGraph(dims), {{name, zeroHead, {}}}),
                                  std::string(name) + " head stride 0");
    }
    // The layouts that failed on gfx942 were dQ batch stride 0 (above) and these.
    const std::vector<int64_t> threeHeads = {2, 3, 256, 128};
    declineTests.emplace_back(withLayouts(createSdpaBwdGraph(threeHeads), {{"dq", zeroHead, {}}}),
                              "dQ head stride 0, H=3");
    declineTests.emplace_back(
        withLayouts(createSdpaBwdGraph(dims), onTensors({"dk", "dv"}, zeroBatch)),
        "dK and dV batch stride 0");
    declineTests.emplace_back(
        withLayouts(createSdpaBwdGraph(threeHeads), onTensors({"dk", "dv"}, zeroHead)),
        "dK and dV head stride 0, H=3");

    for(const auto& test : declineTests)
    {
        EXPECT_FALSE(_planBuilder.isApplicable(_handle, test.graphWrapper())) << test.message;
    }
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
// kernel registry), so the assertions are meaningful on any device. The backward
// isApplicable cannot be used here: it declines devices other than gfx942 and
// gfx950 at its first gate, so on those it would return false before ever
// reaching getMaskType, making an isApplicable-based assertion pass for the
// wrong reason.

// Build a backward SDPA graph that sets the deprecated causal booleans and the
// modern bounds trio explicitly, so contradictory combinations can be
// constructed. Every tensor but the stats takes `dims`, packed. Returns the
// FlatBufferBuilder owning the graph buffer.
flatbuffers::FlatBufferBuilder createSdpaBwdGraphWithMask(
    bool causalMask,
    bool causalMaskBottomRight,
    flatbuffers::Optional<int64_t> leftBound,
    flatbuffers::Optional<int64_t> rightBound,
    hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment diagAlignment,
    const std::vector<int64_t>& dims = {4, 8, 256, 128})
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<TensorAttributes>> tensorAttributes;

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
    const auto doUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, doUid, "do", DataType::BFLOAT16, &strides, &dims));

    const std::vector<int64_t> statsDims = {dims[0], dims[1], dims[2], 1};
    const std::vector<int64_t> statsStrides = {dims[1] * dims[2], dims[2], 1, 1};
    const auto statsUid = uid++;
    tensorAttributes.push_back(CreateTensorAttributesDirect(
        builder, statsUid, "stats", DataType::FLOAT, &statsStrides, &statsDims));

    const auto dqUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, dqUid, "dq", DataType::BFLOAT16, &strides, &dims));
    const auto dkUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, dkUid, "dk", DataType::BFLOAT16, &strides, &dims));
    const auto dvUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, dvUid, "dv", DataType::BFLOAT16, &strides, &dims));

    const auto sdpaAttributes
        = CreateSdpaBackwardAttributes(builder,
                                       qUid,
                                       kUid,
                                       vUid,
                                       oUid,
                                       doUid,
                                       statsUid,
                                       dqUid,
                                       dkUid,
                                       dvUid,
                                       flatbuffers::nullopt, // scale_tensor_uid
                                       flatbuffers::nullopt, // attn_mask_tensor_uid
                                       flatbuffers::nullopt, // seq_len_q_tensor_uid
                                       flatbuffers::nullopt, // seq_len_kv_tensor_uid
                                       flatbuffers::nullopt, // seed_tensor_uid
                                       flatbuffers::nullopt, // offset_tensor_uid
                                       flatbuffers::nullopt, // dropout_mask_tensor_uid
                                       flatbuffers::nullopt, // dropout_scale_tensor_uid
                                       flatbuffers::nullopt, // dropout_scale_inv_tensor_uid
                                       flatbuffers::nullopt, // dbias_tensor_uid
                                       false, // alibi_mask
                                       false, // padding_mask
                                       causalMask,
                                       causalMaskBottomRight,
                                       flatbuffers::nullopt, // dropout_probability
                                       flatbuffers::nullopt, // attn_scale_value
                                       leftBound,
                                       rightBound,
                                       diagAlignment);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "sdpa_bwd",
                                     DataType::BFLOAT16,
                                     NodeAttributes::SdpaBackwardAttributes,
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

// Extract the diagonal band from a backward graph buffer and classify the mask.
plan_utils::MaskType classifyMask(const flatbuffers::FlatBufferBuilder& builder)
{
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());
    const auto& attrs
        = graphWrapper.nodeWrappers()
              .front()
              ->attributesAs<hipdnn_flatbuffers_sdk::data_objects::SdpaBackwardAttributes>();
    const auto band = plan_utils::resolveDiagonalBand(attrs);
    return plan_utils::getMaskType(band.leftBound, band.rightBound, band.alignment);
}

// cuDNN's default: no attn_scale_value and no scale tensor means no scaling.
TEST_F(TestSdpaBwdPlanBuilder, AbsentAttnScaleValueIsOne)
{
    auto builder = createSdpaBwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/false,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment::TOP_LEFT);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());
    const auto& attrs
        = graphWrapper.nodeWrappers()
              .front()
              ->attributesAs<hipdnn_flatbuffers_sdk::data_objects::SdpaBackwardAttributes>();
    ASSERT_FALSE(attrs.attn_scale_value().has_value());
    EXPECT_EQ(plan_utils::attnScaleOrDefault(attrs), 1.0f);
}

class TestSdpaBwdDeprecatedCausalMask : public ::testing::TestWithParam<DeprecatedCausalMaskCase>
{
};

TEST_P(TestSdpaBwdDeprecatedCausalMask, Classifies)
{
    const auto& param = GetParam();
    auto builder = createSdpaBwdGraphWithMask(param.causalMask,
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
                         TestSdpaBwdDeprecatedCausalMask,
                         ::testing::ValuesIn(deprecatedCausalMaskCases()),
                         deprecatedCausalMaskCaseName);

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

TEST(TestSdpaBwdMakeCoveringBoundsUnbound, TopLeftCoveringLeftBoundIsCausal)
{
    EXPECT_EQ(classifyAfterUnbounding(127, 0, DiagonalAlignment::TOP_LEFT),
              plan_utils::MaskType::TOP_LEFT_CAUSAL);
    EXPECT_EQ(classifyAfterUnbounding(126, 0, DiagonalAlignment::TOP_LEFT),
              plan_utils::MaskType::SLIDING_WINDOW);
}

TEST(TestSdpaBwdMakeCoveringBoundsUnbound, TopLeftCoveringRightBoundIsNoMask)
{
    EXPECT_EQ(classifyAfterUnbounding(-1, 511, DiagonalAlignment::TOP_LEFT),
              plan_utils::MaskType::NO_MASK);
    EXPECT_EQ(classifyAfterUnbounding(-1, 510, DiagonalAlignment::TOP_LEFT),
              plan_utils::MaskType::SLIDING_WINDOW);
}

TEST(TestSdpaBwdMakeCoveringBoundsUnbound, BottomRightCoveringLeftBoundIsCausal)
{
    EXPECT_EQ(classifyAfterUnbounding(511, 0, DiagonalAlignment::BOTTOM_RIGHT),
              plan_utils::MaskType::BOTTOM_RIGHT_CAUSAL);
    EXPECT_EQ(classifyAfterUnbounding(510, 0, DiagonalAlignment::BOTTOM_RIGHT),
              plan_utils::MaskType::SLIDING_WINDOW);
}

TEST(TestSdpaBwdMakeCoveringBoundsUnbound, BottomRightCoveringRightBoundIsNoMask)
{
    EXPECT_EQ(classifyAfterUnbounding(-1, 127, DiagonalAlignment::BOTTOM_RIGHT),
              plan_utils::MaskType::NO_MASK);
    EXPECT_EQ(classifyAfterUnbounding(-1, 126, DiagonalAlignment::BOTTOM_RIGHT),
              plan_utils::MaskType::SLIDING_WINDOW);
}

TEST(TestSdpaBwdMakeCoveringBoundsUnbound, BothBoundsCoveringIsNoMask)
{
    EXPECT_EQ(classifyAfterUnbounding(127, 511, DiagonalAlignment::TOP_LEFT),
              plan_utils::MaskType::NO_MASK);
    EXPECT_EQ(classifyAfterUnbounding(-1, -1, DiagonalAlignment::TOP_LEFT),
              plan_utils::MaskType::NO_MASK);
}

TEST(TestSdpaBwdMakeCoveringBoundsUnbound, BoundsInsideSequencePassThroughUnchanged)
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
TEST(TestSdpaBwdMakeCoveringBoundsUnbound, HugeCoveringBoundBecomesUnboundedRequest)
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

TEST(TestSdpaBwdMakeCoveringBoundsUnbound, HugeBoundBelowHugeSequenceStaysSlidingWindow)
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
TEST(TestSdpaBwdMakeCoveringBoundsUnbound, SingleTokenSequenceDropsCausalMaskOnItsAxis)
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

// The backward sliding-window path hands the extracted band to the kernel, so a
// deprecated boolean must contribute its causal right bound to the window.
TEST(TestSdpaBwdDeprecatedCausalMaskBand, CausalWithLeftBoundResolvesToCausalWindow)
{
    auto builder = createSdpaBwdGraphWithMask(true, false, 64, NO_BOUND, ALIGN_BOTTOM_RIGHT);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());
    const auto& attrs
        = graphWrapper.nodeWrappers()
              .front()
              ->attributesAs<hipdnn_flatbuffers_sdk::data_objects::SdpaBackwardAttributes>();

    // left_bound = 64 (cuDNN: 64 keys including the diagonal) is 63 keys left of it for the kernel.
    const auto band = plan_utils::resolveDiagonalBand(attrs);
    EXPECT_EQ(band.leftBound, 63);
    EXPECT_EQ(band.rightBound, 0);
    EXPECT_EQ(band.alignment, ALIGN_BOTTOM_RIGHT);

    constexpr int64_t SEQ_LEN_ABOVE_BOUND = 256;
    const auto windowed
        = plan_utils::makeCoveringBoundsUnbound(band, SEQ_LEN_ABOVE_BOUND, SEQ_LEN_ABOVE_BOUND);
    EXPECT_EQ(windowed.leftBound, 63);
    EXPECT_EQ(windowed.rightBound, 0);

    // 64 keys cover every row of a 64-long sequence; at 65 the last row would drop key 0.
    constexpr int64_t SEQ_LEN_COVERED_BY_BOUND = 64;
    const auto covered = plan_utils::makeCoveringBoundsUnbound(
        band, SEQ_LEN_COVERED_BY_BOUND, SEQ_LEN_COVERED_BY_BOUND);
    EXPECT_EQ(covered.leftBound, -1);
    EXPECT_EQ(covered.rightBound, 0);
}

// Modern bounds-trio path (no deprecated boolean set). An unset bound is treated
// as unbounded (-1), so a partially specified trio still derives the mask it
// describes.

TEST_F(TestSdpaBwdPlanBuilder, MaskBoundsTrioRightZeroLeftUnsetDerivesTopLeftCausal)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    // The canonical causal request: left unbounded (unset) with right_bound=0.
    // An unset left bound must be read as unbounded, not as "no mask".
    auto builder = createSdpaBwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/false,
        flatbuffers::nullopt,
        flatbuffers::Optional<int64_t>(0),
        DiagonalAlignment::TOP_LEFT);

    plan_utils::MaskType maskType = plan_utils::MaskType::NO_MASK;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::TOP_LEFT_CAUSAL);
}

TEST_F(TestSdpaBwdPlanBuilder, MaskBoundsTrioRightZeroBottomRightDerivesBottomRightCausal)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    // Same causal request with bottom-right diagonal alignment.
    auto builder = createSdpaBwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/false,
        flatbuffers::nullopt,
        flatbuffers::Optional<int64_t>(0),
        DiagonalAlignment::BOTTOM_RIGHT);

    plan_utils::MaskType maskType = plan_utils::MaskType::NO_MASK;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::BOTTOM_RIGHT_CAUSAL);
}

TEST_F(TestSdpaBwdPlanBuilder, MaskBoundsTrioExplicitCausalBoundsDeriveTopLeftCausal)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    // Explicit unbounded-left (left=-1) with right_bound=0 matches the unset case.
    auto builder = createSdpaBwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/false,
        flatbuffers::Optional<int64_t>(-1),
        flatbuffers::Optional<int64_t>(0),
        DiagonalAlignment::TOP_LEFT);

    plan_utils::MaskType maskType = plan_utils::MaskType::NO_MASK;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::TOP_LEFT_CAUSAL);
}

TEST_F(TestSdpaBwdPlanBuilder, MaskBoundsTrioBothUnsetDerivesNoMask)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    auto builder = createSdpaBwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/false,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        DiagonalAlignment::TOP_LEFT);

    plan_utils::MaskType maskType = plan_utils::MaskType::SLIDING_WINDOW;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::NO_MASK);
}

TEST_F(TestSdpaBwdPlanBuilder, MaskBoundsTrioBothUnboundedDerivesNoMask)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    auto builder = createSdpaBwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/false,
        flatbuffers::Optional<int64_t>(-1),
        flatbuffers::Optional<int64_t>(-1),
        DiagonalAlignment::TOP_LEFT);

    plan_utils::MaskType maskType = plan_utils::MaskType::SLIDING_WINDOW;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::NO_MASK);
}

TEST_F(TestSdpaBwdPlanBuilder, MaskBoundsTrioSymmetricWindowDerivesSlidingWindow)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    auto builder = createSdpaBwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/false,
        flatbuffers::Optional<int64_t>(64),
        flatbuffers::Optional<int64_t>(64),
        DiagonalAlignment::TOP_LEFT);

    plan_utils::MaskType maskType = plan_utils::MaskType::NO_MASK;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::SLIDING_WINDOW);
}

TEST_F(TestSdpaBwdPlanBuilder, MaskBoundsTrioLeftOnlyDerivesSlidingWindow)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    // A bounded left with an unset (unbounded) right is a one-sided window.
    auto builder = createSdpaBwdGraphWithMask(
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
// Masked attention dispatch (causal, causal-BR, SWA) — isApplicable tests
// =============================================================================
// These tests use createSdpaBwdGraphWithMask (defined above) and require a
// gfx942 or gfx950 device to exercise the full isApplicable() path.

TEST_F(TestSdpaBwdPlanBuilder, IsApplicableAcceptsCausalMaskBottomRight)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    SKIP_IF_NO_DEVICES();

    if(!bwd_dispatch::isSupportedArch(deviceString()))
    {
        GTEST_SKIP();
    }

    auto builder = createSdpaBwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/true,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        DiagonalAlignment::BOTTOM_RIGHT);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaBwdPlanBuilder, IsApplicableAcceptsWindowMask)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    SKIP_IF_NO_DEVICES();

    if(!bwd_dispatch::isSupportedArch(deviceString()))
    {
        GTEST_SKIP();
    }

    // Symmetric sliding window with left=64, right=64, in both alignments
    // (on gfx950 the bottom-right one dispatches the mask-4 kernel).
    for(const auto alignment : {DiagonalAlignment::TOP_LEFT, DiagonalAlignment::BOTTOM_RIGHT})
    {
        SCOPED_TRACE(EnumNameDiagonalAlignment(alignment));
        auto builder = createSdpaBwdGraphWithMask(
            /*causalMask=*/false,
            /*causalMaskBottomRight=*/false,
            flatbuffers::Optional<int64_t>(64),
            flatbuffers::Optional<int64_t>(64),
            alignment);

        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
            builder.GetBufferPointer(), builder.GetSize());

        EXPECT_TRUE(_planBuilder.isApplicable(_handle, graphWrapper));
    }
}

// =============================================================================
// Sliding windows AITER v0.1.29 sends to CK
// =============================================================================

TEST(TestSdpaBwdBatchWindow, WindowRulesPerArch)
{
    using bwd_dispatch::isBatchWindowSupported;

    // gfx950: windows need D_qk <= 128.
    EXPECT_TRUE(isBatchWindowSupported("gfx950", 72));
    EXPECT_TRUE(isBatchWindowSupported("gfx950", 128));
    EXPECT_FALSE(isBatchWindowSupported("gfx950", 192));

    // gfx942: windows need 64 < D_qk <= 128.
    EXPECT_FALSE(isBatchWindowSupported("gfx942", 64));
    EXPECT_TRUE(isBatchWindowSupported("gfx942", 72));
    EXPECT_TRUE(isBatchWindowSupported("gfx942", 128));
    EXPECT_FALSE(isBatchWindowSupported("gfx942", 136));
}

namespace
{
struct WindowCase
{
    const char* label;
    std::vector<int64_t> dims; // Q and K dims; V, O, dO and dV take headDimV.
    int64_t headDimV;
    // Graph bounds (cuDNN convention: the left bound counts the diagonal key).
    flatbuffers::Optional<int64_t> leftBound;
    flatbuffers::Optional<int64_t> rightBound;
    bool gfx942;
    bool gfx950;
};

flatbuffers::FlatBufferBuilder createWindowGraph(const WindowCase& c)
{
    using hipdnn_data_sdk::utilities::generateStrides;

    auto builder = createSdpaBwdGraphWithMask(
        false, false, c.leftBound, c.rightBound, ALIGN_TOP_LEFT, c.dims);
    std::vector<LayoutOverride> layouts;
    if(c.headDimV != c.dims[3])
    {
        const std::vector<int64_t> voDims = {c.dims[0], c.dims[1], c.dims[2], c.headDimV};
        for(const char* name : {"v", "o", "do", "dv"})
        {
            layouts.push_back({name, generateStrides(voDims), voDims});
        }
    }
    return layouts.empty() ? std::move(builder) : withLayouts(builder, layouts);
}
} // namespace

// End to end through isApplicable. A window that keeps only the diagonal
// (graph left 1, right 0 or unset) is a window like any other on both archs:
// AITER's gfx942 Python gate does not classify it as one (left 0 is not > 0),
// but that is not a kernel limit, so the engine serves it.
TEST_F(TestSdpaBwdPlanBuilder, IsApplicableAppliesTheAiterWindowRules)
{
    SKIP_IF_NO_DEVICES();

    const auto arch = deviceString();
    if(!bwd_dispatch::isSupportedArch(arch))
    {
        GTEST_SKIP();
    }

    const flatbuffers::Optional<int64_t> left64(64);
    const flatbuffers::Optional<int64_t> right64(64);
    const flatbuffers::Optional<int64_t> diagonalLeft(1);
    const flatbuffers::Optional<int64_t> diagonalRight(0);
    const std::vector<WindowCase> cases = {
        {"window D=96", {2, 4, 256, 96}, 96, left64, right64, true, true},
        {"window D=64", {2, 4, 256, 64}, 64, left64, right64, false, false},
        {"window D=192x128", {2, 4, 256, 192}, 128, left64, right64, false, false},
        {"causal D=192x128", {2, 4, 256, 192}, 128, NO_BOUND, diagonalRight, false, true},
        {"diagonal D=128", {2, 4, 256, 128}, 128, diagonalLeft, diagonalRight, true, true},
        {"diagonal, right unbounded, D=128",
         {2, 4, 256, 128},
         128,
         diagonalLeft,
         NO_BOUND,
         true,
         true},
        {"diagonal D=128 S=200", {2, 4, 200, 128}, 128, diagonalLeft, diagonalRight, true, true},
        {"diagonal D=96", {2, 4, 256, 96}, 96, diagonalLeft, diagonalRight, true, true},
        {"diagonal D=64", {2, 4, 256, 64}, 64, diagonalLeft, diagonalRight, false, false},
    };

    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.label);
        auto builder = createWindowGraph(c);
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
            builder.GetBufferPointer(), builder.GetSize());
        EXPECT_EQ(_planBuilder.isApplicable(_handle, graphWrapper),
                  arch == "gfx942" ? c.gfx942 : c.gfx950);
    }
}

// =============================================================================
// Kernarg range guards
// =============================================================================
//
// The predicates these guards are built from are covered at their exact
// boundaries by the TestSdpaKernargRanges suite in TestSdpaFwdPlanBuilder.cpp.
// What follows is the backward builder's use of them.

// Build a backward graph with per-tensor dims and one explicit stride set, so a
// dimension can be driven to the limit of its field without the contiguous
// stride for that shape tripping the byte-stride guard first. Only descriptors
// are built, so the dims cost no memory. The LSE/D stride guard is computed from
// Q's dims (H_q * S_q and S_q, as FP32 byte strides), which is why the accepted
// cases below drive the k-side dims: an nhead_q or seqlen_q at its own limit
// makes that guard decline the graph first.
auto createSdpaBwdGraphForDims(const std::vector<int64_t>& qDims,
                               const std::vector<int64_t>& kDims,
                               const std::vector<int64_t>& vDims)
{
    const std::vector<int64_t> oDims = {qDims[0], qDims[1], qDims[2], vDims[3]};
    const std::vector<int64_t> strides = {32768, 32768, 128, 1};
    return hipdnn_test_sdk::utilities::createValidSdpaBwdGraph(
        qDims,
        strides,
        kDims,
        strides,
        vDims,
        strides,
        oDims,
        strides,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16);
}

struct BwdDimRangeCase
{
    const char* label;
    std::vector<int64_t> qDims;
    std::vector<int64_t> kDims;
    std::vector<int64_t> vDims;
};

constexpr int64_t K_U32_DIM_LIMIT = static_cast<int64_t>(UINT32_MAX);
constexpr int64_t K_I32_DIM_LIMIT = static_cast<int64_t>(INT32_MAX);

// Batch reaches the kernels only as gridDimZ, a uint32; seqlen_k takes the
// signed limit because it becomes an int32 mask coordinate on the window path
// and drives a ceil-division evaluated in unsigned int.
TEST_F(TestSdpaBwdPlanBuilder, IsApplicableAcceptsDimensionsAtTheir32BitLimits)
{
    SKIP_IF_NO_DEVICES();

    if(!bwd_dispatch::isSupportedArch(deviceString()))
    {
        GTEST_SKIP();
    }

    const std::vector<BwdDimRangeCase> cases = {
        {"batch at UINT32_MAX",
         {K_U32_DIM_LIMIT, 8, 256, 128},
         {K_U32_DIM_LIMIT, 8, 256, 128},
         {K_U32_DIM_LIMIT, 8, 256, 128}},
        {"seqlen_k at INT32_MAX",
         {4, 8, 256, 128},
         {4, 8, K_I32_DIM_LIMIT, 128},
         {4, 8, K_I32_DIM_LIMIT, 128}},
    };

    for(const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.label);
        auto builder = createSdpaBwdGraphForDims(testCase.qDims, testCase.kDims, testCase.vDims);
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper gw(
            builder.GetBufferPointer(), builder.GetSize());
        EXPECT_TRUE(_planBuilder.isApplicable(_handle, gw));
    }
}

TEST_F(TestSdpaBwdPlanBuilder, IsApplicableDeclinesDimensionsPastTheir32BitLimits)
{
    SKIP_IF_NO_DEVICES();

    if(!bwd_dispatch::isSupportedArch(deviceString()))
    {
        GTEST_SKIP();
    }

    const std::vector<BwdDimRangeCase> cases = {
        {"batch one past UINT32_MAX",
         {K_U32_DIM_LIMIT + 1, 8, 256, 128},
         {K_U32_DIM_LIMIT + 1, 8, 256, 128},
         {K_U32_DIM_LIMIT + 1, 8, 256, 128}},
        {"nhead one past UINT32_MAX",
         {4, K_U32_DIM_LIMIT + 1, 256, 128},
         {4, K_U32_DIM_LIMIT + 1, 256, 128},
         {4, K_U32_DIM_LIMIT + 1, 256, 128}},
        {"seqlen_q one past INT32_MAX",
         {4, 8, K_I32_DIM_LIMIT + 1, 128},
         {4, 8, 256, 128},
         {4, 8, 256, 128}},
        {"seqlen_k one past INT32_MAX",
         {4, 8, 256, 128},
         {4, 8, K_I32_DIM_LIMIT + 1, 128},
         {4, 8, K_I32_DIM_LIMIT + 1, 128}},
    };

    for(const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.label);
        auto builder = createSdpaBwdGraphForDims(testCase.qDims, testCase.kDims, testCase.vDims);
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper gw(
            builder.GetBufferPointer(), builder.GetSize());
        EXPECT_FALSE(_planBuilder.isApplicable(_handle, gw));
    }
}

// The head dims are narrowed to int before the head-dim gate and the registry
// lookup, so a head dim whose low 32 bits are a supported value used to resolve
// a kernel built for that supported value. The head-dim gate is what bounds
// the head dims in practice, which is why there is no case at INT32_MAX here.
TEST_F(TestSdpaBwdPlanBuilder, IsApplicableDeclinesHeadDimsThatTruncateToASupportedValue)
{
    SKIP_IF_NO_DEVICES();

    if(!bwd_dispatch::isSupportedArch(deviceString()))
    {
        GTEST_SKIP();
    }

    constexpr int64_t K_WRAPS_TO_128 = (int64_t{1} << 32) + 128;

    const std::vector<BwdDimRangeCase> cases = {
        {"supported head dims (control)", {4, 8, 256, 128}, {4, 8, 256, 128}, {4, 8, 256, 128}},
        {"head_dim_q wraps to 128",
         {4, 8, 256, K_WRAPS_TO_128},
         {4, 8, 256, K_WRAPS_TO_128},
         {4, 8, 256, 128}},
        {"head_dim_v wraps to 128",
         {4, 8, 256, 128},
         {4, 8, 256, 128},
         {4, 8, 256, K_WRAPS_TO_128}},
    };

    for(const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.label);
        auto builder = createSdpaBwdGraphForDims(testCase.qDims, testCase.kDims, testCase.vDims);
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper gw(
            builder.GetBufferPointer(), builder.GetSize());
        const bool isControl = testCase.label == std::string("supported head dims (control)");
        EXPECT_EQ(_planBuilder.isApplicable(_handle, gw), isControl);
    }
}

// The `ts` of the DQDKDV row isApplicable resolves on arch for a bf16 hd128
// no-mask batch graph with S_q = S_kv = 256 on the default A32 accumulator path
// (RTNE).
std::optional<int> resolveBwdTileSizeKv(const std::string& arch)
{
    const auto row = resolvedDqdkdv(arch, makeRequest("bf16", 128, 128));
    if(!row)
    {
        return std::nullopt;
    }
    return row->ts;
}

// The Ts kernarg is tsKv * stride_k * sizeof(bf16). The per-field stride guard
// only bounds stride_k * 2, so the tile factor is what this guard adds: the
// accepted K sequence stride shrinks by tsKv.
TEST_F(TestSdpaBwdPlanBuilder, IsApplicableBoundsTheTsProductByTheResolvedTileSize)
{
    SKIP_IF_NO_DEVICES();

    if(!bwd_dispatch::isSupportedArch(deviceString()))
    {
        GTEST_SKIP();
    }

    const auto tileSizeKv = resolveBwdTileSizeKv(deviceString());
    ASSERT_TRUE(tileSizeKv.has_value()) << "no bf16 hd128 no-mask a32 dqdkdv row";

    constexpr int64_t K_BF16_BYTES = 2;
    const int64_t limit = static_cast<int64_t>(UINT32_MAX) / (K_BF16_BYTES * *tileSizeKv);
    // DQDKDV indexes K's sequence axis and dK's head axis in 4-byte records, so
    // isApplicable also declines an odd stride there. Both probes are even, so
    // the Ts guard is the only check that separates them.
    const int64_t atLimit = limit - limit % 2;
    const int64_t pastLimit = (limit + 1) + (limit + 1) % 2;
    // The first stride past the Ts limit must still clear the per-field stride
    // guard, or the test would prove nothing about Ts.
    ASSERT_LE(pastLimit * K_BF16_BYTES, static_cast<int64_t>(UINT32_MAX));

    const std::vector<int64_t> dims = {4, 8, 256, 128};
    const std::vector<int64_t> normalStrides = hipdnn_data_sdk::utilities::generateStrides(dims);

    auto buildWithKSeqStride = [&](int64_t kSeqStride) {
        const std::vector<int64_t> kStrides = {kSeqStride, kSeqStride, kSeqStride, 1};
        return hipdnn_test_sdk::utilities::createValidSdpaBwdGraph(
            dims,
            normalStrides,
            dims,
            kStrides,
            dims,
            normalStrides,
            dims,
            normalStrides,
            hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16);
    };

    auto builderAtLimit = buildWithKSeqStride(atLimit);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper gwAtLimit(
        builderAtLimit.GetBufferPointer(), builderAtLimit.GetSize());
    EXPECT_TRUE(_planBuilder.isApplicable(_handle, gwAtLimit));

    auto builderPastLimit = buildWithKSeqStride(pastLimit);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper gwPastLimit(
        builderPastLimit.GetBufferPointer(), builderPastLimit.GetSize());
    EXPECT_FALSE(_planBuilder.isApplicable(_handle, gwPastLimit));
}

// =============================================================================
// Workspace and dq_acc layout (AITER mha_bwd.cu fmha_v3_bwd() :577-596)
// =============================================================================

// A32: FP32 [B, H_q, S_q, D_qk] with the unpadded head dim.
TEST(TestSdpaBwdDqAccLayout, A32UsesFourByteElementsAtTheGraphHeadDim)
{
    const auto layout = sdpaBwdDqAccLayout(2, 8, 200, 96, AccumulatorType::A32);
    EXPECT_EQ(layout.elementBytes, 4u);
    EXPECT_EQ(layout.stride, 96u);
    EXPECT_EQ(layout.nheadStride, 200u * 96);
    EXPECT_EQ(layout.batchStride, 8u * 200 * 96);
    EXPECT_EQ(layout.bytes, 2u * 8 * 200 * 96 * 4);
}

// A16: Q-dtype [B, H_q, ceil16(S_q), 128 or 192] (:579-584).
TEST(TestSdpaBwdDqAccLayout, A16Is16BitWithPaddedRowsAndHeadDim)
{
    const auto layout = sdpaBwdDqAccLayout(2, 8, 200, 96, AccumulatorType::A16);
    EXPECT_EQ(layout.elementBytes, 2u);
    EXPECT_EQ(layout.stride, 128u);
    EXPECT_EQ(layout.nheadStride, 208u * 128);
    EXPECT_EQ(layout.batchStride, 8u * 208 * 128);
    EXPECT_EQ(layout.bytes, 2u * 8 * 208 * 128 * 2);

    const auto hd192 = sdpaBwdDqAccLayout(2, 2, 100, 192, AccumulatorType::A16);
    EXPECT_EQ(hd192.stride, 192u);
    EXPECT_EQ(hd192.nheadStride, 112u * 192);
}

// `batch_stride_dq_acc = (effective_batch == 1) ? 0 : per_batch_elems` (:591).
TEST(TestSdpaBwdDqAccLayout, SingleBatchHasNoBatchStride)
{
    for(const auto accType : {AccumulatorType::A32, AccumulatorType::A16})
    {
        const auto layout = sdpaBwdDqAccLayout(1, 8, 256, 128, accType);
        EXPECT_EQ(layout.batchStride, 0u);
        EXPECT_EQ(layout.bytes, 8u * layout.nheadStride * layout.elementBytes);
    }
}

// B == 1 has no batch stride, but the DQDKDV kernels still form the offset of
// the last head in 32 bits, so the per-batch span must stay below 2^32 bytes.
TEST(TestSdpaBwdDqAccLayout, FitsU32BoundsThePerBatchSpanAtBatchOne)
{
    // A32: 64 * S * 128 * 4 bytes; S = 131071 is 4,294,934,528, S = 131072 is 2^32.
    EXPECT_TRUE(
        sdpaBwdDqAccFitsU32(sdpaBwdDqAccLayout(1, 64, 131071, 128, AccumulatorType::A32), 64));
    EXPECT_FALSE(
        sdpaBwdDqAccFitsU32(sdpaBwdDqAccLayout(1, 64, 131072, 128, AccumulatorType::A32), 64));
    // A16 counts ceil16(S_q) rows of 128: S = 262128 is 4,294,705,152 bytes,
    // S = 262129 pads to 262144 rows, 2^32 bytes.
    EXPECT_TRUE(
        sdpaBwdDqAccFitsU32(sdpaBwdDqAccLayout(1, 64, 262128, 128, AccumulatorType::A16), 64));
    EXPECT_FALSE(
        sdpaBwdDqAccFitsU32(sdpaBwdDqAccLayout(1, 64, 262129, 128, AccumulatorType::A16), 64));
}

TEST(TestSdpaBwdWorkspaceSize, A32IsTheDBufferAndAFourByteDqAcc)
{
    // D buffer 2*8*512*4 = 32768; dq_acc 2*8*512*128*4 = 4194304
    EXPECT_EQ(sdpaBwdWorkspaceSize(2, 8, 512, 128, AccumulatorType::A32, true), 32768u + 4194304u);
}

// gfx950 A16 (dq_shuffle post stage): the D buffer and the 16-bit dq_acc.
TEST(TestSdpaBwdWorkspaceSize, A16WithAPostStageAddsThe16BitDqAcc)
{
    // D buffer 2*8*200*4 = 12800; dq_acc 2*8*208*128*2 = 851968
    EXPECT_EQ(sdpaBwdWorkspaceSize(2, 8, 200, 96, AccumulatorType::A16, true), 12800u + 851968u);
    // D buffer 1*2*100*4 = 800 -> 832 aligned; dq_acc 1*2*112*192*2 = 86016
    EXPECT_EQ(sdpaBwdWorkspaceSize(1, 2, 100, 192, AccumulatorType::A16, true), 832u + 86016u);
}

// gfx942 A16 writes dQ directly: the D buffer only.
TEST(TestSdpaBwdWorkspaceSize, A16WithoutAPostStageIsTheDBufferOnly)
{
    // D buffer 1*3*255*4 = 3060 -> 3072 aligned
    EXPECT_EQ(sdpaBwdWorkspaceSize(1, 3, 255, 128, AccumulatorType::A16, false), 3072u);
}

// Padding S_q to 16 rows and D to 128 makes the A16 dq_acc larger than the
// A32 one for a short sequence with a small head dim.
TEST(TestSdpaBwdWorkspaceSize, A16DqAccCanExceedA32)
{
    // A32: 128 + 17*72*4 = 4896 -> 4928; A16: 128 + 32*128*2 = 8192
    const size_t a32Size = sdpaBwdWorkspaceSize(1, 1, 17, 72, AccumulatorType::A32, true);
    const size_t a16Size = sdpaBwdWorkspaceSize(1, 1, 17, 72, AccumulatorType::A16, true);
    EXPECT_EQ(a32Size, 128u + 4928u);
    EXPECT_EQ(a16Size, 128u + 8192u);
}

// =============================================================================
// Zeroing a strided dQ (planBhsdZero)
// =============================================================================

namespace
{
constexpr size_t K_BF16 = 2;

void expectSingleRegion(const StridedZeroPlan& plan,
                        size_t widthBytes,
                        size_t pitchBytes,
                        size_t height)
{
    EXPECT_EQ(plan.widthBytes, widthBytes);
    EXPECT_EQ(plan.pitchBytes, pitchBytes);
    EXPECT_EQ(plan.height, height);
    for(const auto& loop : plan.loops)
    {
        EXPECT_EQ(loop.extent, 1u);
    }
}
} // namespace

// Packed BHSD and BSHD are one contiguous span.
TEST(TestSdpaBwdStridedZero, DenseLayoutsAreOneSpan)
{
    // [B=2, H=3, S=5, D=8]
    expectSingleRegion(planBhsdZero({2, 3, 5}, {120, 40, 8}, 8, K_BF16), 480, 480, 1);
    expectSingleRegion(planBhsdZero({2, 3, 5}, {120, 8, 24}, 8, K_BF16), 480, 480, 1);
    // Axes of length 1 take any stride.
    expectSingleRegion(planBhsdZero({1, 1, 4}, {0, 0, 8}, 8, K_BF16), 64, 64, 1);
}

// Rows padded past D, and Q's slice of a packed [B, S, 3, H, D] QKV buffer,
// are one 2D region whose gaps are left untouched.
TEST(TestSdpaBwdStridedZero, RowGapsBecomeOnePitchedRegion)
{
    expectSingleRegion(planBhsdZero({2, 3, 5}, {135, 45, 9}, 8, K_BF16), 16, 18, 30);
    expectSingleRegion(planBhsdZero({2, 4, 5}, {480, 8, 96}, 8, K_BF16), 64, 192, 10);
}

// Axes that do not continue the region below them become loops.
TEST(TestSdpaBwdStridedZero, IrregularAxesBecomeLoops)
{
    const auto plan = planBhsdZero({2, 3, 5}, {300, 100, 16}, 8, K_BF16);
    EXPECT_EQ(plan.widthBytes, 16u);
    EXPECT_EQ(plan.pitchBytes, 32u);
    EXPECT_EQ(plan.height, 5u);
    EXPECT_EQ(plan.loops[0].extent, 3u);
    EXPECT_EQ(plan.loops[0].strideBytes, 200u);
    EXPECT_EQ(plan.loops[1].extent, 2u);
    EXPECT_EQ(plan.loops[1].strideBytes, 600u);
    EXPECT_EQ(plan.loops[2].extent, 1u);
}

// =============================================================================
// Engine knob tests
// =============================================================================

TEST_F(TestSdpaBwdPlanBuilder, NonApplicableGraphReturnsNoKnobs)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const auto knobs = _planBuilder.getCustomKnobs(_handle, graphWrapper);
    EXPECT_TRUE(knobs.empty());
}

TEST_F(TestSdpaBwdPlanBuilder, GetCustomKnobsReturnsAccumulatorKnob)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    SKIP_IF_NO_DEVICES();

    if(!bwd_dispatch::isSupportedArch(deviceString()))
    {
        GTEST_SKIP();
    }

    auto graphBuilder = createSdpaBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const auto knobs = _planBuilder.getCustomKnobs(_handle, graphWrapper);
    ASSERT_EQ(knobs.size(), 1u);
    EXPECT_EQ(knobs[0].knob_id, "hipkernel.sdpa.bwd.accumulator_type");

    // Default value is "a32"
    const auto* defaultVal = knobs[0].default_value.AsStringValue();
    ASSERT_NE(defaultVal, nullptr);
    EXPECT_EQ(defaultVal->value, "a32");

    // Valid values are ["a32", "a16"]
    const auto* constraint = knobs[0].constraint.AsStringConstraint();
    ASSERT_NE(constraint, nullptr);
    ASSERT_EQ(constraint->valid_values.size(), 2u);
    EXPECT_EQ(constraint->valid_values[0], "a32");
    EXPECT_EQ(constraint->valid_values[1], "a16");
}

// =============================================================================
// Knob-aware workspace sizing tests
// =============================================================================

TEST_F(TestSdpaBwdPlanBuilder, WorkspaceSizeDefaultSettingsIsA32)
{
    // Default Settings (nullopt) should produce same result as explicit A32
    auto graphBuilder = createSdpaBwdGraph({2, 8, 512, 128});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const Settings defaultSettings;
    Settings a32Settings;
    a32Settings.accumulatorType = AccumulatorType::A32;

    EXPECT_EQ(_planBuilder.getMaxWorkspaceSize(_handle, graphWrapper, defaultSettings),
              _planBuilder.getMaxWorkspaceSize(_handle, graphWrapper, a32Settings));
}

// The A16 workspace holds a 16-bit dq_acc on gfx950, where a dq_shuffle post
// stage reads it, and only the D buffer on gfx942, where dqdkdv writes dQ.
TEST_F(TestSdpaBwdPlanBuilder, A16WorkspaceSizeFollowsTheArch)
{
    SKIP_IF_NO_DEVICES();
    const auto arch = deviceString();
    if(!bwd_dispatch::isSupportedArch(arch))
    {
        GTEST_SKIP();
    }

    auto graphBuilder = createSdpaBwdGraph({2, 8, 200, 128});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    Settings a16Settings;
    a16Settings.accumulatorType = AccumulatorType::A16;

    // D buffer 2*8*200*4 = 12800; gfx950 dq_acc 2*8*208*128*2 = 851968
    const size_t expected = arch == "gfx950" ? 12800u + 851968u : 12800u;
    EXPECT_EQ(_planBuilder.getMaxWorkspaceSize(_handle, graphWrapper, a16Settings), expected);
}

// An A16 request the A16 kernels cannot serve is declined with BAD_PARAM
// before any kernel load, and no plan is set. buildPlan does not fall back to
// A32.
TEST_F(TestSdpaBwdPlanBuilder, BuildPlanDeclinesA16GraphsTheA16KernelsDoNotServe)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;

    SKIP_IF_NO_DEVICES();
    const auto arch = deviceString();
    if(!bwd_dispatch::isSupportedArch(arch))
    {
        GTEST_SKIP();
    }

    const auto packed = [](const std::vector<int64_t>& dims) {
        return hipdnn_data_sdk::utilities::generateStrides(dims);
    };
    std::vector<std::pair<flatbuffers::FlatBufferBuilder, std::string>> declineTests;
    if(arch == "gfx942")
    {
        // Every gfx942 A16 row is pssk = 0: S_q == S_kv, S_kv % 64 == 0.
        const std::vector<int64_t> qDims = {1, 2, 256, 128};
        const std::vector<int64_t> kvDims = {1, 2, 512, 128};
        declineTests.emplace_back(
            hipdnn_test_sdk::utilities::createValidSdpaBwdGraph(qDims,
                                                                packed(qDims),
                                                                kvDims,
                                                                packed(kvDims),
                                                                kvDims,
                                                                packed(kvDims),
                                                                qDims,
                                                                packed(qDims),
                                                                DataType::BFLOAT16),
            "S_q != S_kv");
        declineTests.emplace_back(createSdpaBwdGraph({1, 2, 200, 128}), "S_kv % 64 != 0");
        // No A16 rows above D = 128.
        declineTests.emplace_back(createSdpaBwdGraph({1, 2, 256, 192}), "D = 192");
        // dqdkdv writes dQ with Q's strides.
        const std::vector<int64_t> paddedRows
            = {int64_t{2} * 256 * 136, int64_t{256} * 136, 136, 1};
        declineTests.emplace_back(
            withLayouts(createSdpaBwdGraph({1, 2, 256, 128}), {{"dq", paddedRows, {}}}),
            "dQ rows padded, Q packed");
    }
    else
    {
        // The non-window dqdkdv kernels address dq_acc in 32 bits:
        // ceil16(S_q) * 128 * 2 bytes per head past UINT32_MAX.
        declineTests.emplace_back(createSdpaBwdGraph({1, 1, 16777232, 128}),
                                  "dq_acc head stride past 32 bits");
    }

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper noConfig(nullptr, 0);
    for(const auto& [graphBuilder, name] : declineTests)
    {
        SCOPED_TRACE(name);
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
            graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
        Settings a16Settings;
        a16Settings.accumulatorType = AccumulatorType::A16;
        Context context;
        context.setExecutionSettings(a16Settings);

        try
        {
            _planBuilder.buildPlan(_handle, graphWrapper, noConfig, context);
            ADD_FAILURE() << "buildPlan accepted the a16 accumulator";
        }
        catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
        {
            EXPECT_EQ(e.getStatus(), HIPDNN_PLUGIN_STATUS_BAD_PARAM) << e.what();
        }
        EXPECT_FALSE(context.hasValidPlan());
    }
}

} // namespace
} // namespace asm_sdpa_engine
