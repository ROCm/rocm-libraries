// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "GraphTest.hpp"
#include "asm_fmha_v3_bwd_configs.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaBwdPlanBuilder.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaKernelUtils.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaPlanUtils.hpp"
#include "hip_kernel_provider_common/HipDeviceUtils.hpp"
#include "tests/engines/asm_sdpa_engine/DeprecatedCausalMaskCases.hpp"

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

class TestSdpaBwdPlanBuilder : public ::testing::Test
{
protected:
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

TEST_F(TestSdpaBwdPlanBuilder, IsApplicableSdpaBwdVariations)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    SKIP_IF_NO_DEVICES();

    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
    {
        GTEST_SKIP();
    }

    const std::vector<std::pair<GraphTest, bool>> applicabilityTests = {
        // Valid backward graph: BF16, HD=128, FP32 stats, no masking — the one
        // configuration with a calibrated CPU reference today (logs INFO).
        {GraphTest{createSdpaBwdGraph(), "Valid BF16 HD128 backward"}, true},

        // HD=64: rejected by the registry lookup, not by a dtype/hdim gate. The
        // hd64 registry carries only pddv=0 rows, but the day-one dispatch tuple
        // requests pddv=1, so no dqdkdv row matches.
        {GraphTest{createSdpaBwdGraph({4, 8, 256, 64}), "Head dimension 64"}, false},

        // HD=192 BF16: registry-supported and now dispatched. The CPU reference
        // is not yet calibrated, so buildPlan logs a one-time WARN; isApplicable
        // accepts.
        {GraphTest{createSdpaBwdGraph({4, 8, 256, 192}), "Head dimension 192"}, true},

        // FP16 HD128: registry-supported and dispatched, with a calibrated CPU
        // reference (logs INFO, like bf16 hd128); isApplicable accepts.
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

    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
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

    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
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

    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
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
constexpr int ATOMIC_A32 = 1; // AccumulatorMode::A32
constexpr int ATOMIC_NONE = 0; // odo / dq_convert use atomic32=0
constexpr int PSSK_ON = 1;
constexpr int PDDV_ON = 1;
constexpr int PSSK_OFF = 0;
constexpr int PDDV_OFF = 0;
constexpr int BF16_CVT_RTNE = 0; // RoundingMode::RTNE
using asm_sdpa_engine::bwd_dispatch::BF16_CVT_FP16_SENTINEL;
} // namespace

// POC config: hd128 / bf16 / NO_MASK / BATCH must still resolve across all
// three pipeline-stage registries.
TEST(TestSdpaBwdRegistryLookup, RegistryLookupHd128Bf16NoMaskBatch)
{
    using namespace bwd_dispatch;

    auto odoKey = lookupKernelNameKey(PipelineStage::ODO,
                                      "gfx942",
                                      "bf16",
                                      /*hdimQ=*/128,
                                      /*hdimV=*/128,
                                      MASK_NONE,
                                      ATOMIC_NONE,
                                      PSSK_OFF,
                                      PDDV_OFF,
                                      MODE_BATCH,
                                      BF16_CVT_FP16_SENTINEL);
    EXPECT_FALSE(odoKey.empty()) << "odo lookup should resolve";

    auto dqdkdvKey = lookupKernelNameKey(PipelineStage::DQDKDV,
                                         "gfx942",
                                         "bf16",
                                         /*hdimQ=*/128,
                                         /*hdimV=*/128,
                                         MASK_NONE,
                                         ATOMIC_A32,
                                         PSSK_ON,
                                         PDDV_ON,
                                         MODE_BATCH,
                                         BF16_CVT_RTNE);
    EXPECT_FALSE(dqdkdvKey.empty()) << "dqdkdv lookup should resolve";

    auto dqConvertKey = lookupKernelNameKey(PipelineStage::DQ_CONVERT,
                                            "gfx942",
                                            "bf16",
                                            /*hdimQ=*/128,
                                            /*hdimV=*/128,
                                            MASK_NONE,
                                            ATOMIC_NONE,
                                            PSSK_OFF,
                                            PDDV_OFF,
                                            MODE_BATCH,
                                            BF16_CVT_RTNE);
    EXPECT_FALSE(dqConvertKey.empty()) << "dq_convert lookup should resolve";
}

// FP16 / hd64 row resolves via the unpadded (pssk=1, pddv=0) layout that the
// registry actually carries for that head-dim. This exercises the lookup for
// a hd64 entry even though isApplicable's day-one matrix forces pddv=1 and
// therefore rejects hd64 at dispatch time.
TEST(TestSdpaBwdRegistryLookupFp16, RegistryLookupHd64NoMaskBatch)
{
    using namespace bwd_dispatch;

    auto odoKey = lookupKernelNameKey(PipelineStage::ODO,
                                      "gfx942",
                                      "fp16",
                                      /*hdimQ=*/64,
                                      /*hdimV=*/64,
                                      MASK_NONE,
                                      ATOMIC_NONE,
                                      PSSK_OFF,
                                      PDDV_OFF,
                                      MODE_BATCH,
                                      BF16_CVT_FP16_SENTINEL);
    EXPECT_FALSE(odoKey.empty());

    // hd64 dqdkdv is registered with pssk=1, pddv=0 (no padded-D variant).
    auto dqdkdvKey = lookupKernelNameKey(PipelineStage::DQDKDV,
                                         "gfx942",
                                         "fp16",
                                         /*hdimQ=*/64,
                                         /*hdimV=*/64,
                                         MASK_NONE,
                                         ATOMIC_A32,
                                         PSSK_ON,
                                         PDDV_OFF,
                                         MODE_BATCH,
                                         BF16_CVT_FP16_SENTINEL);
    EXPECT_FALSE(dqdkdvKey.empty());

    auto dqConvertKey = lookupKernelNameKey(PipelineStage::DQ_CONVERT,
                                            "gfx942",
                                            "fp16",
                                            /*hdimQ=*/64,
                                            /*hdimV=*/64,
                                            MASK_NONE,
                                            ATOMIC_NONE,
                                            PSSK_OFF,
                                            PDDV_OFF,
                                            MODE_BATCH,
                                            BF16_CVT_FP16_SENTINEL);
    EXPECT_FALSE(dqConvertKey.empty());
}

// hd192 / bf16 / TOP_LEFT_CAUSAL row resolves via the registry. isApplicable
// rejects causal day-one but the lookup itself must work.
TEST(TestSdpaBwdRegistryLookup, RegistryLookupHd192Bf16CausalBatch)
{
    using namespace bwd_dispatch;

    // Causal hd192/bf16 dqdkdv batch kernels are pssk=1, pddv=1 in the registry.
    auto dqdkdvKey = lookupKernelNameKey(PipelineStage::DQDKDV,
                                         "gfx942",
                                         "bf16",
                                         /*hdimQ=*/192,
                                         /*hdimV=*/192,
                                         MASK_TOP_LEFT_CAUSAL,
                                         ATOMIC_A32,
                                         PSSK_ON,
                                         PDDV_ON,
                                         MODE_BATCH,
                                         BF16_CVT_RTNE);
    EXPECT_FALSE(dqdkdvKey.empty()) << "hd192 bf16 causal_tl dqdkdv lookup should resolve";

    // odo is mask-agnostic.
    auto odoKey = lookupKernelNameKey(PipelineStage::ODO,
                                      "gfx942",
                                      "bf16",
                                      /*hdimQ=*/192,
                                      /*hdimV=*/192,
                                      MASK_NONE,
                                      ATOMIC_NONE,
                                      PSSK_OFF,
                                      PDDV_OFF,
                                      MODE_BATCH,
                                      BF16_CVT_FP16_SENTINEL);
    EXPECT_FALSE(odoKey.empty()) << "hd192 bf16 odo lookup should resolve";
}

// gfx950 BF16 rows store bf16_cvt=3 (the FP16 sentinel) regardless of the
// rounding mode the caller asked for, because gfx950 ships only one kernel
// per (dtype, hdim, mask, atomic, pssk, pddv, mode) tuple.  findKey must
// skip the bf16_cvt comparison on gfx950 so a caller passing the
// graph-derived RoundingMode (e.g. RTNE = 0) still resolves the row.
// isApplicable rejects gfx950 day-one, so we exercise the registry helper
// directly.
TEST(TestSdpaBwdRegistryLookup, RegistryLookupGfx950Hd128Bf16NoMaskBatch)
{
    using namespace bwd_dispatch;

    auto dqdkdvKey = lookupKernelNameKey(PipelineStage::DQDKDV,
                                         "gfx950",
                                         "bf16",
                                         /*hdimQ=*/128,
                                         /*hdimV=*/128,
                                         MASK_NONE,
                                         ATOMIC_A32,
                                         PSSK_ON,
                                         PDDV_ON,
                                         MODE_BATCH,
                                         BF16_CVT_RTNE);
    EXPECT_FALSE(dqdkdvKey.empty())
        << "gfx950 bf16 hd128 a32 dqdkdv lookup should resolve despite bf16_cvt mismatch";

    auto dqConvertKey = lookupKernelNameKey(PipelineStage::DQ_CONVERT,
                                            "gfx950",
                                            "bf16",
                                            /*hdimQ=*/128,
                                            /*hdimV=*/128,
                                            MASK_NONE,
                                            ATOMIC_NONE,
                                            PSSK_OFF,
                                            PDDV_OFF,
                                            MODE_BATCH,
                                            BF16_CVT_RTNE);
    EXPECT_FALSE(dqConvertKey.empty())
        << "gfx950 bf16 hd128 dq_convert lookup should resolve despite bf16_cvt mismatch";

    auto odoKey = lookupKernelNameKey(PipelineStage::ODO,
                                      "gfx950",
                                      "bf16",
                                      /*hdimQ=*/128,
                                      /*hdimV=*/128,
                                      MASK_NONE,
                                      ATOMIC_NONE,
                                      PSSK_OFF,
                                      PDDV_OFF,
                                      MODE_BATCH,
                                      BF16_CVT_FP16_SENTINEL);
    EXPECT_FALSE(odoKey.empty()) << "gfx950 bf16 hd128 odo lookup should resolve";
}

// Causal (mask=1) dispatch: DQDKDV uses mask=1 row; ODO and DQ_CONVERT remain
// mask=0 (mask-agnostic stages).
TEST(TestSdpaBwdRegistryLookup, RegistryLookupHd128Bf16CausalBatch)
{
    using namespace bwd_dispatch;

    auto dqdkdvKey = lookupKernelNameKey(PipelineStage::DQDKDV,
                                         "gfx942",
                                         "bf16",
                                         /*hdimQ=*/128,
                                         /*hdimV=*/128,
                                         MASK_TOP_LEFT_CAUSAL,
                                         ATOMIC_A32,
                                         PSSK_ON,
                                         PDDV_ON,
                                         MODE_BATCH,
                                         BF16_CVT_RTNE);
    EXPECT_FALSE(dqdkdvKey.empty()) << "hd128 bf16 causal_tl dqdkdv lookup should resolve";

    // ODO is mask-agnostic (always mask=0).
    auto odoKey = lookupKernelNameKey(PipelineStage::ODO,
                                      "gfx942",
                                      "bf16",
                                      /*hdimQ=*/128,
                                      /*hdimV=*/128,
                                      MASK_NONE,
                                      ATOMIC_NONE,
                                      PSSK_OFF,
                                      PDDV_OFF,
                                      MODE_BATCH,
                                      BF16_CVT_FP16_SENTINEL);
    EXPECT_FALSE(odoKey.empty()) << "hd128 bf16 odo lookup should resolve for causal dispatch";

    // DQ_CONVERT is mask-agnostic (always mask=0).
    auto dqConvertKey = lookupKernelNameKey(PipelineStage::DQ_CONVERT,
                                            "gfx942",
                                            "bf16",
                                            /*hdimQ=*/128,
                                            /*hdimV=*/128,
                                            MASK_NONE,
                                            ATOMIC_NONE,
                                            PSSK_OFF,
                                            PDDV_OFF,
                                            MODE_BATCH,
                                            BF16_CVT_RTNE);
    EXPECT_FALSE(dqConvertKey.empty())
        << "hd128 bf16 dq_convert lookup should resolve for causal dispatch";
}

// Bottom-right causal (mask=2) dispatch.
TEST(TestSdpaBwdRegistryLookup, RegistryLookupHd128Bf16CausalBrBatch)
{
    using namespace bwd_dispatch;

    auto dqdkdvKey = lookupKernelNameKey(PipelineStage::DQDKDV,
                                         "gfx942",
                                         "bf16",
                                         /*hdimQ=*/128,
                                         /*hdimV=*/128,
                                         MASK_BOTTOM_RIGHT_CAUSAL,
                                         ATOMIC_A32,
                                         PSSK_ON,
                                         PDDV_ON,
                                         MODE_BATCH,
                                         BF16_CVT_RTNE);
    EXPECT_FALSE(dqdkdvKey.empty()) << "hd128 bf16 causal_br dqdkdv lookup should resolve";
}

// SWA / window (mask=3) dispatch — only hd128 gfx942 has SWA rows in the CSV.
TEST(TestSdpaBwdRegistryLookup, RegistryLookupHd128Bf16SwaBatch)
{
    using namespace bwd_dispatch;

    auto dqdkdvKey = lookupKernelNameKey(PipelineStage::DQDKDV,
                                         "gfx942",
                                         "bf16",
                                         /*hdimQ=*/128,
                                         /*hdimV=*/128,
                                         MASK_SLIDING_WINDOW,
                                         ATOMIC_A32,
                                         PSSK_ON,
                                         PDDV_ON,
                                         MODE_BATCH,
                                         BF16_CVT_RTNE);
    EXPECT_FALSE(dqdkdvKey.empty()) << "hd128 bf16 swa dqdkdv lookup should resolve";
}

// FP16 causal (mask=1) dispatch.
TEST(TestSdpaBwdRegistryLookupFp16, RegistryLookupHd128CausalBatch)
{
    using namespace bwd_dispatch;

    auto dqdkdvKey = lookupKernelNameKey(PipelineStage::DQDKDV,
                                         "gfx942",
                                         "fp16",
                                         /*hdimQ=*/128,
                                         /*hdimV=*/128,
                                         MASK_TOP_LEFT_CAUSAL,
                                         ATOMIC_A32,
                                         PSSK_ON,
                                         PDDV_ON,
                                         MODE_BATCH,
                                         BF16_CVT_FP16_SENTINEL);
    EXPECT_FALSE(dqdkdvKey.empty()) << "hd128 fp16 causal_tl dqdkdv lookup should resolve";
}

TEST_F(TestSdpaBwdPlanBuilder, IsApplicableRejectsHd96)
{
    SKIP_IF_NO_DEVICES();

    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
    {
        GTEST_SKIP();
    }

    auto builder = createSdpaBwdGraph({2, 8, 256, 96});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaBwdPlanBuilder, IsApplicableRejectsFp8)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    SKIP_IF_NO_DEVICES();

    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
    {
        GTEST_SKIP();
    }

    auto builder = createSdpaBwdGraph({2, 8, 256, 128}, DataType::FP8_E4M3);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaBwdPlanBuilder, IsApplicableRejectsGfx950)
{
    // Cannot synthesise a different device string from the test harness, so
    // this test only meaningfully runs on a non-gfx942 device. On gfx942 it
    // is skipped (the positive case is covered by IsApplicableSdpaBwdVariations).
    SKIP_IF_NO_DEVICES();

    auto deviceString = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString == "gfx942")
    {
        GTEST_SKIP() << "Test requires non-gfx942 device to assert rejection";
    }

    auto builder = createSdpaBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaBwdPlanBuilder, IsApplicableRejectsFractionalGqaRatio)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    SKIP_IF_NO_DEVICES();

    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
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

    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
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

TEST_F(TestSdpaBwdPlanBuilder, IsApplicableRejectsAsymmetricHdim)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    SKIP_IF_NO_DEVICES();

    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
    {
        GTEST_SKIP();
    }

    // createSdpaBwdGraph enforces D_qk == D_v, so this test builds the graph
    // directly to exercise an asymmetric layout (D_qk=128, D_v=64).
    // V tensor has D_v = 64 while Q/K have D_qk = 128 — backward kernels
    // require square head dimensions (D_qk == D_v).
    const std::vector<int64_t> qDims = {2, 8, 256, 128};
    const std::vector<int64_t> kDims = {2, 8, 256, 128};
    const std::vector<int64_t> vDims = {2, 8, 256, 64};
    const std::vector<int64_t> oDims = {2, 8, 256, 64};

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
// 4-byte record index, any stride on a length-1 axis, zero batch or head strides on
// the inputs (broadcasts), and stats whose size-1 axes have any stride (buildPlan
// derives the LSE/D strides from Q's dims). The B=2, H=2 non-packed cases of
// IntegrationGpuSdpaBackward run BSHD Q, K, V and dO with O and dQ rows padded to
// D+1 on the device.
TEST_F(TestSdpaBwdPlanBuilder, IsApplicableAcceptsAddressableLayouts)
{
    SKIP_IF_NO_DEVICES();

    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
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
    // The stride of a size-1 axis only ever multiplies index 0.
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

    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
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

    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
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

    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
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

    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
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

    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
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
// kernel registry), so the assertions are meaningful on any device — including
// this gfx950 box. The backward isApplicable cannot be used here: it rejects
// non-gfx942 devices at its first gate, so on gfx950 it would return false
// before ever reaching getMaskType, making an isApplicable-based assertion
// pass for the wrong reason.

// Build a backward SDPA graph that sets the deprecated causal booleans and the
// modern bounds trio explicitly, so contradictory combinations can be
// constructed. Returns the FlatBufferBuilder owning the graph buffer.
flatbuffers::FlatBufferBuilder createSdpaBwdGraphWithMask(
    bool causalMask,
    bool causalMaskBottomRight,
    flatbuffers::Optional<int64_t> leftBound,
    flatbuffers::Optional<int64_t> rightBound,
    hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment diagAlignment)
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

    // left_bound = 64 keeps 64 keys (cuDNN), which is FlashAttention's window_size_left 63.
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
// gfx942 device to exercise the full isApplicable() path.

TEST_F(TestSdpaBwdPlanBuilder, IsApplicableAcceptsCausalMaskBottomRight)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    SKIP_IF_NO_DEVICES();

    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
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

    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
    {
        GTEST_SKIP();
    }

    // Symmetric sliding window with left=64, right=64
    auto builder = createSdpaBwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/false,
        flatbuffers::Optional<int64_t>(64),
        flatbuffers::Optional<int64_t>(64),
        DiagonalAlignment::TOP_LEFT);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_handle, graphWrapper));
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

    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
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

    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
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

// The head dims are narrowed to int before the {64, 128, 192} membership test
// and the registry lookup, so a head dim whose low 32 bits are a supported
// value used to resolve a kernel built for that supported value. That registry
// set is what bounds the head dims in practice, which is why there is no case
// at INT32_MAX here.
TEST_F(TestSdpaBwdPlanBuilder, IsApplicableDeclinesHeadDimsThatTruncateToASupportedValue)
{
    SKIP_IF_NO_DEVICES();

    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
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

// The `ts` of the DQDKDV row isApplicable resolves for a bf16 hd128 no-mask
// batch graph on the default A32 accumulator path.
std::optional<int> resolveBwdTileSizeKv()
{
    for(const auto& entry : cfg_fmha_bwd_dqdkdv)
    {
        const auto& cfg = entry.second;
        if(cfg.arch == "gfx942" && cfg.dtype == "bf16" && cfg.hdim_q == 128 && cfg.hdim_v == 128
           && cfg.mask == 0 && cfg.atomic32 == 1 && cfg.pssk == 1 && cfg.pddv == 1 && cfg.mode == 0
           && cfg.bf16_cvt == 0)
        {
            return cfg.ts;
        }
    }
    return std::nullopt;
}

// The Ts kernarg is tsKv * stride_k * sizeof(bf16). The per-field stride guard
// only bounds stride_k * 2, so the tile factor is what this guard adds: the
// accepted K sequence stride shrinks by tsKv.
TEST_F(TestSdpaBwdPlanBuilder, IsApplicableBoundsTheTsProductByTheResolvedTileSize)
{
    SKIP_IF_NO_DEVICES();

    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
    {
        GTEST_SKIP();
    }

    const auto tileSizeKv = resolveBwdTileSizeKv();
    ASSERT_TRUE(tileSizeKv.has_value()) << "no gfx942 bf16 hd128 no-mask a32 dqdkdv row";

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
// Accumulator-aware workspace sizing tests
// =============================================================================

TEST(TestSdpaBwdWorkspaceSize, A32IncludesDqAccBuffer)
{
    // a32: workspace = D buffer + dq_acc buffer
    constexpr size_t K_B = 2;
    constexpr size_t K_H = 8;
    constexpr size_t K_S = 512;
    constexpr size_t K_D = 128;
    const size_t a32Size = sdpaBwdWorkspaceSize(K_B, K_H, K_S, K_D, AccumulatorType::A32);
    const size_t expected
        = sdpaBwdDBufferSize(K_B, K_H, K_S) + sdpaBwdDqAccBufferSize(K_B, K_H, K_S, K_D);
    EXPECT_EQ(a32Size, expected);
    // Verify dq_acc is the dominant term: D buffer << dq_acc buffer
    EXPECT_GT(a32Size, sdpaBwdDBufferSize(K_B, K_H, K_S));
}

TEST(TestSdpaBwdWorkspaceSize, A16IsDBufferOnly)
{
    // a16: workspace = D buffer only (no dq_acc)
    constexpr size_t K_B = 2;
    constexpr size_t K_H = 8;
    constexpr size_t K_S = 512;
    constexpr size_t K_D = 128;
    const size_t a16Size = sdpaBwdWorkspaceSize(K_B, K_H, K_S, K_D, AccumulatorType::A16);
    const size_t dBufferOnly = sdpaBwdDBufferSize(K_B, K_H, K_S);
    EXPECT_EQ(a16Size, dBufferOnly);
}

TEST(TestSdpaBwdWorkspaceSize, A16SmallerThanA32)
{
    constexpr size_t K_B = 4;
    constexpr size_t K_H = 16;
    constexpr size_t K_S = 1024;
    constexpr size_t K_D = 128;
    const size_t a32Size = sdpaBwdWorkspaceSize(K_B, K_H, K_S, K_D, AccumulatorType::A32);
    const size_t a16Size = sdpaBwdWorkspaceSize(K_B, K_H, K_S, K_D, AccumulatorType::A16);
    EXPECT_LT(a16Size, a32Size);
    // a16 savings = dq_acc buffer = B*H*S*D*4
    EXPECT_EQ(a32Size - a16Size, sdpaBwdDqAccBufferSize(K_B, K_H, K_S, K_D));
}

TEST(TestSdpaBwdWorkspaceSize, A16UnalignedDimensions)
{
    // B=1, H=3, S=255 — D buffer raw size (3060) is NOT aligned to 64
    constexpr size_t K_B = 1;
    constexpr size_t K_H = 3;
    constexpr size_t K_S = 255;
    constexpr size_t K_D = 128;
    const size_t a16Size = sdpaBwdWorkspaceSize(K_B, K_H, K_S, K_D, AccumulatorType::A16);
    // D buffer: 1*3*255*4 = 3060, aligned to 64 → 3072
    EXPECT_EQ(a16Size, 3072u);
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

    // TODO(ALMIOPEN-1833): Remove gfx942 restriction once gfx950 is enabled
    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
    {
        GTEST_SKIP();
    }

    auto graphBuilder = createSdpaBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const auto knobs = _planBuilder.getCustomKnobs(_handle, graphWrapper);
    ASSERT_EQ(knobs.size(), 1u);
    EXPECT_EQ(knobs[0].knob_id, "sdpa.bwd.accumulator_type");

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

} // namespace
} // namespace asm_sdpa_engine
