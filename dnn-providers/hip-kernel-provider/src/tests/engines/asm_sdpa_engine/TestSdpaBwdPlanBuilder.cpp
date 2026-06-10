// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "GraphTest.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaBwdPlanBuilder.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaKernelUtils.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaPlanUtils.hpp"
#include "hip_kernel_provider_common/HipDeviceUtils.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

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
                        bool causalMask = false)
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
                                                               causalMask);
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

        // Causal mask not currently dispatched.
        {GraphTest{
             createSdpaBwdGraph({4, 8, 256, 128}, DataType::BFLOAT16, false, false, false, true),
             "causal_mask = true"},
         false},

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
TEST(SdpaBwdRegistryLookup, RegistryLookup_Hd128Bf16NoMaskBatch)
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
TEST(SdpaBwdRegistryLookup, RegistryLookup_Hd64Fp16NoMaskBatch)
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
TEST(SdpaBwdRegistryLookup, RegistryLookup_Hd192Bf16CausalBatch)
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
TEST(SdpaBwdRegistryLookup, RegistryLookup_Gfx950Hd128Bf16NoMaskBatch)
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

TEST_F(TestSdpaBwdPlanBuilder, IsApplicable_RejectsHd96)
{
    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
    {
        GTEST_SKIP();
    }

    auto builder = createSdpaBwdGraph({2, 8, 256, 96});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaBwdPlanBuilder, IsApplicable_RejectsFp8)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
    {
        GTEST_SKIP();
    }

    auto builder = createSdpaBwdGraph({2, 8, 256, 128}, DataType::FP8_E4M3);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_handle, graphWrapper));
}

TEST_F(TestSdpaBwdPlanBuilder, IsApplicable_RejectsGfx950)
{
    // Cannot synthesise a different device string from the test harness, so
    // this test only meaningfully runs on a non-gfx942 device. On gfx942 it
    // is skipped (the positive case is covered by IsApplicableSdpaBwdVariations).
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

TEST_F(TestSdpaBwdPlanBuilder, IsApplicable_RejectsFractionalGqaRatio)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    if(hip_kernel_provider_common::getDeviceString(_handle.getStream()) != "gfx942")
    {
        GTEST_SKIP();
    }

    // nhead_q = 6, nhead_k = 4 → 6 % 4 = 2.  SdpaBwdPlan would silently
    // truncate ratio = 6/4 = 1, dropping K/V heads in dispatch.
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

TEST_F(TestSdpaBwdPlanBuilder, IsApplicable_RejectsAsymmetricHdim)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

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
// Canonical mask-attribute policy (plan_utils::getMaskType)
// =============================================================================
//
// These tests exercise the shared mask-precedence policy directly through
// plan_utils::getMaskType rather than through isApplicable. When a deprecated
// causal boolean is set it takes precedence over the modern bounds trio; only
// setting both deprecated booleans at once throws. The policy is
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

// Resolve the SDPA backward attributes from a graph buffer and classify the mask.
plan_utils::MaskType classifyMask(const flatbuffers::FlatBufferBuilder& builder)
{
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());
    const auto& attrs
        = graphWrapper.nodeWrappers()
              .front()
              ->attributesAs<hipdnn_flatbuffers_sdk::data_objects::SdpaBackwardAttributes>();
    return plan_utils::getMaskType(attrs);
}

TEST_F(TestSdpaBwdPlanBuilder, IsApplicable_RejectsCausalMaskAndBottomRightSetTogether)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    // Both deprecated causal booleans set is a contradiction.
    auto builder = createSdpaBwdGraphWithMask(
        /*causalMask=*/true,
        /*causalMaskBottomRight=*/true,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        DiagonalAlignment::TOP_LEFT);

    EXPECT_THROW(classifyMask(builder), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST_F(TestSdpaBwdPlanBuilder, IsApplicable_PrefersCausalMaskOverWindowBounds)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    // causal_mask=true takes precedence over the bounds trio, even though the
    // bounds describe a sliding window (left=64, right=64): result is causal.
    auto builder = createSdpaBwdGraphWithMask(
        /*causalMask=*/true,
        /*causalMaskBottomRight=*/false,
        flatbuffers::Optional<int64_t>(64),
        flatbuffers::Optional<int64_t>(64),
        DiagonalAlignment::TOP_LEFT);

    plan_utils::MaskType maskType = plan_utils::MaskType::NO_MASK;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::TOP_LEFT_CAUSAL);
}

TEST_F(TestSdpaBwdPlanBuilder, IsApplicable_PrefersBottomRightCausalOverTopLeftBounds)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    // causal_mask_bottom_right=true takes precedence over the bounds trio, even
    // though the trio (left=-1, right=0, diag=TOP_LEFT) would derive
    // TOP_LEFT_CAUSAL: result is bottom-right causal.
    auto builder = createSdpaBwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/true,
        flatbuffers::Optional<int64_t>(-1),
        flatbuffers::Optional<int64_t>(0),
        DiagonalAlignment::TOP_LEFT);

    plan_utils::MaskType maskType = plan_utils::MaskType::NO_MASK;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::BOTTOM_RIGHT_CAUSAL);
}

TEST_F(TestSdpaBwdPlanBuilder, IsApplicable_AcceptsConsistentCausalMaskAndBounds)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    // causal_mask=true with a consistent bounds trio (left=-1, right=0,
    // diag=TOP_LEFT -> TOP_LEFT_CAUSAL) must not throw and classify as causal.
    auto builder = createSdpaBwdGraphWithMask(
        /*causalMask=*/true,
        /*causalMaskBottomRight=*/false,
        flatbuffers::Optional<int64_t>(-1),
        flatbuffers::Optional<int64_t>(0),
        DiagonalAlignment::TOP_LEFT);

    plan_utils::MaskType maskType = plan_utils::MaskType::NO_MASK;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::TOP_LEFT_CAUSAL);
}

TEST_F(TestSdpaBwdPlanBuilder, IsApplicable_PrefersBottomRightCausalOverWindowBounds)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    // causal_mask_bottom_right=true takes precedence over the bounds trio, even
    // though a symmetric sliding window (left=64, right=64) would derive
    // WINDOW_GENERIC: result is bottom-right causal.
    auto builder = createSdpaBwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/true,
        flatbuffers::Optional<int64_t>(64),
        flatbuffers::Optional<int64_t>(64),
        DiagonalAlignment::BOTTOM_RIGHT);

    plan_utils::MaskType maskType = plan_utils::MaskType::NO_MASK;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::BOTTOM_RIGHT_CAUSAL);
}

// Modern bounds-trio path (no deprecated boolean set). An unset bound is treated
// as unbounded (-1), so a partially specified trio still derives the mask it
// describes.

TEST_F(TestSdpaBwdPlanBuilder, MaskBoundsTrio_RightZeroLeftUnsetDerivesTopLeftCausal)
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

TEST_F(TestSdpaBwdPlanBuilder, MaskBoundsTrio_RightZeroBottomRightDerivesBottomRightCausal)
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

TEST_F(TestSdpaBwdPlanBuilder, MaskBoundsTrio_ExplicitCausalBoundsDeriveTopLeftCausal)
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

TEST_F(TestSdpaBwdPlanBuilder, MaskBoundsTrio_BothUnsetDerivesNoMask)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    auto builder = createSdpaBwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/false,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        DiagonalAlignment::TOP_LEFT);

    plan_utils::MaskType maskType = plan_utils::MaskType::WINDOW_GENERIC;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::NO_MASK);
}

TEST_F(TestSdpaBwdPlanBuilder, MaskBoundsTrio_BothUnboundedDerivesNoMask)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    auto builder = createSdpaBwdGraphWithMask(
        /*causalMask=*/false,
        /*causalMaskBottomRight=*/false,
        flatbuffers::Optional<int64_t>(-1),
        flatbuffers::Optional<int64_t>(-1),
        DiagonalAlignment::TOP_LEFT);

    plan_utils::MaskType maskType = plan_utils::MaskType::WINDOW_GENERIC;
    EXPECT_NO_THROW(maskType = classifyMask(builder));
    EXPECT_EQ(maskType, plan_utils::MaskType::NO_MASK);
}

TEST_F(TestSdpaBwdPlanBuilder, MaskBoundsTrio_SymmetricWindowDerivesWindowGeneric)
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
    EXPECT_EQ(maskType, plan_utils::MaskType::WINDOW_GENERIC);
}

TEST_F(TestSdpaBwdPlanBuilder, MaskBoundsTrio_LeftOnlyDerivesWindowGeneric)
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
    EXPECT_EQ(maskType, plan_utils::MaskType::WINDOW_GENERIC);
}

// =============================================================================
// Accumulator-aware workspace sizing tests
// =============================================================================

TEST(SdpaBwdWorkspaceSize, A32IncludesDqAccBuffer)
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

TEST(SdpaBwdWorkspaceSize, A16IsDBufferOnly)
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

TEST(SdpaBwdWorkspaceSize, A16SmallerThanA32)
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

TEST(SdpaBwdWorkspaceSize, A16UnalignedDimensions)
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
