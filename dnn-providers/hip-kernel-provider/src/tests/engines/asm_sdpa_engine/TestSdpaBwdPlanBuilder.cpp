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
#include "hip_kernel_provider_common/HipDeviceUtils.hpp"

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
        // Valid backward graph: BF16, HD=128, FP32 stats, no masking
        {GraphTest{createSdpaBwdGraph(), "Valid BF16 HD128 backward"}, true},

        // Wrong head dimension
        {GraphTest{createSdpaBwdGraph({4, 8, 256, 64}), "Head dimension 64"}, false},

        // Wrong tensor dtype (FP16)
        {GraphTest{createSdpaBwdGraph({4, 8, 256, 128}, DataType::HALF), "FP16 tensors"}, false},

        // Causal mask enabled
        {GraphTest{
             createSdpaBwdGraph({4, 8, 256, 128}, DataType::BFLOAT16, false, false, false, true),
             "causal_mask = true"},
         false},

        // Alibi mask enabled
        {GraphTest{createSdpaBwdGraph({4, 8, 256, 128}, DataType::BFLOAT16, false, true),
                   "alibi_mask = true"},
         false},

        // Padding mask enabled
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
