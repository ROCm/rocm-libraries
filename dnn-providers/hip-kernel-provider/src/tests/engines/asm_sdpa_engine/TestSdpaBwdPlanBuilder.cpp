// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "GraphTest.hpp"
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
    const HipKernelSettings settings;
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
    const HipKernelSettings settings;
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
    const HipKernelSettings settings;
    const size_t workspaceSize = _planBuilder.getMaxWorkspaceSize(_handle, graphWrapper, settings);

    // D buffer: 4*16*1024*4 = 262144, aligned to 64 → 262144
    // dq_acc:   4*16*1024*128*4 = 33554432, aligned to 64 → 33554432
    EXPECT_EQ(workspaceSize, 262144u + 33554432u);
}

} // namespace
} // namespace asm_sdpa_engine
