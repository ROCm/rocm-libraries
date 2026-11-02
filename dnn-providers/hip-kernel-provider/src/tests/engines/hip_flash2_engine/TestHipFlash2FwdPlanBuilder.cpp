// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Unit tests for HipFlash2FwdPlanBuilder.

#include <gtest/gtest.h>

#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Handle.hpp"
#include "engines/hip_flash2_engine/HipFlash2FwdPlanBuilder_v2.hpp"
#include "tests/engines/asm_sdpa_engine/DeprecatedCausalMaskCases.hpp"
#include "tests/engines/asm_sdpa_engine/SdpaSinkTokenGraph.hpp"

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include <cctype>
#include <string>
#include <vector>

namespace hip_flash2_engine
{
namespace
{

class TestHipFlash2FwdPlanBuilder : public ::testing::Test
{
protected:
    Handle _handle;
    HipFlash2FwdPlanBuilder _builder;
};

// cuDNN's default: no attn_scale_value and no scale tensor means no scaling.
TEST(TestHipFlash2AttnScale, AbsentAttnScaleValueIsOne)
{
    const std::vector<int64_t> dims{1, 32, 4096, 128};
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    auto builder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        dims,
        strides,
        dims,
        strides,
        dims,
        strides,
        dims,
        strides,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        /*withAttnMask=*/false,
        /*withScale=*/false);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto& attrs = graph.getNodeWrapper(0)
                            .attributesAs<hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes>();
    ASSERT_FALSE(attrs.attn_scale_value().has_value());
    EXPECT_EQ(attnScaleFor(attrs), 1.0f);
}

// -- isApplicable: valid cases -------------------------------------------------

TEST_F(TestHipFlash2FwdPlanBuilder, AcceptsFP16MHACausal)
{
    SKIP_IF_NO_DEVICES();
    const auto arch = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(arch != "gfx942")
    {
        GTEST_SKIP();
    }

    const std::vector<int64_t> dims{1, 32, 4096, 128};
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    auto builder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        dims,
        strides,
        dims,
        strides,
        dims,
        strides,
        dims,
        strides,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        /*withAttnMask=*/false,
        /*withScale=*/false,
        /*withStats=*/false,
        /*alibiMask=*/false,
        /*paddingMask=*/false,
        /*causalMask=*/true);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipFlash2FwdPlanBuilder, AcceptsFP16MHANonCausal)
{
    SKIP_IF_NO_DEVICES();
    const auto arch = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(arch != "gfx942")
    {
        GTEST_SKIP();
    }

    const std::vector<int64_t> dims{1, 32, 2048, 128};
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    auto builder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        dims,
        strides,
        dims,
        strides,
        dims,
        strides,
        dims,
        strides,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        /*withAttnMask=*/false,
        /*withScale=*/false,
        /*withStats=*/false,
        /*alibiMask=*/false,
        /*paddingMask=*/false,
        /*causalMask=*/false);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipFlash2FwdPlanBuilder, AcceptsFP16HeadDim64)
{
    SKIP_IF_NO_DEVICES();
    const auto arch = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(arch != "gfx942")
    {
        GTEST_SKIP();
    }

    const std::vector<int64_t> dims{1, 32, 2048, 64};
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    auto builder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        dims,
        strides,
        dims,
        strides,
        dims,
        strides,
        dims,
        strides,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(_builder.isApplicable(_handle, graph));
}

// GQA: different Q heads vs KV heads
TEST_F(TestHipFlash2FwdPlanBuilder, AcceptsFP16GQA)
{
    SKIP_IF_NO_DEVICES();
    const auto arch = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(arch != "gfx942")
    {
        GTEST_SKIP();
    }

    const std::vector<int64_t> qDims{1, 32, 4096, 128}; // 32 query heads
    const std::vector<int64_t> kvDims{1, 8, 4096, 128}; // 8 KV heads (GQA ratio=4)
    const auto qStrides = hipdnn_data_sdk::utilities::generateStrides(qDims);
    const auto kvStrides = hipdnn_data_sdk::utilities::generateStrides(kvDims);
    auto builder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        qDims,
        qStrides,
        kvDims,
        kvStrides,
        kvDims,
        kvStrides,
        qDims,
        qStrides,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        /*withAttnMask=*/false,
        /*withScale=*/false,
        /*withStats=*/false,
        /*alibiMask=*/false,
        /*paddingMask=*/false,
        /*causalMask=*/true);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    EXPECT_TRUE(_builder.isApplicable(_handle, graph));
}

// -- isApplicable: rejection cases --------------------------------------------

TEST_F(TestHipFlash2FwdPlanBuilder, RejectsBF16)
{
    SKIP_IF_NO_DEVICES();
    const std::vector<int64_t> dims{1, 32, 2048, 128};
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    auto builder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        dims,
        strides,
        dims,
        strides,
        dims,
        strides,
        dims,
        strides,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipFlash2FwdPlanBuilder, RejectsUnsupportedHeadDim256)
{
    SKIP_IF_NO_DEVICES();
    const auto arch = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(arch != "gfx942")
    {
        GTEST_SKIP();
    }

    const std::vector<int64_t> dims{1, 32, 2048, 256};
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    auto builder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        dims,
        strides,
        dims,
        strides,
        dims,
        strides,
        dims,
        strides,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

TEST_F(TestHipFlash2FwdPlanBuilder, RejectsShortSequenceDecodeLength)
{
    SKIP_IF_NO_DEVICES();
    const auto arch = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(arch != "gfx942")
    {
        GTEST_SKIP();
    }

    // seq_q=1 means decode -- should use batched GEMM, not Flash2
    const std::vector<int64_t> qDims{1, 32, 1, 128};
    const std::vector<int64_t> kvDims{1, 32, 2048, 128};
    const auto qStrides = hipdnn_data_sdk::utilities::generateStrides(qDims);
    const auto kvStrides = hipdnn_data_sdk::utilities::generateStrides(kvDims);
    auto builder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        qDims,
        qStrides,
        kvDims,
        kvStrides,
        kvDims,
        kvStrides,
        qDims,
        qStrides,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(_builder.isApplicable(_handle, graph));
}

// -- isApplicable: 32-bit dimension range -------------------------------------
//
// Flash2FwdParams narrows every dimension to int, so a dimension that does not
// fit is dispatched as a different shape. Only descriptors are built here, so
// the dims cost no memory.

struct Flash2DimRangeCase
{
    const char* label;
    std::vector<int64_t> qDims;
    std::vector<int64_t> kvDims;
    bool applicable;
};

class TestHipFlash2DimRange : public ::testing::TestWithParam<Flash2DimRangeCase>
{
protected:
    Handle _handle;
    HipFlash2FwdPlanBuilder _builder;
};

TEST_P(TestHipFlash2DimRange, IsApplicableMatchesTheDimensionRange)
{
    SKIP_IF_NO_DEVICES();
    const auto arch = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(arch != "gfx942")
    {
        GTEST_SKIP();
    }

    const auto& param = GetParam();
    // One stride set for every tensor: the head-dim stride is 1 as the kernel
    // requires, and nothing else ties the strides to the dims.
    const std::vector<int64_t> strides{32768, 32768, 128, 1};
    const std::vector<int64_t> oDims{
        param.qDims[0], param.qDims[1], param.qDims[2], param.kvDims[3]};
    auto builder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        param.qDims,
        strides,
        param.kvDims,
        strides,
        param.kvDims,
        strides,
        oDims,
        strides,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    EXPECT_EQ(_builder.isApplicable(_handle, graph), param.applicable);
}

// INT32_MAX itself is not reachable for seq_len_q (the engine also requires a
// multiple of 64), so its case sits on the largest multiple of 64 below it.
// seq_len_kv stops one KV tile lower still: K_FLASH2_MAX_SEQ_LEN_KV is the
// largest length whose tile count the kernels can form without overflowing int.
constexpr int64_t K_I32_DIM_LIMIT = static_cast<int64_t>(INT32_MAX);
constexpr int64_t K_MAX_SEQ_MULTIPLE_OF_64 = (K_I32_DIM_LIMIT / 64) * 64;
constexpr int64_t K_WRAPS_TO_4096 = (int64_t{1} << 32) + 4096;
constexpr int64_t K_WRAPS_TO_1 = (int64_t{1} << 32) + 1;

INSTANTIATE_TEST_SUITE_P(
    DimensionRange,
    TestHipFlash2DimRange,
    ::testing::Values(
        Flash2DimRangeCase{
            "ordinary shape (control)", {1, 32, 4096, 128}, {1, 32, 4096, 128}, true},
        Flash2DimRangeCase{"batch at INT32_MAX",
                           {K_I32_DIM_LIMIT, 32, 4096, 128},
                           {K_I32_DIM_LIMIT, 32, 4096, 128},
                           true},
        Flash2DimRangeCase{"batch one past INT32_MAX",
                           {K_I32_DIM_LIMIT + 1, 32, 4096, 128},
                           {K_I32_DIM_LIMIT + 1, 32, 4096, 128},
                           false},
        // Batch has no other guard at all, so a batch that truncates to a legal
        // value would otherwise be dispatched as that value.
        Flash2DimRangeCase{"batch wraps to 1",
                           {K_WRAPS_TO_1, 32, 4096, 128},
                           {K_WRAPS_TO_1, 32, 4096, 128},
                           false},
        Flash2DimRangeCase{
            "num_heads_q at INT32_MAX", {1, K_I32_DIM_LIMIT, 4096, 128}, {1, 1, 4096, 128}, true},
        Flash2DimRangeCase{"num_heads_q one past INT32_MAX",
                           {1, K_I32_DIM_LIMIT + 1, 4096, 128},
                           {1, 1, 4096, 128},
                           false},
        Flash2DimRangeCase{"seq_len_q at its largest accepted value",
                           {1, 32, K_MAX_SEQ_MULTIPLE_OF_64, 128},
                           {1, 32, 4096, 128},
                           true},
        Flash2DimRangeCase{
            "seq_len_q wraps to 4096", {1, 32, K_WRAPS_TO_4096, 128}, {1, 32, 4096, 128}, false},
        Flash2DimRangeCase{"seq_len_kv at its largest accepted value",
                           {1, 32, 4096, 128},
                           {1, 32, K_FLASH2_MAX_SEQ_LEN_KV, 128},
                           true},
        Flash2DimRangeCase{"seq_len_kv one past its largest accepted value",
                           {1, 32, 4096, 128},
                           {1, 32, K_FLASH2_MAX_SEQ_LEN_KV + 1, 128},
                           false},
        Flash2DimRangeCase{"seq_len_kv one past INT32_MAX",
                           {1, 32, 4096, 128},
                           {1, 32, K_I32_DIM_LIMIT + 1, 128},
                           false},
        Flash2DimRangeCase{"head_dim wraps to 128",
                           {1, 32, 4096, (int64_t{1} << 32) + 128},
                           {1, 32, 4096, (int64_t{1} << 32) + 128},
                           false}),
    [](const ::testing::TestParamInfo<Flash2DimRangeCase>& info) {
        std::string name = info.param.label;
        for(auto& c : name)
        {
            if(std::isalnum(static_cast<unsigned char>(c)) == 0)
            {
                c = '_';
            }
        }
        return name;
    });

TEST_F(TestHipFlash2FwdPlanBuilder, RejectsSinkToken)
{
    SKIP_IF_NO_DEVICES();
    const auto arch = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(arch != "gfx942")
    {
        GTEST_SKIP();
    }

    // The same graph is accepted without the sink and declined with it: the flash2 kernel
    // cannot add the sink logit to the softmax denominator.
    const std::vector<int64_t> dims{1, 32, 2048, 128};
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    auto base = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        dims,
        strides,
        dims,
        strides,
        dims,
        strides,
        dims,
        strides,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);
    auto withSinkBuilder = asm_sdpa_engine::addSinkToken(base);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper withoutSink(
        base.GetBufferPointer(), base.GetSize());
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper withSink(
        withSinkBuilder.GetBufferPointer(), withSinkBuilder.GetSize());
    EXPECT_TRUE(_builder.isApplicable(_handle, withoutSink));
    EXPECT_FALSE(_builder.isApplicable(_handle, withSink));
}

using asm_sdpa_engine::DeprecatedCausalMaskCase;
using asm_sdpa_engine::plan_utils::MaskType;

class TestHipFlash2ExtractParamsCausal : public ::testing::TestWithParam<DeprecatedCausalMaskCase>
{
};

TEST_P(TestHipFlash2ExtractParamsCausal, SetsCausalOnlyForTopLeftCausal)
{
    const auto& param = GetParam();
    auto builder = asm_sdpa_engine::createSdpaFwdGraphWithMask(param.causalMask,
                                                               param.causalMaskBottomRight,
                                                               param.leftBound,
                                                               param.rightBound,
                                                               param.alignment);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const Handle handle;

    if(!param.expected.has_value())
    {
        EXPECT_THROW(HipFlash2FwdPlanBuilder::extractParams(handle, graph),
                     hipdnn_plugin_sdk::HipdnnPluginException);
        return;
    }
    const auto params = HipFlash2FwdPlanBuilder::extractParams(handle, graph);
    EXPECT_EQ(params.causal, *param.expected == MaskType::TOP_LEFT_CAUSAL);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestHipFlash2ExtractParamsCausal,
                         ::testing::ValuesIn(asm_sdpa_engine::deprecatedCausalMaskCases()),
                         asm_sdpa_engine::deprecatedCausalMaskCaseName);

// The mask graph is 256x256. left_bound follows cuDNN (L keeps the L keys ending at
// the diagonal), so a top-left left bound of 256 covers the whole sequence and is
// plain causal, while 255 is a genuine sliding window (the last query drops key 0).
bool extractCausalForTopLeftWindow(int64_t leftBound)
{
    auto builder = asm_sdpa_engine::createSdpaFwdGraphWithMask(
        false,
        false,
        leftBound,
        0,
        hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment::TOP_LEFT);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const Handle handle;
    return HipFlash2FwdPlanBuilder::extractParams(handle, graph).causal;
}

TEST(TestHipFlash2ExtractParamsWindow, CoveringLeftBoundIsCausal)
{
    EXPECT_TRUE(extractCausalForTopLeftWindow(256));
    EXPECT_TRUE(extractCausalForTopLeftWindow(4294967298));
}

TEST(TestHipFlash2ExtractParamsWindow, GenuineWindowIsNotCausal)
{
    EXPECT_FALSE(extractCausalForTopLeftWindow(255));
}

} // namespace
} // namespace hip_flash2_engine
