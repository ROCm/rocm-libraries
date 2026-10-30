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
#include "tests/utilities/RaggedGraphTestUtils.hpp"

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include <cctype>
#include <memory>
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

TEST_F(TestHipFlash2FwdPlanBuilder, AcceptsHalfMhaCausal)
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

TEST_F(TestHipFlash2FwdPlanBuilder, AcceptsHalfMhaNonCausal)
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

TEST_F(TestHipFlash2FwdPlanBuilder, AcceptsHalfHeadDim64)
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
TEST_F(TestHipFlash2FwdPlanBuilder, AcceptsHalfGqa)
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

// Bug caught: a ragged SDPA graph accepted and its packed buffers read through dense strides.
// The unmodified graph is accepted, and the same graph is declined with any one of its
// tensors ragged.
TEST_F(TestHipFlash2FwdPlanBuilder, RejectsRaggedTensors)
{
    SKIP_IF_NO_DEVICES();
    const auto arch = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(arch != "gfx942")
    {
        GTEST_SKIP();
    }

    const std::vector<int64_t> dims{1, 32, 2048, 128};
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    const auto base = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        dims,
        strides,
        dims,
        strides,
        dims,
        strides,
        dims,
        strides,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);

    hip_kernel_provider::test_utils::expectRaggedGraphsDeclined(
        base, [this](const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph) {
            return _builder.isApplicable(_handle, graph);
        });
}

// Re-serializes a graph with the declared alignment (bytes) of the tensor `uid` replaced.
flatbuffers::DetachedBuffer
    withDeclaredAlignment(const void* graphBuffer, int64_t uid, int64_t alignment)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    const auto graphT = std::unique_ptr<GraphT>(GetGraph(graphBuffer)->UnPack());
    for(auto& tensor : graphT->tensors)
    {
        if(tensor->uid == uid)
        {
            tensor->alignment = alignment;
        }
    }

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(CreateGraph(builder, graphT.get()));
    return builder.Release();
}

// A tensor of the graph and the alignment (bytes) its kernel needs from the declared value.
struct AlignmentRole
{
    const char* name;
    int64_t uid;
    int64_t needBytes;
};

// Q, K, V and O of the fp16 forward graph used by the declared-alignment tests.
std::vector<AlignmentRole> flash2AlignmentRoles(const flatbuffers::FlatBufferBuilder& graph)
{
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper(
        graph.GetBufferPointer(), graph.GetSize());
    const auto& attrs = wrapper.getNodeWrapper(0)
                            .attributesAs<hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes>();
    // Needs: Q 8 bytes (rocWMMA fragment load of 4 fp16 per lane), K and V 2 bytes (the kernel
    // falls back to scalar fp16 loads unless its own run-time 16-byte check passes), O 8 bytes.
    return {{"q", attrs.q_tensor_uid(), 8},
            {"k", attrs.k_tensor_uid(), 2},
            {"v", attrs.v_tensor_uid(), 2},
            {"o", attrs.o_tensor_uid(), 8}};
}

// Bug caught: a tensor declaring less alignment than the kernel's loads and stores need is
// accepted (the backend then rejects the pointer at execute, or the kernel reads misaligned), a
// declared alignment equal to the need or a larger multiple is wrongly declined, or a declared
// alignment that is not a multiple of the need passes a `>=` style comparison. Each tensor is
// declared at half its need, at its need, at twice its need, at 1.5x its need, and at 1 and 0
// bytes, one tensor at a time; the default-16 graph stays accepted.
TEST_F(TestHipFlash2FwdPlanBuilder, DeclaredAlignmentOfEachTensorIsHonored)
{
    SKIP_IF_NO_DEVICES();
    const auto arch = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(arch != "gfx942")
    {
        GTEST_SKIP();
    }

    const std::vector<int64_t> dims{1, 32, 2048, 128};
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    const auto valid = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        dims,
        strides,
        dims,
        strides,
        dims,
        strides,
        dims,
        strides,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);

    const auto accepts = [&](int64_t uid, int64_t alignment) {
        const auto buffer = withDeclaredAlignment(valid.GetBufferPointer(), uid, alignment);
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(buffer.data(),
                                                                               buffer.size());
        return _builder.isApplicable(_handle, graph);
    };

    {
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper baseline(
            valid.GetBufferPointer(), valid.GetSize());
        EXPECT_TRUE(_builder.isApplicable(_handle, baseline))
            << "the graph with default alignment must be accepted";
    }

    for(const auto& role : flash2AlignmentRoles(valid))
    {
        EXPECT_TRUE(accepts(role.uid, role.needBytes))
            << role.name << " declaring exactly its need (" << role.needBytes
            << " bytes) was declined";
        EXPECT_TRUE(accepts(role.uid, role.needBytes * 2))
            << role.name << " declaring twice its need was declined";
        EXPECT_FALSE(accepts(role.uid, role.needBytes / 2))
            << role.name << " declaring half its need was accepted";
        EXPECT_FALSE(accepts(role.uid, role.needBytes + role.needBytes / 2))
            << role.name << " declaring a non-multiple of its need was accepted";
        EXPECT_FALSE(accepts(role.uid, 1)) << role.name << " declaring 1 byte was accepted";
        EXPECT_FALSE(accepts(role.uid, 0)) << role.name << " declaring 0 bytes was accepted";
    }
}

// Bug caught: K and V are held to a 16-byte vector-load need although the kernel takes a scalar
// path when they are not 16-byte aligned, so a graph declaring K and V at the fp16 element size
// (with Q and O at their needs) is declined instead of served by the scalar path.
TEST_F(TestHipFlash2FwdPlanBuilder, AcceptsKeyValueDeclaredAtElementSize)
{
    SKIP_IF_NO_DEVICES();
    const auto arch = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(arch != "gfx942")
    {
        GTEST_SKIP();
    }

    const std::vector<int64_t> dims{1, 32, 2048, 128};
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    const auto valid = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        dims,
        strides,
        dims,
        strides,
        dims,
        strides,
        dims,
        strides,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);

    const auto roles = flash2AlignmentRoles(valid);
    const auto kNarrowed = withDeclaredAlignment(valid.GetBufferPointer(), roles[1].uid, 2);
    const auto kvNarrowed = withDeclaredAlignment(kNarrowed.data(), roles[2].uid, 2);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(kvNarrowed.data(),
                                                                           kvNarrowed.size());
    EXPECT_TRUE(_builder.isApplicable(_handle, graph));
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
