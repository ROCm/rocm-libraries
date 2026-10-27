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

#include <memory>
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

// The mask graph is 256x256, so a top-left left bound of 255 covers the whole
// sequence and is plain causal, while 254 is a genuine sliding window.
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
    EXPECT_TRUE(extractCausalForTopLeftWindow(255));
    EXPECT_TRUE(extractCausalForTopLeftWindow(4294967298));
}

TEST(TestHipFlash2ExtractParamsWindow, GenuineWindowIsNotCausal)
{
    EXPECT_FALSE(extractCausalForTopLeftWindow(254));
}

} // namespace
} // namespace hip_flash2_engine
