// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdio>
#include <gtest/gtest.h>
#include <limits>
#include <ostream>
#include <string>

#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"
#include "engines/hip_mlops_engine/plans/resample/ResampleBwdPlanBuilder.hpp"
#include "mocks/MockCompiledProgram.hpp"
#include "mocks/MockDevicePropertyProvider.hpp"
#include "mocks/MockKernelCompiler.hpp"
#include "mocks/MockRunnableKernel.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/resample_bwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>

using hipdnn_test_sdk::utilities::MockEngineConfig;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hip_kernel_provider;
using namespace hip_kernel_provider::resample;

namespace
{

// dx is [1, 1, height, width] and dy is [1, 1, outHeight, outWidth].
flatbuffers::FlatBufferBuilder createResampleBwdGraph(int64_t height,
                                                      int64_t width,
                                                      int64_t outHeight,
                                                      int64_t outWidth,
                                                      const std::vector<int64_t>& prePadding,
                                                      const std::vector<int64_t>& postPadding,
                                                      const std::vector<int64_t>& stride,
                                                      const std::vector<int64_t>& window)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int64_t> dims = {1, 1, height, width};
    const std::vector<int64_t> strides = {height * width, height * width, width, 1};
    const std::vector<int64_t> outDims = {1, 1, outHeight, outWidth};
    const std::vector<int64_t> outStrides
        = {outHeight * outWidth, outHeight * outWidth, outWidth, 1};

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        1,
        "dy",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &outStrides,
        &outDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "dx", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &strides, &dims));

    const auto resampleAttr
        = hipdnn_flatbuffers_sdk::data_objects::CreateResampleBwdAttributesDirect(
            builder,
            1,
            2,
            ::flatbuffers::nullopt,
            &prePadding,
            &postPadding,
            &stride,
            &window,
            hipdnn_flatbuffers_sdk::data_objects::ResampleMode::AVGPOOL_EXCLUDE_PADDING,
            hipdnn_flatbuffers_sdk::data_objects::PaddingMode::ZERO_PAD);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "resample_bwd",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ResampleBwdAttributes,
        resampleAttr.Union()));

    builder.Finish(hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &tensorAttributes,
        &nodes));
    return builder;
}

// A 1x1 pooling window with stride 1, so dx has the dims of dy: [1, 1, height, width].
flatbuffers::FlatBufferBuilder createResampleBwdGraphWithGradientSize(int64_t height, int64_t width)
{
    return createResampleBwdGraph(height, width, height, width, {0, 0}, {0, 0}, {1, 1}, {1, 1});
}

} // namespace

class TestResampleBwdPlanBuilder : public ::testing::Test
{
protected:
    MockKernelCompiler _mockKernelCompiler;
    MockDevicePropertyProvider _mockDevicePropertyProvider;
    ResampleBwdPlanBuilder _planBuilder{_mockKernelCompiler, _mockDevicePropertyProvider};
    Handle _dummyHandle;
    MockEngineConfig _mockEngineConfig;

    void setupMockCompileChain()
    {
        hipDeviceProp_t deviceProps = {};
        deviceProps.multiProcessorCount = 60;
        deviceProps.warpSize = 64;
        std::snprintf(deviceProps.gcnArchName, sizeof(deviceProps.gcnArchName), "%s", "gfx942");

        EXPECT_CALL(_mockDevicePropertyProvider, getDeviceProperties())
            .WillOnce(::testing::Return(deviceProps));

        auto mockKernel = std::make_unique<MockRunnableKernel>();
        EXPECT_CALL(*mockKernel, setBlockSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
        EXPECT_CALL(*mockKernel, setGridSize(::testing::_, ::testing::_, ::testing::_)).Times(1);

        auto mockProgram = std::make_unique<MockCompiledProgram>();
        EXPECT_CALL(*mockProgram, getKernel(::testing::_))
            .WillOnce(::testing::Return(::testing::ByMove(std::move(mockKernel))));

        EXPECT_CALL(_mockKernelCompiler, compile(::testing::_, ::testing::_))
            .WillOnce(::testing::Return(::testing::ByMove(std::move(mockProgram))));
    }
};

// ============================================================================
// isApplicable - valid graphs
// ============================================================================

TEST_F(TestResampleBwdPlanBuilder, IsApplicableReturnsTrueForValidGraph)
{
    const auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestResampleBwdPlanBuilder, IsApplicableReturnsTrueWithoutOptionalAttributes)
{
    const auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph(
        false, ResampleMode::AVGPOOL_EXCLUDE_PADDING);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_dummyHandle, graph));
}

// 16777215 * 256 = 2^32 - 256 dx elements is 2^24 - 1 blocks of 256 threads.
TEST_F(TestResampleBwdPlanBuilder, IsApplicableReturnsTrueForGradientAtThreadLimit)
{
    const auto builder = createResampleBwdGraphWithGradientSize(16777215, 256);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_planBuilder.isApplicable(_dummyHandle, graph));
}

// 401 * 10710641 = 2^32 - 255 dx elements needs 2^24 blocks, which is 2^32 threads.
TEST_F(TestResampleBwdPlanBuilder, IsApplicableReturnsFalseForGradientOverThreadLimit)
{
    const auto builder = createResampleBwdGraphWithGradientSize(401, 10710641);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

// dy and dx are [1, 1, 1, 1]. A stride of INT64_MAX keeps dy at one element whatever the padding
// and window, so each row isolates one int64_t limit of the shape inference. An overflowing
// padding row sets the output height to 2, the value the unchecked expression wraps to.
TEST_F(TestResampleBwdPlanBuilder, SpatialParametersAtTheInt64Limit)
{
    constexpr int64_t MAX = std::numeric_limits<int64_t>::max();
    constexpr int64_t TWO_31 = int64_t{1} << 31;
    constexpr int64_t TWO_32 = int64_t{1} << 32;
    struct Row
    {
        const char* name;
        std::vector<int64_t> prePadding;
        std::vector<int64_t> postPadding;
        std::vector<int64_t> window;
        int64_t outHeight;
        bool expected;
    };
    for(const auto& row : {
            Row{"pre_padding fills the padded extent", {MAX - 1, 0}, {0, 0}, {1, 1}, 1, true},
            Row{"pre_padding overflows the padded extent", {MAX, 0}, {0, 0}, {1, 1}, 2, false},
            Row{"post_padding fills the padded extent", {0, 0}, {MAX - 1, 0}, {1, 1}, 1, true},
            Row{"post_padding overflows the padded extent", {0, 0}, {MAX, 0}, {1, 1}, 2, false},
            Row{"window product fits",
                {TWO_32 - 2, TWO_31 - 1},
                {0, 0},
                {TWO_32 - 1, TWO_31},
                1,
                true},
            Row{"window product overflows",
                {TWO_32 - 1, TWO_31 - 1},
                {0, 0},
                {TWO_32, TWO_31},
                1,
                false},
        })
    {
        SCOPED_TRACE(row.name);
        const auto builder = createResampleBwdGraph(
            1, 1, row.outHeight, 1, row.prePadding, row.postPadding, {MAX, MAX}, row.window);
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
            builder.GetBufferPointer(), builder.GetSize());
        EXPECT_EQ(_planBuilder.isApplicable(_dummyHandle, graph), row.expected);
    }
}

// ============================================================================
// isApplicable - invalid graphs
// ============================================================================

TEST_F(TestResampleBwdPlanBuilder, IsNotApplicableForNonResampleGraph)
{
    const auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

TEST_F(TestResampleBwdPlanBuilder, IsNotApplicableForNonF32ComputeType)
{
    const auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph();

    // Set the compute data type of the node to HALF
    auto* mutableGraph
        = hipdnn_flatbuffers_sdk::data_objects::GetMutableGraph(builder.GetBufferPointer());
    mutableGraph->mutable_nodes()->GetMutableObject(0)->mutate_compute_data_type(
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

// ============================================================================
// buildPlan - valid graphs
// ============================================================================

TEST_F(TestResampleBwdPlanBuilder, BuildPlanSetsPlanForSingleNodeGraph)
{
    setupMockCompileChain();

    const auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    Context ctx;

    EXPECT_NO_THROW(_planBuilder.buildPlan(_dummyHandle, graph, _mockEngineConfig, ctx));
    EXPECT_TRUE(ctx.hasValidPlan());
}

// ============================================================================
// getMaxWorkspaceSize
// ============================================================================

TEST_F(TestResampleBwdPlanBuilder, GetMaxWorkspaceSizeReturnsZero)
{
    const auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const Settings settings;

    EXPECT_EQ(_planBuilder.getMaxWorkspaceSize(_dummyHandle, graph, settings), 0u);
}

// ============================================================================
// getCustomKnobs
// ============================================================================

TEST_F(TestResampleBwdPlanBuilder, GetCustomKnobsReturnsEmpty)
{
    const auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto knobs = _planBuilder.getCustomKnobs(_dummyHandle, graph);
    EXPECT_TRUE(knobs.empty());
}

// ============================================================================
// Attributes the kernel reads or writes: a graph that changes them is declined
// ============================================================================

namespace
{
struct ResampleBwdMutation
{
    std::string name;
    int64_t virtualUid;
};

std::ostream& operator<<(std::ostream& os, const ResampleBwdMutation& value)
{
    return os << value.name;
}

flatbuffers::FlatBufferBuilder mutatedResampleBwdGraph(int64_t virtualUid)
{
    const auto base = hipdnn_test_sdk::utilities::createValidResampleBwdGraph();
    return hipdnn_test_sdk::utilities::mutateGraph(base, [virtualUid](GraphT& graph) {
        for(auto& tensor : graph.tensors)
        {
            if(tensor->uid == virtualUid)
            {
                tensor->virtual_ = true;
            }
        }
    });
}
} // namespace

class TestResampleBwdDecline : public TestResampleBwdPlanBuilder,
                               public ::testing::WithParamInterface<ResampleBwdMutation>
{
};

TEST_P(TestResampleBwdDecline, VirtualOperandIsNotApplicable)
{
    const auto builder = mutatedResampleBwdGraph(GetParam().virtualUid);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_planBuilder.isApplicable(_dummyHandle, graph));
}

INSTANTIATE_TEST_SUITE_P(Mutations,
                         TestResampleBwdDecline,
                         ::testing::Values(ResampleBwdMutation{"VirtualDy", 1},
                                           ResampleBwdMutation{"VirtualDx", 2},
                                           ResampleBwdMutation{"VirtualIndex", 3}),
                         [](const ::testing::TestParamInfo<ResampleBwdMutation>& info) {
                             return info.param.name;
                         });
