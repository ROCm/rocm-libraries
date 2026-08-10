// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <optional>
#include <type_traits>

#include <gtest/gtest.h>

#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/plans/resample/ResampleBwdPlan.hpp"
#include "mocks/MockCompiledProgram.hpp"
#include "mocks/MockKernelCompiler.hpp"
#include "mocks/MockRunnableKernel.hpp"

#include "../TestPlanCommon.hpp"

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::resample;

namespace
{

std::pair<flatbuffers::FlatBufferBuilder, ResampleBwdPlan> createPlanFromGraph()
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_ResampleBwdAttributes();

    ResampleBwdParams params(attr, graph.getTensorMap(), node.compute_data_type());
    return {std::move(builder), ResampleBwdPlan{std::move(params)}};
}

} // namespace

// ============================================================================
// ResampleBwdParams - construction from valid graph data
// ============================================================================

TEST(TestResampleBwdParams, ConstructsFromSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_ResampleBwdAttributes();

    EXPECT_NO_THROW(
        const ResampleBwdParams params(attr, graph.getTensorMap(), node.compute_data_type()));
}

TEST(TestResampleBwdParams, HasCorrectTensorPointers)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph(false);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_ResampleBwdAttributes();

    const ResampleBwdParams params(attr, graph.getTensorMap(), node.compute_data_type());

    EXPECT_NE(params.dy(), nullptr);
    EXPECT_NE(params.dx(), nullptr);
    EXPECT_EQ(params.index(), nullptr);
    EXPECT_EQ(params.prePadding(), std::vector<int64_t>({0, 0}));
    EXPECT_EQ(params.postPadding(), std::vector<int64_t>({0, 0}));
    EXPECT_EQ(params.stride(), std::vector<int64_t>({2, 2}));
    EXPECT_EQ(params.window(), std::vector<int64_t>({2, 2}));
    EXPECT_EQ(params.resampleMode(), hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL);
    EXPECT_EQ(params.paddingMode(), hipdnn_flatbuffers_sdk::data_objects::PaddingMode::ZERO_PAD);
}

TEST(TestResampleBwdParams, TensorPointersMatchExpectedUids)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_ResampleBwdAttributes();

    const ResampleBwdParams params(attr, graph.getTensorMap(), node.compute_data_type());

    EXPECT_EQ(params.dy()->uid(), attr.dy_tensor_uid());
    EXPECT_EQ(params.dx()->uid(), attr.dx_tensor_uid());
}

TEST(TestResampleBwdParams, HasOptionalIndexTensor)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph(true);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_ResampleBwdAttributes();

    const ResampleBwdParams params(attr, graph.getTensorMap(), node.compute_data_type());

    ASSERT_NE(params.index(), nullptr);
    EXPECT_EQ(params.index()->uid(), attr.index_tensor_uid().value());
}

TEST(TestResampleBwdParams, IsMoveConstructible)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_ResampleBwdAttributes();

    ResampleBwdParams params(attr, graph.getTensorMap(), node.compute_data_type());
    const ResampleBwdParams moved(std::move(params));

    EXPECT_NE(moved.dy(), nullptr);
    EXPECT_NE(moved.dx(), nullptr);
}

TEST(TestResampleBwdParams, IsNotCopyConstructible)
{
    EXPECT_FALSE(std::is_copy_constructible_v<ResampleBwdParams>);
}

// ============================================================================
// ResampleBwdPlan - basic behavior
// ============================================================================

TEST(TestResampleBwdPlan, ExecuteWithoutCompileThrows)
{
    auto [fbb, plan] = createPlanFromGraph();
    const Handle handle;
    EXPECT_THROW(plan.execute(handle, nullptr, 0), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestResampleBwdPlan, GetWorkspaceSizeReturnsZero)
{
    auto [fbb, plan] = createPlanFromGraph();
    const Handle handle;
    EXPECT_EQ(plan.getWorkspaceSize(handle), 0u);
}

TEST(TestResampleBwdPlan, IsMoveConstructible)
{
    auto [fbb, plan] = createPlanFromGraph();
    const ResampleBwdPlan moved(std::move(plan));
    const Handle handle;
    EXPECT_EQ(moved.getWorkspaceSize(handle), 0u);
}

TEST(TestResampleBwdPlan, IsNotCopyConstructible)
{
    EXPECT_FALSE(std::is_copy_constructible_v<ResampleBwdPlan>);
}

// ============================================================================
// ResampleBwdPlan - compile
// ============================================================================
