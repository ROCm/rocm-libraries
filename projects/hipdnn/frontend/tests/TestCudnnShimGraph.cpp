// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Source-compatibility coverage for the cuDNN-shaped graph wrapper. These tests
// include only the shim umbrella so missing public aliases or overloads fail at
// compile time when HIPDNN_ENABLE_CUDNN_COMPATIBILITY is enabled.
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <gtest/gtest.h>

#include "CudnnShimTestSupport.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace cudnn_frontend = hipdnn_frontend::compatibility::cudnn_frontend;

static_assert(std::is_move_constructible_v<cudnn_frontend::graph::Graph>);
static_assert(!std::is_copy_constructible_v<cudnn_frontend::graph::Graph>);
static_assert(std::is_same_v<cudnn_frontend::graph::TensorAttributes,
                             hipdnn_frontend::graph::TensorAttributes>);
static_assert(std::is_same_v<cudnn_frontend::graph::Tensor_attributes,
                             cudnn_frontend::graph::TensorAttributes>);

namespace
{
namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

TEST(TestCudnnShimGraph, DefaultConstructsAndValidatesEmptyGraph)
{
    fe::graph::Graph graph;

    EXPECT_TRUE(graph.validate().is_good());
    EXPECT_TRUE(graph.build_operation_graph(nullptr).is_good());
    EXPECT_TRUE(graph.build(nullptr).is_good());
    EXPECT_EQ(graph.get_execution_plan_count(), 0);
}

TEST(TestCudnnShimGraph, InvalidOwnedTensorFailsValidate)
{
    fe::graph::Graph graph;
    graph.tensor(fe::graph::Tensor_attributes{}
                     .set_dim({1})
                     .set_data_type(fe::DataType_t::FLOAT)
                     .set_uid(1));

    auto error = graph.validate();
    EXPECT_TRUE(error.is_bad());
    EXPECT_EQ(error.get_code(), fe::error_code_t::INVALID_VALUE);
}

TEST(TestCudnnShimGraph, RecordedSetterErrorSurfacesOnValidate)
{
    fe::graph::Graph graph;
    graph.set_sm_count(1).set_name("still chains");

    auto error = graph.validate();
    EXPECT_TRUE(error.is_bad());
    EXPECT_EQ(error.get_code(), fe::error_code_t::INVALID_VALUE);
    EXPECT_EQ(graph.get_name(), "still chains");
}

TEST(TestCudnnShimGraph, BenignConfigSettersDoNotFailValidation)
{
    fe::graph::Graph graph;
    graph.set_dynamic_shape_enabled(true).set_kernel_cache(nullptr);

    EXPECT_TRUE(graph.validate().is_good());
}

TEST(TestCudnnShimGraph, ScalarTensorFactoriesCompileAndValidateRuntimeParams)
{
    fe::graph::Graph graph;
    graph.tensor(1.0F, fe::graph::ScalarType::RUNTIME_PARAM);
    graph.tensor(fe::graph::half{1.0F}, fe::graph::ScalarType::RUNTIME_PARAM);
    graph.tensor(fe::graph::nv_bfloat16{1.0F}, fe::graph::ScalarType::RUNTIME_PARAM);
    graph.tensor(int32_t{1}, fe::graph::ScalarType::RUNTIME_PARAM);
    graph.tensor(int64_t{1}, fe::graph::ScalarType::RUNTIME_PARAM);
    graph.tensor(1.0, fe::graph::ScalarType::RUNTIME_PARAM);

    EXPECT_TRUE(graph.validate().is_good());
}

TEST(TestCudnnShimGraph, TensorLikeAndQueryByUid)
{
    fe::graph::Graph graph;
    auto tensor = graph.tensor(fe::graph::Tensor_attributes{}
                                   .set_dim({2, 3})
                                   .set_stride({3, 1})
                                   .set_data_type(fe::DataType_t::FLOAT)
                                   .set_uid(7));
    auto tensorLike = graph.tensor_like(tensor, "copy");
    tensorLike->set_uid(8);

    fe::graph::Tensor_attributes queried;
    ASSERT_TRUE(graph.query_tensor_attributes_of_uid(8, queried).is_good());
    EXPECT_EQ(queried.get_uid(), 8);
    EXPECT_EQ(queried.get_name(), "copy");
    EXPECT_EQ(queried.get_dim(), std::vector<int64_t>({2, 3}));
    EXPECT_EQ(queried.get_stride(), std::vector<int64_t>({3, 1}));

    EXPECT_TRUE(graph.query_tensor_attributes_of_uid(999, queried).is_bad());
}

TEST(TestCudnnShimGraph, RequiredGraphSurfaceCompiles)
{
    fe::graph::Graph graph;
    const std::vector<fe::HeurMode_t> modes{fe::HeurMode_t::FALLBACK};

    EXPECT_TRUE(graph.build_operation_graph(nullptr).is_good());
    EXPECT_TRUE(graph.build_operation_graph().is_good());
    EXPECT_TRUE(graph.create_execution_plans(modes).is_good());
    EXPECT_TRUE(graph.check_support().is_good());
    EXPECT_TRUE(graph.check_support(nullptr).is_good());
    EXPECT_TRUE(graph.build_plans().is_good());
    EXPECT_TRUE(graph.build_plans(nullptr).is_good());
    EXPECT_TRUE(graph.build_plan_at_index(0).is_bad());
    EXPECT_TRUE(graph.build_plan_at_index(nullptr, 0).is_bad());
    EXPECT_EQ(graph.get_execution_plan_count(), 0);
    EXPECT_TRUE(graph.build(nullptr, modes).is_good());
    EXPECT_TRUE(graph.build(modes).is_good());

    int64_t workspaceSize = -1;
    EXPECT_TRUE(graph.get_workspace_size(workspaceSize).is_good());
    EXPECT_EQ(workspaceSize, 0);
    EXPECT_EQ(graph.get_workspace_size(), 0);

    // Serializing an operation-graph-less graph is unsupported now that the
    // custom empty-graph blob format is gone; the calls must still compile.
    std::vector<uint8_t> data;
    EXPECT_TRUE(graph.serialize(data).is_bad());
    EXPECT_TRUE(graph.deserialize(data).is_bad());
    EXPECT_TRUE(graph.deserialize(nullptr, data).is_bad());

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> tensorMap;
    std::unordered_map<int64_t, void*> uidMap;
    const std::vector<int64_t> overrideUids;
    const std::vector<std::vector<int64_t>> overrideShapes;
    const std::vector<std::vector<int64_t>> overrideStrides;
    void** sortedUserPtrs = nullptr;

    EXPECT_TRUE(graph.execute(nullptr, tensorMap, nullptr).is_bad());
    EXPECT_TRUE(graph.execute(nullptr, uidMap, nullptr).is_bad());
    EXPECT_TRUE(
        graph.execute(nullptr, uidMap, nullptr, overrideUids, overrideShapes, overrideStrides)
            .is_bad());
    EXPECT_TRUE(graph.execute(nullptr, sortedUserPtrs, 0, nullptr).is_bad());

    auto tensor = graph.tensor(fe::graph::Tensor_attributes{}
                                   .set_dim({1})
                                   .set_stride({1})
                                   .set_data_type(fe::DataType_t::FLOAT)
                                   .set_uid(10));
    auto tensorLike = graph.tensor_like(tensor);
    tensorLike->set_uid(11);
    fe::graph::Tensor_attributes queried;
    EXPECT_TRUE(graph.query_tensor_attributes_of_uid(10, queried).is_good());
}

TEST(TestCudnnShimGraph, ExecuteOnEmptyGraphFails)
{
    const fe::graph::Graph graph;
    std::unordered_map<int64_t, void*> uidMap;

    EXPECT_TRUE(graph.execute(nullptr, uidMap, nullptr).is_bad());
}

TEST(TestCudnnShimGraph, PoisonedGraphSurfacesRecordedErrorOnExecuteAndWorkspace)
{
    // A setter that records an error must surface it ahead of the generic
    // "no execution plan" result on execute()/get_workspace_size().
    fe::graph::Graph graph;
    graph.set_sm_count(1);

    std::unordered_map<int64_t, void*> uidMap;
    auto execError = graph.execute(nullptr, uidMap, nullptr);
    EXPECT_TRUE(execError.is_bad());
    EXPECT_NE(execError.get_message().find("SM count"), std::string::npos);

    int64_t workspaceSize = -1;
    auto wsError = graph.get_workspace_size(workspaceSize);
    EXPECT_TRUE(wsError.is_bad());
    EXPECT_NE(wsError.get_message().find("SM count"), std::string::npos);
}

// Serialize contract: the custom empty-graph blob format is gone. An
// operation-graph-less graph must refuse to serialize, with the exact message,
// on the const overload...
TEST(TestCudnnShimGraph, SerializeEmptyGraphConstIsUnsupported)
{
    const fe::graph::Graph graph;
    std::vector<uint8_t> data;

    auto error = graph.serialize(data);
    ASSERT_TRUE(error.is_bad());
    EXPECT_EQ(error.get_message(),
              "Serializing a graph without a compiled operation graph is unsupported");
    EXPECT_TRUE(data.empty());
}

// ...and on the non-const overload, which runs validate() first but still lands
// on the same unsupported result for an empty graph.
TEST(TestCudnnShimGraph, SerializeEmptyGraphNonConstIsUnsupported)
{
    fe::graph::Graph graph;
    std::vector<uint8_t> data;

    auto error = graph.serialize(data);
    ASSERT_TRUE(error.is_bad());
    EXPECT_EQ(error.get_message(),
              "Serializing a graph without a compiled operation graph is unsupported");
}

// deserialize() of arbitrary bytes forwards to hipDNN; with no backend installed
// it fails, but crucially NOT via the shim's serialize-unsupported path — proving
// the old shim blob branch is gone and the call reaches the native graph.
TEST(TestCudnnShimGraph, DeserializeArbitraryBytesForwardsToNative)
{
    fe::graph::Graph graph;
    const std::vector<uint8_t> bytes{0x01, 0x02, 0x03, 0x04};

    auto error = graph.deserialize(bytes);
    ASSERT_TRUE(error.is_bad());
    EXPECT_EQ(error.get_message().find("without a compiled operation graph"), std::string::npos);
}

// State machine (M4 fix): a fresh empty graph has no execution plan, so the
// error_t& workspace query ERRORS before any build — it must NOT silently report
// 0. After the empty-path build_operation_graph(), the query succeeds with ws==0.
TEST(TestCudnnShimGraph, WorkspaceQueryErrorsBeforeBuildThenZeroAfter)
{
    fe::graph::Graph graph;

    EXPECT_EQ(graph.get_execution_plan_count(), 0);

    int64_t workspaceSize = -1;
    auto preBuild = graph.get_workspace_size(workspaceSize);
    ASSERT_TRUE(preBuild.is_bad());
    EXPECT_NE(preBuild.get_message().find("no compiled execution plan"), std::string::npos);
    EXPECT_EQ(workspaceSize, -1); // untouched on the error path

    ASSERT_TRUE(graph.build_operation_graph(nullptr).is_good());

    workspaceSize = -1;
    auto postBuild = graph.get_workspace_size(workspaceSize);
    EXPECT_TRUE(postBuild.is_good());
    EXPECT_EQ(workspaceSize, 0);
}

// build_plan_at_index message split: on an empty/no-op-graph graph the guard is
// "no compiled execution plan" (noExecutionPlanError), NOT "index is invalid" —
// even for a non-zero index. The Native-graph "index is invalid" counterpart is
// covered in TestCudnnShimGraphSDPA.
TEST(TestCudnnShimGraph, BuildPlanAtIndexOnEmptyGraphReportsNoPlan)
{
    fe::graph::Graph graph;

    auto zero = graph.build_plan_at_index(0);
    ASSERT_TRUE(zero.is_bad());
    EXPECT_NE(zero.get_message().find("no compiled execution plan"), std::string::npos);

    auto nonZero = graph.build_plan_at_index(5);
    ASSERT_TRUE(nonZero.is_bad());
    EXPECT_NE(nonZero.get_message().find("no compiled execution plan"), std::string::npos);
    EXPECT_EQ(nonZero.get_message().find("index is invalid"), std::string::npos);
}

// Deferred SM/device errors surface at validate() (extends the execute/workspace
// coverage): set_sm_count and set_device_properties each poison the graph and the
// SM/device message must appear from validate().
TEST(TestCudnnShimGraph, ValidateSurfacesRecordedSmCountError)
{
    fe::graph::Graph graph;
    graph.set_sm_count(1);

    auto error = graph.validate();
    ASSERT_TRUE(error.is_bad());
    EXPECT_EQ(error.get_code(), fe::error_code_t::INVALID_VALUE);
    EXPECT_NE(error.get_message().find("SM count"), std::string::npos);
}

TEST(TestCudnnShimGraph, ValidateSurfacesRecordedDevicePropertiesError)
{
    fe::graph::Graph graph;
    graph.set_device_properties(nullptr);

    auto error = graph.validate();
    ASSERT_TRUE(error.is_bad());
    EXPECT_EQ(error.get_code(), fe::error_code_t::INVALID_VALUE);
    EXPECT_NE(error.get_message().find("Device properties"), std::string::npos);
}

// Gaps intentionally not tested host-only (documented, not faked):
//  - mma_core_mode default (HALF when unset): the shim moves the attribute into
//    the native graph where the packer consumes it; there is no host-observable
//    getter and the mock fixture does not decode finalized SDPA descriptor
//    attributes, so a test here would assert nothing.
//  - native-deserialize plan-count (handle -> count 1, null handle -> count 0):
//    reaching the is_good() branch needs a real native serialized blob plus the
//    full backendGetAttribute graph-reconstruction mock; that path (and its
//    resulting plan state) is exercised in TestGraph.cpp and is not cheaply
//    reachable through the shim surface without duplicating that machinery.

} // namespace
