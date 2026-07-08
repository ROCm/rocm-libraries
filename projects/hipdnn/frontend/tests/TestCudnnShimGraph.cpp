// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Source-compatibility coverage for the cuDNN-shaped graph wrapper. These tests
// include only the shim umbrella so missing public aliases or overloads fail at
// compile time when HIPDNN_ENABLE_CUDNN_COMPATIBILITY is enabled.
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
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

TEST(TestCudnnShimGraph, EmptyGraphSerializesAndDeserializes)
{
    fe::graph::Graph graph;
    std::vector<uint8_t> data;

    ASSERT_TRUE(graph.serialize(data).is_good());
    EXPECT_FALSE(data.empty());

    fe::graph::Graph roundTripped;
    EXPECT_TRUE(roundTripped.deserialize(data).is_good());
    EXPECT_TRUE(roundTripped.validate().is_good());
    EXPECT_EQ(roundTripped.get_execution_plan_count(), 0);
}

TEST(TestCudnnShimGraph, SingleTensorSerializesAndDeserializes)
{
    fe::graph::Graph graph;
    graph.tensor(fe::graph::Tensor_attributes{}
                     .set_dim({1})
                     .set_stride({1})
                     .set_data_type(fe::DataType_t::FLOAT)
                     .set_uid(1)
                     .set_output(true));

    std::vector<uint8_t> data;
    ASSERT_TRUE(graph.serialize(data).is_good());

    fe::graph::Graph roundTripped;
    ASSERT_TRUE(roundTripped.deserialize(data).is_good());
    EXPECT_TRUE(roundTripped.validate().is_good());

    fe::graph::Tensor_attributes queried;
    ASSERT_TRUE(roundTripped.query_tensor_attributes_of_uid(1, queried).is_good());
    EXPECT_EQ(queried.get_uid(), 1);
    EXPECT_EQ(queried.get_dim(), std::vector<int64_t>{1});
    EXPECT_EQ(queried.get_stride(), std::vector<int64_t>{1});
    EXPECT_EQ(queried.get_data_type(), fe::DataType_t::FLOAT);
    EXPECT_FALSE(queried.get_is_virtual());
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

    std::vector<uint8_t> data;
    EXPECT_TRUE(graph.serialize(data).is_good());
    EXPECT_TRUE(graph.deserialize(data).is_good());
    EXPECT_TRUE(graph.deserialize(nullptr, data).is_good());

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
    auto scalar = graph.tensor(2.0F, fe::graph::ScalarType::RUNTIME_PARAM);
    static_cast<void>(scalar);
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

} // namespace
