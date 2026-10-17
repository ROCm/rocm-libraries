// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Includes the shim umbrella and the two autotune headers PyTorch includes
// next to it, as a ported consumer does, and pins the v0.x builder surface
// they expose next to the v9 graph API.
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_find_plan.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_get_plan.h>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace cudnn_frontend = hipdnn_frontend::compatibility::cudnn_frontend;

namespace
{

template <typename Alias, typename Target>
constexpr bool IS_ALIAS_OF = std::is_same_v<Alias, Target>;

template <typename Descriptor>
constexpr bool IS_MOVE_ONLY_BACKEND_DESCRIPTOR
    = std::is_base_of_v<cudnn_frontend::BackendDescriptor, Descriptor>
      && !std::is_copy_constructible_v<Descriptor> && std::is_move_constructible_v<Descriptor>;

} // namespace

static_assert(IS_ALIAS_OF<cudnn_frontend::Tensor, cudnn_frontend::Tensor_v8>);
static_assert(IS_ALIAS_OF<cudnn_frontend::TensorBuilder, cudnn_frontend::TensorBuilder_v8>);
static_assert(IS_ALIAS_OF<cudnn_frontend::ConvDesc, cudnn_frontend::ConvDesc_v8>);
static_assert(IS_ALIAS_OF<cudnn_frontend::ConvDescBuilder, cudnn_frontend::ConvDescBuilder_v8>);
static_assert(IS_ALIAS_OF<cudnn_frontend::PointWiseDesc, cudnn_frontend::PointWiseDesc_v8>);
static_assert(
    IS_ALIAS_OF<cudnn_frontend::PointWiseDescBuilder, cudnn_frontend::PointWiseDescBuilder_v8>);
static_assert(IS_ALIAS_OF<cudnn_frontend::PointwiseDesc, cudnn_frontend::PointWiseDesc_v8>);
static_assert(
    IS_ALIAS_OF<cudnn_frontend::PointwiseDescBuilder, cudnn_frontend::PointWiseDescBuilder_v8>);
static_assert(IS_ALIAS_OF<cudnn_frontend::MatMulDesc, cudnn_frontend::MatMulDesc_v8>);
static_assert(IS_ALIAS_OF<cudnn_frontend::MatMulDescBuilder, cudnn_frontend::MatMulDescBuilder_v8>);
static_assert(IS_ALIAS_OF<cudnn_frontend::ReductionDesc, cudnn_frontend::ReductionDesc_v8>);
static_assert(
    IS_ALIAS_OF<cudnn_frontend::ReductionDescBuilder, cudnn_frontend::ReductionDescBuilder_v8>);
static_assert(IS_ALIAS_OF<cudnn_frontend::ResampleDesc, cudnn_frontend::ResampleDesc_v8>);
static_assert(
    IS_ALIAS_OF<cudnn_frontend::ResampleDescBuilder, cudnn_frontend::ResampleDescBuilder_v8>);
static_assert(IS_ALIAS_OF<cudnn_frontend::RngDesc, cudnn_frontend::RngDesc_v8>);
static_assert(IS_ALIAS_OF<cudnn_frontend::RngDescBuilder, cudnn_frontend::RngDescBuilder_v8>);
static_assert(IS_ALIAS_OF<cudnn_frontend::VariantPack, cudnn_frontend::VariantPack_v8>);
static_assert(
    IS_ALIAS_OF<cudnn_frontend::VariantPackBuilder, cudnn_frontend::VariantPackBuilder_v8>);
static_assert(IS_ALIAS_OF<cudnn_frontend::Operation, cudnn_frontend::Operation_v8>);
static_assert(IS_ALIAS_OF<cudnn_frontend::OperationBuilder, cudnn_frontend::OperationBuilder_v8>);
static_assert(IS_ALIAS_OF<cudnn_frontend::OperationGraph, cudnn_frontend::OperationGraph_v8>);
static_assert(
    IS_ALIAS_OF<cudnn_frontend::OperationGraphBuilder, cudnn_frontend::OperationGraphBuilder_v8>);
static_assert(IS_ALIAS_OF<cudnn_frontend::EngineHeuristicsBuilder,
                          cudnn_frontend::EngineHeuristicsBuilder_v8>);
static_assert(IS_ALIAS_OF<cudnn_frontend::EngineHeuristicsBuilde,
                          cudnn_frontend::EngineHeuristicsBuilder_v8>);
static_assert(IS_ALIAS_OF<cudnn_frontend::EngineHeuristics, cudnn_frontend::EngineHeuristics_v8>);
static_assert(IS_ALIAS_OF<cudnn_frontend::Engine, cudnn_frontend::Engine_v8>);
static_assert(IS_ALIAS_OF<cudnn_frontend::EngineBuilder, cudnn_frontend::EngineBuilder_v8>);
static_assert(IS_ALIAS_OF<cudnn_frontend::EngineConfig, cudnn_frontend::EngineConfig_v8>);
static_assert(
    IS_ALIAS_OF<cudnn_frontend::EngineConfigBuilder, cudnn_frontend::EngineConfigBuilder_v8>);
static_assert(
    IS_ALIAS_OF<cudnn_frontend::EngineFallbackList, cudnn_frontend::EngineFallbackList_v8>);
static_assert(IS_ALIAS_OF<cudnn_frontend::EngineFallbackListBuilder,
                          cudnn_frontend::EngineFallbackListBuilder_v8>);
static_assert(IS_ALIAS_OF<cudnn_frontend::ExecutionPlan, cudnn_frontend::ExecutionPlan_v8>);
static_assert(
    IS_ALIAS_OF<cudnn_frontend::ExecutionPlanBuilder, cudnn_frontend::ExecutionPlanBuilder_v8>);
static_assert(
    IS_ALIAS_OF<cudnn_frontend::ExecutionPlanCache, cudnn_frontend::ExecutionPlanCache_v2>);
static_assert(std::is_base_of_v<cudnn_frontend::ExecutionPlanCache_v1,
                                cudnn_frontend::ExecutionPlanCache_v2>);
static_assert(IS_ALIAS_OF<cudnn_frontend::EngineConfigList,
                          std::vector<cudnn_frontend::ManagedOpaqueDescriptor>>);
static_assert(IS_ALIAS_OF<cudnn_frontend::ManagedOpaqueDescriptor,
                          std::shared_ptr<cudnn_frontend::OpaqueBackendPointer>>);
static_assert(
    IS_ALIAS_OF<cudnn_frontend::executionPlans_t, std::vector<cudnn_frontend::ExecutionPlan>>);
static_assert(IS_ALIAS_OF<cudnn_frontend::Predicate,
                          std::function<bool(const cudnn_frontend::ExecutionPlan&)>>);
static_assert(
    IS_ALIAS_OF<cudnn_frontend::GeneratorSource,
                std::function<cudnn_frontend::EngineConfigList(cudnn_frontend::OperationGraph&)>>);

// PyTorch stores plans by value and copies them out of its benchmark cache.
static_assert(std::is_copy_constructible_v<cudnn_frontend::ExecutionPlan>);
static_assert(std::is_copy_assignable_v<cudnn_frontend::ExecutionPlan>);
static_assert(std::is_move_constructible_v<cudnn_frontend::ExecutionPlan>);
static_assert(IS_MOVE_ONLY_BACKEND_DESCRIPTOR<cudnn_frontend::EngineHeuristics_v8>);
static_assert(IS_MOVE_ONLY_BACKEND_DESCRIPTOR<cudnn_frontend::Engine_v8>);
static_assert(IS_MOVE_ONLY_BACKEND_DESCRIPTOR<cudnn_frontend::EngineConfig_v8>);
static_assert(
    std::is_base_of_v<cudnn_frontend::BackendDescriptor, cudnn_frontend::EngineFallbackList_v8>);

static_assert(static_cast<int>(cudnn_frontend::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE)
              == 0);
static_assert(
    static_cast<int>(cudnn_frontend::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE)
    == 1);
static_assert(
    static_cast<int>(cudnn_frontend::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_TILL_STABLE)
    == 2);

// The free functions PyTorch calls, with its argument shapes.
static_assert(
    std::is_same_v<decltype(cudnn_frontend::time_sorted_plan<
                            cudnn_frontend::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE>(
                       std::declval<cudnnHandle_t>(),
                       std::declval<cudnn_frontend::executionPlans_t>(),
                       std::declval<const cudnn_frontend::VariantPack&>(),
                       std::declval<std::uint64_t>())),
                   cudnn_frontend::executionPlans_t>);
static_assert(std::is_same_v<decltype(cudnn_frontend::filter(
                                 std::declval<cudnn_frontend::EngineConfigList&>(),
                                 std::declval<cudnn_frontend::EngineConfigList&>(),
                                 std::declval<std::function<bool(cudnnBackendDescriptor_t)>>())),
                             void>);
static_assert(
    std::is_same_v<decltype(cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(
                       std::declval<cudnnBackendDescriptor_t>())),
                   bool>);
static_assert(
    std::is_same_v<decltype(std::declval<cudnn_frontend::EngineConfigGenerator&>().cudnnGetPlan(
                       std::declval<cudnnHandle_t>(),
                       std::declval<cudnn_frontend::OperationGraph&>(),
                       std::declval<cudnn_frontend::Predicate>())),
                   cudnn_frontend::executionPlans_t>);
static_assert(std::is_same_v<decltype(cudnn_frontend::get_heuristics_list(
                                 std::declval<const std::vector<std::string>&>(),
                                 std::declval<cudnn_frontend::OperationGraph&>(),
                                 std::declval<std::function<bool(cudnnBackendDescriptor_t)>>(),
                                 std::declval<cudnn_frontend::EngineConfigList&>())),
                             std::vector<cudnnStatus_t>>);
static_assert(std::is_same_v<decltype(cudnn_frontend::get_heuristics_list<2>(
                                 std::declval<std::array<std::string, 2>>(),
                                 std::declval<cudnn_frontend::OperationGraph&>(),
                                 std::declval<std::function<bool(cudnnBackendDescriptor_t)>>(),
                                 std::declval<cudnn_frontend::EngineConfigList&>())),
                             std::vector<cudnnStatus_t>>);
static_assert(std::is_same_v<decltype(cudnn_frontend::get_heuristics_list<2>(
                                 std::declval<std::array<cudnnBackendHeurMode_t, 2>>(),
                                 std::declval<cudnn_frontend::OperationGraph&>(),
                                 std::declval<std::function<bool(cudnnBackendDescriptor_t)>>())),
                             cudnn_frontend::EngineConfigList>);
static_assert(std::is_same_v<decltype(cudnn_frontend::cudnnReorderFilterAndBiasInt8x32(
                                 std::declval<cudnnHandle_t>(),
                                 std::declval<const cudnn_frontend::Tensor_v8&>(),
                                 std::declval<const cudnn_frontend::ConvDesc_v8&>(),
                                 std::declval<void*>(),
                                 std::declval<void*>(),
                                 std::declval<void*>(),
                                 std::declval<void*>())),
                             cudnnStatus_t>);
#if !defined(CUDNN_FRONTEND_SKIP_JSON_LIB) && !defined(HIPDNN_FRONTEND_SKIP_JSON_LIB)
static_assert(std::is_same_v<decltype(cudnn_frontend::load_from_config(
                                 std::declval<json&>(), std::declval<const std::string&>())),
                             bool>);
static_assert(
    std::is_same_v<decltype(cudnn_frontend::check_errata(std::declval<const json&>(),
                                                         std::declval<const std::string&>(),
                                                         std::declval<cudnnHandle_t>(),
                                                         std::declval<bool (*)()>())),
                   bool>);
#endif

static_assert(IS_MOVE_ONLY_BACKEND_DESCRIPTOR<cudnn_frontend::Tensor_v8>);
static_assert(IS_MOVE_ONLY_BACKEND_DESCRIPTOR<cudnn_frontend::ConvDesc_v8>);
static_assert(IS_MOVE_ONLY_BACKEND_DESCRIPTOR<cudnn_frontend::PointWiseDesc_v8>);
static_assert(IS_MOVE_ONLY_BACKEND_DESCRIPTOR<cudnn_frontend::MatMulDesc_v8>);
static_assert(IS_MOVE_ONLY_BACKEND_DESCRIPTOR<cudnn_frontend::ReductionDesc_v8>);
static_assert(IS_MOVE_ONLY_BACKEND_DESCRIPTOR<cudnn_frontend::ResampleDesc_v8>);
static_assert(IS_MOVE_ONLY_BACKEND_DESCRIPTOR<cudnn_frontend::RngDesc_v8>);
static_assert(IS_MOVE_ONLY_BACKEND_DESCRIPTOR<cudnn_frontend::VariantPack_v8>);
static_assert(IS_MOVE_ONLY_BACKEND_DESCRIPTOR<cudnn_frontend::Operation_v8>);
static_assert(IS_MOVE_ONLY_BACKEND_DESCRIPTOR<cudnn_frontend::OperationGraph_v8>);

static_assert(std::is_same_v<decltype(::MAX_OPGRAPH_OPS), const int64_t>);
static_assert(::MAX_OPGRAPH_OPS == 250);
// Under the JSON opt-outs a pointwise operation's tag is the mode's integer,
// so the ordinals must match upstream's.
static_assert(static_cast<int>(cudnn_frontend::PointwiseMode_t::ADD) == 1);
static_assert(static_cast<int>(cudnn_frontend::PointwiseMode_t::RECIPROCAL) == 50);

static_assert(std::is_base_of_v<std::runtime_error, cudnn_frontend::cudnnException>);
static_assert(std::is_same_v<cudnn_frontend::feature_vector_t, std::vector<int64_t>>);
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
static_assert(std::is_same_v<::json, nlohmann::json>);
#endif
static_assert(cudnn_frontend::detail::get_compiled_version() == 92200);

namespace
{

TEST(TestCudnnShimV8Tu, V8AndV9SurfacesShareOneTranslationUnit)
{
    const std::vector<int64_t> dims{1, 2};
    const std::vector<int64_t> strides{2, 1};
    cudnn_frontend::TensorBuilder builder;
    const auto& chained = builder.setDataType(cudnn_frontend::DataType_t::FLOAT)
                              .setDim(static_cast<int64_t>(dims.size()), dims.data())
                              .setStride(static_cast<int64_t>(strides.size()), strides.data())
                              .setId(1)
                              .setAlignment(16);

    cudnn_frontend::graph::Graph graph;
    const auto& graphChained = graph.set_io_data_type(cudnn_frontend::DataType_t::FLOAT);

    EXPECT_EQ(&chained, &builder);
    EXPECT_EQ(&graphChained, &graph);
}

} // namespace
