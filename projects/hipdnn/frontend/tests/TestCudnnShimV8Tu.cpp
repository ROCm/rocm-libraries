// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Includes only the shim umbrella, as a ported consumer does, and pins the
// v0.x builder surface it exposes next to the v9 graph API.
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>
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

static_assert(IS_MOVE_ONLY_BACKEND_DESCRIPTOR<cudnn_frontend::Tensor_v8>);
static_assert(IS_MOVE_ONLY_BACKEND_DESCRIPTOR<cudnn_frontend::ConvDesc_v8>);
static_assert(IS_MOVE_ONLY_BACKEND_DESCRIPTOR<cudnn_frontend::PointWiseDesc_v8>);
static_assert(IS_MOVE_ONLY_BACKEND_DESCRIPTOR<cudnn_frontend::MatMulDesc_v8>);
static_assert(IS_MOVE_ONLY_BACKEND_DESCRIPTOR<cudnn_frontend::ReductionDesc_v8>);
static_assert(IS_MOVE_ONLY_BACKEND_DESCRIPTOR<cudnn_frontend::ResampleDesc_v8>);
static_assert(IS_MOVE_ONLY_BACKEND_DESCRIPTOR<cudnn_frontend::RngDesc_v8>);
static_assert(IS_MOVE_ONLY_BACKEND_DESCRIPTOR<cudnn_frontend::VariantPack_v8>);

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
