// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Coverage for the free helpers and FE-namespace enums the cuDNN-compatibility
// shim publishes outside the graph wrapper (cudnn_frontend_utils.h). Includes
// only the umbrella, so a missing name fails at compile time.
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

// Upstream defines no distinct fp8 attribute classes — graph_properties.h has
// `using SDPA_fp8_attributes = SDPA_attributes`. The shim must match, or hipified
// source configuring an fp8 graph through the fp8 spelling loses every setter.
// Asserted here rather than in the SDPA-gated test file because the aliases hold
// in both HIPDNN_ENABLE_SDPA states; only the sdpa *nodes* are gated.
static_assert(std::is_same_v<fe::graph::SDPA_fp8_attributes, fe::graph::SDPA_attributes>);
static_assert(
    std::is_same_v<fe::graph::SDPA_fp8_backward_attributes, fe::graph::SDPA_backward_attributes>);

TEST(TestCudnnShimUtils, BackendVersionMatchesClaimedRuntimeVersion)
{
    // The shim has no loadable backend to interrogate, so the frontend's notion
    // of the runtime version is the same compile-time claim cudnnGetVersion()
    // reports. Samples gate features on this number.
    EXPECT_EQ(fe::detail::get_backend_version(), static_cast<size_t>(CUDNN_VERSION));
    EXPECT_EQ(fe::detail::get_backend_version(), cudnnGetVersion());
}

TEST(TestCudnnShimUtils, ElementSizeInBitsCoversSubByteAndPackedTypes)
{
    EXPECT_EQ(fe::detail::get_element_size_in_bits(fe::DataType_t::INT8x32), 256U);
    EXPECT_EQ(fe::detail::get_element_size_in_bits(fe::DataType_t::DOUBLE), 64U);
    EXPECT_EQ(fe::detail::get_element_size_in_bits(fe::DataType_t::FLOAT), 32U);
    EXPECT_EQ(fe::detail::get_element_size_in_bits(fe::DataType_t::HALF), 16U);
    EXPECT_EQ(fe::detail::get_element_size_in_bits(fe::DataType_t::BFLOAT16), 16U);
    EXPECT_EQ(fe::detail::get_element_size_in_bits(fe::DataType_t::FP8_E4M3), 8U);
    EXPECT_EQ(fe::detail::get_element_size_in_bits(fe::DataType_t::FP6_E2M3), 6U);
    EXPECT_EQ(fe::detail::get_element_size_in_bits(fe::DataType_t::FP4_E2M1), 4U);
    EXPECT_EQ(fe::detail::get_element_size_in_bits(fe::DataType_t::INT4), 4U);
    EXPECT_EQ(fe::detail::get_element_size_in_bits(fe::DataType_t::BOOLEAN), 1U);
}

// Sub-byte types are why callers divide by 8 rather than multiplying a byte
// count; a size of 0 for a set type would silently produce zero-length buffers.
TEST(TestCudnnShimUtils, ElementSizeInBitsIsZeroOnlyForUnsetType)
{
    EXPECT_EQ(fe::detail::get_element_size_in_bits(fe::DataType_t::NOT_SET), 0U);
    EXPECT_GT(fe::detail::get_element_size_in_bits(fe::DataType_t::UINT8), 0U);
    EXPECT_GT(fe::detail::get_element_size_in_bits(fe::DataType_t::COMPLEX_FP64), 0U);
}

// MoeGroupedMatmulMode_t is aliased to hipDNN's enum, so a value set through the
// shim spelling is the same object hipDNN sees. The other two are shim-owned.
TEST(TestCudnnShimUtils, MoeGroupedMatmulModeIsAliasedToNative)
{
    static_assert(
        std::is_same_v<fe::MoeGroupedMatmulMode_t, hipdnn_frontend::MoeGroupedMatmulMode_t>);

    EXPECT_EQ(fe::MoeGroupedMatmulMode_t::GATHER, hipdnn_frontend::MoeGroupedMatmulMode::GATHER);
}

TEST(TestCudnnShimUtils, ShimOwnedEnumsExposeUpstreamValueNames)
{
    const fe::TensorReordering_t reordering = fe::TensorReordering_t::F8_128x4;
    const fe::ReshapeMode_t reshape = fe::ReshapeMode_t::LOGICAL;

    EXPECT_NE(reordering, fe::TensorReordering_t::NONE);
    EXPECT_NE(reshape, fe::ReshapeMode_t::NOT_SET);
    EXPECT_NE(fe::ReshapeMode_t::VIEW_ONLY, fe::ReshapeMode_t::LOGICAL);
    EXPECT_NE(fe::TensorReordering_t::INT8x32, fe::TensorReordering_t::F16x16);
}

// The reshape node is still a fail stub, but it now carries the mode so chained
// consumer source compiles and round-trips the value it set. The default is
// upstream's VIEW_ONLY, not the enum's NOT_SET.
TEST(TestCudnnShimUtils, ReshapeAttributesRoundTripsMode)
{
    fe::graph::Reshape_attributes attributes;

    EXPECT_EQ(attributes.get_reshape_mode(), fe::ReshapeMode_t::VIEW_ONLY);

    attributes.set_name("rs").set_reshape_mode(fe::ReshapeMode_t::LOGICAL);

    EXPECT_EQ(attributes.get_reshape_mode(), fe::ReshapeMode_t::LOGICAL);
    EXPECT_EQ(attributes.get_name(), "rs");
}

// Adding the mode setter must not promote the node: it still records
// GRAPH_NOT_SUPPORTED, surfaced at the next validate().
TEST(TestCudnnShimUtils, ReshapeNodeStaysUnsupportedDespiteModeSetter)
{
    fe::graph::Graph graph;
    auto input = graph.tensor(fe::graph::Tensor_attributes{}
                                  .set_dim({1, 2, 3})
                                  .set_stride({6, 3, 1})
                                  .set_data_type(fe::DataType_t::FLOAT)
                                  .set_uid(1));

    graph.reshape(input,
                  fe::graph::Reshape_attributes{}.set_reshape_mode(fe::ReshapeMode_t::LOGICAL));

    auto error = graph.validate();
    ASSERT_TRUE(error.is_bad());
    EXPECT_EQ(error.get_code(), fe::error_code_t::GRAPH_NOT_SUPPORTED);
}

TEST(TestCudnnShimUtils, TransposeAttributesRoundTripsPermutation)
{
    fe::graph::Transpose_attributes attributes;

    EXPECT_TRUE(attributes.get_permutation().empty());

    attributes.set_name("t").set_permutation({0, 2, 1, 3});

    EXPECT_EQ(attributes.get_permutation(), (std::vector<int64_t>{0, 2, 1, 3}));
    EXPECT_EQ(attributes.get_name(), "t");
}

// Upstream defaults slice strides to {1}, and a caller that sets only slices
// relies on that default rather than an empty vector.
TEST(TestCudnnShimUtils, SliceAttributesRoundTripSlicesAndDefaultStrides)
{
    fe::graph::Slice_attributes attributes;

    EXPECT_TRUE(attributes.get_slices().empty());
    EXPECT_EQ(attributes.get_strides(), (std::vector<int64_t>{1}));

    attributes.set_slices({{0, 4}, {2, 6}}).set_strides({2, 1});

    EXPECT_EQ(attributes.get_slices(), (std::vector<std::pair<int64_t, int64_t>>{{0, 4}, {2, 6}}));
    EXPECT_EQ(attributes.get_strides(), (std::vector<int64_t>{2, 1}));
}

TEST(TestCudnnShimUtils, ConcatenateAttributesRoundTripAxisAndInPlaceIndex)
{
    fe::graph::Concatenate_attributes attributes;

    attributes.set_axis(2).set_in_place_index(1);

    EXPECT_EQ(attributes.get_axis(), 2);
    EXPECT_EQ(attributes.get_in_place_index(), 1);
}

TEST(TestCudnnShimUtils, AdaLayernormAttributesRoundTripPhaseAndEpsilon)
{
    fe::graph::Graph graph;
    auto epsilon = graph.tensor(fe::graph::Tensor_attributes{}
                                    .set_dim({1, 1, 1})
                                    .set_stride({1, 1, 1})
                                    .set_data_type(fe::DataType_t::FLOAT)
                                    .set_uid(1));

    fe::graph::AdaLayernorm_attributes attributes;

    EXPECT_EQ(attributes.get_forward_phase(), fe::NormFwdPhase_t::NOT_SET);

    attributes.set_forward_phase(fe::NormFwdPhase_t::TRAINING).set_epsilon(epsilon);

    EXPECT_EQ(attributes.get_forward_phase(), fe::NormFwdPhase_t::TRAINING);
    EXPECT_EQ(attributes.get_epsilon(), epsilon);
}

TEST(TestCudnnShimUtils, BnFinalizeAttributesRoundTripPreviousRunningStats)
{
    fe::graph::Graph graph;
    auto mean = graph.tensor(fe::graph::Tensor_attributes{}.set_uid(1));
    auto variance = graph.tensor(fe::graph::Tensor_attributes{}.set_uid(2));
    auto momentum = graph.tensor(fe::graph::Tensor_attributes{}.set_uid(3));

    fe::graph::BN_finalize_attributes attributes;
    attributes.set_previous_running_stats(mean, variance, momentum);

    EXPECT_EQ(attributes.get_previous_running_mean(), mean);
    EXPECT_EQ(attributes.get_previous_running_variance(), variance);
    EXPECT_EQ(attributes.get_momentum(), momentum);
}

TEST(TestCudnnShimUtils, AdaLayernormBackwardAttributesRoundTripSavedStats)
{
    fe::graph::Graph graph;
    auto mean = graph.tensor(fe::graph::Tensor_attributes{}.set_uid(1));
    auto invVariance = graph.tensor(fe::graph::Tensor_attributes{}.set_uid(2));

    fe::graph::AdaLayernorm_backward_attributes attributes;
    attributes.set_saved_mean_and_inv_variance(mean, invVariance);

    EXPECT_EQ(attributes.get_saved_mean(), mean);
    EXPECT_EQ(attributes.get_saved_inv_variance(), invVariance);
}

} // namespace
