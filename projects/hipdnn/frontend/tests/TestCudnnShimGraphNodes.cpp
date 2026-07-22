// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Node coverage for the cuDNN-shaped graph wrapper: the 39 cuDNN v9
// *_attributes classes and their Graph::* node methods. Three contracts:
//   (a) Tier-1 nodes with a real hipDNN engine build a valid graph that
//       validate()s good (host-only; hipDNN validate() needs no device).
//   (b) Tier-2 fail-stub nodes record GRAPH_NOT_SUPPORTED, surfaced at the next
//       validate(), with a message pointing at the issue tracker.
//   (c) Every one of the 39 attribute classes (tier-1 aliases, SDPA aliases,
//       and the fail-stub/attribute-only stubs) default-constructs and chains
//       .set_name — so a missing alias or stub fails at compile time.
// Include only the umbrella; gated behind HIPDNN_ENABLE_CUDNN_COMPATIBILITY in
// the frontend tests CMakeLists.
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <gtest/gtest.h>

#include "CudnnShimTestSupport.hpp"

#include <memory>
#include <string>

namespace
{
namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

// A fail-stub node records GRAPH_NOT_SUPPORTED on the composition graph; it
// surfaces at the next validate() with a message pointing at the issue tracker.
void expectGraphNotSupported(fe::graph::Graph& graph)
{
    auto error = graph.validate();
    EXPECT_TRUE(error.is_bad());
    EXPECT_EQ(error.get_code(), fe::error_code_t::GRAPH_NOT_SUPPORTED);
    EXPECT_NE(error.get_message().find("github.com/ROCm/rocm-libraries/issues"), std::string::npos);
}

// --- (a) Supported / Tier-1 nodes: a well-shaped graph validates good --------

// Mirrors samples/cudnn_shim/ConvFprop.cpp.
TEST(TestCudnnShimGraphNodes, ConvFpropValidGraphValidates)
{
    const int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF).set_compute_data_type(fe::DataType_t::FLOAT);

    auto x = graph.tensor(fe::graph::Tensor_attributes{}
                              .set_name("image")
                              .set_dim({n, c, h, w})
                              .set_stride({c * h * w, 1, c * w, c})
                              .set_uid(1));
    auto weight = graph.tensor(fe::graph::Tensor_attributes{}
                                   .set_name("filter")
                                   .set_dim({k, c, r, s})
                                   .set_stride({c * r * s, 1, c * s, c})
                                   .set_uid(2));

    auto y = graph.conv_fprop(
        x,
        weight,
        fe::graph::Conv_fprop_attributes{}.set_padding({0, 0}).set_stride({1, 1}).set_dilation(
            {1, 1}));
    ASSERT_NE(y, nullptr);
    y->set_output(true).set_uid(3);

    EXPECT_TRUE(graph.validate().is_good());
}

// Mirrors samples/cudnn_shim/Matmul.cpp.
TEST(TestCudnnShimGraphNodes, MatmulValidGraphValidates)
{
    const int64_t b = 4, m = 64, n = 32, k = 128;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF).set_compute_data_type(fe::DataType_t::FLOAT);

    auto a = graph.tensor(fe::graph::Tensor_attributes{}
                              .set_name("A")
                              .set_dim({b, m, k})
                              .set_stride({m * k, k, 1})
                              .set_uid(1));
    auto bMat = graph.tensor(fe::graph::Tensor_attributes{}
                                 .set_name("B")
                                 .set_dim({b, k, n})
                                 .set_stride({k * n, n, 1})
                                 .set_uid(2));

    auto c = graph.matmul(a, bMat, fe::graph::Matmul_attributes{}.set_name("matmul"));
    ASSERT_NE(c, nullptr);
    c->set_output(true).set_uid(3);

    EXPECT_TRUE(graph.validate().is_good());
}

// Mirrors samples/cudnn_shim/Layernorm.cpp (a norm node).
TEST(TestCudnnShimGraphNodes, LayernormValidGraphValidates)
{
    const int64_t b = 4, s = 1024, d = 128;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto x = graph.tensor(fe::graph::Tensor_attributes{}
                              .set_name("X")
                              .set_dim({b * s, d, 1, 1})
                              .set_stride({d, 1, d, d})
                              .set_uid(1));
    auto scale = graph.tensor(fe::graph::Tensor_attributes{}
                                  .set_name("scale")
                                  .set_dim({1, d, 1, 1})
                                  .set_stride({d, 1, d, d})
                                  .set_data_type(fe::DataType_t::FLOAT)
                                  .set_uid(2));
    auto bias = graph.tensor(fe::graph::Tensor_attributes{}
                                 .set_name("bias")
                                 .set_dim({1, d, 1, 1})
                                 .set_stride({d, 1, d, d})
                                 .set_data_type(fe::DataType_t::FLOAT)
                                 .set_uid(3));
    auto epsilon = graph.tensor(1e-05F, fe::graph::ScalarType::COMPILE_TIME_CONST);

    auto [y, mean, invVariance]
        = graph.layernorm(x,
                          scale,
                          bias,
                          fe::graph::Layernorm_attributes{}
                              .set_forward_phase(fe::NormFwdPhase_t::TRAINING)
                              .set_epsilon(epsilon));
    ASSERT_NE(y, nullptr);
    ASSERT_NE(mean, nullptr);
    ASSERT_NE(invVariance, nullptr);
    y->set_output(true).set_uid(4);
    mean->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_uid(5);
    invVariance->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_uid(6);

    EXPECT_TRUE(graph.validate().is_good());
}

// Mirrors samples/cudnn_shim/Pointwise.cpp (binary ADD).
TEST(TestCudnnShimGraphNodes, PointwiseValidGraphValidates)
{
    const int64_t n = 4;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF).set_compute_data_type(fe::DataType_t::FLOAT);

    auto a = graph.tensor(fe::graph::Tensor_attributes{}
                              .set_name("A")
                              .set_dim({n, n, n, n})
                              .set_stride({n * n * n, n * n, n, 1})
                              .set_uid(1));
    auto b = graph.tensor(fe::graph::Tensor_attributes{}
                              .set_name("B")
                              .set_dim({n, n, n, n})
                              .set_stride({n * n * n, n * n, n, 1})
                              .set_uid(2));

    auto c = graph.pointwise(a,
                             b,
                             fe::graph::Pointwise_attributes{}
                                 .set_mode(fe::PointwiseMode_t::ADD)
                                 .set_compute_data_type(fe::DataType_t::FLOAT));
    ASSERT_NE(c, nullptr);
    c->set_output(true).set_uid(3);

    EXPECT_TRUE(graph.validate().is_good());
}

// Reduction along the trailing dims. The output shape is not inferable from the
// attributes alone, so pin it explicitly (mirrors the native reduction node
// tests' PartialReductionValid shapes).
TEST(TestCudnnShimGraphNodes, ReductionValidGraphValidates)
{
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);

    auto x = graph.tensor(fe::graph::Tensor_attributes{}
                              .set_name("X")
                              .set_dim({2, 8, 16, 64})
                              .set_stride({8192, 1024, 64, 1})
                              .set_uid(1));

    auto y
        = graph.reduction(x, fe::graph::Reduction_attributes{}.set_mode(fe::ReductionMode_t::ADD));
    ASSERT_NE(y, nullptr);
    y->set_output(true)
        .set_dim({2, 8, 1, 1})
        .set_stride({8, 1, 1, 1})
        .set_data_type(fe::DataType_t::FLOAT)
        .set_uid(2);

    EXPECT_TRUE(graph.validate().is_good());
}

// --- (b) Tier-2 fail-stub nodes: recorded GRAPH_NOT_SUPPORTED ---------------
//
// The error is recorded before any tensor validation, so null inputs are fine.

TEST(TestCudnnShimGraphNodes, ReshapeRecordsGraphNotSupported)
{
    fe::graph::Graph graph;
    graph.reshape(nullptr, fe::graph::Reshape_attributes{});
    expectGraphNotSupported(graph);
}

TEST(TestCudnnShimGraphNodes, TransposeRecordsGraphNotSupported)
{
    fe::graph::Graph graph;
    graph.transpose(nullptr, fe::graph::Transpose_attributes{});
    expectGraphNotSupported(graph);
}

TEST(TestCudnnShimGraphNodes, SliceRecordsGraphNotSupported)
{
    fe::graph::Graph graph;
    graph.slice(nullptr, fe::graph::Slice_attributes{});
    expectGraphNotSupported(graph);
}

TEST(TestCudnnShimGraphNodes, RngRecordsGraphNotSupported)
{
    fe::graph::Graph graph;
    graph.rng(nullptr, nullptr, fe::graph::Rng_attributes{});
    expectGraphNotSupported(graph);
}

TEST(TestCudnnShimGraphNodes, InstancenormRecordsGraphNotSupported)
{
    fe::graph::Graph graph;
    graph.instancenorm(nullptr, nullptr, nullptr, fe::graph::Instancenorm_attributes{});
    expectGraphNotSupported(graph);
}

TEST(TestCudnnShimGraphNodes, RopeRecordsGraphNotSupported)
{
    fe::graph::Graph graph;
    graph.rope(nullptr, nullptr, fe::graph::RoPE_attributes{});
    expectGraphNotSupported(graph);
}

// A fail-stub records at add time regardless of the built-up graph: even with a
// valid tier-1 node already present, the recorded error wins at validate().
TEST(TestCudnnShimGraphNodes, FailStubPoisonsAnOtherwiseValidGraph)
{
    const int64_t n = 4;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF).set_compute_data_type(fe::DataType_t::FLOAT);

    auto a = graph.tensor(fe::graph::Tensor_attributes{}
                              .set_dim({n, n, n, n})
                              .set_stride({n * n * n, n * n, n, 1})
                              .set_uid(1));
    auto b = graph.tensor(fe::graph::Tensor_attributes{}
                              .set_dim({n, n, n, n})
                              .set_stride({n * n * n, n * n, n, 1})
                              .set_uid(2));
    auto c = graph.pointwise(
        a, b, fe::graph::Pointwise_attributes{}.set_mode(fe::PointwiseMode_t::ADD));
    ASSERT_NE(c, nullptr);
    c->set_output(true).set_uid(3);

    graph.transpose(nullptr, fe::graph::Transpose_attributes{});

    expectGraphNotSupported(graph);
}

// --- (c) All 39 attribute classes construct and chain .set_name -------------

// Every cuDNN v9 *_attributes class — whether a 1:1 hipDNN alias, an SDPA alias,
// or a fail-stub/attribute-only stub — must default-construct and expose the
// universal fluent .set_name. A missing alias or stub fails to compile here.
template <typename Attributes>
void expectConstructsAndNames()
{
    Attributes attributes;
    attributes.set_name("node");
    EXPECT_EQ(attributes.get_name(), "node");
}

TEST(TestCudnnShimGraphNodes, AllAttributeClassesConstructAndName)
{
    // Tier-1 aliases (real hipDNN engine).
    expectConstructsAndNames<fe::graph::Batchnorm_attributes>();
    expectConstructsAndNames<fe::graph::Batchnorm_backward_attributes>();
    expectConstructsAndNames<fe::graph::Batchnorm_inference_attributes>();
    expectConstructsAndNames<fe::graph::Block_scale_dequantize_attributes>();
    expectConstructsAndNames<fe::graph::Block_scale_quantize_attributes>();
    expectConstructsAndNames<fe::graph::Conv_dgrad_attributes>();
    expectConstructsAndNames<fe::graph::Conv_fprop_attributes>();
    expectConstructsAndNames<fe::graph::Conv_wgrad_attributes>();
    expectConstructsAndNames<fe::graph::Layernorm_attributes>();
    expectConstructsAndNames<fe::graph::Layernorm_backward_attributes>();
    expectConstructsAndNames<fe::graph::Matmul_attributes>();
    expectConstructsAndNames<fe::graph::Pointwise_attributes>();
    expectConstructsAndNames<fe::graph::Reduction_attributes>();
    expectConstructsAndNames<fe::graph::Resample_attributes>();
    expectConstructsAndNames<fe::graph::Rmsnorm_attributes>();
    expectConstructsAndNames<fe::graph::Rmsnorm_backward_attributes>();

#ifdef HIPDNN_ENABLE_SDPA
    // SDPA aliases (only present in an SDPA-enabled build).
    expectConstructsAndNames<fe::graph::SDPA_attributes>();
    expectConstructsAndNames<fe::graph::SDPA_backward_attributes>();
#endif

    // Tier-2 fail-stub attribute classes.
    expectConstructsAndNames<fe::graph::BN_finalize_attributes>();
    expectConstructsAndNames<fe::graph::Genstats_attributes>();
    expectConstructsAndNames<fe::graph::DBN_weight_attributes>();
    expectConstructsAndNames<fe::graph::Instancenorm_attributes>();
    expectConstructsAndNames<fe::graph::Instancenorm_backward_attributes>();
    expectConstructsAndNames<fe::graph::AdaLayernorm_attributes>();
    expectConstructsAndNames<fe::graph::AdaLayernorm_backward_attributes>();
    expectConstructsAndNames<fe::graph::Rng_attributes>();
    expectConstructsAndNames<fe::graph::Reshape_attributes>();
    expectConstructsAndNames<fe::graph::Transpose_attributes>();
    expectConstructsAndNames<fe::graph::RoPE_attributes>();
    expectConstructsAndNames<fe::graph::RoPE_backward_attributes>();
    expectConstructsAndNames<fe::graph::SDPA_fp8_backward_attributes>();
    expectConstructsAndNames<fe::graph::DiagonalBandMask_attributes>();
    expectConstructsAndNames<fe::graph::Slice_attributes>();
    expectConstructsAndNames<fe::graph::Concatenate_attributes>();
    expectConstructsAndNames<fe::graph::Moe_grouped_matmul_attributes>();
    expectConstructsAndNames<fe::graph::Moe_grouped_matmul_bwd_attributes>();

    // Attribute-only stubs (no Graph method): must still exist and chain.
    expectConstructsAndNames<fe::graph::Matmul_fp8_attributes>();
    expectConstructsAndNames<fe::graph::Softmax_attributes>();
    expectConstructsAndNames<fe::graph::PagedCacheLoad_attributes>();
}

} // namespace
