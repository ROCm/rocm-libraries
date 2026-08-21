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
    const int64_t n = 16;
    const int64_t c = 128;
    const int64_t h = 64;
    const int64_t w = 64;
    const int64_t k = 256;
    const int64_t r = 1;
    const int64_t s = 1;

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
    const int64_t b = 4;
    const int64_t m = 64;
    const int64_t n = 32;
    const int64_t k = 128;

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
    const int64_t b = 4;
    const int64_t s = 1024;
    const int64_t d = 128;

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

// All Tier-1 wiring graphs below run FLOAT end to end; factor the repeated
// graph type setup so each test is just tensors + the method under test.
void setFloatGraphTypes(fe::graph::Graph& graph)
{
    graph.set_io_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT);
}

// Mirrors native TestGraph.cpp BatchnormNodeCreation shapes.
TEST(TestCudnnShimGraphNodes, BatchnormValidGraphValidates)
{
    fe::graph::Graph graph;
    setFloatGraphTypes(graph);

    auto x = hipdnn_shim_test::makeTensor(graph, {1, 2, 3, 4}, {24, 12, 4, 1}, 1);
    auto scale = hipdnn_shim_test::makeTensor(graph, {1, 2, 1, 1}, {2, 1, 1, 1}, 2);
    auto bias = hipdnn_shim_test::makeTensor(graph, {1, 2, 1, 1}, {2, 1, 1, 1}, 3);
    auto epsilon = graph.tensor(0.001F, fe::graph::ScalarType::COMPILE_TIME_CONST);

    auto [y, mean, invVariance, nextRunningMean, nextRunningVariance]
        = graph.batchnorm(x, scale, bias, fe::graph::Batchnorm_attributes{}.set_epsilon(epsilon));
    ASSERT_NE(y, nullptr);
    ASSERT_NE(mean, nullptr);
    ASSERT_NE(invVariance, nullptr);
    EXPECT_EQ(nextRunningMean, nullptr);
    EXPECT_EQ(nextRunningVariance, nullptr);
    y->set_output(true).set_uid(4);

    EXPECT_TRUE(graph.validate().is_good());
}

// Mirrors native TestGraph.cpp BatchnormBackwardNodeCreation shapes.
TEST(TestCudnnShimGraphNodes, BatchnormBackwardValidGraphValidates)
{
    fe::graph::Graph graph;
    setFloatGraphTypes(graph);

    auto dy = hipdnn_shim_test::makeTensor(graph, {1, 2, 3, 4}, {24, 12, 4, 1}, 1);
    auto x = hipdnn_shim_test::makeTensor(graph, {1, 2, 3, 4}, {24, 12, 4, 1}, 2);
    auto scale = hipdnn_shim_test::makeTensor(graph, {1, 2, 1, 1}, {2, 1, 1, 1}, 3);

    auto [dx, dscale, dbias]
        = graph.batchnorm_backward(dy, x, scale, fe::graph::Batchnorm_backward_attributes{});
    ASSERT_NE(dx, nullptr);
    ASSERT_NE(dscale, nullptr);
    ASSERT_NE(dbias, nullptr);
    dx->set_output(true).set_uid(4);

    EXPECT_TRUE(graph.validate().is_good());
}

// Mirrors native TestGraph.cpp BatchnormInferenceNodeCreation shapes.
TEST(TestCudnnShimGraphNodes, BatchnormInferenceValidGraphValidates)
{
    fe::graph::Graph graph;
    setFloatGraphTypes(graph);

    auto x = hipdnn_shim_test::makeTensor(graph, {1, 2, 3, 4}, {24, 12, 4, 1}, 1);
    auto mean = hipdnn_shim_test::makeTensor(graph, {1, 2, 1, 1}, {2, 1, 1, 1}, 2);
    auto invVariance = hipdnn_shim_test::makeTensor(graph, {1, 2, 1, 1}, {2, 1, 1, 1}, 3);
    auto scale = hipdnn_shim_test::makeTensor(graph, {1, 2, 1, 1}, {2, 1, 1, 1}, 4);
    auto bias = hipdnn_shim_test::makeTensor(graph, {1, 2, 1, 1}, {2, 1, 1, 1}, 5);

    auto y = graph.batchnorm_inference(
        x, mean, invVariance, scale, bias, fe::graph::Batchnorm_inference_attributes{});
    ASSERT_NE(y, nullptr);
    y->set_output(true).set_uid(6);

    EXPECT_TRUE(graph.validate().is_good());
}

// Mirrors native TestGraphLayernormBackward.cpp shapes.
TEST(TestCudnnShimGraphNodes, LayernormBackwardValidGraphValidates)
{
    fe::graph::Graph graph;
    setFloatGraphTypes(graph);

    auto dy = hipdnn_shim_test::makeTensor(graph, {16, 64, 32, 32}, {65536, 1024, 32, 1}, 1);
    auto x = hipdnn_shim_test::makeTensor(graph, {16, 64, 32, 32}, {65536, 1024, 32, 1}, 2);
    auto scale = hipdnn_shim_test::makeTensor(graph, {1, 64, 32, 32}, {65536, 1024, 32, 1}, 3);
    auto mean = hipdnn_shim_test::makeTensor(graph, {16, 1, 1, 1}, {1, 1, 1, 1}, 4);
    auto invVariance = hipdnn_shim_test::makeTensor(graph, {16, 1, 1, 1}, {1, 1, 1, 1}, 5);
    auto epsilon = hipdnn_shim_test::makeTensor(graph, {1}, {1}, 6);

    auto [dx, dscale, dbias] = graph.layernorm_backward(dy,
                                                        x,
                                                        scale,
                                                        fe::graph::Layernorm_backward_attributes{}
                                                            .set_mean(mean)
                                                            .set_inv_variance(invVariance)
                                                            .set_epsilon(epsilon));
    ASSERT_NE(dx, nullptr);
    ASSERT_NE(dscale, nullptr);
    ASSERT_NE(dbias, nullptr);
    dx->set_output(true).set_uid(7);

    EXPECT_TRUE(graph.validate().is_good());
}

// Mirrors native TestGraph.cpp RMSNormNodeCreation shapes.
TEST(TestCudnnShimGraphNodes, RmsnormValidGraphValidates)
{
    fe::graph::Graph graph;
    setFloatGraphTypes(graph);

    auto x = hipdnn_shim_test::makeTensor(graph, {2, 64, 32, 32}, {65536, 1024, 32, 1}, 1);
    auto scale = hipdnn_shim_test::makeTensor(graph, {1, 64, 32, 32}, {65536, 1024, 32, 1}, 2);
    auto epsilon = graph.tensor(1e-5F, fe::graph::ScalarType::COMPILE_TIME_CONST);

    auto [y, invRms]
        = graph.rmsnorm(x,
                        scale,
                        fe::graph::Rmsnorm_attributes{}.set_epsilon(epsilon).set_forward_phase(
                            fe::NormFwdPhase_t::TRAINING));
    ASSERT_NE(y, nullptr);
    ASSERT_NE(invRms, nullptr);
    y->set_output(true).set_uid(3);

    EXPECT_TRUE(graph.validate().is_good());
}

// Mirrors native TestGraph.cpp RMSNormBackwardNodeCreation shapes.
TEST(TestCudnnShimGraphNodes, RmsnormBackwardValidGraphValidates)
{
    fe::graph::Graph graph;
    setFloatGraphTypes(graph);

    auto dy = hipdnn_shim_test::makeTensor(graph, {1, 64, 32, 32}, {65536, 1024, 32, 1}, 1);
    auto x = hipdnn_shim_test::makeTensor(graph, {1, 64, 32, 32}, {65536, 1024, 32, 1}, 2);
    auto scale = hipdnn_shim_test::makeTensor(graph, {1, 64, 32, 32}, {65536, 1024, 32, 1}, 3);
    auto invRms = hipdnn_shim_test::makeTensor(graph, {1, 1, 1, 1}, {1, 1, 1, 1}, 4);

    auto [dx, dscale, dbias]
        = graph.rmsnorm_backward(dy, x, scale, invRms, fe::graph::Rmsnorm_backward_attributes{});
    ASSERT_NE(dx, nullptr);
    ASSERT_NE(dscale, nullptr);
    EXPECT_EQ(dbias, nullptr); // dbias only produced when set_compute_dbias(true)
    dx->set_output(true).set_uid(5);

    EXPECT_TRUE(graph.validate().is_good());
}

// Mirrors native TestGraph.cpp ConvolutionDgradNodeCreation shapes. The returned
// dx dim must be set before validate(), else it fails ATTRIBUTE_NOT_SET.
TEST(TestCudnnShimGraphNodes, ConvDgradValidGraphValidates)
{
    fe::graph::Graph graph;
    setFloatGraphTypes(graph);

    auto dy = hipdnn_shim_test::makeTensor(graph, {1, 64, 32, 32}, {65536, 1024, 32, 1}, 1);
    auto w = hipdnn_shim_test::makeTensor(graph, {64, 3, 3, 3}, {27, 9, 3, 1}, 2);

    auto dx = graph.conv_dgrad(
        dy,
        w,
        fe::graph::Conv_dgrad_attributes{}.set_padding({1, 1}).set_stride({1, 1}).set_dilation(
            {1, 1}));
    ASSERT_NE(dx, nullptr);
    dx->set_dim({1, 3, 32, 32});
    dx->set_output(true).set_uid(3);

    EXPECT_TRUE(graph.validate().is_good());
}

// Mirrors native TestGraph.cpp ConvolutionWgradNodeCreation shapes. The returned
// dw dim must be set before validate().
TEST(TestCudnnShimGraphNodes, ConvWgradValidGraphValidates)
{
    fe::graph::Graph graph;
    setFloatGraphTypes(graph);

    auto dy = hipdnn_shim_test::makeTensor(graph, {1, 64, 32, 32}, {65536, 1024, 32, 1}, 1);
    auto x = hipdnn_shim_test::makeTensor(graph, {1, 3, 32, 32}, {3072, 1024, 32, 1}, 2);

    auto dw = graph.conv_wgrad(
        dy,
        x,
        fe::graph::Conv_wgrad_attributes{}.set_padding({1, 1}).set_stride({1, 1}).set_dilation(
            {1, 1}));
    ASSERT_NE(dw, nullptr);
    dw->set_dim({64, 3, 3, 3});
    dw->set_output(true).set_uid(3);

    EXPECT_TRUE(graph.validate().is_good());
}

// Mirrors native TestGraph.cpp Resample MAXPOOL shapes.
TEST(TestCudnnShimGraphNodes, ResampleValidGraphValidates)
{
    fe::graph::Graph graph;
    setFloatGraphTypes(graph);

    auto x = hipdnn_shim_test::makeTensor(graph, {1, 3, 4, 4}, {48, 16, 4, 1}, 1);

    auto [y, index] = graph.resample(x,
                                     fe::graph::Resample_attributes{}
                                         .set_resample_mode(fe::ResampleMode_t::MAXPOOL)
                                         .set_padding_mode(fe::PaddingMode_t::ZERO_PAD)
                                         .set_pre_padding({0, 0})
                                         .set_post_padding({0, 0})
                                         .set_stride({2, 2})
                                         .set_window({2, 2}));
    ASSERT_NE(y, nullptr);
    EXPECT_EQ(index, nullptr); // index only produced when generate_index is set
    y->set_output(true).set_uid(2);

    EXPECT_TRUE(graph.validate().is_good());
}

// Mirrors native TestBlockScaleQuantizeNode shapes; y/scale are node-produced.
TEST(TestCudnnShimGraphNodes, BlockScaleQuantizeValidGraphValidates)
{
    fe::graph::Graph graph;
    setFloatGraphTypes(graph);

    auto x = hipdnn_shim_test::makeTensor(graph, {2, 64, 32, 32}, {65536, 1024, 32, 1}, 1);

    auto [y, scale] = graph.block_scale_quantize(
        x, fe::graph::Block_scale_quantize_attributes{}.set_block_size(32));
    ASSERT_NE(y, nullptr);
    ASSERT_NE(scale, nullptr);
    y->set_output(true).set_uid(2);

    EXPECT_TRUE(graph.validate().is_good());
}

// Mirrors native TestBlockScaleDequantizeNode shapes. The dequantize output must
// stay virtual (the node rejects a non-virtual y), so it is left unmarked rather
// than wired with set_output(true) like the other Tier-1 outputs.
TEST(TestCudnnShimGraphNodes, BlockScaleDequantizeValidGraphValidates)
{
    fe::graph::Graph graph;
    setFloatGraphTypes(graph);

    auto x = hipdnn_shim_test::makeTensor(graph, {2, 64, 32, 32}, {65536, 1024, 32, 1}, 1);
    auto scale = hipdnn_shim_test::makeTensor(graph, {2, 2, 32, 32}, {2048, 1024, 32, 1}, 2);

    auto y = graph.block_scale_dequantize(
        x,
        scale,
        fe::graph::Block_scale_dequantize_attributes{}.set_block_size(std::vector<int32_t>{32}));
    ASSERT_NE(y, nullptr);

    EXPECT_TRUE(graph.validate().is_good());
}

// --- (a2) Unhappy path: Tier-1 nodes still enforce constraints --------------
//
// The wrapper forwards to the real hipDNN engines, so a malformed Tier-1 graph
// must fail validate() gracefully with a specific error code -- never crash and
// never masquerade as GRAPH_NOT_SUPPORTED (which is reserved for fail-stubs).

void expectValidateFails(fe::graph::Graph& graph, fe::error_code_t code)
{
    auto error = graph.validate();
    EXPECT_TRUE(error.is_bad());
    EXPECT_EQ(error.get_code(), code);
    EXPECT_NE(error.get_code(), fe::error_code_t::GRAPH_NOT_SUPPORTED);
}

// conv_fprop with default attributes: padding/stride/dilation unset -> the conv
// node's pre-validation rejects it with ATTRIBUTE_NOT_SET.
TEST(TestCudnnShimGraphNodes, ConvFpropMissingParamsFailsValidation)
{
    fe::graph::Graph graph;
    setFloatGraphTypes(graph);

    auto x = hipdnn_shim_test::makeTensor(graph, {16, 128, 64, 64}, {524288, 1, 8192, 128}, 1);
    auto w = hipdnn_shim_test::makeTensor(graph, {256, 128, 1, 1}, {128, 1, 128, 128}, 2);

    auto y = graph.conv_fprop(x, w, fe::graph::Conv_fprop_attributes{});
    ASSERT_NE(y, nullptr);
    y->set_output(true).set_uid(3);

    expectValidateFails(graph, fe::error_code_t::ATTRIBUTE_NOT_SET);
}

// conv_dgrad output dim cannot be inferred from the attributes alone; omitting
// the explicit dx dim fails with ATTRIBUTE_NOT_SET (mirrors the happy-path test
// that sets it, proving the constraint is real).
TEST(TestCudnnShimGraphNodes, ConvDgradMissingOutputDimFailsValidation)
{
    fe::graph::Graph graph;
    setFloatGraphTypes(graph);

    auto dy = hipdnn_shim_test::makeTensor(graph, {1, 64, 32, 32}, {65536, 1024, 32, 1}, 1);
    auto w = hipdnn_shim_test::makeTensor(graph, {64, 3, 3, 3}, {27, 9, 3, 1}, 2);

    auto dx = graph.conv_dgrad(
        dy,
        w,
        fe::graph::Conv_dgrad_attributes{}.set_padding({1, 1}).set_stride({1, 1}).set_dilation(
            {1, 1}));
    ASSERT_NE(dx, nullptr);
    dx->set_output(true).set_uid(3); // deliberately no set_dim

    expectValidateFails(graph, fe::error_code_t::ATTRIBUTE_NOT_SET);
}

// Two graph tensors sharing a UID is a structural error caught before any node
// validation: INVALID_VALUE.
TEST(TestCudnnShimGraphNodes, DuplicateTensorUidFailsValidation)
{
    const int64_t n = 4;

    fe::graph::Graph graph;
    setFloatGraphTypes(graph);

    auto a = hipdnn_shim_test::makeTensor(graph, {n, n, n, n}, {n * n * n, n * n, n, 1}, 1);
    auto b = hipdnn_shim_test::makeTensor(graph, {n, n, n, n}, {n * n * n, n * n, n, 1}, 1);

    auto c = graph.pointwise(
        a, b, fe::graph::Pointwise_attributes{}.set_mode(fe::PointwiseMode_t::ADD));
    ASSERT_NE(c, nullptr);
    c->set_output(true).set_uid(2);

    expectValidateFails(graph, fe::error_code_t::INVALID_VALUE);
}

// A non-positive tensor dimension is rejected by tensor validation: INVALID_VALUE.
TEST(TestCudnnShimGraphNodes, NonPositiveDimFailsValidation)
{
    const int64_t n = 4;

    fe::graph::Graph graph;
    setFloatGraphTypes(graph);

    auto a = hipdnn_shim_test::makeTensor(graph, {n, 0, n, n}, {n * n * n, n * n, n, 1}, 1);
    auto b = hipdnn_shim_test::makeTensor(graph, {n, n, n, n}, {n * n * n, n * n, n, 1}, 2);

    auto c = graph.pointwise(
        a, b, fe::graph::Pointwise_attributes{}.set_mode(fe::PointwiseMode_t::ADD));
    ASSERT_NE(c, nullptr);
    c->set_output(true).set_uid(3);

    expectValidateFails(graph, fe::error_code_t::INVALID_VALUE);
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

// Two sdpa_fp8 overloads (FP8 vs MXFP8) resolve by tensor-arg count.
TEST(TestCudnnShimGraphNodes, SdpaFp8Fp8RecordsGraphNotSupported)
{
    fe::graph::Graph graph;
    graph.sdpa_fp8(nullptr,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr,
                   fe::graph::SDPA_fp8_attributes{});
    expectGraphNotSupported(graph);
}

TEST(TestCudnnShimGraphNodes, SdpaFp8Mxfp8RecordsGraphNotSupported)
{
    fe::graph::Graph graph;
    graph.sdpa_fp8(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, fe::graph::SDPA_fp8_attributes{});
    expectGraphNotSupported(graph);
}

// sdpa_fp8_backward mirrors upstream's two overloads, which differ only in
// tensor-arg count: 18 args -> 7 outputs (FP8), 17 -> 6 (MXFP8). Both the call
// resolving and the output count are source compatibility, so pin both.
TEST(TestCudnnShimGraphNodes, SdpaFp8BackwardFp8RecordsGraphNotSupported)
{
    fe::graph::Graph graph;
    auto outputs = graph.sdpa_fp8_backward(nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           fe::graph::SDPA_fp8_backward_attributes{});
    EXPECT_EQ(outputs.size(), 7U);
    expectGraphNotSupported(graph);
}

TEST(TestCudnnShimGraphNodes, SdpaFp8BackwardMxfp8RecordsGraphNotSupported)
{
    fe::graph::Graph graph;
    auto outputs = graph.sdpa_fp8_backward(nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           fe::graph::SDPA_fp8_backward_attributes{});
    EXPECT_EQ(outputs.size(), 6U);
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

// The single-output fail-stub returns a live placeholder, not null: the
// idiomatic ->set_output(...).set_uid(...) chain (which used to segfault) is
// safe, and the recorded GRAPH_NOT_SUPPORTED still wins at validate().
TEST(TestCudnnShimGraphNodes, ReshapeReturnsChainablePlaceholder)
{
    const int64_t n = 4;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF).set_compute_data_type(fe::DataType_t::FLOAT);

    auto input = graph.tensor(fe::graph::Tensor_attributes{}
                                  .set_dim({n, n, n, n})
                                  .set_stride({n * n * n, n * n, n, 1})
                                  .set_uid(1));

    auto out = graph.reshape(input, fe::graph::Reshape_attributes{});
    ASSERT_NE(out, nullptr);
    out->set_output(true).set_uid(2);

    expectGraphNotSupported(graph);
}

// Every element of a multi-output fail-stub's returned array is a live
// placeholder, so per-element chaining is safe before the error surfaces.
TEST(TestCudnnShimGraphNodes, GenstatsReturnsChainablePlaceholders)
{
    const int64_t n = 4;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF).set_compute_data_type(fe::DataType_t::FLOAT);

    auto input = graph.tensor(fe::graph::Tensor_attributes{}
                                  .set_dim({n, n, n, n})
                                  .set_stride({n * n * n, n * n, n, 1})
                                  .set_uid(1));

    auto outs = graph.genstats(input, fe::graph::Genstats_attributes{});
    int64_t uid = 2;
    for(auto& out : outs)
    {
        ASSERT_NE(out, nullptr);
        out->set_output(true).set_uid(uid++);
    }

    expectGraphNotSupported(graph);
}

// A wider multi-output shape: the 9-arg (FP8) sdpa_fp8 returns an array of four
// live placeholders, each safe to chain before the error surfaces.
TEST(TestCudnnShimGraphNodes, SdpaFp8ReturnsChainablePlaceholders)
{
    fe::graph::Graph graph;

    auto outs = graph.sdpa_fp8(nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               fe::graph::SDPA_fp8_attributes{});
    int64_t uid = 1;
    for(auto& out : outs)
    {
        ASSERT_NE(out, nullptr);
        out->set_output(true).set_uid(uid++);
    }

    expectGraphNotSupported(graph);
}

// --- (c) All 39 attribute classes construct and expose the universal surface -

// Every cuDNN v9 *_attributes class — whether a 1:1 hipDNN alias, an SDPA alias,
// or a fail-stub/attribute-only stub — must default-construct and expose the
// universal fluent accessors (.set_name/.get_name,
// .set_compute_data_type/.get_compute_data_type). A missing alias, stub, or
// accessor fails to compile here.
template <typename Attributes>
void expectConstructsAndExposesUniversalSurface()
{
    Attributes attributes;
    attributes.set_name("node");
    EXPECT_EQ(attributes.get_name(), "node");

    attributes.set_compute_data_type(fe::DataType_t::FLOAT);
    EXPECT_EQ(attributes.get_compute_data_type(), fe::DataType_t::FLOAT);
}

TEST(TestCudnnShimGraphNodes, AllAttributeClassesExposeUniversalSurface)
{
    // Tier-1 aliases (real hipDNN engine).
    expectConstructsAndExposesUniversalSurface<fe::graph::Batchnorm_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::Batchnorm_backward_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::Batchnorm_inference_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::Block_scale_dequantize_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::Block_scale_quantize_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::Conv_dgrad_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::Conv_fprop_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::Conv_wgrad_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::Layernorm_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::Layernorm_backward_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::Matmul_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::Pointwise_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::Reduction_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::Resample_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::Rmsnorm_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::Rmsnorm_backward_attributes>();

#ifdef HIPDNN_ENABLE_SDPA
    // SDPA aliases (only present in an SDPA-enabled build).
    expectConstructsAndExposesUniversalSurface<fe::graph::SDPA_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::SDPA_backward_attributes>();
#endif

    // Tier-2 fail-stub attribute classes.
    expectConstructsAndExposesUniversalSurface<fe::graph::BN_finalize_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::Genstats_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::DBN_weight_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::Instancenorm_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::Instancenorm_backward_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::AdaLayernorm_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::AdaLayernorm_backward_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::Rng_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::Reshape_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::Transpose_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::RoPE_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::RoPE_backward_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::SDPA_fp8_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::SDPA_fp8_backward_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::DiagonalBandMask_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::Slice_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::Concatenate_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::Moe_grouped_matmul_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::Moe_grouped_matmul_bwd_attributes>();

    // Attribute-only stubs (no Graph method): must still exist and chain.
    expectConstructsAndExposesUniversalSurface<fe::graph::Matmul_fp8_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::Softmax_attributes>();
    expectConstructsAndExposesUniversalSurface<fe::graph::PagedCacheLoad_attributes>();
}

} // namespace
