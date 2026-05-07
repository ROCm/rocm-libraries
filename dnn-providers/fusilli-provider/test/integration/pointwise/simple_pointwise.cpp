// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <cstdint>
#include <memory>

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;

class PointwiseIntegrationTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_EQ(hipInit(0), hipSuccess);
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    auto pluginPath = std::filesystem::canonical(
        getCurrentExecutableDirectory() / FUSILLI_PLUGIN_PATH);
    const std::array<const char *, 1> paths = {pluginPath.c_str()};
    ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(),
                                             HIPDNN_PLUGIN_LOADING_ABSOLUTE),
              HIPDNN_STATUS_SUCCESS);

    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnSetStream(handle, stream), HIPDNN_STATUS_SUCCESS);
  }

  void TearDown() override {
    ASSERT_EQ(hipStreamDestroy(stream), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
  }

  // Run an end-to-end unary pointwise op with a uniform input fill, and check
  // the output matches the uniform expected value. Used to give each unary op
  // its own minimal smoke test without copy-pasting the full graph boilerplate.
  void runUnaryPointwiseTest(const char *graphName,
                             hipdnn_frontend::PointwiseMode_t mode,
                             float inputValue, float expectedValue) {
    const int64_t M = 4;
    const int64_t N = 8;
    const int64_t inputUID = 0;
    const int64_t outputUID = 1;

    PinnedTensor<float> inputTensor({M, N});
    PinnedTensor<float> outputTensor({M, N});
    inputTensor.fillWithValue(inputValue);
    outputTensor.fillWithValue(-100.0f);

    PinnedTensor<float> expectedOutput({M, N});
    expectedOutput.fillWithValue(expectedValue);

    auto graph = std::make_shared<graph::Graph>();
    graph->set_name(graphName);
    graph->set_io_data_type(DataType_t::FLOAT)
        .set_intermediate_data_type(DataType_t::FLOAT)
        .set_compute_data_type(DataType_t::FLOAT);

    auto inputAttr = std::make_shared<graph::TensorAttributes>(
        graph::makeTensorAttributes("input", DataType_t::FLOAT, inputTensor));
    inputAttr->set_uid(inputUID);

    graph::PointwiseAttributes pwAttr;
    pwAttr.set_name(graphName).set_mode(mode);

    auto outputAttr = graph->pointwise(inputAttr, pwAttr);
    outputAttr->set_uid(outputUID);
    outputAttr->set_dim(outputTensor.dims())
        .set_stride(outputTensor.strides())
        .set_output(true);

    auto result = graph->validate();
    ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
    result = graph->build_operation_graph(handle);
    ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
    result = graph->create_execution_plans();
    ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
    result = graph->check_support();
    ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
    result = graph->build_plans();
    ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

    std::unordered_map<int64_t, void *> variantPack;
    variantPack[inputUID] = inputTensor.memory().deviceData();
    variantPack[outputUID] = outputTensor.memory().deviceData();

    result = graph->execute(handle, variantPack, nullptr);
    ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
    outputTensor.memory().markDeviceModified();

    CpuFpReferenceValidation<float> validator(1e-5f, 1e-5f);
    EXPECT_TRUE(validator.allClose(expectedOutput, outputTensor));
  }

  // Run an end-to-end binary pointwise op with uniform input fills, and check
  // the output matches the uniform expected value.
  void runBinaryPointwiseTest(const char *graphName,
                              hipdnn_frontend::PointwiseMode_t mode,
                              float in0Value, float in1Value,
                              float expectedValue) {
    const int64_t M = 4;
    const int64_t N = 8;
    const int64_t in0UID = 0;
    const int64_t in1UID = 1;
    const int64_t outputUID = 2;

    PinnedTensor<float> in0Tensor({M, N});
    PinnedTensor<float> in1Tensor({M, N});
    PinnedTensor<float> outputTensor({M, N});
    in0Tensor.fillWithValue(in0Value);
    in1Tensor.fillWithValue(in1Value);
    outputTensor.fillWithValue(-100.0f);

    PinnedTensor<float> expectedOutput({M, N});
    expectedOutput.fillWithValue(expectedValue);

    auto graph = std::make_shared<graph::Graph>();
    graph->set_name(graphName);
    graph->set_io_data_type(DataType_t::FLOAT)
        .set_intermediate_data_type(DataType_t::FLOAT)
        .set_compute_data_type(DataType_t::FLOAT);

    auto in0Attr = std::make_shared<graph::TensorAttributes>(
        graph::makeTensorAttributes("in0", DataType_t::FLOAT, in0Tensor));
    in0Attr->set_uid(in0UID);
    auto in1Attr = std::make_shared<graph::TensorAttributes>(
        graph::makeTensorAttributes("in1", DataType_t::FLOAT, in1Tensor));
    in1Attr->set_uid(in1UID);

    graph::PointwiseAttributes pwAttr;
    pwAttr.set_name(graphName).set_mode(mode);

    auto outputAttr = graph->pointwise(in0Attr, in1Attr, pwAttr);
    outputAttr->set_uid(outputUID);
    outputAttr->set_dim(outputTensor.dims())
        .set_stride(outputTensor.strides())
        .set_output(true);

    auto result = graph->validate();
    ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
    result = graph->build_operation_graph(handle);
    ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
    result = graph->create_execution_plans();
    ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
    result = graph->check_support();
    ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
    result = graph->build_plans();
    ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

    std::unordered_map<int64_t, void *> variantPack;
    variantPack[in0UID] = in0Tensor.memory().deviceData();
    variantPack[in1UID] = in1Tensor.memory().deviceData();
    variantPack[outputUID] = outputTensor.memory().deviceData();

    result = graph->execute(handle, variantPack, nullptr);
    ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
    outputTensor.memory().markDeviceModified();

    CpuFpReferenceValidation<float> validator(1e-5f, 1e-5f);
    EXPECT_TRUE(validator.allClose(expectedOutput, outputTensor));
  }

  hipStream_t stream = nullptr;
  hipdnnHandle_t handle;
};

// Test: Standalone pointwise ReLU forward (unary)
// Graph: input[4,8] -> pointwise(RELU_FWD) -> output[4,8]
TEST_F(PointwiseIntegrationTest, SimpleReluFwd) {
  // ReLU clamps negatives to 0.
  runUnaryPointwiseTest("simple_relu_fwd_test", PointwiseMode_t::RELU_FWD,
                        -3.0f, 0.0f);
}

// Test: Standalone pointwise ADD (binary)
// Graph: in0[4,8] + in1[4,8] -> output[4,8]
TEST_F(PointwiseIntegrationTest, SimpleAdd) {
  runBinaryPointwiseTest("simple_add_test", PointwiseMode_t::ADD, 3.0f, 5.0f,
                         8.0f);
}

// Smoke tests for the remaining unary fusilli pointwise operators. Each test
// picks an input value where the expected output is exact (or near-exact) so
// the assertion uses a tight tolerance.
TEST_F(PointwiseIntegrationTest, SimpleAbs) {
  runUnaryPointwiseTest("simple_abs", PointwiseMode_t::ABS, -3.0f, 3.0f);
}

TEST_F(PointwiseIntegrationTest, SimpleCeil) {
  runUnaryPointwiseTest("simple_ceil", PointwiseMode_t::CEIL, 2.3f, 3.0f);
}

TEST_F(PointwiseIntegrationTest, SimpleErf) {
  // erf(0) = 0.
  runUnaryPointwiseTest("simple_erf", PointwiseMode_t::ERF, 0.0f, 0.0f);
}

TEST_F(PointwiseIntegrationTest, SimpleExp) {
  // exp(0) = 1.
  runUnaryPointwiseTest("simple_exp", PointwiseMode_t::EXP, 0.0f, 1.0f);
}

TEST_F(PointwiseIntegrationTest, SimpleFloor) {
  runUnaryPointwiseTest("simple_floor", PointwiseMode_t::FLOOR, 2.7f, 2.0f);
}

TEST_F(PointwiseIntegrationTest, SimpleNeg) {
  runUnaryPointwiseTest("simple_neg", PointwiseMode_t::NEG, 4.0f, -4.0f);
}

TEST_F(PointwiseIntegrationTest, SimpleReciprocal) {
  runUnaryPointwiseTest("simple_reciprocal", PointwiseMode_t::RECIPROCAL, 4.0f,
                        0.25f);
}

TEST_F(PointwiseIntegrationTest, SimpleSigmoidFwd) {
  // sigmoid(0) = 0.5.
  runUnaryPointwiseTest("simple_sigmoid_fwd", PointwiseMode_t::SIGMOID_FWD,
                        0.0f, 0.5f);
}

TEST_F(PointwiseIntegrationTest, SimpleTanhFwd) {
  // tanh(0) = 0.
  runUnaryPointwiseTest("simple_tanh_fwd", PointwiseMode_t::TANH_FWD, 0.0f,
                        0.0f);
}

// Smoke tests for the remaining binary fusilli pointwise operators.
TEST_F(PointwiseIntegrationTest, SimpleDiv) {
  runBinaryPointwiseTest("simple_div", PointwiseMode_t::DIV, 8.0f, 4.0f, 2.0f);
}

TEST_F(PointwiseIntegrationTest, SimpleMax) {
  runBinaryPointwiseTest("simple_max", PointwiseMode_t::MAX, 3.0f, 5.0f, 5.0f);
}

TEST_F(PointwiseIntegrationTest, SimpleMin) {
  runBinaryPointwiseTest("simple_min", PointwiseMode_t::MIN, 3.0f, 5.0f, 3.0f);
}

TEST_F(PointwiseIntegrationTest, SimpleMul) {
  runBinaryPointwiseTest("simple_mul", PointwiseMode_t::MUL, 3.0f, 5.0f, 15.0f);
}

TEST_F(PointwiseIntegrationTest, SimpleSub) {
  runBinaryPointwiseTest("simple_sub", PointwiseMode_t::SUB, 8.0f, 3.0f, 5.0f);
}
