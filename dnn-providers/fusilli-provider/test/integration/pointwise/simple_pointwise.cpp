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

  hipStream_t stream = nullptr;
  hipdnnHandle_t handle;
};

// Test: Standalone pointwise ReLU forward (unary)
// Graph: input[4,8] -> pointwise(RELU_FWD) -> output[4,8]
TEST_F(PointwiseIntegrationTest, SimpleReluFwd) {
  // Dimensions.
  const int64_t M = 4;
  const int64_t N = 8;

  // UIDs.
  const int64_t inputUID = 0;
  const int64_t outputUID = 1;

  // Initialize tensors.
  PinnedTensor<float> inputTensor({M, N});
  PinnedTensor<float> outputTensor({M, N});
  inputTensor.fillWithValue(-3.0f);
  outputTensor.fillWithValue(-100.0f);

  // Expected output: ReLU clamps negatives to 0.
  PinnedTensor<float> expectedOutput({M, N});
  expectedOutput.fillWithValue(0.0f);

  // Create graph.
  auto graph = std::make_shared<graph::Graph>();
  graph->set_name("simple_relu_fwd_test");
  graph->set_io_data_type(DataType_t::FLOAT)
      .set_intermediate_data_type(DataType_t::FLOAT)
      .set_compute_data_type(DataType_t::FLOAT);

  // Create tensor attributes.
  auto inputAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("input", DataType_t::FLOAT, inputTensor));
  inputAttr->set_uid(inputUID);

  // Create pointwise ReLU attributes.
  graph::PointwiseAttributes reluAttr;
  reluAttr.set_name("relu").set_mode(PointwiseMode_t::RELU_FWD);

  // Create pointwise node (unary).
  auto outputAttr = graph->pointwise(inputAttr, reluAttr);
  outputAttr->set_uid(outputUID);
  outputAttr->set_dim(outputTensor.dims())
      .set_stride(outputTensor.strides())
      .set_output(true);

  // Build + validate + build plans for graph.
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

  // Create variant pack.
  std::unordered_map<int64_t, void *> variantPack;
  variantPack[inputUID] = inputTensor.memory().deviceData();
  variantPack[outputUID] = outputTensor.memory().deviceData();

  // Execute graph.
  result = graph->execute(handle, variantPack, nullptr);
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
  outputTensor.memory().markDeviceModified();

  // Check results.
  CpuFpReferenceValidation<float> validator(1e-6f, 1e-6f);
  EXPECT_TRUE(validator.allClose(expectedOutput, outputTensor));
}

// Test: Standalone pointwise ADD (binary)
// Graph: in0[4,8] + in1[4,8] -> output[4,8]
TEST_F(PointwiseIntegrationTest, SimpleAdd) {
  // Dimensions.
  const int64_t M = 4;
  const int64_t N = 8;

  // UIDs.
  const int64_t in0UID = 0;
  const int64_t in1UID = 1;
  const int64_t outputUID = 2;

  // Initialize tensors.
  PinnedTensor<float> in0Tensor({M, N});
  PinnedTensor<float> in1Tensor({M, N});
  PinnedTensor<float> outputTensor({M, N});
  in0Tensor.fillWithValue(3.0f);
  in1Tensor.fillWithValue(5.0f);
  outputTensor.fillWithValue(-100.0f);

  // Expected output: 3.0 + 5.0 = 8.0.
  PinnedTensor<float> expectedOutput({M, N});
  expectedOutput.fillWithValue(8.0f);

  // Create graph.
  auto graph = std::make_shared<graph::Graph>();
  graph->set_name("simple_add_test");
  graph->set_io_data_type(DataType_t::FLOAT)
      .set_intermediate_data_type(DataType_t::FLOAT)
      .set_compute_data_type(DataType_t::FLOAT);

  // Create tensor attributes.
  auto in0Attr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("in0", DataType_t::FLOAT, in0Tensor));
  in0Attr->set_uid(in0UID);
  auto in1Attr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("in1", DataType_t::FLOAT, in1Tensor));
  in1Attr->set_uid(in1UID);

  // Create pointwise ADD attributes.
  graph::PointwiseAttributes addAttr;
  addAttr.set_name("add").set_mode(PointwiseMode_t::ADD);

  // Create pointwise node (binary).
  auto outputAttr = graph->pointwise(in0Attr, in1Attr, addAttr);
  outputAttr->set_uid(outputUID);
  outputAttr->set_dim(outputTensor.dims())
      .set_stride(outputTensor.strides())
      .set_output(true);

  // Build + validate + build plans for graph.
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

  // Create variant pack.
  std::unordered_map<int64_t, void *> variantPack;
  variantPack[in0UID] = in0Tensor.memory().deviceData();
  variantPack[in1UID] = in1Tensor.memory().deviceData();
  variantPack[outputUID] = outputTensor.memory().deviceData();

  // Execute graph.
  result = graph->execute(handle, variantPack, nullptr);
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
  outputTensor.memory().markDeviceModified();

  // Check results.
  CpuFpReferenceValidation<float> validator(1e-6f, 1e-6f);
  EXPECT_TRUE(validator.allClose(expectedOutput, outputTensor));
}
