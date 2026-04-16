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
#include <hipdnn_frontend/attributes/ReductionAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <cstdint>
#include <memory>

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;

class ReductionIntegrationTest : public ::testing::Test {
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

// Test: Sum reduction over first dimension
// Graph: input[4,8] -> reduction(ADD) -> output[1,8]
TEST_F(ReductionIntegrationTest, SimpleSumReduction) {
  // Dimensions.
  const int64_t M = 4;
  const int64_t N = 8;

  // UIDs.
  const int64_t inputUID = 0;
  const int64_t outputUID = 1;

  // Initialize tensors.
  PinnedTensor<float> inputTensor({M, N});
  PinnedTensor<float> outputTensor({1, N});
  inputTensor.fillWithValue(1.0f);
  outputTensor.fillWithValue(-100.0f);

  // Expected output: sum over dim 0 of all-ones [4,8] = [4.0, ...] shape [1,8]
  PinnedTensor<float> expectedOutput({1, N});
  expectedOutput.fillWithValue(4.0f);

  // Create graph.
  auto graph = std::make_shared<graph::Graph>();
  graph->set_name("simple_sum_reduction_test");
  graph->set_io_data_type(DataType_t::FLOAT)
      .set_intermediate_data_type(DataType_t::FLOAT)
      .set_compute_data_type(DataType_t::FLOAT);

  // Create tensor attributes.
  auto inputAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("input", DataType_t::FLOAT, inputTensor));
  inputAttr->set_uid(inputUID);

  auto outputAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("output", DataType_t::FLOAT, outputTensor));
  outputAttr->set_uid(outputUID);
  outputAttr->set_output(true);

  // Create reduction attributes.
  graph::ReductionAttributes redAttr;
  redAttr.set_name("sum_reduce").set_mode(ReductionMode_t::ADD);

  // Create reduction node.
  auto yAttr = graph->reduction(inputAttr, outputAttr, redAttr);

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

  // Query workspace size.
  int64_t workspaceSize = 0;
  result = graph->get_workspace_size(workspaceSize);
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

  // Allocate workspace if needed.
  void *workspace = nullptr;
  if (workspaceSize > 0) {
    ASSERT_EQ(hipMalloc(&workspace, static_cast<size_t>(workspaceSize)),
              hipSuccess);
  }

  // Create variant pack.
  std::unordered_map<int64_t, void *> variantPack;
  variantPack[inputUID] = inputTensor.memory().deviceData();
  variantPack[outputUID] = outputTensor.memory().deviceData();

  // Execute graph.
  result = graph->execute(handle, variantPack, workspace);
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
  outputTensor.memory().markDeviceModified();

  if (workspace) {
    ASSERT_EQ(hipFree(workspace), hipSuccess);
  }

  // Check results.
  CpuFpReferenceValidation<float> validator(1e-5f, 1e-5f);
  EXPECT_TRUE(validator.allClose(expectedOutput, outputTensor));
}
