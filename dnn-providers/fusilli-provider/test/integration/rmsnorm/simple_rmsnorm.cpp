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
#include <hipdnn_frontend/attributes/RMSNormAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <cmath>
#include <cstdint>
#include <memory>

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;

class RmsnormIntegrationTest : public ::testing::Test {
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

// Test: Simple inference-phase RMSNorm.
// Graph: x[N, C] -> rmsnorm(scale, epsilon) -> y[N, C]
//
// All elements of x are set to a constant value `v`, so
//   mean(x^2) = v^2
//   rms       = sqrt(v^2 + epsilon) ≈ |v|
//   y[b, c]   = x[b, c] * scale[c] / rms = scale[c]  (for v > 0, eps small)
// With scale = 1.0 the expected output is uniformly 1.0.
TEST_F(RmsnormIntegrationTest, SimpleRmsnormInference) {
  // Dimensions.
  const int64_t N = 2;
  const int64_t C = 4;

  // UIDs.
  const int64_t xUID = 0;
  const int64_t scaleUID = 1;
  const int64_t epsilonUID = 2;
  const int64_t yUID = 3;

  // Initialize tensors.
  PinnedTensor<float> xTensor({N, C});
  PinnedTensor<float> scaleTensor({1, C});
  PinnedTensor<float> yTensor({N, C});
  xTensor.fillWithValue(2.0f);
  scaleTensor.fillWithValue(1.0f);
  yTensor.fillWithValue(-100.0f);

  // For x = 2, scale = 1, eps small: y ≈ 1.0 everywhere.
  PinnedTensor<float> expectedOutput({N, C});
  expectedOutput.fillWithValue(1.0f);

  // Create graph.
  auto graph = std::make_shared<graph::Graph>();
  graph->set_name("simple_rmsnorm_inference_test");
  graph->set_io_data_type(DataType_t::FLOAT)
      .set_intermediate_data_type(DataType_t::FLOAT)
      .set_compute_data_type(DataType_t::FLOAT);

  // Create tensor attributes.
  auto xAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("x", DataType_t::FLOAT, xTensor));
  xAttr->set_uid(xUID);

  auto scaleAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("scale", DataType_t::FLOAT, scaleTensor));
  scaleAttr->set_uid(scaleUID);

  // Epsilon (pass-by-value scalar).
  auto epsilonAttr = std::make_shared<graph::TensorAttributes>();
  epsilonAttr->set_name("epsilon").set_value(1e-5f).set_uid(epsilonUID);

  // Create RMSNorm attributes.
  graph::RMSNormAttributes rmsnormAttr;
  rmsnormAttr.set_name("rmsnorm")
      .set_epsilon(epsilonAttr)
      .set_forward_phase(NormFwdPhase::INFERENCE);

  // Create RMSNorm node.
  auto [yAttr, invRmsAttr] = graph->rmsnorm(xAttr, scaleAttr, rmsnormAttr);
  yAttr->set_uid(yUID)
      .set_data_type(DataType_t::FLOAT)
      .set_dim({N, C})
      .set_stride({C, 1})
      .set_output(true);

  // Build + validate + build plans for graph.
  auto result = graph->validate();
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

  result = graph->build(handle);
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
  variantPack[xUID] = xTensor.memory().deviceData();
  variantPack[scaleUID] = scaleTensor.memory().deviceData();
  variantPack[yUID] = yTensor.memory().deviceData();

  // Execute graph.
  result = graph->execute(handle, variantPack, workspace);
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
  yTensor.memory().markDeviceModified();

  if (workspace) {
    ASSERT_EQ(hipFree(workspace), hipSuccess);
  }

  // Check results.
  CpuFpReferenceValidation<float> validator(1e-4f, 1e-4f);
  EXPECT_TRUE(validator.allClose(expectedOutput, yTensor));
}

// Test: Inference-phase RMSNorm with a non-trivial scale.
// Graph: x[N, C] -> rmsnorm(scale, epsilon) -> y[N, C]
//
// With x filled with a constant `v` the normalized activation is 1, so the
// output reduces to `scale[c]` per channel. This exercises the per-channel
// scale broadcast in addition to the normalization.
TEST_F(RmsnormIntegrationTest, RmsnormInferenceWithScale) {
  // Dimensions.
  const int64_t N = 3;
  const int64_t C = 4;

  // UIDs.
  const int64_t xUID = 0;
  const int64_t scaleUID = 1;
  const int64_t epsilonUID = 2;
  const int64_t yUID = 3;

  // Initialize tensors.
  PinnedTensor<float> xTensor({N, C});
  PinnedTensor<float> scaleTensor({1, C});
  PinnedTensor<float> yTensor({N, C});
  xTensor.fillWithValue(3.0f);
  yTensor.fillWithValue(-100.0f);

  // Per-channel scale.
  float *scaleHost = scaleTensor.memory().hostData();
  const std::array<float, 4> scaleValues = {0.5f, 1.0f, 2.0f, 4.0f};
  for (int64_t c = 0; c < C; ++c) {
    scaleHost[c] = scaleValues[static_cast<size_t>(c)];
  }
  scaleTensor.memory().markHostModified();

  // Expected: y[b, c] = x[b, c] * scale[c] / rms ≈ 1.0 * scale[c] = scale[c].
  PinnedTensor<float> expectedOutput({N, C});
  float *expectedHost = expectedOutput.memory().hostData();
  for (int64_t b = 0; b < N; ++b) {
    for (int64_t c = 0; c < C; ++c) {
      expectedHost[b * C + c] = scaleValues[static_cast<size_t>(c)];
    }
  }
  expectedOutput.memory().markHostModified();

  // Create graph.
  auto graph = std::make_shared<graph::Graph>();
  graph->set_name("rmsnorm_inference_with_scale_test");
  graph->set_io_data_type(DataType_t::FLOAT)
      .set_intermediate_data_type(DataType_t::FLOAT)
      .set_compute_data_type(DataType_t::FLOAT);

  // Create tensor attributes.
  auto xAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("x", DataType_t::FLOAT, xTensor));
  xAttr->set_uid(xUID);

  auto scaleAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("scale", DataType_t::FLOAT, scaleTensor));
  scaleAttr->set_uid(scaleUID);

  // Epsilon (pass-by-value scalar).
  auto epsilonAttr = std::make_shared<graph::TensorAttributes>();
  epsilonAttr->set_name("epsilon").set_value(1e-5f).set_uid(epsilonUID);

  // Create RMSNorm attributes.
  graph::RMSNormAttributes rmsnormAttr;
  rmsnormAttr.set_name("rmsnorm")
      .set_epsilon(epsilonAttr)
      .set_forward_phase(NormFwdPhase::INFERENCE);

  // Create RMSNorm node.
  auto [yAttr, invRmsAttr] = graph->rmsnorm(xAttr, scaleAttr, rmsnormAttr);
  yAttr->set_uid(yUID)
      .set_data_type(DataType_t::FLOAT)
      .set_dim({N, C})
      .set_stride({C, 1})
      .set_output(true);

  // Build + validate + build plans for graph.
  auto result = graph->validate();
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

  result = graph->build(handle);
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
  variantPack[xUID] = xTensor.memory().deviceData();
  variantPack[scaleUID] = scaleTensor.memory().deviceData();
  variantPack[yUID] = yTensor.memory().deviceData();

  // Execute graph.
  result = graph->execute(handle, variantPack, workspace);
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
  yTensor.memory().markDeviceModified();

  if (workspace) {
    ASSERT_EQ(hipFree(workspace), hipSuccess);
  }

  // Check results.
  CpuFpReferenceValidation<float> validator(1e-4f, 1e-4f);
  EXPECT_TRUE(validator.allClose(expectedOutput, yTensor));
}
