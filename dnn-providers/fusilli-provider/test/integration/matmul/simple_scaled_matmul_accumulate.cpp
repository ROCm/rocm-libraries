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
#include <hipdnn_frontend/attributes/MatmulAttributes.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <cstdint>
#include <memory>

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;

// Test: Scaled matrix multiplication with accumulation
// Computes: result = alpha * (A @ B) + beta * C
//
// Graph structure:
//   matmul_out = matmul(A[M,K], B[K,N])              -> [M,N]
//   scaled     = pointwise_mul(matmul_out, alpha)     -> alpha * matmul_out
//   accum      = pointwise_mul(C, beta)               -> beta * C
//   result     = pointwise_add(scaled, accum)         -> alpha*(A@B) + beta*C
//
// alpha and beta are pass-by-value scalars, NOT device buffers.
TEST(MatmulIntegrationTest, ScaledMatmulAccumulate) {
  // Initialize HIP.
  ASSERT_EQ(hipInit(0), hipSuccess);
  ASSERT_EQ(hipSetDevice(0), hipSuccess);

  hipStream_t stream = nullptr;
  ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

  // Set plugin paths.
  auto pluginPath = std::filesystem::canonical(getCurrentExecutableDirectory() /
                                               FUSILLI_PLUGIN_PATH);
  const std::array<const char *, 1> paths = {pluginPath.c_str()};
  ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(),
                                           HIPDNN_PLUGIN_LOADING_ABSOLUTE),
            HIPDNN_STATUS_SUCCESS);

  // Create handle.
  hipdnnHandle_t handle;
  ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);
  ASSERT_EQ(hipdnnSetStream(handle, stream), HIPDNN_STATUS_SUCCESS);

  // Dimensions: A[4,8] x B[8,5] = matmul_out[4,5]
  const int64_t M = 4;
  const int64_t K = 8;
  const int64_t N = 5;

  // Scalar values.
  const float alphaVal = 2.0f;
  const float betaVal = 3.0f;

  // UIDs for device-backed tensors.
  const int64_t aUID = 0;
  const int64_t bUID = 1;
  const int64_t cUID = 2;
  const int64_t resultUID = 3;

  // UIDs for scalars (pass-by-value, NOT in variant pack).
  const int64_t alphaUID = 4;
  const int64_t betaUID = 5;

  // Initialize device tensors.
  PinnedTensor<float> aTensor({M, K});
  PinnedTensor<float> bTensor({K, N});
  PinnedTensor<float> cTensor({M, N});
  PinnedTensor<float> resultTensor({M, N});
  aTensor.fillWithValue(1.0f);
  bTensor.fillWithValue(1.0f);
  cTensor.fillWithValue(5.0f);
  resultTensor.fillWithValue(-100.0f);

  // Expected: alpha * (A@B) + beta * C
  //         = 2.0 * (K * 1.0 * 1.0) + 3.0 * 5.0
  //         = 2.0 * 8.0 + 15.0
  //         = 31.0
  PinnedTensor<float> expectedOutput({M, N});
  expectedOutput.fillWithValue(alphaVal * static_cast<float>(K) +
                               betaVal * 5.0f);

  // Create graph.
  auto graph = std::make_shared<graph::Graph>();
  graph->set_name("scaled_matmul_accumulate_test");
  graph->set_io_data_type(DataType_t::FLOAT)
      .set_intermediate_data_type(DataType_t::FLOAT)
      .set_compute_data_type(DataType_t::FLOAT);

  // Create device-backed tensor attributes.
  auto aAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("A", DataType_t::FLOAT, aTensor));
  aAttr->set_uid(aUID);

  auto bAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("B", DataType_t::FLOAT, bTensor));
  bAttr->set_uid(bUID);

  auto cAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("C", DataType_t::FLOAT, cTensor));
  cAttr->set_uid(cUID);

  // Create scalar attributes (pass-by-value, NOT device buffers).
  auto alphaAttr = std::make_shared<graph::TensorAttributes>();
  alphaAttr->set_name("alpha").set_value(alphaVal).set_uid(alphaUID);

  auto betaAttr = std::make_shared<graph::TensorAttributes>();
  betaAttr->set_name("beta").set_value(betaVal).set_uid(betaUID);

  // Build graph: result = alpha * (A @ B) + beta * C

  // matmul_out = A @ B
  graph::MatmulAttributes matmulAttr;
  matmulAttr.set_name("matmul");
  auto matmulOutAttr = graph->matmul(aAttr, bAttr, matmulAttr);
  matmulOutAttr->set_dim(resultTensor.dims())
      .set_stride(resultTensor.strides())
      .set_output(false);

  // scaled = matmul_out * alpha
  graph::PointwiseAttributes mulAlphaAttr;
  mulAlphaAttr.set_name("mul_alpha").set_mode(PointwiseMode_t::MUL);
  auto scaledAttr = graph->pointwise(matmulOutAttr, alphaAttr, mulAlphaAttr);
  scaledAttr->set_dim(resultTensor.dims())
      .set_stride(resultTensor.strides())
      .set_output(false);

  // accum = C * beta
  graph::PointwiseAttributes mulBetaAttr;
  mulBetaAttr.set_name("mul_beta").set_mode(PointwiseMode_t::MUL);
  auto accumAttr = graph->pointwise(cAttr, betaAttr, mulBetaAttr);
  accumAttr->set_dim(resultTensor.dims())
      .set_stride(resultTensor.strides())
      .set_output(false);

  // result = scaled + accum
  graph::PointwiseAttributes addAttr;
  addAttr.set_name("add").set_mode(PointwiseMode_t::ADD);
  auto resultAttr = graph->pointwise(scaledAttr, accumAttr, addAttr);
  resultAttr->set_uid(resultUID);
  resultAttr->set_dim(resultTensor.dims())
      .set_stride(resultTensor.strides())
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

  // Create variant pack for device-backed tensors, not scalars.
  std::unordered_map<int64_t, void *> variantPack;
  variantPack[aUID] = aTensor.memory().deviceData();
  variantPack[bUID] = bTensor.memory().deviceData();
  variantPack[cUID] = cTensor.memory().deviceData();
  variantPack[resultUID] = resultTensor.memory().deviceData();

  // Execute graph.
  result = graph->execute(handle, variantPack, nullptr);
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
  resultTensor.memory().markDeviceModified();

  // Check results.
  CpuFpReferenceValidation<float> validator(1e-6f, 1e-6f);
  EXPECT_TRUE(validator.allClose(expectedOutput, resultTensor));

  // Clean up.
  ASSERT_EQ(hipStreamDestroy(stream), HIPDNN_STATUS_SUCCESS);
  ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}
