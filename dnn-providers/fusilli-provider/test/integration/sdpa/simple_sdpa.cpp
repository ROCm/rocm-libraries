// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <cstdint>
#include <memory>

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using hipdnn_data_sdk::types::half;

// Test: Simple scaled dot-product attention
// Q[B,H,S,D] x K[B,H,S,D] x V[B,H,S,D] -> O[B,H,S,D]
TEST(SdpaIntegrationTest, SimpleSdpa) {
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

  // Dimensions: [batch=2, heads=4, seq_len=8, head_dim=16]
  const int64_t B = 2;
  const int64_t H = 4;
  const int64_t S = 8;
  const int64_t D = 16;

  // UIDs.
  const int64_t qUID = 0;
  const int64_t kUID = 1;
  const int64_t vUID = 2;
  const int64_t oUID = 3;

  // Initialize tensors.
  PinnedTensor<float> qTensor({B, H, S, D});
  PinnedTensor<float> kTensor({B, H, S, D});
  PinnedTensor<float> vTensor({B, H, S, D});
  PinnedTensor<float> oTensor({B, H, S, D});
  qTensor.fillWithValue(0.1f);
  kTensor.fillWithValue(0.1f);
  vTensor.fillWithValue(0.1f);
  oTensor.fillWithValue(-100.0f);

  // Compute expected output using CPU reference.
  PinnedTensor<float> expectedOutput({B, H, S, D});
  CpuFpReferenceSdpa::forward(qTensor, kTensor, vTensor, expectedOutput);

  // Create graph.
  auto graph = std::make_shared<graph::Graph>();
  graph->set_name("simple_sdpa_test");
  graph->set_io_data_type(DataType_t::FLOAT)
      .set_intermediate_data_type(DataType_t::FLOAT)
      .set_compute_data_type(DataType_t::FLOAT);

  // Create tensor attributes.
  auto qAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("Q", DataType_t::FLOAT, qTensor));
  qAttr->set_uid(qUID);
  auto kAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("K", DataType_t::FLOAT, kTensor));
  kAttr->set_uid(kUID);
  auto vAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("V", DataType_t::FLOAT, vTensor));
  vAttr->set_uid(vUID);

  // Create SDPA node.
  graph::SdpaAttributes sdpaAttr;
  sdpaAttr.set_name("sdpa");
  auto [oAttr, statsAttr] = graph->sdpa(qAttr, kAttr, vAttr, sdpaAttr);
  oAttr->set_uid(oUID);
  oAttr->set_dim(oTensor.dims()).set_stride(oTensor.strides()).set_output(true);

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
  variantPack[qUID] = qTensor.memory().deviceData();
  variantPack[kUID] = kTensor.memory().deviceData();
  variantPack[vUID] = vTensor.memory().deviceData();
  variantPack[oUID] = oTensor.memory().deviceData();

  // Allocate workspace if the compiled graph requires it.
  int64_t workspaceSize;
  result = graph->get_workspace_size(workspaceSize);
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
  ASSERT_GE(workspaceSize, 0) << result.err_msg;
  const Workspace workspace(static_cast<size_t>(workspaceSize));

  // Execute graph.
  result = graph->execute(handle, variantPack, workspace.get());
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
  oTensor.memory().markDeviceModified();

  // Check results.
  CpuFpReferenceValidation<float> validator(1e-6f, 1e-6f);
  EXPECT_TRUE(validator.allClose(expectedOutput, oTensor));

  // Clean up.
  ASSERT_EQ(hipStreamDestroy(stream), HIPDNN_STATUS_SUCCESS);
  ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

// Test: Grouped query attention (GQA)
// Q has more heads than K/V: Q[B,H,S,D] x K[B,Hkv,S,D] x V[B,Hkv,S,D]
// where H is a multiple of Hkv.
TEST(SdpaIntegrationTest, SdpaGqa) {
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

  // Dimensions matching fusilli SDPA GQA sample.
  const int64_t B = 1;
  const int64_t H = 8;
  const int64_t Hkv = 2;
  const int64_t S = 64;
  const int64_t D = 64;

  // UIDs.
  const int64_t qUID = 0;
  const int64_t kUID = 1;
  const int64_t vUID = 2;
  const int64_t oUID = 3;

  // Initialize tensors (f16 to match fusilli GQA sample).
  PinnedTensor<half> qTensor({B, H, S, D});
  PinnedTensor<half> kTensor({B, Hkv, S, D});
  PinnedTensor<half> vTensor({B, Hkv, S, D});
  PinnedTensor<half> oTensor({B, H, S, D});
  qTensor.fillWithValue(half(0.01f));
  kTensor.fillWithValue(half(0.01f));
  vTensor.fillWithValue(half(0.01f));
  oTensor.fillWithValue(half(-100.0f));

  // Compute expected output using CPU reference (supports GQA natively).
  PinnedTensor<half> expectedOutput({B, H, S, D});
  CpuFpReferenceSdpa::forward(qTensor, kTensor, vTensor, expectedOutput);

  // Create graph.
  auto graph = std::make_shared<graph::Graph>();
  graph->set_name("sdpa_gqa_test");
  graph->set_io_data_type(DataType_t::HALF)
      .set_intermediate_data_type(DataType_t::HALF)
      .set_compute_data_type(DataType_t::FLOAT);

  // Create tensor attributes.
  auto qAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("Q", DataType_t::HALF, qTensor));
  qAttr->set_uid(qUID);
  auto kAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("K", DataType_t::HALF, kTensor));
  kAttr->set_uid(kUID);
  auto vAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("V", DataType_t::HALF, vTensor));
  vAttr->set_uid(vUID);

  // Create SDPA node.
  graph::SdpaAttributes sdpaAttr;
  sdpaAttr.set_name("sdpa_gqa");
  auto [oAttr, statsAttr] = graph->sdpa(qAttr, kAttr, vAttr, sdpaAttr);
  oAttr->set_uid(oUID);
  oAttr->set_dim(oTensor.dims()).set_stride(oTensor.strides()).set_output(true);

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
  variantPack[qUID] = qTensor.memory().deviceData();
  variantPack[kUID] = kTensor.memory().deviceData();
  variantPack[vUID] = vTensor.memory().deviceData();
  variantPack[oUID] = oTensor.memory().deviceData();

  // Allocate workspace if the compiled graph requires it.
  int64_t workspaceSize;
  result = graph->get_workspace_size(workspaceSize);
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
  ASSERT_GE(workspaceSize, 0) << result.err_msg;
  const Workspace workspace(static_cast<size_t>(workspaceSize));

  // Execute graph.
  result = graph->execute(handle, variantPack, workspace.get());
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
  oTensor.memory().markDeviceModified();

  // Check results (relaxed tolerance for f16 accumulation error).
  CpuFpReferenceValidation<half> validator(1e-2f, 1e-2f);
  EXPECT_TRUE(validator.allClose(expectedOutput, oTensor));

  // Clean up.
  ASSERT_EQ(hipStreamDestroy(stream), HIPDNN_STATUS_SUCCESS);
  ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

// Test: GQA with independent K/V head counts (H_k != H_v)
// Q[B,Hq,S,D] x K[B,Hk,Skv,D] x V[B,Hv,Skv,D] -> O[B,Hq,S,D]
// where Hq % Hk == 0 and Hq % Hv == 0, but Hk != Hv.
TEST(SdpaIntegrationTest, SdpaIndependentKVHeads) {
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

  // Dimensions: Hq=8, Hk=4, Hv=2 (independent K/V head counts).
  const int64_t B = 1;
  const int64_t Hq = 8;
  const int64_t Hk = 4;
  const int64_t Hv = 2;
  const int64_t Sq = 16;
  const int64_t Skv = 16;
  const int64_t D = 32;

  // UIDs.
  const int64_t qUID = 0;
  const int64_t kUID = 1;
  const int64_t vUID = 2;
  const int64_t oUID = 3;

  // Initialize tensors (f16).
  PinnedTensor<half> qTensor({B, Hq, Sq, D});
  PinnedTensor<half> kTensor({B, Hk, Skv, D});
  PinnedTensor<half> vTensor({B, Hv, Skv, D});
  PinnedTensor<half> oTensor({B, Hq, Sq, D});
  qTensor.fillWithValue(half(0.01f));
  kTensor.fillWithValue(half(0.01f));
  vTensor.fillWithValue(half(0.01f));
  oTensor.fillWithValue(half(-100.0f));

  // Compute expected output using CPU reference.
  PinnedTensor<half> expectedOutput({B, Hq, Sq, D});
  CpuFpReferenceSdpa::forward(qTensor, kTensor, vTensor, expectedOutput);

  // Create graph.
  auto graph = std::make_shared<graph::Graph>();
  graph->set_name("sdpa_independent_kv_heads_test");
  graph->set_io_data_type(DataType_t::HALF)
      .set_intermediate_data_type(DataType_t::HALF)
      .set_compute_data_type(DataType_t::FLOAT);

  // Create tensor attributes.
  auto qAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("Q", DataType_t::HALF, qTensor));
  qAttr->set_uid(qUID);
  auto kAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("K", DataType_t::HALF, kTensor));
  kAttr->set_uid(kUID);
  auto vAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("V", DataType_t::HALF, vTensor));
  vAttr->set_uid(vUID);

  // Create SDPA node.
  graph::SdpaAttributes sdpaAttr;
  sdpaAttr.set_name("sdpa_independent_kv");
  auto [oAttr, statsAttr] = graph->sdpa(qAttr, kAttr, vAttr, sdpaAttr);
  oAttr->set_uid(oUID);
  oAttr->set_dim(oTensor.dims()).set_stride(oTensor.strides()).set_output(true);

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
  variantPack[qUID] = qTensor.memory().deviceData();
  variantPack[kUID] = kTensor.memory().deviceData();
  variantPack[vUID] = vTensor.memory().deviceData();
  variantPack[oUID] = oTensor.memory().deviceData();

  // Allocate workspace if the compiled graph requires it.
  int64_t workspaceSize;
  result = graph->get_workspace_size(workspaceSize);
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
  ASSERT_GE(workspaceSize, 0) << result.err_msg;
  const Workspace workspace(static_cast<size_t>(workspaceSize));

  // Execute graph.
  result = graph->execute(handle, variantPack, workspace.get());
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
  oTensor.memory().markDeviceModified();

  // Check results (relaxed tolerance for f16 accumulation error).
  CpuFpReferenceValidation<half> validator(1e-2f, 1e-2f);
  EXPECT_TRUE(validator.allClose(expectedOutput, oTensor));

  // Clean up.
  ASSERT_EQ(hipStreamDestroy(stream), HIPDNN_STATUS_SUCCESS);
  ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}
