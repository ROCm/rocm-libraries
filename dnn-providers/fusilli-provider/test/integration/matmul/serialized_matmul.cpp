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
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <unordered_map>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;

TEST(MatmulIntegrationTest, SerializedCompiledPlanMatmul) {
  ASSERT_EQ(hipInit(0), hipSuccess);
  ASSERT_EQ(hipSetDevice(0), hipSuccess);

  hipStream_t stream = nullptr;
  ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

  auto pluginPath = std::filesystem::canonical(getCurrentExecutableDirectory() /
                                               FUSILLI_PLUGIN_PATH);
  const std::array<const char *, 1> paths = {pluginPath.c_str()};
  ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(),
                                           HIPDNN_PLUGIN_LOADING_ABSOLUTE),
            HIPDNN_STATUS_SUCCESS);

  hipdnnHandle_t handle = nullptr;
  ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);
  ASSERT_EQ(hipdnnSetStream(handle, stream), HIPDNN_STATUS_SUCCESS);

  const int64_t M = 4;
  const int64_t K = 8;
  const int64_t N = 5;

  const int64_t aUID = 0;
  const int64_t bUID = 1;
  const int64_t cUID = 2;

  PinnedTensor<float> aTensor({M, K});
  PinnedTensor<float> bTensor({K, N});
  PinnedTensor<float> cTensor({M, N});
  aTensor.fillWithValue(1.0f);
  bTensor.fillWithValue(1.0f);
  cTensor.fillWithValue(-100.0f);

  PinnedTensor<float> expectedOutput({M, N});
  expectedOutput.fillWithValue(static_cast<float>(K));

  auto graph = std::make_shared<graph::Graph>();
  graph->set_name("serialized_matmul_test");
  graph->set_io_data_type(DataType_t::FLOAT)
      .set_intermediate_data_type(DataType_t::FLOAT)
      .set_compute_data_type(DataType_t::FLOAT);

  auto aAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("A", DataType_t::FLOAT, aTensor));
  aAttr->set_uid(aUID);
  auto bAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("B", DataType_t::FLOAT, bTensor));
  bAttr->set_uid(bUID);

  graph::MatmulAttributes matmulAttr;
  matmulAttr.set_name("matmul");

  auto cAttr = graph->matmul(aAttr, bAttr, matmulAttr);
  cAttr->set_uid(cUID);
  cAttr->set_dim(cTensor.dims()).set_stride(cTensor.strides()).set_output(true);

  auto result = graph->build(handle);
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

  std::vector<uint8_t> serializedPlan;
  result = graph->serialize_compiled_plan(serializedPlan);
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
  ASSERT_FALSE(serializedPlan.empty());

  graph::Graph restoredGraph;
  result = restoredGraph.from_compiled_plan_binary(handle, serializedPlan);
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

  int64_t workspaceSize = 0;
  result = restoredGraph.get_workspace_size(workspaceSize);
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
  ASSERT_GE(workspaceSize, 0);

  void *workspace = nullptr;
  if (workspaceSize > 0) {
    ASSERT_EQ(hipMalloc(&workspace, static_cast<size_t>(workspaceSize)),
              hipSuccess);
  }

  std::unordered_map<int64_t, void *> variantPack;
  variantPack[aUID] = aTensor.memory().deviceData();
  variantPack[bUID] = bTensor.memory().deviceData();
  variantPack[cUID] = cTensor.memory().deviceData();

  result = restoredGraph.execute(handle, variantPack, workspace);
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
  cTensor.memory().markDeviceModified();

  if (workspace != nullptr) {
    ASSERT_EQ(hipFree(workspace), hipSuccess);
  }

  CpuFpReferenceValidation<float> validator(1e-6f, 1e-6f);
  EXPECT_TRUE(validator.allClose(expectedOutput, cTensor));

  ASSERT_EQ(hipStreamDestroy(stream), HIPDNN_STATUS_SUCCESS);
  ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}
