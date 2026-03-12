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
#include <hipdnn_frontend/attributes/ConvolutionWgradAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <cstdint>
#include <memory>

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;

// Test: ConvWGrad 1x1.
TEST(ConvWgradIntegrationTest, ConvWgrad) {
  // Initialize HIP.
  ASSERT_EQ(hipInit(0), hipSuccess);
  ASSERT_EQ(hipSetDevice(0), hipSuccess);

  hipStream_t stream = nullptr;
  ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

  // Set plugin path.
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

  const int64_t n = 16;  // batch
  const int64_t c = 128; // input channels
  const int64_t h = 64;  // spatial height
  const int64_t w = 32;  // spatial width
  const int64_t k = 256; // output channels (gradient channels)
  const int64_t r = 1;   // filter height
  const int64_t s = 1;   // filter width

  // UIDs.
  const int64_t dyUID = 0;
  const int64_t xUID = 1;
  const int64_t dwUID = 2;

  // Initialize tensors.
  PinnedTensor<float> dyTensor({n, k, h, w}, {k * h * w, 1, k * w, k});
  PinnedTensor<float> xTensor({n, c, h, w}, {c * h * w, 1, c * w, c});
  PinnedTensor<float> dwTensor({k, c, r, s});
  dyTensor.fillWithValue(1.0f);
  xTensor.fillWithValue(1.0f);
  dwTensor.fillWithValue(-100.0f);

  // Create graph.
  auto graph = std::make_shared<graph::Graph>();
  graph->set_name("workspace_conv_wgrad_test");
  graph->set_io_data_type(DataType_t::FLOAT)
      .set_intermediate_data_type(DataType_t::FLOAT)
      .set_compute_data_type(DataType_t::FLOAT);

  // Create tensor attributes with NHWC strides.
  auto dyAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("dy", DataType_t::FLOAT, dyTensor));
  dyAttr->set_uid(dyUID);
  auto xAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("x", DataType_t::FLOAT, xTensor));
  xAttr->set_uid(xUID);

  // Create conv wgrad attributes.
  graph::ConvWgradAttributes convWgradAttr;
  convWgradAttr.set_name("conv_wgrad")
      .set_padding({0, 0})
      .set_stride({1, 1})
      .set_dilation({1, 1});

  // Create conv wgrad node.
  auto dwAttr = graph->conv_wgrad(dyAttr, xAttr, convWgradAttr);
  dwAttr->set_uid(dwUID);
  dwAttr->set_dim(dwTensor.dims())
      .set_stride(dwTensor.strides())
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

  // Query workspace size.
  int64_t workspaceSize = 0;
  result = graph->get_workspace_size(workspaceSize);
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
  // Changes in IREE may invalidate this assertion in the future, at the time of
  // this commit this is the only test that validates workspaces sizes. If you
  // change this please ensure there's another test that uses a non-zero
  // workspace.
  ASSERT_GT(workspaceSize, 0)
      << "Conv wgrad should require non-zero workspace for multi-dispatch "
         "execution";

  // Allocate workspace.
  void *workspace = nullptr;
  ASSERT_EQ(hipMalloc(&workspace, static_cast<size_t>(workspaceSize)),
            hipSuccess);

  // Create variant pack.
  std::unordered_map<int64_t, void *> variantPack;
  variantPack[dyUID] = dyTensor.memory().deviceData();
  variantPack[xUID] = xTensor.memory().deviceData();
  variantPack[dwUID] = dwTensor.memory().deviceData();

  // Execute graph with workspace.
  result = graph->execute(handle, variantPack, workspace);
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
  dwTensor.memory().markDeviceModified();

  // Validate results. For 1x1 conv wgrad with all-ones inputs, stride=1,
  // no padding: dw[k,c,0,0] = sum_{n,h,w} dy * x = n * h * w.
  const float expected = static_cast<float>(n * h * w);
  auto *dwData = dwTensor.memory().hostData();
  for (size_t i = 0; i < static_cast<size_t>(k * c * r * s); ++i) {
    EXPECT_FLOAT_EQ(dwData[i], expected) << "mismatch at index " << i;
  }

  // Clean up.
  ASSERT_EQ(hipFree(workspace), hipSuccess);
  ASSERT_EQ(hipStreamDestroy(stream), HIPDNN_STATUS_SUCCESS);
  ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}
