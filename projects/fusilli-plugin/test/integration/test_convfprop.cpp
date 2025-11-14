// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_backend.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>

#include <cstdint>
#include <memory>
#include <ostream>

using namespace hipdnn_frontend;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::test_utilities;

struct TestParams {
  bool shouldSetStream;
  hipDevice_t deviceId;

  // GTest uses the << operator to update the parameterized test name.
  friend std::ostream &operator<<(std::ostream &ss, const TestParams &p) {
    ss << "Device" << p.deviceId << "_";
    ss << (p.shouldSetStream ? "WithStream" : "WithoutStream");
    return ss;
  }
};

class ConvFpropIntegrationTest : public ::testing::TestWithParam<TestParams> {};

TEST_P(ConvFpropIntegrationTest, Basic1x1Convolution) {
  TestParams params = GetParam();
  // Uncomment to enable debug logging
  // setenv("HIPDNN_LOG_LEVEL", "info", 1);

  // Initialize HIP.
  ASSERT_EQ(hipInit(0), hipSuccess);

  // Set device.
  ASSERT_EQ(hipSetDevice(params.deviceId), hipSuccess);

  // Create stream.
  hipStream_t stream = nullptr;
  if (params.shouldSetStream) {
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
  }

  // Set plugin path.
  const std::array<const char *, 1> paths = {FUSILLI_PLUGIN_DIR};
  ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(),
                                           HIPDNN_PLUGIN_LOADING_ABSOLUTE),
            HIPDNN_STATUS_SUCCESS);

  // Create handle.
  hipdnnHandle_t handle;
  ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

  // Check that loading the plugin didn't change the active device.
  hipDevice_t deviceId = -1;
  ASSERT_EQ(hipGetDevice(&deviceId), hipSuccess);
  ASSERT_EQ(deviceId, params.deviceId);

  if (params.shouldSetStream) {
    ASSERT_EQ(hipdnnSetStream(handle, stream), HIPDNN_STATUS_SUCCESS);
  }

  // Dimensions.
  const int64_t n = 16;  // batch
  const int64_t c = 128; // in channels
  const int64_t h = 64;  // image height
  const int64_t w = 64;  // image width
  const int64_t k = 256; // out channels
  const int64_t r = 1;   // filter height
  const int64_t s = 1;   // filter width

  // UIDs.
  const int64_t xUID = 0;
  const int64_t wUID = 1;
  const int64_t yUID = 2;

  // Initialize tensors.
  PinnedTensor<float> xTensor({n, c, h, w});
  PinnedTensor<float> wTensor({k, c, r, s});
  PinnedTensor<float> yTensor({n, k, h, w});
  xTensor.fillWithValue(1.0f);
  wTensor.fillWithValue(1.0f);
  yTensor.fillWithValue(-100.0f);

  // Expected output.
  PinnedTensor<float> expectedOutput({n, k, h, w});
  expectedOutput.fillWithValue(128.0f);

  // Create graph.
  auto graph = std::make_shared<graph::Graph>();
  graph->set_name("conv_1x1_test");
  graph->set_io_data_type(DataType_t::FLOAT)
      .set_compute_data_type(DataType_t::FLOAT);

  // Create tensor attributes.
  auto xAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("input", DataType_t::FLOAT, xTensor));
  xAttr->set_uid(xUID);
  auto wAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("filter", DataType_t::FLOAT, wTensor));
  wAttr->set_uid(wUID);

  // Create convolution attributes.
  graph::ConvFpropAttributes convAttr;
  convAttr.set_name("conv_fprop")
      .set_padding({0, 0})
      .set_stride({1, 1})
      .set_dilation({1, 1});

  // Create graph.
  auto yAttr = graph->conv_fprop(xAttr, wAttr, convAttr);
  yAttr->set_uid(yUID);
  yAttr->set_dim(yTensor.dims()).set_stride(yTensor.strides()).set_output(true);

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
  variantPack[xUID] = xTensor.memory().deviceData();
  variantPack[wUID] = wTensor.memory().deviceData();
  variantPack[yUID] = yTensor.memory().deviceData();

  // Execute graph.
  result = graph->execute(handle, variantPack, nullptr);
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
  // Mark hipDNN tensor CPU cache ask stale, data must be read from device.
  yTensor.memory().markDeviceModified();

  // Check results.
  CpuFpReferenceValidation<float> validator(1e-6f, 1e-6f);
  EXPECT_TRUE(validator.allClose(expectedOutput, yTensor));

  // Clean up.
  if (params.shouldSetStream) {
    ASSERT_EQ(hipStreamDestroy(stream), HIPDNN_STATUS_SUCCESS);
  }
  ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

static std::vector<TestParams> generateTestParams() {
  std::vector<TestParams> params;
  int deviceCount;
  assert(hipGetDeviceCount(&deviceCount) == hipSuccess);

  // Always test with device 0.
  params.push_back({
      .shouldSetStream = false,
      .deviceId = 0,
  });
  params.push_back({
      .shouldSetStream = true,
      .deviceId = 0,
  });

  // Test on last device if multiple devices are available.
  if (deviceCount > 1) {
    params.push_back({
        .shouldSetStream = false,
        .deviceId = deviceCount - 1,
    });
    params.push_back({
        .shouldSetStream = true,
        .deviceId = deviceCount - 1,
    });
  }

  return params;
}

INSTANTIATE_TEST_SUITE_P(, ConvFpropIntegrationTest,
                         ::testing::ValuesIn(generateTestParams()));
