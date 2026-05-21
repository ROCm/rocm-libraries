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
#include <hipdnn_frontend/attributes/CustomOpAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <cstdint>
#include <memory>
#include <string>

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;

// MLIR for element-wise add using type placeholders.
static std::string getCustomAddMlir() {
  return R"(
  func.func private @{FUNC_NAME}(%arg0: {IN0_TYPE},
                                   %arg1: {IN1_TYPE})
                                   -> {OUT0_TYPE} {
    %int1 = torch.constant.int 1
    %0 = torch.aten.add.Tensor %arg0, %arg1, %int1
        : {IN0_TYPE},
          {IN1_TYPE},
          !torch.int
        -> {OUT0_TYPE}
    return %0 : {OUT0_TYPE}
  }
)";
}

// Test: custom_op element-wise add via MLIR.
// Graph: in0[4] + in1[4] -> custom_op(add) -> out[4]
TEST(CustomOpIntegrationTest, SimpleCustomAdd) {
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

  const int64_t N = 4;

  // UIDs.
  const int64_t in0UID = 0;
  const int64_t in1UID = 1;
  const int64_t outUID = 2;

  // Initialize tensors.
  PinnedTensor<float> in0Tensor({N});
  PinnedTensor<float> in1Tensor({N});
  PinnedTensor<float> outTensor({N});
  in0Tensor.fillWithValue(3.0f);
  in1Tensor.fillWithValue(5.0f);
  outTensor.fillWithValue(-100.0f);

  // Expected: 3.0 + 5.0 = 8.0.
  PinnedTensor<float> expectedOutput({N});
  expectedOutput.fillWithValue(8.0f);

  // Create graph.
  auto graphPtr = std::make_shared<graph::Graph>();
  graphPtr->set_name("custom_add_test");
  graphPtr->set_io_data_type(DataType_t::FLOAT)
      .set_intermediate_data_type(DataType_t::FLOAT)
      .set_compute_data_type(DataType_t::FLOAT);

  // Create input tensor attributes.
  auto in0Attr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("in0", DataType_t::FLOAT, in0Tensor));
  in0Attr->set_uid(in0UID);
  auto in1Attr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("in1", DataType_t::FLOAT, in1Tensor));
  in1Attr->set_uid(in1UID);

  // Store MLIR template directly as opaque byte payload (no JSON wrapping).
  std::string mlir = getCustomAddMlir();
  std::vector<uint8_t> opaqueData(mlir.begin(), mlir.end());

  // Create custom op attributes.
  graph::CustomOpAttributes customAttr;
  customAttr.set_name("my_add")
      .set_custom_op_id("fusilli.my_add")
      .set_data(opaqueData);

  // Create custom op node.
  auto outputs = graphPtr->custom_op({in0Attr, in1Attr}, 1, customAttr);
  outputs[0]->set_uid(outUID);
  outputs[0]
      ->set_dim(outTensor.dims())
      .set_stride(outTensor.strides())
      .set_output(true);

  // Build.
  auto result = graphPtr->build(handle);
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

  // Create variant pack.
  std::unordered_map<int64_t, void *> variantPack;
  variantPack[in0UID] = in0Tensor.memory().deviceData();
  variantPack[in1UID] = in1Tensor.memory().deviceData();
  variantPack[outUID] = outTensor.memory().deviceData();

  // Execute graph.
  result = graphPtr->execute(handle, variantPack, nullptr);
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
  outTensor.memory().markDeviceModified();

  // Check results.
  CpuFpReferenceValidation<float> validator(1e-6f, 1e-6f);
  EXPECT_TRUE(validator.allClose(expectedOutput, outTensor));

  // Clean up.
  ASSERT_EQ(hipStreamDestroy(stream), HIPDNN_STATUS_SUCCESS);
  ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}
