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

#include <cstdint>
#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;

// Test: Int4 x fp16 matmul with non-uniform nibble values.
// A[1,M,K] (int4, packed) x B[1,K,N] (fp16) = C[1,M,N] (fp16)
TEST(MatmulIntegrationTest, Int4xFp16Matmul) {
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

  // Dimensions: A[1,4,8] x B[1,8,4] = C[1,4,4]
  // Rank-3 required for mixed-precision matmul (torch.bmm).
  const int64_t batch = 1;
  const int64_t M = 4;
  const int64_t K = 8;
  const int64_t N = 4;

  // UIDs.
  const int64_t aUID = 0;
  const int64_t bUID = 1;
  const int64_t cUID = 2;

  // Initialize tensor A (int4, packed 2 elements per byte, LSB-first nibble).
  // Byte 0x72: low nibble = 2, high nibble = 7.
  // Each row of K=8: [2, 7, 2, 7, 2, 7, 2, 7].
  const size_t aNumElements = static_cast<size_t>(batch * M * K);
  const size_t aPackedBytes = (aNumElements + 1) / 2;
  std::vector<uint8_t> aHostPacked(aPackedBytes, 0x72);

  void *aDevicePtr = nullptr;
  ASSERT_EQ(hipMalloc(&aDevicePtr, aPackedBytes), hipSuccess);
  ASSERT_EQ(hipMemcpyHtoD(aDevicePtr, aHostPacked.data(), aPackedBytes),
            hipSuccess);

  // Initialize tensors B and C.
  PinnedTensor<half> bTensor({batch, K, N});
  bTensor.fillWithValue(static_cast<half>(1.0f));
  PinnedTensor<half> cTensor({batch, M, N});
  cTensor.fillWithValue(static_cast<half>(0.0f));

  // Create graph.
  auto graph = std::make_shared<graph::Graph>();
  graph->set_name("matmul_int4_fp16_test");
  graph->set_io_data_type(DataType_t::HALF)
      .set_intermediate_data_type(DataType_t::HALF)
      .set_compute_data_type(DataType_t::FLOAT);

  // Create tensor attributes.
  auto aAttr =
      std::make_shared<graph::TensorAttributes>(graph::makeTensorAttributes(
          "A", DataType_t::INT4, {batch, M, K}, {M * K, K, 1}));
  aAttr->set_uid(aUID);
  auto bAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("B", DataType_t::HALF, bTensor));
  bAttr->set_uid(bUID);

  // Create matmul node.
  graph::MatmulAttributes matmulAttr;
  matmulAttr.set_name("matmul");

  auto cAttr = graph->matmul(aAttr, bAttr, matmulAttr);
  cAttr->set_uid(cUID);
  cAttr->set_dim(cTensor.dims())
      .set_stride(cTensor.strides())
      .set_data_type(DataType_t::HALF)
      .set_output(true);

  // Build graph.
  auto result = graph->build(handle);
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

  // Create variant pack.
  std::unordered_map<int64_t, void *> variantPack;
  variantPack[aUID] = aDevicePtr;
  variantPack[bUID] = bTensor.memory().deviceData();
  variantPack[cUID] = cTensor.memory().deviceData();

  // Execute graph.
  result = graph->execute(handle, variantPack, nullptr);
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
  cTensor.memory().markDeviceModified();

  // Check results: dot(row, ones) = (K/2)*2 + (K/2)*7 = 8 + 28 = 36.
  const int64_t halfK = K / 2;
  const auto expected = static_cast<half>(static_cast<float>(halfK) * 2.0f +
                                          static_cast<float>(halfK) * 7.0f);
  PinnedTensor<half> expectedOutput({batch, M, N});
  expectedOutput.fillWithValue(expected);

  CpuFpReferenceValidation<half> validator(1e-3f, 1e-3f);
  EXPECT_TRUE(validator.allClose(expectedOutput, cTensor));

  // Clean up.
  ASSERT_EQ(hipFree(aDevicePtr), hipSuccess);
  ASSERT_EQ(hipStreamDestroy(stream), HIPDNN_STATUS_SUCCESS);
  ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}
