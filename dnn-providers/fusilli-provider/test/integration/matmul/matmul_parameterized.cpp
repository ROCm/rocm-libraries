// Copyright 2025 Advanced Micro Devices, Inc.
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
#include <optional>
#include <ostream>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;

namespace {

// 2D matmul: A[M,K] x B[K,N] = C[M,N]
constexpr int64_t M = 4;
constexpr int64_t K = 8;
constexpr int64_t N = 5;

// Bias modes
enum class BiasMode {
  NoBias,            // No bias
  BiasFullBroadcast, // Bias shape {1, N} - broadcasts across M rows
  BiasNoBroadcast    // Bias shape {M, N} - no broadcasting
};

struct MatmulTestCase {
  // When true, B tensor is stored  physically transposed (though we still
  // describe it to the API as a row-major tensor). B will have logical dims
  // {K, N} (row-major), but stride {N*K, 1, K} (column major).
  bool transposeB;

  BiasMode biasMode;

  friend std::ostream &operator<<(std::ostream &os, const MatmulTestCase &tc) {
    os << "M" << M << "K" << K << "N" << N;
    if (tc.transposeB) {
      os << "_TransposeB";
    }
    switch (tc.biasMode) {
    case BiasMode::BiasFullBroadcast:
      os << "_BiasBroadcast";
      break;
    case BiasMode::BiasNoBroadcast:
      os << "_BiasFull";
      break;
    default:
      break;
    }
    return os;
  }
};

std::vector<MatmulTestCase> getMatmulTestCases() {
  return {
      // transposeB=false
      {false, BiasMode::NoBias},
      {false, BiasMode::BiasFullBroadcast},
      {false, BiasMode::BiasNoBroadcast},
      // transposeB=true
      {true, BiasMode::NoBias},
      {true, BiasMode::BiasFullBroadcast},
      {true, BiasMode::BiasNoBroadcast},
  };
}

} // namespace

class MatmulParameterizedTest
    : public ::testing::TestWithParam<MatmulTestCase> {};

TEST_P(MatmulParameterizedTest, Correctness) {
  const MatmulTestCase &tc = GetParam();

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

  // UIDs.
  const int64_t aUID = 0;
  const int64_t bUID = 1;
  const int64_t biasUID = 2;
  const int64_t outUID = 3;

  // Initialize tensors.
  PinnedTensor<float> aTensor({M, K});

  // B tensor - logical dims are {K, N}, stride encodes the physical layout
  //  {N, 1} - (row major) when in standard layout.
  //  {1, K} - (column major) when transposed.
  PinnedTensor<float> bTensor(/*dims=*/{K, N},
                              /*strides=*/tc.transposeB
                                  ? std::vector<int64_t>{1, K}
                                  : std::vector<int64_t>{N, 1});

  PinnedTensor<float> outTensor({M, N});

  aTensor.fillWithValue(1.0f);
  bTensor.fillWithValue(1.0f);
  outTensor.fillWithValue(-100.0f);

  // Bias.
  std::optional<PinnedTensor<float>> biasTensor;
  if (tc.biasMode != BiasMode::NoBias) {
    std::vector<int64_t> biasDims = (tc.biasMode == BiasMode::BiasFullBroadcast)
                                        ? std::vector<int64_t>{1, N}
                                        : std::vector<int64_t>{M, N};
    biasTensor.emplace(std::move(biasDims));
    biasTensor->fillWithValue(2.0f);
  }

  // Expected output: each element = K (dot product of K ones) + bias if
  // present.
  PinnedTensor<float> expectedOutput({M, N});
  float expectedValue = static_cast<float>(K);
  if (tc.biasMode != BiasMode::NoBias) {
    expectedValue += 2.0f;
  }
  expectedOutput.fillWithValue(expectedValue);

  // Create graph.
  auto graph = std::make_shared<graph::Graph>();
  graph->set_name("matmul_parameterized_test");
  graph->set_io_data_type(DataType_t::FLOAT)
      .set_intermediate_data_type(DataType_t::FLOAT)
      .set_compute_data_type(DataType_t::FLOAT);

  // Input tensors.
  auto aAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("A", DataType_t::FLOAT, aTensor));
  aAttr->set_uid(aUID);
  auto bAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("B", DataType_t::FLOAT, bTensor));
  bAttr->set_uid(bUID);

  // Create matmul node.
  graph::MatmulAttributes matmulAttr;
  matmulAttr.set_name("matmul");
  auto matmul = graph->matmul(aAttr, bAttr, matmulAttr);
  matmul->set_dim(outTensor.dims()).set_stride(outTensor.strides());

  // Maybe add bias.
  if (tc.biasMode != BiasMode::NoBias) {
    // Matmul output is intermediate, bias is final.
    matmul->set_output(false);

    // Create bias tensor attributes.
    auto biasAttr = std::make_shared<graph::TensorAttributes>(
        graph::makeTensorAttributes("bias", DataType_t::FLOAT, *biasTensor));
    biasAttr->set_uid(biasUID);

    // Add bias.
    graph::PointwiseAttributes biasAddAttr;
    biasAddAttr.set_name("bias_add").set_mode(PointwiseMode_t::ADD);
    auto biasAdd = graph->pointwise(matmul, biasAttr, biasAddAttr);
    biasAdd->set_uid(outUID)
        .set_dim(outTensor.dims())
        .set_stride(outTensor.strides())
        .set_output(true);
  } else {
    // No bias: matmul output is the final output.
    matmul->set_uid(outUID).set_output(true);
  }

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
  variantPack[aUID] = aTensor.memory().deviceData();
  variantPack[bUID] = bTensor.memory().deviceData();
  variantPack[outUID] = outTensor.memory().deviceData();
  if (tc.biasMode != BiasMode::NoBias) {
    variantPack[biasUID] = biasTensor->memory().deviceData();
  }

  // Execute graph.
  result = graph->execute(handle, variantPack, nullptr);
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
  outTensor.memory().markDeviceModified();

  // Check results.
  CpuFpReferenceValidation<float> validator(1e-6f, 1e-6f);
  EXPECT_TRUE(validator.allClose(expectedOutput, outTensor));

  // Clean up.
  ASSERT_EQ(hipStreamDestroy(stream), HIPDNN_STATUS_SUCCESS);
  ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(Matmul, MatmulParameterizedTest,
                         ::testing::ValuesIn(getMatmulTestCases()));
