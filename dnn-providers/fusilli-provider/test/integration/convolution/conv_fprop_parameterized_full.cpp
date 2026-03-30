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
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <ostream>
#include <random>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;

namespace {

struct ConvTestCase {
  std::vector<int64_t> xDims_; // Input tensor dims [N, C, H, W]
  std::vector<int64_t> wDims_; // Weight tensor dims [K, C/groups, R, S]
  std::vector<int64_t> yDims_; // Output tensor dims (computed)
  std::vector<int64_t> padding_;
  std::vector<int64_t> stride_;
  std::vector<int64_t> dilation_;
  unsigned seed_;
  std::string layoutName_;

  ConvTestCase(std::vector<int64_t> xDims, std::vector<int64_t> wDims,
               std::vector<int64_t> padding, std::vector<int64_t> stride,
               std::vector<int64_t> dilation, unsigned seed,
               std::string layoutName)
      : xDims_(std::move(xDims)), wDims_(std::move(wDims)),
        padding_(std::move(padding)), stride_(std::move(stride)),
        dilation_(std::move(dilation)), seed_(seed),
        layoutName_(std::move(layoutName)) {

    // Indices:
    //   input (x) dimensions
    //     n  - Batch size, always at index 0
    //     ci - Channels, always at index 1
    //     d  - Depth (for 3D convolutions), always at index 2 if present
    //     h  - Height, at index 2 for 2D conv and index 3 for 3D conv
    //     w  - Width, at index 3 for 2D conv and index 4 for 3D conv
    //   weight (w) dimensions
    //     k  - Number of output channels, always at index 0
    //     cw - Weight channels, always at index 1
    //         - for grouped convs groups = ci / cw
    //     t  - Filter Depth (for 3D convolutions), always at index 2 if present
    //     r  - Filter height, at index 2 for 2D conv and index 3 for 3D conv
    //     s  - Filter width, at index 3 for 2D conv and index 4 for 3D conv
    //   output (y) dimensions
    //     n  - Batch size, always at index 0
    //     k  - Output channels, always at index 1
    //     o  - Output depth (for 3D convolutions), always at index 2 if present
    //     p  - Output height, at index 2 for 2D conv and index 3 for 3D conv
    //     q  - Output width, at index 3 for 2D conv and index 4 for 3D conv
    constexpr int kBatchSizeIndex = 0;
    constexpr int kWeightOutputChannelIndex = 0;
    constexpr int kChannelIndex = 1;
    constexpr int kSpatialStartIndex = 2;

    int64_t n = xDims_[kBatchSizeIndex];
    int64_t k = wDims_[kWeightOutputChannelIndex];
    size_t numSpatialDims = xDims_.size() - kSpatialStartIndex;

    // Validate that the convolution parameter vectors match the number of
    // spatial dimensions
    if (padding_.size() != numSpatialDims ||
        dilation_.size() != numSpatialDims ||
        stride_.size() != numSpatialDims) {
      throw std::invalid_argument("Convolution parameter vectors must match "
                                  "the number of spatial dimensions.");
    }

    yDims_.resize(xDims_.size());
    yDims_[kBatchSizeIndex] = n;
    yDims_[kChannelIndex] = k;

    // Formula:
    //   YDim = ((XDim + 2*padding - dilation*(WDim-1) - 1) / stride) + 1
    for (size_t i = 0; i < numSpatialDims; ++i) {
      int64_t inDim = xDims_[i + kSpatialStartIndex];
      int64_t kernelDim = wDims_[i + kSpatialStartIndex];
      int64_t effectiveKernelSize = dilation_[i] * (kernelDim - 1) + 1;
      int64_t paddedInputSize = inDim + 2 * padding_[i];
      yDims_[i + kSpatialStartIndex] =
          ((paddedInputSize - effectiveKernelSize) / stride_[i]) + 1;
    }
  }

  friend std::ostream &operator<<(std::ostream &ss, const ConvTestCase &tc) {
    // Format: N{n}C{c}H{h}W{w}_K{k}R{r}S{s}_Pad{p}_Str{s}_Dil{d}_{layout}
    ss << "N" << tc.xDims_[0] << "C" << tc.xDims_[1] << "H" << tc.xDims_[2]
       << "W" << tc.xDims_[3];
    ss << "_K" << tc.wDims_[0] << "R" << tc.wDims_[2] << "S" << tc.wDims_[3];
    ss << "_Pad" << tc.padding_[0];
    ss << "_Str" << tc.stride_[0];
    ss << "_Dil" << tc.dilation_[0];
    ss << "_" << tc.layoutName_;
    return ss;
  }
};

std::vector<ConvTestCase> getConvTestCases(std::string layoutName) {
  unsigned seed = 42;
  return {
      // Filter 1x1
      {/*xDims=*/{1, 16, 16, 16}, /*wDims=*/{1, 16, 1, 1},
       /*padding=*/{0, 0}, /*stride=*/{1, 1}, /*dilation=*/{1, 1}, seed,
       layoutName},
      // Filter 3x3, no padding
      {/*xDims=*/{1, 16, 16, 16}, /*wDims=*/{1, 16, 3, 3},
       /*padding=*/{0, 0}, /*stride=*/{1, 1}, /*dilation=*/{1, 1}, seed,
       layoutName},
      // Padding = 1
      {/*xDims=*/{1, 16, 16, 16}, /*wDims=*/{1, 16, 3, 3},
       /*padding=*/{1, 1}, /*stride=*/{1, 1}, /*dilation=*/{1, 1}, seed,
       layoutName},
      // Stride = 2
      {/*xDims=*/{1, 16, 16, 16}, /*wDims=*/{1, 16, 3, 3},
       /*padding=*/{1, 1}, /*stride=*/{2, 2}, /*dilation=*/{1, 1}, seed,
       layoutName},
      // Dilation = 2
      {/*xDims=*/{1, 16, 16, 16}, /*wDims=*/{1, 16, 3, 3},
       /*padding=*/{2, 2}, /*stride=*/{1, 1}, /*dilation=*/{2, 2}, seed,
       layoutName},
      // Batched convolution
      {/*xDims=*/{8, 16, 16, 16}, /*wDims=*/{1, 16, 1, 1},
       /*padding=*/{0, 0}, /*stride=*/{1, 1}, /*dilation=*/{1, 1}, seed,
       layoutName},
      // Non-square spatial dims
      {/*xDims=*/{1, 16, 16, 8}, /*wDims=*/{1, 16, 3, 3},
       /*padding=*/{1, 1}, /*stride=*/{1, 1}, /*dilation=*/{1, 1}, seed,
       layoutName},
      // Grouped convolution - 2 groups
      {/*xDims=*/{1, 16, 16, 16}, /*wDims=*/{2, 8, 3, 3},
       /*padding=*/{1, 1}, /*stride=*/{1, 1}, /*dilation=*/{1, 1}, seed,
       layoutName},
      // Grouped convolution - 2 batches, 4 groups, stride, padding, dilation
      {/*xDims=*/{2, 32, 16, 16}, /*wDims=*/{4, 8, 3, 3},
       /*padding=*/{1, 1}, /*stride=*/{2, 2}, /*dilation=*/{2, 2}, seed,
       layoutName},
  };
}

} // namespace

class ConvFpropParameterizedTest
    : public ::testing::TestWithParam<ConvTestCase> {};

TEST_P(ConvFpropParameterizedTest, Correctness) {
  const ::testing::TestInfo *const test_info =
      ::testing::UnitTest::GetInstance()->current_test_info();

  // Get the name of the individual test
  const char *test_name = test_info->name();

  const ConvTestCase &tc = GetParam();

  // Load only the fusilli plugin
  auto pluginPath = std::filesystem::canonical(getCurrentExecutableDirectory() /
                                               FUSILLI_PLUGIN_PATH);
  const std::array<const char *, 1> paths = {pluginPath.c_str()};
  ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(),
                                           HIPDNN_PLUGIN_LOADING_ABSOLUTE),
            HIPDNN_STATUS_SUCCESS);

  // Setup hipdnn and set handle
  hipStream_t stream = nullptr;
  ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
  hipdnnHandle_t handle = nullptr;
  ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);
  ASSERT_EQ(hipdnnSetStream(handle, stream), HIPDNN_STATUS_SUCCESS);

  // See comment on INSTANTIATE_TEST_SUITE_P below
  TensorLayout layout;
  if (tc.layoutName_ == "NCHW") {
    layout = TensorLayout::NCHW;
  } else if (tc.layoutName_ == "NHWC") {
    layout = TensorLayout::NHWC;
  } else {
    throw std::invalid_argument("Unknown layout: " + tc.layoutName_);
  }

  // UIDs
  const int64_t xUID = 0;
  const int64_t wUID = 1;
  const int64_t yUID = 2;

  // Create tensors
  PinnedTensor<float> xTensor(tc.xDims_, layout);
  PinnedTensor<float> wTensor(tc.wDims_, layout);
  PinnedTensor<float> yTensor(tc.yDims_, layout);
  PinnedTensor<float> expectedOutput(tc.yDims_, layout);

  // Initialize with random values
  std::mt19937 gen(tc.seed_);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  auto *xData = xTensor.memory().hostData();
  auto *wData = wTensor.memory().hostData();
  size_t xSize = 1, wSize = 1;
  for (long d : tc.xDims_)
    xSize *= static_cast<size_t>(d);
  for (long d : tc.wDims_)
    wSize *= static_cast<size_t>(d);

  for (size_t i = 0; i < xSize; ++i)
    xData[i] = dist(gen);
  for (size_t i = 0; i < wSize; ++i)
    wData[i] = dist(gen);

  xTensor.memory().markHostModified();
  wTensor.memory().markHostModified();

  // Compute CPU reference
  CpuFpReferenceConvolution::fprop(xTensor, wTensor, expectedOutput, tc.stride_,
                                   tc.dilation_, tc.padding_);

  // Create graph
  auto graph = std::make_shared<graph::Graph>();
  graph->set_name(test_name);
  graph->set_io_data_type(DataType_t::FLOAT)
      .set_compute_data_type(DataType_t::FLOAT);

  // Create tensor attributes
  auto xAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("input", DataType_t::FLOAT, xTensor));
  xAttr->set_uid(xUID);
  auto wAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("filter", DataType_t::FLOAT, wTensor));
  wAttr->set_uid(wUID);

  // Create convolution attributes
  graph::ConvFpropAttributes convAttr;
  convAttr.set_name("conv_fprop")
      .set_padding(tc.padding_)
      .set_stride(tc.stride_)
      .set_dilation(tc.dilation_);

  // Build graph
  auto yAttr = graph->conv_fprop(xAttr, wAttr, convAttr);
  yAttr->set_uid(yUID);
  yAttr->set_dim(yTensor.dims()).set_stride(yTensor.strides()).set_output(true);

  // Validate and build plans
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

  // Create variant pack
  std::unordered_map<int64_t, void *> variantPack;
  variantPack[xUID] = xTensor.memory().deviceData();
  variantPack[wUID] = wTensor.memory().deviceData();
  variantPack[yUID] = yTensor.memory().deviceData();

  // Execute graph
  result = graph->execute(handle, variantPack, nullptr);
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

  // Mark device data as modified so validation reads from device
  yTensor.memory().markDeviceModified();

  // Validate results
  CpuFpReferenceValidation<float> validator(1e-5f, 1e-5f);
  EXPECT_TRUE(validator.allClose(expectedOutput, yTensor));

  // Cleanup
  hipdnnDestroy(handle);
  (void)hipStreamDestroy(stream);
}

// Use string literals instead of TensorLayout::NCHW/NHWC/etc. to avoid static
// initialization order fiasco. TensorLayout is defined in a shared library with
// members that require runtime initialization.  INSTANTIATE_TEST_SUITE_P
// creates static registrations, and if those run before the library's statics
// initialize, we'll get a segfault or similar. The solution would be to
// constexpr-ize TensorLayout::NCHW/NHWC.
INSTANTIATE_TEST_SUITE_P(NCHW, ConvFpropParameterizedTest,
                         ::testing::ValuesIn(getConvTestCases("NCHW")));

INSTANTIATE_TEST_SUITE_P(NHWC, ConvFpropParameterizedTest,
                         ::testing::ValuesIn(getConvTestCases("NHWC")));
