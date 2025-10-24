// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <flatbuffers/flatbuffer_builder.h>
#include <fusilli.h>
#include <gtest/gtest.h>
#include <hipdnn_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_sdk/plugin/EnginePluginApi.h>
#include <hipdnn_sdk/plugin/PluginApi.h>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>
#include <spdlog/spdlog.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unistd.h>
#include <vector>

#include "fusilli/attributes/tensor_attributes.h"
#include "graph_import.h"
#include "hipdnn_engine_plugin_execution_context.h"
#include "utils.h"

bool loggingCallbackCalled = false;
std::vector<std::string> capturedLogMessages;
std::vector<hipdnnSeverity_t> capturedLogSeverities;
std::mutex logMutex;
std::condition_variable logConditionVariable;

void testLoggingCallback(hipdnnSeverity_t severity, const char *msg) {
  // hipDNN sets spdlog up to log in a separate thread, so we need to put our
  // mutual exclusion gloves on before touching any variables the main thread
  // does.
  std::scoped_lock lock(logMutex);

  loggingCallbackCalled = true;
  if (msg) {
    capturedLogMessages.push_back(std::string(msg));
    capturedLogSeverities.push_back(severity);
  }
  logConditionVariable.notify_one();
}

TEST(TestFusilliPluginApi, Logging) {
  // Set tracking variables
  {
    std::scoped_lock lock(logMutex);
    loggingCallbackCalled = false;
    capturedLogMessages.clear();
    capturedLogSeverities.clear();
  }

  // Set up logging callback
  ASSERT_EQ(hipdnnPluginSetLoggingCallback(testLoggingCallback),
            HIPDNN_PLUGIN_STATUS_SUCCESS);

  std::unique_lock lock(logMutex);

  // Wait for the logging callback to signal that it has been called.
  auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  EXPECT_TRUE(logConditionVariable.wait_until(
      lock, timeout, [&]() { return loggingCallbackCalled; }));

  EXPECT_TRUE(loggingCallbackCalled);
  EXPECT_FALSE(capturedLogMessages.empty());
  EXPECT_TRUE(capturedLogMessages.front().find(
                  "logging callback initialized") != std::string::npos);
};

TEST(TestFusilliPluginApi, GetNameSuccess) {
  const char *name = nullptr;
  EXPECT_EQ(hipdnnPluginGetName(&name), HIPDNN_PLUGIN_STATUS_SUCCESS);
  EXPECT_STREQ(name, FUSILLI_PLUGIN_NAME);
}

TEST(TestFusilliPluginApi, GetNameNullptr) {
  EXPECT_EQ(hipdnnPluginGetName(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);

  // Verify error was set
  const char *errorStr = nullptr;
  hipdnnPluginGetLastErrorString(&errorStr);
  ASSERT_NE(errorStr, nullptr);
}

TEST(TestFusilliPluginApi, GetVersionSuccess) {
  const char *version = nullptr;
  EXPECT_EQ(hipdnnPluginGetVersion(&version), HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_NE(version, nullptr);
  // TODO(#2317): check returned version against single source of truth.
}

TEST(TestFusilliPluginApi, GetVersionNullptr) {
  EXPECT_EQ(hipdnnPluginGetVersion(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);

  // Verify error was set
  const char *errorStr = nullptr;
  hipdnnPluginGetLastErrorString(&errorStr);
  ASSERT_NE(errorStr, nullptr);
}

TEST(TestFusilliPluginApi, GetTypeSuccess) {
  hipdnnPluginType_t type;
  EXPECT_EQ(hipdnnPluginGetType(&type), HIPDNN_PLUGIN_STATUS_SUCCESS);
  EXPECT_EQ(type, HIPDNN_PLUGIN_TYPE_ENGINE);
}

TEST(TestFusilliPluginApi, GetTypeNullptr) {
  EXPECT_EQ(hipdnnPluginGetType(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);

  // Verify error was set
  const char *errorStr = nullptr;
  hipdnnPluginGetLastErrorString(&errorStr);
  ASSERT_NE(errorStr, nullptr);
}

TEST(TestFusilliPluginApi, GetLastErrorStringSuccess) {
  const char *errorStr = nullptr;
  hipdnnPluginGetLastErrorString(&errorStr);
  ASSERT_NE(errorStr, nullptr);
  // Initially should be empty or contain a previous error
  EXPECT_GE(strlen(errorStr), 0);
}

TEST(TestFusilliPluginApi, GetLastErrorStringNullptr) {
  // This should not crash even with nullptr
  EXPECT_NO_THROW(hipdnnPluginGetLastErrorString(nullptr));
}

TEST(TestFusilliPluginApi, SetLoggingCallbackNullptr) {
  // Setting nullptr should return BAD_PARAM
  EXPECT_EQ(hipdnnPluginSetLoggingCallback(nullptr),
            HIPDNN_PLUGIN_STATUS_BAD_PARAM);

  // Verify error was set
  const char *errorStr = nullptr;
  hipdnnPluginGetLastErrorString(&errorStr);
  ASSERT_NE(errorStr, nullptr);
}

TEST(TestFusilliPluginApi, GetAllEngineIds) {
  // First call with null buffer to get count
  uint32_t numEngines = 0;
  EXPECT_EQ(hipdnnEnginePluginGetAllEngineIds(nullptr, 0, &numEngines),
            HIPDNN_PLUGIN_STATUS_SUCCESS);
  EXPECT_EQ(numEngines, 1);

  // Second call to get actual engine IDs
  std::vector<int64_t> engineIds(numEngines);
  EXPECT_EQ(hipdnnEnginePluginGetAllEngineIds(engineIds.data(), numEngines,
                                              &numEngines),
            HIPDNN_PLUGIN_STATUS_SUCCESS);
  EXPECT_EQ(numEngines, 1);
  EXPECT_EQ(engineIds[0], FUSILLI_PLUGIN_ENGINE_ID);
}

TEST(TestFusilliPluginApi, GetAllEngineIdsNullNumEngines) {
  EXPECT_EQ(hipdnnEnginePluginGetAllEngineIds(nullptr, 0, nullptr),
            HIPDNN_PLUGIN_STATUS_BAD_PARAM);

  // Verify error was set
  const char *errorStr = nullptr;
  hipdnnPluginGetLastErrorString(&errorStr);
  ASSERT_NE(errorStr, nullptr);
  EXPECT_GT(strlen(errorStr), 0u);
}

// TODO(#2363): investigate using createValidConvFwdGraph from upstream hipDNN
flatbuffers::FlatBufferBuilder
createValidConvFwdGraph(int64_t xUID = 0, int64_t wUID = 1, int64_t yUID = 2,
                        hipdnn_sdk::data_objects::DataType dataType =
                            hipdnn_sdk::data_objects::DataType::FLOAT,
                        const std::vector<int64_t> &xDims = {4, 4, 4, 4},
                        const std::vector<int64_t> &xStrides = {64, 16, 4, 1},
                        const std::vector<int64_t> &wDims = {4, 4, 1, 1},
                        const std::vector<int64_t> &wStrides = {4, 1, 1, 1},
                        const std::vector<int64_t> &yDims = {4, 4, 4, 4},
                        const std::vector<int64_t> &yStrides = {64, 16, 4, 1},
                        const std::vector<int64_t> &convPrePadding = {0, 0},
                        const std::vector<int64_t> &convPostPadding = {0, 0},
                        const std::vector<int64_t> &convStrides = {1, 1},
                        const std::vector<int64_t> &convDilation = {1, 1}) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>>
      tensorAttributes;

  tensorAttributes.push_back(CreateTensorAttributesDirect(
      builder, xUID, "x", dataType, &xStrides, &xDims));

  tensorAttributes.push_back(CreateTensorAttributesDirect(
      builder, wUID, "w", dataType, &wStrides, &wDims));

  tensorAttributes.push_back(CreateTensorAttributesDirect(
      builder, yUID, "y", dataType, &yStrides, &yDims));

  auto convAttributes = CreateConvolutionFwdAttributesDirect(
      builder,
      /*x_tensor_uid*/ xUID,
      /*w_tensor_uid*/ wUID,
      /*y_tensor_uid*/ yUID, &convPrePadding, &convPostPadding, &convStrides,
      &convDilation, hipdnn_sdk::data_objects::ConvMode::CROSS_CORRELATION);

  std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
  auto node = CreateNodeDirect(
      builder, "conv_fwd",
      hipdnn_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes,
      convAttributes.Union());
  nodes.push_back(node);

  auto graphOffset =
      CreateGraphDirect(builder, "test",
                        /*compute_type*/ dataType,
                        /*intermediate_type*/ dataType,
                        /*io_type=*/dataType, &tensorAttributes, &nodes);
  builder.Finish(graphOffset);
  return builder;
}

TEST(TestFusilliPluginApi, GetApplicableEngineIds) {
  // Create plugin handle.
  hipdnnEnginePluginHandle_t handle = nullptr;
  ASSERT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_NE(handle, nullptr);

  // Create a serialized hipDNN bach norm graph.
  auto builder = hipdnn_sdk::test_utilities::createValidBatchnormBwdGraph();
  hipdnnPluginConstData_t opGraph;
  opGraph.ptr = builder.GetBufferPointer();
  opGraph.size = builder.GetSize();

  // Fusilli plugin should not offer to compile and execute bach norm (yet).
  std::array<int64_t, 5> engineIDs;
  uint32_t numEngines = -1;
  ASSERT_EQ(hipdnnEnginePluginGetApplicableEngineIds(
                handle, &opGraph, engineIDs.data(), 5, &numEngines),
            HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_EQ(numEngines, 0);

  // Create a serialized hipDNN conv_fprop graph with symmetric padding.
  builder = createValidConvFwdGraph();
  opGraph.ptr = builder.GetBufferPointer();
  opGraph.size = builder.GetSize();

  // Fusilli plugin should offer to compile and execute single node conv_fprop.
  ASSERT_EQ(hipdnnEnginePluginGetApplicableEngineIds(
                handle, &opGraph, engineIDs.data(), 5, &numEngines),
            HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_EQ(numEngines, 1);
  ASSERT_EQ(engineIDs[0], FUSILLI_PLUGIN_ENGINE_ID);

  // Create a serialized hipDNN conv_fprop graph with asymmetric padding.
  builder = createValidConvFwdGraph(
      /*xUID=*/0, /*wUID=*/1, /*yUID=*/2,
      /*dataType=*/hipdnn_sdk::data_objects::DataType::FLOAT,
      /*xDims=*/{4, 4, 4, 4}, /*xStrides=*/{64, 16, 4, 1},
      /*wDims=*/{4, 4, 1, 1}, /*wStrides=*/{4, 1, 1, 1},
      /*yDims=*/{4, 4, 4, 4}, /*yStrides=*/{64, 16, 4, 1},
      /*convPrePadding=*/{1, 0},  // asymmetric: pre doesn't match post
      /*convPostPadding=*/{2, 1}, // asymmetric: pre doesn't match post
      /*convStrides=*/{1, 1}, /*convDilation=*/{1, 1});
  opGraph.ptr = builder.GetBufferPointer();
  opGraph.size = builder.GetSize();

  // Fusilli plugin should not offer to compile and execute single node
  // conv_fprop with asymmetric padding.
  ASSERT_EQ(hipdnnEnginePluginGetApplicableEngineIds(
                handle, &opGraph, engineIDs.data(), 5, &numEngines),
            HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_EQ(numEngines, 0);
}

TEST(TestFusilliPluginApi, CreateExecutionContext) {
  // Create plugin handle.
  hipdnnEnginePluginHandle_t handle = nullptr;
  ASSERT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_NE(handle, nullptr);

  // UIDs.
  int64_t xUID = 1;
  int64_t wUID = 2;
  int64_t yUID = 3;

  // Dims and strides.
  const std::vector<int64_t> expectedXDims = {4, 4, 4, 4};
  const std::vector<int64_t> expectedXStrides = {64, 16, 4, 1};
  const std::vector<int64_t> expectedWDims = {4, 4, 1, 1};
  const std::vector<int64_t> expectedWStrides = {4, 1, 1, 1};
  const std::vector<int64_t> expectedYDims = {4, 4, 4, 4};
  const std::vector<int64_t> expectedYStrides = {64, 16, 4, 1};
  const hipdnn_sdk::data_objects::DataType dataType =
      hipdnn_sdk::data_objects::DataType::FLOAT;
  fusilli::DataType expectedDataType =
      FUSILLI_PLUGIN_EXPECT_UNWRAP(hipDnnDataTypeToFusilliDataType(dataType));

  // Create a serialized hipDNN conv_fprop.
  auto builder = createValidConvFwdGraph(
      xUID, wUID, yUID, dataType, expectedXDims, expectedXStrides,
      expectedWDims, expectedWStrides, expectedYDims, expectedYStrides);
  hipdnnPluginConstData_t opGraph;
  opGraph.ptr = builder.GetBufferPointer();
  opGraph.size = builder.GetSize();

  // Create engine config.
  flatbuffers::FlatBufferBuilder configBuilder;
  auto engineConfig = hipdnn_sdk::data_objects::CreateEngineConfig(
      configBuilder, FUSILLI_PLUGIN_ENGINE_ID);
  configBuilder.Finish(engineConfig);
  hipdnnPluginConstData_t engineConfigData;
  engineConfigData.ptr = configBuilder.GetBufferPointer();
  engineConfigData.size = configBuilder.GetSize();

  // The function we're actually testing.
  hipdnnEnginePluginExecutionContext_t executionContext = nullptr;
  ASSERT_EQ(hipdnnEnginePluginCreateExecutionContext(
                handle, &engineConfigData, &opGraph, &executionContext),
            HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_NE(executionContext, nullptr);

  auto *ctx =
      static_cast<HipdnnEnginePluginExecutionContext *>(executionContext);

  // Check that we have 3 tensors tracked (x, w, y).
  EXPECT_EQ(ctx->uidToFusilliTensorAttr.size(), 3);

  // Check x tensor properties.
  ASSERT_TRUE(ctx->uidToFusilliTensorAttr.contains(xUID)); // C++ 20
  std::shared_ptr<fusilli::TensorAttr> xTensor =
      ctx->uidToFusilliTensorAttr[xUID];
  EXPECT_EQ(xTensor->getDim(), expectedXDims);
  EXPECT_EQ(xTensor->getStride(), expectedXStrides);
  EXPECT_EQ(xTensor->getDataType(), expectedDataType);
  EXPECT_FALSE(xTensor->isVirtual());

  // Check w tensor properties.
  ASSERT_TRUE(ctx->uidToFusilliTensorAttr.contains(wUID)); // C++ 20
  std::shared_ptr<fusilli::TensorAttr> wTensor =
      ctx->uidToFusilliTensorAttr[wUID];
  EXPECT_EQ(wTensor->getDim(), expectedWDims);
  EXPECT_EQ(wTensor->getStride(), expectedWStrides);
  EXPECT_EQ(wTensor->getDataType(), expectedDataType);
  EXPECT_FALSE(wTensor->isVirtual());

  // Check y tensor properties.
  ASSERT_TRUE(ctx->uidToFusilliTensorAttr.contains(wUID)); // C++ 20
  std::shared_ptr<fusilli::TensorAttr> yTensor =
      ctx->uidToFusilliTensorAttr[yUID];
  EXPECT_EQ(yTensor->getDim(), expectedYDims);
  EXPECT_EQ(yTensor->getStride(), expectedYStrides);
  EXPECT_EQ(yTensor->getDataType(), expectedDataType);
  EXPECT_FALSE(yTensor->isVirtual());

  // Verify graph properties.
  EXPECT_EQ(ctx->graph.context.getIODataType(), expectedDataType);
  EXPECT_EQ(ctx->graph.context.getIntermediateDataType(), expectedDataType);
  EXPECT_EQ(ctx->graph.context.getComputeDataType(), expectedDataType);

  // Clean up.
  EXPECT_EQ(hipdnnEnginePluginDestroyExecutionContext(handle, executionContext),
            HIPDNN_PLUGIN_STATUS_SUCCESS);
  EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestFusilliPluginApi, SetStreamSuccess) {
  // Create plugin handle.
  hipdnnEnginePluginHandle_t handle = nullptr;
  ASSERT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

  // Create a HIP stream.
  hipStream_t stream;
  ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

  // Set the stream on the handle.
  EXPECT_EQ(hipdnnEnginePluginSetStream(handle, stream),
            HIPDNN_PLUGIN_STATUS_SUCCESS);

  // Clean up.
  EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
  EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestFusilliPluginApi, SetStreamNullHandle) {
  // Create a HIP stream.
  hipStream_t stream;
  ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

  // Attempt to set stream with null handle should fail.
  EXPECT_EQ(hipdnnEnginePluginSetStream(nullptr, stream),
            HIPDNN_PLUGIN_STATUS_BAD_PARAM);

  // Verify error was set.
  const char *errorStr = nullptr;
  hipdnnPluginGetLastErrorString(&errorStr);
  ASSERT_NE(errorStr, nullptr);
  EXPECT_GT(strlen(errorStr), 0u);

  // Clean up.
  EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}
