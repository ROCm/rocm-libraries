// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <flatbuffers/flatbuffer_builder.h>
#include <fusilli.h>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/logging/LogLevel.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/MatmulAttributes.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/attributes/RMSNormAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_plugin_sdk/PluginApi.h>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unistd.h>
#include <vector>

#include "fusilli_serialized_plan_payload.h"
#include "graph_import.h"
#include "hipdnn_engine_plugin_execution_context.h"
#include "utils.h"
#include "version.h"

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

// Build matmul + pointwise graph using frontend API.
std::vector<uint8_t>
buildMatmulActivGraph(const std::vector<int64_t> &aDims,
                      const std::vector<int64_t> &bDims,
                      const std::vector<int64_t> &cDims,
                      hipdnn_frontend::PointwiseMode activMode) {
  hipdnn_frontend::graph::Graph graph;
  graph.set_name("MatmulActivTest")
      .set_io_data_type(hipdnn_frontend::DataType::FLOAT)
      .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
      .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT);

  int64_t uid = 1;

  // Input A: [M, K]
  auto aAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
  aAttr->set_uid(uid++)
      .set_name("A")
      .set_data_type(hipdnn_frontend::DataType::FLOAT)
      .set_dim(aDims)
      .set_stride({aDims[1], 1});

  // Input B: [K, N]
  auto bAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
  bAttr->set_uid(uid++)
      .set_name("B")
      .set_data_type(hipdnn_frontend::DataType::FLOAT)
      .set_dim(bDims)
      .set_stride({bDims[1], 1});

  // Matmul: A x B -> C_matmul (virtual)
  hipdnn_frontend::graph::MatmulAttributes matmulAttrs;
  matmulAttrs.set_name("Matmul");
  auto cMatmul = graph.matmul(aAttr, bAttr, matmulAttrs);

  // Pointwise activation: C_matmul -> C
  hipdnn_frontend::graph::PointwiseAttributes pointwiseAttrs;
  pointwiseAttrs.set_mode(activMode);
  auto cOut = graph.pointwise(cMatmul, pointwiseAttrs);
  cOut->set_uid(uid++)
      .set_name("Activ")
      .set_data_type(hipdnn_frontend::DataType::FLOAT)
      .set_dim(cDims)
      .set_stride({cDims[1], 1})
      .set_output(true);

  auto result = graph.validate();
  if (result.is_bad()) {
    throw std::runtime_error("Graph validation failed: " +
                             result.get_message());
  }

  auto [serializedGraph, serErr] = graph.to_binary();
  if (serErr.is_bad()) {
    throw std::runtime_error("Graph serialization failed: " +
                             serErr.get_message());
  }
  return serializedGraph;
}

flatbuffers::DetachedBuffer buildFusilliEngineConfig() {
  flatbuffers::FlatBufferBuilder configBuilder;
  auto engineConfig = hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(
      configBuilder, hipdnn_data_sdk::utilities::FUSILLI_ENGINE_ID);
  configBuilder.Finish(engineConfig);
  return configBuilder.Release();
}

// Build an inference-phase rmsnorm graph using the frontend API.
std::vector<uint8_t> buildRmsnormInferenceGraph() {
  hipdnn_frontend::graph::Graph graph;
  graph.set_name("RmsnormTest")
      .set_io_data_type(hipdnn_frontend::DataType::FLOAT)
      .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
      .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT);

  const std::vector<int64_t> dims = {1, 3, 224, 224};
  const std::vector<int64_t> strides = {150528, 50176, 224, 1};
  const std::vector<int64_t> scaleDims = {1, 3, 224, 224};
  const std::vector<int64_t> scaleStrides = {150528, 50176, 224, 1};

  auto xAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
  xAttr->set_uid(1)
      .set_name("x")
      .set_data_type(hipdnn_frontend::DataType::FLOAT)
      .set_dim(dims)
      .set_stride(strides);

  auto scaleAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
  scaleAttr->set_uid(2)
      .set_name("scale")
      .set_data_type(hipdnn_frontend::DataType::FLOAT)
      .set_dim(scaleDims)
      .set_stride(scaleStrides);

  auto epsilonAttr =
      std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
  epsilonAttr->set_name("epsilon").set_value(1e-5f).set_uid(3);

  hipdnn_frontend::graph::RMSNormAttributes rmsnormAttrs;
  rmsnormAttrs.set_name("rmsnorm")
      .set_epsilon(epsilonAttr)
      .set_forward_phase(hipdnn_frontend::NormFwdPhase::INFERENCE);

  auto [yAttr, invRmsAttr] = graph.rmsnorm(xAttr, scaleAttr, rmsnormAttrs);
  yAttr->set_uid(4)
      .set_name("y")
      .set_data_type(hipdnn_frontend::DataType::FLOAT)
      .set_dim(dims)
      .set_stride(strides)
      .set_output(true);

  auto result = graph.validate();
  if (result.is_bad()) {
    throw std::runtime_error("Graph validation failed: " +
                             result.get_message());
  }

  auto [serializedGraph, serErr] = graph.to_binary();
  if (serErr.is_bad()) {
    throw std::runtime_error("Graph serialization failed: " +
                             serErr.get_message());
  }
  return serializedGraph;
}

// Build a training-phase rmsnorm graph (with inv_rms output) using the
// frontend API. Fusilli does not yet support TRAINING phase for rmsnorm,
// so this graph is expected to be rejected by the plugin.
std::vector<uint8_t> buildRmsnormTrainingGraph() {
  hipdnn_frontend::graph::Graph graph;
  graph.set_name("RmsnormTrainingTest")
      .set_io_data_type(hipdnn_frontend::DataType::FLOAT)
      .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
      .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT);

  const std::vector<int64_t> dims = {1, 3, 224, 224};
  const std::vector<int64_t> strides = {150528, 50176, 224, 1};
  const std::vector<int64_t> scaleDims = {1, 3, 224, 224};
  const std::vector<int64_t> scaleStrides = {150528, 50176, 224, 1};

  auto xAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
  xAttr->set_uid(1)
      .set_name("x")
      .set_data_type(hipdnn_frontend::DataType::FLOAT)
      .set_dim(dims)
      .set_stride(strides);

  auto scaleAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
  scaleAttr->set_uid(2)
      .set_name("scale")
      .set_data_type(hipdnn_frontend::DataType::FLOAT)
      .set_dim(scaleDims)
      .set_stride(scaleStrides);

  auto epsilonAttr =
      std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
  epsilonAttr->set_name("epsilon").set_value(1e-5f).set_uid(3);

  hipdnn_frontend::graph::RMSNormAttributes rmsnormAttrs;
  rmsnormAttrs.set_name("rmsnorm")
      .set_epsilon(epsilonAttr)
      .set_forward_phase(hipdnn_frontend::NormFwdPhase::TRAINING);

  auto [yAttr, invRmsAttr] = graph.rmsnorm(xAttr, scaleAttr, rmsnormAttrs);
  yAttr->set_uid(4)
      .set_name("y")
      .set_data_type(hipdnn_frontend::DataType::FLOAT)
      .set_dim(dims)
      .set_stride(strides)
      .set_output(true);
  invRmsAttr->set_uid(5)
      .set_name("inv_rms")
      .set_data_type(hipdnn_frontend::DataType::FLOAT)
      .set_output(true);

  auto result = graph.validate();
  if (result.is_bad()) {
    throw std::runtime_error("Graph validation failed: " +
                             result.get_message());
  }

  auto [serializedGraph, serErr] = graph.to_binary();
  if (serErr.is_bad()) {
    throw std::runtime_error("Graph serialization failed: " +
                             serErr.get_message());
  }
  return serializedGraph;
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
  EXPECT_STREQ(name, hipdnn_data_sdk::utilities::FUSILLI_ENGINE_NAME);
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
  EXPECT_STREQ(version, FUSILLI_PROVIDER_VERSION_STRING);
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
  EXPECT_EQ(engineIds[0], hipdnn_data_sdk::utilities::FUSILLI_ENGINE_ID);
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

TEST(TestFusilliPluginApi, GetApplicableEngineIds) {
  // Create plugin handle.
  hipdnnEnginePluginHandle_t handle = nullptr;
  ASSERT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_NE(handle, nullptr);

  // Create a serialized hipDNN batch norm graph.
  auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph();
  hipdnnPluginConstData_t opGraph;
  opGraph.ptr = builder.GetBufferPointer();
  opGraph.size = builder.GetSize();

  // Fusilli plugin should not offer to compile and execute bach norm (yet).
  std::array<int64_t, 5> engineIDs;
  uint32_t numEngines = 10;
  ASSERT_EQ(hipdnnEnginePluginGetApplicableEngineIds(
                /*handle=*/handle, /*op_graph=*/&opGraph,
                /*engine_ids=*/engineIDs.data(), /*max_engines=*/5,
                /*num_engines=*/&numEngines),
            HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_EQ(numEngines, 0);

  // Create a serialized hipDNN conv_fprop graph with symmetric padding.
  builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
  opGraph.ptr = builder.GetBufferPointer();
  opGraph.size = builder.GetSize();

  // Fusilli plugin should offer to compile and execute single node conv_fprop.
  ASSERT_EQ(hipdnnEnginePluginGetApplicableEngineIds(
                /*handle=*/handle, /*op_graph=*/&opGraph,
                /*engine_ids=*/engineIDs.data(), /*max_engines=*/5,
                /*num_engines=*/&numEngines),
            HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_EQ(numEngines, 1);
  ASSERT_EQ(engineIDs[0], hipdnn_data_sdk::utilities::FUSILLI_ENGINE_ID);

  // Create a serialized hipDNN conv_fprop graph with asymmetric padding.
  builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph(
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
                /*handle=*/handle, /*op_graph=*/&opGraph,
                /*engine_ids=*/engineIDs.data(), /*max_engines=*/5,
                /*num_engines=*/&numEngines),
            HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_EQ(numEngines, 0);
}

TEST(TestFusilliPluginApi, GetApplicableEngineIdsConvDGrad) {
  // Create plugin handle.
  hipdnnEnginePluginHandle_t handle = nullptr;
  ASSERT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_NE(handle, nullptr);

  std::array<int64_t, 5> engineIDs;
  uint32_t numEngines = 0;

  // Create a serialized hipDNN conv_bwd (dgrad) graph with symmetric padding.
  auto builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph();
  hipdnnPluginConstData_t opGraph;
  opGraph.ptr = builder.GetBufferPointer();
  opGraph.size = builder.GetSize();

  // Fusilli plugin should offer to compile and execute single node conv_dgrad.
  ASSERT_EQ(hipdnnEnginePluginGetApplicableEngineIds(
                /*handle=*/handle, /*op_graph=*/&opGraph,
                /*engine_ids=*/engineIDs.data(), /*max_engines=*/5,
                /*num_engines=*/&numEngines),
            HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_EQ(numEngines, 1);
  ASSERT_EQ(engineIDs[0], hipdnn_data_sdk::utilities::FUSILLI_ENGINE_ID);

  // Create a serialized hipDNN conv_bwd (dgrad) graph with asymmetric padding.
  builder = hipdnn_test_sdk::utilities::createValidConvBwdGraph(
      /*dxDims=*/{4, 4, 4, 4}, /*dxStrides=*/{64, 16, 4, 1},
      /*wDims=*/{4, 4, 1, 1}, /*wStrides=*/{4, 1, 1, 1},
      /*dyDims=*/{4, 4, 4, 4}, /*dyStrides=*/{64, 16, 4, 1},
      /*convPrePadding=*/{1, 0},  // asymmetric: pre doesn't match post
      /*convPostPadding=*/{2, 1}, // asymmetric: pre doesn't match post
      /*convStrides=*/{1, 1}, /*convDilation=*/{1, 1});
  opGraph.ptr = builder.GetBufferPointer();
  opGraph.size = builder.GetSize();

  // Fusilli plugin should not offer to compile and execute conv_dgrad with
  // asymmetric padding.
  ASSERT_EQ(hipdnnEnginePluginGetApplicableEngineIds(
                /*handle=*/handle, /*op_graph=*/&opGraph,
                /*engine_ids=*/engineIDs.data(), /*max_engines=*/5,
                /*num_engines=*/&numEngines),
            HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_EQ(numEngines, 0);

  EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestFusilliPluginApi, GetApplicableEngineIdsConvPointwise) {
  // Create plugin handle.
  hipdnnEnginePluginHandle_t handle = nullptr;
  ASSERT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_NE(handle, nullptr);

  std::array<int64_t, 5> engineIDs;
  uint32_t numEngines = 0;

  // Test conv + unary pointwise activation for various modes.
  // (conv -> binary -> pointwise covered in
  // GetApplicableEngineIdsConvBiasActiv)
  for (auto mode :
       {hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SIGMOID_FWD,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::TANH_FWD,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::GELU_FWD,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ELU_FWD}) {
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdActivGraph(
        /*xDims=*/{4, 4, 4, 4}, /*xStrides=*/{64, 16, 4, 1},
        /*wDims=*/{4, 4, 1, 1}, /*wStrides=*/{4, 1, 1, 1},
        /*yDims=*/{4, 4, 4, 4}, /*yStrides=*/{64, 16, 4, 1},
        /*convPrePadding=*/{0, 0}, /*convPostPadding=*/{0, 0},
        /*convStrides=*/{1, 1}, /*convDilation=*/{1, 1},
        /*activMode=*/mode);
    hipdnnPluginConstData_t opGraph;
    opGraph.ptr = builder.GetBufferPointer();
    opGraph.size = builder.GetSize();

    ASSERT_EQ(hipdnnEnginePluginGetApplicableEngineIds(
                  handle, &opGraph, engineIDs.data(), 5, &numEngines),
              HIPDNN_PLUGIN_STATUS_SUCCESS);

    // If the translation (hipDNN -> fusilli) is supported, the graph should be
    // supported.
    bool modeSupported =
        !fusilli::isError(hipDnnPointwiseModeToFusilliMode(mode));
    uint32_t expectedEngines = modeSupported ? 1 : 0;
    ASSERT_EQ(numEngines, expectedEngines);
  }

  EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestFusilliPluginApi, GetApplicableEngineIdsConvBiasActiv) {
  // Create plugin handle.
  hipdnnEnginePluginHandle_t handle = nullptr;
  ASSERT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_NE(handle, nullptr);

  std::array<int64_t, 5> engineIDs;
  uint32_t numEngines = 0;

  // Graph structure: conv -> bias (ADD) -> activation
  for (auto activMode :
       {hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SIGMOID_FWD,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::TANH_FWD}) {
    auto builder = hipdnn_test_sdk::utilities::createValidConvFwdBiasActivGraph(
        /*xDims=*/{4, 4, 4, 4}, /*xStrides=*/{64, 16, 4, 1},
        /*wDims=*/{4, 4, 1, 1}, /*wStrides=*/{4, 1, 1, 1},
        /*yDims=*/{4, 4, 4, 4}, /*yStrides=*/{64, 16, 4, 1},
        /*convPrePadding=*/{0, 0}, /*convPostPadding=*/{0, 0},
        /*convStrides=*/{1, 1}, /*convDilation=*/{1, 1},
        /*activMode=*/activMode);
    hipdnnPluginConstData_t opGraph;
    opGraph.ptr = builder.GetBufferPointer();
    opGraph.size = builder.GetSize();

    ASSERT_EQ(hipdnnEnginePluginGetApplicableEngineIds(
                  handle, &opGraph, engineIDs.data(), 5, &numEngines),
              HIPDNN_PLUGIN_STATUS_SUCCESS);

    // Graph should be supported if translation for activation is supported.
    bool activSupported =
        !fusilli::isError(hipDnnPointwiseModeToFusilliMode(activMode));
    uint32_t expectedEngines = activSupported ? 1 : 0;
    ASSERT_EQ(numEngines, expectedEngines);
  }

  EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestFusilliPluginApi, GetApplicableEngineIdsMatmulPointwise) {
  hipdnnEnginePluginHandle_t handle = nullptr;
  ASSERT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

  std::array<int64_t, 5> engineIDs;
  uint32_t numEngines = 0;

  // Test matmul + unary pointwise for various activation modes.
  for (auto mode : {hipdnn_frontend::PointwiseMode::RELU_FWD,
                    hipdnn_frontend::PointwiseMode::SIGMOID_FWD,
                    hipdnn_frontend::PointwiseMode::TANH_FWD,
                    hipdnn_frontend::PointwiseMode::GELU_FWD}) {
    auto serializedGraph = buildMatmulActivGraph(
        /*aDims=*/{4, 8}, /*bDims=*/{8, 5}, /*cDims=*/{4, 5}, mode);

    hipdnnPluginConstData_t opGraph;
    opGraph.ptr = serializedGraph.data();
    opGraph.size = serializedGraph.size();

    ASSERT_EQ(hipdnnEnginePluginGetApplicableEngineIds(
                  handle, &opGraph, engineIDs.data(), 5, &numEngines),
              HIPDNN_PLUGIN_STATUS_SUCCESS);

    // Graph supported if pointwise mode translates to fusilli.
    auto sdkMode = hipdnn_test_sdk::utilities::frontendToSdkPointwiseMode(mode);
    bool modeSupported =
        !fusilli::isError(hipDnnPointwiseModeToFusilliMode(sdkMode));
    uint32_t expectedEngines = modeSupported ? 1 : 0;
    ASSERT_EQ(numEngines, expectedEngines);
  }

  EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestFusilliPluginApi, GetApplicableEngineIdsInt4NonBatchedMatmul) {
  hipdnnEnginePluginHandle_t handle = nullptr;
  ASSERT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);

  // Non-batched (2D) mixed-precision int4 x fp16 matmul is not supported —
  // mixed element types require rank-3 tensors (torch.bmm). The plugin should
  // report 0 engines.
  using DT = hipdnn_flatbuffers_sdk::data_objects::DataType;
  flatbuffers::FlatBufferBuilder builder;
  std::vector<int64_t> aDims = {4, 8}, aStrides = {8, 1};
  std::vector<int64_t> bDims = {8, 4}, bStrides = {4, 1};
  std::vector<int64_t> cDims = {4, 4}, cStrides = {4, 1};

  std::vector<::flatbuffers::Offset<
      hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
      tensors;
  tensors.push_back(
      hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
          builder, 1, "A", DT::INT4, &aStrides, &aDims));
  tensors.push_back(
      hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
          builder, 2, "B", DT::HALF, &bStrides, &bDims));
  tensors.push_back(
      hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
          builder, 3, "C", DT::HALF, &cStrides, &cDims));

  auto matmulAttr =
      hipdnn_flatbuffers_sdk::data_objects::CreateMatmulAttributes(builder, 1,
                                                                   2, 3);
  std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>>
      nodes;
  nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
      builder, "matmul", DT::FLOAT,
      hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::MatmulAttributes,
      matmulAttr.Union()));

  auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
      builder, "test", DT::HALF, DT::HALF, DT::FLOAT, &tensors, &nodes);
  builder.Finish(graphOffset);

  hipdnnPluginConstData_t opGraph;
  opGraph.ptr = builder.GetBufferPointer();
  opGraph.size = builder.GetSize();

  std::array<int64_t, 5> engineIDs;
  uint32_t numEngines = 10;
  ASSERT_EQ(hipdnnEnginePluginGetApplicableEngineIds(
                handle, &opGraph, engineIDs.data(), 5, &numEngines),
            HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_EQ(numEngines, 0);

  EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
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
  const hipdnn_flatbuffers_sdk::data_objects::DataType dataType =
      hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
  FUSILLI_PLUGIN_EXPECT_OR_ASSIGN(fusilli::DataType expectedDataType,
                                  hipDnnDataTypeToFusilliDataType(dataType));

  // Create a serialized hipDNN conv_fprop.
  // Note: createValidConvFwdGraph uses hardcoded UIDs 1, 2, 3 for x, w, y
  auto builder = hipdnn_test_sdk::utilities::createValidConvFwdGraph(
      expectedXDims, expectedXStrides, expectedWDims, expectedWStrides,
      expectedYDims, expectedYStrides, /*convPrePadding=*/{0, 0},
      /*convPostPadding=*/{0, 0}, /*convStrides=*/{1, 1},
      /*convDilation=*/{1, 1}, dataType);
  hipdnnPluginConstData_t opGraph;
  opGraph.ptr = builder.GetBufferPointer();
  opGraph.size = builder.GetSize();

  // Create engine config.
  flatbuffers::FlatBufferBuilder configBuilder;
  auto engineConfig = hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(
      configBuilder, hipdnn_data_sdk::utilities::FUSILLI_ENGINE_ID);
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
  ASSERT_TRUE(ctx->uidToFusilliTensorAttr.contains(xUID)); // C++20
  std::shared_ptr<fusilli::TensorAttr> xTensor =
      ctx->uidToFusilliTensorAttr[xUID];
  EXPECT_EQ(xTensor->getDim(), expectedXDims);
  EXPECT_EQ(xTensor->getStride(), expectedXStrides);
  EXPECT_EQ(xTensor->getDataType(), expectedDataType);
  EXPECT_FALSE(xTensor->isVirtual());

  // Check w tensor properties.
  ASSERT_TRUE(ctx->uidToFusilliTensorAttr.contains(wUID)); // C++20
  std::shared_ptr<fusilli::TensorAttr> wTensor =
      ctx->uidToFusilliTensorAttr[wUID];
  EXPECT_EQ(wTensor->getDim(), expectedWDims);
  EXPECT_EQ(wTensor->getStride(), expectedWStrides);
  EXPECT_EQ(wTensor->getDataType(), expectedDataType);
  EXPECT_FALSE(wTensor->isVirtual());

  // Check y tensor properties.
  ASSERT_TRUE(ctx->uidToFusilliTensorAttr.contains(wUID)); // C++20
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

TEST(TestFusilliPluginApi, SerializeExecutionContextAddsPayloadHeader) {
  hipdnnEnginePluginHandle_t handle = nullptr;
  ASSERT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_NE(handle, nullptr);

  auto serializedGraph = buildMatmulActivGraph(
      {4, 8}, {8, 5}, {4, 5}, hipdnn_frontend::PointwiseMode::RELU_FWD);
  hipdnnPluginConstData_t opGraph{serializedGraph.data(),
                                  serializedGraph.size()};

  auto engineConfigBuffer = buildFusilliEngineConfig();
  hipdnnPluginConstData_t engineConfig{engineConfigBuffer.data(),
                                       engineConfigBuffer.size()};

  hipdnnEnginePluginExecutionContext_t executionContext = nullptr;
  ASSERT_EQ(hipdnnEnginePluginCreateExecutionContext(
                handle, &engineConfig, &opGraph, &executionContext),
            HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_NE(executionContext, nullptr);

  hipdnnPluginConstData_t serializedContext{nullptr, 0};
  ASSERT_EQ(hipdnnEnginePluginSerializeExecutionContext(
                handle, executionContext, &serializedContext),
            HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_NE(serializedContext.ptr, nullptr);
  ASSERT_GT(serializedContext.size,
            fusilli_plugin::serialized_plan_payload::HEADER_SIZE);

  const auto *bytes = static_cast<const uint8_t *>(serializedContext.ptr);
  EXPECT_EQ(std::memcmp(bytes,
                        fusilli_plugin::serialized_plan_payload::MAGIC.data(),
                        fusilli_plugin::serialized_plan_payload::MAGIC.size()),
            0);
  EXPECT_EQ(bytes[8], static_cast<uint8_t>(1));
  EXPECT_EQ(bytes[9], static_cast<uint8_t>(0));
  EXPECT_EQ(bytes[12], static_cast<uint8_t>(1));
  EXPECT_EQ(bytes[13], static_cast<uint8_t>(0));
  EXPECT_EQ(bytes[14], static_cast<uint8_t>(0));
  EXPECT_EQ(bytes[15], static_cast<uint8_t>(0));
  EXPECT_EQ(bytes[16],
            static_cast<uint8_t>(
                fusilli_plugin::serialized_plan_payload::HEADER_SIZE));
  EXPECT_EQ(bytes[17], static_cast<uint8_t>(0));
  EXPECT_EQ(bytes[18], static_cast<uint8_t>(0));
  EXPECT_EQ(bytes[19], static_cast<uint8_t>(0));

  EXPECT_EQ(hipdnnEnginePluginDestroySerializedExecutionContext(
                handle, &serializedContext),
            HIPDNN_PLUGIN_STATUS_SUCCESS);
  EXPECT_EQ(hipdnnEnginePluginDestroyExecutionContext(handle, executionContext),
            HIPDNN_PLUGIN_STATUS_SUCCESS);
  EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestFusilliPluginApi,
     CreateExecutionContextFromSerializedRejectsUnsupportedPayloadVersion) {
  hipdnnEnginePluginHandle_t handle = nullptr;
  ASSERT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_NE(handle, nullptr);

  auto serializedGraph = buildMatmulActivGraph(
      {4, 8}, {8, 5}, {4, 5}, hipdnn_frontend::PointwiseMode::RELU_FWD);
  hipdnnPluginConstData_t opGraph{serializedGraph.data(),
                                  serializedGraph.size()};

  auto engineConfigBuffer = buildFusilliEngineConfig();
  hipdnnPluginConstData_t engineConfig{engineConfigBuffer.data(),
                                       engineConfigBuffer.size()};

  hipdnnEnginePluginExecutionContext_t executionContext = nullptr;
  ASSERT_EQ(hipdnnEnginePluginCreateExecutionContext(
                handle, &engineConfig, &opGraph, &executionContext),
            HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_NE(executionContext, nullptr);

  hipdnnPluginConstData_t serializedContext{nullptr, 0};
  ASSERT_EQ(hipdnnEnginePluginSerializeExecutionContext(
                handle, executionContext, &serializedContext),
            HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_NE(serializedContext.ptr, nullptr);

  std::vector<uint8_t> incompatiblePayload(
      static_cast<const uint8_t *>(serializedContext.ptr),
      static_cast<const uint8_t *>(serializedContext.ptr) +
          serializedContext.size);
  incompatiblePayload[8] = 2;
  incompatiblePayload[9] = 0;

  EXPECT_EQ(hipdnnEnginePluginDestroySerializedExecutionContext(
                handle, &serializedContext),
            HIPDNN_PLUGIN_STATUS_SUCCESS);
  EXPECT_EQ(hipdnnEnginePluginDestroyExecutionContext(handle, executionContext),
            HIPDNN_PLUGIN_STATUS_SUCCESS);

  hipdnnPluginConstData_t incompatibleSerializedContext{
      incompatiblePayload.data(), incompatiblePayload.size()};
  hipdnnEnginePluginExecutionContext_t restoredExecutionContext = nullptr;
  EXPECT_EQ(
      hipdnnEnginePluginCreateExecutionContextFromSerialized(
          handle, &incompatibleSerializedContext, &restoredExecutionContext),
      HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
  EXPECT_EQ(restoredExecutionContext, nullptr);

  EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestFusilliPluginApi,
     CreateExecutionContextFromSerializedRejectsLegacyRawGraphPayload) {
  hipdnnEnginePluginHandle_t handle = nullptr;
  ASSERT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_NE(handle, nullptr);

  auto serializedGraph = buildMatmulActivGraph(
      {4, 8}, {8, 5}, {4, 5}, hipdnn_frontend::PointwiseMode::RELU_FWD);
  hipdnnPluginConstData_t rawGraphPayload{serializedGraph.data(),
                                          serializedGraph.size()};

  hipdnnEnginePluginExecutionContext_t executionContext = nullptr;
  EXPECT_EQ(hipdnnEnginePluginCreateExecutionContextFromSerialized(
                handle, &rawGraphPayload, &executionContext),
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
  EXPECT_EQ(executionContext, nullptr);

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

TEST(TestFusilliPluginApi, GetApplicableEngineIdsSdpa) {
  hipdnnEnginePluginHandle_t handle = nullptr;
  ASSERT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_NE(handle, nullptr);

  std::array<int64_t, 5> engineIDs;
  uint32_t numEngines = 0;

  const std::vector<int64_t> qkvDims = {2, 8, 16, 64};
  const std::vector<int64_t> qkvStrides = {qkvDims[1] * qkvDims[2] * qkvDims[3],
                                           qkvDims[2] * qkvDims[3], qkvDims[3],
                                           1};

  // Basic SDPA (Q, K, V -> O) should be supported.
  auto builder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
      qkvDims, qkvStrides, qkvDims, qkvStrides, qkvDims, qkvStrides, qkvDims,
      qkvStrides);
  hipdnnPluginConstData_t opGraph;
  opGraph.ptr = builder.GetBufferPointer();
  opGraph.size = builder.GetSize();

  ASSERT_EQ(hipdnnEnginePluginGetApplicableEngineIds(
                handle, &opGraph, engineIDs.data(), 5, &numEngines),
            HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_EQ(numEngines, 1);
  ASSERT_EQ(engineIDs[0], hipdnn_data_sdk::utilities::FUSILLI_ENGINE_ID);

  // SDPA with attn_mask should be supported.
  builder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
      qkvDims, qkvStrides, qkvDims, qkvStrides, qkvDims, qkvStrides, qkvDims,
      qkvStrides, hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
      /*withAttnMask=*/true);
  opGraph.ptr = builder.GetBufferPointer();
  opGraph.size = builder.GetSize();

  ASSERT_EQ(hipdnnEnginePluginGetApplicableEngineIds(
                handle, &opGraph, engineIDs.data(), 5, &numEngines),
            HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_EQ(numEngines, 1);

  // SDPA with independent K/V head counts (GQA) should be supported.
  // Mirrors test_cudnn_attention_gqa: Q=32 heads, K=8 heads, V=4 heads.
  {
    const std::vector<int64_t> qDims = {4, 32, 512, 128};
    const std::vector<int64_t> qStrides = {32 * 512 * 128, 512 * 128, 128, 1};
    const std::vector<int64_t> kDims = {4, 8, 1024, 128};
    const std::vector<int64_t> kStrides = {8 * 1024 * 128, 1024 * 128, 128, 1};
    const std::vector<int64_t> vDims = {4, 4, 1024, 128};
    const std::vector<int64_t> vStrides = {4 * 1024 * 128, 1024 * 128, 128, 1};
    const std::vector<int64_t> oDims = {4, 32, 512, 128};
    const std::vector<int64_t> oStrides = {32 * 512 * 128, 512 * 128, 128, 1};
    builder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        qDims, qStrides, kDims, kStrides, vDims, vStrides, oDims, oStrides,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16);
    opGraph.ptr = builder.GetBufferPointer();
    opGraph.size = builder.GetSize();

    ASSERT_EQ(hipdnnEnginePluginGetApplicableEngineIds(
                  handle, &opGraph, engineIDs.data(), 5, &numEngines),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_EQ(numEngines, 1);
  }

  // SDPA with stats output is NOT supported (yet).
  builder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
      qkvDims, qkvStrides, qkvDims, qkvStrides, qkvDims, qkvStrides, qkvDims,
      qkvStrides, hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
      /*withAttnMask=*/false, /*withScale=*/false, /*withStats=*/true);
  opGraph.ptr = builder.GetBufferPointer();
  opGraph.size = builder.GetSize();

  ASSERT_EQ(hipdnnEnginePluginGetApplicableEngineIds(
                handle, &opGraph, engineIDs.data(), 5, &numEngines),
            HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_EQ(numEngines, 0);

  EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestFusilliPluginApi, GetApplicableEngineIdsRmsnorm) {
  hipdnnEnginePluginHandle_t handle = nullptr;
  ASSERT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_NE(handle, nullptr);

  std::array<int64_t, 5> engineIDs;
  uint32_t numEngines = 0;

  // A basic rmsnorm inference graph (x, scale, epsilon -> y) should be
  // supported.
  auto serializedGraph = buildRmsnormInferenceGraph();
  hipdnnPluginConstData_t opGraph;
  opGraph.ptr = serializedGraph.data();
  opGraph.size = serializedGraph.size();

  ASSERT_EQ(hipdnnEnginePluginGetApplicableEngineIds(
                handle, &opGraph, engineIDs.data(), 5, &numEngines),
            HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_EQ(numEngines, 1);
  ASSERT_EQ(engineIDs[0], hipdnn_data_sdk::utilities::FUSILLI_ENGINE_ID);

  EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

// Fusilli does not yet support TRAINING phase for rmsnorm. The plugin
// should decline to claim such a graph.
TEST(TestFusilliPluginApi, GetApplicableEngineIdsRmsnormTrainingUnsupported) {
  hipdnnEnginePluginHandle_t handle = nullptr;
  ASSERT_EQ(hipdnnEnginePluginCreate(&handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_NE(handle, nullptr);

  std::array<int64_t, 5> engineIDs;
  uint32_t numEngines = 0;

  auto serializedGraph = buildRmsnormTrainingGraph();
  hipdnnPluginConstData_t opGraph;
  opGraph.ptr = serializedGraph.data();
  opGraph.size = serializedGraph.size();

  ASSERT_EQ(hipdnnEnginePluginGetApplicableEngineIds(
                handle, &opGraph, engineIDs.data(), 5, &numEngines),
            HIPDNN_PLUGIN_STATUS_SUCCESS);
  ASSERT_EQ(numEngines, 0);

  EXPECT_EQ(hipdnnEnginePluginDestroy(handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestFusilliPluginApi, SetLogLevelSuccess) {
  EXPECT_EQ(hipdnnPluginSetLogLevel(HIPDNN_SEV_INFO),
            HIPDNN_PLUGIN_STATUS_SUCCESS);
  EXPECT_EQ(hipdnnPluginSetLogLevel(HIPDNN_SEV_WARN),
            HIPDNN_PLUGIN_STATUS_SUCCESS);
}
