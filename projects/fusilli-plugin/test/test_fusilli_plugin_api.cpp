// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <hipdnn_sdk/plugin/EnginePluginApi.h>
#include <hipdnn_sdk/plugin/PluginApi.h>

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <spdlog/spdlog.h>
#include <string>
#include <unistd.h>
#include <vector>

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
