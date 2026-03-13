// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_frontend/Graph.hpp>

#include <algorithm>
#include <filesystem>
#include <format>
#include <string>

using hipdnn_data_sdk::utilities::getCurrentExecutableDirectory;

static std::vector<std::string> getLoadedPlugins(hipdnnHandle_t handle) {
  size_t numPlugins = 0;
  size_t maxPathLength = 0;
  auto status = hipdnnGetLoadedEnginePluginPaths_ext(handle, &numPlugins,
                                                     nullptr, &maxPathLength);

  if (status != HIPDNN_STATUS_SUCCESS) {
    throw std::runtime_error("Failed to get loaded plugin paths");
  }

  if (numPlugins == 0) {
    return {};
  }

  std::vector<std::vector<char>> pathBuffers(numPlugins,
                                             std::vector<char>(maxPathLength));
  std::vector<char *> pluginPathsC(numPlugins);
  for (size_t i = 0; i < numPlugins; ++i) {
    pluginPathsC[i] = pathBuffers[i].data();
  }

  status = hipdnnGetLoadedEnginePluginPaths_ext(
      handle, &numPlugins, pluginPathsC.data(), &maxPathLength);
  if (status != HIPDNN_STATUS_SUCCESS) {
    throw std::runtime_error("Failed to get loaded plugin paths");
  }

  std::vector<std::string> pluginPaths;
  pluginPaths.reserve(numPlugins);
  for (size_t i = 0; i < numPlugins; ++i) {
    pluginPaths.emplace_back(pluginPathsC[i]);
  }
  return pluginPaths;
}

TEST(IntegrationTests, PluginLoad) {
  // Set plugin paths.
  //
  // FUSILLI_PLUGIN_PATH is a relative path from the executable directory where
  // this test lives to the location of the plugin's .so e.g.
  // "../lib/hipdnn_plugins/engines/fusilli_plugin". The tests will be
  // installed, and therefore re-located, in some build configurations
  // (`TheRock` for example) so we must use a relative path.
  auto pluginPath = std::filesystem::canonical(getCurrentExecutableDirectory() /
                                               FUSILLI_PLUGIN_PATH);
  const std::array<const char *, 1> paths = {pluginPath.c_str()};
  hipdnnStatus_t status = hipdnnSetEnginePluginPaths_ext(
      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE);
  EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

  // Stand up enough of hipDNN to load plugins.
  hipdnnHandle_t handle = nullptr;
  status = hipdnnCreate(&handle);
  ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
  ASSERT_NE(handle, nullptr);

  // If fusilli plugin fails to define a required method it will fail to load.
  auto loadedPlugins = getLoadedPlugins(handle);
  EXPECT_EQ(loadedPlugins.size(), 1);

  // Check that fusilli plugin did load.
  auto expectedPath =
      pluginPath / std::format("lib{}.so", FUSILLI_PLUGIN_TARGET);
  EXPECT_TRUE(std::ranges::any_of(
      loadedPlugins, [&expectedPath](const std::string &loadedPluginPath) {
        return std::filesystem::canonical(loadedPluginPath) ==
               std::filesystem::canonical(expectedPath);
      }));

  EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}
