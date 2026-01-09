#include <gtest/gtest.h>
#include <hipdnn_backend.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::string> getLoadedPlugins(hipdnnHandle_t handle) {
    size_t numPlugins = 0;
    size_t maxPathLength = 0;
    auto status =
        hipdnnGetLoadedEnginePluginPaths_ext(handle, &numPlugins, nullptr, &maxPathLength);

    if (status != HIPDNN_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to get loaded plugin paths");
    }

    if (numPlugins == 0) {
        return {};
    }

    std::vector<std::vector<char>> pathBuffers(numPlugins, std::vector<char>(maxPathLength));
    std::vector<char*> pluginPathsC(numPlugins);
    for (size_t i = 0; i < numPlugins; ++i) {
        pluginPathsC[i] = pathBuffers[i].data();
    }

    status = hipdnnGetLoadedEnginePluginPaths_ext(handle, &numPlugins, pluginPathsC.data(),
                                                  &maxPathLength);
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

}  // anonymous namespace

TEST(HipDNNIntegration, ListLoadedPlugins) {
    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    auto plugins = getLoadedPlugins(handle);

    std::cout << "Loaded plugins: " << plugins.size() << std::endl;
    for (const auto& plugin : plugins) {
        std::cout << "  - " << plugin << std::endl;
    }

    ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}
