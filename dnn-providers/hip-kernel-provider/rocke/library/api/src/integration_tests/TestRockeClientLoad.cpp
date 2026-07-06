// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

#include <array>
#include <filesystem>
#include <string>
#include <vector>

namespace
{

class HipdnnHandle
{
public:
    HipdnnHandle()
    {
        EXPECT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    }

    ~HipdnnHandle()
    {
        if(_handle != nullptr)
        {
            EXPECT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
    }

    hipdnnHandle_t get() const
    {
        return _handle;
    }

private:
    hipdnnHandle_t _handle = nullptr;
};

std::vector<std::string> getLoadedEnginePluginPaths(hipdnnHandle_t handle)
{
    size_t numPluginPaths = 0;
    size_t maxStringLen = 0;
    EXPECT_EQ(hipdnnGetLoadedEnginePluginPaths_ext(handle, &numPluginPaths, nullptr, &maxStringLen),
              HIPDNN_STATUS_SUCCESS);

    std::vector<std::string> storage(numPluginPaths, std::string(maxStringLen, '\0'));
    std::vector<char*> pluginPaths;
    pluginPaths.reserve(numPluginPaths);
    for(auto& path : storage)
    {
        pluginPaths.push_back(path.data());
    }

    EXPECT_EQ(hipdnnGetLoadedEnginePluginPaths_ext(
                  handle, &numPluginPaths, pluginPaths.data(), &maxStringLen),
              HIPDNN_STATUS_SUCCESS);

    for(auto& path : storage)
    {
        path.resize(std::char_traits<char>::length(path.c_str()));
    }

    return storage;
}

} // namespace

TEST(TestRockeClientLoad, HipdnnLoadsPluginFromBuildTreePath)
{
    const auto pluginPath = std::filesystem::weakly_canonical(
        hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / PLUGIN_PATH);
    const std::string pluginPathStr = pluginPath.string();
    const std::array<const char*, 1> paths = {pluginPathStr.c_str()};

    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);

    const HipdnnHandle handle;
    ASSERT_NE(handle.get(), nullptr);

    const auto loadedPaths = getLoadedEnginePluginPaths(handle.get());
    ASSERT_EQ(loadedPaths.size(), 1u);
    EXPECT_NE(loadedPaths[0].find("rocke-client"), std::string::npos);
}
