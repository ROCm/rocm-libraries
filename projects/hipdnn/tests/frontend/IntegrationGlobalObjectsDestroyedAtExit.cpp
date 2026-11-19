// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// A handle or built graph owned by a namespace-scope object is destroyed during process exit,
// after everything the program created in main, including the frontend's backend instance and
// the loaded plugins. Each test re-runs itself in a child process that leaves such an object
// behind, and expects the child to exit with status 0.
//
// The child loads only the test_good engine plugin and runs with HIPDNN_PLUGIN_UNLOAD_EAGER, so
// a late destroy that reaches the plugins releases the last plugin manager and unloads the
// plugin. A probe destroyed after the object checks that the plugin is still loaded and exits
// with PLUGIN_RELEASED_AT_EXIT otherwise, so a late plugin call fails the test even when it
// does not crash.

#include <array>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#define HIPDNN_UNDEF_NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#define HIPDNN_UNDEF_WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#ifdef HIPDNN_UNDEF_NOMINMAX
#undef NOMINMAX
#undef HIPDNN_UNDEF_NOMINMAX
#endif
#ifdef HIPDNN_UNDEF_WIN32_LEAN_AND_MEAN
#undef WIN32_LEAN_AND_MEAN
#undef HIPDNN_UNDEF_WIN32_LEAN_AND_MEAN
#endif
#else
#include <fstream>
#include <unistd.h>
#endif

#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

constexpr const char* CHILD_ENV = "HIPDNN_TEST_EXIT_ORDER_CHILD";

// Child exit statuses other than a crash.
constexpr int PLUGIN_RELEASED_AT_EXIT = 3;
constexpr int PROBE_FAILED_AT_EXIT = 4;
constexpr int DESTROY_FAILED_AT_EXIT = 5;

bool isLibraryLoaded(const std::string& fileName)
{
#if defined(_WIN32)
    return GetModuleHandleA(fileName.c_str()) != nullptr;
#else
    const std::string pathSuffix = "/" + fileName;
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while(std::getline(maps, line))
    {
        if(line.find(pathSuffix) != std::string::npos)
        {
            return true;
        }
    }
    return false;
#endif
}

std::string testGoodPluginFileName()
{
    return std::filesystem::path(hipdnn_tests::plugin_constants::testGoodPluginPath())
        .filename()
        .string();
}

// Defined before the objects below, so it is destroyed after them.
class PluginUnloadProbe
{
public:
    PluginUnloadProbe() = default;
    PluginUnloadProbe(const PluginUnloadProbe&) = delete;
    PluginUnloadProbe& operator=(const PluginUnloadProbe&) = delete;
    PluginUnloadProbe(PluginUnloadProbe&&) = delete;
    PluginUnloadProbe& operator=(PluginUnloadProbe&&) = delete;

    ~PluginUnloadProbe()
    {
        try
        {
            if(!_pluginFile.empty() && !isLibraryLoaded(_pluginFile))
            {
                std::_Exit(PLUGIN_RELEASED_AT_EXIT);
            }
        }
        catch(...)
        {
            std::_Exit(PROBE_FAILED_AT_EXIT);
        }
    }

    void arm(std::string pluginFile)
    {
        _pluginFile = std::move(pluginFile);
    }

private:
    std::string _pluginFile;
};

// Destroys a backend handle from its destructor, as an application object would.
class LateBackendHandle
{
public:
    LateBackendHandle() = default;
    LateBackendHandle(const LateBackendHandle&) = delete;
    LateBackendHandle& operator=(const LateBackendHandle&) = delete;
    LateBackendHandle(LateBackendHandle&&) = delete;
    LateBackendHandle& operator=(LateBackendHandle&&) = delete;

    ~LateBackendHandle()
    {
        if(_handle != nullptr && hipdnnDestroy(_handle) != HIPDNN_STATUS_SUCCESS)
        {
            std::_Exit(DESTROY_FAILED_AT_EXIT);
        }
    }

    void adopt(hipdnnHandle_t handle)
    {
        _handle = handle;
    }

private:
    hipdnnHandle_t _handle = nullptr;
};

PluginUnloadProbe gPluginUnloadProbe;
HipdnnHandlePtr gHandle;
std::shared_ptr<Graph> gGraph;
LateBackendHandle gBackendHandle;

std::shared_ptr<Graph> makeConvFpropGraph()
{
    auto graph = std::make_shared<Graph>();
    graph->set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto makeTensor = [](const std::vector<int64_t>& dims, const std::vector<int64_t>& strides) {
        auto tensor = std::make_shared<TensorAttributes>();
        tensor->set_dim(dims).set_stride(strides).set_data_type(DataType::FLOAT);
        return tensor;
    };
    auto x = makeTensor({2, 3, 8, 8}, {192, 64, 8, 1});
    auto w = makeTensor({4, 3, 3, 3}, {27, 9, 3, 1});

    ConvFpropAttributes conv;
    conv.set_padding({1, 1}).set_stride({1, 1}).set_dilation({1, 1});
    graph->conv_fprop(x, w, conv)->set_output(true);
    return graph;
}

class GlobalObjectDestroyedAtExit : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
    }

    static bool isChild()
    {
        return !hipdnn_data_sdk::utilities::getEnv(CHILD_ENV).empty();
    }

    static void useTestGoodEnginePlugin()
    {
        const std::array<const char*, 1> paths
            = {hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);
    }

    // Configures the child before its first handle: the engine plugin is unloaded as soon as
    // nothing holds it, and the probe checks at exit that something still does.
    static void prepareChild()
    {
        ASSERT_NO_FATAL_FAILURE(useTestGoodEnginePlugin());
        ASSERT_EQ(hipdnnSetPluginUnloadMode_ext(HIPDNN_PLUGIN_UNLOAD_EAGER), HIPDNN_STATUS_SUCCESS);
        gPluginUnloadProbe.arm(testGoodPluginFileName());
    }

    static void currentExecutablePath(std::string& selfPath)
    {
#if defined(_WIN32)
        selfPath.assign(MAX_PATH, '\0');
        const DWORD length
            = GetModuleFileNameA(nullptr, selfPath.data(), static_cast<DWORD>(selfPath.size()));
        ASSERT_GT(length, 0U);
        ASSERT_LT(length, selfPath.size());
        selfPath.resize(length);
#else
        selfPath.assign(4096, '\0');
        const auto length = readlink("/proc/self/exe", selfPath.data(), selfPath.size() - 1);
        ASSERT_GT(length, 0);
        ASSERT_LT(static_cast<size_t>(length), selfPath.size() - 1);
        selfPath.resize(static_cast<size_t>(length));
#endif
    }

    // Re-runs the current test in a child process and expects it to exit with status 0.
    static void expectChildExitsCleanly()
    {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        ASSERT_NE(info, nullptr);
        const std::string testId = std::string(info->test_suite_name()) + "." + info->name();

        std::string selfPath;
        ASSERT_NO_FATAL_FAILURE(currentExecutablePath(selfPath));

        const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter child(CHILD_ENV, "1");
        // Objects released during exit are left to the operating system, which LeakSanitizer
        // would report as leaks.
        const std::string asanOptions = hipdnn_data_sdk::utilities::getEnv("ASAN_OPTIONS");
        const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter noLeakCheck(
            "ASAN_OPTIONS",
            asanOptions.empty() ? "detect_leaks=0" : asanOptions + ":detect_leaks=0");

        const std::string command = "\"" + selfPath + "\" --gtest_filter=" + testId;
        const int status = std::system(command.c_str());
        EXPECT_EQ(status, 0) << "child " << testId << " did not exit cleanly (exit "
                             << PLUGIN_RELEASED_AT_EXIT
                             << ": the engine plugin was released during exit; "
                             << DESTROY_FAILED_AT_EXIT
                             << ": hipdnnDestroy failed during exit; otherwise a crash)";
    }
};

// Suite name of the original exploratory reproduction.
using Issue_A2_01_global_handle_destroyed_at_exit = GlobalObjectDestroyedAtExit;

TEST_F(Issue_A2_01_global_handle_destroyed_at_exit, ProcessExitsCleanly)
{
    if(!isChild())
    {
        expectChildExitsCleanly();
        return;
    }

    ASSERT_NO_FATAL_FAILURE(prepareChild());
    auto [handle, error] = createHipdnnHandle();
    ASSERT_TRUE(error.is_good()) << error.get_message();
    ASSERT_TRUE(isLibraryLoaded(testGoodPluginFileName()));
    gHandle = std::move(handle);
}

TEST_F(Issue_A2_01_global_handle_destroyed_at_exit, GlobalBuiltGraphDestroyedAtExitExitsCleanly)
{
    if(!isChild())
    {
        ASSERT_NO_FATAL_FAILURE(useTestGoodEnginePlugin());
        auto [handle, error] = createHipdnnHandle();
        ASSERT_TRUE(error.is_good()) << error.get_message();
        if(makeConvFpropGraph()->build(*handle).is_bad())
        {
            GTEST_SKIP() << "No engine builds the convolution graph on this device";
        }
        handle.reset();
        expectChildExitsCleanly();
        return;
    }

    ASSERT_NO_FATAL_FAILURE(prepareChild());
    auto [handle, error] = createHipdnnHandle();
    ASSERT_TRUE(error.is_good()) << error.get_message();
    gGraph = makeConvFpropGraph();
    const auto buildError = gGraph->build(*handle);
    ASSERT_TRUE(buildError.is_good()) << buildError.get_message();
    handle.reset();
    // The built plan, not the handle, now holds the engine plugin.
    ASSERT_TRUE(isLibraryLoaded(testGoodPluginFileName()));
}

TEST_F(Issue_A2_01_global_handle_destroyed_at_exit, GlobalBackendHandleDestroyedAtExitExitsCleanly)
{
#if defined(_WIN32)
    GTEST_SKIP() << "The backend's exit handlers run when its DLL detaches, after the "
                    "executable's static destructors, so a late hipdnnDestroy there runs while "
                    "every plugin is still loaded and releases the plugin as before exit";
#else
    if(!isChild())
    {
        expectChildExitsCleanly();
        return;
    }

    ASSERT_NO_FATAL_FAILURE(prepareChild());
    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);
    gBackendHandle.adopt(handle);
    ASSERT_TRUE(isLibraryLoaded(testGoodPluginFileName()));
#endif
}

} // namespace
