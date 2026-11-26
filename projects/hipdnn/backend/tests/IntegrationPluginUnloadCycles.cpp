// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Repeated load and unload of real plugins under HIPDNN_PLUGIN_UNLOAD_EAGER. Every load
// registers the exit handler that marks process exit, so these cases check what that leaves
// behind once the plugins are unloaded again: the exit flag must stay clear, the handlers of
// unloaded plugins must not accumulate, and process exit must still be marked once every
// plugin is unloaded.

#include "PlatformUtils.hpp"
#include "TestPluginConstants.hpp"
#include "hipdnn_backend.h"
#include "plugin/EnginePluginResourceManager.hpp"
#include "plugin/HeuristicPluginResourceManager.hpp"
#include "plugins/codegen_fixture/CodegenFixturePlugin.hpp"

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>
#include <hipdnn_test_sdk/utilities/ProcessUtils.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <set>
#include <string>
#include <thread>
#include <vector>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

using namespace hipdnn_backend::plugin;

namespace
{

std::filesystem::path enginePluginPath()
{
    return hipdnn_backend::platform_utilities::getCurrentModuleDirectory().parent_path()
           / hipdnn_backend::plugin_constants::getTestPluginDefaultDir() / "codegen_fixture"
           / hipdnn_data_sdk::utilities::getLibraryName(TEST_ENGINE_PLUGIN1_NAME);
}

std::filesystem::path heuristicPluginPath()
{
    return hipdnn_backend::plugin_constants::getHeuristicPluginPath(
        TEST_GOOD_HEURISTIC_PLUGIN_NAME);
}

bool isAnyShutdownFlagSet()
{
    return EnginePluginResourceManager::isShuttingDown()
           || HeuristicPluginResourceManager::isShuttingDown();
}

#if defined(__GLIBC__)
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
constexpr bool K_SANITIZED_ALLOCATOR = true;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) \
    || __has_feature(memory_sanitizer)
constexpr bool K_SANITIZED_ALLOCATOR = true;
#else
constexpr bool K_SANITIZED_ALLOCATOR = false;
#endif
#else
constexpr bool K_SANITIZED_ALLOCATOR = false;
#endif

#if defined(HIPDNN_COVERAGE_BUILD)
constexpr bool K_COVERAGE_BUILD = true;
#else
constexpr bool K_COVERAGE_BUILD = false;
#endif

// Bytes the allocator currently hands out, summed over every arena.
size_t heapBytesInUse()
{
#if __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 33)
    return mallinfo2().uordblks;
#else
    return static_cast<size_t>(static_cast<unsigned int>(mallinfo().uordblks));
#endif
}
#endif

#if defined(__linux__)
// Status of the child that HandleCreatedDuringExitLoadsNoPlugin runs, other than a crash.
constexpr int PLUGIN_LOADED_AT_EXIT = 3;
constexpr int PROBE_FAILED_AT_EXIT = 4;

// Creates a handle from its destructor, as an application object destroyed during exit would.
// Exit is already marked by then, so the request must not load any plugin.
class LateHandleCreateProbe
{
public:
    LateHandleCreateProbe() = default;
    LateHandleCreateProbe(const LateHandleCreateProbe&) = delete;
    LateHandleCreateProbe& operator=(const LateHandleCreateProbe&) = delete;
    LateHandleCreateProbe(LateHandleCreateProbe&&) = delete;
    LateHandleCreateProbe& operator=(LateHandleCreateProbe&&) = delete;

    ~LateHandleCreateProbe()
    {
        try
        {
            // Whatever the status, the handle is left to the operating system.
            hipdnnHandle_t handle = nullptr;
            static_cast<void>(hipdnnCreate(&handle));

            if(hipdnn_test_sdk::utilities::isLibraryLoaded(enginePluginPath())
               || hipdnn_test_sdk::utilities::isLibraryLoaded(heuristicPluginPath()))
            {
                std::_Exit(PLUGIN_LOADED_AT_EXIT);
            }
        }
        catch(...)
        {
            std::_Exit(PROBE_FAILED_AT_EXIT);
        }
    }
};
#endif

} // namespace

class IntegrationPluginUnloadCycles : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_TRUE(std::filesystem::exists(enginePluginPath()))
            << "Test precondition: engine plugin missing at " << enginePluginPath();
        ASSERT_TRUE(std::filesystem::exists(heuristicPluginPath()))
            << "Test precondition: heuristic plugin missing at " << heuristicPluginPath();

        _savedEnginePaths = EnginePluginResourceManager::getPluginPaths();
        _savedHeuristicPaths = HeuristicPluginResourceManager::getPluginPaths();

        ASSERT_NO_THROW(EnginePluginResourceManager::setPluginPaths(
            {enginePluginPath()}, HIPDNN_PLUGIN_LOADING_ABSOLUTE));
        ASSERT_NO_THROW(HeuristicPluginResourceManager::setPluginPaths(
            {heuristicPluginPath()}, HIPDNN_PLUGIN_LOADING_ABSOLUTE));
        ASSERT_NO_THROW(
            EnginePluginResourceManager::setPluginUnloadingMode(HIPDNN_PLUGIN_UNLOAD_EAGER));
        ASSERT_NO_THROW(
            HeuristicPluginResourceManager::setPluginUnloadingMode(HIPDNN_PLUGIN_UNLOAD_EAGER));

        ASSERT_FALSE(isAnyShutdownFlagSet()) << "Test precondition: exit flag already set";
    }

    void TearDown() override
    {
        // A child keeps this configuration until it exits, where its exit-time check needs it.
        if(isChild())
        {
            return;
        }

        EnginePluginResourceManager::setPluginUnloadingMode(HIPDNN_DEFAULT_PLUGIN_UNLOADING_MODE);
        HeuristicPluginResourceManager::setPluginUnloadingMode(
            HIPDNN_DEFAULT_PLUGIN_UNLOADING_MODE);
        EnginePluginResourceManager::setPluginPaths(
            std::vector<std::filesystem::path>(_savedEnginePaths.begin(), _savedEnginePaths.end()),
            HIPDNN_PLUGIN_LOADING_ABSOLUTE);
        HeuristicPluginResourceManager::setPluginPaths(
            std::vector<std::filesystem::path>(_savedHeuristicPaths.begin(),
                                               _savedHeuristicPaths.end()),
            HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    }

    static constexpr const char* CHILD_ENV = "HIPDNN_TEST_PLUGIN_UNLOAD_CYCLES_CHILD";

    static bool isChild()
    {
        return !hipdnn_data_sdk::utilities::getEnv(CHILD_ENV).empty();
    }

    // One load and unload: the manager is the only holder of the plugins, so releasing it
    // unloads them.
    static bool engineCycle()
    {
        return EnginePluginResourceManager::create() != nullptr;
    }

    static bool heuristicCycle()
    {
        return HeuristicPluginResourceManager::create() != nullptr;
    }

    static void expectPluginsAnswer()
    {
        const auto engines = EnginePluginResourceManager::create();
        ASSERT_NE(engines, nullptr);
        const auto engineInfos = engines->getEngineInfos();
        EXPECT_TRUE(std::any_of(engineInfos.begin(), engineInfos.end(), [](const auto& info) {
            return info.engineId == codegen_fixture::K_FIXTURE_ENGINE_ID;
        }));

        const auto heuristics = HeuristicPluginResourceManager::create();
        ASSERT_NE(heuristics, nullptr);
        const auto policyId = hipdnn_data_sdk::utilities::policyNameToId("TestGoodHeuristicPolicy");
        EXPECT_NE(heuristics->getPluginForPolicyId(policyId), nullptr);
    }

    std::set<std::filesystem::path> _savedEnginePaths;
    std::set<std::filesystem::path> _savedHeuristicPaths;
};

TEST_F(IntegrationPluginUnloadCycles, ReloadCyclesKeepShutdownFlagClear)
{
    constexpr int CYCLES = 50;
    for(int i = 0; i < CYCLES; ++i)
    {
        ASSERT_TRUE(engineCycle()) << "cycle " << i;
        ASSERT_TRUE(heuristicCycle()) << "cycle " << i;
        ASSERT_FALSE(isAnyShutdownFlagSet()) << "exit flag set after unload cycle " << i;
    }

    EXPECT_FALSE(EnginePluginResourceManager::isShuttingDown());
    EXPECT_FALSE(HeuristicPluginResourceManager::isShuttingDown());
    expectPluginsAnswer();
}

// Loads and unloads race: one thread can load the plugins again while another is still
// unloading the previous load.
TEST_F(IntegrationPluginUnloadCycles, OverlappingReloadsKeepShutdownFlagClear)
{
    constexpr int THREADS = 4;
    constexpr int CYCLES_PER_THREAD = 50;

    std::atomic<int> failedCycles{0};
    std::vector<std::thread> threads;
    threads.reserve(THREADS);
    for(int t = 0; t < THREADS; ++t)
    {
        threads.emplace_back([&failedCycles, t] {
            for(int i = 0; i < CYCLES_PER_THREAD; ++i)
            {
                const bool loaded = (t % 2 == 0) ? engineCycle() : heuristicCycle();
                if(!loaded)
                {
                    failedCycles.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for(auto& thread : threads)
    {
        thread.join();
    }

    EXPECT_EQ(failedCycles.load(), 0);
    EXPECT_FALSE(EnginePluginResourceManager::isShuttingDown());
    EXPECT_FALSE(HeuristicPluginResourceManager::isShuttingDown());
    expectPluginsAnswer();
}

TEST_F(IntegrationPluginUnloadCycles, ReloadCyclesDoNotGrowExitHandlerMemory)
{
#if defined(__GLIBC__)
    if constexpr(K_SANITIZED_ALLOCATOR)
    {
        GTEST_SKIP() << "A sanitizer replaces the allocator, so mallinfo does not measure it";
    }
    if constexpr(K_COVERAGE_BUILD)
    {
        GTEST_SKIP() << "The coverage runtime in the instrumented test plugin retains memory on "
                        "every load, which this heap measurement would count";
    }

    // The first cycles fill lazily created state (function-local statics, loader
    // caches), which is not what is measured here.
    constexpr int WARMUP_CYCLES = 50;
    for(int i = 0; i < WARMUP_CYCLES; ++i)
    {
        ASSERT_TRUE(engineCycle()) << "warm-up cycle " << i;
    }

    // Each load registers one exit handler. A handler left behind by every unload costs
    // tens of bytes, so this many cycles would retain several hundred kilobytes.
    constexpr int CYCLES = 5000;
    constexpr size_t MAX_GROWTH_BYTES = size_t{64} * 1024;

    const size_t before = heapBytesInUse();
    int failedCycles = 0;
    for(int i = 0; i < CYCLES; ++i)
    {
        failedCycles += engineCycle() ? 0 : 1;
    }
    const size_t after = heapBytesInUse();

    EXPECT_EQ(failedCycles, 0);
    const size_t growth = after > before ? after - before : 0;
    EXPECT_LT(growth, MAX_GROWTH_BYTES)
        << "heap in use grew by " << growth << " bytes over " << CYCLES << " unload cycles";
    EXPECT_FALSE(EnginePluginResourceManager::isShuttingDown());
#else
    GTEST_SKIP() << "Exit handler memory is measured through glibc's mallinfo";
#endif
}

// Once every plugin is unloaded no load is left to mark exit, yet a handle requested by an object
// destroyed during exit must still find exit marked and load nothing. The test re-runs itself in
// a child process, which exits with PLUGIN_LOADED_AT_EXIT when that request loads a plugin.
TEST_F(IntegrationPluginUnloadCycles, HandleCreatedDuringExitLoadsNoPlugin)
{
#if defined(__linux__)
    if(!isChild())
    {
        const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter child(CHILD_ENV, "1");
        // The handle created during exit is left to the operating system, which LeakSanitizer
        // would report as a leak.
        const std::string asanOptions = hipdnn_data_sdk::utilities::getEnv("ASAN_OPTIONS");
        const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter noLeakCheck(
            "ASAN_OPTIONS",
            asanOptions.empty() ? "detect_leaks=0" : asanOptions + ":detect_leaks=0");

        const int exitCode = hipdnn_test_sdk::utilities::runCurrentTestInChild();
        EXPECT_EQ(exitCode, 0) << "the child did not exit cleanly. Exit code "
                               << PLUGIN_LOADED_AT_EXIT
                               << ": a handle created during exit loaded a plugin; "
                               << PROBE_FAILED_AT_EXIT
                               << ": the exit-time check itself failed; "
                                  "negative: killed by that signal; anything else: a failed "
                                  "assertion in the child or a crash";
        return;
    }

    // Constructed before the first plugin load of this process, so it is destroyed after
    // every exit handler a load registers.
    static const LateHandleCreateProbe s_probe{};

    // hipdnnCreate runs in the backend library, whose plugin configuration is separate from
    // the copy linked into this binary that the fixture configures, so it is set through the
    // C API here.
    const std::string enginePath = enginePluginPath().string();
    const std::string heuristicPath = heuristicPluginPath().string();
    const std::array<const char*, 1> enginePaths = {enginePath.c_str()};
    const std::array<const char*, 1> heuristicPaths = {heuristicPath.c_str()};
    ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                  enginePaths.size(), enginePaths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnSetHeuristicPluginPaths_ext(
                  heuristicPaths.size(), heuristicPaths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnSetPluginUnloadMode_ext(HIPDNN_PLUGIN_UNLOAD_EAGER), HIPDNN_STATUS_SUCCESS);

    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);
    ASSERT_TRUE(hipdnn_test_sdk::utilities::isLibraryLoaded(enginePluginPath()));
    ASSERT_TRUE(hipdnn_test_sdk::utilities::isLibraryLoaded(heuristicPluginPath()));
    ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);

    // Every plugin manager is released, so nothing of this load is left to mark exit.
    ASSERT_FALSE(hipdnn_test_sdk::utilities::isLibraryLoaded(enginePluginPath()));
    ASSERT_FALSE(hipdnn_test_sdk::utilities::isLibraryLoaded(heuristicPluginPath()));
#else
    GTEST_SKIP() << "Handle creation during exit is checked on Linux only";
#endif
}
