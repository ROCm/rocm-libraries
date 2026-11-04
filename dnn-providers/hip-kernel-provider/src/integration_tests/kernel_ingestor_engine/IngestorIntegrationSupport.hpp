// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

// The two forms in which the frontend saves a built plan.
enum class SaveForm
{
    GRAPH_AND_PLAN,
    PLAN_ONLY
};

namespace detail
{

// The plugin log line of a kernel that plan build selects without benchmarking.
inline constexpr const char* SELECTED_KERNEL_LOG = "' selected kernel ";

// The plugin log line of each path that selects or benchmarks a kernel at plan build.
inline constexpr std::array<const char*, 4> KERNEL_SELECTION_LOG_MARKERS = {
    SELECTED_KERNEL_LOG, "' served kernel ", "' will benchmark ", "benchmarking selected kernel"};

// The plugin log line of a new provider container.
inline constexpr const char* CREATING_CONTAINER_LOG = "Creating Container";

// Clears the process-wide kpack module cache of the loaded provider.
// Makes the loaded provider read kpack code bytes from disk again.
// The provider keeps one module for each archive, entry and architecture for the life of
// the process. A case that must read code bytes again calls this function first.
// The function finds the provider that the harness loaded. It does not load a second copy.
// Returns a description of the failure, or an empty string.
inline std::string resetProviderModuleCaches()
{
    const std::filesystem::path pluginTarget(PLUGIN_PATH);
    const auto pluginFile = hipdnn_data_sdk::utilities::LIB_PREFIX
                            + pluginTarget.filename().string()
                            + hipdnn_data_sdk::utilities::SHARED_LIB_EXT;
    const auto pluginPath = std::filesystem::weakly_canonical(
        hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / pluginTarget.parent_path()
        / pluginFile);

    auto* library = hipdnn_data_sdk::utilities::openLoadedLibrary(pluginPath);
    if(library == nullptr)
    {
        return "the provider at " + pluginPath.string()
               + " is not loaded, so there are no resident modules to reset";
    }

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto* reset = reinterpret_cast<void (*)()>(hipdnn_data_sdk::utilities::getSymbol(
        library, "hipdnnEnginePluginResetKpackModuleCacheForTesting"));
    if(reset == nullptr)
    {
        hipdnn_data_sdk::utilities::closeLibrary(library);
        return "the provider at " + pluginPath.string()
               + " exports no hipdnnEnginePluginResetKpackModuleCacheForTesting; it was "
                 "built without HIPDNN_ENABLE_KERNEL_INGESTOR, or the test-only reset "
                 "hook was removed";
    }

    reset();

    // Releases only the reference that this function took. The harness keeps the provider loaded.
    hipdnn_data_sdk::utilities::closeLibrary(library);
    return {};
}

// Replaces handle with a new hipDNN handle that uses stream.
// Destroy every graph built with the old handle before you call this function.
// When no other handle is alive, the next plugin handle gets a new provider container.
// Wrap the call in ASSERT_NO_FATAL_FAILURE.
inline void replaceHandleWithAFreshOne(hipdnnHandle_t& handle, hipStream_t stream)
{
    ASSERT_EQ(resetProviderModuleCaches(), "");

    ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
    handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnSetStream(handle, stream), HIPDNN_STATUS_SUCCESS);
}

// Saves the built plan of graph in form.
inline std::pair<std::vector<uint8_t>, hipdnn_frontend::Error>
    saveInForm(const hipdnn_frontend::graph::Graph& graph, SaveForm form)
{
    return form == SaveForm::GRAPH_AND_PLAN ? graph.to_binary() : graph.to_compiled_plan_binary();
}

// Loads bytes that saveInForm wrote in form into graph.
inline hipdnn_frontend::Error loadInForm(hipdnn_frontend::graph::Graph& graph,
                                         hipdnnHandle_t handle,
                                         const std::vector<uint8_t>& bytes,
                                         SaveForm form)
{
    return form == SaveForm::GRAPH_AND_PLAN ? graph.from_binary(handle, bytes)
                                            : graph.from_compiled_plan_binary(handle, bytes);
}

// Fails the test for each kernel selection or benchmarking line in recorder.
inline void
    expectNoKernelSelectionLogged(const hipdnn_test_sdk::utilities::LogRecorderBase& recorder)
{
    for(const char* marker : KERNEL_SELECTION_LOG_MARKERS)
    {
        EXPECT_FALSE(recorder.hasLogContaining(marker))
            << "found '" << marker << "'. Captured logs:\n"
            << recorder.getRecordedLogsAsString();
    }
}

} // namespace detail
} // namespace hip_kernel_provider::kernel_ingestor_engine::integration
