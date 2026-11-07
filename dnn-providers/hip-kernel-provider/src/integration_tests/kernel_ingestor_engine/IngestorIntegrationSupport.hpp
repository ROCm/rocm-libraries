// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <cstddef>
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

// A test name's dtype token: "Fp16" for HALF, "Bf16" otherwise (the 16-bit conv engines).
inline std::string dtypeSuffix(hipdnn_frontend::DataType dtype)
{
    return dtype == hipdnn_frontend::DataType::HALF ? "Fp16" : "Bf16";
}

// The engine log's count of the candidates a plan build ranked.
inline std::string candidateCount(size_t candidates)
{
    return std::to_string(candidates) + " candidate(s)";
}

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

// The already-loaded provider's own shared-object path. Does not load a second copy.
inline std::filesystem::path loadedProviderPluginPath()
{
    const std::filesystem::path pluginTarget(PLUGIN_PATH);
    const auto pluginFile = hipdnn_data_sdk::utilities::LIB_PREFIX
                            + pluginTarget.filename().string()
                            + hipdnn_data_sdk::utilities::SHARED_LIB_EXT;
    return std::filesystem::weakly_canonical(
        hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / pluginTarget.parent_path()
        / pluginFile);
}

// Clears the process-wide kpack module cache of the loaded provider.
// Makes the loaded provider read kpack code bytes from disk again.
// The provider keeps one module for each archive, entry and architecture for the life of
// the process. A case that must read code bytes again calls this function first.
// The function finds the provider that the harness loaded. It does not load a second copy.
// Returns a description of the failure, or an empty string.
inline std::string resetProviderModuleCaches()
{
    const auto pluginPath = loadedProviderPluginPath();

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

// Sets the loaded provider's own per-image byte limit for hipkernel:Gfx950ConvDgradAot's
// batch chunker (Gfx950ConvDgradGeometry.hpp::batchChunkByteLimitForTesting()), so a
// small batch exercises the chunked-launch device path without a multi-GiB image. The
// limit lives in a function-local static: a test in another TU that includes the header
// and mutates it directly holds its own copy, never the plugin's, which is why this goes
// through the plugin's exported hipdnnEnginePluginSetGfx950ConvDgradAotBatchChunkByteLimit
// ForTesting symbol instead. Returns a description of the failure, or an empty string.
inline std::string setProviderDgradBatchChunkByteLimit(int64_t limit)
{
    const auto pluginPath = loadedProviderPluginPath();

    auto* library = hipdnn_data_sdk::utilities::openLoadedLibrary(pluginPath);
    if(library == nullptr)
    {
        return "the provider at " + pluginPath.string()
               + " is not loaded, so there is no batch chunk limit to set";
    }

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto* setLimit = reinterpret_cast<void (*)(int64_t)>(hipdnn_data_sdk::utilities::getSymbol(
        library, "hipdnnEnginePluginSetGfx950ConvDgradAotBatchChunkByteLimitForTesting"));
    if(setLimit == nullptr)
    {
        hipdnn_data_sdk::utilities::closeLibrary(library);
        return "the provider at " + pluginPath.string()
               + " exports no "
                 "hipdnnEnginePluginSetGfx950ConvDgradAotBatchChunkByteLimitForTesting; it was "
                 "built without HIPDNN_ENABLE_KERNEL_INGESTOR, or the test-only hook was removed";
    }

    setLimit(limit);

    hipdnn_data_sdk::utilities::closeLibrary(library);
    return {};
}

// The loaded provider's count of hipkernel:Gfx950ConvDgradAot implicit-GEMM launches, one
// per batch slice (exported as
// hipdnnEnginePluginGfx950ConvDgradAotImplicitLaunchCountForTesting). Writes a
// description of the failure to error and returns -1 if the hook is not there.
inline int64_t providerDgradImplicitLaunchCount(std::string& error)
{
    const auto pluginPath = loadedProviderPluginPath();
    auto* library = hipdnn_data_sdk::utilities::openLoadedLibrary(pluginPath);
    if(library == nullptr)
    {
        error = "the provider at " + pluginPath.string() + " is not loaded";
        return -1;
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto* count = reinterpret_cast<int64_t (*)()>(hipdnn_data_sdk::utilities::getSymbol(
        library, "hipdnnEnginePluginGfx950ConvDgradAotImplicitLaunchCountForTesting"));
    const auto value = count == nullptr ? -1 : count();
    if(count == nullptr)
    {
        error = "the provider at " + pluginPath.string()
                + " exports no hipdnnEnginePluginGfx950ConvDgradAotImplicitLaunchCountForTesting";
    }
    hipdnn_data_sdk::utilities::closeLibrary(library);
    return value;
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
