// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Runtime-load static check: this translation unit includes the full hipDNN
// frontend umbrella and links the hipdnn_frontend_dynamic target (see
// CMakeLists.txt), which compiles with HIPDNN_FRONTEND_RUNTIME_LOAD_BACKEND and
// does NOT link libhipdnn_backend. If any frontend code path referenced a backend
// symbol directly, the executable would fail to link with an unresolved-symbol
// error. A successful link therefore proves that, in runtime-load mode, every
// backend entry point is reached only via dlopen/dlsym.
//
// exerciseRuntimeLoadSurface() is defined (so its body is emitted and its
// references must be resolved by the linker) but never executed: it would dlopen
// the backend and call uninitialized handles at runtime.

#include <gtest/gtest.h>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/Logging.hpp>

#include <string>
#include <vector>

namespace
{
[[maybe_unused]] hipdnn_frontend::Error exerciseRuntimeLoadSurface()
{
    using namespace hipdnn_frontend;

    // Backend instance + handle lifecycle (Handle.hpp -> hipdnnBackend() factory
    // -> HipdnnDynamicBackendWrapper create/setStream/destroy).
    auto [handle, handleErr] = createHipdnnHandle();
    if(handleErr.is_bad())
    {
        return handleErr;
    }
    HIPDNN_CHECK_ERROR(setHipdnnHandleStream(handle, nullptr));

    // Graph build path (Graph.hpp and the packer/unpacker detail headers all
    // route descriptor/finalize/execute calls through the backend wrapper).
    graph::Graph graph;
    graph.set_io_data_type(DataType::FLOAT).set_compute_data_type(DataType::FLOAT);
    HIPDNN_CHECK_ERROR(graph.build(*handle));

    // Plugin path API (PluginPaths.hpp -> wrapper plugin path calls).
    const std::vector<std::string> pluginPaths;
    HIPDNN_CHECK_ERROR(setEnginePluginPaths(pluginPaths, PluginLoadingMode::MODE_ABSOLUTE));
    HIPDNN_CHECK_ERROR(setHeuristicPluginPaths(pluginPaths, PluginLoadingMode::MODE_ABSOLUTE));
    // Logging APIs (Logging.hpp) -- the only headers besides the wrapper that
    // previously referenced backend symbols directly.
    hipdnnSeverity_t level = HIPDNN_SEV_OFF;
    HIPDNN_CHECK_ERROR(getGlobalLogLevel(level));
    HIPDNN_CHECK_ERROR(setGlobalLogLevel(level));
    HIPDNN_CHECK_ERROR(setUserLogCallback(nullptr, HIPDNN_SEV_OFF, LogCallbackMode::SYNC, &graph));
    (void)initializeFrontendLogging();

    return {};
}
} // namespace

TEST(TestRuntimeLoadBackendStaticCheck, LinksWithoutBackendLibrary)
{
    // Reference (but never call) the surface so the linker must resolve every
    // backend reference reachable from it.
    using SurfaceFn = hipdnn_frontend::Error (*)();
    volatile SurfaceFn surface = &exerciseRuntimeLoadSurface;
    EXPECT_NE(surface, nullptr);
}
