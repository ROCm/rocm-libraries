// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file DynamicBackendLibrary.hpp
 * @brief Runtime resolution of the hipDNN backend shared library.
 *
 * When the frontend is built in runtime-load mode
 * (@ref HIPDNN_FRONTEND_RUNTIME_LOAD_BACKEND), backend entry points are resolved
 * at first use via the cross-platform loader in @c hipdnn_data_sdk::utilities
 * (`dlopen`/`dlsym` on Linux, `LoadLibrary`/`GetProcAddress` on Windows) instead
 * of being linked directly. This keeps a header-only consumer from inheriting a
 * hard dependency on `libhipdnn_backend.so`.
 *
 * The library handle is opened exactly once and shared by every entry-point
 * resolver. These helpers deliberately never emit log messages: the logging
 * callback itself is resolved through here during frontend logging
 * initialization, so logging from this layer would re-enter the loader.
 */

#pragma once

#include <atomic>
#include <cstdio>
#include <mutex>
#include <string>

#include <hipdnn_data_sdk/Visibility.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

namespace hipdnn_frontend::detail
{

/**
 * @brief Return the lazily-opened handle to the hipDNN backend shared library.
 *
 * The library is opened on first call via `std::call_once`; the result
 * (including a failure, cached as `nullptr`) is reused thereafter. On failure it
 * writes directly to stderr rather than the frontend logging facility, so it is
 * safe to call from the logging-initialization path.
 *
 * `HIPDNN_HIDDEN` gives each shared object its own handle, matching the per-SO
 * isolation of the backend instance accessor.
 *
 * @return The library handle, or `nullptr` if the backend could not be loaded.
 */
HIPDNN_HIDDEN inline hipdnn_data_sdk::utilities::SharedLibraryHandle backendLibraryHandle()
{
    static hipdnn_data_sdk::utilities::SharedLibraryHandle s_handle = nullptr;
    static std::once_flag s_once;

    std::call_once(s_once, [] {
        try
        {
            const std::string libraryName
                = hipdnn_data_sdk::utilities::getLibraryName("hipdnn_backend");
            s_handle = hipdnn_data_sdk::utilities::openLibrary(libraryName);
        }
        catch(const std::exception& e)
        {
            // Report via stderr directly rather than the frontend logging
            // facility: that callback is itself resolved through this loader, so
            // logging here would re-enter it.
            std::fprintf(stderr, "hipDNN: failed to load backend library: %s\n", e.what());
            s_handle = nullptr;
        }
        catch(...)
        {
            std::fprintf(stderr, "hipDNN: failed to load backend library (unknown error)\n");
            s_handle = nullptr;
        }
    });

    return s_handle;
}

/**
 * @brief Resolve a symbol from the already-opened backend library.
 *
 * Returns `nullptr` if the library could not be loaded or the symbol is not
 * found. Never logs.
 */
HIPDNN_HIDDEN inline void* resolveSymbol(const char* symbolName)
{
    const auto handle = backendLibraryHandle();
    if(handle == nullptr)
    {
        return nullptr;
    }
    return hipdnn_data_sdk::utilities::getSymbol(handle, symbolName);
}

/**
 * @brief Resolve a backend entry point on first use and cache the result.
 *
 * @tparam Fn         Function-pointer type of the entry point.
 * @param cache       Per-entry-point cache (start at `nullptr`).
 * @param symbolName  C symbol name to resolve.
 * @return The resolved function pointer, or `nullptr` if it could not be found.
 */
template <typename Fn>
HIPDNN_HIDDEN inline Fn resolveBackendSymbol(std::atomic<void*>& cache, const char* symbolName)
{
    void* resolved = cache.load(std::memory_order_acquire);
    if(resolved == nullptr)
    {
        resolved = resolveSymbol(symbolName);
        if(resolved != nullptr)
        {
            cache.store(resolved, std::memory_order_release);
        }
    }
    return reinterpret_cast<Fn>(resolved);
}

} // namespace hipdnn_frontend::detail
