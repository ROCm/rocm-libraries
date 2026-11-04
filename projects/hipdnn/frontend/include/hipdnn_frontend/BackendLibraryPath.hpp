// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file BackendLibraryPath.hpp
 * @brief Frontend API for pointing backend resolution at a directory
 *
 * Extension API: `_ext` marks a hipDNN addition with no cuDNN counterpart.
 *
 * The backend is resolved at runtime only by consumers linking
 * `hipdnn_frontend_dynamic`; in a direct-link build the backend is satisfied at link
 * time and `<hipdnn_frontend.hpp>` does not include this header. Nothing guards the
 * header itself, so a direct-link consumer can include it explicitly and will get
 * `true` from the setter, but no resolution ever consults the stored directory.
 *
 * @code{.cpp}
 * #include <hipdnn_frontend.hpp>
 *
 * // Record the directory backend resolution searches first. This must run before the
 * // calling module's first hipDNN call, which resolves the backend and closes the setter.
 * if (!hipdnn_frontend::setBackendLibraryPath_ext("/opt/rocm/lib")) {
 *     // resolution already ran; the directory was not stored
 * }
 * @endcode
 */

#pragma once

#include <filesystem>
#include <mutex>

#include <hipdnn_data_sdk/Visibility.hpp>
#include <hipdnn_frontend/detail/DynamicBackendLibrary.hpp>

namespace hipdnn_frontend
{

/**
 * @brief Point backend resolution at @p directory, ahead of every other location.
 *
 * Must be called before the calling module's first hipDNN call: that call resolves the
 * backend and closes the setter for the life of the process.
 *
 * Applies only to the calling module (executable or shared library), not other
 * modules' frontend instances. For process-wide injection, use the
 * `HIPDNN_BACKEND_LIBRARY_PATH` environment variable; this setter takes
 * precedence within its module.
 *
 * @param directory Absolute directory holding the backend shared library; hipDNN
 *     appends the platform's backend filename to it. An empty or relative value is
 *     stored, reported on stderr, and ignored when resolution runs.
 * @return `true` if stored; `false` without changes once resolution has started,
 *     including after a cached failure.
 */
// NOLINTNEXTLINE(readability-identifier-naming)
HIPDNN_HIDDEN inline bool setBackendLibraryPath_ext(const std::filesystem::path& directory)
{
    auto& state = detail::backendLibraryOverrideState();
    const std::lock_guard<std::mutex> lock(state.mutex);
    if(state.resolutionStarted)
    {
        return false;
    }
    state.directory = directory;
    return true;
}

} // namespace hipdnn_frontend
