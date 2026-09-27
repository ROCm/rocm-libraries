// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// A downstream consumer that compiles the public frontend umbrella header as C++20.
//
// std::filesystem::path::u8string() returns std::u8string under C++20, so a diagnostic
// that concatenated its result with a narrow literal would make merely including the
// umbrella ill-formed -- no resolution need ever run, and the consumer need not call
// anything. Compiling this translation unit is therefore the entire test: including
// <hipdnn_frontend.hpp> puts Graph.hpp, Types.hpp, PluginPaths.hpp and everything they
// reach under the C++20 standard. It is built as an object target with nothing to
// execute and no backend to link against. The C++17 build of
// hipdnn_frontend_dynamic_load_tests is the control for the other standard.
//
// This compiles the real shipped header; a copy of it would prove nothing.

#include <hipdnn_frontend.hpp>

#include <filesystem>
#include <string>

namespace
{

/// Odr-uses the runtime-load interface a consumer actually depends on, so the
/// definitions behind it are emitted rather than merely parsed. Never called.
[[maybe_unused]] bool consumeRuntimeLoadInterface(const std::filesystem::path& directory)
{
    const bool stored = hipdnn_frontend::setBackendLibraryPath_ext(directory);
    const std::filesystem::path selected = hipdnn_frontend::detail::resolveBackendLibraryPath();
    const std::string diagnostics = hipdnn_frontend::detail::backendLibraryResolution().diagnostics;
    return stored && !selected.empty() && diagnostics.empty();
}

} // namespace
