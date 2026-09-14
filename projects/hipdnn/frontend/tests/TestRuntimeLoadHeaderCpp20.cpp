// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// A downstream consumer that compiles the shipped runtime-load headers as C++20.
//
// std::filesystem::path::u8string() returns std::u8string under C++20, so a diagnostic
// that concatenated its result with a narrow literal would make merely including these
// non-template inline definitions ill-formed -- no resolution need ever run, and the
// consumer need not even call anything. Compiling this translation unit is therefore
// the entire test: it is built as an object target with nothing to execute and no
// backend to link against. The C++17 build of hipdnn_frontend_dynamic_load_tests is
// the control for the other standard.
//
// This includes the real shipped header. A copy of it would prove nothing.

#include <hipdnn_frontend/detail/DynamicBackendLibrary.hpp>

#include <filesystem>
#include <string>

namespace
{

/// Odr-uses the runtime-load interface a consumer actually depends on, so the
/// definitions behind it are emitted rather than merely parsed. Never called.
[[maybe_unused]] bool consumeRuntimeLoadInterface(const std::filesystem::path& directory)
{
    const bool stored = hipdnn_frontend::setBackendLibraryPath(directory);
    const std::filesystem::path selected = hipdnn_frontend::detail::resolveBackendLibraryPath();
    const std::string diagnostics = hipdnn_frontend::detail::backendLibraryResolution().diagnostics;
    return stored && !selected.empty() && diagnostics.empty();
}

} // namespace
