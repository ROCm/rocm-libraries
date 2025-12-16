// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <miopen/filesystem_checker.hpp>
#include <miopen/expanduser.hpp>

namespace miopen {

bool FilesystemChecker::IsNetworkedFilesystem(const fs::path& path) const
{
    // Call the existing free function from expanduser.cpp
    // Note: Prefer using IFilesystemChecker interface for better testability
    return miopen::IsNetworkedFilesystem(path);
}

namespace {
IFilesystemChecker* g_filesystem_checker = nullptr;
FilesystemChecker g_default_checker;
} // namespace

IFilesystemChecker& GetFilesystemChecker()
{
    return g_filesystem_checker ? *g_filesystem_checker : g_default_checker;
}

void SetFilesystemChecker(IFilesystemChecker* checker) { g_filesystem_checker = checker; }

} // namespace miopen
