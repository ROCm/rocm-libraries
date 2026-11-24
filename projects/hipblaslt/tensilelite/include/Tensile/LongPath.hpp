// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <system_error>

namespace TensileLite
{
    // Windows rejects a path of MAX_PATH (260) characters or more unless the
    // system enables long paths and the executable's manifest opts in. A
    // library cannot opt in for the process that loads it, so such a path has
    // to be opened in its \\?\ (verbatim) form: absolute, with single
    // backslash separators.
    constexpr std::size_t windowsMaxPath = 260;

    // True when `path` is in the \\?\ form, which only wide-path APIs accept.
    inline bool isVerbatimPath(std::filesystem::path const& path)
    {
#ifdef _WIN32
        return path.native().rfind(L"\\\\?\\", 0) == 0;
#else
        static_cast<void>(path);
        return false;
#endif
    }

    // Returns `path` in a form the standard library can open on this platform.
    // Off Windows, and for shorter paths, that is `path` itself.
    inline std::filesystem::path longPathSafe(std::string const& path)
    {
        std::filesystem::path given(path);
#ifdef _WIN32
        if(isVerbatimPath(given))
            return given;

        std::error_code ec;
        auto            absolute = std::filesystem::absolute(given, ec);
        if(ec)
            return given;
        absolute = absolute.lexically_normal();

        std::wstring const& native = absolute.native();
        if(native.size() < windowsMaxPath)
            return given;
        if(native.rfind(L"\\\\", 0) == 0)
            return std::filesystem::path(L"\\\\?\\UNC\\" + native.substr(2));
        return std::filesystem::path(L"\\\\?\\" + native);
#else
        return given;
#endif
    }
} // namespace TensileLite
