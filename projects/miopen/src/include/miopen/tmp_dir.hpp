// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifndef MIOPEN_GUARD_MLOPEN_TMP_DIR_HPP
#define MIOPEN_GUARD_MLOPEN_TMP_DIR_HPP

#include <string_view>
#include <miopen/filesystem.hpp>
#include <miopen/config.hpp>

namespace miopen {

struct TmpDir
{
    fs::path path;
    MIOPEN_INTERNALS_EXPORT explicit TmpDir(std::string_view prefix = "");

    TmpDir(TmpDir&&) noexcept            = default;
    TmpDir& operator=(TmpDir&&) noexcept = default;

    fs::path operator/(std::string_view other) const { return path / other; }

    operator const fs::path&() const { return path; }
    operator std::string() const { return path.string(); }

    MIOPEN_INTERNALS_EXPORT int
    Execute(std::string_view cmd, std::string_view args, bool allowChangingCwd = true) const;

    MIOPEN_INTERNALS_EXPORT ~TmpDir();
};

} // namespace miopen

#endif
