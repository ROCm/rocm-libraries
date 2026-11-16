// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

/// The on-disk cache of HIPRTC code objects (`<cacheRoot>/hip-kernel-provider/hiprtc/<key>.co`).
///
/// A code object is stored under a key that digests everything it was built from: the
/// compiler, the source, every embedded include and every option. Files are written to a
/// temporary name and renamed into place, so a reader in another process never sees a partial
/// file.
namespace hip_kernel_provider::compilation::hiprtc_cache
{

/// The compiler HIPRTC runs in this process, as a string the cache key digests: the HIPRTC
/// version, the HIP runtime version, the comgr version, and a build identity of the HIPRTC
/// and comgr libraries (their ELF build id where the library has one, otherwise the versioned
/// file name and its size). Computed once per process. Independent of the install path, so a
/// toolchain build copied elsewhere keeps its keys.
const std::string& compilerIdentity();

/// The cache key of a code object: a SHA-256 over the cache format, @p compilerIdentity, the
/// program name and source, every embedded include (name and text) and every option.
/// @p includeTexts and @p includeNames are parallel.
std::string codeObjectKey(std::string_view compilerIdentity,
                          std::string_view programName,
                          std::string_view source,
                          const std::vector<std::string_view>& includeTexts,
                          const std::vector<const char*>& includeNames,
                          const std::vector<std::string>& options);

/// Where the code object for @p key lives, or an empty path when the on-disk cache is off
/// (HIPDNN_DISABLE_CACHE) or its root cannot be made.
std::filesystem::path codeObjectPath(const std::string& key);

/// The code object at @p path, or an empty vector when there is none or it cannot be read.
std::vector<char> readCodeObject(const std::filesystem::path& path);

/// Writes @p codeObject to @p path through a temporary file renamed into place. Best effort: a
/// failure is logged at INFO and dropped, and the next process compiles again.
/// @return Whether the entry was placed.
bool writeCodeObject(const std::filesystem::path& path, const std::vector<char>& codeObject);

} // namespace hip_kernel_provider::compilation::hiprtc_cache
