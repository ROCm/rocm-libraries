// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// CacheRoot resolves the single root directory hipDNN's on-disk caches (winner cache,
// autotune cache, and future consumers) share, one subdirectory per consumer beneath it.

#include <filesystem>

namespace hipdnn_data_sdk::utilities
{

/// Resolves and ensures the existence of hipDNN's shared on-disk cache root directory.
///
/// Resolution order:
///  1. `HIPDNN_CACHE_DIR`, if set to a non-empty value (via getEnv()).
///  2. Otherwise, a CMake-baked, per-platform default (`~/.cache/hipdnn/` on Linux,
///     `%USERPROFILE%\.hipdnn\cache\` on Windows).
///
/// Whichever value is chosen is then passed through expandUser() so a leading `~` or
/// `%USERPROFILE%` resolves to the current user's home directory at run time -- this
/// lets a compile-time-baked default still follow `$HOME` across machines that share one
/// build. The resulting directory is created if it does not already exist.
///
/// This function never throws. If the resolved path cannot be created or is otherwise
/// unusable (e.g. it exists as a file, or a permissions/filesystem error prevents
/// directory creation), it degrades to returning an empty/invalid std::filesystem::path
/// rather than throwing or crashing -- callers must treat that as "no on-disk cache is
/// available right now" and fall back to their in-memory-only behavior.
///
/// @return The cache root directory, guaranteed to exist, on success; an empty/invalid
///     path on any resolution or filesystem failure.
inline std::filesystem::path cacheRoot();
// TODO(Stream A): implement in Phase 2

} // namespace hipdnn_data_sdk::utilities
