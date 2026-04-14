// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace rocRoller
{
    class CacheOnlyPolicy // Use cache to look up order. Caller should ensure the cache is valid.
    {
    };
    class UpdateCachePolicy // < Use cache to look up order. Cache will be re-built if invalid.
    {
    };
    class UseCacheIfAvailablePolicy // < Look up in cache if available, otherwise, use traversal.
    {
    };
    class IgnoreCachePolicy // < Use traversal.
    {
    };

    inline constexpr CacheOnlyPolicy           CacheOnly;
    inline constexpr UpdateCachePolicy         UpdateCache;
    inline constexpr UseCacheIfAvailablePolicy UseCacheIfAvailable;
    inline constexpr IgnoreCachePolicy         IgnoreCache;
}
