// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>

namespace TensileLite
{
    /**
     * Counter advanced whenever the library changes after loading, for example
     * when a kernel is registered at runtime.
     *
     * CachingLibrary records the value each memoized result was computed under
     * and treats a mismatch as a miss. Stale entries are overwritten rather than
     * erased, because CacheMap has no erase.
     */
    struct LibraryGeneration
    {
        std::atomic<uint64_t> value{0};
    };

    /**
     * The process-wide counter. A lazily loaded library is many
     * MasterSolutionLibrary objects, each shard with its own CachingLibrary, and
     * one change must invalidate all of them.
     */
    inline std::shared_ptr<LibraryGeneration> globalLibraryGeneration()
    {
        static std::shared_ptr<LibraryGeneration> instance = std::make_shared<LibraryGeneration>();
        return instance;
    }
} // namespace TensileLite
