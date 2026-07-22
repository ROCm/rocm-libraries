// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "SdpaKernelUtils.hpp"
#include "compilation/ModuleCache.hpp"

#include <memory>
#include <string>

namespace asm_sdpa_engine
{

// =============================================================================
// Cached kernel module loader
// =============================================================================
//
// Process-level cache for loaded kernel modules.  hipModuleLoad() is expensive
// (~97% of SDPA execution time in profiling), but the set of distinct .co files
// is small (bounded by CSV config count).  Each SdpaModuleCache instance maps
// (coPath, funcName) pairs to shared_ptr<HipModuleGuard>: on the first call
// the module is loaded and cached; subsequent calls return the cached
// shared_ptr.  Modules are never unloaded until the cache is destroyed.

using CachedModule = std::shared_ptr<HipModuleGuard>;

class SdpaModuleCache
    : public hip_kernel_provider::compilation::
          ModuleCache<SdpaModuleCache, CachedModule, const std::string&, const char*>
{
public:
    static std::string makeKey(const std::string& coPath, const char* funcName)
    {
        return coPath + "::" + funcName;
    }

    static CachedModule load(const std::string& coPath, const char* funcName)
    {
        auto loaded = loadKernelModule(coPath, funcName);
        if(!loaded)
        {
            return nullptr;
        }
        return std::make_shared<HipModuleGuard>(std::move(*loaded));
    }
};

} // namespace asm_sdpa_engine
