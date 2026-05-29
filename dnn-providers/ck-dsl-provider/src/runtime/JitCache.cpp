// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "JitCache.hpp"

#include <exception>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <memory>
#include <utility>

namespace ck_dsl_provider {

std::shared_ptr<HipModule> JitCache::getOrLoad(SignatureHash key, const Loader& loader) {
    if (!loader) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "JitCache::getOrLoad: loader callable is empty");
    }

    // Take the map mutex only long enough to look up or install a
    // future. The loader runs OUTSIDE the mutex so a long compile on
    // one key does not block lookups (hits or misses) on other keys.
    SharedFuture future;
    std::shared_ptr<std::promise<SharedModule>> promise;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _entries.find(key);
        if (it != _entries.end()) {
            future = it->second;
        } else {
            promise = std::make_shared<std::promise<SharedModule>>();
            future = promise->get_future().share();
            _entries.emplace(key, future);
        }
    }

    if (promise) {
        // We installed the entry, so we are responsible for fulfilling
        // its promise. Run the loader outside the cache mutex; concurrent
        // misses on different keys can compile in parallel.
        try {
            KernelArtifact artifact = loader();
            auto module = std::make_shared<HipModule>(artifact);
            HIPDNN_PLUGIN_LOG_INFO("JitCache: loaded kernel '"
                                   << module->kernelName() << "' (kind='" << artifact.kind
                                   << "') for key=0x" << std::hex << key << std::dec);
            promise->set_value(std::move(module));
        } catch (...) {
            // Surface the failure to every waiter, then evict the entry
            // so a subsequent getOrLoad retries rather than caching the
            // exception forever.
            promise->set_exception(std::current_exception());
            {
                std::lock_guard<std::mutex> lock(_mutex);
                _entries.erase(key);
            }
            throw;
        }
    }

    // For the loader thread this returns immediately (promise was just
    // fulfilled); for any racing waiter this blocks until the loader
    // finishes, then either returns the module or re-throws the
    // loader's exception.
    return future.get();
}

bool JitCache::contains(SignatureHash key) const {
    std::lock_guard<std::mutex> lock(_mutex);
    return _entries.find(key) != _entries.end();
}

std::size_t JitCache::size() const {
    std::lock_guard<std::mutex> lock(_mutex);
    return _entries.size();
}

}  // namespace ck_dsl_provider
