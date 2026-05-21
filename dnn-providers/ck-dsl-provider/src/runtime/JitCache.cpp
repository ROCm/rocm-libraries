// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "JitCache.hpp"

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <utility>

namespace ck_dsl_provider {

std::shared_ptr<HipModule> JitCache::getOrLoad(SignatureHash key, const Loader& loader) {
    if (!loader) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "JitCache::getOrLoad: loader callable is empty");
    }

    std::lock_guard<std::mutex> lock(_mutex);

    auto it = _modules.find(key);
    if (it != _modules.end()) {
        return it->second;
    }

    // Cache miss: run the loader under the lock so a second thread
    // racing the same key waits for this compile rather than duplicating
    // it. The loader's typical cost (~150 ms for the bake-off conv per
    // PREP_FINDINGS P-5) is well below the lock-contention threshold
    // that would justify a per-key shared_future scheme.
    KernelArtifact artifact = loader();
    auto module = std::make_shared<HipModule>(artifact);

    HIPDNN_PLUGIN_LOG_INFO("JitCache: loaded kernel '" << module->kernelName() << "' (kind='"
                                                       << artifact.kind << "') for key=0x"
                                                       << std::hex << key << std::dec);

    auto [inserted, _] = _modules.emplace(key, std::move(module));
    return inserted->second;
}

bool JitCache::contains(SignatureHash key) const {
    std::lock_guard<std::mutex> lock(_mutex);
    return _modules.find(key) != _modules.end();
}

std::size_t JitCache::size() const {
    std::lock_guard<std::mutex> lock(_mutex);
    return _modules.size();
}

}  // namespace ck_dsl_provider
