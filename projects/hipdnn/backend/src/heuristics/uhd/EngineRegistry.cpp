// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "EngineRegistry.hpp"

#include "FeatureExtractor.hpp"
#include "adapters/StaticOrderAdapter.hpp"
#include "adapters/TreeDataAdapter.hpp"

#include <algorithm>

namespace hipdnn_backend::heuristics::uhd
{

EngineRegistry& EngineRegistry::instance()
{
    static EngineRegistry s_instance;
    return s_instance;
}

void EngineRegistry::registerEngine(EngineEntry entry)
{
    const std::lock_guard<std::mutex> lock(_mutex);
    _engines[entry.engineId] = std::move(entry);
}

std::optional<std::reference_wrapper<const EngineEntry>>
    EngineRegistry::getEngine(int64_t engineId) const
{
    const std::lock_guard<std::mutex> lock(_mutex);
    const auto it = _engines.find(engineId);
    if(it == _engines.end())
    {
        return std::nullopt;
    }
    return std::cref(it->second);
}

std::shared_ptr<IUhdAdapter> EngineRegistry::getOrCreateAdapter(int64_t engineId) const
{
    const std::lock_guard<std::mutex> lock(_mutex);
    const auto it = _engines.find(engineId);
    if(it == _engines.end())
    {
        return nullptr;
    }

    auto& entry = it->second;

    // Return cached adapter if available
    if(entry.cachedAdapter != nullptr)
    {
        return entry.cachedAdapter;
    }

    // Create adapter based on type
    const auto& cfg = entry.uhdConfig;

    if(cfg.adapterType == "static_order")
    {
        // StaticOrderAdapter needs the features signature to map field names to indices
        auto adapter = StaticOrderAdapter::create(cfg.staticOrderFields, cfg.featuresSignature);
        entry.cachedAdapter = std::move(adapter);
    }
    else if(cfg.adapterType == "tree_data")
    {
        // TreeDataAdapter loads from model file
        if(!cfg.modelArtifactPath.empty())
        {
            auto adapter = TreeDataAdapter::load(cfg.modelArtifactPath, cfg.featuresHash);
            entry.cachedAdapter = std::move(adapter);
        }
    }
    // TODO: Add table, onnx, custom_library adapters when implemented

    return entry.cachedAdapter;
}

bool EngineRegistry::hasEngine(int64_t engineId) const
{
    const std::lock_guard<std::mutex> lock(_mutex);
    return _engines.find(engineId) != _engines.end();
}

std::vector<int64_t> EngineRegistry::getAllEngineIds() const
{
    const std::lock_guard<std::mutex> lock(_mutex);
    std::vector<int64_t> ids;
    ids.reserve(_engines.size());
    for(const auto& [id, _] : _engines)
    {
        ids.push_back(id);
    }
    return ids;
}

void EngineRegistry::clear()
{
    const std::lock_guard<std::mutex> lock(_mutex);
    _engines.clear();
}

} // namespace hipdnn_backend::heuristics::uhd
