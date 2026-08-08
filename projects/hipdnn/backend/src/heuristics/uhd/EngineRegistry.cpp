// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "EngineRegistry.hpp"

#include "FeatureExtractor.hpp"
#include "adapters/StaticOrderAdapter.hpp"
#include "adapters/TreeDataAdapter.hpp"

#include <hipdnn_data_sdk/logging/Logger.hpp>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace hipdnn_backend::heuristics::uhd
{

EngineRegistry& EngineRegistry::instance()
{
    static EngineRegistry s_instance;
    return s_instance;
}

void EngineRegistry::registerEngine(EngineEntry entry)
{
    // Validate KMD field coverage: every $kernel.* in features_signature
    // must exist in at least one candidate's metadata (RFC 0019 §7.3).
    if(!entry.uhdConfig.featuresSignature.empty() && !entry.candidates.empty())
    {
        // Collect all metadata keys across candidates (the KMD field space)
        std::unordered_set<std::string> kmdFields;
        for(const auto& candidate : entry.candidates)
        {
            for(const auto& [key, _] : candidate.metadata)
            {
                kmdFields.insert(key);
            }
        }
        // Also include implicit kernel fields that SelectionEngine adds
        kmdFields.insert("priority");
        kmdFields.insert("id");

        // Build extractor to parse signature and extract $kernel.* refs.
        // A malformed entry throws JsonLogicError; re-raise as invalid_argument so the
        // exception type matches what registerEngine documents.
        std::vector<std::string> missingFields;
        try
        {
            const FeatureExtractor extractor(entry.uhdConfig.featuresSignature);
            missingFields = extractor.getMissingKmdFields(kmdFields);
        }
        catch(const JsonLogicError& e)
        {
            std::ostringstream oss;
            oss << "UHD features_signature contains an entry that is neither a bare $ref "
                << "nor valid JsonLogic. Engine ID: " << entry.engineId << ", uhd='"
                << entry.uhdConfig.uhdId << "': " << e.what();
            throw std::invalid_argument(oss.str());
        }

        if(!missingFields.empty())
        {
            std::ostringstream oss;
            oss << "UHD features_signature references $kernel.* fields not present in "
                << "candidate metadata: ";
            for(size_t i = 0; i < missingFields.size(); ++i)
            {
                if(i > 0)
                {
                    oss << ", ";
                }
                oss << missingFields[i];
            }
            oss << ". Engine ID: " << entry.engineId
                << ". Ensure all $kernel.* fields in features_signature are present in "
                   "candidate metadata.";
            throw std::invalid_argument(oss.str());
        }
    }

    validateFeaturesHash(entry);

    const std::lock_guard<std::mutex> lock(_mutex);
    _engines[entry.engineId] = std::move(entry);
}

void EngineRegistry::validateFeaturesHash(const EngineEntry& entry)
{
    const UhdConfig& cfg = entry.uhdConfig;

    // A declared hash must describe the signature it ships with (RFC 0019 §7.3).
    // Without this the only hash check is model-vs-config, so both could agree while
    // neither matches the signature actually being evaluated — which defeats the
    // point of fingerprinting the feature contract.
    //
    // Checked even for an empty signature: a hash over no features is
    // self-inconsistent, and skipping it there would let that through.
    if(!cfg.featuresHash.empty())
    {
        std::string actual;
        try
        {
            actual = FeatureExtractor::computeHash(cfg.featuresSignature);
        }
        catch(const JsonLogicError& e)
        {
            std::ostringstream oss;
            oss << "UHD features_signature contains an entry that is neither a bare $ref "
                << "nor valid JsonLogic. Engine ID: " << entry.engineId << ", uhd='"
                << cfg.uhdId << "': " << e.what();
            throw std::invalid_argument(oss.str());
        }

        if(actual != cfg.featuresHash)
        {
            std::ostringstream oss;
            oss << "UHD features_hash does not describe its own features_signature. "
                << "Engine ID: " << entry.engineId << ", uhd='" << cfg.uhdId
                << "', declared=" << cfg.featuresHash << ", computed=" << actual
                << ". The signature and hash must be emitted together — regenerate the "
                   "descriptor rather than editing either by hand.";
            throw std::invalid_argument(oss.str());
        }
        return;
    }

    if(cfg.featuresSignature.empty())
    {
        return;
    }

    // Feature-bearing adapters are required to carry a hash (RFC 0019 §7.3). Absent
    // one there is nothing to check the model artifact against.
    if(cfg.adapterType == "tree_data" || cfg.adapterType == "onnx" || cfg.adapterType == "table")
    {
        HIPDNN_SDK_LOG_WARN("UHD: engine "
                            << entry.engineId << " uhd='" << cfg.uhdId << "' adapter '"
                            << cfg.adapterType
                            << "' declares a features_signature but no features_hash; the "
                               "feature contract with the model artifact cannot be enforced");
    }
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

std::shared_ptr<FeatureExtractor> EngineRegistry::getOrCreateExtractor(int64_t engineId) const
{
    const std::lock_guard<std::mutex> lock(_mutex);
    const auto it = _engines.find(engineId);
    if(it == _engines.end())
    {
        return nullptr;
    }

    auto& entry = it->second;

    // Return cached extractor if available
    if(entry.cachedExtractor != nullptr)
    {
        return entry.cachedExtractor;
    }

    // Create extractor from features signature if non-empty
    if(!entry.uhdConfig.featuresSignature.empty())
    {
        entry.cachedExtractor = std::make_shared<FeatureExtractor>(entry.uhdConfig.featuresSignature);
    }

    return entry.cachedExtractor;
}

} // namespace hipdnn_backend::heuristics::uhd
