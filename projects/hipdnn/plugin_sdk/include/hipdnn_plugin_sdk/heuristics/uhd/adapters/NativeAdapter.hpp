// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "IUhdAdapter.hpp"

#include <hipdnn_plugin_sdk/heuristics/uhd/NativeScorerRegistry.hpp>

#include <cstddef>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hipdnn_plugin_sdk::uhd
{

/// @brief Adapter for a scorer compiled into the engine and resolved by symbol (RFC 0019 §7.1).
///
/// The engine registers the scorer with NativeScorerRegistry at init; no artifact is loaded.
/// Construct with numFeatures 0 when the scorer featurizes from bindings; the feature-count
/// check is then skipped. As the RFC 0019 §9 baseline, use the feature-row mode so extraction
/// cost is not mixed into scoring cost.
class NativeAdapter : public IUhdAdapter
{
public:
    /// @brief Resolve a registered scorer by symbol.
    /// @param numFeatures Expected feature count, or 0 when the scorer featurizes itself.
    /// @param expectedFeaturesHash Empty when the UHD carries no `features_signature`.
    /// @return nullptr when the symbol is not registered, so selection degrades to
    ///         `static_order` (RFC 0019 §5).
    static std::unique_ptr<NativeAdapter> resolve(const std::string& symbolName,
                                                  size_t numFeatures,
                                                  const std::string& expectedFeaturesHash);

    double score(const std::vector<double>& features) const override;

    size_t expectedFeatureCount() const override
    {
        return _numFeatures;
    }

    const std::string& getFeaturesHash() const override
    {
        return _featuresHash;
    }

private:
    NativeAdapter(UhdScoreFn scorer,
                  size_t numFeatures,
                  std::string featuresHash,
                  std::string symbolName);

    UhdScoreFn _scorer;
    size_t _numFeatures;
    std::string _featuresHash;
    std::string _symbolName; // for error messages
};

inline std::unique_ptr<NativeAdapter> NativeAdapter::resolve(
    const std::string& symbolName, size_t numFeatures, const std::string& expectedFeaturesHash)
{
    if(symbolName.empty())
    {
        HIPDNN_SDK_LOG_ERROR("NativeAdapter: empty symbol name");
        return nullptr;
    }

    // tryResolve: an unregistered symbol must degrade to static_order, not throw.
    UhdScoreFn scorer = NativeScorerRegistry::tryResolve(symbolName);
    if(scorer == nullptr)
    {
        HIPDNN_SDK_LOG_ERROR("NativeAdapter: no scorer registered under symbol '"
                             << symbolName
                             << "'; the engine must call "
                                "NativeScorerRegistry::registerSymbol before the UHD is loaded");
        return nullptr;
    }

    // Private constructor; make_unique cannot be used.
    return std::unique_ptr<NativeAdapter>(
        new NativeAdapter(scorer, numFeatures, expectedFeaturesHash, symbolName));
}

inline NativeAdapter::NativeAdapter(UhdScoreFn scorer,
                                    size_t numFeatures,
                                    std::string featuresHash,
                                    std::string symbolName)
    : _scorer(scorer)
    , _numFeatures(numFeatures)
    , _featuresHash(std::move(featuresHash))
    , _symbolName(std::move(symbolName))
{
}

inline double NativeAdapter::score(const std::vector<double>& features) const
{
    // numFeatures == 0: the scorer featurizes from bindings; no row to validate.
    if(_numFeatures != 0 && features.size() != _numFeatures)
    {
        std::ostringstream oss;
        oss << "NativeAdapter: feature count mismatch for symbol '" << _symbolName << "'. Expected "
            << _numFeatures << ", got " << features.size();
        throw std::invalid_argument(oss.str());
    }

    return _scorer(features.data(), features.size());
}

} // namespace hipdnn_plugin_sdk::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
