// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "IUhdAdapter.hpp"

#include <hipdnn_plugin_sdk/ingestor/uhd/NativeScorerRegistry.hpp>

#include <memory>
#include <string>
#include <vector>
#include <cstddef>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace hipdnn_plugin_sdk::ingestor::uhd
{

/// @brief Native adapter for scorers compiled into the engine (RFC 0019 §7.1).
///
/// The NATIVE adapter names a scorer by symbol; the engine registers the
/// implementation with NativeScorerRegistry at init and this adapter resolves
/// it. Nothing is loaded from disk and there is no model artifact — the UHD
/// carries a symbol name, and the behaviour is ordinary C++ compiled into the
/// engine.
///
/// Two roles, per RFC 0019 §7.1:
///
/// - **Escape hatch** for heuristics no model expresses.
/// - **Performance baseline.** RFC 0019 §9 defines the `tree_data` overhead
///   target as within 2× of this adapter, so `native` is what that ratio is
///   measured against.
///
/// It is deliberately *not* a drop-in: changing the heuristic means recompiling
/// the engine, so it does not serve the data-driven, independently-shippable
/// goal that `tree_data` exists for.
///
/// `features_signature` is optional here (RFC 0019 §7.1). A scorer may consume
/// the standard feature row, holding it to the same contract as a model, or
/// featurize from the bindings directly. Construct with @p numFeatures of 0 to
/// select the second mode; the feature-count check is then skipped.
///
/// For the baseline role specifically, prefer the feature-row mode: if `native`
/// featurizes from bindings while `tree_data` goes through the generic
/// extractor, a comparison between them conflates extraction cost with scoring
/// cost, which RFC 0019 §9.4 requires be wall-clocked separately.
class NativeAdapter : public IUhdAdapter
{
public:
    /// @brief Resolve a registered scorer by symbol.
    ///
    /// @param symbolName Symbol the engine registered with NativeScorerRegistry.
    /// @param numFeatures Expected feature count, or 0 when the scorer
    ///        featurizes from bindings itself.
    /// @param expectedFeaturesHash SHA-256 of the feature signature; empty when
    ///        the UHD carries no `features_signature`.
    /// @return Adapter on success, nullptr when the symbol is not registered.
    ///
    /// Returning nullptr rather than throwing matches the other adapters and
    /// lets selection degrade to `static_order` (RFC 0019 §5), which is the
    /// required behaviour for any UHD that cannot be brought up.
    static std::unique_ptr<NativeAdapter> resolve(const std::string& symbolName,
                                                  size_t numFeatures,
                                                  const std::string& expectedFeaturesHash);

    double score(const std::vector<double>& features) const override;

    UhdAdapterType type() const override
    {
        return UhdAdapterType::NATIVE;
    }

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


inline std::unique_ptr<NativeAdapter> NativeAdapter::resolve(const std::string& symbolName,
                                                      size_t numFeatures,
                                                      const std::string& expectedFeaturesHash)
{
    if(symbolName.empty())
    {
        HIPDNN_SDK_LOG_ERROR("NativeAdapter: empty symbol name");
        return nullptr;
    }

    // tryResolve rather than resolve: an unregistered symbol degrades this UHD
    // to static_order (RFC 0019 §5) instead of propagating an exception through
    // plan build.
    UhdScoreFn scorer = NativeScorerRegistry::tryResolve(symbolName);
    if(scorer == nullptr)
    {
        HIPDNN_SDK_LOG_ERROR("NativeAdapter: no scorer registered under symbol '"
                             << symbolName
                             << "'; the engine must call "
                                "NativeScorerRegistry::registerSymbol before the UHD is loaded");
        return nullptr;
    }

    // Private constructor, so make_unique is unavailable.
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
    // numFeatures == 0 means the scorer featurizes from bindings itself, so
    // there is no row to validate (RFC 0019 §7.1).
    if(_numFeatures != 0 && features.size() != _numFeatures)
    {
        std::ostringstream oss;
        oss << "NativeAdapter: feature count mismatch for symbol '" << _symbolName << "'. Expected "
            << _numFeatures << ", got " << features.size();
        throw std::invalid_argument(oss.str());
    }

    return _scorer(features.data(), features.size());
}

} // namespace hipdnn_plugin_sdk::ingestor::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
