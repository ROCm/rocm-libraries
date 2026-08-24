// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "NativeAdapter.hpp"

#include <hipdnn_data_sdk/logging/Logger.hpp>

#include <cstddef>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hipdnn_backend::heuristics::uhd
{

std::unique_ptr<NativeAdapter> NativeAdapter::resolve(const std::string& symbolName,
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

NativeAdapter::NativeAdapter(UhdScoreFn scorer,
                             size_t numFeatures,
                             std::string featuresHash,
                             std::string symbolName)
    : _scorer(scorer)
    , _numFeatures(numFeatures)
    , _featuresHash(std::move(featuresHash))
    , _symbolName(std::move(symbolName))
{
}

double NativeAdapter::score(const std::vector<double>& features) const
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

} // namespace hipdnn_backend::heuristics::uhd
