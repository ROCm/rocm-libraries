// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <hipdnn_plugin_sdk/NativeRegistry.hpp>

#include <cstddef>
#include <string>
#include <utility>

/// @file NativeScorerRegistry.hpp
/// @brief Symbol name to compiled scorer, for the UHD `native` adapter (RFC 0019 §7.1).
///
/// Separate from the ingestor's `ScoreRegistry` because a UHD scorer takes the extracted
/// feature row, not the match context.
namespace hipdnn_plugin_sdk::uhd
{

/// @brief Signature of a compiled UHD scorer; the same C ABI `custom_library` calls.
///
/// @param features Row in `features_signature` order; may be null when the signature is
///        empty and the scorer featurizes itself.
/// @return Raw score, with `score.transform` still applied. Must be thread-safe.
using UhdScoreFn = double (*)(const double* features, size_t numFeatures);

/// @brief Process-wide registry of compiled UHD scorers; see NativeRegistry.
using NativeScorerRegistry = hipdnn_plugin_sdk::NativeRegistry<UhdScoreFn>;

/// @brief RAII registration, so a scope cannot leak a symbol into unrelated code.
class ScopedNativeScorer
{
public:
    ScopedNativeScorer(std::string symbol, UhdScoreFn scorer)
        : _symbol(std::move(symbol))
    {
        NativeScorerRegistry::registerSymbol(_symbol, scorer);
    }

    ~ScopedNativeScorer()
    {
        NativeScorerRegistry::unregisterSymbol(_symbol);
    }

    ScopedNativeScorer(const ScopedNativeScorer&) = delete;
    ScopedNativeScorer& operator=(const ScopedNativeScorer&) = delete;
    ScopedNativeScorer(ScopedNativeScorer&&) = delete;
    ScopedNativeScorer& operator=(ScopedNativeScorer&&) = delete;

private:
    std::string _symbol;
};

} // namespace hipdnn_plugin_sdk::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
