// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include <cstddef>
#include <string>
#include <utility>

/// @file NativeScorerRegistry.hpp
/// @brief Symbol name to compiled scorer, for the UHD `native` adapter
/// (RFC 0019 §7.1). Lookup fails closed.
///
/// RFC 0019 §7.1 specifies that a native scorer is "named in the UHD by symbol,
/// resolved through the same symbol-registration mechanism the ingestor uses for
/// matchers and dispatch handlers". That mechanism is
/// `hipdnn_plugin_sdk::ingestor::NativeRegistry<T>`, and this is an
/// instantiation of it.
///
/// A separate instantiation rather than the ingestor's own `ScoreRegistry`,
/// because the two scorer contracts are not interchangeable: an ingestor scorer
/// receives the match context and featurizes itself, while a UHD scorer
/// receives the feature row the extractor already produced from
/// `features_signature`. RFC 0019 §7.1 permits both, so a UHD scorer needs its
/// own signature and therefore its own registry instance.
namespace hipdnn_plugin_sdk::ingestor::uhd
{

/// @brief Signature of a compiled UHD scorer.
///
/// Identical to the C ABI the `custom_library` adapter calls, so one scorer
/// implementation can be reached either way — registered in-process for
/// `native`, or exported from a `.so` for `custom_library` — without changing
/// its signature.
///
/// @param features   Feature row in `features_signature` order. May be null
///                   when the scorer featurizes from bindings itself
///                   (RFC 0019 §7.1 makes `features_signature` optional for
///                   this adapter).
/// @param numFeatures Number of entries in @p features.
/// @return The candidate's score, in the units the UHD's `score` metadata
///         declares. Implementations must be thread-safe.
using UhdScoreFn = double (*)(const double* features, size_t numFeatures);

/// @brief Process-wide registry of compiled UHD scorers, keyed by symbol name.
///
/// Inherits the ingestor registry's semantics: thread-safe, duplicate
/// registration throws, `resolve` throws on a missing symbol, `tryResolve`
/// returns a null function pointer. One instance per registered type per loaded
/// image, so the visibility settings the ingestor documents apply here too.
using NativeScorerRegistry = hipdnn_plugin_sdk::ingestor::NativeRegistry<UhdScoreFn>;

/// @brief RAII registration, so a test or an engine's init scope cannot leak a
/// symbol into an unrelated case.
///
/// The ingestor's `SymbolScope` is not reused: its `add()` overloads are
/// enumerated per ingestor function type and none accepts a `UhdScoreFn`.
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

} // namespace hipdnn_plugin_sdk::ingestor::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
