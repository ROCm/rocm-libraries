// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

/// @file NativeScorerRegistry.hpp
/// @brief Symbol name to compiled scorer, for the UHD `native` adapter
/// (RFC 0019 §7.1). Lookup fails closed.
///
/// RFC 0019 §7.1 specifies that a native scorer is "named in the UHD by symbol,
/// resolved through the same symbol-registration mechanism the ingestor uses for
/// matchers and dispatch handlers". This registry is that mechanism applied at
/// the UHD layer.
///
/// It deliberately does not reuse `hipdnn_plugin_sdk::ingestor::NativeRegistry`
/// directly: that header is compiled only under `HIPDNN_ENABLE_KERNEL_INGESTOR`
/// and its `ScoreFn` is expressed over ingestor types
/// (`KernelDefinition`, `MatchContext`). Binding UHD scoring to an unrelated
/// feature flag, and pulling ingestor types into the backend selection path,
/// would both be worse than duplicating three short methods. Consolidating the
/// two behind one shared template is worthwhile once the ingestor flag is no
/// longer load-bearing.
namespace hipdnn_backend::heuristics::uhd
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

/// @brief Process-wide registry of compiled scorers, keyed by symbol name.
///
/// One instance per loaded image: requires `CXX_VISIBILITY_PRESET hidden` and
/// `--exclude-libs=ALL` so two loaded copies do not share a registry. Thread-safe.
class NativeScorerRegistry
{
public:
    /// @brief Register @p scorer under @p symbol.
    /// @throws std::runtime_error if @p symbol is already registered.
    static void registerSymbol(const std::string& symbol, UhdScoreFn scorer)
    {
        if(scorer == nullptr)
        {
            throw std::runtime_error("null native UHD scorer registered under symbol: " + symbol);
        }

        auto& self = instance();
        const std::lock_guard<std::mutex> lock(self._mutex);

        auto [it, inserted] = self._symbols.emplace(symbol, scorer);
        if(!inserted)
        {
            throw std::runtime_error("duplicate native UHD scorer registration: " + symbol);
        }
    }

    /// @brief Look up @p symbol, throwing when absent.
    /// @param describedBy Optional context for the error message, typically the
    ///        UHD id that named the symbol.
    /// @throws std::runtime_error if nothing is registered under @p symbol.
    static UhdScoreFn resolve(const std::string& symbol, const std::string& describedBy = {})
    {
        auto& self = instance();
        const std::lock_guard<std::mutex> lock(self._mutex);

        auto it = self._symbols.find(symbol);
        if(it == self._symbols.end())
        {
            throw std::runtime_error("unresolved native UHD scorer symbol '" + symbol + "'"
                                     + (describedBy.empty() ? "" : ", named by " + describedBy)
                                     + "; the descriptor names a scorer this build does not "
                                       "ship, which is usually a misspelled symbol");
        }
        return it->second;
    }

    /// @brief Look up @p symbol, returning nullptr when absent.
    static UhdScoreFn tryResolve(const std::string& symbol)
    {
        auto& self = instance();
        const std::lock_guard<std::mutex> lock(self._mutex);

        auto it = self._symbols.find(symbol);
        return it == self._symbols.end() ? nullptr : it->second;
    }

    /// @brief Remove @p symbol if present.
    static void unregisterSymbol(const std::string& symbol)
    {
        auto& self = instance();
        const std::lock_guard<std::mutex> lock(self._mutex);
        self._symbols.erase(symbol);
    }

private:
    static NativeScorerRegistry& instance()
    {
        static NativeScorerRegistry s_instance;
        return s_instance;
    }

    std::mutex _mutex;
    std::unordered_map<std::string, UhdScoreFn> _symbols;
};

/// @brief RAII registration, so a test or an engine's init scope cannot leak a
/// symbol into an unrelated case.
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

} // namespace hipdnn_backend::heuristics::uhd
