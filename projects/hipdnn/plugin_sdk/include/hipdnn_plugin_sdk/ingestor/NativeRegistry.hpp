// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

/// @file NativeRegistry.hpp
/// @brief Symbol name to native function, the escape hatch descriptors resolve
/// through instead of carrying inline code. Lookup fails closed.
namespace hipdnn_plugin_sdk::ingestor
{

/// Implementations of these three must be thread-safe.
using GraphMatcherFn = bool (*)(const MatchContext&, BoundTokens& bound);
using KernelMatcherFn = bool (*)(const MatchContext&, const KernelDefinition&);
using ScoreFn = double (*)(const KernelDefinition&, const MatchContext&);

/// The provider's registry of native implementations, keyed by symbol name. One
/// instance per registered type per loaded image: requires `CXX_VISIBILITY_PRESET
/// hidden` and `--exclude-libs=ALL` so two loaded copies do not share a registry.
/// Thread-safe.
template <typename T>
class NativeRegistry
{
public:
    /// @throws std::runtime_error if @p symbol is already registered.
    static void registerSymbol(const std::string& symbol, T implementation)
    {
        auto& self = instance();
        const std::lock_guard<std::mutex> lock(self._mutex);

        auto [it, inserted] = self._symbols.emplace(symbol, implementation);
        if(!inserted)
        {
            throw std::runtime_error("duplicate native ingestor symbol registration: " + symbol);
        }
    }

    /// @throws std::runtime_error if no implementation is registered under @p symbol.
    static T resolve(const std::string& symbol, const std::string& describedBy = {})
    {
        auto& self = instance();
        const std::lock_guard<std::mutex> lock(self._mutex);

        auto it = self._symbols.find(symbol);
        if(it == self._symbols.end())
        {
            throw std::runtime_error("unresolved native ingestor symbol '" + symbol + "'"
                                     + (describedBy.empty() ? "" : ", named by " + describedBy)
                                     + "; the descriptor names a behaviour this build does "
                                       "not ship, which is usually a misspelled symbol");
        }
        return it->second;
    }

    static T tryResolve(const std::string& symbol)
    {
        auto& self = instance();
        const std::lock_guard<std::mutex> lock(self._mutex);

        auto it = self._symbols.find(symbol);
        return it == self._symbols.end() ? T{} : it->second;
    }

    /// Test-only: replaces and returns the previous entry.
    static T replaceSymbol(const std::string& symbol, T implementation)
    {
        auto& self = instance();
        const std::lock_guard<std::mutex> lock(self._mutex);

        auto it = self._symbols.find(symbol);
        if(it == self._symbols.end())
        {
            self._symbols.emplace(symbol, implementation);
            return T{};
        }

        const T previous = it->second;
        it->second = implementation;
        return previous;
    }

    static void unregisterSymbol(const std::string& symbol)
    {
        auto& self = instance();
        const std::lock_guard<std::mutex> lock(self._mutex);
        self._symbols.erase(symbol);
    }

private:
    static NativeRegistry& instance()
    {
        static NativeRegistry s_instance;
        return s_instance;
    }

    std::mutex _mutex;
    std::unordered_map<std::string, T> _symbols;
};

using GraphMatcherRegistry = NativeRegistry<GraphMatcherFn>;
using KernelMatcherRegistry = NativeRegistry<KernelMatcherFn>;
using ScoreRegistry = NativeRegistry<ScoreFn>;

/// Non-owning: the provider must keep each handler alive for as long as any plan
/// built from it can execute.
template <typename THandle>
using DispatchRegistry = NativeRegistry<const IKernelDispatchHandler<THandle>*>;

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
