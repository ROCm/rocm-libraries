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

/**
 * @file NativeRegistry.hpp
 * @brief Symbol name to native function, the escape hatch descriptors resolve through.
 *
 * A descriptor names a custom predicate or plan by symbol rather than carrying inline
 * code (RFC 0017); the provider resolves it here. Lookup fails closed.
 */
namespace hipdnn_plugin_sdk::ingestor
{

/// Graph-scoped applicability check: reads only graph/device facts, so it runs once per
/// (graph, device); failure prunes every kernel in the pack. On success, writes resolved
/// values into @p bound (RFC 0017 §8.5). Must be thread-safe.
using GraphMatcherFn = bool (*)(const MatchContext&, BoundTokens& bound);

/// Kernel-scoped applicability check: also reads the candidate's metadata, so it runs
/// once per surviving kernel and disqualifies that kernel alone. Must be thread-safe.
using KernelMatcherFn = bool (*)(const MatchContext&, const KernelDefinition&);

/// Scores one kernel for one problem. Higher is better. Never handed the catalog (see
/// IKernelHeuristic). Must be thread-safe.
using ScoreFn = double (*)(const KernelDefinition&, const MatchContext&);

/**
 * @brief The provider's registry of native implementations, keyed by symbol name.
 *
 * One instance per registered type per loaded image: requires `CXX_VISIBILITY_PRESET
 * hidden` and `--exclude-libs=ALL` (`src/CMakeLists.txt`) so two loaded copies do not
 * share a registry.
 *
 * Thread-safe: registration and lookup are mutex-guarded.
 *
 * @tparam T The registered callable or interface pointer type.
 */
template <typename T>
class NativeRegistry
{
public:
    /**
     * @brief Registers @p implementation under @p symbol.
     * @throws std::runtime_error if @p symbol is already registered (which wins would
     *         depend on static-init order).
     */
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

    /**
     * @brief Resolves @p symbol.
     * @throws std::runtime_error if no implementation is registered under it.
     */
    static T resolve(const std::string& symbol)
    {
        auto& self = instance();
        const std::lock_guard<std::mutex> lock(self._mutex);

        auto it = self._symbols.find(symbol);
        if(it == self._symbols.end())
        {
            throw std::runtime_error("unresolved native ingestor symbol: " + symbol);
        }
        return it->second;
    }

    /// @brief Removes @p symbol if present. Used by tests to swap in a temporary
    /// implementation and restore the original; production code never unregisters.
    static void unregisterSymbol(const std::string& symbol)
    {
        auto& self = instance();
        const std::lock_guard<std::mutex> lock(self._mutex);
        self._symbols.erase(symbol);
    }

private:
    static NativeRegistry& instance()
    {
        // Function-local static: avoids depending on unspecified namespace-scope init order.
        static NativeRegistry s_instance;
        return s_instance;
    }

    std::mutex _mutex;
    std::unordered_map<std::string, T> _symbols;
};

using GraphMatcherRegistry = NativeRegistry<GraphMatcherFn>;
using KernelMatcherRegistry = NativeRegistry<KernelMatcherFn>;
using ScoreRegistry = NativeRegistry<ScoreFn>;

/// Non-owning: the provider owns each handler and must keep it alive for as long as
/// any plan built from it can execute.
template <typename THandle>
using DispatchRegistry = NativeRegistry<const IKernelDispatchHandler<THandle>*>;

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
