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
 * code (RFC 0017); the provider resolves it here. Lookup fails closed: a symbol the
 * provider does not ship throws rather than being silently skipped.
 */
namespace hipdnn_plugin_sdk::ingestor
{

/// A graph-scoped applicability check: reads only graph and device facts, so it runs
/// once per (graph, device) and its failure prunes every kernel in the pack.
///
/// On success, writes what it resolved into @p bound (RFC 0017 §8.5), which the catalog
/// keeps and dispatch reads back instead of re-deriving it. A matcher that binds
/// nothing leaves @p bound alone. Kernel-scoped matchers never bind: their result is
/// per-kernel, with nowhere in the catalog to live.
///
/// Must be thread-safe: the registry may call it concurrently, same contract as
/// `IKernelDispatchHandler::launch`.
using GraphMatcherFn = bool (*)(const MatchContext&, BoundTokens& bound);

/// A kernel-scoped applicability check: also reads the candidate's metadata, so it runs
/// once per surviving kernel and disqualifies that kernel alone. Must be thread-safe.
using KernelMatcherFn = bool (*)(const MatchContext&, const KernelDefinition&);

/// Scores one kernel for one problem. Higher is better.
///
/// Deliberately never handed the catalog: see IKernelHeuristic for why that constraint
/// is what keeps a knob-filtered query consistent with the default it reported. Must be
/// thread-safe.
using ScoreFn = double (*)(const KernelDefinition&, const MatchContext&);

/**
 * @brief The provider's registry of native implementations, keyed by symbol name.
 *
 * One instance per registered type per loaded image, not per process: isolation
 * between two loaded copies depends on the provider building with
 * `CXX_VISIBILITY_PRESET hidden` and `--exclude-libs=ALL` (`src/CMakeLists.txt`) — an
 * exported symbol would let two copies share one registry and the second
 * `registerSymbol()` throw.
 *
 * Guarded by a mutex: registration and lookup can race across threads.
 *
 * @tparam T The registered callable or interface pointer type.
 */
template <typename T>
class NativeRegistry
{
public:
    /**
 * @brief Registers @p implementation under @p symbol.
 * @throws std::runtime_error if @p symbol is already registered — always an author
 *         bug, since which implementation would silently win depends on static-init
 *         order.
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

    /// @brief Removes @p symbol if present. Exists so a test can install a counting or
    /// failing implementation and restore the original afterwards; production code
    /// registers once at static-init time and never unregisters.
    static void unregisterSymbol(const std::string& symbol)
    {
        auto& self = instance();
        const std::lock_guard<std::mutex> lock(self._mutex);
        self._symbols.erase(symbol);
    }

private:
    static NativeRegistry& instance()
    {
        // Function-local static: initialized on first use, immune to the unspecified
        // init order of the namespace-scope registrations that populate it.
        static NativeRegistry s_instance;
        return s_instance;
    }

    std::mutex _mutex;
    std::unordered_map<std::string, T> _symbols;
};

using GraphMatcherRegistry = NativeRegistry<GraphMatcherFn>;
using KernelMatcherRegistry = NativeRegistry<KernelMatcherFn>;
using ScoreRegistry = NativeRegistry<ScoreFn>;

/// Registered as non-owning pointers: a handler holds provider state (a compiler,
/// a module cache) whose lifetime is the provider's. The provider must keep the
/// handler alive for as long as any plan built from it can execute.
template <typename THandle>
using DispatchRegistry = NativeRegistry<const IKernelDispatchHandler<THandle>*>;

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
