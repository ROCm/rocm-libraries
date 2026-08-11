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
 * RFC 0017 gives every declarative step a named escape hatch for the case its built-ins
 * cannot express: a criterion may invoke a custom predicate, and a UDD may name a custom
 * plan. In both cases the descriptor carries only a *symbol name*, never inline code,
 * and the provider resolves it from a registry that is part of its published contract.
 *
 * The first implementation uses that escape hatch throughout: every UMD, UHD, and
 * UDD names a symbol here rather than carrying an expression or a model. Replacing a
 * native symbol with real declarative data is a change to one descriptor field, not to
 * the ingestor — which is the property this arrangement exists to demonstrate.
 *
 * Lookup fails closed. A descriptor naming a symbol the provider does not ship throws
 * rather than being silently skipped, so a missing registration surfaces as a clear
 * error at the point of use instead of as an engine that mysteriously matches nothing.
 */
namespace hipdnn_plugin_sdk::ingestor
{

/// A graph-scoped applicability check: reads only graph and device facts, so it runs
/// once per (graph, device) and its failure prunes every kernel in the pack.
///
/// On success it also writes what it resolved about the graph into @p bound, which the
/// catalog keeps and dispatch reads back. RFC 0017 §8.5 gives matching this double duty
/// deliberately: the matcher already walked the graph to decide the kernel applies, so
/// the launch reuses those values rather than re-deriving them. A matcher that binds
/// nothing simply leaves @p bound alone.
///
/// Only graph-scoped matchers bind. Kernel-scoped ones run once per candidate, so what
/// they resolved would be per-kernel state with nowhere in the catalog to live; the
/// tokens a launch needs describe the problem, not the kernel chosen for it.
using GraphMatcherFn = bool (*)(const MatchContext&, BoundTokens& bound);

/// A kernel-scoped applicability check: also reads the candidate's metadata, so it runs
/// once per surviving kernel and disqualifies that kernel alone.
using KernelMatcherFn = bool (*)(const MatchContext&, const KernelDefinition&);

/// Scores one kernel for one problem. Higher is better.
///
/// Deliberately never handed the catalog: see IKernelHeuristic for why that constraint
/// is what keeps a knob-filtered query consistent with the default it reported.
using ScoreFn = double (*)(const KernelDefinition&, const MatchContext&);

/**
 * @brief The provider's registry of native implementations, keyed by symbol name.
 *
 * One instance per registered type per process, reached through the static accessors
 * below. Registration happens at static-init time from the provider's own translation
 * units; lookup happens on the applicability, ranking, and dispatch paths.
 *
 * Guarded by a mutex because registration and lookup can race: a provider loaded on one
 * thread while another is already matching is unusual but not forbidden.
 *
 * @tparam T The registered callable or interface pointer type.
 */
template <typename T>
class NativeRegistry
{
public:
    /**
     * @brief Registers @p implementation under @p symbol.
     * @throws std::runtime_error if @p symbol is already registered. Two implementations
     *         behind one name means one of them is silently unreachable, and which one
     *         wins would depend on static-init order — so this is always an author bug
     *         and is reported rather than resolved.
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

    static bool isRegistered(const std::string& symbol)
    {
        auto& self = instance();
        const std::lock_guard<std::mutex> lock(self._mutex);
        return self._symbols.find(symbol) != self._symbols.end();
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
        // Function-local static: initialized on first use, so this is immune to the
        // unspecified initialization order of the namespace-scope registrations that
        // populate it.
        static NativeRegistry s_instance;
        return s_instance;
    }

    std::mutex _mutex;
    std::unordered_map<std::string, T> _symbols;
};

using GraphMatcherRegistry = NativeRegistry<GraphMatcherFn>;
using KernelMatcherRegistry = NativeRegistry<KernelMatcherFn>;
using ScoreRegistry = NativeRegistry<ScoreFn>;

/// Dispatch handlers are registered as non-owning pointers to provider-owned objects,
/// because a handler holds provider state (a compiler, a module cache) whose lifetime is
/// the provider's, not the registry's. The provider must keep the handler alive for as
/// long as any plan built from it can execute.
template <typename THandle>
using DispatchRegistry = NativeRegistry<const IKernelDispatchHandler<THandle>*>;

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
