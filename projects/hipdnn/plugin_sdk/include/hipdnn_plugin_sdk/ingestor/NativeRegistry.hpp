// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

/// @file NativeRegistry.hpp
/// @brief Symbol name to native function, the escape hatch descriptors resolve
/// through instead of carrying inline code. Lookup fails closed.
namespace hipdnn_plugin_sdk::ingestor
{

/// The four descriptor-driven stages, in pipeline order. Each takes the same operands
/// in the same order, widening as later state exists: the device and graph
/// (`MatchContext`), what the engine's graph match bound (`BoundTokens`), and the
/// kernel under consideration (`KernelDefinition`). A stage takes an operand only when
/// it exists at that point -- there is no kernel yet at graph scope, and the graph
/// match is what produces the bindings -- so the prefix each one takes is the whole of
/// what is knowable there.
///
/// This ordering is the contract a future declarative form mirrors as `$device`,
/// `$graph`, `$<token>`, and `$kernel`: the same access, spelled two ways. Widen every
/// stage together, or the two spellings drift.
///
/// Implementations of these four must be thread-safe.
using GraphMatchFn = std::optional<BoundTokens> (*)(const MatchContext&);
using GraphCriterionFn = bool (*)(const MatchContext&, const BoundTokens&);
using KernelMatcherFn = bool (*)(const MatchContext&, const BoundTokens&, const KernelDefinition&);
using ScoreFn = double (*)(const MatchContext&, const BoundTokens&, const KernelDefinition&);

/// A kernel-scoped matcher's set-level refinement, for an applicability rule that is a
/// property of the candidate set rather than of one kernel ("only the widest variant
/// the graph admits", say). A pack registers one under the symbol of a kernel-scoped
/// UMD it already declares; no descriptor names it. Once every pack has been matched,
/// it sees every kernel that UMD admitted for the graph, across all packs listing it,
/// in catalog order, and returns one keep flag per kernel. It can only narrow the
/// catalog: what it keeps stays in order, and ranking orders it as before. It resolves
/// with its matcher, once per engine's shared state (KernelIngestorSharedState), so it
/// must be registered before the engine loads. Must be thread-safe, like the stages
/// above.
using CandidateFilterFn = std::vector<bool> (*)(const MatchContext&,
                                                const BoundTokens&,
                                                const std::vector<const KernelDefinition*>&);

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

    /// The descriptor loader's pre-flight check: a descriptor naming an unregistered
    /// symbol is dropped with a diagnostic, rather than paying an exception via resolve().
    static bool isRegistered(const std::string& symbol)
    {
        auto& self = instance();
        const std::lock_guard<std::mutex> lock(self._mutex);
        return self._symbols.find(symbol) != self._symbols.end();
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

using GraphMatchRegistry = NativeRegistry<GraphMatchFn>;
using GraphCriterionRegistry = NativeRegistry<GraphCriterionFn>;
using KernelMatcherRegistry = NativeRegistry<KernelMatcherFn>;
using ScoreRegistry = NativeRegistry<ScoreFn>;
using CandidateFilterRegistry = NativeRegistry<CandidateFilterFn>;
using KernelPrepareRegistry = NativeRegistry<KernelPrepareFn>;

/// Non-owning: the provider must keep each handler alive for as long as any plan
/// built from it can execute.
template <typename THandle>
using DispatchRegistry = NativeRegistry<const IKernelDispatchHandler<THandle>*>;

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
