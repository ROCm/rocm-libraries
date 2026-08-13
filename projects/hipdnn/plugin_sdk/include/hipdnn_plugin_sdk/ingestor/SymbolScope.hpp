// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <string>
#include <utility>
#include <vector>

#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/**
 * @file SymbolScope.hpp
 * @brief All-or-nothing registration of one pack's native symbols.
 *
 * A pack registers several symbols that only mean anything together: a descriptor
 * naming a matcher this pack half-registered would resolve to nothing at query time,
 * long after the failure. SymbolScope makes a pack's registration atomic, so a pack
 * either contributes every symbol it declares or none of them, and one pack failing
 * leaves every other pack's symbols intact.
 */

/**
 * @brief A pack's in-progress symbol registrations, rolled back unless committed.
 *
 * Registers eagerly so a duplicate symbol is caught at the point of registration, and
 * unwinds in the destructor unless commit() ran. Move-only; not thread-safe, being a
 * scratch object for one pack's registration.
 *
 * @tparam THandle The provider's handle type, for its dispatch registry.
 */
template <typename THandle>
class SymbolScope
{
public:
    SymbolScope() = default;

    /// Rolls back every symbol added since construction unless commit() ran.
    ~SymbolScope()
    {
        for(const auto& undo : _undo)
        {
            // Registry removal is a no-throw erase, so this is safe from a destructor.
            undo.unregister(undo.symbol);
        }
    }

    SymbolScope(const SymbolScope&) = delete;
    SymbolScope& operator=(const SymbolScope&) = delete;
    SymbolScope(SymbolScope&&) = delete;
    SymbolScope& operator=(SymbolScope&&) = delete;

    /// @throws std::runtime_error if @p symbol is already registered.
    void add(const std::string& symbol, GraphMatcherFn matcher)
    {
        GraphMatcherRegistry::registerSymbol(symbol, matcher);
        recordUndo(&GraphMatcherRegistry::unregisterSymbol, symbol);
    }

    /// @throws std::runtime_error if @p symbol is already registered.
    void add(const std::string& symbol, KernelMatcherFn matcher)
    {
        KernelMatcherRegistry::registerSymbol(symbol, matcher);
        recordUndo(&KernelMatcherRegistry::unregisterSymbol, symbol);
    }

    /// @throws std::runtime_error if @p symbol is already registered.
    void add(const std::string& symbol, ScoreFn scorer)
    {
        ScoreRegistry::registerSymbol(symbol, scorer);
        recordUndo(&ScoreRegistry::unregisterSymbol, symbol);
    }

    /**
     * @brief Registers a dispatch handler.
     *
     * @p handler is held by pointer and must outlive every plan built from it; packs
     * use a process-lifetime static.
     *
     * @throws std::runtime_error if @p symbol is already registered.
     */
    void add(const std::string& symbol, const IKernelDispatchHandler<THandle>* handler)
    {
        DispatchRegistry<THandle>::registerSymbol(symbol, handler);
        recordUndo(&DispatchRegistry<THandle>::unregisterSymbol, symbol);
    }

    /// Keeps every symbol added so far. After this the destructor rolls nothing back.
    void commit() noexcept
    {
        _undo.clear();
    }

private:
    struct Undo
    {
        void (*unregister)(const std::string&);
        std::string symbol;
    };

    /// Records the undo for a symbol that was just registered.
    ///
    /// Reserves before registering would be simpler, but callers add one symbol at a
    /// time; instead this rolls the just-registered symbol back itself if recording
    /// throws, so no path leaves a symbol registered with no undo entry.
    void recordUndo(void (*unregister)(const std::string&), const std::string& symbol)
    {
        try
        {
            _undo.push_back(Undo{unregister, symbol});
        }
        catch(...)
        {
            unregister(symbol);
            throw;
        }
    }

    std::vector<Undo> _undo;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
