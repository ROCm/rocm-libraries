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

/// @file SymbolScope.hpp
/// @brief All-or-nothing registration of one pack's native symbols: a pack
/// contributes every symbol it declares or none, and one pack failing leaves every
/// other pack's symbols intact.

/// Registers eagerly so a duplicate symbol is caught at the point of registration,
/// and rolls back everything added unless commit() runs. Move-only; not thread-safe.
template <typename THandle>
class SymbolScope
{
public:
    SymbolScope() = default;

    ~SymbolScope()
    {
        for(const auto& undo : _undo)
        {
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

    /// @p handler must outlive every plan built from it; packs use a
    /// process-lifetime static.
    /// @throws std::runtime_error if @p symbol is already registered.
    void add(const std::string& symbol, const IKernelDispatchHandler<THandle>* handler)
    {
        DispatchRegistry<THandle>::registerSymbol(symbol, handler);
        recordUndo(&DispatchRegistry<THandle>::unregisterSymbol, symbol);
    }

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
