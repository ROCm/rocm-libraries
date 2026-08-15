// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Opt-in dev diagnostics for kernel authors. The plugin log level defaults to
// OFF (HIPDNN_LOG_LEVEL), so the engine's INFO/WARN/ERROR breadcrumbs are
// invisible unless explicitly enabled -- which makes a mis-pathed catalog, a bad
// family.json, or a non-matching kernel look like a silent no-op. Setting
// HIPDNN_AOT_DEBUG=1 turns on these always-visible stderr diagnostics (catalog
// resolution + load summary + per-graph decline reasons), independent of the
// plugin log level, so an author can see WHY their kernel was not selected.

#pragma once

#include <sstream>
#include <string>

namespace aot_catalog_engine
{

// True when HIPDNN_AOT_DEBUG is set to a value other than empty/0/false/off.
// Evaluated once and cached (the environment does not change mid-run).
bool aotDebugEnabled();

// Write one diagnostic line to stderr, independent of the plugin log level.
void aotDebugEmit(const std::string& message);

} // namespace aot_catalog_engine

// Stream-style, mirrors HIPDNN_PLUGIN_LOG_*: AOT_DEBUG("root=" << dir << " ...").
// The stream expression is only evaluated when HIPDNN_AOT_DEBUG is enabled.
#define AOT_DEBUG(streamExpr)                                                                             \
    do                                                                                                    \
    {                                                                                                     \
        if(::aot_catalog_engine::aotDebugEnabled())                                                       \
        {                                                                                                 \
            std::ostringstream _aotDbg;                                                                   \
            _aotDbg                                                                                       \
                << streamExpr; /* NOLINT(bugprone-macro-parentheses) streamExpr is a stream expression */ \
            ::aot_catalog_engine::aotDebugEmit(_aotDbg.str());                                            \
        }                                                                                                 \
    } while(0)
