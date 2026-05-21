// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/ExecutionContextBase.hpp>
#include <hipdnn_plugin_sdk/PluginBaseTypes.hpp>

#include "CkDslSettings.hpp"

// Forward declaration to break the include cycle with CkDslHandle.
// CkDslHandle lives at global namespace scope to match the
// EnginePluginImpl.inl convention (the SDK casts the opaque
// HipdnnEnginePluginHandle* directly to HIPDNN_PLUGIN_HANDLE_TYPE
// without a namespace qualifier).
struct CkDslHandle;

namespace ck_dsl_provider {

/// Execution context for the CK DSL provider plugin.
///
/// Inherits from:
/// - HipdnnEnginePluginExecutionContext: opaque-pointer compatibility
///   required by the plugin C ABI.
/// - ExecutionContextBase<CkDslHandle, CkDslSettings>: plan + settings
///   storage shared with every other engine plugin.
///
/// Dual inheritance mirrors the reference example_engine_plugin and
/// miopen-provider; the SDK casts opaque pointers through the
/// HipdnnEnginePluginExecutionContext base, then static_casts to the
/// concrete type to access the templated storage.
struct CkDslContext : HipdnnEnginePluginExecutionContext,
                      hipdnn_plugin_sdk::ExecutionContextBase<::CkDslHandle, CkDslSettings> {};

}  // namespace ck_dsl_provider
