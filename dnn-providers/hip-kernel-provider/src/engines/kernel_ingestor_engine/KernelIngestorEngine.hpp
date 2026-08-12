// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <memory>

#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>

#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

/**
 * @brief Registers every native matcher, scorer, and dispatch handler this provider's
 *        ingestor packs resolve through, exactly once for the life of the process.
 *
 * Called eagerly from Container's constructor -- which runs inside the plugin's create
 * entry point -- rather than lazily as a side effect of building today's one hardcoded
 * engine. That laziness is the ALMIOPEN-2401 blocker this closes: a UED loaded from a
 * descriptor file names a symbol like "hipkernel.pointwise_add.graph_match" with no
 * compile-time link to the C++ translation unit that implements it, so nothing after a
 * file-loaded UED can rely on registration happening as a byproduct of constructing the
 * one engine that used to hardcode it.
 *
 * Idempotent across repeated calls: Container can be destroyed and rebuilt many times
 * over a process's life (see SharedContainerManager's weak_ptr), but the process-wide
 * NativeRegistry this populates must be populated only once or the second call throws
 * on the duplicate.
 */
void registerNativeIngestorSymbols();

/// @brief The once_flag-guarded body of registerNativeIngestorSymbols(), callable
/// directly so a test can force and observe a partial-failure/rollback cycle a second
/// time -- something the flag makes unreachable through the wrapper itself.
void registerNativeIngestorSymbolsOnce();

/**
 * @brief Builds the descriptor-backed pointwise-add engine.
 *
 * Assembles this pack's descriptor set once and hands its engine descriptor and state
 * manager straight to GenericEngine, which already satisfies IEngine end to end -- there
 * is no forwarding wrapper left to own. The device resolver and dispatch handler this
 * engine's kernels need are process-lifetime statics (see KernelIngestorEngine.cpp), not
 * members, so nothing about their lifetime constrains how this object is constructed.
 *
 * Registration is not this function's job: registerNativeIngestorSymbols() must already
 * have run, so that a file-loaded UED sharing this pack's symbols resolves the same way
 * a hardcoded one does.
 *
 * This function is what a UED loader replaces: given descriptor files, the same generic
 * engine is constructed from parsed data instead of from a hardcoded pack.
 */
std::unique_ptr<hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>> makePointwiseAddEngine();

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
