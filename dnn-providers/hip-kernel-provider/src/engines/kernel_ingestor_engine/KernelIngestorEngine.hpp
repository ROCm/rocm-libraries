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
 * Called from Container's constructor, so a UED loaded from a descriptor file with no
 * compile-time link to the symbol's translation unit still finds it. Idempotent.
 */
void registerNativeIngestorSymbols();

/// The once_flag-guarded body of registerNativeIngestorSymbols(), callable directly so
/// a test can force and observe a partial-failure/rollback cycle.
void registerNativeIngestorSymbolsOnce();

/**
 * @brief Builds the descriptor-backed pointwise-add engine.
 *
 * The device resolver and dispatch handler its kernels need are process-lifetime
 * statics (see KernelIngestorEngine.cpp), not members.
 *
 * Requires registerNativeIngestorSymbols() to have run first.
 */
std::unique_ptr<hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>> makePointwiseAddEngine();

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
