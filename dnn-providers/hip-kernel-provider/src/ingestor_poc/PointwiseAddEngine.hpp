// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <memory>

#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>

#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"
#include "device/IDevicePropertyProvider.hpp"

namespace hip_kernel_provider::ingestor_poc
{

/**
 * @brief Builds the descriptor-backed pointwise-add engine.
 *
 * Registers this pack's native implementations, assembles its descriptor set, and
 * returns a generic engine driven entirely by that data. The engine owns the kernel
 * compiler and dispatch handler its kernels need, so both outlive every plan it builds.
 *
 * This function is what a UED loader replaces: given descriptor files, the same generic
 * engine is constructed from parsed data instead of from a hardcoded pack.
 */
std::unique_ptr<hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>>
    makePointwiseAddEngine(const device::IDevicePropertyProvider& devicePropertyProvider);

} // namespace hip_kernel_provider::ingestor_poc

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
