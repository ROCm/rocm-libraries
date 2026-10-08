// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <optional>

#include <hipdnn_plugin_sdk/ExecutionContextBase.hpp>
#include <hipdnn_plugin_sdk/PluginBaseTypes.hpp>

#include "Settings.hpp"

// Forward declaration
struct Handle;

/**
 * @brief HIP kernel provider plugin execution context.
 *
 * Inherits from:
 * - HipdnnEnginePluginExecutionContext: For opaque pointer compatibility
 * - ExecutionContextBase: For plan and settings storage
 */
struct Context : HipdnnEnginePluginExecutionContext,
                 hipdnn_plugin_sdk::ExecutionContextBase<Handle, Settings>
{
    /**
     * @brief Records the ID of the engine that built or restored the plan of this context.
     */
    void setEngineId(int64_t engineId)
    {
        _engineId = engineId;
    }

    /**
     * @brief The ID of the engine that built or restored the plan of this context.
     *
     * Set when a kernel ingestor engine builds the plan or the load hook restores it. No value
     * for a context of another engine.
     */
    std::optional<int64_t> engineId() const
    {
        return _engineId;
    }

private:
    std::optional<int64_t> _engineId;
};
