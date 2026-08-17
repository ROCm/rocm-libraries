// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

namespace hipdnn_plugin_sdk
{

/**
 * @brief Base template for execution context implementations.
 *
 * This template provides common functionality for execution contexts:
 * - Plan storage and access
 * - Execution settings storage and access
 *
 * Plugin implementations should inherit from both HipdnnEnginePluginExecutionContext
 * (for the opaque pointer) and this template (for the implementation).
 *
 * @tparam THandle The plugin-specific handle type (e.g., HipdnnMiopenHandle).
 * @tparam TSettings The plugin-specific settings type (e.g., HipdnnMiopenSettings).
 *
 * Example:
 * @code
 * struct HipdnnMiopenContext :
 *     HipdnnEnginePluginExecutionContext,
 *     ExecutionContextBase<HipdnnMiopenHandle, HipdnnMiopenSettings>
 * {};
 * @endcode
 */
template <typename THandle, typename TSettings>
class ExecutionContextBase
{
public:
    virtual ~ExecutionContextBase() = default;

    /**
     * @brief Checks if a valid plan has been set.
     *
     * @return true if a plan has been set, false otherwise.
     */
    bool hasValidPlan() const
    {
        return _plan != nullptr;
    }

    /**
     * @brief Sets the plan for this execution context.
     *
     * @param plan The plan to set. Ownership is transferred.
     */
    void setPlan(std::unique_ptr<IPlan<THandle>> plan)
    {
        _plan = std::move(plan);
    }

    /**
     * @brief Gets the plan for this execution context.
     *
     * @return Reference to the plan.
     * @throws HipdnnPluginException if no plan has been set.
     */
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    virtual IPlan<THandle>& plan() const
    {
        if(!hasValidPlan())
        {
            throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                        "Cannot get plan in execution context, it's not set");
        }
        return *_plan;
    }

    /**
     * @brief Sets the execution settings for this context.
     *
     * @param executionSettings The settings to set.
     */
    void setExecutionSettings(const TSettings& executionSettings)
    {
        _executionSettings = executionSettings;
    }

    /**
     * @brief Gets the execution settings for this context.
     *
     * @return Reference to the execution settings.
     */
    const TSettings& executionSettings() const
    {
        return _executionSettings;
    }

private:
    std::unique_ptr<IPlan<THandle>> _plan;
    TSettings _executionSettings;
};

} // namespace hipdnn_plugin_sdk
