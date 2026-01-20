// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

namespace hipdnn_plugin_sdk
{

/**
 * @brief Interface for an execution context.
 *
 * An IExecutionContext encapsulates the state needed to execute an operation graph.
 * It holds a reference to an IPlan that was built for a specific graph and engine
 * configuration.
 *
 * Execution contexts are created when a plan is finalized and are meant to be
 * immutable and reusable with different device buffers.
 *
 * @note The lifecycle of an execution context is controlled by the consuming
 *       application through the plugin's create/destroy entry points.
 */
class IExecutionContext
{
public:
    virtual ~IExecutionContext() = default;

    /**
     * @brief Checks if this context has a valid plan attached.
     *
     * @return true if a plan has been set, false otherwise.
     */
    virtual bool hasValidPlan() const = 0;

    /**
     * @brief Sets the plan for this execution context.
     *
     * @param plan The plan to attach to this context. Ownership is transferred.
     */
    virtual void setPlan(std::unique_ptr<IPlan> plan) = 0;

    /**
     * @brief Gets the plan attached to this execution context.
     *
     * @return Reference to the attached plan.
     * @throws HipdnnPluginException if no plan has been set.
     */
    virtual IPlan& getPlan() const = 0;
};

/**
 * @brief Default implementation of IExecutionContext.
 *
 * Provides basic plan storage and retrieval functionality that plugins can
 * use directly or extend with additional state.
 */
class ExecutionContextBase : public IExecutionContext
{
public:
    ~ExecutionContextBase() override = default;

    bool hasValidPlan() const override
    {
        return _plan != nullptr;
    }

    void setPlan(std::unique_ptr<IPlan> plan) override
    {
        _plan = std::move(plan);
    }

    IPlan& getPlan() const override
    {
        if(!hasValidPlan())
        {
            throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                        "Cannot get plan from execution context, plan is not set");
        }
        return *_plan;
    }

private:
    std::unique_ptr<IPlan> _plan;
};

} // namespace hipdnn_plugin_sdk
