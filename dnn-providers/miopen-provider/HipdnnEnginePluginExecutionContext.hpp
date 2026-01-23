// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>

#include <hipdnn_data_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

struct HipdnnEnginePluginExecutionContext
{
public:
    virtual ~HipdnnEnginePluginExecutionContext() = default;

    bool hasValidPlan() const
    {
        return _plan != nullptr;
    }

    void setPlan(std::unique_ptr<hipdnn_plugin_sdk::IPlan> plan)
    {
        _plan = std::move(plan);
    }

    virtual hipdnn_plugin_sdk::IPlan& plan() const
    {
        if(!hasValidPlan())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "Cannot get plan in execution context, its not set");
        }
        return *_plan;
    }

    void setBenchmarkingEnabled(bool enabled)
    {
        _benchmarkingEnabled = enabled;
    }

    bool benchmarkingEnabled() const
    {
        return _benchmarkingEnabled;
    }

private:
    std::unique_ptr<hipdnn_plugin_sdk::IPlan> _plan;
    bool _benchmarkingEnabled = false;
};
