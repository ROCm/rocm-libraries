// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <stdexcept>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "CpuReferenceGraphExecutorAdapter.hpp"
#include "IReferenceGraphExecutor.hpp"
#include "TestConfig.hpp"
#include "gpu_graph_executor/GpuReferenceGraphExecutor.hpp"

namespace hipdnn_integration_tests
{

class ReferenceGraphExecutorFactory
{
public:
    static std::unique_ptr<IReferenceGraphExecutor> create(ReferenceExecutorType type)
    {
        switch(type)
        {
        case ReferenceExecutorType::CPU:
            HIPDNN_PLUGIN_LOG_TRACE(
                "ReferenceGraphExecutorFactory: creating CPU reference executor");
            return std::make_unique<CpuReferenceGraphExecutorAdapter>();
        case ReferenceExecutorType::GPU:
            HIPDNN_PLUGIN_LOG_TRACE(
                "ReferenceGraphExecutorFactory: creating GPU reference executor");
            return std::make_unique<gpu_graph_executor::GpuReferenceGraphExecutor>();
        default:
            throw std::runtime_error("Unknown reference executor type");
        }
    }

    /// Convenience: create from TestConfig
    static std::unique_ptr<IReferenceGraphExecutor> createFromConfig()
    {
        return create(TestConfig::get().getReferenceExecutorType());
    }
};

} // namespace hipdnn_integration_tests
