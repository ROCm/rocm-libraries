// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "harness/ReferenceGraphExecutorFactory.hpp"

#include <stdexcept>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "harness/CpuReferenceGraphExecutorAdapter.hpp"
#include "harness/gpu-graph-executor/GpuReferenceGraphExecutor.hpp"

namespace hipdnn_integration_tests
{

std::unique_ptr<IReferenceGraphExecutor>
    ReferenceGraphExecutorFactory::create(ReferenceExecutorType type)
{
    switch(type)
    {
    case ReferenceExecutorType::CPU:
        HIPDNN_PLUGIN_LOG_TRACE("ReferenceGraphExecutorFactory: creating CPU reference executor");
        return std::make_unique<CpuReferenceGraphExecutorAdapter>();
    case ReferenceExecutorType::GPU:
        HIPDNN_PLUGIN_LOG_TRACE("ReferenceGraphExecutorFactory: creating GPU reference executor");
        return std::make_unique<gpu_graph_executor::GpuReferenceGraphExecutor>();
    default:
        throw std::runtime_error("Unknown reference executor type");
    }
}

std::unique_ptr<IReferenceGraphExecutor> ReferenceGraphExecutorFactory::createFromConfig()
{
    return create(TestConfig::get().getReferenceExecutorType());
}

} // namespace hipdnn_integration_tests
