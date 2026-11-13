// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/ReferenceExecutorPool.hpp"

#include <stdexcept>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "harness/CpuReferenceGraphExecutorAdapter.hpp"
#include "harness/gpu-graph-executor/GpuReferenceGraphExecutor.hpp"

namespace hipdnn_integration_tests
{

IReferenceGraphExecutor& ReferenceExecutorPool::get(ReferenceExecutorType type)
{
    switch(type)
    {
    case ReferenceExecutorType::CPU:
        if(_cpu == nullptr)
        {
            HIPDNN_PLUGIN_LOG_TRACE("ReferenceExecutorPool: creating CPU reference executor");
            _cpu = std::make_unique<CpuReferenceGraphExecutorAdapter>();
        }
        return *_cpu;
    case ReferenceExecutorType::GPU:
        if(_gpu == nullptr)
        {
            HIPDNN_PLUGIN_LOG_TRACE("ReferenceExecutorPool: creating GPU reference executor");
            _gpu = std::make_unique<gpu_graph_executor::GpuReferenceGraphExecutor>();
        }
        return *_gpu;
    default:
        throw std::runtime_error("Unknown reference executor type");
    }
}

std::shared_ptr<IReferenceExecutors> sharedReferenceExecutors()
{
    static const std::shared_ptr<IReferenceExecutors> s_pool
        = std::make_shared<ReferenceExecutorPool>();
    return s_pool;
}

} // namespace hipdnn_integration_tests
