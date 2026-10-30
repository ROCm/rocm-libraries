// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>

#include "harness/IReferenceGraphExecutor.hpp"
#include "harness/ReferenceCapabilityError.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor
{

// Runs a serialized graph node by node through the GPU plan builders. The plan
// registry and the flatbuffer machinery sit behind a pimpl so they are compiled
// once (GpuReferenceGraphExecutor.cpp) instead of in every translation unit that
// needs a GPU reference.
class GpuReferenceGraphExecutor : public IReferenceGraphExecutor
{
public:
    GpuReferenceGraphExecutor();
    ~GpuReferenceGraphExecutor() override;

    GpuReferenceGraphExecutor(const GpuReferenceGraphExecutor&) = delete;
    GpuReferenceGraphExecutor& operator=(const GpuReferenceGraphExecutor&) = delete;
    GpuReferenceGraphExecutor(GpuReferenceGraphExecutor&&) = delete;
    GpuReferenceGraphExecutor& operator=(GpuReferenceGraphExecutor&&) = delete;

    bool isApplicable(void* graphBuffer, size_t size) override;

    void execute(void* graphBuffer,
                 size_t size,
                 const std::unordered_map<int64_t, void*>& variantPack) override;

    bool requiresDeviceMemory() const override
    {
        return true;
    }

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace hipdnn_integration_tests::gpu_graph_executor
