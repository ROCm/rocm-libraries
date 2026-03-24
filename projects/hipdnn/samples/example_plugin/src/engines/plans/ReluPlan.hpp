// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "ExamplePluginHandle.hpp"
#include "hip/ICompiledProgram.hpp"
#include "hip/IRunnableKernel.hpp"

#include "ReluParams.hpp"

namespace example_plugin
{

class IKernelCompiler;

/// GPU-based ReLU forward plan.
///
/// Compiles and launches a HIP kernel that applies the ReLU activation
/// function (with optional leaky negative slope) to a float tensor.
class ReluPlan : public hipdnn_plugin_sdk::IPlan<ExamplePluginHandle>
{
public:
    explicit ReluPlan(ReluParams&& params);

    ~ReluPlan() override = default;

    void compile(const IKernelCompiler& kernelCompiler, const hipDeviceProp_t& deviceProperties);

    size_t getWorkspaceSize(const ExamplePluginHandle& handle) const override;

    void execute(const ExamplePluginHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace) const override;

private:
    ReluParams _params;

    std::unique_ptr<ICompiledProgram> _compiledProgram;
    std::unique_ptr<IRunnableKernel> _kernel;
};

} // namespace example_plugin
