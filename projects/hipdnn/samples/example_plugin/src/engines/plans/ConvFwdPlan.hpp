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

#include "ConvFwdParams.hpp"

namespace example_plugin
{

class IKernelCompiler;

/// GPU-based naive convolution forward plan.
///
/// Compiles and launches a HIP kernel that computes a 2D forward
/// convolution (cross-correlation) over NCHW float tensors.
class ConvFwdPlan : public hipdnn_plugin_sdk::IPlan<ExamplePluginHandle>
{
public:
    explicit ConvFwdPlan(ConvFwdParams&& params);

    ~ConvFwdPlan() override = default;

    void compile(const IKernelCompiler& kernelCompiler, const hipDeviceProp_t& deviceProperties);

    size_t getWorkspaceSize(const ExamplePluginHandle& handle) const override;

    void execute(const ExamplePluginHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace) const override;

private:
    ConvFwdParams _params;

    std::unique_ptr<ICompiledProgram> _compiledProgram;
    std::unique_ptr<IRunnableKernel> _kernel;
};

} // namespace example_plugin
