// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// TEMPLATE REFERENCE: Second Plan example demonstrating the same compile/execute pattern as
// ReluPlan but for a convolution kernel with more parameters.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "ExampleProviderHandle.hpp"
#include "hip/ICompiledProgram.hpp"
#include "hip/IRunnableKernel.hpp"

#include "ConvFwdParams.hpp"

namespace example_provider
{

class IKernelCompiler;

/// GPU-based naive convolution forward plan.
///
/// Compiles and launches a HIP kernel that computes a 2D forward
/// convolution (cross-correlation) over NCHW float tensors.
class ConvFwdPlan : public hipdnn_plugin_sdk::IPlan<ExampleProviderHandle>
{
public:
    explicit ConvFwdPlan(const ConvFwdParams& params);

    ~ConvFwdPlan() override = default;

    void compile(const IKernelCompiler& kernelCompiler);

    size_t getWorkspaceSize(const ExampleProviderHandle& handle) const override;

    void execute(const ExampleProviderHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace) const override;

private:
    ConvFwdParams _params;

    std::unique_ptr<ICompiledProgram> _compiledProgram;
    std::unique_ptr<IRunnableKernel> _kernel;
};

} // namespace example_provider
