// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <hipdnn_plugin_sdk/interfaces/ICompilablePlan.hpp>

#include "ExamplePluginHandle.hpp"
#include "hip/ICompiledProgram.hpp"
#include "hip/IKernelCompiler.hpp"
#include "hip/IRunnableKernel.hpp"

namespace example_plugin
{

/// GPU-based ReLU forward plan.
///
/// Compiles and launches a HIP kernel that applies the ReLU activation
/// function (with optional leaky negative slope) to a float tensor.
class ReluPlan : public hipdnn_plugin_sdk::ICompilablePlan<ExamplePluginHandle>
{
public:
    /// @param inputUid  Tensor UID for the input buffer
    /// @param outputUid Tensor UID for the output buffer
    /// @param numElements Total number of float elements
    /// @param negativeSlope Leaky ReLU slope (0.0 for standard ReLU)
    /// @param compiler  Kernel compiler reference (must outlive this plan)
    ReluPlan(int64_t inputUid,
             int64_t outputUid,
             int64_t numElements,
             double negativeSlope,
             const IKernelCompiler& compiler);

    ~ReluPlan() override = default;

    void compile(const hipDeviceProp_t& deviceProperties) override;

    size_t getWorkspaceSize(const ExamplePluginHandle& handle) const override;

    void execute(const ExamplePluginHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace) const override;

private:
    int64_t _inputUid;
    int64_t _outputUid;
    int64_t _numElements;
    double _negativeSlope;

    const IKernelCompiler& _compiler;

    std::unique_ptr<ICompiledProgram> _compiledProgram;
    std::unique_ptr<IRunnableKernel> _kernel;
};

} // namespace example_plugin
