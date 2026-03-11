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

/// GPU-based naive convolution forward plan.
///
/// Compiles and launches a HIP kernel that computes a 2D forward
/// convolution (cross-correlation) over NCHW float tensors.
class ConvFwdPlan : public hipdnn_plugin_sdk::ICompilablePlan<ExamplePluginHandle>
{
public:
    /// @param inputUid   Tensor UID for the input (x) buffer
    /// @param weightUid  Tensor UID for the weight (w) buffer
    /// @param outputUid  Tensor UID for the output (y) buffer
    /// @param n Batch size
    /// @param c Input channels
    /// @param h Input height
    /// @param w Input width
    /// @param k Output channels (number of filters)
    /// @param r Filter height
    /// @param s Filter width
    /// @param outH Output height
    /// @param outW Output width
    /// @param padH Padding height
    /// @param padW Padding width
    /// @param strideH Stride height
    /// @param strideW Stride width
    /// @param blockSize Thread block size for the kernel launch
    /// @param compiler  Kernel compiler reference (must outlive this plan)
    ConvFwdPlan(int64_t inputUid,
                int64_t weightUid,
                int64_t outputUid,
                int64_t n,
                int64_t c,
                int64_t h,
                int64_t w,
                int64_t k,
                int64_t r,
                int64_t s,
                int64_t outH,
                int64_t outW,
                int64_t padH,
                int64_t padW,
                int64_t strideH,
                int64_t strideW,
                int64_t blockSize,
                const IKernelCompiler& compiler);

    ~ConvFwdPlan() override = default;

    void compile(const hipDeviceProp_t& deviceProperties) override;

    size_t getWorkspaceSize(const ExamplePluginHandle& handle) const override;

    void execute(const ExamplePluginHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace) const override;

private:
    int64_t _inputUid;
    int64_t _weightUid;
    int64_t _outputUid;
    int64_t _n;
    int64_t _c;
    int64_t _h;
    int64_t _w;
    int64_t _k;
    int64_t _r;
    int64_t _s;
    int64_t _outH;
    int64_t _outW;
    int64_t _padH;
    int64_t _padW;
    int64_t _strideH;
    int64_t _strideW;
    int64_t _blockSize;

    const IKernelCompiler& _compiler;

    std::unique_ptr<ICompiledProgram> _compiledProgram;
    std::unique_ptr<IRunnableKernel> _kernel;
};

} // namespace example_plugin
