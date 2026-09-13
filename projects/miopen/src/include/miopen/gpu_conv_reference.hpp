// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <miopen/config.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor.hpp>
#include <miopen/convolution.hpp>

namespace miopen {

/// Standalone GPU reference convolution using double-precision accumulators.
///
/// Compiles and launches the naive conv kernel with double accumulators via
/// Handle::AddKernel(), bypassing the solver framework entirely. The kernel
/// is compiled on first use and cached for subsequent calls.
///
/// This replaces the previous approach of toggling AlwaysEnableConvDirectNaive
/// to dual-purpose the naive conv solver as both a solver and a GPU reference.
struct MIOPEN_INTERNALS_EXPORT GpuConvReference
{
    static void RunFwd(const Handle& handle,
                       const TensorDescriptor& xDesc,
                       ConstData_t x,
                       const TensorDescriptor& wDesc,
                       ConstData_t w,
                       const TensorDescriptor& yDesc,
                       Data_t y,
                       const ConvolutionDescriptor& conv,
                       double alpha = 1.0,
                       double beta  = 0.0);

    static void RunBwd(const Handle& handle,
                       const TensorDescriptor& dyDesc,
                       ConstData_t dy,
                       const TensorDescriptor& wDesc,
                       ConstData_t w,
                       const TensorDescriptor& dxDesc,
                       Data_t dx,
                       const ConvolutionDescriptor& conv,
                       double alpha = 1.0,
                       double beta  = 0.0);

    static void RunWrw(const Handle& handle,
                       const TensorDescriptor& dyDesc,
                       ConstData_t dy,
                       const TensorDescriptor& xDesc,
                       ConstData_t x,
                       const TensorDescriptor& dwDesc,
                       Data_t dw,
                       const ConvolutionDescriptor& conv,
                       double alpha = 1.0,
                       double beta  = 0.0);
};

} // namespace miopen
