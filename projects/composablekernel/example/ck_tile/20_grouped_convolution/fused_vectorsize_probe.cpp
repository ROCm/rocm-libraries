// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// AICK-1303: fused-VectorSize conv kernels for the dynamic-VGPR experiment.
// Defines the real GroupedConvolutionForwardKernel for VectorSize 1/2/4/8 as four
// solo kernels plus one fused_conv that branches on a runtime `sel`. This TU is the
// source for the device code object (.hsaco) used by both the static analysis
// (dvgpr/) and the runtime harness (fused_vectorsize_harness.cpp). Compiled for
// gfx1250; not launched directly here.
#include <hip/hip_runtime.h>

#include "fused_vectorsize_probe.hpp"

// One kernel, four real conv load paths selected at runtime. Four kargs params (one
// per VectorSize kernel type) are passed; only the selected path executes. The kargs
// contents are identical (same conv problem); see the static_assert in the header.
extern "C" __global__ void __launch_bounds__(256, 2) fused_conv(K1::GroupedConvFwdKernelArgsSpecialized a1,
                                      K2::GroupedConvFwdKernelArgsSpecialized a2,
                                      K4::GroupedConvFwdKernelArgsSpecialized a4,
                                      K8::GroupedConvFwdKernelArgsSpecialized a8,
                                      int sel)
{
    if(sel == 0)
        K1{}(a1);
    else if(sel == 1)
        K2{}(a2);
    else if(sel == 2)
        K4{}(a4);
    else
        K8{}(a8);
}

// Solo controls: each VectorSize as its own kernel.
extern "C" __global__ void __launch_bounds__(256, 2) solo1(K1::GroupedConvFwdKernelArgsSpecialized a) { K1{}(a); }
extern "C" __global__ void __launch_bounds__(256, 2) solo2(K2::GroupedConvFwdKernelArgsSpecialized a) { K2{}(a); }
extern "C" __global__ void __launch_bounds__(256, 2) solo4(K4::GroupedConvFwdKernelArgsSpecialized a) { K4{}(a); }
extern "C" __global__ void __launch_bounds__(256, 2) solo8(K8::GroupedConvFwdKernelArgsSpecialized a) { K8{}(a); }
