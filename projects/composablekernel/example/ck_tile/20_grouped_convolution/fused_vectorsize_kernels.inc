// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// AICK-1303: the fused-VectorSize conv kernels. Shared by fused_vectorsize_probe.cpp
// (device-code source for the static analysis / .hsaco) and
// fused_vectorsize_harness.cpp (compiled in, launched via <<<>>>). Include after
// fused_vectorsize_probe.hpp (which defines K1/K2/K4/K8).

// One kernel, four real conv load paths selected at runtime. Four kargs params (one
// per VectorSize kernel type); only the selected path executes.
extern "C" __global__ void __launch_bounds__(256, 2)
    fused_conv(K1::GroupedConvFwdKernelArgsSpecialized a1,
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
extern "C" __global__ void __launch_bounds__(256, 2)
    solo1(K1::GroupedConvFwdKernelArgsSpecialized a) { K1{}(a); }
extern "C" __global__ void __launch_bounds__(256, 2)
    solo2(K2::GroupedConvFwdKernelArgsSpecialized a) { K2{}(a); }
extern "C" __global__ void __launch_bounds__(256, 2)
    solo4(K4::GroupedConvFwdKernelArgsSpecialized a) { K4{}(a); }
extern "C" __global__ void __launch_bounds__(256, 2)
    solo8(K8::GroupedConvFwdKernelArgsSpecialized a) { K8{}(a); }
