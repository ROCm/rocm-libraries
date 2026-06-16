// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// AICK-1303: fused-VectorSize conv kernels for the dynamic-VGPR experiment.
// Defines the real GroupedConvolutionForwardKernel for VectorSize 1/2/4/8 as four
// solo kernels plus one fused_conv that branches on a runtime `sel`. This TU is the
// device-code source for the static analysis (dvgpr/) and the .hsaco. Compiled for
// gfx1250; not launched directly here.
#include <hip/hip_runtime.h>

#include "fused_vectorsize_probe.hpp"
#include "fused_vectorsize_kernels.inc"
