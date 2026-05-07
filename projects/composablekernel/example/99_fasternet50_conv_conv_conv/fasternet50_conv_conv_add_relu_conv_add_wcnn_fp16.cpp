// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "common_fasternet50.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/wrapper/layout.hpp"
#include "ck/wrapper/tensor.hpp"

// kernel data types
using InKernelDataType       = FP16;
using WeiKernelDataType      = FP16;
using AccDataType            = FP16;
using CShuffleDataType       = FP16;
using BiasKernelDataType     = FP16;
using ScaleKernelDataType    = FP16;
using ResidualKernelDataType = FP16;
using OutKernelDataType      = FP16;

// tensor data types
using InUserDataType  = InKernelDataType;
using WeiUserDataType = WeiKernelDataType;
using OutUserDataType = OutKernelDataType;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;

// Clamp=False, ActiveFunc=0
using OutElementOp = ck::Tuple<PassThrough, MultiplyAdd, PassThrough>;

#include "run_fasternet50_conv_conv_add_relu_conv_add_wcnn_example.inc"

int main(int argc, char* argv[])
{
    bool is_supported = ck::is_gfx13_supported();
    if(!is_supported)
    {
        std::cout << "WARNING: wcnn example not supported on the platform " << ck::get_device_name()
                  << std::endl;
        return 0;
    }
    return run_fasternet50_conv_conv_add_relu_conv_add_wcnn_example(argc, argv);
}
