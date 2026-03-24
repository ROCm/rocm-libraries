// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>

#include "ck_tile/host.hpp"
#include "grouped_convolution_utils.hpp"
#include "grouped_convolution_forward_invoker.hpp"
#include "depthwise_conv_fwd_invoker.hpp"
#include "run_grouped_convolution_fwd_example.inc"
#include "run_depthwise_conv_fwd_example.inc"

template <template <typename PrecType> typename ConvConfig>
int run_grouped_conv_fwd_example(int argc, char* argv[])
{
    using Invoker = GroupedConvolutionForwardInvoker;

    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    std::string data_type = arg_parser.get_str("prec");

    // Depthwise convolution specialization: C=K=1 per group
    if(arg_parser.get_int("c") == 1 && arg_parser.get_int("k") == 1)
    {
        if(data_type == "fp16")
        {
            return run_depthwise_conv_fwd_example_prec<ck_tile::half_t,
                                                       ck_tile::half_t,
                                                       float,
                                                       ck_tile::half_t>(arg_parser);
        }
        else if(data_type == "fp32")
        {
            return run_depthwise_conv_fwd_example_prec<float, float, float, float>(arg_parser);
        }
        else
        {
            throw std::runtime_error("Unsupported data type for depthwise conv: " + data_type);
        }
    }

    // Grouped convolution path (implicit GEMM)
    std::string in_layout  = arg_parser.get_str("in_layout");
    std::string wei_layout = arg_parser.get_str("wei_layout");
    std::string out_layout = arg_parser.get_str("out_layout");

    if(data_type == "fp16")
    {
        return run_grouped_conv_fwd_example_prec_type<Invoker,
                                                      ConvConfig<ck_tile::half_t>,
                                                      ck_tile::half_t>(
            in_layout, wei_layout, out_layout, argc, argv);
    }
    else if(data_type == "bf16")
    {
        return run_grouped_conv_fwd_example_prec_type<Invoker,
                                                      ConvConfig<ck_tile::bf16_t>,
                                                      ck_tile::bf16_t>(
            in_layout, wei_layout, out_layout, argc, argv);
    }
    else
    {
        throw std::runtime_error("Unsupported data type for this operation !!!");
    }
}

int main(int argc, char* argv[])
{
    try
    {
#if CK_TILE_USE_WMMA
        return !run_grouped_conv_fwd_example<ConvConfigComputeV3_WMMA>(argc, argv);
#else
        return !run_grouped_conv_fwd_example<ConvConfigComputeV3>(argc, argv);
#endif
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
