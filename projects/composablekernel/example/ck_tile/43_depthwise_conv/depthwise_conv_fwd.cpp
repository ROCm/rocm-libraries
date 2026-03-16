// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "depthwise_conv_fwd_invoker.hpp"
#include "run_depthwise_conv_fwd_example.inc"

int run_depthwise_conv_fwd_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    const std::string data_type = arg_parser.get_str("prec");

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
        throw std::runtime_error("Unsupported data type: " + data_type);
    }
}

int main(int argc, char* argv[])
{
    try
    {
        return !run_depthwise_conv_fwd_example(argc, argv);
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
