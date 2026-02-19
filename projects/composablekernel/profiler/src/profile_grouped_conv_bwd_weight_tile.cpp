// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck_tile/builder/testing/conv/ck_tile.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "profiler/grouped_convolution_backward_weight_tile_algs.hpp"

#include "profiler_operation_registry.hpp"

namespace {

enum struct ConvLayout
{
    GNHWC_GKYXC_GNHWK, // 0
    NHWGC_GKYXC_NHWGK, // 1
    NGCHW_GKYXC_NGKHW, // 2
    NGCHW_GKCYX_NGKHW, // 3
};

enum struct ConvDataType
{
    F32_F32_F32,      // 0
    F16_F16_F16,      // 1
    BF16_BF16_BF16,   // 2
    INT8_INT8_INT8,   // 3
    F8_F8_F8,         // 4
    BF8_BF8_F8,       // 5
    F8_BF8_F8,        // 6
    BF8_F8_F8,        // 7
    F32_F32_F32_TF32, // 8
};

enum struct IndexType
{
    INDEX_T,      // 0
    LONG_INDEX_T, // 1
};

#define OP_NAME "grouped_conv_bwd_weight_tile"
#define OP_DESC "Grouped Convolution Backward Weight (CK Tile)"

static void print_helper_msg()
{
    std::cout << "arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n"
              << "arg2: data type (0: Input fp32, Weight fp32, Output fp32\n"
              << "                 1: Input fp16, Weight fp16, Output fp16\n"
              << "                 2: Input bf16, Weight fp32, Output bf16\n"
              << "                 3: Input fp16, Weight fp16, Output fp16, Gemm bf8@fp8\n"
              << "                 4: Input int8, Weight int8, Output int8\n"
              << "                 5: Input bf16, Weight bf16, Output bf16\n"
              << "                 6: Input fp32, Weight fp32, Output fp32, Compute tf32)\n"
              << "arg3: tensor layout (0: Input[G, N, C, Hi, Wi], Weight[G, K, C, Y, X], Output[G, "
                 "N, K, Ho, Wo]\n"
              << "                     1: Input[G, N, Hi, Wi, C], Weight[G, K, Y, X, C], Output[G, "
                 "N, Ho, Wo, K]\n"
              << "                     2: Input[N, Hi, Wi, G, C], Weight[G, K, Y, X, C], Output[N, "
                 "Ho, Wo, G, K]\n"
              << "                     3: Input[N, G, C, Hi, Wi], Weight[G, K, Y, X, C], Output[N, "
                 "G, K, Ho, Wo]\n"
              << "                     4: Input[N, G, C, Hi, Wi], Weight[G, K, C, Y, X], Output[N, "
                 "G, K, Ho, Wo]\n"
              << "arg4: verification (0: no, 1: yes)\n"
              << "arg5: initialization (0: no init, 1: integer value, 2: decimal value)\n"
              << "arg6: print tensor value (0: no; 1: yes)\n"
              << "arg7: time kernel (0: no, 1: yes)\n"
              << ck::utils::conv::get_conv_param_parser_helper_msg()
              << " SplitK (-1 for internally computed split-K value, positive value to set k "
                 "batches explicitly, or 'all' to test all internal split-K values)\n"
              << std::endl;
}

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;
namespace ckp = ck_tile::builder::profiling;

template <auto SIGNATURE>
int call_profiler(const ckt::Args<SIGNATURE>& args, bool time_kernel)
{
    auto inputs  = alloc_inputs(args);
    auto outputs = alloc_outputs(args);
    ckt::init_inputs(args, inputs.get());

    std::cout << args.make_input_descriptor() << std::endl;
    std::cout << args.make_weight_descriptor() << std::endl;
    std::cout << args.make_output_descriptor() << std::endl;
    float avg_time;
    std::string op_name;
    bool valid;
    std::tie(valid, avg_time, op_name) = ckp::run_grouped_conv_backward_weight_tile_algs(
        args, inputs.get(), outputs.get(), ck_tile::stream_config{nullptr, time_kernel});
    if(time_kernel)
    {
        std::cout << "Best configuration parameters:" << "\nname: " << op_name
                  << "\navg_time: " << avg_time << std::endl;
    }
    return !valid;
}

} // namespace

int profile_grouped_conv_bwd_weight_tile(int argc, char* argv[])
{
    // 8 for control, 1 for num_dim_spatial
    if(argc < 9)
    {
        print_helper_msg();
        return 1;
    }

    const auto data_type       = static_cast<ConvDataType>(std::stoi(argv[2]));
    const auto layout          = static_cast<ConvLayout>(std::stoi(argv[3]));
    const bool do_verification = std::stoi(argv[4]);
    const int init_method      = std::stoi(argv[5]);
    const bool do_log          = std::stoi(argv[6]);
    const bool time_kernel     = std::stoi(argv[7]);
    const int num_dim_spatial  = std::stoi(argv[8]);

    // 8 for control, 1 for num_dim_spatial, 4 for G/N/K/C, and 6 * num_dim_spatial, 1 for split-K
    if(argc != 8 + 1 + 4 + 6 * num_dim_spatial + 1)
    {
        print_helper_msg();
        return 1;
    }

    std::cout << "IMPORTANT: Generate instances using: python "
                 "experimental/builder/src/generate_instances.py --mode=profiler and rerun cmake"
              << std::endl;

    const auto params = ck::utils::conv::parse_conv_param(num_dim_spatial, 9, argv);

    const auto& split_k = std::string(argv[8 + 1 + 4 + 6 * num_dim_spatial]);
    if(index_type == IndexType::LONG_INDEX_T)
    {
        std::cout << "this indexing data type is not implemented" << std::endl;
        return 1;
    }

    if(layout == ConvLayout::NHWGC_GKYXC_NHWGK)
    {
        if(num_dim_spatial == 2)
        {
            if(data_type == ConvDataType::F32_F32_F32)
            {
                constexpr auto SIGNATURE = ckp::SIGNATURE_NHWGC_FP32_BWD_WEIGHT;
                return call_profiler<SIGNATURE>(ckp::parse_conv_args<SIGNATURE>(10, argv),
                                                time_kernel);
            }
            else if(data_type == ConvDataType::F16_F16_F16)
            {
                constexpr auto SIGNATURE = ckp::SIGNATURE_NHWGC_FP16_BWD_WEIGHT;
                return call_profiler<SIGNATURE>(ckp::parse_conv_args<SIGNATURE>(10, argv),
                                                time_kernel);
            }
            else if(data_type == ConvDataType::BF16_BF16_BF16)
            {
                constexpr auto SIGNATURE = ckp::SIGNATURE_NHWGC_BF16_BWD_WEIGHT;
                return call_profiler<SIGNATURE>(ckp::parse_conv_args<SIGNATURE>(10, argv),
                                                time_kernel);
            }
        }
        else if(num_dim_spatial == 3)
        {
            if(data_type == ConvDataType::F32_F32_F32)
            {
                constexpr auto SIGNATURE = ckp::SIGNATURE_NDHWGC_FP32_BWD_WEIGHT;
                return call_profiler<SIGNATURE>(ckp::parse_conv_args<SIGNATURE>(10, argv),
                                                time_kernel);
            }
            else if(data_type == ConvDataType::F16_F16_F16)
            {
                constexpr auto SIGNATURE = ckp::SIGNATURE_NDHWGC_FP16_BWD_WEIGHT;
                return call_profiler<SIGNATURE>(ckp::parse_conv_args<SIGNATURE>(10, argv),
                                                time_kernel);
            }
            else if(data_type == ConvDataType::BF16_BF16_BF16)
            {
                constexpr auto SIGNATURE = ckp::SIGNATURE_NDHWGC_BF16_BWD_WEIGHT;
                return call_profiler<SIGNATURE>(ckp::parse_conv_args<SIGNATURE>(10, argv),
                                                time_kernel);
            }
        }
    }

    std::cout << "this data_type & layout is not implemented" << std::endl;

    return 1;
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_grouped_conv_bwd_weight_tile);
