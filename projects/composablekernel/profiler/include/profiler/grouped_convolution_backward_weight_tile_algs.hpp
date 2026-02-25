// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <tuple>

#include "../../experimental/builder/test/utils/conv_algorithm_type_utils.hpp"
#include "grouped_convolution_signatures.hpp"
#include "ck_tile/ref/naive_grouped_conv_bwd_weight_gpu.hpp"

#include "ck_tile/builder/testing/filter_extent.hpp"
#include "ck_tile/builder/testing/conv/fwd.hpp"
#include "ck_tile/builder/testing/conv/ck_tile.hpp"
#include "ck_tile/builder/testing/conv/reference.hpp"
#include "ck_tile/builder/conv_builder.hpp"

namespace ck_tile::builder::profiling {

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;

#include "../../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_ndhwgc_fp32.inc"
#include "../../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_nhwgc_fp32.inc"
#include "../../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_nhwgc_bf16.inc"
#include "../../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_nhwgc_fp16.inc"
#include "../../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_ndhwgc_bf16.inc"
#include "../../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_ndhwgc_fp16.inc"

template <auto SIGNATURE>
auto parse_conv_args(int arg_idx, char* const argv[])
{
    const std::size_t G = static_cast<size_t>(std::stol(argv[arg_idx++]));
    const std::size_t N = static_cast<size_t>(std::stol(argv[arg_idx++]));
    const std::size_t K = static_cast<size_t>(std::stol(argv[arg_idx++]));
    const std::size_t C = static_cast<size_t>(std::stol(argv[arg_idx++]));

    constexpr auto num_dim_spatial = SIGNATURE.spatial_dim;

    std::vector<std::size_t> filter_spatial_lengths(num_dim_spatial);
    std::vector<std::size_t> input_spatial_lengths(num_dim_spatial);
    std::vector<std::size_t> conv_filter_strides(num_dim_spatial);
    std::vector<std::size_t> conv_filter_dilations(num_dim_spatial);
    std::vector<std::size_t> input_left_pads(num_dim_spatial);
    std::vector<std::size_t> input_right_pads(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        filter_spatial_lengths[i] = static_cast<size_t>(std::stol(argv[arg_idx++]));
    }

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        input_spatial_lengths[i] = static_cast<size_t>(std::stol(argv[arg_idx++]));
    }

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        conv_filter_strides[i] = static_cast<size_t>(std::stol(argv[arg_idx++]));
    }

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        conv_filter_dilations[i] = static_cast<size_t>(std::stol(argv[arg_idx++]));
    }

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        input_left_pads[i] = static_cast<size_t>(std::stol(argv[arg_idx++]));
    }

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        input_right_pads[i] = static_cast<size_t>(std::stol(argv[arg_idx++]));
    }

    ckt::Args<SIGNATURE> args = {
        .lengths =
            {
                .batch_size      = N,
                .groups          = G,
                .input_channels  = C,
                .output_channels = K,
                .image  = ckt::filter_extent_from_vector<num_dim_spatial>(input_spatial_lengths),
                .filter = ckt::filter_extent_from_vector<num_dim_spatial>(filter_spatial_lengths),
            },
        .filter_strides   = ckt::filter_extent_from_vector<num_dim_spatial>(conv_filter_strides),
        .filter_dilation  = ckt::filter_extent_from_vector<num_dim_spatial>(conv_filter_dilations),
        .input_left_pad   = ckt::filter_extent_from_vector<num_dim_spatial>(input_left_pads),
        .input_right_pad  = ckt::filter_extent_from_vector<num_dim_spatial>(input_right_pads),
        .a_elementwise_op = {},
        .b_elementwise_op = {},
        .cde_elementwise_op = {},
    };
    return args;
}

std::vector<int> get_split_k_values(const std::string& split_k)
{
    std::vector<int> split_k_list = {/*auto deduce value*/ -1, 1, 2, 4, 8, 16, 32, 64, 128};

    if(split_k != "all")
    {
        try
        {
            int split_k_value = std::stoi(split_k);
            split_k_list      = {split_k_value};
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            exit(EXIT_FAILURE);
        }
    }
    return split_k_list;
}

template <auto SIGNATURE>
void run_cpu_validation(const ckt::Args<SIGNATURE>& args,
                        const ckt::Outputs<SIGNATURE>& outputs,
                        const ckt::Outputs<SIGNATURE>& reference)
{
    using DataType =
        std::conditional_t<SIGNATURE.data_type == ckb::DataType::FP32,
                           float,
                           std::conditional_t<SIGNATURE.data_type == ckb::DataType::FP16,
                                              ck_tile::half_t,
                                              ck_tile::bfloat16_t>>;
    const auto conv_param = args.to_ck_tile_conv_param();

    const std::size_t weight_bytes_num = conv_param.template GetWeightByte<DataType>();
    std::vector<DataType> wei(weight_bytes_num / sizeof(DataType));
    std::vector<DataType> ref(weight_bytes_num / sizeof(DataType));
    HIP_CHECK_ERROR(
        hipMemcpy(&ref.data()[0], reference.weight, weight_bytes_num, hipMemcpyDeviceToHost));
    HIP_CHECK_ERROR(
        hipMemcpy(&wei.data()[0], outputs.weight, weight_bytes_num, hipMemcpyDeviceToHost));
    ck_tile::check_err(wei, ref, "\tError: Incorrect results!");
}

template <auto SIGNATURE>
std::tuple<double, double>
get_rtol_atol(const int num_accums, const int num_accums_split_k, const float max_accumulated_value)
{
    using WeiDataType =
        std::conditional_t<SIGNATURE.data_type == ckb::DataType::FP32,
                           float,
                           std::conditional_t<SIGNATURE.data_type == ckb::DataType::FP16,
                                              ck_tile::half_t,
                                              ck_tile::bfloat16_t>>;
    using ComputeType = WeiDataType;
    using AccDataType = float;

    auto rtol = ck_tile::get_relative_threshold<ComputeType, WeiDataType, AccDataType>(
        num_accums / num_accums_split_k);
    auto atol = ck_tile::get_absolute_threshold<ComputeType, WeiDataType, AccDataType>(
        max_accumulated_value / num_accums_split_k, num_accums / num_accums_split_k);
    // Calculate error due to split_k accumulation
    auto rtol_split_k =
        ck_tile::get_relative_threshold<WeiDataType, WeiDataType, WeiDataType>(num_accums_split_k);
    auto atol_split_k = ck_tile::get_absolute_threshold<WeiDataType, WeiDataType, WeiDataType>(
        max_accumulated_value, num_accums_split_k);
    // Use higher threshold
    rtol = std::max(rtol, rtol_split_k);
    atol = std::max(atol, atol_split_k);
    return std::make_tuple(rtol, atol);
}

/// @brief `run_grouped_conv_backward_weight_tile_algs()` run all grouped conv fwd instances.
///
/// @tparam SIGNATURE Forward convolution signature.
///
/// @see run_grouped_conv_backward_weight_tile_algs()
template <auto SIGNATURE>
std::tuple<bool, float, std::string, int>
run_grouped_conv_backward_weight_tile_algs(const ckt::Args<SIGNATURE>& args,
                                           const std::string& split_k,
                                           const ckt::Inputs<SIGNATURE>& inputs,
                                           const ckt::Outputs<SIGNATURE>& outputs,
                                           const ck_tile::stream_config& s_conf)
{
    float best_avg_time = std::numeric_limits<float>::max();
    std::string best_op_name, op_name;
    int best_split_k;
    bool is_supported;
    float avg_time;
    bool all_instances_valid = true;

    using DataType =
        std::conditional_t<SIGNATURE.data_type == ckb::DataType::FP32,
                           float,
                           std::conditional_t<SIGNATURE.data_type == ckb::DataType::FP16,
                                              ck_tile::half_t,
                                              ck_tile::bfloat16_t>>;

    auto reference = ckt::alloc_outputs(args);
    using ReferenceInstance =
        typename ckb::ConvBuilder<SIGNATURE, ckt::ConvAlgorithm_Reference{}>::Instance;
    auto ref_conv   = ReferenceInstance{};
    auto ref_result = ckt::run(ref_conv, args, inputs, reference.get());

    const auto conv_param = args.to_ck_tile_conv_param();

    // Get max possible value in the output
    const std::size_t weight_bytes_num = conv_param.template GetWeightByte<DataType>();
    std::vector<DataType> ref(weight_bytes_num / sizeof(DataType));
    HIP_CHECK_ERROR(
        hipMemcpy(&ref.data()[0], reference.get().weight, weight_bytes_num, hipMemcpyDeviceToHost));
    const float max_accumulated_value = *std::max_element(ref.begin(), ref.end());
    const index_t num_accums = std::accumulate(std::begin(conv_param.output_spatial_lengths_),
                                               std::end(conv_param.output_spatial_lengths_),
                                               static_cast<std::size_t>(1),
                                               std::multiplies<std::size_t>()) *
                               conv_param.N_;
    const auto split_k_values = get_split_k_values(split_k);

    auto run_alg = [&](auto&& run_alg_func) {
        for(auto& k_batch : split_k_values)
        {
            std::tie(is_supported, avg_time, op_name) = run_alg_func(args, inputs, outputs, s_conf);
            if(is_supported)
            {
                ckt::ValidationReport report;
                auto&& [rtol, atol] =
                    get_rtol_atol<SIGNATURE>(num_accums, k_batch, max_accumulated_value);
                ckt::Outputs<SIGNATURE>::reflect(
                    args,
                    [&](std::string_view name,
                        const auto& desc,
                        void* ckt::Outputs<SIGNATURE>::*ptr) {
                        report.check(name, desc, outputs.*ptr, reference.get().*ptr, rtol, atol);
                    });

                const bool valid = report.get_errors().empty();
                if (valid)
                {
                    best_avg_time = std::min(best_avg_time, avg_time);
                    best_op_name  = best_avg_time < avg_time ? best_op_name : op_name;
                    best_split_k  = best_avg_time < avg_time ? best_split_k : k_batch;
                    std::cout << "[Valid] Perf: " << std::setw(10) << avg_time << " ms," << " " << op_name
                            << ", SplitK " << k_batch << std::endl;
                }
                else {
                    std::cout << "[Error] " << op_name << ", SplitK " << k_batch << std::endl;
                    for(const auto& error : report.get_errors())
                    {
                        std::cout << "\tNumber of incorrect values: " << error.wrong_elements
                                << " Is all zero:" << error.is_all_zero()
                                << " max err: " << error.max_error << std::endl;
                        // Check with cpu verification to get a values
                        run_cpu_validation<SIGNATURE>(args, outputs, reference.get());
                    }
                    all_instances_valid = false;
                }
            }
            else
            {
                std::cout << "[Not supported] " << op_name << ", SplitK " << k_batch << std::endl;
            }
        }
    };

    if constexpr(SIGNATURE == SIGNATURE_NHWGC_FP16_BWD_WEIGHT)
    {
#include "../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_nhwgc_fp16_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NHWGC_BF16_BWD_WEIGHT)
    {
#include "../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_nhwgc_bf16_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NHWGC_FP32_BWD_WEIGHT)
    {
#include "../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_ndhwgc_fp32_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NDHWGC_FP16_BWD_WEIGHT)
    {
#include "../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_ndhwgc_fp16_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NDHWGC_BF16_BWD_WEIGHT)
    {
#include "../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_ndhwgc_bf16_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NDHWGC_FP32_BWD_WEIGHT)
    {
#include "../../experimental/grouped_convolution_tile_instances/instances/backward_weight/grouped_convolution_backward_weight_tile_ndhwgc_fp32_calls.inc"
    }
    else
    {
        std::cout << "Signature not supported" << std::endl;
        return std::make_tuple(false, best_avg_time, best_op_name, best_split_k);
    }
    return std::make_tuple(all_instances_valid, best_avg_time, best_op_name, best_split_k);
}

} // namespace ck_tile::builder::profiling
