// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <tuple>

#include "../../experimental/builder/test/utils/conv_algorithm_type_utils.hpp"
#include "grouped_convolution_signatures.hpp"
#include "common.hpp"
#include "ck_tile/ref/naive_grouped_conv_fwd_gpu.hpp"

#include "ck_tile/builder/testing/filter_extent.hpp"
#include "ck_tile/builder/testing/conv/fwd.hpp"
#include "ck_tile/builder/testing/conv/ck_tile.hpp"
#include "ck_tile/builder/testing/conv/reference.hpp"
#include "ck_tile/builder/conv_builder.hpp"
#include "tile_profiler_common.hpp"
#include "direct_conv_profiler_bridge.hpp"

#define ENABLE_BUILDER_VALIDATE 1

namespace ck_tile::builder::profiling {

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;

#ifndef DISABLE_IMPLICIT_GEMM_INSTANCES
#include "../../experimental/grouped_convolution_tile_instances/instances/forward/grouped_convolution_forward_tile_nhwgc_fp32.inc"
#include "../../experimental/grouped_convolution_tile_instances/instances/forward/grouped_convolution_forward_tile_nhwgc_bf16.inc"
#include "../../experimental/grouped_convolution_tile_instances/instances/forward/grouped_convolution_forward_tile_nhwgc_fp16.inc"
#include "../../experimental/grouped_convolution_tile_instances/instances/forward/grouped_convolution_forward_tile_ndhwgc_fp32.inc"
#include "../../experimental/grouped_convolution_tile_instances/instances/forward/grouped_convolution_forward_tile_ndhwgc_bf16.inc"
#include "../../experimental/grouped_convolution_tile_instances/instances/forward/grouped_convolution_forward_tile_ndhwgc_fp16.inc"
#include "../../experimental/grouped_convolution_tile_instances/instances/forward/grouped_convolution_forward_tile_ngchw_depthwise_fp32.inc"
#include "../../experimental/grouped_convolution_tile_instances/instances/forward/grouped_convolution_forward_tile_ngchw_depthwise_fp16.inc"
#include "../../experimental/grouped_convolution_tile_instances/instances/forward/grouped_convolution_forward_tile_ngchw_depthwise_bf16.inc"
#endif // DISABLE_IMPLICIT_GEMM_INSTANCES

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

    const std::size_t output_bytes_num = conv_param.template GetOutputByte<DataType>();
    std::vector<DataType> out(output_bytes_num / sizeof(DataType));
    std::vector<DataType> ref(output_bytes_num / sizeof(DataType));
    HIP_CHECK_ERROR(
        hipMemcpy(&ref.data()[0], reference.output, output_bytes_num, hipMemcpyDeviceToHost));
    HIP_CHECK_ERROR(
        hipMemcpy(&out.data()[0], outputs.output, output_bytes_num, hipMemcpyDeviceToHost));
    ck_tile::check_err(out, ref, "Error: Incorrect results!");
}

/// @brief `run_grouped_conv_forward_tile_algs()` run all grouped conv fwd instances.
///
/// @tparam SIGNATURE Forward convolution signature.
///
/// @see run_grouped_conv_forward_tile_algs()
template <auto SIGNATURE>
std::tuple<bool, float, float, float, std::string, int>
run_grouped_conv_forward_tile_algs(const ckt::Args<SIGNATURE>& args,
                                   const index_t instance_index,
                                   const ckt::Inputs<SIGNATURE>& inputs,
                                   const ckt::Outputs<SIGNATURE>& outputs,
                                   const ck_tile::stream_config& s_conf,
                                   bool do_verification = true)
{
    using DataType = DeduceDataType<SIGNATURE>;

    // Run first instance as dummy to get proper time from the first instance
    bool dummy_run_executed = false;
    float best_avg_time             = std::numeric_limits<float>::max();
    float best_tflops               = std::numeric_limits<float>::min();
    float best_gbs                  = std::numeric_limits<float>::min();
    std::string best_op_name, op_name;
    ck::index_t best_instance_index = -1;
    bool is_supported;
    float avg_time;
    bool valid = true;

    using DataType =
        std::conditional_t<SIGNATURE.data_type == ckb::DataType::FP32,
                           float,
                           std::conditional_t<SIGNATURE.data_type == ckb::DataType::FP16,
                                              ck_tile::half_t,
                                              ck_tile::bfloat16_t>>;

    auto reference = ckt::alloc_outputs(args);
    if(do_verification)
    {
        using ReferenceInstance =
            typename ckb::ConvBuilder<SIGNATURE, ckt::ConvAlgorithm_Reference{}>::Instance;
        auto ref_conv                    = ReferenceInstance{};
        [[maybe_unused]] auto ref_result = ckt::run(ref_conv, args, inputs, reference.get());
    }
    index_t num_kernel = 0;
    auto run_alg    = [&](auto&& run_alg_func) {
        num_kernel++;
        // Skip if a specific instance was requested and this isn't it
        const bool running_specific_instance = (instance_index != -1);
        const bool current_is_target         = (num_kernel - 1 == instance_index);
        if(running_specific_instance && !current_is_target)
        {
            return;
        }

        std::tie(is_supported, avg_time, op_name) = run_alg_func(args, inputs, outputs, s_conf);
        if(is_supported)
        {
            if((s_conf.time_kernel_ || s_conf.flush_cache_) && !dummy_run_executed)
            {
                // Run first instance twice
                std::tie(is_supported, avg_time, op_name) =
                    run_alg_func(args, inputs, outputs, s_conf);
                dummy_run_executed = true;
            }

            if(do_verification)
            {
                ckt::ValidationReport report;
                ckt::Outputs<SIGNATURE>::reflect(args,
                                                 [&](std::string_view name,
                                                     const auto& desc,
                                                     void* ckt::Outputs<SIGNATURE>::*ptr) {
                                                     report.check(
                                                         name,
                                                         desc,
                                                         outputs.*ptr,
                                                         reference.get().*ptr,
                                                         ck::profiler::get_rtol<DataType>(),
                                                         ck::profiler::get_atol<DataType>());
                                                 });

            const bool instance_valid = report.get_errors().empty();
            if(instance_valid)
            {
                if(avg_time < best_avg_time)
                {
                    best_instance_index = num_kernel - 1;
                }
                best_avg_time = std::min(best_avg_time, avg_time);
                best_op_name  = best_avg_time < avg_time ? best_op_name : op_name;
                const auto conv_param  = args.to_ck_tile_conv_param();
                float tflops           = static_cast<float>(conv_param.GetFlops()) / 1.E9 / avg_time;
                float gb_per_sec       = static_cast<float>(
                    conv_param.template GetByte<DataType, DataType, DataType>()) / 1.E6 / avg_time;
                best_tflops = std::max(best_tflops, tflops);
                best_gbs    = std::max(best_gbs, gb_per_sec);
                std::cout << "[Valid] Perf: " << std::setw(10) << avg_time << " ms, " << tflops
                        << " TFlops, " << gb_per_sec << " GB/s, " << op_name
                        << " (instance " << num_kernel - 1 << ")" << std::endl;
            }
            else
            {
                std::cout << "[Error] " << op_name << std::endl;
                for(const auto& error : report.get_errors())
                {
                    valid = false;
                    std::cout << "\tNumber of incorrect values: " << error.wrong_elements
                              << " Is all zero:" << error.is_all_zero()
                              << " max err: " << error.max_error << std::endl;
                    // Check with cpu verification to get a values
                    run_cpu_validation<SIGNATURE>(args, outputs, reference.get());
                }
            }
        }
        else
        {
            std::cout << "[Not supported] " << op_name << std::endl;
        }
    };

    if constexpr(SIGNATURE == SIGNATURE_NHWGC_FP16_FWD)
    {
#ifndef DISABLE_IMPLICIT_GEMM_INSTANCES
#include "../../experimental/grouped_convolution_tile_instances/instances/forward/grouped_convolution_forward_tile_nhwgc_fp16_calls.inc"
#endif // DISABLE_IMPLICIT_GEMM_INSTANCES
#include "../../experimental/grouped_convolution_tile_instances/instances/forward_direct/grouped_convolution_forward_tile_nhwgc_fp16_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NHWGC_BF16_FWD)
    {
#ifndef DISABLE_IMPLICIT_GEMM_INSTANCES
#include "../../experimental/grouped_convolution_tile_instances/instances/forward/grouped_convolution_forward_tile_nhwgc_bf16_calls.inc"
#endif // DISABLE_IMPLICIT_GEMM_INSTANCES
#include "../../experimental/grouped_convolution_tile_instances/instances/forward_direct/grouped_convolution_forward_tile_nhwgc_bf16_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NHWGC_FP32_FWD)
    {
#ifndef DISABLE_IMPLICIT_GEMM_INSTANCES
#include "../../experimental/grouped_convolution_tile_instances/instances/forward/grouped_convolution_forward_tile_nhwgc_fp32_calls.inc"
#endif // DISABLE_IMPLICIT_GEMM_INSTANCES
    }
    else if constexpr(SIGNATURE == SIGNATURE_NDHWGC_FP16_FWD)
    {
#ifndef DISABLE_IMPLICIT_GEMM_INSTANCES
#include "../../experimental/grouped_convolution_tile_instances/instances/forward/grouped_convolution_forward_tile_ndhwgc_fp16_calls.inc"
#endif // DISABLE_IMPLICIT_GEMM_INSTANCES
    }
    else if constexpr(SIGNATURE == SIGNATURE_NDHWGC_BF16_FWD)
    {
#ifndef DISABLE_IMPLICIT_GEMM_INSTANCES
#include "../../experimental/grouped_convolution_tile_instances/instances/forward/grouped_convolution_forward_tile_ndhwgc_bf16_calls.inc"
#endif // DISABLE_IMPLICIT_GEMM_INSTANCES
    }
    else if constexpr(SIGNATURE == SIGNATURE_NDHWGC_FP32_FWD)
    {
#ifndef DISABLE_IMPLICIT_GEMM_INSTANCES
#include "../../experimental/grouped_convolution_tile_instances/instances/forward/grouped_convolution_forward_tile_ndhwgc_fp32_calls.inc"
#endif // DISABLE_IMPLICIT_GEMM_INSTANCES
    }
    else if constexpr(SIGNATURE == SIGNATURE_NGCHW_FP16_FWD)
    {
#ifndef DISABLE_IMPLICIT_GEMM_INSTANCES
#include "../../experimental/grouped_convolution_tile_instances/instances/forward/grouped_convolution_forward_tile_ngchw_depthwise_fp16_calls.inc"
#endif // DISABLE_IMPLICIT_GEMM_INSTANCES    
    }
    else if constexpr(SIGNATURE == SIGNATURE_NGCHW_BF16_FWD)
    {
#ifndef DISABLE_IMPLICIT_GEMM_INSTANCES
#include "../../experimental/grouped_convolution_tile_instances/instances/forward/grouped_convolution_forward_tile_ngchw_depthwise_bf16_calls.inc"
#endif // DISABLE_IMPLICIT_GEMM_INSTANCES
    }
    else if constexpr(SIGNATURE == SIGNATURE_NGCHW_FP32_FWD)
    {
#ifndef DISABLE_IMPLICIT_GEMM_INSTANCES
#include "../../experimental/grouped_convolution_tile_instances/instances/forward/grouped_convolution_forward_tile_ngchw_depthwise_fp32_calls.inc"
#endif // DISABLE_IMPLICIT_GEMM_INSTANCES 
    }
    else
    {
        std::cout << "Signature not supported" << std::endl;
        return std::make_tuple(false, best_avg_time, best_tflops, best_gbs, best_op_name, best_instance_index);
    }
    return std::make_tuple(valid, best_avg_time, best_tflops, best_gbs, best_op_name, best_instance_index);
}

} // namespace ck_tile::builder::profiling
