// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <iostream>
#include <typeinfo>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_clamp.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_dynamic_op.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_convinvscale.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"
#include "ck/library/reference_tensor_operation/gpu/naive_conv_fwd_gpu.hpp"
#include "ck/library/utility/gpu_verification.hpp"

#include "grouped_convolution_forward_depthwise_dispatch.hpp"

namespace ck {
namespace profiler {

namespace depthwise_ref {

template <typename InDataType, typename WeiDataType, typename AccDataType, typename OutDataType>
void cpu_conv_fwd_ngchw(const InDataType* p_in,
                        const WeiDataType* p_wei,
                        OutDataType* p_out,
                        const ck::utils::conv::ConvParam& p)
{
    const auto G = p.G_, N = p.N_;
    const auto Hi = p.input_spatial_lengths_[0], Wi = p.input_spatial_lengths_[1];
    const auto Ho = p.output_spatial_lengths_[0], Wo = p.output_spatial_lengths_[1];
    const auto Y = p.filter_spatial_lengths_[0], X = p.filter_spatial_lengths_[1];
    const auto Sh = p.conv_filter_strides_[0], Sw = p.conv_filter_strides_[1];
    const auto Dh = p.conv_filter_dilations_[0], Dw = p.conv_filter_dilations_[1];
    const auto Ph = p.input_left_pads_[0], Pw = p.input_left_pads_[1];

    const long long in_g = Hi * Wi, in_n = G * in_g;
    const long long wei_g = Y * X;
    const long long out_g = Ho * Wo, out_n = G * out_g;

    for(long long n = 0; n < N; ++n)
    {
        for(long long g = 0; g < G; ++g)
        {
            for(long long ho = 0; ho < Ho; ++ho)
            {
                for(long long wo = 0; wo < Wo; ++wo)
                {
                    AccDataType acc = 0;
                    for(long long y = 0; y < Y; ++y)
                    {
                        for(long long x = 0; x < X; ++x)
                        {
                            long long hi = ho * Sh + y * Dh - Ph;
                            long long wi = wo * Sw + x * Dw - Pw;
                            if(hi >= 0 && hi < Hi && wi >= 0 && wi < Wi)
                            {
                                acc += static_cast<AccDataType>(
                                           p_in[n * in_n + g * in_g + hi * Wi + wi]) *
                                       static_cast<AccDataType>(p_wei[g * wei_g + y * X + x]);
                            }
                        }
                    }
                    p_out[n * out_n + g * out_g + ho * Wo + wo] = static_cast<OutDataType>(acc);
                }
            }
        }
    }
}

template <typename OutDataType>
bool verify(const OutDataType* p_gpu,
            const OutDataType* p_ref,
            std::size_t size,
            double rtol = 1e-3,
            double atol = 1e-3)
{
    std::size_t error_count = 0;
    double max_err          = 0.0;
    int printed             = 0;

    for(std::size_t i = 0; i < size; ++i)
    {
        double diff = std::abs(static_cast<double>(p_gpu[i]) - static_cast<double>(p_ref[i]));
        if(diff > max_err)
        {
            max_err = diff;
        }
        if(diff > atol + rtol * std::abs(static_cast<double>(p_ref[i])))
        {
            if(printed < 4)
            {
                std::cout << "\tout[" << i << "] != ref[" << i
                          << "]: " << static_cast<float>(p_gpu[i])
                          << " != " << static_cast<float>(p_ref[i]) << std::endl;
                printed++;
            }
            error_count++;
        }
    }
    if(error_count > 0)
    {
        std::cout << "max err: " << max_err << ", errors: " << error_count << " / " << size << " ("
                  << std::fixed << std::setprecision(2) << 100.0 * error_count / size << "%)"
                  << std::endl;
    }
    return error_count == 0;
}

} // namespace depthwise_ref

namespace fwd {
template <ck::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InElementOp,
          typename WeiElementOp,
          typename OutElementOp,
          typename ComputeTypeA,
          typename ComputeTypeB>
void print_instances()
{
    using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<NDimSpatial,
                                                                                   InLayout,
                                                                                   WeiLayout,
                                                                                   ck::Tuple<>,
                                                                                   OutLayout,
                                                                                   InDataType,
                                                                                   WeiDataType,
                                                                                   ck::Tuple<>,
                                                                                   OutDataType,
                                                                                   InElementOp,
                                                                                   WeiElementOp,
                                                                                   OutElementOp,
                                                                                   ComputeTypeA,
                                                                                   ComputeTypeB>;

    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    for(const auto& op_ptr : op_ptrs)
    {
#ifdef CK_EXPERIMENTAL_BUILDER
        const auto& instance_str = op_ptr->GetInstanceString();
        if(!instance_str.empty())
        {
            std::cout << instance_str << std::endl;
        }
        else
        {
            std::cout << op_ptr->GetTypeString() << std::endl;
        }
#else
        std::cout << op_ptr->GetTypeString() << std::endl;
#endif
    }
}
} // namespace fwd

template <ck::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename AComputeType = InDataType,
          typename BComputeType = AComputeType,
          typename IndexType    = ck::index_t,
          typename OutElementOp = ck::tensor_operation::element_wise::PassThrough>
bool profile_grouped_conv_fwd_impl(int do_verification,
                                   int init_method,
                                   bool do_log,
                                   bool time_kernel,
                                   const ck::utils::conv::ConvParam& conv_param,
                                   const OutElementOp out_element_op = OutElementOp{},
                                   index_t instance_index            = -1,
                                   bool list_instances               = false)
{
    using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
    using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;

    const auto in_element_op  = InElementOp{};
    const auto wei_element_op = WeiElementOp{};

    const auto in_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);

    const auto wei_g_k_c_xs_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);

    const auto out_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    std::array<IndexType, NDimSpatial + 3> a_g_n_c_wis_lengths{};
    std::array<IndexType, NDimSpatial + 3> a_g_n_c_wis_strides{};
    std::array<IndexType, NDimSpatial + 3> b_g_k_c_xs_lengths{};
    std::array<IndexType, NDimSpatial + 3> b_g_k_c_xs_strides{};
    std::array<IndexType, NDimSpatial + 3> e_g_n_k_wos_lengths{};
    std::array<IndexType, NDimSpatial + 3> e_g_n_k_wos_strides{};
    std::array<IndexType, NDimSpatial> conv_filter_strides{};
    std::array<IndexType, NDimSpatial> conv_filter_dilations{};
    std::array<IndexType, NDimSpatial> input_left_pads{};
    std::array<IndexType, NDimSpatial> input_right_pads{};

    auto copy = [](const auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

    copy(in_g_n_c_wis_desc.GetLengths(), a_g_n_c_wis_lengths);
    copy(in_g_n_c_wis_desc.GetStrides(), a_g_n_c_wis_strides);
    copy(wei_g_k_c_xs_desc.GetLengths(), b_g_k_c_xs_lengths);
    copy(wei_g_k_c_xs_desc.GetStrides(), b_g_k_c_xs_strides);
    copy(out_g_n_k_wos_desc.GetLengths(), e_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_desc.GetStrides(), e_g_n_k_wos_strides);
    copy(conv_param.conv_filter_strides_, conv_filter_strides);
    copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_param.input_left_pads_, input_left_pads);
    copy(conv_param.input_right_pads_, input_right_pads);

    std::cout << "input: " << in_g_n_c_wis_desc << std::endl;
    std::cout << "weight: " << wei_g_k_c_xs_desc << std::endl;
    std::cout << "output: " << out_g_n_k_wos_desc << std::endl;

    using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<NDimSpatial,
                                                                                   InLayout,
                                                                                   WeiLayout,
                                                                                   ck::Tuple<>,
                                                                                   OutLayout,
                                                                                   InDataType,
                                                                                   WeiDataType,
                                                                                   ck::Tuple<>,
                                                                                   OutDataType,
                                                                                   InElementOp,
                                                                                   WeiElementOp,
                                                                                   OutElementOp,
                                                                                   AComputeType,
                                                                                   BComputeType>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    index_t total_instances = static_cast<index_t>(op_ptrs.size());

    // Create host tensors
    Tensor<InDataType> input(in_g_n_c_wis_desc);
    Tensor<WeiDataType> weight(wei_g_k_c_xs_desc);
    Tensor<OutDataType> host_output(out_g_n_k_wos_desc);
    Tensor<OutDataType> device_output(out_g_n_k_wos_desc);

    // Get element space sizes for allocation
    const auto input_size  = in_g_n_c_wis_desc.GetElementSpaceSize();
    const auto weight_size = wei_g_k_c_xs_desc.GetElementSpaceSize();
    const auto output_size = out_g_n_k_wos_desc.GetElementSpaceSize();

    // Allocate GPU memory
    DeviceMem in_device_buf(sizeof(InDataType) * input_size);
    DeviceMem wei_device_buf(sizeof(WeiDataType) * weight_size);
    DeviceMem out_device_buf(sizeof(OutDataType) * output_size);

    // Don't create reference if we're only listing instances
    if(list_instances)
        do_verification = 0;

    // Initialize tensors based on do_verification:
    // - do_verification=2: GPU-side initialization
    // - do_verification=0,1: CPU-side initialization
    if(do_verification == 2)
    {
        // GPU-side initialization for GPU verification workflow
        switch(init_method)
        {
        case 0:
            // Zero initialization
            in_device_buf.SetZero();
            wei_device_buf.SetZero();
            break;
        case 1:
            // Discrete integer generation: {-5, -4, -3, ..., 3, 4}
            in_device_buf.FillUniformRandInteger<InDataType>(-5, 5);
            wei_device_buf.FillUniformRandInteger<WeiDataType>(-5, 5);
            break;
        default:
            // Continuous float generation
            in_device_buf.FillUniformRandFp<InDataType>(0.0f, 1.0f);
            wei_device_buf.FillUniformRandFp<WeiDataType>(-0.5f, 0.5f);
        }
    }
    else
    {
        // CPU-side initialization for do_verification=0,1
        switch(init_method)
        {
        case 0: break; // Tensors are already zero-initialized by default
        case 1:
            input.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
            weight.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
            break;
        default:
            input.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
            weight.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
        }

        // Copy initialized host data to device
        in_device_buf.ToDevice(input.mData.data());
        wei_device_buf.ToDevice(weight.mData.data());
    }

    // Allocate GPU reference buffer (used only if do_verification == 2)
    DeviceMem gpu_ref_out_buf(
        do_verification == 2 ? sizeof(OutDataType) * device_output.mDesc.GetElementSpaceSize() : 0);

    // run reference op
    if(do_verification == 2)
    {
        // Use GPU reference with GPU verification
        std::cout << "Using GPU reference with GPU verification" << std::endl;

        // Call GPU reference with ConvParam directly
        ref::naive_conv_fwd<InLayout,
                            WeiLayout,
                            OutLayout,
                            InDataType,
                            WeiDataType,
                            OutDataType,
                            InElementOp,
                            WeiElementOp,
                            OutElementOp>(
            reinterpret_cast<const InDataType*>(in_device_buf.GetDeviceBuffer()),
            reinterpret_cast<const WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
            reinterpret_cast<OutDataType*>(gpu_ref_out_buf.GetDeviceBuffer()),
            conv_param,
            in_element_op,
            wei_element_op,
            out_element_op);
    }
    else if(do_verification == 1)
    {
        // Use CPU reference for verification (default)
        auto ref_conv = ck::tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                     InDataType,
                                                                     WeiDataType,
                                                                     OutDataType,
                                                                     InElementOp,
                                                                     WeiElementOp,
                                                                     OutElementOp>{};

        auto ref_invoker  = ref_conv.MakeInvoker();
        auto ref_argument = ref_conv.MakeArgument(input,
                                                  weight,
                                                  host_output,
                                                  conv_param.conv_filter_strides_,
                                                  conv_param.conv_filter_dilations_,
                                                  conv_param.input_left_pads_,
                                                  conv_param.input_right_pads_,
                                                  in_element_op,
                                                  wei_element_op,
                                                  out_element_op);

        // init host output to zero
        host_output.SetZero();

        ref_invoker.Run(ref_argument);
    }

    std::string best_op_name;
    float best_avg_time         = 0;
    float best_tflops           = 0;
    float best_gb_per_sec       = 0;
    index_t num_kernel          = 0;
    index_t valid_instances     = 0;
    index_t best_instance_index = 0;

    // profile device op instances
    bool pass               = true;
    bool dummy_run_executed = false;

    auto run_impl = [&](auto& op_ptr, auto& argument_ptr) {
        // workspace_sz will be equal to 0 for other layout than NGCHW
        const std::size_t workspace_sz = op_ptr->GetWorkSpaceSize(argument_ptr.get());
        DeviceMem workspace_dev(workspace_sz);
        op_ptr->SetWorkSpacePointer(argument_ptr.get(), workspace_dev.GetDeviceBuffer());

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            num_kernel++;

            // List instances mode - just print and continue
            if(list_instances)
            {
                std::cout << "[" << (num_kernel - 1) << "] " << op_ptr->GetTypeString()
                          << std::endl;
                return;
            }

            // Skip if a specific instance was requested and this isn't it
            const bool running_specific_instance = (instance_index != -1);
            const bool current_is_target         = (num_kernel - 1 == instance_index);
            if(running_specific_instance && !current_is_target)
            {
                return;
            }

            std::string op_name = op_ptr->GetTypeString();
            valid_instances++;

            out_device_buf.SetZero();

            auto invoker_ptr = op_ptr->MakeInvokerPointer();

            // Run first instance twice to get proper time
            if(time_kernel && !dummy_run_executed)
            {
                invoker_ptr->Run(argument_ptr.get(),
                                 StreamConfig{nullptr,
                                              time_kernel,
                                              0 /*log_level*/,
                                              5 /*cold_iters*/,
                                              50 /*nrepeat_*/,
                                              time_kernel /*flush_cache*/});
                dummy_run_executed = true;
            }

            float avg_time = invoker_ptr->Run(argument_ptr.get(),
                                              StreamConfig{nullptr,
                                                           time_kernel,
                                                           0 /*log_level*/,
                                                           5 /*cold_iters*/,
                                                           50 /*nrepeat_*/,
                                                           time_kernel /*flush_cache*/});

            std::size_t flop      = conv_param.GetFlops();
            std::size_t num_btype = conv_param.GetByte<InDataType, WeiDataType, OutDataType>();

            float tflops = static_cast<float>(flop) / 1.E9 / avg_time;

            float gb_per_sec = num_btype / 1.E6 / avg_time;

            std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << tflops << " TFlops, "
                      << gb_per_sec << " GB/s, " << op_name << std::endl;

            if(tflops > best_tflops)
            {
                best_op_name        = op_name;
                best_tflops         = tflops;
                best_avg_time       = avg_time;
                best_gb_per_sec     = gb_per_sec;
                best_instance_index = num_kernel - 1;
            }

            // Synchronize before verification to ensure kernel has completed
            if(do_verification > 0 && !time_kernel)
            {
                hip_check_error(hipStreamSynchronize(nullptr));
            }

            if(do_verification == 2)
            {
                // GPU verification path
                // Calculate number of accumulations (C * filter spatial dimensions)
                std::size_t filter_spatial_size = 1;
                for(auto len : conv_param.filter_spatial_lengths_)
                {
                    filter_spatial_size *= len;
                }
                const int num_accums = static_cast<int>(conv_param.C_ * filter_spatial_size);

                // Perform GPU verification (max value computed internally on GPU)
                const std::size_t tensor_size = device_output.mDesc.GetElementSpaceSize();
                auto gpu_result = ck::profiler::gpu_verify<OutDataType, AComputeType, OutDataType>(
                    out_device_buf.GetDeviceBuffer(),
                    gpu_ref_out_buf.GetDeviceBuffer(),
                    num_accums,
                    tensor_size);

                if(!gpu_result)
                {
                    // GPU verification failed - print detailed error summary
                    gpu_result.print_error_summary();
                    pass = false;

                    if(do_log)
                    {
                        // Copy buffers to host for logging
                        out_device_buf.FromDevice(device_output.mData.data());
                        gpu_ref_out_buf.FromDevice(host_output.mData.data());

                        LogRangeAsType<float>(std::cout << "input : ", input.mData, ",")
                            << std::endl;
                        LogRangeAsType<float>(std::cout << "weight: ", weight.mData, ",")
                            << std::endl;
                        LogRangeAsType<float>(
                            std::cout << "host_output  : ", host_output.mData, ",")
                            << std::endl;
                        LogRangeAsType<float>(
                            std::cout << "device_output: ", device_output.mData, ",")
                            << std::endl;
                    }
                }
            }
            else if(do_verification == 1)
            {
                // CPU verification path (original behavior)
                out_device_buf.FromDevice(device_output.mData.data());

                pass = pass & ck::utils::check_err(device_output, host_output);

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "input : ", input.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "weight: ", weight.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "host_output  : ", host_output.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "device_output: ", device_output.mData, ",")
                        << std::endl;
                }
            }
        }
        else if(list_instances || instance_index == -1)
        {
            std::cout << op_ptr->GetTypeString() << " does not support this problem" << std::endl;
        }
    };

    if(list_instances)
    {
        std::cout << "\nValid instances for this problem:" << std::endl;
    }

    for(auto& op_ptr : op_ptrs)
    {
        auto argument_ptr = op_ptr->MakeArgumentPointer(in_device_buf.GetDeviceBuffer(),
                                                        wei_device_buf.GetDeviceBuffer(),
                                                        {},
                                                        out_device_buf.GetDeviceBuffer(),
                                                        a_g_n_c_wis_lengths,
                                                        a_g_n_c_wis_strides,
                                                        b_g_k_c_xs_lengths,
                                                        b_g_k_c_xs_strides,
                                                        {},
                                                        {},
                                                        e_g_n_k_wos_lengths,
                                                        e_g_n_k_wos_strides,
                                                        conv_filter_strides,
                                                        conv_filter_dilations,
                                                        input_left_pads,
                                                        input_right_pads,
                                                        in_element_op,
                                                        wei_element_op,
                                                        out_element_op);

        run_impl(op_ptr, argument_ptr);
    }

    // Depthwise conv: NGCHW tensors + CPU verification
    if constexpr(NDimSpatial == 2)
    {
        constexpr bool is_dw_dtype =
            std::is_same_v<InDataType, ck::half_t> || std::is_same_v<InDataType, float>;

        if constexpr(is_dw_dtype)
        {
            if(conv_param.C_ == 1 && conv_param.K_ == 1)
            {
                const auto G  = conv_param.G_;
                const auto N  = conv_param.N_;
                const auto Hi = conv_param.input_spatial_lengths_[0];
                const auto Wi = conv_param.input_spatial_lengths_[1];
                const auto Ho = conv_param.output_spatial_lengths_[0];
                const auto Wo = conv_param.output_spatial_lengths_[1];
                const auto Y  = conv_param.filter_spatial_lengths_[0];
                const auto X  = conv_param.filter_spatial_lengths_[1];

                const std::size_t dw_in_sz  = static_cast<std::size_t>(N) * G * Hi * Wi;
                const std::size_t dw_wei_sz = static_cast<std::size_t>(G) * Y * X;
                const std::size_t dw_out_sz = static_cast<std::size_t>(N) * G * Ho * Wo;

                DeviceMem dw_in_dev(sizeof(InDataType) * dw_in_sz);
                DeviceMem dw_wei_dev(sizeof(WeiDataType) * dw_wei_sz);
                DeviceMem dw_out_dev(sizeof(OutDataType) * dw_out_sz);

                switch(init_method)
                {
                case 0:
                    dw_in_dev.SetZero();
                    dw_wei_dev.SetZero();
                    break;
                case 1:
                    dw_in_dev.FillUniformRandInteger<InDataType>(-5, 5);
                    dw_wei_dev.FillUniformRandInteger<WeiDataType>(-5, 5);
                    break;
                default:
                    dw_in_dev.FillUniformRandFp<InDataType>(0.0f, 1.0f);
                    dw_wei_dev.FillUniformRandFp<WeiDataType>(-0.5f, 0.5f);
                }

                // GPU reference doesn't support NGCHW; use CPU (do_verification=2 → 1)
                const int dw_verify = (do_verification == 2) ? 1 : do_verification;

                std::vector<OutDataType> dw_ref;
                if(dw_verify == 1)
                {
                    dw_ref.assign(dw_out_sz, OutDataType{0});

                    std::vector<InDataType> dw_in_host(dw_in_sz);
                    std::vector<WeiDataType> dw_wei_host(dw_wei_sz);
                    dw_in_dev.FromDevice(dw_in_host.data());
                    dw_wei_dev.FromDevice(dw_wei_host.data());

                    depthwise_ref::cpu_conv_fwd_ngchw<InDataType, WeiDataType, float, OutDataType>(
                        dw_in_host.data(), dw_wei_host.data(), dw_ref.data(), conv_param);
                }

                ck_tile::conv::ConvParam dw_param(conv_param.num_dim_spatial_,
                                                  conv_param.G_,
                                                  conv_param.N_,
                                                  conv_param.K_,
                                                  conv_param.C_,
                                                  conv_param.filter_spatial_lengths_,
                                                  conv_param.input_spatial_lengths_,
                                                  conv_param.conv_filter_strides_,
                                                  conv_param.conv_filter_dilations_,
                                                  conv_param.input_left_pads_,
                                                  conv_param.input_right_pads_);

                using TileInType  = std::conditional_t<std::is_same_v<InDataType, ck::half_t>,
                                                       ck_tile::half_t,
                                                       InDataType>;
                using TileWeiType = std::conditional_t<std::is_same_v<WeiDataType, ck::half_t>,
                                                       ck_tile::half_t,
                                                       WeiDataType>;
                using TileOutType = std::conditional_t<std::is_same_v<OutDataType, ck::half_t>,
                                                       ck_tile::half_t,
                                                       OutDataType>;

                ck_tile::GroupedConvFwdHostArgs<> dw_host_args(dw_param,
                                                               dw_in_dev.GetDeviceBuffer(),
                                                               dw_wei_dev.GetDeviceBuffer(),
                                                               {},
                                                               dw_out_dev.GetDeviceBuffer(),
                                                               1);

                ck_tile::stream_config dw_stream{nullptr, time_kernel, 1, 5, 50, time_kernel};
                const std::size_t flop = conv_param.GetFlops();
                const std::size_t num_btype =
                    conv_param.GetByte<InDataType, WeiDataType, OutDataType>();

                auto dw_result = ck_tile::grouped_conv_fwd_depthwise_dispatch<TileInType,
                                                                              TileWeiType,
                                                                              float,
                                                                              TileOutType>(
                    dw_host_args, dw_stream, flop, num_btype);

                total_instances += dw_result.total_count;
                valid_instances += dw_result.valid_count;
                num_kernel += dw_result.total_count;

                bool dw_pass = true;
                if(dw_verify == 1 && dw_result.valid_count > 0)
                {
                    std::vector<OutDataType> dw_out_host(dw_out_sz);
                    dw_out_dev.FromDevice(dw_out_host.data());

                    dw_pass = depthwise_ref::verify(dw_out_host.data(), dw_ref.data(), dw_out_sz);
                    if(!dw_pass)
                    {
                        pass = false;
                    }

                    std::cout << "Depthwise verification: " << (dw_pass ? "PASSED" : "FAILED")
                              << std::endl;
                }

                if(dw_pass && dw_result.best_instance_idx >= 0 &&
                   dw_result.best_tflops > best_tflops)
                {
                    best_op_name        = dw_result.best_config;
                    best_avg_time       = dw_result.best_time;
                    best_tflops         = dw_result.best_tflops;
                    best_gb_per_sec     = dw_result.best_gb_per_sec;
                    best_instance_index = dw_result.best_instance_idx;
                }
            }
        }
    }

    std::cout << "ckProfiler found " << total_instances << " instances" << std::endl;

    if(list_instances)
    {
        std::cout << "\nTotal: " << num_kernel << " valid instances" << std::endl;
        return true;
    }

    printf("\033[36mvalids: %ld\033[0m\n", static_cast<long>(valid_instances));

    if(instance_index != -1 && valid_instances == 0)
    {
        std::cerr << "Error: instance_index " << instance_index
                  << " exceeds the number of valid instances (" << num_kernel << ")" << std::endl;
        return false;
    }

    std::cout << "Best configuration parameters:" << "\nname: " << best_op_name << " (instance "
              << best_instance_index << ")" << "\navg_time: " << best_avg_time
              << "\ntflops: " << best_tflops << "\nGB/s: " << best_gb_per_sec << std::endl;
    if(instance_index != -1)
    {
        std::cout << "grouped_conv_fwd_instance (" << instance_index << "/" << num_kernel
                  << "): Passed" << std::endl;
    }
    return pass;
}

} // namespace profiler
} // namespace ck
