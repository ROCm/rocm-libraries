// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_operation/gpu/device/device_gemm_streamk.hpp"

template <typename ProblemType>
bool run_gemm(const ProblemType& problem_size, const ExecutionConfig& config)
{
    static_assert(ck::is_mx_type_t_v<ADataType> && ck::is_mx_type_t_v<BDataType>,
                  "this version only supports MX data type");
    using namespace ck::literals;

    auto M       = problem_size.M;
    auto N       = problem_size.N;
    auto K       = problem_size.K;
    auto StrideA = problem_size.StrideA;
    auto StrideB = problem_size.StrideB;
    auto StrideC = problem_size.StrideC;

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    auto f_get_default_stride =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if(stride == 0)
            {
                // give a chance if stride is zero, return a default packed stride
                if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
                {
                    return col;
                }
                else
                {
                    return row;
                }
            }
            else
                return stride;
        };

    StrideA = f_get_default_stride(M, K, StrideA, ALayout{});
    StrideB = f_get_default_stride(K, N, StrideB, BLayout{});
    StrideC = f_get_default_stride(M, N, StrideC, CLayout{});

    Tensor<ADataType::type_t> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType::type_t> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));
    // for each row in A matrix's M or column in B matrix's N, will need 2 VGPRs to store the
    // exp-scale vector
    const ck::index_t ScaleK = ck::math::integer_divide_ceil(K, 256) * 2;
    Tensor<int32_t> a_scale(f_host_tensor_descriptor(M, ScaleK, ScaleK, ALayout{}));
    Tensor<int32_t> b_scale(f_host_tensor_descriptor(ScaleK, N, ScaleK, BLayout{}));
    switch(config.init_method)
    {
    case 0:
        ck::utils::FillConstant<ADataType::type_t>{ck::type_convert<ADataType::type_t, float>(1.f)}(
            a_m_k);
        ck::utils::FillConstant<BDataType::type_t>{ck::type_convert<BDataType::type_t, float>(1.f)}(
            b_k_n);
        break;
    case 1:
        ck::utils::FillUniformDistributionIntegerValue<ADataType::type_t>{-5.f, 5.f}(a_m_k);
        ck::utils::FillUniformDistributionIntegerValue<BDataType::type_t>{-5.f, 5.f}(b_k_n);
        break;
    case 2:
        ck::utils::FillUniformDistribution<ADataType::type_t>{-1.f, 1.f}(a_m_k);
        ck::utils::FillUniformDistribution<BDataType::type_t>{-1.f, 1.f}(b_k_n);
        break;
    case 3:
        ck::utils::FillUniformDistributionIntegerValue<ADataType::type_t>{1.f, 1.f}(a_m_k);
        ck::utils::FillUniformDistributionIntegerValue<BDataType::type_t>{-5.f, 5.f}(b_k_n);
        break;
    case 4:
        ck::utils::FillUniformDistributionIntegerValue<ADataType::type_t>{-5.f, 5.f}(a_m_k);
        ck::utils::FillUniformDistributionIntegerValue<BDataType::type_t>{1.f, 1.f}(b_k_n);
        break;
    case 5:
        ck::utils::FillUniformDistributionIntegerValue<ADataType::type_t>{-2.f, 2.f}(a_m_k);
        ck::utils::FillUniformDistributionIntegerValue<BDataType::type_t>{-2.f, 2.f}(b_k_n);
        break;
    default:
        ck::utils::FillUniformDistribution<ADataType::type_t>{-0.1f, 0.1f}(a_m_k);
        ck::utils::FillUniformDistribution<BDataType::type_t>{-0.1f, 0.1f}(b_k_n);
    }
    ck::utils::FillConstant<int32_t>{0x7f7f7f7f}(a_scale);
    ck::utils::FillConstant<int32_t>{0x7f7f7f7f}(b_scale);

    // pack from 8bit to original MX data type
    std::vector<ADataType::type_t> a_m_k_packed_data = ADataType::compact_to_raw(a_m_k.mData);
    std::vector<ADataType::type_t> b_k_n_packed_data = BDataType::compact_to_raw(b_k_n.mData);

    Tensor<CDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<CDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;

    DeviceMem a_m_k_device_buf(sizeof(ADataType::type_t) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_k_n_device_buf(sizeof(BDataType::type_t) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpaceSize());
    DeviceMem a_scale_device_buf(sizeof(int32_t) * a_scale.mDesc.GetElementSpaceSize());
    DeviceMem b_scale_device_buf(sizeof(int32_t) * b_scale.mDesc.GetElementSpaceSize());

    a_m_k_device_buf.ToDevice(a_m_k_packed_data.data());
    b_k_n_device_buf.ToDevice(b_k_n_packed_data.data());
    a_scale_device_buf.ToDevice(a_scale.mData.data());
    b_scale_device_buf.ToDevice(b_scale.mData.data());

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};
    // do GEMM
    auto gemm      = DeviceGemmInstance{};
    auto invoker   = gemm.MakeInvoker();
    float ave_time = 0;

    auto argument =
        gemm.MakeArgument(static_cast<ADataType::type_t*>(a_m_k_device_buf.GetDeviceBuffer()),
                          static_cast<BDataType::type_t*>(b_k_n_device_buf.GetDeviceBuffer()),
                          static_cast<int32_t*>(a_scale_device_buf.GetDeviceBuffer()),
                          static_cast<int32_t*>(b_scale_device_buf.GetDeviceBuffer()),
                          static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
                          M,
                          N,
                          K,
                          StrideA,
                          StrideB,
                          StrideC,
                          a_element_op,
                          b_element_op,
                          c_element_op);

    if(!gemm.IsSupportedArgument(argument))
    {
        std::cerr << gemm.GetTypeString() << " does not support this problem" << std::endl;

        return true;
    }

    ave_time = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});

    std::size_t flop = 2_uz * M * N * K;
    std::size_t num_btype =
        (M * K * ADataType::BITS / 8) + (K * N * BDataType::BITS / 8) + sizeof(CDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << gemm.GetTypeString() << std::endl;

    if(config.do_verification)
    {
        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n, c_m_n_host_result, a_element_op, b_element_op, c_element_op);

        ref_invoker.Run(ref_argument);

        c_m_n_device_buf.FromDevice(c_m_n_device_result.mData.data());

        return ck::utils::check_err(c_m_n_device_result,
                                    c_m_n_host_result,
                                    "Error: Incorrect results!",
                                    get_rtol<CDataType>(),
                                    get_atol<CDataType>());
    }

    return true;
}

bool run_gemm_example(int argc, char* argv[])
{
    ProblemSize problem_size;
    ExecutionConfig config;

    return !parse_cmd_args(argc, argv, problem_size, config) || run_gemm(problem_size, config);
}

bool run_gemm_streamk_example(int argc, char* argv[])
{
    ProblemSizeStreamK problem_size;
    ExecutionConfig config;

    return !parse_cmd_args(argc, argv, problem_size, config) || run_gemm(problem_size, config);
}
