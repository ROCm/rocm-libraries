// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

#include "batched_gemm_common.hpp"

#ifdef BATCHED_GEMM_SINGLE_INSTANCE_HPP
#include BATCHED_GEMM_SINGLE_INSTANCE_HPP
#endif

inline auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "1024", "The value for m dimension. Default is 1024.")
        .insert("n", "1024", "The value for n dimension. Default is 1024.")
        .insert("k", "1024", "The value for k dimension. Default is 1024.")
        .insert("stride_a", "0", "The stride value for tensor A. Default is 0.")
        .insert("stride_b", "0", "The stride value for tensor B. Default is 0.")
        .insert("stride_c", "0", "The stride value for tensor C. Default is 0.")
        .insert("batch_stride_a", "0", "Batch stride for tensor A. Default is 0.")
        .insert("batch_stride_b", "0", "Batch stride for tensor B. Default is 0.")
        .insert("batch_stride_c", "0", "Batch stride for tensor C. Default is 0.")
        .insert("batch_count", "8", "Batch count. Default is 8.")
        .insert("split_k", "1", "The split value for k dimension. Default is 1.")
        .insert("verify",
                "1",
                "Set to 0 for no validation, or 1 for validation on CPU. Default is 1.")
        .insert("log",
                "false",
                "Whether to output kernel information. Values: true or false.")
        .insert("warmup", "25", "The number of warmup iterations. Default is 25.")
        .insert("repeat", "50", "The number of benchmark iterations. Default is 50.")
        .insert("timer", "true", "Whether if the timer is GPU timer. Values: true or false.")
        .insert("init",
                "0",
                "Tensor initialization. 0: random, 1: linear, 2: constant(1). Default is 0.");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;

    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));

    const auto rtol_split_k =
        ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
        max_accumulated_value, kbatch);

    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

int benchmark_single(const ck_tile::ArgParser& arg_parser)
{
    const int m = arg_parser.get_int("m");
    const int n = arg_parser.get_int("n");
    const int k = arg_parser.get_int("k");

    int stride_a = arg_parser.get_int("stride_a");
    int stride_b = arg_parser.get_int("stride_b");
    int stride_c = arg_parser.get_int("stride_c");

    int batch_stride_a = arg_parser.get_int("batch_stride_a");
    int batch_stride_b = arg_parser.get_int("batch_stride_b");
    int batch_stride_c = arg_parser.get_int("batch_stride_c");

    const int batch_count = arg_parser.get_int("batch_count");
    const int split_k = arg_parser.get_int("split_k");

    const int verify = arg_parser.get_int("verify");
    const int init_method = arg_parser.get_int("init");

    const int warmup = arg_parser.get_int("warmup");
    const int repeat = arg_parser.get_int("repeat");
    const bool timer = arg_parser.get_bool("timer");
    const bool log = arg_parser.get_bool("log");

    const auto a_layout = ALayout{};
    const auto b_layout = BLayout{};
    const auto c_layout = CLayout{};

    stride_a = ck_tile::get_default_stride(m, k, stride_a, is_row_major(a_layout));
    stride_b = ck_tile::get_default_stride(k, n, stride_b, is_row_major(b_layout));
    stride_c = ck_tile::get_default_stride(m, n, stride_c, is_row_major(c_layout));

    if(batch_stride_a == 0)
    {
        batch_stride_a = m * k;
    }
    if(batch_stride_b == 0)
    {
        batch_stride_b = k * n;
    }
    if(batch_stride_c == 0)
    {
        batch_stride_c = m * n;
    }

    ck_tile::HostTensor<ADataType> a_b_m_k(make_batched_host_tensor_descriptor(
        batch_count, m, k, stride_a, batch_stride_a, a_layout));
    ck_tile::HostTensor<BDataType> b_b_k_n(make_batched_host_tensor_descriptor(
        batch_count, k, n, stride_b, batch_stride_b, b_layout));
    ck_tile::HostTensor<CDataType> c_b_m_n_dev(make_batched_host_tensor_descriptor(
        batch_count, m, n, stride_c, batch_stride_c, c_layout));

    if(init_method == 0)
    {
        ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_b_m_k);
        ck_tile::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_b_k_n);
    }
    else if(init_method == 1)
    {
        ck_tile::FillMonotonicSeq<ADataType>{}(a_b_m_k);
        ck_tile::FillMonotonicSeq<BDataType>{}(b_b_k_n);
    }
    else
    {
        ck_tile::FillConstant<ADataType>{static_cast<ADataType>(1)}(a_b_m_k);
        ck_tile::FillConstant<BDataType>{static_cast<BDataType>(1)}(b_b_k_n);
    }

    ck_tile::DeviceMem a_dev(a_b_m_k.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_dev(b_b_k_n.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_dev(c_b_m_n_dev.get_element_space_size_in_bytes());

    a_dev.ToDevice(a_b_m_k.data());
    b_dev.ToDevice(b_b_k_n.data());
    c_dev.SetZero();

    ck_tile::BatchedGemmHostArgs args{a_dev.GetDeviceBuffer(),
                                      b_dev.GetDeviceBuffer(),
                                      c_dev.GetDeviceBuffer(),
                                      split_k,
                                      m,
                                      n,
                                      k,
                                      stride_a,
                                      stride_b,
                                      stride_c,
                                      batch_stride_a,
                                      batch_stride_b,
                                      batch_stride_c,
                                      batch_count};

    float ave_time = 0.0f;
    try
    {
        ave_time = SelectedKernel::launch(
            args, ck_tile::stream_config{nullptr, true, log ? 1 : 0, warmup, repeat, timer});
    }
    catch(const std::exception& e)
    {
        std::cerr << "Kernel launch failed: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    c_dev.FromDevice(c_b_m_n_dev.data());

    std::size_t flop = std::size_t(2) * batch_count * m * n * k;
    std::size_t num_byte = sizeof(ADataType) * batch_count * m * k +
                           sizeof(BDataType) * batch_count * k * n +
                           sizeof(CDataType) * batch_count * m * n;

    const float tflops = static_cast<float>(flop) / 1.E9f / ave_time;
    const float gb_per_sec = static_cast<float>(num_byte) / 1.E6f / ave_time;

    std::cout << "Kernel: " << KERNEL_NAME << std::endl;
    std::cout << "M=" << m << " N=" << n << " K=" << k << " batch_count=" << batch_count
              << " split_k=" << split_k << " time=" << ave_time << " ms, " << tflops
              << " TFlops, " << gb_per_sec << " GB/s" << std::endl;

    if(verify)
    {
        ck_tile::HostTensor<CDataType> c_b_m_n_ref(make_batched_host_tensor_descriptor(
            batch_count, m, n, stride_c, batch_stride_c, c_layout));
        c_b_m_n_ref.SetZero();

        const auto b_b_n_k = b_b_k_n.transpose({0, 2, 1});
        ck_tile::reference_batched_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_b_m_k, b_b_n_k, c_b_m_n_ref);

        const float max_accumulated_value =
            *std::max_element(c_b_m_n_ref.mData.begin(), c_b_m_n_ref.mData.end());

        const auto rtol_atol =
            calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
                k, split_k, max_accumulated_value);

        const bool pass = ck_tile::check_err(c_b_m_n_dev,
                                             c_b_m_n_ref,
                                             "Error: Incorrect results!",
                                             rtol_atol.at(ck_tile::number<0>{}),
                                             rtol_atol.at(ck_tile::number<1>{}));

        std::cout << "Verification: " << (pass ? "PASS" : "FAIL") << std::endl;

        if(!pass)
        {
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}

int main(int argc, char* argv[])
{
    auto [result, parser] = create_args(argc, argv);
    if(!result)
    {
        return EXIT_FAILURE;
    }

    return benchmark_single(parser);
}
