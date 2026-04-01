// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "flatmm_common.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"

enum class Metric
{
    LATENCY   = 0,
    TFLOPS    = 1,
    BANDWIDTH = 2
};

inline constexpr auto get_metric_name(Metric metric)
{
    switch(metric)
    {
    case Metric::LATENCY: return "latency";
    case Metric::TFLOPS: return "tflops";
    case Metric::BANDWIDTH: return "bandwidth";
    default: throw std::invalid_argument("Unsupported metric type");
    }
}

struct KernelConfig
{
    std::tuple<int, int, int> tile_dims;
    std::tuple<int, int, int> warp_dims;
    std::tuple<int, int, int> warp_tile_dims;
    bool permute_n;
};

struct FlatmmProblem
{
    int split_k_;
    int m_;
    int n_;
    int k_;
    int stride_a_;
    int stride_b_;
    int stride_c_;

    std::string dtype_a_;
    std::string dtype_b_;
    std::string dtype_acc_;
    std::string dtype_c_;
    std::string layout_a_;
    std::string layout_b_;
    std::string layout_c_;

    friend std::ostream& operator<<(std::ostream& os, const FlatmmProblem& problem)
    {
        os << "{\n"
           << "   \"split_k\": " << problem.split_k_ << ",\n"
           << "   \"m\": " << problem.m_ << ",\n"
           << "   \"n\": " << problem.n_ << ",\n"
           << "   \"k\": " << problem.k_ << ",\n"
           << "   \"stride_a\": " << problem.stride_a_ << ",\n"
           << "   \"stride_b\": " << problem.stride_b_ << ",\n"
           << "   \"stride_c\": " << problem.stride_c_ << ",\n"
           << "   \"dtype_a\": \"" << problem.dtype_a_ << "\",\n"
           << "   \"dtype_b\": \"" << problem.dtype_b_ << "\",\n"
           << "   \"dtype_acc\": \"" << problem.dtype_acc_ << "\",\n"
           << "   \"dtype_c\": \"" << problem.dtype_c_ << "\",\n"
           << "   \"layout_a\": \"" << problem.layout_a_ << "\",\n"
           << "   \"layout_b\": \"" << problem.layout_b_ << "\",\n"
           << "   \"layout_c\": \"" << problem.layout_c_ << "\"\n"
           << "}";
        return os;
    }
};

struct PerformanceResult
{
    double latency_;
    double tflops_;
    double bandwidth_;

    static bool compare(const PerformanceResult& lhs, const PerformanceResult& rhs, Metric metric)
    {
        switch(metric)
        {
        case Metric::LATENCY: return lhs.latency_ < rhs.latency_;
        case Metric::TFLOPS: return lhs.tflops_ > rhs.tflops_;
        case Metric::BANDWIDTH: return lhs.bandwidth_ > rhs.bandwidth_;
        default: throw std::invalid_argument("Unsupported metric type");
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const PerformanceResult& result)
    {
        os << "{\n"
           << "   \"latency(ms)\": " << std::fixed << std::setprecision(2) << result.latency_
           << ",\n"
           << "   \"tflops(TFlops)\": " << result.tflops_ << ",\n"
           << "   \"bandwidth(GB/s)\": " << result.bandwidth_ << "\n"
           << "}";
        return os;
    }
};

struct KernelInstance
{
    std::string name_;
    FlatmmProblem problem_;
    PerformanceResult perf_result_;

    static bool compare(const KernelInstance& lhs, const KernelInstance& rhs, Metric metric)
    {
        return PerformanceResult::compare(lhs.perf_result_, rhs.perf_result_, metric);
    }

    friend std::ostream& operator<<(std::ostream& os, const KernelInstance& instance)
    {
        os << "{\n"
           << " \"name\": \"" << instance.name_ << "\",\n"
           << " \"problem\": " << instance.problem_ << ",\n"
           << " \"perf_result\": " << instance.perf_result_ << "\n"
           << "}";
        return os;
    }
};

struct Setting
{
    int n_warmup_;
    int n_repeat_;
    bool is_gpu_timer_;
    int verify_;
    int init_method_;
    bool log_;
    std::string csv_filename_;
    bool flush_cache_;
    int rotating_count_;
    bool json_output_;
};

inline std::string get_rocm_version()
{
    std::ifstream version_file("/opt/rocm/.info/version");
    if(version_file.is_open())
    {
        std::string version;
        std::getline(version_file, version);
        return version;
    }
    return "Unknown";
}

template <typename AType, typename BType, typename AccType, typename CType>
auto calculate_rtol_atol(const ck_tile::index_t k,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeType = std::conditional_t<sizeof(AType) < sizeof(BType), AType, BType>;

    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CType, AccType>(
        ck_tile::integer_divide_ceil(k, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CType, AccType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(k, kbatch));
    const auto rtol_split_k = ck_tile::get_relative_threshold<CType, CType, CType>(kbatch);
    const auto atol_split_k =
        ck_tile::get_absolute_threshold<CType, CType, CType>(max_accumulated_value, kbatch);

    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

inline bool compare(std::string instance_name,
                    ck_tile::index_t k,
                    ck_tile::index_t kbatch,
                    ck_tile::HostTensor<CDataType>& device_result,
                    ck_tile::HostTensor<CDataType>& reference_result)
{
    const float max_accumulated_value =
        *std::max_element(reference_result.mData.begin(), reference_result.mData.end());
    const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
        k, kbatch, max_accumulated_value);

    const bool pass = ck_tile::check_err(device_result,
                                         reference_result,
                                         "Error: Incorrect results!",
                                         rtol_atol.at(ck_tile::number<0>{}),
                                         rtol_atol.at(ck_tile::number<1>{}));

    std::cout << "For " << instance_name << " relative error threshold is "
              << rtol_atol.at(ck_tile::number<0>{}) << " absolute error threshold is "
              << rtol_atol.at(ck_tile::number<1>{}) << std::endl;
    std::cout << "The verification result is: " << (pass ? "correct" : "fail") << std::endl;
    return pass;
}

inline void flatmm_host_reference(int verify,
                                  ck_tile::HostTensor<ADataType>& a_m_k,
                                  ck_tile::HostTensor<BDataType>& b_k_n,
                                  ck_tile::HostTensor<CDataType>& c_m_n_ref,
                                  ck_tile::DeviceMem& a_m_k_dev_buf,
                                  ck_tile::DeviceMem& b_k_n_dev_buf,
                                  ck_tile::index_t m,
                                  ck_tile::index_t n,
                                  ck_tile::index_t k,
                                  ck_tile::index_t stride_a,
                                  ck_tile::index_t stride_b,
                                  ck_tile::index_t stride_c)
{
    if(verify == 1)
    {
        ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_m_k, b_k_n, c_m_n_ref);
    }
    else if(verify == 2)
    {
        a_m_k_dev_buf.ToDevice(a_m_k.data());
        b_k_n_dev_buf.ToDevice(b_k_n.data());

        ck_tile::DeviceMem c_gpu_ref_buf(c_m_n_ref.get_element_space_size_in_bytes());
        c_gpu_ref_buf.SetZero();

        ADataType* d_a = static_cast<ADataType*>(a_m_k_dev_buf.GetDeviceBuffer());
        BDataType* d_b = static_cast<BDataType*>(b_k_n_dev_buf.GetDeviceBuffer());
        CDataType* d_c = static_cast<CDataType*>(c_gpu_ref_buf.GetDeviceBuffer());

        ck_tile::reference_gemm_gpu<ADataType,
                                    BDataType,
                                    AccDataType,
                                    CDataType,
                                    ALayout,
                                    BLayout,
                                    CLayout>(d_a, d_b, d_c, m, n, k, stride_a, stride_b, stride_c);

        c_gpu_ref_buf.FromDevice(c_m_n_ref.data());
    }
}

#pragma clang diagnostic pop
