// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <stdexcept>
#include <vector>
#include <array>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/reference/reference_gemm.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_multi_abd_kernel.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

#include "gemm_multi_abd_common.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-seggestions"

enum class Metric
{
    LATENCY   = 0,
    TFLOPS    = 1,
    BANDWIDTH = 2
};

inline constexpr auto get_metric_name(Metric m)
{
    switch(m)
    {
    case Metric::LATENCY: return "latency";
    case Metric::TFLOPS: return "tflops";
    case Metric::BANDWIDTH: return "bandwidth";
    default: throw std::invalid_argument("Unsupported metric type");
    }
}

struct GemmMultiABDProblem
{
    int split_k_;
    int m_, n_, k_;
    std::vector<int> stride_as_;
    std::vector<int> stride_bs_;
    std::vector<int> stride_ds_;
    int stride_e_;
    std::vector<std::string> dtype_as_;
    std::vector<std::string> dtype_bs_;
    std::vector<std::string> dtype_ds_;
    std::string dtype_acc_, dtype_e_;
    std::vector<std::string> layout_as_;
    std::vector<std::string> layout_bs_;
    std::vector<std::string> layout_ds_;
    std::string layout_e_;
    std::string a_elementwise_;
    std::string b_elementwise_;
    std::string cde_elementwise_;

    friend std::ostream& operator<<(std::ostream& os, const GemmMultiABDProblem& p)
    {
        os << "{\n"
           << "   \"split_k\":" << p.split_k_ << ",\n"
           << "   \"m\":" << p.m_ << ",\n"
           << "   \"n\":" << p.n_ << ",\n"
           << "   \"k\":" << p.k_ << ",\n";
        for(std::size_t i = 0; i < p.stride_as_.size(); i++)
            os << "   \"stride_a" << i << "\":" << p.stride_as_[i] << ",\n";
        for(std::size_t i = 0; i < p.stride_bs_.size(); i++)
            os << "   \"stride_b" << i << "\":" << p.stride_bs_[i] << ",\n";
        for(std::size_t i = 0; i < p.stride_ds_.size(); i++)
            os << "   \"stride_d" << i << "\":" << p.stride_ds_[i] << ",\n";
        os << "   \"stride_e\":" << p.stride_e_ << ",\n";
        for(std::size_t i = 0; i < p.dtype_as_.size(); i++)
            os << "   \"dtype_a" << i << "\":\"" << p.dtype_as_[i] << "\",\n";
        for(std::size_t i = 0; i < p.dtype_bs_.size(); i++)
            os << "   \"dtype_b" << i << "\":\"" << p.dtype_bs_[i] << "\",\n";
        for(std::size_t i = 0; i < p.dtype_ds_.size(); i++)
            os << "   \"dtype_d" << i << "\":\"" << p.dtype_ds_[i] << "\",\n";
        os << "   \"dtype_acc\":\"" << p.dtype_acc_ << "\",\n"
           << "   \"dtype_e\":\"" << p.dtype_e_ << "\",\n";
        for(std::size_t i = 0; i < p.layout_as_.size(); i++)
            os << "   \"layout_a" << i << "\":\"" << p.layout_as_[i] << "\",\n";
        for(std::size_t i = 0; i < p.layout_bs_.size(); i++)
            os << "   \"layout_b" << i << "\":\"" << p.layout_bs_[i] << "\",\n";
        for(std::size_t i = 0; i < p.layout_ds_.size(); i++)
            os << "   \"layout_d" << i << "\":\"" << p.layout_ds_[i] << "\",\n";
        os << "   \"layout_e\":\"" << p.layout_e_ << "\",\n"
           << "   \"a_elementwise\":\"" << p.a_elementwise_ << "\",\n"
           << "   \"b_elementwise\":\"" << p.b_elementwise_ << "\",\n"
           << "   \"cde_elementwise\":\"" << p.cde_elementwise_ << "\"\n"
           << "}";
        return os;
    }
};

struct PerformanceResult
{
    double latency_;
    double tflops_;
    double bandwidth_;

    static bool compare(const PerformanceResult& a, const PerformanceResult& b, Metric m)
    {
        switch(m)
        {
        case Metric::LATENCY: return a.latency_ < b.latency_;
        case Metric::TFLOPS: return a.tflops_ > b.tflops_;
        case Metric::BANDWIDTH: return a.bandwidth_ > b.bandwidth_;
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
    GemmMultiABDProblem problem_;
    PerformanceResult perf_result_;

    static bool compare(const KernelInstance& a, const KernelInstance& b, Metric m)
    {
        return PerformanceResult::compare(a.perf_result_, b.perf_result_, m);
    }

    friend std::ostream& operator<<(std::ostream& os, const KernelInstance& obj)
    {
        os << "{\n"
           << " \"name\": \"" << obj.name_ << "\",\n"
           << " \"problem\": " << obj.problem_ << ",\n"
           << " \"perf_result\": " << obj.perf_result_ << "\n"
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

template <typename AType, typename BType, typename DType, typename AccType, typename EType>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeTypeAB =
        std::conditional_t<sizeof(AType) < sizeof(BType), AType, BType>;

    using ComputeType =
        std::conditional_t<sizeof(ComputeTypeAB) < sizeof(DType), ComputeTypeAB, DType>;

    const auto rtol = ck_tile::get_relative_threshold<ComputeType, EType, AccType>(
        ck_tile::integer_divide_ceil(K, kbatch));

    const auto atol = ck_tile::get_absolute_threshold<ComputeType, EType, AccType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));

    const auto rtol_split_k =
        ck_tile::get_relative_threshold<EType, EType, EType>(kbatch);

    const auto atol_split_k = ck_tile::get_absolute_threshold<EType, EType, EType>(
        max_accumulated_value, kbatch);

    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

/// @brief Compare device and host results
bool compare(std::string instanceName,
             ck_tile::index_t K,
             ck_tile::index_t kbatch,
             ck_tile::HostTensor<EDataType>& e_m_n_dev_result,
             ck_tile::HostTensor<EDataType>& e_m_n_host_result)
{
    const float max_accumulated_value =
        *std::max_element(e_m_n_host_result.mData.begin(), e_m_n_host_result.mData.end());

    const auto rtol_atol =
        calculate_rtol_atol<ABaseDataType, BBaseDataType, DBaseDataType, AccDataType, EDataType>(
            K, kbatch, max_accumulated_value);

    bool pass = ck_tile::check_err(e_m_n_dev_result,
                                   e_m_n_host_result,
                                   "Error: Incorrect results!",
                                   rtol_atol.at(ck_tile::number<0>{}),
                                   rtol_atol.at(ck_tile::number<1>{}));

    std::cout << "For " << instanceName << " Relative error threshold is "
              << rtol_atol.at(ck_tile::number<0>{}) << " Absolute error threshold is "
              << rtol_atol.at(ck_tile::number<1>{}) << std::endl;
    std::cout << "The verification result is:" << (pass ? "correct" : "fail") << std::endl;

    return pass;
}

/// @brief Host reference computation
template <std::size_t NumA, std::size_t NumB, std::size_t NumD>
void gemm_multi_abd_host_reference(
    int verify,
    const std::array<ck_tile::HostTensor<ABaseDataType>, NumA>& as_tensors,
    const std::array<ck_tile::HostTensor<BBaseDataType>, NumB>& bs_tensors,
    const std::array<ck_tile::HostTensor<DBaseDataType>, NumD>& ds_tensors,
    ck_tile::HostTensor<EDataType>& e_m_n_host_result)
{
    if(verify > 0)
    {
        ck_tile::index_t M = as_tensors[0].get_length(0);
        ck_tile::index_t K = as_tensors[0].get_length(1);
        ck_tile::index_t N = bs_tensors[0].get_length(1);

        ck_tile::HostTensor<ABaseDataType> a_m_k({M, K});
        ck_tile::HostTensor<BBaseDataType> b_k_n({K, N});

        ck_tile::reference_gemm_multiple_abd<
            AsDataType, BsDataType, DsDataType,
            AccDataType, EDataType, 
            AElementWiseFn, BElementWiseFn, CDEElementWiseFn,
            ABaseDataType, BBaseDataType, DBaseDataType>(
            as_tensors, bs_tensors, ds_tensors,
            a_m_k, b_k_n, e_m_n_host_result);
    }
}

#pragma clang diagnostic pop
