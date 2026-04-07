// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <vector>
#include <sstream>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "batched_contraction_common.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-seggestions"

// Data types and Layouts are defined by the generated kernel headers
// No hardcoded type definitions here to avoid conflicts

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

// Helper to parse comma-separated dimension strings
inline std::vector<ck_tile::index_t> parse_dims_string(const std::string& dims_str)
{
    std::vector<ck_tile::index_t> dims;
    if(dims_str.empty())
        return dims;
    std::stringstream ss(dims_str);
    std::string token;
    while(std::getline(ss, token, ','))
    {
        dims.push_back(std::stoi(token));
    }
    return dims;
}

// Helper to calculate total elements from dimension vector
inline ck_tile::index_t calculate_total(const std::vector<ck_tile::index_t>& dims)
{
    ck_tile::index_t total = 1;
    for(auto d : dims)
    {
        total *= d;
    }
    return total;
}

// Helper to concatenate dimension vectors
inline std::vector<ck_tile::index_t>
concatenate_dims(const std::vector<std::vector<ck_tile::index_t>>& dim_components)
{
    std::vector<ck_tile::index_t> result;
    for(const auto& component : dim_components)
    {
        result.insert(result.end(), component.begin(), component.end());
    }
    return result;
}

// Helper to format a dimension vector as a string
inline std::string dims_to_string(const std::vector<ck_tile::index_t>& dims)
{
    std::string result;
    for(size_t i = 0; i < dims.size(); ++i)
    {
        if(i > 0)
            result += ",";
        result += std::to_string(dims[i]);
    }
    return result;
}

struct BatchedContractionProblem
{
    int split_k_;
    std::vector<ck_tile::index_t> g_dims_;
    std::vector<ck_tile::index_t> m_dims_;
    std::vector<ck_tile::index_t> n_dims_;
    std::vector<ck_tile::index_t> k_dims_;
    int num_d_tensors_;

    std::string dtype_a_, dtype_b_, dtype_d_, dtype_acc_, dtype_e_;
    std::string layout_a_, layout_b_, layout_e_;

    // Derived totals
    ck_tile::index_t G_total() const { return calculate_total(g_dims_); }
    ck_tile::index_t M_total() const { return calculate_total(m_dims_); }
    ck_tile::index_t N_total() const { return calculate_total(n_dims_); }
    ck_tile::index_t K_total() const { return calculate_total(k_dims_); }

    friend std::ostream& operator<<(std::ostream& os, const BatchedContractionProblem& problem)
    {
        os << "{\n"
           << "   \"split_k\":" << problem.split_k_ << ",\n"
           << "   \"g_dims\":\"" << dims_to_string(problem.g_dims_) << "\",\n"
           << "   \"m_dims\":\"" << dims_to_string(problem.m_dims_) << "\",\n"
           << "   \"n_dims\":\"" << dims_to_string(problem.n_dims_) << "\",\n"
           << "   \"k_dims\":\"" << dims_to_string(problem.k_dims_) << "\",\n"
           << "   \"G_total\":" << problem.G_total() << ",\n"
           << "   \"M_total\":" << problem.M_total() << ",\n"
           << "   \"N_total\":" << problem.N_total() << ",\n"
           << "   \"K_total\":" << problem.K_total() << ",\n"
           << "   \"num_d_tensors\":" << problem.num_d_tensors_ << ",\n"
           << "   \"dtype_a\":\"" << problem.dtype_a_ << "\",\n"
           << "   \"dtype_b\":\"" << problem.dtype_b_ << "\",\n"
           << "   \"dtype_d\":\"" << problem.dtype_d_ << "\",\n"
           << "   \"dtype_acc\":\"" << problem.dtype_acc_ << "\",\n"
           << "   \"dtype_e\":\"" << problem.dtype_e_ << "\",\n"
           << "   \"layout_a\":\"" << problem.layout_a_ << "\",\n"
           << "   \"layout_b\":\"" << problem.layout_b_ << "\",\n"
           << "   \"layout_e\":\"" << problem.layout_e_ << "\"\n"
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
    BatchedContractionProblem problem_;
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

template <typename ADataType, typename BDataType, typename AccDataType, typename EDataType>
auto batched_contraction_calculate_rtol_atol(const ck_tile::index_t K,
                                             const ck_tile::index_t kbatch,
                                             const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;

    // Calculate thresholds
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, EDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));

    const auto atol = ck_tile::get_absolute_threshold<ComputeType, EDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));

    // Calculate error due to split_k accumulation
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<EDataType, EDataType, EDataType>(kbatch);

    const auto atol_split_k = ck_tile::get_absolute_threshold<EDataType, EDataType, EDataType>(
        max_accumulated_value, kbatch);

    // Use higher threshold
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

/// @brief Function to compare the results of the device and host computations
template <typename ADataType, typename BDataType, typename AccDataType, typename EDataType>
bool batched_contraction_compare(std::string instanceName,
                                 ck_tile::index_t K,
                                 ck_tile::index_t kbatch,
                                 ck_tile::HostTensor<EDataType>& e_dev_result,
                                 ck_tile::HostTensor<EDataType>& e_host_result)
{
    const float max_accumulated_value =
        *std::max_element(e_host_result.mData.begin(), e_host_result.mData.end());

    const auto rtol_atol =
        batched_contraction_calculate_rtol_atol<ADataType, BDataType, AccDataType, EDataType>(
            K, kbatch, max_accumulated_value);

    bool pass = ck_tile::check_err(e_dev_result,
                                   e_host_result,
                                   "Error: Incorrect results!",
                                   rtol_atol.at(ck_tile::number<0>{}),
                                   rtol_atol.at(ck_tile::number<1>{}));

    std::cout << "For " << instanceName << " Relative error threshold is "
              << rtol_atol.at(ck_tile::number<0>{}) << " Absolute error threshold is "
              << rtol_atol.at(ck_tile::number<1>{}) << std::endl;
    std::cout << "The verification result is:" << (pass ? "correct" : "fail") << std::endl;

    return pass;
}

/// @brief Function to get the kernel output with reference implementation on CPU
template <typename ADataType,
          typename BDataType,
          typename DDataType,
          typename AccDataType,
          typename EDataType,
          typename CDEElementWise,
          ck_tile::index_t NumDTensor>
void batched_contraction_host_reference(
    int verify,
    const ck_tile::HostTensor<ADataType>& a_tensor,
    const ck_tile::HostTensor<BDataType>& b_tensor,
    const std::array<ck_tile::HostTensor<DDataType>, NumDTensor>& ds_tensors,
    ck_tile::HostTensor<EDataType>& e_host_result,
    ck_tile::index_t G_total,
    ck_tile::index_t M_total,
    ck_tile::index_t N_total,
    ck_tile::index_t K_total,
    const CDEElementWise& cde_elementwise,
    const std::vector<ck_tile::index_t>& G_dims,
    const std::vector<ck_tile::index_t>& M_dims,
    const std::vector<ck_tile::index_t>& N_dims,
    const std::vector<ck_tile::index_t>& K_dims)
{
    if(verify > 0)
    {
        ck_tile::compute_reference_batched_contraction<ADataType,
                                                       BDataType,
                                                       DDataType,
                                                       EDataType,
                                                       AccDataType,
                                                       CDEElementWise,
                                                       NumDTensor>(a_tensor,
                                                                   b_tensor,
                                                                   ds_tensors,
                                                                   e_host_result,
                                                                   G_total,
                                                                   M_total,
                                                                   N_total,
                                                                   K_total,
                                                                   cde_elementwise,
                                                                   G_dims,
                                                                   M_dims,
                                                                   N_dims,
                                                                   K_dims);
    }
}
#pragma clang diagnostic pop
