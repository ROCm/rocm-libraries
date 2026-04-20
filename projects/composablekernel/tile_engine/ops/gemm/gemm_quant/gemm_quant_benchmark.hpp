// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <iostream>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <tuple>
#include <sstream>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm_quant.hpp"
#include "gemm/gemm_benchmark.hpp"
#include "gemm/gemm_common.hpp"

struct GemmQuantProblem : GemmProblem
{
    int stride_aq_;
    int stride_bq_;
    int qk_a_;
    int qk_b_;

    std::string dtype_q_;
    std::string layout_aq_;
    std::string layout_bq_;
    std::string quant_mode_;
    std::string quant_profile_;
    std::string aq_group_;
    std::string bq_group_;

    friend std::ostream& operator<<(std::ostream& os, const GemmQuantProblem& problem)
    {
        os << "{\n"
           << "   \"split_k\":" << problem.split_k_ << ",\n"
           << "   \"m\":" << problem.m_ << ",\n"
           << "   \"n\":" << problem.n_ << ",\n"
           << "   \"k\":" << problem.k_ << ",\n"
           << "   \"stride_a\":" << problem.stride_a_ << ",\n"
           << "   \"stride_aq\":" << problem.stride_aq_ << ",\n"
           << "   \"stride_b\":" << problem.stride_b_ << ",\n"
           << "   \"stride_bq\":" << problem.stride_bq_ << ",\n"
           << "   \"stride_c\":" << problem.stride_c_ << ",\n"
           << "   \"qk_a\":" << problem.qk_a_ << ",\n"
           << "   \"qk_b\":" << problem.qk_b_ << ",\n"
           << "   \"dtype_a\":\"" << problem.dtype_a_ << "\",\n"
           << "   \"dtype_q\":\"" << problem.dtype_q_ << "\",\n"
           << "   \"dtype_b\":\"" << problem.dtype_b_ << "\",\n"
           << "   \"dtype_acc\":\"" << problem.dtype_acc_ << "\",\n"
           << "   \"dtype_c\":\"" << problem.dtype_c_ << "\",\n"
           << "   \"layout_a\":\"" << problem.layout_a_ << "\",\n"
           << "   \"layout_aq\":\"" << problem.layout_aq_ << "\",\n"
           << "   \"layout_b\":\"" << problem.layout_b_ << "\",\n"
           << "   \"layout_bq\":\"" << problem.layout_bq_ << "\",\n"
           << "   \"layout_c\":\"" << problem.layout_c_ << "\",\n"
           << "   \"quant_mode\":\"" << problem.quant_mode_ << "\",\n"
           << "   \"quant_profile\":\"" << problem.quant_profile_ << "\",\n"
           << "   \"aq_group\":\"" << problem.aq_group_ << "\",\n"
           << "   \"bq_group\":\"" << problem.bq_group_ << "\",\n"
           << "   \"structured_sparsity\":"
           << (problem.structured_sparsity_ ? "true" : "false") << "\n"
           << "}";
        return os;
    }
};

inline std::array<int, 3> parse_quant_group(const std::string& group_name)
{
    std::array<int, 3> dims = {1, 1, 1};
    std::stringstream ss(group_name);
    std::string token;
    int idx = 0;
    while(std::getline(ss, token, 'x') && idx < 3)
    {
        dims[idx++] = std::stoi(token);
    }
    return dims;
}

inline void add_quant_benchmark_args(ck_tile::ArgParser& arg_parser)
{
    arg_parser.insert("stride_q", "0", "Fallback stride for AQ/BQ tensors. Default is 0.")
        .insert("stride_aq", "0", "The stride value for tensor AQ. Default is 0.")
        .insert("stride_bq", "0", "The stride value for tensor BQ. Default is 0.");
}

inline auto create_quant_args(int argc, char* argv[])
{
    return create_args(argc, argv, 1, add_quant_benchmark_args);
}

inline void gemm_quant_host_reference(
    int verify,
    ck_tile::HostTensor<ADataType>& a_m_k,
    ck_tile::HostTensor<AQDataType>* aq_tensor,
    ck_tile::HostTensor<BDataType>& b_k_n,
    ck_tile::HostTensor<BQDataType>* bq_tensor,
    ck_tile::HostTensor<CDataType>& c_m_n_host_result)
{
    if(verify == 0)
    {
        return;
    }

    c_m_n_host_result.SetZero();

    if constexpr(SelectedKernel::QuantMode == ck_tile::QuantType::AQuantGrouped)
    {
        ck_tile::reference_gemm_quant<ADataType,
                                      AQDataType,
                                      BDataType,
                                      AccDataType,
                                      CDataType,
                                      AQuantGroupSize,
                                      true>(
            a_m_k, *aq_tensor, b_k_n, c_m_n_host_result);
    }
    else if constexpr(SelectedKernel::QuantMode == ck_tile::QuantType::BQuantGrouped)
    {
        ck_tile::reference_gemm_quant<ADataType,
                                      BQDataType,
                                      BDataType,
                                      AccDataType,
                                      CDataType,
                                      BQuantGroupSize,
                                      false>(
            a_m_k, *bq_tensor, b_k_n, c_m_n_host_result);
    }
    else if constexpr(SelectedKernel::QuantMode == ck_tile::QuantType::RowColQuant)
    {
        ck_tile::reference_gemm_rowcol_quant<ADataType,
                                             AQDataType,
                                             BDataType,
                                             BQDataType,
                                             AccDataType,
                                             CDataType>(
            a_m_k, *aq_tensor, b_k_n, *bq_tensor, c_m_n_host_result);
    }
    else if constexpr(SelectedKernel::QuantMode == ck_tile::QuantType::TensorQuant)
    {
        ck_tile::reference_gemm_tensor_quant<ADataType,
                                             AQDataType,
                                             BDataType,
                                             BQDataType,
                                             AccDataType,
                                             CDataType>(
            a_m_k, *aq_tensor, b_k_n, *bq_tensor, c_m_n_host_result);
    }
    else
    {
        throw std::runtime_error("Unsupported gemm_quant mode in tile engine");
    }
}
