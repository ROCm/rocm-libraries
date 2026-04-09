// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <sstream>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"

// 2 A tensors (fp16, fp16), 1 B tensor (fp16), 1 D tensor (fp16), E = fp16
using A0DataType = ck_tile::half_t;
using A1DataType = ck_tile::half_t;
using B0DataType = ck_tile::half_t;
using D0DataType = ck_tile::half_t;
using EDataType  = ck_tile::half_t;

using AsDataType = ck_tile::tuple<A0DataType, A1DataType>;
using BsDataType = ck_tile::tuple<B0DataType>;
using DsDataType = ck_tile::tuple<D0DataType>;

using AccDataType = float;

static constexpr ck_tile::index_t NumATensor = AsDataType::size();
static constexpr ck_tile::index_t NumBTensor = BsDataType::size();
static constexpr ck_tile::index_t NumDTensor = DsDataType::size();

#define GEMM_PIPELINE ck_tile::GemmPipelineAgBgCrCompV3
#define UNIVERSAL_GEMM_PIPELINE ck_tile::BaseGemmPipelineAgBgCrCompV3
#define GEMM_PIPELINE_SCHEDULER ck_tile::GemmPipelineScheduler::Intrawave

struct AddDs
{
    template <typename E, typename C, typename... Ds>
    CK_TILE_HOST_DEVICE auto operator()(E& e, const C& c, const Ds&... ds) const -> void
    {
        const float x0_f =
            ck_tile::type_convert<float>(c) + (ck_tile::type_convert<float>(ds) + ...);
        e = ck_tile::type_convert<E>(x0_f);
    }
};

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m_dims", "4,256", "M dimensions separated by comma")
        .insert("n_dims", "16,128", "N dimensions separated by comma")
        .insert("k_dims", "64", "K dimensions separated by comma")
        .insert("g_dims", "1,2", "G dimensions separated by comma")
        .insert("v", "1", "0. No validation, 1. Validation on CPU")
        .insert("warmup", "5", "number of iterations before benchmark the kernel")
        .insert("repeat", "10", "number of iterations to benchmark the kernel")
        .insert("log", "1", "log level for debugging");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

std::vector<ck_tile::index_t> parse_dimensions(const std::string& dims_str)
{
    std::vector<ck_tile::index_t> dims;
    std::stringstream ss(dims_str);
    std::string token;
    while(std::getline(ss, token, ','))
    {
        dims.push_back(std::stoi(token));
    }
    if(dims.empty())
    {
        throw std::invalid_argument("Dimensions cannot be empty");
    }
    return dims;
}

ck_tile::index_t calculate_total_elements(const std::vector<ck_tile::index_t>& dims)
{
    ck_tile::index_t total = 1;
    for(auto dim : dims)
        total *= dim;
    return total;
}

std::vector<ck_tile::index_t>
concatenate_dim_components(const std::vector<std::vector<ck_tile::index_t>>& dim_components)
{
    std::vector<ck_tile::index_t> result;
    for(const auto& component : dim_components)
        result.insert(result.end(), component.begin(), component.end());
    return result;
}

void print_dims(const std::string& name,
                const std::vector<ck_tile::index_t>& dims,
                ck_tile::index_t total)
{
    std::cout << name << ": [";
    for(size_t i = 0; i < dims.size(); ++i)
    {
        std::cout << dims[i];
        if(i < dims.size() - 1)
            std::cout << ",";
    }
    std::cout << "]";
    if(total != 0)
        std::cout << " (total=" << total << ")";
    std::cout << std::endl;
}
