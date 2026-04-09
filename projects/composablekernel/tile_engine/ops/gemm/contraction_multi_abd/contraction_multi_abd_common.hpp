// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>
#include <sstream>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

template <typename T>
struct DataTypeTraits;

template <>
struct DataTypeTraits<float>
{
    static constexpr const char* name = "fp32";
};

template <>
struct DataTypeTraits<double>
{
    static constexpr const char* name = "fp64";
};

template <>
struct DataTypeTraits<ck_tile::half_t>
{
    static constexpr const char* name = "fp16";
};

template <>
struct DataTypeTraits<ck_tile::bf16_t>
{
    static constexpr const char* name = "bf16";
};

template <typename Layout>
constexpr auto is_row_major(Layout)
{
    return ck_tile::bool_constant<std::is_same_v<Layout, ck_tile::tensor_layout::gemm::RowMajor>>{};
}

struct KernelTraits
{
    std::string pipeline;
    std::string scheduler;
    std::string epilogue;
    bool pad_m;
    bool pad_n;
    bool pad_k;

    KernelTraits()
        : pipeline("compv3"),
          scheduler("intrawave"),
          epilogue("cshuffle"),
          pad_m(false),
          pad_n(false),
          pad_k(false)
    {
    }
};

inline std::vector<ck_tile::index_t> parse_dimensions(const std::string& dims_str)
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

inline ck_tile::index_t calculate_total(const std::vector<ck_tile::index_t>& dims)
{
    ck_tile::index_t total = 1;
    for(auto d : dims)
        total *= d;
    return total;
}

inline std::vector<ck_tile::index_t>
concatenate_dims(const std::vector<std::vector<ck_tile::index_t>>& components)
{
    std::vector<ck_tile::index_t> result;
    for(const auto& c : components)
        result.insert(result.end(), c.begin(), c.end());
    return result;
}

inline std::vector<ck_tile::index_t>
compute_row_major_strides(const std::vector<ck_tile::index_t>& dims)
{
    std::vector<ck_tile::index_t> strides(dims.size(), 1);
    for(int i = static_cast<int>(dims.size()) - 2; i >= 0; --i)
    {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    return strides;
}
