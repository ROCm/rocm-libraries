// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>
#include <sstream>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/common/utils.hpp"
#include "common/utils.hpp"

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

template <std::size_t N>
inline std::array<ck_tile::index_t, N> to_fixed_dims(const std::vector<ck_tile::index_t>& dims)
{
    if(dims.size() != N)
    {
        throw std::invalid_argument("Expected " + std::to_string(N) + " dimensions, got " +
                                    std::to_string(dims.size()));
    }

    std::array<ck_tile::index_t, N> result{};
    std::copy(dims.begin(), dims.end(), result.begin());
    return result;
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
