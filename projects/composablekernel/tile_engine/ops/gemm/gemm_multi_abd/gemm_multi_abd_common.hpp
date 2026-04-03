// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <array>
#include <vector>
#include <utility>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/pk_int4.hpp"

//[TODO] This can be moved to commons
// DataTypeTraits for all supported types
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

template <>
struct DataTypeTraits<ck_tile::fp8_t>
{
    static constexpr const char* name = "fp8";
};

template <>
struct DataTypeTraits<ck_tile::bf8_t>
{
    static constexpr const char* name = "bf8";
};

template <>
struct DataTypeTraits<ck_tile::int8_t>
{
    static constexpr const char* name = "int8";
};

template <>
struct DataTypeTraits<ck_tile::int32_t>
{
    static constexpr const char* name = "int32";
};

template <>
struct DataTypeTraits<ck_tile::pk_int4_t>
{
    static constexpr const char* name = "pk_int4_t";
};

// Helper function to determine if a layout is row-major
template <typename Layout>
constexpr auto is_row_major(Layout)
{ return ck_tile::bool_constant<std::is_same_v<Layout, ck_tile::tensor_layout::gemm::RowMajor>>{}; }

// Structure to hold kernel traits for dispatcher
struct KernelTraits
{
    std::string pipeline;  // compv3, compv4, mem
    std::string scheduler; // intrawave, interwave
    std::string epilogue;  // cshuffle, default
    bool pad_m;
    bool pad_n;
    bool pad_k;
    bool persistent;

    // Constructor with defaults
    KernelTraits()
        : pipeline("compv4"),
          scheduler("intrawave"),
          epilogue("cshuffle"),
          pad_m(false),
          pad_n(false),
          pad_k(false),
          persistent(false)
    {
    }
};

// Helper to create an std::array of HostTensors from a vector of strides.
// All tensors share the same DataType and Layout.
template <typename DataType, typename Layout, std::size_t N, std::size_t... Is>
auto make_host_tensor_array_impl(ck_tile::index_t rows,
                                 ck_tile::index_t cols,
                                 const std::vector<int>& strides,
                                 std::index_sequence<Is...>)
{
    return std::array<ck_tile::HostTensor<DataType>, N>{
        ck_tile::HostTensor<DataType>(ck_tile::host_tensor_descriptor(
            rows, cols, static_cast<ck_tile::index_t>(strides[Is]), is_row_major(Layout{})))...};
}

template <typename DataType, typename Layout, std::size_t N>
auto make_host_tensor_array(ck_tile::index_t rows,
                            ck_tile::index_t cols,
                            const std::vector<int>& strides)
{
    return make_host_tensor_array_impl<DataType, Layout, N>(
        rows, cols, strides, std::make_index_sequence<N>{});
}
