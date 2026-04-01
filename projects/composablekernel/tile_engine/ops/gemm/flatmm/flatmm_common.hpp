// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/host.hpp"

template <typename T>
struct DataTypeTraits;

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
struct DataTypeTraits<float>
{
    static constexpr const char* name = "fp32";
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
    bool persistent;

    KernelTraits()
        : pipeline("flatmmv1"),
          scheduler("default"),
          epilogue("cshuffle"),
          pad_m(false),
          pad_n(false),
          pad_k(false),
          persistent(false)
    {
    }
};

inline KernelTraits extract_traits_from_name(const std::string& kernel_name)
{
    KernelTraits traits;
    if(kernel_name.find("True") != std::string::npos)
    {
        traits.persistent = kernel_name.find("_True_") != std::string::npos ||
                            kernel_name.rfind("_True") == kernel_name.size() - 5;
    }
    return traits;
}

template <typename T>
auto shuffle_b_v0(const ck_tile::HostTensor<T>& tensor,
                  ck_tile::index_t n_warp_tile,
                  ck_tile::index_t k_warp_tile)
{
    assert(tensor.get_lengths().size() == 2);

    const int n = tensor.get_lengths()[1];
    const int k = tensor.get_lengths()[0];

    const int max_vec_size     = 16 / sizeof(T);
    const int k_lane           = ck_tile::get_warp_size() / n_warp_tile;
    const int items_per_access = std::min(max_vec_size, static_cast<int>(k_warp_tile / k_lane));

    ck_tile::HostTensor<T> tensor_view(
        {n / n_warp_tile, n_warp_tile, k / items_per_access, items_per_access});
    std::copy(tensor.begin(), tensor.end(), tensor_view.begin());
    return ck_tile::reference_permute(tensor_view, {0, 2, 1, 3});
}

template <typename T>
auto shuffle_b_v1(const ck_tile::HostTensor<T>& tensor,
                  ck_tile::index_t tile_n,
                  ck_tile::index_t warp_n,
                  ck_tile::index_t warp_tile_n,
                  ck_tile::index_t warp_tile_k)
{
    assert(tensor.get_lengths().size() == 2);

    const int n = tensor.get_lengths()[1];
    const int k = tensor.get_lengths()[0];

    const int max_vec_size     = 16 / sizeof(T);
    const int k_lane           = ck_tile::get_warp_size() / warp_tile_n;
    const int items_per_access = std::min(max_vec_size, static_cast<int>(warp_tile_k / k_lane));
    const int n_repeat         = tile_n / warp_tile_n / warp_n;

    ck_tile::HostTensor<T> tensor_view(
        {n / tile_n, warp_n, warp_tile_n, n_repeat, k / items_per_access, items_per_access});
    std::copy(tensor.begin(), tensor.end(), tensor_view.begin());
    return ck_tile::reference_permute(tensor_view, {0, 3, 1, 4, 2, 5});
}
