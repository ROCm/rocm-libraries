// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>

template <auto KernelDescriptor>
constexpr void instantiate_kernel(std::vector<BaseOperatorPtr>& kernels)
{
    using Builder = ckb::ConvBuilder<KernelDescriptor.signature, KernelDescriptor.algorithm>;
    do_builder_checks<Builder>();

    kernels.push_back(std::make_unique<typename Builder::Instance>());
}

template <typename T, T... values>
constexpr void build_kernels_helper(std::vector<BaseOperatorPtr>& kernels)
{
    // std::array<BaseOperatorPtr, sizeof...(values)> result{};
    ((instantiate_kernel<values>(kernels)), ...);
}

template <typename T, std::size_t N, std::array<T, N> arr, std::size_t... I>
constexpr void build_kernels_impl(std::vector<BaseOperatorPtr>& kernels, std::index_sequence<I...>)
{
    build_kernels_helper<T, arr[I]...>(kernels);
}

template <typename ArrayType>
struct array_traits;

template <typename T, std::size_t N>
struct array_traits<std::array<T, N>>
{
    using value_type                  = T;
    static constexpr std::size_t size = N;
};

template <auto arr>
constexpr void build_kernels(std::vector<BaseOperatorPtr>& kernels)
{
    using T                 = typename array_traits<decltype(arr)>::value_type;
    constexpr std::size_t N = array_traits<decltype(arr)>::size;
    build_kernels_impl<T, N, arr>(kernels, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N1, std::size_t N2, std::size_t... I1, std::size_t... I2>
constexpr std::array<T, N1 + N2> concat2_impl(const std::array<T, N1>& a,
                                              const std::array<T, N2>& b,
                                              std::index_sequence<I1...>,
                                              std::index_sequence<I2...>)
{
    return {a[I1]..., b[I2]...};
}

template <typename T, std::size_t N1, std::size_t N2>
constexpr std::array<T, N1 + N2> concat2(const std::array<T, N1>& a, const std::array<T, N2>& b)
{
    return concat2_impl(a, b, std::make_index_sequence<N1>{}, std::make_index_sequence<N2>{});
}

// Variadic: concatenate many arrays recursively
template <typename T, std::size_t N>
constexpr std::array<T, N> concat(const std::array<T, N>& a)
{
    return a;
}

template <typename T, std::size_t N1, std::size_t N2, std::size_t... Ns>
constexpr auto
concat(const std::array<T, N1>& a, const std::array<T, N2>& b, const std::array<T, Ns>&... rest)
{
    return concat(concat2(a, b), rest...);
}
