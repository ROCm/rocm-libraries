// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <miopen/ck_builder/shared.hpp>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

namespace ckb = ck_tile::builder;

// Takes a kernel descriptor and creates a kernel instance using CK Builder, adding it to a
// collection. The descriptor contains the signature and algorithm for the convolution kernel.
// Performs compile-time checks to ensure the Builder type is valid before instantiation.
template <auto KernelDescriptor, typename DeviceOp>
constexpr void instantiate_kernel(std::vector<std::unique_ptr<DeviceOp>>& kernels)
{
    using Builder = ckb::ConvBuilder<KernelDescriptor.signature, KernelDescriptor.algorithm>;
    static_assert(is_valid_builder<Builder>, "The derived Builder is not valid");

    kernels.push_back(std::make_unique<typename Builder::Instance>());
}

// Helper function that instantiates multiple kernels using variadic templates.
// Expands the parameter pack to call instantiate_kernel for each value.
template <typename DeviceOp, typename T, T... values>
constexpr void build_kernels_helper(std::vector<std::unique_ptr<DeviceOp>>& kernels)
{
    ((instantiate_kernel<values>(kernels)), ...);
}

// Implementation detail that expands an array into individual kernel instantiations.
// Uses index_sequence to unpack array elements at compile-time.
template <typename DeviceOp, typename T, std::size_t N, std::array<T, N> arr, std::size_t... I>
constexpr void build_kernels_impl(std::vector<std::unique_ptr<DeviceOp>>& kernels,
                                  std::index_sequence<I...>)
{
    build_kernels_helper<DeviceOp, T, arr[I]...>(kernels);
}

// Type trait to extract array properties (element type and size).
// Used to deduce template parameters from std::array types.
template <typename ArrayType>
struct array_traits;

template <typename T, std::size_t N>
struct array_traits<std::array<T, N>>
{
    using value_type                  = T;
    static constexpr std::size_t size = N;
};

// Main entry point that builds all kernels from an array of descriptors.
// Takes a compile-time array of kernel descriptors and instantiates each one.
// This is the primary function used by factory implementations.
template <auto arr, typename DeviceOp>
constexpr void build_kernels(std::vector<std::unique_ptr<DeviceOp>>& kernels)
{
    using T                 = typename array_traits<decltype(arr)>::value_type;
    constexpr std::size_t N = array_traits<decltype(arr)>::size;
    build_kernels_impl<DeviceOp, T, N, arr>(kernels, std::make_index_sequence<N>{});
}

// Implementation detail for concatenating two arrays at compile-time.
// Uses index sequences to unpack both arrays into a single new array.
template <typename T, std::size_t N1, std::size_t N2, std::size_t... I1, std::size_t... I2>
constexpr std::array<T, N1 + N2> concat2_impl(const std::array<T, N1>& a,
                                              const std::array<T, N2>& b,
                                              std::index_sequence<I1...>,
                                              std::index_sequence<I2...>)
{
    return {a[I1]..., b[I2]...};
}

// Concatenates two compile-time arrays into a single array.
// The result has size N1 + N2 and contains all elements from both input arrays.
template <typename T, std::size_t N1, std::size_t N2>
constexpr std::array<T, N1 + N2> concat2(const std::array<T, N1>& a, const std::array<T, N2>& b)
{
    return concat2_impl(a, b, std::make_index_sequence<N1>{}, std::make_index_sequence<N2>{});
}

// Base case for variadic concatenation - returns a single array unchanged.
template <typename T, std::size_t N>
constexpr std::array<T, N> concat(const std::array<T, N>& a)
{
    return a;
}

// Concatenates multiple compile-time arrays recursively.
// Combines arrays from left to right, allowing factory functions to merge
// kernel descriptors from different specializations (e.g., default, 1x1 filters, etc.).
template <typename T, std::size_t N1, std::size_t N2, std::size_t... Ns>
constexpr auto
concat(const std::array<T, N1>& a, const std::array<T, N2>& b, const std::array<T, Ns>&... rest)
{
    return concat(concat2(a, b), rest...);
}
} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
