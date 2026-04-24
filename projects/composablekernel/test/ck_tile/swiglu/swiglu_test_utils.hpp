// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <gtest/gtest.h>

#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/fill.hpp"
#include "ck_tile/host/host_tensor.hpp"
// #include "ck_tile/utils.hpp"

#include <cstddef>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace ck_tile::utils {
template <bool... Bs>
struct boolseq
{
    using type       = boolseq;
    using value_type = bool;
};

struct SwiGLUShape
{
    size_t m{};
    size_t n{};
    size_t k{};

    bool a_is_row_major{};
    bool b_is_row_major{};
    bool c_is_row_major{};

    [[nodiscard]] auto mnk_lengths() const -> tuple<size_t, size_t, size_t> { return {m, n, k}; }

    [[nodiscard]] auto abc_layouts() const -> tuple<bool, bool, bool>
    {
        return {a_is_row_major, b_is_row_major, c_is_row_major};
    }

    friend std::ostream& operator<<(std::ostream& stream, const SwiGLUShape& shape)
    {
        constexpr auto maj_str = [](bool is_row_major) -> std::string {
            return is_row_major ? " (Row major)" : " (Col major)";
        };

        auto [m_, n_, k_] = shape.mnk_lengths();
        stream << "SwiGLUShape: \n"
               << "  A: " << m_ << " x " << k_ << maj_str(shape.a_is_row_major) << "\n"
               << "  B: " << k_ << " x " << n_ << maj_str(shape.b_is_row_major) << "\n"
               << "  C: " << m_ << " x " << n_ << maj_str(shape.c_is_row_major) << "\n";
        return stream;
    }

    [[nodiscard]] auto to_string() const -> std::string
    {
        std::stringstream stream;
        stream << *this;
        return stream.str();
    }
};

[[nodiscard]] static inline auto get_flops(const SwiGLUShape& shape) -> std::size_t
{
    auto [m, n, k] = shape.mnk_lengths();

    auto gemm_flops = 2 * m * n * k;
    auto elem_flops = m * n;
    auto flops      = (2 * gemm_flops) + elem_flops;
    return flops;
}

template <typename ADataType, typename BDataType, typename CDataType>
[[nodiscard]] auto get_num_bytes(const SwiGLUShape& shape) -> std::size_t
{
    auto [m, n, k] = shape.mnk_lengths();

    std::size_t a_size = sizeof(ADataType);
    std::size_t b_size = sizeof(BDataType);
    std::size_t c_size = sizeof(CDataType);
    auto gemm_bytes    = (a_size * m * k) + (b_size * k * n) + (c_size * m * n);
    auto elem_bytes    = c_size * m * n;
    auto bytes         = (2 * gemm_bytes) + elem_bytes;
    return bytes;
}

[[nodiscard]] auto
make_tensor_descriptor(index_t rows, index_t cols, bool is_row_major, index_t stride)
{
    using T = bool_constant<true>;
    using F = bool_constant<false>;
    HostTensorDescriptor dsc;
    return is_row_major ? host_tensor_descriptor(rows, cols, stride, T{})
                        : host_tensor_descriptor(rows, cols, stride, F{});
}

template <typename T>
[[nodiscard]] auto make_host_tensor(index_t rows, index_t cols, bool is_row_major, index_t stride)
    -> std::unique_ptr<HostTensor<T>>
{
    auto desc = make_tensor_descriptor(rows, cols, is_row_major, stride);
    auto data = std::make_unique<HostTensor<T>>(desc);
    return data;
}

template <typename T>
[[nodiscard]] auto make_device_buffer(const HostTensor<T>& data_host) -> std::unique_ptr<DeviceMem>
{
    auto data = std::make_unique<DeviceMem>(data_host.get_element_space_size_in_bytes());
    data->ToDevice(data_host.data());
    return data;
}

template <typename T>
auto device_to_host(const DeviceMem& data_dev, HostTensor<T>& data_host) -> void
{
    data_dev.FromDevice(data_host.data());
}

[[nodiscard]] static auto
default_stride(index_t stride, size_t rows, size_t cols, bool is_row_major) -> index_t
{
    if(stride != 0)
        return stride;
    return is_row_major ? index_t(cols) : index_t(rows);
}

template <typename T>
auto init_normal(HostTensor<T>& data, std::optional<uint32_t> seed, float mean = 0, float var = 0.1)
    -> void
{
    FillNormalDistribution<T>{mean, var, seed}(data);
}

template <typename T>
auto init_constant(HostTensor<T>& data, double val) -> void
{
    FillConstant<T>{static_cast<T>(val)}(data);
}
} // namespace ck_tile::utils
