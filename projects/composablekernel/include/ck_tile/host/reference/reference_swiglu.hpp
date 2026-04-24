// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/host/reference/reference_elementwise.hpp"
#include "ck_tile/host/reference/reference_gemm.hpp"

#include <algorithm>
#include <optional>

namespace ck_tile {
template <typename T>
static constexpr auto swish_mul_reference = [](const T& y0, const T& y1) {
    constexpr T one{1};

    const T y1_act = y1 / (one + ck_tile::exp(-y1));
    return y0 * y1_act;
};

template <typename Acc, typename A, typename B0, typename B1, typename C>
auto swiglu_reference(const HostTensor<A>& a,
                      const HostTensor<B0>& b0,
                      const HostTensor<B1>& b1,
                      HostTensor<C>& c,
                      std::optional<std::function<C(C, C)>> act_mul = {},
                      double* max_val                               = nullptr) -> void
{
    /// SwiGLU(x, W, V, b, c, β)  = Swish(xV + b) ⊙ (xW + c)

    HostTensor<Acc> c0(c.mDesc);
    ck_tile::reference_gemm<A, B0, Acc, Acc>(a, b0, c0);

    HostTensor<Acc> c1(c.mDesc);
    ck_tile::reference_gemm<A, B1, Acc, Acc>(a, b1, c1);

    auto op = act_mul.value_or(swish_mul_reference<Acc>);
    ck_tile::reference_binary_elementwise<Acc, Acc, C, Acc>(c0, c1, c, op);

    if(max_val)
    {
        const auto max_val_c0 = *std::max_element(c0.mData.begin(), c0.mData.end());
        const auto max_val_c1 = *std::max_element(c1.mData.begin(), c1.mData.end());
        const auto max_val_c  = *std::max_element(c.mData.begin(), c.mData.end());
        *max_val = std::max({double(max_val_c0), double(max_val_c1), double(max_val_c)});
    }
}
} // namespace ck_tile
