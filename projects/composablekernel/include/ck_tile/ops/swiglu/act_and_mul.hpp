// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/math.hpp"

namespace ck_tile {
template <typename Act>
struct ActMul
{
    constexpr static Act act{};

    template <typename T>
    CK_TILE_HOST_DEVICE auto operator()(const T& x0, const T& x1) const -> T
    {
        return x0 * act(x1);
    }

    template <typename T>
    CK_TILE_HOST_DEVICE auto operator()(T& y, const T& x0, const T& x1) const -> void
    {
        y = this->operator()(x0, x1);
    }
};

template <double Beta = 1.0>
struct Swish
{
    constexpr static double beta = Beta;

    template <typename T>
    CK_TILE_HOST_DEVICE auto operator()(const T& x) const -> T
    {
        constexpr static T b   = static_cast<T>(beta);
        constexpr static T one = 1;

        return x / (one + ck_tile::exp(b * -x));
    };

    template <typename T>
    CK_TILE_HOST_DEVICE auto operator()(T& y, const T& x) const -> void
    {
        y = this->operator()(x);
    };
};

using SwishMul = ActMul<Swish<1.0>>;
} // namespace ck_tile
