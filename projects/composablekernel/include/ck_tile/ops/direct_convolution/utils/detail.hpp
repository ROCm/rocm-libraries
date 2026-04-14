// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>

namespace ck_tile::direct_conv
{

// Invoke f<0>(), f<1>(), ..., f<N-1>() in sequence at compile time.
template <int N, typename F>
__device__ __forceinline__ void static_for(F f)
{
    [&]<int... Is>(std::integer_sequence<int, Is...>)
    { (f.template operator()<Is>(), ...); }(std::make_integer_sequence<int, N>{});
}

// Call f<I>() for the unique I that matches the runtime idx.
template <int N, typename F>
__device__ __forceinline__ void dispatch(int idx, F f)
{
    static_for<N>(
        [&]<int I>()
        {
            if(idx == I)
                f.template operator()<I>();
        });
}

} // namespace ck_tile::direct_conv
