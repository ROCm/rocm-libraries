// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile::direct_conv
{

// builtin max is not constexpr, so we define our own.
inline constexpr __device__ __host__ int maximum(int a, int b)
{ return (a > b) ? a : b; }

inline constexpr __device__ __host__ int divup(int x, int y)
{ return (x + y - 1) / y; }

} // namespace ck_tile::direct_conv
