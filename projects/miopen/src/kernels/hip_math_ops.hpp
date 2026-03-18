// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Given the value of x & y, computes the value r such that r = x - k*y,
// where is the quotient `k` is returned as an output parameter.
// The `k*y` multiplication is done in 24 bits. It is the responsibility of the solver
// to ensure `x` & `y` don't exceed 2^24.
inline __device__ unsigned int iRemquo(unsigned int x, unsigned int y, unsigned int& k)
{
    k = x / y;
    return x - __mul24(k, y);
}

// Responsibility of caller to ensure that `u`, `d`, and `u*d` don't exceed the 24-bit range.
inline __device__ unsigned iMod(unsigned v, unsigned u, unsigned d) { return v - __mul24(u, d); }

inline __device__ unsigned iDiv(unsigned v, unsigned d) { return v / d; }