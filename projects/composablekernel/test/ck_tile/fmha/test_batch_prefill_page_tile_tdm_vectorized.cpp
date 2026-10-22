// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// TDM descriptors for the vectorized (16-bit, x = 8) KV cache layout, read back from LDS:
//   K page-head [D/8][P][8]: element (t, d) at ((d/8)*P + t)*8 + d%8
//   V page-head [P/8][D][8]: element (t, d) at ((t/8)*D + d)*8 + t%8
// K3  rank 3, dims (8, D/8, rows), strides (P*8, 8): the K slice lands row-major [rows][D+8]
//     (the linear K LDS layout; the extra chunk and rows past the valid count are zero filled).
// V4  rank 4, dims (1, D, 8, groups), strides (8, 1, D*8): the V slice lands row-major [rows][D].
//     Also guards the group-2 tile_dim3 field (rank >= 4).
// Vg  rank 2, dims (D*8, groups): a partial last group of 8 tokens is NOT zero filled - the
//     tokens past the valid count are read from the page (the V tail must be cleared by the
//     caller).

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <vector>

#include "ck_tile/core.hpp"

namespace {

using ck_tile::index_t;
using T = uint16_t;

constexpr index_t kLdsElems = 16384;

struct Case
{
    int rank;
    uint32_t dims[4];
    uint64_t strides[4];
    uint16_t box[4];
    long src_offset;
    int lds_elems;
};

__global__ void read_back(const T* src, T* out, Case c)
{
    __shared__ T lds[kLdsElems];
    for(int i = threadIdx.x; i < kLdsElems; i += blockDim.x)
        lds[i] = 0xDEAD;
    __syncthreads();
    ck_tile::TDMConfig cfg;
    if(c.rank == 2)
        ck_tile::amd_tdm_load(ck_tile::createTDMDescriptor<T, 2>(
            src + c.src_offset, lds, c.dims, c.strides, c.box, cfg));
    else if(c.rank == 3)
        ck_tile::amd_tdm_load(ck_tile::createTDMDescriptor<T, 3>(
            src + c.src_offset, lds, c.dims, c.strides, c.box, cfg));
    else
        ck_tile::amd_tdm_load(ck_tile::createTDMDescriptor<T, 4>(
            src + c.src_offset, lds, c.dims, c.strides, c.box, cfg));
    ck_tile::s_wait_tensorcnt<0>();
    __syncthreads();
    for(int i = threadIdx.x; i < c.lds_elems; i += blockDim.x)
        out[i] = lds[i];
}

// expect[i] < 0: zero fill.
int CountMismatches(const std::vector<T>& h_src,
                    const T* d_src,
                    const Case& c,
                    const std::vector<long>& expect)
{
    T* d_out = nullptr;
    EXPECT_EQ(hipMalloc(&d_out, c.lds_elems * sizeof(T)), hipSuccess);
    hipLaunchKernelGGL(read_back, dim3(1), dim3(32), 0, nullptr, d_src, d_out, c);
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
    std::vector<T> h(c.lds_elems);
    EXPECT_EQ(hipMemcpy(h.data(), d_out, c.lds_elems * sizeof(T), hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(hipFree(d_out), hipSuccess);
    int bad = 0;
    for(int i = 0; i < c.lds_elems; ++i)
        bad += h[i] != (expect[i] < 0 ? T(0) : h_src[expect[i]]);
    return bad;
}

class VectorizedKvTdm : public ::testing::TestWithParam<std::tuple<int, int, int>>
{
};

TEST_P(VectorizedKvTdm, ReadBack)
{
    const auto [P, D, R] = GetParam();
    const int rows       = P == 16 ? 16 : 32; // one wave slice / P16 fragment
    if(R > rows)
        GTEST_SKIP();
    const int t0 = P == 16 ? 0 : 32; // first page-local token of the slice
    const long n = long(P) * D;
    std::vector<T> h_src(n);
    for(long i = 0; i < n; ++i)
        h_src[i] = T((i * 7 + 1) % 65521 + 1); // non-zero, distinct within a page-head
    T* d_src = nullptr;
    ASSERT_EQ(hipMalloc(&d_src, n * sizeof(T)), hipSuccess);
    ASSERT_EQ(hipMemcpy(d_src, h_src.data(), n * sizeof(T), hipMemcpyHostToDevice), hipSuccess);
    auto kidx   = [&](int t, int d) { return long((d / 8) * P + t) * 8 + d % 8; };
    auto vidx   = [&](int t, int d) { return long((t / 8) * D + d) * 8 + t % 8; };
    const int G = (R + 7) / 8, GB = rows / 8;

    {
        const Case c{3,
                     {8u, uint32_t(D / 8), uint32_t(R), 0u},
                     {uint64_t(P) * 8, 8, 0, 0},
                     {8, uint16_t(D / 8 + 1), uint16_t(rows), 0},
                     long(t0) * 8,
                     rows * (D + 8)};
        std::vector<long> e(c.lds_elems);
        for(int r = 0; r < rows; ++r)
            for(int col = 0; col < D + 8; ++col)
                e[r * (D + 8) + col] = (r < R && col < D) ? kidx(t0 + r, col) : -1;
        EXPECT_EQ(CountMismatches(h_src, d_src, c, e), 0) << "K3";
    }
    {
        const Case c{4,
                     {1u, uint32_t(D), 8u, uint32_t(G)},
                     {8, 1, uint64_t(D) * 8, 0},
                     {1, uint16_t(D), 8, uint16_t(GB)},
                     long(t0 / 8) * D * 8,
                     rows * D};
        std::vector<long> e(c.lds_elems);
        for(int r = 0; r < rows; ++r)
            for(int d = 0; d < D; ++d)
                e[r * D + d] = r / 8 < G ? vidx(t0 + r, d) : -1;
        EXPECT_EQ(CountMismatches(h_src, d_src, c, e), 0) << "V4";
    }
    {
        const Case c{2,
                     {uint32_t(D * 8), uint32_t(G), 0u, 0u},
                     {uint64_t(D) * 8, 1, 0, 0},
                     {uint16_t(D * 8), uint16_t(GB), 0, 0},
                     long(t0 / 8) * D * 8,
                     rows * D};
        std::vector<long> e(c.lds_elems);
        for(int g = 0; g < GB; ++g)
            for(int d = 0; d < D; ++d)
                for(int x = 0; x < 8; ++x)
                    e[(g * D + d) * 8 + x] = g < G ? vidx(t0 + g * 8 + x, d) : -1;
        EXPECT_EQ(CountMismatches(h_src, d_src, c, e), 0) << "Vg";
    }
    ASSERT_EQ(hipFree(d_src), hipSuccess);
}

INSTANTIATE_TEST_SUITE_P(PagesHeadsRows,
                         VectorizedKvTdm,
                         ::testing::Combine(::testing::Values(16, 64, 128, 1024),
                                            ::testing::Values(64, 128),
                                            ::testing::Values(32, 16, 13, 1)));

} // namespace
