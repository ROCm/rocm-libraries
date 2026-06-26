// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// SHORT chain (3 transforms) — V1 type-based, buffer-loaded.
// Mirrors test_medium_v1.cpp (4D Embed + Merge + Merge) but with all values
// from a device buffer (no const literals, no in-kernel arithmetic).
//
// Buffer layout (9 ints) shared with test_short_v2_runtime_ctx_buffer.cpp:
//   [0] K_DIV  [1] MPB    [2] K_MOD  [3] VEC    [4] KMV
//   [5] USER_K [6] s0     [7] s1     [8] s2
//
// USER_K is unused by V1's chain. ASM verification (2026-05) confirmed V2/V3
// also DCE USER_K — value flows through inputs(...) but offset arithmetic
// only uses derived strides + magic-divs. V1 drops it too: no load, no pin.

#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/tensor/tensor_adaptor.hpp"
#include "ck_tile/core/tensor/tensor_descriptor.hpp"
#include "ck_tile/core/container/container_helper.hpp"

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <random>

namespace {
using namespace ck_tile;

CK_TILE_HOST_DEVICE constexpr auto make_short_v1(
    index_t K_DIV, index_t MPB, index_t K_MOD, index_t VEC, index_t KMV,
    index_t s0, index_t s1, index_t s2)
{
    const auto desc_0 = make_naive_tensor_descriptor(
        make_tuple(K_DIV, MPB, K_MOD, VEC),
        make_tuple(s0, s1, s2, 1));

    const auto desc_1 = transform_tensor_descriptor(desc_0,
        make_tuple(make_pass_through_transform(K_DIV),
                   make_pass_through_transform(MPB),
                   make_merge_transform(make_tuple(K_MOD, VEC))),
        make_tuple(sequence<0>{}, sequence<1>{}, sequence<2, 3>{}),
        make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

    return transform_tensor_descriptor(desc_1,
        make_tuple(make_pass_through_transform(MPB),
                   make_merge_transform(make_tuple(K_DIV, KMV))),
        make_tuple(sequence<1>{}, sequence<0, 2>{}),
        make_tuple(sequence<0>{}, sequence<1>{}));
}

__global__ void test_kernel(const index_t* m_in, const index_t* k_in,
                             index_t* out,
                             const index_t* runtime_args,
                             const index_t* n_iters_ptr)
{
    // Build the descriptor ONCE (matches V2/V3's "build setup once,
    // hot-loop reuse" pattern + how production V1 actually uses descriptors).
    const auto desc = make_short_v1(
        runtime_args[0],   // K_DIV
        runtime_args[1],   // MPB
        runtime_args[2],   // K_MOD
        runtime_args[3],   // VEC
        runtime_args[4],   // KMV
        runtime_args[6],   // s0
        runtime_args[7],   // s1
        runtime_args[8]);  // s2

    const index_t tid     = blockIdx.x * blockDim.x + threadIdx.x;
    const index_t m_base  = m_in[tid];
    const index_t k_base  = k_in[tid];
    const index_t n_iters = *n_iters_ptr;   // runtime — opaque

    index_t s = 0;
    for(index_t i = 0; i < n_iters; ++i)
    {
        const index_t m = m_base + (i & 0xff);
        const index_t k = k_base + ((i >> 4) & 0xff);

        s += desc.calculate_offset(make_multi_index(m, k));
        s += desc.calculate_offset(make_multi_index(m + 1, k));
        s += desc.calculate_offset(make_multi_index(m + 2, k));
        s += desc.calculate_offset(make_multi_index(m + 3, k));
        s += desc.calculate_offset(make_multi_index(m + 4, k));
        s += desc.calculate_offset(make_multi_index(m, k + 1));
        s += desc.calculate_offset(make_multi_index(m, k + 2));
        s += desc.calculate_offset(make_multi_index(m, k + 3));
        s += desc.calculate_offset(make_multi_index(m, k + 4));
        s += desc.calculate_offset(make_multi_index(m + 1, k + 1));
    }
    out[tid] = s;
}
} // namespace

int main()
{
    std::mt19937 rng{42};
    constexpr int choices[]   = {32, 64, 128, 256};
    constexpr int k_choices[] = {32, 64, 128};
    std::uniform_int_distribution<int> dist_m{0, 3};
    std::uniform_int_distribution<int> dist_k{0, 2};
    const index_t MPerBlock = choices[dist_m(rng)];
    const index_t KPerBlock = k_choices[dist_k(rng)];

    constexpr index_t NA = 9;
    index_t h_args[NA];
    h_args[0] = KPerBlock / 16;
    h_args[1] = MPerBlock;
    h_args[2] = 8;
    h_args[3] = 2;
    h_args[4] = 16;
    h_args[5] = KPerBlock;
    h_args[6] = (MPerBlock + 1) * 16;
    h_args[7] = 16;
    h_args[8] = 2;

    constexpr index_t N = 1024;
    index_t* h_m = new index_t[N];
    index_t* h_k = new index_t[N];
    std::uniform_int_distribution<int> dist_coord{0, 31};
    for(index_t i = 0; i < N; ++i)
    {
        h_m[i] = dist_coord(rng);
        h_k[i] = dist_coord(rng);
    }
    const char* env_loop = std::getenv("LOOP_ITERS");
    const index_t loop_iters = env_loop ? static_cast<index_t>(std::atoi(env_loop)) : 10000;


    index_t *d_m = nullptr, *d_k = nullptr, *d_out = nullptr, *d_args = nullptr, *d_iters=nullptr;
    (void)hipMalloc(&d_m, N * sizeof(index_t));
    (void)hipMalloc(&d_k, N * sizeof(index_t));
    (void)hipMalloc(&d_out, N * sizeof(index_t));
    (void)hipMalloc(&d_args, NA * sizeof(index_t));
    (void)hipMemcpy(d_m, h_m, N * sizeof(index_t), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_k, h_k, N * sizeof(index_t), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_args, h_args, NA * sizeof(index_t), hipMemcpyHostToDevice);
    (void)hipMalloc(&d_iters, sizeof(index_t));
    (void)hipMemcpy(d_iters, &loop_iters, sizeof(index_t), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(test_kernel, dim3(4), dim3(256), 0, nullptr,
                       d_m, d_k, d_out, d_args, d_iters);
    (void)hipDeviceSynchronize();

    hipEvent_t start, stop;
    (void)hipEventCreate(&start);
    (void)hipEventCreate(&stop);
    const char* env_n     = std::getenv("N_TRIALS");
    const char* env_b     = std::getenv("TRIAL_BASE");
    const int   n_trials   = env_n ? std::atoi(env_n) : 100;
    const int   trial_base = env_b ? std::atoi(env_b) : 0;
    for(int trial = 1; trial <= n_trials; ++trial)
    {
        (void)hipEventRecord(start, nullptr);
        hipLaunchKernelGGL(test_kernel, dim3(4), dim3(256), 0, nullptr,
                           d_m, d_k, d_out, d_args, d_iters);
        (void)hipEventRecord(stop, nullptr);
        (void)hipEventSynchronize(stop);
        float ms = 0.0f;
        (void)hipEventElapsedTime(&ms, start, stop);
        std::fprintf(stderr, "medium v1 trial %d: %.4f ms\n", trial_base + trial, ms);
    }
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);

    index_t* h_out = new index_t[N];
    (void)hipMemcpy(h_out, d_out, N * sizeof(index_t), hipMemcpyDeviceToHost);
    int rc = static_cast<int>(h_out[0]);
    (void)hipFree(d_m); (void)hipFree(d_k); (void)hipFree(d_out); (void)hipFree(d_args); (void)hipFree(d_iters);
    delete[] h_m; delete[] h_k; delete[] h_out;
    return rc;
}
