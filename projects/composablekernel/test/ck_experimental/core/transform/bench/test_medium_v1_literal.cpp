// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Workload B (medium) — V1 type-based descriptor.
// 4-transform 4D LDS chain with composed merge.

#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/tensor/tensor_adaptor.hpp"
#include "ck_tile/core/tensor/tensor_descriptor.hpp"
#include "ck_tile/core/container/container_helper.hpp"

namespace {
using namespace ck_tile;

template <index_t MPerBlock, index_t KPerBlock>
CK_TILE_HOST_DEVICE constexpr auto make_medium_v1()
{
    constexpr index_t K_DIV = KPerBlock / 16;
    constexpr index_t K_MOD = 8;
    constexpr index_t VEC   = 2;

    // Type 1: 4D base (K_div, M, K_mod, Vec) with padded strides
    constexpr auto desc_0 = make_naive_tensor_descriptor(
        make_tuple(number<K_DIV>{}, number<MPerBlock>{}, number<K_MOD>{}, number<VEC>{}),
        make_tuple(number<(MPerBlock + 1) * K_MOD * VEC>{},
                   number<K_MOD * VEC>{},
                   number<VEC>{},
                   number<1>{}));

    // Type 2: merge K_mod + Vec → KModVec (4D → 3D)
    constexpr auto desc_1 = transform_tensor_descriptor(
        desc_0,
        make_tuple(make_pass_through_transform(K_DIV),
                   make_pass_through_transform(MPerBlock),
                   make_merge_transform(make_tuple(K_MOD, VEC))),
        make_tuple(sequence<0>{}, sequence<1>{}, sequence<2, 3>{}),
        make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

    // Type 3: merge K_div + KModVec → K, passthrough M (3D → 2D)
    constexpr auto desc_2 = transform_tensor_descriptor(
        desc_1,
        make_tuple(make_pass_through_transform(MPerBlock),
                   make_merge_transform(make_tuple(K_DIV, K_MOD * VEC))),
        make_tuple(sequence<1>{}, sequence<0, 2>{}),
        make_tuple(sequence<0>{}, sequence<1>{}));

    return desc_2;
}

template <index_t M, index_t K>
CK_TILE_HOST_DEVICE index_t use_v1(index_t m, index_t k)
{
    constexpr auto desc = make_medium_v1<M, K>();
    return desc.calculate_offset(make_multi_index(m, k));
}

__global__ void test_kernel(const index_t* m_in, const index_t* k_in, index_t* out,
                             const index_t* n_iters_ptr)
{

    const index_t tid     = blockIdx.x * blockDim.x + threadIdx.x;
    const index_t m_base  = m_in[tid];
    const index_t k_base  = k_in[tid];
    const index_t n_iters = *n_iters_ptr;   // runtime — opaque

    index_t s = 0;
    for(index_t i = 0; i < n_iters; ++i)
    {
        const index_t m = m_base + (i & 0xff);
        const index_t k = k_base + ((i >> 4) & 0xff);

        s += use_v1< 32,   32>(m, k);
        s += use_v1< 32,   64>(m, k);
        s += use_v1< 64,   32>(m, k);
        s += use_v1< 64,   64>(m, k);
        s += use_v1<128,   32>(m, k);
        s += use_v1<128,   64>(m, k);
        s += use_v1<128,  128>(m, k);
        s += use_v1<256,   32>(m, k);
        s += use_v1<256,   64>(m, k);
        s += use_v1<256,  128>(m, k);
    }
    out[tid] = s;
}
} // namespace

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <random>

int main()
{
    std::mt19937 rng{42};
    constexpr index_t N = 1024;
    index_t* h_m = new index_t[N];
    index_t* h_k = new index_t[N];
    std::uniform_int_distribution<int> dist_coord{0, 31};
    for(index_t i = 0; i < N; ++i) { h_m[i] = dist_coord(rng); h_k[i] = dist_coord(rng); }
    const char* env_loop = std::getenv("LOOP_ITERS");
    const index_t loop_iters = env_loop ? static_cast<index_t>(std::atoi(env_loop)) : 10000;


    index_t *d_m=nullptr, *d_k=nullptr, *d_out=nullptr, *d_iters=nullptr;
    (void)hipMalloc(&d_m, N * sizeof(index_t));
    (void)hipMalloc(&d_k, N * sizeof(index_t));
    (void)hipMalloc(&d_out, N * sizeof(index_t));
    (void)hipMemcpy(d_m, h_m, N * sizeof(index_t), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_k, h_k, N * sizeof(index_t), hipMemcpyHostToDevice);
    (void)hipMalloc(&d_iters, sizeof(index_t));
    (void)hipMemcpy(d_iters, &loop_iters, sizeof(index_t), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(test_kernel, dim3(4), dim3(256), 0, nullptr, d_m, d_k, d_out, d_iters);
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
        hipLaunchKernelGGL(test_kernel, dim3(4), dim3(256), 0, nullptr, d_m, d_k, d_out, d_iters);
        (void)hipEventRecord(stop, nullptr);
        (void)hipEventSynchronize(stop);
        float ms = 0.0f;
        (void)hipEventElapsedTime(&ms, start, stop);
        std::fprintf(stderr, "medium v1 literal trial %d: %.4f ms\n", trial_base + trial, ms);
    }
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);

    index_t* h_out = new index_t[N];
    (void)hipMemcpy(h_out, d_out, N * sizeof(index_t), hipMemcpyDeviceToHost);
    int rc = static_cast<int>(h_out[0]);
    (void)hipFree(d_m); (void)hipFree(d_k); (void)hipFree(d_out); (void)hipFree(d_iters);
    delete[] h_m; delete[] h_k; delete[] h_out;
    return rc;
}
