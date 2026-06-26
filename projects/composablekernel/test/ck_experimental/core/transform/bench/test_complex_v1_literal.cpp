// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Workload C (complex) — V1 type-based descriptor.
// 10-transform production-complexity GEMM universal LDS chain (6D base, peak 8D).
// Modeled on gemm_universal_pipeline_ag_bg_cr_policy.hpp ColumnMajor Wave64.

#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/tensor/tensor_adaptor.hpp"
#include "ck_tile/core/tensor/tensor_descriptor.hpp"
#include "ck_tile/core/container/container_helper.hpp"

namespace {
using namespace ck_tile;

template <index_t MPerBlock, index_t KPerBlock>
CK_TILE_HOST_DEVICE constexpr auto make_complex_v1()
{
    constexpr index_t KThreadWrite = KPerBlock / 4;
    constexpr index_t K0PerThreadWrite = 1;
    constexpr index_t KThreadReadPerm = 4;
    constexpr index_t M0 = MPerBlock / 4;
    constexpr index_t M1 = 4;
    constexpr index_t kfold = 2;
    constexpr index_t mpair = 2;
    constexpr index_t AK1 = 1;

    constexpr auto desc_0 = make_naive_tensor_descriptor(
        make_tuple(number<KThreadWrite / kfold / KThreadReadPerm>{},
                   number<K0PerThreadWrite>{},
                   number<KThreadReadPerm * M1>{},
                   number<kfold * M0 / mpair>{},
                   number<mpair>{}, number<AK1>{}),
        make_tuple(number<K0PerThreadWrite * KThreadReadPerm * M1 * kfold * M0 / mpair * mpair * AK1>{},
                   number<KThreadReadPerm * M1 * kfold * M0 / mpair * mpair * AK1>{},
                   number<kfold * M0 / mpair * mpair * AK1>{},
                   number<mpair * AK1>{}, number<AK1>{}, number<1>{}),
        number<AK1>{}, number<1>{});

    constexpr auto desc_1 = transform_tensor_descriptor(desc_0,
        make_tuple(make_pass_through_transform(KThreadWrite / kfold / KThreadReadPerm),
                   make_pass_through_transform(K0PerThreadWrite),
                   make_xor_transform(make_tuple(number<KThreadReadPerm * M1>{},
                                                 number<kfold * M0 / mpair>{})),
                   make_pass_through_transform(mpair),
                   make_pass_through_transform(AK1)),
        make_tuple(sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}),
        make_tuple(sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}));

    constexpr auto desc_2 = transform_tensor_descriptor(desc_1,
        make_tuple(
            make_pass_through_transform(KThreadWrite / kfold / KThreadReadPerm),
            make_pass_through_transform(K0PerThreadWrite),
            make_unmerge_transform(make_tuple(number<KThreadReadPerm>{}, number<M1>{})),
            make_unmerge_transform(make_tuple(number<kfold>{}, number<M0 / mpair>{})),
            make_pass_through_transform(mpair), make_pass_through_transform(AK1)),
        make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{},
                   sequence<3>{}, sequence<4>{}, sequence<5>{}),
        make_tuple(sequence<1>{}, sequence<2>{}, sequence<0, 3>{},
                   sequence<4, 5>{}, sequence<6>{}, sequence<7>{}));

    return transform_tensor_descriptor(desc_2,
        make_tuple(make_merge_transform(make_tuple(number<KThreadReadPerm>{},
                       number<KThreadWrite / kfold / KThreadReadPerm>{},
                       number<kfold>{}, number<K0PerThreadWrite>{}, number<AK1>{})),
                   make_merge_transform(make_tuple(
                       number<M0 / mpair>{}, number<mpair>{}, number<M1>{}))),
        make_tuple(sequence<0, 1, 4, 2, 7>{}, sequence<5, 6, 3>{}),
        make_tuple(sequence<1>{}, sequence<0>{}));
}

template <index_t M, index_t K>
CK_TILE_HOST_DEVICE index_t use_v1(index_t m, index_t k)
{
    constexpr auto desc = make_complex_v1<M, K>();
    return desc.calculate_offset(make_multi_index(m, k));
}

// GPU kernel modelled after real ck_tile production usage:
//
//   - 10 descriptor template specializations exercised in one kernel
//     (matches the host-side test_all from the original Confluence spec).
//     This stresses the compiler with 10 distinct constexpr descriptors
//     each producing its own magic-divisor tables and arithmetic.
//   - 256 threads per block × 4 blocks = 1024 threads (typical block size
//     for compv6-style kernels on gfx942).
//   - Each thread computes its own (m, k) → offset for each descriptor.
//   - Per-thread sum is written to a global output array — prevents
//     the compiler from constant-folding the entire computation away.
__global__ void test_kernel(const index_t* m_in, const index_t* k_in,
                             index_t* out,
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
        std::fprintf(stderr, "complex v1 literal trial %d: %.4f ms\n", trial_base + trial, ms);
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
