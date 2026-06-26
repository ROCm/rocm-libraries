// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Workload C (complex), TRUE-RUNTIME variant — V1 type-based descriptor.
//
// IDENTICAL test setup to test_complex_v2_runtime_ctx_apples_buffer.cpp:
//   - Same kernel-arg signature: (m_in*, k_in*, out*, runtime_args*)
//   - Same 14-int device buffer
//   - Same load pattern: all 14 values loaded into local index_t consts
//   - Same descriptor-builder call shape: takes all 14 values
//   - Same 10 calculate_offset calls per thread
//
// V2 declares 6 structural integers (KTRPerm=4, M1=4, kfold=2, mpair=2, AK1=1,
// K0PerTW=1) as constexpr inside its graph definition. V1 mirrors this exactly:
// same 6 values are constexpr at namespace scope, used in the same
// compile-time positions (unmerge/merge tuple args).
//
// V2 declares 14 runtime placeholders. V1 uses 12 of them directly in its
// descriptor builder (d0..d5, s0..s4, M0_div_mpair). ASM verification (2026-05)
// confirmed V2/V3 also DCE USER_M and USER_K — they flow through inputs(...)
// slot routing but offset arithmetic only consumes derived strides + magic-divs,
// never the user-supplied dim values. So V1 drops them too: no load, no pin.
// Symmetric kernarg-load set across V1/V2/V3. NO in-kernel arithmetic on
// tile-config values in either design.

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

// Build the complex V1 descriptor from runtime values. USER_M/USER_K aren't
// part of V1's chain and (per ASM verification) V2/V3 DCE them too, so V1
// drops them with no load. Descriptor is built ONCE in test_kernel (outside
// the hot loop), matching V2/V3's build-once pattern.
CK_TILE_HOST_DEVICE constexpr auto make_complex_desc_v1(
    index_t d0, index_t d1, index_t d2, index_t d3, index_t d4, index_t d5,
    index_t s0, index_t s1, index_t s2, index_t s3, index_t s4,
    index_t M0_div_mpair)
{
    // Structural literals — function-local to mirror V2/V3 placement.
    constexpr index_t KTRPerm  = 4;
    constexpr index_t M1       = 4;
    constexpr index_t kfold    = 2;
    constexpr index_t mpair    = 2;
    constexpr index_t AK1      = 1;
    constexpr index_t K0PerTW  = 1;

    const auto desc_0 = make_naive_tensor_descriptor(
        make_tuple(d0, d1, d2, d3, d4, d5),
        make_tuple(s0, s1, s2, s3, s4, 1));

    const auto desc_1 = transform_tensor_descriptor(desc_0,
        make_tuple(make_pass_through_transform(d0),
                   make_pass_through_transform(d1),
                   make_xor_transform(make_tuple(d2, d3)),
                   make_pass_through_transform(d4),
                   make_pass_through_transform(d5)),
        make_tuple(sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}),
        make_tuple(sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}));

    const auto desc_2 = transform_tensor_descriptor(desc_1,
        make_tuple(make_pass_through_transform(d0),
                   make_pass_through_transform(d1),
                   make_unmerge_transform(make_tuple(KTRPerm, M1)),
                   make_unmerge_transform(make_tuple(kfold, M0_div_mpair)),
                   make_pass_through_transform(d4),
                   make_pass_through_transform(d5)),
        make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{},
                   sequence<3>{}, sequence<4>{}, sequence<5>{}),
        make_tuple(sequence<1>{}, sequence<2>{}, sequence<0, 3>{},
                   sequence<4, 5>{}, sequence<6>{}, sequence<7>{}));

    return transform_tensor_descriptor(desc_2,
        make_tuple(make_merge_transform(make_tuple(KTRPerm, d0, kfold, K0PerTW, AK1)),
                   make_merge_transform(make_tuple(M0_div_mpair, mpair, M1))),
        make_tuple(sequence<0, 1, 4, 2, 7>{}, sequence<5, 6, 3>{}),
        make_tuple(sequence<1>{}, sequence<0>{}));
}

// Identical kernel-arg signature to test_complex_v2_runtime_ctx_apples_buffer.cpp.
// Loads all 14 values from the same 14-int device buffer.
__global__ void test_kernel(const index_t* m_in, const index_t* k_in,
                             index_t* out,
                             const index_t* runtime_args,
                             const index_t* n_iters_ptr)
{
    // Build the descriptor ONCE (matches V2/V3's "build setup once,
    // hot-loop reuse" pattern + how production V1 actually uses descriptors).
    const auto desc = make_complex_desc_v1(
        runtime_args[0],   // d0
        runtime_args[1],   // d1
        runtime_args[2],   // d2
        runtime_args[3],   // d3
        runtime_args[4],   // d4
        runtime_args[5],   // d5
        runtime_args[6],   // s0
        runtime_args[7],   // s1
        runtime_args[8],   // s2
        runtime_args[9],   // s3
        runtime_args[10],  // s4
        runtime_args[11]); // M0_div_mpair

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

    // 14-int buffer — identical layout to test_complex_v2_runtime_ctx_apples_buffer.cpp.
    constexpr index_t NA = 14;
    index_t h_args[NA];
    h_args[0]  = KPerBlock / 32;          // d0
    h_args[1]  = 1;                        // d1
    h_args[2]  = 16;                       // d2
    h_args[3]  = MPerBlock / 4;            // d3
    h_args[4]  = 2;                        // d4
    h_args[5]  = 1;                        // d5
    h_args[6]  = 8 * MPerBlock;            // s0
    h_args[7]  = 8 * MPerBlock;            // s1
    h_args[8]  = MPerBlock / 2;            // s2
    h_args[9]  = 2;                        // s3
    h_args[10] = 1;                        // s4
    h_args[11] = MPerBlock / 8;            // M0_div_mpair
    h_args[12] = MPerBlock;                // USER_M
    h_args[13] = KPerBlock / 4;            // USER_K

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
        std::fprintf(stderr, "complex v1 trial %d: %.4f ms\n", trial_base + trial, ms);
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
