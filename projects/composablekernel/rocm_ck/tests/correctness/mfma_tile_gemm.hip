// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// MFMA-tiled GEMM kernel for the split-accumulation reference.
// Computes one (blockM × blockN) tile of C[M,N] = sum_k A[M,K] * B[N,K].
// Adapted from CK tutorial 03_mfma_16x16x16 for FP32×FP32→FP32.

#include "mfma_tile_gemm.hpp"
#include "mfma_gemm_policy.hpp"

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/host/hip_check_error.hpp"

#include <hip/hip_runtime.h>

namespace rocm_ck::test {

template <typename AccumPolicy>
__global__ void
#if CK_TILE_WORKAROUND_LAUNCH_BOUND
    __launch_bounds__(kMfmaBlockSize)
#endif
        tile_gemm_mfma_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ partial,
                              float* __restrict__ err_out,
                              ck_tile::index_t M,
                              ck_tile::index_t N,
                              ck_tile::index_t K_full,
                              ck_tile::index_t k_offset,
                              ck_tile::index_t K_tile,
                              ck_tile::index_t num_tile_n)
{
    using namespace ck_tile;

    const index_t i_block = get_block_id();
    const index_t iM = __builtin_amdgcn_readfirstlane((i_block / num_tile_n) * kMfmaBlockM);
    const index_t iN = __builtin_amdgcn_readfirstlane((i_block % num_tile_n) * kMfmaBlockN);

    // Global memory views — A[M, K_full], B[N, K_full]
    const auto a_global = make_naive_tensor_view<address_space_enum::global>(
        A, make_tuple(M, K_full), make_tuple(K_full, 1), number<kKPack>{}, number<1>{});

    const auto b_global = make_naive_tensor_view<address_space_enum::global>(
        B, make_tuple(N, K_full), make_tuple(K_full, 1), number<kKPack>{}, number<1>{});

    // DRAM copy windows with tile distributions for coalesced loads
    auto a_copy_window = make_tile_window(
        a_global,
        make_tuple(number<kMfmaBlockM>{}, number<kMfmaBlockK>{}),
        {iM, k_offset},
        MakeADramDist());
    auto b_copy_window = make_tile_window(
        b_global,
        make_tuple(number<kMfmaBlockN>{}, number<kMfmaBlockK>{}),
        {iN, k_offset},
        MakeBDramDist());

    // LDS
    __shared__ char lds_raw[kLdsBytes];
    float* p_a_lds = reinterpret_cast<float*>(lds_raw);
    float* p_b_lds = reinterpret_cast<float*>(lds_raw + kALdsBytes);

    constexpr auto a_lds_desc = MakeALdsDesc();
    constexpr auto b_lds_desc = MakeBLdsDesc();

    auto a_lds = make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_desc);
    auto b_lds = make_tensor_view<address_space_enum::lds>(p_b_lds, b_lds_desc);

    auto a_lds_store_window = make_tile_window(
        a_lds,
        make_tuple(number<kMfmaBlockM>{}, number<kMfmaBlockK>{}),
        {0, 0},
        a_copy_window.get_tile_distribution());
    auto b_lds_store_window = make_tile_window(
        b_lds,
        make_tuple(number<kMfmaBlockN>{}, number<kMfmaBlockK>{}),
        {0, 0},
        b_copy_window.get_tile_distribution());

    auto a_lds_gemm_window = make_tile_window(
        a_lds, make_tuple(number<kMfmaBlockM>{}, number<kMfmaBlockK>{}), {0, 0});
    auto b_lds_gemm_window = make_tile_window(
        b_lds, make_tuple(number<kMfmaBlockN>{}, number<kMfmaBlockK>{}), {0, 0});

    // Block GEMM with accumulation policy
    MfmaBlockGemmAccum<AccumPolicy> block_gemm;
    auto c_block = decltype(block_gemm)::MakeCBlockTile();
    auto err_block = decltype(c_block){};
    tile_elementwise_inout([](auto& c) { c = 0.0f; }, c_block);
    tile_elementwise_inout([](auto& c) { c = 0.0f; }, err_block);

    // K-loop over inner tiles
    index_t num_k_inner = K_tile / kMfmaBlockK;

    for(index_t ik = 0; ik < num_k_inner; ++ik)
    {
        auto a_tile = load_tile(a_copy_window);
        auto b_tile = load_tile(b_copy_window);

        move_tile_window(a_copy_window, {0, kMfmaBlockK});
        move_tile_window(b_copy_window, {0, kMfmaBlockK});

        store_tile(a_lds_store_window, a_tile);
        store_tile(b_lds_store_window, b_tile);

        block_sync_lds();
        block_gemm(c_block, err_block, a_lds_gemm_window, b_lds_gemm_window);
        block_sync_lds();
    }

    const auto c_global = make_naive_tensor_view<address_space_enum::global>(
        partial, make_tuple(M, N), make_tuple(N, 1), number<kKPack>{}, number<1>{});
    auto c_window = make_tile_window(
        c_global, make_tuple(number<kMfmaBlockM>{}, number<kMfmaBlockN>{}), {iM, iN});

    if(err_out)
    {
        const auto c_out = tile_elementwise_in(
            [](const auto& acc) { return ck_tile::type_convert<float>(acc); }, c_block);
        store_tile(c_window, c_out);

        const auto err_global = make_naive_tensor_view<address_space_enum::global>(
            err_out, make_tuple(M, N), make_tuple(N, 1), number<kKPack>{}, number<1>{});
        auto err_window = make_tile_window(
            err_global, make_tuple(number<kMfmaBlockM>{}, number<kMfmaBlockN>{}), {iM, iN});
        const auto e_out = tile_elementwise_in(
            [](const auto& acc) { return ck_tile::type_convert<float>(acc); }, err_block);
        store_tile(err_window, e_out);
    }
    else
    {
        AccumPolicy::finalize(c_block, err_block);
        const auto c_out = tile_elementwise_in(
            [](const auto& acc) { return ck_tile::type_convert<float>(acc); }, c_block);
        store_tile(c_window, c_out);
    }
}

// =========================================================================
// K4-decomposed MFMA kernel — same LDS loads, K4 warp GEMM reads
// =========================================================================

template <typename AccumPolicy>
__global__ void
#if CK_TILE_WORKAROUND_LAUNCH_BOUND
    __launch_bounds__(kMfmaBlockSize)
#endif
        tile_gemm_mfma_k4_kernel(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ partial,
                                 float* __restrict__ err_out,
                                 ck_tile::index_t M,
                                 ck_tile::index_t N,
                                 ck_tile::index_t K_full,
                                 ck_tile::index_t k_offset,
                                 ck_tile::index_t K_tile,
                                 ck_tile::index_t num_tile_n)
{
    using namespace ck_tile;

    const index_t i_block = get_block_id();
    const index_t iM = __builtin_amdgcn_readfirstlane((i_block / num_tile_n) * kMfmaBlockM);
    const index_t iN = __builtin_amdgcn_readfirstlane((i_block % num_tile_n) * kMfmaBlockN);

    const auto a_global = make_naive_tensor_view<address_space_enum::global>(
        A, make_tuple(M, K_full), make_tuple(K_full, 1), number<kKPack>{}, number<1>{});

    const auto b_global = make_naive_tensor_view<address_space_enum::global>(
        B, make_tuple(N, K_full), make_tuple(K_full, 1), number<kKPack>{}, number<1>{});

    auto a_copy_window = make_tile_window(
        a_global,
        make_tuple(number<kMfmaBlockM>{}, number<kMfmaBlockK>{}),
        {iM, k_offset},
        MakeADramDist());
    auto b_copy_window = make_tile_window(
        b_global,
        make_tuple(number<kMfmaBlockN>{}, number<kMfmaBlockK>{}),
        {iN, k_offset},
        MakeBDramDist());

    __shared__ char lds_raw[kLdsBytes];
    float* p_a_lds = reinterpret_cast<float*>(lds_raw);
    float* p_b_lds = reinterpret_cast<float*>(lds_raw + kALdsBytes);

    constexpr auto a_lds_desc = MakeALdsDesc();
    constexpr auto b_lds_desc = MakeBLdsDesc();

    auto a_lds = make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_desc);
    auto b_lds = make_tensor_view<address_space_enum::lds>(p_b_lds, b_lds_desc);

    auto a_lds_store_window = make_tile_window(
        a_lds,
        make_tuple(number<kMfmaBlockM>{}, number<kMfmaBlockK>{}),
        {0, 0},
        a_copy_window.get_tile_distribution());
    auto b_lds_store_window = make_tile_window(
        b_lds,
        make_tuple(number<kMfmaBlockN>{}, number<kMfmaBlockK>{}),
        {0, 0},
        b_copy_window.get_tile_distribution());

    auto a_lds_gemm_window = make_tile_window(
        a_lds, make_tuple(number<kMfmaBlockM>{}, number<kMfmaBlockK>{}), {0, 0});
    auto b_lds_gemm_window = make_tile_window(
        b_lds, make_tuple(number<kMfmaBlockN>{}, number<kMfmaBlockK>{}), {0, 0});

    MfmaBlockGemmAccumK4<AccumPolicy> block_gemm;
    auto c_block = decltype(block_gemm)::MakeCBlockTile();
    auto err_block = decltype(c_block){};
    tile_elementwise_inout([](auto& c) { c = 0.0f; }, c_block);
    tile_elementwise_inout([](auto& c) { c = 0.0f; }, err_block);

    index_t num_k_inner = K_tile / kMfmaBlockK;

    for(index_t ik = 0; ik < num_k_inner; ++ik)
    {
        auto a_tile = load_tile(a_copy_window);
        auto b_tile = load_tile(b_copy_window);

        move_tile_window(a_copy_window, {0, kMfmaBlockK});
        move_tile_window(b_copy_window, {0, kMfmaBlockK});

        store_tile(a_lds_store_window, a_tile);
        store_tile(b_lds_store_window, b_tile);

        block_sync_lds();
        block_gemm(c_block, err_block, a_lds_gemm_window, b_lds_gemm_window);
        block_sync_lds();
    }

    const auto c_global = make_naive_tensor_view<address_space_enum::global>(
        partial, make_tuple(M, N), make_tuple(N, 1), number<kKPack>{}, number<1>{});
    auto c_window = make_tile_window(
        c_global, make_tuple(number<kMfmaBlockM>{}, number<kMfmaBlockN>{}), {iM, iN});

    if(err_out)
    {
        const auto c_out = tile_elementwise_in(
            [](const auto& acc) { return ck_tile::type_convert<float>(acc); }, c_block);
        store_tile(c_window, c_out);

        const auto err_global = make_naive_tensor_view<address_space_enum::global>(
            err_out, make_tuple(M, N), make_tuple(N, 1), number<kKPack>{}, number<1>{});
        auto err_window = make_tile_window(
            err_global, make_tuple(number<kMfmaBlockM>{}, number<kMfmaBlockN>{}), {iM, iN});
        const auto e_out = tile_elementwise_in(
            [](const auto& acc) { return ck_tile::type_convert<float>(acc); }, err_block);
        store_tile(err_window, e_out);
    }
    else
    {
        AccumPolicy::finalize(c_block, err_block);
        const auto c_out = tile_elementwise_in(
            [](const auto& acc) { return ck_tile::type_convert<float>(acc); }, c_block);
        store_tile(c_window, c_out);
    }
}

template <typename AccumPolicy>
static void launch_mfma_tile_gemm_k4_impl(const float* A,
                                           const float* B,
                                           float* partial,
                                           int M,
                                           int N,
                                           int K_full,
                                           int k_offset,
                                           int K_tile,
                                           hipStream_t stream,
                                           float* err_out)
{
    int num_tile_m = M / kMfmaBlockM;
    int num_tile_n = N / kMfmaBlockN;
    int grid       = num_tile_m * num_tile_n;

    tile_gemm_mfma_k4_kernel<AccumPolicy><<<grid, kMfmaBlockSize, 0, stream>>>(
        A, B, partial, err_out, M, N, K_full, k_offset, K_tile, num_tile_n);
    HIP_CHECK_ERROR(hipGetLastError());
}

// =========================================================================
// K16 launch
// =========================================================================

template <typename AccumPolicy>
static void launch_mfma_tile_gemm_impl(const float* A,
                                       const float* B,
                                       float* partial,
                                       int M,
                                       int N,
                                       int K_full,
                                       int k_offset,
                                       int K_tile,
                                       hipStream_t stream,
                                       float* err_out)
{
    int num_tile_m = M / kMfmaBlockM;
    int num_tile_n = N / kMfmaBlockN;
    int grid       = num_tile_m * num_tile_n;

    tile_gemm_mfma_kernel<AccumPolicy><<<grid, kMfmaBlockSize, 0, stream>>>(
        A, B, partial, err_out, M, N, K_full, k_offset, K_tile, num_tile_n);
    HIP_CHECK_ERROR(hipGetLastError());
}

void launch_mfma_tile_gemm(const float* A,
                           const float* B,
                           float* partial,
                           int M,
                           int N,
                           int K_full,
                           int k_offset,
                           int K_tile,
                           hipStream_t stream,
                           AccumKLoopPolicy accum_policy,
                           float* err_out)
{
    switch(accum_policy)
    {
    case AccumKLoopPolicy::TwoSum:
        launch_mfma_tile_gemm_impl<AccumTwoSum>(
            A, B, partial, M, N, K_full, k_offset, K_tile, stream, err_out);
        break;
    case AccumKLoopPolicy::Kahan:
        launch_mfma_tile_gemm_impl<AccumKahan>(
            A, B, partial, M, N, K_full, k_offset, K_tile, stream, err_out);
        break;
    case AccumKLoopPolicy::Naive:
        launch_mfma_tile_gemm_impl<AccumNaive>(
            A, B, partial, M, N, K_full, k_offset, K_tile, stream, err_out);
        break;
    case AccumKLoopPolicy::NaiveK4:
    case AccumKLoopPolicy::TwoSumK4:
    case AccumKLoopPolicy::KahanK4:
    case AccumKLoopPolicy::VeltkampK4:
    case AccumKLoopPolicy::VeltkampNaiveK4:
        break;
    }
}

void launch_mfma_tile_gemm_k4(const float* A,
                              const float* B,
                              float* partial,
                              int M,
                              int N,
                              int K_full,
                              int k_offset,
                              int K_tile,
                              hipStream_t stream,
                              AccumKLoopPolicy accum_policy,
                              float* err_out)
{
    switch(accum_policy)
    {
    case AccumKLoopPolicy::TwoSumK4:
        launch_mfma_tile_gemm_k4_impl<AccumTwoSum>(
            A, B, partial, M, N, K_full, k_offset, K_tile, stream, err_out);
        break;
    case AccumKLoopPolicy::KahanK4:
        launch_mfma_tile_gemm_k4_impl<AccumKahan>(
            A, B, partial, M, N, K_full, k_offset, K_tile, stream, err_out);
        break;
    case AccumKLoopPolicy::NaiveK4:
        launch_mfma_tile_gemm_k4_impl<AccumNaive>(
            A, B, partial, M, N, K_full, k_offset, K_tile, stream, err_out);
        break;
    case AccumKLoopPolicy::Naive:
    case AccumKLoopPolicy::TwoSum:
    case AccumKLoopPolicy::Kahan:
    case AccumKLoopPolicy::VeltkampK4:
    case AccumKLoopPolicy::VeltkampNaiveK4:
        break;
    }
}

// =========================================================================
// Veltkamp K4 kernel — exact products + pluggable accumulation
// =========================================================================

template <typename AccumPolicy>
__global__ void
#if CK_TILE_WORKAROUND_LAUNCH_BOUND
    __launch_bounds__(kMfmaBlockSize)
#endif
        tile_gemm_mfma_veltkamp_k4_kernel(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ partial,
                                          float* __restrict__ err_out,
                                          ck_tile::index_t M,
                                          ck_tile::index_t N,
                                          ck_tile::index_t K_full,
                                          ck_tile::index_t k_offset,
                                          ck_tile::index_t K_tile,
                                          ck_tile::index_t num_tile_n)
{
    using namespace ck_tile;

    const index_t i_block = get_block_id();
    const index_t iM = __builtin_amdgcn_readfirstlane((i_block / num_tile_n) * kMfmaBlockM);
    const index_t iN = __builtin_amdgcn_readfirstlane((i_block % num_tile_n) * kMfmaBlockN);

    const auto a_global = make_naive_tensor_view<address_space_enum::global>(
        A, make_tuple(M, K_full), make_tuple(K_full, 1), number<kKPack>{}, number<1>{});
    const auto b_global = make_naive_tensor_view<address_space_enum::global>(
        B, make_tuple(N, K_full), make_tuple(K_full, 1), number<kKPack>{}, number<1>{});

    auto a_copy_window = make_tile_window(
        a_global, make_tuple(number<kMfmaBlockM>{}, number<kMfmaBlockK>{}),
        {iM, k_offset}, MakeADramDist());
    auto b_copy_window = make_tile_window(
        b_global, make_tuple(number<kMfmaBlockN>{}, number<kMfmaBlockK>{}),
        {iN, k_offset}, MakeBDramDist());

    __shared__ char lds_raw[kLdsBytes];
    float* p_a_lds = reinterpret_cast<float*>(lds_raw);
    float* p_b_lds = reinterpret_cast<float*>(lds_raw + kALdsBytes);

    constexpr auto a_lds_desc = MakeALdsDesc();
    constexpr auto b_lds_desc = MakeBLdsDesc();

    auto a_lds = make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_desc);
    auto b_lds = make_tensor_view<address_space_enum::lds>(p_b_lds, b_lds_desc);

    auto a_lds_store_window = make_tile_window(
        a_lds, make_tuple(number<kMfmaBlockM>{}, number<kMfmaBlockK>{}),
        {0, 0}, a_copy_window.get_tile_distribution());
    auto b_lds_store_window = make_tile_window(
        b_lds, make_tuple(number<kMfmaBlockN>{}, number<kMfmaBlockK>{}),
        {0, 0}, b_copy_window.get_tile_distribution());

    auto a_lds_gemm_window = make_tile_window(
        a_lds, make_tuple(number<kMfmaBlockM>{}, number<kMfmaBlockK>{}), {0, 0});
    auto b_lds_gemm_window = make_tile_window(
        b_lds, make_tuple(number<kMfmaBlockN>{}, number<kMfmaBlockK>{}), {0, 0});

    MfmaBlockGemmVeltkampK4<AccumPolicy> block_gemm;
    auto c_block = decltype(block_gemm)::MakeCBlockTile();
    auto err_block = decltype(c_block){};
    tile_elementwise_inout([](auto& c) { c = 0.0f; }, c_block);
    tile_elementwise_inout([](auto& c) { c = 0.0f; }, err_block);

    index_t num_k_inner = K_tile / kMfmaBlockK;

    for(index_t ik = 0; ik < num_k_inner; ++ik)
    {
        auto a_tile = load_tile(a_copy_window);
        auto b_tile = load_tile(b_copy_window);
        move_tile_window(a_copy_window, {0, kMfmaBlockK});
        move_tile_window(b_copy_window, {0, kMfmaBlockK});
        store_tile(a_lds_store_window, a_tile);
        store_tile(b_lds_store_window, b_tile);
        block_sync_lds();
        block_gemm(c_block, err_block, a_lds_gemm_window, b_lds_gemm_window);
        block_sync_lds();
    }

    const auto c_global = make_naive_tensor_view<address_space_enum::global>(
        partial, make_tuple(M, N), make_tuple(N, 1), number<kKPack>{}, number<1>{});
    auto c_window = make_tile_window(
        c_global, make_tuple(number<kMfmaBlockM>{}, number<kMfmaBlockN>{}), {iM, iN});

    if(err_out)
    {
        // Write c and err separately — no finalize, no precision loss
        const auto c_out = tile_elementwise_in(
            [](const auto& acc) { return ck_tile::type_convert<float>(acc); }, c_block);
        store_tile(c_window, c_out);

        const auto err_global = make_naive_tensor_view<address_space_enum::global>(
            err_out, make_tuple(M, N), make_tuple(N, 1), number<kKPack>{}, number<1>{});
        auto err_window = make_tile_window(
            err_global, make_tuple(number<kMfmaBlockM>{}, number<kMfmaBlockN>{}), {iM, iN});
        const auto e_out = tile_elementwise_in(
            [](const auto& acc) { return ck_tile::type_convert<float>(acc); }, err_block);
        store_tile(err_window, e_out);
    }
    else
    {
        AccumPolicy::finalize(c_block, err_block);
        const auto c_out = tile_elementwise_in(
            [](const auto& acc) { return ck_tile::type_convert<float>(acc); }, c_block);
        store_tile(c_window, c_out);
    }
}

template <typename AccumPolicy>
static void launch_mfma_tile_gemm_veltkamp_k4_impl(const float* A,
                                                    const float* B,
                                                    float* partial,
                                                    int M,
                                                    int N,
                                                    int K_full,
                                                    int k_offset,
                                                    int K_tile,
                                                    hipStream_t stream,
                                                    float* err_out)
{
    int num_tile_m = M / kMfmaBlockM;
    int num_tile_n = N / kMfmaBlockN;
    int grid       = num_tile_m * num_tile_n;

    tile_gemm_mfma_veltkamp_k4_kernel<AccumPolicy><<<grid, kMfmaBlockSize, 0, stream>>>(
        A, B, partial, err_out, M, N, K_full, k_offset, K_tile, num_tile_n);
    HIP_CHECK_ERROR(hipGetLastError());
}

void launch_mfma_tile_gemm_veltkamp_k4(const float* A,
                                       const float* B,
                                       float* partial,
                                       int M,
                                       int N,
                                       int K_full,
                                       int k_offset,
                                       int K_tile,
                                       hipStream_t stream,
                                       float* err_out)
{
    launch_mfma_tile_gemm_veltkamp_k4_impl<AccumTwoSum>(
        A, B, partial, M, N, K_full, k_offset, K_tile, stream, err_out);
}

void launch_mfma_tile_gemm_veltkamp_naive_k4(const float* A,
                                             const float* B,
                                             float* partial,
                                             int M,
                                             int N,
                                             int K_full,
                                             int k_offset,
                                             int K_tile,
                                             hipStream_t stream,
                                             float* err_out)
{
    launch_mfma_tile_gemm_veltkamp_k4_impl<AccumNaive>(
        A, B, partial, M, N, K_full, k_offset, K_tile, stream, err_out);
}

bool can_use_mfma(int M, int N, int K_tile)
{
    return (M % kMfmaBlockM == 0) && (N % kMfmaBlockN == 0) && (K_tile % kMfmaBlockK == 0);
}

} // namespace rocm_ck::test
