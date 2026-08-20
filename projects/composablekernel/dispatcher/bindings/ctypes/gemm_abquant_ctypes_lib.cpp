// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Gemm ABQuant (A+B block-scale) ctypes Library
 *
 * One .so per kernel variant; the kernel is force-included at compile time:
 *   hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE gemm_abquant_ctypes_lib.cpp
 * Force-include defines: SelectedKernel, KERNEL_NAME, ADataType, BDataType,
 * CDataType, QDataType, AccDataType, AQuantGroupSize, BQuantGroupSize.
 *
 * Direct launch -- SelectedKernel::launch(QuantGemmHostArgs, stream_config) is
 * called directly; no dispatcher registry is used.
 *
 * ABQuant quantizes BOTH A and B: aq_ptr AND bq_ptr are non-null. AQ is stored
 * RowMajor [M, QK_A] (QK_A = ceil(K / AGroupSizeK)); BQ is stored ColumnMajor
 * [QK_B, QN_B] (QK_B = ceil(K / BGroupSizeK), QN_B = ceil(N / BGroupSizeN);
 * BQLayout==ColumnMajor is enforced by a static_assert in gemm_quant_kernel.hpp).
 *
 * Shared infrastructure lives in quant_bridge_common.hpp; host-load primitives
 * in quant_bridge_shuffle.hpp. Memory model: host-pointer.
 */

#include <hip/hip_runtime.h>
#include <cstdint>
#include <iostream>
#include <type_traits>

#include "quant_bridge_common.hpp"
#include "quant_bridge_shuffle.hpp"

extern "C" {

QUANT_BRIDGE_C_API()

/**
 * Run ABQuant GEMM:
 *   C[M,N] = dequant(A[M,K], AQ[M,QK_A]) @ dequant(B[K,N], BQ[QK_B,QN_B])
 * A, B, AQ, BQ, C are host pointers; device memory is managed internally. QK_A,
 * QK_B, QN_B are the A K-group / B K-group / B N-group counts. time_ms is
 * optional. Returns 0 on success, negative on error.
 */
int dispatcher_run_abquant_gemm(const void* A,
                                const void* B,
                                const void* AQ,
                                const void* BQ,
                                void* C,
                                int64_t M,
                                int64_t N,
                                int64_t K,
                                int64_t stride_A,
                                int64_t stride_B,
                                int64_t stride_AQ,
                                int64_t stride_BQ,
                                int64_t stride_C,
                                int64_t QK_A,
                                int64_t QK_B,
                                int64_t QN_B,
                                int k_batch,
                                float* time_ms)
{
    using namespace quant_bridge;
    const char* kFn = "dispatcher_run_abquant_gemm";

    if(!check_initialized(kFn, g_initialized) || !check_non_null(kFn, {A, B, AQ, BQ, C}) ||
       !check_positive_dims(kFn, {M, N, K, QK_A, QK_B, QN_B}))
        return -1;

    // Graceful reject: PreshuffleB is not supported for fp4 (BDataType==pk_fp4_t),
    // exactly as Old-TE THROWS in run_gemm_quant_example.inc:994-1001. The fp4
    // preshuffle host path would otherwise allocate/copy a mis-sized B buffer and
    // heap-corrupt. Compile-time branch: can only fire in an fp4 PreshuffleB .so.
    if constexpr(SelectedKernel::PreshuffleB && std::is_same_v<BDataType, ck_tile::pk_fp4_t>)
    {
        std::cerr << kFn
                  << ": Preshuffling weight matrix is not supported for bf16_fp4_gemm "
                     "(matches Old-TE reject)\n";
        return -3;
    }

    if(!validate_supported_arch(kFn))
        return -1;

    // Validate QK_A/QK_B/QN_B against the compile-time quant group sizes.
    {
        const int64_t expected_QK_A =
            (K + static_cast<int64_t>(AQuantGroupSize::kK) - 1) / AQuantGroupSize::kK;
        const int64_t expected_QK_B =
            (K + static_cast<int64_t>(BQuantGroupSize::kK) - 1) / BQuantGroupSize::kK;
        const int64_t expected_QN_B =
            (N + static_cast<int64_t>(BQuantGroupSize::kN) - 1) / BQuantGroupSize::kN;
        if(QK_A != expected_QK_A || QK_B != expected_QK_B || QN_B != expected_QN_B)
        {
            std::cerr << kFn << ": QK_A/QK_B/QN_B mismatch. Got (" << QK_A << ", " << QK_B << ", "
                      << QN_B << "), expected (" << expected_QK_A << ", " << expected_QK_B << ", "
                      << expected_QN_B << ") for K=" << K << " N=" << N
                      << " with AQuantGroupSize kK=" << AQuantGroupSize::kK
                      << " BQuantGroupSize kK=" << BQuantGroupSize::kK
                      << " kN=" << BQuantGroupSize::kN << "\n";
            return -1;
        }
    }

    // Only packed layouts are supported. AQ leading dim depends on AQLayout: the
    // n=128 EightWaves fast path uses ColumnMajor [M, QK_A] -> M; otherwise
    // RowMajor -> QK_A. BQ is ColumnMajor [QK_B, QN_B] -> leading dim QK_B.
    const int64_t expected_stride_AQ = SelectedKernel::AQIsColumnMajor ? M : QK_A;
    if(stride_A != K || stride_B != K || stride_AQ != expected_stride_AQ || stride_BQ != QK_B ||
       stride_C != N)
    {
        std::cerr << kFn << ": non-packed strides are not supported. Expected stride_A=" << K
                  << " stride_B=" << K << " stride_AQ=" << expected_stride_AQ
                  << " stride_BQ=" << QK_B << " stride_C=" << N << ", got stride_A=" << stride_A
                  << " stride_B=" << stride_B << " stride_AQ=" << stride_AQ
                  << " stride_BQ=" << stride_BQ << " stride_C=" << stride_C << "\n";
        return -1;
    }

    const BDataType* B_host  = static_cast<const BDataType*>(B);
    const QDataType* AQ_host = static_cast<const QDataType*>(AQ);
    const QDataType* BQ_host = static_cast<const QDataType*>(BQ);

    DeviceBuffer<ADataType> A_dev;
    DeviceBuffer<BDataType> B_dev;
    DeviceBuffer<QDataType> AQ_dev;
    DeviceBuffer<QDataType> BQ_dev;
    DeviceBuffer<CDataType> C_dev;
    BRIDGE_HIP_CHECK(kFn, A_dev.allocate(elements_to_bytes<ADataType>(M * K)));
    BRIDGE_HIP_CHECK(kFn, B_dev.allocate(elements_to_bytes<BDataType>(K * N)));
    BRIDGE_HIP_CHECK(kFn, AQ_dev.allocate(elements_to_bytes<QDataType>(M * QK_A)));
    BRIDGE_HIP_CHECK(kFn, BQ_dev.allocate(elements_to_bytes<QDataType>(QK_B * QN_B)));
    BRIDGE_HIP_CHECK(kFn, C_dev.allocate(elements_to_bytes<CDataType>(M * N)));

    BRIDGE_HIP_CHECK(
        kFn, hipMemcpy(A_dev, A, elements_to_bytes<ADataType>(M * K), hipMemcpyHostToDevice));

    // Host-side B prep: PreshuffleB kernels shuffle B first (shuffle_b_permuteN
    // when TiledMMAPermuteN && kN==1, else shuffle_b); plain copy otherwise.
    if constexpr(SelectedKernel::PreshuffleB)
    {
        auto b_k_n = load_host_tensor<false>(
            B_host, static_cast<int>(K), static_cast<int>(N), static_cast<int>(K));
        constexpr bool use_permute_n = SelectedKernel::TiledMMAPermuteN && (BGroupSizeN == 1);
        auto b_shuffled              = [&]() {
            if constexpr(use_permute_n)
                return ck_tile::shuffle_b_permuteN<typename SelectedKernel::BShuffleConfig>(b_k_n);
            else
                return ck_tile::shuffle_b<typename SelectedKernel::BShuffleConfig>(b_k_n);
        }();
        BRIDGE_HIP_CHECK(kFn,
                         hipMemcpy(B_dev,
                                   b_shuffled.data(),
                                   elements_to_bytes<BDataType>(K * N),
                                   hipMemcpyHostToDevice));
    }
    else
    {
        BRIDGE_HIP_CHECK(
            kFn, hipMemcpy(B_dev, B, elements_to_bytes<BDataType>(K * N), hipMemcpyHostToDevice));
    }

    // Host-side AQ prep: APreshuffleQuant reorders AQ (shuffle_aq); else plain.
    if constexpr(SelectedKernel::APreshuffleQuant)
    {
        const int block_aq_k =
            static_cast<int>(SelectedKernel::TileK) / static_cast<int>(AQuantGroupSize::kK);
        auto aq_h = load_host_tensor<true>(
            AQ_host, static_cast<int>(M), static_cast<int>(QK_A), static_cast<int>(QK_A));
        auto aq_shuffled = ck_tile::shuffle_aq(&aq_h, block_aq_k);
        BRIDGE_HIP_CHECK(kFn,
                         hipMemcpy(AQ_dev,
                                   aq_shuffled.data(),
                                   elements_to_bytes<QDataType>(M * QK_A),
                                   hipMemcpyHostToDevice));
    }
    else
    {
        BRIDGE_HIP_CHECK(
            kFn,
            hipMemcpy(AQ_dev, AQ, elements_to_bytes<QDataType>(M * QK_A), hipMemcpyHostToDevice));
    }

    // Host-side BQ prep (run_gemm_quant_example.inc:799-825): three cases as in
    // bquant -- bq_permuteN (+ optional shuffle_bq), shuffle_bq only, or plain.
    // Only build a host tensor when a permute/shuffle is applied; the plain case
    // copies raw BQ straight to device.
    constexpr bool bq_use_permute_n =
        SelectedKernel::PreshuffleB && SelectedKernel::TiledMMAPermuteN && (BGroupSizeN == 1);
    {
        const std::size_t bq_bytes = elements_to_bytes<QDataType>(QK_B * QN_B);
        if constexpr(bq_use_permute_n || SelectedKernel::BPreshuffleQuant)
        {
            const int block_bq_k =
                static_cast<int>(SelectedKernel::TileK) / static_cast<int>(BQuantGroupSize::kK);
            auto bq_h = load_host_tensor<false>(
                BQ_host, static_cast<int>(QK_B), static_cast<int>(QN_B), static_cast<int>(QK_B));
            if constexpr(bq_use_permute_n)
            {
                auto bq_permuted = ck_tile::bq_permuteN<typename SelectedKernel::BShuffleConfig>(
                    bq_h, static_cast<ck_tile::index_t>(BGroupSizeN));
                if constexpr(SelectedKernel::BPreshuffleQuant)
                {
                    auto bq_shuffled = ck_tile::shuffle_bq(&bq_permuted, block_bq_k);
                    BRIDGE_HIP_CHECK(
                        kFn,
                        hipMemcpy(BQ_dev, bq_shuffled.data(), bq_bytes, hipMemcpyHostToDevice));
                }
                else
                {
                    BRIDGE_HIP_CHECK(
                        kFn,
                        hipMemcpy(BQ_dev, bq_permuted.data(), bq_bytes, hipMemcpyHostToDevice));
                }
            }
            else // BPreshuffleQuant only
            {
                auto bq_shuffled = ck_tile::shuffle_bq(&bq_h, block_bq_k);
                BRIDGE_HIP_CHECK(
                    kFn, hipMemcpy(BQ_dev, bq_shuffled.data(), bq_bytes, hipMemcpyHostToDevice));
            }
        }
        else
        {
            BRIDGE_HIP_CHECK(kFn, hipMemcpy(BQ_dev, BQ, bq_bytes, hipMemcpyHostToDevice));
        }
    }
    BRIDGE_HIP_CHECK(kFn, hipMemset(C_dev, 0, elements_to_bytes<CDataType>(M * N)));

    // ABQuant: both aq_ptr and bq_ptr are non-null.
    ck_tile::QuantGemmHostArgs args;
    args.a_ptr     = A_dev;
    args.b_ptr     = B_dev;
    args.aq_ptr    = AQ_dev;
    args.bq_ptr    = BQ_dev;
    args.c_ptr     = C_dev;
    args.k_batch   = k_batch;
    args.M         = static_cast<ck_tile::index_t>(M);
    args.N         = static_cast<ck_tile::index_t>(N);
    args.K         = static_cast<ck_tile::index_t>(K);
    args.QK_A      = static_cast<ck_tile::index_t>(QK_A);
    args.QK_B      = static_cast<ck_tile::index_t>(QK_B);
    args.stride_A  = static_cast<ck_tile::index_t>(stride_A);
    args.stride_B  = static_cast<ck_tile::index_t>(stride_B);
    args.stride_C  = static_cast<ck_tile::index_t>(stride_C);
    args.stride_AQ = static_cast<ck_tile::index_t>(stride_AQ);
    args.stride_BQ = static_cast<ck_tile::index_t>(stride_BQ);

    return launch_and_copyback<SelectedKernel, CDataType>(
        kFn, args, C, C_dev, static_cast<std::size_t>(M) * N, time_ms);
}

} // extern "C"
