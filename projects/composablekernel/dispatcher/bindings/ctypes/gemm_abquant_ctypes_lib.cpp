// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Gemm ABQuant (A+B block-scale) ctypes Library
 *
 * Provides a C API for Python ctypes integration. One .so is compiled per
 * kernel variant; the kernel is force-included at compile time:
 *   hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE gemm_abquant_ctypes_lib.cpp
 *
 * Force-include defines (from generated kernel header):
 *   SelectedKernel, KERNEL_NAME
 *   ADataType, BDataType, CDataType, QDataType, AccDataType
 *   AQuantGroupSize, BQuantGroupSize
 *
 * Design: direct launch -- SelectedKernel::launch(QuantGemmHostArgs, stream_config) is
 * called directly. No dispatcher registry is used: ABQuant kernels take QuantGemmHostArgs,
 * which is incompatible with the GeneratedTileKernelInstance::run() signature used by
 * the dispatcher's registry backend.
 *
 * ABQuant quantizes BOTH A and B: aq_ptr AND bq_ptr are non-null. AQ is stored
 * RowMajor [M, QK_A] with QK_A = ceil(K / AGroupSizeK); BQ is stored ColumnMajor
 * [QK_B, QN_B] with QK_B = ceil(K / BGroupSizeK), QN_B = ceil(N / BGroupSizeN)
 * (BQLayout==ColumnMajor is enforced by a static_assert in gemm_quant_kernel.hpp;
 * see the stride handling at lines ~194-196 below).
 *
 * Memory model: host-pointer (this library owns hipMalloc/hipMemcpy/hipFree).
 */

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "ck_tile/host/tensor_shuffle_utils.hpp"

// Kernel header force-included via -include compiler flag.
// Defines: ADataType, BDataType, CDataType, QDataType, AccDataType,
//          AQuantGroupSize, BQuantGroupSize, SelectedKernel, KERNEL_NAME

// Compute the byte count for N logical elements of type T.
// For packed types (pk_int4_t, pk_fp4_t) PackedSize=2, so N logical values
// occupy N/2 bytes even though sizeof(T)==1.  For all other types PackedSize=1.
template <typename T>
static constexpr std::size_t elements_to_bytes(std::size_t n)
{
    return n * sizeof(T) / ck_tile::numeric_traits<T>::PackedSize;
}

// GPU architecture is derived from the running device at launch time (see the
// runtime check in dispatcher_run_abquant_gemm) rather than assumed at compile
// time -- do not hardcode a default architecture here.

static bool g_initialized = false;

#define HIP_CHECK(call)                                                                         \
    {                                                                                           \
        hipError_t _err = (call);                                                               \
        if(_err != hipSuccess)                                                                  \
        {                                                                                       \
            std::cerr << "HIP error: " << hipGetErrorString(_err) << " at " << __FILE__ << ":" \
                      << __LINE__ << "\n";                                                      \
            cleanup();                                                                          \
            return -1;                                                                          \
        }                                                                                       \
    }

extern "C" {

/**
 * Initialize the ctypes lib. Must be called before dispatcher_run_abquant_gemm.
 *
 * This library uses a single-kernel-per-.so model: SelectedKernel is
 * force-included at compile time and invoked directly via SelectedKernel::launch().
 * No dispatcher registry is involved -- ABQuant kernels require QuantGemmHostArgs
 * which is incompatible with the GeneratedTileKernelInstance::run() signature that
 * the dispatcher's registry backend uses.
 *
 * Returns 0 on success.
 */
int dispatcher_initialize()
{
    if(g_initialized)
        return 0;
    g_initialized = true;
    return 0;
}

/**
 * Run ABQuant GEMM:
 *   C[M,N] = dequant(A[M,K], AQ[M,QK_A]) @ dequant(B[K,N], BQ[QK_B,QN_B])
 *
 * A, B, AQ, BQ, C are host pointers. This function manages device memory internally.
 *
 * Parameters:
 *   A, B, AQ, BQ, C - host data pointers
 *   M, N, K         - matrix dimensions
 *   stride_A        - leading dimension of A (row-major: K)
 *   stride_B        - leading dimension of B (col-major: K)
 *   stride_AQ       - leading dimension of AQ (row-major: QK_A)
 *   stride_BQ       - leading dimension of BQ (col-major: QK_B)
 *   stride_C        - leading dimension of C (row-major: N)
 *   QK_A            - number of A K-groups = ceil(K / AGroupSizeK)
 *   QK_B            - number of B K-groups = ceil(K / BGroupSizeK)
 *   QN_B            - number of B N-groups = ceil(N / BGroupSizeN)
 *   k_batch         - split-K factor (1 = no split)
 *   time_ms         - output: kernel execution time in ms (may be NULL)
 *
 * Returns 0 on success, negative on error.
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
    if(!g_initialized)
    {
        std::cerr << "dispatcher_run_abquant_gemm: not initialized\n";
        return -1;
    }
    if(!A || !B || !AQ || !BQ || !C)
    {
        std::cerr << "dispatcher_run_abquant_gemm: null pointer argument\n";
        return -1;
    }
    if(M <= 0 || N <= 0 || K <= 0 || QK_A <= 0 || QK_B <= 0 || QN_B <= 0)
    {
        std::cerr << "dispatcher_run_abquant_gemm: invalid dimensions\n";
        return -1;
    }

    // Graceful reject: PreshuffleB is not supported for fp4 (BDataType==pk_fp4_t),
    // exactly as Old-TE THROWS in run_gemm_quant_example.inc:994-1001
    //   "Preshuffling weight matrix is not supported for AQuant, RowColQuant or
    //    bf16_fp4_gemm".
    // The fp4 preshuffle host path would otherwise allocate/copy a mis-sized B
    // buffer and heap-corrupt (malloc abort). Return an error code instead so the
    // Python runner raises rather than crashing the process. This is a compile-time
    // branch: it can only ever fire in an fp4 PreshuffleB .so.
    if constexpr(SelectedKernel::PreshuffleB && std::is_same_v<BDataType, ck_tile::pk_fp4_t>)
    {
        std::cerr << "dispatcher_run_abquant_gemm: Preshuffling weight matrix is not "
                     "supported for bf16_fp4_gemm (matches Old-TE reject)\n";
        return -3;
    }

    // Derive the GPU architecture from the running device (do not assume one at
    // compile time) and reject unsupported archs, per review feedback.
    {
        int dev = 0;
        hipDeviceProp_t props{};
        if(hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&props, dev) != hipSuccess)
        {
            std::cerr << "dispatcher_run_abquant_gemm: could not query device architecture\n";
            return -1;
        }
        const std::string arch(props.gcnArchName);
        if(arch.rfind("gfx950", 0) != 0 && arch.rfind("gfx942", 0) != 0)
        {
            std::cerr << "dispatcher_run_abquant_gemm: unsupported GPU architecture '" << arch
                      << "' (supported: gfx942, gfx950)\n";
            return -1;
        }
    }

    // Validate that the caller's QK_A/QK_B/QN_B match the compile-time quant group
    // sizes baked into this .so.  A mismatch means the AQ/BQ device buffers would be
    // allocated with the wrong size while the kernel indexes them with different
    // strides.
    {
        const int64_t expected_QK_A =
            (K + static_cast<int64_t>(AQuantGroupSize::kK) - 1) / AQuantGroupSize::kK;
        const int64_t expected_QK_B =
            (K + static_cast<int64_t>(BQuantGroupSize::kK) - 1) / BQuantGroupSize::kK;
        const int64_t expected_QN_B =
            (N + static_cast<int64_t>(BQuantGroupSize::kN) - 1) / BQuantGroupSize::kN;
        if(QK_A != expected_QK_A || QK_B != expected_QK_B || QN_B != expected_QN_B)
        {
            std::cerr << "dispatcher_run_abquant_gemm: QK_A/QK_B/QN_B mismatch. " << "Got (" << QK_A
                      << ", " << QK_B << ", " << QN_B << "), " << "expected (" << expected_QK_A
                      << ", " << expected_QK_B << ", " << expected_QN_B << ") " << "for K=" << K
                      << " N=" << N << " with AQuantGroupSize kK=" << AQuantGroupSize::kK
                      << " BQuantGroupSize kK=" << BQuantGroupSize::kK
                      << " kN=" << BQuantGroupSize::kN << "\n";
            return -1;
        }
    }

    // This implementation only supports packed (contiguous) layouts.
    // Device buffers are allocated and copied as M*K, K*N, M*QK_A, QK_B*QN_B, M*N
    // packed arrays.  Non-packed strides would cause the kernel to index into a
    // differently-sized buffer, producing incorrect results or OOB accesses.
    // AQ leading dim depends on AQLayout: RowMajor [M, QK_A] -> QK_A; the n=128
    //   EightWaves fast path uses ColumnMajor [M, QK_A] -> M (StrideAQ=M, matching
    //   Old-TE run_gemm_quant_example.inc:1013-1021 + get_default_stride).
    // BQ is ColumnMajor [QK_B, QN_B]  -> leading dim = QK_B (ABQuant kernel
    //                                    requires ColumnMajor BQ; see static_assert
    //                                    in gemm_quant_kernel.hpp).
    const int64_t expected_stride_AQ = SelectedKernel::AQIsColumnMajor ? M : QK_A;
    if(stride_A != K || stride_B != K || stride_AQ != expected_stride_AQ || stride_BQ != QK_B ||
       stride_C != N)
    {
        std::cerr << "dispatcher_run_abquant_gemm: non-packed strides are not supported. "
                  << "Expected stride_A=" << K << " stride_B=" << K
                  << " stride_AQ=" << expected_stride_AQ << " stride_BQ=" << QK_B
                  << " stride_C=" << N << ", got stride_A=" << stride_A << " stride_B=" << stride_B
                  << " stride_AQ=" << stride_AQ << " stride_BQ=" << stride_BQ
                  << " stride_C=" << stride_C << "\n";
        return -1;
    }

    const ADataType* A_host  = static_cast<const ADataType*>(A);
    const BDataType* B_host  = static_cast<const BDataType*>(B);
    const QDataType* AQ_host = static_cast<const QDataType*>(AQ);
    const QDataType* BQ_host = static_cast<const QDataType*>(BQ);
    CDataType* C_host        = static_cast<CDataType*>(C);

    ADataType* A_dev  = nullptr;
    BDataType* B_dev  = nullptr;
    QDataType* AQ_dev = nullptr;
    QDataType* BQ_dev = nullptr;
    CDataType* C_dev  = nullptr;

    auto cleanup = [&]() {
        if(A_dev)
            (void)hipFree(A_dev);
        if(B_dev)
            (void)hipFree(B_dev);
        if(AQ_dev)
            (void)hipFree(AQ_dev);
        if(BQ_dev)
            (void)hipFree(BQ_dev);
        if(C_dev)
            (void)hipFree(C_dev);
    };

    // Allocate device buffers.
    // A/B may be packed types (pk_fp4_t): 2 logical values per byte.
    // elements_to_bytes<T>(n) handles the packed case via numeric_traits::PackedSize.
    HIP_CHECK(hipMalloc(&A_dev,  elements_to_bytes<ADataType>(M * K)));
    HIP_CHECK(hipMalloc(&B_dev,  elements_to_bytes<BDataType>(K * N)));
    HIP_CHECK(hipMalloc(&AQ_dev, elements_to_bytes<QDataType>(M * QK_A)));
    HIP_CHECK(hipMalloc(&BQ_dev, elements_to_bytes<QDataType>(QK_B * QN_B)));
    HIP_CHECK(hipMalloc(&C_dev,  elements_to_bytes<CDataType>(M * N)));

    // Copy A input to device.
    HIP_CHECK(hipMemcpy(A_dev, A_host, elements_to_bytes<ADataType>(M * K), hipMemcpyHostToDevice));

    // Copy the B weight matrix to device. For PreshuffleB kernels the B matrix
    // must be pre-shuffled on host FIRST, exactly as Old-TE does before its device
    // copy (run_gemm_quant_example.inc:770-789):
    //   * shuffle_b_permuteN<GemmConfig>(B) when TiledMMAPermuteN && kN == 1
    //   * shuffle_b<GemmConfig>(B)          otherwise
    // The kernel reads B in this interleaved layout; without the shuffle the
    // PreshuffleB kernels produce garbage (max_rel ~50-78 on gfx950).
    if constexpr(SelectedKernel::PreshuffleB)
    {
        // B is supplied ColumnMajor [K, N]; shuffle_b indexes it as [K rows, N cols].
        ck_tile::HostTensor<BDataType> b_k_n(
            ck_tile::host_tensor_descriptor(static_cast<int>(K),
                                            static_cast<int>(N),
                                            static_cast<int>(K),
                                            ck_tile::bool_constant<false>{} /*col-major*/));
        std::copy(B_host, B_host + K * N, b_k_n.begin());

        constexpr bool use_permute_n = SelectedKernel::TiledMMAPermuteN && (BGroupSizeN == 1);
        auto b_shuffled              = [&]() {
            if constexpr(use_permute_n)
                return ck_tile::shuffle_b_permuteN<typename SelectedKernel::BShuffleConfig>(b_k_n);
            else
                return ck_tile::shuffle_b<typename SelectedKernel::BShuffleConfig>(b_k_n);
        }();

        HIP_CHECK(hipMemcpy(B_dev, b_shuffled.data(), elements_to_bytes<BDataType>(K * N), hipMemcpyHostToDevice));
    }
    else
    {
        HIP_CHECK(hipMemcpy(B_dev, B_host, elements_to_bytes<BDataType>(K * N), hipMemcpyHostToDevice));
    }

    // Apply AQ preshuffle when required -- mirrors the profiler's shuffle_aq path.
    // APreshuffleQuant reorders AQ in host memory before the device copy so the
    // kernel finds the scale values in the interleaved layout it expects.
    if constexpr(SelectedKernel::APreshuffleQuant)
    {
        constexpr int block_aq_k =
            static_cast<int>(SelectedKernel::TileK) / static_cast<int>(AQuantGroupSize::kK);
        ck_tile::HostTensor<QDataType> aq_h(
            ck_tile::host_tensor_descriptor(static_cast<int>(M),
                                            static_cast<int>(QK_A),
                                            static_cast<int>(QK_A),
                                            ck_tile::bool_constant<true>{} /*row-major*/));
        std::copy(AQ_host, AQ_host + M * QK_A, aq_h.begin());
        auto aq_shuffled = ck_tile::shuffle_aq(&aq_h, block_aq_k);
        HIP_CHECK(hipMemcpy(AQ_dev, aq_shuffled.data(), elements_to_bytes<QDataType>(M * QK_A), hipMemcpyHostToDevice));
    }
    else
    {
        HIP_CHECK(hipMemcpy(AQ_dev, AQ_host, elements_to_bytes<QDataType>(M * QK_A), hipMemcpyHostToDevice));
    }

    // Apply the BQ scale-tensor preshuffle when required. This mirrors Old-TE's
    // three-way branch in run_gemm_quant_example.inc:799-825 exactly:
    //   1. PreshuffleB && TiledMMAPermuteN && kN==1 (the "permute_n" kernel):
    //        bq_permuteN<BShuffleConfig>(BQ, kN) FIRST, then shuffle_bq if
    //        BPreshuffleQuant. The permute_n B-epilogue riffles the N columns, so
    //        the BQ scales MUST be permuted the same way or every column except the
    //        first is read with the wrong scale (max_rel ~large, col 0 exact).
    //   2. else if BPreshuffleQuant: shuffle_bq only.
    //   3. else: no shuffle.
    // BQ arrives ColumnMajor [QK_B, QN_B]: BQ[k,n] at offset n*QK_B + k.
    constexpr bool bq_use_permute_n =
        SelectedKernel::PreshuffleB && SelectedKernel::TiledMMAPermuteN && (BGroupSizeN == 1);
    if constexpr(bq_use_permute_n)
    {
        ck_tile::HostTensor<QDataType> bq_h(
            ck_tile::host_tensor_descriptor(static_cast<int>(QK_B),
                                            static_cast<int>(QN_B),
                                            static_cast<int>(QK_B),
                                            ck_tile::bool_constant<false>{} /*col-major*/));
        std::copy(BQ_host, BQ_host + QK_B * QN_B, bq_h.begin());
        // Old-TE: bq_permuteN<GemmConfig>(*bq_tensor_ptr, BQuantGroupSize::kN).
        auto bq_permuted = ck_tile::bq_permuteN<typename SelectedKernel::BShuffleConfig>(
            bq_h, static_cast<ck_tile::index_t>(BGroupSizeN));
        auto bq_final = [&]() {
            if constexpr(SelectedKernel::BPreshuffleQuant)
            {
                constexpr int block_bq_k =
                    static_cast<int>(SelectedKernel::TileK) / static_cast<int>(BQuantGroupSize::kK);
                return ck_tile::shuffle_bq(&bq_permuted, block_bq_k);
            }
            else
            {
                return bq_permuted;
            }
        }();
        HIP_CHECK(hipMemcpy(BQ_dev, bq_final.data(), elements_to_bytes<QDataType>(QK_B * QN_B), hipMemcpyHostToDevice));
    }
    else if constexpr(SelectedKernel::BPreshuffleQuant)
    {
        constexpr int block_bq_k =
            static_cast<int>(SelectedKernel::TileK) / static_cast<int>(BQuantGroupSize::kK);
        ck_tile::HostTensor<QDataType> bq_h(
            ck_tile::host_tensor_descriptor(static_cast<int>(QK_B),
                                            static_cast<int>(QN_B),
                                            static_cast<int>(QK_B),
                                            ck_tile::bool_constant<false>{} /*col-major*/));
        std::copy(BQ_host, BQ_host + QK_B * QN_B, bq_h.begin());
        auto bq_shuffled = ck_tile::shuffle_bq(&bq_h, block_bq_k);
        HIP_CHECK(hipMemcpy(BQ_dev, bq_shuffled.data(), elements_to_bytes<QDataType>(QK_B * QN_B), hipMemcpyHostToDevice));
    }
    else
    {
        HIP_CHECK(hipMemcpy(BQ_dev, BQ_host, elements_to_bytes<QDataType>(QK_B * QN_B), hipMemcpyHostToDevice));
    }

    HIP_CHECK(hipMemset(C_dev, 0, elements_to_bytes<CDataType>(M * N)));

    // Build QuantGemmHostArgs -- both aq_ptr and bq_ptr are non-null for ABQuant.
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

    const bool do_time = (time_ms != nullptr);
    // When timing is requested use GPU timer with warmup (cold_niters=3, nrepeat=10).
    // Otherwise run once with no overhead.
    ck_tile::stream_config stream_cfg{
        nullptr,          // stream_id_
        do_time,          // time_kernel_
        0,                // log_level_
        do_time ? 3 : 0,  // cold_niters_
        do_time ? 10 : 1, // nrepeat_
        do_time,          // is_gpu_timer_
        false,            // flush_cache_
        1,                // rotating_count_
    };

    float exec_time = SelectedKernel::launch(args, stream_cfg);

    if(exec_time < 0.0f)
    {
        std::cerr << "dispatcher_run_abquant_gemm: kernel reported unsupported args\n";
        cleanup();
        return -2;
    }

    // Copy result back
    HIP_CHECK(hipMemcpy(C_host, C_dev, elements_to_bytes<CDataType>(M * N), hipMemcpyDeviceToHost));

    if(time_ms)
        *time_ms = exec_time;

    cleanup();
    return 0;
}

/**
 * Return the compile-time KERNEL_NAME of the force-included kernel.
 */
const char* dispatcher_get_kernel_name() { return KERNEL_NAME; }

/**
 * Initialize dispatcher (alias kept for consistency with gemm_ctypes_lib).
 */
int dispatcher_init() { return dispatcher_initialize(); }

/**
 * Number of kernels in this .so (always 1: the force-included SelectedKernel).
 */
int dispatcher_get_kernel_count() { return 1; }

/**
 * Release resources.
 */
void dispatcher_cleanup() { g_initialized = false; }

} // extern "C"
