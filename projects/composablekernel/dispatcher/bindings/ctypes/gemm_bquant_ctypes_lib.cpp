// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Gemm BQuant ctypes Library (non-grouped, block-scale GEMM)
 *
 * C API for the plain (non-grouped) B-only quantized block-scale GEMM operator
 * from example/ck_tile/38_block_scale_gemm. Distinct from the multi-problem
 * grouped_gemm_bquant bridge -- this handles a single GEMM problem whose weight
 * matrix B is quantized with grouped scales BQ.
 *
 * One .so is compiled per kernel variant; the kernel is force-included at
 * compile time:
 *   hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE gemm_bquant_ctypes_lib.cpp
 *
 * Force-include defines (from generated kernel header):
 *   SelectedKernel, KERNEL_NAME
 *   ADataType, BDataType, CDataType, QDataType, AccDataType, QuantGroupSize
 *
 * Design: direct launch -- SelectedKernel::launch(QuantGemmHostArgs, stream_config) is
 * called directly. No dispatcher registry is used: BQuant kernels take QuantGemmHostArgs,
 * which is incompatible with the GeneratedTileKernelInstance::run() signature used by
 * the dispatcher's registry backend.
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
#include "ck_tile/host/permute_pk_int4.hpp"

// Kernel header force-included via -include compiler flag.
// Defines: ADataType, BDataType, CDataType, QDataType, AccDataType,
//          QuantGroupSize, SelectedKernel, KERNEL_NAME

// Compute the byte count for N logical elements of type T.
// For packed types (pk_int4_t, pk_fp4_t) PackedSize=2, so N logical values
// occupy N/2 bytes even though sizeof(T)==1.  For all other types PackedSize=1.
template <typename T>
static constexpr std::size_t elements_to_bytes(std::size_t n)
{
    return n * sizeof(T) / ck_tile::numeric_traits<T>::PackedSize;
}

// GPU architecture is derived from the running device at launch time (see the
// runtime check in dispatcher_run_bquant_gemm) rather than assumed at compile
// time -- do not hardcode a default architecture here.

static bool g_initialized = false;

#define HIP_CHECK(call)                                                                        \
    {                                                                                          \
        hipError_t _err = (call);                                                              \
        if(_err != hipSuccess)                                                                 \
        {                                                                                      \
            std::cerr << "HIP error: " << hipGetErrorString(_err) << " at " << __FILE__ << ":" \
                      << __LINE__ << "\n";                                                     \
            cleanup();                                                                         \
            return -1;                                                                         \
        }                                                                                      \
    }

extern "C" {

/**
 * Initialize the ctypes lib. Must be called before dispatcher_run_bquant_gemm.
 *
 * This library uses a single-kernel-per-.so model: SelectedKernel is
 * force-included at compile time and invoked directly via SelectedKernel::launch().
 * No dispatcher registry is involved -- BQuant kernels require QuantGemmHostArgs
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
 * Run non-grouped BQuant GEMM:
 *   C[M,N] = A[M,K] @ dequant(B[K,N], BQ[ceil(K/gK), ceil(N/gN)])
 *
 * A, B, BQ, C are host pointers. This function manages device memory internally.
 *
 * Parameters:
 *   A, B, BQ, C  - host data pointers
 *   M, N, K      - matrix dimensions
 *   stride_A     - leading dimension of A (row-major: K)
 *   stride_B     - leading dimension of B (col-major: K)
 *   stride_BQ    - leading dimension of BQ (col-major: ceil(K/gK) = QK_B)
 *   stride_C     - leading dimension of C (row-major: N)
 *   QK_B         - number of K-groups = ceil(K / quant_group_k)
 *   QN_B         - number of N-groups = ceil(N / quant_group_n)
 *   k_batch      - split-K factor (1 = no split)
 *   time_ms      - output: kernel execution time in ms (may be NULL)
 *
 * Returns 0 on success, negative on error.
 */
int dispatcher_run_bquant_gemm(const void* A,
                               const void* B,
                               const void* BQ,
                               void* C,
                               int64_t M,
                               int64_t N,
                               int64_t K,
                               int64_t stride_A,
                               int64_t stride_B,
                               int64_t stride_BQ,
                               int64_t stride_C,
                               int64_t QK_B,
                               int64_t QN_B,
                               int k_batch,
                               float* time_ms)
{
    if(!g_initialized)
    {
        std::cerr << "dispatcher_run_bquant_gemm: not initialized\n";
        return -1;
    }
    if(!A || !B || !BQ || !C)
    {
        std::cerr << "dispatcher_run_bquant_gemm: null pointer argument\n";
        return -1;
    }
    if(M <= 0 || N <= 0 || K <= 0 || QK_B <= 0 || QN_B <= 0)
    {
        std::cerr << "dispatcher_run_bquant_gemm: invalid dimensions\n";
        return -1;
    }

    // Derive the GPU architecture from the running device (do not assume one at
    // compile time) and reject unsupported archs, per review feedback.
    {
        int dev = 0;
        hipDeviceProp_t props{};
        if(hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&props, dev) != hipSuccess)
        {
            std::cerr << "dispatcher_run_bquant_gemm: could not query device architecture\n";
            return -1;
        }
        const std::string arch(props.gcnArchName);
        if(arch.rfind("gfx950", 0) != 0 && arch.rfind("gfx942", 0) != 0 &&
           arch.rfind("gfx90a", 0) != 0)
        {
            std::cerr << "dispatcher_run_bquant_gemm: unsupported GPU architecture '" << arch
                      << "' (supported: gfx90a, gfx942, gfx950)\n";
            return -1;
        }
    }

    // Validate that the caller's QK_B/QN_B match the compile-time quant group sizes
    // baked into this .so.  A mismatch means the BQ device buffer would be allocated
    // with the wrong size while the kernel indexes it with different strides.
    {
        const int64_t expected_QK_B =
            (K + static_cast<int64_t>(QuantGroupSize::kK) - 1) / QuantGroupSize::kK;
        const int64_t expected_QN_B =
            (N + static_cast<int64_t>(QuantGroupSize::kN) - 1) / QuantGroupSize::kN;
        if(QK_B != expected_QK_B || QN_B != expected_QN_B)
        {
            std::cerr << "dispatcher_run_bquant_gemm: QK_B/QN_B mismatch. " << "Got (" << QK_B
                      << ", " << QN_B << "), " << "expected (" << expected_QK_B << ", "
                      << expected_QN_B << ") " << "for K=" << K << " N=" << N
                      << " with QuantGroupSize kK=" << QuantGroupSize::kK
                      << " kN=" << QuantGroupSize::kN << "\n";
            return -1;
        }
    }

    // This implementation only supports packed (contiguous) layouts.
    // Device buffers are allocated and copied as M*K, K*N, QK_B*QN_B, M*N packed arrays.
    // BQ is ColumnMajor with logical shape [QK_B, QN_B] (matching Old-TE's rcr
    // path and the WPQuantB pipeline's col-major requirement), so its packed
    // leading dim is QK_B.  Non-packed strides would cause the kernel to index
    // into a differently-sized buffer, producing incorrect results or OOB.
    if(stride_A != K || stride_B != K || stride_BQ != QK_B || stride_C != N)
    {
        std::cerr << "dispatcher_run_bquant_gemm: non-packed strides are not supported. "
                  << "Expected stride_A=" << K << " stride_B=" << K << " stride_BQ=" << QK_B
                  << " stride_C=" << N << ", got stride_A=" << stride_A << " stride_B=" << stride_B
                  << " stride_BQ=" << stride_BQ << " stride_C=" << stride_C << "\n";
        return -1;
    }

    const ADataType* A_host  = static_cast<const ADataType*>(A);
    const BDataType* B_host  = static_cast<const BDataType*>(B);
    const QDataType* BQ_host = static_cast<const QDataType*>(BQ);
    CDataType* C_host        = static_cast<CDataType*>(C);

    ADataType* A_dev  = nullptr;
    BDataType* B_dev  = nullptr;
    QDataType* BQ_dev = nullptr;
    CDataType* C_dev  = nullptr;

    auto cleanup = [&]() {
        if(A_dev)
            (void)hipFree(A_dev);
        if(B_dev)
            (void)hipFree(B_dev);
        if(BQ_dev)
            (void)hipFree(BQ_dev);
        if(C_dev)
            (void)hipFree(C_dev);
    };

    // Allocate device buffers.
    // B may be a packed type (pk_int4_t, pk_fp4_t): 2 logical values per byte.
    // elements_to_bytes<T>(n) handles the packed case via numeric_traits::PackedSize.
    HIP_CHECK(hipMalloc(&A_dev, elements_to_bytes<ADataType>(M * K)));
    HIP_CHECK(hipMalloc(&B_dev, elements_to_bytes<BDataType>(K * N)));
    HIP_CHECK(hipMalloc(&BQ_dev, elements_to_bytes<QDataType>(QK_B * QN_B)));
    HIP_CHECK(hipMalloc(&C_dev, elements_to_bytes<CDataType>(M * N)));

    // Copy inputs to device
    HIP_CHECK(hipMemcpy(A_dev, A_host, elements_to_bytes<ADataType>(M * K), hipMemcpyHostToDevice));
    // Copy the B weight matrix to device. This mirrors Old-TE's host-side B prep
    // in run_gemm_quant_example.inc:770-789 exactly:
    //   1. For PreshuffleB kernels, pre-shuffle B into the interleaved layout the
    //      WPQuantB pipeline reads:
    //        * shuffle_b_permuteN<BShuffleConfig>(B) when TiledMMAPermuteN && kN==1
    //        * shuffle_b<BShuffleConfig>(B)          otherwise
    //      Without this the PreshuffleB kernels return garbage (max_rel ~67-69 for
    //      fp8/bf8 preshuffleb on gfx950).
    //   2. For pk_int4 B (fp8i4/bf8i4), permute_vectors_i4x4_b is applied
    //      UNCONDITIONALLY (run_gemm_quant_example.inc:784-787) so the device
    //      i4->fp8/bf8 conversion sees data in 0x75316420 order. Skipping it made
    //      all fp8i4/bf8i4 phases wrong (NaN on random, zeros on constant).
    {
        // B is supplied ColumnMajor [K, N]; the shuffle/permute helpers index it
        // as [K rows, N cols].
        ck_tile::HostTensor<BDataType> b_k_n(
            ck_tile::host_tensor_descriptor(static_cast<int>(K),
                                            static_cast<int>(N),
                                            static_cast<int>(K),
                                            ck_tile::bool_constant<false>{} /*col-major*/));
        // For packed B (pk_int4_t, pk_fp4_t; PackedSize=2) the HostTensor holds
        // only K*N/PackedSize elements and B_host already contains the packed
        // representation, so copy b_k_n.size() elements (== elements_to_bytes
        // count) -- copying K*N here overran the buffer and corrupted the heap
        // before permute_vectors_i4x4_b ran, crashing all i4/fp4 configs.
        std::copy(B_host, B_host + b_k_n.size(), b_k_n.begin());

        ck_tile::HostTensor<BDataType> b_k_n_dev = b_k_n;
        if constexpr(SelectedKernel::PreshuffleB)
        {
            constexpr bool use_permute_n =
                SelectedKernel::TiledMMAPermuteN && (QuantGroupSize::kN == 1);
            if constexpr(use_permute_n)
                b_k_n_dev =
                    ck_tile::shuffle_b_permuteN<typename SelectedKernel::BShuffleConfig>(b_k_n);
            else
                b_k_n_dev = ck_tile::shuffle_b<typename SelectedKernel::BShuffleConfig>(b_k_n);
        }
        // pk_int4 B is always permuted (Old-TE run_gemm_quant_example.inc:784-787).
        if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
        {
            ck_tile::permute_vectors_i4x4_b(b_k_n_dev);
        }

        HIP_CHECK(hipMemcpy(B_dev, b_k_n_dev.data(), elements_to_bytes<BDataType>(K * N), hipMemcpyHostToDevice));
    }
    // Apply BQ preshuffle when required -- mirrors run_gemm_quant_example.inc:794-825
    // exactly. There are three cases:
    //   (a) PreshuffleB && TiledMMAPermuteN && kN==1: bq_permuteN the BQ scales
    //       first, then shuffle_bq if BPreshuffleQuant (else use the permuted BQ).
    //   (b) BPreshuffleQuant (no permuteN): shuffle_bq only.
    //   (c) neither: plain copy.
    // BQ is ColumnMajor [QK_B, QN_B] (leading dim QK_B) -- build every host tensor
    // col-major so the shuffle/permute helpers see the layout the kernel expects.
    {
        constexpr int block_bq_k =
            static_cast<int>(SelectedKernel::TileK) / static_cast<int>(QuantGroupSize::kK);
        constexpr bool use_permute_n = SelectedKernel::PreshuffleB &&
                                       SelectedKernel::TiledMMAPermuteN &&
                                       (QuantGroupSize::kN == 1);

        ck_tile::HostTensor<QDataType> bq_h(
            ck_tile::host_tensor_descriptor(static_cast<int>(QK_B),
                                            static_cast<int>(QN_B),
                                            static_cast<int>(QK_B),
                                            ck_tile::bool_constant<false>{} /*col-major*/));
        std::copy(BQ_host, BQ_host + QK_B * QN_B, bq_h.begin());

        // HostTensor has a deleted default constructor, so each branch owns its
        // result and copies within scope (mirrors Old-TE's per-branch ToDevice).
        const std::size_t bq_bytes = elements_to_bytes<QDataType>(QK_B * QN_B);
        hipError_t copy_rc         = hipSuccess;
        if constexpr(use_permute_n)
        {
            auto bq_permuted = ck_tile::bq_permuteN<typename SelectedKernel::BShuffleConfig>(
                bq_h, static_cast<ck_tile::index_t>(QuantGroupSize::kN));
            if constexpr(SelectedKernel::BPreshuffleQuant)
            {
                auto bq_shuffled = ck_tile::shuffle_bq(&bq_permuted, block_bq_k);
                copy_rc = hipMemcpy(BQ_dev, bq_shuffled.data(), bq_bytes, hipMemcpyHostToDevice);
            }
            else
            {
                copy_rc = hipMemcpy(BQ_dev, bq_permuted.data(), bq_bytes, hipMemcpyHostToDevice);
            }
        }
        else if constexpr(SelectedKernel::BPreshuffleQuant)
        {
            auto bq_shuffled = ck_tile::shuffle_bq(&bq_h, block_bq_k);
            copy_rc = hipMemcpy(BQ_dev, bq_shuffled.data(), bq_bytes, hipMemcpyHostToDevice);
        }
        else
        {
            copy_rc = hipMemcpy(BQ_dev, bq_h.data(), bq_bytes, hipMemcpyHostToDevice);
        }

        HIP_CHECK(copy_rc);
    }
    HIP_CHECK(hipMemset(C_dev, 0, elements_to_bytes<CDataType>(M * N)));

    // Build QuantGemmHostArgs (aq_ptr = nullptr, QK_A = 0, stride_AQ = 0 for BQuant-only)
    ck_tile::QuantGemmHostArgs args;
    args.a_ptr     = A_dev;
    args.b_ptr     = B_dev;
    args.aq_ptr    = nullptr;
    args.bq_ptr    = BQ_dev;
    args.c_ptr     = C_dev;
    args.k_batch   = k_batch;
    args.M         = static_cast<ck_tile::index_t>(M);
    args.N         = static_cast<ck_tile::index_t>(N);
    args.K         = static_cast<ck_tile::index_t>(K);
    args.QK_A      = 0;
    args.QK_B      = static_cast<ck_tile::index_t>(QK_B);
    args.stride_A  = static_cast<ck_tile::index_t>(stride_A);
    args.stride_B  = static_cast<ck_tile::index_t>(stride_B);
    args.stride_C  = static_cast<ck_tile::index_t>(stride_C);
    args.stride_AQ = 0;
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
        std::cerr << "dispatcher_run_bquant_gemm: kernel reported unsupported args\n";
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
