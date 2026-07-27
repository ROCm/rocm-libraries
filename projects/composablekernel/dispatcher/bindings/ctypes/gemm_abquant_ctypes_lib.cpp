// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * ABQuant GEMM ctypes Library
 *
 * Provides a C API for Python ctypes integration. One .so is compiled per
 * kernel variant; the kernel is force-included at compile time:
 *   hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE gemm_abquant_ctypes_lib.cpp
 *
 * Force-include defines (from generated kernel header):
 *   SelectedKernel, KERNEL_NAME
 *   ADataType, BDataType, CDataType, AQDataType, BQDataType, AccDataType
 *   AQuantGroupSize, BQuantGroupSize, GroupSizeK, GroupSizeN
 *
 * Design: direct launch -- SelectedKernel::launch(QuantGemmHostArgs, stream_config) is
 * called directly. No dispatcher registry is used.
 *
 * ABQuant specifics vs AQuant/BQuant:
 *   - Both AQ (A scale) and BQ (B scale) are populated
 *   - AQ is RowMajor [M, QK_A]: stride_AQ = QK_A
 *   - BQ is ColumnMajor [QK_B, QN_B]: stride_BQ = QK_B (leading dim = QK_B rows)
 *
 * NOTE: Do NOT include dispatcher/include here — it conflicts with QuantGemmHostArgs.
 */

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <type_traits>

#include "ck_tile/host/tensor_shuffle_utils.hpp"

template <typename T>
static constexpr std::size_t elements_to_bytes(std::size_t n)
{
    return n * sizeof(T) / ck_tile::numeric_traits<T>::PackedSize;
}

static bool g_initialized = false;

extern "C" {

int dispatcher_initialize()
{
    if(g_initialized)
        return 0;
    g_initialized = true;
    return 0;
}

/**
 * Run ABQuant GEMM: C[M,N] = dequant(A[M,K], AQ[M,QK_A]) @ dequant(B[K,N], BQ[QK_B,QN_B])
 *
 * Parameters:
 *   A           - host A matrix (quantized), row-major [M,K]
 *   B           - host B matrix (quantized), col-major [K,N]
 *   AQ          - host A scale tensor [M, QK_A], float32, row-major; stride=QK_A
 *   BQ          - host B scale tensor [QK_B, QN_B], float32, col-major; stride=QK_B
 *   C           - host output C matrix [M, N], row-major
 *   M, N, K     - matrix dimensions
 *   stride_A    - leading dim of A (row-major: K; col-major: M)
 *   stride_B    - leading dim of B (col-major: K; row-major: N)
 *   stride_AQ   - leading dim of AQ row-major [M, QK_A] = QK_A
 *   stride_BQ   - leading dim of BQ col-major [QK_B, QN_B] = QK_B
 *   stride_C    - leading dim of C row-major [M, N] = N
 *   QK_A        - ceil(K / group_size_k) — K-groups for A scale
 *   QK_B        - ceil(K / group_size_k) — K-groups for B scale (same group_k as A)
 *   QN_B        - ceil(N / group_size_n) — N-groups for B scale
 *   k_batch     - split-K factor (1 = no split)
 *   time_ms     - output execution time in ms (may be NULL)
 */
int dispatcher_run_abquant_gemm(const void* A,
                                const void* B,
                                const void* AQ,
                                const void* BQ,
                                void*       C,
                                int64_t     M,
                                int64_t     N,
                                int64_t     K,
                                int64_t     stride_A,
                                int64_t     stride_B,
                                int64_t     stride_AQ,
                                int64_t     stride_BQ,
                                int64_t     stride_C,
                                int64_t     QK_A,
                                int64_t     QK_B,
                                int64_t     QN_B,
                                int         k_batch,
                                float*      time_ms)
{
#ifndef CK_TILE_SINGLE_KERNEL_INCLUDE
    std::cerr << "dispatcher_run_abquant_gemm: library built without a kernel; unsupported\n";
    (void)A; (void)B; (void)AQ; (void)BQ; (void)C;
    (void)M; (void)N; (void)K;
    (void)stride_A; (void)stride_B; (void)stride_AQ; (void)stride_BQ; (void)stride_C;
    (void)QK_A; (void)QK_B; (void)QN_B; (void)k_batch; (void)time_ms;
    return -2;
#else
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

    {
        int dev = 0;
        hipDeviceProp_t props{};
        if(hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&props, dev) != hipSuccess)
        {
            std::cerr << "dispatcher_run_abquant_gemm: could not query device architecture\n";
            return -1;
        }
        const std::string arch(props.gcnArchName);
        if(arch.rfind("gfx950", 0) != 0 && arch.rfind("gfx942", 0) != 0 &&
           arch.rfind("gfx90a", 0) != 0)
        {
            std::cerr << "dispatcher_run_abquant_gemm: unsupported GPU architecture '" << arch
                      << "' (supported: gfx90a, gfx942, gfx950)\n";
            return -1;
        }
    }

    // Validate QK_A and QK_B match compile-time group sizes.
    {
        const int64_t expected_QK_A =
            (K + static_cast<int64_t>(GroupSizeK) - 1) / static_cast<int64_t>(GroupSizeK);
        const int64_t expected_QK_B = expected_QK_A;  // same GroupSizeK for A and B
        const int64_t expected_QN_B =
            (N + static_cast<int64_t>(GroupSizeN) - 1) / static_cast<int64_t>(GroupSizeN);
        if(QK_A != expected_QK_A || QK_B != expected_QK_B || QN_B != expected_QN_B)
        {
            std::cerr << "dispatcher_run_abquant_gemm: QK_A/QK_B/QN_B mismatch. "
                      << "Got (" << QK_A << "," << QK_B << "," << QN_B << "), "
                      << "expected (" << expected_QK_A << "," << expected_QK_B << ","
                      << expected_QN_B << ") for K=" << K << " N=" << N
                      << " GroupSizeK=" << GroupSizeK << " GroupSizeN=" << GroupSizeN << "\n";
            return -1;
        }
    }

    // Validate packed strides.
    // A: row-major -> stride=K; col-major -> stride=M
    const bool a_is_col_major = (stride_A == M);
    const int64_t expected_stride_A  = a_is_col_major ? M : K;
    const int64_t expected_stride_AQ = QK_A;   // AQ row-major [M, QK_A]
    const int64_t expected_stride_B  = K;       // B col-major: leading dim = K
    const int64_t expected_stride_BQ = QK_B;   // BQ col-major [QK_B, QN_B]: leading dim = QK_B
    const int64_t expected_stride_C  = N;       // C row-major
    if(stride_A != expected_stride_A || stride_AQ != expected_stride_AQ ||
       stride_B != expected_stride_B || stride_BQ != expected_stride_BQ ||
       stride_C != expected_stride_C)
    {
        std::cerr << "dispatcher_run_abquant_gemm: non-packed strides not supported.\n";
        return -1;
    }

    ADataType*  A_dev  = nullptr;
    BDataType*  B_dev  = nullptr;
    AQDataType* AQ_dev = nullptr;
    BQDataType* BQ_dev = nullptr;
    CDataType*  C_dev  = nullptr;

    auto cleanup = [&]() {
        if(A_dev)  (void)hipFree(A_dev);
        if(B_dev)  (void)hipFree(B_dev);
        if(AQ_dev) (void)hipFree(AQ_dev);
        if(BQ_dev) (void)hipFree(BQ_dev);
        if(C_dev)  (void)hipFree(C_dev);
    };

    if(hipMalloc(&A_dev,  elements_to_bytes<ADataType>(M * K)) != hipSuccess)
    { cleanup(); return -1; }
    if(hipMalloc(&B_dev,  elements_to_bytes<BDataType>(K * N)) != hipSuccess)
    { cleanup(); return -1; }
    if(hipMalloc(&AQ_dev, elements_to_bytes<AQDataType>(M * QK_A)) != hipSuccess)
    { cleanup(); return -1; }
    if(hipMalloc(&BQ_dev, elements_to_bytes<BQDataType>(QK_B * QN_B)) != hipSuccess)
    { cleanup(); return -1; }
    if(hipMalloc(&C_dev,  elements_to_bytes<CDataType>(M * N)) != hipSuccess)
    { cleanup(); return -1; }

    // Copy A to device.
    if(hipMemcpy(A_dev, static_cast<const ADataType*>(A),
                 elements_to_bytes<ADataType>(M * K), hipMemcpyHostToDevice) != hipSuccess)
    { cleanup(); return -1; }

    // AQ: preshuffle if APreshuffleQuant, otherwise direct copy.
    if constexpr(SelectedKernel::APreshuffleQuant)
    {
        constexpr int block_aq_k =
            static_cast<int>(SelectedKernel::TileK) / static_cast<int>(GroupSizeK);
        ck_tile::HostTensor<AQDataType> aq_h(
            ck_tile::host_tensor_descriptor(static_cast<int>(M),
                                            static_cast<int>(QK_A),
                                            static_cast<int>(QK_A),
                                            ck_tile::bool_constant<true>{}));
        std::copy(static_cast<const AQDataType*>(AQ),
                  static_cast<const AQDataType*>(AQ) + M * QK_A,
                  aq_h.begin());
        auto aq_shuffled = ck_tile::shuffle_aq(&aq_h, block_aq_k);
        if(hipMemcpy(AQ_dev, aq_shuffled.data(),
                     elements_to_bytes<AQDataType>(M * QK_A),
                     hipMemcpyHostToDevice) != hipSuccess)
        { cleanup(); return -1; }
    }
    else
    {
        if(hipMemcpy(AQ_dev, static_cast<const AQDataType*>(AQ),
                     elements_to_bytes<AQDataType>(M * QK_A),
                     hipMemcpyHostToDevice) != hipSuccess)
        { cleanup(); return -1; }
    }

    // B to device.
    if(hipMemcpy(B_dev, static_cast<const BDataType*>(B),
                 elements_to_bytes<BDataType>(K * N), hipMemcpyHostToDevice) != hipSuccess)
    { cleanup(); return -1; }

    // BQ: preshuffle if BPreshuffleQuant, otherwise direct copy.
    if constexpr(SelectedKernel::BPreshuffleQuant)
    {
        constexpr int block_bq_k =
            static_cast<int>(SelectedKernel::TileK) / static_cast<int>(GroupSizeK);
        ck_tile::HostTensor<BQDataType> bq_h(
            ck_tile::host_tensor_descriptor(static_cast<int>(QK_B),
                                            static_cast<int>(QN_B),
                                            static_cast<int>(QN_B),
                                            ck_tile::bool_constant<true>{}));
        std::copy(static_cast<const BQDataType*>(BQ),
                  static_cast<const BQDataType*>(BQ) + QK_B * QN_B,
                  bq_h.begin());
        auto bq_shuffled = ck_tile::shuffle_bq(&bq_h, block_bq_k);
        if(hipMemcpy(BQ_dev, bq_shuffled.data(),
                     elements_to_bytes<BQDataType>(QK_B * QN_B),
                     hipMemcpyHostToDevice) != hipSuccess)
        { cleanup(); return -1; }
    }
    else
    {
        if(hipMemcpy(BQ_dev, static_cast<const BQDataType*>(BQ),
                     elements_to_bytes<BQDataType>(QK_B * QN_B),
                     hipMemcpyHostToDevice) != hipSuccess)
        { cleanup(); return -1; }
    }

    if(hipMemset(C_dev, 0, elements_to_bytes<CDataType>(M * N)) != hipSuccess)
    { cleanup(); return -1; }

    // Build QuantGemmHostArgs for ABQuant.
    // BQ stride: col-major [QK_B, QN_B] -> stride_BQ = QK_B.
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
    ck_tile::stream_config stream_cfg{
        nullptr, do_time, 0, do_time ? 3 : 0, do_time ? 10 : 1, do_time, false, 1,
    };

    float exec_time = SelectedKernel::launch(args, stream_cfg);

    if(exec_time < 0.0f)
    {
        std::cerr << "dispatcher_run_abquant_gemm: kernel reported unsupported args\n";
        cleanup();
        return -2;
    }

    if(hipMemcpy(static_cast<CDataType*>(C), C_dev,
                 elements_to_bytes<CDataType>(M * N), hipMemcpyDeviceToHost) != hipSuccess)
    { cleanup(); return -1; }

    if(time_ms)
        *time_ms = exec_time;

    cleanup();
    return 0;
#endif // CK_TILE_SINGLE_KERNEL_INCLUDE
}

const char* dispatcher_get_kernel_name()
{
#ifdef CK_TILE_SINGLE_KERNEL_INCLUDE
    return KERNEL_NAME;
#else
    return "";
#endif
}

int dispatcher_init() { return dispatcher_initialize(); }
int dispatcher_get_kernel_count() { return 1; }
void dispatcher_cleanup() { g_initialized = false; }

} // extern "C"
