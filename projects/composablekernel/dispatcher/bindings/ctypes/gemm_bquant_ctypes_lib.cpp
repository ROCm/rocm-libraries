// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * BQuant GEMM ctypes Library (block_scale_gemm operator, gemm_bquant_* naming)
 *
 * Provides a C API for Python ctypes integration. One .so is compiled per
 * kernel variant; the kernel is force-included at compile time:
 *   hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE gemm_bquant_ctypes_lib.cpp
 *
 * Force-include defines (from generated kernel header):
 *   SelectedKernel, KERNEL_NAME
 *   ADataType, BDataType, CDataType, BQDataType, AccDataType, QuantGroupSize, GroupSizeK
 *
 * Design: direct launch -- SelectedKernel::launch(QuantGemmHostArgs, stream_config).
 * No dispatcher registry used: BQuant kernels take QuantGemmHostArgs which is
 * incompatible with GeneratedTileKernelInstance::run().
 *
 * Difference from grouped_gemm_bquant_ctypes_lib.cpp:
 *   - Uses BQDataType (not QDataType) as the scale type — matches the generated header
 *   - Uses "dispatcher_run_bquant_gemm" (same function name for compatibility)
 *   - No QN_B parameter — gemm_bquant uses 1D quant (K-grouped only, N-group = 1)
 *
 * NOTE: Do NOT include dispatcher/include — conflicts with QuantGemmHostArgs.
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

#define HIP_CHECK(call)                                                                        \
    {                                                                                          \
        hipError_t _err = (call);                                                              \
        if(_err != hipSuccess)                                                                 \
        {                                                                                      \
            std::cerr << "HIP error: " << hipGetErrorString(_err) << " at " << __FILE__ << ":" \
                      << __LINE__ << "\n";                                                     \
            return -1;                                                                         \
        }                                                                                      \
    }

extern "C" {

int dispatcher_initialize()
{
    if(g_initialized) return 0;
    g_initialized = true;
    return 0;
}

/**
 * Run BQuant GEMM: C[M,N] = A[M,K] @ dequant(B[K,N], BQ[ceil(K/gK), ceil(N/gN)])
 *
 * A, B, BQ, C are host pointers. This function manages device memory internally.
 *
 * Parameters:
 *   A, B, BQ, C  - host data pointers
 *   M, N, K      - matrix dimensions
 *   stride_A     - leading dimension of A (row-major: K; col-major: M)
 *   stride_B     - leading dimension of B (col-major: K; row-major: N)
 *   stride_BQ    - leading dimension of BQ (row-major: ceil(N/gN))
 *   stride_C     - leading dimension of C (row-major: N)
 *   QK_B         - number of K-groups = ceil(K / quant_group_k)
 *   QN_B         - number of N-groups = ceil(N / quant_group_n)  [usually 1]
 *   k_batch      - split-K factor (1 = no split)
 *   time_ms      - output: kernel execution time in ms (may be NULL)
 *
 * Returns 0 on success, negative on error.
 */
int dispatcher_run_bquant_gemm(const void* A,
                               const void* B,
                               const void* BQ,
                               void*       C,
                               int64_t     M,
                               int64_t     N,
                               int64_t     K,
                               int64_t     stride_A,
                               int64_t     stride_B,
                               int64_t     stride_BQ,
                               int64_t     stride_C,
                               int64_t     QK_B,
                               int64_t     QN_B,
                               int         k_batch,
                               float*      time_ms)
{
#ifndef CK_TILE_SINGLE_KERNEL_INCLUDE
    std::cerr << "dispatcher_run_bquant_gemm: library built without a kernel; unsupported\n";
    (void)A; (void)B; (void)BQ; (void)C;
    (void)M; (void)N; (void)K;
    (void)stride_A; (void)stride_B; (void)stride_BQ; (void)stride_C;
    (void)QK_B; (void)QN_B; (void)k_batch; (void)time_ms;
    return -2;
#else
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

    // Validate QK_B matches compile-time quant group size.
    {
        const int64_t expected_QK_B =
            (K + static_cast<int64_t>(GroupSizeK) - 1) / static_cast<int64_t>(GroupSizeK);
        if(QK_B != expected_QK_B)
        {
            std::cerr << "dispatcher_run_bquant_gemm: QK_B mismatch. "
                      << "Got " << QK_B << ", expected " << expected_QK_B
                      << " for K=" << K << " GroupSizeK=" << GroupSizeK << "\n";
            return -1;
        }
    }

    if(stride_A != K || stride_B != K || stride_BQ != QN_B || stride_C != N)
    {
        std::cerr << "dispatcher_run_bquant_gemm: non-packed strides are not supported. "
                  << "Expected stride_A=" << K << " stride_B=" << K << " stride_BQ=" << QN_B
                  << " stride_C=" << N << ", got stride_A=" << stride_A << " stride_B=" << stride_B
                  << " stride_BQ=" << stride_BQ << " stride_C=" << stride_C << "\n";
        return -1;
    }

    ADataType*  A_dev  = nullptr;
    BDataType*  B_dev  = nullptr;
    BQDataType* BQ_dev = nullptr;
    CDataType*  C_dev  = nullptr;

    auto cleanup = [&]() {
        if(A_dev)  (void)hipFree(A_dev);
        if(B_dev)  (void)hipFree(B_dev);
        if(BQ_dev) (void)hipFree(BQ_dev);
        if(C_dev)  (void)hipFree(C_dev);
    };

    HIP_CHECK(hipMalloc(&A_dev,  elements_to_bytes<ADataType>(M * K)));
    HIP_CHECK(hipMalloc(&B_dev,  elements_to_bytes<BDataType>(K * N)));
    HIP_CHECK(hipMalloc(&BQ_dev, elements_to_bytes<BQDataType>(QK_B * QN_B)));
    HIP_CHECK(hipMalloc(&C_dev,  elements_to_bytes<CDataType>(M * N)));

    HIP_CHECK(hipMemcpy(A_dev, static_cast<const ADataType*>(A),
                        elements_to_bytes<ADataType>(M * K), hipMemcpyHostToDevice));

    // BQ preshuffle when required
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
        HIP_CHECK(hipMemcpy(BQ_dev, bq_shuffled.data(),
                            elements_to_bytes<BQDataType>(QK_B * QN_B), hipMemcpyHostToDevice));
    }
    else
    {
        HIP_CHECK(hipMemcpy(BQ_dev, static_cast<const BQDataType*>(BQ),
                            elements_to_bytes<BQDataType>(QK_B * QN_B), hipMemcpyHostToDevice));
    }

    HIP_CHECK(hipMemcpy(B_dev, static_cast<const BDataType*>(B),
                        elements_to_bytes<BDataType>(K * N), hipMemcpyHostToDevice));
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
    ck_tile::stream_config stream_cfg{
        nullptr, do_time, 0, do_time ? 3 : 0, do_time ? 10 : 1, do_time, false, 1,
    };

    float exec_time = SelectedKernel::launch(args, stream_cfg);

    if(exec_time < 0.0f)
    {
        std::cerr << "dispatcher_run_bquant_gemm: kernel reported unsupported args\n";
        cleanup();
        return -2;
    }

    HIP_CHECK(hipMemcpy(static_cast<CDataType*>(C), C_dev,
                        elements_to_bytes<CDataType>(M * N), hipMemcpyDeviceToHost));

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
