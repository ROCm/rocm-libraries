// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*
 * emulation.cpp
 *
 * FP32/FP64 GEMM emulation via Ozaki Scheme II (accurate mode) using INT8 Tensor Cores.
 *
 * Algorithm (paper: Ozaki, Uchino, Imamura, arXiv:2504.08009)
 *
 * This file MUST be compiled as HIP (LANGUAGE HIP in CMakeLists.txt).
 * Inner INT8 GEMMs use hipblasLtMatmul (INT8 tensor cores, INT32 accumulate).
 */

#include "emulation.hpp"
#include "handle.h" /* _rocblaslt_handle */
#include "hipblaslt_ostream.hpp" /* hipblaslt_cerr */
#include "kernel_common.hpp" /* shared device helpers + tuning constants */
#include "prelim_kernels.hpp" /* accu_prelim_A/B_T/N kernels */
#include "reconstruct_kernels.hpp" /* chunk_accum + accum_finalize kernels */
#include "scale_kernels.hpp" /* scale_A/B_T/N kernels */
#include "shift_refine_kernels.hpp" /* refine / col_max / adp kernels */
#include "tables.hpp"

#include "hipblaslt/hipblaslt.h"
#include <hip/hip_runtime.h>

#include <atomic> // std::atomic (ADP overflow warning rate-limiter)
#include <cassert> // assert
#include <chrono> // std::chrono::steady_clock (CPU-side profiling)
#include <cmath> // std::log2, std::floor, etc.
#include <cstdio> // std::fopen / std::fprintf / std::fclose / std::ftell
#include <cstdlib> // std::getenv
#include <cstring> // std::strcmp
#include <limits> // std::numeric_limits
#include <mutex> // std::mutex  (per-device INT8 handle/desc cache)
#include <optional> // std::optional
#include <unordered_map> // std::unordered_map
#include <utility> // std::index_sequence, std::make_index_sequence

namespace FixedPointEmulation
{
    /* =========================================================================
     * Tuning constants
     * ========================================================================= */
    /* Workspace reserved for the INT8 GEMM algorithms (preliminary and batch).
     * Passing a non-null workspace allows hipBLASLt to select algorithms that
     * require workspace, which is necessary for very large k values where no
     * zero-workspace INT8 GEMM algorithm is available.
     * 128 MiB matches the default value of HIPBLASLT_TUNING_USER_MAX_WORKSPACE. */
    /* TODO some INT8 GEMMs return wrong results for larger K when given a
     * larger workspace, but are correct without workspace.
     * This is why this workspace is set to size 0.
     * For larger K this might fail to find a solution,
     * and then we fall back to native DGEMM */
    static constexpr size_t OZ2_INT8_GEMM_WS_BYTES = 0; //128ull << 20; /* 128 MiB */

    /* Number of moduli processed per pass: the largest value ≤ s such that
     * the simultaneous allocation of A8i + B8i + C32i fits in the total budget.
     * Scale and GEMM share the same chunk — one scale launch followed by one
     * batched GEMM per pass.  The final pass handles the remainder naturally
     * via min(chunk, effective_s - chunk_start).                             */
    static unsigned compute_chunk_size(int64_t m, int64_t n, int64_t k, unsigned s, size_t budget)
    {
        const size_t mn4    = static_cast<size_t>(m) * static_cast<size_t>(n) * 4u;
        const size_t lda8i  = pad(static_cast<size_t>(k));
        const size_t cola8i = pad(static_cast<size_t>(m));
        const size_t slc    = lda8i * cola8i + lda8i * static_cast<size_t>(n);

        size_t chunk = s;
        if(mn4 > 0u || slc > 0u)
            chunk = std::min(chunk, budget / (mn4 + slc));
        return static_cast<unsigned>(std::max(size_t(1u), chunk));
    }

    /* Compute the effective INT8 workspace budget for the monolithic path given
     * workspace_bytes, sub-problem dimensions (m, n), and the moduli count s.
     * Mirrors the two-step budget derivation in emulated_gemm_impl so that
     * effective_time_ms accurately models the available budget at each recursive
     * split level.                                                              */
    static size_t
        compute_budget_from_ws(int64_t m, int64_t n, int64_t k, unsigned s, size_t workspace_bytes)
    {
        const size_t co  = pad(static_cast<size_t>(m));
        const size_t pn  = pad(static_cast<size_t>(n));
        const size_t boh = co * sizeof(int16_t) + pn * sizeof(int16_t) + sizeof(uint32_t)
                           + co * sizeof(int32_t) + pn * sizeof(int32_t) + 2 * sizeof(float)
                           + OZ2_INT8_GEMM_WS_BYTES;
        const size_t   W1  = (workspace_bytes > boh) ? workspace_bytes - boh : 0u;
        const unsigned cs1 = compute_chunk_size(m, n, k, s, W1);
        if(cs1 < s)
        {
            const size_t zoh = 2u * co * static_cast<size_t>(n) * sizeof(double);
            return (workspace_bytes > boh + zoh) ? workspace_bytes - boh - zoh : 0u;
        }
        return W1;
    }

    struct PerfModelKernelEffs
    {
        static constexpr double eff_prelim[2][2] = {{0.697, 0.561}, /* [N][N], [N][T] */
                                                    {0.902, 0.682}}; /* [T][N], [T][T] */
        static constexpr double eff_scale[2][2]  = {{0.595, 0.520}, /* [N][N], [N][T] */
                                                    {0.683, 0.583}}; /* [T][N], [T][T] */
        static constexpr double eff_refine       = 0.586;
        static constexpr double eff_accum        = 0.938;
        static constexpr double host_overhead_s = 1.0e-4; /* host overhead (s) = 0.1 ms           */
        static constexpr double latency_kernel_s
            = 5.0e-6; /* GPU kernel scheduling overhead (s)   */
        static constexpr double latency_matmul_s
            = 10.0e-6; /* hipBLASLt matmul launch overhead (s) */
        static constexpr double latency_memset_s
            = 2.0e-6; /* hipMemsetAsync overhead (s)          */
        static constexpr double latency_sync_s = 50.0e-6; /* hipStreamSynchronize cost (s)        */
    };

    struct PerfModelDeviceParams
    {
        double bw;
        double int8;
        double fp64;
        double fp32;
    };

    /* Returns the perf-model parameters for the given HIP device, or nullopt if
     * the device is not in the table (in which case emulation should not run). */
    static std::optional<PerfModelDeviceParams> get_perf_model_params(int device)
    {
        static const std::unordered_map<uint32_t, PerfModelDeviceParams> hw_params_by_pci_id = {
            {0x75a0u, {6.08e12, 3.172e15, 6.824e13, 1.34e14}}, // gfx950-mi350x
            {0x75b0u, {6.08e12, 3.172e15, 6.824e13, 1.34e14}}, // gfx950-mi350x
            {0x75a3u, {6.82e12, 3.301e15, 7.726e13, 1.54e14}}, // gfx950-mi355x
            {0x75b3u, {6.82e12, 3.301e15, 7.726e13, 1.54e14}}, // gfx950-mi355x
        };
        static std::optional<std::optional<PerfModelDeviceParams>> device_params_cache[64];

        if(device < 0 || device >= 64)
            return std::nullopt;
        auto& entry = device_params_cache[device];
        if(!entry)
        {
            int              chip_id = 0;
            const hipError_t attr_err
                = hipDeviceGetAttribute(&chip_id, hipDeviceAttributePciChipId, device);
            const uint32_t pci_device_id = static_cast<uint32_t>(chip_id) & 0xFFFFu;
            auto           it            = hw_params_by_pci_id.find(pci_device_id);
            entry                        = (it != hw_params_by_pci_id.end())
                                               ? std::optional<PerfModelDeviceParams>{it->second}
                                               : std::optional<PerfModelDeviceParams>{};
        }
        return *entry;
    }

    /* Returns the minimum number of moduli s ∈ [2, S_MAX] such that
     * log2P(s-2) >= adp_bits, used ONLY in the PERFORMANCE MODEL to predict
     * which s ADP will select at runtime.  Falls back to S_MAX when
     * adp_bits <= 0 (sentinel) or when all moduli are needed to reach the
     * target.                                                                */
    static unsigned adp_expected_num_moduli(int adp_bits, unsigned max_s) noexcept
    {
        if(adp_bits <= 0)
            return max_s;
        for(unsigned s = 2u; s <= max_s; ++s)
            if(log2P(s - 2u) >= static_cast<float>(adp_bits))
                return s;
        return max_s;
    }

    /* =========================================================================
     * Performance-model predicted times
     * Returns all sub-times in milliseconds.  Used both for the profiling CSV
     * and (via comparison of t_total_ms vs t_native_ms) for the performance
     * heuristic in fixedPointEmulationPerformanceCheck.
     * ========================================================================= */
    struct PerfModelTimes
    {
        double t_prelim_ms; /* prelim kernel (shift + extraction)  */
        double t_prelim_gemm_ms; /* preliminary INT8 GEMM               */
        double t_refine_ms; /* sft-refinement kernels              */
        double t_adp_ms; /* ADP reduce kernels + hipStreamSynchronize (dynamic mode only) */
        double t_scale_ms; /* multi-modulus scaling kernels       */
        double t_int8_gemms_ms; /* all INT8 GEMMs                      */
        double t_accum_ms; /* CRT accumulation / finalize kernels */
        double t_host_ms; /* per-call host overhead              */
        double t_total_ms; /* total predicted emulation time      */
        double t_native_ms; /* predicted native DGEMM/SGEMM time    */
    };

    static PerfModelTimes perf_model_times(bool        tA,
                                           bool        tB,
                                           int64_t     m,
                                           int64_t     n,
                                           int64_t     k,
                                           unsigned    num_moduli,
                                           int         device,
                                           size_t      budget,
                                           hipDataType type_a)
    {
        using K           = PerfModelKernelEffs;
        const auto hw_opt = get_perf_model_params(device);
        assert(hw_opt.has_value()
               && "perf_model_times called for a device not in hw_params_by_pci_id");
        const PerfModelDeviceParams& hw = *hw_opt;

        const bool   is_fp32        = (type_a == HIP_R_32F);
        const double c0             = hw.bw;
        const double c1             = is_fp32 ? hw.fp32 : hw.fp64;
        const double c2             = hw.int8;
        const double bytes_per_elem = is_fp32 ? 4.0 : 8.0;
        const double s              = static_cast<double>(num_moduli);
        const double mn             = static_cast<double>(m) * static_cast<double>(n);
        const double mk             = static_cast<double>(m) * static_cast<double>(k);
        const double kn             = static_cast<double>(k) * static_cast<double>(n);
        const double mnk            = mn * static_cast<double>(k);

        const double chunk_sz
            = static_cast<double>(compute_chunk_size(m, n, k, num_moduli, budget));
        const double n_chunks       = std::ceil(s / chunk_sz);
        const double n_scale_chunks = n_chunks; /* scale and GEMM share the same chunk */

        const double t_int8_bw = (mk + kn + 4.0 * mn) / c0;
        const double t_prelim_kern
            = ((mk + kn) * 17.0 / c0 + 2.0 * K::latency_kernel_s) / K::eff_prelim[tA][tB];
        const double t_prelim_gemm = std::max(2.0 * mnk / c2, t_int8_bw) + K::latency_matmul_s;
        const double t_refine_kern
            = (mn * 8.0 / c0 + 3.0 * K::latency_kernel_s + K::latency_memset_s) / K::eff_refine;
        const double t_scale_kern = ((mk + kn) * (8.0 * n_scale_chunks + s) / c0
                                     + 2.0 * n_scale_chunks * K::latency_kernel_s)
                                    / K::eff_scale[tA][tB];
        const double t_int8_gemms
            = s * std::max(2.0 * mnk / c2, t_int8_bw) + n_chunks * K::latency_matmul_s;
        const double t_accum_kern
            = (mn * (4.0 * s + 32.0 * n_chunks - 16.0) / c0 + n_chunks * K::latency_kernel_s)
              / K::eff_accum;
        const double t_host       = K::host_overhead_s;
        const double t_gemm_accum = t_int8_gemms + t_accum_kern;
        const double t_adp
            = 2.0 * K::latency_kernel_s /* adp_reduce_A_kernel + adp_reduce_B_kernel */
              + K::latency_sync_s; /* hipStreamSynchronize — dominant cost */
        const double t_total = t_prelim_kern + t_prelim_gemm + t_refine_kern + t_scale_kern
                               + t_gemm_accum + t_host + t_adp;
        const double t_native
            = std::max(2.0 * mnk / c1, bytes_per_elem * (mk + kn + mn) / c0) + K::latency_matmul_s;

        constexpr double s2ms = 1000.0;
        return {t_prelim_kern * s2ms,
                t_prelim_gemm * s2ms,
                t_refine_kern * s2ms,
                t_adp * s2ms,
                t_scale_kern * s2ms,
                t_int8_gemms * s2ms,
                t_accum_kern * s2ms,
                t_host * s2ms,
                t_total * s2ms,
                t_native * s2ms};
    }

    /* Minimum problem size for efficient INT8 tensor core execution.
     * Below this threshold MFMA tiles are under-utilised and native DGEMM wins.
     * Both effective_time_ms and emulated_gemm_impl fall back immediately.      */
    static constexpr int64_t FIXED_POINT_EMUL_MIN_MN = 16;

    /* Returns the minimum achievable emulation time in ms, accounting for the
     * recursive binary-halving that fp64EmulatedGemm applies when n_chunks > 1.
     * Both halves execute sequentially so the effective time is additive.
     *
     * The gate comparison uses the RECURSIVE effective times of each half
     * (not the flat perf-model times).  This correctly handles the case where
     * one split does not yet reduce n_chunks but further splitting would: the
     * recursive sub-call for the half discovers and accounts for those deeper
     * splits, returning the true best achievable time for that half.       */
    /* workspace_bytes is passed through to each recursive sub-call so that every
     * level recomputes its own budget from the current sub-problem dimensions —
     * mirroring the behaviour of emulated_gemm_impl, where smaller sub-problems
     * have smaller Zhi/Zlo requirements and thus a larger effective budget.    */
    static double effective_time_ms(bool        tA,
                                    bool        tB,
                                    int64_t     m,
                                    int64_t     n,
                                    int64_t     k,
                                    unsigned    s,
                                    int         device,
                                    size_t      workspace_bytes,
                                    hipDataType type_a)
    {
        if(m < FIXED_POINT_EMUL_MIN_MN || n < FIXED_POINT_EMUL_MIN_MN)
        {
            const double t_nat
                = perf_model_times(tA, tB, m, n, k, s, device, ~size_t{0}, type_a).t_native_ms;
            return t_nat * (1.0 + 1e-6);
        }
        const size_t budget = compute_budget_from_ws(m, n, k, s, workspace_bytes);
        const double t_mono
            = perf_model_times(tA, tB, m, n, k, s, device, budget, type_a).t_total_ms;

        const unsigned chunk_sz = compute_chunk_size(m, n, k, s, budget);
        const unsigned n_chunks = (s + chunk_sz - 1u) / chunk_sz;
        if(n_chunks > 1u)
        {
            const bool    split_m = (m >= n);
            const int64_t half_m  = split_m ? m / 2 : m;
            const int64_t half_n  = split_m ? n : n / 2;

            const double t_split
                = 2.
                  * effective_time_ms(
                      tA, tB, half_m, half_n, k, s, device, workspace_bytes, type_a);
            if(t_split < t_mono)
                return t_split;
        }
        return t_mono;
    }

    /* Returns per-component predicted times summed across ALL leaf sub-GEMMs,
     * mirroring the recursive binary-halving of effective_time_ms.
     * t_native_ms is always set to the top-level (m,n,k) native GEMM time
     * because native GEMM does not split.                                    */
    static PerfModelTimes effective_perf_model_times(bool        tA,
                                                     bool        tB,
                                                     int64_t     m,
                                                     int64_t     n,
                                                     int64_t     k,
                                                     unsigned    s,
                                                     int         device,
                                                     size_t      workspace_bytes,
                                                     hipDataType type_a)
    {
        const size_t   budget = compute_budget_from_ws(m, n, k, s, workspace_bytes);
        PerfModelTimes mono   = perf_model_times(tA, tB, m, n, k, s, device, budget, type_a);

        const unsigned chunk_sz = compute_chunk_size(m, n, k, s, budget);
        const unsigned n_chunks = (s + chunk_sz - 1u) / chunk_sz;
        if(n_chunks > 1u)
        {
            const bool    split_m = (m >= n);
            const int64_t half_m  = split_m ? m / 2 : m;
            const int64_t half_n  = split_m ? n : n / 2;

            const double t_split
                = 2.
                  * effective_time_ms(
                      tA, tB, half_m, half_n, k, s, device, workspace_bytes, type_a);
            if(t_split < mono.t_total_ms)
            {
                PerfModelTimes half = effective_perf_model_times(
                    tA, tB, half_m, half_n, k, s, device, workspace_bytes, type_a);
                half.t_prelim_ms *= 2.0;
                half.t_prelim_gemm_ms *= 2.0;
                half.t_refine_ms *= 2.0;
                half.t_adp_ms *= 2.0;
                half.t_scale_ms *= 2.0;
                half.t_int8_gemms_ms *= 2.0;
                half.t_accum_ms *= 2.0;
                half.t_host_ms *= 2.0;
                half.t_total_ms *= 2.0;
                /* Native DGEMM does not split: keep the top-level prediction. */
                half.t_native_ms = mono.t_native_ms;
                return half;
            }
        }
        return mono;
    }

    static const char* profile_file()
    {
        static const char* const fn = std::getenv("HIPBLASLT_EMULATION_PROFILE");
        return fn;
    }

    /* =========================================================================
     * native_gemm_fallback<T> — run a plain hipblasLtMatmul (DGEMM or SGEMM)
     * for a sub-block.  Used by the split path in emulated_gemm_impl when the
     * second half's emulation fails after the first half has already written D,
     * so that D is fully correct without corrupting the first half's output.
     *
     * T = double → DGEMM (HIP_R_64F, HIPBLAS_COMPUTE_64F)
     * T = float  → SGEMM (HIP_R_32F, HIPBLAS_COMPUTE_32F)
     *
     * A fresh hipblasLtHandle is created with emulation explicitly disabled
     * on BOTH sub-structs so that env-var triggers do not cause recursive
     * re-entry regardless of which type triggered the original fallback.
     * ========================================================================= */
    template <typename T>
    static rocblaslt_status native_gemm_fallback(const _rocblaslt_handle* h,
                                                 hipblasOperation_t       opA,
                                                 hipblasOperation_t       opB,
                                                 int64_t                  m,
                                                 int64_t                  n,
                                                 int64_t                  k,
                                                 const T*                 alpha,
                                                 const T*                 A,
                                                 int64_t                  lda,
                                                 const T*                 B,
                                                 int64_t                  ldb,
                                                 const T*                 beta,
                                                 const T*                 C,
                                                 int64_t                  ldc,
                                                 T*                       D,
                                                 int64_t                  ldd,
                                                 hipStream_t              stream)
    {
        static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>,
                      "native_gemm_fallback<T>: T must be double (FP64) or float (FP32)");
        /* Resolve hip data type and compute type from T at compile time. */
        constexpr hipDataType          hip_type = std::is_same_v<T, double> ? HIP_R_64F : HIP_R_32F;
        constexpr hipblasComputeType_t ctype
            = std::is_same_v<T, double> ? HIPBLAS_COMPUTE_64F : HIPBLAS_COMPUTE_32F;

        hipblasLtHandle_t       native_handle = nullptr;
        hipblasLtMatrixLayout_t layoutA       = nullptr;
        hipblasLtMatrixLayout_t layoutB       = nullptr;
        hipblasLtMatrixLayout_t layoutC       = nullptr;
        hipblasLtMatrixLayout_t layoutD       = nullptr;
        hipblasLtMatmulDesc_t   desc          = nullptr;

        auto cleanup = [&]() noexcept {
            if(desc)
                (void)hipblasLtMatmulDescDestroy(desc);
            if(layoutD)
                (void)hipblasLtMatrixLayoutDestroy(layoutD);
            if(layoutC)
                (void)hipblasLtMatrixLayoutDestroy(layoutC);
            if(layoutB)
                (void)hipblasLtMatrixLayoutDestroy(layoutB);
            if(layoutA)
                (void)hipblasLtMatrixLayoutDestroy(layoutA);
            if(native_handle)
                (void)hipblasLtDestroy(native_handle);
        };

        if(hipblasLtCreate(&native_handle) != HIPBLAS_STATUS_SUCCESS)
            return rocblaslt_status_internal_error;

        /* Physical (stored) matrix dimensions for column-major layout:
         *   opA=N → A is m×k; opA=T → A is k×m (transposed in matmulDesc).
         *   opB=N → B is k×n; opB=T → B is n×k.                             */
        const uint64_t rows_A
            = (opA == HIPBLAS_OP_N) ? static_cast<uint64_t>(m) : static_cast<uint64_t>(k);
        const uint64_t cols_A
            = (opA == HIPBLAS_OP_N) ? static_cast<uint64_t>(k) : static_cast<uint64_t>(m);
        const uint64_t rows_B
            = (opB == HIPBLAS_OP_N) ? static_cast<uint64_t>(k) : static_cast<uint64_t>(n);
        const uint64_t cols_B
            = (opB == HIPBLAS_OP_N) ? static_cast<uint64_t>(n) : static_cast<uint64_t>(k);

        hipblasLtMatrixLayoutCreate(&layoutA, hip_type, rows_A, cols_A, lda);
        hipblasLtMatrixLayoutCreate(&layoutB, hip_type, rows_B, cols_B, ldb);
        hipblasLtMatrixLayoutCreate(
            &layoutC, hip_type, static_cast<uint64_t>(m), static_cast<uint64_t>(n), ldc);
        hipblasLtMatrixLayoutCreate(
            &layoutD, hip_type, static_cast<uint64_t>(m), static_cast<uint64_t>(n), ldd);
        hipblasLtMatmulDescCreate(&desc, ctype, hip_type);
        {
            /* Force-disable emulation on this desc to prevent recursive re-entry. */
            const int32_t emul_off = 0;
            hipblasLtMatmulDescSetAttribute(
                desc, HIPBLASLT_MATMUL_DESC_EMULATION_ENABLED_EXT, &emul_off, sizeof(emul_off));
        }
        hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
        hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

        const hipblasStatus_t st = hipblasLtMatmul(native_handle,
                                                   desc,
                                                   alpha,
                                                   A,
                                                   layoutA,
                                                   B,
                                                   layoutB,
                                                   beta,
                                                   C,
                                                   layoutC,
                                                   D,
                                                   layoutD,
                                                   nullptr,
                                                   nullptr,
                                                   0,
                                                   stream);
        /* Synchronize the stream before destroying the handle.
         * hipblasLtMatmul() is asynchronous: the kernel is enqueued on the
         * stream and control returns immediately.  The subsequent cleanup()
         * calls hipblasLtDestroy(native_handle), which calls hipModuleUnload()
         * for all GPU kernel modules.  If the kernel is still executing when
         * its module is unloaded, the in-flight kernel reads from freed GPU
         * memory → GPU fault → all subsequent hipModuleUnload() calls fail.  */
        const hipError_t sync_err = hipStreamSynchronize(stream);
        (void)hipGetLastError();
        cleanup();
        return (st == HIPBLAS_STATUS_SUCCESS && sync_err == hipSuccess)
                   ? rocblaslt_status_success
                   : rocblaslt_status_internal_error;
    }

    /* =========================================================================
     * fp64EmulatedGemm — profiling accumulator + implementation
     * ========================================================================= */

    /* Aggregates per-component GPU times across all leaf sub-GEMMs and counts
     * how many leaf (monolithic) sub-GEMMs were executed.  Owned by the public
     * fp64EmulatedGemm wrapper; passed by pointer through recursive calls.     */
    struct ProfileAccum
    {
        float t_prelim      = 0.f;
        float t_prelim_gemm = 0.f;
        float t_refine      = 0.f;
        float t_adp   = 0.f; /* ADP reduce kernels + hipStreamSynchronize (dynamic mode only) */
        float t_scale = 0.f;
        float t_int8  = 0.f;
        float t_accum = 0.f;
        /* CPU-side overhead timers (wall-clock, not GPU events) */
        float    t_ctx_create_ms  = 0.f; /* hipblasLtCreate                               */
        float    t_layout_ms      = 0.f; /* 3× MatrixLayoutCreate + DescCreate + 2× SetAttr */
        float    t_svmask_sync_ms = 0.f; /* NaN-flag hipStreamSynchronize + hipMemcpy       */
        float    t_batch_attr_ms  = 0.f; /* 6× MatrixLayoutSetAttribute for strided-batch   */
        unsigned effective_s_used = 0u; /* ADP: actual s chosen (= num_moduli in fixed mode) */
        unsigned n_sub_gemms      = 0u;
    };

    /* ── Generic index dispatcher ──────────────────────────────────────────────────────── */
    /* Calls fn(integral_constant<unsigned, N>{}) for the unique N in [1, N_MAX]     */
    /* such that actual == N.  Uses a C++17 fold expression to avoid a switch.       */
    template <unsigned N_MAX, typename Fn, std::size_t... Ns>
    static void dispatch_by_index_impl(unsigned actual, Fn&& fn, std::index_sequence<Ns...>)
    {
        (...,
         (void)((actual == Ns + 1u) && (fn(std::integral_constant<unsigned, Ns + 1u>{}), true)));
    }

    template <unsigned N_MAX, typename Fn>
    static void dispatch_by_index(unsigned actual, Fn&& fn)
    {
        dispatch_by_index_impl<N_MAX>(
            actual, std::forward<Fn>(fn), std::make_index_sequence<N_MAX>{});
    }

    /* ── Preliminary shift+extraction kernel launch helper ────────────────────── */
    /* Launches the four accu_prelim kernels for A and B (transpose-aware).
     * CHECK_NAN: true when NaN/Inf detection is active.
     * T: element type of the input matrices (double or float).                    */
    template <bool CHECK_NAN, typename T>
    static void launch_prelim_kernels(bool        tA,
                                      bool        tB,
                                      const T*    A,
                                      int64_t     m,
                                      int64_t     k,
                                      int64_t     lda,
                                      int8_t*     A8i_high,
                                      size_t      lda8i,
                                      int16_t*    sftA,
                                      uint32_t*   nan_flag,
                                      const T*    B,
                                      int64_t     n,
                                      int64_t     ldb,
                                      int8_t*     B8i_high,
                                      size_t      ldb8i,
                                      int16_t*    sftB,
                                      hipStream_t stream)
    {
        const unsigned m_blks_A_T = static_cast<unsigned>(m);
        const unsigned m_blks_A_N
            = static_cast<unsigned>((m + OZ2_PRELIM_SHMEM_TILE_M - 1) / OZ2_PRELIM_SHMEM_TILE_M);
        const unsigned n_blks_B_N = static_cast<unsigned>(n);
        const unsigned n_blks_B_T
            = static_cast<unsigned>((n + OZ2_PRELIM_SHMEM_TILE_M - 1) / OZ2_PRELIM_SHMEM_TILE_M);
        if(tA)
            hipLaunchKernelGGL((accu_prelim_A_T_kernel<CHECK_NAN, T>),
                               dim3(m_blks_A_T),
                               dim3(OZ2_PRELIM_COALESC_THRS),
                               0,
                               stream,
                               A,
                               m,
                               k,
                               lda,
                               A8i_high,
                               lda8i,
                               sftA,
                               nan_flag);
        else
            hipLaunchKernelGGL((accu_prelim_A_N_kernel<CHECK_NAN, T>),
                               dim3(m_blks_A_N),
                               dim3(OZ2_PRELIM_TILE_K * OZ2_PRELIM_SHMEM_TILE_M),
                               0,
                               stream,
                               A,
                               m,
                               k,
                               lda,
                               A8i_high,
                               lda8i,
                               sftA,
                               nan_flag);
        if(!tB)
            hipLaunchKernelGGL((accu_prelim_B_N_kernel<CHECK_NAN, T>),
                               dim3(n_blks_B_N),
                               dim3(OZ2_PRELIM_COALESC_THRS),
                               0,
                               stream,
                               B,
                               n,
                               k,
                               ldb,
                               B8i_high,
                               ldb8i,
                               sftB,
                               nan_flag);
        else
            hipLaunchKernelGGL((accu_prelim_B_T_kernel<CHECK_NAN, T>),
                               dim3(n_blks_B_T),
                               dim3(OZ2_PRELIM_TILE_K * OZ2_PRELIM_SHMEM_TILE_M),
                               0,
                               stream,
                               B,
                               n,
                               k,
                               ldb,
                               B8i_high,
                               ldb8i,
                               sftB,
                               nan_flag);
    }

    /* ── Scale kernel launch helpers ──────────────────────────────────────────── */
    /* T: input element type (double or float).  The scale kernels load T values
     * and widen to double immediately — all residue arithmetic stays in double.  */
    template <unsigned TC, typename T>
    static void launch_scale_A(bool        tA,
                               const T*    A,
                               int64_t     m,
                               int64_t     lda,
                               int8_t*     A8i,
                               size_t      lda8i,
                               size_t      cola8i,
                               int16_t*    sftA,
                               int64_t     k,
                               unsigned    sc_start,
                               hipStream_t stream)
    {
        const unsigned k_c
            = static_cast<unsigned>((k + 4u * OZ2_SCALE_TILE_K - 1u) / (4u * OZ2_SCALE_TILE_K));
        const unsigned k_x = static_cast<unsigned>((k + OZ2_SCALE_TILE_K - 1) / OZ2_SCALE_TILE_K);
        if(tA)
            hipLaunchKernelGGL((scale_A_T_kernel<TC, T>),
                               dim3(k_c,
                                    static_cast<unsigned>((m + OZ2_SCALE_COALESC_TILE_M - 1)
                                                          / OZ2_SCALE_COALESC_TILE_M)),
                               dim3(OZ2_SCALE_TILE_K * OZ2_SCALE_COALESC_TILE_M),
                               0,
                               stream,
                               A,
                               m,
                               lda,
                               A8i,
                               lda8i,
                               cola8i,
                               sftA,
                               k,
                               sc_start);
        else
            hipLaunchKernelGGL((scale_A_N_kernel<TC, T>),
                               dim3(k_x,
                                    static_cast<unsigned>((m + OZ2_SCALE_SHMEM_TILE_M - 1)
                                                          / OZ2_SCALE_SHMEM_TILE_M)),
                               dim3(OZ2_SCALE_SHMEM_BLOCK_DIM),
                               0,
                               stream,
                               A,
                               m,
                               lda,
                               A8i,
                               lda8i,
                               cola8i,
                               sftA,
                               k,
                               sc_start);
    }

    template <unsigned TC, typename T>
    static void launch_scale_B(bool        tB,
                               const T*    B,
                               int64_t     n,
                               int64_t     ldb,
                               int8_t*     B8i,
                               size_t      ldb8i,
                               int16_t*    sftB,
                               int64_t     k,
                               unsigned    sc_start,
                               hipStream_t stream)
    {
        const unsigned k_c
            = static_cast<unsigned>((k + 4u * OZ2_SCALE_TILE_K - 1u) / (4u * OZ2_SCALE_TILE_K));
        const unsigned k_x = static_cast<unsigned>((k + OZ2_SCALE_TILE_K - 1) / OZ2_SCALE_TILE_K);
        if(!tB)
            hipLaunchKernelGGL((scale_B_N_kernel<TC, T>),
                               dim3(k_c,
                                    static_cast<unsigned>((n + OZ2_SCALE_COALESC_TILE_M - 1)
                                                          / OZ2_SCALE_COALESC_TILE_M)),
                               dim3(OZ2_SCALE_TILE_K * OZ2_SCALE_COALESC_TILE_M),
                               0,
                               stream,
                               B,
                               n,
                               ldb,
                               B8i,
                               ldb8i,
                               sftB,
                               k,
                               sc_start);
        else
            hipLaunchKernelGGL((scale_B_T_kernel<TC, T>),
                               dim3(k_x,
                                    static_cast<unsigned>((n + OZ2_SCALE_SHMEM_TILE_M - 1)
                                                          / OZ2_SCALE_SHMEM_TILE_M)),
                               dim3(OZ2_SCALE_SHMEM_BLOCK_DIM),
                               0,
                               stream,
                               B,
                               n,
                               ldb,
                               B8i,
                               ldb8i,
                               sftB,
                               k,
                               sc_start);
    }

    /* ── Accumulate/finalize dispatch helper ─────────────────────────────────── */
    /* Encapsulates the CRT accumulation and finalization kernel dispatch.  The   */
    /* grid is computed from m and n; CS is the compile-time chunk size (1..18). */
    template <unsigned CS, typename T_out = double>
    static void dispatch_accum_chunk(bool           is_first,
                                     bool           is_last,
                                     bool           has_lo,
                                     const int32_t* C32i_batch,
                                     double*        Zhi,
                                     double*        Zlo,
                                     int64_t        m,
                                     int64_t        n,
                                     size_t         ldc32i,
                                     unsigned       chunk_start,
                                     unsigned       effective_s,
                                     const T_out*   C,
                                     T_out*         D,
                                     int64_t        ldc,
                                     int64_t        ldd,
                                     double         alpha,
                                     double         beta,
                                     const int16_t* sftA,
                                     const int16_t* sftB,
                                     hipStream_t    stream)
    {
        const dim3 blk_acc(64, 8);
        const dim3 grid_acc(static_cast<unsigned>((m + 63) / 64),
                            static_cast<unsigned>((n + 7) / 8));

        if(is_last)
        {
            if(has_lo)
            {
                if(is_first)
                    hipLaunchKernelGGL((accum_finalize_kernel<true, CS, true, T_out>),
                                       grid_acc,
                                       blk_acc,
                                       0,
                                       stream,
                                       C32i_batch,
                                       Zhi,
                                       Zlo,
                                       C,
                                       D,
                                       m,
                                       n,
                                       ldc32i,
                                       ldc,
                                       ldd,
                                       alpha,
                                       beta,
                                       sftA,
                                       sftB,
                                       chunk_start,
                                       effective_s);
                else
                    hipLaunchKernelGGL((accum_finalize_kernel<true, CS, false, T_out>),
                                       grid_acc,
                                       blk_acc,
                                       0,
                                       stream,
                                       C32i_batch,
                                       Zhi,
                                       Zlo,
                                       C,
                                       D,
                                       m,
                                       n,
                                       ldc32i,
                                       ldc,
                                       ldd,
                                       alpha,
                                       beta,
                                       sftA,
                                       sftB,
                                       chunk_start,
                                       effective_s);
            }
            else
            {
                if(is_first)
                    hipLaunchKernelGGL((accum_finalize_kernel<false, CS, true, T_out>),
                                       grid_acc,
                                       blk_acc,
                                       0,
                                       stream,
                                       C32i_batch,
                                       Zhi,
                                       Zlo,
                                       C,
                                       D,
                                       m,
                                       n,
                                       ldc32i,
                                       ldc,
                                       ldd,
                                       alpha,
                                       beta,
                                       sftA,
                                       sftB,
                                       chunk_start,
                                       effective_s);
                else
                    hipLaunchKernelGGL((accum_finalize_kernel<false, CS, false, T_out>),
                                       grid_acc,
                                       blk_acc,
                                       0,
                                       stream,
                                       C32i_batch,
                                       Zhi,
                                       Zlo,
                                       C,
                                       D,
                                       m,
                                       n,
                                       ldc32i,
                                       ldc,
                                       ldd,
                                       alpha,
                                       beta,
                                       sftA,
                                       sftB,
                                       chunk_start,
                                       effective_s);
            }
        }
        else
        {
            if(has_lo)
            {
                if(is_first)
                    hipLaunchKernelGGL((chunk_accum_kernel<true, CS, true>),
                                       grid_acc,
                                       blk_acc,
                                       0,
                                       stream,
                                       C32i_batch,
                                       Zhi,
                                       Zlo,
                                       m,
                                       n,
                                       ldc32i,
                                       chunk_start,
                                       effective_s);
                else
                    hipLaunchKernelGGL((chunk_accum_kernel<true, CS, false>),
                                       grid_acc,
                                       blk_acc,
                                       0,
                                       stream,
                                       C32i_batch,
                                       Zhi,
                                       Zlo,
                                       m,
                                       n,
                                       ldc32i,
                                       chunk_start,
                                       effective_s);
            }
            else
            {
                if(is_first)
                    hipLaunchKernelGGL((chunk_accum_kernel<false, CS, true>),
                                       grid_acc,
                                       blk_acc,
                                       0,
                                       stream,
                                       C32i_batch,
                                       Zhi,
                                       Zlo,
                                       m,
                                       n,
                                       ldc32i,
                                       chunk_start,
                                       effective_s);
                else
                    hipLaunchKernelGGL((chunk_accum_kernel<false, CS, false>),
                                       grid_acc,
                                       blk_acc,
                                       0,
                                       stream,
                                       C32i_batch,
                                       Zhi,
                                       Zlo,
                                       m,
                                       n,
                                       ldc32i,
                                       chunk_start,
                                       effective_s);
            }
        }
    }

    /* =========================================================================
     * Per-device cache for the INT8 hipblasLtHandle and matmulDesc.
     *
     * hipblasLtCreate (profiled as cpu_ctx_create_ms) is the dominant overhead
     * for small problems because it is called once per emulated_gemm_impl leaf
     * invocation and can take several hundred microseconds.
     *
     * Both objects are completely device-specific but problem-shape–agnostic:
     *   • int8_handle — a plain hipblasLt context; no per-call mutable state.
     *   • matmulDesc  — always TRANSA=T, TRANSB=N, COMPUTE_32I, R_32I; never
     *                   mutated after creation (batch attributes live in the
     *                   per-call MatrixLayout objects, not in matmulDesc).
     *
     * layoutA/B/CD are NOT cached: they encode (k, m, n) dimensions and their
     * batch_count attribute is updated per-chunk, so they remain per-call.
     *
     * Thread safety: the mutex is held only during the one-time initialisation.
     * After warm-up the mutex is taken and released immediately (~30 ns) and
     * the pointers read under it never change, so there is no contention on
     * the hot path.  hipblasLtMatmul is safe to call concurrently from multiple
     * threads sharing the same handle (the library protects its internal
     * algorithm-selection cache, mirroring the cublasLt threading model).
     * ========================================================================= */
    struct DeviceInt8Cache
    {
        std::mutex            mu;
        hipblasLtHandle_t     int8_handle = nullptr;
        hipblasLtMatmulDesc_t matmulDesc  = nullptr;
    };
    static DeviceInt8Cache s_device_cache[64]; /* one slot per HIP device index */

    /* ── RAII guard for INT8 GEMM handles and profiling events ───────────────── */
    /* Destroyed automatically on scope exit — covers both normal return and  */
    /* all early-return error paths, eliminating explicit oz2_cleanup() calls. */
    struct Ozaki2Context
    {
        hipblasLtHandle_t       int8_handle = nullptr;
        hipblasLtMatrixLayout_t layoutA     = nullptr;
        hipblasLtMatrixLayout_t layoutB     = nullptr;
        hipblasLtMatrixLayout_t layoutCD    = nullptr;
        hipblasLtMatmulDesc_t   matmulDesc  = nullptr;
        hipEvent_t              ev0         = nullptr;
        hipEvent_t              ev1         = nullptr;
        /* When true the destructor owns and destroys the object.
         * When false the per-device cache owns the lifetime and the
         * destructor skips it — avoiding a premature hipblasLtDestroy
         * that would invalidate the cached handle for all future calls. */
        bool owns_handle = true;
        bool owns_desc   = true;

        Ozaki2Context()                                = default;
        Ozaki2Context(const Ozaki2Context&)            = delete;
        Ozaki2Context& operator=(const Ozaki2Context&) = delete;

        ~Ozaki2Context() noexcept
        {
            if(matmulDesc && owns_desc)
                (void)hipblasLtMatmulDescDestroy(matmulDesc);
            if(layoutCD)
                (void)hipblasLtMatrixLayoutDestroy(layoutCD);
            if(layoutB)
                (void)hipblasLtMatrixLayoutDestroy(layoutB);
            if(layoutA)
                (void)hipblasLtMatrixLayoutDestroy(layoutA);
            if(int8_handle && owns_handle)
                (void)hipblasLtDestroy(int8_handle);
            if(ev1)
                (void)hipEventDestroy(ev1);
            if(ev0)
                (void)hipEventDestroy(ev0);
        }
    };

    /* Internal implementation — called recursively during binary-halving.
     * T = double → FP64 path; T = float → FP32 path.
     * All GPU kernels (prelim, scale, finalize) are templated on T so no
     * intermediate copies are needed for either type.
     * prof != nullptr enables per-component accumulation across all leaves.   */
    template <typename T>
    static rocblaslt_status emulated_gemm_impl(const _rocblaslt_handle*           h,
                                               hipblasOperation_t                 opA,
                                               hipblasOperation_t                 opB,
                                               int64_t                            m,
                                               int64_t                            n,
                                               int64_t                            k,
                                               const T*                           alpha,
                                               const T*                           A,
                                               int64_t                            lda,
                                               const T*                           B,
                                               int64_t                            ldb,
                                               const T*                           beta,
                                               const T*                           C,
                                               int64_t                            ldc,
                                               T*                                 D,
                                               int64_t                            ldd,
                                               hipStream_t                        stream,
                                               const FixedPointEmulationSettings& settings,
                                               ProfileAccum*                      prof)
    {
        static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>,
                      "emulated_gemm_impl<T>: T must be double (FP64) or float (FP32)");
        /* Type-specific label strings for warning messages. */
        constexpr const char* type_name  = std::is_same_v<T, double> ? "FP64" : "FP32";
        constexpr const char* native_str = std::is_same_v<T, double> ? "DGEMM" : "SGEMM";
        /* Always ADP mode: workspace layout always covers max_num_moduli<T>() moduli.
         * fixedPointEmulationNumModuli() post-ADP override is applied later. */
        const unsigned num_moduli = max_num_moduli<T>();
        /* Workspace budget: use max_num_moduli<T>() because ADP selects effective_s
         * at runtime — the workspace must accommodate up to that many moduli. */
        const size_t ws_budget
            = compute_budget_from_ws(m, n, k, num_moduli, settings.workspace_bytes);

        {
            const unsigned chunk_sz = compute_chunk_size(m, n, k, num_moduli, ws_budget);
            const unsigned n_chunks = (num_moduli + chunk_sz - 1u) / chunk_sz;

            /* ws_budget == 0 means the Zhi/Zlo double-accumulator arrays (each m×n
             * doubles) cannot fit alongside the INT8 arrays in the workspace.
             * The monolithic path would write beyond the workspace buffer → unsafe.
             * Force a binary spatial split on the larger of m or n to reduce m×n
             * (and hence the Zhi/Zlo requirement) until ws_budget > 0.
             * This case is also entered when n_chunks > 1 (multiple chunk passes). */
            if(n_chunks > 1u || ws_budget == 0u)
            {
                const bool    split_m = (m >= n);
                const int64_t half_m  = split_m ? m / 2 : m;
                const int64_t half_n  = split_m ? n : n / 2;
                const int64_t m2      = split_m ? (m - m / 2) : m;
                const int64_t n2      = split_m ? n : (n - n / 2);

                const int  device = h->device;
                const bool tA     = (opA != HIPBLAS_OP_N);
                const bool tB     = (opB != HIPBLAS_OP_N);

                /* Resolve the HIP data type from T for the performance model. */
                constexpr hipDataType type_a_perf
                    = std::is_same_v<T, double> ? HIP_R_64F : HIP_R_32F;

                /* When ws_budget == 0, the performance model would return degenerate
                 * predictions (chunk_size=1, always-split to tiny sub-GEMMs that look
                 * fast but are not).  Skip the model and force the split immediately. */
                const bool force_split = (ws_budget == 0u);
                /* ADP-aware performance model s: use the minimum s that satisfies the
                 * precision target instead of S_MAX, to avoid overestimating emulation
                 * cost in dynamic mode.                                               */
                const int      adp_bits_impl = (settings.adp_mantissa_bits > 0)
                                                   ? static_cast<int>(settings.adp_mantissa_bits)
                                                   : fixedPointEmulationAdpMantissaBits(type_a_perf);
                const unsigned pm_s_impl     = adp_expected_num_moduli(adp_bits_impl, num_moduli);
                const double   t_mono
                    = force_split ? 0.0
                                  : perf_model_times(
                                        tA, tB, m, n, k, pm_s_impl, device, ws_budget, type_a_perf)
                                        .t_total_ms;
                /* Pass workspace_bytes so the sub-call recomputes its own budget
                 * from the smaller (half_m, half_n) dimensions.                  */
                const double t_split = force_split
                                           ? 0.0
                                           : 2.
                                                 * effective_time_ms(tA,
                                                                     tB,
                                                                     half_m,
                                                                     half_n,
                                                                     k,
                                                                     pm_s_impl,
                                                                     device,
                                                                     settings.workspace_bytes,
                                                                     type_a_perf);

                if(force_split || t_split < t_mono)
                {
                    {
                        rocblaslt_status st = emulated_gemm_impl<T>(h,
                                                                    opA,
                                                                    opB,
                                                                    half_m,
                                                                    half_n,
                                                                    k,
                                                                    alpha,
                                                                    A,
                                                                    lda,
                                                                    B,
                                                                    ldb,
                                                                    beta,
                                                                    C,
                                                                    ldc,
                                                                    D,
                                                                    ldd,
                                                                    stream,
                                                                    settings,
                                                                    prof);
                        if(st != rocblaslt_status_success)
                            return st;
                    }

                    const T* const         A2  = split_m ? (tA ? A + half_m * lda : A + half_m) : A;
                    const T* const         B2  = split_m ? B : (tB ? B + half_n : B + half_n * ldb);
                    const T* const         C2  = split_m ? C + half_m : C + half_n * ldc;
                    T* const               D2  = split_m ? D + half_m : D + half_n * ldd;
                    const rocblaslt_status st2 = emulated_gemm_impl<T>(h,
                                                                       opA,
                                                                       opB,
                                                                       m2,
                                                                       n2,
                                                                       k,
                                                                       alpha,
                                                                       A2,
                                                                       lda,
                                                                       B2,
                                                                       ldb,
                                                                       beta,
                                                                       C2,
                                                                       ldc,
                                                                       D2,
                                                                       ldd,
                                                                       stream,
                                                                       settings,
                                                                       prof);
                    if(st2 != rocblaslt_status_success)
                    {
                        hipblaslt_cerr
                            << "[hipBLASLt " << type_name
                            << " emulation] WARNING: second-half emulation failed " << "(m=" << m2
                            << ", n=" << n2 << ", k=" << k << ", st=" << (int)st2 << "). "
                            << "Running native " << native_str
                            << " for second half to preserve first-half output." << std::endl;
                        return native_gemm_fallback<T>(h,
                                                       opA,
                                                       opB,
                                                       m2,
                                                       n2,
                                                       k,
                                                       alpha,
                                                       A2,
                                                       lda,
                                                       B2,
                                                       ldb,
                                                       beta,
                                                       C2,
                                                       ldc,
                                                       D2,
                                                       ldd,
                                                       stream);
                    }
                    return rocblaslt_status_success;
                }
            }
        }
        /* ── Existing monolithic path ────────────────────────────────────── */

        /* Minimum problem size guard — see FIXED_POINT_EMUL_MIN_MN. */
        if(m < FIXED_POINT_EMUL_MIN_MN || n < FIXED_POINT_EMUL_MIN_MN)
            return native_gemm_fallback<T>(
                h, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd, stream);

        const bool    _prof = (prof != nullptr);
        Ozaki2Context context;
        float         _t_prelim = 0, _t_prelim_gemm = 0, _t_refine = 0, _t_adp = 0, _t_scale = 0,
              _t_int8 = 0, _t_accum = 0;
        /* CPU-side overhead timers — wall-clock, only used when profiling. */
        using WallClock     = std::chrono::steady_clock;
        using WallMs        = std::chrono::duration<float, std::milli>;
        float _t_ctx_create = 0.f, _t_layout = 0.f, _t_svmask_sync = 0.f, _t_batch_attr = 0.f;
        if(_prof)
        {
            (void)hipEventCreate(&context.ev0);
            (void)hipEventCreate(&context.ev1);
        }
        auto _pstart = [&]() noexcept {
            if(_prof)
                (void)hipEventRecord(context.ev0, stream);
        };
        auto _pstop = [&](float& t) noexcept {
            if(_prof)
            {
                (void)hipEventRecord(context.ev1, stream);
                (void)hipStreamSynchronize(stream);
                float ms = 0.f;
                (void)hipEventElapsedTime(&ms, context.ev0, context.ev1);
                t += ms;
            }
        };

        /* Always ADP (dynamic) mode: layout always covers max_num_moduli<T>() moduli. */
        const unsigned layout_moduli = num_moduli;
        const unsigned chunk_size    = compute_chunk_size(m, n, k, layout_moduli, ws_budget);

        const size_t lda8i  = pad(static_cast<size_t>(k));
        const size_t cola8i = pad(static_cast<size_t>(m));
        const size_t ldb8i  = lda8i;
        const size_t ldc32i = cola8i;
        const size_t padn   = pad(static_cast<size_t>(n));
        const size_t szC32i = ldc32i * static_cast<size_t>(n);

        const size_t szA8i        = chunk_size * lda8i * cola8i;
        const size_t szB8i        = chunk_size * ldb8i * static_cast<size_t>(n);
        const size_t szZhi        = (chunk_size < layout_moduli) ? szC32i : 0u;
        const size_t szZlo        = szZhi;
        const size_t n_c32i_slots = static_cast<size_t>(chunk_size);
        const size_t szSftA       = cola8i;
        const size_t szSftB       = padn;
        const size_t szNanFlag    = 1;
        const size_t szRowMax     = cola8i;
        const size_t szColMax     = padn;

        char* const ws = static_cast<char*>(settings.workspace);

        int8_t* const    A8i        = reinterpret_cast<int8_t*>(ws);
        int8_t* const    B8i        = A8i + szA8i;
        int32_t* const   C32i_batch = reinterpret_cast<int32_t*>(B8i + szB8i);
        double* const    Zhi        = reinterpret_cast<double*>(C32i_batch + n_c32i_slots * szC32i);
        double* const    Zlo        = Zhi + szZhi;
        int16_t* const   sftA       = reinterpret_cast<int16_t*>(Zlo + szZlo);
        int16_t* const   sftB       = sftA + szSftA;
        uint32_t* const  nan_flag   = reinterpret_cast<uint32_t*>(sftB + szSftB);
        int32_t* const   row_max    = reinterpret_cast<int32_t*>(nan_flag + szNanFlag);
        int32_t* const   col_max    = row_max + szRowMax;
        float* const     adp_buf    = reinterpret_cast<float*>(col_max + szColMax);
        void* const      int8_ws    = static_cast<void*>(adp_buf + 2);
        constexpr size_t int8_ws_size = OZ2_INT8_GEMM_WS_BYTES;
        int32_t* const   C32i         = C32i_batch;

        int8_t* const A8i_high = A8i;
        int8_t* const B8i_high = B8i;

        const bool tA = (opA != HIPBLAS_OP_N);
        const bool tB = (opB != HIPBLAS_OP_N);

        const uint32_t svmask
            = (settings.sv_mask != ~0u) ? settings.sv_mask : fixedPointEmulationSpecialValuesMask();

        if(svmask != 0u)
        {
            if(hipMemsetAsync(nan_flag, 0, sizeof(uint32_t), stream) != hipSuccess)
            {
                return rocblaslt_status_internal_error;
            }
        }

        {
            /* Obtain the INT8 handle from the per-device cache.
             * First call for this device: create and cache under the mutex.
             * Subsequent calls: mutex acquired/released (~30 ns), pointer copied.
             * Exotic device index (≥64): fall back to per-call creation.        */
            auto      _wt0 = WallClock::now();
            const int dev  = h->device;
            if(dev >= 0 && dev < 64)
            {
                auto&                       slot = s_device_cache[dev];
                std::lock_guard<std::mutex> lk(slot.mu);
                if(slot.int8_handle == nullptr)
                {
                    if(hipblasLtCreate(&slot.int8_handle) != HIPBLAS_STATUS_SUCCESS)
                        return rocblaslt_status_internal_error;
                }
                context.int8_handle = slot.int8_handle;
                context.owns_handle = false; /* cache owns the lifetime */
            }
            else
            {
                /* Exotic device index: fall back to per-call creation. */
                if(hipblasLtCreate(&context.int8_handle) != HIPBLAS_STATUS_SUCCESS)
                    return rocblaslt_status_internal_error;
                /* owns_handle remains true; destructor will destroy it. */
            }
            if(_prof)
                _t_ctx_create = WallMs(WallClock::now() - _wt0).count();
        }

        {
            auto _wt0 = WallClock::now();
            hipblasLtMatrixLayoutCreate(&context.layoutA,
                                        HIP_R_8I,
                                        static_cast<uint64_t>(k),
                                        static_cast<uint64_t>(m),
                                        static_cast<int64_t>(lda8i));
            hipblasLtMatrixLayoutCreate(&context.layoutB,
                                        HIP_R_8I,
                                        static_cast<uint64_t>(k),
                                        static_cast<uint64_t>(n),
                                        static_cast<int64_t>(ldb8i));
            hipblasLtMatrixLayoutCreate(&context.layoutCD,
                                        HIP_R_32I,
                                        static_cast<uint64_t>(m),
                                        static_cast<uint64_t>(n),
                                        static_cast<int64_t>(ldc32i));
            /* matmulDesc is always TRANSA=T, TRANSB=N, COMPUTE_32I, R_32I —
             * identical for every emulated_gemm_impl call on a given device.
             * Obtain it from the per-device cache (same slot used for int8_handle
             * above; mutex guards the one-time initialisation only).             */
            {
                const int dev = h->device;
                if(dev >= 0 && dev < 64)
                {
                    auto&                       slot = s_device_cache[dev];
                    std::lock_guard<std::mutex> lk(slot.mu);
                    if(slot.matmulDesc == nullptr)
                    {
                        hipblasLtMatmulDescCreate(&slot.matmulDesc, HIPBLAS_COMPUTE_32I, HIP_R_32I);
                        hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
                        hipblasLtMatmulDescSetAttribute(
                            slot.matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
                        hipblasLtMatmulDescSetAttribute(
                            slot.matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
                    }
                    context.matmulDesc = slot.matmulDesc;
                    context.owns_desc  = false; /* cache owns the lifetime */
                }
                else
                {
                    /* Exotic device index: fall back to per-call creation. */
                    hipblasLtMatmulDescCreate(&context.matmulDesc, HIPBLAS_COMPUTE_32I, HIP_R_32I);
                    hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
                    hipblasLtMatmulDescSetAttribute(
                        context.matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
                    hipblasLtMatmulDescSetAttribute(
                        context.matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
                    /* owns_desc remains true; destructor will destroy it. */
                }
            }
            if(_prof)
                _t_layout = WallMs(WallClock::now() - _wt0).count();
        }

        const int32_t one_i = 1, zero_i = 0;

        /* Preliminary shift + extraction: separate A and B kernels.
         * Coalesced paths (TRANS_A=T → A_T, TRANS_B=N → B_N): blockDim=256, 1 block/row or col.
         * SHMEM paths    (TRANS_A=N → A_N, TRANS_B=T → B_T): blockDim=1024, TILE_M=16.    */
        _pstart();
        if(svmask == 0u)
            launch_prelim_kernels<false>(tA,
                                         tB,
                                         A,
                                         m,
                                         k,
                                         lda,
                                         A8i_high,
                                         lda8i,
                                         sftA,
                                         nan_flag,
                                         B,
                                         n,
                                         ldb,
                                         B8i_high,
                                         ldb8i,
                                         sftB,
                                         stream);
        else
            launch_prelim_kernels<true>(tA,
                                        tB,
                                        A,
                                        m,
                                        k,
                                        lda,
                                        A8i_high,
                                        lda8i,
                                        sftA,
                                        nan_flag,
                                        B,
                                        n,
                                        ldb,
                                        B8i_high,
                                        ldb8i,
                                        sftB,
                                        stream);
        _pstop(_t_prelim);

        if(svmask != 0u)
        {
            auto _wt0 = WallClock::now();
            if(hipStreamSynchronize(stream) != hipSuccess)
            {
                return rocblaslt_status_internal_error;
            }
            uint32_t detected = 0u;
            if(hipMemcpy(&detected, nan_flag, sizeof(uint32_t), hipMemcpyDeviceToHost)
               != hipSuccess)
            {
                return rocblaslt_status_internal_error;
            }
            if(_prof)
                _t_svmask_sync = WallMs(WallClock::now() - _wt0).count();
            if(detected & svmask)
            {
                return rocblaslt_status_invalid_value;
            }
        }

        /* Preliminary INT8 GEMM: C32i_prelim = A8i_high^T × B8i_high */
        _pstart();
        {
            const hipblasStatus_t prelim_st = hipblasLtMatmul(context.int8_handle,
                                                              context.matmulDesc,
                                                              &one_i,
                                                              A8i_high,
                                                              context.layoutA,
                                                              B8i_high,
                                                              context.layoutB,
                                                              &zero_i,
                                                              C32i,
                                                              context.layoutCD,
                                                              C32i,
                                                              context.layoutCD,
                                                              nullptr,
                                                              int8_ws,
                                                              int8_ws_size,
                                                              stream);
            _pstop(_t_prelim_gemm);
            if(prelim_st != HIPBLAS_STATUS_SUCCESS)
            {
                hipblaslt_cerr << "[hipBLASLt " << type_name
                               << " emulation] WARNING: preliminary INT8 GEMM failed " << "(m=" << m
                               << ", n=" << n << ", k=" << k << ", status=" << (int)prelim_st
                               << "). " << "Falling back to native " << native_str << "."
                               << std::endl;
                /* Drain the stream before clearing the error so that the stream is in
                 * a clean state for future operations on this stream/context.       */
                (void)hipStreamSynchronize(stream);
                (void)hipGetLastError();
                return rocblaslt_status_internal_error;
            }
        }

        _pstart();
        const unsigned sftA_m_blks = static_cast<unsigned>((m + 63) / 64);
        const unsigned sftA_n_blks = static_cast<unsigned>((n + 63) / 64);
        /* adp_buf: 2 floats used by ADP (Adaptive Precision) kernels (see above).
         * Initialised to 0.0f (= biased −200 → effective_s=2 if all rows/cols zero). */

        /* effective_s: ADP may reduce this below num_moduli; determined BEFORE the
         * shift-refinement delta is applied so we can use the correct log2P.       */
        unsigned effective_s = num_moduli;

        (void)hipMemsetAsync(row_max, 0, szRowMax * sizeof(int32_t), stream);
        /* col_max is written directly by col_max_kernel (no atomicMax → no memset needed). */
        hipLaunchKernelGGL(refine_sftA_partial_kernel,
                           dim3(sftA_m_blks, sftA_n_blks),
                           dim3(64),
                           0,
                           stream,
                           C32i,
                           m,
                           n,
                           ldc32i,
                           row_max);
        /* Compute col_max[n] from C32i in a separate kernel (column-major read, no LDS contention).
         * Runs concurrently with the above in the same stream; both read the same C32i buffer.
         * adp_reduce_B and refine_sftB then read col_max instead of C32i. */
        hipLaunchKernelGGL(col_max_kernel,
                           dim3(static_cast<unsigned>(n)),
                           dim3(256),
                           0,
                           stream,
                           C32i,
                           m,
                           ldc32i,
                           col_max);
        _pstop(_t_refine); /* partial: refine_sftA_partial only */

        /* ADP always runs: determine effective_s from the preliminary GEMM result
         * BEFORE applying any shift delta.  Both ADP kernels read sftA_init and
         * sftB_init (the values set in step 1a/1b, before any += delta).
         * The refine delta is then applied using log2P_{effective_s} so that
         * X_true is sized to fit within M_{effective_s}/2.                        */
        {
            _pstart();
            (void)hipMemsetAsync(adp_buf, 0, 2 * sizeof(float), stream);
            /* Resolve ADP target precision: settings sentinel 0 → env var default (52). */
            const float adp_bits
                = static_cast<float>((settings.adp_mantissa_bits > 0)
                                         ? settings.adp_mantissa_bits
                                         : fixedPointEmulationAdpMantissaBits(
                                               std::is_same_v<T, double> ? HIP_R_64F : HIP_R_32F));

            /* A-side: reads row_max[] and sftA[] before apply_kernel modifies sftA. */
            hipLaunchKernelGGL(adp_reduce_A_kernel,
                               dim3(sftA_m_blks),
                               dim3(OZ2_PRELIM_COALESC_THRS),
                               0,
                               stream,
                               row_max,
                               sftA,
                               m,
                               adp_buf + 0,
                               adp_bits);
            /* B-side: reads precomputed col_max[] (not C32i) before refine_sftB_kernel modifies sftB. */
            hipLaunchKernelGGL(adp_reduce_B_kernel,
                               dim3((static_cast<unsigned>(n) + 255u) / 256u),
                               dim3(256),
                               0,
                               stream,
                               col_max,
                               n,
                               sftB,
                               adp_buf + 1,
                               adp_bits);

            /* Sync, copy 2 floats, compute effective_s on host.
             * hipGetLastError() clears any sticky thread-level error that may have
             * been set by the ADP kernels (e.g. from a previous GPU fault on the
             * same stream).  Without this, a subsequent hipMallocAsync on the same
             * stream may fail even though the device has plenty of free memory.   */
            (void)hipStreamSynchronize(stream);
            (void)hipGetLastError();
            _pstop(_t_adp); /* ADP reduce kernels + hipStreamSynchronize */
            float h_adp[2] = {0.0f, 0.0f};
            (void)hipMemcpy(h_adp, adp_buf, 2 * sizeof(float), hipMemcpyDeviceToHost);

            const float log2P_needed = std::max(h_adp[0], h_adp[1]) - 200.0f;

            if(log2P_needed > log2P(num_moduli - 2u))
            {
                /* ADP determined the type's max moduli are still insufficient.
                 * Before giving up, check whether HIPBLASLT_EMULATION_NUM_MODULI
                 * provides a forced override — if so, use it and proceed.
                 * This allows test/debug code to force a specific s even when the
                 * ADP analysis says no standard s is sufficient (accuracy is not
                 * guaranteed, but the computation will run rather than returning
                 * invalid_value).                                                 */
                const unsigned override_s_adp = fixedPointEmulationNumModuli();
                if(override_s_adp >= 2u && override_s_adp <= num_moduli)
                {
                    effective_s = override_s_adp;
                }
                else
                {
                    /* No valid override: fall back to native GEMM.
                     * This occurs for matrices with extremely large dynamic range within
                     * a single row/column (e.g. condition number ≫ 2^{2×log2P_18}).
                     * The Ozaki shift-refinement would set A8i_final or B8i_final to
                     * near-zero for the elements providing cancellation, giving wrong
                     * results regardless of s.  Rate-limited warning (≤5 per process). */
                    hipblaslt_cerr << "[hipBLASLt " << type_name
                                   << " emulation] WARNING: ADP overflow for GEMM " << "(m=" << m
                                   << ", n=" << n << ", k=" << k
                                   << "): " << "A-side log2P_req=" << (h_adp[0] - 200.0f)
                                   << " bits, " << "B-side=" << (h_adp[1] - 200.0f) << " bits, "
                                   << "max required=" << log2P_needed
                                   << " > supported max=" << log2P(num_moduli - 2u)
                                   << " (s=" << num_moduli << " moduli). Falling back to native "
                                   << native_str << "." << std::endl;
                    return rocblaslt_status_invalid_value;
                }
            }

            for(unsigned s = 2u; s <= num_moduli; ++s)
            {
                if(log2P(s - 2u) >= log2P_needed)
                {
                    effective_s = s;
                    break;
                }
            }
        }
        /* Post-ADP testing override: HIPBLASLT_EMULATION_NUM_MODULI overrides the
         * ADP-selected s.  Only for testing; accuracy is not guaranteed below ADP value. */
        {
            const unsigned override_s = fixedPointEmulationNumModuli();
            if(override_s >= 2u && override_s <= num_moduli)
                effective_s = override_s;
        }

        /* Apply shift-refinement delta using the correct log2P for effective_s.
         * This ensures X_true ≤ M_{effective_s}/4 < M_{effective_s}/2 (CRT safe). */
        const float refine_log2P = log2P(effective_s - 2u);

        _pstart();
        hipLaunchKernelGGL(refine_sftA_apply_kernel,
                           dim3(sftA_m_blks),
                           dim3(64),
                           0,
                           stream,
                           row_max,
                           sftA,
                           m,
                           refine_log2P);
        hipLaunchKernelGGL(refine_sftB_kernel,
                           dim3((static_cast<unsigned>(n) + 255u) / 256u),
                           dim3(256),
                           0,
                           stream,
                           col_max,
                           n,
                           sftB,
                           refine_log2P);
        _pstop(_t_refine); /* partial: refine_sftA_apply + refine_sftB */

        /* Scale → INT8 GEMM → CRT accum/finalize.
         * Flat single loop: each pass scales `actual` moduli into A8i[0..actual-1]
         * then runs one batched GEMM of batch_count=actual.                       */
        int32_t       batch_cur  = 0; /* set on first iteration */
        const int64_t stride_A_b = static_cast<int64_t>(lda8i * cola8i);
        const int64_t stride_B_b = static_cast<int64_t>(ldb8i * static_cast<size_t>(n));
        const int64_t stride_C_b = static_cast<int64_t>(szC32i);
        {
            auto _wt0 = WallClock::now();
            hipblasLtMatrixLayoutSetAttribute(context.layoutA,
                                              HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                              &stride_A_b,
                                              sizeof(stride_A_b));
            hipblasLtMatrixLayoutSetAttribute(context.layoutB,
                                              HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                              &stride_B_b,
                                              sizeof(stride_B_b));
            hipblasLtMatrixLayoutSetAttribute(context.layoutCD,
                                              HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                              &stride_C_b,
                                              sizeof(stride_C_b));
            if(_prof)
                _t_batch_attr += WallMs(WallClock::now() - _wt0).count();
        }

        for(unsigned chunk_start = 0; chunk_start < effective_s; chunk_start += chunk_size)
        {
            const unsigned actual = std::min(chunk_size, effective_s - chunk_start);

            /* Scale: write actual moduli into A8i[0..actual-1] / B8i[0..actual-1] */
            _pstart();
            dispatch_by_index<max_num_moduli<T>()>(actual, [&](auto Count) {
                constexpr unsigned TC = Count.value;
                launch_scale_A<TC>(tA, A, m, lda, A8i, lda8i, cola8i, sftA, k, chunk_start, stream);
                launch_scale_B<TC>(tB, B, n, ldb, B8i, ldb8i, sftB, k, chunk_start, stream);
            });
            _pstop(_t_scale);

            /* Update batch_count when it changes (normally constant; may differ on
             * the final pass when effective_s is not a multiple of chunk_size).    */
            if(static_cast<int32_t>(actual) != batch_cur)
            {
                batch_cur = static_cast<int32_t>(actual);
                auto _wt0 = WallClock::now();
                hipblasLtMatrixLayoutSetAttribute(context.layoutA,
                                                  HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                  &batch_cur,
                                                  sizeof(batch_cur));
                hipblasLtMatrixLayoutSetAttribute(context.layoutB,
                                                  HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                  &batch_cur,
                                                  sizeof(batch_cur));
                hipblasLtMatrixLayoutSetAttribute(context.layoutCD,
                                                  HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                  &batch_cur,
                                                  sizeof(batch_cur));
                if(_prof)
                    _t_batch_attr += WallMs(WallClock::now() - _wt0).count();
            }

            _pstart();
            {
                /* A8i and B8i both start at position 0: the scale pass wrote actual
                 * moduli into A8i[0..actual-1] / B8i[0..actual-1].                */
                const hipblasStatus_t batch_st = hipblasLtMatmul(context.int8_handle,
                                                                 context.matmulDesc,
                                                                 &one_i,
                                                                 A8i,
                                                                 context.layoutA,
                                                                 B8i,
                                                                 context.layoutB,
                                                                 &zero_i,
                                                                 C32i_batch,
                                                                 context.layoutCD,
                                                                 C32i_batch,
                                                                 context.layoutCD,
                                                                 nullptr,
                                                                 int8_ws,
                                                                 int8_ws_size,
                                                                 stream);
                _pstop(_t_int8);
                if(batch_st != HIPBLAS_STATUS_SUCCESS)
                {
                    hipblaslt_cerr << "[hipBLASLt " << type_name
                                   << " emulation] WARNING: INT8 batch GEMM failed " << "(m=" << m
                                   << ", n=" << n << ", k=" << k << ", batch_count=" << batch_cur
                                   << ", moduli_offset=" << chunk_start
                                   << ", status=" << (int)batch_st << "). "
                                   << "Falling back to native " << native_str << "." << std::endl;
                    (void)hipStreamSynchronize(stream);
                    (void)hipGetLastError();
                    return rocblaslt_status_internal_error;
                }
            }

            const bool is_first = (chunk_start == 0);
            const bool is_last  = (chunk_start + actual >= effective_s);
            const bool has_lo   = (effective_s > 7u);
            _pstart();
            dispatch_by_index<max_num_moduli<T>()>(actual, [&](auto Count) {
                dispatch_accum_chunk<Count.value>(is_first,
                                                  is_last,
                                                  has_lo,
                                                  C32i_batch,
                                                  Zhi,
                                                  Zlo,
                                                  m,
                                                  n,
                                                  ldc32i,
                                                  chunk_start,
                                                  effective_s,
                                                  C,
                                                  D,
                                                  ldc,
                                                  ldd,
                                                  *alpha,
                                                  *beta,
                                                  sftA,
                                                  sftB,
                                                  stream);
            });

            _pstop(_t_accum);
        }

        if(_prof)
        {
            /* Accumulate component times into the caller's accumulator. */
            prof->t_prelim += _t_prelim;
            prof->t_prelim_gemm += _t_prelim_gemm;
            prof->t_refine += _t_refine;
            prof->t_adp += _t_adp;
            prof->t_scale += _t_scale;
            prof->t_int8 += _t_int8;
            prof->t_accum += _t_accum;
            prof->t_ctx_create_ms += _t_ctx_create;
            prof->t_layout_ms += _t_layout;
            prof->t_svmask_sync_ms += _t_svmask_sync;
            prof->t_batch_attr_ms += _t_batch_attr;
            prof->effective_s_used = std::max(prof->effective_s_used, effective_s);
            prof->n_sub_gemms += 1u;
        }
        return rocblaslt_status_success;
    }

} // namespace FixedPointEmulation

bool fixedPointEmulationIsEager()
{
    static const bool v = []() -> bool {
        const auto parsed
            = fixedPointEmulationParseStrategyEnv(std::getenv("HIPBLASLT_EMULATION_STRATEGY"));
        return parsed.state == FIXED_POINT_EMULATION_ENV_VALID
               && parsed.value == static_cast<unsigned>(HIPBLASLT_EMULATION_STRATEGY_EAGER);
    }();
    return v;
}

uint32_t fixedPointEmulationSpecialValuesMask()
{
    static const uint32_t v = []() -> uint32_t {
        const auto parsed = fixedPointEmulationParseSpecialValuesMaskEnv(
            std::getenv("HIPBLASLT_EMULATION_SPECIAL_VALUES_SUPPORT_MASK"));
        return (parsed.state == FIXED_POINT_EMULATION_ENV_VALID) ? parsed.value : 0x3u;
    }();
    return v;
}

FixedPointEmulationEnvValue fixedPointEmulationParseEnabledEnv(const char* value)
{
    using namespace FixedPointEmulation;
    if(value == nullptr)
        return {FIXED_POINT_EMULATION_ENV_UNSET, 0u};
    if(std::strcmp(value, "1") == 0)
        return {FIXED_POINT_EMULATION_ENV_VALID, 1u};
    if(std::strcmp(value, "0") == 0)
        return {FIXED_POINT_EMULATION_ENV_VALID, 0u};
    return {FIXED_POINT_EMULATION_ENV_INVALID, 0u};
}

FixedPointEmulationEnvValue fixedPointEmulationParseStrategyEnv(const char* value)
{
    using namespace FixedPointEmulation;
    if(value == nullptr)
        return {FIXED_POINT_EMULATION_ENV_UNSET, 0u};
    if(std::strcmp(value, "performant") == 0)
        return {FIXED_POINT_EMULATION_ENV_VALID,
                static_cast<unsigned>(HIPBLASLT_EMULATION_STRATEGY_PERFORMANT)};
    if(std::strcmp(value, "eager") == 0)
        return {FIXED_POINT_EMULATION_ENV_VALID,
                static_cast<unsigned>(HIPBLASLT_EMULATION_STRATEGY_EAGER)};
    return {FIXED_POINT_EMULATION_ENV_INVALID, 0u};
}

FixedPointEmulationEnvValue fixedPointEmulationParseSpecialValuesMaskEnv(const char* value)
{
    using namespace FixedPointEmulation;
    if(value == nullptr)
        return {FIXED_POINT_EMULATION_ENV_UNSET, 0x3u};
    char*      endp = nullptr;
    const long v    = std::strtol(value, &endp, 0);
    if(endp == value || *endp != '\0' || v < 0)
        return {FIXED_POINT_EMULATION_ENV_INVALID, 0u};
    return {FIXED_POINT_EMULATION_ENV_VALID, static_cast<unsigned>(v)};
}

FixedPointEmulationEnvValue fixedPointEmulationParseNumModuliEnv(const char* value)
{
    using namespace FixedPointEmulation;
    if(value == nullptr)
        return {FIXED_POINT_EMULATION_ENV_UNSET, 0u};
    char*      endp = nullptr;
    const long n    = std::strtol(value, &endp, 10);
    if(endp == value || *endp != '\0' || n < 2 || n > static_cast<long>(S_MAX))
        return {FIXED_POINT_EMULATION_ENV_INVALID, 0u};
    return {FIXED_POINT_EMULATION_ENV_VALID, static_cast<unsigned>(n)};
}

FixedPointEmulationDecision fixedPointEmulationDecision(const _rocblaslt_handle*      h,
                                                        const _rocblaslt_matmul_desc* desc,
                                                        hipDataType                   type_a,
                                                        hipblasOperation_t            opA,
                                                        hipblasOperation_t            opB,
                                                        int64_t                       m,
                                                        int64_t                       n,
                                                        int64_t                       k,
                                                        int32_t                       batch_count,
                                                        size_t workspace_bytes)
{
    /* Setting Precedence (highest to lowest):
     *   1. Per-matmul descriptor fields (desc->emulation_*; sentinel = inherit).
     *   2. Environment variable — read once at first use, process-wide default.
     *   3. Built-in default (ADP mode, S_MAX moduli, special-values mask = 0x3).
     *
     * Descriptor fields are initialized to sentinel values (-1 / ~0u / 0).
     * A sentinel means "inherit from env var"; a non-sentinel means "use this
     * value", disabling the env-var fallback for that setting.               */
    using namespace FixedPointEmulation;
    FixedPointEmulationDecision result{};
    result.status  = rocblaslt_status_success;
    result.apply   = false;
    result.sv_mask = fixedPointEmulationSpecialValuesMask();

    /* Type and batch-count pre-checks (no env-var validation needed). */
    if((type_a != HIP_R_64F && type_a != HIP_R_32F) || batch_count != 1)
        return result; /* apply=false, success */

    /* Emulation enabled check.
     * Priority: desc->emulation_enabled (non-sentinel) > env var.
     *   desc_enabled == 1  -> force on
     *   desc_enabled == 0  -> force off (returns false)
     *   desc_enabled == -1 -> inherit from env var                          */
    const int  desc_enabled = desc ? desc->emulation_enabled : -1;
    const bool emulEnabled
        = (desc_enabled == 1) || (desc_enabled != 0 && fixedPointEmulationIsEnabled(type_a));
    if(!emulEnabled)
        return result;

    /* Device must be in the supported table. */
    const int dev = h->device;
    if(!get_perf_model_params(dev))
        return result;

    /* Resolve special_values_mask.
     * Priority: desc->emulation_sv_mask (non-~0u) > env var. */
    {
        const unsigned int desc_sv = desc ? desc->emulation_sv_mask : ~0u;
        if(desc_sv != ~0u)
            result.sv_mask = desc_sv;
    }

    /* ADP target precision.
     * Priority: desc->emulation_mantissa_bits (> 0) > type-specific env var. */
    {
        const int desc_bits = desc ? desc->emulation_mantissa_bits : 0;
        result.adp_mantissa_bits
            = (desc_bits > 0) ? desc_bits : fixedPointEmulationAdpMantissaBits(type_a);
    }

    /* Strategy (eager vs performant).
     * Priority: desc->emulation_strategy (non-sentinel) > env var. */
    {
        const int  desc_strat = desc ? desc->emulation_strategy : -1;
        const bool eager = (desc_strat == 2) || (desc_strat != 1 && fixedPointEmulationIsEager());
        if(eager
           || fixedPointEmulationPerformanceCheck(
               h, desc, type_a, opA, opB, m, n, k, workspace_bytes))
        {
            result.apply = true;
            /* Store the caller's workspace preference so fixedPointEmulationWorkspaceSize
         * can cap the returned size at min(optimal, workspace_cap).  This
         * ensures the library never reports a workspace larger than the user
         * allocated.                                       */
            result.workspace_cap = workspace_bytes;
        }
    } /* end strategy block */

    return result;
}

size_t fixedPointEmulationWorkspaceSize(const _rocblaslt_handle*           h,
                                        hipDataType                        type_a,
                                        hipblasOperation_t                 opA,
                                        hipblasOperation_t                 opB,
                                        int64_t                            m,
                                        int64_t                            n,
                                        int64_t                            k,
                                        const FixedPointEmulationDecision& decision)
{
    using namespace FixedPointEmulation;
    assert(h != nullptr && "fixedPointEmulationWorkspaceSize requires a valid handle");
    /* Always ADP mode: workspace covers max_num_moduli(type_a) moduli. */
    const unsigned num_moduli = max_num_moduli(type_a);
    /* Optimal workspace: chunk_size = split_num_moduli (single pass, no budget constraint).
     * n_chunks = 1 → split check never fires.  Always return monolithic workspace.  */

    /* Monolithic path workspace. */
    const size_t lda8i  = pad(static_cast<size_t>(k));
    const size_t cola8i = pad(static_cast<size_t>(m));
    const size_t ldb8i  = lda8i;
    const size_t ldc32i = cola8i;
    const size_t padn   = pad(static_cast<size_t>(n));
    const size_t szC32i = ldc32i * static_cast<size_t>(n);
    /* chunk_ws = num_moduli: optimal single-pass workspace (no budget constraint). */
    const unsigned chunk_ws = num_moduli;

    /* Zhi/Zlo accumulators are only needed when there are multiple passes
     * (chunk_ws < num_moduli).  Single-pass finalize writes D directly.  */
    const size_t szZhi_ws = (chunk_ws < num_moduli) ? szC32i : 0u;

    /* C32i production slots: chunk_ws slots for the batched INT8 GEMMs. */
    const size_t n_c32i_ws = static_cast<size_t>(chunk_ws);

    const size_t optimal
        = chunk_ws * lda8i * cola8i * sizeof(int8_t)
          + chunk_ws * ldb8i * static_cast<size_t>(n) * sizeof(int8_t)
          + n_c32i_ws * szC32i * sizeof(int32_t) + szZhi_ws * sizeof(double) * 2
          + cola8i * sizeof(int16_t) + padn * sizeof(int16_t) + sizeof(uint32_t)
          + cola8i * sizeof(int32_t) /* row_max[m] */
          + padn * sizeof(int32_t) /* col_max[n] — precomputed by partial kernel */
          + 2 * sizeof(float) /* ADP float buffer: adp_buf[0..1] (bias ±200) */
          + OZ2_INT8_GEMM_WS_BYTES; /* INT8 GEMM workspace (preliminary + batch)  */
    return std::min(optimal, decision.workspace_cap);
}

unsigned fixedPointEmulationNumModuli()
{
    /* Re-read on every call (no static cache) so that test code can use
     * setenv("HIPBLASLT_EMULATION_NUM_MODULI", "8", 1) between runs.
     * Returns 0 when the env var is absent or invalid (pure ADP mode).
     * Returns [2..S_MAX] when the env var specifies a post-ADP override. */
    const auto parsed
        = fixedPointEmulationParseNumModuliEnv(std::getenv("HIPBLASLT_EMULATION_NUM_MODULI"));
    return (parsed.state == FIXED_POINT_EMULATION_ENV_VALID) ? parsed.value : 0u;
}

FixedPointEmulationEnvValue fixedPointEmulationParseToleranceEnv(const char* value)
{
    /* Converts a positive tolerance string to a mantissa-bit count.
     * bits = clamp(floor(-log2(tol)), 1, 52).
     * Examples: "1e-16" → 53 (1e-16 < eps, slightly conservative)
     *           "1e-8"  → 26
     *           "1e-4"  → 13                                             */
    if(value == nullptr)
        return {FIXED_POINT_EMULATION_ENV_UNSET, 52u};
    char*        endp = nullptr;
    const double tol  = std::strtod(value, &endp);
    if(endp == value || *endp != '\0' || tol <= 0.0 || tol > 1.0)
        return {FIXED_POINT_EMULATION_ENV_INVALID, 0u};
    const int      bits    = static_cast<int>(std::floor(-std::log2(tol)));
    const unsigned clamped = static_cast<unsigned>(std::max(1, std::min(bits, 52)));
    return {FIXED_POINT_EMULATION_ENV_VALID, clamped};
}

/* =========================================================================
 * fp64EmulatedGemm — public wrapper
 *
 * Owns the profiling accumulator.  Records a single HIP event pair around
 * the entire call (including all recursive sub-GEMMs) to measure the true
 * GPU wall-clock time, then writes one summary CSV row with:
 *   – summed component times across all leaf sub-GEMMs
 *   – the measured t_total_ms for the full call
 *   – num_sub_gemms (number of monolithic leaf calls executed)
 * ========================================================================= */
rocblaslt_status fp64EmulatedGemm(hipblasLtHandle_t                  handle,
                                  hipblasOperation_t                 opA,
                                  hipblasOperation_t                 opB,
                                  int64_t                            m,
                                  int64_t                            n,
                                  int64_t                            k,
                                  const double*                      alpha,
                                  const double*                      A,
                                  int64_t                            lda,
                                  const double*                      B,
                                  int64_t                            ldb,
                                  const double*                      beta,
                                  const double*                      C,
                                  int64_t                            ldc,
                                  double*                            D,
                                  int64_t                            ldd,
                                  hipStream_t                        stream,
                                  const FixedPointEmulationSettings& settings)
{
    using namespace FixedPointEmulation;
    /* ── Change 4: Workspace Graceful Degradation ────────────────────────────
     * No internal hipMalloc.  The caller must provide workspace via
     * settings.workspace / settings.workspace_bytes.  If none is provided
     * or the budget makes emulation slower than native, fall back to native. */
    const _rocblaslt_handle* h = reinterpret_cast<const _rocblaslt_handle*>(handle);
    /* Always ADP mode: layout uses max_num_moduli<double>() moduli. */
    const unsigned num_moduli = max_num_moduli<double>();
    const int      device     = h->device;

    const FixedPointEmulationSettings& effectiveSettings = settings;

    {
        const size_t W = settings.workspace_bytes;
        const void*  P = settings.workspace;

        static std::atomic<int> s_ws_absent_warns{0};
        static std::atomic<int> s_ws_small_warns{0};

        /* Case 1: no workspace — warn + fall back immediately. */
        if(P == nullptr || W == 0)
        {
            if(s_ws_absent_warns.fetch_add(1, std::memory_order_relaxed) < 5)
                hipblaslt_cerr << "[hipBLASLt WARNING] FP64 emulation requires a workspace.\n"
                               << "  Call hipblasLtEmulationWorkspaceSize() for the optimal size.\n"
                               << "  Falling back to native DGEMM." << std::endl;
            return rocblaslt_status_memory_error;
        }

        /* Case 2: workspace provided — compute budget-constrained chunk_size. */
        const bool   tA_g = (opA != HIPBLAS_OP_N), tB_g = (opB != HIPBLAS_OP_N);
        const size_t cola8i_g = pad(static_cast<size_t>(m));
        const size_t padn_g   = pad(static_cast<size_t>(n));
        const size_t base_oh  = cola8i_g * sizeof(int16_t) /* sftA       */
                               + padn_g * sizeof(int16_t) /* sftB       */
                               + sizeof(uint32_t) /* nan_flag   */
                               + cola8i_g * sizeof(int32_t) /* row_max    */
                               + padn_g * sizeof(int32_t) /* col_max    */
                               + 2 * sizeof(float) /* adp_buf    */
                               + OZ2_INT8_GEMM_WS_BYTES; /* = 0        */

        const size_t   W_var1 = (W > base_oh) ? W - base_oh : 0u;
        const unsigned cs1    = compute_chunk_size(m, n, k, num_moduli, W_var1);

        size_t W_var;
        if(cs1 < num_moduli)
        {
            const size_t szC32i_g = cola8i_g * static_cast<size_t>(n);
            const size_t zlo_oh   = 2u * szC32i_g * sizeof(double);
            W_var                 = (W > base_oh + zlo_oh) ? W - base_oh - zlo_oh : 0u;
        }
        else
        {
            W_var = W_var1;
        }

        /* EAGER strategy means "always emulate, bypass the performance model".
         * Only apply the performance gate in PERFORMANT mode.                */
        const bool is_eager_g = settings.eager;

        /* ADP-aware performance model s. */
        const int      adp_bits_g = (settings.adp_mantissa_bits > 0)
                                        ? static_cast<int>(settings.adp_mantissa_bits)
                                        : fixedPointEmulationAdpMantissaBits(HIP_R_64F);
        const unsigned pm_s_g     = adp_expected_num_moduli(adp_bits_g, num_moduli);

        const double t_emul_W
            = is_eager_g ? 0.0
                         : effective_time_ms(tA_g, tB_g, m, n, k, pm_s_g, device, W, HIP_R_64F);
        const double t_native_g
            = is_eager_g
                  ? 1.0
                  : perf_model_times(tA_g, tB_g, m, n, k, pm_s_g, device, ~size_t{0}, HIP_R_64F)
                        .t_native_ms;
        if(!is_eager_g && t_emul_W > t_native_g)
        {
            const double t_emul_opt
                = effective_time_ms(tA_g, tB_g, m, n, k, pm_s_g, device, ~size_t{0}, HIP_R_64F);
            if(t_emul_opt <= t_native_g
               && s_ws_small_warns.fetch_add(1, std::memory_order_relaxed) < 5)
            {
                FixedPointEmulationDecision ws_dec{};
                ws_dec.workspace_cap = ~size_t{0};
                const size_t opt_ws
                    = fixedPointEmulationWorkspaceSize(h, HIP_R_64F, opA, opB, m, n, k, ws_dec);
                hipblaslt_cerr << "[hipBLASLt WARNING] FP64 emulation workspace (" << (W >> 20)
                               << " MiB) too small (m=" << m << ",n=" << n << ",k=" << k << ").\n"
                               << "  Optimal: " << (opt_ws >> 20) << " MiB."
                               << " Falling back to native DGEMM." << std::endl;
            }
            return rocblaslt_status_memory_error;
        }
    }
    /* ────────────────────────────────────────────────────────────────────── */

    const char* const _pf   = profile_file();
    const bool        _prof = (_pf != nullptr);

    ProfileAccum accum{};
    hipEvent_t   ev_start{}, ev_end{};
    if(_prof)
    {
        (void)hipEventCreate(&ev_start);
        (void)hipEventCreate(&ev_end);
        (void)hipEventRecord(ev_start, stream);
    }

    const rocblaslt_status st = emulated_gemm_impl<double>(h,
                                                           opA,
                                                           opB,
                                                           m,
                                                           n,
                                                           k,
                                                           alpha,
                                                           A,
                                                           lda,
                                                           B,
                                                           ldb,
                                                           beta,
                                                           C,
                                                           ldc,
                                                           D,
                                                           ldd,
                                                           stream,
                                                           effectiveSettings,
                                                           _prof ? &accum : nullptr);

    if(_prof)
    {
        (void)hipEventRecord(ev_end, stream);
        (void)hipStreamSynchronize(stream);
        float t_total = 0.f;
        (void)hipEventElapsedTime(&t_total, ev_start, ev_end);

        const unsigned       prof_s = accum.effective_s_used ? accum.effective_s_used : num_moduli;
        const unsigned       chunk_size       = compute_chunk_size(m, n, k, prof_s, ~size_t{0});
        const unsigned       scale_chunk_size = chunk_size;
        const bool           tA               = (opA != HIPBLAS_OP_N);
        const bool           tB               = (opB != HIPBLAS_OP_N);
        const int            adp_bits_prof_64 = (settings.adp_mantissa_bits > 0)
                                                    ? static_cast<int>(settings.adp_mantissa_bits)
                                                    : fixedPointEmulationAdpMantissaBits(HIP_R_64F);
        const PerfModelTimes pm
            = effective_perf_model_times(tA,
                                         tB,
                                         m,
                                         n,
                                         k,
                                         adp_expected_num_moduli(adp_bits_prof_64, num_moduli),
                                         device,
                                         ~size_t{0},
                                         HIP_R_64F);

        std::FILE* _f = std::fopen(_pf, "a");
        if(_f)
        {
            if(std::ftell(_f) == 0)
                std::fprintf(
                    _f,
                    "m,n,k,transA,transB,num_moduli,effective_s,scale_chunk_size,gemm_chunk_size,"
                    "workspace_bytes,num_sub_gemms,"
                    "t_prelim_ms,t_prelim_gemm_ms,t_refine_ms,"
                    "t_adp_ms,t_scale_ms,t_int8_gemm_ms,t_accum_ms,"
                    "t_total_ms,"
                    "cpu_ctx_create_ms,cpu_layout_ms,cpu_svmask_sync_ms,cpu_batch_attr_ms,"
                    "pred_prelim_ms,pred_prelim_gemm_ms,pred_refine_ms,pred_adp_ms,"
                    "pred_scale_ms,pred_int8_gemm_ms,pred_accum_ms,"
                    "pred_host_ms,pred_total_ms,pred_native_dgemm_ms\n");
            std::fprintf(_f,
                         "%lld,%lld,%lld,%c,%c,%u,%u,%u,%u,"
                         "%llu,%u,"
                         "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                         "%.4f,%.4f,%.4f,%.4f,"
                         "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                         (long long)m,
                         (long long)n,
                         (long long)k,
                         tA ? 'T' : 'N',
                         tB ? 'T' : 'N',
                         num_moduli,
                         accum.effective_s_used,
                         scale_chunk_size,
                         chunk_size,
                         (unsigned long long)settings.workspace_bytes,
                         accum.n_sub_gemms,
                         accum.t_prelim,
                         accum.t_prelim_gemm,
                         accum.t_refine,
                         accum.t_adp,
                         accum.t_scale,
                         accum.t_int8,
                         accum.t_accum,
                         t_total,
                         accum.t_ctx_create_ms,
                         accum.t_layout_ms,
                         accum.t_svmask_sync_ms,
                         accum.t_batch_attr_ms,
                         pm.t_prelim_ms,
                         pm.t_prelim_gemm_ms,
                         pm.t_refine_ms,
                         pm.t_adp_ms,
                         pm.t_scale_ms,
                         pm.t_int8_gemms_ms,
                         pm.t_accum_ms,
                         pm.t_host_ms,
                         pm.t_total_ms,
                         pm.t_native_ms);
            std::fclose(_f);
        }
        (void)hipEventDestroy(ev_end);
        (void)hipEventDestroy(ev_start);
    }
    return st;
}

/* Parser for HIPBLASLT_EMULATION_FP{32,64}_MANTISSA_BIT_COUNT.
 * Accepts decimal integers in [1..52]; values outside that range return INVALID. */
FixedPointEmulationEnvValue fixedPointEmulationParseMantissaBitCountEnv(const char* value)
{
    using namespace FixedPointEmulation;
    if(value == nullptr)
        return {FIXED_POINT_EMULATION_ENV_UNSET, 0u};
    char*      endp = nullptr;
    const long n    = std::strtol(value, &endp, 10);
    if(endp == value || *endp != '\0' || n < 1 || n > 52)
        return {FIXED_POINT_EMULATION_ENV_INVALID, 0u};
    return {FIXED_POINT_EMULATION_ENV_VALID, static_cast<unsigned>(n)};
}

/* Returns true when emulation is enabled for the given type.
 *   HIP_R_64F: reads HIPBLASLT_EMULATE_DOUBLE_PRECISION
 *   HIP_R_32F: reads HIPBLASLT_EMULATE_SINGLE_PRECISION
 * Cached separately per type on first call.                                 */
bool fixedPointEmulationIsEnabled(hipDataType type_a)
{
    if(type_a == HIP_R_64F)
    {
        static const bool v = []() -> bool {
            const auto parsed = fixedPointEmulationParseEnabledEnv(
                std::getenv("HIPBLASLT_EMULATE_DOUBLE_PRECISION"));
            return parsed.state == FIXED_POINT_EMULATION_ENV_VALID && parsed.value != 0u;
        }();
        return v;
    }
    if(type_a == HIP_R_32F)
    {
        static const bool v = []() -> bool {
            const auto parsed = fixedPointEmulationParseEnabledEnv(
                std::getenv("HIPBLASLT_EMULATE_SINGLE_PRECISION"));
            return parsed.state == FIXED_POINT_EMULATION_ENV_VALID && parsed.value != 0u;
        }();
        return v;
    }
    return false;
}

/* Returns the ADP target mantissa-bit count for the given type.
 * FP64: checks HIPBLASLT_EMULATION_FP64_MANTISSA_BIT_COUNT (integer bits),
 *       then HIPBLASLT_EMULATION_FP64_TOLERANCE (tolerance → bits),
 *       then defaults to 52 (full IEEE 754 FP64 precision).
 * FP32: checks HIPBLASLT_EMULATION_FP32_MANTISSA_BIT_COUNT (integer bits),
 *       then HIPBLASLT_EMULATION_FP32_TOLERANCE (tolerance → bits),
 *       then defaults to 23 (full IEEE 754 FP32 precision).               */
int fixedPointEmulationAdpMantissaBits(hipDataType type_a)
{
    if(type_a == HIP_R_64F)
    {
        static const int v = []() -> int {
            /* 1. Per-type bit-count env var. */
            const auto p = fixedPointEmulationParseMantissaBitCountEnv(
                std::getenv("HIPBLASLT_EMULATION_FP64_MANTISSA_BIT_COUNT"));
            if(p.state == FIXED_POINT_EMULATION_ENV_VALID)
                return static_cast<int>(p.value);
            /* 2. Per-type tolerance env var (converted to bit count). */
            const auto t = fixedPointEmulationParseToleranceEnv(
                std::getenv("HIPBLASLT_EMULATION_FP64_TOLERANCE"));
            if(t.state == FIXED_POINT_EMULATION_ENV_VALID)
                return static_cast<int>(t.value);
            /* 3. Default: full IEEE 754 FP64 precision = 52 mantissa bits. */
            return 52;
        }();
        return v;
    }
    if(type_a == HIP_R_32F)
    {
        static const int v = []() -> int {
            /* 1. Per-type bit-count env var. */
            const auto p = fixedPointEmulationParseMantissaBitCountEnv(
                std::getenv("HIPBLASLT_EMULATION_FP32_MANTISSA_BIT_COUNT"));
            if(p.state == FIXED_POINT_EMULATION_ENV_VALID)
                return static_cast<int>(p.value);
            /* 2. Per-type tolerance env var (converted to bit count). */
            const auto t = fixedPointEmulationParseToleranceEnv(
                std::getenv("HIPBLASLT_EMULATION_FP32_TOLERANCE"));
            if(t.state == FIXED_POINT_EMULATION_ENV_VALID)
                return static_cast<int>(t.value);
            /* 3. Default: full IEEE 754 FP32 precision = 23 mantissa bits. */
            return 23;
        }();
        return v;
    }
    hipblaslt_cerr
        << "[hipBLASLt] ERROR: fixedPointEmulationAdpMantissaBits called with unsupported type "
        << static_cast<int>(type_a) << std::endl;
    return 0;
}

/* Unified performance check — dispatches on type_a.
 * Uses FP32-specific ai_fp32/ratio_fp32 for HIP_R_32F (Step 7),
 * and FP64-specific ai/ratio for HIP_R_64F.                              */
bool fixedPointEmulationPerformanceCheck(const _rocblaslt_handle*      h,
                                         const _rocblaslt_matmul_desc* desc,
                                         hipDataType                   type_a,
                                         hipblasOperation_t            opA,
                                         hipblasOperation_t            opB,
                                         int64_t                       m,
                                         int64_t                       n,
                                         int64_t                       k,
                                         size_t                        workspace_bytes)
{
    using namespace FixedPointEmulation;
    const int  device = h->device;
    const bool tA     = (opA != HIPBLAS_OP_N);
    const bool tB     = (opB != HIPBLAS_OP_N);
    /* ADP mode: no fixed moduli set in desc or env var. */
    /* Always ADP mode. */
    const int      adp_bits_pc = fixedPointEmulationAdpMantissaBits(type_a);
    const unsigned pm_num      = adp_expected_num_moduli(adp_bits_pc, max_num_moduli(type_a));
    const double   t_emul
        = effective_time_ms(tA, tB, m, n, k, pm_num, device, workspace_bytes, type_a);
    const double t_native
        = perf_model_times(tA, tB, m, n, k, pm_num, device, ~size_t{0}, type_a).t_native_ms;
    return t_emul <= t_native;
}

/* Convenience wrapper: returns true when emulation would be applied for this
 * type and problem (enabled + device supported + strategy gate passes).      */
bool fixedPointEmulationWouldApply(const _rocblaslt_handle*      h,
                                   const _rocblaslt_matmul_desc* desc,
                                   hipDataType                   type_a,
                                   hipblasOperation_t            opA,
                                   hipblasOperation_t            opB,
                                   int64_t                       m,
                                   int64_t                       n,
                                   int64_t                       k,
                                   int32_t                       batch_count)
{
    return fixedPointEmulationDecision(h,
                                       desc,
                                       type_a,
                                       opA,
                                       opB,
                                       m,
                                       n,
                                       k,
                                       batch_count,
                                       /*workspace_bytes=*/~size_t{0u})
        .apply;
}

/* =========================================================================
 * fp32EmulatedGemm — FP32 public entry point (Step 6 — native float path)
 *
 * Mirrors fp64EmulatedGemm exactly.  All kernels are now templated on T=float
 * so no intermediate copies are needed.  alpha and beta are passed as
 * const float* directly; emulated_gemm_impl<float> reads them as float and
 * widens to double at arithmetic time (inside accum_finalize_kernel).
 * ========================================================================= */
rocblaslt_status fp32EmulatedGemm(hipblasLtHandle_t                  handle,
                                  hipblasOperation_t                 opA,
                                  hipblasOperation_t                 opB,
                                  int64_t                            m,
                                  int64_t                            n,
                                  int64_t                            k,
                                  const float*                       alpha,
                                  const float*                       A,
                                  int64_t                            lda,
                                  const float*                       B,
                                  int64_t                            ldb,
                                  const float*                       beta,
                                  const float*                       C,
                                  int64_t                            ldc,
                                  float*                             D,
                                  int64_t                            ldd,
                                  hipStream_t                        stream,
                                  const FixedPointEmulationSettings& settings)
{
    using namespace FixedPointEmulation;

    const _rocblaslt_handle* h = reinterpret_cast<const _rocblaslt_handle*>(handle);
    /* Always ADP mode: layout uses max_num_moduli<float>() moduli. */
    const unsigned num_moduli = max_num_moduli<float>();
    const int      device     = h->device;

    const FixedPointEmulationSettings& effectiveSettings = settings;

    {
        const size_t W = settings.workspace_bytes;
        const void*  P = settings.workspace;

        static std::atomic<int> s_ws_absent_warns{0};
        static std::atomic<int> s_ws_small_warns{0};

        if(P == nullptr || W == 0)
        {
            if(s_ws_absent_warns.fetch_add(1, std::memory_order_relaxed) < 5)
                hipblaslt_cerr << "[hipBLASLt WARNING] FP32 emulation requires a workspace.\n"
                               << "  Call hipblasLtEmulationWorkspaceSize() for the optimal size.\n"
                               << "  Falling back to native SGEMM." << std::endl;
            return rocblaslt_status_memory_error;
        }

        const bool   tA_g = (opA != HIPBLAS_OP_N), tB_g = (opB != HIPBLAS_OP_N);
        const size_t cola8i_g = pad(static_cast<size_t>(m));
        const size_t padn_g   = pad(static_cast<size_t>(n));
        const size_t base_oh  = cola8i_g * sizeof(int16_t) + padn_g * sizeof(int16_t)
                               + sizeof(uint32_t) + cola8i_g * sizeof(int32_t)
                               + padn_g * sizeof(int32_t) + 2 * sizeof(float)
                               + OZ2_INT8_GEMM_WS_BYTES;

        const size_t   W_var1 = (W > base_oh) ? W - base_oh : 0u;
        const unsigned cs1    = compute_chunk_size(m, n, k, num_moduli, W_var1);
        size_t         W_var;
        if(cs1 < num_moduli)
        {
            const size_t szC32i_g = cola8i_g * static_cast<size_t>(n);
            const size_t zlo_oh   = 2u * szC32i_g * sizeof(double);
            W_var                 = (W > base_oh + zlo_oh) ? W - base_oh - zlo_oh : 0u;
        }
        else
        {
            W_var = W_var1;
        }

        const bool is_eager_g = settings.eager;

        /* ADP-aware performance model s. */
        const int      adp_bits_g = (settings.adp_mantissa_bits > 0)
                                        ? static_cast<int>(settings.adp_mantissa_bits)
                                        : fixedPointEmulationAdpMantissaBits(HIP_R_32F);
        const unsigned pm_s_g     = adp_expected_num_moduli(adp_bits_g, num_moduli);

        /* FP32 performance model: compare against native SGEMM, not DGEMM. */
        const double t_emul_W
            = is_eager_g ? 0.0
                         : effective_time_ms(tA_g, tB_g, m, n, k, pm_s_g, device, W, HIP_R_32F);
        const double t_native_g
            = is_eager_g
                  ? 1.0
                  : perf_model_times(tA_g, tB_g, m, n, k, pm_s_g, device, ~size_t{0}, HIP_R_32F)
                        .t_native_ms;
        if(!is_eager_g && t_emul_W > t_native_g)
        {
            const double t_emul_opt
                = effective_time_ms(tA_g, tB_g, m, n, k, pm_s_g, device, ~size_t{0}, HIP_R_32F);
            if(t_emul_opt <= t_native_g
               && s_ws_small_warns.fetch_add(1, std::memory_order_relaxed) < 5)
            {
                FixedPointEmulationDecision ws_dec{};
                ws_dec.workspace_cap = ~size_t{0};
                const size_t opt_ws
                    = fixedPointEmulationWorkspaceSize(h, HIP_R_32F, opA, opB, m, n, k, ws_dec);
                hipblaslt_cerr << "[hipBLASLt WARNING] FP32 emulation workspace (" << (W >> 20)
                               << " MiB) too small (m=" << m << ",n=" << n << ",k=" << k << ").\n"
                               << "  Optimal: " << (opt_ws >> 20) << " MiB."
                               << " Falling back to native SGEMM." << std::endl;
            }
            return rocblaslt_status_memory_error;
        }
    }
    /* ────────────────────────────────────────────────────────────────────── */

    const char* const _pf   = profile_file();
    const bool        _prof = (_pf != nullptr);

    ProfileAccum accum{};
    hipEvent_t   ev_start{}, ev_end{};
    if(_prof)
    {
        (void)hipEventCreate(&ev_start);
        (void)hipEventCreate(&ev_end);
        (void)hipEventRecord(ev_start, stream);
    }

    /* Native FP32 path: alpha and beta are passed as const float*; emulated_gemm_impl<float>
     * dereferences them as float and widens to double inside accum_finalize_kernel.
     * No intermediate float→double copies needed.                             */
    const rocblaslt_status st = emulated_gemm_impl<float>(h,
                                                          opA,
                                                          opB,
                                                          m,
                                                          n,
                                                          k,
                                                          alpha,
                                                          A,
                                                          lda,
                                                          B,
                                                          ldb,
                                                          beta,
                                                          C,
                                                          ldc,
                                                          D,
                                                          ldd,
                                                          stream,
                                                          effectiveSettings,
                                                          _prof ? &accum : nullptr);

    if(_prof)
    {
        (void)hipEventRecord(ev_end, stream);
        (void)hipStreamSynchronize(stream);
        float t_total = 0.f;
        (void)hipEventElapsedTime(&t_total, ev_start, ev_end);

        const unsigned       prof_s = accum.effective_s_used ? accum.effective_s_used : num_moduli;
        const unsigned       chunk_size       = compute_chunk_size(m, n, k, prof_s, ~size_t{0});
        const unsigned       scale_chunk_size = chunk_size;
        const bool           tA               = (opA != HIPBLAS_OP_N);
        const bool           tB               = (opB != HIPBLAS_OP_N);
        const int            adp_bits_prof_32 = (settings.adp_mantissa_bits > 0)
                                                    ? static_cast<int>(settings.adp_mantissa_bits)
                                                    : fixedPointEmulationAdpMantissaBits(HIP_R_32F);
        const PerfModelTimes pm
            = effective_perf_model_times(tA,
                                         tB,
                                         m,
                                         n,
                                         k,
                                         adp_expected_num_moduli(adp_bits_prof_32, num_moduli),
                                         device,
                                         ~size_t{0},
                                         HIP_R_32F);

        std::FILE* _f = std::fopen(_pf, "a");
        if(_f)
        {
            if(std::ftell(_f) == 0)
                std::fprintf(
                    _f,
                    "m,n,k,transA,transB,num_moduli,effective_s,scale_chunk_size,gemm_chunk_size,"
                    "workspace_bytes,num_sub_gemms,"
                    "t_prelim_ms,t_prelim_gemm_ms,t_refine_ms,"
                    "t_adp_ms,t_scale_ms,t_int8_gemm_ms,t_accum_ms,"
                    "t_total_ms,"
                    "cpu_ctx_create_ms,cpu_layout_ms,cpu_svmask_sync_ms,cpu_batch_attr_ms,"
                    "pred_prelim_ms,pred_prelim_gemm_ms,pred_refine_ms,pred_adp_ms,"
                    "pred_scale_ms,pred_int8_gemm_ms,pred_accum_ms,"
                    "pred_host_ms,pred_total_ms,pred_native_sgemm_ms\n");
            std::fprintf(_f,
                         "%lld,%lld,%lld,%c,%c,%u,%u,%u,%u,"
                         "%llu,%u,"
                         "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                         "%.4f,%.4f,%.4f,%.4f,"
                         "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                         (long long)m,
                         (long long)n,
                         (long long)k,
                         tA ? 'T' : 'N',
                         tB ? 'T' : 'N',
                         num_moduli,
                         accum.effective_s_used,
                         scale_chunk_size,
                         chunk_size,
                         (unsigned long long)settings.workspace_bytes,
                         accum.n_sub_gemms,
                         accum.t_prelim,
                         accum.t_prelim_gemm,
                         accum.t_refine,
                         accum.t_adp,
                         accum.t_scale,
                         accum.t_int8,
                         accum.t_accum,
                         t_total,
                         accum.t_ctx_create_ms,
                         accum.t_layout_ms,
                         accum.t_svmask_sync_ms,
                         accum.t_batch_attr_ms,
                         pm.t_prelim_ms,
                         pm.t_prelim_gemm_ms,
                         pm.t_refine_ms,
                         pm.t_adp_ms,
                         pm.t_scale_ms,
                         pm.t_int8_gemms_ms,
                         pm.t_accum_ms,
                         pm.t_host_ms,
                         pm.t_total_ms,
                         pm.t_native_ms);
            std::fclose(_f);
        }
        (void)hipEventDestroy(ev_end);
        (void)hipEventDestroy(ev_start);
    }
    return st;
}
