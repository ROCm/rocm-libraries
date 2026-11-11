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
#include "utility.hpp" /* isFixedPointEmulatedComputeType & friends */

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
    static constexpr size_t OZ2_INT8_GEMM_WS_BYTES = 128ull << 20; /* 128 MiB */
    /* The INT8 GEMM workspace follows small scratch buffers in the emulation
     * workspace and is rounded up to OZ2_INT8_WS_ALIGN; reserve the slack.   */
    static constexpr size_t OZ2_INT8_WS_ALIGN   = 256u;
    static constexpr size_t OZ2_INT8_WS_RESERVE
        = OZ2_INT8_GEMM_WS_BYTES + (OZ2_INT8_GEMM_WS_BYTES > 0 ? OZ2_INT8_WS_ALIGN : 0u);

    /* Number of moduli processed per pass: the largest value ≤ s such that
     * the simultaneous allocation of A8i + B8i + C32i fits in the total budget.
     * Scale and GEMM share the same chunk — one scale launch followed by one
     * batched GEMM per pass.  The final pass handles the remainder naturally
     * via min(chunk, effective_s - chunk_start).                             */
    /* Workspace bytes needed per modulus in one pass: A8i slice (pad(k) x pad(m)),
     * B8i slice (pad(k) x n) and one INT32 output slot (pad(m) x n), matching the
     * layout in emulated_gemm_impl.                                           */
    static size_t bytes_per_modulus(int64_t m, int64_t n, int64_t k)
    {
        const size_t lda8i  = pad(static_cast<size_t>(k));
        const size_t cola8i = pad(static_cast<size_t>(m));
        const size_t nn     = static_cast<size_t>(n);
        return lda8i * cola8i + lda8i * nn + cola8i * nn * sizeof(int32_t);
    }

    /* True when at least one modulus fits in the budget.  When false, the
     * monolithic path cannot run without overflowing the workspace and the
     * problem must be split.                                                  */
    static bool chunk_fits(int64_t m, int64_t n, int64_t k, size_t budget)
    {
        return budget >= bytes_per_modulus(m, n, k);
    }

    static unsigned compute_chunk_size(int64_t m, int64_t n, int64_t k, unsigned s, size_t budget)
    {
        const size_t per = bytes_per_modulus(m, n, k);

        size_t chunk = s;
        if(per > 0u)
            chunk = std::min(chunk, budget / per);
        /* Clamped to 1 for the chunk-count arithmetic; callers must check
         * chunk_fits() before running the monolithic path.                  */
        return static_cast<unsigned>(std::max(size_t(1u), chunk));
    }

    /* Workspace needed independently of the moduli count: sftA/sftB, nan_flag,
     * row/col max, the ADP buffer, and the INT8 GEMM reserve.                 */
    static size_t fixed_overhead_bytes(int64_t m, int64_t n)
    {
        const size_t co = pad(static_cast<size_t>(m));
        const size_t pn = pad(static_cast<size_t>(n));
        return co * sizeof(int16_t) + pn * sizeof(int16_t) + sizeof(uint32_t)
               + co * sizeof(int32_t) + pn * sizeof(int32_t) + 2 * sizeof(float)
               + OZ2_INT8_WS_RESERVE;
    }

    /* Compute the effective INT8 workspace budget for the monolithic path given
     * workspace_bytes, sub-problem dimensions (m, n), and the moduli count s.
     * Mirrors the two-step budget derivation in emulated_gemm_impl so that
     * effective_time accurately models the available budget at each recursive
     * split level.                                                              */
    static size_t
        compute_budget_from_ws(int64_t m, int64_t n, int64_t k, unsigned s, size_t workspace_bytes)
    {
        const size_t   co  = pad(static_cast<size_t>(m));
        const size_t   boh = fixed_overhead_bytes(m, n);
        const size_t   W1  = (workspace_bytes > boh) ? workspace_bytes - boh : 0u;
        const unsigned cs1 = compute_chunk_size(m, n, k, s, W1);
        if(cs1 < s)
        {
            const size_t zoh = 2u * co * static_cast<size_t>(n) * sizeof(double);
            return (workspace_bytes > boh + zoh) ? workspace_bytes - boh - zoh : 0u;
        }
        return W1;
    }

    /* Smallest workspace that emulates (m, n, k) without a spatial split, i.e.
     * the smallest W with chunk_fits(m, n, k, compute_budget_from_ws(..., W)).
     * Either all s modulus slices fit in one pass (no Zhi/Zlo accumulators
     * needed), or one slice per pass plus the Zhi/Zlo accumulators; for s >= 6
     * the latter is always smaller.                                           */
    static size_t min_workspace_bytes(int64_t m, int64_t n, int64_t k, unsigned s)
    {
        const size_t per = bytes_per_modulus(m, n, k);
        const size_t zoh
            = 2u * pad(static_cast<size_t>(m)) * static_cast<size_t>(n) * sizeof(double);
        return fixed_overhead_bytes(m, n) + std::min(static_cast<size_t>(s) * per, zoh + per);
    }

    struct PerfModelKernelEffs
    {
        static constexpr double eff_prelim[2][2] = {{0.697, 0.561}, /* [N][N], [N][T] */
                                                    {0.902, 0.682}}; /* [T][N], [T][T] */
        static constexpr double eff_scale[2][2]  = {{0.595, 0.520}, /* [N][N], [N][T] */
                                                    {0.683, 0.583}}; /* [T][N], [T][T] */
        static constexpr double eff_refine       = 0.586;
        static constexpr double eff_accum        = 0.938;
    };

    struct PerfModelDeviceParams
    {
        double ai_fp64;
        double ai_fp32;
        double ai_int8;
        double ovh_kernel;
        double ovh_matmul;
        double ovh_memset;
        double ovh_sync;
        double ovh_host;
    };

    /* Device support tables.  Performance-model values are relative (only
     * ratios matter).  Field order: ai_fp64, ai_fp32, ai_int8, ovh_kernel,
     * ovh_matmul, ovh_memset, ovh_sync, ovh_host.
     * See fp_emulation/README.md, "Adding support for a new device".
     *
     * Chip rows, keyed by PCI device ID, hold performance-model values
     * measured on that card; the performance model decides for that card.
     * Architecture rows, keyed by gfx target, hold arithmetic-intensity
     * thresholds for cards of that target without a chip row.  A device with
     * neither row is not supported.                                          */
    static const std::unordered_map<uint32_t, PerfModelDeviceParams>& chip_rows()
    {
        static const std::unordered_map<uint32_t, PerfModelDeviceParams> rows = {
            {0x75a0u, {11.22, 22.04, 521.7, 3.0e7, 6.1e7, 1.2e7, 3.0e8, 6.1e8}}, // gfx950-mi350x
            {0x75b0u, {11.22, 22.04, 521.7, 3.0e7, 6.1e7, 1.2e7, 3.0e8, 6.1e8}}, // gfx950-mi350x
            {0x75a3u, {11.33, 22.58, 484.0, 3.4e7, 6.8e7, 1.4e7, 3.4e8, 6.8e8}}, // gfx950-mi355x
            {0x75b3u, {11.33, 22.58, 484.0, 3.4e7, 6.8e7, 1.4e7, 3.4e8, 6.8e8}}, // gfx950-mi355x
            {0x745eu, {1.591, 15.62, 159.7, 1.3e6, 3.3e6, 1.4e6, 6.6e6, 2.7e7}}, // gfx1100-w7800
            {0x7550u, {1.36, 39.86, 332.0, 1.4e6, 4.2e6, 1.8e6, 1.6e7, 5.6e7}}, // gfx1201-rx9070xt
        };
        return rows;
    }

    struct ArchRow
    {
        const char* arch; /* gfx target */
        /* Unless the strategy is eager, emulate exactly when the arithmetic
         * intensity (flop/byte) is at least this value.  0 = always,
         * FIXED_POINT_EMULATION_NEVER = never.                               */
        double fp64_min_ai;
        double fp32_min_ai;
    };

    /* Residue arithmetic.  The scale and accumulation kernels compute their
     * residues either with FP64 operations or with integer and FP32 ones (see
     * kernel_common.hpp).  The FP64 versions need fewer instructions and are
     * faster on GPUs with a fast FP64 rate; on GPUs with a low FP64 rate they
     * make those kernels FP64-bound.  A device uses the FP64 versions when its
     * native FP64 rate is at least this fraction of its INT8 rate
     * (ai_fp64 / ai_int8): MI350X and MI355X are above it, the measured Radeon
     * cards below.  Architecture rows (ai_fp64 = 0) always use integer.      */
    static constexpr double FP64_RESIDUE_MIN_FP64_INT8_RATIO = 0.015;

    /* HIPBLASLT_EMULATION_INT_RESIDUES=0|1 forces the FP64 or the integer
     * residues on every device, for testing; unset or invalid keeps the
     * per-device choice.  Returns -1 for no override.                          */
    static int int_residues_override()
    {
        static const int v = []() -> int {
            const char* e = std::getenv("HIPBLASLT_EMULATION_INT_RESIDUES");
            if(e && std::strcmp(e, "0") == 0)
                return 0;
            if(e && std::strcmp(e, "1") == 0)
                return 1;
            return -1;
        }();
        return v;
    }

    /* Generic performance-model values for cards without a chip row.  They
     * do not decide whether to emulate (the thresholds do); they only plan
     * how an emulated GEMM is split when the workspace is too small, and fill
     * the predictions in the profiling CSV.                                 */
    static constexpr PerfModelDeviceParams generic_arch_params
        = {0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    static const ArchRow* find_arch_row(const char* gcn_arch_name)
    {
        /* Thresholds are arithmetic intensities, not sizes: emulation costs
         * work proportional to m * n per modulus, so a GEMM with a small k
         * stays faster natively however large m * n * k is.  A square FP64
         * GEMM of size n has an intensity of n / 12.
         * FP64: the smallest measured intensity from which emulated FP64 is
         * faster than native, for square GEMMs (TN and NN) and for
         * 8192 x 8192 x k:
         *   RDNA 3:   on the Radeon Pro W7800, emulation is slower at 512^3
         *             (43) and faster from 576^3 (48) on; at 8192 x 8192 x k
         *             it is slower up to k = 128 (31) and faster at k = 256.
         *   RDNA 3.5: copied from RDNA 3; not yet measured on RDNA 3.5.
         *   RDNA 4:   on the Radeon RX 9070 XT, emulation is faster from
         *             384^3 (32) on; at 8192 x 8192 x k it is slower up to
         *             k = 64 (16) and faster from k = 128 (31) on.
         * FP32: never; emulated FP32 is slower than native FP32 in the TN and
         * NN layouts on both cards.                                          */
        constexpr double NEVER = FIXED_POINT_EMULATION_NEVER;
        static const ArchRow rows[] = {
            {"gfx1100", 48.0, NEVER}, // RDNA 3
            {"gfx1101", 48.0, NEVER}, // RDNA 3
            {"gfx1102", 48.0, NEVER}, // RDNA 3
            {"gfx1150", 48.0, NEVER}, // RDNA 3.5
            {"gfx1151", 48.0, NEVER}, // RDNA 3.5
            {"gfx1152", 48.0, NEVER}, // RDNA 3.5
            {"gfx1200", 30.0, NEVER}, // RDNA 4
            {"gfx1201", 30.0, NEVER}, // RDNA 4
        };
        if(gcn_arch_name == nullptr)
            return nullptr;
        /* gcnArchName may carry target features, e.g. "gfx1100:sramecc-:xnack-". */
        const size_t len = std::strcspn(gcn_arch_name, ":");
        for(const auto& row : rows)
            if(std::strlen(row.arch) == len && std::strncmp(row.arch, gcn_arch_name, len) == 0)
                return &row;
        return nullptr;
    }

    struct DeviceEmulationInfo
    {
        std::optional<PerfModelDeviceParams> params; /* nullopt: not supported */
        FixedPointEmulationDeviceSupport     support;
    };

    static DeviceEmulationInfo lookup_device(uint32_t pci_device_id, const char* gcn_arch_name)
    {
        DeviceEmulationInfo info{std::nullopt,
                                 {FIXED_POINT_EMULATION_DEVICE_NONE,
                                  FIXED_POINT_EMULATION_NEVER,
                                  FIXED_POINT_EMULATION_NEVER,
                                  true}};
        const auto& chips = chip_rows();
        const auto  it    = chips.find(pci_device_id & 0xFFFFu);
        if(it != chips.end())
        {
            info.params        = it->second;
            info.support.match = FIXED_POINT_EMULATION_DEVICE_CHIP;
        }
        else if(const ArchRow* arch = find_arch_row(gcn_arch_name))
        {
            info.params               = generic_arch_params;
            info.support.match        = FIXED_POINT_EMULATION_DEVICE_ARCH;
            info.support.fp64_min_ai  = arch->fp64_min_ai;
            info.support.fp32_min_ai  = arch->fp32_min_ai;
        }
        if(info.params)
            info.support.int_residues
                = !(info.params->ai_int8 > 0.0
                    && info.params->ai_fp64 / info.params->ai_int8
                           >= FP64_RESIDUE_MIN_FP64_INT8_RATIO);
        return info;
    }

    /* Device lookup, cached per device; nullptr for an out-of-range index. */
    static const DeviceEmulationInfo* device_info(int device)
    {
        static std::optional<DeviceEmulationInfo> cache[64];
        if(device < 0 || device >= 64)
            return nullptr;
        auto& entry = cache[device];
        if(!entry)
        {
            int chip_id = 0;
            if(hipDeviceGetAttribute(&chip_id, hipDeviceAttributePciChipId, device) != hipSuccess)
                chip_id = 0;
            hipDeviceProp_t prop{};
            const char*     arch = (hipGetDeviceProperties(&prop, device) == hipSuccess)
                                       ? prop.gcnArchName
                                       : nullptr;
            entry                = lookup_device(static_cast<uint32_t>(chip_id), arch);
        }
        return &*entry;
    }

    /* Returns the perf-model parameters for the given HIP device, or nullopt if
     * the device is not supported (in which case emulation should not run).   */
    static std::optional<PerfModelDeviceParams> get_perf_model_params(int device)
    {
        const DeviceEmulationInfo* info = device_info(device);
        return info ? info->params : std::nullopt;
    }

    /* Returns the minimum number of moduli s ∈ [2, S_MAX] such that
     * crt_capacity_bits(s-2) >= adp_bits, used ONLY in the PERFORMANCE MODEL to predict
     * which s ADP will select at runtime.  Falls back to S_MAX when
     * adp_bits <= 0 (sentinel) or when all moduli are needed to reach the
     * target.                                                                */
    static unsigned adp_expected_num_moduli(int adp_bits, unsigned max_s) noexcept
    {
        if(adp_bits <= 0)
            return max_s;
        for(unsigned s = 2u; s <= max_s; ++s)
            if(crt_capacity_bits(s - 2u) >= static_cast<float>(adp_bits))
                return s;
        return max_s;
    }

    /* =========================================================================
     * Performance-model predicted costs
     * Costs are in device-relative units; only ratios between them are
     * meaningful.  Used both for the profiling CSV (reported relative to the
     * native GEMM cost) and (via comparison of t_total vs t_native) for the
     * performance heuristic in fixedPointEmulationPerformanceCheck.
     * ========================================================================= */
    struct PerfModelTimes
    {
        double t_prelim; /* prelim kernel (shift + extraction)  */
        double t_prelim_gemm; /* preliminary INT8 GEMM               */
        double t_refine; /* sft-refinement kernels              */
        double t_adp; /* ADP reduce kernels + hipStreamSynchronize */
        double t_scale; /* multi-modulus scaling kernels       */
        double t_int8_gemms; /* all INT8 GEMMs                      */
        double t_accum; /* CRT accumulation / finalize kernels */
        double t_host; /* per-call host overhead              */
        double t_total; /* total predicted emulation cost      */
        double t_native; /* predicted native DGEMM/SGEMM cost   */
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
        const double c0             = 1.0;
        const double c1             = is_fp32 ? hw.ai_fp32 : hw.ai_fp64;
        const double c2             = hw.ai_int8;
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

        /* One INT8 GEMM (k x m)^T (k x n) -> INT32 (m x n).  The INT8 operand
         * reads overlap with the MFMA work, but the INT32 output write (4 B per
         * element) does not: it is charged additively at HBM bandwidth.  With
         * max(compute, reads + writes) the model underestimated INT8 GEMMs by
         * up to ~2x for large m, n and small k (where the output write
         * dominates), letting the gate pick emulation where it is slower.     */
        const double t_int8_one = std::max(2.0 * mnk / c2, (mk + kn) / c0) + 4.0 * mn / c0;
        const double t_prelim_kern
            = ((mk + kn) * 17.0 / c0 + 2.0 * hw.ovh_kernel) / K::eff_prelim[tA][tB];
        const double t_prelim_gemm = t_int8_one + hw.ovh_matmul;
        const double t_refine_kern
            = (mn * 8.0 / c0 + 3.0 * hw.ovh_kernel + hw.ovh_memset) / K::eff_refine;
        const double t_scale_kern = ((mk + kn) * (8.0 * n_scale_chunks + s) / c0
                                     + 2.0 * n_scale_chunks * hw.ovh_kernel)
                                    / K::eff_scale[tA][tB];
        const double t_int8_gemms = s * t_int8_one + n_chunks * hw.ovh_matmul;
        const double t_accum_kern
            = (mn * (4.0 * s + 32.0 * n_chunks - 16.0) / c0 + n_chunks * hw.ovh_kernel)
              / K::eff_accum;
        const double t_host       = hw.ovh_host;
        const double t_gemm_accum = t_int8_gemms + t_accum_kern;
        const double t_adp
            = 2.0 * hw.ovh_kernel /* adp_reduce_A_kernel + adp_reduce_B_kernel */
              + hw.ovh_sync; /* hipStreamSynchronize — dominant cost */
        const double t_total = t_prelim_kern + t_prelim_gemm + t_refine_kern + t_scale_kern
                               + t_gemm_accum + t_host + t_adp;
        /* A native rate of 0 (generic architecture values) marks native GEMM of
         * this type as never competitive: its predicted cost is infinite.     */
        const double t_native
            = (c1 > 0.0)
                  ? std::max(2.0 * mnk / c1, bytes_per_elem * (mk + kn + mn) / c0) + hw.ovh_matmul
                  : std::numeric_limits<double>::infinity();

        return {t_prelim_kern,
                t_prelim_gemm,
                t_refine_kern,
                t_adp,
                t_scale_kern,
                t_int8_gemms,
                t_accum_kern,
                t_host,
                t_total,
                t_native};
    }

    /* Minimum problem size for efficient INT8 tensor core execution.
     * Below this threshold MFMA tiles are under-utilised and native DGEMM wins.
     * Both effective_time and emulated_gemm_impl fall back immediately.         */
    static constexpr int64_t FIXED_POINT_EMUL_MIN_MN = 16;

    /* Returns the minimum achievable predicted emulation cost, accounting for the
     * recursive binary-halving that fp64EmulatedGemm applies when n_chunks > 1.
     * Both halves execute sequentially so the effective cost is additive.
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
    static double effective_time(bool        tA,
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
                = perf_model_times(tA, tB, m, n, k, s, device, ~size_t{0}, type_a).t_native;
            return t_nat * (1.0 + 1e-6);
        }
        const size_t budget = compute_budget_from_ws(m, n, k, s, workspace_bytes);
        const bool   fits   = chunk_fits(m, n, k, budget);

        const unsigned chunk_sz = compute_chunk_size(m, n, k, s, budget);
        const unsigned n_chunks = (s + chunk_sz - 1u) / chunk_sz;
        if(n_chunks > 1u || !fits)
        {
            const bool    split_m = (m >= n);
            const int64_t half_m  = split_m ? m / 2 : m;
            const int64_t half_n  = split_m ? n : n / 2;

            const double t_split
                = 2.
                  * effective_time(
                      tA, tB, half_m, half_n, k, s, device, workspace_bytes, type_a);
            /* If not even one modulus fits, the split is mandatory. */
            if(!fits)
                return t_split;
            const double t_mono
                = perf_model_times(tA, tB, m, n, k, s, device, budget, type_a).t_total;
            if(t_split < t_mono)
                return t_split;
            return t_mono;
        }
        return perf_model_times(tA, tB, m, n, k, s, device, budget, type_a).t_total;
    }

    /* Returns per-component predicted costs summed across ALL leaf sub-GEMMs,
     * mirroring the recursive binary-halving of effective_time.
     * t_native is always set to the top-level (m,n,k) native GEMM cost
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
        const bool     fits   = chunk_fits(m, n, k, budget);

        const unsigned chunk_sz = compute_chunk_size(m, n, k, s, budget);
        const unsigned n_chunks = (s + chunk_sz - 1u) / chunk_sz;
        if((n_chunks > 1u || !fits) && m >= FIXED_POINT_EMUL_MIN_MN
           && n >= FIXED_POINT_EMUL_MIN_MN)
        {
            const bool    split_m = (m >= n);
            const int64_t half_m  = split_m ? m / 2 : m;
            const int64_t half_n  = split_m ? n : n / 2;

            const double t_split
                = 2.
                  * effective_time(
                      tA, tB, half_m, half_n, k, s, device, workspace_bytes, type_a);
            if(!fits || t_split < mono.t_total)
            {
                PerfModelTimes half = effective_perf_model_times(
                    tA, tB, half_m, half_n, k, s, device, workspace_bytes, type_a);
                half.t_prelim *= 2.0;
                half.t_prelim_gemm *= 2.0;
                half.t_refine *= 2.0;
                half.t_adp *= 2.0;
                half.t_scale *= 2.0;
                half.t_int8_gemms *= 2.0;
                half.t_accum *= 2.0;
                half.t_host *= 2.0;
                half.t_total *= 2.0;
                /* Native DGEMM does not split: keep the top-level prediction. */
                half.t_native = mono.t_native;
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
        /* Explicitly disable emulation via force_off flag to prevent infinite recursion:
         * native_gemm_fallback → hipblasLtMatmul → emulation → native_gemm_fallback.
         * The stack-local emulation descriptor outlives the (synchronous) hipblasLtMatmul call. */
        hipblasLtEmulationDescriptor emul_off_desc;
        emul_off_desc.force_off = true;
        struct hipblasLtEmulationDescriptor* emul_off = &emul_off_desc;
        hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR,
                                        &emul_off, sizeof(emul_off));
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
        unsigned chunk_size_used  = 0u; /* moduli per INT8 GEMM pass, smallest over sub-GEMMs */
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
    template <unsigned TC, bool INT_RES, typename T>
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
        const unsigned k_x
            = static_cast<unsigned>((k + OZ2_SCALE_SHMEM_TILE_K - 1) / OZ2_SCALE_SHMEM_TILE_K);
        if(tA)
            hipLaunchKernelGGL((scale_A_T_kernel<TC, T, INT_RES>),
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
            hipLaunchKernelGGL((scale_A_N_kernel<TC, T, INT_RES>),
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

    template <unsigned TC, bool INT_RES, typename T>
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
        const unsigned k_x
            = static_cast<unsigned>((k + OZ2_SCALE_SHMEM_TILE_K - 1) / OZ2_SCALE_SHMEM_TILE_K);
        if(!tB)
            hipLaunchKernelGGL((scale_B_N_kernel<TC, T, INT_RES>),
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
            hipLaunchKernelGGL((scale_B_T_kernel<TC, T, INT_RES>),
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
    template <unsigned CS, bool INT_RES, typename T_out = double>
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
                    hipLaunchKernelGGL((accum_finalize_kernel<true, CS, true, INT_RES, T_out>),
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
                    hipLaunchKernelGGL((accum_finalize_kernel<true, CS, false, INT_RES, T_out>),
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
                    hipLaunchKernelGGL((accum_finalize_kernel<false, CS, true, INT_RES, T_out>),
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
                    hipLaunchKernelGGL((accum_finalize_kernel<false, CS, false, INT_RES, T_out>),
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
                    hipLaunchKernelGGL((chunk_accum_kernel<true, CS, true, INT_RES>),
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
                    hipLaunchKernelGGL((chunk_accum_kernel<true, CS, false, INT_RES>),
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
                    hipLaunchKernelGGL((chunk_accum_kernel<false, CS, true, INT_RES>),
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
                    hipLaunchKernelGGL((chunk_accum_kernel<false, CS, false, INT_RES>),
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
                                               const FixedPointEmulationSettings& decision,
                                               ProfileAccum*                      prof)
    {
        static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>,
                      "emulated_gemm_impl<T>: T must be double (FP64) or float (FP32)");
        /* Type-specific label strings for warning messages. */
        constexpr const char* type_name  = std::is_same_v<T, double> ? "FP64" : "FP32";
        constexpr const char* native_str = std::is_same_v<T, double> ? "DGEMM" : "SGEMM";

        /* Minimum problem size guard — see FIXED_POINT_EMUL_MIN_MN.  Must come
         * before the split below (as in effective_time): splitting only shrinks
         * m and n, and without this guard a sub-problem that never fits, e.g.
         * (m, n) = (0, 1), recurses into an identical copy of itself.
         * k == 0 is an empty product (D = beta * C): the scale kernels would
         * be launched with grid.x = 0 (hipErrorInvalidConfiguration).         */
        if(m < FIXED_POINT_EMUL_MIN_MN || n < FIXED_POINT_EMUL_MIN_MN || k == 0)
            return native_gemm_fallback<T>(
                h, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd, stream);

        /* Always ADP mode: workspace layout always covers max_num_moduli<T>() moduli.
         * fixedPointEmulationNumModuli() post-ADP override is applied later. */
        const unsigned num_moduli = max_num_moduli<T>();
        /* Workspace budget: use max_num_moduli<T>() because ADP selects effective_s
         * at runtime — the workspace must accommodate up to that many moduli. */
        const size_t ws_budget
            = compute_budget_from_ws(m, n, k, num_moduli, decision.workspace_bytes);

        {
            const unsigned chunk_sz = compute_chunk_size(m, n, k, num_moduli, ws_budget);
            const unsigned n_chunks = (num_moduli + chunk_sz - 1u) / chunk_sz;

            /* If not even one modulus (A8i + B8i slices + one INT32 slot) fits in
             * the budget left after the fixed buffers and the Zhi/Zlo accumulators,
             * the monolithic path would write beyond the workspace buffer.  Force
             * a binary spatial split on the larger of m or n until it fits.
             * This branch is also entered when n_chunks > 1 (multiple passes),
             * where the split is optional and decided by the performance model. */
            const bool fits = chunk_fits(m, n, k, ws_budget);
            if(n_chunks > 1u || !fits)
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

                /* When one modulus does not fit, the monolithic path is not an
                 * option: skip the performance model and split immediately.      */
                const bool force_split = !fits;
                /* ADP-aware performance model s: use the minimum s that satisfies the
                 * precision target instead of S_MAX, to avoid overestimating emulation
                 * cost in dynamic mode.                                               */
                const int      adp_bits_impl = (decision.adp_mantissa_bits > 0)
                                                   ? static_cast<int>(decision.adp_mantissa_bits)
                                                   : fixedPointEmulationAdpMantissaBits(type_a_perf);
                const unsigned pm_s_impl     = adp_expected_num_moduli(adp_bits_impl, num_moduli);
                const double   t_mono
                    = force_split ? 0.0
                                  : perf_model_times(
                                        tA, tB, m, n, k, pm_s_impl, device, ws_budget, type_a_perf)
                                        .t_total;
                /* Pass workspace_bytes so the sub-call recomputes its own budget
                 * from the smaller (half_m, half_n) dimensions.                  */
                const double t_split = force_split
                                           ? 0.0
                                           : 2.
                                                 * effective_time(tA,
                                                                     tB,
                                                                     half_m,
                                                                     half_n,
                                                                     k,
                                                                     pm_s_impl,
                                                                     device,
                                                                     decision.workspace_bytes,
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
                                                                    decision,
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
                                                                       decision,
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

        /* Residue arithmetic for this device (see int_residues). */
        const bool int_res = fixedPointEmulationDeviceSupport(h->device).int_residues;

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

        char* const ws = static_cast<char*>(decision.workspace);

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
        /* Round up to OZ2_INT8_WS_ALIGN: GEMM kernels that use workspace
         * (e.g. split-K / Stream-K partials) expect an aligned buffer.  The
         * slack is included in OZ2_INT8_WS_RESERVE.                       */
        void* const int8_ws = reinterpret_cast<void*>(
            (reinterpret_cast<uintptr_t>(adp_buf + 2) + OZ2_INT8_WS_ALIGN - 1u)
            & ~static_cast<uintptr_t>(OZ2_INT8_WS_ALIGN - 1u));
        constexpr size_t int8_ws_size = OZ2_INT8_GEMM_WS_BYTES;
        int32_t* const   C32i         = C32i_batch;

        int8_t* const A8i_high = A8i;
        int8_t* const B8i_high = B8i;

        const bool tA = (opA != HIPBLAS_OP_N);
        const bool tB = (opB != HIPBLAS_OP_N);

        /* sv_support is a bitmask (bit 0 = Inf, bit 1 = NaN); the descriptor
         * accepts any 16-bit mask, so keep only the bits the kernels understand. */
        const uint32_t svmask = decision.sv_support & 0x3u;

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
         * shift-refinement delta is applied so we can use the correct crt_capacity_bits. */
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
         * The refine delta is then applied using crt_capacity_bits(effective_s-2) so that
         * X_true is sized to fit within M_{effective_s}/2.                        */
        {
            _pstart();
            (void)hipMemsetAsync(adp_buf, 0, 2 * sizeof(float), stream);
            /* Resolve ADP target precision: settings sentinel 0 → env var default (52). */
            const float adp_bits
                = static_cast<float>((decision.adp_mantissa_bits > 0)
                                         ? decision.adp_mantissa_bits
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

            const float crt_capacity_bits_needed = std::max(h_adp[0], h_adp[1]) - 200.0f;

            if(crt_capacity_bits_needed > crt_capacity_bits(num_moduli - 2u))
            {
                /* ADP determined the type's max moduli are still insufficient.
                 * Before giving up, check whether HIPBLASLT_EMULATION_NUM_MODULI
                 * provides a forced override — if so, use it and proceed.
                 * This allows test/debug code to force a specific s even when the
                 * ADP analysis says no standard s is sufficient (accuracy is not
                 * guaranteed, but the computation will run rather than returning
                 * invalid_value).                                                 */
                const unsigned override_s_adp = fixedPointEmulationNumModuli(
                    std::is_same_v<T, double> ? HIP_R_64F : HIP_R_32F);
                if(override_s_adp >= 2u && override_s_adp <= num_moduli)
                {
                    effective_s = override_s_adp;
                }
                else
                {
                    /* No valid override: fall back to native GEMM.
                     * This occurs for matrices with extremely large dynamic range within
                     * a single row/column (e.g. condition number ≫ 2^{2×crt_capacity_bits(18)}).
                     * The Ozaki shift-refinement would set A8i_final or B8i_final to
                     * near-zero for the elements providing cancellation, giving wrong
                     * results regardless of s.  Rate-limited warning (≤5 per process). */
                    hipblaslt_cerr << "[hipBLASLt " << type_name
                                   << " emulation] WARNING: ADP overflow for GEMM " << "(m=" << m
                                   << ", n=" << n << ", k=" << k
                                   << "): " << "A-side crt_capacity_bits_req=" << (h_adp[0] - 200.0f)
                                   << " bits, " << "B-side=" << (h_adp[1] - 200.0f) << " bits, "
                                   << "max required=" << crt_capacity_bits_needed
                                   << " > supported max=" << crt_capacity_bits(num_moduli - 2u)
                                   << " (s=" << num_moduli << " moduli). Falling back to native "
                                   << native_str << "." << std::endl;
                    return rocblaslt_status_invalid_value;
                }
            }

            for(unsigned s = 2u; s <= num_moduli; ++s)
            {
                if(crt_capacity_bits(s - 2u) >= crt_capacity_bits_needed)
                {
                    effective_s = s;
                    break;
                }
            }
        }
        /* Post-ADP testing override: HIPBLASLT_EMULATION_NUM_MODULI overrides the
         * ADP-selected s.  Only for testing; accuracy is not guaranteed below ADP value. */
        {
            const unsigned override_s = fixedPointEmulationNumModuli(
                std::is_same_v<T, double> ? HIP_R_64F : HIP_R_32F);
            if(override_s >= 2u && override_s <= num_moduli)
                effective_s = override_s;
        }

        /* Apply shift-refinement delta using the correct crt_capacity_bits for effective_s.
         * This ensures X_true ≤ M_{effective_s}/4 < M_{effective_s}/2 (CRT safe). */
        const float refine_crt_capacity_bits = crt_capacity_bits(effective_s - 2u);

        _pstart();
        hipLaunchKernelGGL(refine_sftA_apply_kernel,
                           dim3(sftA_m_blks),
                           dim3(64),
                           0,
                           stream,
                           row_max,
                           sftA,
                           m,
                           refine_crt_capacity_bits);
        hipLaunchKernelGGL(refine_sftB_kernel,
                           dim3((static_cast<unsigned>(n) + 255u) / 256u),
                           dim3(256),
                           0,
                           stream,
                           col_max,
                           n,
                           sftB,
                           refine_crt_capacity_bits);
        _pstop(_t_refine); /* partial: refine_sftA_apply + refine_sftB */

        /* Scale → INT8 GEMM → CRT accum/finalize.
         * Flat single loop: each pass scales `actual` moduli into A8i[0..actual-1]
         * then runs one batched GEMM of batch_count=actual.                       */
        int32_t       batch_cur  = 0; /* set on first iteration */
        const int64_t stride_A_b = static_cast<int64_t>(lda8i * cola8i);
        const int64_t stride_B_b = static_cast<int64_t>(ldb8i * static_cast<size_t>(n));
        const int64_t stride_C_b = static_cast<int64_t>(szC32i);
        /* TensileLite passes every tensor stride, including the batch stride,
         * to its kernels as a 32-bit argument, so hipBLASLt reports no
         * solution for a strided-batched GEMM whose A, B or C/D batch stride
         * reaches 2^32 elements (a batch_count=1 GEMM of that size is fine).
         * In that case run the moduli of each chunk as separate
         * batch_count=1 GEMMs.  The C/D stride (pad(m) * n INT32 elements)
         * reaches the limit at about m = n = 65536.                        */
        constexpr int64_t OZ2_INT8_MAX_BATCH_STRIDE = int64_t(1) << 32;
        const bool        use_batched = stride_A_b < OZ2_INT8_MAX_BATCH_STRIDE
                                 && stride_B_b < OZ2_INT8_MAX_BATCH_STRIDE
                                 && stride_C_b < OZ2_INT8_MAX_BATCH_STRIDE;
        if(use_batched)
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
                if(int_res)
                {
                    launch_scale_A<TC, true>(
                        tA, A, m, lda, A8i, lda8i, cola8i, sftA, k, chunk_start, stream);
                    launch_scale_B<TC, true>(
                        tB, B, n, ldb, B8i, ldb8i, sftB, k, chunk_start, stream);
                }
                else
                {
                    launch_scale_A<TC, false>(
                        tA, A, m, lda, A8i, lda8i, cola8i, sftA, k, chunk_start, stream);
                    launch_scale_B<TC, false>(
                        tB, B, n, ldb, B8i, ldb8i, sftB, k, chunk_start, stream);
                }
            });
            _pstop(_t_scale);

            /* Update batch_count when it changes (normally constant; may differ on
             * the final pass when effective_s is not a multiple of chunk_size).    */
            if(use_batched && static_cast<int32_t>(actual) != batch_cur)
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
                hipblasStatus_t batch_st = HIPBLAS_STATUS_SUCCESS;
                const unsigned  n_calls  = use_batched ? 1u : actual;
                for(unsigned t = 0; t < n_calls && batch_st == HIPBLAS_STATUS_SUCCESS; ++t)
                {
                    int32_t* const c_t = C32i_batch + t * stride_C_b;
                    batch_st           = hipblasLtMatmul(context.int8_handle,
                                               context.matmulDesc,
                                               &one_i,
                                               A8i + t * stride_A_b,
                                               context.layoutA,
                                               B8i + t * stride_B_b,
                                               context.layoutB,
                                               &zero_i,
                                               c_t,
                                               context.layoutCD,
                                               c_t,
                                               context.layoutCD,
                                               nullptr,
                                               int8_ws,
                                               int8_ws_size,
                                               stream);
                }
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
            /* qpi_lo is non-zero from 6 moduli on (see qpi() in tables.hpp). */
            const bool has_lo   = (effective_s > 5u);
            _pstart();
            dispatch_by_index<max_num_moduli<T>()>(actual, [&](auto Count) {
                constexpr unsigned CS = Count.value;
                if(int_res)
                    dispatch_accum_chunk<CS, true>(is_first,
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
                else
                    dispatch_accum_chunk<CS, false>(is_first,
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
            const unsigned used_chunk = std::min(chunk_size, effective_s);
            prof->chunk_size_used     = prof->chunk_size_used
                                            ? std::min(prof->chunk_size_used, used_chunk)
                                            : used_chunk;
        }
        return rocblaslt_status_success;
    }

} // namespace FixedPointEmulation

bool fixedPointEmulationIsEager()
{
    static const bool v = []() -> bool {
        const auto parsed
            = fixedPointEmulationParseStrategyEnv(std::getenv("HIPBLAS_EMULATION_STRATEGY"));
        return parsed.state == FIXED_POINT_EMULATION_ENV_VALID
               && parsed.value == static_cast<unsigned>(rocblaslt_emulation_strategy_eager);
    }();
    return v;
}

FixedPointEmulationDeviceSupport fixedPointEmulationLookupDevice(uint32_t    pci_device_id,
                                                                 const char* gcn_arch_name)
{
    return FixedPointEmulation::lookup_device(pci_device_id, gcn_arch_name).support;
}

FixedPointEmulationDeviceSupport fixedPointEmulationDeviceSupport(int device)
{
    const auto* info = FixedPointEmulation::device_info(device);
    if(info)
    {
        FixedPointEmulationDeviceSupport sup = info->support;
        const int                        ov  = FixedPointEmulation::int_residues_override();
        if(ov >= 0)
            sup.int_residues = (ov == 1);
        return sup;
    }
    return info ? info->support
                : FixedPointEmulationDeviceSupport{FIXED_POINT_EMULATION_DEVICE_NONE,
                                                   FIXED_POINT_EMULATION_NEVER,
                                                   FIXED_POINT_EMULATION_NEVER,
                                                   true};
}

FixedPointEmulationPolicy fixedPointEmulationPolicy(const FixedPointEmulationDeviceSupport& support,
                                                    bool                                    eager,
                                                    hipDataType                             type_a,
                                                    int64_t                                 m,
                                                    int64_t                                 n,
                                                    int64_t                                 k)
{
    if(support.match == FIXED_POINT_EMULATION_DEVICE_NONE
       || (type_a != HIP_R_64F && type_a != HIP_R_32F))
        return FIXED_POINT_EMULATION_POLICY_NATIVE;
    if(eager)
        return FIXED_POINT_EMULATION_POLICY_EMULATE;
    if(support.match == FIXED_POINT_EMULATION_DEVICE_CHIP)
        return FIXED_POINT_EMULATION_POLICY_MODEL;
    const bool   fp64   = (type_a == HIP_R_64F);
    const double min_ai = fp64 ? support.fp64_min_ai : support.fp32_min_ai;
    const double md     = static_cast<double>(m);
    const double nd     = static_cast<double>(n);
    const double kd     = static_cast<double>(k);
    /* Arithmetic intensity: flops per byte of A, B and C. */
    const double bytes = (fp64 ? 8.0 : 4.0) * (md * kd + kd * nd + md * nd);
    const double ai    = (bytes > 0.0) ? 2.0 * md * nd * kd / bytes : 0.0;
    return (ai >= min_ai) ? FIXED_POINT_EMULATION_POLICY_EMULATE
                          : FIXED_POINT_EMULATION_POLICY_NATIVE;
}

uint32_t fixedPointEmulationSpecialValuesMask()
{
    static const uint32_t v = []() -> uint32_t {
        const auto parsed = fixedPointEmulationParseSpecialValuesMaskEnv(
            std::getenv("HIPBLAS_EMULATION_SPECIAL_VALUES_SUPPORT_MASK"));
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
                static_cast<unsigned>(rocblaslt_emulation_strategy_performant)};
    if(std::strcmp(value, "eager") == 0)
        return {FIXED_POINT_EMULATION_ENV_VALID,
                static_cast<unsigned>(rocblaslt_emulation_strategy_eager)};
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

FixedPointEmulationEnvValue fixedPointEmulationParseNumModuliEnv(const char* value, unsigned max_s)
{
    using namespace FixedPointEmulation;
    if(value == nullptr)
        return {FIXED_POINT_EMULATION_ENV_UNSET, 0u};
    /* Never accept more moduli than the tables provide. */
    const long limit = static_cast<long>(std::min(max_s, S_MAX));
    char*      endp  = nullptr;
    const long n     = std::strtol(value, &endp, 10);
    if(endp == value || *endp != '\0' || n < 2 || n > limit)
        return {FIXED_POINT_EMULATION_ENV_INVALID, 0u};
    return {FIXED_POINT_EMULATION_ENV_VALID, static_cast<unsigned>(n)};
}

FixedPointEmulationSettings fixedPointEmulationDecision(const _rocblaslt_handle*       h,
                                                        const _rocblaslt_matmul_desc*  desc,
                                                        rocblaslt_compute_type         compute_type,
                                                        hipDataType                    a_type,
                                                        hipDataType                    b_type,
                                                        hipDataType                    c_type,
                                                        hipDataType                    d_type,
                                                        int64_t                        m,
                                                        int64_t                        n,
                                                        int64_t                        k,
                                                        int32_t                        batch_count,
                                                        size_t workspace_bytes)
{
    /* Data type being emulated: HIP_R_64F / HIP_R_32F for the fixed-point
     * emulated compute types and for plain f64 / f32 (env-var path);
     * HIPBLASLT_DATATYPE_INVALID for every other compute type.            */
    const hipDataType type_a = fixedPointEmulatedComputeDataType(compute_type);
    /* Derive transpose flags from the descriptor (or default to OP_N when nullptr). */
    const hipblasOperation_t opA = desc ? desc->op_A : HIPBLAS_OP_N;
    const hipblasOperation_t opB = desc ? desc->op_B : HIPBLAS_OP_N;
    /* Setting Precedence (highest to lowest):
     *   1. Per-matmul descriptor fields (desc->emulation_*; sentinel = inherit).
     *   2. Environment variable — read once at first use, process-wide default.
     *   3. Built-in default (ADP mode, S_MAX moduli, special-values mask = 0x3).
     *
     * Descriptor fields are initialized to sentinel values (-1 / ~0u / 0).
     * A sentinel means "inherit from env var"; a non-sentinel means "use this
     * value", disabling the env-var fallback for that setting.               */
    using namespace FixedPointEmulation;
    FixedPointEmulationSettings result{};
    result.status    = rocblaslt_status_success;
    result.apply     = false;
    result.workspace = nullptr;
    /* Resolve sv_support from env var (default: both Inf+NaN). */
    {
        /* The env-var mask uses the same bit layout as the descriptor attribute
         * (bit 0 = Inf, bit 1 = NaN), matching cudaEmulationSpecialValuesSupport. */
        const uint32_t env_mask = fixedPointEmulationSpecialValuesMask();
        result.sv_support       = env_mask & 0x3u;
    }

    /* Type, batch-count and size pre-checks (no env-var validation needed).
     * All four matrices must be of the emulated real type: the plain 64F/32F
     * compute types are also used for complex GEMMs and for FP16/BF16 with
     * 32F compute, which the emulation kernels would read and write as
     * FP64/FP32 (wrong results, and for FP16/BF16 writes past the end of D).
     * An empty product (m, n or k == 0; with k == 0, D = beta * C) never
     * needs emulation, so it is left to the native path and reports no
     * emulation workspace.                                                 */
    if((type_a != HIP_R_64F && type_a != HIP_R_32F) || a_type != type_a || b_type != type_a
       || c_type != type_a || d_type != type_a || batch_count != 1 || m == 0 || n == 0
       || k == 0)
        return result; /* apply=false, success */

    const hipblasLtEmulationDescriptor* ed = (desc && desc->emulation_desc) ? desc->emulation_desc : nullptr;

    /* force_off = internal flag set by native_gemm_fallback to prevent recursive re-entry. */
    if(ed && ed->force_off)
        return result; /* apply=false */

    /* Emulation enabled:
     *   - compute type is {64F,32F}_EMULATED_FIXEDPOINT -> explicitly requested;
     *   - plain 64F / 32F compute type -> only via the env var
     *     HIPBLAS_EMULATE_{DOUBLE,SINGLE}_PRECISION=1 (transparent mode).
     * The emulation descriptor only tunes settings; it never enables. */
    const bool emulEnabled = isFixedPointEmulatedComputeType(compute_type)
                             || fixedPointEmulationIsEnabled(type_a);
    if(!emulEnabled)
        return result;

    /* Device must have a chip row or an architecture row. */
    const int dev = h->device;
    if(!get_perf_model_params(dev))
        return result;

    /* Special-values mask.
     * Priority: desc->emulation_sv_support (non-DEFAULT) > env var > 0x3.
     * DEFAULT (0xFFFF, like cudaEmulationSpecialValuesSupport's DEFAULT)
     * keeps the env-var value resolved above, matching how DEFAULT/0 inherit
     * the env var for the strategy and the precision target.               */
    if(ed && ed->emulation_sv_support != HIPBLASLT_EMULATION_SV_SUPPORT_DEFAULT)
        result.sv_support = ed->emulation_sv_support;

    /* ADP target precision.
     * Priority: desc->emulation_fixedpoint_max_mantissa_bits (> 0) > type-specific env var.
     * The descriptor accepts 0..52 because its data type is unknown when the
     * attribute is set; for FP32 the value is clamped here to 23 bits.      */
    {
        int desc_bits = ed ? ed->emulation_fixedpoint_max_mantissa_bits : 0;
        if(type_a == HIP_R_32F && desc_bits > FP32_MAX_MANTISSA_BITS)
        {
            static std::atomic<bool> s_warned{false};
            if(!s_warned.exchange(true))
                hipblaslt_cerr << "[hipBLASLt FP32 emulation] WARNING: "
                                  "HIPBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT="
                               << desc_bits << " exceeds the FP32 maximum of "
                               << FP32_MAX_MANTISSA_BITS << "; using "
                               << FP32_MAX_MANTISSA_BITS << "." << std::endl;
            desc_bits = FP32_MAX_MANTISSA_BITS;
        }
        result.adp_mantissa_bits
            = (desc_bits > 0) ? desc_bits : fixedPointEmulationAdpMantissaBits(type_a);
    }

    /* Strategy (eager vs performant).
     * Priority: desc->emulation_strategy (non-DEFAULT) > env var.
     * force_off is already handled above; only DEFAULT/PERFORMANT/EAGER reach here.
     * Eager always emulates.  Otherwise a device with a chip row asks the
     * performance model, and a device with only an architecture row emulates
     * exactly the problems at or above the row's intensity threshold.        */
    {
        const rocblaslt_emulation_strategy desc_strat
            = ed ? ed->emulation_strategy : rocblaslt_emulation_strategy_default;
        const bool eager
            = desc_strat == rocblaslt_emulation_strategy_eager
              || (desc_strat == rocblaslt_emulation_strategy_default && fixedPointEmulationIsEager());
        result.workspace_bytes = workspace_bytes;
        const FixedPointEmulationPolicy policy = fixedPointEmulationPolicy(
            fixedPointEmulationDeviceSupport(dev), eager, type_a, m, n, k);
        /* A threshold decision is final: like eager, it skips the
         * performance-model workspace check in fp{32,64}EmulatedGemm.        */
        result.strategy = (policy == FIXED_POINT_EMULATION_POLICY_MODEL)
                              ? rocblaslt_emulation_strategy_performant
                              : rocblaslt_emulation_strategy_eager;
        if(policy == FIXED_POINT_EMULATION_POLICY_EMULATE
           || (policy == FIXED_POINT_EMULATION_POLICY_MODEL
               && fixedPointEmulationPerformanceCheck(
                   h, desc, type_a, opA, opB, m, n, k, workspace_bytes)))
        {
            result.apply = true;
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
                                        const FixedPointEmulationSettings& decision)
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
          + OZ2_INT8_WS_RESERVE; /* INT8 GEMM workspace (preliminary + batch)  */
    return std::min(optimal, decision.workspace_bytes);
}

unsigned fixedPointEmulationNumModuli(hipDataType type_a)
{
    using namespace FixedPointEmulation;
    /* Re-read on every call (no static cache) so that test code can use
     * setenv("HIPBLASLT_EMULATION_NUM_MODULI", "8", 1) between runs.
     * Returns 0 when the env var is absent or invalid (pure ADP mode).
     * Returns [2..max_num_moduli(type_a)] when the env var specifies a
     * post-ADP override: [2..20] for FP64, [2..12] for FP32.             */
    const unsigned type_max = max_num_moduli(type_a);
    const unsigned max_s    = type_max ? type_max : S_MAX_FP64;
    const char*    env      = std::getenv("HIPBLASLT_EMULATION_NUM_MODULI");
    const auto     parsed   = fixedPointEmulationParseNumModuliEnv(env, max_s);
    if(parsed.state == FIXED_POINT_EMULATION_ENV_INVALID)
    {
        /* Warn once per type; the value is ignored and ADP selects s. */
        static std::atomic<bool> s_warned_fp32{false};
        static std::atomic<bool> s_warned_fp64{false};
        std::atomic<bool>&       warned = (type_a == HIP_R_32F) ? s_warned_fp32 : s_warned_fp64;
        if(!warned.exchange(true))
            hipblaslt_cerr << "[hipBLASLt " << ((type_a == HIP_R_32F) ? "FP32" : "FP64")
                           << " emulation] WARNING: HIPBLASLT_EMULATION_NUM_MODULI=" << env
                           << " is outside the supported range [2.." << max_s
                           << "]; ignoring it." << std::endl;
    }
    return (parsed.state == FIXED_POINT_EMULATION_ENV_VALID) ? parsed.value : 0u;
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
                                  const FixedPointEmulationSettings& decision)
{
    using namespace FixedPointEmulation;
    /* ── Change 4: Workspace Graceful Degradation ────────────────────────────
     * No internal hipMalloc.  The caller must provide workspace via
     * decision.workspace / decision.workspace_bytes.  If none is provided
     * or the budget makes emulation slower than native, fall back to native. */
    const _rocblaslt_handle* h = reinterpret_cast<const _rocblaslt_handle*>(handle);
    /* Always ADP mode: layout uses max_num_moduli<double>() moduli. */
    const unsigned num_moduli = max_num_moduli<double>();
    const int      device     = h->device;

    /* Empty product: D = beta * C.  Nothing to emulate (and the scale
     * kernels can't be launched with a zero-sized grid).                   */
    if(k == 0)
        return native_gemm_fallback<double>(
            h, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd, stream);

    {
        const size_t W = decision.workspace_bytes;
        const void*  P = decision.workspace;

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

        /* Case 2: workspace provided.  Hard capacity check, for every strategy
         * (EAGER bypasses the performance model below, not this check): if not
         * even a FIXED_POINT_EMUL_MIN_MN^2 tile fits, splitting cannot help.  */
        const bool   tA_g  = (opA != HIPBLAS_OP_N), tB_g = (opB != HIPBLAS_OP_N);
        const size_t W_min = min_workspace_bytes(
            FIXED_POINT_EMUL_MIN_MN, FIXED_POINT_EMUL_MIN_MN, k, num_moduli);
        if(W < W_min)
        {
            static std::atomic<int> s_ws_min_warns{0};
            if(s_ws_min_warns.fetch_add(1, std::memory_order_relaxed) < 5)
            {
                FixedPointEmulationSettings ws_dec{};
                ws_dec.workspace_bytes = ~size_t{0};
                const size_t opt_ws
                    = fixedPointEmulationWorkspaceSize(h, HIP_R_64F, opA, opB, m, n, k, ws_dec);
                const size_t nosplit_ws = min_workspace_bytes(m, n, k, num_moduli);
                constexpr size_t MiB    = size_t(1) << 20;
                hipblaslt_cerr
                    << "[hipBLASLt WARNING] FP64 emulation workspace (" << (W / MiB)
                    << " MiB) is below the minimum of " << ((W_min + MiB - 1) / MiB)
                    << " MiB, which includes a " << (OZ2_INT8_GEMM_WS_BYTES / MiB)
                    << " MiB INT8 GEMM reserve (m=" << m << ",n=" << n << ",k=" << k << ").\n"
                    << "  Without spatial splitting: " << ((nosplit_ws + MiB - 1) / MiB)
                    << " MiB. Optimal: " << (opt_ws / MiB) << " MiB.\n"
                    << "  Falling back to native DGEMM." << std::endl;
            }
            return rocblaslt_status_memory_error;
        }

        /* EAGER strategy means "always emulate, bypass the performance model".
         * Only apply the performance gate in PERFORMANT mode.                */
        const bool is_eager_g = (decision.strategy == rocblaslt_emulation_strategy_eager);

        /* ADP-aware performance model s. */
        const int      adp_bits_g = (decision.adp_mantissa_bits > 0)
                                        ? static_cast<int>(decision.adp_mantissa_bits)
                                        : fixedPointEmulationAdpMantissaBits(HIP_R_64F);
        const unsigned pm_s_g     = adp_expected_num_moduli(adp_bits_g, num_moduli);

        const double t_emul_W
            = is_eager_g ? 0.0
                         : effective_time(tA_g, tB_g, m, n, k, pm_s_g, device, W, HIP_R_64F);
        const double t_native_g
            = is_eager_g
                  ? 1.0
                  : perf_model_times(tA_g, tB_g, m, n, k, pm_s_g, device, ~size_t{0}, HIP_R_64F)
                        .t_native;
        if(!is_eager_g && t_emul_W > t_native_g)
        {
            const double t_emul_opt
                = effective_time(tA_g, tB_g, m, n, k, pm_s_g, device, ~size_t{0}, HIP_R_64F);
            if(t_emul_opt <= t_native_g
               && s_ws_small_warns.fetch_add(1, std::memory_order_relaxed) < 5)
            {
                FixedPointEmulationSettings ws_dec{};
                ws_dec.workspace_bytes = ~size_t{0};
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
                                                           decision,
                                                           _prof ? &accum : nullptr);

    if(_prof)
    {
        (void)hipEventRecord(ev_end, stream);
        (void)hipStreamSynchronize(stream);
        float t_total = 0.f;
        (void)hipEventElapsedTime(&t_total, ev_start, ev_end);

        const unsigned       prof_s = accum.effective_s_used ? accum.effective_s_used : num_moduli;
        /* Report what ran with the caller's workspace, not an unlimited one. */
        const unsigned       chunk_size       = accum.chunk_size_used ? accum.chunk_size_used : prof_s;
        const unsigned       scale_chunk_size = chunk_size;
        const bool           tA               = (opA != HIPBLAS_OP_N);
        const bool           tB               = (opB != HIPBLAS_OP_N);
        const int            adp_bits_prof_64 = (decision.adp_mantissa_bits > 0)
                                                    ? static_cast<int>(decision.adp_mantissa_bits)
                                                    : fixedPointEmulationAdpMantissaBits(HIP_R_64F);
        const PerfModelTimes pm
            = effective_perf_model_times(tA,
                                         tB,
                                         m,
                                         n,
                                         k,
                                         adp_expected_num_moduli(adp_bits_prof_64, num_moduli),
                                         device,
                                         decision.workspace_bytes,
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
                    "pred_prelim_rel,pred_prelim_gemm_rel,pred_refine_rel,pred_adp_rel,"
                    "pred_scale_rel,pred_int8_gemm_rel,pred_accum_rel,"
                    "pred_host_rel,pred_total_rel\n");
            /* pred_*_rel: predicted cost relative to the predicted native DGEMM cost
             * (pred_total_rel < 1 means emulation is predicted to be faster). */
            std::fprintf(_f,
                         "%lld,%lld,%lld,%c,%c,%u,%u,%u,%u,"
                         "%llu,%u,"
                         "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                         "%.4f,%.4f,%.4f,%.4f,"
                         "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                         (long long)m,
                         (long long)n,
                         (long long)k,
                         tA ? 'T' : 'N',
                         tB ? 'T' : 'N',
                         num_moduli,
                         accum.effective_s_used,
                         scale_chunk_size,
                         chunk_size,
                         (unsigned long long)decision.workspace_bytes,
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
                         pm.t_prelim / pm.t_native,
                         pm.t_prelim_gemm / pm.t_native,
                         pm.t_refine / pm.t_native,
                         pm.t_adp / pm.t_native,
                         pm.t_scale / pm.t_native,
                         pm.t_int8_gemms / pm.t_native,
                         pm.t_accum / pm.t_native,
                         pm.t_host / pm.t_native,
                         pm.t_total / pm.t_native);
            std::fclose(_f);
        }
        (void)hipEventDestroy(ev_end);
        (void)hipEventDestroy(ev_start);
    }
    return st;
}

/* Parser for HIPBLAS_FIXEDPOINT_EMULATION_FP{32,64}_MANTISSA_BIT_COUNT.
 * Accepts decimal integers in [1..max_bits] (52 for FP64, 23 for FP32);
 * values outside that range return INVALID.                               */
FixedPointEmulationEnvValue fixedPointEmulationParseMantissaBitCountEnv(const char* value,
                                                                        unsigned    max_bits)
{
    using namespace FixedPointEmulation;
    if(value == nullptr)
        return {FIXED_POINT_EMULATION_ENV_UNSET, 0u};
    char*      endp = nullptr;
    const long n    = std::strtol(value, &endp, 10);
    if(endp == value || *endp != '\0' || n < 1 || n > static_cast<long>(max_bits))
        return {FIXED_POINT_EMULATION_ENV_INVALID, 0u};
    return {FIXED_POINT_EMULATION_ENV_VALID, static_cast<unsigned>(n)};
}

/* Returns true when emulation is enabled for the given type.
 *   HIP_R_64F: reads HIPBLAS_EMULATE_DOUBLE_PRECISION
 *   HIP_R_32F: reads HIPBLAS_EMULATE_SINGLE_PRECISION
 * Cached separately per type on first call.                                 */
bool fixedPointEmulationIsEnabled(hipDataType type_a)
{
    if(type_a == HIP_R_64F)
    {
        static const bool v = []() -> bool {
            const auto parsed = fixedPointEmulationParseEnabledEnv(
                std::getenv("HIPBLAS_EMULATE_DOUBLE_PRECISION"));
            return parsed.state == FIXED_POINT_EMULATION_ENV_VALID && parsed.value != 0u;
        }();
        return v;
    }
    if(type_a == HIP_R_32F)
    {
        static const bool v = []() -> bool {
            const auto parsed = fixedPointEmulationParseEnabledEnv(
                std::getenv("HIPBLAS_EMULATE_SINGLE_PRECISION"));
            return parsed.state == FIXED_POINT_EMULATION_ENV_VALID && parsed.value != 0u;
        }();
        return v;
    }
    return false;
}

/* Returns the ADP target mantissa-bit count for the given type.
 * FP64: checks HIPBLAS_FIXEDPOINT_EMULATION_FP64_MANTISSA_BIT_COUNT, then
 *       HIPBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT (same name as CUDA's
 *       CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT), then defaults to 52
 *       (full IEEE 754 FP64 precision).
 * FP32: checks HIPBLAS_FIXEDPOINT_EMULATION_FP32_MANTISSA_BIT_COUNT (integer bits),
 *       then defaults to 23 (full IEEE 754 FP32 precision).               */
int fixedPointEmulationAdpMantissaBits(hipDataType type_a)
{
    if(type_a == HIP_R_64F)
    {
        static const int v = []() -> int {
            /* 1. FP64-specific bit-count env var. */
            const auto p = fixedPointEmulationParseMantissaBitCountEnv(
                std::getenv("HIPBLAS_FIXEDPOINT_EMULATION_FP64_MANTISSA_BIT_COUNT"),
                FP64_MAX_MANTISSA_BITS);
            if(p.state == FIXED_POINT_EMULATION_ENV_VALID)
                return static_cast<int>(p.value);
            /* 2. Generic bit-count env var (CUDA-compatible name; FP64 only, like
             *    cuBLAS, where fixed-point emulation is only used for FP64). */
            const auto g = fixedPointEmulationParseMantissaBitCountEnv(
                std::getenv("HIPBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT"),
                FP64_MAX_MANTISSA_BITS);
            if(g.state == FIXED_POINT_EMULATION_ENV_VALID)
                return static_cast<int>(g.value);
            /* 3. Default: full IEEE 754 FP64 precision = 52 mantissa bits. */
            return 52;
        }();
        return v;
    }
    if(type_a == HIP_R_32F)
    {
        static const int v = []() -> int {
            /* 1. Per-type bit-count env var, limited to [1..23]. */
            const char* env = std::getenv("HIPBLAS_FIXEDPOINT_EMULATION_FP32_MANTISSA_BIT_COUNT");
            const auto  p = fixedPointEmulationParseMantissaBitCountEnv(env, FP32_MAX_MANTISSA_BITS);
            if(p.state == FIXED_POINT_EMULATION_ENV_VALID)
                return static_cast<int>(p.value);
            if(p.state == FIXED_POINT_EMULATION_ENV_INVALID)
                hipblaslt_cerr << "[hipBLASLt FP32 emulation] WARNING: "
                                  "HIPBLAS_FIXEDPOINT_EMULATION_FP32_MANTISSA_BIT_COUNT="
                               << env << " is outside the supported range [1.."
                               << FP32_MAX_MANTISSA_BITS << "]; using the default of "
                               << FP32_MAX_MANTISSA_BITS << "." << std::endl;
            /* 2. Default: full IEEE 754 FP32 precision = 23 mantissa bits. */
            return FP32_MAX_MANTISSA_BITS;
        }();
        return v;
    }
    hipblaslt_cerr
        << "[hipBLASLt] ERROR: fixedPointEmulationAdpMantissaBits called with unsupported type "
        << static_cast<int>(type_a) << std::endl;
    return 0;
}

/* Unified performance check — dispatches on type_a.
 * Compares the predicted emulation cost against native SGEMM (HIP_R_32F)
 * or native DGEMM (HIP_R_64F).                                           */
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
        = effective_time(tA, tB, m, n, k, pm_num, device, workspace_bytes, type_a);
    const double t_native
        = perf_model_times(tA, tB, m, n, k, pm_num, device, ~size_t{0}, type_a).t_native;
    return t_emul <= t_native;
}

/* Convenience wrapper: returns true when emulation would be applied for this
 * type and problem (enabled + device supported + strategy gate passes).      */
bool fixedPointEmulationWouldApply(const _rocblaslt_handle*       h,
                                   const _rocblaslt_matmul_desc*  desc,
                                   rocblaslt_compute_type         compute_type,
                                   hipDataType                    a_type,
                                   hipDataType                    b_type,
                                   hipDataType                    c_type,
                                   hipDataType                    d_type,
                                   int64_t                        m,
                                   int64_t                        n,
                                   int64_t                        k,
                                   int32_t                        batch_count)
{
    return fixedPointEmulationDecision(h,
                                       desc,
                                       compute_type,
                                       a_type,
                                       b_type,
                                       c_type,
                                       d_type,
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
                                  const FixedPointEmulationSettings& decision)
{
    using namespace FixedPointEmulation;

    const _rocblaslt_handle* h = reinterpret_cast<const _rocblaslt_handle*>(handle);
    /* Always ADP mode: layout uses max_num_moduli<float>() moduli. */
    const unsigned num_moduli = max_num_moduli<float>();
    const int      device     = h->device;

    /* Empty product: D = beta * C (see fp64EmulatedGemm). */
    if(k == 0)
        return native_gemm_fallback<float>(
            h, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd, stream);

    {
        const size_t W = decision.workspace_bytes;
        const void*  P = decision.workspace;

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

        /* Hard capacity check for every strategy — see fp64EmulatedGemm. */
        const bool   tA_g  = (opA != HIPBLAS_OP_N), tB_g = (opB != HIPBLAS_OP_N);
        const size_t W_min = min_workspace_bytes(
            FIXED_POINT_EMUL_MIN_MN, FIXED_POINT_EMUL_MIN_MN, k, num_moduli);
        if(W < W_min)
        {
            static std::atomic<int> s_ws_min_warns{0};
            if(s_ws_min_warns.fetch_add(1, std::memory_order_relaxed) < 5)
            {
                FixedPointEmulationSettings ws_dec{};
                ws_dec.workspace_bytes = ~size_t{0};
                const size_t opt_ws
                    = fixedPointEmulationWorkspaceSize(h, HIP_R_32F, opA, opB, m, n, k, ws_dec);
                const size_t nosplit_ws = min_workspace_bytes(m, n, k, num_moduli);
                constexpr size_t MiB    = size_t(1) << 20;
                hipblaslt_cerr
                    << "[hipBLASLt WARNING] FP32 emulation workspace (" << (W / MiB)
                    << " MiB) is below the minimum of " << ((W_min + MiB - 1) / MiB)
                    << " MiB, which includes a " << (OZ2_INT8_GEMM_WS_BYTES / MiB)
                    << " MiB INT8 GEMM reserve (m=" << m << ",n=" << n << ",k=" << k << ").\n"
                    << "  Without spatial splitting: " << ((nosplit_ws + MiB - 1) / MiB)
                    << " MiB. Optimal: " << (opt_ws / MiB) << " MiB.\n"
                    << "  Falling back to native SGEMM." << std::endl;
            }
            return rocblaslt_status_memory_error;
        }

        const bool is_eager_g = (decision.strategy == rocblaslt_emulation_strategy_eager);

        /* ADP-aware performance model s. */
        const int      adp_bits_g = (decision.adp_mantissa_bits > 0)
                                        ? static_cast<int>(decision.adp_mantissa_bits)
                                        : fixedPointEmulationAdpMantissaBits(HIP_R_32F);
        const unsigned pm_s_g     = adp_expected_num_moduli(adp_bits_g, num_moduli);

        /* FP32 performance model: compare against native SGEMM, not DGEMM. */
        const double t_emul_W
            = is_eager_g ? 0.0
                         : effective_time(tA_g, tB_g, m, n, k, pm_s_g, device, W, HIP_R_32F);
        const double t_native_g
            = is_eager_g
                  ? 1.0
                  : perf_model_times(tA_g, tB_g, m, n, k, pm_s_g, device, ~size_t{0}, HIP_R_32F)
                        .t_native;
        if(!is_eager_g && t_emul_W > t_native_g)
        {
            const double t_emul_opt
                = effective_time(tA_g, tB_g, m, n, k, pm_s_g, device, ~size_t{0}, HIP_R_32F);
            if(t_emul_opt <= t_native_g
               && s_ws_small_warns.fetch_add(1, std::memory_order_relaxed) < 5)
            {
                FixedPointEmulationSettings ws_dec{};
                ws_dec.workspace_bytes = ~size_t{0};
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
                                                          decision,
                                                          _prof ? &accum : nullptr);

    if(_prof)
    {
        (void)hipEventRecord(ev_end, stream);
        (void)hipStreamSynchronize(stream);
        float t_total = 0.f;
        (void)hipEventElapsedTime(&t_total, ev_start, ev_end);

        const unsigned       prof_s = accum.effective_s_used ? accum.effective_s_used : num_moduli;
        /* Report what ran with the caller's workspace, not an unlimited one. */
        const unsigned       chunk_size       = accum.chunk_size_used ? accum.chunk_size_used : prof_s;
        const unsigned       scale_chunk_size = chunk_size;
        const bool           tA               = (opA != HIPBLAS_OP_N);
        const bool           tB               = (opB != HIPBLAS_OP_N);
        const int            adp_bits_prof_32 = (decision.adp_mantissa_bits > 0)
                                                    ? static_cast<int>(decision.adp_mantissa_bits)
                                                    : fixedPointEmulationAdpMantissaBits(HIP_R_32F);
        const PerfModelTimes pm
            = effective_perf_model_times(tA,
                                         tB,
                                         m,
                                         n,
                                         k,
                                         adp_expected_num_moduli(adp_bits_prof_32, num_moduli),
                                         device,
                                         decision.workspace_bytes,
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
                    "pred_prelim_rel,pred_prelim_gemm_rel,pred_refine_rel,pred_adp_rel,"
                    "pred_scale_rel,pred_int8_gemm_rel,pred_accum_rel,"
                    "pred_host_rel,pred_total_rel\n");
            /* pred_*_rel: predicted cost relative to the predicted native SGEMM cost
             * (pred_total_rel < 1 means emulation is predicted to be faster). */
            std::fprintf(_f,
                         "%lld,%lld,%lld,%c,%c,%u,%u,%u,%u,"
                         "%llu,%u,"
                         "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                         "%.4f,%.4f,%.4f,%.4f,"
                         "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                         (long long)m,
                         (long long)n,
                         (long long)k,
                         tA ? 'T' : 'N',
                         tB ? 'T' : 'N',
                         num_moduli,
                         accum.effective_s_used,
                         scale_chunk_size,
                         chunk_size,
                         (unsigned long long)decision.workspace_bytes,
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
                         pm.t_prelim / pm.t_native,
                         pm.t_prelim_gemm / pm.t_native,
                         pm.t_refine / pm.t_native,
                         pm.t_adp / pm.t_native,
                         pm.t_scale / pm.t_native,
                         pm.t_int8_gemms / pm.t_native,
                         pm.t_accum / pm.t_native,
                         pm.t_host / pm.t_native,
                         pm.t_total / pm.t_native);
            std::fclose(_f);
        }
        (void)hipEventDestroy(ev_end);
        (void)hipEventDestroy(ev_start);
    }
    return st;
}
