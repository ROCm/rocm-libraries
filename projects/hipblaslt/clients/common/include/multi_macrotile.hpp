/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
 *
 * Multi-MacroTile: splits a GEMM into sub-problems solved with per-subproblem
 * kernel selection.  Only the Origami-based empirical split search (S17/S18)
 * is retained; all legacy uniform / heuristic strategies have been removed.
 *
 *******************************************************************************/

#pragma once

#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <memory>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>

// Forward declaration -- implementation in multi_macrotile_origami_improved.hpp
std::vector<int64_t> computeOrigamiOptimizedSplitsWithHandle(
    hipblasLtHandle_t handle, int64_t total_size, int num_splits,
    int macrotile_size, int64_t other_dim, int64_t K, bool is_m_split,
    hipblasOperation_t transA, hipblasOperation_t transB,
    hipDataType a_type, hipDataType b_type,
    hipDataType c_type, hipDataType d_type,
    hipblasComputeType_t compute_type,
    bool brute_force = false,
    bool xcd_aware = false,
    bool greedy = false);

// ── Sub-problem descriptor ──────────────────────────────────────────────────

struct GemmSubProblem
{
    int64_t m_size, n_size, k_size;
    int64_t m_offset, n_offset;
    size_t offset_A_bytes, offset_B_bytes;
    size_t offset_C_bytes, offset_D_bytes;
    size_t offset_E_bytes, offset_bias_bytes;
    int expected_workgroups;
    int macrotile_m, macrotile_n;
};

// Pre-created context used by the timing hot-loop (avoids per-iteration
// layout create / destroy / heuristic query overhead).
//
// Two operating modes:
//   (1) Default ("C-API mode"): matA/B/C/D + matmul_desc + algo + A/B/C/D_ptr
//       are used with hipblasLtMatmul on the hot path.
//   (2) "Ext-API mode" (when use_ext_gemm=true and ext_gemm is non-null):
//       hipblasLtMatmul is bypassed and ext_gemm->run(stream) is called
//       instead.  This is the only path that lets us override WGM at runtime
//       via hipblaslt_ext::GemmTuning::setWgm().
struct SubProblemContext
{
    // C-API state (always populated; used as fallback even in ext mode)
    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    hipblasLtMatmulDesc_t   matmul_desc; // per-subproblem descriptor for correct kernel selection
    hipblasLtMatmulAlgo_t   algo;
    void *A_ptr, *B_ptr, *C_ptr, *D_ptr;
    bool valid;
    bool owns_matmul_desc; // true if we created matmul_desc and need to destroy it

    // Ext-API state (populated only when --origami_wgm or --multi_mt_aware_wgm is set)
    bool    use_ext_gemm    = false;
    int16_t origami_wgm     = 0;  // Origami-recommended WGM (0 = kernel default)
    int16_t origami_wgmxcc  = 0;  // Origami-recommended WGMXCC
    int16_t origami_wgmxccchunk = 0; // Origami-recommended WGMXCCCHUNK
    std::shared_ptr<hipblaslt_ext::Gemm> ext_gemm;          // initialized once
    std::shared_ptr<hipblaslt_ext::GemmTuning> ext_tuning;  // configured per-sub
};

// Dispatch a single sub-problem on the given stream.  Picks ext_gemm.run()
// if available (i.e. --origami_wgm path), otherwise falls back to the
// classic C-API hipblasLtMatmul.
inline hipblasStatus_t dispatchSubProblem(
    hipblasLtHandle_t handle,
    const SubProblemContext& ctx,
    const void* alpha, const void* beta,
    void* workspace, size_t workspace_size,
    hipStream_t stream)
{
    if (ctx.use_ext_gemm && ctx.ext_gemm)
    {
        // Inputs were already bound during initialization; just run.
        return ctx.ext_gemm->run(stream);
    }
    return hipblasLtMatmul(handle, ctx.matmul_desc, alpha,
                           ctx.A_ptr, ctx.matA, ctx.B_ptr, ctx.matB,
                           beta, ctx.C_ptr, ctx.matC, ctx.D_ptr, ctx.matD,
                           &ctx.algo, workspace, workspace_size, stream);
}

// ── Helpers ─────────────────────────────────────────────────────────────────

inline size_t getDataTypeSize(hipDataType dt)
{
    switch (dt)
    {
    case HIP_R_64F:                      return 8;
    case HIP_R_32F: case HIP_R_32I:     return 4;
    case HIP_R_16F: case HIP_R_16BF:    return 2;
    case HIP_R_8I:  case HIP_R_8U:
    case HIP_R_8F_E4M3: case HIP_R_8F_E5M2:
    case HIP_R_8F_E4M3_FNUZ: case HIP_R_8F_E5M2_FNUZ:
                                         return 1;
    default:                             return 2;
    }
}

inline size_t calculateOffsetA(int64_t m_off, int64_t k_off, int64_t lda,
                                hipblasOperation_t transA, hipDataType dt)
{
    size_t e = getDataTypeSize(dt);
    // transA=N: A is [M x K] col-major, lda >= M.  Element (m, k) at m + k*lda.
    //           Skipping m_off rows: offset = m_off + k_off * lda.
    // transA=T: A is [K x M] col-major, lda >= K.  Element (k, m) at k + m*lda.
    //           Skipping m_off columns: offset = k_off + m_off * lda.
    return (transA == HIPBLAS_OP_N)
        ? (m_off + k_off * lda) * e
        : (k_off + m_off * lda) * e;
}

inline size_t calculateOffsetB(int64_t n_off, int64_t k_off, int64_t ldb,
                                hipblasOperation_t transB, hipDataType dt)
{
    size_t e = getDataTypeSize(dt);
    return (transB == HIPBLAS_OP_N) ? n_off * ldb * e : n_off * e;
}

inline size_t calculateOffsetCD(int64_t m_off, int64_t n_off, int64_t ld,
                                 hipDataType dt)
{
    return (m_off + n_off * ld) * getDataTypeSize(dt);
}

inline size_t calculateOffsetBias(int64_t m_off, hipDataType dt)
{
    return m_off * getDataTypeSize(dt);
}

inline int estimateWorkgroups(int64_t M, int64_t N, int mt_m, int mt_n)
{
    if (mt_m <= 0 || mt_n <= 0) return 0;
    return (int)(((M + mt_m - 1) / mt_m) * ((N + mt_n - 1) / mt_n));
}

// ── Pre-split guard: reject obviously bad problem shapes ─────────────────────
// Derived from 1440-point benchmark across all 4 layouts on MI350X (gfx950).

inline bool shouldUseMultiMacroTile(int64_t M, int64_t N, int64_t K)
{
    if (std::min(M, N) < 5120) return false;
    if (K < 2048) return false;
    return true;
}

// ── MT-aware heuristic: the main decision function ──────────────────────────
// Call AFTER querying the baseline kernel's MacroTile via
// hipblasLtMatmulAlgoGetHeuristic + parseMacroTileFromName.
//
// Based on 1440-point analysis (728 valid with baseline):
//   MT_N=256 baseline → 11% win rate, -1.3% avg → DON'T SPLIT
//   MT_N∈{240,224,208,176} → 66% win rate, +8.6% avg → SPLIT
//   transB=T (NT/TT layouts) → 14-19% win rate → DON'T SPLIT
//
// Applying this heuristic to the full dataset:
//   ENABLED:  61 pts → 40 wins (66%), 4 losses (7%), avg +8.6%
//   DISABLED: 299 pts → 55 wins (18%), 101 losses (34%), avg -1.1%
// Net effect: eliminates most regressions while keeping the best gains.

inline bool shouldEnableMultiMT_MTAware(
    size_t mt_m, size_t mt_n,
    hipblasOperation_t transB,
    int64_t M, int64_t N,
    bool is_m_split)
{
    if (std::min(M, N) < 5120) return false;

    // Check the MacroTile component along the NON-SPLIT axis.
    // M-split preserves N → check MT_N. N-split preserves M → check MT_M.
    size_t mt_check = is_m_split ? mt_n : mt_m;

    if (mt_check == 256) return false;
    if (mt_check == 240 || mt_check == 224 || mt_check == 208 || mt_check == 176
        || mt_check == 192) return true;
    return false;
}

// ── Main entry point ────────────────────────────────────────────────────────
// Splits a GEMM along M (strategy 17) or N (strategy 18) using the Origami
// empirical search.  Returns a single-element vector when splitting is
// disabled or Origami declines.

inline std::vector<GemmSubProblem> splitGemmProblem(
    int64_t M, int64_t N, int64_t K,
    int64_t lda, int64_t ldb, int64_t ldc, int64_t ldd, int64_t lde,
    hipblasOperation_t transA, hipblasOperation_t transB,
    hipDataType a_type, hipDataType b_type,
    hipDataType c_type, hipDataType d_type,
    hipDataType aux_type, hipDataType bias_type,
    int split_strategy,
    int num_splits,
    int macrotile_m = 128, int macrotile_n = 128,
    hipblasLtHandle_t handle = nullptr,
    hipblasComputeType_t compute_type = HIPBLAS_COMPUTE_32F)
{
    std::vector<GemmSubProblem> subs;

    // Single-problem fallback
    auto makeSingle = [&]() {
        GemmSubProblem s{};
        s.m_size = M; s.n_size = N; s.k_size = K;
        s.expected_workgroups = estimateWorkgroups(M, N, macrotile_m, macrotile_n);
        s.macrotile_m = macrotile_m; s.macrotile_n = macrotile_n;
        subs.push_back(s);
        return subs;
    };

    // Strategy mapping:
    // 17/19 = M-split (19 = brute-force)
    // 18/20 = N-split (20 = brute-force)
    // 21 = 3-way M-split, 22 = 3-way N-split
    // 23 = XCD-aware M-split, 24 = XCD-aware N-split
    // 25 = Greedy biggest-MT M-split (single deterministic candidate)
    // 26 = Greedy biggest-MT N-split (single deterministic candidate)
    // 27 = Greedy auto: try both axes, pick the one with the smaller leftover
    bool is_m_split = (split_strategy == 17 || split_strategy == 19
                       || split_strategy == 21 || split_strategy == 23
                       || split_strategy == 25);
    bool brute_force = (split_strategy == 19 || split_strategy == 20);
    bool three_way = (split_strategy == 21 || split_strategy == 22);
    bool xcd_aware = (split_strategy == 23 || split_strategy == 24);
    bool greedy    = (split_strategy == 25 || split_strategy == 26 || split_strategy == 27);
    bool greedy_auto = (split_strategy == 27);

    if (three_way && num_splits <= 0) num_splits = 3;
    else if (num_splits <= 0) num_splits = 2;
    if (greedy) num_splits = 2; // greedy is always 2-way: big-MT piece + leftover
    num_splits = std::min(num_splits, 16);

    std::vector<int64_t> split_sizes;
    if (handle)
    {
        if (greedy_auto)
        {
            // Try both axes; prefer the one with the smaller leftover (less
            // work assigned to the slow tail kernel).  If only one axis
            // produces a useful split, take it.  Tie-break: prefer M-split
            // (it tends to produce better-balanced kernels for transB=N
            // layouts based on the existing benchmark data).
            std::vector<int64_t> m_split = computeOrigamiOptimizedSplitsWithHandle(
                handle, M, num_splits, macrotile_m, N, K, /*is_m_split=*/true,
                transA, transB, a_type, b_type, c_type, d_type, compute_type,
                /*brute_force=*/false, /*xcd_aware=*/false, /*greedy=*/true);
            std::vector<int64_t> n_split = computeOrigamiOptimizedSplitsWithHandle(
                handle, N, num_splits, macrotile_n, M, K, /*is_m_split=*/false,
                transA, transB, a_type, b_type, c_type, d_type, compute_type,
                /*brute_force=*/false, /*xcd_aware=*/false, /*greedy=*/true);

            int64_t m_leftover = m_split.size() == 2 ? m_split[1] : INT64_MAX;
            int64_t n_leftover = n_split.size() == 2 ? n_split[1] : INT64_MAX;

            if (m_leftover == INT64_MAX && n_leftover == INT64_MAX)
            {
                split_sizes.clear();
            }
            else if (m_leftover <= n_leftover)
            {
                split_sizes = m_split;
                is_m_split  = true;
            }
            else
            {
                split_sizes = n_split;
                is_m_split  = false;
            }
        }
        else
        {
            int64_t dim   = is_m_split ? M : N;
            int64_t other = is_m_split ? N : M;
            int     mt    = is_m_split ? macrotile_m : macrotile_n;

            split_sizes = computeOrigamiOptimizedSplitsWithHandle(
                handle, dim, num_splits, mt, other, K, is_m_split,
                transA, transB, a_type, b_type, c_type, d_type, compute_type,
                brute_force, xcd_aware, greedy);
        }
    }

    if (split_sizes.empty())
        return makeSingle();

    num_splits = (int)split_sizes.size();

    // Build sub-problems from split sizes
    int64_t offset = 0;
    for (int i = 0; i < num_splits; i++)
    {
        GemmSubProblem s{};
        int64_t sz = split_sizes[i];

        if (is_m_split)
        {
            s.m_size = sz;        s.n_size = N;   s.k_size = K;
            s.m_offset = offset;  s.n_offset = 0;
            s.offset_A_bytes    = calculateOffsetA(offset, 0, lda, transA, a_type);
            s.offset_B_bytes    = 0;
            s.offset_C_bytes    = calculateOffsetCD(offset, 0, ldc, c_type);
            s.offset_D_bytes    = calculateOffsetCD(offset, 0, ldd, d_type);
            s.offset_E_bytes    = calculateOffsetCD(offset, 0, lde, aux_type);
            s.offset_bias_bytes = calculateOffsetBias(offset, bias_type);
        }
        else
        {
            s.m_size = M;         s.n_size = sz;  s.k_size = K;
            s.m_offset = 0;       s.n_offset = offset;
            s.offset_A_bytes    = 0;
            s.offset_B_bytes    = calculateOffsetB(offset, 0, ldb, transB, b_type);
            s.offset_C_bytes    = calculateOffsetCD(0, offset, ldc, c_type);
            s.offset_D_bytes    = calculateOffsetCD(0, offset, ldd, d_type);
            s.offset_E_bytes    = calculateOffsetCD(0, offset, lde, aux_type);
            s.offset_bias_bytes = 0;
        }

        s.expected_workgroups = estimateWorkgroups(s.m_size, s.n_size, macrotile_m, macrotile_n);
        s.macrotile_m = macrotile_m;
        s.macrotile_n = macrotile_n;

        subs.push_back(s);
        offset += sz;
    }

    return subs;
}
