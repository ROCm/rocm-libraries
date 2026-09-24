// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/*
 * FP32/FP64 GEMM emulation via Ozaki Scheme II (Fixed-Point Emulation).
 *
 * This header declares the host-side interface used by the library and tests.
 * It is intentionally free of any HIP device types so it can be included
 * from plain C++ translation units.
 *
 * Environment variables:
 *   HIPBLASLT_EMULATE_DOUBLE_PRECISION=0|1  enables FP64 emulation.
 *   HIPBLASLT_EMULATE_SINGLE_PRECISION=0|1  enables FP32 emulation.
 *   HIPBLASLT_EMULATION_NUM_MODULI=N        post-ADP testing override [2..20].
 *     After ADP selects the minimum sufficient s, this env var overrides it.
 *     Useful for testing specific moduli counts; accuracy is not guaranteed
 *     for all inputs when set below the ADP-selected value.
 *   HIPBLASLT_EMULATION_STRATEGY=performant|eager
 *     performant (default): use emulation only when predicted to outperform native GEMM.
 *     eager: always emulate regardless of problem size or arithmetic intensity.
 *   HIPBLASLT_EMULATION_SPECIAL_VALUES_SUPPORT_MASK=<hex>
 *     bitmask controlling NaN/Inf detection: bit 0 = Inf, bit 1 = NaN.  Default 0x3.
 *     Set to 0 to skip special-value detection entirely.
 *   HIPBLASLT_EMULATION_FP64_MANTISSA_BIT_COUNT=N
 *     ADP precision target for FP64 inputs in mantissa bits [1..52].
 *     Default 52 (full IEEE 754 FP64 precision).  Lower values -> fewer moduli -> faster.
 *     Takes precedence over HIPBLASLT_EMULATION_FP64_TOLERANCE.
 *   HIPBLASLT_EMULATION_FP32_MANTISSA_BIT_COUNT=N
 *     ADP precision target for FP32 inputs in mantissa bits [1..23].
 *     Default 23 (full IEEE 754 FP32 precision).  Lower values -> fewer moduli -> faster.
 *   HIPBLASLT_EMULATION_FP64_TOLERANCE=<value>
 *     target relative accuracy for ADP (dynamic) mode for FP64 inputs, expressed as a
 *     positive floating-point value (e.g. 1e-8, 1e-16).  ADP selects the minimum number
 *     of CRT moduli needed to achieve this accuracy; fewer moduli means faster GEMMs.
 *     Default (absent): full IEEE 754 double precision (~1.11e-16, 52 mantissa bits).
 *     If HIPBLASLT_EMULATION_FP64_MANTISSA_BIT_COUNT is also set, that takes precedence.
 *   HIPBLASLT_EMULATION_FP32_TOLERANCE=<value>
 *     target relative accuracy for ADP (dynamic) mode for FP32 inputs.
 *     Default (absent): full IEEE 754 single precision (~5.96e-8, 23 mantissa bits).
 *     If HIPBLASLT_EMULATION_FP32_MANTISSA_BIT_COUNT is also set, that takes precedence.
 *   HIPBLASLT_EMULATION_PROFILE=<path>
 *     append per-call profiling CSV rows to the given file path.
 *
 * Setting precedence (per-matmul desc attributes > env vars > built-in defaults).
 * Per-matmul emulation settings live on the matmul descriptor via
 * HIPBLASLT_MATMUL_DESC_EMULATION_*_EXT attributes; sentinel values
 * (-1 / ~0u / 0) fall back to the process-wide env var default.
 * See fixedPointEmulationDecision() for the full resolution logic.
 *
 * See docs/how-to/fp-emulation.rst
 */

#include "rocblaslt.h"
#include <hip/hip_runtime_api.h>

/* =========================================================================
 * Environment variable parsing infrastructure
 * ========================================================================= */
enum FixedPointEmulationEnvState
{
    FIXED_POINT_EMULATION_ENV_UNSET   = 0,
    FIXED_POINT_EMULATION_ENV_VALID   = 1,
    FIXED_POINT_EMULATION_ENV_INVALID = 2,
};

struct FixedPointEmulationEnvValue
{
    FixedPointEmulationEnvState state;
    unsigned int                value;
};

/* Pure parsers -- used by tests and by the cached runtime readers below.
 * All are stateless; the caller supplies the raw env-var string value.    */
FixedPointEmulationEnvValue fixedPointEmulationParseEnabledEnv(const char* value);
FixedPointEmulationEnvValue fixedPointEmulationParseStrategyEnv(const char* value);
FixedPointEmulationEnvValue fixedPointEmulationParseSpecialValuesMaskEnv(const char* value);
FixedPointEmulationEnvValue fixedPointEmulationParseNumModuliEnv(const char* value);
FixedPointEmulationEnvValue fixedPointEmulationParseToleranceEnv(const char* value);
/* Parser for HIPBLASLT_EMULATION_FP{32,64}_MANTISSA_BIT_COUNT.
 * Accepts decimal integers in [1..52]; returns INVALID for out-of-range.  */
FixedPointEmulationEnvValue fixedPointEmulationParseMantissaBitCountEnv(const char* value);

/* Forward declarations -- callers already include handle.h which provides
 * the full definitions.  Declared here so the functions below can use them. */
struct _rocblaslt_handle;
struct _rocblaslt_matmul_desc;

/* Returns true when emulation is enabled for the given type.
 *   type_a == HIP_R_64F: reads HIPBLASLT_EMULATE_DOUBLE_PRECISION
 *   type_a == HIP_R_32F: reads HIPBLASLT_EMULATE_SINGLE_PRECISION
 * Env var is read once and cached per type.
 * Any value other than "1" (including absent) returns false.              */
bool fixedPointEmulationIsEnabled(hipDataType type_a);

/* Returns true when emulation is estimated to be at least as fast as
 * native GEMM for the given problem size and input type.
 * Uses a Roofline performance model calibrated for the target hardware.
 * workspace_bytes -- caller workspace size; ~size_t{0} (SIZE_MAX) means no
 *   budget constraint (optimal, single-pass performance model).
 * desc carries the per-matmul emulation settings (may be nullptr).         */
bool fixedPointEmulationPerformanceCheck(const _rocblaslt_handle*      h,
                                         const _rocblaslt_matmul_desc* desc,
                                         hipDataType                   type_a,
                                         hipblasOperation_t            opA,
                                         hipblasOperation_t            opB,
                                         int64_t                       m,
                                         int64_t                       n,
                                         int64_t                       k,
                                         size_t                        workspace_bytes);

/* Returns true when HIPBLASLT_EMULATION_STRATEGY=eager is set, or when the
 * per-matmul desc requests EAGER strategy.
 * Shared for both FP32 and FP64.  Env var cached on first call.           */
bool fixedPointEmulationIsEager();

/* Returns the special-values support mask from
 * HIPBLASLT_EMULATION_SPECIAL_VALUES_SUPPORT_MASK (default: 0x3).
 *   bit 0 = Inf detection enabled
 *   bit 1 = NaN detection enabled
 * A return value of 0 means no Inf/NaN checking is performed.
 * Shared for both FP32 and FP64.                                           */
uint32_t fixedPointEmulationSpecialValuesMask();

/* Returns the post-ADP testing override from HIPBLASLT_EMULATION_NUM_MODULI
 * [2..S_MAX], or 0 if the env var is absent or invalid.
 * When non-zero, emulated_gemm_impl applies this as an override AFTER ADP
 * selects the minimum sufficient s.  Accuracy is not guaranteed for all
 * inputs when set below the ADP-selected value.
 * Maximum supported: 20 moduli (~155 bits of CRT capacity).               */
unsigned fixedPointEmulationNumModuli();

/* Returns the ADP target mantissa-bit count for the given input type.
 *   type_a == HIP_R_64F:
 *     1. Reads HIPBLASLT_EMULATION_FP64_MANTISSA_BIT_COUNT.
 *     2. Falls back to HIPBLASLT_EMULATION_FP64_TOLERANCE.
 *     3. Default: 52 (full IEEE 754 FP64 precision).
 *   type_a == HIP_R_32F:
 *     1. Reads HIPBLASLT_EMULATION_FP32_MANTISSA_BIT_COUNT.
 *     2. Falls back to HIPBLASLT_EMULATION_FP32_TOLERANCE.
 *     3. Default: 23 (full IEEE 754 FP32 precision).
 * Value range: [1..52] for FP64, [1..23] for FP32.
 * Cached on first call per type.  Only consulted in ADP (dynamic) mode.   */
int fixedPointEmulationAdpMantissaBits(hipDataType type_a);

/* =========================================================================
 * Decision struct and gate function -- status-propagating
 * ========================================================================= */
struct FixedPointEmulationDecision
{
    rocblaslt_status status; /* rocblaslt_status_invalid_value on bad env   */
    bool             apply; /* true -> use emulation; false -> native path */
    unsigned int     sv_mask; /* resolved special-values mask                */
    size_t           workspace_cap; /* caller workspace limit; ~size_t{0} = none  */
    /* ADP target precision in mantissa bits.
     * 52 = full IEEE 754 FP64 precision; 23 = full IEEE 754 FP32 precision.
     * Derived from the per-matmul desc attribute, then from the type-specific
     * env var (FP64/FP32_MANTISSA_BIT_COUNT or FP64/FP32_TOLERANCE).
     * Only consulted when dynamic_mode = true.                              */
    int adp_mantissa_bits; /* [1..52]; type default = full precision */
};

/* Status-returning emulation gate.  Invalid env-var values return
 * rocblaslt_status_invalid_value so callers do not silently fall back to
 * native GEMM.  On success, apply=false means the native path should be
 * used without error.  Dispatches on type_a (HIP_R_64F or HIP_R_32F).
 *
 * Setting precedence (highest to lowest):
 *   1. Per-matmul descriptor fields (desc->emulation_*; sentinel = inherit)
 *   2. Environment variables (read once, process-wide)
 *   3. Built-in defaults (ADP mode, S_MAX moduli, sv_mask = 0x3)
 *
 * desc may be nullptr when called from code that has no matmul desc
 * (e.g. workspace size query with no attached desc).
 * Always selects ADP (dynamic) mode; num_moduli is always S_MAX.          */
FixedPointEmulationDecision fixedPointEmulationDecision(const _rocblaslt_handle*      h,
                                                        const _rocblaslt_matmul_desc* desc,
                                                        hipDataType                   type_a,
                                                        hipblasOperation_t            opA,
                                                        hipblasOperation_t            opB,
                                                        int64_t                       m,
                                                        int64_t                       n,
                                                        int64_t                       k,
                                                        int32_t                       batch_count,
                                                        size_t workspace_bytes);

/* Convenience wrapper: returns true when fixedPointEmulationDecision(...).apply
 * is true for this type and problem. Cheaper than the full decision when only
 * the apply/reject outcome is needed.                                       */
bool fixedPointEmulationWouldApply(const _rocblaslt_handle*      h,
                                   const _rocblaslt_matmul_desc* desc,
                                   hipDataType                   type_a,
                                   hipblasOperation_t            opA,
                                   hipblasOperation_t            opB,
                                   int64_t                       m,
                                   int64_t                       n,
                                   int64_t                       k,
                                   int32_t                       batch_count);

/* Returns the workspace size in bytes for the given problem and input type.
 * Always uses S_MAX (= 20) moduli for the layout (always ADP mode).
 * When decision.workspace_cap != ~size_t{0}, the returned size is capped at
 * min(optimal, decision.workspace_cap).  This ensures the library never
 * reports a workspace larger than the user's allocation.                   */
size_t fixedPointEmulationWorkspaceSize(const _rocblaslt_handle*           h,
                                        hipDataType                        type_a,
                                        hipblasOperation_t                 opA,
                                        hipblasOperation_t                 opB,
                                        int64_t                            m,
                                        int64_t                            n,
                                        int64_t                            k,
                                        const FixedPointEmulationDecision& decision);

/* Per-call emulation settings.
 * Fields with sentinel values (0 for num_moduli, ~0u for sv_mask) cause the
 * function to fall back to the process-wide env var defaults.              */
struct FixedPointEmulationSettings
{
    unsigned int sv_mask; /* special-values mask; ~0u = env var       */
    bool         eager; /* true -> skip workspace performance gate  */
    void*        workspace; /* caller workspace; nullptr = no workspace */
    size_t       workspace_bytes; /* size of caller workspace                 */
    /* ADP target precision in mantissa bits.
     * For FP64: [1..52]; 52 = full FP64 precision.
     * For FP32: [1..23]; 23 = full FP32 precision.
     * 0 = sentinel: derive from env var (FP64/FP32_MANTISSA_BIT_COUNT
     *   or FP64/FP32_TOLERANCE) or type default.
     * Only consulted when dynamic_mode = true.                              */
    int adp_mantissa_bits; /* 0 = env var default                    */
};

/* Run an emulated FP64 GEMM using Ozaki Scheme II (accurate mode).
 *
 * Computes D = alpha * op(A) * op(B) + beta * C  where A, B, C, D are
 * device pointers to double arrays.  C and D may be the same pointer.
 * Only non-batched (batch count == 1) FP64 GEMM is supported.
 *
 * Returns rocblaslt_status_success on success,
 *         rocblaslt_status_memory_error if no workspace is provided or too small,
 *         rocblaslt_status_invalid_value if Inf/NaN is detected (sv_mask) or
 *             if ADP determines s=20 is insufficient for the input's range.
 * In invalid_value cases the caller should fall back to native FP64.       */
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
                                  const FixedPointEmulationSettings& settings);

/* Run an emulated FP32 GEMM using Ozaki Scheme II (accurate mode).
 *
 * Computes D = alpha * op(A) * op(B) + beta * C  where A, B, C, D are
 * device pointers to float arrays.  C and D may be the same pointer.
 * Only non-batched (batch count == 1) FP32 GEMM is supported.
 *
 * Returns rocblaslt_status_success on success,
 *         rocblaslt_status_memory_error if no workspace is provided or too small,
 *         rocblaslt_status_invalid_value if Inf/NaN is detected (sv_mask) or
 *             if ADP determines s=20 is insufficient for the input's range.
 * In invalid_value cases the caller should fall back to native FP32.       */
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
                                  const FixedPointEmulationSettings& settings);
