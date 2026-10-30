/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gemm_cf32.h -- the named universal GEMM bodies of the
 * emulated-f32 routes (dispatch/gemm/cf32.py: cf32_body, _adjust_k), the
 * constants of its body rules (gfx950 "waves", gfx942 "cost"), the gfx942
 * operand split modes and native-route thresholds, and the launch limits of a
 * body (gemm_launch_limits), for C++ callers that build those kernels without
 * the Python dispatch layer (hipTensor's kernel pack). The gemm_cf32 parity
 * family builds every body through this API and through cf32.py and compares
 * the IR.
 */
#ifndef ROCKE_INSTANCE_GEMM_CF32_H
#define ROCKE_INSTANCE_GEMM_CF32_H

#include <stdbool.h>

#include "rocke/instance_gemm_universal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The thresholds of the body rule (cf32.py CF32_SK_ALWAYS_WAVES, CF32_SK_MAX_WAVES,
 * CF32_SK_MAX_TAIL, CF32_SMALL_MAX_WAVES). */
#define ROCKE_GEMM_CF32_SK_ALWAYS_WAVES 4.0
#define ROCKE_GEMM_CF32_SK_MAX_WAVES 32.0
#define ROCKE_GEMM_CF32_SK_MAX_TAIL 0.65
#define ROCKE_GEMM_CF32_SMALL_MAX_WAVES 0.5

/* Python cf32_body(name, arch): tile, trait and block size of the named body
 * (gfx950: "RIW", "RIW_SK", "Q3_SK", "C3"; gfx942, the register-staged bodies
 * of _VP_BODIES: "G<M>x<N>x<K><variant>") written into `spec` (every other
 * field of `spec` is left as it is). False for an unknown name. */
bool rocke_gemm_cf32_body(const char* name, const char* arch, rocke_gemm_universal_spec_t* spec);

/* Python _adjust_k(spec, M, Nk, wide): the launch adjustments of a body for
 * kernel extents M x Nk, applied in place: the 64-bit-address form with `wide`
 * (wide_addr, a non-persistent grid, dtl_soff0 unless the resources are tight).
 * Ragged extents (M or Nk not a tile multiple) need the Python-engine-only
 * levers dtl_tight_ragged_only / rve_pad: returns false and leaves `spec`
 * unchanged then. A register-staged body (vgpr_prefetch) takes pad_m / pad_n
 * for ragged extents and wide_addr with `wide` (true). */
bool rocke_gemm_cf32_adjust_k(rocke_gemm_universal_spec_t* spec,
                              long long M,
                              long long Nk,
                              bool wide);

/* Python gemm_body_coef(body, route): the coefficients (a, b, occ) of a body
 * in the gfx942 "cost" rule (CF32_COST of the route or of its CF32_ROUTE_ALIAS).
 * False when there are none. */
bool rocke_gemm_cf32_body_cost(const char* body, const char* route, double* a, double* b, int* occ);

/* Python split_mode(arch, route): the operand split of an f32 route's prologue
 * on `arch` (CF32_SPLIT_MODE: "trunc" or "rne"). */
const char* rocke_gemm_cf32_split_mode(const char* arch, const char* route);

/* Python CF32_NATIVE_FIRST[arch][route] ("x6" real, "cf32" complex): the real
 * multiply-adds below which the x6 recipe's native f32 route goes first; 0
 * when the recipe does not list that route on `arch`. */
double rocke_gemm_cf32_native_first(const char* arch, const char* route);

/* Python gemm_launch_limits(spec): the launch limits of a body launched on
 * extents padded to tile multiples (the limit_* attributes of hipTensor's
 * kernel pack): operand bytes (all batches) below operand_bytes, C (all
 * batches) at most c_elements; the 64-bit-address form: one tile's operand
 * rows at most tile_operand_bytes, one tile's C rows at most c_tile_elements;
 * rows of tiles and batches at most grid_yz. */
typedef struct rocke_gemm_cf32_limits
{
    long long operand_bytes;
    long long c_elements;
    long long tile_operand_bytes;
    long long c_tile_elements;
    long long grid_yz;
} rocke_gemm_cf32_limits_t;

void rocke_gemm_cf32_launch_limits(const rocke_gemm_universal_spec_t* spec,
                                   rocke_gemm_cf32_limits_t* out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GEMM_CF32_H */
