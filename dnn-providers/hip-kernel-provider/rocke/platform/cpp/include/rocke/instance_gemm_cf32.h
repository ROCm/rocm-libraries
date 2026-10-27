/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gemm_cf32.h -- the named universal GEMM bodies of the
 * emulated-f32 routes (dispatch/gemm/cf32.py: cf32_body, _adjust_k) and the
 * thresholds of its body rule, for C++ callers that build those kernels
 * without the Python dispatch layer (hipTensor's kernel pack). The gemm parity
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
 * ("RIW", "RIW_SK", "Q3_SK", "C3") written into `spec` (every other field of
 * `spec` is left as it is). False for an unknown name. */
bool rocke_gemm_cf32_body(const char* name, const char* arch, rocke_gemm_universal_spec_t* spec);

/* Python _adjust_k(spec, M, Nk, wide): the launch adjustments of a body for
 * kernel extents M x Nk, applied in place: the 64-bit-address form with `wide`
 * (wide_addr, a non-persistent grid, dtl_soff0 unless the resources are tight).
 * Ragged extents (M or Nk not a tile multiple) need the Python-engine-only
 * levers dtl_tight_ragged_only / rve_pad: returns false and leaves `spec`
 * unchanged then. */
bool rocke_gemm_cf32_adjust_k(rocke_gemm_universal_spec_t* spec,
                              long long M,
                              long long Nk,
                              bool wide);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GEMM_CF32_H */
