/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_mx_gemm.h -- C99 port of the MX (microscaling) GEMM kernel
 * instance builder rocke/instances/common/mx_gemm.py (CK Tile 42_mx_gemm
 * parity).
 *
 *   Python (mx_gemm.py)               C99 (this header)
 *   --------------------------------  -----------------------------------------
 *   class MxGemmSpec                  rocke_mx_gemm_spec_t
 *   MxGemmSpec.kernel_name()          rocke_mx_gemm_kernel_name(...)
 *   is_valid_spec(spec, arch)         rocke_mx_gemm_is_valid_spec(...)
 *   build_mx_gemm(spec, arch)         rocke_build_mx_gemm(...)
 *   mx_gemm_grid(spec)                rocke_mx_gemm_grid(...)
 *   (+ convenience: init-from-spec)   rocke_build_mx_gemm_new(...)
 *   (+ convenience: build -> lower)   rocke_mx_gemm_lower_to_llvm(...)
 *
 * The MX GEMM uses the OCP MX-spec shared-exponent format: each 32-element
 * mantissa block carries one 8-bit unbiased E8M0 scale. v1 ships fp8e4m3 /
 * bf8e5m2 mantissa, group_k=32, 16x16 tiles, wave64, f32 output. The body
 * uses the fp8/bf8 16x16x32 MFMA atom plus an explicit per-group E8M0 decode +
 * scale-apply chain (decode_mx_scale_e8m0).
 *
 * SPEC AS AN EXPLICIT C STRUCT. The Python dataclass is a frozen value type with
 * defaults. In C the caller fills a rocke_mx_gemm_spec_t;
 * rocke_mx_gemm_spec_default() returns a struct with every field at the Python
 * dataclass default (M/N/K must still be set).
 *
 * Error model mirrors the rest of the C port: build/lower route errors through
 * the sticky-error IRBuilder (rocke_b_*); the validity gate returns a bool + a
 * reason string; the convenience lower returns a rocke_status_t.
 */
#ifndef ROCKE_INSTANCE_MX_GEMM_H
#define ROCKE_INSTANCE_MX_GEMM_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ MxGemmSpec *
 *
 * Mirror of Python MxGemmSpec (frozen dataclass). Field order tracks the Python
 * declaration order. `mantissa_dtype` and `name` are const char* string
 * literals compared by strcmp like the Python Literal / str. */
typedef struct rocke_mx_gemm_spec
{
    int M;
    int N;
    int K;
    const char* mantissa_dtype; /* "fp8e4m3" (default) | "bf8e5m2"     */
    int group_k; /* MX shared-exponent block size, default 32 */
    int block_tile_m; /* default 16 */
    int block_tile_n; /* default 16 */
    const char* name; /* default "rocke_mx_gemm" */
    bool per_input_row; /* default true (v1 experimental scale mode) */
} rocke_mx_gemm_spec_t;

/* Default-constructed spec (every field == Python dataclass default). The
 * caller must still set M / N / K. */
rocke_mx_gemm_spec_t rocke_mx_gemm_spec_default(void);

/* MxGemmSpec.block_size property: MFMA path uses one wave64 warp per CTA = 64. */
int rocke_mx_gemm_block_size(const rocke_mx_gemm_spec_t* spec);

/* MxGemmSpec.atom property: the fp8 / bf8 16x16x32 MFMA atom selected from
 * mantissa_dtype. Returns a pointer into the static MFMA catalog, or NULL on
 * the Python ValueError path ((block_tile_m, block_tile_n) != (16, 16) or an
 * unsupported mantissa_dtype). Do NOT free/mutate the returned pointer. */
const struct rocke_mfma_atom* rocke_mx_gemm_atom(const rocke_mx_gemm_spec_t* spec);

/* MxGemmSpec.kernel_name() -> NUL-terminated into out (capacity out_cap).
 * Joins name, "M{M}N{N}K{K}", mantissa_dtype, "gk{group_k}",
 * "t{block_tile_m}x{block_tile_n}" via kernel_name_join. Returns ROCKE_OK or
 * ROCKE_ERR_VALUE (buffer too small / NULL args). */
rocke_status_t
    rocke_mx_gemm_kernel_name(const rocke_mx_gemm_spec_t* spec, char* out, size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950". On a
 * reject, `reason` (if non-NULL, capacity reason_cap) receives the Python
 * message and false is returned. On accept returns true and writes "ok". */
bool rocke_mx_gemm_is_valid_spec(const rocke_mx_gemm_spec_t* spec,
                                 const char* arch,
                                 char* reason,
                                 size_t reason_cap);

/* build_mx_gemm(spec, arch). Builds the IR into the supplied (already
 * rocke_ir_builder_init'd with spec.kernel_name()) builder `b`, exactly as the
 * Python build does, and returns the kernel (b->kernel) on success or NULL with
 * b's sticky error set. `arch` NULL => "gfx950". Does NOT re-init the builder. */
rocke_kernel_def_t*
    rocke_build_mx_gemm(rocke_ir_builder_t* b, const rocke_mx_gemm_spec_t* spec, const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with rocke_ir_builder_free(). Returns the kernel or NULL. */
rocke_kernel_def_t* rocke_build_mx_gemm_new(rocke_ir_builder_t* b,
                                            const rocke_mx_gemm_spec_t* spec,
                                            const char* arch);

/* Convenience: given a spec, init a builder, build, and lower to LLVM .ll text.
 * `arch` NULL => "gfx950". On ROCKE_OK *out_ll receives a malloc'd NUL-terminated
 * string the caller frees with free(); on failure it is left NULL and (if
 * err!=NULL, capacity err_cap) a diagnostic is written. Internally owns and
 * frees its IRBuilder. */
rocke_status_t rocke_mx_gemm_lower_to_llvm(const rocke_mx_gemm_spec_t* spec,
                                           const char* arch,
                                           rocke_llvm_flavor_t flavor,
                                           char** out_ll,
                                           char* err,
                                           size_t err_cap);

/* mx_gemm_grid(spec) -> (n_tiles, m_tiles, 1). Any out-param may be NULL. */
void rocke_mx_gemm_grid(const rocke_mx_gemm_spec_t* spec, int* out_gx, int* out_gy, int* out_gz);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_MX_GEMM_H */
