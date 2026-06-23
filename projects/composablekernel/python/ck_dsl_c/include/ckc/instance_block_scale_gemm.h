/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/instance_block_scale_gemm.h -- C99 port of the block-scaled GEMM kernel
 * instance builder ck_dsl/instances/common/block_scale_gemm.py (CK Tile
 * 38_block_scale_gemm parity, v1 scalar/MFMA inner).
 *
 *   Python (block_scale_gemm.py)              C99 (this header)
 *   ---------------------------------------   ------------------------------------
 *   class BlockScaleGemmSpec                  ckc_block_scale_gemm_spec_t
 *   spec.atom (@property)                     ckc_block_scale_gemm_spec_atom(...)
 *   spec.block_size (@property)               ckc_block_scale_gemm_spec_block_size
 *   spec.kernel_name()                        ckc_block_scale_gemm_kernel_name(...)
 *   is_valid_spec(spec, arch)                 ckc_block_scale_gemm_is_valid_spec
 *   _mantissa_storage_dtype(spec)             ckc_block_scale_gemm_mantissa_store
 *   _storage_ir_type(store)                   ckc_block_scale_gemm_storage_ir_type
 *   build_block_scale_gemm(spec, arch)        ckc_build_block_scale_gemm(...)
 *   block_scale_gemm_grid(spec)               ckc_block_scale_gemm_grid(...)
 *   block_scale_gemm_signature(spec)          ckc_block_scale_gemm_signature(...)
 *   (+ convenience: build -> lower .ll)       ckc_block_scale_gemm_lower_to_llvm
 *
 * SPEC AS AN EXPLICIT C STRUCT. The Python dataclass is a frozen value type with
 * defaults. In C the caller fills a ckc_block_scale_gemm_spec_t.
 * ckc_block_scale_gemm_spec_default() returns a struct with every field set to
 * the Python dataclass default; ckc_block_scale_gemm_spec_new() is a convenience
 * matching the doc'd call pattern. The two computed @property values
 * (atom, block_size) are recomputed by accessors below, never stored.
 *
 * Error model mirrors the rest of the C port: build/lower routes errors through
 * the sticky-error IRBuilder (ckc_b_*); the validity gate returns a bool + a
 * reason string; the convenience lower returns a ckc_status_t.
 */
#ifndef CKC_INSTANCE_BLOCK_SCALE_GEMM_H
#define CKC_INSTANCE_BLOCK_SCALE_GEMM_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
#include "ckc/helper_ck_dsl.helpers.atoms.h" /* ckc_mfma_atom_t */
#include "ckc/helper_ck_dsl.helpers.spec.h"  /* ckc_sig_entry_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------- BlockScaleGemmSpec *
 *
 * Mirror of the Python frozen dataclass, field order identical:
 *   M, N, K, quant_mode, mantissa_dtype, preshuffle_b, group_size_mnk,
 *   block_tile_m, block_tile_n, name, per_input_row.
 *
 * quant_mode  : "aquant" | "bquant" | "abquant"  (default "bquant")
 * mantissa_dtype : "fp8e4m3" | "bf8e5m2" | "i4_fp8" | "i4_bf8" (default fp8e4m3)
 * Strings are const char* (string literals); compared by strcmp like Python. */
typedef struct ckc_block_scale_gemm_spec
{
    int M;
    int N;
    int K;
    const char* quant_mode;     /* default "bquant"   */
    const char* mantissa_dtype; /* default "fp8e4m3"  */
    bool preshuffle_b;          /* default false      */
    int group_m;                /* group_size_mnk[0], default 1   */
    int group_n;                /* group_size_mnk[1], default 1   */
    int group_k;                /* group_size_mnk[2], default 128 */
    int block_tile_m;           /* default 16 */
    int block_tile_n;           /* default 16 */
    const char* name;           /* default "ck_dsl_block_scale_gemm" */
    bool per_input_row;         /* default true */
} ckc_block_scale_gemm_spec_t;

/* Default-constructed spec (every field == Python dataclass default). The caller
 * must still set M/N/K and any non-default fields. */
ckc_block_scale_gemm_spec_t ckc_block_scale_gemm_spec_default(void);

/* Convenience matching the doc'd call pattern:
 *   spec = ckc_block_scale_gemm_spec_new(M, N, K, "abquant", "fp8e4m3",
 *                                        group_m=1, group_n=1, group_k=128)
 * All other fields take their dataclass defaults. Returns the spec by value. */
ckc_block_scale_gemm_spec_t ckc_block_scale_gemm_spec_new(int M,
                                                          int N,
                                                          int K,
                                                          const char* quant_mode,
                                                          const char* mantissa_dtype,
                                                          int group_m,
                                                          int group_n,
                                                          int group_k);

/* spec.atom (@property): MFMA atom from (block_tile_m, block_tile_n,
 * mantissa_dtype). Returns a pointer into the static MFMA catalog (do NOT
 * free/mutate), or NULL on the Python ValueError path (non-16x16 tile or
 * unknown mantissa). */
const ckc_mfma_atom_t* ckc_block_scale_gemm_spec_atom(const ckc_block_scale_gemm_spec_t* spec);

/* spec.block_size (@property): always 64 (one wave64 warp per CTA). */
int ckc_block_scale_gemm_spec_block_size(const ckc_block_scale_gemm_spec_t* spec);

/* spec.kernel_name() -> NUL-terminated into out (capacity out_cap).
 * Returns CKC_OK or CKC_ERR_VALUE (buffer too small / bad spec). */
ckc_status_t ckc_block_scale_gemm_kernel_name(const ckc_block_scale_gemm_spec_t* spec,
                                              char* out,
                                              size_t out_cap);

/* _mantissa_storage_dtype(spec): "fp8e4m3"/"bf8e5m2" pass through; the i4_*
 * variants map to "i8". Returns a static string. */
const char* ckc_block_scale_gemm_mantissa_store(const ckc_block_scale_gemm_spec_t* spec);

/* _storage_ir_type(store): "f16"->io_ir_type, "i8"->I8, else quant_ir_type.
 * Returns the interned scalar singleton, or NULL on an unsupported store. */
const ckc_type_t* ckc_block_scale_gemm_storage_ir_type(const char* store);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950". On a
 * reject, `reason` (if non-NULL, capacity reason_cap) receives the structured
 * message and false is returned. On accept returns true and writes "ok".
 * `b` is used only for arena ownership of the block-size cap reason string (the
 * validate_arch_and_block_size contract); pass a live builder, or NULL to skip
 * that arena-owned path (then a generic message is used). */
bool ckc_block_scale_gemm_is_valid_spec(ckc_ir_builder_t* b,
                                        const ckc_block_scale_gemm_spec_t* spec,
                                        const char* arch,
                                        char* reason,
                                        size_t reason_cap);

/* build_block_scale_gemm(spec, arch). Builds the IR into the supplied (already
 * ckc_ir_builder_init'd) builder `b`, exactly as the Python build does, and
 * returns the kernel (b->kernel) on success or NULL with b's sticky error set.
 * `arch` NULL => "gfx950". This routine does NOT re-init the builder. */
ckc_kernel_def_t* ckc_build_block_scale_gemm(ckc_ir_builder_t* b,
                                             const ckc_block_scale_gemm_spec_t* spec,
                                             const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with ckc_ir_builder_free(). Returns the kernel or NULL. */
ckc_kernel_def_t* ckc_build_block_scale_gemm_new(ckc_ir_builder_t* b,
                                                 const ckc_block_scale_gemm_spec_t* spec,
                                                 const char* arch);

/* block_scale_gemm_grid(spec) -> (ceil(N/tile_n), ceil(M/tile_m), 1). Writes
 * the three axes to out[0..2]. Returns CKC_OK, or CKC_ERR_VALUE on NULL args. */
ckc_status_t ckc_block_scale_gemm_grid(const ckc_block_scale_gemm_spec_t* spec, int out[3]);

/* block_scale_gemm_signature(spec): the manifest signature entry list. The
 * entries (and their copied strings) live in `arena`. On success *out_items /
 * *out_count receive the arena-owned array; returns CKC_OK. */
ckc_status_t ckc_block_scale_gemm_signature(struct ckc_arena* arena,
                                            const ckc_block_scale_gemm_spec_t* spec,
                                            const ckc_sig_entry_t** out_items,
                                            size_t* out_count);

/* Convenience: given a spec, init a builder, build, and lower to LLVM .ll text.
 * `arch` NULL => "gfx950". On CKC_OK *out_ll receives a malloc'd NUL-terminated
 * string the caller frees with free(); on failure it is left NULL and (if
 * err!=NULL, capacity err_cap) a diagnostic is written. Internally owns and
 * frees its IRBuilder. */
ckc_status_t ckc_block_scale_gemm_lower_to_llvm(const ckc_block_scale_gemm_spec_t* spec,
                                                const char* arch,
                                                ckc_llvm_flavor_t flavor,
                                                char** out_ll,
                                                char* err,
                                                size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_INSTANCE_BLOCK_SCALE_GEMM_H */
