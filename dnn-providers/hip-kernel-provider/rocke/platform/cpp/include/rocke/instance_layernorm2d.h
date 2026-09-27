/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_layernorm2d.h -- C99 port of the LayerNorm2D forward kernel
 * instance builder rocke/instances/common/layernorm2d.py (CK Tile
 * 02_layernorm2d parity).
 *
 * For each row of an (M, N) activation tensor the kernel computes:
 *
 *     mean[m]    = sum_n(X[m,n]) / N
 *     var[m]     = sum_n((X[m,n] - mean[m])^2) / N        (stable Welford)
 *     inv_std[m] = 1 / sqrt(var[m] + eps)
 *     Y[m,n]     = (X[m,n] - mean[m]) * inv_std[m] * gamma[n] + beta[n]
 *
 * The build entry mirrors build_layernorm2d(spec) -> KernelDef:
 *
 *   Python (layernorm2d.py)               C99 (this header)
 *   -----------------------------------   --------------------------------------
 *   class LayerNorm2DSpec                  rocke_layernorm2d_spec_t
 *   is_valid_spec(spec, arch)             rocke_layernorm2d_is_valid_spec(...)
 *   build_layernorm2d(spec)               rocke_build_layernorm2d(b, spec)
 *   layernorm2d_grid(m, spec)             rocke_layernorm2d_grid(...)
 *   layernorm2d_signature(spec)           rocke_layernorm2d_signature(...)
 *   (+ convenience: build -> lower .ll)   rocke_layernorm2d_lower_to_llvm(...)
 *
 * SPEC AS AN EXPLICIT C STRUCT. The Python frozen dataclass carries defaults;
 * in C the caller fills a rocke_layernorm2d_spec_t. rocke_layernorm2d_spec_default()
 * returns a struct with every field set to the Python dataclass default; the
 * caller then overrides what it cares about (n_per_block is required).
 *
 * Error model mirrors the rest of the C port: build/lower routes errors through
 * the sticky-error IRBuilder (rocke_b_*); the validity gate returns a bool + a
 * reason string; the convenience lower returns a rocke_status_t.
 */
#ifndef ROCKE_INSTANCE_LAYERNORM2D_H
#define ROCKE_INSTANCE_LAYERNORM2D_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------- LayerNorm2DSpec
 *
 * Mirror of Python LayerNorm2DSpec. ``dtype`` is the Literal["f16","bf16"]
 * string (compared by strcmp like Python). The computed @property
 * ``elems_per_thread`` (n_per_block // block_size) is NOT stored; use
 * rocke_layernorm2d_elems_per_thread(). */
typedef struct rocke_layernorm2d_spec
{
    int n_per_block;
    int block_size; /* default 256                       */
    int vec; /* default 4                         */
    const char* dtype; /* default "f16" ("f16"|"bf16")      */
    bool save_mean_invstd; /* default false                     */
    int wave_size; /* default 64                        */
    const char* name; /* default "rocke_layernorm2d_fwd"  */
} rocke_layernorm2d_spec_t;

/* Default-constructed spec (every field == Python dataclass default). The
 * caller must still set `n_per_block`. */
rocke_layernorm2d_spec_t rocke_layernorm2d_spec_default(void);

/* LayerNorm2DSpec.elems_per_thread property: n_per_block // block_size. */
int rocke_layernorm2d_elems_per_thread(const rocke_layernorm2d_spec_t* spec);

/* LayerNorm2DSpec.kernel_name() -> NUL-terminated into out (capacity out_cap).
 * Returns ROCKE_OK or ROCKE_ERR_VALUE (buffer too small). */
rocke_status_t
    rocke_layernorm2d_kernel_name(const rocke_layernorm2d_spec_t* spec, char* out, size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950". On a
 * reject, `reason` (if non-NULL, capacity reason_cap) receives the structured
 * message; returns false. On accept returns true and writes "". */
bool rocke_layernorm2d_is_valid_spec(const rocke_layernorm2d_spec_t* spec,
                                     const char* arch,
                                     char* reason,
                                     size_t reason_cap);

/* build_layernorm2d(spec). Builds the IR into the supplied (already
 * rocke_ir_builder_init'd with spec.kernel_name()) builder `b`, exactly as the
 * Python build does, and returns the kernel (b->kernel) on success or NULL with
 * b's sticky error set. The caller owns `b`'s lifetime. */
rocke_kernel_def_t* rocke_build_layernorm2d(rocke_ir_builder_t* b,
                                            const rocke_layernorm2d_spec_t* spec);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with rocke_ir_builder_free(). Returns the kernel or NULL. */
rocke_kernel_def_t* rocke_build_layernorm2d_new(rocke_ir_builder_t* b,
                                                const rocke_layernorm2d_spec_t* spec);

/* layernorm2d_grid(m, spec) -> (x, y, z). Returns ceil_div_grid((m, 1)). */
rocke_status_t rocke_layernorm2d_grid(int m, const rocke_layernorm2d_spec_t* spec, int out[3]);

/* layernorm2d_signature(spec): build the {name,type} manifest into the
 * caller-provided `sb` (already rocke_signature_builder_init'd). Returns the
 * accumulated entries via *out_items / *out_count (arena-owned by sb). */
rocke_status_t rocke_layernorm2d_signature(rocke_signature_builder_t* sb,
                                           const rocke_layernorm2d_spec_t* spec,
                                           const rocke_sig_entry_t** out_items,
                                           size_t* out_count);

/* Convenience: given a spec, init a builder, build, and lower to LLVM .ll text.
 * `arch` NULL => "gfx950". On ROCKE_OK *out_ll receives a malloc'd NUL-terminated
 * string the caller frees with free(); on failure it is left NULL and (if
 * err!=NULL, capacity err_cap) a diagnostic is written. Internally owns and
 * frees its IRBuilder. */
rocke_status_t rocke_layernorm2d_lower_to_llvm(const rocke_layernorm2d_spec_t* spec,
                                               const char* arch,
                                               rocke_llvm_flavor_t flavor,
                                               char** out_ll,
                                               char* err,
                                               size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_LAYERNORM2D_H */
