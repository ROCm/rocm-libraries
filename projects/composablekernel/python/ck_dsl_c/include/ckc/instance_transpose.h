/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/instance_transpose.h -- C99 port of the 2D transpose kernel instance
 * builder ck_dsl/instances/common/transpose.py.
 *
 * transpose.py is a pure memory-movement kernel (coalesced HBM read -> LDS
 * stage -> coalesced HBM write) with NO MFMA atoms. The build entry mirrors
 * the Python op-by-op so the emitted IR op stream is byte-identical.
 *
 *   Python (transpose.py)                  C99 (this header)
 *   -----------------------------------    -------------------------------------
 *   class Transpose2DSpec                  ckc_transpose2d_spec_t
 *   Transpose2DSpec.block_size             ckc_transpose2d_block_size()
 *   Transpose2DSpec.kernel_name()          ckc_transpose2d_kernel_name()
 *   is_valid_spec(spec, arch)              ckc_transpose2d_is_valid_spec(...)
 *   build_transpose2d(spec)                ckc_build_transpose2d(...)
 *   transpose2d_grid(h, w, spec)           ckc_transpose2d_grid(...)
 *   (+ convenience: build -> lower .ll)    ckc_transpose2d_lower_to_llvm(...)
 *
 * SPEC AS AN EXPLICIT C STRUCT. The Python Transpose2DSpec is a frozen
 * dataclass with defaults. ckc_transpose2d_spec_default() returns a struct
 * with every field at the Python dataclass default (tile_m=64, tile_n=64,
 * vec=8, dtype="f16", lds_pad=8, grid_order="row", name="ck_dsl_transpose2d").
 * block_size is a derived @property recomputed by the accessor (not stored).
 *
 * Error model mirrors the rest of the C port: build/lower routes errors through
 * the sticky-error IRBuilder (ckc_b_*); the validity gate returns a bool + a
 * reason string; the convenience lower returns a ckc_status_t.
 */
#ifndef CKC_INSTANCE_TRANSPOSE_H
#define CKC_INSTANCE_TRANSPOSE_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------- Transpose2DSpec
 *
 * Mirror of the Python frozen dataclass. dtype is "f16" | "bf16"; grid_order
 * is "row" | "morton" (compared by strcmp, like the Python Literal). */
typedef struct ckc_transpose2d_spec
{
    int tile_m;             /* default 64               */
    int tile_n;             /* default 64               */
    int vec;                /* default 8                */
    const char* dtype;      /* default "f16"            */
    int lds_pad;            /* default 8                */
    const char* grid_order; /* default "row"            */
    const char* name;       /* default "ck_dsl_transpose2d" */
} ckc_transpose2d_spec_t;

/* Default-constructed spec (every field == Python dataclass default). */
ckc_transpose2d_spec_t ckc_transpose2d_spec_default(void);

/* Transpose2DSpec.block_size @property: (tile_m * tile_n) // vec. */
int ckc_transpose2d_block_size(const ckc_transpose2d_spec_t* spec);

/* Transpose2DSpec.kernel_name() -> NUL-terminated into `out` (capacity
 * out_cap). Returns CKC_OK or CKC_ERR_VALUE (buffer too small / NULL). */
ckc_status_t
ckc_transpose2d_kernel_name(const ckc_transpose2d_spec_t* spec, char* out, size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950". On a
 * reject, `reason` (if non-NULL, capacity reason_cap) receives the structured
 * message; returns false. On accept returns true and writes "ok". */
bool ckc_transpose2d_is_valid_spec(const ckc_transpose2d_spec_t* spec,
                                   const char* arch,
                                   char* reason,
                                   size_t reason_cap);

/* build_transpose2d(spec). Builds the IR into the supplied (already
 * ckc_ir_builder_init'd) builder `b`, exactly as the Python build does, and
 * returns the kernel (b->kernel) on success or NULL with b's sticky error set.
 * `arch` is accepted for parity (the Python build always validates against
 * gfx950); NULL => "gfx950".
 *
 * Like the Python, this expects the builder to have been created with the
 * spec's kernel_name(). Use ckc_build_transpose2d_new() for the init-from-spec
 * convenience. */
ckc_kernel_def_t*
ckc_build_transpose2d(ckc_ir_builder_t* b, const ckc_transpose2d_spec_t* spec, const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with ckc_ir_builder_free(). Returns the kernel or NULL. */
ckc_kernel_def_t* ckc_build_transpose2d_new(ckc_ir_builder_t* b,
                                            const ckc_transpose2d_spec_t* spec,
                                            const char* arch);

/* transpose2d_grid(h, w, spec) -> (x, y, z) launch grid into out[3]. */
ckc_status_t ckc_transpose2d_grid(int h, int w, const ckc_transpose2d_spec_t* spec, int out[3]);

/* Convenience: given a spec, init a builder, build, and lower to LLVM .ll text.
 * `arch` NULL => "gfx950". On CKC_OK *out_ll receives a malloc'd NUL-terminated
 * string the caller frees with free(); on failure it is left NULL and (if
 * err!=NULL, capacity err_cap) a diagnostic is written. Internally owns and
 * frees its IRBuilder. */
ckc_status_t ckc_transpose2d_lower_to_llvm(const ckc_transpose2d_spec_t* spec,
                                           const char* arch,
                                           ckc_llvm_flavor_t flavor,
                                           char** out_ll,
                                           char* err,
                                           size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_INSTANCE_TRANSPOSE_H */
