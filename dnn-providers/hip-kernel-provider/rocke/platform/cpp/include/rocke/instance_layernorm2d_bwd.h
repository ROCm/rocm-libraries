/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_layernorm2d_bwd.h -- C99 port of the LayerNorm2D backward
 * kernel instance builder rocke/instances/common/layernorm2d_bwd.py.
 *
 * Three AOT-generic kernels (runtime M / N / P), selected by spec.part:
 *
 *   "dx"           dX from dY, X, gamma and the saved mean / rstd (one CTA
 *                  per row, twin row reduction of dY*gamma and dY*gamma*X)
 *   "dwdb"         dgamma / dbeta column reduction over a row slice per CTA;
 *                  writes the outputs directly when P == 1, else f32 partials
 *                  into WS[P][2][N]
 *   "dwdb_reduce"  folds the P workspace partials into dgamma / dbeta
 *
 *   Python (layernorm2d_bwd.py)           C99 (this header)
 *   -----------------------------------   ------------------------------------------
 *   class LayerNorm2DBwdSpec               rocke_layernorm2d_bwd_spec_t
 *   is_valid_spec(spec, arch)             rocke_layernorm2d_bwd_is_valid_spec(...)
 *   build_layernorm2d_bwd(spec)           rocke_build_layernorm2d_bwd(b, spec)
 *   (+ convenience: build -> lower .ll)   rocke_layernorm2d_bwd_lower_to_llvm(...)
 *
 * Error model as instance_layernorm2d.h.
 */
#ifndef ROCKE_INSTANCE_LAYERNORM2D_BWD_H
#define ROCKE_INSTANCE_LAYERNORM2D_BWD_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Mirror of Python LayerNorm2DBwdSpec. ``part`` and ``dtype`` are the Literal
 * strings (compared by strcmp like Python). */
typedef struct rocke_layernorm2d_bwd_spec
{
    const char* part; /* default "dx" ("dx"|"dwdb"|"dwdb_reduce") */
    int block_size; /* default 256                                */
    int vec; /* default 4                                  */
    const char* dtype; /* default "f16" ("f16"|"bf16")               */
    int wave_size; /* default 64                                 */
    bool fast_reduce; /* default false; dx only, wave-prologue fold */
    int unroll; /* default 4; dwdb / dwdb_reduce only        */
    const char* name; /* default "rocke_layernorm2d_bwd"            */
} rocke_layernorm2d_bwd_spec_t;

/* Default-constructed spec (every field == Python dataclass default). */
rocke_layernorm2d_bwd_spec_t rocke_layernorm2d_bwd_spec_default(void);

/* LayerNorm2DBwdSpec.kernel_name() -> NUL-terminated into out (capacity
 * out_cap). Returns ROCKE_OK or ROCKE_ERR_VALUE (buffer too small). */
rocke_status_t rocke_layernorm2d_bwd_kernel_name(const rocke_layernorm2d_bwd_spec_t* spec,
                                                 char* out,
                                                 size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950". */
bool rocke_layernorm2d_bwd_is_valid_spec(const rocke_layernorm2d_bwd_spec_t* spec,
                                         const char* arch,
                                         char* reason,
                                         size_t reason_cap);

/* build_layernorm2d_bwd(spec) into an already-initialised builder `b`.
 * Returns b->kernel or NULL with b's sticky error set. */
rocke_kernel_def_t* rocke_build_layernorm2d_bwd(rocke_ir_builder_t* b,
                                                const rocke_layernorm2d_bwd_spec_t* spec);

/* Convenience: init `b` with spec.kernel_name(), then build. */
rocke_kernel_def_t* rocke_build_layernorm2d_bwd_new(rocke_ir_builder_t* b,
                                                    const rocke_layernorm2d_bwd_spec_t* spec);

/* Convenience: build and lower to LLVM .ll text (malloc'd, caller frees). */
rocke_status_t rocke_layernorm2d_bwd_lower_to_llvm(const rocke_layernorm2d_bwd_spec_t* spec,
                                                   const char* arch,
                                                   rocke_llvm_flavor_t flavor,
                                                   char** out_ll,
                                                   char* err,
                                                   size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_LAYERNORM2D_BWD_H */
