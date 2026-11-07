/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_transpose_bc.h -- C++ twin of the no-LDS in-register
 * transpose kernel instance builder rocke/instances/common/transpose_bc.py.
 *
 * transpose_bc.py is a pure memory-movement kernel: each lane loads a vec x vec
 * sub-tile, transposes it in registers and stores it, with NO LDS and NO MFMA
 * atoms. The build entry mirrors the Python op-by-op so the emitted IR op
 * stream is byte-identical.
 *
 *   Python (transpose_bc.py)              C (this header)
 *   -----------------------------------   -------------------------------------
 *   class TransposeBcSpec                 rocke_transpose_bc_spec_t
 *   TransposeBcSpec()                     rocke_transpose_bc_spec_default()
 *   TransposeBcSpec.block_size            rocke_transpose_bc_block_size()
 *   TransposeBcSpec.kernel_name()         rocke_transpose_bc_kernel_name()
 *   is_valid_spec(spec, arch)             rocke_transpose_bc_is_valid_spec(...)
 *   build_transpose_bc(spec, arch)        rocke_build_transpose_bc_new(...)
 *
 * transpose_bc_grid and transpose_bc_signature are host-side launch helpers
 * with no C twin.
 *
 * Error model mirrors the rest of the C port: the build entry records the
 * Python raise on the builder and returns NULL; the validity gate returns a
 * bool + a reason string.
 */
#ifndef ROCKE_INSTANCE_TRANSPOSE_BC_H
#define ROCKE_INSTANCE_TRANSPOSE_BC_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Mirror of the Python frozen dataclass TransposeBcSpec. dtype is "f16" |
 * "bf16" (the Python Literal DType). */
typedef struct rocke_transpose_bc_spec
{
    int tile_m; /* default 64                   */
    int tile_n; /* default 64                   */
    int vec; /* default 8                    */
    const char* dtype; /* default "f16"                */
    bool use_buffer_io; /* default false                */
    const char* name; /* default "rocke_transpose_bc" */
} rocke_transpose_bc_spec_t;

/* Default-constructed spec (every field == Python dataclass default). */
rocke_transpose_bc_spec_t rocke_transpose_bc_spec_default(void);

/* TransposeBcSpec.block_size @property: (tile_m // vec) * (tile_n // vec),
 * with Python floor division. Returns 0 for a NULL spec or vec == 0. */
int rocke_transpose_bc_block_size(const rocke_transpose_bc_spec_t* spec);

/* TransposeBcSpec.kernel_name() -> NUL-terminated into `out` (capacity
 * out_cap). Returns ROCKE_OK or ROCKE_ERR_VALUE (buffer too small / NULL). */
rocke_status_t rocke_transpose_bc_kernel_name(const rocke_transpose_bc_spec_t* spec,
                                              char* out,
                                              size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950". On a
 * reject, `reason` (if non-NULL, capacity reason_cap) receives the Python
 * reason text and the call returns false; on accept it returns true and writes
 * "ok". */
bool rocke_transpose_bc_is_valid_spec(const rocke_transpose_bc_spec_t* spec,
                                      const char* arch,
                                      char* reason,
                                      size_t reason_cap);

/* build_transpose_bc(spec, arch). Inits `b` with spec.kernel_name() and builds
 * the kernel into it. `arch` NULL => "gfx950". Returns the kernel (b->kernel),
 * or NULL with b's sticky error set (an invalid spec records "invalid
 * transpose_bc spec: <reason>"). The caller owns `b` and frees it with
 * rocke_ir_builder_free(). */
rocke_kernel_def_t* rocke_build_transpose_bc_new(rocke_ir_builder_t* b,
                                                 const rocke_transpose_bc_spec_t* spec,
                                                 const char* arch);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_TRANSPOSE_BC_H */
