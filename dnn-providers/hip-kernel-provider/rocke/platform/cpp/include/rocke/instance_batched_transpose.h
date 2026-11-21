/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_batched_transpose.h -- C++ twin of the batched 2D transpose
 * kernel instance builder rocke/instances/common/batched_transpose.py.
 *
 * batched_transpose.py is the 2D transpose (coalesced HBM read -> LDS stage ->
 * coalesced HBM write, no MFMA atoms) with a batch axis on block_id_z. The
 * build entry mirrors the Python op-by-op so the emitted IR op stream is
 * byte-identical.
 *
 *   Python (batched_transpose.py)          C (this header)
 *   -----------------------------------    -------------------------------------
 *   class BatchedTranspose2DSpec           rocke_batched_transpose2d_spec_t
 *   BatchedTranspose2DSpec()               rocke_batched_transpose2d_spec_default()
 *   BatchedTranspose2DSpec.block_size      rocke_batched_transpose2d_block_size()
 *   BatchedTranspose2DSpec.kernel_name()   rocke_batched_transpose2d_kernel_name()
 *   is_valid_spec(spec, arch)              rocke_batched_transpose2d_is_valid_spec(...)
 *   build_batched_transpose2d(spec, arch)  rocke_build_batched_transpose2d_new(...)
 *
 * batched_transpose2d_grid and batched_transpose2d_signature are host-side
 * launch helpers with no C twin.
 *
 * Error model mirrors the rest of the C port: the build entry records the
 * Python raise on the builder and returns NULL; the validity gate returns a
 * bool + a reason string.
 */
#ifndef ROCKE_INSTANCE_BATCHED_TRANSPOSE_H
#define ROCKE_INSTANCE_BATCHED_TRANSPOSE_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Mirror of the Python frozen dataclass. dtype is "f16" | "bf16" (the DType
 * Literal; "fp16" is accepted as an alias of "f16"). */
typedef struct rocke_batched_transpose2d_spec
{
    int tile_m; /* default 64                          */
    int tile_n; /* default 64                          */
    int vec; /* default 8                           */
    const char* dtype; /* default "f16"                       */
    int lds_pad; /* default 8                           */
    const char* name; /* default "rocke_batched_transpose2d" */
} rocke_batched_transpose2d_spec_t;

/* Default-constructed spec (every field == Python dataclass default). */
rocke_batched_transpose2d_spec_t rocke_batched_transpose2d_spec_default(void);

/* BatchedTranspose2DSpec.block_size @property: (tile_m * tile_n) // vec, with
 * Python floor division. Returns 0 for a NULL spec or vec == 0 (where the
 * Python raises ZeroDivisionError). */
int rocke_batched_transpose2d_block_size(const rocke_batched_transpose2d_spec_t* spec);

/* BatchedTranspose2DSpec.kernel_name() -> NUL-terminated into `out` (capacity
 * out_cap). Returns ROCKE_OK or ROCKE_ERR_VALUE (buffer too small / NULL). */
rocke_status_t rocke_batched_transpose2d_kernel_name(const rocke_batched_transpose2d_spec_t* spec,
                                                     char* out,
                                                     size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950". On a
 * reject, `reason` (if non-NULL, capacity reason_cap) receives the Python
 * reason text and the call returns false; on accept it returns true and writes
 * "ok". vec == 0 is a reject carrying the Python ZeroDivisionError text. */
bool rocke_batched_transpose2d_is_valid_spec(const rocke_batched_transpose2d_spec_t* spec,
                                             const char* arch,
                                             char* reason,
                                             size_t reason_cap);

/* build_batched_transpose2d(spec, arch). Inits `b` with spec.kernel_name() and
 * builds the kernel into it. `arch` NULL => "gfx950". Returns the kernel
 * (b->kernel), or NULL with b's sticky error set (an invalid spec records
 * "invalid batched_transpose2d spec: <reason>"). The caller owns `b` and frees
 * it with rocke_ir_builder_free(). */
rocke_kernel_def_t* rocke_build_batched_transpose2d_new(
    rocke_ir_builder_t* b, const rocke_batched_transpose2d_spec_t* spec, const char* arch);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_BATCHED_TRANSPOSE_H */
